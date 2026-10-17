#!/usr/bin/env python3
"""Sequential AV1 decode -> q95 4:4:4 JPEG, one video per worker."""
import argparse, concurrent.futures as cf, json, math, os, pathlib, random, shutil, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import av
import numpy as np
from PIL import Image

def atomic_json(path, value):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def build_file(job):
    t0 = time.time()
    source, target, frame_count, samples, deadline = job
    target = pathlib.Path(target); target.mkdir(parents=True, exist_ok=True)
    selected = set(samples)
    observations = []
    total_bytes = 0
    mismatch_count = 0
    n = 0
    with av.open(source) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "NONE"
        fps = float(stream.average_rate)
        for n, frame in enumerate(container.decode(stream), start=1):
            ordinal = n - 1
            if ordinal % 200 == 0:
                if time.time() > deadline:
                    raise TimeoutError("Cache build deadline reached")
                if shutil.disk_usage(target).free < 3 * 1024**3:
                    raise RuntimeError("Less than 3 GiB remains: retaining partial cache")
                atomic_json(target / "progress.json", {"frames": ordinal, "expected_frames": frame_count, "bytes": total_bytes, "elapsed_seconds": time.time()-t0})
            if frame.width != 640 or frame.height != 480:
                raise ValueError(f"Unexpected shape {frame.width}x{frame.height}")
            if abs(float(frame.time) - ordinal / fps) > 1e-5:
                mismatch_count += 1
            rgb = frame.to_ndarray(format="rgb24")
            destination = target / f"frame-{ordinal:08d}.jpg"
            Image.fromarray(rgb).save(destination, format="JPEG", quality=95, subsampling=0, optimize=False)
            total_bytes += destination.stat().st_size
            if ordinal in selected:
                with Image.open(destination) as im:
                    recovered = np.asarray(im.convert("RGB"))
                d = recovered.astype(np.float32) - rgb.astype(np.float32)
                mse = float(np.mean(d*d))
                observations.append({"frame_index": ordinal, "mae_uint8": float(np.mean(np.abs(d))), "mse_uint8": mse, "psnr_db": float(10*math.log10(255**2 / mse)) if mse else 99.0, "max_abs_diff": int(np.max(np.abs(d)))})
    if n != frame_count:
        raise ValueError(f"Frame count mismatch for {source}: {n} != {frame_count}")
    result = {"source": source, "frame_count": n, "total_bytes": total_bytes, "elapsed_seconds": time.time()-t0, "pts_mismatch_count": mismatch_count, "fidelity": observations, "shape_hwc":[480,640,3]}
    atomic_json(target / "progress.json", result)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", required=True); p.add_argument("--cache",required=True)
    p.add_argument("--workers",type=int,default=12);p.add_argument("--seconds",type=float,default=900)
    p.add_argument("--deadline-unix",type=float,required=True)
    a=p.parse_args()
    t0=time.time();deadline=min(a.deadline_unix,t0+a.seconds)
    dataset=pathlib.Path(a.dataset);cache=pathlib.Path(a.cache)
    info=json.loads((dataset/"meta/info.json").read_text())
    files=sorted(dataset.glob("videos/**/*.mp4"))
    metadata={"status":"building","version":1,"dataset":str(dataset),"cache":str(cache),"quality":95,"subsampling":0,"jpeg_chroma":"4:4:4","fps":info["fps"],"shape_hwc":[480,640,3],"frame_index_rule":"round(video_timestamp * fps); all source PTS verified against ordinal/fps","video_files":{}}
    jobs=[]
    for f in files:
        with av.open(str(f)) as c:
            s=c.streams.video[0];count=s.frames;fps=float(s.average_rate)
            if fps != info["fps"] or s.start_time != 0:
                raise ValueError(f"Unsupported timestamps {f}: fps={fps}, start={s.start_time}")
        rel=f.relative_to(dataset).as_posix()
        d=f.relative_to(dataset/"videos").with_suffix("").as_posix()
        metadata["video_files"][rel]={"directory":d,"pattern":"frame-{frame_index:08d}.jpg","frame_count":count,"fps":fps,"start_time":0}
        jobs.append([str(f),str(cache/d),count,[],deadline])
    total=sum(x[2] for x in jobs)
    chosen=random.Random(20261004).sample(range(total),200)
    offset=0
    for job in jobs:
        job[3]=[n-offset for n in chosen if offset <= n < offset+job[2]]
        offset+=job[2]
    atomic_json(cache/"metadata.json",metadata)
    results=[]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(build_file,j):j for j in jobs}
        for f in cf.as_completed(futures):
            result=f.result();results.append(result)
            metadata["completed_files"]=len(results)
            metadata["completed_frames"]=sum(r["frame_count"] for r in results)
            metadata["completed_bytes"]=sum(r["total_bytes"] for r in results)
            metadata["elapsed_seconds"]=time.time()-t0
            atomic_json(cache/"metadata.json",metadata)
            print(json.dumps({"completed_files":len(results),"file":result["source"],"frames":result["frame_count"],"elapsed_seconds":round(time.time()-t0,2),"bytes":metadata["completed_bytes"]}),flush=True)
    camera_totals={}
    for r in results:
        cam=pathlib.Path(r["source"]).relative_to(dataset/"videos").parts[0]
        camera_totals[cam]=camera_totals.get(cam,0)+r["frame_count"]
    if set(camera_totals.values()) != {info["total_frames"]}:
        raise ValueError(f"Camera count mismatch {camera_totals}")
    fidelity=[{"video":r["source"],**v} for r in results for v in r["fidelity"]]
    mse=float(np.mean([v["mse_uint8"] for v in fidelity]))
    summary={"status":"complete","cache":str(cache),"build_wall_seconds":time.time()-t0,"workers":a.workers,"quality":95,"subsampling":0,"camera_frame_counts":camera_totals,"total_frames":total,"total_jpeg_bytes":sum(r["total_bytes"] for r in results),"total_jpeg_gib":sum(r["total_bytes"] for r in results)/1024**3,"pts_mismatch_count":sum(r["pts_mismatch_count"] for r in results),"fidelity":{"n":len(fidelity),"mae_uint8":float(np.mean([v["mae_uint8"] for v in fidelity])),"mae_unit":float(np.mean([v["mae_uint8"] for v in fidelity]))/255,"psnr_db_from_pooled_mse":10*math.log10(255**2/mse),"mean_frame_psnr_db":float(np.mean([v["psnr_db"] for v in fidelity])),"max_frame_mae_uint8":max(v["mae_uint8"] for v in fidelity),"min_frame_psnr_db":min(v["psnr_db"] for v in fidelity),"decoder":"PyAV rgb24 to Pillow libjpeg decode RGB"},"fidelity_samples":fidelity,"files":results}
    metadata.update({"status":"complete","completed_files":len(results),"completed_frames":total,"completed_bytes":summary["total_jpeg_bytes"],"camera_frame_counts":camera_totals,"pts_mismatch_count":summary["pts_mismatch_count"],"elapsed_seconds":summary["build_wall_seconds"]})
    atomic_json(cache/"metadata.json",metadata)
    atomic_json(cache/"build_summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("files","fidelity_samples")}),flush=True)
if __name__=="__main__":
    main()

