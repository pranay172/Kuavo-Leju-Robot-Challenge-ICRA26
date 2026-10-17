"""Atomic complete ACT resume states, RNG and deterministic epoch-offset sampler."""
from __future__ import annotations
import copy, json, math, os, random, shutil, time, uuid
from pathlib import Path

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp-"+uuid.uuid4().hex[:8])
    temporary.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+"\n")
    os.replace(temporary,path)

def resolve_resume(path):
    path=Path(path).resolve()
    if path.is_file() and path.suffix==".pt": return path
    if (path/"resume.json").exists():
        metadata=json.loads((path/"resume.json").read_text())
        if metadata.get("status")!="complete": raise ValueError("Resume state is not committed")
        return path/metadata.get("state_file","state.pt")
    for pointer in (path/"latest_resume.json",path/"resume_states/latest_resume.json"):
        if pointer.exists():
            value=json.loads(pointer.read_text())
            directory=Path(value["directory"])
            if not directory.is_absolute(): directory=path/directory
            return resolve_resume(directory)
    raise ValueError(f"No complete resume state at {path}")

def load_resume(path, torch):
    location=resolve_resume(path)
    return torch.load(location,map_location="cpu",weights_only=False)

def cpu_tree(value, torch):
    if isinstance(value,torch.Tensor): return value.detach().cpu()
    if isinstance(value,dict): return {key:cpu_tree(item,torch) for key,item in value.items()}
    if isinstance(value,tuple): return tuple(cpu_tree(item,torch) for item in value)
    if isinstance(value,list): return [cpu_tree(item,torch) for item in value]
    return value

def capture_rng(torch,np):
    state={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"]=torch.cuda.get_rng_state_all()
    return state

def restore_rng(torch,np,state):
    messages=[]
    for label,restore,key in (("python",random.setstate,"python"),("numpy",np.random.set_state,"numpy"),
                             ("torch",torch.set_rng_state,"torch")):
        try: restore(state[key])
        except Exception as error: messages.append(f"{label}: {error}")
    if "cuda" in state and torch.cuda.is_available():
        try: torch.cuda.set_rng_state_all(state["cuda"])
        except Exception as error: messages.append(f"cuda: {error}")
    return messages

def save_resume(output,step,payload,torch,keep=2):
    output=Path(output).resolve();root=output/"resume_states";root.mkdir(parents=True,exist_ok=True)
    destination=root/f"step{step:06d}"
    if destination.exists(): destination=root/f"step{step:06d}_{time.time_ns()}"
    temporary=root/(".tmp_"+destination.name+"_"+uuid.uuid4().hex[:8]);temporary.mkdir()
    portable=cpu_tree(payload,torch)
    torch.save(portable,temporary/"state.pt")
    metadata={key:payload.get(key) for key in ("optimizer_step","phase","phase_kind","phase_training_seconds",
        "total_training_seconds","elapsed_seconds","ema_updates","consumed_samples","learning_rates","scaler_scale",
        "epoch","batches_in_epoch","schedule_spec","phase_wall_budget")}
    metadata.update(step=step,optimizer_step=step,status="complete",state_file="state.pt",
                    managed_by="act_resume_state_v1",saved_unix=time.time())
    atomic_json(temporary/"resume.json",metadata)
    os.replace(temporary,destination)
    atomic_json(output/"latest_resume.json",{"directory":str(destination),"optimizer_step":step})
    complete=[]
    for directory in root.iterdir():
        if not directory.is_dir() or directory.name.startswith("."): continue
        try:
            value=json.loads((directory/"resume.json").read_text())
            if value.get("managed_by")=="act_resume_state_v1" and value.get("status")=="complete" and (directory/"state.pt").exists():
                complete.append((value["saved_unix"],directory))
        except (OSError,ValueError,KeyError): pass
    for _,directory in sorted(complete)[:-keep]:
        shutil.rmtree(directory)
    return str(destination)

class EpochOffsetSampler:
    def __init__(self,length,seed,epoch=0,offset=0):
        self.length,self.seed,self.epoch,self.offset=length,seed,epoch,offset
    def __iter__(self):
        import torch
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        yield from torch.randperm(self.length,generator=generator).tolist()[self.offset:]
    def __len__(self): return max(0,self.length-self.offset)

def make_scheduler(optimizer,spec):
    from torch.optim.lr_scheduler import LambdaLR
    def multiplier(step):
        if spec["kind"]=="decay":
            progress=min(1.,max(0.,step/max(1,spec["decay_steps"])))
            return .5*(1+math.cos(math.pi*progress))
        warmup=spec["warmup_steps"]
        if step<warmup: return step/max(1,warmup)
        if spec["kind"]=="constant": return 1.
        progress=min(1.,max(0.,(step-warmup)/max(1,spec["total_steps"]-warmup)))
        return .5*(1+math.cos(math.pi*progress))
    return LambdaLR(optimizer,multiplier)

def restore_arguments(args,provided,payload):
    protected={"output","resume","decay_steps","deadline_unix"}
    # Explicit operational flags may override; absent defaults inherit the saved run.
    for name,value in payload.get("arguments",{}).items():
        if name.startswith("_") or name in protected: continue
        if name not in provided and hasattr(args,name): setattr(args,name,value)
    # Optimizer/schedule hparams always describe the loaded state until explicit decay.
    for name in ("lr","lr_backbone","weight_decay","warmup_steps","lr_schedule","ema_decay"):
        if name in payload.get("arguments",{}): setattr(args,name,payload["arguments"][name])
