#!/usr/bin/env python3
"""Tile one recorded episode's three camera videos into an animation: head camera on top, wrists below.

  python make_demo_gif.py <episode video dir> <output.gif|.webp> [--speed 2] [--width 480] [--title TEXT]
                         [--front <run>/front_view]

With --front, the viewer frames captured by record_demo.sh (FRONT_VIEW=1) are matched to the
camera frames by time and placed to the left of them; --width is then the total width.

The format follows the output extension. Animated WebP is several times smaller than GIF for these
renders, and GitHub shows both inline.

Needs numpy, imageio and Pillow (all in the policy image). The episode directory is one of
videos/episode_*_seed*/ written by record_demo.sh.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

CAMERAS = ("head_cam_h", "wrist_cam_l", "wrist_cam_r")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--speed", type=float, default=2.0, help="Playback speed relative to real time")
    p.add_argument("--width", type=int, default=480, help="Output width in pixels")
    p.add_argument("--title", default="", help="Caption drawn in the top-left corner")
    p.add_argument("--max-fps", type=float, default=10.0, help="Drop frames to stay at or below this rate")
    p.add_argument("--colors", type=int, default=128, help="GIF palette size")
    p.add_argument("--quality", type=int, default=70, help="WebP quality")
    p.add_argument("--front", type=Path, help="Viewer frames saved by record_demo.sh FRONT_VIEW=1")
    a = p.parse_args()

    videos = {c: imageio.mimread(a.episode / f"observation.images.{c}.mp4", memtest=False) for c in CAMERAS}
    count = min(len(v) for v in videos.values())
    times = json.loads((a.episode / "frame_times.json").read_text())
    real_fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 10.0
    height, width = videos["head_cam_h"][0].shape[:2]
    step = max(1, round(real_fps * a.speed / a.max_fps))
    front = None
    if a.front:
        start = json.loads((a.episode / "episode_start.json").read_text())["unix_time"]
        shots = sorted(a.front.glob("*.jpg"), key=lambda f: float(f.stem))
        stamps = np.array([float(f.stem) for f in shots])
        front = (start, shots, stamps)
    frames = []
    for i in range(0, count, step):
        head = videos["head_cam_h"][i]
        wrists = np.concatenate([videos["wrist_cam_l"][i], videos["wrist_cam_r"][i]], axis=1)
        wrists = np.asarray(Image.fromarray(wrists).resize((width, height // 2), Image.LANCZOS))
        tile = Image.fromarray(np.concatenate([head, wrists], axis=0))
        if front:
            start, shots, stamps = front
            view = Image.open(shots[int(np.abs(stamps - (start + times[i])).argmin())]).convert("RGB")
            view = view.resize((round(view.width * tile.height / view.height), tile.height), Image.LANCZOS)
            joined = Image.new("RGB", (view.width + tile.width, tile.height))
            joined.paste(view, (0, 0))
            joined.paste(tile, (view.width, 0))
            tile = joined
        scale = a.width / tile.width
        tile = tile.resize((a.width, round(tile.height * scale)), Image.LANCZOS)
        if a.title:
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 8 + 7 * len(a.title), 18), fill=(0, 0, 0))
            draw.text((4, 3), a.title, fill=(255, 255, 255))
        frames.append(tile)
    duration_ms = round(1000 * step / (real_fps * a.speed))
    if a.output.suffix.lower() == ".webp":
        frames[0].save(a.output, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0,
                       quality=a.quality, method=6)
    else:
        frames = [f.convert("P", palette=Image.ADAPTIVE, colors=a.colors) for f in frames]
        frames[0].save(a.output, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0,
                       optimize=True, disposal=1)
    print(f"{a.output}: {len(frames)} frames from {count}, {real_fps:.1f} fps recorded, {a.speed}x playback, "
          f"{a.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
