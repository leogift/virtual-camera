#!/usr/bin/env python3
"""
把 NFOCC / WoodScape 鱼眼虚拟成理想针孔相机（可带偏航/俯仰/横滚增量）。

示例（FV 前视，虚拟出 左45 / 中 / 右45 三个 150x120 度视口）::

    python tools/convert_nfocc.py \
        --root /Volumes/Workspace/NFOCC/datasets/fisheye \
        --camera FV --yaws -45 0 45 --output-dir out/fv

全部四路一次做完并拼大图::

    python tools/convert_nfocc.py --all --yaws -45 0 45 --mosaic

这个脚本只做参数解析与接线，几何全部在 vcam 里。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2                                                    # noqa: E402
from vcam import FisheyeCamera, GridConverter, VirtualCamera   # noqa: E402
from vcam.io import NFOCCDataset, NFOCC_CAM_DESC               # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="NFOCC fisheye -> ideal pinhole virtualization")
    ap.add_argument("--root", default="/Volumes/Workspace/NFOCC/datasets/fisheye")
    ap.add_argument("--camera", default="FV", help="FV / RV / MVL / MVR")
    ap.add_argument("--all", action="store_true", help="四路相机全做")
    ap.add_argument("--yaws", type=float, nargs="+", default=[-45.0, 0.0, 45.0],
                    help="相对源相机的偏航增量(度)，正值 = 向左看")
    ap.add_argument("--pitch", type=float, default=0.0, help="附加俯仰增量(度)，正值=抬头")
    ap.add_argument("--roll", type=float, default=0.0, help="附加横滚增量(度)")
    ap.add_argument("--fov-h", type=float, default=150.0)
    ap.add_argument("--fov-v", type=float, default=120.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=None,
                    help="默认按 FOV 与宽高比自动算（保持方形像素）")
    ap.add_argument("--square-pixels", action="store_true", help="按 FOV 比例反推宽度")
    ap.add_argument("--inherited-view", action="store_true",
                    help="额外输出一帧'完全继承源姿态'的视图（fx 由 FOV 决定）")
    ap.add_argument("--output-dir", default="out/nfocc")
    ap.add_argument("--mosaic", action="store_true", help="生成多视角对比大图")
    ap.add_argument("--save-maps", action="store_true", help="保存 map_x/map_y 文本")
    ap.add_argument("--frame", type=int, default=0, help="同名相机取第几份标定")
    return ap.parse_args()


def height_for_fov(width, fov_h, fov_v):
    """由水平 FOV 与目标垂直 FOV 反推高度，使 fx/fy 一致（方形像素）。"""
    tan_h = np.tan(np.radians(fov_h / 2))
    tan_v = np.tan(np.radians(fov_v / 2))
    return int(round(width * tan_v / tan_h))


def build_virtual(cam, fov_h, fov_v, width, height, square_pixels, name):
    h = height or height_for_fov(width, fov_h, fov_v)
    return VirtualCamera.from_fov(width, h, fov_h, fov_v, square_pixels=square_pixels,
                                  name=name)


def render_one(cam, view, yaw, out_dir, tag, save_maps, inherited):
    img = cv2.imread(cam.image_path)
    if img is None:
        raise IOError(f"读不到图像 {cam.image_path}")
    conv = GridConverter(view, view, dst_pose=view.pose) if inherited else \
        GridConverter(cam, view, yaw_delta_deg=yaw)
    res = conv.remap(img)
    path = os.path.join(out_dir, f"{tag}_yaw{yaw:+.0f}.png")
    cv2.imwrite(path, res.image)
    if save_maps:
        res.save_maps(path.replace(".png", "_map_x.txt"),
                      path.replace(".png", "_map_y.txt"))
    print(f"  {tag:14s} yaw={yaw:+6.1f} -> {res.dst.pose.facing():22s} "
          f"valid={res.stats.valid_ratio * 100:5.1f}% "
          f"maxTheta={res.stats.max_theta_deg:5.1f}deg  {os.path.basename(path)}")
    if res.stats.note:
        print(f"      警告: {res.stats.note}")
    return res


def annotate(img, text, upscale=1):
    out = img.copy()
    if upscale != 1:
        out = cv2.resize(out, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    ds = NFOCCDataset(args.root)
    names = ds.camera_names() if args.all else [args.camera]
    os.makedirs(args.output_dir, exist_ok=True)
    meta = {"root": ds.root, "fov_h": args.fov_h, "fov_v": args.fov_v,
            "yaws": args.yaws, "cameras": {}}

    for name in names:
        cam = ds.camera(name, frame=args.frame)
        print(f"\n=== {name} ({NFOCC_CAM_DESC.get(name, '')}) ===")
        print(f"  {cam.summary()}")
        print(f"  图像: {cam.image_path}")

        base_v = build_virtual(cam, args.fov_h, args.fov_v, args.width,
                               args.height, args.square_pixels, f"{name}_vp")
        views = []
        head = base_v.pose.with_delta(0.0, args.pitch, args.roll)
        for yaw in args.yaws:
            vp = FisheyeCamera(name=f"{name}_vp", model=base_v.model,
                               pose=cam.pose.with_delta(yaw + 0.0, args.pitch, args.roll),
                               is_virtual=True)
            r = render_one(cam, vp, yaw + 0.0, args.output_dir, f"{name}",
                           args.save_maps, inherited=False)
            views.append((yaw, r))

        meta["cameras"][name] = dict(
            calib=cam.describe(), image=cam.image_path,
            virtual_resolution=[base_v.width, base_v.height],
            virtual_focal=[float(base_v.model.fx), float(base_v.model.fy)],
            views=[dict(yaw=yaw, facing=r.dst.pose.facing(),
                        valid_ratio=float(r.stats.valid_ratio),
                        out_of_source=int(r.stats.out_of_source),
                        max_theta_deg=float(r.stats.max_theta_deg))
                   for yaw, r in views])

        if args.mosaic:
            tiles = []
            for yaw, r in views:
                tiles.append(annotate(r.image,
                                      f"{name} yaw{yaw:+.0f} [{r.dst.pose.facing()}] "
                                      f"valid {r.stats.valid_ratio * 100:.0f}%"))
            row = np.hstack(tiles)
            # 顶部插入缩略的源图，便于对照
            thumb_w = row.shape[1] // 4
            src = cv2.imread(cam.image_path)
            th = int(src.shape[0] * thumb_w / src.shape[1])
            thumb = cv2.resize(src, (thumb_w, th))
            canvas = np.zeros((th + row.shape[0], row.shape[1], 3), np.uint8)
            canvas[:th, :thumb_w] = thumb
            canvas[th:, :] = row
            mp = os.path.join(args.output_dir, f"{name}_mosaic.png")
            cv2.imwrite(mp, canvas)
            print(f"  mosaic -> {mp}")

    with open(os.path.join(args.output_dir, "convert_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n输出目录: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
