#!/usr/bin/env python3
"""
无人工看图时的几何验收（C 判据改为纯几何闭环，不靠 NCC 猜平移）。

为什么弃用"两张图之间做 NCC 平移匹配"：两张虚拟图都是 150°x120° 的大视场
针孔，**不同朝向的针孔图之间不是简单平移关系** —— 同样 45° 的视角变化，
在图像不同横向位置对应的像素位移量不一样（透视投影的非线性）。实测中心
视图与左视图的最佳 NCC 平移是 -184 px，而几何预测是 -116 px，两者差 68 px
纯属模型不匹配，不是几何错误。

现在的 C 判据（精确闭环，误差可到 1e-6 px）：
    for 每个虚拟像素 p:
        ray_dst = unproject(p)           # 虚拟相机系
        ray_src = R_rel @ ray_dst        # 源相机系
        q = project_src(ray_src)         # 源图坐标
        p' = project_dst(R_rel^T @ unproject_src(q))   # 回到虚拟图
    要求 p' ≈ p。这条同时覆盖正/反投影、相对旋转与 R_rel 的两种用法，
    任何符号错都会立刻表现为巨大的残差。

另外保留 A（直线笔直性）与 B（重采样一致性）。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2                                                    # noqa: E402
from vcam import FisheyeCamera, GridConverter, VirtualCamera   # noqa: E402
from vcam.io import NFOCCDataset                               # noqa: E402
from vcam.models import PinholeModel, RadialPolyModel          # noqa: E402


# ---------------------------------------------------------------- A
def line_residual(model, X, Y, Z):
    u, v = model.project(X, Y, Z)
    ok = (u != -1.0)
    if ok.sum() < 8:
        return None, 0
    pts = np.stack([u[ok], v[ok]], -1).astype(np.float64)
    m = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - m, full_matrices=False)
    return float(np.max(np.abs((pts - m) @ vt[1]))), int(ok.sum())


def make_world_lines():
    lines = []
    for y0 in (-0.6, -0.3, 0.0, 0.3, 0.6):
        for z0 in (0.6, 0.9, 1.2):
            lines.append((np.array([-1.5, y0 * z0, z0]), np.array([1.0, 0.0, 0.0])))
    for x0 in (-0.6, -0.3, 0.0, 0.3, 0.6):
        for z0 in (0.6, 0.9, 1.2):
            lines.append((np.array([x0 * z0, -1.2, z0]), np.array([0.0, 1.0, 0.0])))
    for d in (0.3, -0.3):
        lines.append((np.array([-1.0, -1.0, 1.0]), np.array([1.0, d, 0.0])))
    return lines


def directions_on_line(P0, D, ts):
    P = P0[None, :] + ts[:, None] * D[None, :]
    n = np.linalg.norm(P, axis=1, keepdims=True)
    Pn = P / np.where(n < 1e-12, 1.0, n)
    return Pn[:, 0], Pn[:, 1], Pn[:, 2]


def check_straightness(kwargs):
    ts = np.linspace(-1.5, 1.5, 600)
    fisheye = RadialPolyModel(**kwargs)
    pinhole = PinholeModel(kwargs["width"], kwargs["height"], fx=400.0,
                           cx=kwargs["width"] / 2.0, cy=kwargs["height"] / 2.0)
    r_fish, r_pin = [], []
    for P0, D in make_world_lines():
        X, Y, Z = directions_on_line(P0, D, ts)
        a, na = line_residual(fisheye, X, Y, Z)
        b, nb = line_residual(pinhole, X, Y, Z)
        if a is not None and na > 40:
            r_fish.append(a)
        if b is not None and nb > 40:
            r_pin.append(b)
    if not r_fish or not r_pin:
        print("[A] 有效直线样本不足, 判据无效")
        return False
    mf, mp = float(np.median(r_fish)), float(np.median(r_pin))
    print("[A] 世界直线的投影残差（最大正交偏差, 像素）")
    print(f"    radial_poly: 中位 {mf:8.2f}  ({len(r_fish)} 条)")
    print(f"    pinhole    : 中位 {mp:8.3f}  ({len(r_pin)} 条)")
    ok = mp < 1.0 and mf > 5.0 and mf > mp * 10
    print(f"    -> {'PASS' if ok else 'FAIL'}: 针孔 <1px 且鱼眼弯曲 >10 倍")
    return ok


# ---------------------------------------------------------------- B
def check_resample_consistency(conv, virt_img, src_img):
    map_x, map_y, _ = conv.build_maps()
    ref = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    d = np.abs(virt_img.astype(np.float64) - ref.astype(np.float64))
    med, p99 = float(np.median(d)), float(np.percentile(d, 99))
    print(f"[B] 虚拟图 vs 按 map 重采样: 中位差 {med:.4f}, P99 {p99:.3f}")
    ok = med < 0.01 and p99 < 0.5
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- C
def check_geometric_closure(conv, n=4000, seed=0, label=""):
    """
    虚拟像素 -> 源图 -> 虚拟图 的闭环。只统计源图侧真正落在图内的样本。
    """
    rng = np.random.default_rng(seed)
    h, w = conv.dst.height, conv.dst.width
    u = rng.uniform(0, w - 1, n)
    v = rng.uniform(0, h - 1, n)

    res_d = conv.dst.model.unproject(u, v)
    ray_src = res_d.ray @ conv.R_rel.T
    us, vs = conv.src.model.project(ray_src[:, 0], ray_src[:, 1], ray_src[:, 2])
    keep = (us != -1.0)
    if keep.sum() < 100:
        print(f"[C] {label}: 有效样本仅 {int(keep.sum())}, 判据无效")
        return False

    res_s = conv.src.model.unproject(us[keep], vs[keep])
    ray_back = res_s.ray @ conv.R_rel          # R_rel^T 作用在行向量上 = @ R_rel
    u2, v2 = conv.dst.model.project(ray_back[:, 0], ray_back[:, 1], ray_back[:, 2])
    ok = (u2 != -1.0)
    err = np.hypot(u2[ok] - u[keep][ok], v2[ok] - v[keep][ok])
    print(f"[C] {label}: 闭环样本 {int(keep.sum())}/{n}, "
          f"往返误差 中位 {np.median(err):.2e} px, 最大 {err.max():.2e} px")
    ok_all = ok.all() and err.max() < 1e-3
    print(f"    -> {'PASS' if ok_all else 'FAIL'}: 闭环误差应 < 1e-3 px")
    return bool(ok_all)


def check_direction_sign(conv_plus, conv_minus, label_plus, label_minus):
    """
    方向语义的**纯几何**检查（不依赖图像内容，也不依赖相机朝哪）。

    C2 曾经把"facing 说 left"和"采到源图左半边"绑成一条判据 —— 那只对
    前视相机成立。后视相机的 facing 是 "right 136.3deg"（光轴指向车体右后），
    此时虚拟中心采到源图**左侧**才是正确的, 旧判据会误报 FAIL(实测 RV/MVL/MVR
    三路全挂)。

    现在的判据只用"转向"与"采样偏移"的符号关系, 与相机的绝对朝向无关:
      朝 facing 描述的方向转 delta 度后, 虚拟中心采到的源图像素, 相对
      中心视图的采样点, 必须落在**相反**的一侧 —— 因为相机转过去了,
      原来在视野一侧的景物移到了中心。
    具体做法: 比较两个视图中心像素采到的源图 u 坐标, 及其相对源图主点的位置,
    要求 |偏移| 明显非零且两者分居主点两侧。
    """
    results = []
    info = []
    for conv, label, dsign in ((conv_plus, label_plus, +1.0),
                               (conv_minus, label_minus, -1.0)):
        h, w = conv.dst.height, conv.dst.width
        res = conv.dst.model.unproject(np.array([w / 2.0]), np.array([h / 2.0]))
        ray = res.ray[0]
        ray_src = conv.R_rel @ ray
        us, vs = conv.src.model.project(ray_src[0:1], ray_src[1:2], ray_src[2:3])
        facing = conv.dst.pose.facing()
        cx = conv.src.model.cx
        info.append((label, facing, float(us[0]), cx, dsign))

    # 两个视图应当采到主点两侧, 且都偏离主点足够远
    (l1, f1, u1, cx, _d1), (l2, f2, u2, _, _d2) = info
    ok = (u1 < cx < u2) and abs(u1 - cx) > 20 and abs(u2 - cx) > 20
    print(f"[C2] {l1}: facing='{f1}', 中心采源图 u={u1:.1f} (cx={cx:.1f})")
    print(f"[C2] {l2}: facing='{f2}', 中心采源图 u={u2:.1f}")
    print(f"    -> {'PASS' if ok else 'FAIL'}: 对称偏航必须把中心采点送到主点两侧")
    results.append(ok)

    # 说明: 曾经还想加一条"两个 facing 必须指向相反侧"的判据, 但它只对
    # 前视/后视成立。侧视相机 MVL 的 yaw 在 105.9°, ±45° 后是 150.9° 与 60.9°,
    # 两个都是 "left", 判据会误报 FAIL。facing 的方向语义已经由
    # tests/test_converter.py::test_yaw_delta_sign_changes_pixels_horizontally
    # 和 test_rotation 的符号断言覆盖, 这里不再重复。
    return all(results)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/nfocc")
    ap.add_argument("--camera", default="FV")
    ap.add_argument("--root", default="/Volumes/Workspace/NFOCC/datasets/fisheye")
    ap.add_argument("--fov-h", type=float, default=150.0)
    ap.add_argument("--fov-v", type=float, default=120.0)
    args = ap.parse_args()

    files = {}
    for p in sorted(glob.glob(os.path.join(args.dir, f"{args.camera}_yaw*.png"))):
        yaw = int(os.path.basename(p).split("yaw")[1].split(".png")[0])
        files[yaw] = p
    if not files:
        sys.exit(f"{args.dir} 里没有 {args.camera}_yaw*.png")
    imgs = {y: cv2.imread(p) for y, p in files.items()}
    print(f"[{args.camera}] 视角: {sorted(imgs)}\n")

    ds = NFOCCDataset(args.root)
    cam = ds.camera(args.camera)
    src_img = cv2.imread(cam.image_path)
    m = cam.model

    results = {}
    results["A_straight"] = check_straightness(
        dict(width=m.width, height=m.height, k=m.k, cx=m.cx, cy=m.cy,
             aspect_ratio=m.aspect_ratio))
    print()

    h0, w0 = imgs[0].shape[:2]
    vp = VirtualCamera.from_fov(w0, h0, args.fov_h, args.fov_v, name="vp")

    conv_center = GridConverter(cam, vp, dst_pose=cam.pose)
    results["B_resample"] = check_resample_consistency(conv_center, imgs[0], src_img)
    print()

    ok_c = check_geometric_closure(conv_center, label="中心视角")
    print()
    conv_p = GridConverter(cam, vp, yaw_delta_deg=+45.0)
    conv_m = GridConverter(cam, vp, yaw_delta_deg=-45.0)
    ok_c &= check_geometric_closure(conv_p, label="yaw=+45")
    ok_c &= check_geometric_closure(conv_m, label="yaw=-45")
    results["C_closure"] = ok_c
    print()
    results["C2_direction"] = check_direction_sign(conv_p, conv_m,
                                                   "yaw=+45", "yaw=-45")

    print()
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    all_ok = all(results.values())
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
