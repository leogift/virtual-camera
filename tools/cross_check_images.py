#!/usr/bin/env python3
"""
C++ 与 Python 的**端到端**图像对拍。

上面已经有 test_parity（map/投影的数值对拍），这里补的是最后一公里:
两边真的把 PNG 跑出来了, 图是不是同一张。

判据: 平均绝对差 < 0.01 灰度级, P100 <= 1 (只允许双线性舍入)。
阈值不是"看着差不多" —— 双线性插值在两侧是同一套公式, 差异只能来自
坐标的 float32/float64 边界, 超过 1 就说明坐标本身不一致 (实测曾经因为
C++ 侧漏用源相机姿态而差 166 px, 图上表现为"整体偏了一截")。

用法:
    python tools/cross_check_images.py --pairs out/nfocc/FV_yaw-45.png:out/cpp/yaw-45.png ...
或:
    python tools/cross_check_images.py --py-dir out/nfocc --cpp-dir out/cpp \
        --py-prefix FV --yaws -45 0 45
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("需要 opencv-python")


def compare(a_path, b_path, max_mean=0.01, max_peak=1):
    a = cv2.imread(a_path, cv2.IMREAD_COLOR)
    b = cv2.imread(b_path, cv2.IMREAD_COLOR)
    if a is None:
        return False, f"读不到 {a_path}"
    if b is None:
        return False, f"读不到 {b_path}"
    if a.shape != b.shape:
        return False, f"尺寸不同 {a.shape} vs {b.shape}"
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    mean, peak = float(d.mean()), int(d.max())
    ok = mean <= max_mean and peak <= max_peak
    return ok, (f"mean|d|={mean:.5f} max={peak} "
                f"P99={np.percentile(d, 99):.1f} >2:{(d > 2).mean() * 100:.4f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[], help="py_path:cpp_path")
    ap.add_argument("--py-dir", default=None)
    ap.add_argument("--cpp-dir", default=None)
    ap.add_argument("--py-prefix", default="FV")
    ap.add_argument("--yaws", nargs="*", type=int, default=[-45, 0, 45])
    ap.add_argument("--max-mean", type=float, default=0.01)
    ap.add_argument("--max-peak", type=int, default=1)
    args = ap.parse_args()

    pairs = []
    for p in args.pairs:
        if ":" not in p:
            sys.exit(f"--pairs 需要 py:cpp 形式, 收到 {p}")
        a, b = p.split(":", 1)
        pairs.append((a, b))
    if args.py_dir and args.cpp_dir:
        for y in args.yaws:
            # Python 侧文件名带正号: FV_yaw+0.png; C++ 侧: yaw+0.png
            py_name = f"{args.py_prefix}_yaw{y:+d}.png"
            cpp_name = f"yaw{y:+d}.png"
            pairs.append((os.path.join(args.py_dir, py_name),
                          os.path.join(args.cpp_dir, cpp_name)))
    if not pairs:
        sys.exit("没有可比的图对")

    all_ok = True
    for a, b in pairs:
        ok, msg = compare(a, b, args.max_mean, args.max_peak)
        all_ok &= ok
        print(f"[{'ok' if ok else 'FAIL'}] {os.path.basename(a)} vs "
              f"{os.path.basename(b)}: {msg}")
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
