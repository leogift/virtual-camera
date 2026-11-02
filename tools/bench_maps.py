#!/usr/bin/env python3
"""
测 C++ 核心库的网格构建吞吐与 OpenMP 加速比。

只测"建 map"这一段 (不含 PNG 读写), 因为它才是几何成本所在,
也才是 OpenMP 并行的那部分。这样测出来的数字才能说明并行有没有用。

用法:
    python tools/bench_maps.py --bin cpp/build/tests/bench_maps
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys


def run(bin_path, threads, width, height, repeat):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    out = subprocess.run([bin_path, "--width", str(width), "--height", str(height),
                          "--repeat", str(repeat)],
                         env=env, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"bench 失败: {out.stderr.strip()}")
    m = re.search(r"ms_per_map=([0-9.]+)", out.stdout)
    if not m:
        sys.exit(f"无法解析输出: {out.stdout!r}")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="cpp/build/tests/bench_maps")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()

    print(f"map {args.width}x{args.height}, {args.repeat} 次取平均")
    print(f"{'threads':>8} {'ms/map':>10} {'maps/s':>10} {'speedup':>8}")
    base = None
    for t in args.threads:
        ms = run(args.bin, t, args.width, args.height, args.repeat)
        if base is None:
            base = ms
        print(f"{t:>8} {ms:>10.2f} {1000.0 / ms:>10.1f} {base / ms:>8.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
