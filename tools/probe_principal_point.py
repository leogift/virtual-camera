"""探针: 判定 NFOCC/WoodScape 主点约定 (是否减 0.5)。

背景: WoodScape readme 的投影公式写成
    u = u' + cx_offset + width/2 - 0.5
若把 "-0.5" 理解成"主点本身要再减 0.5"，那 max(rho) 就应恰好等于
(width/2 + offx - 0.5) 与 (height/2 + offy - 0.5)；若只是说明"像素索引
(0,0) 是左上角像素中心"（即索引即中心），那 max(rho) 应等于不减 0.5 的值。

判据: 圆周方向的暗环最外边缘到主点的距离，在 360 个方位角上的极差。
主点取对时，最近的那些暗环应当近似同心 -> 各向同性。两个候选只差 0.5 px，
所以这个判据的灵敏度必须由"人为平移主点"的对照曲线给出，否则无法解释结果。
"""

import json
import sys
import argparse

import numpy as np
import cv2


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    i = d["intrinsic"]
    w, h = float(i["width"]), float(i["height"])
    k = [i["k1"], i["k2"], i["k3"], i["k4"]]
    cx = w / 2.0 + float(i["cx_offset"])
    cy = h / 2.0 + float(i["cy_offset"])
    return w, h, k, cx, cy


def ring_isotropy(gray, cx, cy, r_min=60.0, r_max=280.0, n_ang=360):
    """
    对每个方位角, 找出 [r_min, r_max] 内**最暗**的局部极小半径; 返回这些半径的
    最大最小差(极差)。同心 -> 极差小。
    """
    h, w = gray.shape
    rr = np.arange(r_min, r_max, 0.25, dtype=np.float32)
    angs = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    best = []
    for a in angs:
        xs = (cx + rr * np.cos(a)).astype(np.float32)
        ys = (cy + rr * np.sin(a)).astype(np.float32)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        prof = cv2.remap(gray, xs[None], ys[None], cv2.INTER_LINEAR)[0]
        # 平滑以抑制噪声, 再取最暗处
        prof = np.convolve(prof, np.ones(9) / 9.0, mode="same")
        best.append(rr[int(np.argmin(prof[9:-9])) + 9])
    best = np.array(best, dtype=np.float64)
    return float(np.percentile(best, 95) - np.percentile(best, 5)), float(np.median(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--shift-grid", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0])
    args = ap.parse_args()

    w, h, k, cx0, cy0 = load(args.calib)
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"cannot read {args.image}")
    img = np.ascontiguousarray(img.astype(np.float32))

    print(f"image {img.shape[1]}x{img.shape[0]}, calib {w:.0f}x{h:.0f}, k={k}")
    print("shift  center(u,v)          ring-radius spread(px)  median radius")
    results = {}
    for s in args.shift_grid:
        cx, cy = cx0 - s, cy0 - s
        spread, med = ring_isotropy(img, cx, cy)
        results[s] = spread
        print(f"{s:5.2f}  ({cx:8.3f},{cy:8.3f})  {spread:12.3f}      {med:8.2f}")

    s0, s05 = 0.0, 0.5
    if s0 in results and s05 in results:
        d = results[s05] - results[s0]
        print(f"\nshift0 minus shift0.5 spread = {d:+.3f} px "
              f"({'不减 0.5 更同心' if d < 0 else '减 0.5 更同心'})")


if __name__ == "__main__":
    main()
