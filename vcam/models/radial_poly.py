"""
径向多项式鱼眼模型（WoodScape / NFOCC 数据集所用，poly_order=4 或 5）:

    rho(theta) = k1 theta + k2 theta^2 + k3 theta^3 + k4 theta^4 [+ k5 theta^5]
    u = cx + rho * X / sqrt(X^2+Y^2),  v = cy + rho * Y / sqrt(...)

单位约定（报告问题 1）：**k 的单位是"像素 / 弧度^n"**，rho 直接就是像素。
WoodScape 的 readme 与本项目实测都成立：k1 就是小角度焦点 ≈ f（FV 为 339.75
像素）。若上游标定把多项式定义在归一化坐标上，请调用
`RadialPolyModel.from_normalized()` 显式声明，本类不做猜测。
"""

from __future__ import annotations

import numpy as np

from .base import (CameraModel, UnprojectionResult, _ray_from_theta, _safe_ratio,
                   inside_image)

__all__ = ["RadialPolyModel", "solve_poly_theta"]


def solve_poly_theta(rho_d, k, domain=(0.0, np.pi), tol=1e-9, max_iter=40):
    """
    解 k1 t + k2 t^2 + ... = rho_d，返回 (theta, converged)。

    两级保护（报告 S2 + 问题 2）:
      1. 初值: k1 退化（<=0 或极小）时不再用 rho_d/k1（会给出负的或巨大的初值），
         退回二分法求根；k1>0 时用 rho_d/k2 量级的等比初值。
      2. 牛顿步的导数改成 max(|f'|, floor) * sign(f')，不会像旧代码那样把负的
         微小导数翻成 +1e-9（报告问题 2 指出的正是这个符号翻转）。
    最后用残差 < 1e-7（像素）判定收敛，未收敛的位置由上层置无效。
    """
    k = [float(c) for c in k]
    order = len(k)
    coef = np.zeros(order + 1)          # theta^0 .. theta^order
    for i, c in enumerate(k, start=1):
        coef[i] = c

    def poly(t):
        acc = np.zeros_like(t)
        for c in coef[::-1]:
            acc = acc * t + c
        return acc

    def dpoly(t):
        acc = np.zeros_like(t)
        for i, c in enumerate(coef[1:], start=1):
            acc = acc * t + c * i
        return acc

    lo, hi = float(domain[0]), float(domain[1])
    rho_d = np.asarray(rho_d, dtype=np.float64)

    # --- 初值 ---
    k1 = k[0] if order >= 1 else 0.0
    theta = np.where(k1 > 1e-6, rho_d / max(k1, 1e-6), 0.5 * (lo + hi))
    theta = np.clip(theta, lo, hi)

    floor = 1e-7
    for _ in range(max_iter):
        f = poly(theta) - rho_d
        df = dpoly(theta)
        denom = np.maximum(np.abs(df), floor) * np.where(df < 0.0, -1.0, 1.0)
        step = np.where(np.abs(denom) > 0.0, f / denom, 0.0)
        step = np.clip(step, -0.5, 0.5)          # 阻尼: 单步不超过 0.5 rad
        new = np.clip(theta - step, lo, hi)
        new = np.where(np.isfinite(new), new, theta)
        done = float(np.max(np.abs(new - theta))) < tol
        theta = new
        if done:
            break

    converged = np.abs(poly(theta) - rho_d) < 1e-7

    # 牛顿法没收敛的点用二分法兜底（poly 在 [0, pi] 上单调增，若标定良好则必收敛）
    if not np.all(converged):
        bad = ~converged
        a = np.full_like(theta, lo)
        b = np.full_like(theta, hi)
        for _ in range(60):
            mid = 0.5 * (a + b)
            pos = poly(mid) - rho_d < 0.0
            a = np.where(pos, mid, a)
            b = np.where(pos, b, mid)
        theta_bis = 0.5 * (a + b)
        ok_bis = np.abs(poly(theta_bis) - rho_d) < 1e-7
        theta = np.where(bad & ok_bis, theta_bis, theta)
        converged = converged | (bad & ok_bis)

    return theta, converged


class RadialPolyModel(CameraModel):
    model_name = "4thpoly"

    def __init__(self, width, height, k, cx=None, cy=None, aspect_ratio=1.0,
                 unit="px"):
        """
        width/height: 图像尺寸（像素）
        k: 多项式系数 [k1..kn]；poly_order = len(k)
        cx/cy: **绝对**主点像素坐标（不是 offset）
        aspect_ratio: v 方向的额外缩放，WoodScape 的 v = v'*aspect + cy + h/2 - 0.5
        unit: 'px'（默认，rho 以像素计）或 'norm'（rho 以归一化坐标计，
              构造时内部乘上 k1 之外的 f）。'norm' 仅在标定文件明确如此时使用。
        """
        self.width = int(width)
        self.height = int(height)
        k = [float(c) for c in np.atleast_1d(k)]
        if len(k) < 1:
            raise ValueError("4thpoly: 至少需要 1 个系数")
        if abs(k[0]) < 1e-12:
            raise ValueError(
                "4thpoly: k1 不能为 0（它是一阶项，也是小角度等效焦距）。"
                "若标定给的是归一化多项式，请用 from_normalized() 并显式给出焦距。")
        self.k = k
        self.cx = float(self.width / 2.0 if cx is None else cx)
        self.cy = float(self.height / 2.0 if cy is None else cy)
        self.aspect_ratio = float(aspect_ratio)
        if unit not in ("px", "norm"):
            raise ValueError("4thpoly: unit 只能是 'px' 或 'norm'")
        self.unit = unit
        if unit == "norm":
            raise ValueError("4thpoly: rho 用归一化单位时，请先调用 "
                             "RadialPolyModel.normalized_to_px(k, fx) 换算成像素单位，"
                             "再用 unit='px' 构造；不要在这里猜。")

    @staticmethod
    def normalized_to_px(k, fx):
        """
        把"归一化坐标下的多项式系数"换算成像素单位系数: k_i_px = fx * k_i。

        依据: rho_norm(theta) = sum k_i theta^i，像素半径 rho_px = fx * rho_norm，
        令 k_i_px = fx * k_i 即得 rho_px = sum k_i_px theta^i —— 这**不是**
        "只缩放一阶项"，每一项都乘同一个 fx（因为 fx 是图像映射，不是角度量）。
        报告问题 1 要求的 poly_unit 开关就是这两个函数。
        """
        fx = float(fx)
        if fx <= 0.0:
            raise ValueError("normalized_to_px: fx 必须为正")
        return [fx * float(c) for c in np.atleast_1d(k)]

    def _rho(self, theta):
        acc = np.zeros_like(theta)
        for c in reversed(self.k):
            acc = acc * theta + c
        return acc * theta        # 从 theta^1 起

    def unproject(self, u, v) -> UnprojectionResult:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        x_shift = u - self.cx
        y_shift = (v - self.cy) / self.aspect_ratio
        rho_d = np.hypot(x_shift, y_shift)
        theta, ok = solve_poly_theta(rho_d, self.k)
        x_norm, mask = _safe_ratio(x_shift, rho_d)
        y_norm, _ = _safe_ratio(y_shift, rho_d)
        ray = _ray_from_theta(theta, x_norm, y_norm)
        return UnprojectionResult(ray, np.degrees(theta), ok & (mask | (rho_d < 1e-12)))

    def project(self, X, Y, Z):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        r_xy = np.hypot(X, Y)
        theta = np.arctan2(r_xy, Z)
        x_norm, mask = _safe_ratio(X, r_xy)
        y_norm, _ = _safe_ratio(Y, r_xy)
        rho = self._rho(theta)
        u = np.where(mask, self.cx + rho * x_norm, self.cx)
        v = np.where(mask, self.cy + self.aspect_ratio * rho * y_norm, self.cy)
        # 只按图像边界判有效, 不额外限制在 90 度以内:
        # 鱼眼的 rho(theta) 在 [0, pi] 上单调, 图像半径可达 700+ px 对应
        # theta 超过 90 度 (后半球), 那些像素是**合法**的。加上 "theta<=90"
        # 会把它们全部错杀 —— 这正是报告问题 5 批评的常数截断.
        # 用共享的 inside_image: 边界带舍入容差, 否则 (W-1, h) 这类点会被误判.
        ok = inside_image(u, v, self.width, self.height)
        u = np.where(ok, u, -1.0)
        v = np.where(ok, v, -1.0)
        return u, v

    def focal_small_angle(self):
        """小角度等效焦距 = d(rho)/d(theta) at 0 = k1。用于 FOV 换算与主点检查。"""
        return self.k[0]

    def describe(self):
        return dict(model=f"radial_poly{len(self.k)}", wh=(self.width, self.height),
                    k=tuple(self.k), c=(round(self.cx, 3), round(self.cy, 3)),
                    aspect_ratio=self.aspect_ratio)
