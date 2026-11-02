"""
增强型多项式鱼眼 (e-ftheta):

    [u - cx]   [c d] [x_norm]
    [v - cy] = [e 1] [y_norm] * rho(theta),   rho = k1 t + k2 t^2 + ...

d 项是"斜切"(decentering/skew)，用行列式 det = c - d*e 求逆。旧实现的反解
公式经核算正确（报告第四节确认），这里保留其代数形式，只补两件事：
  * det 接近 0 时不再悄悄用 1.0 兜底（旧代码 `det = det if abs(det)>1e-8 else 1.0`），
    而是构造期直接报错，因为那样的标定本来就不可逆。
  * 求根与 4thpoly 共用 solve_poly_theta（同一套导数保护与残差回验）。
"""

from __future__ import annotations

import numpy as np

from .base import (CameraModel, UnprojectionResult, _ray_from_theta, _safe_ratio,
                   inside_image)
from .radial_poly import solve_poly_theta

__all__ = ["EnhancedFThetaModel"]


class EnhancedFThetaModel(CameraModel):
    model_name = "eftheta"

    def __init__(self, width, height, k, c=1.0, d=0.0, e=0.0, cx=None, cy=None):
        self.width = int(width)
        self.height = int(height)
        self.k = [float(x) for x in np.atleast_1d(k)]
        self.c, self.d, self.e = float(c), float(d), float(e)
        self.det = self.c - self.d * self.e
        if abs(self.det) < 1e-8:
            raise ValueError(
                f"eftheta: 仿射矩阵行列式 det=c-d*e={self.det:.3e} 接近 0，不可逆。"
                "旧实现会用 det=1.0 静默替代并给出错误像素，这里改为构造期失败。")
        if len(self.k) < 1 or abs(self.k[0]) < 1e-12:
            raise ValueError("eftheta: k1 不能为 0")
        self.cx = float(self.width / 2.0 if cx is None else cx)
        self.cy = float(self.height / 2.0 if cy is None else cy)

    def _rho(self, theta):
        acc = np.zeros_like(theta)
        for coef in reversed(self.k):
            acc = acc * theta + coef
        return acc * theta

    def unproject(self, u, v) -> UnprojectionResult:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        u_s = u - self.cx
        v_s = v - self.cy
        # 仿射逆: [x_aff, y_aff] = A^-1 [u_s, v_s]，A = [[c, d], [e, 1]]
        x_aff = (u_s - self.d * v_s) / self.det
        y_aff = (-self.e * u_s + self.c * v_s) / self.det
        rho_d = np.hypot(x_aff, y_aff)
        theta, ok = solve_poly_theta(rho_d, self.k)
        x_norm, mask = _safe_ratio(x_aff, rho_d)
        y_norm, _ = _safe_ratio(y_aff, rho_d)
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
        u = np.where(mask, self.cx + (self.c * x_norm + self.d * y_norm) * rho, self.cx)
        v = np.where(mask, self.cy + (self.e * x_norm + 1.0 * y_norm) * rho, self.cy)
        # 只按图像边界判有效, 不额外限制在 90 度以内:
        # 鱼眼的 rho(theta) 在 [0, pi] 上单调, 图像半径可达 700+ px 对应
        # theta 超过 90 度 (后半球), 那些像素是**合法**的。加上 "theta<=90"
        # 会把它们全部错杀 —— 这正是报告问题 5 批评的常数截断.
        # 用共享的 inside_image: 边界带舍入容差, 否则 (W-1, h) 这类点会被误判.
        ok = inside_image(u, v, self.width, self.height)
        u = np.where(ok, u, -1.0)
        v = np.where(ok, v, -1.0)
        return u, v

    def describe(self):
        return dict(model="eftheta", wh=(self.width, self.height),
                    k=tuple(self.k), affine=(self.c, self.d, self.e),
                    c=(round(self.cx, 3), round(self.cy, 3)))
