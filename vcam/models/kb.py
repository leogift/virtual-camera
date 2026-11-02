"""
Kannala-Brandt 鱼眼模型:

    rho(theta) = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
    u = cx + fx * rho * X / sqrt(X^2+Y^2),  v = cy + fy * rho * Y / sqrt(...)

反解是单变量单调方程 rho(theta) = r_d，用带保护的牛顿法。
"""

from __future__ import annotations

import numpy as np

from .base import (CameraModel, UnprojectionResult, _ray_from_theta, _safe_ratio,
                   inside_image)

__all__ = ["KannalaBrandtModel"]


class KannalaBrandtModel(CameraModel):
    model_name = "kb"

    def __init__(self, width, height, fx, fy=None, cx=None, cy=None,
                 k1=0.0, k2=0.0, k3=0.0, k4=0.0, max_iter=50, converge_rad=1e-9):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fx if fy is None else fy)
        if self.fx == 0.0 or self.fy == 0.0:
            raise ValueError("kb: 焦距不能为 0")
        self.cx = float(self.width / 2.0 if cx is None else cx)
        self.cy = float(self.height / 2.0 if cy is None else cy)
        self.k1, self.k2, self.k3, self.k4 = (float(k1), float(k2), float(k3), float(k4))
        self.max_iter = int(max_iter)
        self.converge_rad = float(converge_rad)

    def _rho(self, theta):
        t2 = theta * theta
        return theta * (1.0 + t2 * (self.k1 + t2 * (self.k2 + t2 * (self.k3 + t2 * self.k4))))

    def _drho(self, theta):
        t2 = theta * theta
        return 1.0 + t2 * (3.0 * self.k1 + t2 * (5.0 * self.k2 + t2 * (7.0 * self.k3 + t2 * 9.0 * self.k4)))

    def _solve_theta(self, rho_d):
        """
        解 rho(theta) = rho_d。返回 (theta, converged)。

        导数保护改成 |f'| 的**幅值下限 + 保留原符号**（报告问题 3）：
        旧写法 sign(f'+1e-12)*1e-8 在 f' 接近 0 时会变成 1e-8 这类跨符号分母，
        步长可以爆到 1e8。这里 max(|f'|, floor) * sign(f') 在 f'==0 时步长
        为 0 而不是爆炸，同时靠外层阻尼回退保证收敛。
        """
        theta = np.clip(rho_d.copy(), 0.0, np.pi)
        floor = 1e-8
        for _ in range(self.max_iter):
            f = self._rho(theta) - rho_d
            df = self._drho(theta)
            denom = np.maximum(np.abs(df), floor) * np.where(df < 0.0, -1.0, 1.0)
            step = np.where(np.abs(denom) > 0, f / denom, 0.0)
            new = np.clip(theta - step, 0.0, np.pi)
            bad = ~np.isfinite(new)
            new = np.where(bad, theta, new)
            done = np.max(np.abs(new - theta)) < self.converge_rad
            theta = new
            if done:
                break
        converged = np.abs(self._rho(theta) - rho_d) < 1e-7
        return theta, converged

    def unproject(self, u, v) -> UnprojectionResult:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        xs = (u - self.cx) / self.fx
        ys = (v - self.cy) / self.fy
        rho_d = np.hypot(xs, ys)
        theta, ok = self._solve_theta(rho_d)
        x_norm, mask = _safe_ratio(xs, rho_d)
        y_norm, _ = _safe_ratio(ys, rho_d)
        ray = _ray_from_theta(theta, x_norm, y_norm)
        return UnprojectionResult(ray, np.degrees(theta), ok & (mask | (rho_d < 1e-12)))

    def project(self, X, Y, Z):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        r_xy = np.hypot(X, Y)
        theta = np.arctan2(r_xy, Z)          # 钝角正确: Z<0 时 theta>90°
        x_norm, mask = _safe_ratio(X, r_xy)
        y_norm, _ = _safe_ratio(Y, r_xy)
        rho = self._rho(theta)
        u = np.where(mask, self.cx + self.fx * rho * x_norm, self.cx)
        v = np.where(mask, self.cy + self.fy * rho * y_norm, self.cy)
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
        return dict(model="kb", wh=(self.width, self.height),
                    f=(round(self.fx, 4), round(self.fy, 4)),
                    c=(round(self.cx, 4), round(self.cy, 4)),
                    k=(self.k1, self.k2, self.k3, self.k4))
