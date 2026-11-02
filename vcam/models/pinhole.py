"""
针孔 (pinhole) 模型，含 Brown-Conrady 径向 + 切向畸变:

    x_d = x_u * (1 + k1 r^2 + k2 r^4 + k3 r^6 + k4 r^8) + 2 p1 x_u y_u + p2 (r^2 + 2 x_u^2)
    y_d = y_u * (1 + k1 r^2 + k2 r^4 + k3 r^6 + k4 r^8) + p1 (r^2 + 2 y_u^2) + 2 p2 x_u y_u
    u   = cx + fx * x_d,   v = cy + fy * y_d

反解用固定点迭代 + **残差回验**：旧实现在 20 次迭代后直接采纳当前值，强桶形
畸变下不收敛也照用（报告问题 4）。这里迭代完再把结果正投影一次，残差超过
收敛阈值的像素进带 converged=False，由上层置无效。
"""

from __future__ import annotations

import numpy as np

from .base import CameraModel, UnprojectionResult, _unit_ray, inside_image

__all__ = ["PinholeModel"]


class PinholeModel(CameraModel):
    model_name = "pinhole"

    def __init__(self, width, height, fx, fy=None, cx=None, cy=None,
                 k1=0.0, k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0,
                 max_iter=30, converge_px=1e-4):
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"pinhole: 图像尺寸非法 {width}x{height}")
        self.fx = float(fx)
        self.fy = float(fx if fy is None else fy)
        if self.fx == 0.0 or self.fy == 0.0:
            raise ValueError("pinhole: 焦距不能为 0")
        self.cx = float(self.width / 2.0 if cx is None else cx)
        self.cy = float(self.height / 2.0 if cy is None else cy)
        (self.k1, self.k2, self.k3, self.k4,
         self.p1, self.p2) = (float(k1), float(k2), float(k3), float(k4),
                              float(p1), float(p2))
        self.max_iter = int(max_iter)
        self.converge_px = float(converge_px)
        self._has_dist = any(abs(k) > 1e-12 for k in
                             (self.k1, self.k2, self.k3, self.k4, self.p1, self.p2))

    # ---- 内部 ---------------------------------------------------------------
    def _distort(self, xu, yu):
        r2 = xu * xu + yu * yu
        radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * (self.k3 + r2 * self.k4)))
        xd = xu * radial + 2.0 * self.p1 * xu * yu + self.p2 * (r2 + 2.0 * xu * xu)
        yd = yu * radial + self.p1 * (r2 + 2.0 * yu * yu) + 2.0 * self.p2 * xu * yu
        return xd, yd, r2

    def _undistort(self, xd, yd):
        """固定点迭代求 x_u，返回 (x_u, y_u, 残差像素 (H,W), 收敛标记)。"""
        if not self._has_dist:
            return xd, yd, np.zeros_like(xd), np.ones(xd.shape, dtype=bool)

        xu, yu = xd.copy(), yd.copy()
        for _ in range(self.max_iter):
            dx, dy, _ = self._distort(xu, yu)
            ex, ey = xd - dx, yd - dy
            xu = xu + ex
            yu = yu + ey
            if np.max(np.abs(ex)) * max(self.fx, self.fy) < self.converge_px and \
                    np.max(np.abs(ey)) * max(self.fx, self.fy) < self.converge_px:
                break
        else:
            dx, dy, _ = self._distort(xu, yu)
            res = np.hypot(xd - dx, yd - dy) * max(self.fx, self.fy)
            ok = res < max(self.converge_px, 1e-3)
            return xu, yu, res, ok
        dx, dy, _ = self._distort(xu, yu)
        res = np.hypot(xd - dx, yd - dy) * max(self.fx, self.fy)
        return xu, yu, res, np.ones(xd.shape, dtype=bool)

    # ---- 接口 ---------------------------------------------------------------
    def unproject(self, u, v) -> UnprojectionResult:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        xd = (u - self.cx) / self.fx
        yd = (v - self.cy) / self.fy
        xu, yu, _res, ok = self._undistort(xd, yd)
        ray = _unit_ray(xu, yu, np.ones_like(xu))
        theta = np.degrees(np.arctan2(np.hypot(xu, yu), 1.0))
        return UnprojectionResult(ray, theta, ok)

    def project(self, X, Y, Z):
        """
        相机系向量 -> 像素。**落在图像外与后半球一律返回 -1**，与鱼眼模型一致。

        为什么在模型层做边界判定：Convertor 判"有效"只看 map>=0。若 pinhole
        把 85° 入射角投到 u=115443（图像宽 2287），rect(mX=115443) 仍是个
        合法正数，remap 会去读一张不存在的像素；cv2 的 BORDER_CONSTANT 恰好
        兜住了它，于是"能跑、图也对"，但地图里的数已经没意义。把判定放在这里，
        调用方拿到 -1 就知道该像素不可用。
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        valid = Z > 1e-9
        Zs = np.where(valid, Z, 1.0)
        xd, yd, _r2 = self._distort(X / Zs, Y / Zs)
        u = self.cx + self.fx * xd
        v = self.cy + self.fy * yd
        ok = valid & inside_image(u, v, self.width, self.height)
        return np.where(ok, u, -1.0), np.where(ok, v, -1.0)

    def max_fov_deg(self):
        # 不截断: tan(theta) 在 90° 前单调，超出图像范围的自然由 u,v<0 或 >=W 判定。
        # 全视场上限由**图像边界**决定，而不是一个 r2<50 的常数。
        return 2.0 * np.degrees(np.arctan(
            0.5 * max(self.width / self.fx, self.height / self.fy)))

    def project_in_image(self, X, Y, Z):
        u, v = self.project(X, Y, Z)
        inside = (u >= 0) & (u <= self.width - 1.0) & (v >= 0) & (v <= self.height - 1.0)
        return u, v, inside

    def describe(self):
        return dict(model="pinhole", wh=(self.width, self.height),
                    f=(round(self.fx, 4), round(self.fy, 4)),
                    c=(round(self.cx, 4), round(self.cy, 4)),
                    dist=dict(k1=self.k1, k2=self.k2, k3=self.k3, k4=self.k4,
                              p1=self.p1, p2=self.p2))
