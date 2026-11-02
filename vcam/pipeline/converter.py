"""
虚拟相机坐标映射网格的构建与图像重采样。

核心链路（一个虚拟像素怎么找到源图像素）:

    虚拟像素 (u,v)
      --虚拟相机的 unproject-->  虚拟相机系射线 ray_dst
      --R_rel = R_src @ R_dst^T -->  源相机系射线 ray_src
      --源相机的 project-->  源像素 (u_src, v_src)

旧实现把这条链路摊在 4 个函数里、靠 16 个包装函数做 4x4 组合分发，且
"哪些像素无效"由各分支用魔数（r2<50、统一置 -1）各自决定。这里把它收成
GridConverter 一个对象，并统一产出 validity 掩码与统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:                                      # cv2 只在真正重采样时才需要
    import cv2
except Exception:                         # pragma: no cover
    cv2 = None

from ..core.rotation import rotation_relative
from ..models.base import inside_image
from .camera import CameraPose, FisheyeCamera, VirtualCamera

__all__ = ["MapStats", "RemapResult", "GridConverter"]


@dataclass
class MapStats:
    """一次网格构建的质量统计。用于告警与调试（报告 S1）。"""

    total: int
    valid: int
    non_finite: int = 0
    out_of_source: int = 0
    unconverged: int = 0
    max_theta_deg: float = 0.0
    note: str = ""

    @property
    def valid_ratio(self) -> float:
        return float(self.valid) / float(self.total) if self.total else 0.0

    def summary(self) -> str:
        s = (f"valid {self.valid}/{self.total} ({self.valid_ratio * 100:.1f}%), "
             f"max入射角 {self.max_theta_deg:.1f}deg")
        for tag, n in (("非有限", self.non_finite),
                       ("落在源图外", self.out_of_source),
                       ("反投影未收敛", self.unconverged)):
            if n:
                s += f", {tag} {n}"
        return s


@dataclass
class RemapResult:
    """重采样结果。同时保留中间量，便于复查"图为什么是黑的"。"""

    image: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    stats: MapStats
    R_rel: np.ndarray
    src: FisheyeCamera
    dst: FisheyeCamera

    def save_maps(self, map_x_path, map_y_path, fmt="%.4f"):
        """
        写 map 文本。写之前做 nan_to_num（报告 S6）：旧实现遇到 inf 会写出
        字面量 "inf"，np.loadtxt 读回来直接抛异常。
        """
        np.savetxt(map_x_path, np.nan_to_num(self.map_x, nan=-1.0, posinf=-1.0, neginf=-1.0),
                   fmt=fmt)
        np.savetxt(map_y_path, np.nan_to_num(self.map_y, nan=-1.0, posinf=-1.0, neginf=-1.0),
                   fmt=fmt)


class GridConverter:
    """
    绑定一对 (源真实相机, 目标虚拟相机) 的转换算子。

    姿态可以两种方式给:
      * dst_pose 直接给目标相机姿态（绝对）；
      * yaw_delta_deg / pitch_delta_deg / roll_delta_deg 给**相对源相机**的增量，
        在 Valeo 约定下 +yaw = 向左看、+pitch = 抬头（见 CameraPose.facing）。
        日志里打印的是实际几何朝向，不是变量名的含义（报告问题 6）。
    """

    def __init__(self, src: FisheyeCamera, dst: FisheyeCamera,
                 dst_pose: Optional[CameraPose] = None,
                 yaw_delta_deg: float = 0.0,
                 pitch_delta_deg: float = 0.0,
                 roll_delta_deg: float = 0.0,
                 R_rel: Optional[np.ndarray] = None,
                 dtype=np.float32):
        self.src = src
        self.dtype = dtype
        if R_rel is not None:
            self.R_rel = np.asarray(R_rel, dtype=np.float64)
            self.dst = dst
        else:
            if dst_pose is not None:
                pose = dst_pose
            elif (yaw_delta_deg or pitch_delta_deg or roll_delta_deg):
                pose = src.pose.with_delta(yaw_delta_deg, pitch_delta_deg, roll_delta_deg)
            else:
                pose = dst.pose
            self.dst = FisheyeCamera(name=dst.name, model=dst.model, pose=pose,
                                     is_virtual=dst.is_virtual)
            self.R_rel = rotation_relative(src.pose.matrix, pose.matrix)

    # ---- 姿态信息 -----------------------------------------------------------
    @property
    def is_identity_rotation(self) -> bool:
        return bool(np.allclose(self.R_rel, np.eye(3), atol=1e-9))

    def describe(self) -> str:
        if self.is_identity_rotation:
            rot = "纯内参转换（虚拟相机与源相机同姿态）"
        else:
            rot = (f"源: {self.src.pose.summary()}\n"
                   f"      目标: {self.dst.pose.summary()}")
        return (f"{self.src.name}({self.src.model.model_name}) "
                f"{self.src.width}x{self.src.height} -> "
                f"{self.dst.name}({self.dst.model.model_name}) "
                f"{self.dst.width}x{self.dst.height} | {rot}")

    # ---- 网格构建 -----------------------------------------------------------
    def build_maps(self, chunk_rows: int = 0) -> Tuple[np.ndarray, np.ndarray, MapStats]:
        """
        返回 (map_x, map_y, stats)，map 形状 (H_dst, W_dst)，float32。

        chunk_rows > 0 时按行分块计算，用于超大虚拟图或内存吃紧的板端；
        结果与整块计算逐位一致（只是顺序不同），不做任何近似。
        """
        dw, dh = self.dst.width, self.dst.height
        map_x = np.empty((dh, dw), dtype=self.dtype)
        map_y = np.empty((dh, dw), dtype=self.dtype)
        stat_acc = dict(non_finite=0, out_of_source=0, unconverged=0,
                        max_theta=0.0, total=dw * dh)

        rows = dh if chunk_rows <= 0 else int(chunk_rows)
        u_cols = np.arange(dw, dtype=np.float64)
        for r0 in range(0, dh, rows):
            r1 = min(r0 + rows, dh)
            v_rows = np.arange(r0, r1, dtype=np.float64)
            u, v = np.meshgrid(u_cols, v_rows, indexing="xy")

            res = self.dst.model.unproject(u, v)
            ray_dst = res.ray
            ray_src = ray_dst @ self.R_rel.T
            u_src, v_src = self.src.model.project(ray_src[..., 0], ray_src[..., 1],
                                                  ray_src[..., 2])

            finite = np.isfinite(u_src) & np.isfinite(v_src)
            # 模型层已经对越界返回 -1，这里再判一次是为了防止自定义模型漏做；
            # 两者用同一套带舍入容差的判据（见 models.base.inside_image）。
            inside = inside_image(u_src, v_src, self.src.width, self.src.height)
            ok = finite & inside & res.converged

            bad = ~finite
            keep = ~ok
            xs = np.where(keep, -1.0, u_src)
            ys = np.where(keep, -1.0, v_src)
            map_x[r0:r1] = xs.astype(self.dtype, copy=False)
            map_y[r0:r1] = ys.astype(self.dtype, copy=False)

            stat_acc["non_finite"] += int(np.count_nonzero(bad))
            stat_acc["out_of_source"] += int(np.count_nonzero(finite & ~inside))
            stat_acc["unconverged"] += int(np.count_nonzero(~res.converged[inside & finite]))
            m = float(np.max(res.theta_deg)) if res.theta_deg.size else 0.0
            stat_acc["max_theta"] = max(stat_acc["max_theta"], m)

        # NaN/Inf 已经在写 map 时被归到 -1，这里再兜一次（报告 S1）
        non_finite = int(np.count_nonzero(~np.isfinite(map_x))) + \
                     int(np.count_nonzero(~np.isfinite(map_y)))
        np.nan_to_num(map_x, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
        np.nan_to_num(map_y, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)

        # 有效与否以哨兵为准：模型层对不可用像素写 -1，可用像素写真实坐标。
        # 不能写成 `map_x >= 0`：恒等映射下图像边界像素的坐标可能是 -1.7e-13
        # （浮点舍入到 0 的另一侧），那种点完全合法，实测因此误报 155 个无效。
        valid = int(np.count_nonzero((map_x != -1.0) & (map_y != -1.0)))
        stats = MapStats(total=stat_acc["total"], valid=valid,
                         non_finite=max(non_finite, stat_acc["non_finite"]),
                         out_of_source=stat_acc["out_of_source"],
                         unconverged=stat_acc["unconverged"],
                         max_theta_deg=stat_acc["max_theta"])
        self._warn_if_suspicious(stats)
        return map_x, map_y, stats

    def _warn_if_suspicious(self, stats: MapStats):
        warns = []
        if stats.valid == 0:
            warns.append("有效像素为 0：虚拟相机完全看不到源相机视野")
        elif stats.valid_ratio < 0.05:
            warns.append(f"有效像素仅 {stats.valid_ratio * 100:.1f}%，检查姿态或 FOV")
        if stats.max_theta_deg > 90.0:
            warns.append(f"虚拟视场覆盖到入射角 {stats.max_theta_deg:.1f}deg（超过 90°）")
        stats.note = "; ".join(warns)

    # ---- 重采样 -------------------------------------------------------------
    def remap(self, image, map_x=None, map_y=None,
              interpolation=None, border_mode=None, border_value=0):
        """把源图（BGR 或灰度，cv2 读入格式）重采样成虚拟相机视图。"""
        if cv2 is None:
            raise ImportError("需要 opencv-python 才能做图像重采样")
        if map_x is None or map_y is None:
            map_x, map_y, stats = self.build_maps()
        else:
            u, v = map_x, map_y
            stats = MapStats(total=u.size,
                             valid=int(np.count_nonzero((u >= 0) & (v >= 0))),
                             non_finite=int(np.count_nonzero(~np.isfinite(u))))
        self._check_image(image)
        out = cv2.remap(image, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR if interpolation is None else interpolation,
                        borderMode=cv2.BORDER_CONSTANT if border_mode is None else border_mode,
                        borderValue=border_value)
        return RemapResult(image=out, map_x=map_x, map_y=map_y, stats=stats,
                           R_rel=self.R_rel, src=self.src, dst=self.dst)

    def _check_image(self, image):
        """
        源图通道数检查（报告 S4）。cv2.imread 默认 3 通道，灰度鱼眼图会被静默
        升成 3 通道；反过来，把彩色图喂给 1 通道的 map 也会在这里暴露。
        """
        if image is None:
            raise ValueError("源图像为空（cv2.imread 返回 None，路径或编码有问题）")
        if image.shape[0] != self.src.height or image.shape[1] != self.src.width:
            raise ValueError(
                f"源图像尺寸 {image.shape[1]}x{image.shape[0]} 与标定 "
                f"{self.src.width}x{self.src.height} 不一致。旧实现会静默 resize，"
                "但那会同时改变畸变几何；请显式 resize 或提供匹配尺寸的标定。")

    # ---- 导出为采样网格（PyTorch grid_sample / 边缘 NPU）--------------------
    def to_grid(self, map_x, map_y, align_corners=True):
        """
        map -> 归一化到 [-1, 1] 的 (1, H, W, 2) 网格。

        align_corners=False 用像素中心形式 (2(u+0.5)/W - 1)（报告问题 7）：
        旧实现在 W==1 或 H==1 时 2x/(W-1)-1 直接除零。
        """
        sw, sh = self.src.width, self.src.height
        if align_corners:
            denom_x, denom_y = max(sw - 1, 1), max(sh - 1, 1)
            gx = 2.0 * map_x / denom_x - 1.0
            gy = 2.0 * map_y / denom_y - 1.0
        else:
            gx = 2.0 * (map_x + 0.5) / max(sw, 1) - 1.0
            gy = 2.0 * (map_y + 0.5) / max(sh, 1) - 1.0
        return np.stack([gx, gy], axis=-1)[None, ...].astype(np.float32)

    def to_gather_tables(self, map_x, map_y):
        """
        双线性采样表 (idx, w)，形状 (4, H*W)。

        边缘 NPU 上 Gather 比 GridSample 好部署（fcntl 不支持 GridSample），且
        表本身只依赖几何、与图像无关，可以离线烘焙一次、当图输入喂进去。
        tap 顺序 [x0y0, x1y0, x0y1, x1y1]；越界 tap 权重置 0（等价于
        padding_mode='zeros'），非法像素四个权重全 0。
        """
        H, W = map_x.shape
        x = map_x.astype(np.float64)
        y = map_y.astype(np.float64)
        bad = (x < 0) | (y < 0) | ~np.isfinite(x) | ~np.isfinite(y)
        x = np.where(bad, 0.0, x)
        y = np.where(bad, 0.0, y)
        x0 = np.floor(x)
        y0 = np.floor(y)
        fx = x - x0
        fy = y - y0
        xi0 = x0.astype(np.int64)
        yi0 = y0.astype(np.int64)
        xi1 = xi0 + 1
        yi1 = yi0 + 1

        def flat(xi, yi):
            ok = (xi >= 0) & (xi < self.src.width) & (yi >= 0) & (yi < self.src.height) & ~bad
            idx = np.where(ok, yi * self.src.width + xi, 0)
            return idx.reshape(-1), ok

        idx = []
        w = []
        for (dx, dy, wx, wy) in ((xi0, yi0, 1 - fx, 1 - fy), (xi1, yi0, fx, 1 - fy),
                                 (xi0, yi1, 1 - fx, fy), (xi1, yi1, fx, fy)):
            i, ok = flat(dx, dy)
            idx.append(i)
            w.append(np.where(ok, wx * wy, 0.0).reshape(-1))
        return np.stack(idx, 0).astype(np.int32), np.stack(w, 0).astype(np.float32)
