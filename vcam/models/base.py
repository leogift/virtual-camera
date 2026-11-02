"""
相机模型公共基类。

设计要点：
  * 一个模型 = 一个类，构造时校验参数，构造后只提供 project / unproject 两个
    纯函数，不读配置、不碰图像。
  * 反投影 (unproject) 的签名固定为
        unproject(u, v) -> (ray, angle_deg, converged)
    返回的三个量都是 (H, W) 的 ndarray。angle_deg 是入射角 theta（单位度），
    converged 是数值求根的收敛标记 —— 旧代码在强畸变下收敛失败时会**静默**
    返回未收敛值（报告问题 4），这里把它显式带出来，由上层决定是否置无效。
  * 像素坐标约定：整数索引即像素中心，(0,0) 是左上角像素第一行第一列的
    中心。与 OpenCV、与 WoodScape readme 的 "-0.5" 写法一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = ["CameraModel", "ProjectionResult", "UnprojectionResult"]


class UnprojectionResult(tuple):
    """(ray, theta_deg, converged)，同时可按字段访问。"""

    __slots__ = ()

    def __new__(cls, ray, theta_deg, converged):
        return super().__new__(cls, (ray, theta_deg, converged))

    @property
    def ray(self):
        return self[0]

    @property
    def theta_deg(self):
        return self[1]

    @property
    def converged(self):
        return self[2]


class CameraModel(ABC):
    """相机模型基类：图像像素 <-> 相机系视线。"""

    #: 注册名（小写、无连字符），见 vcam.models.registry
    model_name: str = "base"

    # ---- 必备接口 -----------------------------------------------------------
    @abstractmethod
    def unproject(self, u, v) -> UnprojectionResult:
        """像素 -> 相机系单位射线 (H,W,3) / 入射角 (H,W) [deg] / 收敛标记 (H,W)。"""

    @abstractmethod
    def project(self, X, Y, Z) -> Tuple[np.ndarray, np.ndarray]:
        """相机系三维向量 (H,W,3 分量) -> 像素坐标 (u, v)，禁用位置填 -1。"""

    # ---- 公共能力 -----------------------------------------------------------
    def max_fov_deg(self) -> Optional[float]:
        """
        该模型可表达的**最大全视场角**（度）。pinhole 有限，鱼眼类为 None（不受限）。

        用途：校验虚拟相机 FOV 是否超出目标模型能力。旧代码用 r2 < 50 这种
        魔数截断（报告问题 5），等价于 164° 的 pinhole 全视场却又没有说明。
        """
        return None

    def is_visible(self, X, Y, Z):
        """该模型下哪些方向能成像（pinhole: Z>0；鱼眼: 全半球以上）。"""
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        return Z > 1e-9

    def describe(self) -> Dict:
        return {"model": self.model_name}

    def __repr__(self) -> str:
        d = self.describe()
        kvs = ", ".join(f"{k}={v}" for k, v in d.items() if k != "model")
        return f"{type(self).__name__}({kvs})"


# ---- 数值工具，供各模型共用 -------------------------------------------------

def _safe_ratio(numer, denom, eps=1e-12):
    """denom 近零处返回 0（径向方向的提取只在下游乘 sin(theta) 时才有意义）。"""
    out = np.zeros_like(np.asarray(numer, dtype=np.float64))
    mask = np.abs(denom) > eps
    out[mask] = numer[mask] / denom[mask]
    return out, mask


def _ray_from_theta(theta, x_norm, y_norm):
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    return np.stack([sin_t * x_norm, sin_t * y_norm, cos_t], axis=-1)


def _unit_ray(X, Y, Z):
    norm = np.sqrt(X * X + Y * Y + Z * Z)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return np.stack([X / norm, Y / norm, Z / norm], axis=-1)


def _visible(Z):
    """
    鱼眼只能看到前半球。判据用 Z 的符号本身，不要用 `theta < pi/2 + 1e-9`
    这类浮点容差：theta 恰在 90° 附近时，随机生成的方向会得到 Z=-7e-18，
    被误判为不可见而直接返回 -1（实测 2000 个方向里错杀 107 个）。
    90° 边界本身属于可见（rho 有限），所以取 Z >= 0。
    """
    return Z >= 0.0


#: 越界判定容差（像素）。必须留这一丁点余量：图像右下边界的整数坐标在
#: 浮点里可能变成 639.0000000000001，用它判越界会把"恰好落在最后一个像素
#: 中心"的点判无效。实测恒等映射下 (639, 0) 因此变成 -1，整张 map 出现
#: 640 px 的偏差。容差只吸收舍入，不吸收真实越界。
EDGE_EPS = 1e-6


def inside_image(u, v, width, height):
    """像素是否落在图像范围内（含边界，带 EDGE_EPS 舍入容差）。"""
    return ((u >= -EDGE_EPS) & (u <= width - 1.0 + EDGE_EPS) &
            (v >= -EDGE_EPS) & (v <= height - 1.0 + EDGE_EPS))


def _theta_from_ray(X, Y, Z):
    """入射角 theta = atan2(sqrt(X^2+Y^2), Z)，超过 90° 时也能正确返回钝角。"""
    return np.arctan2(np.sqrt(X * X + Y * Y), Z)
