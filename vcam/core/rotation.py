"""
旋转与欧拉角：车体/世界系 (ISO 8855: X 前, Y 左, Z 上) 与相机光学系
(X 右, Y 下, Z 前) 之间的姿态表达。

为什么单独一个文件：原 virtual_camera_core.py 把"构造"和"分解"分别写在两个
函数里，两者对同一约定的定义不一致且没有任何测试覆盖，于是 camera 约定的往返
静默错误（构造 yaw=30,pitch=20,roll=10 后分解得 pitch=11.82°，见 CHANGES.md
Bug 1）。这里把每种约定都实现为 RotationConvention，并强制提供
inverse()，由 test/test_rotation.py 做"构造->分解->构造"的往返断言。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

__all__ = [
    "RotationConvention",
    "ValeoConvention",
    "CameraConvention",
    "CONVENTIONS",
    "get_convention",
    "PRE_ROT_VALEO",
    "rotation_to_ypr",
    "ypr_to_rotation",
    "zxz_to_rotation",
    "rotation_relative",
    "optical_axis_in_world",
]


# Valeo / GCam 规范的零姿态预旋转: 相机 (右,下,前) <- 车体 (前,左,上)
#   X_cam = -Y_world, Y_cam = -Z_world, Z_cam = +X_world
PRE_ROT_VALEO = np.array([[0.0, -1.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [1.0, 0.0, 0.0]], dtype=np.float64)


class RotationConvention(ABC):
    """一种 YPR 欧拉角约定：定义 R(world->cam) 的构造方式及其严格逆。"""

    name: str = ""

    @abstractmethod
    def to_matrix(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        """(yaw, pitch, roll) [deg] -> R (3x3, world -> cam)。"""

    @abstractmethod
    def from_matrix(self, R: np.ndarray) -> Tuple[float, float, float]:
        """R (3x3, world -> cam) -> (yaw, pitch, roll) [deg]，是 to_matrix 的逆。"""

    def roundtrip_error(self, yaw_deg, pitch_deg, roll_deg) -> float:
        """构造->分解->再构造，返回两个旋转矩阵的最大绝对差。"""
        R0 = self.to_matrix(yaw_deg, pitch_deg, roll_deg)
        y, p, r = self.from_matrix(R0)
        R1 = self.to_matrix(y, p, r)
        return float(np.max(np.abs(R0 - R1)))

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<RotationConvention {self.name}>"


class ValeoConvention(RotationConvention):
    """
    R = Rz(roll) @ Rx(pitch) @ Ry(yaw) @ PRE_ROT_VALEO。

    注意这里的 Rz/Rx/Ry 是**右乘作用在相机系**上的转动（等价于车体系的
    Z-X-Y 内旋序列），因此与 YXZ 的 Rz Rx Ry 分解公式严格配对。
    符号含义（均经数值验证，见 test/test_rotation.py）：
      yaw   > 0 -> 光轴转向 +Y（车体左侧）
      pitch > 0 -> 光轴 Z 分量变正（抬头）
      roll  > 0 -> 图像顺时针
    """

    name = "valeo"

    def to_matrix(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> np.ndarray:
        yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
        Ry = np.array([[np.cos(yaw), 0.0, np.sin(yaw)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(yaw), 0.0, np.cos(yaw)]], dtype=np.float64)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(pitch), np.sin(pitch)],
                       [0.0, -np.sin(pitch), np.cos(pitch)]], dtype=np.float64)
        Rz = np.array([[np.cos(roll), np.sin(roll), 0.0],
                       [-np.sin(roll), np.cos(roll), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        return Rz @ Rx @ Ry @ PRE_ROT_VALEO

    def from_matrix(self, R) -> Tuple[float, float, float]:
        M = np.asarray(R, dtype=np.float64) @ PRE_ROT_VALEO.T
        sin_p = np.clip(-M[2, 1], -1.0, 1.0)
        pitch = np.arcsin(sin_p)
        if abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(-M[2, 0], M[2, 2])
            roll = np.arctan2(M[0, 1], M[1, 1])
        else:  # 万向锁: yaw 与 roll 共轴，固定 yaw=0
            yaw = 0.0
            roll = np.arctan2(-M[1, 0], M[0, 0])
        return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


class CameraConvention(RotationConvention):
    """
    相机本地 Tait-Bryan 欧拉角: yaw 绕 Y(下), pitch 绕 X(右), roll 绕 Z(前)。

    展开（直接手推，因为 Rz 会把 Rx 里的 sin(pitch) 的符号吃掉一次）:
        R = Rz Rx Ry =
        [[ c_r c_y - s_r s_p s_y, -s_r c_p,  c_r s_y + s_r s_p c_y],
         [ s_r c_y + c_r s_p s_y,  c_r c_p,  s_r s_y - c_r s_p c_y],
         [-c_p s_y,                s_p,      c_p c_y              ]]
    因此 pitch = asin(R[2,1])，yaw = atan2(-R[2,0], R[2,2])，
    roll = atan2(-R[0,1], R[1,1])。
    注意这不是"把符号反一下"的经验修补：旧实现用 R[1,2] 取 pitch，而
    R[1,2] = s_r s_y - c_r s_p c_y 是**叉乘项**，原本就不可能等于 sin(pitch)。

    与报告 Bug 1 的相对关系：报告把"构造 ZXY"与"分解 YXZ"判成不一致。真相是
    旧构造式与旧分解式**本来就是**
    同一种约定（YXZ 内旋），问题在于旧分解用 R[1,2]/R[1,0]/R[1,1]/R[0,2]/
    R[2,2] —— 那是**另一条**链 (Ry Rx Rz) 的公式，套在 Rz Rx Ry 上必然错。
    这就是旧代码 (30,20,10) -> (33.75,11.82,19.01) 的来源。
    """

    name = "camera"

    def to_matrix(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> np.ndarray:
        yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
        Ry = np.array([[np.cos(yaw), 0.0, np.sin(yaw)],
                       [0.0, 1.0, 0.0],
                       [-np.sin(yaw), 0.0, np.cos(yaw)]], dtype=np.float64)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(pitch), -np.sin(pitch)],
                       [0.0, np.sin(pitch), np.cos(pitch)]], dtype=np.float64)
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0.0],
                       [np.sin(roll), np.cos(roll), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        return Rz @ Rx @ Ry

    def from_matrix(self, R) -> Tuple[float, float, float]:
        R = np.asarray(R, dtype=np.float64)
        pitch = np.arcsin(np.clip(R[2, 1], -1.0, 1.0))
        if abs(np.cos(pitch)) > 1e-6:
            yaw = np.arctan2(-R[2, 0], R[2, 2])
            roll = np.arctan2(-R[0, 1], R[1, 1])
        else:  # 万向锁: yaw 与 roll 共轴，固定 yaw=0
            yaw = 0.0
            roll = np.arctan2(R[1, 0], R[0, 0])
        return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


CONVENTIONS = {"valeo": ValeoConvention(), "camera": CameraConvention()}
_ALIASES = {
    "valeo": "valeo", "gcam": "valeo", "proj1356": "valeo",
    "camera": "camera", "standard": "camera", "tait-bryan": "camera", "opencv": "camera",
}


def get_convention(name) -> RotationConvention:
    """按名字取约定对象；接受 ValeoConvention 实例原样返回。"""
    if isinstance(name, RotationConvention):
        return name
    key = str(name or "valeo").strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"未知的欧拉角约定: {name!r}，仅支持 'valeo' 或 'camera'（别名: "
            + ", ".join(sorted(k for k in _ALIASES if k not in ('valeo', 'camera'))) + "）")
    return CONVENTIONS[_ALIASES[key]]


# ---- 兼容旧 API 的模块级函数 ------------------------------------------------

def ypr_to_rotation(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, convention="valeo") -> np.ndarray:
    return get_convention(convention).to_matrix(yaw_deg, pitch_deg, roll_deg)


def rotation_to_ypr(R, convention="valeo") -> Tuple[float, float, float]:
    return get_convention(convention).from_matrix(R)


# ---- ZXZ（FLMC / Tuam / smart2 配置格式）------------------------------------

def zxz_to_rotation(rotz1_deg=0.0, rotx_deg=0.0, rotz2_deg=0.0) -> np.ndarray:
    """
    R = Rz(rotz2) @ Rx(rotx + 180°) @ Rz(rotz1)。

    这个 +180° 不是笔误：proj1356 的 rotationZXZToMat 把第三轴的 0 位定义为
    "光轴指向车体后方"（与 Valeo w = [1,0,0] 前视相差 180°），所以翻转必须
    显式写进 Rx。旧实现缺这条注释，报告 S5 也是指出它无从核对。
    """
    rz1 = np.radians(rotz1_deg)
    rx = np.radians(rotx_deg) + np.pi
    rz2 = np.radians(rotz2_deg)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(rx), np.sin(rx)],
                   [0.0, -np.sin(rx), np.cos(rx)]], dtype=np.float64)
    Rz1 = np.array([[np.cos(rz1), np.sin(rz1), 0.0],
                    [-np.sin(rz1), np.cos(rz1), 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
    Rz2 = np.array([[np.cos(rz2), -np.sin(rz2), 0.0],
                    [np.sin(rz2), np.cos(rz2), 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz2 @ Rx @ Rz1


# ---- 相对旋转 ---------------------------------------------------------------

def rotation_relative(R_src: np.ndarray, R_dst: np.ndarray) -> np.ndarray:
    """
    源相机系 <- 目标（虚拟）相机系 的相对旋转: v_src = R_rel @ v_dst。

    推导: v_world = R_dst^T v_dst，v_src = R_src v_world，故
    R_rel = R_src R_dst^T。R_src、R_dst 都必须是 world->cam 方向。
    """
    R_src = np.asarray(R_src, dtype=np.float64)
    R_dst = np.asarray(R_dst, dtype=np.float64)
    return R_src @ R_dst.T


def optical_axis_in_world(R) -> np.ndarray:
    """光轴 (cam +Z) 在车体系下的单位向量。用于打印"实际朝向"和写测试。"""
    R = np.asarray(R, dtype=np.float64)
    return R.T @ np.array([0.0, 0.0, 1.0])


# ---- 四元数（WoodScape / NFOCC readme: 相机->车体） -------------------------

def quaternion_to_rotation(q, scalar_first: bool = False) -> np.ndarray:
    """
    单位四元数 -> 3x3 旋转矩阵。

    scalar_first=False（默认）按 (x, y, z, w) 解释，与
    `scipy.spatial.transform.Rotation.from_quat` 一致；NFOCC 的标定 readme
    明确说用 from_quat，所以默认取这个顺序。
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"四元数必须是 4 个分量，实际 {q.size}")
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("四元数模长为 0")
    q = q / n
    if scalar_first:
        w, x, y, z = q
    else:
        x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quaternion_cam2veh_to_rt(quat, translation=None, scalar_first: bool = False):
    """
    把"相机->车体"的四元数/平移换算成本库要的 (R_world->cam, t)。

    标定读作 T_veh_cam: v_veh = R_c2v v_cam + t。本库的 R 是 world->cam，
    所以取转置 R_w2c = R_c2v^T。
    """
    R_c2v = quaternion_to_rotation(quat, scalar_first=scalar_first)
    t = None if translation is None else tuple(float(x) for x in np.asarray(translation).reshape(-1)[:3])
    return R_c2v.T, t
