"""vcam.core: 无 IO 依赖的几何核心（旋转 + 相对姿态）。"""

from .rotation import (CONVENTIONS, PRE_ROT_VALEO, CameraConvention,
                       RotationConvention, ValeoConvention, get_convention,
                       optical_axis_in_world, rotation_relative,
                       rotation_to_ypr, ypr_to_rotation, zxz_to_rotation)

__all__ = [
    "RotationConvention", "ValeoConvention", "CameraConvention", "CONVENTIONS",
    "PRE_ROT_VALEO", "get_convention", "ypr_to_rotation", "rotation_to_ypr",
    "zxz_to_rotation", "rotation_relative", "optical_axis_in_world",
]
