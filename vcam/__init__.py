"""vcam: 虚拟相机几何库（内参转换 + 外参旋转 + remap 采样）的统一出口。

最小用法::

    from vcam import FisheyeCamera, VirtualCamera, GridConverter

    src = FisheyeCamera.from_nfocc_json('datasets/fisheye/calibration/.../00164_FV.json')
    dst = VirtualCamera.from_fov(640, 480, fov_h_deg=150, fov_v_deg=120)
    conv = GridConverter(src, dst, yaw_delta_deg=+45)     # +45 = 向左看
    map_x, map_y = conv.build_maps()
    out = conv.remap(image_bgr)
"""

from .core.rotation import (CameraConvention, RotationConvention,
                            ValeoConvention, get_convention,
                            optical_axis_in_world, rotation_relative,
                            rotation_to_ypr, ypr_to_rotation, zxz_to_rotation)
from .models import (CAMERA_MODELS, CameraModel, EnhancedFThetaModel,
                     KannalaBrandtModel, PinholeModel, RadialPolyModel,
                     build_model, normalize_model_name, register_model)
from .pipeline.camera import CameraPose, FisheyeCamera, VirtualCamera
from .pipeline.converter import GridConverter, MapStats, RemapResult

__version__ = "1.0.0"

__all__ = [
    "FisheyeCamera", "VirtualCamera", "CameraPose", "GridConverter",
    "MapStats", "RemapResult",
    "CameraModel", "PinholeModel", "KannalaBrandtModel", "RadialPolyModel",
    "EnhancedFThetaModel", "build_model", "normalize_model_name",
    "register_model", "CAMERA_MODELS",
    "RotationConvention", "ValeoConvention", "CameraConvention",
    "get_convention", "ypr_to_rotation", "rotation_to_ypr", "zxz_to_rotation",
    "rotation_relative", "optical_axis_in_world",
    "__version__",
]
