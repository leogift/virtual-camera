"""相机模型子包: 一个模型一个文件，通过 registry 统一构造。"""

from .base import CameraModel, UnprojectionResult
from .eftheta import EnhancedFThetaModel
from .kb import KannalaBrandtModel
from .pinhole import PinholeModel
from .radial_poly import RadialPolyModel, solve_poly_theta
from .registry import (CAMERA_MODELS, build_model, get_model_class,
                       normalize_model_name, register_model)

__all__ = [
    "CameraModel", "UnprojectionResult",
    "PinholeModel", "KannalaBrandtModel", "RadialPolyModel", "EnhancedFThetaModel",
    "CAMERA_MODELS", "build_model", "get_model_class", "normalize_model_name",
    "register_model", "solve_poly_theta",
]
