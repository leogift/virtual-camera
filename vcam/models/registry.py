"""
相机模型注册表与统一构造入口。

为什么用注册表而不是 if/elif 分发：旧代码里"模型名规范化"、"按配置猜模型"、
"取转换函数"是三张各自维护的映射，加一个模型要改四处而且漏一处不会报错。
这里模型名 -> 类的映射只有一份，别名表也只有一份，`build_model()` 是唯一
构造入口。
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base import CameraModel
from .eftheta import EnhancedFThetaModel
from .kb import KannalaBrandtModel
from .pinhole import PinholeModel
from .radial_poly import RadialPolyModel

__all__ = [
    "CAMERA_MODELS", "register_model", "normalize_model_name",
    "get_model_class", "build_model", "CameraModelABC",
    "PinholeModel", "KannalaBrandtModel", "RadialPolyModel", "EnhancedFThetaModel",
]

CameraModelABC = CameraModel

CAMERA_MODELS: Dict[str, Type[CameraModel]] = {}


def register_model(*aliases):
    """类装饰器: 把一个 CameraModel 子类登记到主名 + 别名上。"""
    def deco(cls):
        names = {cls.model_name} | {a.lower() for a in aliases}
        for n in names:
            CAMERA_MODELS[n] = cls
        cls.aliases = tuple(sorted(names))
        return cls
    return deco


# 主名 + 常见拼写变体。规范化前会先去掉 '-' 和 '_' 并转小写。
for _cls, _extra in (
    (PinholeModel, ("perspective", "standard", "rectilinear")),
    (KannalaBrandtModel, ("kannala", "kannalabrandt", "fisheyekb")),
    (RadialPolyModel, ("poly4", "poly4th", "polynomial", "fisheye",
                       "radialpoly", "woodsccapepoly")),
    (EnhancedFThetaModel, ("ftheta", "enhancedftheta")),
):
    register_model(*_extra)(_cls)


def normalize_model_name(name) -> str:
    """别名/大小写/连字符一律归一；未知名字直接报错并列出可用项。"""
    if isinstance(name, CameraModel):
        return name.model_name
    key = str(name or "pinhole").strip().lower().replace("-", "").replace("_", "")
    key = key.replace(" ", "")
    if key not in CAMERA_MODELS:
        known = ", ".join(sorted(set(k for k in CAMERA_MODELS)))
        raise ValueError(f"未知相机模型: {name!r}。可用: {known}")
    return CAMERA_MODELS[key].model_name


def get_model_class(name) -> Type[CameraModel]:
    cls = CAMERA_MODELS.get(normalize_model_name(name))
    if cls is None:                       # pragma: no cover - normalize 已拦截
        raise ValueError(f"未知相机模型: {name!r}")
    return cls


def build_model(model, width=None, height=None, **params) -> CameraModel:
    """
    统一构造入口。`model` 可以是:
      * 已有的 CameraModel 实例（原样返回，允许 width/height 覆盖检查）
      * 模型名字符串（'pinhole' / 'kb' / '4thpoly' / 'eftheta' 及其别名）
      * 一个含 'model' 或 'cameraModel' 键的 dict（形如相机配置的扁平字典）
    params 直接透传给模型构造函数；未给出的 width/height 从参数里补。
    """
    if isinstance(model, CameraModel):
        return model
    if isinstance(model, dict):
        from ..io.config_schema import camera_model_from_cfg_dict
        return camera_model_from_cfg_dict(model)
    cls = get_model_class(model)
    kwargs = dict(params)
    if width is not None:
        kwargs.setdefault("width", width)
    if height is not None:
        kwargs.setdefault("height", height)
    missing = [k for k in ("width", "height") if k not in kwargs]
    if missing:
        raise TypeError(f"build_model('{model}') 缺少必填参数: {missing}")
    return cls(**kwargs)
