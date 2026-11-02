"""
旧 virtual_camera_core.py 的兼容 API。

目的只有一个：让已有的调用脚本（test_yaw45.py 之类）不改一行就能切到新库，
并让"新旧实现逐像素对比"成为可能。这些函数不做任何新的几何计算，全部转发到
vcam 的对象上。

与旧实现的**行为差异**（都是修正，不是等价替代）：
  * camera 约定的 YPR 往返现在自洽（旧代码静默错）；
  * 反投影失败/未收敛像素不再静默使用，而是标为 -1 并计入统计；
  * do_convert 不再偷偷 resize 源图（尺寸不符直接报错）；
  * 16 个 convert_*_grid 包装函数用**动态生成**，不再逐个手写（旧版本加一个
    模型要写 8 个几乎相同的函数）。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .core import rotation as _rot
from .io.camera_factory import camera_from_cfg_dict
from .pipeline.camera import CameraPose, FisheyeCamera, VirtualCamera
from .pipeline.converter import GridConverter

__all__ = [
    "rotation_yaw_pitch_roll_to_mat", "mat_to_yaw_pitch_roll", "rotation_zxz_to_mat",
    "compute_relative_rotation", "convert_camera_grid", "convert_grid",
    "get_convert_func", "do_convert", "do_convert_cv2", "maps_to_grid",
    "normalize_camera_model", "get_camera_model", "solve_theta_robust_4thpoly",
    "unproject_pixel_to_ray", "project_ray_to_pixel", "rotate_ray",
    "convert_pinhole_to_pinhole_grid",
]


# ---- 旋转 -----------------------------------------------------------------

def rotation_yaw_pitch_roll_to_mat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, convention="valeo"):
    return _rot.ypr_to_rotation(yaw_deg, pitch_deg, roll_deg, convention)


def mat_to_yaw_pitch_roll(R, convention="valeo"):
    return _rot.rotation_to_ypr(R, convention)


def rotation_zxz_to_mat(rotz1_deg=0.0, rotx_deg=0.0, rotz2_deg=0.0):
    return _rot.zxz_to_rotation(rotz1_deg, rotx_deg, rotz2_deg)


def compute_relative_rotation(src_extrinsic=None, dst_extrinsic=None, delta_rpy=None,
                              convention="valeo"):
    """返回 (R_rel, src_rpy, dst_rpy)，语义与旧实现一致。"""
    from .io.config_schema import parse_extrinsic
    src = parse_extrinsic(src_extrinsic, convention=convention, name="src")
    if delta_rpy is not None:
        dy, dp, dr = delta_rpy
        dst_rpy = (src.rpy_deg[0] + dy, src.rpy_deg[1] + dp, src.rpy_deg[2] + dr)
        R_dst = _rot.ypr_to_rotation(*dst_rpy, convention=convention)
    elif dst_extrinsic is not None:
        if isinstance(dst_extrinsic, dict) and any(
                k in dst_extrinsic for k in ("delta_yaw", "delta_pitch", "delta_roll")):
            dst_rpy = (src.rpy_deg[0] + float(dst_extrinsic.get("delta_yaw", 0.0)),
                       src.rpy_deg[1] + float(dst_extrinsic.get("delta_pitch", 0.0)),
                       src.rpy_deg[2] + float(dst_extrinsic.get("delta_roll", 0.0)))
            R_dst = _rot.ypr_to_rotation(*dst_rpy, convention=convention)
        else:
            dst = parse_extrinsic(dst_extrinsic, convention=convention, name="dst")
            dst_rpy, R_dst = dst.rpy_deg, dst.R
    else:
        dst_rpy, R_dst = src.rpy_deg, src.R
    return _rot.rotation_relative(src.R, R_dst), src.rpy_deg, dst_rpy


# ---- 相机模型名 ------------------------------------------------------------

def normalize_camera_model(model_name):
    from .models.registry import normalize_model_name
    return normalize_model_name(model_name)


def get_camera_model(cfg):
    from .io.config_schema import infer_model_name
    return infer_model_name(cfg)


# ---- 网格构建 --------------------------------------------------------------

def _build_pair(src_cfg, dst_cfg, src_model=None, dst_model=None, src_extrinsic=None):
    src = camera_from_cfg_dict(src_cfg, src_extrinsic, name="src")
    if src_model is not None:
        from .models.registry import normalize_model_name
        if normalize_model_name(src_model) != src.model.model_name:
            raise ValueError(
                f"src_model={src_model!r} 与配置里声明的模型 {src.model.model_name!r} 冲突。"
                "旧实现让显式参数静默覆盖配置，掩盖了配置写错的情况。")
    dst = camera_from_cfg_dict(dst_cfg, None, name="dst", is_virtual=True)
    if dst_model is not None:
        from .models.registry import normalize_model_name
        if normalize_model_name(dst_model) != dst.model.model_name:
            raise ValueError(
                f"dst_model={dst_model!r} 与配置里声明的模型 {dst.model.model_name!r} 冲突")
    return src, dst


def convert_camera_grid(src_cfg, dst_cfg, R_rel=None, src_extrinsic=None, dst_extrinsic=None,
                        delta_rpy=None, convention="valeo", src_model=None, dst_model=None):
    """返回 (map_x, map_y)。"""
    src, dst = _build_pair(src_cfg, dst_cfg, src_model, dst_model, src_extrinsic)
    if R_rel is None:
        R_rel = compute_relative_rotation(src_extrinsic, dst_extrinsic,
                                         delta_rpy=delta_rpy, convention=convention)[0]
    conv = GridConverter(src, dst, R_rel=R_rel)
    map_x, map_y, _ = conv.build_maps()
    return map_x, map_y


def convert_grid(src_cfg, dst_cfg, convert_func=None, **kwargs):
    if convert_func is not None:
        return convert_func(src_cfg, dst_cfg, **kwargs)
    return convert_camera_grid(src_cfg, dst_cfg, **kwargs)


def get_convert_func(src_cfg, dst_cfg):
    """16 个组合的包装函数改为按模型名动态生成。"""
    src_m = get_camera_model(src_cfg)
    dst_m = get_camera_model(dst_cfg)

    def _fn(s_cfg, d_cfg, **kw):
        kw.setdefault("src_model", src_m)
        kw.setdefault("dst_model", dst_m)
        return convert_camera_grid(s_cfg, d_cfg, **kw)

    _fn.__name__ = f"convert_{src_m}_to_{dst_m}_grid"
    return _fn


# 旧代码里被显式 import 的名字
def _make_named(src_m, dst_m):
    def fn(src_cfg, dst_cfg, R_rel=None, src_extrinsic=None, dst_extrinsic=None,
           delta_rpy=None, convention="valeo", **kwargs):
        return convert_camera_grid(src_cfg, dst_cfg, R_rel=R_rel,
                                   src_extrinsic=src_extrinsic,
                                   dst_extrinsic=dst_extrinsic, delta_rpy=delta_rpy,
                                   convention=convention,
                                   src_model=src_m, dst_model=dst_m)
    fn.__name__ = f"convert_{src_m}_to_{dst_m}_grid"
    return fn


_MODELS = ("pinhole", "kb", "4thpoly", "eftheta")
for _s in _MODELS:
    for _d in _MODELS:
        globals()[f"convert_{_s}_to_{_d}_grid"] = _make_named(_s, _d)
del _s, _d
__all__.extend(f"convert_{a}_to_{b}_grid" for a in _MODELS for b in _MODELS)


# ---- 图像重采样 ------------------------------------------------------------

def do_convert_cv2(img, src_cfg, dst_cfg, convert_func=None, R_rel=None,
                   src_extrinsic=None, dst_extrinsic=None, delta_rpy=None,
                   convention="valeo", interpolation=None, border_mode=None,
                   border_value=0):
    """返回 (dst_img, map_x, map_y)，与旧实现签名一致。"""
    src, dst = _build_pair(src_cfg, dst_cfg, None, None, src_extrinsic)
    if R_rel is None:
        R_rel = compute_relative_rotation(src_extrinsic, dst_extrinsic,
                                         delta_rpy=delta_rpy, convention=convention)[0]
    conv = GridConverter(src, dst, R_rel=R_rel)
    map_x, map_y, _ = conv.build_maps()
    res = conv.remap(img, map_x, map_y, interpolation=interpolation,
                     border_mode=border_mode, border_value=border_value)
    return res.image, res.map_x, res.map_y


do_convert = do_convert_cv2


def maps_to_grid(map_x, map_y, sw, sh, normalize=True, align_corners=True):
    """旧签名保留；内部用 GridConverter.to_grid 的除零保护（报告问题 7）。"""
    if not normalize:
        return np.stack([map_x, map_y], axis=-1)[np.newaxis, ...]
    if align_corners:
        dx, dy = max(int(sw) - 1, 1), max(int(sh) - 1, 1)
        gx = 2.0 * map_x / dx - 1.0
        gy = 2.0 * map_y / dy - 1.0
    else:
        gx = 2.0 * (map_x + 0.5) / max(int(sw), 1) - 1.0
        gy = 2.0 * (map_y + 0.5) / max(int(sh), 1) - 1.0
    return np.stack([gx, gy], axis=-1)[np.newaxis, ...]


# ---- 低层函数（给需要逐像素控制的调用方） ----------------------------------

def unproject_pixel_to_ray(u, v, p, model="pinhole"):
    from .models.registry import build_model
    if not hasattr(p, "unproject"):
        p = _cfg_to_model(p, model)
    return p.unproject(u, v).ray


def project_ray_to_pixel(ray, p, model="pinhole"):
    if not hasattr(p, "project"):
        p = _cfg_to_model(p, model)
    return p.project(ray[..., 0], ray[..., 1], ray[..., 2])


def rotate_ray(ray, R_rel):
    if R_rel is None:
        return ray
    return np.asarray(ray) @ np.asarray(R_rel, dtype=np.float64).T


def solve_theta_robust_4thpoly(r_d, k1, k2, k3, k4, k5=0.0):
    from .models.radial_poly import solve_poly_theta
    theta, _ = solve_poly_theta(r_d, [k1, k2, k3, k4, k5])
    return theta


solve_theta_robust = solve_theta_robust_4thpoly
solve_theta_robust_5thpoly = solve_theta_robust_4thpoly


def _cfg_to_model(cfg: dict, model="pinhole"):
    """旧 API 传进来的是 _get_cam_params() 那样的扁平字典（含 w/h/fx/fy/cx/cy）。"""
    from .io.config_schema import camera_model_from_cfg_dict
    if "w" in cfg and "h" in cfg:
        model = cfg.get("model", model)
        cfg = dict(model=model, width_px=cfg["w"], height_px=cfg["h"],
                   focal_x_px=cfg["fx"], focal_y_px=cfg["fy"],
                   center_x_px=cfg["cx"], center_y_px=cfg["cy"],
                   k1=cfg.get("k1", 0.0), k2=cfg.get("k2", 0.0),
                   k3=cfg.get("k3", 0.0), k4=cfg.get("k4", 0.0),
                   p1=cfg.get("p1", 0.0), p2=cfg.get("p2", 0.0),
                   elem_c=cfg.get("c", 1.0), elem_d=cfg.get("d", 0.0),
                   elem_e=cfg.get("e", 0.0))
    return camera_model_from_cfg_dict(cfg)
