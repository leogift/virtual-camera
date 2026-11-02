"""
配置解析：把各种第三方标定/配置字典翻译成 vcam 的模型对象。

旧 virtual_camera_core.py 把这件事写成了三层 `.get()` 兜底链
（`intr.get('width_px', intr.get('img_width_px', intr.get('width', 1920)))`），
后果是**缺字段不报错，而是悄悄用 1920x1080 或焦距 1000**。这里换一种写法：
按格式显式分派（WoodScape / 扁平 / 嵌套），每个格式只认自己那几个键，
缺必填项直接抛异常并说明缺哪个。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..models.base import CameraModel

__all__ = [
    "parse_woodsccape_intrinsic",
    "parse_flat_intrinsic",
    "camera_model_from_cfg_dict",
    "infer_model_name",
    "parse_extrinsic",
    "ExtrinsicInfo",
]


# --------------------------------------------------------------------------
# 内参
# --------------------------------------------------------------------------

def parse_woodsccape_intrinsic(intr: dict, name: str = "") -> CameraModel:
    """
    WoodScape / NFOCC 数据集格式::

        {"width":1280.0,"height":966.0,"k1":..,"k2":..,"k3":..,"k4":..,
         "cx_offset":3.942,"cy_offset":-3.093,"aspect_ratio":1.0,
         "model":"radial_poly","poly_order":4}

    主点约定: cx = width/2 + cx_offset, cy = height/2 + cy_offset。

    **已知存疑**（记录在案，不做猜测）: readme 的公式里写的是
    `u' + cx_offset + width/2 - 0.5`。实测用 00164_FV 图像做同心度判据，
    两种解释（减 0.5 / 不减）的差异只有 0.5 px，判据本身噪声 ±5 px，
    无法分辨。取不减 0.5 的理由只有一条且是结构性的：此时 max(rho) 恰好
    等于主点到图像边缘的距离，即"最外一圈入射角恰好在边缘"。若上游确认
    需要减 0.5，只需给本函数加 principal_shift=-0.5，不改其它任何地方。
    """
    need = ("width", "height", "k1", "k2", "k3", "k4")
    missing = [k for k in need if k not in intr]
    if missing:
        raise KeyError(f"[{name}] WoodScape 内参缺少字段: {missing}")
    w, h = int(round(float(intr["width"]))), int(round(float(intr["height"])))
    offx = float(intr.get("cx_offset", 0.0))
    offy = float(intr.get("cy_offset", 0.0))
    aspect = float(intr.get("aspect_ratio", 1.0) or 1.0)
    order = int(intr.get("poly_order", 4))
    keys = [f"k{i}" for i in range(1, order + 1)]
    for kk in keys:
        if kk not in intr:
            raise KeyError(f"[{name}] model=radial_poly 声明 poly_order={order}，"
                           f"但缺少系数 {kk}")

    from ..models.radial_poly import RadialPolyModel
    return RadialPolyModel(w, h, [float(intr[kk]) for kk in keys],
                           cx=w / 2.0 + offx, cy=h / 2.0 + offy,
                           aspect_ratio=aspect)


def parse_flat_intrinsic(intr: dict, name: str = "") -> CameraModel:
    """
    扁平格式（旧 convert.json 的 src_intrinsic / dst_intrinsic）::

        {"cameraModel":"eftheta","width_px":1920,"height_px":1536,
         "center_x_px":960,"center_y_px":768,
         "focal_x_px":..,"focal_y_px":..,"k1":..,"k2":..,"k3":..,"k4":..,
         "elem_c":1.0,"elem_d":0.0,"elem_e":0.0,"p1":..,"p2":..}

    兼容的别名只保留**真实出现过**的那些（width_px/img_width_px/width 等），
    并且一旦某个模型必需的量缺失就报错，不再落到 1920x1080 默认值。
    """
    inner = intr.get("intrinsics") or intr.get("intrinsic")
    if isinstance(inner, dict):
        # 嵌套结构: 合并两层，内层优先
        merged = dict(intr)
        merged.update(inner)
        intr = merged
    dist = intr.get("distortion") or intr.get("distortions")

    def pick(*keys, default=None, required=False):
        for k in keys:
            if k in intr and intr[k] is not None:
                return intr[k]
        if required:
            raise KeyError(f"[{name}] 内参缺少必填字段，候选键名: {list(keys)}")
        return default

    w = pick("width_px", "img_width_px", "width", required=True)
    h = pick("height_px", "img_height_px", "height", required=True)
    w, h = int(round(float(w))), int(round(float(h)))
    fx = pick("focal_x_px", "fx_px", "f_x_px", "fx")
    fx = float(fx) if fx is not None else None
    fy = pick("focal_y_px", "fy_px", "f_y_px", "fy")
    fy = float(fy) if fy is not None else None
    cx = pick("center_x_px", "cx_px", "cx", "cu_px")
    cy = pick("center_y_px", "cy_px", "cy", "cv_px")
    cx = float(cx) if cx is not None else w / 2.0
    cy = float(cy) if cy is not None else h / 2.0

    def dget(*keys, default=0.0):
        if isinstance(dist, (list, tuple)):
            idx = {"k1": 0, "k2": 1, "k3": 2, "k4": 3, "k5": 4, "p1": 5, "p2": 6}
            for k in keys:
                if k in idx and len(dist) > idx[k]:
                    return float(dist[idx[k]])
            return default
        for k in keys:
            src = dist if isinstance(dist, dict) else intr
            if k in src and src[k] is not None:
                return float(src[k])
        return default

    k1, k2, k3, k4 = dget("k1"), dget("k2"), dget("k3"), dget("k4")
    p1, p2 = dget("p1"), dget("p2")
    c = dget("elem_c", "c", default=1.0)
    d = dget("elem_d", "d", default=0.0)
    e = dget("elem_e", "e", default=0.0)
    k5 = dget("k5")

    model = infer_model_name(intr)
    if model == "pinhole":
        from ..models.pinhole import PinholeModel
        if fx is None:
            raise KeyError(f"[{name}] pinhole 内参必须有焦距 focal_x_px/fx")
        return PinholeModel(w, h, fx=fx, fy=fy, cx=cx, cy=cy,
                            k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2)
    if model == "kb":
        from ..models.kb import KannalaBrandtModel
        if fx is None:
            raise KeyError(f"[{name}] kb 内参必须有焦距 focal_x_px/fx")
        return KannalaBrandtModel(w, h, fx=fx, fy=fy, cx=cx, cy=cy,
                                  k1=k1, k2=k2, k3=k3, k4=k4)
    if model == "eftheta":
        from ..models.eftheta import EnhancedFThetaModel
        return EnhancedFThetaModel(w, h, [k1, k2, k3, k4, k5], c=c, d=d, e=e,
                                   cx=cx, cy=cy)
    # radial poly
    from ..models.radial_poly import RadialPolyModel
    return RadialPolyModel(w, h, [k1, k2, k3, k4, k5], cx=cx, cy=cy)


def infer_model_name(cfg: dict) -> str:
    """
    只读**显式**字段；不靠"有哪些键"去猜。

    旧实现的启发式（有 elem_c 就 eftheta、有 fisheyeCoeff1 就 4thpoly、
    有 f_x_px 就 kb）在真实配置上会误判：WoodScape 的 radial_poly 也有
    "focal 量级"的 k1，一旦有人补个 fx 字段就会被判成 kb。缺 model 字段
    时这里直接报错，要求调用方写清楚。
    """
    inner = cfg.get("intrinsics") if isinstance(cfg.get("intrinsics"), dict) else {}
    inner2 = cfg.get("intrinsic") if isinstance(cfg.get("intrinsic"), dict) else {}
    for src in (cfg, inner, inner2):
        for key in ("cameraModel", "model", "type", "camera_model"):
            v = src.get(key) if isinstance(src, dict) else None
            if v:
                from ..models.registry import normalize_model_name
                return normalize_model_name(v)
    raise KeyError("配置里没有相机模型字段（cameraModel/model/type），"
                   "本库不按参数键名猜模型，请显式写明。")


def camera_model_from_cfg_dict(cfg: dict) -> CameraModel:
    """主入口: 任意相机字典 -> CameraModel。"""
    if not isinstance(cfg, dict):
        raise TypeError(f"相机配置必须是 dict，实际 {type(cfg)}")
    # WoodScape/NFOCC 内参（model=radial_poly 且带 cx_offset）
    intr = cfg.get("intrinsic") if isinstance(cfg.get("intrinsic"), dict) else cfg
    if isinstance(intr, dict) and "cx_offset" in intr and "k1" in intr:
        return parse_woodsccape_intrinsic(intr, name=cfg.get("name", ""))
    return parse_flat_intrinsic(cfg, name=cfg.get("name", ""))


# --------------------------------------------------------------------------
# 外参
# --------------------------------------------------------------------------

class ExtrinsicInfo:
    """解析结果: world->cam 旋转矩阵 + 可读的 yaw/pitch/roll + 来源标记。"""

    __slots__ = ("R", "rpy_deg", "source", "translation", "convention")

    def __init__(self, R, rpy_deg, source, translation=None, convention="valeo"):
        self.R = np.asarray(R, dtype=np.float64)
        self.rpy_deg = tuple(float(x) for x in rpy_deg)
        self.source = source
        self.translation = None if translation is None else tuple(float(x) for x in translation)
        self.convention = convention

    def __repr__(self):
        return (f"ExtrinsicInfo(source={self.source!r}, "
                f"ypr={tuple(round(v, 3) for v in self.rpy_deg)})")


def _as_4x4(data, row_major: bool, what: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (16,):
        return arr.reshape((4, 4), order="C" if row_major else "F")
    if arr.ndim == 2 and arr.shape == (1, 16):
        return arr.reshape((4, 4), order="C" if row_major else "F")
    raise ValueError(
        f"{what}: 期望 4x4 矩阵或 16 个数的扁平列表，实际 shape={arr.shape}。"
        "旧实现遇到嵌套 [[..],[..]] 会静默跳过并回落到默认姿态（报告 S3），"
        "这里直接报错。")


def parse_extrinsic(cfg, convention="valeo", name: str = "") -> ExtrinsicInfo:
    """
    解析外参配置，支持：
      * 3x3 旋转矩阵 (世界->相机)
      * 4x4 变换矩阵（16 数扁平 / 4x4 嵌套，rowMajor 可切换）
      * ZXZ 欧拉角 rotx/rotz1/rotz2
      * 角度制 yaw/pitch/roll（别名 rot_yaw_deg / yaw_deg ...）
      * 弧度制 rot_yaw_rad / rot_pitch_rad / rot_roll_rad
      * (yaw, pitch, roll) 三元组
      * None -> 单位姿态
    """
    from ..core.rotation import (get_convention, rotation_to_ypr,
                                 ypr_to_rotation, zxz_to_rotation)

    conv = get_convention(convention)
    if cfg is None:
        return ExtrinsicInfo(conv.to_matrix(0, 0, 0), (0.0, 0.0, 0.0),
                             "default-identity", convention=conv.name)

    if isinstance(cfg, np.ndarray) and cfg.shape == (3, 3):
        return ExtrinsicInfo(cfg, rotation_to_ypr(cfg, conv), "rotation-matrix-3x3",
                             convention=conv.name)

    if isinstance(cfg, (list, tuple)) and len(cfg) == 3 and not isinstance(cfg[0], (list, tuple)):
        rpy = tuple(float(x) for x in cfg)
        return ExtrinsicInfo(ypr_to_rotation(*rpy, convention=conv), rpy,
                             "ypr-triplet", convention=conv.name)

    if isinstance(cfg, (list, tuple)) and len(cfg) in (16,):
        M = _as_4x4(cfg, False, f"[{name}] extrinsic")
        return ExtrinsicInfo(M[:3, :3], rotation_to_ypr(M[:3, :3], conv),
                             "matrix4x4-colmajor", tuple(M[:3, 3]), conv.name)

    if not isinstance(cfg, dict):
        raise TypeError(f"[{name}] 无法解析的外参类型: {type(cfg)}")

    ext = cfg.get("extrinsic") or cfg.get("extrinsics")
    if isinstance(ext, dict):
        cfg = ext
    row_major = bool(cfg.get("rowMajor", cfg.get("row_major", False)))

    # 4x4
    for key in ("camera_external", "camera2vehicle", "T", "matrix", "transform"):
        if key in cfg:
            M = _as_4x4(cfg[key], row_major, f"[{name}] {key}")
            return ExtrinsicInfo(M[:3, :3], rotation_to_ypr(M[:3, :3], conv),
                                 f"matrix4x4-{key}", tuple(M[:3, 3]), conv.name)

    # 3x3
    if "R" in cfg or "rotation" in cfg:
        R = np.asarray(cfg.get("R", cfg.get("rotation")), dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"[{name}] 旋转矩阵应为 3x3，实际 {R.shape}")
        return ExtrinsicInfo(R, rotation_to_ypr(R, conv), "rotation-matrix-3x3",
                             convention=conv.name)

    # ZXZ
    if any(k in cfg for k in ("rotx", "rotz1", "rotz2",
                              "rot_x_deg", "rot_z1_deg", "rot_z2_deg")):
        rx = float(cfg.get("rotx", cfg.get("rot_x_deg", 0.0)))
        rz1 = float(cfg.get("rotz1", cfg.get("rot_z1_deg", 0.0)))
        rz2 = float(cfg.get("rotz2", cfg.get("rot_z2_deg", 0.0)))
        R = zxz_to_rotation(rz1, rx, rz2)
        return ExtrinsicInfo(R, rotation_to_ypr(R, conv), "zxz-euler",
                             convention=conv.name)

    # 弧度制 YPR
    if any(k in cfg for k in ("rot_yaw_rad", "rot_pitch_rad", "rot_roll_rad")):
        rpy = (float(np.degrees(cfg.get("rot_yaw_rad", 0.0))),
               float(np.degrees(cfg.get("rot_pitch_rad", 0.0))),
               float(np.degrees(cfg.get("rot_roll_rad", 0.0))))
        return ExtrinsicInfo(ypr_to_rotation(*rpy, convention=conv), rpy,
                             "ypr-rad", convention=conv.name)

    # 角度制 YPR
    if any(k in cfg for k in ("yaw", "pitch", "roll",
                              "rot_yaw_deg", "rot_pitch_deg", "rot_roll_deg",
                              "yaw_deg", "pitch_deg", "roll_deg")):
        rpy = (float(cfg.get("yaw", cfg.get("rot_yaw_deg", cfg.get("yaw_deg", 0.0)))),
               float(cfg.get("pitch", cfg.get("rot_pitch_deg", cfg.get("pitch_deg", 0.0)))),
               float(cfg.get("roll", cfg.get("rot_roll_deg", cfg.get("roll_deg", 0.0)))))
        return ExtrinsicInfo(ypr_to_rotation(*rpy, convention=conv), rpy,
                             "ypr-deg", convention=conv.name)

    # 四元数 (WoodScape 风格: 相机->车体, 需要转置成 车体->相机)
    if "quaternion" in cfg:
        from ..core.rotation import quaternion_cam2veh_to_rt
        R_w2c, t = quaternion_cam2veh_to_rt(cfg["quaternion"], cfg.get("translation"))
        return ExtrinsicInfo(R_w2c, rotation_to_ypr(R_w2c, conv),
                             "quaternion-cam2veh", t, conv.name)

    raise KeyError(f"[{name}] 外参字典里没有任何可识别的字段，键: {sorted(cfg)}")
