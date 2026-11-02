"""
相机工厂：从标定文件 / 配置字典 / 数据集目录构造 FisheyeCamera。

这一层是外部调用方唯一需要知道"文件长什么样"的地方；vcam 的其余部分
（core / models / pipeline）都不认识任何第三方 JSON。
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from ..core.rotation import get_convention, rotation_to_ypr
from ..pipeline.camera import CameraPose, FisheyeCamera
from .config_schema import ExtrinsicInfo, camera_model_from_cfg_dict, parse_extrinsic

__all__ = [
    "load_json",
    "camera_from_calibration",
    "camera_from_cfg_dict",
    "NFOCCDataset",
    "load_convert_json",
]

# NFOCC / WoodScape 相机名 -> (cam_id, 说明)。cam_id 用来对齐 rgb_images 的序号
# 前缀，不能靠列表顺序猜（数据里 FV=164 / MVL=165 / MVR=166 / RV=167 并不左右
# 交替）。
NFOCC_CAM_ORDER = {"FV": 0, "RV": 1, "MVL": 2, "MVR": 3}
NFOCC_CAM_DESC = {"FV": "Front View", "RV": "Rear View",
                  "MVL": "Mirror View Left", "MVR": "Mirror View Right"}


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pose_from_extrinsic(info: ExtrinsicInfo, convention) -> CameraPose:
    conv = get_convention(convention)
    rpy = info.rpy_deg if info.convention == conv.name else rotation_to_ypr(info.R, conv)
    return CameraPose(yaw_deg=rpy[0], pitch_deg=rpy[1], roll_deg=rpy[2],
                      convention=conv, translation=info.translation)


def camera_from_calibration(calib: dict, convention="valeo",
                            name: Optional[str] = None,
                            extrinsic: Optional[dict] = None) -> FisheyeCamera:
    """
    从一份"intrinsic + extrinsic"标定字典构造相机。

    WoodScape/NFOCC 一份 JSON 同时带内参和外参，直接传整个 dict 即可；
    若内外参分开存（旧 convert.json 风格），用 extrinsic= 显式给。
    """
    name = name or calib.get("name", "cam")
    model = camera_model_from_cfg_dict(calib)
    ext_cfg = extrinsic if extrinsic is not None else calib.get("extrinsic")
    info = parse_extrinsic(ext_cfg, convention=convention, name=name)
    pose = _pose_from_extrinsic(info, convention)
    return FisheyeCamera(name=name, model=model, pose=pose, is_virtual=False)


def camera_from_cfg_dict(intr_cfg: dict, ext_cfg=None, convention="valeo",
                         name="cam", is_virtual=False) -> FisheyeCamera:
    """旧 convert.json 的 (src_intrinsic, src_extrinsic) 组合 -> 相机对象。"""
    model = camera_model_from_cfg_dict(intr_cfg)
    info = parse_extrinsic(ext_cfg, convention=convention, name=name)
    pose = _pose_from_extrinsic(info, convention)
    return FisheyeCamera(name=name, model=model, pose=pose, is_virtual=is_virtual)


class NFOCCDataset:
    """
    NFOCC / WoodScape 鱼眼数据集目录::

        <root>/calibration/calibration/00164_FV.json ...
        <root>/rgb_images/00164_FV.png
        <root>/vehicle_info/rgb_images/00164_FV.json   (vehicle-split 元数据)

    标定文件在 00164..00175 上重复三次（同一套相机），所以按相机名去重，
    取**序号最小**的那份作为标定，并把它对应的图像作为默认图像。
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.calib_dir = self._first_existing(
            [os.path.join(self.root, "calibration", "calibration"),
             os.path.join(self.root, "calibration")])
        self.image_dir = self._first_existing(
            [os.path.join(self.root, "rgb_images"), self.root])

    @staticmethod
    def _first_existing(cands):
        for c in cands:
            if os.path.isdir(c):
                return c
        raise FileNotFoundError(f"找不到标定/图像目录，试过: {cands}")

    def frames(self) -> Dict[str, List[dict]]:
        """按相机名分组: {'FV': [{'index':164,'calib':path,'image':path}, ...]}"""
        out: Dict[str, List[dict]] = {}
        for calib_path in sorted(glob.glob(os.path.join(self.calib_dir, "*.json"))):
            base = os.path.splitext(os.path.basename(calib_path))[0]
            idx_str, _, cam = base.partition("_")
            if not idx_str.isdigit() or cam not in NFOCC_CAM_ORDER:
                continue
            img = os.path.join(self.image_dir, f"{base}.png")
            out.setdefault(cam, []).append(
                dict(index=int(idx_str), calib=calib_path,
                     image=img if os.path.exists(img) else None))
        for v in out.values():
            v.sort(key=lambda d: d["index"])
        return out

    def camera_names(self) -> List[str]:
        names = list(self.frames().keys())
        return sorted(names, key=lambda n: NFOCC_CAM_ORDER.get(n, 99))

    def camera(self, name: str, convention="valeo", frame: int = 0) -> FisheyeCamera:
        """
        取某台相机的标定。frame 是"同名相机第几帧"，默认 0 = 序号最小的那份。
        """
        if name not in NFOCC_CAM_ORDER and name not in self.frames():
            raise KeyError(f"数据集里没有相机 {name!r}，可用: {self.camera_names()}")
        entries = self.frames()[name]
        if frame >= len(entries):
            raise IndexError(f"{name} 只有 {len(entries)} 份标定，请求 frame={frame}")
        e = entries[frame]
        calib = load_json(e["calib"])
        cam = camera_from_calibration(calib, convention=convention, name=name)
        cam.index = e["index"]          # 记下数据序号，写元数据时用
        cam.image_path = e["image"]
        return cam

    def image_path(self, name: str, frame: int = 0) -> str:
        e = self.frames()[name][frame]
        if e["image"] is None:
            raise FileNotFoundError(f"{name} frame={frame} 没有对应图像")
        return e["image"]

    def summary(self) -> str:
        lines = [f"NFOCC dataset root: {self.root}"]
        for cam, entries in self.frames().items():
            lines.append(f"  {cam:4s} ({NFOCC_CAM_DESC.get(cam, '?')}): "
                         f"{len(entries)} 份标定, 序号 {[e['index'] for e in entries]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 旧 convert.json 兼容层
# --------------------------------------------------------------------------

def load_convert_json(path, convention="valeo", ext_map=None) -> List[dict]:
    """
    读取旧版 convert.json 的任务列表，返回:
        [{'name':..., 'src':FisheyeCamera, 'dst_intrinsic_cfg':dict,
          'image_path':..., 'dst_path':..., 'dst_ext':...}]

    与旧实现的差异：字段缺失一律报错（旧实现 print 一行 "配置字段不完整，跳过"
    就继续，跑完的"成功 0 路"看起来毫无问题）。
    """
    data = load_json(path)
    folder = os.path.dirname(os.path.abspath(path)) or "."
    ext_map = ext_map or {}
    tasks = []
    for i, cam in enumerate(data.get("cameras", [])):
        name = cam.get("name", f"cam_{i}")
        src_cfg = cam.get("src_intrinsic") or cam.get("intrinsic")
        dst_cfg = cam.get("dst_intrinsic") or cam.get("dst_intrinsics")
        if src_cfg is None or dst_cfg is None:
            raise KeyError(f"[{name}] convert.json 需要 src_intrinsic 与 dst_intrinsic")
        ext = (cam.get("src_extrinsic") or cam.get("extrinsic")
               or ext_map.get(name) or ext_map.get(i) or src_cfg.get("extrinsic"))
        src = camera_from_cfg_dict(src_cfg, ext, convention=convention, name=name)
        img = cam.get("img_path")
        if img and not os.path.isabs(img):
            cand = os.path.join(folder, img)
            img = cand if os.path.exists(cand) else img
        tasks.append(dict(name=name, src=src, dst_intrinsic_cfg=dst_cfg,
                          image_path=img,
                          dst_path=cam.get("dst_path") or cam.get("output_path"),
                          dst_ext=cam.get("dst_extrinsic") or dst_cfg.get("extrinsic"),
                          index=i))
    return tasks
