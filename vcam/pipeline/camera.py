"""
相机对象与位姿。

三个类，各自只有一件事：
  * CameraPose   —— 车体系下的姿态（yaw/pitch/roll），持有 3x3 旋转矩阵与约定，
                    可算出"光轴朝向哪边"，用于打印与断言。
  * FisheyeCamera —— 真实相机：模型参数 + 位姿。名字里的 Fisheye 是历史包袱，
                    它同样能持有 pinhole 模型（虚拟相机也用它）。
  * VirtualCamera —— 虚拟相机工厂：从 FOV / 从现有相机继承 / 从原始参数构造。

外部调用方只需要关心这两个构造方式::

    src = FisheyeCamera.from_nfocc_json(path, position=14)   # 见 vcam.io
    dst = VirtualCamera.from_fov(640, 297, fov_h_deg=150, fov_v_deg=120)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..core.rotation import (RotationConvention, get_convention,
                             optical_axis_in_world, ypr_to_rotation)
from ..models.base import CameraModel

__all__ = ["CameraPose", "FisheyeCamera", "VirtualCamera"]


@dataclass
class CameraPose:
    """车体系 (ISO 8855, X 前 / Y 左 / Z 上) 下的相机姿态。"""

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    convention: RotationConvention = field(default_factory=lambda: get_convention("valeo"))
    #: 可选的外参平移 (相机在车体系下的位置, 米)。仅用于诊断与生成元数据，
    #: 不参与纯旋转的 remap 计算。
    translation: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        self.convention = get_convention(self.convention)

    @property
    def matrix(self) -> np.ndarray:
        """world -> cam 的 3x3 旋转矩阵。"""
        return self.convention.to_matrix(self.yaw_deg, self.pitch_deg, self.roll_deg)

    def with_delta(self, dyaw=0.0, dpitch=0.0, droll=0.0) -> "CameraPose":
        return CameraPose(self.yaw_deg + dyaw, self.pitch_deg + dpitch,
                          self.roll_deg + droll, self.convention, self.translation)

    def with_absolute(self, yaw=None, pitch=None, roll=None) -> "CameraPose":
        return CameraPose(
            self.yaw_deg if yaw is None else yaw,
            self.pitch_deg if pitch is None else pitch,
            self.roll_deg if roll is None else roll,
            self.convention, self.translation)

    def optical_axis(self) -> np.ndarray:
        """光轴在车体系下的单位向量 (x 前 / y 左 / z 上)。"""
        return optical_axis_in_world(self.matrix)

    def facing(self, tol_deg: float = 1e-6) -> str:
        """
        把光轴翻译成人类可读的朝向。

        存在的理由：旧 test_yaw45.py 把 `yaw_real - 45` 标成 "left"，而 Valeo
        约定正 yaw 才向左（数值验证: yaw=+90° 时光轴 = +Y 即左）。命名与物理
        方向相反、日志又把错的方向打给用户看（报告问题 6）。这里让日志直接来自
        几何量，而不是来自变量名。
        """
        ax = self.optical_axis()
        yaw = np.degrees(np.arctan2(ax[1], ax[0]))       # +Y 为左
        elev = np.degrees(np.arcsin(np.clip(ax[2], -1.0, 1.0)))
        if abs(yaw) < tol_deg:
            horiz = "front"
        elif abs(abs(yaw) - 180.0) < tol_deg:
            horiz = "rear"
        elif yaw > 0:
            horiz = f"left {abs(yaw):.1f}deg"
        else:
            horiz = f"right {abs(yaw):.1f}deg"
        vert = "" if abs(elev) < tol_deg else (
            f", {'up' if elev > 0 else 'down'} {abs(elev):.1f}deg")
        return f"{horiz}{vert}"

    def summary(self) -> str:
        return (f"yaw={self.yaw_deg:+.2f} pitch={self.pitch_deg:+.2f} "
                f"roll={self.roll_deg:+.2f} [{self.convention.name}] -> {self.facing()}")


@dataclass
class FisheyeCamera:
    """
    一台已标定相机：投影模型 + 姿态 + 名字。

    不叫 "Camera" 是因为这个库里所有相机（含虚拟）都走同一个类；真实相机的
    特殊之处只有"从标定文件加载"，那属于 vcam.io。
    """

    name: str
    model: CameraModel
    pose: CameraPose = field(default_factory=CameraPose)
    #: 是否虚拟相机。生成的地图/元数据里会带这个标记。
    is_virtual: bool = False

    # ---- 便捷属性 -----------------------------------------------------------
    @property
    def width(self) -> int:
        return int(self.model.width)

    @property
    def height(self) -> int:
        return int(self.model.height)

    def pixel_grid(self, dtype=np.float64):
        """(u, v) 像素坐标网格，indexing='ij'，行=v、列=u，形状 (H, W)。"""
        v, u = np.meshgrid(np.arange(self.height, dtype=dtype),
                           np.arange(self.width, dtype=dtype), indexing="ij")
        return u, v

    def describe(self) -> dict:
        d = dict(name=self.name, virtual=bool(self.is_virtual))
        d.update(self.model.describe())
        d.update(pose=dict(yaw_deg=self.pose.yaw_deg, pitch_deg=self.pose.pitch_deg,
                           roll_deg=self.pose.roll_deg,
                           convention=self.pose.convention.name),
                 facing=self.pose.facing())
        if self.pose.translation is not None:
            d["translation_m"] = list(self.pose.translation)
        return d

    def summary(self) -> str:
        k = f"{self.width}x{self.height}"
        tag = "virtual" if self.is_virtual else "real"
        return f"[{tag}] {self.name}: {self.model.model_name} {k}, {self.pose.summary()}"

    def __repr__(self):
        return f"FisheyeCamera({self.summary()})"


class VirtualCamera:
    """虚拟相机工厂。返回的都是 FisheyeCamera(is_virtual=True)。"""

    @staticmethod
    def from_fov(width, height, fov_h_deg, fov_v_deg=None, square_pixels=False,
                 name="virtual", model="pinhole", pose=None, **model_kwargs):
        """
        按视场角构造。fov_v_deg 省略时由 fov_h_deg 与宽高比推出。

        square_pixels=True 时用 fov_h/fov_v 的 tan 比值反推宽度，保证 fx==fy
        （旧脚本的 --square_pixels 行为）。注意 150°x120° 在 640x480 下
        fx≈85.7 / fy≈138.6，像素是**非方形**的；remap 本身没问题，但如果下游
        假设方形像素（例如按 fx 统一换算角度）就会出错，所以这里显式提醒。
        """
        fov_h = float(fov_h_deg)
        if not (0.0 < fov_h < 179.99):
            raise ValueError(f"fov_h_deg 必须在 (0, 180) 内，实际 {fov_h}")
        tan_h = np.tan(np.radians(fov_h / 2.0))
        if fov_v_deg is None:
            fov_v = np.degrees(2.0 * np.arctan(tan_h * float(height) / float(width)))
        else:
            fov_v = float(fov_v_deg)
            if not (0.0 < fov_v < 179.99):
                raise ValueError(f"fov_v_deg 必须在 (0, 180) 内，实际 {fov_v}")

        tan_v = np.tan(np.radians(fov_v / 2.0))
        if square_pixels:
            width = int(round(float(height) * tan_h / tan_v))
        width, height = int(width), int(height)
        fx = (width / 2.0) / tan_h
        fy = (height / 2.0) / tan_v

        from ..models.registry import normalize_model_name, get_model_class
        canonical = normalize_model_name(model)
        if canonical == "pinhole":
            from ..models.pinhole import PinholeModel
            m = PinholeModel(width, height, fx=fx, fy=fy,
                             cx=(width - 1.0) / 2.0 if "cx" not in model_kwargs else model_kwargs.pop("cx"),
                             cy=(height - 1.0) / 2.0 if "cy" not in model_kwargs else model_kwargs.pop("cy"),
                             **model_kwargs)
        else:
            cls = get_model_class(canonical)
            m = cls(width, height, fx=fx, fy=fy, **model_kwargs)
        return FisheyeCamera(name=name, model=m,
                             pose=pose or CameraPose(), is_virtual=True)

    @staticmethod
    def inherit_from(src: FisheyeCamera, name=None, pose=None) -> FisheyeCamera:
        """
        复制源相机的模型（同一套内参），可选换姿态。用于"同镜头、不同朝向"。
        姿态默认继承源相机的 yaw/pitch/roll —— 这是修复报告问题 8 的落点：
        旧脚本把虚拟 pitch 硬编码成 -10°，而真实鱼眼 pitch 若为 +2°，
        则虚拟中心视角与真实光轴差 12°，且没有任何提示。
        """
        return FisheyeCamera(name=name or f"{src.name}_virtual",
                             model=src.model,
                             pose=pose or src.pose,
                             is_virtual=True)
