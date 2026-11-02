"""
修正上一版对"后半球"的错误断言。

时间线（写下来免得下次又改回去）：
  1. 第一版把鱼眼 project 限制在 theta < 90°（`_visible(Z)`），理由是"鱼眼只
     看前半球"。测试断言"后半球返回 -1"。
  2. 真实数据打脸：WoodScape FV 的 rho(theta) 在 [0, pi] 上单调递增，
     rho(90°)=598 px、rho(103.5°)=~720 px，而图像半径有 700+ px。也就是说
     **图像里本来就有 theta>90° 的像素**。限制半球把这些合法像素判成无效，
     实测 2000 个图像内像素里有 370 个往返失败。
  3. 结论：鱼眼模型只按**图像边界**判有效，不限制半球。这正是报告问题 5
     批评的"用常数/角度截断代替图像边界"的同一类错误，我自己重犯了一次。
  4. 现在：这三个模型的行为是"图像内有值、图像外 -1"。
"""

import numpy as np
import pytest

from vcam.models import EnhancedFThetaModel, KannalaBrandtModel, RadialPolyModel

FS_K = [339.749, -31.988, 48.275, -7.201]


def test_back_hemisphere_inside_image_is_kept():
    """
    theta > 90° 但 rho 仍落在图像内 -> 必须返回有效像素，而不是 -1。

    FV 实测（k=[339.749,-31.988,48.275,-7.201], cx=643.9, cy=479.9）:
        rho(90°)=598.0  rho(95°)=641.0  rho(100°)=685.4
    主点到左边界的距离是 643.9，所以 95° 恰好还能落在图像左边缘内，100° 就
    出界了。也就是说这台 FV 的实际覆盖到约 95° 入射角 —— 这正是"用图像边界
    判有效、不用 90° 常数截断"的意义。
    """
    m = RadialPolyModel(1280, 966, FS_K, cx=643.942, cy=479.907)
    th = np.radians(95.0)
    dirs = np.array([[-np.sin(th), 0.0, np.cos(th)]])   # 朝图像左侧
    u, v = m.project(dirs[:, 0], dirs[:, 1], dirs[:, 2])
    assert u[0] != -1.0, u
    res = m.unproject(u, v)
    assert np.max(np.abs(res.ray - dirs)) < 1e-6
    assert abs(res.theta_deg[0] - 95.0) < 1e-4


def test_theta_beyond_image_extent_is_invalid():
    """rho 超出主点到边界距离时（这里 100° -> 685 px > 644 px）必须 -1。"""
    m = RadialPolyModel(1280, 966, FS_K, cx=643.942, cy=479.907)
    th = np.radians(100.0)
    dirs = np.array([[-np.sin(th), 0.0, np.cos(th)]])
    u, v = m.project(dirs[:, 0], dirs[:, 1], dirs[:, 2])
    assert u[0] == -1.0 and v[0] == -1.0


def test_ninety_degree_boundary_is_in_image():
    """恰 90°: rho(90°)=598 < 640，必须在图像内。"""
    m = RadialPolyModel(1280, 966, FS_K, cx=643.942, cy=479.907)
    u, v = m.project(np.array([1.0]), np.zeros(1), np.array([0.0]))
    assert u[0] != -1.0
    assert abs(u[0] - (643.942 + 598.01)) < 0.5


@pytest.mark.parametrize("model_factory,theta_deg", [
    (lambda: KannalaBrandtModel(1280, 966, fx=339.7, cx=643.9, cy=479.9), 92.0),
    (lambda: EnhancedFThetaModel(1280, 966, [339.7, -31.0, 47.0, -7.0],
                                 cx=643.9, cy=479.9), 92.0),
])
def test_other_fisheye_models_also_keep_back_hemisphere(model_factory, theta_deg):
    """
    同一条语义对 KB 与 e-ftheta 成立：看图不看半球。
    入射角必须挑在"rho 还在图像内"的范围（这两套参数在 92° 时 rho≈600，
    主点到左边界 643.9，够用；100° 的 rho≈683 就已出界）。
    """
    m = model_factory()
    th = np.radians(theta_deg)
    dirs = np.array([[-np.sin(th), 0.0, np.cos(th)]])
    u, v = m.project(dirs[:, 0], dirs[:, 1], dirs[:, 2])
    assert u[0] != -1.0, (u, m._rho(th))
    res = m.unproject(u, v)
    assert np.max(np.abs(res.ray - dirs)) < 1e-6


def test_pinhole_still_rejects_back_hemisphere():
    """
    pinhole 是**真**的只能看前半球（除法 x = X/Z），这条语义不变：
    Z<=0 必须 -1，与鱼眼区分开。
    """
    from vcam.models import PinholeModel
    m = PinholeModel(640, 480, fx=300, cx=320, cy=240)
    u, v = m.project(np.array([1.0, -1.0]), np.zeros(2), np.array([0.0, -1.0]))
    assert (u == -1.0).all() and (v == -1.0).all()
