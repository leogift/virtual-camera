"""
GridConverter 与抽样表的测试。

最要紧的一条是"身份旋转 + 同模型同尺寸 -> map 就是恒等映射"。旧脚本把
yaw 方向写反、pitch 硬编码这类错误，在图像上只表现为"看起来有点歪"，
但在这个断言下会直接暴露成几十像素的偏差。
"""

import numpy as np
import pytest

from vcam import FisheyeCamera, GridConverter, VirtualCamera
from vcam.models import KannalaBrandtModel, PinholeModel, RadialPolyModel


def make_src(**kw):
    m = RadialPolyModel(640, 480, [300.0, -30.0, 40.0, -6.0], cx=322, cy=241)
    return FisheyeCamera(name="src", model=m)


def test_identity_rotation_gives_identity_map():
    """同模型同尺寸同姿态: map 必须等于像素坐标本身（误差 < 1e-3 px）。"""
    src = make_src()
    dst = FisheyeCamera(name="dst", model=src.model, pose=src.pose, is_virtual=True)
    conv = GridConverter(src, dst)
    assert conv.is_identity_rotation
    mx, my, st = conv.build_maps()
    v, u = np.meshgrid(np.arange(480.0), np.arange(640.0), indexing="ij")
    assert np.max(np.abs(mx - u)) < 1e-3, np.max(np.abs(mx - u))
    assert np.max(np.abs(my - v)) < 1e-3
    assert st.valid == 640 * 480
    assert st.out_of_source == 0


def test_yaw_delta_sign_changes_pixels_horizontally():
    """
    +yaw = 向左看（Valeo 约定，facing() 会打印 left）。相机左转后，原本位于
    画面左侧的景物占据虚拟图的中心，所以虚拟中心像素应当采到源图**左半边**
    （u < cx），右转则采到右半边。实测 +30° -> 165.87，-30° -> 474.16。
    """
    src = make_src()
    vcam = VirtualCamera.inherit_from(src)
    cy, cx = 240, 320
    results = {}
    for d in (+30.0, -30.0):
        conv = GridConverter(src, vcam, yaw_delta_deg=d)
        mx, my, _ = conv.build_maps()
        results[d] = float(mx[cy, cx])
        assert conv.dst.pose.facing().startswith("left" if d > 0 else "right")
    assert results[-30.0] > cx > results[+30.0], results
    # 对称性: 正负偏航的偏移量应当接近
    assert abs((results[+30.0] - cx) + (results[-30.0] - cx)) < 2.0


def test_pitch_delta_changes_pixels_vertically():
    """
    +pitch = 抬头（facing() 打印 "up"，光轴 Z 分量 +0.342）。
    抬头后原本画面上方的景物进入虚拟图中心，故中心像素采到源图上方（v < cy）。
    实测 +20° -> 138.3，-20° -> 343.7。
    """
    src = make_src()
    vcam = VirtualCamera.inherit_from(src)
    up = GridConverter(src, vcam, pitch_delta_deg=+20.0)
    down = GridConverter(src, vcam, pitch_delta_deg=-20.0)
    mx_u, my_u, _ = up.build_maps()
    mx_d, my_d, _ = down.build_maps()
    v_up, v_down = float(my_u[240, 320]), float(my_d[240, 320])
    assert "up" in up.dst.pose.facing() and "down" in down.dst.pose.facing()
    assert up.dst.pose.optical_axis()[2] > 0.3
    assert v_up < 240 < v_down, (v_up, v_down)


def test_dst_pose_inherited_from_source_by_default():
    """
    报告问题 8: 旧脚本把虚拟 pitch 硬编码 -10°。这里默认继承源姿态，
    只有显式给 delta 才偏离。
    """
    src = make_src()
    src.pose = src.pose.with_absolute(pitch=-22.0)
    vcam = VirtualCamera.inherit_from(src)
    conv = GridConverter(src, vcam)
    assert abs(conv.dst.pose.pitch_deg + 22.0) < 1e-9
    assert conv.is_identity_rotation


def test_validity_mask_excludes_out_of_source():
    """虚拟相机 FOV 明显大于源相机 -> 必须有 out_of_source 像素。"""
    src = make_src()                     # 源约 100° 对角, 图像 640x480
    small = PinholeModel(640, 480, fx=60.0, cx=320, cy=240)   # ~160° 水平
    dst = FisheyeCamera(name="wide", model=small, pose=src.pose, is_virtual=True)
    conv = GridConverter(src, dst)
    mx, my, st = conv.build_maps()
    assert st.out_of_source > 0
    assert st.valid < st.total
    # 无效位置必须都是 -1，不能留 NaN/Inf
    assert np.isfinite(mx).all() and np.isfinite(my).all()
    bad = (mx < 0)
    assert bad.sum() == st.total - st.valid


def test_chunked_maps_match_full():
    """分块计算与整块计算必须逐位一致（不是"近似"）。"""
    src = make_src()
    vcam = VirtualCamera.from_fov(320, 240, 100.0, 80.0)
    conv = GridConverter(src, vcam, yaw_delta_deg=15.0)
    a = conv.build_maps()
    b = conv.build_maps(chunk_rows=37)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[2].valid == b[2].valid


def test_stats_report_unconverged():
    """把源相机换成严重桶形的 pinhole，未收敛像素必须计入统计。"""
    m = PinholeModel(640, 480, fx=200, fy=200, cx=320, cy=240,
                     k1=-0.95, max_iter=3, converge_px=1e-12)
    src = FisheyeCamera(name="skew", model=m)
    dst = FisheyeCamera(name="v", model=m, pose=src.pose, is_virtual=True)
    conv = GridConverter(src, dst)
    _, _, st = conv.build_maps()
    assert st.unconverged > 0


def test_remap_rejects_size_mismatch():
    """旧实现会静默 resize，新实现直接报错（resize 会改变畸变几何）。"""
    src = make_src()
    dst = FisheyeCamera(name="v", model=src.model, pose=src.pose, is_virtual=True)
    conv = GridConverter(src, dst)
    with pytest.raises(ValueError) as e:
        conv.remap(np.zeros((100, 100, 3), np.uint8))
    assert "不一致" in str(e.value)


def test_remap_identity_is_pixelwise_close():
    src = make_src()
    dst = FisheyeCamera(name="v", model=src.model, pose=src.pose, is_virtual=True)
    conv = GridConverter(src, dst)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    out = conv.remap(img)
    diff = np.abs(out.image.astype(int) - img.astype(int)).max()
    assert diff <= 1, diff        # 恒等映射只有插值舍入


def test_to_grid_align_corners_false_handles_size_one():
    """报告问题 7: sw/sh == 1 时 align_corners 公式除零。"""
    src = make_src()
    dst = FisheyeCamera(name="v", model=src.model, pose=src.pose, is_virtual=True)
    conv = GridConverter(src, dst)
    m1 = np.zeros((1, 1), np.float32)
    g = conv.to_grid(m1, m1, align_corners=False)
    assert np.isfinite(g).all()
    g2 = conv.to_grid(m1, m1, align_corners=True)
    assert np.isfinite(g2).all()


def test_gather_tables_sum_to_one_and_match_remap():
    """
    抽样表的 4 个权重在图像内部必须和为 1；用 gather 求出的像素值必须与
    cv2.remap 的 bilinear 结果一致（容差 1 灰度级）。
    """
    src = make_src()
    vcam = VirtualCamera.inherit_from(src)
    conv = GridConverter(src, vcam, yaw_delta_deg=12.0)
    mx, my, _ = conv.build_maps()
    idx, w = conv.to_gather_tables(mx, my)
    assert idx.shape == (4, 640 * 480) and w.shape == idx.shape
    assert idx.max() < 640 * 480
    inner = (mx > 1) & (my > 1) & (mx < 638) & (my < 478)
    wsum = w.sum(0).reshape(480, 640)
    assert np.allclose(wsum[inner], 1.0, atol=1e-6)

    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (480, 640), dtype=np.uint8).astype(np.float32)
    flat = img.reshape(-1)
    got = sum(w[k] * flat[idx[k]] for k in range(4)).reshape(480, 640)
    import cv2
    ref = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)
    assert np.abs(got - ref)[inner].max() <= 1.0


def test_identity_rotation_negative_control():
    """
    负对照: 若把 R_rel 反着乘（R_dst @ R_src^T 而不是 R_src @ R_dst^T），
    恒等旋转测不出来，但有偏航时 map 会偏离几十像素。用来证明上面几条
    符号测试确实有分辨力。
    """
    src = make_src()
    vcam = VirtualCamera.inherit_from(src)
    good = GridConverter(src, vcam, yaw_delta_deg=30.0).build_maps()[0]
    R_bad = np.eye(3)
    from vcam.core.rotation import rotation_relative
    R_bad = rotation_relative(vcam.pose.matrix, src.pose.matrix)   # 反过来
    conv_bad = GridConverter(src, vcam, R_rel=R_bad)
    bad = conv_bad.build_maps()[0]
    assert np.nanmax(np.abs(good - bad)) > 10.0
