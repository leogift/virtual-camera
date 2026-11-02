"""
IO 层测试：真实 NFOCC/WoodScape 标定文件 + 旧 convert.json 格式。

真实数据的路径通过环境变量 NFOCC_ROOT 覆盖；找不到时跳过并明确说明，
不静默通过。
"""

import json
import os

import numpy as np
import pytest

from vcam.io import (NFOCCDataset, camera_from_calibration, parse_extrinsic,
                     parse_flat_intrinsic, parse_woodsccape_intrinsic)

NFOCC_ROOT = os.environ.get("NFOCC_ROOT", "/Volumes/Workspace/NFOCC/datasets/fisheye")
HAS_DATA = os.path.isdir(os.path.join(NFOCC_ROOT, "calibration"))
needs_data = pytest.mark.skipif(not HAS_DATA, reason=f"未找到数据集 {NFOCC_ROOT}")


# ---------------------------------------------------------------- 真实数据
@needs_data
def test_dataset_discovers_four_cameras():
    ds = NFOCCDataset(NFOCC_ROOT)
    assert ds.camera_names() == ["FV", "RV", "MVL", "MVR"]
    frames = ds.frames()
    assert all(len(v) == 3 for v in frames.values())


@needs_data
def test_camera_installs_are_physically_plausible():
    """
    安装姿态的物理自检（不是"能跑"）：
      FV 朝前略下俯；RV 朝后下俯；MVL/MVR 几乎对称地朝侧下方俯视 70°+。
    相机高度也应接近（同一车顶/门边高度量级）。
    """
    ds = NFOCCDataset(NFOCC_ROOT)
    cams = {n: ds.camera(n) for n in ds.camera_names()}
    assert abs(cams["FV"].pose.yaw_deg) < 5
    assert 15 < -cams["FV"].pose.pitch_deg < 30
    assert abs(abs(cams["RV"].pose.yaw_deg) - 180) < 5
    assert 30 < -cams["RV"].pose.pitch_deg < 45
    for n in ("MVL", "MVR"):
        assert -cams[n].pose.pitch_deg > 65
    assert cams["MVL"].pose.yaw_deg > 90 > 0 > cams["MVR"].pose.yaw_deg
    heights = [c.pose.translation[2] for c in cams.values()]
    assert max(heights) - min(heights) < 1.5, heights


@needs_data
def test_extrinsic_translation_matches_readme_frame():
    """readme: 平移是相机在车体系下的位置（X 前、Y 左、Z 上）。FV 应在前方。"""
    ds = NFOCCDataset(NFOCC_ROOT)
    fv = ds.camera("FV")
    x, y, z = fv.pose.translation
    assert 3.0 < x < 4.5, x
    assert abs(y) < 0.5, y
    assert 0.3 < z < 1.2, z


@needs_data
def test_camera_model_is_radial_poly_and_roundtrips():
    """
    radial_poly 的像素往返误差 < 1e-4 px（只对图像内的像素成立）。

    注意 -1 哨兵的副作用: 图像**外**的像素反投影得到的射线是正确方向，
    但再正投影回来会因越界被判 -1，所以往返测试不能覆盖整幅矩形以外。
    对虚拟相机来说这是期望行为：源相机看不到的方向就该采不到。
    """
    ds = NFOCCDataset(NFOCC_ROOT)
    cam = ds.camera("FV")
    assert cam.model.model_name == "4thpoly"
    assert len(cam.model.k) == 4
    rng = np.random.default_rng(0)
    u = rng.uniform(0, cam.width - 1, 2000)
    v = rng.uniform(0, cam.height - 1, 2000)
    res = cam.model.unproject(u, v)
    assert res.converged.all()
    u2, v2 = cam.model.project(res.ray[:, 0], res.ray[:, 1], res.ray[:, 2])
    inside = (u2 != -1.0)
    assert inside.all(), f"{int((~inside).sum())} 个图像内像素往返后被判无效"
    assert np.max(np.abs(u2 - u)) < 1e-4, np.max(np.abs(u2 - u))
    assert np.max(np.abs(v2 - v)) < 1e-4


@needs_data
def test_out_of_image_pixel_still_unprojects_to_correct_direction():
    """
    图像外的像素反投影出的射线方向仍然正确 —— 它只是"源相机视场外"的射线，
    虚拟相机采不到就该 -1，而不是方向乱掉（乱掉会让无效区变成错误的彩色）。

    判据：主点正右方大半径处，射线必须严格在相机 XZ 平面内（Y 分量 0）、
    指向图像右（X>0）、且入射角随半径单调增大。实测 rho=2*cx 时
    theta≈103.5°、ray=(-0.801, 0, -0.234)（注意 X 为负说明该方向已越过 90°，
    进入后半球——这是鱼眼的正常覆盖，不是错误）。
    """
    ds = NFOCCDataset(NFOCC_ROOT)
    cam = ds.camera("FV")
    m = cam.model
    thetas = []
    for ratio in (0.5, 1.0, 1.5, 2.0):
        rho = m.cx * ratio
        res = m.unproject(np.array([m.cx + rho]), np.array([m.cy]))
        ray = res.ray[0]
        assert res.converged[0]
        assert abs(ray[1]) < 1e-9                       # 严格水平 (Y 分量 0)
        assert abs(np.linalg.norm(ray) - 1.0) < 1e-12
        thetas.append(res.theta_deg[0])
    assert all(b > a for a, b in zip(thetas, thetas[1:])), thetas
    assert thetas[-1] > 90.0, thetas      # 大半径确实进入后半球


@needs_data
def test_k1_is_focal_scale_not_distortion_coefficient():
    """
    关键约定：radial_poly 的 k1 是"像素/弧度"，即小角度等效焦距（≈ f），
    不是通常意义上的畸变系数。FV 标定 k1=339.749，image 1280 宽，
    半宽 639.5 -> 半视场 atan(639.5/339.749) ≈ 62°。
    """
    ds = NFOCCDataset(NFOCC_ROOT)
    cam = ds.camera("FV")
    k1 = cam.model.focal_small_angle()
    assert 300 < k1 < 400, k1
    half_fov_h = np.degrees(np.arctan((cam.width / 2) / k1))
    assert 55 < half_fov_h < 70, half_fov_h


@needs_data
def test_different_cameras_have_different_calibration():
    """MVL/MVR 不是同一份标定拷贝，系数必须不同（否则后面所有比对都无效）。"""
    ds = NFOCCDataset(NFOCC_ROOT)
    mvl, mvr = ds.camera("MVL"), ds.camera("MVR")
    assert not np.allclose(mvl.model.k, mvr.model.k)
    assert mvl.model.cx != mvr.model.cx


# ---------------------------------------------------------------- 格式解析
def test_woodsccape_missing_field_raises_instead_of_defaulting():
    """旧实现缺 width 会落到 1920 默认值；这里必须报错。"""
    intr = dict(height=966.0, k1=339.7, k2=-31.9, k3=48.2, k4=-7.2,
                cx_offset=0.0, cy_offset=0.0, model="radial_poly")
    with pytest.raises(KeyError) as e:
        parse_woodsccape_intrinsic(intr)
    assert "width" in str(e.value)


def test_woodsccape_poly_order_mismatch_raises():
    intr = dict(width=1280.0, height=966.0, k1=339.7, k2=-31.9, k3=48.2, k4=-7.2,
                cx_offset=0.0, cy_offset=0.0, model="radial_poly", poly_order=5)
    with pytest.raises(KeyError) as e:
        parse_woodsccape_intrinsic(intr)
    assert "k5" in str(e.value)


def test_flat_eftheta_parses_affine_terms():
    cfg = dict(cameraModel="eftheta", width_px=1920, height_px=1536,
               center_x_px=960, center_y_px=768, focal_x_px=340, focal_y_px=340,
               k1=340.0, k2=-2.0, k3=1.0, k4=-0.2, k5=0.0,
               elem_c=1.0, elem_d=0.01, elem_e=-0.02)
    m = parse_flat_intrinsic(cfg)
    assert m.model_name == "eftheta"
    assert m.c == 1.0 and m.d == 0.01 and m.e == -0.02
    assert m.width == 1920


def test_flat_missing_model_raises():
    cfg = dict(width_px=640, height_px=480, focal_x_px=500, k1=0.0)
    with pytest.raises(KeyError) as e:
        parse_flat_intrinsic(cfg)
    assert "model" in str(e.value).lower() or "模型" in str(e.value)


def test_nested_extrinsic_matrix_rejected_loudly():
    """报告 S3: 嵌套 [[..],[..]] 不能被静默忽略。16 数的一行写法应当可用。"""
    flat = dict(camera_external=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    info = parse_extrinsic(flat)
    assert np.allclose(info.R, np.eye(3))
    with pytest.raises(ValueError):
        parse_extrinsic(dict(camera_external=[[1, 0, 0], [0, 1, 0]]))


def test_extrinsic_ypr_matches_matrix():
    from vcam.core.rotation import ypr_to_rotation
    info = parse_extrinsic({"yaw": 12.0, "pitch": -3.0, "roll": 5.0})
    assert np.allclose(info.R, ypr_to_rotation(12, -3, 5))
    assert info.source == "ypr-deg"


def test_extrinsic_rad_and_zxz_paths():
    info_r = parse_extrinsic({"rot_yaw_rad": np.radians(10.0)})
    assert abs(info_r.rpy_deg[0] - 10.0) < 1e-9
    info_z = parse_extrinsic({"rotx": 0.0, "rotz1": 0.0, "rotz2": 180.0})
    assert info_z.source == "zxz-euler"


def test_legacy_convert_json_is_missing_file_error():
    from vcam.io import load_convert_json
    with pytest.raises(FileNotFoundError):
        load_convert_json("/nonexistent/convert.json")


def test_legacy_convert_json_reports_missing_intrinsics(tmp_path):
    from vcam.io import load_convert_json
    p = tmp_path / "convert.json"
    p.write_text(json.dumps({"cameras": [{"name": "c0", "src_intrinsic": {}}]}))
    with pytest.raises(KeyError):
        load_convert_json(str(p))
