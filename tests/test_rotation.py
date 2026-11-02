"""
旋转与欧拉角的回归测试。

包含报告 S7 要求的"构造-分解往返一致性"，这是能自动抓住旧代码 camera 约定
Bug 的那类测试。同时用**闭式期望值**校验符号含义，避免"往返一致但定义反了"
这种更隐蔽的错误：往返只证明自洽，不证明朝向对。
"""

import numpy as np
import pytest

from vcam.core.rotation import (PRE_ROT_VALEO, get_convention,
                                optical_axis_in_world, quaternion_cam2veh_to_rt,
                                quaternion_to_rotation, rotation_relative,
                                zxz_to_rotation)

CONV_NAMES = ["valeo", "camera"]
GRID = [(y, p, r)
        for y in (-179.0, -90.0, -33.0, 0.0, 17.0, 90.0, 179.0)
        for p in (-89.0, -45.0, 0.0, 20.0, 60.0)
        for r in (-170.0, -10.0, 0.0, 45.0, 150.0)]


@pytest.mark.parametrize("name", CONV_NAMES)
def test_roundtrip_matrix(name):
    conv = get_convention(name)
    worst = max(conv.roundtrip_error(*g) for g in GRID)
    assert worst < 1e-12, f"{name} 往返误差 {worst:.3e}"


@pytest.mark.parametrize("name", CONV_NAMES)
def test_roundtrip_angles(name):
    """角度本身也要回来（矩阵一致但角度跳 360° 的情况下仍要报出来）。"""
    conv = get_convention(name)
    y, p, r = 30.0, 20.0, 10.0
    got = conv.from_matrix(conv.to_matrix(y, p, r))
    assert np.allclose(got, (y, p, r), atol=1e-9), got


def test_camera_convention_old_bug_is_gone():
    """
    旧实现: camera 约定 (30,20,10) 分解回 (33.75, 11.82, 19.01)。
    这个用例钉住"不再出现 11.82"这一事实。
    """
    conv = get_convention("camera")
    y, p, r = conv.from_matrix(conv.to_matrix(30.0, 20.0, 10.0))
    assert abs(p - 20.0) < 1e-9
    assert abs(p - 11.8221) > 1.0


@pytest.mark.parametrize("name", CONV_NAMES)
def test_rotation_matrix_is_orthonormal(name):
    conv = get_convention(name)
    for g in GRID[::7]:
        R = conv.to_matrix(*g)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_valeo_positive_yaw_turns_left():
    """Valeo 正 yaw = 向左（车体 +Y）。报告问题 6 的命名错误由这条钉死。"""
    for yaw, expect_y in ((45.0, +1), (-45.0, -1), (90.0, +1), (0.0, None)):
        ax = optical_axis_in_world(get_convention("valeo").to_matrix(yaw, 0, 0))
        if expect_y is not None:
            assert np.sign(ax[1]) == expect_y, (yaw, ax)
        assert ax[0] > 0.0 - 1e-6 or abs(yaw) > 89.9


def test_valeo_zero_pose_points_forward():
    """
    零姿态光轴 = 车体 +X（前），且相机 +X 指向车体 -Y（右）、+Y 指向 -Z（下）。
    注意 R 是 world->cam，所以"相机轴在车体里指向哪"要用 R.T @ e。
    """
    R = get_convention("valeo").to_matrix(0, 0, 0)
    assert np.allclose(R.T @ np.array([0, 0, 1.0]), [1.0, 0.0, 0.0])
    assert np.allclose(R.T @ np.array([1.0, 0, 0]), [0.0, -1.0, 0.0])
    assert np.allclose(R.T @ np.array([0, 1.0, 0]), [0.0, 0.0, -1.0])
    assert np.allclose(R, PRE_ROT_VALEO)


def test_valeo_positive_pitch_looks_up():
    R = get_convention("valeo").to_matrix(0, 15.0, 0)
    ax = optical_axis_in_world(R)
    assert ax[2] > 0.2 and abs(ax[1]) < 1e-9


def test_zxz_zero_points_down():
    """
    ZXZ 的 rotx 带 +180°，所以 (0,0,0) 不是前视也不是后视，而是**朝下**
    （光轴 = 车体 -Z）。实测值 [0,-1.2e-16,-1]。这条把那个 π 偏移的后果
    钉住；旧代码没有任何注释说明它，报告 S5 指出无从核对。
    """
    ax = optical_axis_in_world(zxz_to_rotation(0, 0, 0))
    assert np.allclose(ax, [0.0, 0.0, -1.0], atol=1e-12)


def test_zxz_rotx_controls_axis_angle():
    """
    ZXZ 里真正决定光轴的是 rotx：+180° 让光轴从"前"翻到"后"再落到"下"，
    所以 (0,0,0) = 朝下，(0,+90,0) = 朝车体左 (+Y)，(0,-90,0) = 朝右 (-Y)。
    rotz1/rotz2 只绕光轴自身转，不改变光轴指向 —— 这也是"朝下"这个结果
    与它们无关的原因。旧代码那个 π 偏移没有任何注释（报告 S5），
    这里用实测值把它固定下来。
    """
    assert np.allclose(optical_axis_in_world(zxz_to_rotation(0, 0, 0)),
                       [0.0, 0.0, -1.0], atol=1e-12)
    assert np.allclose(optical_axis_in_world(zxz_to_rotation(0, 90, 0)),
                       [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(optical_axis_in_world(zxz_to_rotation(0, -90, 0)),
                       [0.0, -1.0, 0.0], atol=1e-12)
    for rz1, rz2 in ((90, 0), (0, 90), (180, 180)):
        assert np.allclose(optical_axis_in_world(zxz_to_rotation(rz1, 0, rz2)),
                           [0.0, 0.0, -1.0], atol=1e-12)


def test_relative_rotation_identity():
    R = get_convention("valeo").to_matrix(12, -3, 5)
    assert np.allclose(rotation_relative(R, R), np.eye(3), atol=1e-12)


def test_relative_rotation_composition():
    """R_rel @ v_dst 应等于 R_src @ R_dst^T @ v_dst（直接定义核对）。"""
    rng = np.random.default_rng(0)
    conv = get_convention("valeo")
    R_src = conv.to_matrix(20, -10, 3)
    R_dst = conv.to_matrix(65, 5, -2)
    R_rel = rotation_relative(R_src, R_dst)
    v = rng.normal(size=(50, 3))
    assert np.allclose(R_rel @ v.T, (R_src @ R_dst.T) @ v.T)


def test_quaternion_roundtrip_matches_scipy_if_available():
    q = [0.592188, -0.584691, 0.395043, -0.389090]   # 取自 00164_FV.json
    R = quaternion_to_rotation(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        pytest.skip("scipy 不可用")
    assert np.allclose(R, Rotation.from_quat(q).as_matrix(), atol=1e-9)


def test_quaternion_cam2veh_is_transposed():
    """readme 说四元数是 cam->veh，本库要 veh->cam，必须转置。"""
    q = [0.0, 0.0, 0.0, 1.0]
    R_w2c, _ = quaternion_cam2veh_to_rt(q)
    assert np.allclose(R_w2c, np.eye(3))


def test_unknown_convention_raises():
    with pytest.raises(ValueError):
        get_convention("nonsense")


def test_gimbal_lock_does_not_blow_up():
    for name in CONV_NAMES:
        conv = get_convention(name)
        R = conv.to_matrix(40.0, 90.0, 25.0)
        y, p, r = conv.from_matrix(R)
        assert np.isfinite([y, p, r]).all()
        # 万向锁下 yaw 被固定为 0，但矩阵要能重建
        assert np.allclose(conv.to_matrix(y, p, r), R, atol=1e-9)
