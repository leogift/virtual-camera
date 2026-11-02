"""
负对照：把报告里指出的**旧实现写法**原样搬回来，断言测试确实会抓住它们。

没有这组用例，"测试通过"只说明测试没报错。每条都是"旧代码 -> 必须被检测到"。
注意：这里的负对照本身也要能失败。凡是断言"旧写法有问题"的用例，都先把旧
写法的输出打印/量化出来，确认差异是真实数量级，而不是我记错了旧代码。
"""

import os
import tempfile

import numpy as np
import pytest

from vcam.core.rotation import get_convention
from vcam.models import PinholeModel


def old_camera_decompose(R):
    """旧 mat_to_yaw_pitch_roll 的 camera 分支（原文照抄）。"""
    pitch = np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))
    roll = np.arctan2(R[1, 0], R[1, 1])
    yaw = np.arctan2(R[0, 2], R[2, 2])
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


def test_old_camera_decompose_is_actually_broken():
    """若这条不再是 11.82°，说明"旧写法"被我记错了，后面所有对照都无意义。"""
    conv = get_convention("camera")
    R = conv.to_matrix(30.0, 20.0, 10.0)
    y, p, r = old_camera_decompose(R)
    assert abs(p - 11.8221) < 1e-3, p


def test_new_camera_decompose_passes_where_old_fails():
    conv = get_convention("camera")
    y, p, r = 30.0, 20.0, 10.0
    R = conv.to_matrix(y, p, r)
    assert np.allclose(conv.from_matrix(R), (y, p, r), atol=1e-9)
    assert not np.allclose(old_camera_decompose(R), (y, p, r), atol=1e-3)


def test_old_derivative_guard_flips_sign():
    """报告问题 2: np.where(|d|<1e-9, 1e-9, d) 会把负的微小导数翻成正号。"""
    d = np.array([-1e-12, +1e-12, -1e-3])
    old = np.where(np.abs(d) < 1e-9, 1e-9, d)
    new = np.maximum(np.abs(d), 1e-9) * np.sign(d)
    assert old[0] > 0 and new[0] < 0
    assert np.all(np.sign(new) == np.sign(d))


def test_kb_old_derivative_guard_double_counts_at_negative_epsilon():
    """
    报告问题 3 的机制: 分母是 f' + sign(f'+1e-12)*1e-8。
    f' = -1e-12 时 sign 仍为 -1，分母 = -1e-12 - 1e-8 ≈ -1e-8（被本身量级放大
    了一万倍）；而 f' = +1e-12 时分母 = +1e-12 + 1e-8 ≈ +1e-8。看起来有下界，
    但 f' 从 -1e-3 增加到 -2e-12 时分母从 -1.000001e-3 突然坍到 -1e-8，
    步长被放大 1e5 倍 —— 这才是"步长爆炸"的真实路径，不是符号翻转。
    """
    fp = np.array([-1e-3, -1e-6, -1e-11, -2e-12])
    denom = fp + np.sign(fp + 1e-12) * 1e-8
    step_scale = 1.0 / np.abs(denom)
    assert step_scale[-1] / step_scale[0] > 1e4      # 放大一万倍以上

    new = np.maximum(np.abs(fp), 1e-8) * np.sign(fp)
    new_scale = 1.0 / np.abs(new)
    assert new_scale.max() / new_scale.min() < 1e6


def test_old_pinhole_iteration_adopts_unconverged_values():
    """
    报告问题 4: 旧实现迭代若干次后无论是否收敛都直接采纳。
    用迭代次数 2 + 强桶形构造不收敛条件：新实现返回 converged=False 的像素，
    且把这些像素正投影回去能看到明显残差。
    """
    m = PinholeModel(640, 480, fx=200, fy=200, cx=320, cy=240,
                     k1=-0.9, k2=0.0, max_iter=2, converge_px=1e-9)
    u, v = np.meshgrid(np.arange(0, 640, 32.0), np.arange(0, 480, 32.0))
    res = m.unproject(u, v)
    bad = ~res.converged
    assert bad.any(), "构造的发散条件没有触发未收敛标记"
    u2, v2 = m.project(res.ray[..., 0], res.ray[..., 1], res.ray[..., 2])
    resid = np.hypot(np.where(bad, u2 - u, 0.0), np.where(bad, v2 - v, 0.0))
    assert resid.max() > 1.0, "未收敛像素的残差应当可见"


def test_pinhole_does_not_hard_cut_wide_fov():
    """
    报告问题 5: 旧代码 r2 < 50 的截断约等于 164° 全视场。
    本实现不按常数截断：170° 视场的理想针孔，图像范围内的像素全部保留，
    只有真正超出图像边界或后半球的方向才返回 -1。
    """
    half_fov = 85.0
    fx = 100.0
    w = int(2 * fx * np.tan(np.radians(half_fov))) + 1
    m = PinholeModel(w, w, fx=fx, fy=fx, cx=(w - 1) / 2, cy=(w - 1) / 2)

    # 角方向的归一化坐标 = 主点到边缘的像素数 / fx。
    # 注意: 这里必须用 像素数/fx，不能把像素偏移直接当光线分量 —— 我第一版
    # 就是这么写错的（把 1143 px 当 s，得 s=1143 而不是 11.43），结果四个角
    # 全部超出图像被判 -1，被误读成实现 bug。
    s_max = m.cx / fx
    u, v = m.project(np.array([s_max, -s_max]), np.array([s_max, -s_max]), np.ones(2))
    assert (u >= 0).all() and (v >= 0).all(), (u, v)
    theta_corner = np.degrees(np.arctan(np.hypot(s_max, s_max)))
    assert theta_corner > 82.0, theta_corner
    xu, yu = s_max, s_max
    assert xu * xu + yu * yu > 50.0, "旧的 r2<50 判据会把这些角像素判无效"

    # 超出图像边界必须无效, 而不是返回一个大于宽度的正数
    u_out, _ = m.project(np.array([s_max * 3.0]), np.zeros(1), np.ones(1))
    assert u_out[0] == -1.0
    # 后半球必须无效
    u90, _ = m.project(np.array([1.0]), np.zeros(1), np.array([0.0]))
    assert u90[0] == -1.0


def test_old_maps_to_grid_divides_by_zero():
    """报告问题 7: sw==1 时 2x/(sw-1) 除零。"""
    map_x = np.zeros((1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        old = 2.0 * map_x / (1 - 1) - 1.0
    assert not np.isfinite(old).all()

    from vcam.compat import maps_to_grid
    assert np.isfinite(maps_to_grid(map_x, map_x, 1, 1)).all()


def test_old_savetxt_writes_inf_that_loadtxt_rejects():
    """报告 S6: 写出 'inf' 会让 np.loadtxt 失败；新实现先 nan_to_num。"""
    arr = np.array([[np.inf, np.nan], [1.0, 2.0]])
    with tempfile.TemporaryDirectory() as d:
        p_old = os.path.join(d, "old.txt")
        np.savetxt(p_old, arr, fmt="%f")
        with open(p_old) as f:
            assert "inf" in f.read()

        p_new = os.path.join(d, "new.txt")
        np.savetxt(p_new, np.nan_to_num(arr, nan=-1.0, posinf=-1.0, neginf=-1.0),
                   fmt="%f")
        assert np.isfinite(np.loadtxt(p_new)).all()


def test_radial_poly_solver_handles_negative_k1():
    """
    报告 S2: k1<=0 时旧初值 r_d/(k1+1e-8) 得到负值并被 clip 到 0，
    牛顿法停在 0。本实现用"k1 退化 -> 二分"来兜底。

    注意"根是否正确"只能按**残差**判定：当多项式在 [0,pi] 上非单调时，
    同一个 rho 可能有多个 theta 解（下面的 k 在 theta≈0.089 处有一个转折），
    此时"my θ == reference θ"不是合法断言 —— 这是我第一次写这条时犯的错。
    """
    from vcam.models.radial_poly import solve_poly_theta
    k = [-0.5, 3.0, -1.5, 0.3]
    th_ref = np.linspace(0.05, 1.2, 100)
    rho = np.abs(sum(c * th_ref ** (i + 1) for i, c in enumerate(k)))
    th, ok = solve_poly_theta(rho, k)
    assert ok.all(), f"{int((~ok).sum())} 个点未收敛"
    resid = np.abs(sum(c * th ** (i + 1) for i, c in enumerate(k)) - rho)
    assert resid.max() < 1e-6, resid.max()


def test_naive_initial_guess_would_fail_on_same_case():
    """
    把旧初值 + 无二分兜底的简化牛顿法写出来，证明上一条用例确实是"有难度"的：
    它在 k1<0 的输入上要么停住、要么给出残差很大的值。
    """
    k = [-0.5, 3.0, -1.5, 0.3]

    def poly(t):
        return sum(c * t ** (i + 1) for i, c in enumerate(k))

    def dpoly(t):
        return sum((i + 1) * c * t ** i for i, c in enumerate(k))

    rho = abs(poly(0.6))
    theta = float(np.clip(rho / (k[0] + 1e-8), 0.0, np.pi))   # 旧初值 -> 负 -> clip 到 0
    assert theta == 0.0
    for _ in range(12):
        f = poly(theta) - rho
        d = dpoly(theta) or 1e-9
        theta = float(np.clip(theta - f / d, 0.0, np.pi))
        if abs(poly(theta) - rho) < 1e-6:
            break
    assert abs(poly(theta) - rho) > 1e-3, "旧式牛顿法本应无法收敛"
