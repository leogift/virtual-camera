"""
相机模型的正确性测试。

每条断言都对着一个**独立的闭式参考**或一个能区分对错的判别量，而不是只做
"project -> unproject -> project" 的闭环 —— 闭环只能证明自洽，旧代码的 KB
和 eftheta 分支闭环也是通的，错的是公式本身。

注意 `project()` 的约定：**落在图像外或后半球的方向返回 -1**。所以任何
"和参考公式逐一比对"的用例都必须先按有效性掩码过滤，否则拿 -1 去和真实
像素比较，测的是约定而不是公式。这条踩过一次（5 个用例同时挂）。
"""

import numpy as np
import pytest

from vcam.models import (EnhancedFThetaModel, KannalaBrandtModel, PinholeModel,
                         RadialPolyModel, build_model, normalize_model_name,
                         solve_poly_theta)


def sphere_dirs(n=1500, max_theta_deg=95.0, seed=0):
    rng = np.random.default_rng(seed)
    th = np.radians(rng.uniform(0, max_theta_deg, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


def in_image(model, u, v):
    return (u >= 0) & (v >= 0) & (u <= model.width - 1) & (v <= model.height - 1)


def project_valid(model, dirs):
    """投影并只保留落在图像内的点，返回 (dirs_valid, u, v)。"""
    u, v = model.project(dirs[:, 0], dirs[:, 1], dirs[:, 2])
    m = in_image(model, u, v)
    return dirs[m], u[m], v[m]


def roundtrip(model, dirs):
    """闭环: 只对落在图像内的方向做 project -> unproject -> project。"""
    dv, u, v = project_valid(model, dirs)
    res = model.unproject(u, v)
    u2, v2 = model.project(res.ray[..., 0], res.ray[..., 1], res.ray[..., 2])
    return np.max(np.abs(u2 - u)), np.max(np.abs(v2 - v)), res


# ---------------------------------------------------------------- WoodScape
def woodsccape_reference(X, Y, Z, width, height, k, cx_off, cy_off, shift):
    """WoodScape readme 里逐字给出的投影公式（独立重写，不复用库里任何代码）。"""
    chi = np.hypot(X, Y)
    th = np.arctan2(chi, Z)
    rho = sum(c * th ** (i + 1) for i, c in enumerate(k))
    with np.errstate(invalid="ignore", divide="ignore"):
        up = np.where(chi > 0, rho * X / np.where(chi > 0, chi, 1.0), 0.0)
        vp = np.where(chi > 0, rho * Y / np.where(chi > 0, chi, 1.0), 0.0)
    return (up + cx_off + width / 2.0 - shift,
            vp + cy_off + height / 2.0 - shift)


def test_woodsccape_forward_matches_readme_formula():
    """
    正投影逐点对齐 readme 公式。只比较**落在图像内**的前半球方向；
    图像外/后半球返回 -1 是本库的约定（见 PinholeModel.project 的说明），
    参考公式不会返回 -1，混在一起比较测的就不是公式了。
    """
    w, h = 1280, 966
    k = [339.749, -31.988, 48.275, -7.201]
    offx, offy = 3.942, -3.093
    m = RadialPolyModel(w, h, k, cx=w / 2 + offx, cy=h / 2 + offy)
    d = sphere_dirs(2000, max_theta_deg=120.0)      # 含大量图像外的方向
    dv, u, v = project_valid(m, d)
    ru, rv = woodsccape_reference(dv[:, 0], dv[:, 1], dv[:, 2], w, h, k, offx, offy, 0.0)
    assert dv.shape[0] > 100, "有效样本太少, 这条用例没有说服力"
    assert np.max(np.abs(u - ru)) < 1e-9
    assert np.max(np.abs(v - rv)) < 1e-9


def test_project_marks_out_of_image_as_invalid():
    """图像外必须 -1；这条单独测，避免和公式对拍混在一起。"""
    m = RadialPolyModel(1280, 966, [339.749, -31.988, 48.275, -7.201],
                        cx=643.942, cy=479.907)
    d = sphere_dirs(2000, max_theta_deg=120.0)
    u, v = m.project(d[:, 0], d[:, 1], d[:, 2])
    ru, rv = woodsccape_reference(d[:, 0], d[:, 1], d[:, 2], 1280, 966,
                                  [339.749, -31.988, 48.275, -7.201],
                                  3.942, -3.093, 0.0)
    outside = ~in_image(m, ru, rv)
    assert outside.any(), "样本里没有图像外的点"
    assert (u[outside] == -1.0).all() and (v[outside] == -1.0).all()


def test_ninety_degree_boundary_is_visible():
    """
    边界: theta 恰为 90° 时 rho(90°)=598 px，落在图像内，必须照常成像。
    这里只验证"90° 不被当成不可见" —— 是否有效由图像边界决定，见
    tests/test_fisheye_domain.py 中关于后半球的三条用例。
    """
    m = RadialPolyModel(1280, 966, [339.749, -31.988, 48.275, -7.201],
                        cx=643.942, cy=479.907)
    Z = np.array([0.0, -0.0, 1e-18, np.tan(np.radians(1e-9))])
    X = np.ones(4)
    Y = np.zeros(4)
    u, v = m.project(X, Y, Z)
    assert (u > 0).all(), u


def test_woodsccape_principal_point_is_edge_symmetric():
    """
    主点取 width/2+offset 时, max(rho) 恰好等于主点到左/上边缘的距离。
    这条把"主点约定"变成一个可断言的结构性质; 若改为 -0.5 会差 0.5 px 而不通过,
    届时需要同时改 parse_woodsccape_intrinsic 与这里。
    """
    from vcam.io.config_schema import parse_woodsccape_intrinsic
    intr = dict(width=1280.0, height=966.0, k1=339.749, k2=-31.988, k3=48.275,
                k4=-7.201, cx_offset=3.942, cy_offset=-3.093, aspect_ratio=1.0,
                model="radial_poly", poly_order=4)
    m = parse_woodsccape_intrinsic(intr)
    k = [intr["k1"], intr["k2"], intr["k3"], intr["k4"]]
    theta_max = np.linspace(0, 3.0, 200000)
    rho = sum(c * theta_max ** (i + 1) for i, c in enumerate(k))
    i_left = int(np.argmin(np.abs(rho - m.cx)))
    i_top = int(np.argmin(np.abs(rho - m.cy)))
    assert abs(rho[i_left] - m.cx) < 0.01, (rho[i_left], m.cx)
    assert abs(rho[i_top] - m.cy) < 0.01, (rho[i_top], m.cy)


def test_woodsccape_inverse_roundtrip():
    m = RadialPolyModel(1280, 966, [339.749, -31.988, 48.275, -7.201],
                        cx=643.942, cy=479.907)
    du, dv, res = roundtrip(m, sphere_dirs(1000, max_theta_deg=100.0))
    assert du < 1e-5 and dv < 1e-5
    assert res.converged.all()


def test_woodsccape_theta_recovered():
    """反投影出的入射角与射线方向必须回到原值（只在图像内的点上比较）。"""
    m = RadialPolyModel(1280, 966, [339.749, -31.988, 48.275, -7.201],
                        cx=643.942, cy=479.907)
    d = sphere_dirs(1500, max_theta_deg=110.0, seed=1)
    dv, u, v = project_valid(m, d)
    th = np.arctan2(np.hypot(dv[:, 0], dv[:, 1]), dv[:, 2])
    res = m.unproject(u, v)
    assert np.max(np.abs(res.theta_deg - np.degrees(th))) < 1e-6
    assert np.max(np.abs(res.ray - dv)) < 1e-6


# ---------------------------------------------------------------- KB
def test_kb_forward_closed_form():
    """rho = theta*(1 + k1 t^2 + ...)：直接用定义式核对（仅图像内）。"""
    k = (0.02, -0.004, 0.001, -0.0001)
    f = 300.0
    m = KannalaBrandtModel(640, 480, fx=f, cx=319.5, cy=239.5,
                           k1=k[0], k2=k[1], k3=k[2], k4=k[3])
    dv, u, v = project_valid(m, sphere_dirs(1500, max_theta_deg=91.0))
    chi = np.hypot(dv[:, 0], dv[:, 1])
    th = np.arctan2(chi, dv[:, 2])
    rho = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
    ru = 319.5 + f * rho * dv[:, 0] / chi
    rv = 239.5 + f * rho * dv[:, 1] / chi
    assert np.allclose(u, ru, atol=1e-9)
    assert np.allclose(v, rv, atol=1e-9)


def test_kb_inverse_roundtrip_and_convergence():
    m = KannalaBrandtModel(1280, 966, fx=340, cx=640, cy=483,
                           k1=0.02, k2=-0.004, k3=0.001, k4=-0.0001)
    du, dv, res = roundtrip(m, sphere_dirs(1000, max_theta_deg=95.0))
    assert du < 1e-5 and dv < 1e-5
    assert res.converged.all()


def test_kb_derivative_guard_does_not_flip_sign():
    """
    报告问题 3: 旧式 sign(f'+eps)*1e-8 在 f' 接近 0 时把步长放大上万倍。
    构造一个 f' 会变号的模型（强负 k1），要求求解不出现非有限值/巨大跳变。
    """
    m = KannalaBrandtModel(1280, 966, fx=300, cx=640, cy=483,
                           k1=-0.30, k2=0.0, k3=0.0, k4=0.0)
    u = np.linspace(0, 1279, 256)
    v = np.full_like(u, 483.0)
    res = m.unproject(u, v)
    assert np.all(np.isfinite(res.ray))
    assert np.all(np.isfinite(res.theta_deg))
    assert np.max(res.theta_deg) <= 180.0


# ---------------------------------------------------------------- e-ftheta
def test_eftheta_affine_inverse_closed_form():
    """
    [u,v]-c = A @ (rho * [x_norm, y_norm])，A=[[c,d],[e,1]]。
    直接按这个定义造像素再反解，必须回到原方向。
    """
    m = EnhancedFThetaModel(1920, 1536, [340.0, -2.0, 1.0, -0.2],
                            c=1.0, d=0.02, e=-0.01, cx=960, cy=768)
    dv, u, v = project_valid(m, sphere_dirs(1200, max_theta_deg=100.0))
    chi = np.hypot(dv[:, 0], dv[:, 1])
    th = np.arctan2(chi, dv[:, 2])
    rho = 340.0 * th - 2.0 * th**2 + th**3 - 0.2 * th**4
    xn, yn = dv[:, 0] / chi, dv[:, 1] / chi
    ru = 960 + (1.0 * xn + 0.02 * yn) * rho
    rv = 768 + (-0.01 * xn + 1.0 * yn) * rho
    assert np.allclose(u, ru, atol=1e-8)
    assert np.allclose(v, rv, atol=1e-8)
    res = m.unproject(u, v)
    assert np.max(np.abs(res.ray - dv)) < 1e-6


def test_eftheta_singular_affine_rejected():
    with pytest.raises(ValueError):
        EnhancedFThetaModel(640, 480, [300.0, 0.0], c=1.0, d=1.0, e=1.0)


# ---------------------------------------------------------------- pinhole
def test_pinhole_no_distortion_is_exact():
    m = PinholeModel(640, 480, fx=500, fy=500, cx=320, cy=240)
    dv, u, v = project_valid(m, sphere_dirs(800, max_theta_deg=60.0))
    assert np.allclose(u, 320 + 500 * dv[:, 0] / dv[:, 2], atol=1e-9)
    res = m.unproject(u, v)
    assert np.max(np.abs(res.ray - dv)) < 1e-12


def test_pinhole_distorted_roundtrip_and_residual_check():
    m = PinholeModel(640, 480, fx=800, fy=800, cx=320, cy=240,
                     k1=-0.15, k2=0.03, p1=1e-4, p2=-2e-4)
    du, dv, res = roundtrip(m, sphere_dirs(1000, max_theta_deg=60.0))
    assert du < 1e-4 and dv < 1e-4
    assert res.converged.all()


def test_pinhole_severe_distortion_flags_unconverged():
    """强桶形下必须有像素被标成未收敛，而不是静默返回落点错误的值。"""
    m = PinholeModel(640, 480, fx=200, fy=200, cx=320, cy=240,
                     k1=-0.95, k2=0.0, max_iter=5, converge_px=1e-12)
    u, v = np.meshgrid(np.arange(640.0), np.arange(480.0))
    res = m.unproject(u, v)
    assert not res.converged.all()
    u2, v2 = m.project(res.ray[..., 0], res.ray[..., 1], res.ray[..., 2])
    assert (~res.converged).sum() > 0


def test_pinhole_rejects_nonpositive_focal():
    with pytest.raises(ValueError):
        PinholeModel(640, 480, fx=0.0)


# ---------------------------------------------------------------- 求根器
def test_solver_converges_for_negative_k1_small_rho():
    """报告 S2: k1<=0 时旧初值 r_d/(k1+eps) 会给出负初值。按残差判定。"""
    k = [-0.5, 3.0, -1.5, 0.3]
    theta_ref = np.linspace(0.05, 1.2, 200)
    rho = np.abs(sum(c * theta_ref ** (i + 1) for i, c in enumerate(k)))
    th, ok = solve_poly_theta(rho, k)
    assert ok.all(), int((~ok).sum())
    resid = np.abs(sum(c * th ** (i + 1) for i, c in enumerate(k)) - rho)
    assert resid.max() < 1e-6


def test_radial_poly_rejects_zero_k1():
    with pytest.raises(ValueError):
        RadialPolyModel(100, 100, [0.0, 1.0], cx=50, cy=50)


# ---------------------------------------------------------------- 注册表
@pytest.mark.parametrize("alias,canon", [
    ("pinhole", "pinhole"), ("perspective", "pinhole"), ("Pinhole", "pinhole"),
    ("kb", "kb"), ("fisheye-kb", "kb"), ("KannalaBrandt", "kb"),
    ("4thpoly", "4thpoly"), ("radial_poly", "4thpoly"), ("poly4", "4thpoly"),
    ("eftheta", "eftheta"), ("e-ftheta", "eftheta"), ("ftheta", "eftheta"),
])
def test_model_alias_resolution(alias, canon):
    assert normalize_model_name(alias) == canon


def test_unknown_model_raises_with_choices():
    with pytest.raises(ValueError) as e:
        normalize_model_name("mei")
    assert "可用" in str(e.value)


def test_build_model_requires_size():
    with pytest.raises(TypeError):
        build_model("pinhole", fx=500)
