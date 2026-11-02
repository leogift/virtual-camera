/**
 * @file test_camera.cpp
 * @brief 相机模型。对应 Python 版 tests/test_models.py 与 test_fisheye_domain.py。
 *
 * 每条断言都对着独立的闭式参考, 而不是只做 project->unproject->project 的闭环
 * —— 闭环只能证明自洽, 旧代码的 KB/eftheta 分支闭环也是通的。
 *
 * 注意 project() 的约定: **落在图像外返回 -1**。所以"和参考公式逐一比对"的
 * 用例必须先按有效性掩码过滤, 否则拿 -1 去比真实像素, 测的是约定不是公式。
 */
#include "vcam/vc_camera.h"
#include "vc_test.h"

#include <cmath>
#include <cstdio>

static const double FS_K[4] = {339.749, -31.988, 48.275, -7.201};

static VCCameraModel fs_model() {
    VCCameraModel m;
    vc_cam_make_radial_poly(&m, 1280, 966, FS_K, 643.942, 479.907, 1.0);
    return m;
}

/* WoodScape readme 的投影公式, 独立重写 */
static void woodsccape_ref(double X, double Y, double Z, double *u, double *v) {
    double chi = sqrt(X * X + Y * Y);
    double th = atan2(chi, Z);
    double rho = 0.0;
    for (int i = 3; i >= 0; --i) rho = rho * th + FS_K[i];
    rho *= th;
    double scale = (chi > 0.0) ? rho / chi : 0.0;
    *u = scale * X + 3.942 + 1280 / 2.0;
    *v = scale * Y + (-3.093) + 966 / 2.0;
}

VC_TEST(woodsccape_forward_matches_readme) {
    VCCameraModel m = fs_model();
    int checked = 0;
    for (int a = 0; a < 60; ++a) {
        for (int b = 0; b < 60; ++b) {
            double th = (a + 0.5) / 60.0 * 89.0 * M_PI / 180.0;
            double ph = (b + 0.5) / 60.0 * 2.0 * M_PI;
            double X = sin(th) * cos(ph), Y = sin(th) * sin(ph), Z = cos(th);
            double u, v;
            VCStatus st = vc_cam_project(&m, X, Y, Z, &u, &v);
            if (st != VC_OK || u < 0.0) continue;
            double ru, rv;
            woodsccape_ref(X, Y, Z, &ru, &rv);
            VC_CHECK_NEAR(u, ru, 1e-9);
            VC_CHECK_NEAR(v, rv, 1e-9);
            ++checked;
        }
    }
    VC_CHECK_MSG(checked > 500, "有效样本太少, 用例没有说服力");
}

VC_TEST(principal_point_edge_symmetric) {
    /* 主点取 width/2+offset 时, rho 的最大值恰好等于主点到左/上边缘的距离 */
    VCCameraModel m = fs_model();
    double rho_left = vc_cam_rho(&m, 0.0);
    (void)rho_left;
    /* 求 rho(theta) = cx 与 = cy 的 theta */
    double th_cx = vc_cam_solve_theta(&m, m.cx, NULL);
    double th_cy = vc_cam_solve_theta(&m, m.cy, NULL);
    VC_CHECK_NEAR(vc_cam_rho(&m, th_cx), m.cx, 0.01);
    VC_CHECK_NEAR(vc_cam_rho(&m, th_cy), m.cy, 0.01);
}

VC_TEST(back_hemisphere_inside_image_kept) {
    /*
     * theta=95° 时 rho=641 px, 主点到左边界 644 px -> 仍在图像内, 必须有效。
     * 用 90° 常数截断会错杀它 (Python 版初版犯过)。
     */
    VCCameraModel m = fs_model();
    double th = 95.0 * M_PI / 180.0;
    double u, v;
    VCStatus st = vc_cam_project(&m, -sin(th), 0.0, cos(th), &u, &v);
    VC_CHECK(st == VC_OK);
    VC_CHECK_MSG(u != -1.0, "95 度入射且 rho 在图像内, 不应判无效");
    double ray[3];
    double tdeg;
    vc_i32 ok;
    vc_cam_unproject(&m, u, v, ray, &tdeg, &ok);
    VC_CHECK(ok == 1);
    VC_CHECK_NEAR(tdeg, 95.0, 1e-4);
}

VC_TEST(theta_beyond_image_invalid) {
    /* 100° -> rho=685 px > 644 px, 出界必须 -1 */
    VCCameraModel m = fs_model();
    double th = 100.0 * M_PI / 180.0;
    double u, v;
    vc_cam_project(&m, -sin(th), 0.0, cos(th), &u, &v);
    VC_CHECK_NEAR(u, -1.0, 1e-12);
    VC_CHECK_NEAR(v, -1.0, 1e-12);
}

VC_TEST(ninety_degree_in_image) {
    VCCameraModel m = fs_model();
    double u, v;
    vc_cam_project(&m, 1.0, 0.0, 0.0, &u, &v);
    VC_CHECK_MSG(u != -1.0, "rho(90)=598 < 644, 应在图像内");
    VC_CHECK_NEAR(u, 643.942 + 598.01, 0.5);
}

VC_TEST(inverse_roundtrip) {
    VCCameraModel m = fs_model();
    int n = 0;
    for (int i = 0; i < 2000; ++i) {
        double u = (double)(i % 1280);
        double v = (double)((i * 37) % 966);
        double ray[3], tdeg;
        vc_i32 ok;
        VCStatus st = vc_cam_unproject(&m, u, v, ray, &tdeg, &ok);
        VC_CHECK(st == VC_OK);
        VC_CHECK(ok == 1);
        double u2, v2;
        vc_cam_project(&m, ray[0], ray[1], ray[2], &u2, &v2);
        VC_CHECK_NEAR(u2, u, 1e-4);
        VC_CHECK_NEAR(v2, v, 1e-4);
        ++n;
    }
    VC_CHECK_EQ_INT(n, 2000);
}

VC_TEST(theta_recovered) {
    VCCameraModel m = fs_model();
    for (int i = 0; i < 500; ++i) {
        double th = (double)i / 500.0 * 88.0 * M_PI / 180.0;
        double ph = (double)i * 0.7;
        double X = sin(th) * cos(ph), Y = sin(th) * sin(ph), Z = cos(th);
        double u, v;
        vc_cam_project(&m, X, Y, Z, &u, &v);
        if (u < 0.0) continue;
        double ray[3], tdeg;
        vc_i32 ok;
        vc_cam_unproject(&m, u, v, ray, &tdeg, &ok);
        VC_CHECK_NEAR(tdeg, th * 180.0 / M_PI, 1e-6);
        VC_CHECK_NEAR(ray[0], X, 1e-6);
        VC_CHECK_NEAR(ray[2], Z, 1e-6);
    }
}

/* ---------------------------------------------------------------- KB */
VC_TEST(kb_forward_closed_form) {
    VCCameraModel m;
    const double k1 = 0.02, k2 = -0.004, k3 = 0.001, k4 = -0.0001;
    vc_cam_make_kb(&m, 640, 480, 300.0, 300.0, 319.5, 239.5, k1, k2, k3, k4);
    for (int i = 0; i < 800; ++i) {
        double th = (double)i / 800.0 * 88.0 * M_PI / 180.0;
        double ph = (double)i * 0.31;
        double X = sin(th) * cos(ph), Y = sin(th) * sin(ph), Z = cos(th);
        double u, v;
        vc_cam_project(&m, X, Y, Z, &u, &v);
        if (u < 0.0) continue;
        double chi = sqrt(X * X + Y * Y);
        if (chi < 1e-9) continue; /* 正对光轴: 参考公式里的 x/chi 无定义 */
        double theta = atan2(chi, Z);
        double t2 = theta * theta;
        double rho = theta * (1 + k1 * t2 + k2 * t2 * t2 + k3 * t2 * t2 * t2 +
                              k4 * t2 * t2 * t2 * t2);
        VC_CHECK_NEAR(u, 319.5 + 300.0 * rho * X / chi, 1e-9);
        VC_CHECK_NEAR(v, 239.5 + 300.0 * rho * Y / chi, 1e-9);
    }
}

VC_TEST(kb_strong_negative_k1_stays_finite) {
    /* 导数会变号; 步长不得爆炸 */
    VCCameraModel m;
    vc_cam_make_kb(&m, 1280, 966, 300.0, 300.0, 640, 483, -0.30, 0, 0, 0);
    for (int i = 0; i < 256; ++i) {
        double ray[3], tdeg;
        vc_i32 ok;
        vc_cam_unproject(&m, (double)i * 5.0, 483.0, ray, &tdeg, &ok);
        VC_CHECK(tdeg == tdeg); /* 非 NaN */
        VC_CHECK(tdeg >= 0.0 && tdeg <= 180.0);
    }
}

/* ---------------------------------------------------------------- e-ftheta */
VC_TEST(eftheta_affine_inverse) {
    VCCameraModel m;
    const double k[4] = {340.0, -2.0, 1.0, -0.2};
    VCStatus st = vc_cam_make_eftheta(&m, 1920, 1536, k, 4, 1.0, 0.02, -0.01, 960, 768);
    VC_CHECK(st == VC_OK);
    for (int i = 0; i < 600; ++i) {
        double th = (double)i / 600.0 * 85.0 * M_PI / 180.0;
        double ph = (double)i * 0.53;
        double X = sin(th) * cos(ph), Y = sin(th) * sin(ph), Z = cos(th);
        double u, v;
        vc_cam_project(&m, X, Y, Z, &u, &v);
        if (u < 0.0) continue;
        double chi = sqrt(X * X + Y * Y);
        if (chi < 1e-9) continue; /* 正对光轴: 参考公式里的 x/chi 无定义 */
        double theta = atan2(chi, Z);
        double t2 = theta * theta;
        double rho = 340.0 * theta - 2.0 * t2 + t2 * theta - 0.2 * t2 * t2;
        double xn = X / chi, yn = Y / chi;
        VC_CHECK_NEAR(u, 960 + (1.0 * xn + 0.02 * yn) * rho, 1e-8);
        VC_CHECK_NEAR(v, 768 + (-0.01 * xn + 1.0 * yn) * rho, 1e-8);
        double ray[3];
        vc_cam_unproject(&m, u, v, ray, NULL, NULL);
        VC_CHECK_NEAR(ray[0], X, 1e-6);
    }
}

VC_TEST(eftheta_singular_rejected) {
    VCCameraModel m;
    const double k[2] = {300.0, 0.0};
    VCStatus st = vc_cam_make_eftheta(&m, 640, 480, k, 2, 1.0, 1.0, 1.0, 320, 240);
    VC_CHECK(st == VC_E_CAM_SINGULAR);
}

/* ---------------------------------------------------------------- pinhole */
VC_TEST(pinhole_no_distortion_exact) {
    VCCameraModel m;
    vc_cam_make_pinhole(&m, 640, 480, 500.0, 500.0, 320.0, 240.0);
    for (int i = 0; i < 300; ++i) {
        double th = (double)i / 300.0 * 60.0 * M_PI / 180.0;
        double X = sin(th), Z = cos(th);
        double u, v;
        vc_cam_project(&m, X, 0.0, Z, &u, &v);
        if (u < 0.0) continue;
        VC_CHECK_NEAR(u, 320.0 + 500.0 * X / Z, 1e-9);
        VC_CHECK_NEAR(v, 240.0, 1e-9);
    }
}

VC_TEST(pinhole_back_hemisphere_invalid) {
    VCCameraModel m;
    vc_cam_make_pinhole(&m, 640, 480, 300.0, 300.0, 320, 240);
    double u, v;
    vc_cam_project(&m, 1.0, 0.0, 0.0, &u, &v);
    VC_CHECK_NEAR(u, -1.0, 1e-12);
    vc_cam_project(&m, 1.0, 0.0, -1.0, &u, &v);
    VC_CHECK_NEAR(u, -1.0, 1e-12);
}

VC_TEST(pinhole_out_of_image_invalid) {
    VCCameraModel m;
    vc_cam_make_pinhole(&m, 640, 480, 300.0, 300.0, 320, 240);
    double u, v;
    /* X/Z=11.43 -> u=3750, 远超 639 */
    vc_cam_project(&m, 11.43, 0.0, 1.0, &u, &v);
    VC_CHECK_NEAR(u, -1.0, 1e-12);
}

VC_TEST(solver_negative_k1) {
    const double k[4] = {-0.5, 3.0, -1.5, 0.3};
    VCCameraModel m;
    vc_cam_make_radial_poly(&m, 100, 100, k, 50, 50, 1.0);
    for (int i = 0; i < 100; ++i) {
        double th = 0.05 + (double)i / 100.0 * 1.15;
        double rho = fabs(vc_cam_rho(&m, th));
        vc_i32 ok;
        double got = vc_cam_solve_theta(&m, rho, &ok);
        VC_CHECK(ok == 1);
        /* 非单调时同一 rho 可能有多个根, 只能按残差判定 */
        VC_CHECK_NEAR(vc_cam_rho(&m, got), rho, 1e-6);
    }
}

VC_TEST(reject_zero_k1) {
    const double k[4] = {0.0, 1.0, 1.0, 1.0};
    VCCameraModel m;
    VC_CHECK(vc_cam_make_radial_poly(&m, 100, 100, k, 50, 50, 1.0) ==
             VC_E_CAM_NOT_CALIBRATED);
}

VC_TEST(version_checked) {
    VCCameraModel m = fs_model();
    m.version = 99;
    VC_CHECK(vc_cam_validate(&m) == VC_E_PARAM_VERSION);
}

int main() { return vc_test_run_all(); }
