/**
 * @file test_rotation.cpp
 * @brief 姿态旋转与欧拉角。对应 Python 版 tests/test_rotation.py。
 *
 * 最重要的两条:
 *   1. 构造->分解->构造 往返误差 (这才是能抓住 old Bug 1 的测试)
 *   2. **闭式期望值**校验符号语义 —— 往返只证明自洽, 不证明朝向对
 */
#include "vcam/vc_rotation.h"
#include "vc_test.h"

#include <cmath>

static double max_roundtrip_err(VCConventionKind conv) {
    double worst = 0.0;
    const double ys[] = {-179, -90, -33, 0, 17, 90, 179};
    const double ps[] = {-89, -45, 0, 20, 60};
    const double rs[] = {-170, -10, 0, 45, 150};
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 5; ++k) {
                double R0[9], y, p, r, R1[9];
                vc_rot_ypr_to_mat(ys[i], ps[j], rs[k], conv, R0);
                vc_rot_mat_to_ypr(R0, conv, &y, &p, &r);
                vc_rot_ypr_to_mat(y, p, r, conv, R1);
                for (int m = 0; m < 9; ++m)
                    if (std::fabs(R0[m] - R1[m]) > worst) worst = std::fabs(R0[m] - R1[m]);
            }
    return worst;
}

VC_TEST(roundtrip_valeo) { VC_CHECK(max_roundtrip_err(VC_CONV_VALEO) < 1e-12); }
VC_TEST(roundtrip_camera) { VC_CHECK(max_roundtrip_err(VC_CONV_CAMERA) < 1e-12); }

VC_TEST(roundtrip_angles_valeo) {
    double R[9], y, p, r;
    vc_rot_ypr_to_mat(30, 20, 10, VC_CONV_VALEO, R);
    vc_rot_mat_to_ypr(R, VC_CONV_VALEO, &y, &p, &r);
    VC_CHECK_NEAR(y, 30.0, 1e-9);
    VC_CHECK_NEAR(p, 20.0, 1e-9);
    VC_CHECK_NEAR(r, 10.0, 1e-9);
}

VC_TEST(roundtrip_angles_camera) {
    double R[9], y, p, r;
    vc_rot_ypr_to_mat(30, 20, 10, VC_CONV_CAMERA, R);
    vc_rot_mat_to_ypr(R, VC_CONV_CAMERA, &y, &p, &r);
    VC_CHECK_NEAR(y, 30.0, 1e-9);
    VC_CHECK_NEAR(p, 20.0, 1e-9);
    VC_CHECK_NEAR(r, 10.0, 1e-9);
}

/** 旧 Python 实现: camera 约定 (30,20,10) 分解回 pitch=11.82。这里必须不再是它。 */
VC_TEST(camera_convention_old_bug_gone) {
    double R[9], y, p, r;
    vc_rot_ypr_to_mat(30, 20, 10, VC_CONV_CAMERA, R);
    vc_rot_mat_to_ypr(R, VC_CONV_CAMERA, &y, &p, &r);
    VC_CHECK_NEAR(p, 20.0, 1e-9);
    VC_CHECK(std::fabs(p - 11.8221) > 1.0);
}

VC_TEST(orthonormal) {
    const VCConventionKind convs[2] = {VC_CONV_VALEO, VC_CONV_CAMERA};
    for (int c = 0; c < 2; ++c) {
        double R[9];
        vc_rot_ypr_to_mat(37, -21, 63, convs[c], R);
        double Rt[9];
        vc_rot_transpose(R, Rt);
        /* R R^T = I */
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += R[i * 3 + k] * Rt[k * 3 + j];
                VC_CHECK_NEAR(s, (i == j) ? 1.0 : 0.0, 1e-12);
            }
    }
}

/** Valeo 正 yaw = 向左 (+Y)。报告问题 6 的命名错误由这条钉死。 */
VC_TEST(valeo_positive_yaw_turns_left) {
    double R[9], ax[3];
    vc_rot_ypr_to_mat(45, 0, 0, VC_CONV_VALEO, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK(ax[1] > 0.5);
    VC_CHECK(ax[0] > 0.5);
    vc_rot_ypr_to_mat(-45, 0, 0, VC_CONV_VALEO, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK(ax[1] < -0.5);
    vc_rot_ypr_to_mat(90, 0, 0, VC_CONV_VALEO, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK_NEAR(ax[0], 0.0, 1e-12);
    VC_CHECK_NEAR(ax[1], 1.0, 1e-12);
}

/** 零姿态: 光轴 = 车体 +X; 相机 +X -> -Y(右); 相机 +Y -> -Z(下)。 */
VC_TEST(valeo_zero_pose) {
    double R[9], v[3], o[3];
    vc_rot_ypr_to_mat(0, 0, 0, VC_CONV_VALEO, R);
    /* cam 轴在车体系的方向: R^T @ e */
    double Rt[9];
    vc_rot_transpose(R, Rt);
    v[0] = 0; v[1] = 0; v[2] = 1;
    vc_rot_matvec(Rt, v, o);
    VC_CHECK_NEAR(o[0], 1.0, 1e-12);
    VC_CHECK_NEAR(o[1], 0.0, 1e-12);
    VC_CHECK_NEAR(o[2], 0.0, 1e-12);
    v[0] = 1; v[1] = 0; v[2] = 0;
    vc_rot_matvec(Rt, v, o);
    VC_CHECK_NEAR(o[0], 0.0, 1e-12);
    VC_CHECK_NEAR(o[1], -1.0, 1e-12);
    VC_CHECK_NEAR(o[2], 0.0, 1e-12);
    v[0] = 0; v[1] = 1; v[2] = 0;
    vc_rot_matvec(Rt, v, o);
    VC_CHECK_NEAR(o[2], -1.0, 1e-12);
}

VC_TEST(valeo_positive_pitch_looks_up) {
    double R[9], ax[3];
    vc_rot_ypr_to_mat(0, 15, 0, VC_CONV_VALEO, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK(ax[2] > 0.2);
    VC_CHECK(std::fabs(ax[1]) < 1e-12);
}

/** ZXZ 的 rotx 带 +180°: (0,0,0) 朝下, rotx=±90 分别朝左/右。 */
VC_TEST(zxz_axis) {
    double R[9], ax[3];
    vc_rot_zxz_to_mat(0, 0, 0, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK_NEAR(ax[2], -1.0, 1e-12);
    vc_rot_zxz_to_mat(0, 90, 0, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK_NEAR(ax[1], 1.0, 1e-12);
    vc_rot_zxz_to_mat(0, -90, 0, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK_NEAR(ax[1], -1.0, 1e-12);
    /* rotz1/rotz2 只绕光轴自转, 不改变指向 */
    vc_rot_zxz_to_mat(90, 0, 0, R);
    vc_rot_optical_axis(R, ax);
    VC_CHECK_NEAR(ax[2], -1.0, 1e-12);
}

VC_TEST(relative_rotation) {
    double Rs[9], Rd[9], Rrel[9], v[3], out[3], expect[3];
    vc_rot_ypr_to_mat(20, -10, 3, VC_CONV_VALEO, Rs);
    vc_rot_ypr_to_mat(65, 5, -2, VC_CONV_VALEO, Rd);
    vc_rot_relative(Rs, Rd, Rrel);
    v[0] = 0.3; v[1] = -0.5; v[2] = 0.8;
    vc_rot_matvec(Rrel, v, out);
    /* 直接算 R_src @ R_dst^T @ v */
    double RdT[9];
    vc_rot_transpose(Rd, RdT);
    double t[3];
    vc_rot_matvec(RdT, v, t);
    vc_rot_matvec(Rs, t, expect);
    VC_CHECK_NEAR(out[0], expect[0], 1e-12);
    VC_CHECK_NEAR(out[1], expect[1], 1e-12);
    VC_CHECK_NEAR(out[2], expect[2], 1e-12);
    /* 同姿态 -> 单位阵 */
    vc_rot_relative(Rs, Rs, Rrel);
    VC_CHECK(vc_rot_is_identity(Rrel, 1e-12));
}

VC_TEST(facing_text) {
    char buf[64];
    double R[9];
    vc_rot_ypr_to_mat(45, 0, 0, VC_CONV_VALEO, R);
    vc_rot_facing(R, buf, sizeof(buf));
    VC_CHECK_STR(buf, "left 45.0deg");
    vc_rot_ypr_to_mat(-45, -20, 0, VC_CONV_VALEO, R);
    vc_rot_facing(R, buf, sizeof(buf));
    VC_CHECK_STR(buf, "right 45.0deg, down 20.0deg");
    vc_rot_ypr_to_mat(0, 0, 0, VC_CONV_VALEO, R);
    vc_rot_facing(R, buf, sizeof(buf));
    VC_CHECK_STR(buf, "front");
}

int main() { return vc_test_run_all(); }
