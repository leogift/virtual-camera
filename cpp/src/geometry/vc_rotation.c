/**
 * @file vc_rotation.c
 * @brief 工体/相机系姿态与欧拉角。
 *
 * 与 Python 版 vcam/core/rotation.py 逐字对应。
 *
 * ## Valeo 约定 (与 Python/C++ 两侧保持一致的唯一来源)
 *   R = Rz(roll) @ Rx(pitch) @ Ry(yaw) @ PRE_ROT
 *   PRE_ROT: X_cam=-Y_w, Y_cam=-Z_w, Z_cam=+X_w
 *
 *   **注意** 这里的 Rz/Rx/Ry 是 Python 版里右手系写法:
 *     Ry = [[c,0,s],[0,1,0],[-s,0,c]]   (绕 Y 轴, s 在 [0][2])
 *     Rx = [[1,0,0],[0,c,s],[0,-s,c]]
 *     Rz = [[c,s,0],[-s,c,0],[0,0,1]]
 *   与 numpy 逐元素一致, 不要"顺手"换成另一个手性。
 *
 * ## 符号含义 (均有数值断言)
 *   yaw>0   光轴转向车体 +Y (左)
 *   pitch>0 光轴 Z 分量为正 (抬头)
 *   roll>0  图像顺时针
 *
 * ## camera 约定 (Tait-Bryan, R = Rz(roll) Rx(pitch) Ry(yaw), 无预旋转)
 *   展开后 R[2][1] = sin(pitch) —— 与 Python 版注释里的手推式一致。
 *   旧 Python 实现用 R[1][2] 取 pitch, 那是叉乘项, 往返必然错 (30/20/10 得
 *   11.82°)。这里的分解式必须与构造式严格配对, 由 test_rotation 的往返用例守。
 */
#include "vcam/vc_rotation.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define VC_PI 3.14159265358979323846

static const double PRE_ROT_VALEO[9] = {0.0, -1.0, 0.0,
                                        0.0, 0.0,  -1.0,
                                        1.0, 0.0,  0.0};

static void mat3_mul(const double a[9], const double b[9], double out[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[r * 3 + c] = a[r * 3 + 0] * b[0 * 3 + c] +
                             a[r * 3 + 1] * b[1 * 3 + c] +
                             a[r * 3 + 2] * b[2 * 3 + c];
        }
    }
}

void vc_rot_ypr_to_mat(vc_f64 yaw_deg, vc_f64 pitch_deg, vc_f64 roll_deg,
                       VCConventionKind conv, vc_f64 R[9]) {
    const vc_f64 cy = cos(yaw_deg * VC_PI / 180.0), sy = sin(yaw_deg * VC_PI / 180.0);
    const vc_f64 cp = cos(pitch_deg * VC_PI / 180.0), sp = sin(pitch_deg * VC_PI / 180.0);
    const vc_f64 cr = cos(roll_deg * VC_PI / 180.0), sr = sin(roll_deg * VC_PI / 180.0);

    if (conv == VC_CONV_CAMERA) {
        /* Ry, Rx, Rz 用 numpy 的右手系定义, R = Rz @ Rx @ Ry */
        const double Ry[9] = {cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy};
        const double Rx[9] = {1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp};
        const double Rz[9] = {cr, -sr, 0.0, sr, cr, 0.0, 0.0, 0.0, 1.0};
        double tmp[9];
        mat3_mul(Rx, Ry, tmp);
        mat3_mul(Rz, tmp, R);
        return;
    }

    /* Valeo: 注意符号与 numpy 版逐元素一致 */
    const double Ry[9] = {cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy};
    const double Rx[9] = {1.0, 0.0, 0.0, 0.0, cp, sp, 0.0, -sp, cp};
    const double Rz[9] = {cr, sr, 0.0, -sr, cr, 0.0, 0.0, 0.0, 1.0};
    double tmp[9], tmp2[9];
    mat3_mul(Rx, Ry, tmp);
    mat3_mul(Rz, tmp, tmp2);
    mat3_mul(tmp2, PRE_ROT_VALEO, R);
}

void vc_rot_mat_to_ypr(const vc_f64 R[9], VCConventionKind conv, vc_f64 *yaw_deg,
                       vc_f64 *pitch_deg, vc_f64 *roll_deg) {
    if (conv == VC_CONV_CAMERA) {
        vc_f64 sp = R[2 * 3 + 1];
        if (sp > 1.0) sp = 1.0;
        if (sp < -1.0) sp = -1.0;
        vc_f64 p = asin(sp);
        vc_f64 y, r;
        if (fabs(cos(p)) > 1e-6) {
            y = atan2(-R[2 * 3 + 0], R[2 * 3 + 2]);
            r = atan2(-R[0 * 3 + 1], R[1 * 3 + 1]);
        } else { /* 万向锁 */
            y = 0.0;
            r = atan2(R[1 * 3 + 0], R[0 * 3 + 0]);
        }
        *yaw_deg = y * 180.0 / VC_PI;
        *pitch_deg = p * 180.0 / VC_PI;
        *roll_deg = r * 180.0 / VC_PI;
        return;
    }

    /* Valeo: M = R @ PRE_ROT^T。PRE_ROT 正交, 转置即逆 */
    double M[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            M[r * 3 + c] = R[r * 3 + 0] * PRE_ROT_VALEO[c * 3 + 0] +
                           R[r * 3 + 1] * PRE_ROT_VALEO[c * 3 + 1] +
                           R[r * 3 + 2] * PRE_ROT_VALEO[c * 3 + 2];
    vc_f64 sp = -M[2 * 3 + 1];
    if (sp > 1.0) sp = 1.0;
    if (sp < -1.0) sp = -1.0;
    vc_f64 p = asin(sp);
    vc_f64 y, r;
    if (fabs(cos(p)) > 1e-6) {
        y = atan2(-M[2 * 3 + 0], M[2 * 3 + 2]);
        r = atan2(M[0 * 3 + 1], M[1 * 3 + 1]);
    } else {
        y = 0.0;
        r = atan2(-M[1 * 3 + 0], M[0 * 3 + 0]);
    }
    *yaw_deg = y * 180.0 / VC_PI;
    *pitch_deg = p * 180.0 / VC_PI;
    *roll_deg = r * 180.0 / VC_PI;
}

void vc_rot_zxz_to_mat(vc_f64 rotz1_deg, vc_f64 rotx_deg, vc_f64 rotz2_deg,
                       vc_f64 R[9]) {
    const vc_f64 rz1 = rotz1_deg * VC_PI / 180.0;
    const vc_f64 rx = rotx_deg * VC_PI / 180.0 + VC_PI; /* +180: 见 Python 版注释 */
    const vc_f64 rz2 = rotz2_deg * VC_PI / 180.0;
    const double Rx[9] = {1.0, 0.0, 0.0, 0.0, cos(rx), sin(rx), 0.0, -sin(rx), cos(rx)};
    const double Rz1[9] = {cos(rz1), sin(rz1), 0.0, -sin(rz1), cos(rz1), 0.0, 0.0, 0.0, 1.0};
    const double Rz2[9] = {cos(rz2), -sin(rz2), 0.0, sin(rz2), cos(rz2), 0.0, 0.0, 0.0, 1.0};
    double tmp[9];
    mat3_mul(Rx, Rz1, tmp);
    mat3_mul(Rz2, tmp, R);
}

void vc_rot_relative(const vc_f64 R_src[9], const vc_f64 R_dst[9], vc_f64 out[9]) {
    /* R_rel = R_src @ R_dst^T */
    double Rt[9];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) Rt[r * 3 + c] = R_dst[c * 3 + r];
    mat3_mul(R_src, Rt, out);
}

void vc_rot_delta_to_mat(vc_f64 yaw_deg, vc_f64 pitch_deg, vc_f64 roll_deg,
                         VCConventionKind conv, vc_f64 R[9]) {
    /*
     * 增量旋转: 既满足"零增量 -> 单位阵", 又保持与绝对姿态一致的符号语义。
     *
     * 推导(与 Python 版 GridConverter 完全一致):
     *   源相机绝对姿态 R_src = R_abs(0,0,0)
     *   目标虚拟相机     R_dst = R_abs(yaw, pitch, roll)     <- 不是"再叠增量"!
     *   R_rel = R_src @ R_dst^T
     *
     * 两点都靠这个式子拿到:
     *   1) 零增量: R_src @ R_src^T = I
     *   2) +yaw = 向左看。PRE_ROT 在 R_src @ R_dst^T 里被共轭掉, 剩下
     *      PRE_ROT^T·Rz(yaw)·PRE_ROT —— 符号与绝对姿态那条链一致。
     *
     * 试过的两个错法(都会让 yaw/pitch 符号反过来, 实测 center u 从 144.47
     * 变成 495.50):
     *   a) 直接用 R_abs(yaw,pitch,roll) 当增量 -> 零增量得到 PRE_ROT 而非 I
     *   b) 自己写 Rz(roll)Rx(pitch)Ry(yaw) 不乘 PRE_ROT -> 零增量对了, 但
     *      yaw 绕的是相机 Y 而不是车体 Z, 符号与物理方向脱钩
     */
    vc_f64 R_src[9], R_dst[9];
    vc_rot_ypr_to_mat(0.0, 0.0, 0.0, conv, R_src);
    vc_rot_ypr_to_mat(yaw_deg, pitch_deg, roll_deg, conv, R_dst);
    vc_rot_relative(R_src, R_dst, R);
}

void vc_rot_conjugate_delta(const vc_f64 R_rel_cam[9], vc_f64 R_delta_world[9]) {
    /* 相机系里的相对旋转 -> 车体系里可对绝对姿态左乘的增量:
     *     R_delta_world = PRE_ROT^T @ R_rel @ PRE_ROT
     * 这三步是同一个相似变换, 别拆成两次 matvec 记不清谁乘谁。 */
    const vc_f64 PRE[9] = {0, -1, 0, 0, 0, -1, 1, 0, 0};
    vc_f64 PREt[9], t1[9];
    vc_rot_transpose(PRE, PREt);
    mat3_mul(PREt, R_rel_cam, t1);
    mat3_mul(t1, PRE, R_delta_world);
}

void vc_rot_optical_axis(const vc_f64 R[9], vc_f64 axis[3]) {
    /* 光轴 = cam +Z 在车体系的方向 = R^T @ [0,0,1] = R 的第 **3 行**。
     * 曾写成第 3 列 (R[0*3+2], R[1*3+2], R[2*3+2]), 那是 R @ e_z 的方向,
     * 也就是"world +Z 在相机系里指向哪", 完全是另一件事。零姿态下 R 恰好
     * 对称 (PRE_ROT 是置换阵), 所以只有非零姿态才暴露 —— 测试里 yaw=90
     * 才报错就是因为这个。 */
    axis[0] = R[2 * 3 + 0];
    axis[1] = R[2 * 3 + 1];
    axis[2] = R[2 * 3 + 2];
}

void vc_rot_transpose(const vc_f64 R[9], vc_f64 out[9]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[r * 3 + c] = R[c * 3 + r];
}

void vc_rot_matvec(const vc_f64 R[9], const vc_f64 v[3], vc_f64 out[3]) {
    out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

int vc_rot_is_identity(const vc_f64 R[9], vc_f64 tol) {
    for (int i = 0; i < 9; ++i) {
        vc_f64 expect = (i % 4 == 0) ? 1.0 : 0.0;
        if (fabs(R[i] - expect) > tol) return 0;
    }
    return 1;
}

void vc_rot_facing(const vc_f64 R[9], char *buf, int buflen) {
    vc_f64 ax[3];
    vc_rot_optical_axis(R, ax);
    /* yaw 在车体系 X-Y 平面: +Y 为左 */
    vc_f64 yaw = atan2(ax[1], ax[0]) * 180.0 / VC_PI;
    vc_f64 elev = asin(ax[2] > 1.0 ? 1.0 : (ax[2] < -1.0 ? -1.0 : ax[2])) * 180.0 / VC_PI;
    const char *horiz;
    char hbuf[32];
    if (fabs(yaw) < 1e-6) {
        snprintf(hbuf, sizeof(hbuf), "front");
    } else if (fabs(fabs(yaw) - 180.0) < 1e-6) {
        snprintf(hbuf, sizeof(hbuf), "rear");
    } else if (yaw > 0.0) {
        snprintf(hbuf, sizeof(hbuf), "left %.1fdeg", fabs(yaw));
    } else {
        snprintf(hbuf, sizeof(hbuf), "right %.1fdeg", fabs(yaw));
    }
    horiz = hbuf;
    if (fabs(elev) < 1e-6) {
        snprintf(buf, buflen, "%s", horiz);
    } else {
        snprintf(buf, buflen, "%s, %s %.1fdeg", horiz, elev > 0 ? "up" : "down",
                 fabs(elev));
    }
}
