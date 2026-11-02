/**
 * @file vc_camera.c
 * @brief 相机模型的 C 实现 (pinhole / kb / radial_poly / eftheta)。
 *
 * 所有公式与 Python 版逐字对应, 数值对拍见 tests/test_parity.cpp。
 * 这里刻意写成标量版本: 网格构建按像素并行 (OpenMP), 标量代码在
 * -O3 下比手写 SIMD 更容易被编译器向量化, 且可读性高、便于对拍。
 */
#include "vcam/vc_camera.h"

#include <math.h>
#include <string.h>

#define VC_PI 3.14159265358979323846
#define VC_EDGE_EPS 1e-6 /* 越界判定的舍入容差, 见 vc_inside_image */

static const char *k_status[] = {
    "OK",
    "E_PARAM_NULL",
    "E_PARAM_RANGE",
    "E_PARAM_VERSION",
    "E_UNKNOWN_MODEL",
    "-", "-", "-", "-", "-",
    "E_CAM_NOT_CALIBRATED",
    "E_CAM_INVERSE_DIVERGED",
    "E_CAM_SINGULAR",
    "E_EXTRINSIC_INVALID",
    "-", "-", "-", "-", "-", "-",
    "E_MAP_EMPTY",
    "E_MAP_NO_OVERLAP",
    "-", "-", "-", "-", "-", "-", "-", "-",
    "E_BUFFER_SMALL",
    "E_IMAGE_SHAPE",
};

const char *vc_status_str(VCStatus st) {
    int i = (int)st;
    if (i < 0 || i >= (int)(sizeof(k_status) / sizeof(k_status[0]))) return "E_UNKNOWN";
    return k_status[i];
}

const char *vc_model_str(VCModelKind m) {
    switch (m) {
    case VC_MODEL_PINHOLE: return "pinhole";
    case VC_MODEL_KB: return "kb";
    case VC_MODEL_RADIAL_POLY: return "radial_poly";
    case VC_MODEL_EFTHETA: return "eftheta";
    }
    return "unknown";
}

const char *vc_convention_str(VCConventionKind c) {
    return (c == VC_CONV_CAMERA) ? "camera" : "valeo";
}

/* ---------------------------------------------------------------- 内部工具 */

/** 像素是否在图像内 (含边界, 带舍入容差)。
 *
 * 为什么需要 EPS: 恒等映射下边界像素坐标可能算成 -1.7e-13 或
 * 639.0000000000001, 用 u>=0 && u<=W-1 会把它们判成越界。Python 版实测
 * 因此整张 map 出现 640px 偏差。容差只吸收舍入, 不吸收真实越界。 */
static int vc_inside_image(vc_f64 u, vc_f64 v, vc_i32 w, vc_i32 h) {
    return (u >= -VC_EDGE_EPS) && (u <= (vc_f64)(w - 1) + VC_EDGE_EPS) &&
           (v >= -VC_EDGE_EPS) && (v <= (vc_f64)(h - 1) + VC_EDGE_EPS);
}

static vc_f64 vc_clamp(vc_f64 x, vc_f64 lo, vc_f64 hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

/* ---------------------------------------------------------------- 构造 */

static void vc_cam_init(VCCameraModel *out) {
    memset(out, 0, sizeof(*out));
    out->version = VC_CAMERA_MODEL_VERSION;
    out->aspect_ratio = 1.0;
    out->efc = 1.0;
}

VCStatus vc_cam_make_radial_poly(VCCameraModel *out, vc_i32 width, vc_i32 height,
                                 const vc_f64 k[4], vc_f64 cx, vc_f64 cy,
                                 vc_f64 aspect_ratio) {
    if (!out || !k) return VC_E_PARAM_NULL;
    if (width <= 0 || height <= 0) return VC_E_PARAM_RANGE;
    if (!(k[0] > 1e-9)) return VC_E_CAM_NOT_CALIBRATED;
    vc_cam_init(out);
    out->kind = VC_MODEL_RADIAL_POLY;
    out->width = width;
    out->height = height;
    out->cx = cx;
    out->cy = cy;
    out->fx = k[0]; /* 约定: 小角度焦距 = k1, 便于与 pinhole 共用上层代码 */
    out->fy = k[0];
    for (int i = 0; i < 4; ++i) out->k[i] = k[i];
    out->aspect_ratio = (aspect_ratio > 0.0) ? aspect_ratio : 1.0;
    return VC_OK;
}

VCStatus vc_cam_make_pinhole(VCCameraModel *out, vc_i32 width, vc_i32 height,
                             vc_f64 fx, vc_f64 fy, vc_f64 cx, vc_f64 cy) {
    if (!out) return VC_E_PARAM_NULL;
    if (width <= 0 || height <= 0) return VC_E_PARAM_RANGE;
    if (!(fx > 0.0) || !(fy > 0.0)) return VC_E_CAM_NOT_CALIBRATED;
    vc_cam_init(out);
    out->kind = VC_MODEL_PINHOLE;
    out->width = width;
    out->height = height;
    out->fx = fx;
    out->fy = fy;
    out->cx = cx;
    out->cy = cy;
    return VC_OK;
}

VCStatus vc_cam_make_kb(VCCameraModel *out, vc_i32 width, vc_i32 height,
                        vc_f64 fx, vc_f64 fy, vc_f64 cx, vc_f64 cy, vc_f64 k1,
                        vc_f64 k2, vc_f64 k3, vc_f64 k4) {
    if (!out) return VC_E_PARAM_NULL;
    if (width <= 0 || height <= 0) return VC_E_PARAM_RANGE;
    if (!(fx > 0.0) || !(fy > 0.0)) return VC_E_CAM_NOT_CALIBRATED;
    vc_cam_init(out);
    out->kind = VC_MODEL_KB;
    out->width = width;
    out->height = height;
    out->fx = fx;
    out->fy = fy;
    out->cx = cx;
    out->cy = cy;
    out->k[0] = k1;
    out->k[1] = k2;
    out->k[2] = k3;
    out->k[3] = k4;
    return VC_OK;
}

VCStatus vc_cam_make_eftheta(VCCameraModel *out, vc_i32 width, vc_i32 height,
                             const vc_f64 *k, vc_i32 n_k, vc_f64 c, vc_f64 d,
                             vc_f64 e, vc_f64 cx, vc_f64 cy) {
    if (!out || !k) return VC_E_PARAM_NULL;
    if (width <= 0 || height <= 0) return VC_E_PARAM_RANGE;
    if (n_k < 2 || n_k > 5) return VC_E_PARAM_RANGE;
    if (!(k[0] > 1e-9)) return VC_E_CAM_NOT_CALIBRATED;
    if (fabs(c - d * e) < 1e-8) return VC_E_CAM_SINGULAR;
    vc_cam_init(out);
    out->kind = VC_MODEL_EFTHETA;
    out->width = width;
    out->height = height;
    out->cx = cx;
    out->cy = cy;
    out->fx = k[0];
    out->fy = k[0];
    for (int i = 0; i < n_k; ++i) out->k[i] = k[i];
    out->efc = c;
    out->efd = d;
    out->efe = e;
    return VC_OK;
}

VCStatus vc_cam_validate(const VCCameraModel *cam) {
    if (!cam) return VC_E_PARAM_NULL;
    if (cam->version != VC_CAMERA_MODEL_VERSION) return VC_E_PARAM_VERSION;
    if (cam->width <= 0 || cam->height <= 0) return VC_E_PARAM_RANGE;
    switch (cam->kind) {
    case VC_MODEL_PINHOLE:
        if (!(cam->fx > 0.0) || !(cam->fy > 0.0)) return VC_E_CAM_NOT_CALIBRATED;
        break;
    case VC_MODEL_KB:
        if (!(cam->fx > 0.0) || !(cam->fy > 0.0)) return VC_E_CAM_NOT_CALIBRATED;
        break;
    case VC_MODEL_RADIAL_POLY:
        if (!(cam->k[0] > 1e-9)) return VC_E_CAM_NOT_CALIBRATED;
        break;
    case VC_MODEL_EFTHETA:
        if (!(cam->k[0] > 1e-9)) return VC_E_CAM_NOT_CALIBRATED;
        if (fabs(cam->efc - cam->efd * cam->efe) < 1e-8) return VC_E_CAM_SINGULAR;
        break;
    default:
        return VC_E_UNKNOWN_MODEL;
    }
    return VC_OK;
}

/* ---------------------------------------------------------------- rho */

vc_f64 vc_cam_rho(const VCCameraModel *cam, vc_f64 theta) {
    switch (cam->kind) {
    case VC_MODEL_KB: {
        vc_f64 t2 = theta * theta;
        return theta * (1.0 + t2 * (cam->k[0] + t2 * (cam->k[1] + t2 * (cam->k[2] + t2 * cam->k[3]))));
    }
    case VC_MODEL_RADIAL_POLY:
    case VC_MODEL_EFTHETA: {
        vc_f64 acc = 0.0;
        for (int i = 4; i >= 0; --i) acc = acc * theta + cam->k[i];
        return acc * theta;
    }
    case VC_MODEL_PINHOLE:
    default:
        return tan(theta);
    }
}

vc_f64 vc_cam_rho_prime(const VCCameraModel *cam, vc_f64 theta) {
    switch (cam->kind) {
    case VC_MODEL_KB: {
        vc_f64 t2 = theta * theta;
        return 1.0 + t2 * (3.0 * cam->k[0] +
                           t2 * (5.0 * cam->k[1] + t2 * (7.0 * cam->k[2] + t2 * 9.0 * cam->k[3])));
    }
    case VC_MODEL_RADIAL_POLY:
    case VC_MODEL_EFTHETA: {
        vc_f64 acc = 0.0;
        for (int i = 4; i >= 1; --i) acc = acc * theta + cam->k[i] * (vc_f64)(i + 1);
        return acc * theta + cam->k[0];
    }
    case VC_MODEL_PINHOLE:
    default: {
        vc_f64 c = cos(theta);
        return 1.0 / (c * c);
    }
    }
}

/**
 * rho(theta) = rho_d 求根。
 *
 * 导数保护改成 max(|f'|, floor) * sign(f') —— 与 Python 版一致。
 * 为什么不用 `f' + eps`: 旧写法在 f' 从 -1e-3 增到 -2e-12 时分母突然坍到
 * 1e-8, 步长被放大 1e5 倍(Python 版负对照实测)。
 * 牛顿失败再二分兜底: rho 在 [0,pi] 上若单调则二分必收敛。
 */
vc_f64 vc_cam_solve_theta(const VCCameraModel *cam, vc_f64 rho, vc_i32 *ok) {
    const vc_f64 lo = 0.0, hi = VC_PI;
    const vc_f64 floor_d = 1e-7;
    vc_f64 theta;
    if (ok) *ok = 0;

    /* 初值: k1 退化时不用 rho/k1 (会得到负值或巨大值), 直接给中点 */
    vc_f64 scale = (cam->kind == VC_MODEL_KB) ? 1.0 : cam->k[0];
    if (scale > 1e-6) {
        theta = vc_clamp(rho / scale, lo, hi);
    } else {
        theta = 0.5 * (lo + hi);
    }

    int i;
    for (i = 0; i < 40; ++i) {
        vc_f64 f = vc_cam_rho(cam, theta) - rho;
        vc_f64 df = vc_cam_rho_prime(cam, theta);
        vc_f64 denom = (fabs(df) > floor_d ? fabs(df) : floor_d) * (df < 0.0 ? -1.0 : 1.0);
        vc_f64 step = (denom != 0.0) ? f / denom : 0.0;
        step = vc_clamp(step, -0.5, 0.5); /* 阻尼: 单步不超过 0.5 rad */
        vc_f64 next = vc_clamp(theta - step, lo, hi);
        if (!(next == next)) next = theta; /* NaN 保护 */
        vc_f64 d = fabs(next - theta);
        theta = next;
        if (d < 1e-9) break;
    }

    if (fabs(vc_cam_rho(cam, theta) - rho) < 1e-7) {
        if (ok) *ok = 1;
        return theta;
    }

    /* 二分兜底 */
    vc_f64 a = lo, b = hi;
    for (i = 0; i < 60; ++i) {
        vc_f64 mid = 0.5 * (a + b);
        if (vc_cam_rho(cam, mid) - rho < 0.0) a = mid; else b = mid;
    }
    theta = 0.5 * (a + b);
    if (ok) *ok = (fabs(vc_cam_rho(cam, theta) - rho) < 1e-7) ? 1 : 0;
    return theta;
}

/* ---------------------------------------------------------------- 反投影 */

static VCStatus vc_unproject_pinhole(const VCCameraModel *cam, vc_f64 u, vc_f64 v,
                                     vc_f64 ray[3], vc_i32 *conv) {
    vc_f64 xd = (u - cam->cx) / cam->fx;
    vc_f64 yd = (v - cam->cy) / cam->fy;
    vc_f64 xu = xd, yu = yd;
    const vc_f64 k1 = cam->k[0], k2 = cam->k[1], k3 = cam->k[2], k4 = cam->k[3];
    const vc_f64 p1 = cam->p1, p2 = cam->p2;
    int has_dist = (fabs(k1) > 1e-12) || (fabs(k2) > 1e-12) || (fabs(k3) > 1e-12) ||
                   (fabs(k4) > 1e-12) || (fabs(p1) > 1e-12) || (fabs(p2) > 1e-12);

    if (conv) *conv = 1;
    if (has_dist) {
        int it;
        for (it = 0; it < 30; ++it) {
            vc_f64 r2 = xu * xu + yu * yu;
            vc_f64 radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)));
            vc_f64 dxt = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu);
            vc_f64 dyt = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu;
            vc_f64 diffx = xd - (xu * radial + dxt);
            vc_f64 diffy = yd - (yu * radial + dyt);
            xu += diffx;
            yu += diffy;
            vc_f64 lim = 1e-4 / (cam->fx > cam->fy ? cam->fx : cam->fy);
            if (fabs(diffx) < lim && fabs(diffy) < lim) break;
        }
        if (it == 30 && conv) {
            /* 残差回验: 迭代到上限还没收敛, 用残差判定 */
            vc_f64 r2 = xu * xu + yu * yu;
            vc_f64 radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)));
            vc_f64 dxt = 2.0 * p1 * xu * yu + p2 * (r2 + 2.0 * xu * xu);
            vc_f64 dyt = p1 * (r2 + 2.0 * yu * yu) + 2.0 * p2 * xu * yu;
            vc_f64 res = hypot(xd - (xu * radial + dxt), yd - (yu * radial + dyt)) *
                         (cam->fx > cam->fy ? cam->fx : cam->fy);
            *conv = (res < 1e-3) ? 1 : 0;
        }
    }
    vc_f64 n = sqrt(xu * xu + yu * yu + 1.0);
    ray[0] = xu / n;
    ray[1] = yu / n;
    ray[2] = 1.0 / n;
    return VC_OK;
}

static VCStatus vc_unproject_fisheye(const VCCameraModel *cam, vc_f64 u, vc_f64 v,
                                     vc_f64 ray[3], vc_i32 *conv) {
    vc_f64 u_s = u - cam->cx;
    vc_f64 v_s = v - cam->cy;
    vc_f64 theta_src;
    if (cam->kind == VC_MODEL_RADIAL_POLY) {
        /* radial_poly 的 rho 本身就是**像素** (k1 的单位是 px/rad),
         * 因此不能再用 fx 去除。曾在这里沿用 KB 的 (u-cx)/fx, 结果
         * 多除一次焦距: theta=0.5 的像素反解成 0.083。 */
        v_s /= cam->aspect_ratio;
        theta_src = hypot(u_s, v_s);
    } else if (cam->kind == VC_MODEL_EFTHETA) {
        /* 仿射逆: A=[[c,d],[e,1]], det=c-d*e */
        vc_f64 det = cam->efc - cam->efd * cam->efe;
        if (fabs(det) < 1e-12) return VC_E_CAM_SINGULAR;
        vc_f64 xa = (u_s - cam->efd * v_s) / det;
        vc_f64 ya = (-cam->efe * u_s + cam->efc * v_s) / det;
        u_s = xa;
        v_s = ya;
        theta_src = hypot(xa, ya);
    } else { /* KB: rho 以归一化半径计, 要乘 fx */
        u_s /= cam->fx;
        v_s /= cam->fy;
        theta_src = hypot(u_s, v_s);
    }
    vc_f64 rho_d = theta_src;

    vc_i32 ok = 0;
    vc_f64 theta = vc_cam_solve_theta(cam, rho_d, &ok);

    vc_f64 xn = 0.0, yn = 0.0;
    if (rho_d > 1e-12) {
        xn = u_s / rho_d;
        yn = v_s / rho_d;
    }
    vc_f64 st = sin(theta), ct = cos(theta);
    ray[0] = st * xn;
    ray[1] = st * yn;
    ray[2] = ct;
    if (conv) *conv = ok;
    return VC_OK;
}

VCStatus vc_cam_unproject(const VCCameraModel *cam, vc_f64 u, vc_f64 v,
                          vc_f64 ray_out[3], vc_f64 *theta_deg_out,
                          vc_i32 *converged_out) {
    if (!cam || !ray_out) return VC_E_PARAM_NULL;
    VCStatus st = vc_cam_validate(cam);
    if (st != VC_OK) return st;
    vc_i32 conv = 1;

    switch (cam->kind) {
    case VC_MODEL_PINHOLE:
        st = vc_unproject_pinhole(cam, u, v, ray_out, &conv);
        break;
    case VC_MODEL_KB:
    case VC_MODEL_RADIAL_POLY:
    case VC_MODEL_EFTHETA:
        st = vc_unproject_fisheye(cam, u, v, ray_out, &conv);
        break;
    default:
        return VC_E_UNKNOWN_MODEL;
    }
    if (st != VC_OK) return st;

    if (theta_deg_out) {
        vc_f64 ct = vc_clamp(ray_out[2], -1.0, 1.0);
        *theta_deg_out = acos(ct) * 180.0 / VC_PI;
    }
    if (converged_out) *converged_out = conv;
    return VC_OK;
}

/* ---------------------------------------------------------------- 正投影 */

static VCStatus vc_project_pinhole(const VCCameraModel *cam, vc_f64 X, vc_f64 Y,
                                   vc_f64 Z, vc_f64 *u_out, vc_f64 *v_out) {
    if (!(Z > 1e-9)) {
        *u_out = -1.0;
        *v_out = -1.0;
        return VC_OK;
    }
    vc_f64 xu = X / Z, yu = Y / Z;
    vc_f64 r2 = xu * xu + yu * yu;
    vc_f64 radial = 1.0 + r2 * (cam->k[0] + r2 * (cam->k[1] + r2 * (cam->k[2] + r2 * cam->k[3])));
    vc_f64 dxt = 2.0 * cam->p1 * xu * yu + cam->p2 * (r2 + 2.0 * xu * xu);
    vc_f64 dyt = cam->p1 * (r2 + 2.0 * yu * yu) + 2.0 * cam->p2 * xu * yu;
    vc_f64 u = cam->cx + cam->fx * (xu * radial + dxt);
    vc_f64 v = cam->cy + cam->fy * (yu * radial + dyt);
    if (!vc_inside_image(u, v, cam->width, cam->height)) {
        *u_out = -1.0;
        *v_out = -1.0;
    } else {
        *u_out = u;
        *v_out = v;
    }
    return VC_OK;
}

static VCStatus vc_project_fisheye(const VCCameraModel *cam, vc_f64 X, vc_f64 Y,
                                   vc_f64 Z, vc_f64 *u_out, vc_f64 *v_out) {
    vc_f64 r_xy = hypot(X, Y);
    vc_f64 theta = atan2(r_xy, Z); /* Z<0 时给钝角, 正确 */
    vc_f64 rho = vc_cam_rho(cam, theta);

    vc_f64 xn = 0.0, yn = 0.0;
    if (r_xy > 1e-12) {
        xn = X / r_xy;
        yn = Y / r_xy;
    }
    vc_f64 u, v;
    if (cam->kind == VC_MODEL_EFTHETA) {
        u = cam->cx + (cam->efc * xn + cam->efd * yn) * rho;
        v = cam->cy + (cam->efe * xn + 1.0 * yn) * rho;
    } else if (cam->kind == VC_MODEL_RADIAL_POLY) {
        u = cam->cx + rho * xn;
        v = cam->cy + cam->aspect_ratio * rho * yn;
    } else { /* KB */
        u = cam->cx + cam->fx * rho * xn;
        v = cam->cy + cam->fy * rho * yn;
    }
    if (!vc_inside_image(u, v, cam->width, cam->height)) {
        *u_out = -1.0;
        *v_out = -1.0;
    } else {
        *u_out = u;
        *v_out = v;
    }
    return VC_OK;
}

VCStatus vc_cam_project(const VCCameraModel *cam, vc_f64 X, vc_f64 Y, vc_f64 Z,
                        vc_f64 *u_out, vc_f64 *v_out) {
    if (!cam || !u_out || !v_out) return VC_E_PARAM_NULL;
    VCStatus st = vc_cam_validate(cam);
    if (st != VC_OK) return st;
    switch (cam->kind) {
    case VC_MODEL_PINHOLE:
        return vc_project_pinhole(cam, X, Y, Z, u_out, v_out);
    case VC_MODEL_KB:
    case VC_MODEL_RADIAL_POLY:
    case VC_MODEL_EFTHETA:
        return vc_project_fisheye(cam, X, Y, Z, u_out, v_out);
    default:
        return VC_E_UNKNOWN_MODEL;
    }
}

vc_f64 vc_cam_focal_scale(const VCCameraModel *cam) {
    if (!cam) return 0.0;
    if (cam->kind == VC_MODEL_RADIAL_POLY || cam->kind == VC_MODEL_EFTHETA) return cam->k[0];
    return cam->fx;
}
