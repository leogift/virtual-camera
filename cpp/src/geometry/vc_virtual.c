/**
 * @file vc_virtual.c
 * @brief 虚拟相机网格构建。OpenMP 按行块并行。
 */
#include "vcam/vc_virtual.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef VCAM_HAS_OPENMP
#include <omp.h>
#endif

#define VC_DEFAULT_CHUNK 16

int vc_have_openmp(void) {
#ifdef VCAM_HAS_OPENMP
    return 1;
#else
    return 0;
#endif
}

VCStatus vc_conv_init(VCVirtualConversion *out, const VCCameraModel *src,
                      const VCCameraModel *dst, vc_f64 yaw_delta_deg,
                      vc_f64 pitch_delta_deg, vc_f64 roll_delta_deg,
                      VCConventionKind conv) {
    if (!out || !src || !dst) return VC_E_PARAM_NULL;
    VCStatus st = vc_cam_validate(src);
    if (st != VC_OK) return st;
    st = vc_cam_validate(dst);
    if (st != VC_OK) return st;

    memset(out, 0, sizeof(*out));
    out->version = VC_VIRTUAL_CONVERSION_VERSION;
    out->src = *src;
    out->dst = *dst;

    /* 相对增量语义: 目标姿态 = 源姿态再叠一个增量, 化简后 R_rel **就是增量本身**。
     *
     * 必须用 vc_rot_delta_to_mat 而不是 vc_rot_ypr_to_mat: 后者是绝对姿态,
     * Valeo 约定下零姿态给出 PRE_ROT 而非单位阵。曾用错, 零增量的恒等映射
     * 变成 90 度旋转, 实测 valid 只剩 21877/307200。 */
    vc_f64 R_delta[9];
    vc_rot_delta_to_mat(yaw_delta_deg, pitch_delta_deg, roll_delta_deg, conv, R_delta);
    memcpy(out->R_rel, R_delta, sizeof(out->R_rel));
    return VC_OK;
}

VCStatus vc_conv_init_from_R(VCVirtualConversion *out, const VCCameraModel *src,
                             const VCCameraModel *dst, const vc_f64 R_rel[9]) {
    if (!out || !src || !dst || !R_rel) return VC_E_PARAM_NULL;
    VCStatus st = vc_cam_validate(src);
    if (st != VC_OK) return st;
    st = vc_cam_validate(dst);
    if (st != VC_OK) return st;
    memset(out, 0, sizeof(*out));
    out->version = VC_VIRTUAL_CONVERSION_VERSION;
    out->src = *src;
    out->dst = *dst;
    memcpy(out->R_rel, R_rel, sizeof(out->R_rel));
    return VC_OK;
}

/**
 * 单个虚拟像素 -> 源像素。
 * 返回 1 表示有效, 0 表示无效并给出原因码 (out_reason: 0 越界, 1 未收敛, 2 非有限)。
 */
static int vc_conv_pixel(const VCVirtualConversion *conv, vc_i32 row, vc_i32 col,
                         vc_f32 *out_x, vc_f32 *out_y, vc_f64 *theta_deg,
                         int *out_reason) {
    vc_f64 ray_d[3], ray_s[3];
    vc_i32 conv_ok = 1;
    VCStatus st = vc_cam_unproject(&conv->dst, (vc_f64)col, (vc_f64)row, ray_d,
                                   theta_deg, &conv_ok);
    if (st != VC_OK) {
        *out_x = -1.0f;
        *out_y = -1.0f;
        *out_reason = 2;
        return 0;
    }
    /* ray_src = R_rel @ ray_dst。matvec 对 out 与 v 别名不安全时结果错,
     * 所以用独立数组再拷贝, 不要图省事直接传 ray_d。 */
    vc_rot_matvec(conv->R_rel, ray_d, ray_s);

    vc_f64 u, v;
    st = vc_cam_project(&conv->src, ray_s[0], ray_s[1], ray_s[2], &u, &v);
    if (st != VC_OK || !(u == u) || !(v == v)) {
        *out_x = -1.0f;
        *out_y = -1.0f;
        *out_reason = 2;
        return 0;
    }
    if (u == -1.0 && v == -1.0) {
        /* 模型层用 -1 单一哨兵表示"不可见"。**不能用 u<0 判**:
         * 恒等映射下 (0,0) 往返得到 -6.7e-14, 那是落在 0 另一侧的正确值,
         * 判成越界会让整张 map 的第一个像素无效 (实测 640px 偏差)。 */
        *out_x = -1.0f;
        *out_y = -1.0f;
        *out_reason = 0;
        return 0;
    }
    if (!conv_ok) {
        *out_x = -1.0f;
        *out_y = -1.0f;
        *out_reason = 1;
        return 0;
    }
    *out_x = (vc_f32)u;
    *out_y = (vc_f32)v;
    return 1;
}

VCStatus vc_conv_build_rows(const VCVirtualConversion *conv, vc_i32 row_begin,
                            vc_i32 row_end, vc_f32 *map_x, vc_f32 *map_y,
                            vc_i32 map_stride) {
    if (!conv || !map_x || !map_y) return VC_E_PARAM_NULL;
    const vc_i32 w = conv->dst.width;
    for (vc_i32 r = row_begin; r < row_end; ++r) {
        for (vc_i32 c = 0; c < w; ++c) {
            vc_f32 mx, my;
            vc_f64 th = 0.0;
            int reason = 0;
            vc_conv_pixel(conv, r, c, &mx, &my, &th, &reason);
            map_x[(size_t)r * map_stride + c] = mx;
            map_y[(size_t)r * map_stride + c] = my;
        }
    }
    return VC_OK;
}

VCStatus vc_conv_build_maps(const VCVirtualConversion *conv, vc_f32 *map_x,
                            vc_f32 *map_y, VCMapStats *stats, vc_i32 chunk_rows) {
    if (!conv || !map_x || !map_y) return VC_E_PARAM_NULL;
    /* 尺寸自检: dst 的 width/height 决定要写多少元素, 调用方必须按它分配。
     * 曾出现"记录尺寸 320x240 但 dst 是 1280x966"的调用, 越界写把进程搞崩,
     * 崩溃点落在 libomp 里, 完全看不出是这个原因。这里至少挡住非法尺寸。 */
    if (conv->dst.width <= 0 || conv->dst.height <= 0) return VC_E_PARAM_RANGE;
    if (conv->src.width <= 0 || conv->src.height <= 0) return VC_E_PARAM_RANGE;
    const vc_i32 w = conv->dst.width;
    const vc_i32 h = conv->dst.height;
    const vc_i32 chunk = (chunk_rows > 0) ? chunk_rows : VC_DEFAULT_CHUNK;
    const int nchunk = (h + chunk - 1) / chunk;

    /* 每块的局部统计, 避免原子操作 */
    vc_i32 *loc_valid = NULL, *loc_oor = NULL, *loc_unc = NULL, *loc_nf = NULL;
    vc_f64 *loc_theta = NULL;
    int allocs_ok = 1;
    loc_valid = (vc_i32 *)calloc((size_t)nchunk, sizeof(vc_i32));
    loc_oor = (vc_i32 *)calloc((size_t)nchunk, sizeof(vc_i32));
    loc_unc = (vc_i32 *)calloc((size_t)nchunk, sizeof(vc_i32));
    loc_nf = (vc_i32 *)calloc((size_t)nchunk, sizeof(vc_i32));
    loc_theta = (vc_f64 *)calloc((size_t)nchunk, sizeof(vc_f64));
    if (!loc_valid || !loc_oor || !loc_unc || !loc_nf || !loc_theta) allocs_ok = 0;

    if (!allocs_ok) {
        free(loc_valid);
        free(loc_oor);
        free(loc_unc);
        free(loc_nf);
        free(loc_theta);
        return VC_E_BUFFER_SMALL;
    }

#ifdef VCAM_HAS_OPENMP
#pragma omp parallel for schedule(dynamic, 2)
#endif
    for (int ci = 0; ci < nchunk; ++ci) {
        const vc_i32 r0 = ci * chunk;
        const vc_i32 r1 = (r0 + chunk < h) ? r0 + chunk : h;
        vc_i32 nv = 0, noor = 0, nunc = 0, nnf = 0;
        vc_f64 maxt = 0.0;
        for (vc_i32 r = r0; r < r1; ++r) {
            const size_t base = (size_t)r * w;
            for (vc_i32 c = 0; c < w; ++c) {
                vc_f32 mx, my;
                vc_f64 th = 0.0;
                int reason = 0;
                int ok = vc_conv_pixel(conv, r, c, &mx, &my, &th, &reason);
                map_x[base + c] = mx;
                map_y[base + c] = my;
                if (ok) {
                    ++nv;
                } else if (reason == 0) {
                    ++noor;
                } else if (reason == 1) {
                    ++nunc;
                } else {
                    ++nnf;
                }
                if (th > maxt) maxt = th;
            }
        }
        loc_valid[ci] = nv;
        loc_oor[ci] = noor;
        loc_unc[ci] = nunc;
        loc_nf[ci] = nnf;
        loc_theta[ci] = maxt;
    }

    VCMapStats s;
    memset(&s, 0, sizeof(s));
    s.version = VC_MAP_STATS_VERSION;
    s.width = w;
    s.height = h;
    s.total = w * h;
    for (int i = 0; i < nchunk; ++i) {
        s.valid += loc_valid[i];
        s.out_of_source += loc_oor[i];
        s.unconverged += loc_unc[i];
        s.non_finite += loc_nf[i];
        if (loc_theta[i] > s.max_theta_deg) s.max_theta_deg = loc_theta[i];
    }
    free(loc_valid);
    free(loc_oor);
    free(loc_unc);
    free(loc_nf);
    free(loc_theta);

    /* 告警: 这些情况几乎总是配置错, 而不是"图正好没重叠" */
    if (s.valid == 0) {
        snprintf(s.note, sizeof(s.note), "有效像素为 0: 虚拟相机完全看不到源相机视野");
    } else if ((vc_f64)s.valid < 0.05 * (vc_f64)s.total) {
        snprintf(s.note, sizeof(s.note), "有效像素仅 %.1f%%, 检查姿态或 FOV",
                 100.0 * s.valid / s.total);
    } else if (s.max_theta_deg > 90.0) {
        snprintf(s.note, sizeof(s.note), "虚拟视场覆盖到入射角 %.1fdeg (超过 90)",
                 s.max_theta_deg);
    }

    if (stats) *stats = s;
    /* 判"无重叠"用相对比例而不是 valid==0: 180 度反向时恰好有 1 个像素
     * (光轴)落在源图主点上, valid=1, 但它显然不是"有重叠"。
     * 阈值取 1e-4 (0.01%), 即典型 30 万像素图里 <30 个 */
    if ((vc_f64)s.valid < 1e-4 * (vc_f64)s.total) {
        if (stats) snprintf(stats->note, sizeof(stats->note),
                            "有效像素仅 %d/%d: 视为无重叠", (int)s.valid, (int)s.total);
        return VC_E_MAP_NO_OVERLAP;
    }
    return VC_OK;
}

VCStatus vc_conv_build_tables(const VCVirtualConversion *conv, const vc_f32 *map_x,
                              const vc_f32 *map_y, VCSampleTable *tables) {
    if (!conv || !map_x || !map_y || !tables || !tables->idx || !tables->w)
        return VC_E_PARAM_NULL;
    const vc_i32 w = conv->dst.width, h = conv->dst.height;
    const vc_i32 sw = conv->src.width, sh = conv->src.height;
    const vc_i32 n = w * h;

#ifdef VCAM_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (vc_i32 i = 0; i < n; ++i) {
        vc_f64 x = map_x[i], y = map_y[i];
        /* 只认 -1 哨兵, 不用 x<0 (见 vc_conv_pixel 的说明) */
        int bad = (x == -1.0) || (y == -1.0) || !(x == x) || !(y == y);
        if (bad) {
            x = 0.0;
            y = 0.0;
        }
        vc_i32 x0 = (vc_i32)floor(x);
        vc_i32 y0 = (vc_i32)floor(y);
        vc_f64 fx = x - x0, fy = y - y0;
        vc_i32 xi0 = x0, yi0 = y0, xi1 = x0 + 1, yi1 = y0 + 1;
        const vc_i32 xs[4] = {xi0, xi1, xi0, xi1};
        const vc_i32 ys[4] = {yi0, yi0, yi1, yi1};
        const vc_f64 wxs[4] = {1.0 - fx, fx, 1.0 - fx, fx};
        const vc_f64 wys[4] = {1.0 - fy, 1.0 - fy, fy, fy};
        for (int k = 0; k < 4; ++k) {
            int inb = !bad && xs[k] >= 0 && xs[k] < sw && ys[k] >= 0 && ys[k] < sh;
            tables->idx[k * n + i] = inb ? (ys[k] * sw + xs[k]) : 0;
            tables->w[k * n + i] = inb ? (vc_f32)(wxs[k] * wys[k]) : 0.0f;
        }
    }
    tables->count = n;
    return VC_OK;
}

VCStatus vc_sample_bilinear(const vc_u8 *src, vc_i32 src_w, vc_i32 src_h,
                            vc_i32 img_channels, const VCSampleTable *tables,
                            vc_u8 *dst, vc_i32 dst_w, vc_i32 dst_h) {
    if (!src || !tables || !dst) return VC_E_PARAM_NULL;
    if (img_channels != 1 && img_channels != 3) return VC_E_PARAM_RANGE;
    const vc_i32 n = dst_w * dst_h;
    if (tables->count != n) return VC_E_BUFFER_SMALL;

#ifdef VCAM_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (vc_i32 i = 0; i < n; ++i) {
        for (vc_i32 ch = 0; ch < img_channels; ++ch) {
            vc_f32 acc = 0.0f;
            for (int k = 0; k < 4; ++k) {
                vc_i32 pi = tables->idx[k * n + i];
                acc += tables->w[k * n + i] *
                       (vc_f32)src[(size_t)pi * img_channels + ch];
            }
            vc_f32 v = acc + 0.5f;
            dst[(size_t)i * img_channels + ch] =
                (vc_u8)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
        }
    }
    (void)src_w;
    (void)src_h;
    return VC_OK;
}
