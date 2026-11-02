/**
 * @file test_virtual.cpp
 * @brief 网格构建。对应 Python 版 tests/test_converter.py。
 *
 * 最要紧的一条: **身份旋转 + 同模型同尺寸 -> map 必须是恒等映射**。
 * 旧脚本把 yaw 方向写反、pitch 硬编码这类错误, 在图像上只表现为"看起来
 * 有点歪", 但在这个断言下会直接暴露成几十像素的偏差。
 */
#include "vcam/vc_virtual.h"
#include "vc_test.h"

#include <cmath>
#include <cstdlib>
#include <vector>

static const double FS_K[4] = {339.749, -31.988, 48.275, -7.201};

static VCCameraModel make_src() {
    VCCameraModel m;
    vc_cam_make_radial_poly(&m, 640, 480, FS_K, 322.0, 241.0, 1.0);
    return m;
}

VC_TEST(identity_rotation_gives_identity_map) {
    VCCameraModel src = make_src(), dst = make_src();
    VCVirtualConversion conv;
    VC_CHECK(vc_conv_init(&conv, &src, &dst, 0, 0, 0, VC_CONV_VALEO) == VC_OK);
    VC_CHECK(vc_rot_is_identity(conv.R_rel, 1e-12));

    std::vector<vc_f32> mx(640 * 480), my(640 * 480);
    VCMapStats st;
    VC_CHECK(vc_conv_build_maps(&conv, &mx[0], &my[0], &st, 0) == VC_OK);
    VC_CHECK_EQ_INT(st.valid, 640 * 480);
    VC_CHECK_EQ_INT(st.out_of_source, 0);
    VC_CHECK_EQ_INT(st.unconverged, 0);

    double worst = 0.0;
    for (int r = 0; r < 480; ++r)
        for (int c = 0; c < 640; ++c) {
            double d = std::fabs((double)mx[r * 640 + c] - (double)c);
            double e = std::fabs((double)my[r * 640 + c] - (double)r);
            if (d > worst) worst = d;
            if (e > worst) worst = e;
        }
    VC_CHECK_MSG(worst < 1e-2, "恒等映射偏差应 < 0.01 px (float32 存储)");
}

/** +yaw = 向左看: 虚拟中心采到源图左侧 (u < cx)。 */
VC_TEST(yaw_sign) {
    VCCameraModel src = make_src(), dst = make_src();
    VCVirtualConversion conv;
    vc_conv_init(&conv, &src, &dst, 30.0, 0, 0, VC_CONV_VALEO);
    std::vector<vc_f32> mx(640 * 480), my(640 * 480);
    vc_conv_build_maps(&conv, &mx[0], &my[0], NULL, 0);
    double u_plus = mx[240 * 640 + 320];

    vc_conv_init(&conv, &src, &dst, -30.0, 0, 0, VC_CONV_VALEO);
    vc_conv_build_maps(&conv, &mx[0], &my[0], NULL, 0);
    double u_minus = mx[240 * 640 + 320];

    VC_CHECK_MSG(u_plus < 320.0, "+30 度(左) 中心应采到源图左侧");
    VC_CHECK_MSG(u_minus > 320.0, "-30 度(右) 中心应采到源图右侧");
    VC_CHECK_MSG(std::fabs((u_plus - 320.0) + (u_minus - 320.0)) < 2.0,
                 "左右偏航应当对称");
}

/** +pitch = 抬头: 虚拟中心采到源图上方 (v < cy)。 */
VC_TEST(pitch_sign) {
    VCCameraModel src = make_src(), dst = make_src();
    VCVirtualConversion conv;
    std::vector<vc_f32> mx(640 * 480), my(640 * 480);
    vc_conv_init(&conv, &src, &dst, 0, 20.0, 0, VC_CONV_VALEO);
    vc_conv_build_maps(&conv, &mx[0], &my[0], NULL, 0);
    double v_up = my[240 * 640 + 320];
    vc_conv_init(&conv, &src, &dst, 0, -20.0, 0, VC_CONV_VALEO);
    vc_conv_build_maps(&conv, &mx[0], &my[0], NULL, 0);
    double v_dn = my[240 * 640 + 320];
    VC_CHECK_MSG(v_up < 240.0, "+20 度(抬头) 中心应采到源图上方");
    VC_CHECK_MSG(v_dn > 240.0, "-20 度(低头) 中心应采到源图下方");
}

VC_TEST(chunked_equals_full) {
    /* 分块并行与整块串行必须逐位一致, 不是"近似" */
    VCCameraModel src = make_src(), dst = make_src();
    VCVirtualConversion conv;
    vc_conv_init(&conv, &src, &dst, 15.0, 0, 0, VC_CONV_VALEO);
    std::vector<vc_f32> a_x(640 * 480), a_y(640 * 480), b_x(640 * 480), b_y(640 * 480);
    VCMapStats sa, sb;
    vc_conv_build_maps(&conv, &a_x[0], &a_y[0], &sa, 0);      /* 默认块高 */
    vc_conv_build_maps(&conv, &b_x[0], &b_y[0], &sb, 480);    /* 单块 = 串行 */
    int diff = 0;
    for (size_t i = 0; i < a_x.size(); ++i)
        if (a_x[i] != b_x[i] || a_y[i] != b_y[i]) ++diff;
    VC_CHECK_EQ_INT(diff, 0);
    VC_CHECK_EQ_INT(sa.valid, sb.valid);
    VC_CHECK_EQ_INT(sa.out_of_source, sb.out_of_source);
}

VC_TEST(out_of_source_counted) {
    /* 虚拟 FOV 远大于源 -> 必须有 out_of_source */
    VCCameraModel src = make_src(), dst;
    vc_cam_make_pinhole(&dst, 640, 480, 60.0, 60.0, 320, 240); /* ~160 度 */
    VCVirtualConversion conv;
    vc_conv_init(&conv, &src, &dst, 0, 0, 0, VC_CONV_VALEO);
    std::vector<vc_f32> mx(640 * 480), my(640 * 480);
    VCMapStats st;
    VCStatus s = vc_conv_build_maps(&conv, &mx[0], &my[0], &st, 0);
    VC_CHECK(s == VC_OK);
    VC_CHECK(st.out_of_source > 0);
    VC_CHECK(st.valid < st.total);
    VC_CHECK_EQ_INT(st.valid + st.out_of_source + st.unconverged + st.non_finite,
                    st.total);
    /* 无效位置必须是 -1, 不能留 NaN */
    for (size_t i = 0; i < mx.size(); ++i) VC_CHECK(mx[i] == mx[i]);
}

VC_TEST(no_overlap_reported) {
    /*
     * 虚拟相机朝后看, 源相机朝前 -> 几乎完全无重叠。
     *
     * 注意不是严格的 0: 光轴那一个像素 (241,322) 采到源图主点, 因为
     * 180° 旋转把虚拟光轴正好转到源光轴上。Python 版同样给 valid=1,
     * 两侧一致 —— 所以这里判"有效像素 < 0.001%", 而不是 ==0。
     */
    VCCameraModel src = make_src(), dst = make_src();
    VCVirtualConversion conv;
    vc_conv_init(&conv, &src, &dst, 180.0, 0, 0, VC_CONV_VALEO);
    std::vector<vc_f32> mx(640 * 480), my(640 * 480);
    VCMapStats st;
    VCStatus s = vc_conv_build_maps(&conv, &mx[0], &my[0], &st, 0);
    VC_CHECK(s == VC_E_MAP_NO_OVERLAP);
    VC_CHECK_MSG(st.valid <= 4, "180 度反向应当只剩光轴附近极少数像素");
}

VC_TEST(tables_sum_to_one_inside) {
    VCCameraModel src = make_src(), dst = make_src();
    VCVirtualConversion conv;
    vc_conv_init(&conv, &src, &dst, 12.0, 0, 0, VC_CONV_VALEO);
    std::vector<vc_f32> mx(640 * 480), my(640 * 480);
    vc_conv_build_maps(&conv, &mx[0], &my[0], NULL, 0);

    const int n = 640 * 480;
    std::vector<vc_i32> idx(4 * n);
    std::vector<vc_f32> w(4 * n);
    VCSampleTable t;
    t.idx = &idx[0];
    t.w = &w[0];
    t.count = 0;
    VC_CHECK(vc_conv_build_tables(&conv, &mx[0], &my[0], &t) == VC_OK);

    int checked = 0, badsum = 0;
    for (int r = 2; r < 478; ++r)
        for (int c = 2; c < 638; ++c) {
            int i = r * 640 + c;
            if (mx[i] <= 1 || my[i] <= 1 || mx[i] >= 638 || my[i] >= 478) continue;
            float s = w[i] + w[n + i] + w[2 * n + i] + w[3 * n + i];
            if (std::fabs(s - 1.0f) > 1e-5f) ++badsum;
            if (idx[i] < 0 || idx[i] >= 640 * 480) ++badsum;
            ++checked;
        }
    VC_CHECK(checked > 1000);
    VC_CHECK_EQ_INT(badsum, 0);
}

VC_TEST(bilinear_matches_manual) {
    /* 恒等 map: 采样结果必须等于原图 (只有舍入) */
    VCCameraModel src = make_src(), dst = make_src();
    VCVirtualConversion conv;
    vc_conv_init(&conv, &src, &dst, 0, 0, 0, VC_CONV_VALEO);
    const int n = 640 * 480;
    std::vector<vc_f32> mx(n), my(n);
    vc_conv_build_maps(&conv, &mx[0], &my[0], NULL, 0);
    std::vector<vc_i32> idx(4 * n);
    std::vector<vc_f32> w(4 * n);
    VCSampleTable t;
    t.idx = &idx[0];
    t.w = &w[0];
    vc_conv_build_tables(&conv, &mx[0], &my[0], &t);

    std::vector<vc_u8> srcimg(n), dstimg(n);
    for (int i = 0; i < n; ++i) srcimg[i] = (vc_u8)(i * 7919 % 256);
    VC_CHECK(vc_sample_bilinear(&srcimg[0], 640, 480, 1, &t, &dstimg[0], 640, 480) ==
             VC_OK);
    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (std::abs((int)dstimg[i] - (int)srcimg[i]) > 1) ++bad;
    VC_CHECK_EQ_INT(bad, 0);
}

VC_TEST(openmp_reported) {
    /* 只是把实际编译配置打出来; 不判断值, 避免误以为并行一定开着 */
    int omp = vc_have_openmp();
    VC_CHECK(omp == 0 || omp == 1);
}

int main() { return vc_test_run_all(); }
