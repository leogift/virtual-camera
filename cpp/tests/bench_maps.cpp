/**
 * @file bench_maps.cpp
 * @brief 网格构建吞吐基准 (只测建 map, 不含 IO)。
 *
 * 为什么单独一个程序而不是塞进测试: 计时受机器负载影响, 放进 ctest 会让
 * 测试不稳定; 而"并行到底有没有用"又是必须回答的问题 —— 见 README
 * 的实测加速比。
 */
#include "vcam/vc_virtual.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
const double FS_K[4] = {339.749, -31.988, 48.275, -7.201};

VCCameraModel make_virtual(int w, int h) {
    /* 150x120 FOV 的理想针孔: fx=(w/2)/tan75, fy=(h/2)/tan60 */
    const double tan_h = 3.7320508075688772; /* tan(75 deg) */
    const double tan_v = 1.7320508075688772; /* tan(60 deg) */
    VCCameraModel m;
    vc_cam_make_pinhole(&m, w, h, (w / 2.0) / tan_h, (h / 2.0) / tan_v,
                        (w - 1.0) / 2.0, (h - 1.0) / 2.0);
    return m;
}
} // namespace

int main(int argc, char **argv) {
    int w = 1920, h = 1080, repeat = 20;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--width") && i + 1 < argc) w = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--height") && i + 1 < argc) h = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::atoi(argv[++i]);
    }

    VCCameraModel src;
    vc_cam_make_radial_poly(&src, 1280, 966, FS_K, 643.942, 479.907, 1.0);
    VCCameraModel dst = make_virtual(w, h);
    VCVirtualConversion conv;
    vc_conv_init(&conv, &src, &dst, -45.0, 0.0, 0.0, VC_CONV_VALEO);

    std::vector<vc_f32> mx((size_t)w * h), my((size_t)w * h);
    VCMapStats st;

    /* 预热一次: 首次调用会触发 OpenMP 线程池创建, 计进去就不是纯计算耗时 */
    vc_conv_build_maps(&conv, &mx[0], &my[0], &st, 0);

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r)
        vc_conv_build_maps(&conv, &mx[0], &my[0], &st, 0);
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0)
            .count();
    const double per = total_ms / (double)repeat;

    std::printf("width=%d height=%d repeat=%d openmp=%d valid=%d ms_per_map=%.3f\n",
                w, h, repeat, vc_have_openmp(), (int)st.valid, per);
    return 0;
}
