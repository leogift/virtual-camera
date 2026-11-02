/**
 * @file test_parity.cpp
 * @brief C++ 与 Python 的逐点对拍。
 *
 * 参考数据由 tools/dump_reference.py 用 Python 版生成。比对三件事:
 *   PROJ   正投影 (含图像外的 -1)
 *   UNPROJ 反投影 (射线三分量 + 入射角 + 收敛标记)
 *   MAP    整张 map 的抽样点 + 有效像素计数
 *
 * 容差为什么是这些数:
 *   PROJ/UNPROJ 1e-9  —— 两侧都是双精度、同一组公式, 只差运算顺序
 *   MAP         0.01  —— map 存 float32, 0.01 已是 ULP 量级之上, 能容忍
 *                        OpenMP 分块顺序差异带来的最后一位抖动
 *   计数必须**完全相等** —— 有效/越界计数差一个就说明判定边界不一致
 */
#include "vcam/vc_camera.h"
#include "vcam/vc_rotation.h"
#include "vcam/vc_virtual.h"
#include "vc_test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

const double FS_K[4] = {339.749, -31.988, 48.275, -7.201};

/* 与 dump_reference.py 的 build_models() 一一对应 */
bool build_model(const std::string &name, VCCameraModel *m) {
    if (name == "radial") {
        return vc_cam_make_radial_poly(m, 1280, 966, FS_K, 643.942, 479.907, 1.0) ==
               VC_OK;
    }
    if (name == "kb") {
        return vc_cam_make_kb(m, 640, 480, 300.0, 300.0, 319.5, 239.5, 0.02,
                              -0.004, 0.001, -0.0001) == VC_OK;
    }
    if (name == "pinhole") {
        return vc_cam_make_pinhole(m, 640, 480, 500.0, 500.0, 320.0, 240.0) == VC_OK;
    }
    if (name == "eftheta") {
        const double k[4] = {340.0, -2.0, 1.0, -0.2};
        return vc_cam_make_eftheta(m, 1920, 1536, k, 4, 1.0, 0.02, -0.01, 960.0,
                                   768.0) == VC_OK;
    }
    return false;
}

/* 与 dump_reference.py 的虚拟相机一一对应: 320x240, FOV 150x120 */
VCCameraModel make_virtual_pinhole(int w, int h) {
    const double tan_h = std::tan(150.0 / 2.0 * M_PI / 180.0);
    const double tan_v = std::tan(120.0 / 2.0 * M_PI / 180.0);
    const double fx = (w / 2.0) / tan_h;
    const double fy = (h / 2.0) / tan_v;
    VCCameraModel m;
    vc_cam_make_pinhole(&m, w, h, fx, fy, (w - 1.0) / 2.0, (h - 1.0) / 2.0);
    return m;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::printf("usage: %s <ref.txt>\n", argv[0]);
        return 77; /* SKIP_RETURN_CODE */
    }
    FILE *f = std::fopen(argv[1], "r");
    if (!f) {
        std::printf("cannot open %s -> skip\n", argv[1]);
        return 77;
    }

    std::map<std::string, VCCameraModel> models;
    int n_proj = 0, n_unproj = 0, n_map = 0;
    int bad_proj = 0, bad_unproj = 0, bad_map = 0;
    double worst_proj = 0.0, worst_unproj = 0.0, worst_map = 0.0;

    char line[65536];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        char tag[16];
        if (std::sscanf(line, "%15s", tag) != 1) continue;

        if (std::strcmp(tag, "CAMERA") == 0) {
            char name[32];
            int w, h;
            double k1, k2, k3, k4, cx, cy;
            std::sscanf(line, "%15s %31s %d %d %lf %lf %lf %lf %lf %lf", tag, name,
                        &w, &h, &k1, &k2, &k3, &k4, &cx, &cy);
            VCCameraModel m;
            if (build_model(name, &m)) models[name] = m;
        } else if (std::strcmp(tag, "PROJ") == 0) {
            char name[32];
            double X, Y, Z, ru, rv;
            std::sscanf(line, "%15s %31s %lf %lf %lf %lf %lf", tag, name, &X, &Y, &Z,
                        &ru, &rv);
            if (!models.count(name)) continue;
            double u, v;
            VCStatus st = vc_cam_project(&models[name], X, Y, Z, &u, &v);
            ++n_proj;
            if (st != VC_OK) {
                ++bad_proj;
                continue;
            }
            bool ref_invalid = (ru < 0.0 || rv < 0.0);
            bool got_invalid = (u < 0.0 || v < 0.0);
            if (ref_invalid != got_invalid) {
                ++bad_proj;
                continue;
            }
            if (!ref_invalid) {
                double d = std::max(std::fabs(u - ru), std::fabs(v - rv));
                if (d > worst_proj) worst_proj = d;
                if (d > 1e-9) ++bad_proj;
            }
        } else if (std::strcmp(tag, "UNPROJ") == 0) {
            char name[32];
            double u, v, rx, ry, rz, rt;
            int conv;
            std::sscanf(line, "%15s %31s %lf %lf %lf %lf %lf %lf %d", tag, name, &u,
                        &v, &rx, &ry, &rz, &rt, &conv);
            if (!models.count(name)) continue;
            double ray[3], tdeg;
            vc_i32 ok;
            VCStatus st = vc_cam_unproject(&models[name], u, v, ray, &tdeg, &ok);
            ++n_unproj;
            if (st != VC_OK) {
                ++bad_unproj;
                continue;
            }
            if ((int)ok != conv) {
                ++bad_unproj;
                continue;
            }
            double d = std::max(std::max(std::fabs(ray[0] - rx),
                                         std::fabs(ray[1] - ry)),
                                std::fabs(ray[2] - rz));
            if (d > worst_unproj) worst_unproj = d;
            if (d > 1e-9) ++bad_unproj;
            if (std::fabs(tdeg - rt) > 1e-6) ++bad_unproj;
        } else if (std::strcmp(tag, "MAP") == 0) {
            char tagname[32];
            double yaw = 0, pitch = 0, roll = 0;
            int w, h, nvalid, noor;
            int consumed = 0;
            if (std::sscanf(line, "%15s %31s %lf %lf %lf %d %d %d %d%n", tag, tagname,
                            &yaw, &pitch, &roll, &w, &h, &nvalid, &noor,
                            &consumed) != 9) {
                ++bad_map;
                continue;
            }
            /* 从 consumed 之后是第一个点对 */
            const char *p = line + consumed;
            while (*p == ' ') ++p;

            VCCameraModel src, dst;
            VCVirtualConversion conv;
            if (std::strcmp(tagname, "identity") == 0) {
                /* 恒等映射: 源与目标同一模型。**尺寸必须与记录里的 w/h 一致**,
                 * 否则 build_maps 按 src.width 写缓冲区而缓冲区只有 w*h,
                 * 直接越界 (实测 SIGSEGV 在 libomp 里表现出来, 极难定位)。 */
                src = models["radial"];
                dst = src;
                if (w != src.width || h != src.height) {
                    std::printf("  [MAP identity] 记录尺寸 %dx%d 与模型 %dx%d 不符\n",
                                w, h, (int)src.width, (int)src.height);
                    ++bad_map;
                    continue;
                }
            } else {
                src = models["radial"];
                dst = make_virtual_pinhole(w, h);
            }
            vc_conv_init(&conv, &src, &dst, yaw, pitch, roll, VC_CONV_VALEO);

            /* **按 dst 的实际尺寸分配**, 不是按记录里的 w/h。两者不一致时
             * 后面会显式报错, 而不是越界写。 */
            const int NW = conv.dst.width, NH = conv.dst.height;
            std::vector<vc_f32> mx((size_t)NW * NH), my((size_t)NW * NH);
            VCMapStats st;
            VCStatus s = vc_conv_build_maps(&conv, &mx[0], &my[0], &st, 0);
            ++n_map;
            if (s != VC_OK && s != VC_E_MAP_NO_OVERLAP) {
                ++bad_map;
                continue;
            }
            if (NW != w || NH != h) {
                std::printf("  [MAP %s yaw=%g] dst %dx%d != record %dx%d\n", tagname,
                            yaw, NW, NH, w, h);
                ++bad_map;
                continue;
            }
            if (st.valid != nvalid || st.out_of_source != noor) {
                std::printf("  [MAP %s yaw=%g] count mismatch: valid %d vs %d, "
                            "oor %d vs %d\n", tagname, yaw, (int)st.valid, nvalid,
                            (int)st.out_of_source, noor);
                ++bad_map;
            }
            const int N = NW * NH;
            int k = 0;
            while (*p && *p != '\n' && k < 1000) {
                int i = 0;
                double ex = 0, ey = 0;
                int c2 = 0;
                /* 记录格式: idx,map_x,map_y —— idx 由 Python 给出 */
                if (std::sscanf(p, "%d,%lf,%lf%n", &i, &ex, &ey, &c2) != 3) break;
                if (i < 0 || i >= N) {
                    ++bad_map;
                    break;
                }
                double d = std::max(std::fabs((double)mx[i] - ex),
                                    std::fabs((double)my[i] - ey));
                if (d > worst_map) worst_map = d;
                if (d > 0.01) ++bad_map;
                p += c2;
                if (*p == ';') ++p;
                ++k;
            }
        }
    }
    std::fclose(f);

    std::printf("PARITY  proj  %6d samples, worst %.3e, bad %d\n", n_proj, worst_proj,
                bad_proj);
    std::printf("PARITY  unproj%6d samples, worst %.3e, bad %d\n", n_unproj,
                worst_unproj, bad_unproj);
    std::printf("PARITY  map   %6d maps,    worst %.3e, bad %d\n", n_map, worst_map,
                bad_map);

    bool ok = (n_proj > 0 && n_unproj > 0 && n_map > 0) &&
              (bad_proj == 0) && (bad_unproj == 0) && (bad_map == 0);
    std::printf("PARITY  %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
