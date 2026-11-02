/**
 * @file vcam_convert.cpp
 * @brief 把 WoodScape/NFOCC 鱼眼虚拟成理想针孔相机。
 *
 * 与 Python 版 tools/convert_nfocc.py 等价, 用同一套 C 核心库。
 *
 * 示例:
 *   vcam_convert --calib datasets/fisheye/calibration/calibration/00164_FV.json \
 *                --image datasets/fisheye/rgb_images/00164_FV.png \
 *                --yaws -45 0 45 --fov-h 150 --fov-v 120 \
 *                --width 640 --output-dir out/cpp
 *
 * 这个文件只做参数解析与接线, 几何全在 vc_core 里。
 */
#include "vcam/vc_camera.h"
#include "vcam/vc_io_ws.h"
#include "vcam/vc_rotation.h"
#include "vcam/vc_virtual.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace {

struct Args {
    std::string calib;
    std::string image;
    std::string output_dir = "out/cpp";
    std::vector<double> yaws;
    double pitch = 0.0;
    double roll = 0.0;
    double fov_h = 150.0;
    double fov_v = 120.0;
    int width = 640;
    int height = 0; /* 0 -> 由 FOV 反推, 保证方形像素 */
    int chunk_rows = 0;
    bool save_maps = false;
    bool quiet = false;
};

void usage(const char *prog) {
    std::printf(
        "usage: %s --calib <json> --image <png> [options]\n"
        "  --yaws a b c ...   相对源相机的 yaw 增量(度), 正值=向左看 (默认 -45 0 45)\n"
        "  --pitch deg        附加俯仰增量, 正值=抬头\n"
        "  --roll deg         附加横滚增量\n"
        "  --fov-h / --fov-v  虚拟针孔视场角 (默认 150 / 120)\n"
        "  --width px         默认 640; --height 省略时按 FOV 反推保持方形像素\n"
        "  --chunk-rows n     并行块高 (默认由库里定)\n"
        "  --save-maps        额外写出 map_x/map_y 文本\n"
        "  --quiet            只打印结果行\n",
        prog);
}

/** 递归建目录 (mkdir -p)。返回 0 表示目录可用。 */
int make_dirs(const std::string &path) {
    if (path.empty()) return 0;
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty()) {
                if (mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) return -1;
            }
        }
        if (i < path.size()) cur.push_back(path[i]);
    }
    return 0;
}

bool parse(int argc, char **argv, Args *a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](double *v) {
            if (i + 1 >= argc) return false;
            *v = std::atof(argv[++i]);
            return true;
        };
        auto nexti = [&](int *v) {
            if (i + 1 >= argc) return false;
            *v = std::atoi(argv[++i]);
            return true;
        };
        if (k == "--calib") {
            if (++i >= argc) return false;
            a->calib = argv[i];
        } else if (k == "--image") {
            if (++i >= argc) return false;
            a->image = argv[i];
        } else if (k == "--output-dir") {
            if (++i >= argc) return false;
            a->output_dir = argv[i];
        } else if (k == "--yaws") {
            a->yaws.clear();
            /* 收集后续所有"能解析成数"的 token。不能只看首字符:
             * -45 以 '-' 开头, 但它是一个数, 不是下一个选项。 */
            while (i + 1 < argc) {
                const char *nx = argv[i + 1];
                char *end = NULL;
                double v = std::strtod(nx, &end);
                if (end == nx || *end != '\0') break;
                a->yaws.push_back(v);
                ++i;
            }
            if (a->yaws.empty()) {
                std::fprintf(stderr, "--yaws 后面至少要跟一个数\n");
                return false;
            }
        } else if (k == "--pitch") {
            next(&a->pitch);
        } else if (k == "--roll") {
            next(&a->roll);
        } else if (k == "--fov-h") {
            next(&a->fov_h);
        } else if (k == "--fov-v") {
            next(&a->fov_v);
        } else if (k == "--width") {
            nexti(&a->width);
        } else if (k == "--height") {
            nexti(&a->height);
        } else if (k == "--chunk-rows") {
            nexti(&a->chunk_rows);
        } else if (k == "--save-maps") {
            a->save_maps = true;
        } else if (k == "--quiet") {
            a->quiet = true;
        } else if (k == "-h" || k == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            return false;
        }
    }
    if (a->yaws.empty()) a->yaws = {-45.0, 0.0, 45.0};
    return !a->calib.empty() && !a->image.empty();
}

/** 由 FOV 与宽度反推高度, 使 fx==fy。 */
int height_for_fov(int width, double fov_h, double fov_v) {
    const double th = std::tan(fov_h / 2.0 * M_PI / 180.0);
    const double tv = std::tan(fov_v / 2.0 * M_PI / 180.0);
    return (int)(width * tv / th + 0.5);
}

VCCameraModel make_virtual(int w, int h, double fov_h, double fov_v) {
    const double fx = (w / 2.0) / std::tan(fov_h / 2.0 * M_PI / 180.0);
    const double fy = (h / 2.0) / std::tan(fov_v / 2.0 * M_PI / 180.0);
    VCCameraModel m;
    vc_cam_make_pinhole(&m, w, h, fx, fy, (w - 1.0) / 2.0, (h - 1.0) / 2.0);
    return m;
}

void save_map_text(const std::string &path, const std::vector<vc_f32> &m, int w,
                   int h) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) return;
    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            std::fprintf(f, "%.4f%c", (double)m[(size_t)r * w + c],
                         c + 1 == w ? '\n' : ' ');
        }
    }
    std::fclose(f);
}

} // namespace

int main(int argc, char **argv) {
    Args args;
    if (!parse(argc, argv, &args)) {
        usage(argv[0]);
        return 2;
    }
    if (make_dirs(args.output_dir) != 0) {
        std::fprintf(stderr, "无法创建输出目录: %s\n", args.output_dir.c_str());
        return 1;
    }

    VCWoodScapeCalib calib;
    VCStatus st = vc_ws_load(args.calib.c_str(), &calib);
    if (st != VC_OK) {
        std::fprintf(stderr, "load calib failed: %s\n", vc_status_str(st));
        return 1;
    }
    VCCameraModel src;
    st = vc_ws_to_camera(&calib, &src);
    if (st != VC_OK) {
        std::fprintf(stderr, "build camera failed: %s\n", vc_status_str(st));
        return 1;
    }

    cv::Mat img = cv::imread(args.image, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "cannot read image: %s\n", args.image.c_str());
        return 1;
    }
    if (img.cols != src.width || img.rows != src.height) {
        std::fprintf(stderr,
                     "image %dx%d != calib %dx%d (拒绝静默 resize: 会同时改变畸变几何)\n",
                     img.cols, img.rows, (int)src.width, (int)src.height);
        return 1;
    }

    const int H = args.height > 0 ? args.height
                                  : height_for_fov(args.width, args.fov_h, args.fov_v);
    VCCameraModel dst = make_virtual(args.width, H, args.fov_h, args.fov_v);

    char facing_src[64];
    vc_f64 Rs[9];
    if (calib.has_extrinsic) {
        vc_ws_pose(&calib, Rs);
        vc_rot_facing(Rs, facing_src, sizeof(facing_src));
    } else {
        std::snprintf(facing_src, sizeof(facing_src), "(no extrinsic)");
    }

    if (!args.quiet) {
        std::printf("=== %s ===\n", calib.name[0] ? calib.name : "camera");
        std::printf("  source: radial_poly %dx%d, k=[%.3f %.3f %.3f %.3f], "
                    "principal=(%.3f, %.3f)\n", (int)src.width, (int)src.height,
                    calib.k1, calib.k2, calib.k3, calib.k4, src.cx, src.cy);
        std::printf("  source facing: %s\n", facing_src);
        std::printf("  virtual: pinhole %dx%d, fx=%.2f fy=%.2f, FOV %.0fx%.0f\n",
                    (int)dst.width, (int)dst.height, dst.fx, dst.fy, args.fov_h,
                    args.fov_v);
        std::printf("  OpenMP: %s\n", vc_have_openmp() ? "on" : "off (serial)");
    }

    const int N = args.width * H;
    std::vector<vc_f32> mx((size_t)N), my((size_t)N);
    std::vector<vc_i32> idx(4 * (size_t)N);
    std::vector<vc_f32> wt(4 * (size_t)N);
    VCSampleTable tables;
    tables.idx = &idx[0];
    tables.w = &wt[0];

    for (size_t i = 0; i < args.yaws.size(); ++i) {
        const double yaw = args.yaws[i];

        /*
         * R_rel 必须由**源相机的真实绝对姿态**与目标虚拟姿态一起算:
         *      R_src = R_abs(yaw_src, pitch_src, roll_src)        <- 来自标定外参
         *      R_dst = R_abs(yaw_src+dy, pitch_src+dp, roll_src+dr)
         *      R_rel = R_src @ R_dst^T
         *
         * 不能用 vc_conv_init 的"零姿态增量"版本: 那条路径假设源相机是
         * 零偏航零俯仰的理想相机, 而实际 FV 有 -22.65° 俯角。实测直接用量:
         *   py/x 同一点差 166 px, 有效掩码只有 91% 一致 —— 图看上去"也像那么
         *   回事", 但几何全错。
         */
        vc_f64 R_src[9], R_dst_abs[9], R_rel[9];
        if (calib.has_extrinsic) {
            vc_ws_pose(&calib, R_src);
            double sy, sp, sr;
            vc_rot_mat_to_ypr(R_src, VC_CONV_VALEO, &sy, &sp, &sr);
            vc_rot_ypr_to_mat(sy + yaw, sp + args.pitch, sr + args.roll,
                              VC_CONV_VALEO, R_dst_abs);
            vc_rot_relative(R_src, R_dst_abs, R_rel);
        } else {
            /* 没有外参时退回零姿态增量 (与 vc_conv_init 等价) */
            vc_rot_delta_to_mat(yaw, args.pitch, args.roll, VC_CONV_VALEO, R_rel);
        }

        VCVirtualConversion conv;
        st = vc_conv_init_from_R(&conv, &src, &dst, R_rel);
        if (st != VC_OK) {
            std::fprintf(stderr, "conv init failed: %s\n", vc_status_str(st));
            return 1;
        }

        VCMapStats stats;
        st = vc_conv_build_maps(&conv, &mx[0], &my[0], &stats, args.chunk_rows);
        if (st != VC_OK && st != VC_E_MAP_NO_OVERLAP) {
            std::fprintf(stderr, "build maps failed: %s\n", vc_status_str(st));
            return 1;
        }
        vc_conv_build_tables(&conv, &mx[0], &my[0], &tables);

        cv::Mat dst_img(H, args.width, CV_8UC3);
        st = vc_sample_bilinear(img.data, img.cols, img.rows, img.channels(), &tables,
                                dst_img.data, args.width, H);
        if (st != VC_OK) {
            std::fprintf(stderr, "sample failed: %s\n", vc_status_str(st));
            return 1;
        }

        /* 与 Python 一样, 无效像素写 -1 时采样结果是 0; 这里再显式清零, 避免
         * 表里的 "0 权重 tap" 在某些实现下漏网 */
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < args.width; ++c)
                if (mx[(size_t)r * args.width + c] < 0) {
                    dst_img.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
                }

        char tag[32];
        std::snprintf(tag, sizeof(tag), "yaw%+d", (int)yaw);
        const std::string out_png = args.output_dir + "/" + tag + ".png";
        /* imwrite 在目录不存在时**静默返回 false**, 不抛也不打印。
         * 实测输出目录没建, 程序照常打印了每一条 "out/xxx/yaw0.png",
         * 回头一看目录不存在。这里必须查返回值并给出可操作的提示。 */
        if (!cv::imwrite(out_png, dst_img)) {
            std::fprintf(stderr,
                         "写图失败: %s\n"
                         "  常见原因: 输出目录不存在。请先 mkdir -p %s\n",
                         out_png.c_str(), args.output_dir.c_str());
            return 1;
        }
        if (args.save_maps) {
            save_map_text(args.output_dir + "/" + tag + "_map_x.txt", mx, args.width, H);
            save_map_text(args.output_dir + "/" + tag + "_map_y.txt", my, args.width, H);
        }

        char facing[64];
        if (calib.has_extrinsic) {
            /*
             * 目标绝对姿态 = 源姿态再叠一个"车体坐标系下的 yaw/pitch/roll 增量":
             *   R_abs = R_abs(yaw_src + dy, pitch_src + dp, roll_src + dr)
             *
             * 为什么不在 R_rel 上做矩阵共轭: R_rel 是**相机系**里的旋转,
             * 它经 mat_to_ypr 反解出来的 (dy,dp,dr) 已经不是车体系欧拉角
             * (实测 45° 的 R_rel 反解后 yaw 项丢掉, 共轭左乘后光轴原地不动,
             * 打印仍是 "right 0.1deg")。最直接、也最不容易错的写法就是
             * 在欧拉角域相加 —— Python 版 GridConverter 就是这么做的。
             * 这里只影响**打印**, 几何仍走 conv.R_rel。
             */
            double sy, sp, sr;
            vc_rot_mat_to_ypr(Rs, VC_CONV_VALEO, &sy, &sp, &sr);
            vc_f64 Rdst[9];
            vc_rot_ypr_to_mat(sy + yaw, sp + args.pitch, sr + args.roll,
                              VC_CONV_VALEO, Rdst);
            vc_rot_facing(Rdst, facing, sizeof(facing));
        } else {
            std::snprintf(facing, sizeof(facing), "%s", vc_convention_str(VC_CONV_VALEO));
        }

        std::printf("%-12s yaw=%+6.1f -> %-24s valid=%6d/%d (%5.1f%%) "
                    "maxTheta=%5.1f  %s%s\n",
                    calib.name[0] ? calib.name : "cam", yaw, facing, (int)stats.valid,
                    stats.total, 100.0 * stats.valid / stats.total,
                    stats.max_theta_deg, out_png.c_str(),
                    stats.note[0] ? "  [WARN]" : "");
        if (stats.note[0] && !args.quiet) {
            std::printf("             warning: %s\n", stats.note);
        }
    }

    return 0;
}
