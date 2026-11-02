/**
 * @file vc_io_ws.c
 * @brief WoodScape 标定 JSON 读取 (零依赖, 面向本格式).
 *
 * 解析策略: 把整个文件读进内存, 用 strstr 定位键, 再向后找冒号与数值。
 * 不是通用 JSON —— 但对固定格式足够, 且不需要引入第三方库。
 * 关键约束: **必需字段缺失就报错**, 不做默认值兜底。
 */
#include "vcam/vc_io_ws.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n <= 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if (out_len) *out_len = got;
    return buf;
}

/** 在 from 之后找 "key" 并取其后的第一个数字 (可跳过冒号与空白). */
static int find_number(const char *from, const char *key, double *out) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(from, pat);
    if (!p) return 0;
    p += strlen(pat);
    while (*p && *p != ':' && *p != '-' && *p != '.' && (*p < '0' || *p > '9')) {
        if (*p == '}') return 0;
        ++p;
    }
    if (*p == ':') ++p;
    char *end = NULL;
    double v = strtod(p, &end);
    if (end == p) return 0;
    *out = v;
    return 1;
}

/** 取 "key": [a, b, c, d] 这类数组. 最多 n 个, 返回实际读到的个数. */
static int find_array(const char *from, const char *key, double *out, int n) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(from, pat);
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    ++p;
    int cnt = 0;
    while (cnt < n && *p) {
        while (*p && *p != '-' && *p != '.' && (*p < '0' || *p > '9')) {
            if (*p == ']') return cnt;
            ++p;
        }
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p) break;
        out[cnt++] = v;
        p = end;
        while (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
        if (*p == ']') break;
    }
    return cnt;
}

static int find_string(const char *from, const char *key, char *out, int n) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(from, pat);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p = strchr(p, '"');
    if (!p) return 0;
    ++p;
    const char *e = strchr(p, '"');
    if (!e) return 0;
    int len = (int)(e - p);
    if (len >= n) len = n - 1;
    memcpy(out, p, (size_t)len);
    out[len] = '\0';
    return 1;
}

VCStatus vc_ws_load(const char *path, VCWoodScapeCalib *out) {
    if (!path || !out) return VC_E_PARAM_NULL;
    size_t len = 0;
    char *buf = read_file(path, &len);
    if (!buf) return VC_E_PARAM_NULL;
    memset(out, 0, sizeof(*out));
    out->aspect_ratio = 1.0;
    out->poly_order = 4;

    /* 内参在 "intrinsic" 之后; 外参在 "extrinsic" 之后 */
    const char *pint = strstr(buf, "\"intrinsic\"");
    const char *pext = strstr(buf, "\"extrinsic\"");
    if (!pint) {
        free(buf);
        return VC_E_CAM_NOT_CALIBRATED;
    }

    double w = 0, h = 0;
    const char *missing = NULL;
    if (!find_number(pint, "width", &w)) missing = "width";
    if (!missing && !find_number(pint, "height", &h)) missing = "height";
    if (!missing && !find_number(pint, "k1", &out->k1)) missing = "k1";
    if (!missing && !find_number(pint, "k2", &out->k2)) missing = "k2";
    if (!missing && !find_number(pint, "k3", &out->k3)) missing = "k3";
    if (!missing && !find_number(pint, "k4", &out->k4)) missing = "k4";
    if (!missing && !find_number(pint, "cx_offset", &out->cx_offset))
        missing = "cx_offset";
    if (!missing && !find_number(pint, "cy_offset", &out->cy_offset))
        missing = "cy_offset";
    find_number(pint, "aspect_ratio", &out->aspect_ratio);
    double po = 0;
    if (find_number(pint, "poly_order", &po)) out->poly_order = (vc_i32)po;
    if (!missing) {
        out->width = (vc_i32)(w + 0.5);
        out->height = (vc_i32)(h + 0.5);
    }
    if (!missing && out->aspect_ratio <= 0.0) out->aspect_ratio = 1.0;

    find_string(buf, "name", out->name, sizeof(out->name));

    if (pext) {
        int nq = find_array(pext, "quaternion", out->quat, 4);
        int nt = find_array(pext, "translation", out->trans, 3);
        out->has_extrinsic = (nq == 4) ? 1 : 0;
        (void)nt;
    }

    free(buf);
    if (missing) {
        fprintf(stderr, "vc_ws_load: %s 缺少必需字段 \"%s\"\n", path, missing);
        return VC_E_CAM_NOT_CALIBRATED;
    }
    if (out->poly_order != 4) {
        fprintf(stderr, "vc_ws_load: %s 的 poly_order=%d, 本实现只支持 4\n", path,
                (int)out->poly_order);
        return VC_E_CAM_NOT_CALIBRATED;
    }
    return VC_OK;
}

VCStatus vc_ws_to_camera(const VCWoodScapeCalib *calib, VCCameraModel *out) {
    if (!calib || !out) return VC_E_PARAM_NULL;
    const vc_f64 k[4] = {calib->k1, calib->k2, calib->k3, calib->k4};
    /* 主点 = w/2 + offset。WoodScape readme 里的 "-0.5" 有两种读法, 差 0.5px,
     * 现有图像判据分辨不出来; 取不减的理由是结构性的: 这样 max(rho) 恰好
     * 等于主点到边界的距离。详见 README "主点约定"。 */
    const vc_f64 cx = calib->width / 2.0 + calib->cx_offset;
    const vc_f64 cy = calib->height / 2.0 + calib->cy_offset;
    return vc_cam_make_radial_poly(out, calib->width, calib->height, k, cx, cy,
                                   calib->aspect_ratio);
}

void vc_quat_to_mat(const vc_f64 q[4], vc_f64 R[9]) {
    /* q = (x, y, z, w), 与 scipy Rotation.from_quat 一致 */
    vc_f64 n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n < 1e-12) {
        R[0] = 1; R[1] = 0; R[2] = 0;
        R[3] = 0; R[4] = 1; R[5] = 0;
        R[6] = 0; R[7] = 0; R[8] = 1;
        return;
    }
    const vc_f64 x = q[0] / n, y = q[1] / n, z = q[2] / n, w = q[3] / n;
    R[0] = 1 - 2 * (y * y + z * z);
    R[1] = 2 * (x * y - z * w);
    R[2] = 2 * (x * z + y * w);
    R[3] = 2 * (x * y + z * w);
    R[4] = 1 - 2 * (x * x + z * z);
    R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w);
    R[7] = 2 * (y * z + x * w);
    R[8] = 1 - 2 * (x * x + y * y);
}

void vc_ws_pose(const VCWoodScapeCalib *calib, vc_f64 R_out[9]) {
    vc_f64 Rc2v[9];
    vc_quat_to_mat(calib->quat, Rc2v);
    /* R_w2c = R_c2v^T */
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R_out[r * 3 + c] = Rc2v[c * 3 + r];
}
