# vcam — 虚拟相机几何库

把车载鱼眼相机虚拟成任意姿态、任意视场的理想针孔相机（或反向），
生成 OpenCV `remap` 用的坐标映射表。**Python 与 C++ 双实现，数值逐位一致。**

```
虚拟像素 (u,v)
  --虚拟相机 unproject-->  虚拟相机系射线
  --R_rel = R_src · R_dst^T-->  源相机系射线
  --源相机 project-->  源图像素
```

支持 4 种相机模型：针孔（含 Brown-Conrady 畸变）、Kannala-Brandt、
径向多项式（WoodScape / NFOCC 数据集格式）、增强型 e-ftheta。

## 为什么有这个库

原始实现 `virtual_camera_core.py` 存在一个会静默污染结果的 Bug：`camera`
欧拉角约定的"构造"与"分解"用了**不同**的旋转链，往返必然失败
（`yaw=30, pitch=20, roll=10` 分解回来 `pitch=11.82°`），没有任何测试捕获它。

本库把每种约定实现为一个 `RotationConvention` 对象，**强制成对提供**
`to_matrix` / `from_matrix`，并用往返测试守住。同时在真实数据上发现并修正
了 9 个"能跑、图看着也对、但几何错"的问题——逐条记录在 `CHANGES.md` 第三节。

## 快速开始（Python）

```python
from vcam import FisheyeCamera, VirtualCamera, GridConverter
from vcam.io import NFOCCDataset

ds  = NFOCCDataset('/path/to/NFOCC/datasets/fisheye')
src = ds.camera('FV')                                    # 真实鱼眼，读标定
dst = VirtualCamera.from_fov(640, 297, fov_h_deg=150, fov_v_deg=120)

conv = GridConverter(src, dst, yaw_delta_deg=+45)        # +45 = 向左看
print(conv.dst.pose.facing())                            # "left 44.9deg, down 22.7deg"

out = conv.remap(image_bgr)                              # cv2.remap
out.save_maps('map_x.txt', 'map_y.txt')
```

命令行：

```bash
python tools/convert_nfocc.py --all --yaws -45 0 45 --mosaic
python tools/verify_virtualization.py --camera FV
```

## 快速开始（C++）

```bash
cd cpp
python3 tools/sync_third_party.py        # 首次：把本机 libomp/OpenCV 链进 third_party/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

第三方依赖统一由 `cpp/cmake/vcam_deps.cmake` 管理，三档可选：

| `-DVCAM_DEPS=` | 行为 |
| :--- | :--- |
| `vendored`（默认） | 优先 `third_party/` 里的链接，树内自包含 |
| `system` | 只用 `find_package` / 系统路径 |
| `off` | 不引入任何三方库，核心库仍可构建（工具跳过） |

`third_party/` 里存的是**符号链接**而不是实体库——仓库干净，换机器重跑
`sync_third_party.py` 即可。这套做法照搬 NFOCC/cpp 的 `nf_deps.cmake`。

```cpp
#include "vcam/vc_virtual.h"

VCCameraModel src;
vc_cam_make_radial_poly(&src, 1280, 966, k, cx, cy, 1.0);

VCCameraModel dst;
vc_cam_make_pinhole(&dst, 640, 297, fx, fy, cx_v, cy_v);

VCVirtualConversion conv;
vc_conv_init(&conv, &src, &dst, /*yaw*/45.0, /*pitch*/0.0, /*roll*/0.0,
             VC_CONV_VALEO);

VCMapStats stats;
vc_conv_build_maps(&conv, map_x, map_y, &stats, /*chunk_rows*/0);
```

命令行：

```bash
cpp/build/vcam_convert --calib calib.json --image img.png \
    --yaws -45 0 45 --fov-h 150 --fov-v 120 --width 640 --output-dir out/
```

## 坐标约定

| 坐标系 | X | Y | Z |
| :--- | :--- | :--- | :--- |
| 车体（ISO 8855） | 前 | 左 | 上 |
| 相机（光学） | 右 | 下 | 前（光轴） |

`R` 一律是 **world → cam** 的 3×3。光轴方向 = `R.T @ [0,0,1]`（即 R 的第 3 行）。

### 方向语义（有数值断言）

`valeo` 约定下（虚拟相机默认）：

| 参数 | 正值含义 | 断言位置 |
| :--- | :--- | :--- |
| `yaw` | 向**左**看（光轴转向车体 +Y） | `test_valeo_positive_yaw_turns_left` |
| `pitch` | **抬**头（光轴 Z 分量为正） | `test_valeo_positive_pitch_looks_up` |
| `roll` | 图像**顺**时针 | `test_rotation.py::orthonormal` |

**不要依赖变量名判断方向，看 `facing()` 的输出**——它从光轴几何量直接算，
不经过任何命名约定。旧代码正是在这里出错：`yaw_left = yaw_real - 45`
实际输出的是右侧视图。

### 旋转的三种量，别混用

| 量 | 含义 | 零值 | 用途 |
| :--- | :--- | :--- | :--- |
| `R_abs(yaw,pitch,roll)` | 绝对姿态（含 `PRE_ROT`） | `PRE_ROT`，**不是 I** | 从标定外参描述相机朝向 |
| `R_rel` | 源相机系内的相对旋转 | **I** | 网格构建（几何用它） |
| `R_delta_world` | 车体系内可左乘的增量 | I | 由 `R_rel` 求目标绝对姿态（仅打印用） |

混用会产出"看起来正常但几何错"的结果：把 `R_abs` 当增量用，零增量的恒等
映射会变成 90° 旋转；自己写 `Rz·Rx·Ry` 当增量，零增量对了但 yaw 变成绕相机
Y 轴转，方向与物理脱钩。细节见 `CHANGES.md` §3.5。

## 相机模型

| 名称 | 公式 | 典型参数 |
| :--- | :--- | :--- |
| `pinhole` | `x_d = x(1+k1r²+k2r⁴+k3r⁶+k4r⁸) + 切向` | 内部 / 虚拟相机 |
| `kb` | `ρ = θ(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸)`，乘 `f` | 常见鱼眼标定 |
| `4thpoly` | `ρ = k1θ+k2θ²+k3θ³+k4θ⁴`，**k1 单位是像素/弧度** | WoodScape / NFOCC |
| `eftheta` | 同上 + 仿射斜切 `A=[[c,d],[e,1]]` | 部分国产鱼眼 |

**`4thpoly` 的 k1 就是小角度等效焦距**（NFOCC FV 实测 339.749，图像宽 1280），
不是通常意义的畸变系数。按 KB 公式套用会完全失真。

### 有效性判据

- **鱼眼**：只看是否落在图像内。**不限制 θ ≤ 90°**——WoodScape FV 的
  `ρ(95°)=641 px` 仍在图像内（主点到左边界 644 px），θ>90° 的像素是合法的。
- **针孔**：`Z > 0` 且落在图像内。

不可见的像素统一返回 **`-1`**（不是 NaN、不是 `u<0`）。判断有效性请用
`== -1` 哨兵，不要用 `>= 0`：边界像素的浮点坐标可能是 `-1.7e-13`。

## 项目结构

```
vcam/                    Python 实现
  core/rotation.py       旋转与欧拉角（约定成对实现 + 往返测试）
  models/                pinhole / kb / radial_poly / eftheta，一个模型一个文件
  pipeline/              CameraPose / FisheyeCamera / VirtualCamera / GridConverter
  io/                    外部格式解析（唯一认识第三方 JSON 的地方）
  compat.py              旧 virtual_camera_core.py 的兼容 API

cpp/                     C++11 实现
  include/vcam/          纯 C 接口（extern "C"），与 NFOCC/cpp 同一套命名规范
  src/camera/            相机模型
  src/geometry/          旋转 + 网格构建（OpenMP 按行块并行）
  src/io/                WoodScape 标定读取（零依赖）
  tools/vcam_convert.cpp 转换 CLI（OpenCV 只在这一层）
  tests/                 单元测试 + 与 Python 的数值对拍

tools/                   命令行入口与探针（见下表）
tests/                   Python 测试
```

**核心库零第三方依赖**：`ldd libvcam_core.a` 里没有 OpenCV。OpenMP 是编译器
运行时，不算依赖。这与 NFOCC/cpp 的 `V-BUILD-2` 约定一致。

## 工具

| 脚本 | 用途 |
| :--- | :--- |
| `tools/convert_nfocc.py` | 批量转换 NFOCC/WoodScape 四路相机 |
| `tools/verify_virtualization.py` | 图像验收（4 条判据，见下） |
| `tools/dump_reference.py` | 转储 Python 几何参考数据供 C++ 对拍 |
| `tools/cross_check_images.py` | C++ 与 Python 出图逐像素对比 |
| `tools/bench_maps.py` | 网格构建吞吐与 OpenMP 加速比 |
| `tools/probe_principal_point.py` | 主点约定探针 |

## 验证

```bash
# Python: 92 passed, 1 skipped
.venv/bin/python -m pytest tests/ -q

# C++: 4/4 passed（38 用例，327k 次断言）
ctest --test-dir cpp/build --output-on-failure

# 数值对拍
.venv/bin/python tools/dump_reference.py --out cpp/tests/data/ref.txt
cpp/build/tests/test_parity cpp/tests/data/ref.txt
```

实测结果：

| 项目 | 结果 |
| :--- | :--- |
| 旋转往返一致性 | 两约定各 175 组角度，误差 < 1.8e-15 |
| 正投影对拍 | 1200 点，最大差 **2.27e-13** |
| 反投影对拍 | 812 点，最大差 **1.99e-10** |
| 网格对拍 | 6 张 map，最大差 **5.00e-06** |
| 端到端出图 | 12 张图（4 路 × 3 视角），最大差 **1 灰度级** |
| 几何闭环 | 中位 5.7e-14 px，最大 1.7e-07 px |

图像验收的 4 条判据（`tools/verify_virtualization.py`）：

- **A 直线笔直性**：世界直线在针孔图里残差 0.000 px，在鱼眼图里 17.06 px
  → 证明去畸变确实发生，且用的是真几何而非近似。
- **B 重采样一致性**：虚拟图 vs 按 map 重采样，中位差 0.0000。
- **C 几何闭环**：虚拟像素 → 源图 → 虚拟像素，最大 1.7e-07 px。
- **C2 方向语义**：对称偏航必须把虚拟中心的采点送到源图主点两侧
  （`u1 < cx < u2`），与相机的绝对朝向无关。

四路相机 × 4 条判据全部 PASS。

> 早期版本用过"Hough 最长直线"和"连通域提网格线"做判据，两者都被证明
> **无效**（前者测的是场景里有几条长边，后者在网格相交时连成一整块，
> 残差 523 px 无分辨力）。见 `CHANGES.md` §3.9。

## 性能

1920×1080 虚拟针孔，150°×120° FOV，20 次平均：

| 线程 | ms/map | 加速比 |
| ---: | ---: | ---: |
| 1 | 43.84 – 44.91 | 1.00× |
| 2 | 23.16 | 1.94× |
| 4 | 12.33 | 3.64× |
| 8 | 6.88 – 6.93 | **6.37 – 6.48×** |

（8 线程一列是两次独立运行的范围；单次测量有 ±2% 波动。）

OpenMP 按**行块**并行（默认 16 行/块）。不按行并行是因为每行只有 1920 像素，
行级 fork/join 开销会吃掉收益。

## 已知限制

1. **主点约定的 0.5 px 存疑未决。** WoodScape readme 的
   `u = u' + cx_offset + width/2 - 0.5` 有两种读法，现有图像判据分辨不出来
   （噪声 ±5 px，两者差 0.5 px）。本实现取**不减**，理由是结构性的：
   此时 `max(ρ)` 恰好等于主点到图像边界的距离。若上游确认需要减 0.5，
   改两处即可。
2. **C++ 的 WoodScape 解析器只认本格式**，不是通用 JSON。
3. **`vcam_convert` 只支持径向多项式源相机**；KB / e-ftheta 需自行接线。

## 许可

未指定。

---

English version: [README_EN.md](README_EN.md)
