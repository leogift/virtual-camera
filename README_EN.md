# vcam — Virtual Camera Geometry Library

Renders an automotive fisheye camera as an ideal pinhole camera with arbitrary
pose and field of view, producing coordinate maps for OpenCV `remap`.
**Reference implementations in Python and C++11, numerically bit-compatible.**

```
virtual pixel (u,v)
  --virtual camera unproject-->  ray in virtual camera frame
  --R_rel = R_src · R_dst^T-->   ray in source camera frame
  --source camera project-->     source pixel
```

Four camera models are supported: pinhole (with Brown-Conrady distortion),
Kannala-Brandt, radial polynomial (WoodScape / NFOCC dataset format), and
enhanced e-ftheta.

## Why this library exists

The original `virtual_camera_core.py` carried a bug that silently corrupted
results: the `camera` Euler-angle convention built its rotation with one chain
and decomposed it with a *different* one, so a round trip was always wrong
(`yaw=30, pitch=20, roll=10` decomposed back to `pitch=11.82°`). No test
caught it.

Here each convention is a `RotationConvention` object that **must** provide both
`to_matrix` and `from_matrix`, guarded by round-trip tests. Running against real
data also surfaced nine further defects of the "it runs, the image looks fine,
but the geometry is wrong" kind — catalogued in `CHANGES.md` §3.

## Quick start (Python)

```python
from vcam import FisheyeCamera, VirtualCamera, GridConverter
from vcam.io import NFOCCDataset

ds  = NFOCCDataset('/path/to/NFOCC/datasets/fisheye')
src = ds.camera('FV')                                    # real fisheye + calibration
dst = VirtualCamera.from_fov(640, 297, fov_h_deg=150, fov_v_deg=120)

conv = GridConverter(src, dst, yaw_delta_deg=+45)        # +45 = look left
print(conv.dst.pose.facing())                            # "left 44.9deg, down 22.7deg"

out = conv.remap(image_bgr)                              # cv2.remap
out.save_maps('map_x.txt', 'map_y.txt')
```

Command line:

```bash
python tools/convert_nfocc.py --all --yaws -45 0 45 --mosaic
python tools/verify_virtualization.py --camera FV
```

## Quick start (C++)

```bash
cd cpp
python3 tools/sync_third_party.py        # first run: link local libomp/OpenCV into third_party/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Third-party dependencies are managed by `cpp/cmake/vcam_deps.cmake`, with
three modes:

| `-DVCAM_DEPS=` | Behaviour |
| :--- | :--- |
| `vendored` (default) | prefer links in `third_party/`, tree is self-contained |
| `system` | use `find_package` / system paths only |
| `off` | no third-party libs; the core library still builds (tools skipped) |

`third_party/` holds **symbolic links**, not real libraries — the repo stays
clean and a new machine just reruns `sync_third_party.py`. This mirrors
NFOCC/cpp's `nf_deps.cmake`.

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

Command line:

```bash
cpp/build/vcam_convert --calib calib.json --image img.png \
    --yaws -45 0 45 --fov-h 150 --fov-v 120 --width 640 --output-dir out/
```

## Coordinate conventions

| Frame | X | Y | Z |
| :--- | :--- | :--- | :--- |
| Vehicle (ISO 8855) | forward | left | up |
| Camera (optical) | right | down | forward (optical axis) |

`R` is always the **world → cam** 3×3. The optical axis is `R.T @ [0,0,1]`,
i.e. row 3 of `R`.

### Direction semantics (numerically asserted)

Under the `valeo` convention (the default for virtual cameras):

| Parameter | Positive means | Assertion |
| :--- | :--- | :--- |
| `yaw` | look **left** (axis rotates toward +Y) | `test_valeo_positive_yaw_turns_left` |
| `pitch` | look **up** (axis Z component positive) | `test_valeo_positive_pitch_looks_up` |
| `roll` | image rotates **clockwise** | `test_rotation.py::orthonormal` |

**Do not infer direction from a variable name — read `facing()`.** It is
computed from the optical-axis geometry and goes through no naming convention.
The original code failed exactly here: `yaw_left = yaw_real - 45` actually
produced the *right* view.

### Three kinds of rotation, do not mix them

| Quantity | Meaning | Zero value | Used for |
| :--- | :--- | :--- | :--- |
| `R_abs(yaw,pitch,roll)` | absolute pose (includes `PRE_ROT`) | `PRE_ROT`, **not I** | describing camera orientation from calibration |
| `R_rel` | relative rotation in the source camera frame | **I** | map construction (this is the geometry) |
| `R_delta_world` | increment, left-multipliable in vehicle frame | I | deriving target absolute pose from `R_rel` (display only) |

Mixing them yields plausible-looking but geometrically wrong output. Treating
`R_abs` as an increment turns the identity map into a 90° rotation; hand-writing
`Rz·Rx·Ry` as an increment gets zero correct but makes yaw rotate about the
camera Y axis, detaching the sign from physical direction. See `CHANGES.md` §3.5.

## Camera models

| Name | Formula | Typical use |
| :--- | :--- | :--- |
| `pinhole` | `x_d = x(1+k1r²+k2r⁴+k3r⁶+k4r⁸) + tangential` | internal / virtual cameras |
| `kb` | `ρ = θ(1+k1θ²+k2θ⁴+k3θ⁶+k4θ⁸)`, times `f` | common fisheye calibration |
| `4thpoly` | `ρ = k1θ+k2θ²+k3θ³+k4θ⁴`, **k1 in px/rad** | WoodScape / NFOCC |
| `eftheta` | as above plus affine skew `A=[[c,d],[e,1]]` | some production fisheyes |

**For `4thpoly`, k1 *is* the small-angle focal length** (measured 339.749 on
NFOCC FV, image width 1280), not a distortion coefficient. Feeding it to the KB
formula distorts everything beyond θ ≈ 0.2 rad.

### Validity

- **Fisheye**: only "is it inside the image". **No θ ≤ 90° restriction** —
  WoodScape FV has `ρ(95°) = 641 px` still inside the image (principal point to
  left edge is 644 px), so pixels with θ > 90° are legitimate.
- **Pinhole**: `Z > 0` and inside the image.

Invisible pixels return **`-1`** (not NaN, not `u < 0`). Test validity with the
`== -1` sentinel, never `>= 0`: a boundary pixel's float coordinate may be
`-1.7e-13`.

## Layout

```
vcam/                    Python implementation
  core/rotation.py       rotations and Euler angles (paired to_matrix/from_matrix)
  models/                pinhole / kb / radial_poly / eftheta, one model per file
  pipeline/              CameraPose / FisheyeCamera / VirtualCamera / GridConverter
  io/                    external format parsing (the only place that knows JSON)
  compat.py              backward-compatible API for the old virtual_camera_core.py

cpp/                     C++11 implementation
  include/vcam/          plain C interface (extern "C"), same naming as NFOCC/cpp
  src/camera/            camera models
  src/geometry/          rotation + map construction (OpenMP row-chunked)
  src/io/                WoodScape calibration reader (zero dependencies)
  tools/vcam_convert.cpp conversion CLI (OpenCV only in this layer)
  tests/                 unit tests + numeric parity against Python

tools/                   CLI entry points and probes
tests/                   Python tests
```

**The core library has zero third-party dependencies**: no OpenCV in
`libvcam_core.a`. OpenMP is a compiler runtime and does not count. This matches
NFOCC/cpp's `V-BUILD-2` rule.

## Tools

| Script | Purpose |
| :--- | :--- |
| `tools/convert_nfocc.py` | batch-convert the four NFOCC/WoodScape cameras |
| `tools/verify_virtualization.py` | image acceptance (4 criteria, below) |
| `tools/dump_reference.py` | dump Python geometry reference data for C++ parity |
| `tools/cross_check_images.py` | pixel-wise comparison of C++ vs Python output |
| `tools/bench_maps.py` | map build throughput and OpenMP speedup |
| `tools/probe_principal_point.py` | principal-point convention probe |

## Verification

```bash
# Python: 92 passed, 1 skipped
.venv/bin/python -m pytest tests/ -q

# C++: 4/4 passed (38 cases, 327k assertions)
ctest --test-dir cpp/build --output-on-failure

# numeric parity
.venv/bin/python tools/dump_reference.py --out cpp/tests/data/ref.txt
cpp/build/tests/test_parity cpp/tests/data/ref.txt
```

Measured:

| Item | Result |
| :--- | :--- |
| Rotation round trip | 175 angle triples per convention, error < 1.8e-15 |
| Forward projection parity | 1200 points, worst **2.27e-13** |
| Inverse projection parity | 812 points, worst **1.99e-10** |
| Map parity | 6 maps, worst **5.00e-06** |
| End-to-end images | 12 images (4 cameras x 3 views), worst **1 grey level** |
| Geometric closure | median 5.7e-14 px, max 1.7e-07 px |

The four acceptance criteria in `tools/verify_virtualization.py`:

- **A: line straightness** — world lines land with 0.000 px residual in the
  pinhole image and 17.06 px in the fisheye image, proving undistortion actually
  happens and uses the real geometry rather than an approximation.
- **B: resampling consistency** — virtual image vs remap from the map, median 0.0000.
- **C: geometric closure** — virtual pixel → source → virtual pixel, max 1.7e-07 px.
- **C2: direction semantics** — symmetric yaw must place the virtual centre's
  sample point on opposite sides of the source principal point
  (`u1 < cx < u2`), independent of the camera's absolute heading.

All four cameras pass all four criteria.

> Earlier probe versions used "longest Hough line" and "connected-component
> grid extraction". Both were shown to be **invalid** (the first measures how
> many long edges the scene has; the second merges intersecting grid lines into
> one blob, giving a 523 px residual with no discriminating power). See
> `CHANGES.md` §3.9.

## Performance

1920×1080 virtual pinhole, 150°×120° FOV, mean of 20 runs:

| Threads | ms/map | Speedup |
| ---: | ---: | ---: |
| 1 | 43.84 – 44.91 | 1.00x |
| 2 | 23.16 | 1.94x |
| 4 | 12.33 | 3.64x |
| 8 | 6.88 – 6.93 | **6.37 – 6.48x** |

(The 8-thread column is the range over two independent runs; single
measurements vary by about ±2%.)

OpenMP parallelises over **row chunks** (16 rows each). Per-row parallelism was
rejected: a row is only 1920 pixels of a few dozen flops, so fork/join overhead
eats the gain.

## Known limitations

1. **A 0.5 px principal-point ambiguity is unresolved.** WoodScape's readme
   writes `u = u' + cx_offset + width/2 - 0.5`, which admits two readings. No
   image-based criterion in this project can separate them (noise ±5 px vs a
   0.5 px difference). This implementation does **not** subtract 0.5, for one
   structural reason: then `max(ρ)` equals exactly the distance from the
   principal point to the image border. If upstream confirms the subtraction,
   two places need changing.
2. **The C++ WoodScape reader is format-specific**, not a general JSON parser.
3. **`vcam_convert` only supports radial-polynomial source cameras**; KB and
   e-ftheta require wiring through the core API.

## License

Unspecified.

---

Chinese version: [README.md](README.md)
