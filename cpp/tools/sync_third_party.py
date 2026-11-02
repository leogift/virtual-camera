#!/usr/bin/env python3
"""
把本机安装的三方库同步进 `third_party/`（符号链接方式）。

## 为什么用符号链接而不是拷贝

`third_party/` 要进 git。实体库是几十 MB 的二进制，进仓库既臃肿又会在
不同机器上产生无意义的 diff。符号链接只存一个路径字符串，仓库干净，
换机器时重跑本脚本即可。

代价：链接指向本机绝对路径，**换机器必须重跑**。所以 `third_party/*/`
下的实体内容一律 gitignore，只提交本脚本与 `MANIFEST.txt`。

## 平台

在 macOS 上用 Homebrew 前缀；在 Linux 上找系统路径或 `$VCAM_DEPS_ROOT`。
交叉编译（目标非宿主）时不要跑本脚本 —— 宿主库链不进目标。

## 用法

    python3 tools/sync_third_party.py            # 建立链接
    python3 tools/sync_third_party.py --check    # 只检查, 不改动
    python3 tools/sync_third_party.py --clean    # 删除链接, 保留目录

## 另一个副作用

在 exFAT/网络卷上（如 /Volumes/Workspace）创建/复制文件会生成 macOS 的
`._xxx` 资源分支文件。本脚本结束时顺手清理掉，否则每个链接旁边都会
多出一个 4 KB 垃圾文件。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CPP_ROOT = os.path.dirname(HERE)
THIRD_PARTY = os.path.join(CPP_ROOT, "third_party")

# vcam 实际用到的 OpenCV 模块。**只链接用得到的**：OpenCV 5 装了 40+ 个
# 模块，全链进来会让部署包膨胀几倍，而本工程只用 core/imgproc/imgcodecs。
OPENCV_MODULES = ("core", "imgproc", "imgcodecs")


def candidate_roots():
    """按优先级给出可能的依赖安装前缀。"""
    roots = []
    env = os.environ.get("VCAM_DEPS_ROOT")
    if env:
        roots.append(env)
    if sys.platform == "darwin":
        roots += ["/opt/homebrew/opt", "/usr/local/opt"]
    else:
        roots += ["/usr", "/usr/local"]
    return roots


def find_omp(root):
    for base in ("libomp", "llvm", "openmp"):
        inc = os.path.join(root, base, "include", "omp.h")
        if os.path.isfile(inc):
            libdir = os.path.join(root, base, "lib")
            for name in ("libomp.dylib", "libomp.so", "libomp.so.5"):
                lib = os.path.join(libdir, name)
                if os.path.isfile(lib):
                    return inc, lib
    return None, None


def find_opencv(root):
    """返回 (include 目录, {module: dylib/so 路径})。"""
    inc_cands = [os.path.join(root, "opencv", "include", "opencv5"),
                 os.path.join(root, "opencv", "include", "opencv4"),
                 os.path.join(root, "include", "opencv5"),
                 os.path.join(root, "include", "opencv4")]
    inc = None
    for c in inc_cands:
        if os.path.isdir(c):
            inc = c
            break
    if inc is None:
        return None, {}
    libdirs = [os.path.join(root, "opencv", "lib"), os.path.join(root, "lib")]
    libs = {}
    for m in OPENCV_MODULES:
        for d in libdirs:
            for ext in (".dylib", ".so"):
                p = os.path.join(d, f"libopencv_{m}{ext}")
                if os.path.isfile(p):
                    libs[m] = p
                    break
                # Linux 带版本后缀
                for cand in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                    if cand.startswith(f"libopencv_{m}{ext}"):
                        libs[m] = os.path.join(d, cand)
                        break
                if m in libs:
                    break
            if m in libs:
                break
    return inc, libs


def link(src, dst):
    """建立符号链接；已存在则先删。"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def clean_dot_underscore(path):
    """删掉 macOS 在非 HFS 卷上生成的 ._xxx 资源分支。"""
    n = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            if f.startswith("._"):
                try:
                    os.remove(os.path.join(dirpath, f))
                    n += 1
                except OSError:
                    pass
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查，不改动")
    ap.add_argument("--clean", action="store_true", help="删除 third_party 下的链接")
    args = ap.parse_args()

    if args.clean:
        for sub in ("libomp", "opencv"):
            d = os.path.join(THIRD_PARTY, sub)
            if os.path.isdir(d):
                for dirpath, _dn, filenames in os.walk(d):
                    for f in filenames:
                        p = os.path.join(dirpath, f)
                        if os.path.islink(p):
                            os.remove(p)
        print("已删除 third_party 下的符号链接")
        return 0

    found = {}
    for root in candidate_roots():
        if not os.path.isdir(root):
            continue
        if "libomp" not in found:
            inc, lib = find_omp(root)
            if inc and lib:
                found["libomp"] = (inc, lib, root)
        if "opencv" not in found:
            inc, libs = find_opencv(root)
            if inc and libs:
                found["opencv"] = (inc, libs, root)

    ok = True
    if "libomp" in found:
        inc, lib, root = found["libomp"]
        print(f"libomp   <- {root}")
        if not args.check:
            link(inc, os.path.join(THIRD_PARTY, "libomp/include/omp.h"))
            link(lib, os.path.join(THIRD_PARTY, "libomp/lib", os.path.basename(lib)))
    else:
        print("libomp   未找到（OpenMP 并行将被关闭）")
        ok = False

    if "opencv" in found:
        inc, libs, root = found["opencv"]
        print(f"opencv   <- {root}  ({len(libs)} 个模块: {', '.join(sorted(libs))})")
        if not args.check:
            link(inc, os.path.join(THIRD_PARTY, "opencv/include",
                                   os.path.basename(inc)))
            for m, p in libs.items():
                # 链接到**真实文件**（解析一层符号链接），并同时给短名。
                # Homebrew 里 libopencv_core.dylib 本身是指向
                # libopencv_core.500.dylib 的链接, 直接链它会绕两层;
                # 这里保留原始名字, 链接期用 -lopencv_core 时靠短名。
                real = os.path.realpath(p)
                link(real, os.path.join(THIRD_PARTY, "opencv/lib",
                                        os.path.basename(real)))
                ext = ".dylib" if real.endswith(".dylib") else ".so"
                link(real, os.path.join(THIRD_PARTY, "opencv/lib",
                                        f"libopencv_{m}{ext}"))
    else:
        print("opencv   未找到（转换工具将不构建）")
        ok = False

    if not args.check:
        with open(os.path.join(THIRD_PARTY, "MANIFEST.txt"), "w") as f:
            f.write("# 由 tools/sync_third_party.py 生成\n")
            for k, v in sorted(found.items()):
                f.write(f"{k}\n")
        # **清理必须放在所有写操作之后**：写 MANIFEST 本身也会在 exFAT 卷上
        # 产生 ._MANIFEST.txt，先清后写等于没清。
        n = clean_dot_underscore(THIRD_PARTY)
        if n:
            print(f"清理 {n} 个 ._ 资源分支文件")

    print("\n结果:", "OK" if ok else "部分缺失（见上）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
