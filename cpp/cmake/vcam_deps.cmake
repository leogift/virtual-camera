# vcam 第三方依赖管理
#
# ## 与 NFOCC/cpp 的关系
# 这套 `nf_deps.cmake` 的同构精简版：保留了**三层来源选择**与 **RPATH**，
# 去掉了本工程用不到的 vcpkg/Conan 工具链探测。
#
# ## 三种来源，由 -DVCAM_DEPS=... 选择
#   vendored (默认)  优先 third_party/ 里的符号链接（由 tools/sync_third_party.py
#                    建立），树内自包含，换机器只需重跑同步脚本
#   system           只用 find_package / 系统路径
#   off              不引入任何三方库（核心库仍可构建，工具层跳过）
#
# ## 为什么必须管 RPATH
# 部署时 `vcam_convert` 依赖的 `libopencv_core.5.0.0.dylib` / `libomp.dylib`
# 不在系统默认搜索路径。不写 RPATH 就要用户手工设 `LD_LIBRARY_PATH` /
# `DYLD_LIBRARY_PATH`，量产镜像里不可接受。这里用**相对** RPATH，整个目录
# 拷过去就能跑：
#   macOS: @loader_path/../lib
#   Linux: $ORIGIN/../lib
#
# ## 交叉编译守卫
# 宿主（macOS）Homebrew 里是 arm64-darwin 的库，链不进 linux 目标。交叉编译
# 时跳过所有宿主探测，避免"配置成功、到链接期才报架构不匹配"。

set(VCAM_DEPS "vendored" CACHE STRING "三方库来源: vendored | system | off")
set_property(CACHE VCAM_DEPS PROPERTY STRINGS vendored system off)

set(VCAM_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# ---------------------------------------------------------------- RPATH
if(APPLE)
  set(VCAM_RPATH_BUILD   "@loader_path/../lib\;@loader_path/../../lib")
  set(VCAM_RPATH_INSTALL "@loader_path/../lib")
else()
  set(VCAM_RPATH_BUILD   "\$ORIGIN/../lib;\$ORIGIN/../../lib")
  set(VCAM_RPATH_INSTALL "\$ORIGIN/../lib")
endif()
set(CMAKE_SKIP_BUILD_RPATH            FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH    FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
set(CMAKE_INSTALL_RPATH               "${VCAM_RPATH_INSTALL}")
set(CMAKE_BUILD_RPATH                 "${VCAM_RPATH_BUILD}")

# ---------------------------------------------------------------- 交叉编译
set(VCAM_CROSS_COMPILING OFF)
if(CMAKE_CROSSCOMPILING)
  set(VCAM_CROSS_COMPILING ON)
  message(STATUS "vcam: 交叉编译模式 -> 跳过宿主依赖探测 (${CMAKE_SYSTEM_PROCESSOR})")
endif()

# ---------------------------------------------------------------- OpenMP
set(VCAM_OMP_FOUND OFF)
if(NOT VCAM_DEPS STREQUAL "off" AND NOT VCAM_CROSS_COMPILING)
  # (1) vendored: third_party/libomp/
  if(VCAM_DEPS STREQUAL "vendored")
    find_path(VCAM_OMP_INC omp.h
      PATHS "${VCAM_THIRD_PARTY_DIR}/libomp/include" NO_DEFAULT_PATH)
    find_library(VCAM_OMP_LIB NAMES omp libomp
      PATHS "${VCAM_THIRD_PARTY_DIR}/libomp/lib" NO_DEFAULT_PATH)
    if(VCAM_OMP_INC AND VCAM_OMP_LIB)
      set(VCAM_OMP_FOUND ON)
      set(VCAM_OMP_FROM "third_party/")
    endif()
  endif()

  # (2) system: OpenMP::OpenMP_C 或 find_library 兜底
  if(NOT VCAM_OMP_FOUND)
    find_package(OpenMP QUIET COMPONENTS C CXX)
    if(OpenMP_C_FOUND)
      set(VCAM_OMP_FOUND ON)
      set(VCAM_OMP_FROM "find_package(OpenMP)")
    else()
      # macOS/AppleClang 默认不带 libomp。**不加 -Xpreprocessor -fopenmp
      # 的话 `#pragma omp` 会被静默忽略** —— 编译通过、完全串行。这是
      # 全工程最容易漏的一条。
      find_path(VCAM_OMP_INC omp.h
        HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
      find_library(VCAM_OMP_LIB omp
        HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
      if(VCAM_OMP_INC AND VCAM_OMP_LIB)
        set(VCAM_OMP_FOUND ON)
        set(VCAM_OMP_FROM "系统路径")
      endif()
    endif()
  endif()
endif()

if(VCAM_OMP_FOUND)
  message(STATUS "vcam: OpenMP <- ${VCAM_OMP_FROM}")
elseif(NOT VCAM_DEPS STREQUAL "off" AND NOT VCAM_CROSS_COMPILING)
  message(STATUS "vcam: OpenMP 不可用, 串行编译 (跑 tools/sync_third_party.py 可补)")
endif()

# 把 OpenMP 接到某个 target 上（含 macOS 的显式编译选项）
#
# **include 目录与编译选项一律 PRIVATE**：公共头文件 vc_*.h 不 include
# omp.h，所以第三方 include 不该进 INTERFACE。否则 `install(EXPORT)` 会报
# "INTERFACE_INCLUDE_DIRECTORIES contains path prefixed in the source
# directory" —— 而这条错误只在装包时才暴露，构建本身是好的。
# 宏 VCAM_HAS_OPENMP 要 PUBLIC：调用方用 vc_have_openmp() 看它。
function(vcam_apply_openmp target)
  if(NOT VCAM_OMP_FOUND)
    return()
  endif()
  target_compile_definitions(${target} PUBLIC VCAM_HAS_OPENMP=1)
  if(OpenMP_C_FOUND AND NOT VCAM_OMP_FROM STREQUAL "third_party/"
     AND NOT VCAM_OMP_FROM STREQUAL "系统路径")
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_C)
    return()
  endif()
  target_include_directories(${target} PRIVATE ${VCAM_OMP_INC})
  target_compile_options(${target} PRIVATE -Xpreprocessor -fopenmp)
  # libomp 头在 C23 下对自己的枚举值报 -Wc23-extensions,
  # 不是我们的代码, 屏蔽掉以免淹没真正的警告
  target_compile_options(${target} PRIVATE -Wno-c23-extensions)

  # **静态库的链接依赖会进导出表**（下游必须补上这些符号，否则
  # `___kmpc_*` undefined reference）。但直接写 ${VCAM_OMP_LIB} 会把
  # 本机绝对路径写进 vcamTargets.cmake，换机器即断。
  #
  # 分两个作用域：
  #   BUILD_INTERFACE   -> 真实路径, 本工程内部构建与测试用
  #   INSTALL_INTERFACE -> OpenMP::OpenMP_C, 由下游的 find_dependency
  #                        拿到(见 cmake/vcamConfig.cmake.in)。
  # 写裸库名 `omp` 是不够的: 它不在默认搜索路径, ld 直接报
  # "library 'omp' not found"。
  target_link_libraries(${target} PRIVATE
    "$<BUILD_INTERFACE:${VCAM_OMP_LIB}>")
  set_property(TARGET ${target} APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES
    "$<BUILD_INTERFACE:${VCAM_OMP_LIB}>"
    "$<INSTALL_INTERFACE:OpenMP::OpenMP_C>")
endfunction()

# ---------------------------------------------------------------- OpenCV
set(VCAM_OPENCV_FOUND OFF)
set(VCAM_OPENCV_INCLUDE "")
set(VCAM_OPENCV_LIBS "")
set(VCAM_OPENCV_FROM "")

if(NOT VCAM_DEPS STREQUAL "off" AND NOT VCAM_CROSS_COMPILING)
  # 只找用得到的模块: core / imgproc / imgcodecs
  set(VCAM_OPENCV_MODULES core imgproc imgcodecs)

  if(VCAM_DEPS STREQUAL "vendored")
    # third_party/opencv/include/opencv5 与 lib/libopencv_*.dylib
    file(GLOB _vci "${VCAM_THIRD_PARTY_DIR}/opencv/include/opencv*")
    if(_vci)
      list(GET _vci 0 _vci0)
      set(_vc_inc "${_vci0}")
      set(_vc_libs "")
      set(_vc_ok TRUE)
      foreach(m ${VCAM_OPENCV_MODULES})
        find_library(_vcl_${m} NAMES opencv_${m}
          PATHS "${VCAM_THIRD_PARTY_DIR}/opencv/lib" NO_DEFAULT_PATH)
        if(_vcl_${m})
          list(APPEND _vc_libs "${_vcl_${m}}")
        else()
          set(_vc_ok FALSE)
        endif()
      endforeach()
      if(_vc_ok)
        set(VCAM_OPENCV_FOUND ON)
        set(VCAM_OPENCV_INCLUDE "${_vc_inc}")
        set(VCAM_OPENCV_LIBS "${_vc_libs}")
        set(VCAM_OPENCV_FROM "third_party/")
      endif()
    endif()
  endif()

  if(NOT VCAM_OPENCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS ${VCAM_OPENCV_MODULES})
    if(OpenCV_FOUND)
      set(VCAM_OPENCV_FOUND ON)
      set(VCAM_OPENCV_INCLUDE "${OpenCV_INCLUDE_DIRS}")
      set(VCAM_OPENCV_LIBS "${OpenCV_LIBS}")
      set(VCAM_OPENCV_FROM "find_package(OpenCV ${OpenCV_VERSION})")
    endif()
  endif()
endif()

if(VCAM_OPENCV_FOUND)
  message(STATUS "vcam: OpenCV <- ${VCAM_OPENCV_FROM}")
elseif(NOT VCAM_DEPS STREQUAL "off" AND NOT VCAM_CROSS_COMPILING)
  message(STATUS "vcam: 未找到 OpenCV, 转换工具跳过 (跑 tools/sync_third_party.py 可补)")
endif()

function(vcam_apply_opencv target)
  if(NOT VCAM_OPENCV_FOUND)
    return()
  endif()
  # **SYSTEM 而非 PRIVATE**: OpenCV 5 的头文件用了 C++17 的 [[nodiscard]]
  # 和 C11 的 _Atomic, 在 -std=c++11 下每条都报 -Wc++17-attribute-extensions,
  # 首次构建刷出四十多条警告, 把工程自己的警告全淹了。
  # SYSTEM 让编译器对三方头静默, 我们自己的代码仍然 -Wall -Wextra。
  target_include_directories(${target} SYSTEM PRIVATE ${VCAM_OPENCV_INCLUDE})
  target_link_libraries(${target} PRIVATE ${VCAM_OPENCV_LIBS})
  # OpenCV 会抛 cv::Exception。我们**不在库里 catch**(跨 C++ 标准边界的
  # 异常 ABI 不匹配, 实测 SIGTRAP), 改为调用前自查。但包含 OpenCV 头文件的
  # TU 必须开异常支持才能编译。
  target_compile_options(${target} PRIVATE -fexceptions)
endfunction()
