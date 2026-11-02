# vcam C++ — 虚拟相机几何库 (鱼眼/针孔 内参转换 + 外参旋转 + remap)
#
# ## 快速开始
#   python3 tools/sync_third_party.py          # 建立 third_party/ 链接 (首次)
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j && ctest --test-dir build --output-on-failure
#
# ## 依赖
#   -DVCAM_DEPS=vendored   优先 third_party/ (默认)
#   -DVCAM_DEPS=system     只用 find_package / 系统路径
#   -DVCAM_DEPS=off        不引入任何三方库 (核心库仍可构建)
#
#   核心库 vcam_core **零第三方依赖**: 库文件里不得出现 OpenCV 符号
#   (同 NFOCC/cpp 的 V-BUILD-2)。OpenMP 是编译器运行时, 不算依赖。
#   OpenCV 只出现在 tools/vcam_convert 这一层。
#
# ## 加速
#   -DVCAM_USE_OPENMP=ON  (默认) 网格按行块并行
cmake_minimum_required(VERSION 3.16)

project(vcam LANGUAGES C CXX VERSION 1.0.0)

# ---------------------------------------------------------------- 代码规范
# NFOCC/cpp 的硬要求: C11 / C++11, 禁异常与 RTTI (三方库边界除外)。
# 本工程是纯几何计算, 没有理由开异常。
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # 供 clangd / IDE

if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

option(VCAM_BUILD_TESTS    "构建测试"                 ON)
option(VCAM_BUILD_TOOLS    "构建工具"                 ON)
option(VCAM_USE_OPENMP     "启用 OpenMP 并行"         ON)
option(VCAM_WITH_OPENCV    "构建需要 OpenCV 的工具"   ON)
option(VCAM_STRICT_WERROR  "警告即错误 (CI 用)"       OFF)
if(VCAM_STRICT_WERROR AND NOT MSVC)
  add_compile_options(-Werror)
endif()

# ---------------------------------------------------------------- 三方库
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcam_deps.cmake)

# ---------------------------------------------------------------- 核心库
file(GLOB_RECURSE VCAM_CORE_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/camera/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io/*.c
)
# macOS 会在每个文件旁生成 ._xxx 资源分支, GLOB 会把它当源文件 -> 编译报错
list(FILTER VCAM_CORE_SRC EXCLUDE REGEX "/\\.[^/]*$")

add_library(vcam_core STATIC ${VCAM_CORE_SRC})
add_library(vcam::core ALIAS vcam_core)
# 导出名必须显式设为 core，否则 install(EXPORT NAMESPACE vcam::) 会生成
# vcam::vcam_core —— 而文档和下游都写 vcam::core，配置期就报
# "but the target was not found"。ALIAS 不进导出表，光加 alias 不够。
set_target_properties(vcam_core PROPERTIES EXPORT_NAME core)
target_include_directories(vcam_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(NOT MSVC)
  target_compile_options(vcam_core PRIVATE -O3 -fno-exceptions -fno-rtti)
endif()

find_package(Threads REQUIRED)
target_link_libraries(vcam_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

set(VCAM_EXPORT_NEEDS_OPENMP OFF)
if(VCAM_USE_OPENMP)
  vcam_apply_openmp(vcam_core)
  # 供 cmake/vcamConfig.cmake.in 生成时使用：决定下游要不要 find OpenMP
  if(VCAM_OMP_FOUND)
    set(VCAM_EXPORT_NEEDS_OPENMP ON)
  endif()
endif()

# ---------------------------------------------------------------- 工具 (需要 OpenCV)
if(VCAM_BUILD_TOOLS AND VCAM_WITH_OPENCV AND VCAM_OPENCV_FOUND)
  add_executable(vcam_convert tools/vcam_convert.cpp)
  target_link_libraries(vcam_convert PRIVATE vcam_core)
  vcam_apply_opencv(vcam_convert)
  message(STATUS "vcam: vcam_convert 已启用")
endif()

# ---------------------------------------------------------------- 测试
if(VCAM_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------- 安装
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS vcam_core EXPORT vcamTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vcamTargets
        FILE vcamTargets.cmake
        NAMESPACE vcam::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vcam)

# 没有 Version 文件时，下游 `find_package(vcam 1.0 REQUIRED)` 直接失败，
# 且报错信息是 "could not find a configuration file matching the
# requested version" —— 看起来像没安装，其实是缺这一个文件。
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/vcamConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

# 模板必须**渲染**成 vcamConfig.cmake 再装。直接 install(FILES
# vcamConfig.cmake.in) 会把 .in 原样拷过去, 下游 find_package 找不到
# vcamConfig.cmake, 报 "Could not find a package configuration file"。
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vcamConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/vcamConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vcam)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vcamConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/vcamConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vcam)
