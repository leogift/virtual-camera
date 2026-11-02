# vcam C++ 测试。分层: unit / parity
#
# 运行: ctest --test-dir build --output-on-failure
#       ctest --test-dir build -L unit

set(VCAM_TEST_INC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_rotation test_rotation.cpp)
target_link_libraries(test_rotation PRIVATE vcam_core)
target_compile_options(test_rotation PRIVATE -fexceptions)
add_test(NAME unit_rotation COMMAND test_rotation)
set_tests_properties(unit_rotation PROPERTIES LABELS unit)

add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera PRIVATE vcam_core)
target_compile_options(test_camera PRIVATE -fexceptions)
add_test(NAME unit_camera COMMAND test_camera)
set_tests_properties(unit_camera PROPERTIES LABELS unit)

add_executable(test_virtual test_virtual.cpp)
target_link_libraries(test_virtual PRIVATE vcam_core)
target_compile_options(test_virtual PRIVATE -fexceptions)
add_test(NAME unit_virtual COMMAND test_virtual)
set_tests_properties(unit_virtual PROPERTIES LABELS unit)

# ---------------------------------------------------------------- 对拍
# C++ 与 Python 逐像素对比。由外部脚本生成参考数据后再跑:
#   python tools/dump_reference.py --out tests/data/ref.txt
#   ctest -R parity
add_executable(bench_maps bench_maps.cpp)
target_link_libraries(bench_maps PRIVATE vcam_core)
target_compile_options(bench_maps PRIVATE -fexceptions)
# 基准不进 ctest: 计时受机器负载影响, 放进去只会制造不稳定测试

add_executable(test_parity test_parity.cpp)
target_link_libraries(test_parity PRIVATE vcam_core)
target_compile_options(test_parity PRIVATE -fexceptions)
add_test(NAME parity_python COMMAND test_parity ${CMAKE_CURRENT_SOURCE_DIR}/data/ref.txt)
set_tests_properties(parity_python PROPERTIES LABELS parity)
set_tests_properties(parity_python PROPERTIES SKIP_RETURN_CODE 77)
