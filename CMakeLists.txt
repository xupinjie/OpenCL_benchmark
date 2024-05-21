cmake_minimum_required(VERSION 3.14)
project(opencl_benchmark)

set(CMAKE_CXX_STANDARD 11)

set(PPLCOMMON_BUILD_TESTS OFF CACHE BOOL "disable pplcommon tests")
set(PPLCOMMON_BUILD_BENCHMARK OFF CACHE BOOL "disable pplcommon benchmark")
set(PPLCOMMON_ENABLE_PYTHON_API OFF)
set(PPLCOMMON_ENABLE_LUA_API OFF)
set(PPLCOMMON_HOLD_DEPS OFF)
set(PPLCOMMON_USE_X86_64 OFF)
set(PPLCOMMON_USE_AARCH64 ${PPLNN_USE_AARCH64})
set(PPLCOMMON_USE_ARMV7 ${PPLNN_USE_ARMV7})
set(PPLCOMMON_USE_CUDA OFF)
set(PPLCOMMON_USE_OPENCL ${PPLNN_USE_OCL})

add_subdirectory(deps/ppl.common)

add_executable(fp16_diff ${CMAKE_CURRENT_SOURCE_DIR}/tools/fp16_diff.cc)
target_link_libraries(fp16_diff pplcommon_static)
