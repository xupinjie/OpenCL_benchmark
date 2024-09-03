cmake_minimum_required(VERSION 3.14)
project(opencl_benchmark)

set(CMAKE_CXX_STANDARD 11)

### 初始化ppl.common flag
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

set(g_crypto_key "pplkernelopencl9")
set(g_crypto_iv "inferenceengine9")

add_definitions(-DG_CRYPTO_KEY="${g_crypto_key}")
add_definitions(-DG_CRYPTO_IV="${g_crypto_iv}")

### OpenCL file转换成HEX文件
# 查找所有内核文件
file(GLOB_RECURSE KERNEL_FILES clcode/*.cl)
# 处理每个内核文件
foreach(KERNEL_FILE IN ITEMS ${KERNEL_FILES})
    # 读取内核文件内容并转换为 HEX
    file(STRINGS ${KERNEL_FILE} KERNEL_STRING NEWLINE_CONSUME)
    string(HEX ${KERNEL_STRING} OCLKERNEL_HEX)

    # 将十六进制内容格式化为 C++ 数组初始化格式
    string(REGEX REPLACE "(.)(.)" "0x\\1\\2, " CONTENT_HEX_FORMATTED ${OCLKERNEL_HEX})

    # 提取内核文件名并创建有效的 C++ 标识符
    get_filename_component(KERNEL_FILENAME ${KERNEL_FILE} NAME)
    string(REGEX REPLACE "\\.cl$" "" KERNEL_BASE ${KERNEL_FILENAME})

    # 创建 C++ 内容
    set(KERNEL_CONTENT "static const char ${KERNEL_BASE}_string[] = {${CONTENT_HEX_FORMATTED}0x00}\;")

    # 将内容写入二进制目录中的新文件
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/kernels/${KERNEL_FILENAME}.h ${KERNEL_CONTENT})
endforeach()

### 构建编译目录结构
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/ppl.common/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/ppl.common/deps/opencl_headers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/oclbenchmark-build/)

add_subdirectory(deps/ppl.common)

add_executable(fp16_diff ${CMAKE_CURRENT_SOURCE_DIR}/tools/fp16_diff.cc)
target_link_libraries(fp16_diff pplcommon_static)

add_executable(cl_bandwidth ${CMAKE_CURRENT_SOURCE_DIR}/tools/cl_bandwidth.cc)
target_link_libraries(cl_bandwidth pplcommon_static)

add_executable(cl_peak ${CMAKE_CURRENT_SOURCE_DIR}/tools/cl_peak.cc)
target_link_libraries(cl_peak pplcommon_static)

add_executable(cl_gemm_fp16 ${CMAKE_CURRENT_SOURCE_DIR}/tools/cl_gemm_fp16.cc)
target_link_libraries(cl_gemm_fp16 pplcommon_static)
