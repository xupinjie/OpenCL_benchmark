#include "ppl/common/log.h"
#include "ppl/common/destructor.h"
#include "ppl/common/ocl/device.h"
#include "ppl/common/ocl/framechain.h"
#include "ppl/common/ocl/runkernel.h"
#include "ppl/common/ocl/kernelpool.h"

#include "CL/opencl.h"

#include "kernels/demo.cl.h"
#include "kernels/benchmark_bandwidth_buffer.cl.h"
#include "kernels/benchmark_bandwidth_sharememory.cl.h"

#include <string>

//不是所有厂商都会返回正确的最大malloc size，比如mali
#define MAX_BUFFER_FIX 1 * 1024 * 1024 * 1024
#define MIN(a, b) (a) < (b) ? (a) : (b);

#define CHECK_ERROR(info, ret)             \
    if (ret != 0) {                        \
        LOG(ERROR) << info << ": " << ret; \
        exit(-1);                          \
    }
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define LOOP_KERNEL_SYN(cmd)                                 \
    {                                                        \
        for (int i = 0; i < warms; i++) {                    \
            cmd;                                             \
            ret = clFinish(frame_chain->getQueue());         \
        }                                                    \
        for (int i = 0; i < loops; i++) {                    \
            cmd ret = clFinish(frame_chain->getQueue());     \
            CHECK_ERROR("clFinish failed", ret);             \
            uint64_t time_ns = frame_chain->getKernelTime(); \
            if (time_ns == UINT64_MAX) {                     \
                LOG(ERROR) << "run kernel failed";           \
                exit(-1);                                    \
            }                                                \
            ave_time_us += time_ns / 1000;                   \
            min_time_us = MIN(min_time_us, time_ns / 1000);  \
        }                                                    \
        ave_time_us /= loops;                                \
    }

#define LOOP_KERNEL_SYN_NS(cmd)                              \
    {                                                        \
        for (int i = 0; i < warms; i++) {                    \
            cmd;                                             \
            ret = clFinish(frame_chain->getQueue());         \
        }                                                    \
        for (int i = 0; i < loops; i++) {                    \
            cmd ret = clFinish(frame_chain->getQueue());     \
            CHECK_ERROR("clFinish failed", ret);             \
            uint64_t time_ns = frame_chain->getKernelTime(); \
            if (time_ns == UINT64_MAX) {                     \
                LOG(ERROR) << "run kernel failed";           \
                exit(-1);                                    \
            }                                                \
            ave_time_ns += time_ns;                          \
            min_time_ns = MIN(min_time_ns, time_ns);         \
        }                                                    \
        ave_time_ns /= loops;                                \
    }

#define LOOP_KERNEL_NON_SYN(cmd)                             \
    {                                                        \
        for (int i = 0; i < warms; i++) {                    \
            cmd;                                             \
        }                                                    \
        for (int i = 0; i < loops; i++) {                    \
            cmd;                                             \
            uint64_t time_ns = frame_chain->getKernelTime(); \
            if (time_ns == UINT64_MAX) {                     \
                LOG(ERROR) << "run kernel failed";           \
                exit(-1);                                    \
            }                                                \
            ave_time_us += time_ns / 1000;                   \
            min_time_us = MIN(min_time_us, time_ns / 1000);  \
        }                                                    \
        ave_time_us /= loops;                                \
    }

size_t get_elem_size(std::string dtype_options) {
    if (dtype_options == "float")
        return sizeof(float);
    if (dtype_options == "float2")
        return sizeof(float) * 2;
    if (dtype_options == "float4")
        return sizeof(float) * 4;
    if (dtype_options == "float8")
        return sizeof(float) * 8;
    if (dtype_options == "float16")
        return sizeof(float) * 16;

    if (dtype_options == "half")
        return sizeof(cl_half);
    if (dtype_options == "half2")
        return sizeof(cl_half) * 2;
    if (dtype_options == "half4")
        return sizeof(cl_half) * 4;
    if (dtype_options == "half8")
        return sizeof(cl_half) * 8;
    if (dtype_options == "half16")
        return sizeof(cl_half) * 16;

    if (dtype_options == "int")
        return sizeof(int);
    if (dtype_options == "int2")
        return sizeof(int) * 2;
    if (dtype_options == "int4")
        return sizeof(int) * 4;
    if (dtype_options == "int8")
        return sizeof(int) * 8;
    if (dtype_options == "int16")
        return sizeof(int) * 16;

    return -1;
}

static void BandtwidhTest_demo(ppl::common::ocl::FrameChain* frame_chain) {
    cl_int ret = 0;
    std::string options;
    frame_chain->setCompileOptions(options.c_str());
    SET_PROGRAM_SOURCE(frame_chain, demo);
    size_t gs[] = {100};
    int a = 1;
    runOclKernel(frame_chain, "demo", 1, gs, nullptr, a);
    ret = clFinish(frame_chain->getQueue());
    if (ret) {
        LOG(ERROR) << "clFinsh failed.";
    }

    uint64_t time_ns = frame_chain->getKernelTime();
    LOG(ERROR) << time_ns;
}

static void benchmark_stream_copy_buffer(ppl::common::ocl::FrameChain* frame_chain, std::string dtype_options,
                                         int READ_TIMES = 1, int WRITE_TIMES = 1, size_t buffer_bytes = 0) {
    if (WRITE_TIMES != 0 && WRITE_TIMES != 1) {
        LOG(ERROR) << "WRITE_TIMES must be 0 or 1.";
        return;
    }

    cl_int ret = 0;
    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();
    size_t max_mem_alloc_size = MIN(device->getMaxMemAllocSize(), MAX_BUFFER_FIX);

    if (buffer_bytes == 0)
        buffer_bytes = max_mem_alloc_size;
    size_t elems = buffer_bytes / get_elem_size(dtype_options);

    size_t read_bytes, write_bytes;
    if (READ_TIMES == 0) {
        read_bytes = 0;
    } else {
        read_bytes = buffer_bytes;
    }
    if (WRITE_TIMES == 0) {
        write_bytes = 0;
    } else {
        write_bytes = buffer_bytes / READ_TIMES;
    }

    cl_mem read_buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, buffer_bytes, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    cl_mem write_buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_WRITE_ONLY, buffer_bytes, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    ppl::common::Destructor __guard([&read_buffer, &write_buffer]() -> void {
        if (read_buffer)
            clReleaseMemObject(read_buffer);
        if (write_buffer)
            clReleaseMemObject(write_buffer);
    });

    std::string options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    options += (" -DT=" + dtype_options);
    options += (" -DREAD_TIMES=" + std::to_string(READ_TIMES));
    options += (" -DWRITE_TIMES=" + std::to_string(WRITE_TIMES));
    options += (" -DOUT_LOOPS=" + std::to_string(1));
    cl_kernel kernel;

    frame_chain->setCompileOptions(options.c_str());
    SET_PROGRAM_SOURCE(frame_chain, benchmark_bandwidth_buffer);

    size_t gs[] = {elems / READ_TIMES, 1, 1};

    uint64_t ave_time_us = 0;
    uint64_t min_time_us = UINT64_MAX;
    const int loops = 10;
    const int warms = 5;
    LOOP_KERNEL_SYN(
        runOclKernel(frame_chain, "bandwidth_copy_buffer", 1, gs, nullptr, elems, read_buffer, write_buffer);)

    double ave_bandwidth = (read_bytes + write_bytes) * 1.0 / (ave_time_us * 1e-06) / 1024 / 1024 / 1024;
    double max_bandwidth = (read_bytes + write_bytes) * 1.0 / (min_time_us * 1e-06) / 1024 / 1024 / 1024;
    size_t bytesMB = (read_bytes + write_bytes) / 1024 / 1024;
    size_t bytesKB = (read_bytes + write_bytes) / 1024;
    if (bytesMB > 0)
        printf("stream copy[r%d,w%d] buffer bandwidth: %zuMB %s. max:%f GB/s, ave: %f\n", READ_TIMES, WRITE_TIMES,
               bytesMB, dtype_options.c_str(), max_bandwidth, ave_bandwidth);
    else
        printf("stream copy[r%d,w%d] buffer bandwidth: %zuKB %s. max:%f GB/s, ave: %f\n", READ_TIMES, WRITE_TIMES,
               bytesKB, dtype_options.c_str(), max_bandwidth, ave_bandwidth);
}

static void benchmark_stream_copy_buffer_nblock(ppl::common::ocl::FrameChain* frame_chain, std::string dtype_options,
                                                size_t buffer_bytes = 0, size_t blocknum = 1) {
    cl_int ret = 0;
    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();
    size_t max_mem_alloc_size = MIN(device->getMaxMemAllocSize(), MAX_BUFFER_FIX);

    int outloops = 64;

    if (buffer_bytes == 0)
        buffer_bytes = max_mem_alloc_size;
    size_t elems = buffer_bytes / get_elem_size(dtype_options);

    size_t read_bytes, write_bytes;
    read_bytes = buffer_bytes;
    write_bytes = 0;

    cl_mem read_buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, buffer_bytes, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    cl_mem write_buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_WRITE_ONLY, buffer_bytes, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    ppl::common::Destructor __guard([&read_buffer, &write_buffer]() -> void {
        if (read_buffer)
            clReleaseMemObject(read_buffer);
        if (write_buffer)
            clReleaseMemObject(write_buffer);
    });

    std::string options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    options += (" -DT=" + dtype_options);
    options += (" -DREAD_TIMES=" + std::to_string(1));
    options += (" -DWRITE_TIMES=" + std::to_string(0));
    options += (" -DOUT_LOOPS=" + std::to_string(outloops));
    cl_kernel kernel;

    frame_chain->setCompileOptions(options.c_str());
    SET_PROGRAM_SOURCE(frame_chain, benchmark_bandwidth_buffer);

    size_t blocksize = 512;
    size_t ls[] = {blocksize, 1, 1};
    size_t gs[] = {blocksize * blocknum, 1, 1};

    uint64_t ave_time_us = 0;
    uint64_t min_time_us = UINT64_MAX;
    const int loops = 10;
    const int warms = 5;

    LOOP_KERNEL_NON_SYN(
        runOclKernel(frame_chain, "bandwidth_copy_buffer_nblock", 1, gs, ls, elems, read_buffer, write_buffer););

    double ave_bandwidth =
        blocknum * outloops * (read_bytes + write_bytes) * 1.0 / (ave_time_us * 1e-06) / 1024 / 1024 / 1024;
    double max_bandwidth =
        blocknum * outloops * (read_bytes + write_bytes) * 1.0 / (min_time_us * 1e-06) / 1024 / 1024 / 1024;
    size_t bytesMB = (read_bytes + write_bytes) / 1024 / 1024;
    size_t bytesKB = (read_bytes + write_bytes) / 1024;
    printf("stream copy buffer bandwidth: %zuKB loops:%d blocknum:%d %s. max:%f GB/s, ave: %f\n", bytesKB, outloops,
           blocknum, dtype_options.c_str(), max_bandwidth, ave_bandwidth);
}

static void benchmark_stream_sharedmemory_nblock(ppl::common::ocl::FrameChain* frame_chain, std::string dtype_options,
                                                 size_t lds_bytes_per_block, size_t blocknum) {
    cl_int ret = 0;
    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();
    size_t max_mem_alloc_size = device->getMaxMemAllocSize();
    size_t elems_per_block = lds_bytes_per_block / get_elem_size(dtype_options);
    size_t elems = elems_per_block * blocknum;

    size_t OUT_LOOPS = 1;
    size_t read_bytes, write_bytes;
    read_bytes = lds_bytes_per_block * blocknum;
    write_bytes = 0;

    cl_mem write_buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_WRITE_ONLY, max_mem_alloc_size, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    ppl::common::Destructor __guard([&write_buffer]() -> void {
        if (write_buffer)
            clReleaseMemObject(write_buffer);
    });

    std::string options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    options += (" -DT=" + dtype_options);
    options += (" -DBLOCKSIZE=" + std::to_string(elems_per_block));
    options += (" -DOUT_LOOPS=" + std::to_string(OUT_LOOPS));
    cl_kernel kernel;

    frame_chain->setCompileOptions(options.c_str());
    SET_PROGRAM_SOURCE(frame_chain, benchmark_bandwidth_sharememory);

    size_t ls[] = {elems_per_block, 1, 1};
    size_t gs[] = {elems_per_block * blocknum, 1, 1};

    uint64_t ave_time_ns = 0;
    uint64_t min_time_ns = UINT64_MAX;
    const int loops = 10;
    const int warms = 5;

    LOOP_KERNEL_SYN_NS(
        runOclKernel(frame_chain, "bandwidth_read_sharememory_nblock", 1, gs, ls, elems_per_block, write_buffer););

    double ave_bandwidth = OUT_LOOPS * (read_bytes + write_bytes) * 1.0 / (ave_time_ns * 1e-09) / 1024 / 1024 / 1024;
    double max_bandwidth = OUT_LOOPS * (read_bytes + write_bytes) * 1.0 / (min_time_ns * 1e-09) / 1024 / 1024 / 1024;
    size_t bytesKB = (read_bytes + write_bytes) / 1024;
    printf("shared memory bandwidth: %zuKB lds_bytes_per_block:%d blocknum:%d %s. max:%f GB/s, ave: %f times:%d\n",
           bytesKB, lds_bytes_per_block, blocknum, dtype_options.c_str(), max_bandwidth, ave_bandwidth, ave_time_ns);
}

/**
 * @brief 测试读和写的次数，对带宽的影响
 */
static void BandtwidhTest_writem_readn(ppl::common::ocl::FrameChain* frame_chain) {
    LOG(INFO) << "测试读和写的次数，对带宽的影响";
    benchmark_stream_copy_buffer(frame_chain, "float8", 1, 1, 0);
    benchmark_stream_copy_buffer(frame_chain, "float8", 2, 1, 0);
    benchmark_stream_copy_buffer(frame_chain, "float8", 4, 1, 0);
    benchmark_stream_copy_buffer(frame_chain, "float8", 8, 1, 0);

    benchmark_stream_copy_buffer(frame_chain, "float8", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "float8", 2, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "float8", 4, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "float8", 8, 0, 0);
}

/**
 * @brief 数据类型对带宽的影响
 */
static void BandtwidhTest_datatype(ppl::common::ocl::FrameChain* frame_chain) {
    LOG(INFO) << "数据类型对带宽的影响";
    benchmark_stream_copy_buffer(frame_chain, "float", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "float2", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "float4", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "float8", 1, 0, 0);

    benchmark_stream_copy_buffer(frame_chain, "half", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "half2", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "half4", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "half8", 1, 0, 0);

    benchmark_stream_copy_buffer(frame_chain, "int", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "int2", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "int4", 1, 0, 0);
    benchmark_stream_copy_buffer(frame_chain, "int8", 1, 0, 0);
}

/**
 * @brief 访存量对带宽的影响
 */
static void BandtwidhTest_datasize(ppl::common::ocl::FrameChain* frame_chain) {
    LOG(INFO) << "访存量对带宽的影响";
    size_t max_size = ppl::common::ocl::getSharedDevice()->getMaxMemAllocSize();
    max_size = MIN(max_size, MAX_BUFFER_FIX);
    // for (size_t buffer_size = 1024; buffer_size <= max_size; buffer_size*=2) {
    for (size_t buffer_size = max_size; buffer_size >= 1024; buffer_size /= 2) {
        benchmark_stream_copy_buffer(frame_chain, "half8", 1, 0, buffer_size);
    }
}

/**
 * @brief 测试多层存储的大小和带宽
 */
static void BandtwidhTest_cachesize(ppl::common::ocl::FrameChain* frame_chain) {
    LOG(INFO) << "测试多层存储的大小和带宽";
    size_t stride = 4096;
    // size_t max_size = ppl::common::ocl::getSharedDevice()->getMaxMemAllocSize();
    size_t max_size = 32 * 1024 * 1024;
    // for (size_t buffer_size = 4096; buffer_size < max_size; buffer_size += stride) {
    for (size_t buffer_size = 4096; buffer_size < max_size; buffer_size *= 2) {
        // for (size_t buffer_size = max_size; buffer_size <= max_size; buffer_size *= 2) {
        benchmark_stream_copy_buffer_nblock(frame_chain, "float4", buffer_size, 256);
    }
}

/**
 * @brief 测试shared memory带宽/延迟
 */
static void BandtwidhTest_sharedmemory(ppl::common::ocl::FrameChain* frame_chain) {
    LOG(INFO) << "测试shared memory带宽/延迟";
    size_t stride = 4096;
    // size_t max_size = ppl::common::ocl::getSharedDevice()->getMaxMemAllocSize();
    size_t max_size = 128 * 1024;
    size_t lds_bytes_per_block = 4 * 1024;
    size_t block_num = 1;
    // for (; lds_bytes_per_block < max_size; lds_bytes_per_block *= 2) {
    //     benchmark_stream_sharedmemory_nblock(frame_chain, "float4", lds_bytes_per_block, block_num);
    // }
    for (; block_num < 1024; block_num++) {
        benchmark_stream_sharedmemory_nblock(frame_chain, "float8", lds_bytes_per_block, block_num);
    }
}

int main() {
    ppl::common::ocl::createSharedFrameChain(false);
    ppl::common::ocl::FrameChain* frame_chain = ppl::common::ocl::getSharedFrameChain();
    frame_chain->setTuningQueueStatus(true);
    frame_chain->setProjectName("cl_bandwidth");

    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();

    // BandtwidhTest_demo(frame_chain);
    BandtwidhTest_writem_readn(frame_chain);
    BandtwidhTest_datatype(frame_chain);
    BandtwidhTest_datasize(frame_chain);
    BandtwidhTest_cachesize(frame_chain);
    // benchmark_stream_copy_buffer_nblock(frame_chain, "float4", 4 * 1024 * 1024, 256);
    // BandtwidhTest_sharedmemory(frame_chain);

    ppl::common::ocl::removeAllKernelsFromPool();

    return 0;
}
