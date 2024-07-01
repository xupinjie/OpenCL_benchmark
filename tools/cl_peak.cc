#include "ppl/common/log.h"
#include "ppl/common/destructor.h"
#include "ppl/common/ocl/device.h"
#include "ppl/common/ocl/framechain.h"
#include "ppl/common/ocl/runkernel.h"
#include "ppl/common/ocl/kernelpool.h"

#include "CL/opencl.h"

#include "kernels/benchmark_peak.cl.h"
#include "kernels/benchmark_peak_v2.cl.h"

#include <string>
#include <limits>

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
            ave_time_ns += time_ns / 1000;                   \
            min_time_ns = MIN(min_time_us, time_ns / 1000);  \
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

size_t get_elem_num(std::string dtype_options) {
    if (dtype_options == "float")
        return 1;
    if (dtype_options == "float2")
        return 2;
    if (dtype_options == "float4")
        return 4;
    if (dtype_options == "float8")
        return 8;
    if (dtype_options == "float16")
        return 16;

    if (dtype_options == "half")
        return 1;
    if (dtype_options == "half2")
        return 2;
    if (dtype_options == "half4")
        return 4;
    if (dtype_options == "half8")
        return 8;
    if (dtype_options == "half16")
        return 16;

    if (dtype_options == "int")
        return 1;
    if (dtype_options == "int2")
        return 2;
    if (dtype_options == "int4")
        return 4;
    if (dtype_options == "int8")
        return 8;
    if (dtype_options == "int16")
        return 16;

    return -1;
}

static void benchmark_peak(ppl::common::ocl::FrameChain* frame_chain, std::string dtype_options) {
    cl_int ret = 0;
    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();
    size_t max_mem_alloc_size = device->getMaxMemAllocSize();

    cl_mem read_buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, max_mem_alloc_size, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    cl_mem write_buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_WRITE_ONLY, max_mem_alloc_size, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    ppl::common::Destructor __guard([&read_buffer, &write_buffer]() -> void {
        if (read_buffer)
            clReleaseMemObject(read_buffer);
        if (write_buffer)
            clReleaseMemObject(write_buffer);
    });

    const size_t threads = 16384;
    const size_t LOOP_ITEMS = 128*100;
    const size_t elems = threads * LOOP_ITEMS;

    std::string options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    options += (" -DT=" + dtype_options);
    options += (" -DLOOP_ITEMS=" + std::to_string(LOOP_ITEMS));
    cl_kernel kernel;

    frame_chain->setCompileOptions(options.c_str());
    SET_PROGRAM_SOURCE(frame_chain, benchmark_peak);

    size_t gs[] = {threads, 1, 1};

    double ave_time_ns = 0;
    double min_time_ns = __DBL_MAX__;
    const int loops = 10;
    const int warms = 5;
    LOOP_KERNEL_SYN(runOclKernel(frame_chain, "benchmark_peak", 1, gs, nullptr, elems, read_buffer, write_buffer);)

    size_t ops = elems * get_elem_num(dtype_options) * 2;
    double gfops = ops / (ave_time_ns);
    printf("peak %s gfops:%f %f ns %dops\n", dtype_options.c_str(), gfops, ave_time_ns, ops);
}

static void benchmark_peak_v2(ppl::common::ocl::FrameChain* frame_chain) {
    cl_int ret = 0;
    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();
    size_t max_mem_alloc_size = device->getMaxMemAllocSize();

    cl_mem buffer = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, max_mem_alloc_size, NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    ppl::common::Destructor __guard([&buffer]() -> void {
        if (buffer)
            clReleaseMemObject(buffer);
    });

    std::string options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    frame_chain->setCompileOptions(options.c_str());
    SET_PROGRAM_SOURCE(frame_chain, benchmark_peak_v2);

    const size_t globalWIs = 16384;
    const size_t workPerWI = 4096;
    size_t gs[] = {globalWIs, 1, 1};

    double ave_time_ns = 0;
    double min_time_ns = __DBL_MAX__;
    const int loops = 10;
    const int warms = 5;
    cl_float A = 1.3f;
    LOOP_KERNEL_SYN(runOclKernel(frame_chain, "compute_hp_v1", 1, gs, nullptr, buffer, A);)

    const double gflops = ((double)(globalWIs) * (double)(workPerWI)) / ave_time_ns;

    printf("half peak gfops:%f %fns\n", gflops, ave_time_ns);
}

int main() {
    ppl::common::ocl::createSharedFrameChain(false);
    ppl::common::ocl::FrameChain* frame_chain = ppl::common::ocl::getSharedFrameChain();
    frame_chain->setTuningQueueStatus(true);
    frame_chain->setProjectName("cl_peak");

    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();

    // benchmark_peak(frame_chain, "half");
    benchmark_peak_v2(frame_chain);

    ppl::common::ocl::removeAllKernelsFromPool();

    return 0;
}
