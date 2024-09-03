#include "ppl/common/log.h"
#include "ppl/common/destructor.h"
#include "ppl/common/ocl/device.h"
#include "ppl/common/ocl/framechain.h"
#include "ppl/common/ocl/runkernel.h"
#include "ppl/common/ocl/kernelpool.h"

#include "fp16_diff.h"

#include "CL/opencl.h"

#include "kernels/gemm_fp16_plain.cl.h"
#include "kernels/gemm_fp16_v2.cl.h"

#include <string>
#include <limits>
#include <stdio.h>
#include <stdlib.h>

// #include <arm_fp16.h>

#define CHECK_ERROR(info, ret)             \
    if (ret != 0) {                        \
        LOG(ERROR) << info << ": " << ret; \
        exit(-1);                          \
    }
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define LOOP_KERNEL_SYN(cmd)                                   \
    {                                                          \
        for (int i = 0; i < warms; i++) {                      \
            cmd;                                               \
            ret = clFinish(frame_chain->getTuningQueue());     \
        }                                                      \
        for (int i = 0; i < loops; i++) {                      \
            cmd ret = clFinish(frame_chain->getTuningQueue()); \
            CHECK_ERROR("clFinish failed", ret);               \
            uint64_t time_ns = frame_chain->getKernelTime();   \
            if (time_ns == UINT64_MAX) {                       \
                LOG(ERROR) << "run kernel failed";             \
                exit(-1);                                      \
            }                                                  \
            ave_time_ns += time_ns;                            \
            min_time_ns = MIN(min_time_ns, time_ns);           \
        }                                                      \
        ave_time_ns /= loops;                                  \
    }

static void gemm_fp16_cpu(int M, int N, int K, const __fp16* matA, int lda, const __fp16* matB, int ldb, __fp16* matC,
                          int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __fp16 sum = 0.;
            for (int k = 0; k < K; k++) {
                sum += matA[i * lda + k] * matB[k * ldb + j];
            }
            matC[i * ldc + j] = sum;
        }
    }
}

static void matrix_fill_random_fp16(
    int M,
    int N,
    __fp16 *mat,
    int ld)
{
    srand(666);
    for (int i = 0 ; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __fp16 val = (__fp16)((float)rand() / RAND_MAX);
            mat[i * ld + j] = val;
        }
    }
}

static void gemm_fp16_ocl_v1(int M, int N, int K) {
    cl_int ret = 0;
    ppl::common::ocl::FrameChain* frame_chain = ppl::common::ocl::getSharedFrameChain();
    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();

    __fp16* matA_h = (__fp16*)malloc(M * K * sizeof(__fp16));
    __fp16* matB_h = (__fp16*)malloc(K * N * sizeof(__fp16));
    __fp16* matC_h = (__fp16*)malloc(M * N * sizeof(__fp16));
    __fp16* matC_ref = (__fp16*)malloc(M * N * sizeof(__fp16));

    matrix_fill_random_fp16(M, K, matA_h, K);
    matrix_fill_random_fp16(K, N, matB_h, N);

    // size_t max_mem_alloc_size = device->getMaxMemAllocSize();

    cl_mem matA_d = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, M * K * sizeof(__fp16), NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    cl_mem matB_d = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, K * N * sizeof(__fp16), NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    cl_mem matC_d = clCreateBuffer(frame_chain->getContext(), CL_MEM_WRITE_ONLY, M * N * sizeof(__fp16), NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    ppl::common::Destructor __guard([&matA_h, &matB_h, &matC_h, &matA_d, &matB_d, &matC_d]() -> void {
        if (matA_d)
            clReleaseMemObject(matA_d);
        if (matB_d)
            clReleaseMemObject(matB_d);
        if (matC_d)
            clReleaseMemObject(matC_d);
        if (matA_h)
            free(matA_h);
        if (matB_h)
            free(matB_h);
        if (matC_h)
            free(matC_h);
    });

    ret = clEnqueueWriteBuffer(frame_chain->getTuningQueue(), matA_d, CL_TRUE, 0, M * K *sizeof(__fp16), matA_h, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer failed", ret);
    ret = clEnqueueWriteBuffer(frame_chain->getTuningQueue(), matB_d, CL_TRUE, 0, K * N *sizeof(__fp16), matB_h, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer failed", ret);

    // int mItem = 4;
    // int nItem = 4;
    // int kItem = 4;

    std::string options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    // options += (" -DT=" + mItem);

    frame_chain->setCompileOptions(options.c_str());


    double ave_time_ns = 0;
    double min_time_ns = __DBL_MAX__;
    const int loops = 10;
    const int warms = 5;

    SET_PROGRAM_SOURCE(frame_chain, gemm_fp16_plain);
    
    for (size_t TPB = 64; TPB <= device->getMaxWorkItemsInGroup(); TPB*=2) {
        size_t gs[] = {(size_t)N, (size_t)M, 1};
        size_t ls[] = {TPB, 1, 1};
        LOOP_KERNEL_SYN(runOclKernel(frame_chain, "gemm_fp16_v1", 2, gs, ls, M, N, K, matA_d, matB_d, matC_d);)

        size_t ops = (size_t)M * N * K * 2;
        double gfops = ops / (ave_time_ns);
        printf("plain algo, gs[%d,%d] ls[%d,%d]\n", gs[0], gs[1], ls[0], ls[1]);
        printf("gfops:%f %fns %zuops\n", gfops, ave_time_ns, ops);
    }

    gemm_fp16_cpu(M, N, K, matA_h, K, matB_h, N, matC_ref, N);
    ret = clEnqueueReadBuffer(frame_chain->getTuningQueue(), matC_d, CL_TRUE, 0, M * N * sizeof(__fp16), matC_h, 0, NULL, NULL);

    fp_diff(matC_h, matC_ref, M * N , 1e-2f, 1e-2f, 1e-1f);

}

static void gemm_fp16_ocl_v2(int M, int N, int K) {
    cl_int ret = 0;
    ppl::common::ocl::FrameChain* frame_chain = ppl::common::ocl::getSharedFrameChain();
    ppl::common::ocl::Device* device = ppl::common::ocl::getSharedDevice();

    __fp16* matA_h = (__fp16*)malloc(M * K * sizeof(__fp16));
    __fp16* matB_h = (__fp16*)malloc(K * N * sizeof(__fp16));
    __fp16* matC_h = (__fp16*)malloc(M * N * sizeof(__fp16));
    __fp16* matC_ref = (__fp16*)malloc(M * N * sizeof(__fp16));

    matrix_fill_random_fp16(M, K, matA_h, K);
    matrix_fill_random_fp16(K, N, matB_h, N);

    // size_t max_mem_alloc_size = device->getMaxMemAllocSize();

    cl_mem matA_d = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, M * K * sizeof(__fp16), NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    cl_mem matB_d = clCreateBuffer(frame_chain->getContext(), CL_MEM_READ_ONLY, K * N * sizeof(__fp16), NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    cl_mem matC_d = clCreateBuffer(frame_chain->getContext(), CL_MEM_WRITE_ONLY, M * N * sizeof(__fp16), NULL, &ret);
    CHECK_ERROR("clCreateBuffer failed", ret);
    ppl::common::Destructor __guard([&matA_h, &matB_h, &matC_h, &matA_d, &matB_d, &matC_d]() -> void {
        if (matA_d)
            clReleaseMemObject(matA_d);
        if (matB_d)
            clReleaseMemObject(matB_d);
        if (matC_d)
            clReleaseMemObject(matC_d);
        if (matA_h)
            free(matA_h);
        if (matB_h)
            free(matB_h);
        if (matC_h)
            free(matC_h);
    });

    ret = clEnqueueWriteBuffer(frame_chain->getTuningQueue(), matA_d, CL_TRUE, 0, M * K *sizeof(__fp16), matA_h, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer failed", ret);
    ret = clEnqueueWriteBuffer(frame_chain->getTuningQueue(), matB_d, CL_TRUE, 0, K * N *sizeof(__fp16), matB_h, 0, NULL, NULL);
    CHECK_ERROR("clEnqueueWriteBuffer failed", ret);

    int mItem = 8;
    int nItem = 1;
    int kItem = 8;

    std::string options = "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math";
    options += (" -DmItem=" + std::to_string(mItem));
    options += (" -DnItem=" + std::to_string(nItem));
    options += (" -DkItem=" + std::to_string(kItem));

    frame_chain->setCompileOptions(options.c_str());


    double ave_time_ns = 0;
    double min_time_ns = __DBL_MAX__;
    const int loops = 10;
    const int warms = 5;

    SET_PROGRAM_SOURCE(frame_chain, gemm_fp16_v2);
    
    for (size_t TPB = 64; TPB <= device->getMaxWorkItemsInGroup(); TPB+=64) {
        size_t gs[] = {(size_t)N, (size_t)M / mItem, 1};
        size_t ls[] = {TPB, 1, 1};
        LOOP_KERNEL_SYN(runOclKernel(frame_chain, "gemm_fp16_v2", 2, gs, ls, M, N, K, matA_d, matB_d, matC_d);)

        size_t ops = (size_t)M * N * K * 2;
        double gfops = ops / (ave_time_ns);
        printf("v2 alg0. mItem %d, nItem %d, kItem %d, ls[%d,%d], gs[%d,%d]\n", mItem, nItem, kItem, gs[0], gs[1], ls[0], ls[2]);
        printf("gfops:%f %fns %zuops\n", gfops, ave_time_ns, ops);
    }

    gemm_fp16_cpu(M, N, K, matA_h, K, matB_h, N, matC_ref, N);
    ret = clEnqueueReadBuffer(frame_chain->getTuningQueue(), matC_d, CL_TRUE, 0, M * N * sizeof(__fp16), matC_h, 0, NULL, NULL);

    fp_diff(matC_h, matC_ref, M * N , 1e-2f, 1e-2f, 1e-1f);

}

int main() {
    ppl::common::ocl::createSharedFrameChain(false);
    ppl::common::ocl::FrameChain* frame_chain = ppl::common::ocl::getSharedFrameChain();
    frame_chain->setTuningQueueStatus(true);
    frame_chain->setProjectName("cl_gemm");

    gemm_fp16_ocl_v1(256, 1024, 1024);

    gemm_fp16_ocl_v2(256, 1024, 1024);

    ppl::common::ocl::removeAllKernelsFromPool();

    return 0;
}
