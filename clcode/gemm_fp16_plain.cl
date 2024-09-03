#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void gemm_fp16_plain(
    int M,
    int N,
    int K,
    __global half *matA,
    __global half *matB,
    __global half *matC)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    half sum = 0.;
    for (int k = 0; k < K; k++) {
        sum += matA[i * K + k] * matB[k * N + j];
    }
    matC[i * N + j] = sum;
}
