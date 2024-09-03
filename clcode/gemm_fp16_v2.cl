#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef USE_FLOAT_ACC
#define accT1  float
#define accT2  float2
#define accT4  float4
#define accT8  float8
#define accT16 float16
#else
#define accT1 half
#define accT2 half2
#define accT4 half4
#define accT8 half8
#endif

#if (kItem == 8)
#define TKVec half8
#define vloadTKVec vload8
#define convert_TKVec convert_T8
#elif (kItem == 4)
#define TKVec half4
#define vloadTKVec vload4
#define convert_TKVec convert_T4
#elif (kItem == 2)
#define TKVec half2
#define vloadTKVec vload2
#define convert_TKVec convert_T2
#endif

#if (nItem == 8)
#define TNVec T8
#define accTNVec accT8
#define vloadTNVec vload8
#define vstoreTNVec vstore8
#define convert_TNVec convert_T8
#define convert_accTNVec convert_accT8
#elif (nItem == 4)
#define TNVec T4
#define accTNVec accT4
#define vloadTNVec vload4
#define vstoreTNVec vstore4
#define convert_TNVec convert_T4
#define convert_accTNVec convert_accT4
#elif (nItem == 2)
#define TNVec T2
#define accTNVec accT2
#define vloadTNVec vload2
#define vstoreTNVec vstore2
#define convert_TNVec convert_T2
#define convert_accTNVec convert_accT2
#endif


__kernel void gemm_fp16_v2(
    int M,
    int N,
    int K,
    __global half *matA,
    __global half *matB,
    __global half *matC)
{
    int j = get_global_id(0);
    int i = get_global_id(1) * mItem;

    half tmpC[mItem];
    for (int m = 0; m < mItem; m++) {
        tmpC[m] = 0.;
    }

    for (int k = 0; k < K; k += kItem) {
        TKVec tmpA[mItem];
        for (int m = 0; m < mItem; m++) {
            tmpA[m] = matA[(i + m) * K + k];
        }
        for (int m = 0; m < mItem; m++) {
            tmpC[m] += tmpA[m] * matB[k * N + j];
        }
    }

    for (int m = 0; m < mItem; m++) {
        matC[(i+m) * N + j] = tmpC[m];
    }
    
    // half sum = 0.;
    // for (int k = 0; k < K; k++) {
    //     sum += matA[i * K + k] * matB[k * N + j];
    // }
    // matC[i * N + j] = sum;
}
