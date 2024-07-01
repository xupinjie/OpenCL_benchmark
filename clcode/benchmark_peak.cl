#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void benchmark_peak(long len, __global T* src, __global T* dst) {
    int gidx = get_global_id(0); // len

    T vc = (T)0;
    T va = (T)1 * 0.005;
    T vb = (T)1 * 0.005;

#pragma unroll
    for (int i = 0; i < LOOP_ITEMS; i++) {
        vc += va * vb;
    }

    // if (((float*)&vc)[0] == 3.1415926535)
        dst[gidx] = vc;
        // printf("%f\n", vc);
}
