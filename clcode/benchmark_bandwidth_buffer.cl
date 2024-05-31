#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void bandwidth_copy_buffer(long len, __global T* src, __global T* dst) {
    int gidx = get_global_id(0); // len
    int tidx = get_local_id(0);
    int bidx = get_group_id(0);
    int blocksize = get_local_size(0);

    int ridx = bidx * blocksize * READ_TIMES + tidx;
    int widx = bidx * blocksize * WRITE_TIMES + tidx;

    T res = (T)0;
#pragma unroll
    for (int i = 0; i < READ_TIMES; i++) {
        res += src[ridx + i * blocksize];
    }

#if WRITE_TIMES == 0
    if (((float*)&res)[0] == 3.1415926535)
#endif
        dst[widx] = res;
}

__kernel void bandwidth_copy_buffer_nblock(long len, __global T* src, __global T* dst) {
    int gidx = get_global_id(0);
    int tidx = get_local_id(0);
    int blocksize = get_local_size(0);
    int bidx = get_group_id(0);

    if (tidx > len)
        return;

    for (int lp = 0; lp < OUT_LOOPS; lp++) {
        for (int idx = tidx; idx < len; idx += blocksize) {
            T data = src[idx];
            if (((float*)&data)[0] == 3.1415926535)
                dst[gidx] = data;
        }
    }

    // int base_idx = bidx * (len / get_num_groups(0));
    // T* src_l = src + base_idx;

    // for (int lp = 0; lp < OUT_LOOPS; lp++) {
    //     for (int idx = tidx; idx < len; idx += blocksize) {
    //         T data = src_l[idx];
    //         if (((float*)&data)[0] == 3.1415926535)
    //             dst[idx] = data;
    //     }
    // }
}

// __kernel void bandwidth_reducesum_buffer(
//     long len,
//     __global TSRC* src,
//     __global float* dst)
// {
//     int gidx = get_global_id(0); //len
//     int tidx = get_local_id(0);
//     int bidx = get_group_id(0);
//     int blocksizex = get_local_size(0);

//     int idx = bidx * blocksizex * FETCH_NUM + tidx;

//     TSRC tmp = (TSRC)0;
//     #pragma unroll
//     for (int i = 0; i < FETCH_NUM; i++) {
//         tmp += src[idx+i*blocksizex];
//     }
// #if SIMD_LEN==1
//     if (tmp == 3.1415926535) dst[gidx] = tmp;
// #else
//     if (tmp.s0 == 3.1415926535) dst[gidx] = tmp.S0;
// #endif
// }
