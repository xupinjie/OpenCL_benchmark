#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void bandwidth_read_sharememory_nblock(long len, __global T* dst) {
    int gidx = get_global_id(0);
    int tidx = get_local_id(0);
    // int blocksize = get_local_size(0);
    int bidx = get_group_id(0);

    if (tidx > len)
        return;

    __local T lds[BLOCKSIZE];
    lds[tidx] = (T)(3+gidx);

    for (int lp = 0; lp < OUT_LOOPS; lp++) {
        for (int idx = tidx; idx < len; idx += BLOCKSIZE) {
            T data = lds[idx];
            if (((float*)&data)[0] == 3.1415926535)
                dst[gidx] = data;
        }
    }
}
