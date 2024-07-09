#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef VENDOR_QUALCOMM
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size: enable
#endif

__kernel void compute_float_2_32(
        __read_only image2d_t A,
        __read_only image2d_t scaleA,
        __read_only image2d_t B,
        __read_only image2d_t scaleB,
        __global float* C) 
{
   
    float4 x = read_imagef(A, (int2)(0, 1));
    float4 y   = read_imagef(A, (int2)(0, 2));

    for (int i = 0; i < 20; i+=1) {

        for (int j = 0; j < 32; j+=1) 
        {
           x = mad(y, x, y);   y = mad(x, y, x);   x = mad(y, x, y);   y = mad(x, y, x);x = mad(y, x, y);
        }
    }

   vstore4(  x,0,C + 0);

}

// #ifdef VENDOR_QUALCOMM
// __attribute__((qcom_reqd_sub_group_size("full"))) __kernel
// #endif
__kernel void compute_half_2_32(
        __read_only image2d_t A,
        __read_only image2d_t scaleA,
        __read_only image2d_t B,
        __read_only image2d_t scaleB,
        __global half* C) 
{
   
    half4 x = read_imageh(A, (int2)(0, 1));
    half4 y   = read_imageh(A, (int2)(0, 2));

    for (int i = 0; i < 20; i+=1) {

        for (int j = 0; j < 32; j+=1) 
        {
            
              x = (y*x) + y;   y = (x*y) + x;      x = (y*x) + y;      y = (x*y) + x;   x = (y*x) + y; 
           //x = mad(y, x, y);   y = mad(x, y, x);   x = mad(y, x, y);   y = mad(x, y, x);x = mad(y, x, y);
        }
    }
   vstore4(  x,0,C + 0);
}


__kernel void compute_int_2_32(
        __read_only image2d_t A,
        __read_only image2d_t scaleA,
        __read_only image2d_t B,
        __read_only image2d_t scaleB,
        __global int* C) 
{
   
    int4 x = read_imagei(A, (int2)(0, 1));
    int4 y   = read_imagei(A, (int2)(0, 2));

    for (int i = 0; i < 20; i+=1) {

        for (int j = 0; j < 32; j+=1) 
        {
            
             x = (y*x) + y;   y = (x*y) + x;      x = (y*x) + y;      y = (x*y) + x;   x = (y*x) + y; 
           //x = mad(y, x, y);   y = mad(x, y, x);   x = mad(y, x, y);   y = mad(x, y, x);x = mad(y, x, y);
        }
    }

   vstore4(  x,0,C + 0);
    
}




#ifdef VENDOR_QUALCOMM
#pragma OPENCL EXTENSION cl_qcom_dot_product8 : enable
#endif

#ifdef VENDOR_ARM
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
#endif

__kernel void compute_dot_2_32(
        __read_only image2d_t A,
        __read_only image2d_t scaleA,
        __read_only image2d_t B,
        __read_only image2d_t scaleB,
        __global int* C) 
{
    int idx = get_local_id(0);
    int idx0 = idx%16;
    int idx1 = (idx/16)%16;
   
    uint4 x = read_imageui(A, (int2)(idx0, idx1));
    
    //float4 x2 = read_imagef(B, (int2)(0, 1));
    //float4 y2   = read_imagef(B, (int2)(0, 2));
    int acc = 0;
    half accf = 0;
//qcom_dot8_acc(uint p0, uint

    for (int i = 0; i < 20; i+=1) {
        uint4 y   = read_imageui(A, (int2)(idx0, i));

        for (int j = 0; j < 8; j+=1) 
        {
#ifdef VENDOR_QUALCOMM
            acc = qcom_dot8_acc(x.s0, y.s0,acc);


            //accf += convert_half(acc)*0.2f;


            acc = qcom_dot8_acc(x.s1, y.s1,acc);
            //accf += convert_half(acc)*0.2f;
            acc = qcom_dot8_acc(x.s2, y.s2,acc);
            //accf += convert_half(acc)*0.2f;
            acc = qcom_dot8_acc(x.s3, y.s3,acc);
            //accf += convert_half(acc)*0.2f;
            
              //x = (y*x) + y;   y = (x*y) + x;      x = (y*x) + y;      y = (x*y) + x;   x = (y*x) + y; 
              // x2 = (y2*x2) + y2;   y2 = (x2*y2) + x2;      x2 = (y2*x2) + y2;
                 
           //x = mad(y, x, y);   y = mad(x, y, x);   x = mad(y, x, y);   y = mad(x, y, x);x = mad(y, x, y);
#endif

#ifdef VENDOR_ARM
            acc = arm_dot_acc(x.s0, y.s0,acc);
            acc = arm_dot_acc(x.s1, y.s0,acc);
            acc = arm_dot_acc(x.s2, y.s0,acc);
            acc = arm_dot_acc(x.s3, y.s0,acc);
#endif
        }
    }

    //x += convert_half4(x2);
    C[0] = acc;
   //vstore4(  x,0,C + 0);
    

}
