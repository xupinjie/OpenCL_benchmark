processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`
export PPL_BUILD_THREAD_NUM=$processor_num
echo "processor_num is $processor_num"

sh ./build.sh \
-DCL_TARGET_OPENCL_VERSION=220 \
-DPPLNN_USE_OCL=ON
