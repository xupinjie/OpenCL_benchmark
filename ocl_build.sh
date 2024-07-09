export ANDROID_NDK=/your/path/android-ndk-rxx

echo "aarch64"
aar="arm64-v8a"

processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`
export PPL_BUILD_THREAD_NUM=$processor_num
echo "processor_num is $processor_num"

sh ./build.sh -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=$aar \
-DANDROID_NDK=$ANDROID_NDK \
-DCMAKE_ANDROID_NDK=$ANDROID_NDK \
-DANDROID_PLATFORM=android-24 \
-DCL_TARGET_OPENCL_VERSION=220 \
-DPPLNN_USE_OCL=ON \
-DPPLNN_USE_AARCH64=ON \
