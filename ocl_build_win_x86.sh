mkdir oclbenchmark-build
cd ./oclbenchmark-build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DCL_TARGET_OPENCL_VERSION=220 -DPPLNN_USE_OCL=ON .. && cmake --build . -j 12 --config Release && cmake --build . --target install -j 12 --config Release
