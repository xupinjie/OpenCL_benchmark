#include <cstring>
#include <fstream>
#include <memory>
#include <algorithm>

#include "fp16_diff.h"

using std::unique_ptr;

static void CvtFp16ToFp32(int64_t counts, void const* src, void* dst) {
    auto src_ptr = (__fp16*)src;
    auto dst_ptr = (float*)dst;
    for (int64_t i = 0; i < counts; i += 1) {
        dst_ptr[i] = src_ptr[i];
    }
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: " << argv[0] << " file1 file2 [fp32]" << std::endl;
        return -1;
    }

    std::ifstream ifs0(argv[1], std::ios::binary | std::ios::in);
    std::ifstream ifs1(argv[2], std::ios::binary | std::ios::in);
    if (!ifs0.is_open()) {
        std::cerr << "Fail to open " << argv[1] << std::endl;
        return 1;
    }
    if (!ifs1.is_open()) {
        std::cerr << "Fail to open " << argv[2] << std::endl;
        return 1;
    }
    ifs0.seekg(0, std::ios::end);
    ifs1.seekg(0, std::ios::end);
    intptr_t len0 = ifs0.tellg();
    intptr_t len1 = ifs1.tellg();

    if (len0 != len1) {
        std::cerr << "==Warning== file length not equal, " << len0 << " vs " << len1 << std::endl;
    }

    ifs0.seekg(0, std::ios::beg);
    ifs1.seekg(0, std::ios::beg);

    if (argc==4 && strcmp(argv[3], "fp32")==0) {
        unique_ptr<float[]> buf0(new float[len0 / sizeof(float)]);
        unique_ptr<float[]> buf1(new float[len1 / sizeof(float)]);

        ifs0.read((char*)buf0.get(), len0);
        ifs1.read((char*)buf1.get(), len1);

        len0 = ifs0.gcount();
        len1 = ifs1.gcount();
        if (len0 != len1) {
            std::cerr << "==Warning== file read size not equal, " << len0 << " vs " << len1 << std::endl;
            return -1;
        }

        size_t ret = fp_diff(buf0.get(), buf1.get(), std::min(len0, len1) / sizeof(float), 1e-6f, 1e-6f, 1e-3f);
        if (ret == 0) {
            std::cerr << "PASS" << std::endl;
        } else {
            std::cerr << "FAILED" << std::endl;
        }
    } else {
        unique_ptr<uint16_t[]> buf0(new uint16_t[len0 / sizeof(uint16_t)]);
        unique_ptr<uint16_t[]> buf1(new uint16_t[len1 / sizeof(uint16_t)]);

        ifs0.read((char*)buf0.get(), len0);
        ifs1.read((char*)buf1.get(), len1);

        len0 = ifs0.gcount();
        len1 = ifs1.gcount();
        if (len0 != len1) {
            std::cerr << "==Warning== file read size not equal, " << len0 << " vs " << len1 << std::endl;
            return -1;
        }

        float *src0 = (float *)malloc(len0 / sizeof(uint16_t) * sizeof(float));
        float *src1 = (float *)malloc(len1 / sizeof(uint16_t) * sizeof(float));
        
        CvtFp16ToFp32(len0 / sizeof(uint16_t), buf0.get(), src0);
        CvtFp16ToFp32(len1 / sizeof(uint16_t), buf1.get(), src1);

        size_t ret = fp_diff(src0, src1, std::min(len0, len1) / sizeof(uint16_t), 1e-2f, 1e-2f, 1e-1f);
        if (ret == 0) {
            std::cerr << "PASS" << std::endl;
        } else {
            std::cerr << "FAILED" << std::endl;
        }
                
        free(src0);
        free(src1);
    }

    return 0;
}
