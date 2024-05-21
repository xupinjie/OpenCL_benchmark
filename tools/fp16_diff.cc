#include <cmath>
#include <cfloat>
#include <cstring>
#include <climits>
#include <fstream>
#include <iostream>

// BECAREFUL! do not use -ffast-math on g++
static inline void kahan_step(const double& in, double *sum, double *c) {
    double y = in - c[0];
    double t = sum[0] + y;
    c[0] = (t - sum[0]) - y;
    sum[0] = t;
    return;
}

static size_t fp_diff(const float *in0, const float *in1, const size_t &length, const float& eps_abs, const float& eps_rel, const float& eps_cos) {
    size_t min_abs_idx = (size_t)UINT64_MAX;
    size_t max_abs_idx = (size_t)UINT64_MAX;
    float min_abs_err = FLT_MAX;
    float max_abs_err = -FLT_MAX;
    float min_abs_val0;
    float max_abs_val0;
    float min_abs_val1;
    float max_abs_val1;

    size_t min_rel_idx = (size_t)UINT64_MAX;
    size_t max_rel_idx = (size_t)UINT64_MAX;
    float min_rel_err = FLT_MAX;
    float max_rel_err = -FLT_MAX;
    float min_rel_val0;
    float max_rel_val0;
    float min_rel_val1;
    float max_rel_val1;

    size_t abs_en1_cnt = 0;
    size_t abs_en2_cnt = 0;
    size_t abs_en3_cnt = 0;
    size_t abs_en4_cnt = 0;
    size_t abs_en5_cnt = 0;
    size_t abs_en6_cnt = 0;

    size_t rel_en1_cnt = 0;
    size_t rel_en2_cnt = 0;
    size_t rel_en3_cnt = 0;
    size_t rel_en4_cnt = 0;
    size_t rel_en5_cnt = 0;
    size_t rel_en6_cnt = 0;

    size_t err_val_cnt = 0;

    double sum_ab = 0.;
    double sum_a2 = 0.;
    double sum_b2 = 0.;
    double N = length;
    double rN = 1.0 / N;

    double sum_ab_c = 0.;
    double sum_a2_c = 0.;
    double sum_b2_c = 0.;

    for (size_t i = 0; i < length; ++i) {
        float val0 = in0[i];
        float val1 = in1[i];

        int nan0 = std::isnan(val0);
        int nan1 = std::isnan(val1);
        int inf0 = std::isinf(val0);
        int inf1 = std::isinf(val1);

        if (nan0 && nan1) {
            continue;
        }

        if (inf0 && inf1) {
            continue;
        }

        kahan_step((double)val0 * val1 * rN, &sum_ab, &sum_ab_c);
        kahan_step((double)val0 * val0 * rN, &sum_a2, &sum_a2_c);
        kahan_step((double)val1 * val1 * rN, &sum_b2, &sum_b2_c);
        // sum_ab += val0 * val1 * rN;
        // sum_a2 += val0 * val0 * rN;
        // sum_b2 += val1 * val1 * rN;

        float err = val0 - val1;
        float abs_err = fabs(err);
        float rel_err = fabs(val0) > fabs(val1) ? fabs(err / val0) : fabs(err / val1);
        if (abs_err > 1e-1f) {
            ++abs_en1_cnt;
        } else if (abs_err > 1e-2f) {
            ++abs_en2_cnt;
        } else if (abs_err > 1e-3f) {
            ++abs_en3_cnt;
        } else if (abs_err > 1e-4f) {
            ++abs_en4_cnt;
        } else if (abs_err > 1e-5f) {
            ++abs_en5_cnt;
        } else if (abs_err > 1e-6f) {
            ++abs_en6_cnt;
        }
        if (rel_err > 1e-1f) {
            ++rel_en1_cnt;
        } else if (rel_err > 1e-2f) {
            ++rel_en2_cnt;
        } else if (rel_err > 1e-3f) {
            ++rel_en3_cnt;
        } else if (rel_err > 1e-4f) {
            ++rel_en4_cnt;
        } else if (rel_err > 1e-5f) {
            ++rel_en5_cnt;
        } else if (rel_err > 1e-6f) {
            ++rel_en6_cnt;
        }
        if ((abs_err >= eps_abs && rel_err >= eps_rel) || nan0 || nan1 || inf0 || inf1) {
            ++err_val_cnt;
        }
        if (abs_err > max_abs_err) {
            max_abs_idx = i;
            max_abs_err = abs_err;
            max_abs_val0 = val0;
            max_abs_val1 = val1;
        }
        if (rel_err > max_rel_err) {
            max_rel_idx = i;
            max_rel_err = rel_err;
            max_rel_val0 = val0;
            max_rel_val1 = val1;
        }
        if (abs_err < min_abs_err) {
            min_abs_idx = i;
            min_abs_err = abs_err;
            min_abs_val0 = val0;
            min_abs_val1 = val1;
        }
        if (rel_err < min_rel_err) {
            min_rel_idx = i;
            min_rel_err = rel_err;
            min_rel_val0 = val0;
            min_rel_val1 = val1;
        }
    }

    double cos_distance = 1.0 - (sum_ab / (sqrt(sum_a2) * sqrt(sum_b2)));

    std::cerr << "Number of values: " << length << std::endl;
    std::cerr << "Number of equal values: " << length - err_val_cnt << std::endl;
    std::cerr << "Number of not equal values: " << err_val_cnt << std::endl;
    std::cerr << "COS simulate: " << 1-cos_distance << ", with SUM(ab)/N: " << sum_ab << ", SUM(a^2)/N:" << sum_a2 << ", SUM(b^2)/N: " << sum_b2 << std::endl;

    if (min_abs_idx != (size_t)UINT64_MAX) std::cerr << "MIN abs error: " << min_abs_err << " --> " << min_abs_val0 << " vs. " << min_abs_val1 << ", at " << min_abs_idx << std::endl;
    if (max_abs_idx != (size_t)UINT64_MAX) std::cerr << "MAX abs error: " << max_abs_err << " --> " << max_abs_val0 << " vs. " << max_abs_val1 << ", at " << max_abs_idx << std::endl;
    if (min_rel_idx != (size_t)UINT64_MAX) std::cerr << "MIN rel error: " << min_rel_err << " --> " << min_rel_val0 << " vs. " << min_rel_val1 << ", at " << min_rel_idx << std::endl;
    if (max_rel_idx != (size_t)UINT64_MAX) std::cerr << "MAX rel error: " << max_rel_err << " --> " << max_rel_val0 << " vs. " << max_rel_val1 << ", at " << max_rel_idx << std::endl;

    std::cerr << "Statistics of abs error" << std::endl;
    std::cerr << "(..., 1e-1f): " << abs_en1_cnt << std::endl;
    std::cerr << "(..., 1e-2f): " << abs_en2_cnt << std::endl;
    std::cerr << "(..., 1e-3f): " << abs_en3_cnt << std::endl;
    std::cerr << "(..., 1e-4f): " << abs_en4_cnt << std::endl;
    std::cerr << "(..., 1e-5f): " << abs_en5_cnt << std::endl;
    std::cerr << "(..., 1e-6f): " << abs_en6_cnt << std::endl;

    std::cerr << "Statistics of rel error" << std::endl;
    std::cerr << "(..., 1e-1f): " << rel_en1_cnt << std::endl;
    std::cerr << "(..., 1e-2f): " << rel_en2_cnt << std::endl;
    std::cerr << "(..., 1e-3f): " << rel_en3_cnt << std::endl;
    std::cerr << "(..., 1e-4f): " << rel_en4_cnt << std::endl;
    std::cerr << "(..., 1e-5f): " << rel_en5_cnt << std::endl;
    std::cerr << "(..., 1e-6f): " << rel_en6_cnt << std::endl;

    if (std::isnan(cos_distance) || std::isinf(cos_distance)) {
        return err_val_cnt;
    }

    if (cos_distance < eps_cos) {
        return 0;
    } else {
        return err_val_cnt;
    }
}

#include <memory>
#include <algorithm>

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
