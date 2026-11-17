// Host test of the hc_mma FP16-range admission (include/strix/cuda/qwen_hc_fp16_range.hpp): admitted inside the range,
// refused with a reason at the activation bound, on an inject weight that rounds to FP16 zero, and on non-finite weights.
#include "strix/cuda/qwen_hc_fp16_range.hpp"

#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace hr = strix::cuda::qwen_hc::fp16_range;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

int main() {
    std::vector<float> norm(10240, 1.5F), inject(40960, 0.01F);
    inject[7] = 0.0F;  // exact zeros are not an underflow
    {
        hr::Scan s;
        hr::scan_norm(s, norm.data(), norm.size());
        hr::scan_inject(s, inject.data(), inject.size());
        check(hr::refusal(s).empty(), "typical weights admitted: " + hr::refusal(s));
        check(s.norm_max_abs == 1.5F && s.inject_min_nonzero_abs == 0.01F, "scan extremes");
    }
    {  // sqrt(2560) * 161 = 8146 < 8188 admitted; 162 -> 8197 refused
        hr::Scan a, b;
        norm[3] = -161.0F;
        hr::scan_norm(a, norm.data(), norm.size());
        check(hr::refusal(a).empty(), "161 admitted: " + hr::refusal(a));
        norm[3] = -162.0F;
        hr::scan_norm(b, norm.data(), norm.size());
        check(hr::refusal(b).find("FP16 activation range") != std::string::npos, "162 refused: " + hr::refusal(b));
        norm[3] = 1.5F;
    }
    {  // 2^-24 is the smallest FP16 subnormal (kept); 2^-25 rounds to zero (refused)
        hr::Scan a, b;
        inject[11] = -0x1p-24F;
        hr::scan_inject(a, inject.data(), inject.size());
        check(hr::refusal(a).empty(), "2^-24 admitted: " + hr::refusal(a));
        inject[11] = 0x1p-25F;
        hr::scan_inject(b, inject.data(), inject.size());
        check(b.inject_underflow == 1 && hr::refusal(b).find("round to FP16 zero") != std::string::npos,
              "2^-25 refused: " + hr::refusal(b));
        inject[11] = 0.01F;
    }
    {  // non-finite weights refuse the form
        hr::Scan s;
        norm[0] = std::numeric_limits<float>::quiet_NaN();
        inject[0] = std::numeric_limits<float>::infinity();
        hr::scan_norm(s, norm.data(), norm.size());
        hr::scan_inject(s, inject.data(), inject.size());
        check(s.nonfinite == 2 && hr::refusal(s).find("non-finite") != std::string::npos, "non-finite refused: " + hr::refusal(s));
    }
    if (failures == 0) std::printf("qwen_hc_fp16_range_host_test: OK\n");
    return failures == 0 ? 0 : 1;
}
