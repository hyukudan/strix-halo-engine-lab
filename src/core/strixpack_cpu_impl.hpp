#pragma once

#include "strix/weight/strixpack_cpu.hpp"

namespace strix::weight::sp::cpu_detail {

struct Operations {
    DecodeRows decode;
    ProjectRows project;
};

#if defined(STRIX_SP_X86_SIMD)
extern const Operations avx2[3], avx512[3];
#endif

template <class S>
inline float level(const Header& h, unsigned code) {
    const auto mag = static_cast<float>(h.table[code % S::levels]);
    return code < S::levels ? mag : -mag;
}

}  // namespace strix::weight::sp::cpu_detail
