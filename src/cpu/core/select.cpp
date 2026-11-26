#include "cpu/core/backend.hpp"

namespace strix::cpu {

Kernels select(const Caps& c) noexcept {
    Kernels table = reference_kernels();
    table.isa = c.neon ? Isa::neon : select_isa(c, {.avx2 = true, .avx512 = c.avx512bw});
    return table;
}

}  // namespace strix::cpu
