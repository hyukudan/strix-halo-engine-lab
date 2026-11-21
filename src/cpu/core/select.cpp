// Kernel-table selection (layer 3 is picked by capability).
#include "cpu/core/backend.hpp"
#include "cpu/arch/avx512/kernels.hpp"
#include "cpu/arch/neon/kernels.hpp"

namespace strix::cpu {

namespace {
void fill(Kernels& k, const Kernels* spec) {
    if (!spec) return;
    k.isa = spec->isa;
    if (spec->quantize_a8) k.quantize_a8 = spec->quantize_a8;
    if (spec->gemv_q8_0_a8) k.gemv_q8_0_a8 = spec->gemv_q8_0_a8;
    if (spec->rmsnorm_f32) k.rmsnorm_f32 = spec->rmsnorm_f32;
}
}  // namespace

Kernels select(const Caps& c) noexcept {
    Kernels table = reference_kernels();
    if (c.avx512f && c.avx512bw) fill(table, arch::avx512::kernels());
    if (c.neon) fill(table, arch::neon::kernels());
    return table;
}

const Kernels* registered_weight_kernels(Isa isa) noexcept {
    switch (isa) {
    case Isa::scalar: return &reference_kernels();
    case Isa::avx512: return caps().avx512f && caps().avx512bw ? arch::avx512::kernels() : nullptr;
    case Isa::neon: return caps().neon ? arch::neon::kernels() : nullptr;
    default: return nullptr;
    }
}

}  // namespace strix::cpu
