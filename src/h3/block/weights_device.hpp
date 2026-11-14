#pragma once
#include "h3/block/weights.hpp"
#include "h3/gemm/api.hpp"
#include <array>
#include <cstdlib>
namespace strix::h3::block {
// Shared metadata upload for either driver; bounded lookup preparation replaces
// repeated scale decoding/RNE in each K stage. Never retain expanded INT8 weights.
template <class Buffer, class Upload>
std::size_t upload_operand(const QLinear& q, gemm::Weights& w, std::array<Buffer,3>& storage, Upload&& upload) {
    if (q.operand==gemm::Operand::int8_convrot) return 0;
    q.packed.validate(q.n,q.k,q.w.size());
    // Process-level A/B control; restart readers when changing it.
    static const bool use_lut = [] {
        const char* v = std::getenv("STRIX_H3_W4A8_LUT");
        return !v || v[0] != '0' || v[1] != '\0';
    }();
    const auto prepared = use_lut ? gemm::prepare_w4a8(q.packed) : gemm::PreparedW4A8{};
    w.operand=q.operand;w.group_size=q.packed.group_size;
    w.group_fp8=nullptr; w.group_f32=nullptr; w.codebook=nullptr;
    w.scale_ids=nullptr; w.pair_lut=nullptr; w.group_shift=0;
    if (!prepared.pairs.empty()) {
        storage[0]=upload(prepared.scale_ids);
        storage[1]=upload(prepared.pairs);
        w.scale_ids=storage[0].template as<std::uint8_t>();
        w.pair_lut=storage[1].template as<std::uint16_t>();
        w.group_shift=prepared.group_shift;
        return prepared.bytes();
    }
    storage[0]=upload(q.packed.scales_fp8);
    storage[1]=upload(q.packed.scales_f32);
    storage[2]=upload(q.packed.codebook);
    w.group_fp8=q.packed.scales_fp8.empty()?nullptr:storage[0].template as<std::uint8_t>();
    w.group_f32=q.packed.scales_f32.empty()?nullptr:storage[1].template as<float>();
    w.codebook=q.packed.codebook.empty()?nullptr:storage[2].template as<float>();
    return q.packed.scales_fp8.size()+4ULL*(q.packed.scales_f32.size()+q.packed.codebook.size());
}
} // namespace strix::h3::block
