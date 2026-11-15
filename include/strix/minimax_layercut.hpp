#pragma once
#include "strix/pair/layercut.hpp"
namespace strix::minimax::layercut {
inline constexpr std::uint32_t layers = 60, hidden = 6144;
inline constexpr std::uint64_t seam_bytes = 24576, state_bytes = 2560, dedup_state_bytes = 2304;
inline constexpr const char* format = "minimax-m3.1/layercut-v1";
inline constexpr std::uint32_t rank_state_widths[] = {512, 512, 256};
inline pair::chain::ModelShape chain_shape() {
    return {layers, double(seam_bytes), std::vector<double>(layers, double(state_bytes)), std::vector<double>(layers, 0), {}};
}
// The pair-alone prefill chunk (engine Options::max_rows default): production bits depend on it, so the default pair
// identity names it (pair::layercut::identity_grain); a layer-cut block is a multiple of it.
inline constexpr std::uint32_t pair_chunk_rows = 4096;
inline std::string default_pair_identity() { return "default/c" + std::to_string(pair_chunk_rows); }
inline std::string identity(unsigned cut, const std::string& pair) {
    return "mm-layercut-v1 rtx=cuda-fp32id L[0," + std::to_string(cut) + ") pair=" + pair + " L[" + std::to_string(cut) + ",60)";
}
} // namespace strix::minimax::layercut
