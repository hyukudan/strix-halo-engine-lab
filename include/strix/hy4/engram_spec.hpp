#pragma once

// Hy4-flash Engram adapter: the shared Engram Spec (strix/engram/spec.hpp) from the pack's metadata. No hashing
// code here: hash, row reader and row decode are the shared component.
//
// Pack keys (GGUF): hy_v4_flash.ple.layers (0-based Engram layers), ple.ngram_size, ple.heads_per_ngram,
// ple.layer_multipliers ([layer][lookback], the odd int64 draws of the reference stored as data: no RNG here),
// ple.head_vocab_sizes ([layer][order][head] prime moduli), ple.head_offsets (flat-table row of each head, checked),
// engram.pad_token_id, engram.token_map (I32), embedding_length_per_layer_input (row elements).

#include "strix/engram/spec.hpp"

#include <cstdint>
#include <vector>

namespace strix::hy4 {

struct EngramPackMeta {
    std::vector<std::uint32_t> layers;
    std::uint32_t ngram_size = 0, heads_per_ngram = 0, row_elements = 0;
    std::vector<std::int64_t> layer_multipliers;
    std::vector<std::uint64_t> head_vocab_sizes;
    std::vector<std::uint64_t> head_offsets;  // optional (empty: not checked)
    std::uint32_t pad_token_id = 0;
    std::vector<std::int32_t> token_map;
    // Reference lookup table size (configuration_hy_v4_flash engram_lookup_table_size): ids at or past it read the
    // pad's compressed id. The pack's map covers the whole vocabulary (120832), the reference table 120818.
    std::uint32_t lookup_table_size = 120818;
};

// Throws std::invalid_argument on an inconsistent pack (negative map entry or multiplier, head offsets that are
// not the running sum of the moduli in layer order).
[[nodiscard]] engram::Spec engram_spec(const EngramPackMeta& meta);

} // namespace strix::hy4
