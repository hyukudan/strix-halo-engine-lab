#pragma once

#include <cstdint>

namespace strix::cuda::core::select {

enum class KeyOrder : std::uint8_t { ordered, canonical_float, raw_float, eligible_float };

// A pick can emit a group ID, expand groups into tokens with a partial tail,
// or reserve the newest (possibly incomplete) block. Keys and IDs are in
// chronological order. A join supplies global IDs in that same order.
struct PickSpec {
    unsigned top_k;
    unsigned expansion = 1;
    KeyOrder order = KeyOrder::canonical_float;
    bool force_newest = false;
};

enum class Pool : std::uint8_t { none, source, user };

struct PickRow {
    const float* scores = nullptr;
    const std::uint32_t* keys = nullptr;
    const std::uint32_t* ids = nullptr;
    unsigned groups = 0, tail = 0;
    std::uint32_t* selected = nullptr;
    std::uint32_t* selected_keys = nullptr; // Optional ordered keys, one per selected group.
    std::uint32_t* count = nullptr;
    float* mask_scores = nullptr;
    Pool pool = Pool::none;
    unsigned pool_width = 0, pool_top = 0;
    std::uint8_t* candidates = nullptr;
    std::uint32_t* pool_keys = nullptr;
};

// Query codes contain four signed, doubled E2M1 values per word. Cache entries
// contain 64 packed E2M1 bytes then four E8M0 exponent bytes.
struct PackedQuery {
    int codes[32][32];
    float scales[32][4];
    float weights[32];
};

struct PackedFp8Query {
    alignas(4) std::uint8_t codes[32][128];
    float weights[32];
};

struct ScoreRow {
    const float* query = nullptr;
    const float* keys = nullptr;
    const float* weights = nullptr;
    const PackedQuery* packed_query = nullptr;
    const std::uint8_t* packed_keys = nullptr;
    const PackedFp8Query* fp8_query = nullptr;
    const float* key_scales = nullptr;
    const std::uint16_t* bf16_keys = nullptr;
    unsigned visible = 0;
    float* scores = nullptr;
    std::uint32_t* ordered = nullptr;
    unsigned* nonfinite = nullptr;
};

// These are numerical forms. A different scheduling tile may use the same
// form only when it preserves the declared reduction and each write guard.
enum class ScoreForm : std::uint8_t {
    relu_tree, weighted_fma, weighted_bf16, weighted_bf16x3, weighted_fp4, weighted_fp8, block_max, block_max_tiled,
};

struct ScoreSpec {
    unsigned top_k;
    unsigned heads;
    unsigned dim;
    unsigned block_width = 1;
};

} // namespace strix::cuda::core::select
