#pragma once
// Host-only TP2 token ownership and wire offsets. Odd chunks use the legacy path:
// Channel exchanges equal byte counts in each direction.
#include "glm_traits.hpp"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <string_view>

namespace strix::glm::xfer {
inline unsigned prefill_pipeline(unsigned requested, const char* value) {
    if (value) {
        const std::string_view v(value);
        if (v != "1" && v != "2") throw std::invalid_argument("STRIX_GLM_PF_PIPELINE must be 1 or 2");
        requested = v == "2" ? 2U : 1U;
    }
    if (requested != 1 && requested != 2) throw std::invalid_argument("pf_pipeline must be 1 or 2");
    return requested;
}
// Actual SeqparExchange owners: two depth-sized coherent channel rings,
// one full-row normalized consumer, and runtime/event reserve. Pipeline 2
// takes disjoint halves of the existing engine rows; column scratch is shared.
inline std::size_t seqpar_workspace_bytes(unsigned rows, unsigned depth, std::size_t capacity) {
    if (depth < 2 || depth > 64) throw std::invalid_argument("glm seqpar: channel depth must be 2..64");
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    if (capacity > max-sizeof(Trailer)-4095U) throw std::overflow_error("glm seqpar stride overflow");
    const auto stride = (capacity+sizeof(Trailer)+4095U) & ~std::size_t{4095};
    const auto side = GlmSchedule::row_bytes(rows) + (8U << 20U);
    if (stride > (max-side)/(2U*depth)) throw std::overflow_error("glm seqpar workspace overflow");
    return 2U*depth*stride + side;
}
struct SeqparRows {
    unsigned first, count;
    [[nodiscard]] constexpr std::size_t elements(unsigned width = 4096) const {
        return std::size_t{first} * width;
    }
    [[nodiscard]] constexpr std::size_t bytes(std::size_t row_bytes = kRowF32) const {
        return std::size_t{count} * row_bytes;
    }
};
[[nodiscard]] constexpr bool seqpar_admits(unsigned rows) { return rows >= 2 && rows % 2 == 0; }
[[nodiscard]] constexpr SeqparRows seqpar_rows(unsigned rows, unsigned rank) {
    if (!seqpar_admits(rows) || rank > 1) throw std::invalid_argument("glm seqpar: even rows >= 2 and rank 0/1 required");
    return {rank * (rows / 2), rows / 2};
}
[[nodiscard]] constexpr unsigned seqpar_last_source(unsigned rows, unsigned rank) {
    const auto own = seqpar_rows(rows, rank);
    return rank == 0 ? own.first : own.first + own.count - 1U;
}

// Keep the pipeline's column calls at R/2 rows. Each call owns a contiguous
// quarter on each rank, so its RS need not wait for the other column call.
// Odd half sizes keep the original SP schedule, without changing admission.
[[nodiscard]] constexpr unsigned seqpar_tiles(unsigned rows, unsigned pipeline) {
    return pipeline == 2 && rows >= 4 && rows % 4 == 0 ? 2U : 1U;
}
[[nodiscard]] constexpr SeqparRows seqpar_tile(unsigned rows, unsigned pipeline, unsigned tile) {
    if (!seqpar_admits(rows) || tile >= seqpar_tiles(rows, pipeline))
        throw std::invalid_argument("glm seqpar: invalid tile");
    const unsigned count = rows / seqpar_tiles(rows, pipeline);
    return {tile * count, count};
}
[[nodiscard]] constexpr SeqparRows seqpar_owner(SeqparRows tile, unsigned rank) {
    const auto own = seqpar_rows(tile.count, rank);
    return {tile.first + own.first, own.count};
}

// Scheduling only: ownership, wire tiles and numerical contracts are unchanged.
// Keep the qualified two-stream organization as the default during measurement.
[[nodiscard]] constexpr unsigned seqpar_compute_streams(std::string_view value = "2") {
    if (value == "1") return 1;
    if (value == "2") return 2;
    throw std::invalid_argument("STRIX_GLM_PF_STREAMS must be 1 or 2");
}
[[nodiscard]] constexpr SeqparRows seqpar_moe_piece(unsigned rows, unsigned first) {
    if (first >= rows) throw std::invalid_argument("glm seqpar: invalid MoE piece");
    return {first, rows - first < 8192U ? rows - first : 8192U};
}

enum class SeqparStep { pre_attn, attn, reduce_attn, post_attn, pre_ffn, ffn, reduce_ffn, post_ffn };
// One compute queue, independent DMA queue. Publish attention A's RS before
// computing B; publish AG A before RS B, as in the legacy wire protocol. The
// grouped FFN consumes BOTH gathers, then publishes each half's RS separately.
// Keep this host-only schedule executable by the ownership/lifetime tests.
template<class Emit> void seqpar_single_layer(unsigned tiles, Emit&& emit) {
    if (tiles != 1 && tiles != 2) throw std::invalid_argument("glm seqpar: invalid tile count");
    emit(SeqparStep::pre_attn, 0);
    if (tiles == 2) emit(SeqparStep::pre_attn, 1);
    emit(SeqparStep::attn, 0);
    emit(SeqparStep::reduce_attn, 0);
    if (tiles == 2) emit(SeqparStep::attn, 1);
    emit(SeqparStep::post_attn, 0);
    emit(SeqparStep::pre_ffn, 0);
    if (tiles == 2) {
        emit(SeqparStep::reduce_attn, 1);
        emit(SeqparStep::post_attn, 1);
        emit(SeqparStep::pre_ffn, 1);
    }
    emit(SeqparStep::ffn, 0);
    emit(SeqparStep::reduce_ffn, 0);
    if (tiles == 2) emit(SeqparStep::reduce_ffn, 1);
    emit(SeqparStep::post_ffn, 0);
    if (tiles == 2) emit(SeqparStep::post_ffn, 1);
}

// A separate plan prevents an AR peer from silently interpreting an SP phase.
// Reuse Channel's publication, framing, CRC and drain/seal lifecycle unchanged.
struct GlmSeqparTraits : GlmTraits {
    static constexpr std::uint64_t kPlan = 0x3350534d4c47ULL; // GLMSP3: direct AG payload + separate trailer
};
}  // namespace strix::glm::xfer
