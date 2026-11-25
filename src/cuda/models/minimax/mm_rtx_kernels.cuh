#pragma once
// MiniMax M3.1 Flash prefill on the RTX PRO: the non-linear kernels, each in the declared order
// of the engine's portable host references, so the results equal the nodes' bit for bit:
//   input norm              engine mm_kernels.hpp rmsnorm_gemma_kernel
//   post-exchange + router  strix/minimax_moe.hpp post_exchange_row / route_row
//   index projections       shared MMIDX (per output: fmaf chain over the 6144 inputs, ascending)
//   MSA                     strix/minimax_msa.hpp head_norm_rope / index_block_scores / select_blocks /
//                           sparse_attention (K/V/index-key caches BF16, RNE)
// Build with --fmad=false -prec-div=true -prec-sqrt=true (IEEE division and sqrt, no contraction).
#include <cuda_runtime.h>

#include "mm_rtx_attn.hpp"
#include "strix/model/bind.hpp"
#include "strix/model_data/minimax_m3_1_flash.hpp"
#include "strix/moe/route_policy.hpp"

namespace strix::cuda::core::chain {
struct Norm;
struct PrepSequenceView;
struct ScalarStore;
}

namespace strix::minimax::rtxpf::cuda {

inline constexpr unsigned kHidden = 6144, kExperts = 128, kTopE = 4;
inline constexpr unsigned kLayers = 60;

// The router, bound from models/minimax-m3.1-flash.toml [moe.route] (the host engine binds the same fields).
namespace route_data = ::strix::model_data::minimax_m3_1_flash::moe;
struct RouterPolicy {
    static constexpr ::strix::moe::RoutePolicy value = ::strix::moe::route_policy_named(
        route_data::route::score, route_data::route::selection_bias, route_data::route::key, route_data::route::order,
        ::strix::model::narrow<std::uint32_t>(route_data::experts), ::strix::model::narrow<std::uint32_t>(route_data::top_k),
        route_data::route::renormalize, route_data::route::renorm_single,
        ::strix::model::nearest_float(route_data::route::renorm_eps), ::strix::model::nearest_float(route_data::route::scale));
};
static_assert(RouterPolicy::value.experts == kExperts && RouterPolicy::value.top_k == kTopE);

// Device descriptor uses the effective FP32 gain (1+w) and the 256-lane halving tree.
void rmsnorm(const float* x, const ::strix::cuda::core::chain::Norm* norm, float* y, std::size_t rows, cudaStream_t s);
// Store h + (p0 + p1) through the device residual descriptor targeting h.
// Then write its Gemma RMSNorm to y using the same descriptor contract as rmsnorm.
void post_exchange(const float* h, const float* p0, const float* p1, const ::strix::cuda::core::chain::ScalarStore* residual,
                   const ::strix::cuda::core::chain::Norm* norm, float* y, std::size_t rows, cudaStream_t s);
// Router of route_row: ids / weights [rows][4] (choice order), scratch [rows][128][2].
void route(const float* y, const float* router_w, const float* bias, std::size_t rows, float* scratch, std::uint32_t* ids,
           float* weights, cudaStream_t s);
// Project the two input-major IQ rank slices into one plane and IK into its own plane.
void index_project(const float* x, const float* wiq0, const float* wiq1, const float* wik,
                   float* iq, float* ik, std::size_t rows, cudaStream_t s);

// MSA of one layer for all rows of all sequences, including layer-cut blocks at nonzero positions:
//  prep    q (64 heads) and iq (4 heads) normed + roped in place (FP32); k (4 heads) normed + roped, v, ik normed +
//          roped stored into the sequence caches (BF16) of layer L.
//  attend  per (row, KV head g): index scores of index head g over the visible blocks, top-16 selection (all blocks when
//          at most 16), sparse attention of query heads 16 g .. 16 g + 15 -> o [rows][64 * 128].
struct MsaPrepGains {
    const float *q_w, *k_w, *iq_w, *ik_w;  // 128 Gemma weights each
};

class MsaPrep {
public:
    MsaPrep() = default;
    ~MsaPrep();
    MsaPrep(const MsaPrep&) = delete;
    MsaPrep& operator=(const MsaPrep&) = delete;

    // The supplied device row map matches seqs. The rotary table and cache planes
    // are borrowed through launch. Prepare/reset require the previous use to be complete.
    void prepare(std::span<const SeqView> seqs, const std::uint32_t* row_seq, std::size_t rows,
                 unsigned layers, std::size_t index_layer_stride, const float* rotary, std::size_t rotary_positions);
    void launch(unsigned layer, const MsaPrepGains& gains, float* q, const float* k, const float* v,
                float* iq, const float* ik, cudaStream_t stream) const;
    void reset() noexcept;

private:
    std::vector<ops::chain::PrepSequence> layouts_;
    ::strix::cuda::core::chain::PrepSequenceView* views_d_ = nullptr;
    std::size_t view_slots_ = 0;
    std::uint32_t* compact_map_d_ = nullptr;
    const std::uint32_t* row_seq_ = nullptr;
    const float* rotary_ = nullptr;
    std::size_t rows_ = 0, rotary_positions_ = 0, index_layer_stride_ = 0;
    unsigned layers_ = 0;
};

void prepare_msa_score_tiles();
// Prepare staged attention in the current context before launching it.
void prepare_msa_attention();
// Sequence slots match the row map. Live K/V caches have cudaMalloc alignment.
void launch_msa_attention(const attn::SelectedBlockSequence* seqs, const std::uint32_t* row_seq, unsigned rows,
                          unsigned layer, const float* q, const std::uint32_t* sel, const std::uint32_t* sel_count,
                          float* o, cudaStream_t stream, bool legacy);
// Legacy uses the packed-warp score and per-head attention schedules (MM_RTX_ATTN_LEGACY=1).
void msa_attend(const attn::SelectedBlockSequence* seqs_d, select::StridedScoreArgs score_args, const float* q,
                const select::PickRow* picks, const std::uint32_t* sel, const std::uint32_t* sel_count,
                float* o, cudaStream_t s, bool legacy);

// This rank's state rows of one layer from a sequence's caches into a host-layout buffer (Engine::export_rows: per layer
// K [n][256], V [n][256], index key [n][128], rank r = KV heads 2r, 2r + 1). dst points at the layer's block.
extern double g_attn_prof[3];  // MM_RTX_PROFILE: msa_attend sub-phases (scores, select, attend), seconds
void export_layer(const SeqView& seq, std::size_t index_layer_stride, unsigned layer, unsigned rank,
                  std::uint64_t r0, std::uint64_t n, std::uint8_t* dst, cudaStream_t s);

}  // namespace strix::minimax::rtxpf::cuda
