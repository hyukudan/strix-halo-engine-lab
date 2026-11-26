// MiniMax M3.1 Flash prefill on the RTX PRO: non-linear kernels in the engine's declared orders.
// See mm_rtx_kernels.cuh for the reference of each kernel.
#include "mm_rtx_kernels.cuh"
#include "mm_rtx_index.hpp"
#include "cuda/core/attn_selected_blocks.cuh"
#include "cuda/core/chain/launch.cuh"
#include "cuda/core/chain/prep.cuh"
#include "cuda/core/dense/rows_f32_input_major.cuh"
#include "cuda/core/route_project.cuh"
#include "cuda/core/route_select.cuh"
#include "cuda/core/select_score.cuh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

namespace strix::minimax::rtxpf::cuda {
namespace chain = ::strix::cuda::core::chain;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void rmsnorm(const float* x, const chain::Norm* norm, float* y, std::size_t rows, cudaStream_t s) {
    chain::ChainArgs args{};
    args.aux[0] = norm;
    using Rms = chain::List<chain::Step<chain::Form::norm_rms>>;
    check(chain::launch<kHidden / chain::kThreads>(Rms{}, {x, y, static_cast<unsigned>(rows), kHidden}, args, s), "input norm");
}
void post_exchange(const float* h, const float* p0, const float* p1, const chain::ScalarStore* residual,
                   const chain::Norm* norm, float* y, std::size_t rows, cudaStream_t s) {
    chain::ChainArgs args{};
    args.aux[0] = p1;
    args.aux[1] = h;
    args.aux[2] = residual;
    args.aux[3] = norm;
    using Exchange = chain::List<chain::Step<chain::Form::residual_add, 0>, chain::Step<chain::Form::residual_add, 1>,
                                 chain::Step<chain::Form::store_f32, 2>, chain::Step<chain::Form::norm_rms, 3>>;
    check(chain::launch<kHidden / chain::kThreads>(Exchange{}, {p0, y, static_cast<unsigned>(rows), kHidden}, args, s),
          "post-exchange norm");
}
void route(const float* y, const float* router_w, const float* bias, std::size_t rows, float* scratch, std::uint32_t* ids, float* weights,
           cudaStream_t s) {
    if (!rows) return;
    namespace core = ::strix::cuda::core;
    const moe::RouteScorePlanes<2> scores{scratch, scratch + 1};
    check(core::launch_route_sigmoid_projection<kHidden, kExperts, core::RouteStridedFmaXor32, moe::RouteCodyWaiteExpV1>(
              y, router_w, bias, rows, scores, s),
          "router projection");
    check(core::launch_route_score_planes<RouterPolicy>(scores, rows, ids, weights, s), "router selection");
}
void index_project(const float* x, const float* wiq0, const float* wiq1, const float* wik,
                   float* iq, float* ik, std::size_t rows, cudaStream_t s) {
    if (rows == 0) return;
    const auto launch = [&](IndexProjection projection, const float* weights, float* output) {
        check(::strix::cuda_core::dense_f32_input_major_rows(x, weights, output, rows, kIndexInput,
                  projection.outputs, projection.stride, s), projection.tensor);
    };
    launch(kIndexQuery, wiq0, iq);
    launch(kIndexQuery, wiq1, iq + kIndexQuery.outputs);
    launch(kIndexKey, wik, ik);
}
MsaPrep::~MsaPrep() { reset(); }

void MsaPrep::reset() noexcept {
    if (views_d_) (void)cudaFree(views_d_);
    if (compact_map_d_) (void)cudaFree(compact_map_d_);
    views_d_ = nullptr;
    compact_map_d_ = nullptr;
    view_slots_ = 0;
    layers_ = 0;
}

void MsaPrep::prepare(std::span<const SeqView> seqs, const std::uint32_t* row_seq, std::size_t rows,
                      unsigned layers, std::size_t index_layer_stride, const float* rotary, std::size_t rotary_positions) {
    layers_ = 0;
    layouts_.clear();
    layouts_.reserve(seqs.size());
    std::vector<chain::PrepSequenceView> views;
    views.reserve(seqs.size());
    for (const auto& seq : seqs) {
        // A one-token prompt exports zero rows. It has an attention slot but no prep work.
        if (seq.rows == 0) continue;
        if (!seq.k || !seq.v || !seq.ik) throw std::invalid_argument("MMPREP refused MiniMax RTX missing sequence cache");
        const auto layout = msa_prep_sequence(seq);
        layouts_.push_back(layout);
        views.push_back({layout, seq.k, seq.v, seq.ik});
    }
    // Check the highest layer at setup to validate the full footprint.
    const unsigned last_layer = layers == 0 ? 0 : layers - 1;
    (void)plan_msa_prep(rows, rotary_positions, layouts_, last_layer, layers, index_layer_stride);
    if (rows != 0 && (!row_seq || !rotary))
        throw std::invalid_argument("MMPREP refused MiniMax RTX missing row map or rotary table");

    if (views.size() > view_slots_) {
        if (views_d_) check(cudaFree(views_d_), "MMPREP free sequence views");
        views_d_ = nullptr;
        view_slots_ = 0;
        check(cudaMalloc(&views_d_, views.size() * sizeof(views.front())), "MMPREP allocate sequence views");
        view_slots_ = views.size();
    }
    if (!views.empty())
        check(cudaMemcpy(views_d_, views.data(), views.size() * sizeof(views.front()), cudaMemcpyHostToDevice),
              "MMPREP upload sequence views");

    if (compact_map_d_) {
        check(cudaFree(compact_map_d_), "MMPREP free compact row map");
        compact_map_d_ = nullptr;
    }
    row_seq_ = row_seq;
    if (rows != 0 && views.size() != seqs.size()) {
        std::vector<std::uint32_t> compact_map(rows);
        std::uint32_t sequence = 0;
        for (const auto& layout : layouts_)
            std::fill_n(compact_map.begin() + layout.first_row, layout.rows, sequence++);
        check(cudaMalloc(&compact_map_d_, rows * sizeof(compact_map.front())), "MMPREP allocate compact row map");
        check(cudaMemcpy(compact_map_d_, compact_map.data(), rows * sizeof(compact_map.front()), cudaMemcpyHostToDevice),
              "MMPREP upload compact row map");
        row_seq_ = compact_map_d_;
    }
    rows_ = rows;
    rotary_ = rotary;
    rotary_positions_ = rotary_positions;
    index_layer_stride_ = index_layer_stride;
    layers_ = layers;
}

void MsaPrep::launch(unsigned layer, const MsaPrepGains& gains, float* q, const float* k, const float* v,
                     float* iq, const float* ik, cudaStream_t stream) const {
    const auto plan = plan_msa_prep(rows_, rotary_positions_, layouts_, layer, layers_, index_layer_stride_);
    const chain::PrepBindings args{views_d_, row_seq_, rotary_, q, k, v, iq, ik,
                                  gains.q_w, gains.k_w, gains.iq_w, gains.ik_w};
    check(chain::launch_prep(plan, args, stream), "MMPREP MiniMax RTX prep launch");
}
double g_attn_prof[3] = {0, 0, 0};  // MM_RTX_PROFILE: scores / select / attend seconds (synchronised)
namespace {
struct SubLap {
    bool on;
    std::chrono::steady_clock::time_point t;
    explicit SubLap(bool o) : on(o) { if (on) { cudaDeviceSynchronize(); t = std::chrono::steady_clock::now(); } }
    void lap(int i) {
        if (!on) return;
        cudaDeviceSynchronize();
        const auto n = std::chrono::steady_clock::now();
        g_attn_prof[i] += std::chrono::duration<double>(n - t).count();
        t = n;
    }
};
}  // namespace
void prepare_msa_score_tiles() {
    check(select::prepare_strided_score_tiles(), "MSA score tile shared memory");
}

void prepare_msa_attention() {
    const char* value = std::getenv("MM_RTX_ATTN_DIAG");
    const long diag = value ? std::strtol(value, nullptr, 10) : 0;
    if (diag == 1 || diag == 2)
        throw std::invalid_argument("MMATTN refused MiniMax RTX MM_RTX_ATTN_DIAG");
    check(::strix::cuda::core::attn::prepare_selected_blocks<MsaAttentionShape>(), "MMATTN refused MiniMax RTX staged setup");
}

void launch_msa_attention(const attn::SelectedBlockSequence* seqs, const std::uint32_t* row_seq, unsigned rows,
                          unsigned layer, const float* q, const std::uint32_t* sel, const std::uint32_t* sel_count,
                          float* o, cudaStream_t stream, bool legacy) {
    constexpr float kScale = 0.08838834764831845F;
    check(::strix::cuda::core::attn::enqueue_selected_blocks<MsaAttentionShape>(msa_attention_schedule(legacy), rows,
              kKvHeads, {seqs, row_seq, q, sel, sel_count, o, layer, kScale}, stream),
          "MMATTN refused MiniMax RTX attention launch");
}

void msa_attend(const attn::SelectedBlockSequence* seqs_d, select::StridedScoreArgs score_args, const float* q,
                const select::PickRow* picks, const std::uint32_t* sel, const std::uint32_t* sel_count,
                float* o, cudaStream_t s, bool legacy) {
    const unsigned rows = score_args.row_count;
    if (!rows) return;
    static const bool prof = std::getenv("MM_RTX_PROFILE") != nullptr;
    SubLap sl(prof);
    using select::ScoreForm;
    check(legacy ? select::enqueue_scores<ScoreForm::block_max>(score_args, kMsaScore, s)
                 : select::enqueue_scores<ScoreForm::block_max_tiled>(score_args, kMsaScore, s),
          "MSA block scores");
    sl.lap(0);
    check(select::enqueue_pick(picks, rows * kIdxHeads, kMsaPick, s), "MSA block selection");
    sl.lap(1);
    launch_msa_attention(seqs_d, score_args.row_sequence, rows, score_args.layer, q, sel, sel_count, o, s, legacy);
    sl.lap(2);
}
void export_layer(const SeqView& seq, std::size_t index_layer_stride, unsigned layer, unsigned rank,
                  std::uint64_t r0, std::uint64_t n, std::uint8_t* dst, cudaStream_t s) {
    if (r0 > seq.cap || n > seq.cap - r0 || rank >= 2) throw std::invalid_argument("export_layer: bad range/rank");
    const std::size_t base = static_cast<std::size_t>(layer) * seq.cap + r0;
    // K and V: rows of 4 heads x 128 BF16 (1024 B) in the cache, this rank's 2 heads (512 B) per row in the blob
    check(cudaMemcpy2DAsync(dst, 512, reinterpret_cast<const std::uint8_t*>(seq.k + base * 512) + rank * 512, 1024, 512, n,
                            cudaMemcpyDeviceToHost, s), "export k");
    check(cudaMemcpy2DAsync(dst + n * 512, 512, reinterpret_cast<const std::uint8_t*>(seq.v + base * 512) + rank * 512, 1024, 512, n,
                            cudaMemcpyDeviceToHost, s), "export v");
    const auto index_offset = layer * index_layer_stride + r0 * kHeadDim;
    check(cudaMemcpyAsync(dst + n * 1024, seq.ik + index_offset, n * 256, cudaMemcpyDeviceToHost, s), "export ik");
}

}  // namespace strix::minimax::rtxpf::cuda
