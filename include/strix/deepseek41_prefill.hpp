#pragma once

#include "strix/deepseek41_attention_state.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <array>
#include <vector>

// Device prefill of a deepseek41 prompt: T rows of the decode plan at once, layer by layer, on the fronts' many-row
// kernels (layer sites, the prefill experts' router / routed / shared WMMA kernels, the layer piece's dense prefill
// projections) and the attention piece's kernels in windows of consecutive rows. Every row gets the bits the decode
// step of its position would give, and after the prompt every layer's attention state (window ring, compressed KV,
// index keys, ratio-2 carry) is the state T decode steps leave, so decode continues from it.
//
// world 1: one process holds every expert of the range (a trimmed model; the full model does not fit one node) and
// emulates the two expert halves (canonical (shared + low) + high). world 2: rank `rank` of two keeps only its expert
// half and its vocabulary slice; the half sums of each MoE chunk and the argmax candidate are exchanged with the peer.
namespace strix::dist {
class Communicator;
}

namespace strix::deepseek41 {

struct PrefillOptions {
    std::uint32_t layers = 40;           // target layers [0, layers)
    std::uint32_t max_positions = 16384; // prompt rows the state and the buffers are sized for
    // Rows of one run (0: max_positions). With extend() a long prompt runs in pieces of at most max_rows rows: the row
    // buffers hold max_rows rows, the attention caches max_positions positions.
    std::uint32_t max_rows = 0;
    std::uint32_t rank = 0, world = 1;
    dist::Communicator* communicator = nullptr;  // world 2 (not owned)
    std::uint32_t chunk_rows = 1024;     // rows per MoE / attention-projection / Engram chunk (half sums of a chunk <= 21 MB)
    std::uint32_t window_rows = 8;       // rows per attention window (1..attn::kMaxCallRows; also the unit of the row split)
    // rows per routed-MoE chunk (a multiple of chunk_rows; the expert weights are read once per chunk); world 2 sends
    // a chunk's half sums in pieces of chunk_rows rows on a communication stream while the next chunk runs
    std::uint32_t moe_chunk_rows = 4096;
    std::uint32_t attention_lanes = 4;   // streams the windows of a chunk are spread over (index, select, attention)
    // world 2: tensor-parallel attention as the two-rank decode forward: rank r runs q_b of heads [32r, 32r + 32), their
    // attention, wo_a of o-groups [4r, 4r + 4) and wo_b's K-half r; the halves of every chunk are exchanged and
    // y = h0 + h1 (the declared K-split, so the same bits as one process). Window, compressor and indexer replicated.
    bool tp_attention = true;
    // world 2: the rows of the prompt split between the ranks (8-row windows alternate, window k -> rank k % 2). Each
    // rank runs the row-local work (sites, norms, Engram, projections, attention with all heads, router, shared expert)
    // on its own rows only; the window ring and the compressor run on both ranks over every row from the owners' K/V
    // rows (all-gathered), the routed experts stay split by expert (the MoE input all-gathered, each half sum sent to
    // the row's owner). Same bits as one process; replaces tp_attention.
    bool row_split = true;  // the world-2 default (main 07:20); false: tensor-parallel attention, every row on both ranks
    // The routed experts are not resident: two layer slots, the next layer's experts read from the file (O_DIRECT) on a
    // thread while the current layer runs (one node: the whole model does not fit). Same bits.
    bool stream_experts = false;
    bool engram_direct_io = true;        // Engram rows from the NVMe with O_DIRECT (else the file mapping)
    bool stream_hashes = false;          // FNV of the last row's residual stream after each layer (gate against decode)
    bool gather_logits = true;           // world 2: exchange the vocabulary slices so both ranks hold every logit
    bool timing = false;                 // per-layer wall times (synchronises at every layer end)
    // Exact suffix replay (the full 40 layers only): decoder layers compute only the cone of rows the final state and
    // the last row's logits depend on (ds41_prefill.hpp plan_suffix_replay); every result and the live attention state
    // equal those of a full prefill.
    bool cone = true;
    std::uint32_t draft_rows = 0;        // rows of the drafter's seed the cone must also cover (0: none)
    bool routing_loads = false;          // record the routed experts' entry counts per layer (load-balance study)
    // world 2 with row_split: a run of fewer rows takes the tensor-parallel attention form instead (same bits; short
    // prompts: fewer exchanges per layer). Default 9: runs of one window (<= 8 rows) take the TP form (pair: 8 rows 115
    // vs 124 ms; at 32 rows and above row_split + joint is faster). 0: always row_split. STRIX_DS41_PREFILL_SPLIT_MIN_ROWS
    // overrides.
    std::uint32_t split_min_rows = 9;
    // world 2 row_split MoE: a run of at most this many rows takes the joint expert pass over [peer rows; own rows] (one
    // read of the expert half per layer instead of two: the short prompts are bound by those reads; same bits). 0: only
    // STRIX_DS41_PREFILL_JOINT=1 (every run). Default 2048 (pair: -10 % at 32, -12 % at 128, -6 % at 512, -8 % at 2K
    // against two passes; P11: neutral at 16K). STRIX_DS41_PREFILL_JOINT_MAX_ROWS overrides.
    std::uint32_t joint_max_rows = 512;  // P14: the joint pass loses 3 % at 2K, wins at 128-512 rows
    // Rows of a dummy prompt run once at construction (first kernel launches, scratch first touch), so the first real
    // run does not pay them; the next run starts a fresh sequence as usual (world 2: both ranks must agree). 0: none.
    std::uint32_t warmup_rows = 0;
    // Requantization overlay (ds41::WeightOverlay: tensors of the same type and size replacing the model's by name),
    // loaded in its stored Q8_0 form: the overlay's bits, the base kernels. Empty: STRIX_DS41_OVERLAY, else none.
    std::filesystem::path weights_overlay;
    // Device tensors of another forward in this process (BatchForward::file_form_weights), by name: a tensor found here
    // with the bytes this prefill would load (its expert range, its vocabulary slice) is read in place, not loaded nor
    // owned; the others are loaded as usual. Serving keeps one copy of the weights per node this way. The owner must
    // outlive the prefill.
    std::map<std::string, std::pair<const void*, std::uint64_t>> shared_weights;
};

// timing (diagnostic: the stream is synchronized around each part, which removes overlap across parts). attention
// and moe are wall intervals that INCLUDE their own exchanges; exchange is the communicator's wall time inside the
// layer (start to end of every exchange, overlapped work included) and exchange_blocked the part of it the CPU spent
// in finish() waiting for the transfer. Do not add exchange to attention + moe.
struct PrefillLayerTiming {
    double seconds = 0.0, attention = 0.0, moe = 0.0, exchange = 0.0, exchange_blocked = 0.0;
};

struct PrefillResult {
    std::vector<float> logits;        // last row (world 2 without gather_logits: this rank's slice, other entries zero)
    std::uint32_t next_token = 0;
    std::vector<std::uint64_t> stream_hashes;  // last row, after each layer (stream_hashes)
    double seconds = 0.0, engram_seconds = 0.0, head_seconds = 0.0;
    double engram_wait_seconds = 0.0;  // overlapped Engram read: the part the run waited for (before the first Engram layer)
    std::vector<PrefillLayerTiming> layers;
    std::uint64_t row_layers = 0;        // rows x layers computed (the cone's saving against rows x 40)
    std::vector<std::vector<std::uint32_t>> routing_loads;  // routing_loads: [layer][expert] entries of the output rows
    double expert_read_seconds = 0.0, expert_wait_seconds = 0.0;  // stream_experts: file reads (thread), waits for them
    // row_split: whether the last row is this rank's (stream_hashes and the last row's top-k only there), and the rows
    // this rank computed in the encoder layers
    bool last_row_local = true;
    std::uint32_t local_rows = 0;
    // timing: attention of every layer in parts (projections for compressor / indexer, window K/V + compressor,
    // q_b, windows (index, select, attention; TP: the previous chunk's halves), o_a + wo_b)
    std::array<double, 5> attention_parts{};
};

class DevicePrefill {
public:
    DevicePrefill(const std::filesystem::path& gguf, const PrefillOptions& options);
    ~DevicePrefill();
    // The overlay's path, identity, tensor count, bytes and content FNV (empty without an overlay).
    [[nodiscard]] std::string overlay_description() const;
    DevicePrefill(const DevicePrefill&) = delete;
    DevicePrefill& operator=(const DevicePrefill&) = delete;

    // The whole prompt from position 0 (a fresh sequence).
    PrefillResult run(std::span<const std::uint32_t> tokens);
    // The next piece of the same sequence: tokens at positions [rows so far, + tokens.size()), reading the attention
    // state (rings, compressed KV and index keys, ratio-2 carries) and the Engram history the earlier runs left. The rows
    // so far must be a multiple of window_rows. Every row and the final state equal those of one run of the whole prompt.
    PrefillResult extend(std::span<const std::uint32_t> tokens);
    [[nodiscard]] std::uint32_t positions() const noexcept;  // rows of the sequence so far
    // FNV of every layer's attention state after the last run (the ring's live window, the used compressed KV and
    // index-key entries, carry, the last row's top-k), labelled; the same labels and order as DeviceForward::state_hashes.
    [[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>> state_hashes() const;
    // The attention state after the last run (deepseek41_attention_state.hpp), for a decode forward to continue from:
    // every layer's ring, the KV sources' caches and carries, and the last row's top-k on the rank that computed it
    // (row_split: topk_known false on the other). position = the prompt length; tokens and next_token stay empty / 0
    // (the caller holds the prompt and PrefillResult::next_token).
    [[nodiscard]] AttentionState export_state() const;
    // Continue a sequence from a state (another run's export, read from a file): rings, caches, carries, the Engram
    // history of state.tokens and the rows so far = state.position (a multiple of window_rows); then extend().
    void import_state(const AttentionState& state);
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;         // loaded (owned) by this prefill
    [[nodiscard]] std::uint64_t shared_weight_bytes() const noexcept;  // read in place from shared_weights
    [[nodiscard]] double load_seconds() const noexcept;

    struct Impl;

private:
    PrefillResult run_from(std::span<const std::uint32_t> tokens, std::uint32_t start);
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::deepseek41
