#pragma once

#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_host_forward.hpp"
#include "strix/ds41_pipeline_wire.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Device run of the deepseek41 decode plan (one token per step) on the fronts' kernels: the layer sites on the layer
// piece's, the MoE on the formats/MoE piece's, the attention on the attention piece's. Host work per step: the Engram
// row gather, the routes of the selected experts and the RoPE rows.
//
// One process (world 1) runs every expert and the whole head and emulates the two ranks: both expert half sums, the
// canonical (shared + low) + high, and the split-vocabulary argmax. With world 2 the process is rank `rank` of two:
// it loads only its expert half (ids [rank * n/2, (rank + 1) * n/2)) and its vocabulary slice of the head, computes
// its half sum and its argmax candidate, and exchanges them with the peer through the communicator; every other
// operation is replicated. Both ranks then hold the same y and the same token, bitwise those of world 1.
//
// Every operation's outputs are hashed exactly as HostForward hashes them, so a lockstep run proves bitwise equality op
// by op. With world 2, `raw` holds only the rank's expert slots (its hash is rank-local), and the logits are complete
// only with `gather_logits`.
namespace strix::dist {
class Communicator;
}

namespace strix::deepseek41 {

struct DeviceForwardOptions {
    std::uint32_t layers = 4;  // target layers [0, layers)
    std::uint32_t max_positions = 4096;
    std::uint32_t rank = 0, world = 1;           // world 1 or 2
    dist::Communicator* communicator = nullptr;  // world 2: the peer exchange (not owned)
    bool op_hashes = true;        // hash every op output (downloads each buffer); off for timing runs
    bool gather_logits = true;    // world 2: exchange the logits slices so both ranks hold every logit
    bool engram_direct_io = false;  // Engram rows from the NVMe with O_DIRECT (else from the file mapping)
    bool layer_timing = false;
    // Device events around every plan step of a step (also STRIX_DS41_OP_TIMING=1): per-op device time, host time in
    // the communicator and the weight bytes the op reads (last_op_timing). No synchronisation inside the step.
    bool op_timing = false;
    // world 2: tensor-parallel attention. Rank r runs q_b and the attention of query heads [32r, 32r + 32), wo_a of
    // o-groups [4r, 4r + 4) and its K-half [4096r, 4096r + 4096) of wo_b; the two partial outputs are exchanged and
    // y = y0 + y1 on both ranks (the K-split declaration every run uses for wo_b). q, o and o_low are then rank-local.
    bool tp_attention = false;
    // world 2: the shared expert split by rows over the ranks (opt-in, also STRIX_DS41_SPLIT_SHARED=1). Rank r runs gate/up of
    // intermediate rows [I/2 r, I/2 r + I/2) and down rows [H/2 r, H/2 r + H/2); the A8 activation halves are exchanged
    // while the routed experts run and the down rows travel with the half sum. Every value is the same product: same bits.
    bool split_shared = false;
    bool record_routes = false;
    // world 1, decode (R = 1): the routed experts stay in the model file (page cache / NVMe) and every layer copies its
    // selected experts into device slots (one-node baseline when the experts do not fit in memory). Same bits.
    bool stream_experts = false;  // keep every layer's selected expert ids of the last step (one download per layer)    // per-layer wall times (synchronises the stream at every layer end)
    bool capture_main_hidden = false;  // keep the inputs of layers [layers - 3, layers) of every row (DSpark drafter input)
    std::filesystem::path weights_overlay;  // optional requantization candidate (ds41::WeightOverlay), read instead of the model's tensors
    // Overlay tensors of the Q8_0 x A8 sites (q_a, q_b, kv, o_a, o_b, head) whose codes take at most 16 values are
    // packed at load into CB4 rows (ds41::Q8A8View::cb4) and read by the CB4 kernels: same bits, fewer bytes.
    bool cb4_overlay = false;
    // Routers (ffn_gate_inp, F32) stored as BF16 on the device when every value is a BF16 value (checked at load,
    // else the F32 copy is kept): half the bytes, same bits. Default on; STRIX_DS41_ROUTER_BF16=0 turns it off.
    bool router_bf16 = true;
    // engram_kv (F16) held as its exact F8 rows when every 32-value block is E4M3 x 2^k (checked
    // at load, else the F16 copy is kept); the F16 lanes kernel widens them to the same F16 values.
    // Default on; STRIX_DS41_F16_AS_F8=0 turns it off.
    bool f16_as_f8 = true;
    // cb4_overlay: pack the Q8_0 x A8 site tensors in the planar layout (the shared expert stays interleaved).
    bool packed_planar = false;
    // Native FP8 dense weights (ds41_native_fp8.hpp), off when empty: a directory with the original checkpoint's
    // model.safetensors.index.json and shards, or an extract of its dense tensors (tools/ds41_extract_native_fp8.py).
    // The model's Q8_0 tensors of the selected site classes are replaced at load by the checkpoint's E4M3 codes and
    // 32 x 32 UE8M0 block exponents (the exact original weights, 8.008 bits per weight against 8.5) and run the native
    // FP8 x A8 products (identities e4m3-b32x32-a8-int-*, a different arithmetic from the Q8_0 form: no bitwise gate
    // against the host forward). Takes precedence over an overlay for the same tensors. Decode only (DeviceForward);
    // BatchForward and DevicePrefill keep the Q8_0 form. Also STRIX_DS41_NATIVE_FP8=DIR.
    std::filesystem::path native_fp8;
    // Site classes read natively: q_a, q_b, kv, o_a, o_b, shexp (gate, up and down together), or all. Also
    // STRIX_DS41_NATIVE_FP8_SITES=list.
    std::string native_fp8_sites = "all";
    // Pipeline stages (strix/ds41-pipeline-v1, ds41_pipeline_wire.hpp). first_layer > 0: this forward is the TAIL of a cut
    // at first_layer: it runs target layers [first_layer, layers) and the head on a boundary (step_boundary) instead of
    // the embedding and the layers before (their KV / top-k / candidate outputs that later layers read arrive in the
    // boundary's sections and are kept in mirrors of the source layers' caches). 0 (default): the whole forward.
    std::uint32_t first_layer = 0;
    // false: no head and no argmax: a FRONT stage [0, layers) of a pipeline (export_boundary() after each step).
    bool head = true;
    // trellis weights for the dense decode sites (ds41_trellis.hpp, identity ds41-trellis-k5-v1,
    // docs/ds41-trellis-k5-v1.md), off when empty: a pack directory (dense.L<l>.safetensors + dense.head.safetensors). The
    // selected sites run the trellis library on F32 activations (no A8 at those sites) instead of their Q8_0 / packed
    // copy, which stays loaded. Also STRIX_DS41_TRELLIS=DIR; the build must link the library (CMake STRIX_DS41_TRELLIS).
    std::filesystem::path trellis;
    // Site classes read from the pack: q_a, q_b, kv, o_a, o_b, shexp, head, or all (each class only where its hook
    // exists). Default EMPTY: on v12 no class wins in-app yet, so STRIX_DS41_TRELLIS alone selects nothing (a startup line
    // says so); name the classes explicitly ("shexp", "all", ...). Also STRIX_DS41_TRELLIS_SITES=list.
    std::string trellis_sites;
    // With a trellis shared expert (class shexp): split it over the two ranks (identity ds41-trellis-k5-v1:shexp-tp,
    // ds41_shexp_tp.hpp): rank r runs gate / up rows [1152 r, +1152) and the down K-half [1152 r, +1152), adds its partial
    // onto its half sum before the half-sum exchange; y = (S0 + P0) + (S1 + P1). One process runs both ranks' partials.
    // Default off (the replicated ds41-trellis-k5-v1:shexp). Also STRIX_DS41_SHEXP_TP=1.
    bool trellis_shexp_tp = false;
    // With op_hashes: called with every hashed op output ("op@layer/site:buffer", values) of single steps (calibration).
    std::function<void(const std::string&, const std::vector<float>&)> buffer_tap;
};

// One plan step of the last step (op_timing). ms: device time from the previous step's end event to this step's
// (launch gaps and waits for the host included); exchange_ms: host time inside the communicator while the step was
// enqueued; weight_bytes: bytes of weights the step reads on this rank (routed experts: the experts of this rank the
// rows selected; tensor-parallel attention and the split shared expert: this rank's half; the embedding: its rows).
struct OpTiming {
    Op op = Op::embed;
    std::int32_t layer = -1;
    std::string site;
    double ms = 0.0, exchange_ms = 0.0;
    std::uint64_t weight_bytes = 0;
};

struct LayerTiming {
    double seconds = 0.0;   // wall time of the layer, including its exchange
    double exchange = 0.0;  // inside Communicator::exchange (world 2)
};

class DeviceForward {
public:
    DeviceForward(const std::filesystem::path& gguf, const DeviceForwardOptions& options);
    DeviceForward(const std::filesystem::path& gguf, const HostForwardOptions& options);
    ~DeviceForward();
    DeviceForward(const DeviceForward&) = delete;
    DeviceForward& operator=(const DeviceForward&) = delete;

    const std::vector<float>& step(std::uint32_t token);
    [[nodiscard]] std::uint32_t next_token() const noexcept { return next_token_; }

    // Verify window: tokens[0..R) at positions next_position() .. + R - 1 in one batched pass (R <= max_window_rows),
    // every row bitwise the row of R single steps. Returns each row's greedy token. The state then holds all R rows;
    // commit_window(kept) keeps rows [0, kept) and rolls the others back (position, the ratio-2 compressor carry, the
    // Engram history; ring and compressed entries of dropped rows are rewritten before any later read). No logits
    // are kept for R > 1 and op hashes cover single steps only.
    static constexpr std::uint32_t max_window_rows = 8;
    const std::vector<std::uint32_t>& step_window(std::span<const std::uint32_t> tokens);
    void commit_window(std::uint32_t kept);
    // New sequence: position 0, every ring, compressed/index cache, carry, selection and the Engram history cleared
    // (the state a fresh DeviceForward starts from; weights stay resident).
    void reset();
    [[nodiscard]] const std::vector<std::uint32_t>& last_tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::uint32_t next_position() const noexcept;
    // capture_main_hidden: the input streams of layers layers-3, layers-2, layers-1 of the last step's rows, device
    // [R][hc][hidden] each, ordered on stream() (valid until the next step).
    [[nodiscard]] std::array<const float*, 3> captured_streams() const;
    [[nodiscard]] void* stream() const noexcept;
    // Per-op output hashes of the last step (labels as HostForward's); empty without op_hashes.
    [[nodiscard]] const std::vector<std::pair<std::string, std::uint64_t>>& last_step_hashes() const noexcept { return hashes_; }
    // FNV-1a of the residual stream after each layer of the last step (op_hashes or layer_timing).
    [[nodiscard]] const std::vector<std::uint64_t>& last_layer_stream_hashes() const noexcept { return stream_hashes_; }
    // record_routes: per layer, the rows x top_k expert ids selected by the router in the last step.
    [[nodiscard]] const std::vector<std::vector<std::uint32_t>>& last_step_routes() const noexcept { return routes_; }
    [[nodiscard]] const std::vector<LayerTiming>& last_layer_timing() const noexcept { return timing_; }
    [[nodiscard]] const std::vector<OpTiming>& last_op_timing() const noexcept { return op_timing_; }
    // Seconds of the last step spent reading Engram rows, in the head + argmax (with their exchanges), in total.
    [[nodiscard]] double last_engram_seconds() const noexcept { return engram_seconds_; }
    [[nodiscard]] double last_head_seconds() const noexcept { return head_seconds_; }
    [[nodiscard]] double last_step_seconds() const noexcept { return step_seconds_; }
    // stream_experts: seconds of the last step copying expert bytes from the file to pinned memory.
    [[nodiscard]] double last_stream_seconds() const noexcept;
    // FNV of every layer's attention state after the steps so far (the ring's live window, the used compressed KV and
    // index-key entries, the ratio-2 carry, the last row's top-k), labelled as DevicePrefill::state_hashes: a prefill of
    // the same tokens must leave the same state.
    [[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>> state_hashes() const;
    // The attention state after the positions so far (deepseek41_attention_state.hpp), downloaded; tokens hold the
    // sequence this forward was fed (or imported).
    [[nodiscard]] AttentionState export_state() const;
    // Continue a sequence from a state (a prefill's or another forward's): rings, the KV sources' caches and carries, the
    // index sources' last top-k, the position, the Engram history and next_token(). Then step(next_token()) is the step
    // of position state.position, bitwise. Needs state.position + the rows to come <= max_positions.
    void import_state(const AttentionState& state);
    // Pipeline FRONT (layers < the model's, head off): what the cut after layer layers - 1 carries for the last step (the
    // stream, the FFN mix of the last layer, the KV entries, top-k and candidate blocks later layers read).
    [[nodiscard]] ds41::pipe::Boundary export_boundary() const;
    // Pipeline TAIL (first_layer > 0): one step of the boundary's rows (position must be next_position()); returns each
    // row's greedy token, like step_window (R = 1: the logits are kept as step() keeps them).
    const std::vector<std::uint32_t>& step_boundary(const ds41::pipe::Boundary& boundary);
    [[nodiscard]] const ds41::pipe::BoundaryPlan& boundary_plan() const;
    // Every logit of every row of the last step with the head, [row][vocab] on the host. World 2: a collective (both ranks
    // call it after the same step; each sends its vocabulary slice of every row to the peer). For checks, not a hot path.
    [[nodiscard]] std::vector<float> window_logits();
    // Host logits of the last single-row step (as step() returns them; empty when the step kept none).
    [[nodiscard]] const std::vector<float>& last_logits() const noexcept;
    // World 2: host bytes to the peer and the peer's back (same size on both ranks), through the communicator; an empty
    // `in` drops the peer's bytes (no download).
    void exchange_host(std::span<const std::byte> out, std::span<std::byte> in);
    // Pinned host memory of at least `bytes` owned by the forward (grown on demand; the previous contents are lost when
    // it grows): a pipeline TAIL receives each STEP payload here, so the boundary uploads read pinned memory directly.
    [[nodiscard]] std::span<std::uint8_t> receive_buffer(std::size_t bytes);
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;
    // Tensors packed to CB4 at load (cb4_overlay) and their device bytes.
    [[nodiscard]] std::size_t cb4_tensors() const noexcept;
    // The weight overlay's description line (empty without one).
    [[nodiscard]] std::string overlay_description() const;
    // The exact storage forms in use: "packed sites N, BF16 routers N, F8 rows N".
    [[nodiscard]] std::string storage_description() const;
    // Device logits of the last step, [row][vocab] on stream(); with world 2 only this rank's vocabulary slice is valid.
    [[nodiscard]] const float* device_logits() const noexcept;
    // The last step's logits rows of the vocabulary slice starting at `first` (a vocab_slice first): device_logits() + first
    // with row stride vocab, except under STRIX_DS41_TRELLIS_HEADOUT at R > 1 (the trellis head's per-slice rows, stride =
    // the slice's count; device_logits() then holds no rows of that step). Readers of R > 1 logits use this.
    struct LogitsRows {
        const float* p = nullptr;
        std::size_t stride = 0;  // floats between rows
    };
    [[nodiscard]] LogitsRows device_logits_rows(std::uint32_t first) const;
    [[nodiscard]] double load_seconds() const noexcept { return load_seconds_; }
    // Operations of the plan still bound to a host stand-in (none since the attention piece landed).
    [[nodiscard]] static std::vector<Op> host_ops();

    struct Impl;

private:
    const std::vector<std::uint32_t>& run_step(std::span<const std::uint32_t> tokens, const ds41::pipe::Boundary* boundary);
    std::unique_ptr<Impl> impl_;
    std::uint32_t next_token_ = 0;
    std::vector<std::uint32_t> tokens_;
    std::vector<std::pair<std::string, std::uint64_t>> hashes_;
    std::vector<std::uint64_t> stream_hashes_;
    std::vector<std::vector<std::uint32_t>> routes_;
    std::vector<LayerTiming> timing_;
    std::vector<OpTiming> op_timing_;
    void collect_op_timing(const std::vector<double>& exchange);
    double engram_seconds_ = 0.0, head_seconds_ = 0.0, step_seconds_ = 0.0, load_seconds_ = 0.0;
};

}  // namespace strix::deepseek41
