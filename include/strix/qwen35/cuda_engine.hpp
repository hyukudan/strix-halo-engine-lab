#pragma once
// Qwen3.8-27B dense family (`qwen35`) on one CUDA device: reusable engine over the shared cuda_core kernels.
// Home: specific (model layer composition); every arithmetic kernel lives in src/cuda_core (shared).
//
// Drafting contract (Gemma-style target):
//   forward_rows(tokens[R]) runs R rows at positions committed() .. committed()+R-1 and leaves per-row logits / argmax /
//   post-final-norm hidden on the device. NOTHING is committed: the GDN recurrent state and conv state are written to a
//   shadow copy, the KV rows are written past the committed length. commit(k) then accepts the first k rows of the LAST
//   window (0..R): k == R flips the state parity (free); 0 < k < R re-applies the first k rows of the window to the
//   pre-window state in one state pass (the window's per-row GDN inputs are kept on the device); k == 0 discards.
//   Rows 1..R are bitwise equal to row-by-row decode (every reduction order is independent of the row count).
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strix/cuda_core/types.hpp"
#include "strix/qwen35/config.hpp"

namespace strix::qwen35 {

enum class PackFormat : std::uint8_t { bf16, q8_0 };
enum class LogitRows : std::uint8_t { none, last, all };

struct EngineOptions {
    std::uint32_t max_positions = 4096;  // KV capacity per attention layer (and for the MTP layer)
    bool use_graphs = true;              // CUDA graph per (rows, logits mode)
    bool multi_gemv = true;              // one launch for GEMVs sharing x (q|k|v, qkv|z|beta|alpha); bitwise equal either way
    bool pdl = false;                    // (measured +0.27 ms loss at C1: off) programmatic dependent launch + weight L2 prefetch between kernels
    bool load_mtp = true;                // read blk.<layers> (MTP) tensors when present
    bool quiet = false;
};

// Outputs of the last forward (device pointers, valid until the next forward on this engine).
struct RowsOut {
    const float* logits = nullptr;    // [rows][vocab] (all) or [1][vocab] (last), null for none
    const std::uint32_t* top = nullptr;  // [rows] argmax (lowest index on ties) (all) or [1] (last)
    const float* hidden_post = nullptr;  // [rows][hidden] after the final norm (always)
    const float* hidden_pre = nullptr;   // [rows][hidden] residual stream before the final norm
};

class CudaEngine {
public:
    ~CudaEngine();
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    static std::unique_ptr<CudaEngine> from_gguf(const std::string& path, const EngineOptions& opt);
    // Real dimensions of `cfg` (defaults = Qwen3.8-27B) with weights generated on the device by a counter-based RNG
    // (deterministic in seed, scaled so activations stay finite). No disk. Timing equals the real model (memory bound).
    static std::unique_ptr<CudaEngine> synthetic(const Config& cfg, PackFormat pack, const EngineOptions& opt,
                                                 std::uint64_t seed);

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] bool has_mtp() const;
    [[nodiscard]] cudaStream_t stream() const;
    [[nodiscard]] std::uint32_t committed() const;             // committed sequence length
    [[nodiscard]] std::uint64_t weight_bytes() const;          // bytes read per pass by the trunk + lm_head
    [[nodiscard]] std::uint64_t state_bytes_per_pass() const;  // GDN state read + write per pass

    void reset();  // committed length 0, all recurrent state zero, window discarded

    // rows 1..32 (cuda_core::kMaxRows). tokens: host array. Async on stream(); read outputs after synchronizing.
    RowsOut forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);
    // Accepts the first k rows of the last forward_rows window (0 <= k <= rows). Async on stream().
    void commit(std::uint32_t k);

    // MTP layer (single full-attention block with its own KV; positions are explicit, no hidden state to commit).
    // Row r runs at position pos0 + r with input token next_tokens[r] (the token FOLLOWING the hidden's position) and
    // hidden_dev[r] = the trunk hidden of position pos0 + r (use RowsOut::hidden_post or hidden_pre: the convention is the
    // caller's choice). Rows are causal inside the call; earlier positions must have been written by earlier calls.
    struct MtpOut {
        const float* logits = nullptr;  // [rows][vocab]
        const std::uint32_t* top = nullptr;
        const float* hidden_pre = nullptr;   // MTP block output before mtp.norm (feed back for chains, convention 'pre')
        const float* hidden_post = nullptr;  // after mtp.norm
    };
    MtpOut mtp_forward(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0);

    // Host copies (synchronize the stream).
    void read_logits(const RowsOut& o, std::uint32_t row, float* host_vocab) const;
    void read_top(const RowsOut& o, std::uint32_t rows, std::uint32_t* host) const;
    void read_hidden(const float* dev, std::uint32_t rows, float* host) const;
    void sync() const;

    // Diagnostics (no effect on arithmetic).
    void set_graphs(bool on);
    [[nodiscard]] std::uint64_t launches_last_forward() const;

private:
    CudaEngine();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strix::qwen35
