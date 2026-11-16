#pragma once
// Qwen3.8-27B dense family (`qwen35`) on one HIP device: window engine over the shared HIP cores (dense_linear,
// attn_core dense_decode, hip_gdn_decode_step_dims, ops/norm). Same drafting contract as CudaEngine:
//   forward_rows(tokens[R]) runs R rows at positions committed() .. committed()+R-1 (R <= max_rows()) and leaves per-row
//   logits / argmax on the device. Nothing is committed: the GDN matrix + conv state are snapshotted first, the KV rows are
//   written past the committed length (dead until committed). commit(k) accepts the first k rows of the LAST window:
//   k == R keeps the advanced state (free); 0 < k < R restores the snapshot and replays the first k rows of every GDN layer
//   from the saved raw inputs; k == 0 restores only. Rows are bitwise equal to row-by-row decode (every reduction order is
//   independent of the row count).
// Home: specific (model layer composition: qwen35 GDN/attention interleave); every kernel is shared (backend core).
#include <cstdint>
#include <memory>
#include <string>

#include "strix/qwen35/config.hpp"
#include "strix/qwen35/engine_types.hpp"

namespace strix::qwen35 {

struct HipEngineOptions {
    std::uint32_t max_positions = 4096;  // KV capacity per full-attention layer
    bool quiet = false;
    // forward_commit window width. 16 = exact (the serial arithmetic). > 16 = FAST policy dense-q8a8-f16-v1 (shared
    // core::f16_dense_tall::q8a8: A8 activations x Q8_0 weights, FP16 WMMA) for the dense projections of those wide windows;
    // GDN and attention keep their exact kernels. Decode windows (forward_rows) stay <= 16 rows.
    std::uint32_t prefill_rows = 16;
};

struct HipRowsOut {
    const float* logits = nullptr;       // device [rows][vocab] (all) or [1][vocab] (last); null for none
    const std::uint32_t* top = nullptr;  // device [rows] argmax (lowest index on ties) (all) or [1] (last)
};

class HipEngine {
public:
    using LogitRows = qwen35::LogitRows;
    ~HipEngine();
    HipEngine(const HipEngine&) = delete;
    HipEngine& operator=(const HipEngine&) = delete;
    static std::unique_ptr<HipEngine> from_gguf(const std::string& path, const HipEngineOptions& opt);

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] std::uint32_t vocab() const;
    [[nodiscard]] static constexpr std::uint32_t max_rows() { return 16U; }
    [[nodiscard]] std::uint32_t prefill_rows() const;  // widest forward_commit window (options)
    [[nodiscard]] std::uint32_t committed() const;
    [[nodiscard]] std::uint64_t weight_bytes() const;          // bytes read per pass (all weights but the embedding table)
    [[nodiscard]] std::uint64_t state_bytes_per_pass() const;  // GDN state read + write per pass
    [[nodiscard]] std::uint32_t kv_bytes_per_position() const;
    void reset();  // committed length 0, recurrent state zero, window discarded

    // rows 1..max_rows(). tokens: host array. Async on the stream; read outputs after read_top / sync.
    HipRowsOut forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);
    void commit(std::uint32_t k);  // accepts the first k rows of the last forward_rows window (0 <= k <= rows)
    // Same arithmetic, every row accepted immediately: no snapshot, no replay inputs (prefill chunks, serial decode).
    HipRowsOut forward_commit(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);

    void read_top(const HipRowsOut& o, std::uint32_t rows, std::uint32_t* host) const;  // synchronizes
    void read_logits(const HipRowsOut& o, std::uint32_t rows, float* host) const;       // [rows][vocab], synchronizes
    void sync() const;
    // Timing only: set the committed length to `pos` WITHOUT computing (KV / state keep their contents; positions below pos are
    // whatever the cache holds). Used to sample window cost at long context without running the whole prefill.
    void bench_seek(std::uint32_t pos);

private:
    HipEngine();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strix::qwen35
