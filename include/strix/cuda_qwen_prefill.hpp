// CUDA prefill of Qwen3.8-Flash-Next, bit-identical to the HIP tiled prefill under the single A8 profile
// (hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1) of a Strix Halo node, for hand-off to that node.
//
// The prefill computes prompt rows [0, p) and leaves the node's decode state for position p: per row component
// (12 attention KV slices, 12 raw index slices, MTP KV and raw index with shift 1) and the fixed blob (GDN recurrent
// matrices, GDN conv histories, PLE conv history, n-gram history, position word, held wide row h_{p-1}). Each piece is
// handed to the caller (`Sink`) on the host as soon as it is final, so it can be streamed while later rows compute.
// The node then runs row p itself (first token, MTP adoption). Under this profile the bits do not depend on the row
// blocking (verified on the node: T = 1024 and T = 256 give identical state), so the blocks here are free.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strix::cuda::qwen_prefill {

struct Config {
    std::string model_gguf;            // production UD-Q4_K_XL merged GGUF
    std::string mtp_sidecar_gguf;      // MTP sidecar (distillG/D Q8_0); empty = no MTP rows (non-MTP node route)
    std::string tables_dir;            // gfx1151 math tables (qwen_amd_math_tables.hpp)
    std::string rope_table;            // cs[position][32] float2 from the node (rope_cs_262144.bin)
    std::uint32_t max_positions = 32768;
    std::vector<std::uint32_t> block_schedule{1024, 2048, 4096, 8192};  // rows per block; the last entry repeats
    int device = 0;
    bool ple_host_pinned = true;       // per_layer_token_embd (28.8 GB) in pinned host memory, gathered on the host
    // Release per layer: every block's KV/index rows of layer l, and in the last block each GDN layer's recurrent/conv
    // slice (final once that layer ran over the last rows), are copied on a second stream and handed to the sink while
    // the next layers compute, from a host thread (the sink must then be thread-safe; pieces of one component still
    // arrive in ascending order). Same bytes; only when each piece is released changes. false: after each block.
    // Default ON (gate 2026-10-24 05:33: all state bit-identical; net3: faster at every length 1K-32K).
    bool release_by_layer = true;
};

// One finished piece of the node state (host memory owned by the prefill until the next run() starts).
struct Piece {
    bool fixed = false;
    std::string component;             // node component name: kv.L3, index.L3, mtp.kv, mtp.index | recurrent_matrix, ...
    std::uint64_t first = 0;           // first row (row component) or byte offset inside the component (fixed)
    std::uint64_t count = 0;           // rows (row component), 0 for fixed
    const std::uint8_t* data = nullptr;
    std::uint64_t bytes = 0;
};
using Sink = std::function<void(const Piece&)>;

struct Timing {
    double total_s = 0, first_block_s = 0;
    std::vector<double> block_s;       // per block, device time
    std::uint64_t rows = 0;
};

class Prefill {
public:
    explicit Prefill(const Config& config);
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    // Computes rows [0, position) of `tokens` (position <= tokens.size()). Pieces go to `sink` in block order; the
    // fixed pieces come last. Returns after every piece was handed over.
    Timing run(const std::vector<std::uint32_t>& tokens, std::uint64_t position, const Sink& sink);

    // Diagnostics for the bitwise gates: the wide residual rows after layer 47 of the last run (BF16 [rows][10240]).
    [[nodiscard]] std::string describe() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::cuda::qwen_prefill
