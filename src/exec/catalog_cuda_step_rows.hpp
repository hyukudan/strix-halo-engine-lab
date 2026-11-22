#pragma once

#include "strix/exec/compiler.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace strix::exec {
class CudaProgramRecipe;
}

namespace strix::exec::cuda_step_rows {

enum class Format : std::uint8_t { bf16, q8, f32, unsupported };

struct Matrix {
    const void* data;
    std::uint32_t in, out;
    Format format;
    bool operator==(const Matrix&) const = default;
};

struct Embedding {
    Matrix weight;
    const std::uint32_t* ids;
    float* output;
    bool operator==(const Embedding&) const = default;
};

struct View {
    std::uint32_t groups, width, row_stride, group_stride;
    bool operator==(const View&) const = default;
};

struct Rms {
    const float* input;
    View input_view;
    const float* gain;
    float* output;
    View output_view;
    float epsilon;
    bool operator==(const Rms&) const = default;
};

struct Dense {
    std::array<Matrix, 4> weights;
    std::array<float*, 4> outputs;
    std::uint32_t count;
    const float* input;
    const float* residual;
    const void* up;  // Fused SiLU(gate) * up, when present.
    bool mma;
    bool operator==(const Dense&) const = default;
};

struct Rope {
    float* image;
    View view;
    std::uint32_t rotary_width;
    const float* cosines;
    const float* sines;
    const std::uint32_t* position;
    bool operator==(const Rope&) const = default;
};

struct AttentionShape {
    std::uint32_t heads, kv_heads, width, capacity;
    float scale;
    bool operator==(const AttentionShape&) const = default;
};

struct KvStore {
    AttentionShape shape;
    const float* key;
    const float* value;
    void* key_cache;
    void* value_cache;
    const std::uint32_t* position;
    bool operator==(const KvStore&) const = default;
};

struct Attention {
    AttentionShape shape;
    const float* query;
    View query_view;
    const float* gate;
    View gate_view;
    const void* key_cache;
    const void* value_cache;
    const std::uint32_t* position;
    float* output;
    float* scratch;
    bool operator==(const Attention&) const = default;
};

struct Gdn {
    std::uint32_t key_heads, key_width, value_heads, value_width, conv_taps;
    const float* convolution_weights;
    const float* ssm_a;
    const float* time_bias;
    const float* gain;
    float* convolution_state;
    float* recurrent_state;
    const float* raw;
    float* prepared;
    const float* alpha;
    const float* beta;
    const float* z;
    float* output;
    const std::uint32_t* parity;
    float epsilon;
    bool operator==(const Gdn&) const = default;
};

enum class AttentionPhase { scores, values, combine };
enum class GdnPhase { convolution, recurrence };

struct Pick {
    const float* logits;
    void* workspace;
    std::uint32_t vocabulary;
    std::uint32_t* ids;
    bool operator==(const Pick&) const = default;
};

enum class Leaf : std::size_t {
    embedding, rms, dense, rope, kv_store, pick_partial, pick_final,
    gdn_convolution, gdn_recurrence, attention_scores, attention_values, attention_combine, count
};
using Thunks = std::array<LaunchFn, static_cast<std::size_t>(Leaf::count)>;

// The native table is supplied by the CUDA body. The host can compile recipes
// with trap thunks using the same SDK-free argument storage and grids.
class Bindings {
public:
    explicit Bindings(const Thunks& thunks) : thunks_(thunks) {}
    [[nodiscard]] KernelBinding bind(const Embedding&) const;
    [[nodiscard]] KernelBinding bind(const Rms&) const;
    [[nodiscard]] KernelBinding bind(const Dense&) const;
    [[nodiscard]] KernelBinding bind(const Rope&) const;
    [[nodiscard]] KernelBinding bind(const KvStore&) const;
    [[nodiscard]] KernelBinding bind(const Pick&, Leaf) const;
    [[nodiscard]] KernelBinding bind(const Gdn&, GdnPhase) const;
    [[nodiscard]] KernelBinding bind(const Attention&, AttentionPhase) const;

private:
    const Thunks& thunks_;
};

void project(CudaProgramRecipe&, const Bindings&, bool programmatic_edges,
             std::span<const Matrix> weights, std::span<float* const> outputs, const float* input,
             const float* residual = nullptr, const Matrix* up = nullptr);
[[nodiscard]] const Thunks& native_thunks();

}  // namespace strix::exec::cuda_step_rows
