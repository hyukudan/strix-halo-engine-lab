#pragma once

#include <cstdint>
#include <array>

namespace strix {

// Experimental signed-scale, block-Hadamard projection. The caller owns every
// device allocation and keeps it alive through stream completion. Only integer
// rates 5 and 6 are admitted. Matrices are input-major 16x16 circular tiles;
// this interface does not reinterpret a Q8 tensor or imply model qualification.
struct HipDenseTrellisWeights {
    const std::uint32_t* tiles = nullptr;
    const std::uint16_t* input_scales = nullptr;
    const std::uint16_t* output_scales = nullptr;
    std::uint32_t inputs = 0, outputs = 0, bits = 0;
};

struct HipDenseTrellisWorkspace {
    float* transformed = nullptr; // rows * inputs
    float* partials = nullptr;    // 4 * rows * outputs, split-major
    std::uint64_t transformed_bytes = 0, partial_bytes = 0;
};

// Native GDN/QSA row views. Outputs are BF16-rounded values stored in F32, as
// those input projections require; no gather/scatter allocation or kernel.
struct HipDenseTrellisRows {
    std::array<const std::uint16_t*, 8> input{};
    std::array<float*, 8> output{};
};
void launch_hip_dense_trellis_rows(const HipDenseTrellisWeights& weights,
    const HipDenseTrellisRows& views, std::uint32_t rows,
    const HipDenseTrellisWorkspace& workspace, void* stream);

// BF16 input rows -> FP32 output rows. Input/output strides are elements.
// FP32 signed scaling and Hadamard surround FP32 dot products of exact decoded
// F16 codebook values. This differs from rounding materialized weights to F16;
// a comparison with that reference and a deployed-model quality gate are required.
void launch_hip_dense_trellis(const HipDenseTrellisWeights& weights,
    const std::uint16_t* input, std::uint64_t input_stride, float* output,
    std::uint64_t output_stride, std::uint32_t rows,
    const HipDenseTrellisWorkspace& workspace, void* stream);
// Diagnostic reference organization, preserving the same scalar-codec FMA tree.
void launch_hip_dense_trellis_scalar_reference(const HipDenseTrellisWeights& weights,
    const std::uint16_t* input, std::uint64_t input_stride, float* output,
    std::uint64_t output_stride, std::uint32_t rows,
    const HipDenseTrellisWorkspace& workspace, void* stream);

// Qualification-only reconstruction of the rotated codebook matrix [input,output].
// The caller supplies inputs * outputs F16 cells; no Hadamard/scales are applied.
void launch_hip_dense_trellis_decode(const HipDenseTrellisWeights& weights,
    std::uint16_t* output, std::uint64_t output_bytes, void* stream);

// Borrowed registry, published only by the pinned experimental pack owner.
bool hip_dense_trellis_active() noexcept;
bool hip_dense_trellis_permitted() noexcept;
std::uint64_t hip_dense_trellis_used_layers() noexcept;
class HipDenseTrellisScope {
    bool previous_;
public:
    explicit HipDenseTrellisScope(bool enabled);
    ~HipDenseTrellisScope();
    HipDenseTrellisScope(const HipDenseTrellisScope&) = delete;
};
bool project_hip_dense_trellis_registered(const void* original, const HipDenseTrellisRows& views,
    unsigned rows, unsigned inputs, unsigned outputs, void* stream);

} // namespace strix
