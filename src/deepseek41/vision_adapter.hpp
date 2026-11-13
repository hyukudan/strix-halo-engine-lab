#pragma once
// DeepSeek V4.1 image tower on the shared vision stack. This file only declares what is specific to the family:
// the tower layout, the tensor names, the preprocessing parameters, the 3x3 unshuffle + aligner projector, the span
// the language model sees and the identity that keys caches. Every computation runs in the shared encoder
// (strix/vision/encoder.hpp) and the shared preprocessing (strix/vision/preprocess.hpp).
#include "strix/deepseek41_model.hpp"
#include "strix/vision/adapter.hpp"
#include "strix/vision/preprocess.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace strix::deepseek41 {

// What the checkpoint does not state. Each field is part of the adapter identity, so a correction after a comparison
// with the upstream processor invalidates cached rows and prefix entries by construction.
struct VisionRecipe {
    enum class Activation : std::uint8_t { gelu, gelu_tanh, silu };
    std::array<float, 3> mean{0.5F, 0.5F, 0.5F}, stddev{0.5F, 0.5F, 0.5F};
    vision::Resampler resampler{vision::Filter::bicubic, true};
    bool swiglu_gate_first = true;                    // first half of mlp.w1's outputs is the SiLU gate
    vision::EncoderConfig::RopePairing rope_pairing = vision::EncoderConfig::RopePairing::half;
    Activation aligner_activation = Activation::gelu;  // between aligner.w1 and aligner.w2
    bool unshuffle_row_major = true;                   // aligner input index = c * r * r + dy * r + dx (else dx * r + dy)
    bool newline_after_last_row = true;                // the row-end embedding also closes the last grid row
    [[nodiscard]] std::string id() const;
};

class VisionAdapter final : public vision::VisionAdapter {
public:
    // weights_sha256_hex: SHA-256 of the admitted vision file (lower-case hex).
    VisionAdapter(VisionConfig config, std::string weights_sha256_hex, VisionRecipe recipe = {});

    vision::EncoderConfig config() const override;
    std::string identity() const override;
    vision::ResizeRule resize() const override;
    vision::PatchGrid patches(const vision::Image& image) const override;
    void bind(vision::WeightSource& source, vision::EncoderWeights& weights) override;
    void project(vision::EncoderOutput& hidden, const vision::PatchGrid& grid, vision::DenseOps& dense,
                 vision::ProjectedRows& out) override;
    vision::SpanLayout layout(const vision::PatchGrid& grid) const override;
    // Split activations: on images with flat areas this tower forms a few rows of very large norm in its middle
    // blocks, and which rows those are flips under single-FP16 operand rounding (measured on the real file).
    vision::EncoderOptions device_options() const override;

    // The same resize contract as resize(), in the form the shared preprocessing takes.
    [[nodiscard]] vision::ResizePolicy resize_policy() const;
    // Size the image is resampled to. Throws std::invalid_argument when its aspect cannot fit max_image_tokens.
    [[nodiscard]] vision::Size target(int width, int height) const;
    // Rows of the span for a merged grid: the grid, one row end per grid row, start and end.
    [[nodiscard]] unsigned span_rows(unsigned grid_h, unsigned grid_w) const noexcept;
    [[nodiscard]] unsigned max_span_rows() const noexcept;
    // Elements of the projector linears (panels the device runner builds besides the tower's own).
    [[nodiscard]] std::uint64_t projector_weight_elements() const noexcept;
    [[nodiscard]] const VisionConfig& model() const noexcept { return config_; }
    [[nodiscard]] const VisionRecipe& recipe() const noexcept { return recipe_; }

private:
    VisionConfig config_;
    std::string sha_;
    VisionRecipe recipe_;
    vision::Tensor aligner_w1_, aligner_b1_, aligner_w2_, aligner_b2_, image_start_, image_newline_, image_end_;
};

}  // namespace strix::deepseek41
