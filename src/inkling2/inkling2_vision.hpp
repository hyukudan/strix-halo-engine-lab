#pragma once
// Inkling 2 vision tower adapter (strix::vision::VisionAdapter) for the shared image runtime.
//
// The reference is transformers 5.24 `Inkling2VisionModel` / `Inkling2ImageProcessor` (the checkpoint's own model
// code): images are not resized, each side is padded to a multiple of 40 with the value -1 on the 0..255 scale and
// normalised as (v - 127.5) / 127.5; every 40 x 40 cell is one language-model token made of 2 x 2 native 20 x 20
// patches, patch features in (y, x, channel) order, patches in cell order (cell row, cell column, dy, dx). A class
// token sits in front of the patches at rotary position (0, 0); patch (y, x) has position (y + 1, x + 1). Learned
// positions: the 32 x 32 table resampled bilinearly with align_corners=False. 23 pre-LayerNorm blocks with LayerScale,
// exact GELU MLP, axial rotary positions (width angles then height angles, adjacent channel pairs, base 10000), a
// LayerNorm before block 0 and no final norm. The class row is dropped; the four patch rows of a cell are
// concatenated (slot-major, slot = dy * 2 + dx) and projected by proj_in (4096 -> 4096), GELU, proj_out
// (4096 -> 2304), no biases. The rows replace the image placeholder token 200054 and still pass through the LM embedding norm.
//
// Mapping onto the shared tower (no shared-kernel change): the class token is patch row 0 with zero pixels (the patch
// embedding has no bias) and its embedding as that row's position addend; LayerScale is folded into the out and
// second MLP linears (FP32, before the device builds its FP16 panels); q, k and v are concatenated.
#include "strix/vision/adapter.hpp"
#include "strix/vision/preprocess.hpp"
#include "strix/weight/tensor_access.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strix::inkling2 {

inline constexpr std::uint32_t kVisionImageTokenId = 200054;  // <|unused_200054|>, Inkling2Config::image_token_id
inline constexpr unsigned kVisionCell = 40;                     // pixels per image token side (patch 20 x merge 2)
inline constexpr unsigned kVisionDefaultMaxTokens = 4096;

// vision_config of the checkpoint, checked against the one tower this adapter implements.
struct VisionSpec {
    unsigned layers = 23, width = 1024, heads = 16, mlp_width = 4096, patch = 20, merge = 2, pos_grid = 32;
    unsigned projection = 4096, text_width = 2304;
    float eps = 1e-5F, rope_theta = 10000.0F;
    bool projector_bias = false;
};
// Parse and validate config.json's vision_config (throws std::invalid_argument on any difference).
VisionSpec parse_vision_spec(const std::string& config_json_text);

// Exact schema of the checkpoint's vision tensors (model.visual.*, BF16) for a weight::TensorLookup over the one
// safetensors file; the language-model tensors are the ignored namespace kVisionIgnoredPrefix.
inline constexpr std::string_view kVisionIgnoredPrefix = "model.llm.";
[[nodiscard]] std::vector<weight::TensorSpec> vision_tensor_specs(const VisionSpec&);

// The image rule: the number of tokens of an image of the given size (no resize, padded to the cell).
[[nodiscard]] unsigned vision_tokens(unsigned width, unsigned height, unsigned cell = kVisionCell);
// The size an image is scaled to so that it fits `max_tokens` (aspect kept; unchanged when it already fits).
struct VisionSize {
    unsigned width = 0, height = 0;
};
[[nodiscard]] VisionSize vision_fit(unsigned width, unsigned height, unsigned max_tokens, unsigned cell = kVisionCell);

// Processing policy as data. Fields marked UNCONFIRMED are not stated by the checkpoint files on this machine
// (no processor or modelling code ships with it); each default is the assumption, documented as such.
struct VisionRecipe {
    std::array<float, 3> mean{0.48145466F, 0.4578275F, 0.40821073F};    // UNCONFIRMED
    std::array<float, 3> stddev{0.26862954F, 0.26130258F, 0.27577711F};  // UNCONFIRMED
    vision::Resampler resampler{vision::Filter::bicubic, true};          // only used when an image exceeds the limits
    std::uint8_t pad_fill = 0;                     // UNCONFIRMED: grey of the bottom / right margin up to the 40 px cells
    std::uint32_t max_tokens = 4096;               // UNCONFIRMED: cells per image; larger images are scaled down
    std::uint32_t min_side_cells = 1;
    bool class_row = true;                         // UNCONFIRMED: the class row takes part in attention
    bool merge_slot_major = true;                  // UNCONFIRMED: proj_in input index = (dy * 2 + dx) * W + c
    vision::Activation projector_activation = vision::Activation::gelu;  // UNCONFIRMED (between proj_in and proj_out)
    vision::EncoderConfig::Mlp mlp = vision::EncoderConfig::Mlp::gelu;  // UNCONFIRMED (exact erf GELU)
    std::uint32_t placeholder_token = 200054;      // <|unused_200054|> after <|content_image|> in the chat template
    [[nodiscard]] std::string id() const;
};
// The image identity the adapter runs. Production entry points never pick it here: it comes from the row
// inkling2-vision-recipe of the model's profile table (include/strix/profiles/inkling2_mini.hpp, mini_vision_axial),
// resolved with the run's other rows; the direct constructor is for the adapter's own tests and the reference gate.
enum class VisionIdentity { legacy, axial_v1 };

class VisionAdapter final : public vision::VisionAdapter {
public:
    // weights_digest: identity of the checkpoint's vision tensors (part of the cache key; required).
    VisionAdapter(VisionSpec spec, std::string weights_digest, VisionRecipe recipe = {});
    VisionAdapter(VisionSpec spec, std::string weights_digest, VisionIdentity identity,
                  unsigned max_tokens = kVisionDefaultMaxTokens);

    vision::EncoderConfig config() const override;
    std::string identity() const override;
    vision::ResizeRule resize() const override;
    vision::PatchGrid patches(const vision::Image&) const override;
    void bind(vision::WeightSource&, vision::EncoderWeights&) override;
    vision::PositionInputs positions(const vision::EncoderConfig&, const vision::EncoderWeights&,
                                     const vision::PatchGrid&) const override;
    void project(vision::EncoderOutput&, const vision::PatchGrid&, vision::DenseOps&, vision::ProjectedRows&) override;
    vision::SpanLayout layout(const vision::PatchGrid&) const override;
    std::vector<std::pair<vision::Tensor, vision::Tensor>> projector_linears() const override;
    unsigned max_encoder_rows() const override { return spec_.merge * spec_.merge * max_tokens_ + (identity_ == VisionIdentity::axial_v1 || legacy_.class_row ? 1U : 0U); }

    vision::ColorPolicy color_policy() const override;
    [[nodiscard]] vision::Size scaled(int width, int height) const;
    [[nodiscard]] vision::Size canvas(int width, int height) const;
    [[nodiscard]] unsigned tokens(int width, int height) const;
    [[nodiscard]] const VisionRecipe& recipe() const noexcept { return legacy_; }
    // Merged grid (token rows, token columns) of a patch grid made by patches().
    struct TokenGrid {
        unsigned rows = 0, cols = 0;
    };
    TokenGrid token_grid(const vision::PatchGrid&) const;

private:
    [[nodiscard]] vision::PatchGrid legacy_patches(const vision::Image&) const;
    [[nodiscard]] vision::PatchGrid image_grid(const vision::PatchGrid&) const;
    VisionSpec spec_;
    VisionIdentity identity_;
    VisionRecipe legacy_;
    std::string digest_;
    unsigned max_tokens_;
    std::vector<float> class_row_;  // class_embedding + cls_position_embedding
    vision::Tensor proj_in_, proj_out_;

};

}  // namespace strix::inkling2
