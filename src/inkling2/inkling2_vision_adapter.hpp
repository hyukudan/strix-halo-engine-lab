#pragma once
// Inkling 2 Mini image tower binding on the shared vision runtime (include/strix/vision). Model-specific content
// only: the tensor names, the pad-to-40 geometry, the patch order, the folds that map this checkpoint onto the
// shared tower graph, and the span the language model sees. The tower and the projector arithmetic are shared.
//
// Checkpoint facts (config.json vision_config, safetensors header): ViT width 1024, 23 blocks, 16 heads, MLP 4096,
// LayerNorm eps 1e-5, patch 20 x 20 over 2 frames (an image fills both), a 32 x 32 learned position table, a class
// row, ln_pre, LayerScale ls_1 / ls_2, no final norm; projector proj_in [4096][4096], proj_out [2304][4096], no bias.
// One language-model row per 2 x 2 patch cell = one 40 x 40 pixel cell.
//
// Mapping onto the shared graph (no private kernels):
//   ln_pre       -> EncoderConfig::embed_norm
//   LayerScale   -> folded: out_w' = diag(ls_1) out_w, out_b' = ls_1 * out_b; mlp2 likewise with ls_2
//   q / k / v    -> concatenated into the fused [3 W][W] qkv
//   class row    -> one extra tower row of zero pixels (no patch bias: its embedding is exactly 0) whose addend is
//                   class_embedding + cls_position_embedding; it is the LAST row and is dropped before the merge
//                   (the tower is permutation-equivariant: positions enter only through the addend)
//   2 x 2 merge  -> MergeProjector (channel-major unshuffle); a slot-major checkpoint layout is a column permutation
//                   of proj_in, done once at bind()
#include "strix/vision/adapter.hpp"
#include "strix/vision/preprocess.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace strix::inkling2 {

// Geometry of the checkpoint (config.json vision_config); the defaults are revision ed40e7d.
struct VisionConfig {
    unsigned hidden_size = 1024, num_hidden_layers = 23, num_attention_heads = 16, mlp_width = 4096;
    float layer_norm_eps = 1e-5F;
    unsigned native_patch_size = 20, temporal_patch_size = 2, adapter_merge_size = 2, posemb_grid_size = 32;
    unsigned projection_hidden_dim = 4096, text_hidden_size = 2304;
    [[nodiscard]] unsigned cell() const noexcept { return native_patch_size * adapter_merge_size; }  // 40 px
};

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

class VisionAdapter final : public vision::VisionAdapter {
public:
    VisionAdapter(VisionConfig config, std::string weights_sha256_hex, VisionRecipe recipe = {});

    vision::EncoderConfig config() const override;
    std::string identity() const override;
    vision::ResizeRule resize() const override;
    vision::PatchGrid patches(const vision::Image& image) const override;
    void bind(vision::WeightSource& source, vision::EncoderWeights& weights) override;
    vision::PositionInputs positions(const vision::EncoderConfig& c, const vision::EncoderWeights& w,
                                     const vision::PatchGrid& g) const override;
    void project(vision::EncoderOutput& hidden, const vision::PatchGrid& grid, vision::DenseOps& dense,
                 vision::ProjectedRows& out) override;
    vision::SpanLayout layout(const vision::PatchGrid& grid) const override;
    std::vector<std::pair<vision::Tensor, vision::Tensor>> projector_linears() const override;
    unsigned max_encoder_rows() const override;

    // Canvas of an image: its own size when inside max_tokens cells, else scaled down keeping the aspect ratio;
    // then each side rounded UP to a multiple of the 40 px cell (the margin is padded, not resampled).
    [[nodiscard]] vision::Size scaled(int width, int height) const;
    [[nodiscard]] vision::Size canvas(int width, int height) const;
    [[nodiscard]] unsigned tokens(int width, int height) const;  // LM rows of an image
    [[nodiscard]] const VisionRecipe& recipe() const noexcept { return recipe_; }

private:
    [[nodiscard]] vision::PatchGrid image_grid(const vision::PatchGrid& tower_grid) const;
    VisionConfig config_;
    std::string sha_;
    VisionRecipe recipe_;
    vision::Tensor cls_embed_, cls_pos_, proj_in_, proj_out_;
};

}  // namespace strix::inkling2
