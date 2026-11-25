#pragma once
// GLM-5.3 Flash image tower on the shared vision stack. The tower ships as a separate GGUF projector file
// (`general.architecture = clip`, `clip.projector_type = glm5v`): a pre-norm ViT with RMSNorm, per-head q/k norm, 2D
// rotary positions and a clamped SwiGLU MLP over 2x2-ordered patches, then a projector (2x2 merge linear, linear,
// LayerNorm, GELU, clamped SwiGLU MLP) into the language width. This file declares only what is specific to the
// family: the file contract, the tensor names, the canvas rule, the projector, the span and the identity that keys
// caches. Every tower computation runs in the shared encoder (strix/vision/encoder.hpp, host reference: the device
// towers refuse per-head q/k norm) and the shared preprocessing (strix/vision/preprocess.hpp).
//
// Provenance: the file contract (metadata keys, tensor names) is that of the format's writer, see the GLM-5.3 Flash
// image tower row of docs/reference/sources-ledger.md. No source is copied.
#include "strix/gguf_reader.hpp"
#include "strix/model_data/glm_5_3_flash.hpp"
#include "strix/vision/adapter.hpp"
#include "strix/vision/preprocess.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strix::glm {

// Vocabulary ids of the language model's image tokens (tokenizer.json): an image is <|begin_of_image|>, one
// <|image|> row per projected row, <|end_of_image|>.
inline constexpr std::uint32_t kImageBeginToken = 154830;
inline constexpr std::uint32_t kImageEndToken = 154831;
inline constexpr std::uint32_t kImageToken = 154854;
inline constexpr auto kLanguageHidden = static_cast<std::uint32_t>(model_data::glm_5_3_flash::dims::hidden);
inline constexpr auto kLanguageVocab = static_cast<std::uint32_t>(model_data::glm_5_3_flash::dims::vocab);
static_assert(kImageBeginToken < kLanguageVocab && kImageEndToken < kLanguageVocab && kImageToken < kLanguageVocab);

// Everything read from the projector file (metadata and tensor shapes).
struct VisionConfig {
    std::uint32_t layers = 0, width = 0, heads = 0, mlp_width = 0;
    std::uint32_t patch = 0, merge = 0;
    std::uint32_t merged_width = 0;      // outputs of the 2x2 merge linear
    std::uint32_t projection = 0;        // language width
    std::uint32_t projector_mlp = 0;     // inner width of the projector's SwiGLU MLP
    std::uint64_t min_pixels = 0, max_pixels = 0;  // canvas area bounds, single frame
    float eps = 0.0F;                    // tower norms (RMS), per-head q/k norm included
    float swiglu_limit = 0.0F;           // 0: no clamp
    std::array<float, 3> mean{}, stddev{};
    bool fused_qkv = false;              // one qkv tensor per block, else q, k and v
    // A bias some block carries; a block without it gets zeros (the shared tower has one flag per kind).
    bool qkv_bias = false, out_bias = false, mlp_bias = false;
    bool embed_norm = false, final_norm = false;  // `v.pre_ln`, `v.post_ln`

    [[nodiscard]] std::uint32_t head_dim() const noexcept { return heads ? width / heads : 0U; }
    [[nodiscard]] std::uint32_t cell() const noexcept { return patch * merge; }  // pixels per image row side
    [[nodiscard]] std::uint64_t cell_pixels() const noexcept { return std::uint64_t{cell()} * cell(); }
    [[nodiscard]] std::uint32_t max_image_tokens() const noexcept {
        return cell() ? static_cast<std::uint32_t>(max_pixels / cell_pixels()) : 0U;
    }
};

// Reader limits the projector file needs (the mean and standard deviation arrays are kept).
[[nodiscard]] gguf::Limits vision_gguf_limits();
// Strict: architecture and projector type, every metadata key with its type, value ranges and the tensor inventory
// (names, shapes, FP32 / FP16 / BF16 storage; no unknown tensor). Throws std::invalid_argument naming the first
// violation.
[[nodiscard]] VisionConfig validate_vision(const gguf::FileInfo& info);
// Byte budget of a projector file, judged from its header after validate_vision (so before any tensor byte is read or
// hashed): every tensor lies inside the file, the tensors sum to at most kMaxProjectorTensorBytes and the file ends
// within kMaxProjectorTrailingBytes of its last tensor (no padded or sparse tail). The tensor cap is 4 GiB: one
// billion parameters at FP32, a ceiling above any tower the validated schema is meant for, not a measured size; the
// generic host budget of GgufWeights (also 4 GiB) then applies to the whole file. Returns the file size. Throws
// std::invalid_argument.
inline constexpr std::uint64_t kMaxProjectorTensorBytes = std::uint64_t{4} << 30U;
inline constexpr std::uint64_t kMaxProjectorTrailingBytes = std::uint64_t{1} << 20U;
// The file length is also capped independently of the declared offsets: header (metadata and tensor table) at most
// kMaxProjectorHeaderBytes, and every byte not in a tensor accounted for by the header, one alignment pad per tensor
// (at most kMaxProjectorAlignment) or the trailing allowance, so no gap before or between tensors can grow the file.
inline constexpr std::uint64_t kMaxProjectorHeaderBytes = std::uint64_t{64} << 20U;
inline constexpr std::uint64_t kMaxProjectorAlignment = 4096;
std::uint64_t check_projector_bytes(const gguf::FileInfo& info);
// The tower must belong to the language model: projection width and a token id inside the vocabulary.
void require_vision_matches_language(const VisionConfig& vision, std::uint32_t hidden, std::uint32_t vocab,
                                     std::uint32_t image_token_id);

// What the file does not state. Each field is part of the adapter identity.
struct VisionRecipe {
    vision::Resampler resampler{vision::Filter::bicubic, true};
    std::uint8_t pad_fill = 0;            // canvas right of and below the image
    float rope_theta = 10000.0F;
    float post_norm_eps = 1e-5F;          // LayerNorm of the projector
    // Diagnostic model of the file format's reference implementation, which rounds the inputs of its two
    // convolution-shaped linears (patch embedding, 2x2 merge) to FP16. Off: every operand stays FP32.
    bool conv_inputs_f16 = false;
    [[nodiscard]] std::string id() const;
};

class VisionAdapter final : public vision::VisionAdapter {
public:
    // weights_sha256_hex: SHA-256 of the projector file (lower-case hex). image_token_id: the vocabulary id every
    // image row counts as.
    VisionAdapter(VisionConfig config, std::string weights_sha256_hex, std::uint32_t image_token_id,
                  VisionRecipe recipe = {});

    vision::EncoderConfig config() const override;
    std::string identity() const override;
    vision::ResizeRule resize() const override;
    vision::PatchGrid patches(const vision::Image& image) const override;
    void bind(vision::WeightSource& source, vision::EncoderWeights& weights) override;
    // Rotary angles as the reference forms them: per axis a FP32 product chain pos * s * s ..., s = theta^(-2 / (D / 2)).
    vision::PositionInputs positions(const vision::EncoderConfig& c, const vision::EncoderWeights& w,
                                     const vision::PatchGrid& g) const override;
    void project(vision::EncoderOutput& hidden, const vision::PatchGrid& grid, vision::DenseOps& dense,
                 vision::ProjectedRows& out) override;
    // The rows project() stores, before their BF16 rounding: [image rows][projection] FP32 (gates and comparisons).
    [[nodiscard]] std::vector<float> project_f32(const vision::EncoderOutput& hidden, const vision::PatchGrid& grid,
                                                 vision::DenseOps& dense) const;
    vision::SpanLayout layout(const vision::PatchGrid& grid) const override;
    // Image rows of a width x height image: (canvas / cell) squared per axis, as layout() will state them.
    std::uint64_t rows_for_size(int width, int height) const override;
    std::vector<std::pair<vision::Tensor, vision::Tensor>> projector_linears() const override;
    unsigned max_encoder_rows() const override;

    // The canvas and the image inside it. Each canvas side is the image side rounded up to a multiple of the cell
    // (patch x merge); an image under min_pixels is scaled up first; a canvas over max_pixels is replaced by the
    // largest one of the image's aspect ratio that fits (search over the content height). The image is scaled to fit
    // the canvas (never up once it has min_pixels) and sits at the top left. Throws std::invalid_argument.
    struct Canvas {
        vision::Size canvas, content;
    };
    [[nodiscard]] Canvas target(int width, int height) const;
    [[nodiscard]] const VisionConfig& model() const noexcept { return config_; }
    [[nodiscard]] const VisionRecipe& recipe() const noexcept { return recipe_; }
    [[nodiscard]] std::uint32_t image_token_id() const noexcept { return image_token_id_; }

private:
    [[nodiscard]] vision::MergedGrid checked_grid(const vision::PatchGrid& grid) const;
    VisionConfig config_;
    std::string sha_;
    std::uint32_t image_token_id_;
    VisionRecipe recipe_;
    vision::Tensor merge_w_, merge_b_, fc_w_, post_norm_w_, post_norm_b_;
    vision::Tensor up_w_, up_b_, gate_w_, gate_b_, down_w_, down_b_;
};

}  // namespace strix::glm
