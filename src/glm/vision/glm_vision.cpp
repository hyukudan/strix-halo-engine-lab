#include "glm_vision.hpp"

#include "strix/vision/host_math.hpp"
#include "strix/vision/image.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>

namespace strix::glm {
namespace {

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument("glm vision: " + what); }
void require(bool ok, const std::string& what) {
    if (!ok) bad(what);
}

const gguf::MetadataEntry& meta(const gguf::FileInfo& info, const std::string& key) {
    const auto* entry = info.find_metadata(key);
    require(entry != nullptr, "missing metadata " + key);
    return *entry;
}
std::uint32_t meta_u32(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.unsigned_value.has_value(), "metadata " + key + " must be unsigned");
    require(*e.unsigned_value <= 0xffffffffULL, "metadata " + key + " is out of range");
    return static_cast<std::uint32_t>(*e.unsigned_value);
}
float meta_f32(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.floating_value.has_value(), "metadata " + key + " must be a float");
    return static_cast<float>(*e.floating_value);
}
bool meta_flag(const gguf::FileInfo& info, const std::string& key) {  // an absent flag is false
    const auto* e = info.find_metadata(key);
    if (e == nullptr) return false;
    require(e->boolean_value.has_value(), "metadata " + key + " must be a boolean");
    return *e->boolean_value;
}
std::array<float, 3> meta_rgb(const gguf::FileInfo& info, const std::string& key) {
    const auto& e = meta(info, key);
    require(e.array_length.has_value() && *e.array_length >= 3 && e.floating_array.size() == *e.array_length,
            "metadata " + key + " must be a float array of at least 3 values");
    return {static_cast<float>(e.floating_array[0]), static_cast<float>(e.floating_array[1]),
            static_cast<float>(e.floating_array[2])};
}

using Dims = std::vector<std::uint64_t>;
struct Expected {
    Dims dims;
    bool required = true;
};

const char* filter_name(vision::Filter f) {
    switch (f) {
        case vision::Filter::bilinear: return "bilinear";
        case vision::Filter::bicubic: return "bicubic";
        case vision::Filter::lanczos: return "lanczos";
    }
    return "?";
}

void copy_values(const vision::Tensor& t, float* dst) {
    for (std::size_t i = 0; i < t.size(); ++i) dst[i] = t.at(i);
}
void expect_shape(const vision::Tensor& t, std::size_t rows, std::size_t cols, const char* name) {
    if (!t || t.rows != rows || t.cols != cols) bad(std::string("tensor shape of ") + name + " differs from the configuration");
}

}  // namespace

gguf::Limits vision_gguf_limits() {
    gguf::Limits limits;
    limits.max_retained_array_elements = 16;
    return limits;
}

VisionConfig validate_vision(const gguf::FileInfo& info) {
    const auto text = [&](const std::string& key) {
        const auto& e = meta(info, key);
        require(e.string_value.has_value(), "metadata " + key + " must be a string");
        return *e.string_value;
    };
    require(text("general.architecture") == "clip", "general.architecture must be clip");
    require(text("clip.projector_type") == "glm5v", "clip.projector_type must be glm5v");
    require(meta_flag(info, "clip.has_vision_encoder"), "the file has no vision encoder");
    require(!meta_flag(info, "clip.has_audio_encoder"), "a file with an audio encoder is not supported");
    require(meta_flag(info, "clip.use_silu") && !meta_flag(info, "clip.use_gelu"), "the MLP activation must be SiLU (clip.use_silu)");

    VisionConfig c;
    c.layers = meta_u32(info, "clip.vision.block_count");
    c.width = meta_u32(info, "clip.vision.embedding_length");
    c.heads = meta_u32(info, "clip.vision.attention.head_count");
    c.mlp_width = meta_u32(info, "clip.vision.feed_forward_length");
    c.projection = meta_u32(info, "clip.vision.projection_dim");
    c.patch = meta_u32(info, "clip.vision.patch_size");
    c.merge = info.find_metadata("clip.vision.spatial_merge_size") ? meta_u32(info, "clip.vision.spatial_merge_size") : 2U;
    c.min_pixels = meta_u32(info, "clip.vision.image_min_pixels");
    c.max_pixels = meta_u32(info, "clip.vision.image_max_pixels");
    c.eps = meta_f32(info, "clip.vision.attention.layer_norm_epsilon");
    c.swiglu_limit = meta_f32(info, "clip.vision.swiglu_clamp");
    c.mean = meta_rgb(info, "clip.vision.image_mean");
    c.stddev = meta_rgb(info, "clip.vision.image_std");

    require(c.layers > 0 && c.layers <= 256, "block_count is out of range");
    require(c.heads > 0 && c.width > 0 && c.width <= 16384 && c.width % c.heads == 0, "width must be a multiple of the head count");
    require(c.head_dim() % 4U == 0, "head dimension must be a multiple of 4 (two rotary axes)");
    require(c.mlp_width > 0 && c.mlp_width <= 65536, "feed_forward_length is out of range");
    require(c.projection > 0 && c.projection <= 65536, "projection_dim is out of range");
    require(c.patch > 0 && c.patch <= 64, "patch_size is out of range");
    require(c.merge > 0 && c.merge <= 8, "spatial_merge_size is out of range");
    require(c.min_pixels >= c.cell_pixels() && c.min_pixels <= c.max_pixels, "image_min_pixels is outside the token range");
    require(c.max_image_tokens() <= 16384, "image_max_pixels is out of range");
    require(std::isfinite(c.eps) && c.eps > 0.0F && c.eps < 1e-2F, "norm epsilon is out of range");
    require(std::isfinite(c.swiglu_limit) && c.swiglu_limit >= 0.0F, "swiglu_clamp is out of range");
    for (std::size_t i = 0; i < 3; ++i)
        require(std::isfinite(c.mean[i]) && std::isfinite(c.stddev[i]) && c.stddev[i] > 0.0F, "image mean / std is out of range");

    std::map<std::string, const gguf::TensorInfo*> have;
    for (const auto& t : info.tensors) {
        require(have.emplace(t.name, &t).second, "duplicate tensor " + t.name);
        require(t.ggml_type == 0 || t.ggml_type == 1 || t.ggml_type == 30,
                "tensor " + t.name + " must be stored as F32, F16 or BF16 (a quantized projector file is not supported)");
    }
    const auto dims_of = [&](const std::string& name) -> const Dims& {
        const auto it = have.find(name);
        require(it != have.end(), "missing tensor " + name);
        return it->second->dimensions;
    };
    const auto has = [&](const std::string& name) { return have.find(name) != have.end(); };

    // Widths the metadata does not state come from the projector tensors.
    const Dims& merge_dims = dims_of("mm.patch_merger.weight");
    require(merge_dims.size() == 4, "mm.patch_merger.weight must have 4 dimensions");
    require(merge_dims[3] > 0 && merge_dims[3] <= 65536, "mm.patch_merger.weight output width is out of range");
    c.merged_width = static_cast<std::uint32_t>(merge_dims[3]);
    const Dims& up_dims = dims_of("mm.up.weight");
    require(up_dims.size() == 2 && up_dims[1] > 0 && up_dims[1] <= 262144, "mm.up.weight shape is out of range");
    c.projector_mlp = static_cast<std::uint32_t>(up_dims[1]);

    const std::uint64_t W = c.width, M = c.mlp_width, D = c.head_dim(), P = c.projection, F = c.projector_mlp;
    std::map<std::string, Expected> expected;
    const auto add = [&](const std::string& name, Dims dims, bool required = true) { expected[name] = {std::move(dims), required}; };
    add("v.patch_embd.weight", {c.patch, c.patch, 3, W});
    add("v.patch_embd.weight.1", {c.patch, c.patch, 3, W});
    add("v.patch_embd.bias", {W});
    for (const char* norm : {"v.pre_ln", "v.post_ln"}) {
        add(std::string(norm) + ".weight", {W}, false);
        add(std::string(norm) + ".bias", {W}, false);
    }
    c.embed_norm = has("v.pre_ln.weight");
    c.final_norm = has("v.post_ln.weight");
    require(c.embed_norm || !has("v.pre_ln.bias"), "v.pre_ln.bias without its weight");
    require(c.final_norm || !has("v.post_ln.bias"), "v.post_ln.bias without its weight");
    c.fused_qkv = has("v.blk.0.attn_qkv.weight");
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        add(p + "ln1.weight", {W});
        add(p + "ln1.bias", {W}, false);
        add(p + "ln2.weight", {W});
        add(p + "ln2.bias", {W}, false);
        if (c.fused_qkv) {
            add(p + "attn_qkv.weight", {W, 3U * W});
            add(p + "attn_qkv.bias", {3U * W}, false);
            c.qkv_bias = c.qkv_bias || has(p + "attn_qkv.bias");
        } else {
            for (const char* part : {"attn_q", "attn_k", "attn_v"}) {
                add(p + part + ".weight", {W, W});
                add(p + part + ".bias", {W}, false);
                c.qkv_bias = c.qkv_bias || has(p + part + ".bias");
            }
        }
        add(p + "attn_q_norm.weight", {D});
        add(p + "attn_k_norm.weight", {D});
        add(p + "attn_out.weight", {W, W});
        add(p + "attn_out.bias", {W}, false);
        c.out_bias = c.out_bias || has(p + "attn_out.bias");
        add(p + "ffn_up.weight", {W, M});
        add(p + "ffn_gate.weight", {W, M});
        add(p + "ffn_down.weight", {M, W});
        for (const char* part : {"ffn_up", "ffn_gate"}) {
            add(p + part + ".bias", {M}, false);
            c.mlp_bias = c.mlp_bias || has(p + part + ".bias");
        }
        add(p + "ffn_down.bias", {W}, false);
        c.mlp_bias = c.mlp_bias || has(p + "ffn_down.bias");
    }
    add("mm.patch_merger.weight", {c.merge, c.merge, W, c.merged_width});
    add("mm.patch_merger.bias", {c.merged_width});
    add("mm.model.fc.weight", {c.merged_width, P});
    add("mm.post_norm.weight", {P});
    add("mm.post_norm.bias", {P}, false);
    add("mm.up.weight", {P, F});
    add("mm.up.bias", {F}, false);
    add("mm.gate.weight", {P, F});
    add("mm.gate.bias", {F}, false);
    add("mm.down.weight", {F, P});
    add("mm.down.bias", {P}, false);

    for (const auto& [name, e] : expected) {
        if (!has(name)) {
            require(!e.required, "missing tensor " + name);
            continue;
        }
        require(dims_of(name) == e.dims, "tensor " + name + " has an unexpected shape");
    }
    for (const auto& t : info.tensors) require(expected.find(t.name) != expected.end(), "unexpected tensor " + t.name);
    return c;
}

std::uint64_t check_projector_bytes(const gguf::FileInfo& info) {
    require(info.data_offset <= info.file_size, "the projector file is shorter than its header");
    std::uint64_t total = 0, last = info.data_offset;
    for (const auto& t : info.tensors) {
        require(t.absolute_end <= info.file_size && t.absolute_offset <= t.absolute_end, "tensor " + t.name + " lies outside the file");
        require(t.byte_size <= kMaxProjectorTensorBytes - total, "the projector tensors exceed " + std::to_string(kMaxProjectorTensorBytes) + " bytes");
        total += t.byte_size;
        last = std::max(last, t.absolute_end);
    }
    require(info.file_size - last <= kMaxProjectorTrailingBytes, "the projector file has bytes after its last tensor");
    // The file's length is bounded on its own, whatever offsets the header declares: a header of at most
    // kMaxProjectorHeaderBytes, the tensors, at most one alignment pad (capped) per tensor and the trailing allowance.
    // A sparse gap before or between tensors is refused here, before the host budget takes the file's length.
    require(info.data_offset <= kMaxProjectorHeaderBytes, "the projector header exceeds " + std::to_string(kMaxProjectorHeaderBytes) + " bytes");
    const std::uint64_t pad = std::min<std::uint64_t>(std::max<std::uint64_t>(info.alignment, 1U), kMaxProjectorAlignment);
    const std::uint64_t slack = info.data_offset + kMaxProjectorTrailingBytes + pad * info.tensors.size();
    require(info.file_size - total <= slack, "the projector file has unaccounted bytes (a gap before or between tensors)");
    return info.file_size;
}

void require_vision_matches_language(const VisionConfig& vision, std::uint32_t hidden, std::uint32_t vocab,
                                     std::uint32_t image_token_id) {
    require(vision.projection == hidden, "projection width differs from the language hidden size");
    require(image_token_id < vocab, "image token id is outside the vocabulary");
}

std::string VisionRecipe::id() const {
    const auto bits = [](float v) { return std::bit_cast<std::uint32_t>(v); };
    // "c1": the canvas rule of VisionAdapter::target (cell-multiple canvas, image at the top left).
    char text[96];
    std::snprintf(text, sizeof text, "%s%s-f%d-t%08x-e%08x-h%d-c1", filter_name(resampler.filter), resampler.antialias ? "-aa" : "",
                  static_cast<int>(pad_fill), bits(rope_theta), bits(post_norm_eps), conv_inputs_f16 ? 1 : 0);
    return text;
}

VisionAdapter::VisionAdapter(VisionConfig config, std::string weights_sha256_hex, std::uint32_t image_token_id,
                             VisionRecipe recipe)
    : config_(std::move(config)), sha_(std::move(weights_sha256_hex)), image_token_id_(image_token_id), recipe_(recipe) {
    if (sha_.size() != 64) bad("the weights digest must be 64 hex characters");
    if (config_.layers == 0 || config_.heads == 0 || config_.cell() == 0 || config_.max_image_tokens() == 0 ||
        config_.projection == 0 || config_.merged_width == 0 || config_.projector_mlp == 0 ||
        config_.min_pixels > config_.max_pixels)
        bad("incomplete configuration");
    if (!std::isfinite(recipe_.rope_theta) || !(recipe_.rope_theta > 1.0F)) bad("recipe rope_theta");
    if (!std::isfinite(recipe_.post_norm_eps) || !(recipe_.post_norm_eps > 0.0F)) bad("recipe post_norm_eps");
}

vision::EncoderConfig VisionAdapter::config() const {
    vision::EncoderConfig c;
    c.layers = config_.layers;
    c.width = config_.width;
    c.heads = config_.heads;
    c.mlp_width = config_.mlp_width;
    c.norm = vision::EncoderConfig::Norm::rmsnorm;
    c.eps = config_.eps;
    c.mlp = vision::EncoderConfig::Mlp::swiglu;
    c.swiglu_limit = config_.swiglu_limit;
    c.swiglu_gate_first = true;  // bind() lays the fused first MLP linear out as [gate | up]
    c.qkv_bias = config_.qkv_bias;
    c.out_bias = config_.out_bias;
    c.mlp_bias = config_.mlp_bias;
    c.patch_bias = true;
    c.qk_norm = true;
    c.pos = vision::EncoderConfig::Pos::rope2d;
    c.rope_theta = recipe_.rope_theta;
    c.rope_pairing = vision::EncoderConfig::RopePairing::half;
    c.patch = config_.patch;
    c.temporal = 2;  // two patch kernels over the one frame of an image
    c.channels = 3;
    c.embed_norm = config_.embed_norm;
    c.final_norm = config_.final_norm;
    return c;
}

std::string VisionAdapter::identity() const {
    return "glm5-vision/" + sha_ + "/t" + std::to_string(image_token_id_) + "/" + recipe_.id();
}

vision::ResizeRule VisionAdapter::resize() const {
    vision::ResizeRule r;
    r.multiple = config_.cell();
    r.min_pixels = config_.min_pixels;
    r.max_pixels = config_.max_pixels;
    r.max_rows = config_.max_image_tokens();
    r.max_aspect = 0.0F;
    return r;
}

unsigned VisionAdapter::max_encoder_rows() const {
    const std::uint64_t rows = std::uint64_t{config_.max_image_tokens()} * config_.merge * config_.merge;
    return rows > 0x7FFFFFFFULL ? 0U : static_cast<unsigned>(rows);
}

vision::MergedGrid VisionAdapter::checked_grid(const vision::PatchGrid& grid) const {
    return vision::checked_merged_grid(grid, config_.merge, config_.max_image_tokens());
}

VisionAdapter::Canvas VisionAdapter::target(int width, int height) const {
    if (width <= 0 || height <= 0) bad("empty image");
    // Integers where the reference has integers, IEEE doubles (sqrt, ceil, floor, true division) where it has floats.
    const std::int64_t factor = config_.cell();
    const auto min_px = static_cast<std::int64_t>(config_.min_pixels), max_px = static_cast<std::int64_t>(config_.max_pixels);
    const std::int64_t h = height, w = width;
    const auto align = [factor](std::int64_t v) { return (v + factor - 1) / factor * factor; };
    const auto whole = [](double v) {
        if (!std::isfinite(v) || v < 0.0 || v >= 0x1p62) bad("resize size is out of range");
        return static_cast<std::int64_t>(v);
    };

    std::int64_t canvas_h = align(h), canvas_w = align(w);
    if (canvas_h * canvas_w < min_px) {
        const double scale = std::sqrt(static_cast<double>(min_px) / static_cast<double>(h * w));
        canvas_h = align(std::max<std::int64_t>(1, whole(std::ceil(static_cast<double>(h) * scale))));
        canvas_w = align(std::max<std::int64_t>(1, whole(std::ceil(static_cast<double>(w) * scale))));
    }
    if (canvas_h > max_px || canvas_w > max_px || canvas_h * canvas_w > max_px) {
        // The largest content height whose aligned canvas fits the budget; the width follows the aspect ratio.
        std::int64_t lo = 1, hi = h, best_h = factor, best_w = factor;
        while (lo <= hi) {
            const std::int64_t ch = lo + (hi - lo) / 2;
            const std::int64_t cw = std::max<std::int64_t>(1, w * ch / h);
            const std::int64_t ah = align(ch), aw = align(cw);
            if (ah <= max_px && aw <= max_px && ah * aw <= max_px) {
                best_h = ah;
                best_w = aw;
                lo = ch + 1;
            } else {
                hi = ch - 1;
            }
        }
        canvas_h = best_h;
        canvas_w = best_w;
    }
    if (canvas_h * canvas_w > max_px || canvas_h > std::numeric_limits<int>::max() || canvas_w > std::numeric_limits<int>::max())
        bad("the image does not fit " + std::to_string(config_.max_image_tokens()) + " image rows");

    double scale = std::min(static_cast<double>(canvas_h) / static_cast<double>(h), static_cast<double>(canvas_w) / static_cast<double>(w));
    if (h * w >= min_px) scale = std::min(1.0, scale);
    const std::int64_t content_h = std::max<std::int64_t>(1, std::min(canvas_h, whole(std::floor(static_cast<double>(h) * scale))));
    const std::int64_t content_w = std::max<std::int64_t>(1, std::min(canvas_w, whole(std::floor(static_cast<double>(w) * scale))));
    return {{static_cast<int>(canvas_w), static_cast<int>(canvas_h)}, {static_cast<int>(content_w), static_cast<int>(content_h)}};
}

std::uint64_t VisionAdapter::rows_for_size(int width, int height) const {
    const Canvas t = target(width, height);
    return (static_cast<std::uint64_t>(t.canvas.height) / config_.cell()) * (static_cast<std::uint64_t>(t.canvas.width) / config_.cell());
}

vision::PatchGrid VisionAdapter::patches(const vision::Image& image) const {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != image.pixels() * 3U) bad("empty image");
    const Canvas t = target(image.width, image.height);
    // Before any resample or allocation: the grid must fit the 16-bit patch coordinates of the shared representation.
    vision::check_grid_axes(static_cast<std::uint64_t>(t.canvas.height) / config_.patch,
                            static_cast<std::uint64_t>(t.canvas.width) / config_.patch);
    // Memory cost of this call from the canvas size, before it is built: padded RGB8 (3 B/px), the resized content
    // (<= 3 B/px), FP32 pixels (12 B/px) and temporal patches (2 frames x 3 channels x 4 B = 24 B/px). The config
    // bounds the canvas to 16384 rows (about 540 MB at 42 B/px); 2 GiB is a ceiling no valid config reaches.
    constexpr std::uint64_t kBytesPerCanvasPixel = 3 + 3 + 12 + 24, kMaxPatchBytes = std::uint64_t{2} << 30U;
    if (static_cast<std::uint64_t>(t.canvas.height) * static_cast<std::uint64_t>(t.canvas.width) > kMaxPatchBytes / kBytesPerCanvasPixel)
        bad("the image needs more than " + std::to_string(kMaxPatchBytes) + " bytes of working memory");
    vision::Image resized;
    const vision::Image* content = &image;
    if (t.content.width != image.width || t.content.height != image.height) {
        resized = vision::resize(image, t.content.width, t.content.height, recipe_.resampler);
        content = &resized;
    }
    vision::Image canvas;
    canvas.width = t.canvas.width;
    canvas.height = t.canvas.height;
    canvas.rgb.assign(canvas.pixels() * 3U, recipe_.pad_fill);
    const std::size_t line = static_cast<std::size_t>(content->width) * 3U;
    for (int y = 0; y < content->height; ++y)
        std::copy_n(content->rgb.data() + static_cast<std::size_t>(y) * line, line,
                    canvas.rgb.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(canvas.width) * 3U);

    const std::vector<float> pixels = vision::normalize_div255(canvas, config_.mean, config_.stddev, false);
    const float* frame = pixels.data();
    vision::PatchSpec spec;
    spec.patch = static_cast<int>(config_.patch);
    spec.temporal = 2;
    spec.merge = static_cast<int>(config_.merge);  // rows in merge x merge block order, as the projector groups them
    vision::Patches p = vision::patchify(std::span<const float* const>(&frame, 1), canvas.height, canvas.width, spec);

    vision::PatchGrid g;
    g.grid_h = static_cast<unsigned>(p.grid_h);
    g.grid_w = static_cast<unsigned>(p.grid_w);
    g.rows = g.grid_h * g.grid_w;
    g.patch_dim = static_cast<unsigned>(p.feature);
    g.data = std::move(p.data);
    if (recipe_.conv_inputs_f16)
        for (float& v : g.data) v = vision::round_to_f16(v);
    g.y.resize(g.rows);
    g.x.resize(g.rows);
    const unsigned m = config_.merge, mw = g.grid_w / m;
    for (unsigned r = 0; r < g.rows; ++r) {
        const unsigned block = r / (m * m), slot = r % (m * m);
        g.y[r] = static_cast<std::uint16_t>(block / mw * m + slot / m);
        g.x[r] = static_cast<std::uint16_t>(block % mw * m + slot % m);
    }
    (void)checked_grid(g);
    return g;
}

vision::PositionInputs VisionAdapter::positions(const vision::EncoderConfig& c, const vision::EncoderWeights&,
                                                const vision::PatchGrid& g) const {
    vision::validate(c);
    if (g.y.size() != g.rows || g.x.size() != g.rows) bad("patch coordinates");
    const unsigned half = c.head_dim() / 2U, per_axis = half / 2U;
    const float step = std::pow(c.rope_theta, -2.0F / static_cast<float>(half));
    vision::PositionInputs p;
    p.angles.resize(std::size_t{g.rows} * half);
    for (unsigned r = 0; r < g.rows; ++r) {
        float ty = static_cast<float>(g.y[r]), tx = static_cast<float>(g.x[r]);
        float* row = &p.angles[std::size_t{r} * half];
        for (unsigned f = 0; f < per_axis; ++f) {
            row[f] = ty;
            row[per_axis + f] = tx;
            ty *= step;
            tx *= step;
        }
    }
    return p;
}

void VisionAdapter::bind(vision::WeightSource& s, vision::EncoderWeights& w) {
    w = {};
    const std::size_t W = config_.width, M = config_.mlp_width, D = config_.head_dim();
    const std::size_t pp = std::size_t{config_.patch} * config_.patch;
    const auto own = [&w](std::size_t rows, std::size_t cols) {
        auto& v = w.owned.emplace_back(rows * cols, 0.0F);
        return vision::Tensor{v.data(), vision::Tensor::Type::f32, rows, cols};
    };
    const auto mut = [](const vision::Tensor& t) { return const_cast<float*>(static_cast<const float*>(t.data)); };
    const auto optional = [&s](const std::string& name) { return s.has(name) ? s.tensor(name) : vision::Tensor{}; };

    {  // the two patch kernels as one [W][channel][frame][y][x] linear
        const vision::Tensor k0 = s.tensor("v.patch_embd.weight"), k1 = s.tensor("v.patch_embd.weight.1");
        expect_shape(k0, W, 3U * pp, "v.patch_embd.weight");
        expect_shape(k1, W, 3U * pp, "v.patch_embd.weight.1");
        w.patch_w = own(W, 6U * pp);
        float* dst = mut(w.patch_w);
        for (std::size_t o = 0; o < W; ++o)
            for (std::size_t c = 0; c < 3U; ++c)
                for (std::size_t k = 0; k < pp; ++k) {
                    dst[((o * 3U + c) * 2U + 0U) * pp + k] = k0.at((o * 3U + c) * pp + k);
                    dst[((o * 3U + c) * 2U + 1U) * pp + k] = k1.at((o * 3U + c) * pp + k);
                }
    }
    w.patch_b = s.tensor("v.patch_embd.bias");
    if (config_.embed_norm) {
        w.embed_norm_w = s.tensor("v.pre_ln.weight");
        w.embed_norm_b = optional("v.pre_ln.bias");
    }
    if (config_.final_norm) {
        w.final_norm_w = s.tensor("v.post_ln.weight");
        w.final_norm_b = optional("v.post_ln.bias");
    }
    w.blocks.resize(config_.layers);
    for (std::uint32_t l = 0; l < config_.layers; ++l) {
        const std::string p = "v.blk." + std::to_string(l) + ".";
        vision::BlockWeights& b = w.blocks[l];
        b.norm1_w = s.tensor(p + "ln1.weight");
        b.norm1_b = optional(p + "ln1.bias");
        b.norm2_w = s.tensor(p + "ln2.weight");
        b.norm2_b = optional(p + "ln2.bias");
        if (config_.fused_qkv) {
            b.qkv_w = s.tensor(p + "attn_qkv.weight");
            b.qkv_b = optional(p + "attn_qkv.bias");
            if (config_.qkv_bias && !b.qkv_b) b.qkv_b = own(3U * W, 1);
        } else {
            b.qkv_w = own(3U * W, W);
            if (config_.qkv_bias) b.qkv_b = own(3U * W, 1);
            std::size_t part = 0;
            for (const char* name : {"attn_q", "attn_k", "attn_v"}) {
                const vision::Tensor t = s.tensor(p + name + ".weight");
                expect_shape(t, W, W, name);
                copy_values(t, mut(b.qkv_w) + part * W * W);
                if (const vision::Tensor bias = optional(p + name + ".bias")) {
                    expect_shape(bias, W, 1, name);
                    copy_values(bias, mut(b.qkv_b) + part * W);
                }
                ++part;
            }
        }
        b.q_norm_w = s.tensor(p + "attn_q_norm.weight");
        b.k_norm_w = s.tensor(p + "attn_k_norm.weight");
        expect_shape(b.q_norm_w, D, 1, "attn_q_norm");
        expect_shape(b.k_norm_w, D, 1, "attn_k_norm");
        b.out_w = s.tensor(p + "attn_out.weight");
        b.out_b = optional(p + "attn_out.bias");
        if (config_.out_bias && !b.out_b) b.out_b = own(W, 1);

        // First MLP linear of the shared tower: [gate | up] outputs, the SiLU on the gate half.
        const vision::Tensor gate = s.tensor(p + "ffn_gate.weight"), up = s.tensor(p + "ffn_up.weight");
        expect_shape(gate, M, W, "ffn_gate");
        expect_shape(up, M, W, "ffn_up");
        b.mlp1_w = own(2U * M, W);
        copy_values(gate, mut(b.mlp1_w));
        copy_values(up, mut(b.mlp1_w) + M * W);
        b.mlp2_w = s.tensor(p + "ffn_down.weight");
        if (config_.mlp_bias) {
            b.mlp1_b = own(2U * M, 1);
            if (const vision::Tensor bias = optional(p + "ffn_gate.bias")) {
                expect_shape(bias, M, 1, "ffn_gate bias");
                copy_values(bias, mut(b.mlp1_b));
            }
            if (const vision::Tensor bias = optional(p + "ffn_up.bias")) {
                expect_shape(bias, M, 1, "ffn_up bias");
                copy_values(bias, mut(b.mlp1_b) + M);
            }
            b.mlp2_b = optional(p + "ffn_down.bias");
            if (!b.mlp2_b) b.mlp2_b = own(W, 1);
        }
    }

    merge_w_ = s.tensor("mm.patch_merger.weight");
    merge_b_ = s.tensor("mm.patch_merger.bias");
    fc_w_ = s.tensor("mm.model.fc.weight");
    post_norm_w_ = s.tensor("mm.post_norm.weight");
    post_norm_b_ = optional("mm.post_norm.bias");
    up_w_ = s.tensor("mm.up.weight");
    up_b_ = optional("mm.up.bias");
    gate_w_ = s.tensor("mm.gate.weight");
    gate_b_ = optional("mm.gate.bias");
    down_w_ = s.tensor("mm.down.weight");
    down_b_ = optional("mm.down.bias");
    const std::size_t C = config_.merged_width, P = config_.projection, F = config_.projector_mlp;
    const std::size_t cell = std::size_t{config_.merge} * config_.merge;
    expect_shape(merge_w_, C, W * cell, "mm.patch_merger.weight");
    expect_shape(merge_b_, C, 1, "mm.patch_merger.bias");
    expect_shape(fc_w_, P, C, "mm.model.fc.weight");
    expect_shape(post_norm_w_, P, 1, "mm.post_norm.weight");
    expect_shape(up_w_, F, P, "mm.up.weight");
    expect_shape(gate_w_, F, P, "mm.gate.weight");
    expect_shape(down_w_, P, F, "mm.down.weight");
    if (post_norm_b_) expect_shape(post_norm_b_, P, 1, "mm.post_norm.bias");
    if (up_b_) expect_shape(up_b_, F, 1, "mm.up.bias");
    if (gate_b_) expect_shape(gate_b_, F, 1, "mm.gate.bias");
    if (down_b_) expect_shape(down_b_, P, 1, "mm.down.bias");
}

std::vector<std::pair<vision::Tensor, vision::Tensor>> VisionAdapter::projector_linears() const {
    if (!merge_w_) return {};
    return {{merge_w_, merge_b_}, {fc_w_, {}}, {gate_w_, gate_b_}, {up_w_, up_b_}, {down_w_, down_b_}};
}

void VisionAdapter::project(vision::EncoderOutput& hidden, const vision::PatchGrid& grid, vision::DenseOps& dense,
                            vision::ProjectedRows& out) {
    const std::vector<float> rows = project_f32(hidden, grid, dense);
    out = {};
    out.width = config_.projection;
    out.rows = static_cast<unsigned>(rows.size() / config_.projection);
    out.bf16 = vision::to_bf16_rows(rows.data(), rows.size());
}

std::vector<float> VisionAdapter::project_f32(const vision::EncoderOutput& hidden, const vision::PatchGrid& grid,
                                              vision::DenseOps& dense) const {
    if (!merge_w_) bad("project() before bind()");
    const std::size_t W = config_.width, m = config_.merge, cell = m * m;
    const std::size_t C = config_.merged_width, P = config_.projection, F = config_.projector_mlp;
    const vision::MergedGrid merged = checked_grid(grid);
    if (hidden.device_epoch != 0) bad("the projector reads the tower output on the host");
    if (hidden.width != W || hidden.rows != grid.rows || hidden.hidden.size() != std::size_t{hidden.rows} * W)
        bad("the encoder output does not match the patch grid");
    // The merge reads each cell as consecutive rows: the grid must be in the block order patches() emits.
    for (std::size_t r = 0; r < grid.y.size(); ++r) {
        const std::size_t block = r / cell, slot = r % cell;
        if (grid.y[r] != block / merged.grid_w * m + slot / m || grid.x[r] != block % merged.grid_w * m + slot % m)
            bad("the patch grid is not in merge-block order");
    }
    const std::size_t T = merged.tokens;

    // 2x2 merge: feature c * cell + dy * m + dx of image row t is tower row t * cell + dy * m + dx, channel c.
    std::vector<float> x(T * W * cell), y(T * C);
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t slot = 0; slot < cell; ++slot) {
            const float* src = &hidden.hidden[(t * cell + slot) * W];
            float* dst = &x[t * W * cell + slot];
            for (std::size_t c = 0; c < W; ++c) dst[c * cell] = src[c];
        }
    if (recipe_.conv_inputs_f16)
        for (float& v : x) v = vision::round_to_f16(v);
    dense.linear(merge_w_, merge_b_, x.data(), T, y.data());
    std::vector<float> z(T * P);
    dense.linear(fc_w_, vision::Tensor{}, y.data(), T, z.data());

    // LayerNorm, then GELU (exact).
    // The shared row norm (vision/host_math.hpp), then GELU per value.
    std::vector<float> norm_w(P), norm_b(post_norm_b_ ? P : 0U);
    copy_values(post_norm_w_, norm_w.data());
    if (post_norm_b_) copy_values(post_norm_b_, norm_b.data());
    vision::norm_rows(z.data(), T, P, norm_w.data(), post_norm_b_ ? norm_b.data() : nullptr, recipe_.post_norm_eps, true, z.data());
    for (float& v : z) v = vision::activate(vision::Activation::gelu, v);

    // Clamped SwiGLU MLP: silu(min(gate, limit)) * clamp(up, -limit, limit), then the down projection.
    std::vector<float> gate(T * F), up(T * F);
    dense.linear(gate_w_, gate_b_, z.data(), T, gate.data());
    dense.linear(up_w_, up_b_, z.data(), T, up.data());
    const double limit = static_cast<double>(config_.swiglu_limit);
    for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = vision::clamped_swiglu(gate[i], up[i], limit);
    dense.linear(down_w_, down_b_, gate.data(), T, z.data());
    return z;
}

vision::SpanLayout VisionAdapter::layout(const vision::PatchGrid& grid) const {
    const vision::MergedGrid merged = checked_grid(grid);
    vision::SpanLayout s;
    s.grid_h = merged.grid_h;
    s.grid_w = merged.grid_w;
    s.real_token_id = image_token_id_;
    // Image rows only: the begin and end markers of an image are vocabulary tokens of the prompt.
    s.role.assign(merged.tokens, vision::SpanLayout::Role::image);
    s.lm_pos.resize(merged.tokens);
    for (std::uint32_t i = 0; i < merged.tokens; ++i) s.lm_pos[i] = {i, 0U, 0U};
    return s;
}

}  // namespace strix::glm
