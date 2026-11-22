#include "inkling2_vision_adapter.hpp"

#include "strix/vision/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <utility>

namespace strix::inkling2 {
namespace {

[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument("inkling2 vision: " + what); }

const char* filter_name(vision::Filter f) {
    switch (f) {
        case vision::Filter::bilinear: return "bilinear";
        case vision::Filter::bicubic: return "bicubic";
        case vision::Filter::lanczos: return "lanczos";
    }
    return "?";
}

unsigned ceil_div(unsigned a, unsigned b) { return (a + b - 1U) / b; }

vision::Tensor expect(vision::WeightSource& s, const std::string& name, std::size_t rows, std::size_t cols) {
    vision::Tensor t = s.tensor(name);
    if (t.size() != rows * cols) bad(name + ": " + std::to_string(t.size()) + " elements, expected " + std::to_string(rows * cols));
    t.rows = rows;
    t.cols = cols;
    return t;
}

vision::Tensor own(vision::EncoderWeights& w, std::vector<float>&& v, std::size_t rows, std::size_t cols) {
    auto& stored = w.owned.emplace_back(std::move(v));
    return vision::Tensor{stored.data(), vision::Tensor::Type::f32, rows, cols};
}

// diag(gamma) * m: row o scaled by gamma[o] (FP32 product of the stored values).
vision::Tensor scale_rows(vision::EncoderWeights& w, const vision::Tensor& m, const vision::Tensor& gamma) {
    std::vector<float> v(m.size());
    for (std::size_t o = 0; o < m.rows; ++o) {
        const float g = gamma.at(o);
        for (std::size_t i = 0; i < m.cols; ++i) v[o * m.cols + i] = g * m.at(o * m.cols + i);
    }
    return own(w, std::move(v), m.rows, m.cols);
}

}  // namespace

std::string VisionRecipe::id() const {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "mean=%.8g,%.8g,%.8g;std=%.8g,%.8g,%.8g;resample=%s%s;pad=%u;max_tokens=%u;min_cells=%u;cls=%d;"
                  "slot_major=%d;proj_act=%u;mlp=%u;placeholder=%u",
                  static_cast<double>(mean[0]), static_cast<double>(mean[1]), static_cast<double>(mean[2]),
                  static_cast<double>(stddev[0]), static_cast<double>(stddev[1]), static_cast<double>(stddev[2]),
                  filter_name(resampler.filter), resampler.antialias ? "+aa" : "", unsigned{pad_fill}, max_tokens,
                  min_side_cells, class_row ? 1 : 0, merge_slot_major ? 1 : 0, static_cast<unsigned>(projector_activation),
                  static_cast<unsigned>(mlp), placeholder_token);
    return buf;
}

VisionAdapter::VisionAdapter(VisionConfig config, std::string weights_sha256_hex, VisionRecipe recipe)
    : config_(config), sha_(std::move(weights_sha256_hex)), recipe_(std::move(recipe)) {
    if (!config_.native_patch_size || !config_.adapter_merge_size || !config_.temporal_patch_size)
        bad("zero patch, merge or temporal size");
    if (!recipe_.max_tokens || !recipe_.min_side_cells) bad("zero token limits");
    vision::validate(this->config());
}

vision::EncoderConfig VisionAdapter::config() const {
    vision::EncoderConfig c;
    c.layers = config_.num_hidden_layers;
    c.width = config_.hidden_size;
    c.heads = config_.num_attention_heads;
    c.mlp_width = config_.mlp_width;
    c.norm = vision::EncoderConfig::Norm::layernorm;
    c.eps = config_.layer_norm_eps;
    c.mlp = recipe_.mlp;
    c.qkv_bias = c.out_bias = c.mlp_bias = true;
    c.patch_bias = false;
    c.pos = vision::EncoderConfig::Pos::learned_interp;
    c.pos_grid = config_.posemb_grid_size;
    c.patch = config_.native_patch_size;
    c.temporal = 1;  // the stored patch weight is 2-D ([W][3 * 20 * 20]): a still image is one frame
    c.channels = 3;
    c.embed_norm = true;
    c.final_norm = false;
    return c;
}

std::string VisionAdapter::identity() const { return "inkling2-vision-v1|" + sha_ + "|" + recipe_.id(); }

vision::ResizeRule VisionAdapter::resize() const {
    const std::uint64_t cell = config_.cell();
    vision::ResizeRule r;
    r.multiple = config_.cell();
    r.min_pixels = cell * cell;
    r.max_pixels = cell * cell * recipe_.max_tokens;
    r.max_rows = recipe_.max_tokens;
    return r;
}

vision::Size VisionAdapter::scaled(int width, int height) const {
    if (width <= 0 || height <= 0) bad("empty image");
    const unsigned cell = config_.cell();
    const auto cells = [cell](double w, double h) {
        return std::uint64_t{ceil_div(static_cast<unsigned>(w), cell)} * ceil_div(static_cast<unsigned>(h), cell);
    };
    if (cells(width, height) <= recipe_.max_tokens) return {width, height};
    // Scale both sides by one factor until the padded cell count fits; the aspect ratio is kept.
    double s = std::sqrt(static_cast<double>(recipe_.max_tokens) * cell * cell / (static_cast<double>(width) * height));
    for (;;) {
        const int w = std::max(1, static_cast<int>(std::floor(width * s)));
        const int h = std::max(1, static_cast<int>(std::floor(height * s)));
        if (cells(w, h) <= recipe_.max_tokens) return {w, h};
        s *= 0.995;
    }
}

vision::Size VisionAdapter::canvas(int width, int height) const {
    const vision::Size s = scaled(width, height);
    const unsigned cell = config_.cell();
    const unsigned cw = std::max(recipe_.min_side_cells, ceil_div(static_cast<unsigned>(s.width), cell));
    const unsigned ch = std::max(recipe_.min_side_cells, ceil_div(static_cast<unsigned>(s.height), cell));
    return {static_cast<int>(cw * cell), static_cast<int>(ch * cell)};
}

unsigned VisionAdapter::tokens(int width, int height) const {
    const vision::Size c = canvas(width, height);
    return static_cast<unsigned>(c.width / static_cast<int>(config_.cell())) *
           static_cast<unsigned>(c.height / static_cast<int>(config_.cell()));
}

vision::PatchGrid VisionAdapter::patches(const vision::Image& image) const {
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != image.pixels() * 3U) bad("empty image");
    const vision::Size s = scaled(image.width, image.height);
    const vision::Size c = canvas(image.width, image.height);
    const unsigned P = config_.native_patch_size;
    vision::check_grid_axes(static_cast<std::uint64_t>(c.height) / P, static_cast<std::uint64_t>(c.width) / P);
    vision::Image src = (s.width == image.width && s.height == image.height)
                            ? image
                            : vision::resize(image, s.width, s.height, recipe_.resampler);
    // The picture at the top left, the margin up to whole 40 px cells filled with pad_fill.
    vision::Image padded;
    padded.width = c.width;
    padded.height = c.height;
    padded.rgb.assign(padded.pixels() * 3U, recipe_.pad_fill);
    for (int y = 0; y < src.height; ++y)
        std::copy_n(&src.rgb[std::size_t(y) * std::size_t(src.width) * 3U], std::size_t(src.width) * 3U,
                    &padded.rgb[std::size_t(y) * std::size_t(c.width) * 3U]);
    const std::vector<float> pixels = vision::normalize_div255(padded, recipe_.mean, recipe_.stddev, false);
    const float* frame = pixels.data();
    vision::PatchSpec spec;
    spec.patch = static_cast<int>(P);
    spec.temporal = 1;
    spec.merge = 1;  // raster order: the 2 x 2 grouping happens in project()
    vision::Patches p = vision::patchify(std::span<const float* const>(&frame, 1), c.height, c.width, spec);

    vision::PatchGrid g;
    g.grid_h = static_cast<unsigned>(p.grid_h);
    g.grid_w = static_cast<unsigned>(p.grid_w);
    const unsigned n = g.grid_h * g.grid_w;
    (void)vision::checked_merged_grid({n, static_cast<unsigned>(p.feature), g.grid_h, g.grid_w, {}, {}, {}, {}},
                                      config_.adapter_merge_size, recipe_.max_tokens);
    g.rows = n + (recipe_.class_row ? 1U : 0U);
    g.patch_dim = static_cast<unsigned>(p.feature);
    g.data = std::move(p.data);
    g.data.resize(std::size_t{g.rows} * g.patch_dim, 0.0F);  // the class row: zero pixels
    g.y.resize(g.rows, 0);
    g.x.resize(g.rows, 0);
    for (unsigned r = 0; r < n; ++r) {
        g.y[r] = static_cast<std::uint16_t>(r / g.grid_w);
        g.x[r] = static_cast<std::uint16_t>(r % g.grid_w);
    }
    return g;
}

vision::PatchGrid VisionAdapter::image_grid(const vision::PatchGrid& tower) const {
    const unsigned n = tower.grid_h * tower.grid_w;
    if (tower.rows != n + (recipe_.class_row ? 1U : 0U)) bad("tower rows differ from the image grid");
    vision::PatchGrid g;
    g.rows = n;
    g.patch_dim = tower.patch_dim;
    g.grid_h = tower.grid_h;
    g.grid_w = tower.grid_w;
    g.y.assign(tower.y.begin(), tower.y.begin() + n);
    g.x.assign(tower.x.begin(), tower.x.begin() + n);
    return g;
}

void VisionAdapter::bind(vision::WeightSource& s, vision::EncoderWeights& w) {
    w = {};
    const std::size_t W = config_.hidden_size, M = config_.mlp_width, G = config_.posemb_grid_size;
    const vision::EncoderConfig c = config();
    const std::string e = "model.visual.backbone.embeddings.";
    w.patch_w = expect(s, e + "patch_embedding.weight", W, c.patch_dim());
    w.pos_embed = expect(s, e + "position_embedding", G * G, W);
    cls_embed_ = expect(s, e + "class_embedding", W, 1);
    cls_pos_ = expect(s, e + "cls_position_embedding", W, 1);
    w.embed_norm_w = expect(s, "model.visual.backbone.ln_pre.weight", W, 1);
    w.embed_norm_b = expect(s, "model.visual.backbone.ln_pre.bias", W, 1);
    w.blocks.resize(config_.num_hidden_layers);
    for (unsigned l = 0; l < config_.num_hidden_layers; ++l) {
        const std::string p = "model.visual.backbone.layers." + std::to_string(l) + ".";
        vision::BlockWeights& b = w.blocks[l];
        b.norm1_w = expect(s, p + "ln_1.weight", W, 1);
        b.norm1_b = expect(s, p + "ln_1.bias", W, 1);
        b.norm2_w = expect(s, p + "ln_2.weight", W, 1);
        b.norm2_b = expect(s, p + "ln_2.bias", W, 1);
        // q, k, v -> one [3 W][W] matrix and [3 W] bias (FP32 copies of the stored values).
        std::vector<float> qkv(3U * W * W), qkv_b(3U * W);
        const char* const names[3] = {"q_proj", "k_proj", "v_proj"};
        for (std::size_t j = 0; j < 3; ++j) {
            const vision::Tensor m = expect(s, p + "self_attn." + names[j] + ".weight", W, W);
            const vision::Tensor bias = expect(s, p + "self_attn." + names[j] + ".bias", W, 1);
            for (std::size_t i = 0; i < W * W; ++i) qkv[j * W * W + i] = m.at(i);
            for (std::size_t i = 0; i < W; ++i) qkv_b[j * W + i] = bias.at(i);
        }
        b.qkv_w = own(w, std::move(qkv), 3U * W, W);
        b.qkv_b = own(w, std::move(qkv_b), 3U * W, 1);
        // LayerScale folded into the residual branch's last linear.
        const vision::Tensor ls1 = expect(s, p + "ls_1.gamma", W, 1);
        const vision::Tensor ls2 = expect(s, p + "ls_2.gamma", W, 1);
        b.out_w = scale_rows(w, expect(s, p + "self_attn.out_proj.weight", W, W), ls1);
        b.out_b = scale_rows(w, expect(s, p + "self_attn.out_proj.bias", W, 1), ls1);
        b.mlp1_w = expect(s, p + "mlp.c_fc.weight", M, W);
        b.mlp1_b = expect(s, p + "mlp.c_fc.bias", M, 1);
        b.mlp2_w = scale_rows(w, expect(s, p + "mlp.c_proj.weight", W, M), ls2);
        b.mlp2_b = scale_rows(w, expect(s, p + "mlp.c_proj.bias", W, 1), ls2);
    }
    const unsigned R = config_.adapter_merge_size;
    const std::size_t in = W * R * R, H = config_.projection_hidden_dim, T = config_.text_hidden_size;
    vision::Tensor pin = expect(s, "model.visual.proj_in.weight", H, in);
    if (recipe_.merge_slot_major) {
        // Checkpoint input index slot * W + ch -> the shared unshuffle's ch * R * R + slot.
        std::vector<float> v(H * in);
        const std::size_t RR = std::size_t{R} * R;
        for (std::size_t o = 0; o < H; ++o)
            for (std::size_t slot = 0; slot < RR; ++slot)
                for (std::size_t ch = 0; ch < W; ++ch) v[o * in + ch * RR + slot] = pin.at(o * in + slot * W + ch);
        pin = own(w, std::move(v), H, in);
    }
    proj_in_ = pin;
    proj_out_ = expect(s, "model.visual.proj_out.weight", T, H);
    vision::validate(c, w);
}

vision::PositionInputs VisionAdapter::positions(const vision::EncoderConfig& c, const vision::EncoderWeights& w,
                                                const vision::PatchGrid& g) const {
    vision::PositionInputs p = vision::default_positions(c, w, image_grid(g));
    if (recipe_.class_row) {
        if (!cls_embed_ || !cls_pos_) bad("positions() before bind()");
        const std::size_t W = c.width;
        p.addend.resize(p.addend.size() + W);
        float* row = &p.addend[p.addend.size() - W];
        for (std::size_t i = 0; i < W; ++i) row[i] = cls_embed_.at(i) + cls_pos_.at(i);
    }
    return p;
}

void VisionAdapter::project(vision::EncoderOutput& hidden, const vision::PatchGrid& grid, vision::DenseOps& dense,
                            vision::ProjectedRows& out) {
    if (!proj_in_) bad("project() before bind()");
    const vision::PatchGrid ig = image_grid(grid);
    const vision::MergedGrid mg = vision::checked_merged_grid(ig, config_.adapter_merge_size, recipe_.max_tokens);
    const unsigned W = config_.hidden_size, T = config_.text_hidden_size;
    if (hidden.width != W || hidden.rows != grid.rows ||
        (hidden.device_epoch == 0 && hidden.hidden.size() != std::size_t{hidden.rows} * W))
        bad("the encoder output does not match the patch grid");
    // The class row is the last tower row: drop it (rows are row-major with the row pitch, so the image rows stay).
    hidden.rows = ig.rows;
    if (!hidden.hidden.empty()) hidden.hidden.resize(std::size_t{ig.rows} * W);
    for (auto& tap : hidden.taps) tap.resize(std::min(tap.size(), std::size_t{ig.rows} * W));

    vision::MergeProjector m;
    m.ratio = config_.adapter_merge_size;
    m.row_major = true;
    m.act = recipe_.projector_activation;
    m.w1 = proj_in_;
    m.w2 = proj_out_;
    if (m.tokens(ig.grid_h, ig.grid_w) != mg.tokens) bad("the merged grid differs from the projector's");
    std::vector<float> rows(std::size_t{mg.tokens} * T);
    dense.merge_project(hidden, ig.grid_h, ig.grid_w, m, rows.data());
    out = {};
    out.rows = mg.tokens;
    out.width = T;
    out.bf16 = vision::to_bf16_rows(rows.data(), rows.size());
}

vision::SpanLayout VisionAdapter::layout(const vision::PatchGrid& grid) const {
    const vision::MergedGrid mg = vision::checked_merged_grid(image_grid(grid), config_.adapter_merge_size, recipe_.max_tokens);
    vision::SpanLayout s;
    s.role.assign(mg.tokens, vision::SpanLayout::Role::image);
    s.lm_pos.resize(mg.tokens);
    for (unsigned i = 0; i < mg.tokens; ++i) s.lm_pos[i] = {i, 0U, 0U};  // 1-D positions, one per row
    s.real_token_id = recipe_.placeholder_token;
    s.grid_h = mg.grid_h;
    s.grid_w = mg.grid_w;
    return s;
}

std::vector<std::pair<vision::Tensor, vision::Tensor>> VisionAdapter::projector_linears() const {
    if (!proj_in_) return {};
    return {{proj_in_, vision::Tensor{}}, {proj_out_, vision::Tensor{}}};
}

unsigned VisionAdapter::max_encoder_rows() const {
    const std::uint64_t r = std::uint64_t{recipe_.max_tokens} * config_.adapter_merge_size * config_.adapter_merge_size +
                            (recipe_.class_row ? 1U : 0U);
    return r > 0x7FFFFFFFULL ? 0U : static_cast<unsigned>(r);
}

}  // namespace strix::inkling2
