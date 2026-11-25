// Host tests of the GLM-5.3 Flash image tower adapter: strict file validation, the canvas table, patch order and
// padding, tensor binding (separate and fused q/k/v give the same rows), the projector against a direct computation,
// the span layout, the cache key and the embed rows file. The tower is a small synthetic one with the real metadata
// keys and tensor names.
#include "glm_vision_toy.hpp"
#include "private_test_dir.hpp"

#include "strix/session/embed_rows_file.hpp"
#include "strix/vision/embed_cache.hpp"
#include "strix/vision/embed_span.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/vision/gguf_weights.hpp"
#include "strix/vision/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace strix;
using namespace glm_vision_toy;

int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            ++failures;                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (false)

bool throws(const std::function<void()>& fn, const char* needle = "") {
    try {
        fn();
    } catch (const std::exception& e) {
        if (std::strstr(e.what(), needle) != nullptr) return true;
        std::printf("  unexpected message: %s\n", e.what());
        return false;
    }
    return false;
}

const std::string kSha(64, 'a');
constexpr std::uint32_t kToken = 7;

vision::Image noise_image(int width, int height, std::uint32_t seed) {
    vision::Image im;
    im.width = width;
    im.height = height;
    im.rgb.resize(im.pixels() * 3U);
    for (auto& v : im.rgb) {
        seed = seed * 1664525U + 1013904223U;
        v = static_cast<std::uint8_t>(seed >> 24U);
    }
    return im;
}

void test_validate() {
    for (const bool fused : {false, true})
        for (const bool biases : {false, true}) {
            const glm::VisionConfig c = glm::validate_vision(small_file(fused, biases).info());
            CHECK(c.layers == 2 && c.width == kW && c.heads == 2 && c.mlp_width == kM && c.head_dim() == kD);
            CHECK(c.patch == kPatch && c.merge == kMerge && c.cell() == 4 && c.max_image_tokens() == 36);
            CHECK(c.merged_width == kC && c.projection == kP && c.projector_mlp == kF);
            CHECK(c.fused_qkv == fused && c.qkv_bias == biases && c.out_bias == biases && c.mlp_bias == biases);
            CHECK(!c.embed_norm && !c.final_norm && c.swiglu_limit == 0.6F && c.mean[1] == 0.4F && c.stddev[2] == 0.2F);
        }
    const File good = small_file(false, true);
    const auto bad_meta = [&](const std::function<void(File&)>& edit, const char* needle) {
        File f = good;
        edit(f);
        CHECK(throws([&] { (void)glm::validate_vision(f.info()); }, needle));
    };
    bad_meta([](File& f) { f.set("general.architecture", std::string("llama")); }, "general.architecture");
    bad_meta([](File& f) { f.set("clip.projector_type", std::string("glm4v")); }, "glm5v");
    bad_meta([](File& f) { f.erase("clip.has_vision_encoder"); }, "no vision encoder");
    bad_meta([](File& f) { f.set("clip.has_audio_encoder", true); }, "audio");
    bad_meta([](File& f) { f.erase("clip.use_silu"); }, "SiLU");
    bad_meta([](File& f) { f.set("clip.use_gelu", true); }, "SiLU");
    bad_meta([](File& f) { f.erase("clip.vision.swiglu_clamp"); }, "missing metadata clip.vision.swiglu_clamp");
    bad_meta([](File& f) { f.erase("clip.vision.image_min_pixels"); }, "image_min_pixels");
    bad_meta([](File& f) { f.set("clip.vision.block_count", 2.0F); }, "must be unsigned");
    bad_meta([](File& f) { f.set("clip.vision.attention.head_count", std::uint32_t{3}); }, "multiple of the head count");
    bad_meta([](File& f) { f.set("clip.vision.image_min_pixels", std::uint32_t{8}); }, "image_min_pixels");
    bad_meta([](File& f) { f.set("clip.vision.image_max_pixels", std::uint32_t{48}); }, "image_min_pixels");
    bad_meta([](File& f) { f.set("clip.vision.image_std", std::vector<float>{0.5F, 0.0F, 0.5F}); }, "mean / std");
    bad_meta([](File& f) { f.set("clip.vision.image_mean", std::vector<float>{0.5F, 0.5F}); }, "at least 3");
    bad_meta([](File& f) { f.set("clip.vision.projection_dim", std::uint32_t{41}); }, "unexpected shape");
    bad_meta([](File& f) { f.drop("v.blk.1.attn_k_norm.weight"); }, "missing tensor v.blk.1.attn_k_norm.weight");
    bad_meta([](File& f) { f.drop("v.patch_embd.weight.1"); }, "missing tensor v.patch_embd.weight.1");
    bad_meta([](File& f) { f.drop("mm.patch_merger.bias"); }, "missing tensor mm.patch_merger.bias");
    bad_meta([](File& f) { f.add("v.position_embd.weight", {kW, 16}, 0.1F); }, "unexpected tensor v.position_embd.weight");
    bad_meta([](File& f) { f.add("mm.model.fc.bias", {kP}, 0.1F); }, "unexpected tensor mm.model.fc.bias");
    bad_meta([](File& f) { f.tensor("v.blk.0.attn_q_norm.weight").dims.assign(1, kW); f.tensor("v.blk.0.attn_q_norm.weight").data.resize(kW); },
             "unexpected shape");
    // One layout for every block: a fused tensor in block 0 makes the file a fused one, which block 1 then is not.
    bad_meta([](File& f) { f.add("v.blk.0.attn_qkv.weight", {kW, 3 * kW}, 0.1F); }, "missing tensor v.blk.1.attn_qkv.weight");
    // An optional merge size defaults to 2; a pre / post norm is read from the inventory.
    File optional = good;
    optional.erase("clip.vision.spatial_merge_size");
    optional.add("v.pre_ln.weight", {kW}, 0.2F, 1.0F);
    optional.add("v.post_ln.weight", {kW}, 0.2F, 1.0F);
    const glm::VisionConfig oc = glm::validate_vision(optional.info());
    CHECK(oc.merge == 2 && oc.embed_norm && oc.final_norm);
    // A quantized tensor is refused by its storage type.
    gguf::FileInfo quantized = good.info();
    quantized.tensors[3].ggml_type = 7;
    CHECK(throws([&] { (void)glm::validate_vision(quantized); }, "quantized"));

    const glm::VisionConfig c = glm::validate_vision(good.info());
    glm::require_vision_matches_language(c, kP, 100, kToken);
    CHECK(throws([&] { glm::require_vision_matches_language(c, kP + 1, 100, kToken); }, "projection width"));
    CHECK(throws([&] { glm::require_vision_matches_language(c, kP, kToken, kToken); }, "outside the vocabulary"));
    CHECK(throws([&] { glm::VisionAdapter(c, "abc", kToken); }, "64 hex"));
}

// The canvas rule on the geometry of a patch 14, merge 2 tower with 16 to 144 image rows. The canvas sizes are those
// of the file format's reference implementation for the same images.
void test_canvas() {
    glm::VisionConfig c = glm::validate_vision(small_file(false, true).info());
    c.patch = 14;
    c.min_pixels = 16 * 784;
    c.max_pixels = 144 * 784;
    const glm::VisionAdapter a(c, kSha, kToken);
    struct Case {
        int w, h, canvas_w, canvas_h;
    };
    for (const Case k : {Case{112, 84, 140, 112}, Case{131, 97, 140, 112}, Case{23, 17, 140, 112}, Case{640, 480, 392, 280},
                         Case{900, 40, 924, 56}, Case{30, 700, 56, 700}, Case{336, 336, 336, 336}, Case{28, 28, 112, 112},
                         Case{337, 336, 336, 336}, Case{1, 1, 112, 112}, Case{100000, 3, 28, 28}}) {
        const auto t = a.target(k.w, k.h);
        CHECK(t.canvas.width == k.canvas_w && t.canvas.height == k.canvas_h);
        CHECK(t.canvas.width % 28 == 0 && t.canvas.height % 28 == 0);
        CHECK(std::uint64_t(t.canvas.width) * std::uint64_t(t.canvas.height) <= c.max_pixels);
        CHECK(t.content.width >= 1 && t.content.height >= 1 && t.content.width <= t.canvas.width && t.content.height <= t.canvas.height);
    }
    // An aspect ratio no canvas of the budget holds collapses to one cell (the reference's rule).
    CHECK(a.target(100000, 3).content.height == 1 && a.target(100000, 3).content.width >= 27);
    // Inside the budget the image is not resampled: it sits on the canvas as it is.
    CHECK((a.target(131, 97).content == vision::Size{131, 97}));
    CHECK((a.target(336, 336).content == vision::Size{336, 336}));
    // Under the minimum it grows to touch the canvas on one axis.
    const auto tiny = a.target(23, 17);
    CHECK(tiny.content.width == 140 || tiny.content.height == 112);
    CHECK(throws([&] { (void)a.target(0, 5); }, "empty image"));
    const auto r = a.resize();
    CHECK(r.multiple == 28 && r.min_pixels == 16 * 784 && r.max_pixels == 144 * 784 && r.max_rows == 144);
    CHECK(a.max_encoder_rows() == 576);
}

void test_patches_and_layout() {
    const glm::VisionConfig c = glm::validate_vision(small_file(false, true).info());
    const glm::VisionAdapter a(c, kSha, kToken);
    const vision::Image im = noise_image(10, 7, 3);  // canvas 12 x 8, no resample
    const auto t = a.target(10, 7);
    CHECK((t.canvas == vision::Size{12, 8}) && (t.content == vision::Size{10, 7}));
    const vision::PatchGrid g = a.patches(im);
    CHECK(g.grid_w == 6 && g.grid_h == 4 && g.rows == 24 && g.patch_dim == 3 * 2 * 2 * 2);
    // Rows in 2x2 block order; a value is (v / 255 - mean) / std; both frames of a patch hold the one image; the
    // margin right of and below the image is black.
    const auto pixel = [&](unsigned y, unsigned x, unsigned ch) {
        const float level = x < 10 && y < 7 ? static_cast<float>(im.rgb[(std::size_t{y} * 10U + x) * 3U + ch]) : 0.0F;
        return (level / 255.0F - c.mean[ch]) / c.stddev[ch];
    };
    bool order = true, values = true;
    for (unsigned r = 0; r < g.rows; ++r) {
        const unsigned block = r / 4, slot = r % 4, y = block / 3 * 2 + slot / 2, x = block % 3 * 2 + slot % 2;
        order = order && g.y[r] == y && g.x[r] == x;
        for (unsigned ch = 0; ch < 3; ++ch)
            for (unsigned frame = 0; frame < 2; ++frame)
                for (unsigned py = 0; py < 2; ++py)
                    for (unsigned px = 0; px < 2; ++px)
                        values = values && g.data[std::size_t{r} * 24U + ((ch * 2U + frame) * 2U + py) * 2U + px] ==
                                               pixel(y * 2U + py, x * 2U + px, ch);
    }
    CHECK(order);
    CHECK(values);

    const vision::SpanLayout s = a.layout(g);
    // The row count from the size alone (before any pixel exists) is the layout's row count.
    CHECK(a.rows_for_size(10, 7) == s.rows());
    CHECK(throws([&] { (void)a.rows_for_size(0, 7); }, "empty image"));
    CHECK(s.rows() == 6 && s.grid_w == 3 && s.grid_h == 2 && s.real_token_id == kToken);
    CHECK(std::all_of(s.role.begin(), s.role.end(), [](auto role) { return role == vision::SpanLayout::Role::image; }));
    CHECK(s.lm_pos[5][0] == 5);

    // Rotary angles: per axis pos * step^f as a FP32 chain, [row | column]; close to the shared default.
    const vision::EncoderConfig ec = a.config();
    CHECK(ec.qk_norm && ec.temporal == 2 && ec.mlp == vision::EncoderConfig::Mlp::swiglu && ec.swiglu_limit == 0.6F);
    const vision::PositionInputs p = a.positions(ec, vision::EncoderWeights{}, g);
    const unsigned half = ec.head_dim() / 2U, per_axis = half / 2U;
    CHECK(p.addend.empty() && p.angles.size() == std::size_t{g.rows} * half);
    const float step = std::pow(10000.0F, -2.0F / static_cast<float>(half));
    bool chain = true;
    for (unsigned r = 0; r < g.rows; ++r) {
        float ty = static_cast<float>(g.y[r]), tx = static_cast<float>(g.x[r]);
        for (unsigned f = 0; f < per_axis; ++f, ty *= step, tx *= step)
            chain = chain && p.angles[std::size_t{r} * half + f] == ty && p.angles[std::size_t{r} * half + per_axis + f] == tx;
    }
    CHECK(chain);
    const vision::PositionInputs d = vision::default_positions(ec, vision::EncoderWeights{}, g);
    double far = 0.0;
    for (std::size_t i = 0; i < d.angles.size(); ++i) far = std::max(far, std::fabs(double(d.angles[i]) - double(p.angles[i])));
    CHECK(far < 1e-5);

    // An image over the budget is refused by the grid check of a caller's own grid, not resized silently there.
    vision::PatchGrid big = g;
    big.grid_h = 14;
    big.grid_w = 12;
    big.rows = 168;
    big.y.clear();
    big.x.clear();
    CHECK(throws([&] { (void)a.layout(big); }, "image rows"));
    vision::PatchGrid odd = g;
    odd.grid_h = 3;
    odd.rows = 18;
    odd.y.resize(18);
    odd.x.resize(18);
    CHECK(throws([&] { (void)a.layout(odd); }, "multiple of the merge factor"));
}

struct Bound {
    File file;
    vision::GgufWeights source;
    glm::VisionConfig config;
    glm::VisionAdapter adapter;
    vision::EncoderWeights weights;
    explicit Bound(File f, glm::VisionRecipe recipe = {})
        : file(std::move(f)), source(file.weights()), config(glm::validate_vision(source.info())),
          adapter(config, kSha, kToken, recipe) {
        adapter.bind(source, weights);
    }
};

// The direct computation of the projector from the tower output, in FP64, reading the file's own tensors.
std::vector<float> direct_projector(File& f, const glm::VisionConfig& c, const vision::EncoderOutput& h, unsigned tokens) {
    const auto lin = [](const TensorData& w, const TensorData* b, const std::vector<double>& x, std::size_t out, std::size_t in) {
        std::vector<double> y(out);
        for (std::size_t o = 0; o < out; ++o) {
            double acc = b ? static_cast<double>(b->data[o]) : 0.0;
            for (std::size_t k = 0; k < in; ++k) acc += static_cast<double>(w.data[o * in + k]) * x[k];
            y[o] = acc;
        }
        return y;
    };
    const double limit = static_cast<double>(c.swiglu_limit);
    std::vector<float> rows(std::size_t{tokens} * kP);
    for (unsigned t = 0; t < tokens; ++t) {
        std::vector<double> x(kW * 4);
        for (unsigned dy = 0; dy < 2; ++dy)
            for (unsigned dx = 0; dx < 2; ++dx)
                for (unsigned ch = 0; ch < kW; ++ch)
                    x[ch * 4 + dy * 2 + dx] = static_cast<double>(h.hidden[(std::size_t{t} * 4 + dy * 2 + dx) * kW + ch]);
        // The intermediate rows are FP32 in the adapter (one rounding per linear): mirrored here.
        const auto f32 = [](std::vector<double> v) {
            for (auto& e : v) e = static_cast<double>(static_cast<float>(e));
            return v;
        };
        const auto y = f32(lin(f.tensor("mm.patch_merger.weight"), &f.tensor("mm.patch_merger.bias"), x, kC, kW * 4));
        auto z = f32(lin(f.tensor("mm.model.fc.weight"), nullptr, y, kP, kC));
        double mean = 0.0, sq = 0.0;
        for (const double v : z) mean += v;
        mean /= static_cast<double>(kP);
        for (const double v : z) sq += (v - mean) * (v - mean);
        for (std::size_t i = 0; i < kP; ++i) {
            const double n = static_cast<double>(static_cast<float>(
                (z[i] - mean) / std::sqrt(sq / static_cast<double>(kP) + 1e-5F) * static_cast<double>(f.tensor("mm.post_norm.weight").data[i]) +
                static_cast<double>(f.tensor("mm.post_norm.bias").data[i])));
            z[i] = static_cast<double>(static_cast<float>(0.5 * n * (1.0 + std::erf(n / std::sqrt(2.0)))));
        }
        const auto gate = f32(lin(f.tensor("mm.gate.weight"), nullptr, z, kF, kP));
        const auto up = f32(lin(f.tensor("mm.up.weight"), nullptr, z, kF, kP));
        std::vector<double> act(kF);
        for (std::size_t i = 0; i < kF; ++i) {
            const double g = std::min(gate[i], limit), u = std::clamp(up[i], -limit, limit);
            act[i] = static_cast<double>(static_cast<float>(g / (1.0 + std::exp(-g)) * u));
        }
        const auto out = lin(f.tensor("mm.down.weight"), &f.tensor("mm.down.bias"), act, kP, kF);
        for (std::size_t i = 0; i < kP; ++i) rows[std::size_t{t} * kP + i] = static_cast<float>(out[i]);
    }
    return rows;
}

void test_encode_and_project() {
    const vision::Image im = noise_image(13, 9, 11);  // canvas 16 x 12: 8 x 6 patches, 12 image rows
    Bound sep(small_file(false, true)), fused(small_file(true, true));
    const vision::PatchGrid g = sep.adapter.patches(im);
    CHECK(g.rows == 48);

    vision::ReferenceEncoder enc(sep.adapter.config(), sep.weights);
    const vision::EncoderConfig ec = sep.adapter.config();
    const vision::EncoderOutput hidden = enc.encode(g, sep.adapter.positions(ec, sep.weights, g));
    const std::vector<float> rows = sep.adapter.project_f32(hidden, g, enc);
    CHECK(rows.size() == 12U * kP);
    // The clamp is active on this tower: without it the rows differ.
    const std::vector<float> want = direct_projector(sep.file, sep.config, hidden, 12);
    double worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        worst = std::max(worst, std::fabs(double(rows[i]) - double(want[i])));
        scale = std::max(scale, std::fabs(double(want[i])));
    }
    CHECK(scale > 0.05 && worst < 2e-6 * std::max(1.0, scale));

    // project() stores exactly those rows as BF16; encode_cached runs the same steps.
    vision::EncoderOutput again = hidden;
    vision::ProjectedRows stored;
    sep.adapter.project(again, g, enc, stored);
    CHECK(stored.rows == 12 && stored.width == kP && stored.bf16 == vision::to_bf16_rows(rows.data(), rows.size()));
    const auto cached = vision::encode_cached(sep.adapter, enc, sep.weights, nullptr, vision::Digest{}, g);
    CHECK(cached.rows->bf16 == stored.bf16);

    // Fused q/k/v with the concatenated tensors: the same tower, the same bits.
    vision::ReferenceEncoder enc_fused(fused.adapter.config(), fused.weights);
    const auto cached_fused = vision::encode_cached(fused.adapter, enc_fused, fused.weights, nullptr, vision::Digest{}, g);
    CHECK(cached_fused.rows->bf16 == stored.bf16);

    // Biases a block lacks are zeros: a file without them equals the file with zeroed ones.
    File zeroed = small_file(false, true);
    for (auto& t : zeroed.tensors)
        if (t.name.find("attn_q.bias") != std::string::npos || t.name.find("attn_k.bias") != std::string::npos ||
            t.name.find("attn_v.bias") != std::string::npos || t.name.find("attn_out.bias") != std::string::npos ||
            t.name.find("ffn_gate.bias") != std::string::npos || t.name.find("ffn_down.bias") != std::string::npos)
            std::fill(t.data.begin(), t.data.end(), 0.0F);
    Bound with_zero(zeroed), without(small_file(false, false));
    vision::ReferenceEncoder e0(with_zero.adapter.config(), with_zero.weights), e1(without.adapter.config(), without.weights);
    CHECK(vision::encode_cached(with_zero.adapter, e0, with_zero.weights, nullptr, vision::Digest{}, g).rows->bf16 ==
          vision::encode_cached(without.adapter, e1, without.weights, nullptr, vision::Digest{}, g).rows->bf16);

    // The second patch kernel takes part (the image fills both frames).
    File one_kernel = small_file(false, true);
    auto& k1 = one_kernel.tensor("v.patch_embd.weight.1").data;
    std::fill(k1.begin(), k1.end(), 0.0F);
    Bound single(one_kernel);
    vision::ReferenceEncoder e2(single.adapter.config(), single.weights);
    CHECK(vision::encode_cached(single.adapter, e2, single.weights, nullptr, vision::Digest{}, g).rows->bf16 != stored.bf16);

    // Refusals of project(): a grid that is not in block order, rows that do not match, hidden rows left on a device.
    vision::PatchGrid raster = g;
    for (unsigned r = 0; r < raster.rows; ++r) {
        raster.y[r] = static_cast<std::uint16_t>(r / raster.grid_w);
        raster.x[r] = static_cast<std::uint16_t>(r % raster.grid_w);
    }
    CHECK(throws([&] { (void)sep.adapter.project_f32(hidden, raster, enc); }, "merge-block order"));
    vision::EncoderOutput short_rows = hidden;
    short_rows.hidden.pop_back();
    CHECK(throws([&] { (void)sep.adapter.project_f32(short_rows, g, enc); }, "does not match the patch grid"));
    vision::EncoderOutput resident = hidden;
    resident.device_epoch = 1;
    CHECK(throws([&] { (void)sep.adapter.project_f32(resident, g, enc); }, "on the host"));
    const glm::VisionAdapter unbound(sep.config, kSha, kToken);
    CHECK(throws([&] { (void)unbound.project_f32(hidden, g, enc); }, "before bind"));
    CHECK(sep.adapter.projector_linears().size() == 5 && unbound.projector_linears().empty());

    // The device towers refuse this tower: it runs on the host reference.
    vision::EncoderConfig device_shape = ec;
    device_shape.width = 128;  // head dimension 64, a shape the device plan takes
    std::string why;
    CHECK(!vision::hip_supported(device_shape, &why) && why.find("q/k norm") != std::string::npos);

    // The diagnostic FP16 arm is another recipe: other rows, another identity.
    glm::VisionRecipe f16;
    f16.conv_inputs_f16 = true;
    Bound model(small_file(false, true), f16);
    vision::ReferenceEncoder e3(model.adapter.config(), model.weights);
    const vision::PatchGrid g16 = model.adapter.patches(im);
    CHECK(g16.data != g.data);
    CHECK(model.adapter.identity() != sep.adapter.identity());
}

void test_identity() {
    const glm::VisionConfig c = glm::validate_vision(small_file(false, true).info());
    const glm::VisionAdapter a(c, kSha, kToken);
    CHECK(a.identity() == glm::VisionAdapter(c, kSha, kToken).identity());
    CHECK(a.identity() != glm::VisionAdapter(c, std::string(64, 'b'), kToken).identity());
    CHECK(a.identity() != glm::VisionAdapter(c, kSha, kToken + 1).identity());
    glm::VisionRecipe r;
    r.pad_fill = 127;
    CHECK(a.identity() != glm::VisionAdapter(c, kSha, kToken, r).identity());
    r = {};
    r.rope_theta = std::nextafter(10000.0F, 20000.0F);
    CHECK(a.identity() != glm::VisionAdapter(c, kSha, kToken, r).identity());
    r = {};
    r.resampler.antialias = false;
    CHECK(a.identity() != glm::VisionAdapter(c, kSha, kToken, r).identity());
    r = {};
    r.post_norm_eps = 0.0F;
    CHECK(throws([&] { glm::VisionAdapter(c, kSha, kToken, r); }, "post_norm_eps"));
    CHECK(glm::kImageBeginToken == 154830 && glm::kImageEndToken == 154831 && glm::kImageToken == 154854);
    CHECK(glm::kLanguageHidden == 4096 && glm::kLanguageVocab == 154880);
}

void test_rows_file() {
    Bound b(small_file(false, true));
    const vision::Image im = noise_image(8, 8, 5);
    const vision::PatchGrid g = b.adapter.patches(im);
    vision::ReferenceEncoder enc(b.adapter.config(), b.weights);
    const auto encoded = vision::encode_cached(b.adapter, enc, b.weights, nullptr, vision::Digest{}, g);
    session::EmbedDigest digest{};
    digest[0] = 42;
    const session::EmbedRows rows = vision::span_embed_rows(b.adapter.layout(g), *encoded.rows, digest);
    CHECK(rows.rows == 4 && rows.width == kP && rows.real_token_id == kToken && rows.axes == 0);

    const strix::test::PrivateTestDir dir("strix-glm-vision-rows");
    const auto path = dir.fresh("rows.embed");
    session::write_embed_rows_file(path, rows);
    const session::EmbedRows back = session::read_embed_rows_file(path, kP);
    CHECK(back.rows == rows.rows && back.width == rows.width && back.real_token_id == kToken && back.digest == digest &&
          back.data == rows.data);
    CHECK(throws([&] { (void)session::read_embed_rows_file(path, kP + 1); }, "the model takes"));
    {  // the payload pass holds the admitted header: a file replaced by a larger valid one in between is refused
        const auto admitted = session::read_embed_rows_file_info(path, kP);
        CHECK(session::read_embed_rows_file(path, admitted).data == rows.data);
        session::EmbedRows larger = rows;
        larger.rows = rows.rows * 2U;
        larger.data.resize(std::size_t{larger.rows} * larger.width, 0x3C00U);
        session::write_embed_rows_file(path, larger);
        CHECK(throws([&] { (void)session::read_embed_rows_file(path, admitted); }, "changed after its header was admitted"));
        session::write_embed_rows_file(path, rows);
    }
    {  // one flipped payload bit fails the checksum; a cut file fails the length
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(static_cast<std::streamoff>(session::kEmbedRowsFileHeader + 5));
        char byte = 0;
        f.read(&byte, 1);
        byte = static_cast<char>(byte ^ 1);
        f.seekp(static_cast<std::streamoff>(session::kEmbedRowsFileHeader + 5));
        f.write(&byte, 1);
    }
    CHECK(throws([&] { (void)session::read_embed_rows_file(path); }, "checksum"));
    std::filesystem::resize_file(path, session::kEmbedRowsFileHeader + 6);
    CHECK(throws([&] { (void)session::read_embed_rows_file(path); }, "length"));
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f << "not an embed rows file, but long enough to hold a whole header of the format";
    }
    CHECK(throws([&] { (void)session::read_embed_rows_file(path); }, "version 1"));
    session::EmbedRows deep = rows;
    deep.axes = 1;
    CHECK(throws([&] { session::write_embed_rows_file(path, deep); }, "plain content rows"));
}

// The projector byte budget is judged from the parsed header alone (no tensor byte is needed).
void test_projector_bytes() {
    const auto file = small_file(false, true);
    const gguf::FileInfo& info = file.info();
    CHECK(glm::check_projector_bytes(info) == info.file_size);
    const auto mutated = [&](const std::function<void(gguf::FileInfo&)>& edit) {
        gguf::FileInfo copy = info;
        edit(copy);
        return copy;
    };
    CHECK(throws([&] { (void)glm::check_projector_bytes(mutated([](gguf::FileInfo& i) { i.tensors.front().absolute_end = i.file_size + 1U; })); },
                 "outside the file"));
    CHECK(throws([&] { (void)glm::check_projector_bytes(mutated([](gguf::FileInfo& i) { i.file_size += glm::kMaxProjectorTrailingBytes + 1U; })); },
                 "after its last tensor"));
    CHECK(throws([&] { (void)glm::check_projector_bytes(mutated([](gguf::FileInfo& i) { i.tensors.front().byte_size = glm::kMaxProjectorTensorBytes + 1U; })); },
                 "exceed"));
    CHECK(throws([&] { (void)glm::check_projector_bytes(mutated([](gguf::FileInfo& i) { i.data_offset = i.file_size + 1U; })); }, "shorter than its header"));
    // A large aligned gap before the tensors or between two of them: every tensor still lies inside the file and the
    // tail is within the allowance, but the length is not accounted for by the header and tensors.
    constexpr std::uint64_t gap = std::uint64_t{1} << 30U;
    const auto shift_from = [&](std::size_t first) {
        return mutated([&](gguf::FileInfo& i) {
            for (std::size_t t = first; t < i.tensors.size(); ++t) {
                i.tensors[t].absolute_offset += gap;
                i.tensors[t].absolute_end += gap;
            }
            i.file_size += gap;
        });
    };
    CHECK(throws([&] { (void)glm::check_projector_bytes(shift_from(0)); }, "unaccounted bytes"));
    if (info.tensors.size() > 1) CHECK(throws([&] { (void)glm::check_projector_bytes(shift_from(1)); }, "unaccounted bytes"));
    CHECK(throws([&] { (void)glm::check_projector_bytes(mutated([](gguf::FileInfo& i) {
                     i.data_offset = glm::kMaxProjectorHeaderBytes + 1U;
                     i.file_size += glm::kMaxProjectorHeaderBytes + 1U;
                 })); }, "header exceeds"));
}

}  // namespace

int main() {
    try {
        test_projector_bytes();
        test_validate();
        test_canvas();
        test_patches_and_layout();
        test_encode_and_project();
        test_identity();
        test_rows_file();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) return 1;
    std::puts("glm5-vision: PASS");
    return 0;
}
