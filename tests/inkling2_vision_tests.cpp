// Host tests of the Inkling 2 image tower adapter on a small synthetic tower with the real tensor names: config
// validation, the tensor schema, the patch rule (padding, normalisation, cell order, feature order, the class row),
// the position inputs (class embedding, align_corners=False resampling, axial angles), the weight binding (q/k/v
// concatenation, LayerScale folding), the projector against a direct computation, the span layout, the token rule
// and the refusals. The reference encoder runs the tower end to end.
#include "inkling2/inkling2_vision.hpp"
#include "inkling2/vision_test_dir.hpp"
#include "inkling2_vision_gate_checks.hpp"

#include "strix/vision/encoder.hpp"
#include "strix/vision/image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {

using namespace strix;
namespace ink = strix::inkling2;

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++failures;                                                          \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
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

// Tiny tower: two blocks, width 64 (one 64-wide head), patch 2, merge 2 (4-pixel cells), 4 x 4 position table.
ink::VisionSpec tiny() {
    ink::VisionSpec s;
    s.layers = 2;
    s.width = 64;
    s.heads = 1;
    s.mlp_width = 96;
    s.patch = 2;
    s.merge = 2;
    s.pos_grid = 4;
    s.projection = 40;
    s.text_width = 24;
    return s;
}

// Deterministic values in [-0.5, 0.5).
float value(std::uint64_t seed, std::size_t i) {
    std::uint64_t x = seed * 0x9E3779B97F4A7C15ULL + i * 0xBF58476D1CE4E5B9ULL + 1U;
    x ^= x >> 31U;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 29U;
    return static_cast<float>(x % 1000003U) / 1000003.0F - 0.5F;
}

class MemoryWeights final : public vision::WeightSource {
public:
    explicit MemoryWeights(const ink::VisionSpec& spec) {
        std::uint64_t seed = 1;
        for (const auto& t : ink::vision_tensor_specs(spec)) {
            std::size_t n = 1;
            for (const auto e : t.extents) n *= static_cast<std::size_t>(e);
            Entry e;
            e.data.resize(n);
            for (std::size_t i = 0; i < n; ++i) e.data[i] = value(seed, i);
            if (t.name.find("ln_") != std::string::npos && t.name.ends_with(".weight"))
                for (auto& v : e.data) v += 1.0F;  // norm gains near one
            e.rows = static_cast<std::size_t>(t.extents[0]);
            e.cols = n / e.rows;
            map_.emplace(t.name, std::move(e));
            ++seed;
        }
    }
    bool has(std::string_view name) const override { return map_.find(std::string(name)) != map_.end(); }
    vision::Tensor tensor(std::string_view name) override {
        const auto& e = map_.at(std::string(name));
        return vision::Tensor{e.data.data(), vision::Tensor::Type::f32, e.rows, e.cols};
    }
    const std::vector<float>& data(const std::string& name) const { return map_.at(name).data; }
    void erase(const std::string& name) { map_.erase(name); }

private:
    struct Entry {
        std::vector<float> data;
        std::size_t rows = 0, cols = 0;
    };
    std::map<std::string, Entry> map_;
};

// y = W x + b, sequential sums: the adapter's projector runs through it, and the direct computation uses it too.
class HostDense final : public vision::DenseOps {
public:
    void linear(const vision::Tensor& w, const vision::Tensor& b, const float* x, std::size_t rows, float* y) override {
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t o = 0; o < w.rows; ++o) {
                float acc = 0.0F;
                for (std::size_t k = 0; k < w.cols; ++k) acc += w.at(o * w.cols + k) * x[r * w.cols + k];
                y[r * w.rows + o] = acc + (b ? b.at(o) : 0.0F);
            }
    }
};

vision::Image image(int w, int h) {
    vision::Image im;
    im.width = w;
    im.height = h;
    im.rgb.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3U);
    for (std::size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = static_cast<std::uint8_t>((i * 37U + 11U) % 256U);
    return im;
}

void test_config() {
    const std::string good = R"({"vision_config":{"model_type":"inkling2_vision","hidden_size":1024,"num_hidden_layers":23,
        "num_attention_heads":16,"mlp_ratio":4.0,"native_patch_size":20,"adapter_merge_size":2,"posemb_grid_size":32,
        "projection_hidden_dim":4096,"projector_bias":false,"text_hidden_size":2304,"temporal_patch_size":2,
        "layer_norm_eps":1e-05}})";
    const auto spec = ink::parse_vision_spec(good);
    CHECK(spec.width == 1024U && spec.layers == 23U && spec.patch == 20U && spec.merge == 2U && spec.text_width == 2304U);
    const auto fallback = ink::parse_vision_spec(R"({"vision_config":{"model_type":"inkling2_vision"}})");
    CHECK(fallback.text_width == spec.text_width);
    CHECK(throws([] {
        (void)ink::parse_vision_spec(R"({"vision_config":{"model_type":"inkling2_vision","text_hidden_size":2305}})");
    }, "text_hidden_size"));
    auto changed = good;
    changed.replace(changed.find("\"hidden_size\":1024"), 18, "\"hidden_size\":1152");
    CHECK(throws([&] { (void)ink::parse_vision_spec(changed); }, "hidden_size"));
    auto other = good;
    other.replace(other.find("inkling2_vision"), 15, "something_else1");
    CHECK(throws([&] { (void)ink::parse_vision_spec(other); }, "model_type"));
    auto bias = good;
    bias.replace(bias.find("\"projector_bias\":false"), 22, "\"projector_bias\":true ");
    CHECK(throws([&] { (void)ink::parse_vision_spec(bias); }, "projector biases"));
    CHECK(throws([&] { (void)ink::parse_vision_spec(R"({"text_config":{}})"); }, "vision_config"));
    // 4 embedding tensors, 2 pre-norm, 18 per block, 2 projector linears
    CHECK(ink::vision_tensor_specs(spec).size() == 4U + 2U + 18U * 23U + 2U);
    CHECK(throws([] { ink::VisionAdapter(tiny(), ""); }, "digest"));
}

void test_tokens() {
    CHECK(ink::vision_tokens(640, 480) == 16U * 12U);
    CHECK(ink::vision_tokens(1, 1) == 1U);
    CHECK(ink::vision_tokens(41, 40) == 2U);
    const auto same = ink::vision_fit(640, 480, 4096);
    CHECK(same.width == 640U && same.height == 480U);
    const auto fit = ink::vision_fit(8000, 2000, 4096);
    CHECK(ink::vision_tokens(fit.width, fit.height) <= 4096U);
    CHECK(fit.width >= 3 * fit.height && fit.width <= 5 * fit.height);  // aspect kept (4:1)
    CHECK(ink::vision_tokens(fit.width + 41U, fit.height + 11U) > 4096U || fit.width > 7900U);  // near the largest fit
}

void test_patches() {
    ink::VisionAdapter adapter(tiny(), "test", ink::VisionIdentity::axial_v1);
    const auto im = image(5, 3);  // 4-pixel cells: 2 x 1 tokens, 4 x 2 patches
    const auto g = adapter.patches(im);
    CHECK(g.grid_h == 2U && g.grid_w == 4U && g.rows == 1U + 8U && g.patch_dim == 12U);
    const auto t = adapter.token_grid(g);
    CHECK(t.rows == 1U && t.cols == 2U);
    bool cls_zero = true;
    for (unsigned i = 0; i < g.patch_dim; ++i) cls_zero = cls_zero && g.data[i] == 0.0F;
    CHECK(cls_zero);
    const float pad = (-1.0F - 127.5F) / 127.5F;
    bool order = true, values = true;
    unsigned row = 1;
    for (unsigned tx = 0; tx < 2U; ++tx)
        for (unsigned dy = 0; dy < 2U; ++dy)
            for (unsigned dx = 0; dx < 2U; ++dx, ++row) {
                const unsigned py0 = dy * 2U, px0 = (tx * 2U + dx) * 2U;
                order = order && g.y[row] == dy && g.x[row] == tx * 2U + dx;
                for (unsigned py = 0; py < 2U; ++py)
                    for (unsigned px = 0; px < 2U; ++px)
                        for (unsigned c = 0; c < 3U; ++c) {
                            const unsigned y = py0 + py, x = px0 + px;
                            const float want = y < 3U && x < 5U
                                ? (static_cast<float>(im.rgb[(y * 5U + x) * 3U + c]) - 127.5F) / 127.5F : pad;
                            values = values && g.data[row * g.patch_dim + (py * 2U + px) * 3U + c] == want;
                        }
            }
    CHECK(order);
    CHECK(values);
    // A larger image under a tight limit is scaled to fit it.
    ink::VisionAdapter small(tiny(), "test", ink::VisionIdentity::axial_v1, 6U);
    const auto big = small.patches(image(40, 10));
    const auto bt = small.token_grid(big);
    CHECK(bt.rows * bt.cols <= 6U);
}

void test_weights_and_positions() {
    const auto spec = tiny();
    MemoryWeights source(spec);
    ink::VisionAdapter adapter(spec, "test", ink::VisionIdentity::axial_v1);
    vision::EncoderWeights w;
    adapter.bind(source, w);
    const auto cfg = adapter.config();
    CHECK(cfg.embed_norm && !cfg.final_norm && !cfg.patch_bias && cfg.qkv_bias &&
          cfg.rope_pairing == vision::EncoderConfig::RopePairing::adjacent);
    const std::string p = "model.visual.backbone.layers.1.";
    const auto& q = source.data(p + "self_attn.q_proj.weight");
    const auto& v = source.data(p + "self_attn.v_proj.weight");
    const auto& kb = source.data(p + "self_attn.k_proj.bias");
    const auto& b1 = w.blocks[1];
    CHECK(b1.qkv_w.rows == 3U * spec.width && b1.qkv_w.at(5) == q[5] &&
          b1.qkv_w.at(2U * spec.width * spec.width + 7U) == v[7] && b1.qkv_b.at(spec.width + 3U) == kb[3]);
    const auto& out = source.data(p + "self_attn.out_proj.weight");
    const auto& ls1 = source.data(p + "ls_1.gamma");
    const auto& fc2b = source.data(p + "mlp.c_proj.bias");
    const auto& ls2 = source.data(p + "ls_2.gamma");
    CHECK(b1.out_w.at(9U * spec.width + 4U) == out[9U * spec.width + 4U] * ls1[9]);
    CHECK(b1.mlp2_b.at(11) == fc2b[11] * ls2[11]);

    const auto im = image(13, 6);  // 4 x 2 tokens, 8 x 4 patches
    const auto g = adapter.patches(im);
    const auto pos = adapter.positions(cfg, w, g);
    vision::validate(cfg, g, pos);
    const auto& cls = source.data("model.visual.backbone.embeddings.class_embedding");
    const auto& clsp = source.data("model.visual.backbone.embeddings.cls_position_embedding");
    CHECK(pos.addend[3] == cls[3] + clsp[3]);
    // Patch rows: F.interpolate(bilinear, align_corners=False) of the 4 x 4 table to the 4 x 8 patch grid.
    const auto& table = source.data("model.visual.backbone.embeddings.position_embedding");
    const auto taps = [](unsigned i, unsigned n, unsigned side, unsigned k, float& weight) {
        const double src = (i + 0.5) * side / n - 0.5;
        const double f = std::floor(src);
        weight = static_cast<float>(std::max(0.0, 1.0 - std::fabs(src - f - k)));
        return static_cast<unsigned>(std::clamp(static_cast<long>(f) + static_cast<long>(k), 0L, static_cast<long>(side) - 1L));
    };
    double worst = 0.0;
    for (unsigned r = 1; r < g.rows; ++r)
        for (unsigned c = 0; c < spec.width; c += 7U) {
            double want = 0.0;
            for (unsigned a = 0; a < 2U; ++a)
                for (unsigned b = 0; b < 2U; ++b) {
                    float wy{}, wx{};
                    const unsigned ty = taps(g.y[r], g.grid_h, spec.pos_grid, a, wy);
                    const unsigned tx = taps(g.x[r], g.grid_w, spec.pos_grid, b, wx);
                    want += static_cast<double>(table[(ty * spec.pos_grid + tx) * spec.width + c]) * wy * wx;
                }
            worst = std::max(worst, std::fabs(want - pos.addend[r * spec.width + c]));
        }
    CHECK(worst < 1e-6);
    // Axial angles: width angles then height angles, positions 1-based, the class row at (0, 0).
    const unsigned half = cfg.head_dim() / 2U, nf = half / 2U;
    bool angles = true;
    for (unsigned f = 0; f < half; ++f) angles = angles && pos.angles[f] == 0.0F;
    const unsigned r = 6;
    for (unsigned f = 0; f < nf; ++f) {
        const float inv = 1.0F / std::pow(10000.0F, static_cast<float>(2U * f) / static_cast<float>(half));
        angles = angles && pos.angles[r * half + f] == static_cast<float>(g.x[r] + 1U) * inv &&
                 pos.angles[r * half + nf + f] == static_cast<float>(g.y[r] + 1U) * inv;
    }
    CHECK(angles);

    // The tower end to end on the host reference.
    vision::ReferenceEncoder encoder(cfg, w);
    auto hidden = encoder.encode(g, pos);
    CHECK(hidden.rows == g.rows && hidden.width == spec.width);
    bool finite = true;
    for (const float x : hidden.hidden) finite = finite && std::isfinite(x);
    CHECK(finite);

    // Projector: drop the class row, slot-major 2 x 2 cells, proj_in, GELU, proj_out (no biases).
    HostDense dense;
    vision::ProjectedRows rows;
    auto copy = hidden;
    adapter.project(copy, g, dense, rows);
    const unsigned tokens = 8U;
    CHECK(rows.rows == tokens && rows.width == spec.text_width && rows.bf16.size() == std::size_t{tokens} * spec.text_width);
    const auto in = source.tensor("model.visual.proj_in.weight");
    const auto outw = source.tensor("model.visual.proj_out.weight");
    vision::Tensor w1 = in, w2 = outw;
    w1.rows = spec.projection;
    w1.cols = 4U * spec.width;
    w2.rows = spec.text_width;
    w2.cols = spec.projection;
    std::vector<float> mid(std::size_t{tokens} * spec.projection), fin(std::size_t{tokens} * spec.text_width);
    dense.linear(w1, {}, hidden.hidden.data() + spec.width, tokens, mid.data());
    for (auto& x : mid) x = vision::activate(vision::Activation::gelu, x);
    dense.linear(w2, {}, mid.data(), tokens, fin.data());
    CHECK(rows.bf16 == vision::to_bf16_rows(fin.data(), fin.size()));

    const auto span = adapter.layout(g);
    CHECK(span.rows() == tokens && span.real_token_id == ink::kVisionImageTokenId && span.grid_h == 2U && span.grid_w == 4U);
    CHECK(span.lm_pos[5][0] == 5U && span.role[7] == vision::SpanLayout::Role::image);
    CHECK(adapter.projector_linears().size() == 2U);
    CHECK(adapter.identity().find("test") != std::string::npos);
}

void test_refusals() {
    const auto spec = tiny();
    ink::VisionAdapter adapter(spec, "test", ink::VisionIdentity::axial_v1, 4U);
    auto g = adapter.patches(image(8, 4));  // 2 x 1 tokens
    auto odd = g;
    odd.grid_w = 3U;
    CHECK(throws([&] { (void)adapter.token_grid(odd); }, "whole number of cells"));
    auto rows = g;
    rows.rows += 1U;
    CHECK(throws([&] { (void)adapter.token_grid(rows); }, "rows differ"));
    auto many = g;
    many.grid_w = 12U;
    many.rows = 1U + many.grid_h * many.grid_w;
    CHECK(throws([&] { (void)adapter.token_grid(many); }, "token limit"));
    MemoryWeights missing(spec);
    missing.erase("model.visual.backbone.layers.0.ls_2.gamma");
    vision::EncoderWeights w;
    CHECK(throws([&] { adapter.bind(missing, w); }, "missing tensor"));
    vision::Image empty;
    CHECK(throws([&] { (void)adapter.patches(empty); }, "invalid image"));
}

void test_reference_admission() {
    namespace fs = std::filesystem;
    namespace gate = ink::gate;
    const auto write = [](const fs::path& path, std::string_view data) {
        std::ofstream out(path, std::ios::binary);
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!out) throw std::runtime_error("write reference fixture");
    };
    VisionTestDir fixture;
    const auto root = fixture.path / "ref";
    fs::create_directory(root);
    const weight::ConfinedDirectory ref(root);
    write(root / "rows.f32", std::string(8, '\0'));
    const auto zero = gate::read_f32(ref, "rows.f32", 2);
    const std::array<float, 2> nonzero{1.0F, 0.0F};
    const auto zero_error = gate::compare_rows(zero, zero, 2);
    const auto mismatch = gate::compare_rows(nonzero, zero, 2);
    CHECK(gate::passes(0.0, zero_error, zero_error, 0.02));
    CHECK(!gate::passes(0.0, mismatch, zero_error, 0.02));  // hidden mismatch alone fails
    CHECK(!gate::passes(0.0, zero_error, mismatch, 0.02));  // all-zero reference cannot bless a nonzero candidate
    CHECK(!gate::passes(0.0, zero_error, zero_error, std::numeric_limits<double>::infinity()));
    const std::array<float, 2> invalid{std::numeric_limits<float>::quiet_NaN(), 0.0F};
    CHECK(throws([&] { (void)gate::compare_rows(invalid, zero, 2); }, "non-finite"));

    const auto shape = gate::shape(serve::parse_json(R"({"rows":2,"cols":1,"tokens":2})"), tiny());
    CHECK(shape.patches == 9 && shape.hidden() == 9U * 64U && shape.projected() == 2U * 24U);
    vision::PatchGrid patch_grid;
    patch_grid.rows = shape.patches;
    patch_grid.patch_dim = shape.patch_dim;
    patch_grid.data.resize(std::size_t{shape.patches} * shape.patch_dim);
    shape.check_patches(patch_grid, 2, 1);
    patch_grid.data[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(throws([&] { shape.check_patches(patch_grid, 2, 1); }, "class patch"));
    vision::EncoderOutput truncated;
    truncated.rows = 8;
    truncated.width = shape.hidden_width;
    truncated.hidden.resize(8U * truncated.width);
    CHECK(throws([&] { shape.check_hidden(truncated); }, "grid/spec"));
    vision::ProjectedRows wrong_width;
    wrong_width.rows = 1;
    wrong_width.width = 2U * shape.projected_width;
    wrong_width.bf16.resize(shape.projected());
    CHECK(throws([&] { shape.check_projected(wrong_width); }, "grid/spec"));
    CHECK(throws([&] { (void)gate::shape(serve::parse_json(R"({"rows":103,"cols":103})"), tiny()); }, "token limit"));
    CHECK(!gate::simple_name("../rows") && !gate::simple_name("x:y") && gate::simple_name("rgba_noise"));
    for (const std::size_t size : {7U, 9U}) {
        write(root / "wrong.f32", std::string(size, '\0'));
        CHECK(throws([&] { (void)gate::read_f32(ref, "wrong.f32", 2); }, "length"));
    }
    write(root / "nan.f32", std::string{"\0\0\xc0\x7f", 4});
    CHECK(throws([&] { (void)gate::read_f32(ref, "nan.f32", 1); }, "non-finite"));
    write(fixture.path / "other", "outside");
    CHECK(throws([&] { (void)ref.read("../other", 16); }, "basename"));
    fs::create_directory(root / "directory");
    CHECK(throws([&] { (void)ref.read("directory", 16); }));
#ifndef _WIN32
    CHECK(::mkfifo((root / "fifo").c_str(), 0600) == 0);
    CHECK(throws([&] { (void)ref.read("fifo", 16); }));
#endif
    std::error_code link_error;
    fs::create_symlink(fixture.path / "other", root / "link", link_error);
#ifndef _WIN32
    CHECK(!link_error);
#endif
    if (!link_error) CHECK(throws([&] { (void)ref.read("link", 16); }));
    else std::printf("symbolic link fixture unavailable: %s\n", link_error.message().c_str());

    link_error.clear();
    fs::create_directory_symlink(root, fixture.path / "ref-link", link_error);
#ifndef _WIN32
    CHECK(!link_error);
#endif
    if (!link_error) CHECK(throws([&] { (void)weight::ConfinedDirectory(fixture.path / "ref-link"); }));

    // Replacing a directory name cannot retarget opens below its held handle.
    fs::rename(root, fixture.path / "moved");
    fs::create_directory(root);
    write(root / "rows.f32", "replacement");
    CHECK(gate::read_f32(ref, "rows.f32", 2) == zero);
    const auto source = ref.open("rows.f32");
    fs::resize_file(fixture.path / "moved" / "rows.f32", 7);
    std::array<weight::Byte, 8> data{};
    CHECK(!source->read({0, data.size()}, data));
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        test_config();
        test_tokens();
        test_patches();
        test_weights_and_positions();
        test_refusals();
        test_reference_admission();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf("inkling2-vision: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
