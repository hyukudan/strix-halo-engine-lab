// Host test of the Inkling 2 Mini vision binding on the shared tower: geometry (pad to whole 40 px cells, one LM row
// per cell, scaling above the token limit), the class row, and the mapping of the checkpoint graph onto the shared
// graph. The mapped pipeline (embed_norm, folded LayerScale, fused qkv, class row last, permuted proj_in, shared
// MergeProjector) is compared with an independent double-precision forward of the unfolded checkpoint graph
// (class row first, explicit ln_pre and LayerScale, separate q / k / v, slot-major 2 x 2 concatenation).
// No GPU, no checkpoint.
#include "inkling2/inkling2_vision.hpp"

#include "strix/vision/image.hpp"
#include "strix/vision/lookup_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace vis = strix::vision;
namespace ink = strix::inkling2;

namespace {

int failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

struct MapSource final : vis::WeightSource {
    std::map<std::string, std::vector<float>, std::less<>> t;
    std::map<std::string, std::pair<std::size_t, std::size_t>, std::less<>> shape;
    std::uint64_t s = 99;
    float rnd() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<double>(s >> 40U) / 8388608.0 - 1.0);
    }
    void add(const std::string& name, std::size_t rows, std::size_t cols, float scale, float offset = 0.0F) {
        auto& v = t[name];
        v.resize(rows * cols);
        for (auto& e : v) e = offset + scale * rnd();
        shape[name] = {rows, cols};
    }
    bool has(std::string_view name) const override { return t.find(name) != t.end(); }
    vis::Tensor tensor(std::string_view name) override {
        const auto it = t.find(name);
        if (it == t.end()) throw std::out_of_range(std::string(name));
        const auto sh = shape.find(name)->second;
        return vis::Tensor{it->second.data(), vis::Tensor::Type::f32, sh.first, sh.second};
    }
    const std::vector<float>& v(const std::string& name) const { return t.find(name)->second; }
};

ink::VisionSpec toy_config() {
    ink::VisionSpec c;
    c.width = 16;
    c.layers = 2;
    c.heads = 2;
    c.mlp_width = 24;
    c.patch = 2;
    c.merge = 2;
    c.pos_grid = 4;
    c.projection = 20;
    c.text_width = 12;
    return c;
}

void fill_source(MapSource& s, const ink::VisionSpec& c) {
    const std::size_t W = c.width, M = c.mlp_width, G = c.pos_grid;
    const std::size_t pd = 3U * c.patch * c.patch;  // 2-D stored patch weight
    const std::string e = "model.visual.backbone.embeddings.";
    s.add(e + "patch_embedding.weight", W, pd, 0.1F);
    s.add(e + "position_embedding", G * G, W, 0.3F);
    s.add(e + "class_embedding", W, 1, 0.5F);
    s.add(e + "cls_position_embedding", W, 1, 0.3F);
    s.add("model.visual.backbone.ln_pre.weight", W, 1, 0.2F, 1.0F);
    s.add("model.visual.backbone.ln_pre.bias", W, 1, 0.1F);
    const float ws = 1.0F / static_cast<float>(c.heads * 4U);
    for (unsigned l = 0; l < c.layers; ++l) {
        const std::string p = "model.visual.backbone.layers." + std::to_string(l) + ".";
        for (const char* n : {"ln_1", "ln_2"}) {
            s.add(p + n + ".weight", W, 1, 0.2F, 1.0F);
            s.add(p + n + ".bias", W, 1, 0.1F);
        }
        s.add(p + "ls_1.gamma", W, 1, 0.3F, 0.5F);
        s.add(p + "ls_2.gamma", W, 1, 0.3F, 0.5F);
        for (const char* n : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            s.add(p + "self_attn." + n + ".weight", W, W, ws * 2.0F);
            s.add(p + "self_attn." + n + ".bias", W, 1, 0.1F);
        }
        s.add(p + "mlp.c_fc.weight", M, W, 0.3F);
        s.add(p + "mlp.c_fc.bias", M, 1, 0.1F);
        s.add(p + "mlp.c_proj.weight", W, M, 0.2F);
        s.add(p + "mlp.c_proj.bias", W, 1, 0.1F);
    }
    const std::size_t R = c.merge;
    s.add("model.visual.proj_in.weight", c.projection, W * R * R, 0.15F);
    s.add("model.visual.proj_out.weight", c.text_width, c.projection, 0.2F);
}

vis::Image test_image(int w, int h) {
    vis::Image im;
    im.width = w;
    im.height = h;
    im.rgb.resize(im.pixels() * 3U);
    for (std::size_t i = 0; i < im.rgb.size(); ++i) im.rgb[i] = static_cast<std::uint8_t>((i * 37U + 11U) % 251U);
    return im;
}

using Rows = std::vector<std::vector<double>>;

std::vector<double> lin(const std::vector<float>& w, const std::vector<float>* b, const std::vector<double>& x, std::size_t out) {
    const std::size_t in = x.size();
    std::vector<double> y(out);
    for (std::size_t o = 0; o < out; ++o) {
        double acc = b ? static_cast<double>((*b)[o]) : 0.0;
        for (std::size_t i = 0; i < in; ++i) acc += static_cast<double>(w[o * in + i]) * x[i];
        y[o] = acc;
    }
    return y;
}

std::vector<double> layernorm(const std::vector<double>& x, const std::vector<float>& w, const std::vector<float>& b, double eps) {
    const double n = static_cast<double>(x.size());
    double mean = 0.0, sq = 0.0;
    for (const double v : x) mean += v;
    mean /= n;
    for (const double v : x) sq += (v - mean) * (v - mean);
    const double inv = 1.0 / std::sqrt(sq / n + eps);
    std::vector<double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = (x[i] - mean) * inv * w[i] + b[i];
    return y;
}

double gelu(double v) { return 0.5 * v * (1.0 + std::erf(v * 0.70710678118654752440)); }

// Unfolded checkpoint graph in raster order with the class row first. The axial
// arm computes its own pixels, learned positions and rotations.
std::vector<double> independent(const MapSource& s, const ink::VisionSpec& c, const vis::PatchGrid& g,
                                const vis::PositionInputs& pos, const vis::Image* axial_image = nullptr) {
    const std::size_t W = c.width, H = c.heads, D = W / H, M = c.mlp_width;
    const std::size_t N = std::size_t{g.grid_h} * g.grid_w, pd = g.patch_dim;
    const std::string e = "model.visual.backbone.embeddings.";
    Rows x(N + 1U, std::vector<double>(W));
    for (std::size_t i = 0; i < W; ++i)
        x[0][i] = static_cast<double>(s.v(e + "class_embedding")[i]) + static_cast<double>(s.v(e + "cls_position_embedding")[i]);
    for (std::size_t r = 0; r < N; ++r) {
        std::vector<double> p(pd);
        const std::size_t patch_y = r / g.grid_w, patch_x = r % g.grid_w;
        if (axial_image) {
            const auto width = static_cast<std::size_t>(axial_image->width);
            const auto height = static_cast<std::size_t>(axial_image->height);
            for (std::size_t i = 0; i < pd; ++i) {
                const auto py = patch_y * c.patch + i / (3U * c.patch);
                const auto px = patch_x * c.patch + (i / 3U) % c.patch;
                const double raw = py < height && px < width ? axial_image->rgb[(py * width + px) * 3U + i % 3U] : -1.0;
                p[i] = (raw - 127.5) / 127.5;
            }
        } else {
            for (std::size_t i = 0; i < pd; ++i) p[i] = g.data[r * pd + i];
        }
        const auto y = lin(s.v(e + "patch_embedding.weight"), nullptr, p, W);
        if (axial_image) {
            const auto coordinate = [side = c.pos_grid](std::size_t index, unsigned length) {
                return std::clamp((static_cast<double>(index) + 0.5) * side / length - 0.5, 0.0, side - 1.0);
            };
            const auto iy = coordinate(patch_y, g.grid_h), ix = coordinate(patch_x, g.grid_w);
            const auto y0 = static_cast<std::size_t>(iy), x0 = static_cast<std::size_t>(ix);
            const auto y1 = std::min<std::size_t>(y0 + 1U, c.pos_grid - 1U);
            const auto x1 = std::min<std::size_t>(x0 + 1U, c.pos_grid - 1U);
            const auto& table = s.v(e + "position_embedding");
            for (std::size_t i = 0; i < W; ++i) {
                const auto at = [&](std::size_t ty, std::size_t tx) { return table[(ty * c.pos_grid + tx) * W + i]; };
                const auto frac_x = ix - static_cast<double>(x0), frac_y = iy - static_cast<double>(y0);
                const double top = std::lerp(static_cast<double>(at(y0, x0)), static_cast<double>(at(y0, x1)), frac_x);
                const double bottom = std::lerp(static_cast<double>(at(y1, x0)), static_cast<double>(at(y1, x1)), frac_x);
                x[r + 1U][i] = y[i] + std::lerp(top, bottom, frac_y);
            }
        } else {
            for (std::size_t i = 0; i < W; ++i) x[r + 1U][i] = y[i] + static_cast<double>(pos.addend[r * W + i]);
        }
    }
    for (auto& row : x)
        row = layernorm(row, s.v("model.visual.backbone.ln_pre.weight"), s.v("model.visual.backbone.ln_pre.bias"), c.eps);
    for (unsigned l = 0; l < c.layers; ++l) {
        const std::string p = "model.visual.backbone.layers." + std::to_string(l) + ".";
        const std::string a = p + "self_attn.";
        Rows q(N + 1U), k(N + 1U), v(N + 1U);
        for (std::size_t r = 0; r <= N; ++r) {
            const auto h = layernorm(x[r], s.v(p + "ln_1.weight"), s.v(p + "ln_1.bias"), c.eps);
            q[r] = lin(s.v(a + "q_proj.weight"), &s.v(a + "q_proj.bias"), h, W);
            k[r] = lin(s.v(a + "k_proj.weight"), &s.v(a + "k_proj.bias"), h, W);
            v[r] = lin(s.v(a + "v_proj.weight"), &s.v(a + "v_proj.bias"), h, W);
            if (axial_image && r > 0U) {
                const std::size_t frequencies = D / 4U;
                const std::size_t raster = r - 1U;
                for (std::size_t head = 0; head < H; ++head)
                    for (std::size_t axis = 0; axis < 2U; ++axis)
                        for (std::size_t f = 0; f < frequencies; ++f) {
                            const auto index = head * D + 2U * (axis * frequencies + f);
                            const auto position = axis == 0U ? raster % g.grid_w : raster / g.grid_w;
                            const auto angle = static_cast<double>(position + 1U) /
                                               std::pow(static_cast<double>(c.rope_theta), static_cast<double>(f) /
                                                        static_cast<double>(frequencies));
                            for (auto* row : {&q[r], &k[r]}) {
                                const auto even = (*row)[index], odd = (*row)[index + 1U];
                                (*row)[index] = even * std::cos(angle) - odd * std::sin(angle);
                                (*row)[index + 1U] = odd * std::cos(angle) + even * std::sin(angle);
                            }
                        }
            }
        }
        Rows att(N + 1U, std::vector<double>(W, 0.0));
        for (std::size_t hd = 0; hd < H; ++hd)
            for (std::size_t r = 0; r <= N; ++r) {
                std::vector<double> sc(N + 1U);
                double mx = -1e300;
                for (std::size_t j = 0; j <= N; ++j) {
                    double d = 0.0;
                    for (std::size_t i = 0; i < D; ++i) d += q[r][hd * D + i] * k[j][hd * D + i];
                    sc[j] = d / std::sqrt(static_cast<double>(D));
                    mx = std::max(mx, sc[j]);
                }
                double z = 0.0;
                for (auto& t : sc) z += (t = std::exp(t - mx));
                for (std::size_t j = 0; j <= N; ++j)
                    for (std::size_t i = 0; i < D; ++i) att[r][hd * D + i] += sc[j] / z * v[j][hd * D + i];
            }
        for (std::size_t r = 0; r <= N; ++r) {
            const auto o = lin(s.v(a + "out_proj.weight"), &s.v(a + "out_proj.bias"), att[r], W);
            for (std::size_t i = 0; i < W; ++i) x[r][i] += static_cast<double>(s.v(p + "ls_1.gamma")[i]) * o[i];
            const auto h = layernorm(x[r], s.v(p + "ln_2.weight"), s.v(p + "ln_2.bias"), c.eps);
            auto m = lin(s.v(p + "mlp.c_fc.weight"), &s.v(p + "mlp.c_fc.bias"), h, M);
            for (auto& t : m) t = gelu(t);
            const auto y = lin(s.v(p + "mlp.c_proj.weight"), &s.v(p + "mlp.c_proj.bias"), m, W);
            for (std::size_t i = 0; i < W; ++i) x[r][i] += static_cast<double>(s.v(p + "ls_2.gamma")[i]) * y[i];
        }
    }
    // Drop the class row; slot-major 2 x 2 concatenation in raster cells; proj_in, GELU, proj_out.
    const std::size_t R = c.merge, mh = g.grid_h / R, mw = g.grid_w / R;
    std::vector<double> out;
    for (std::size_t my = 0; my < mh; ++my)
        for (std::size_t mx = 0; mx < mw; ++mx) {
            std::vector<double> f(W * R * R);
            for (std::size_t dy = 0; dy < R; ++dy)
                for (std::size_t dx = 0; dx < R; ++dx) {
                    const std::size_t row = 1U + (my * R + dy) * g.grid_w + mx * R + dx;
                    for (std::size_t i = 0; i < W; ++i) f[(dy * R + dx) * W + i] = x[row][i];
                }
            auto h = lin(s.v("model.visual.proj_in.weight"), nullptr, f, c.projection);
            for (auto& t : h) t = gelu(t);
            const auto y = lin(s.v("model.visual.proj_out.weight"), nullptr, h, c.text_width);
            out.insert(out.end(), y.begin(), y.end());
        }
    return out;
}

void test_geometry() {
    const ink::VisionSpec c = toy_config();  // cell = 4 px
    ink::VisionAdapter a(c, "sha");
    CHECK(a.canvas(10, 6) == (vis::Size{12, 8}));
    CHECK(a.tokens(10, 6) == 6U);
    CHECK(a.tokens(8, 8) == 4U);  // exact multiples stay unpadded
    const vis::ResizeRule r = a.resize();
    CHECK(r.multiple == 4U && r.max_rows == a.recipe().max_tokens);
    ink::VisionRecipe small;
    small.max_tokens = 4;
    ink::VisionAdapter b(c, "sha", small);
    const vis::Size s = b.canvas(40, 20);
    CHECK(s.width % 4 == 0 && s.height % 4 == 0);
    CHECK(b.tokens(40, 20) <= 4U && b.tokens(40, 20) >= 2U);
    CHECK(s.width >= s.height);  // aspect kept
    CHECK(a.identity() != b.identity());
    // Real geometry: 1024 x 768 -> 1040 x 800 -> 26 x 20 = 520 rows; 40 x 40 -> one row.
    ink::VisionAdapter real(ink::VisionSpec{}, "sha");
    CHECK(real.canvas(1024, 768) == (vis::Size{1040, 800}));
    CHECK(real.tokens(1024, 768) == 520U);
    CHECK(real.tokens(40, 40) == 1U);
    CHECK(real.max_encoder_rows() == 4096U * 4U + 1U);
    const vis::EncoderConfig ec = real.config();
    CHECK(ec.patch_dim() == 1200U && ec.temporal == 1U && ec.head_dim() == 64U && ec.embed_norm && !ec.final_norm && !ec.patch_bias);
    CHECK(vis::hip_supported(ec));
}

void test_patches_and_layout() {
    const ink::VisionSpec c = toy_config();
    ink::VisionAdapter a(c, "sha");
    const vis::PatchGrid g = a.patches(test_image(10, 6));
    CHECK(g.grid_w == 6U && g.grid_h == 4U);
    CHECK(g.rows == 25U);  // 24 patches + the class row
    CHECK(g.patch_dim == 12U);
    bool zero = true;
    for (unsigned i = 0; i < g.patch_dim; ++i) zero = zero && g.data[std::size_t{24} * g.patch_dim + i] == 0.0F;
    CHECK(zero);
    const vis::SpanLayout s = a.layout(g);
    CHECK(s.rows() == 6U && s.grid_h == 2U && s.grid_w == 3U && s.real_token_id == 200054U);
    CHECK(s.lm_pos[5][0] == 5U);
}

void test_mapping_matches_unfolded_graph() {
    const ink::VisionSpec c = toy_config();
    MapSource src;
    fill_source(src, c);
    ink::VisionAdapter a(c, "sha");
    vis::EncoderWeights w;
    a.bind(src, w);
    const vis::EncoderConfig ec = a.config();
    const vis::PatchGrid g = a.patches(test_image(10, 6));
    const vis::PositionInputs p = a.positions(ec, w, g);
    vis::ReferenceEncoder enc(ec, w);
    vis::EncoderOutput h = enc.encode(g, p);
    vis::ProjectedRows rows;
    a.project(h, g, enc, rows);
    CHECK(rows.rows == 6U && rows.width == c.text_width);

    vis::PositionInputs img_pos = p;  // the image rows' addend (the class row is the last one)
    img_pos.addend.resize(std::size_t{24} * c.width);
    const std::vector<double> want = independent(src, c, g, img_pos);
    CHECK(want.size() == rows.bf16.size());
    double num = 0.0, den = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < want.size() && i < rows.bf16.size(); ++i) {
        const double got = static_cast<double>(vis::from_bf16(rows.bf16[i]));
        num += (got - want[i]) * (got - want[i]);
        den += want[i] * want[i];
        worst = std::max(worst, std::fabs(got - want[i]) / (std::fabs(want[i]) + 1e-3));
    }
    const double rel = std::sqrt(num / den);
    std::printf("mapping vs unfolded graph: relative L2 %.3g, worst relative %.3g (BF16 output rounding)\n", rel, worst);
    CHECK(rel < 4e-3);    // BF16 rounding of the rows: 2^-9 relative per element at most
    CHECK(worst < 8e-3);

    // A slot-major checkpoint read as channel-major (wrong order) must not pass: the test can see the order.
    ink::VisionRecipe wrong;
    wrong.merge_slot_major = false;
    ink::VisionAdapter b(c, "sha", wrong);
    vis::EncoderWeights w2;
    b.bind(src, w2);
    vis::ReferenceEncoder enc2(b.config(), w2);
    vis::EncoderOutput h2 = enc2.encode(g, b.positions(b.config(), w2, g));
    vis::ProjectedRows rows2;
    b.project(h2, g, enc2, rows2);
    double num2 = 0.0;
    for (std::size_t i = 0; i < want.size() && i < rows2.bf16.size(); ++i) {
        const double d = static_cast<double>(vis::from_bf16(rows2.bf16[i])) - want[i];
        num2 += d * d;
    }
    CHECK(std::sqrt(num2 / den) > 1e-2);
}

class ProjectorProbe final : public vis::DenseOps {
public:
    explicit ProjectorProbe(vis::ReferenceEncoder& encoder) : encoder_(encoder) {}
    void linear(const vis::Tensor& w, const vis::Tensor& b, const float* x, std::size_t rows, float* y) override {
        encoder_.linear(w, b, x, rows, y);
    }
    void merge_project(const vis::EncoderOutput& hidden, unsigned h, unsigned w, const vis::MergeProjector& m,
                       float* out) override {
        CHECK(m.layout == vis::MergeProjector::Layout::packed_cells && m.row_offset == 1U);
        ++calls;
        vis::DenseOps::merge_project(hidden, h, w, m, out);
    }
    unsigned calls = 0;

private:
    vis::ReferenceEncoder& encoder_;
};

void test_axial_mapping_matches_unfolded_graph() {
    const auto spec = toy_config();
    MapSource source;
    fill_source(source, spec);
    ink::VisionAdapter adapter(spec, "sha", ink::VisionIdentity::axial_v1);
    const ink::VisionAdapter incumbent(spec, "sha");
    vis::EncoderWeights weights;
    adapter.bind(source, weights);
    const auto config = adapter.config();
    const auto image = test_image(10, 6);
    const auto grid = adapter.patches(image);
    const auto legacy_grid = incumbent.patches(image);
    CHECK(incumbent.config().pos == vis::EncoderConfig::Pos::learned_interp &&
          config.pos == vis::EncoderConfig::Pos::rope2d_plus_learned);
    CHECK(legacy_grid.rows == grid.rows && legacy_grid.data != grid.data);
    const ink::VisionAdapter capped_legacy(spec, "sha", ink::VisionIdentity::legacy, 28);
    const ink::VisionAdapter capped_axial(spec, "sha", ink::VisionIdentity::axial_v1, 28);
    const auto tall = test_image(20, 82);
    CHECK(capped_legacy.layout(capped_legacy.patches(tall)).rows() == 18U);
    CHECK(capped_axial.layout(capped_axial.patches(tall)).rows() == 27U);
    vis::ReferenceEncoder encoder(config, weights);
    auto hidden = encoder.encode(grid, adapter.positions(config, weights, grid));
    ProjectorProbe projector(encoder);
    vis::ProjectedRows got;
    adapter.project(hidden, grid, projector, got);
    CHECK(projector.calls == 1U);
    CHECK(grid.grid_w == 6U && grid.grid_h == 4U && grid.patch_dim == 12U && got.rows == 6U);
    const auto want = independent(source, spec, grid, {}, &image);
    CHECK(want.size() == got.bf16.size());
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < want.size() && i < got.bf16.size(); ++i) {
        const double error = vis::from_bf16(got.bf16[i]) - want[i];
        diff += error * error;
        norm += want[i] * want[i];
    }
    CHECK(norm > 0.0 && std::sqrt(diff / norm) < 4e-3);
}

void test_bind_rejects_shapes() {
    const ink::VisionSpec c = toy_config();
    MapSource src;
    fill_source(src, c);
    src.add("model.visual.proj_out.weight", c.text_width + 1U, c.projection, 0.1F);
    ink::VisionAdapter a(c, "sha");
    vis::EncoderWeights w;
    bool threw = false;
    try {
        a.bind(src, w);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// --checkpoint SHARD: the exact tower inventory of the official file, then bind + host reference on an 80 x 40
// image (2 LM rows, 8 tower rows + the class row). Prints the row RMS; checks shapes and finite values. CPU only.
int probe(const char* shard) {
    const ink::VisionSpec c{};
    const auto specs = ink::vision_tensor_specs(c);
    strix::weight::TensorLookup lookup;
    const strix::weight::TensorSchema schema{strix::weight::Container::safetensors, "safetensors@1", specs,
                                             ink::kVisionIgnoredPrefix};
    const strix::weight::Status st = lookup.open_safetensors(shard, schema);
    if (!st) {
        std::fprintf(stderr, "FAIL inventory: %s\n", st.detail.c_str());
        return 1;
    }
    std::printf("inventory: %zu tower tensors match\n", specs.size());
    vis::LookupWeights src(lookup);
    ink::VisionAdapter a(c, "probe");
    vis::EncoderWeights w;
    a.bind(src, w);
    const vis::PatchGrid g = a.patches(test_image(80, 40));
    const vis::EncoderConfig ec = a.config();
    vis::ReferenceEncoder enc(ec, w);
    vis::EncoderOutput h = enc.encode(g, a.positions(ec, w, g));
    vis::ProjectedRows rows;
    a.project(h, g, enc, rows);
    CHECK(rows.rows == 2U && rows.width == 2304U);
    for (unsigned r = 0; r < rows.rows; ++r) {
        double sq = 0.0;
        bool finite = true;
        for (unsigned i = 0; i < rows.width; ++i) {
            const double v = static_cast<double>(vis::from_bf16(rows.bf16[std::size_t{r} * rows.width + i]));
            finite = finite && std::isfinite(v);
            sq += v * v;
        }
        CHECK(finite);
        std::printf("row %u rms %.5g\n", r, std::sqrt(sq / rows.width));
    }
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    if (argc == 3 && std::string(argv[1]) == "--checkpoint") {
        try {
            return probe(argv[2]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL probe: %s\n", e.what());
            return 1;
        }
    }
    try {
        test_geometry();
        test_patches_and_layout();
        test_mapping_matches_unfolded_graph();
        test_axial_mapping_matches_unfolded_graph();
        test_bind_rejects_shapes();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("inkling2 vision adapter: all tests passed");
    return 0;
}
