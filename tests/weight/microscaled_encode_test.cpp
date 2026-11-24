// Host test of the NVFP4 encoder and the MXFP8 -> NVFP4 transcode (strix/weight/formats/microscaled_encode.hpp):
// (1) NVFP4-representable rows round-trip bitwise in value; (2) the mse@2 block scale is never worse than the amax rule
// alone and visits its neighbors; (3) finite scale limits and midpoint rounding; (4) MXFP8 matrix error matches a
// recomputation; (5) slab layout, validation and worker failure handling.
#include "strix/weight/formats/microscaled_encode.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace ms = strix::weight::microscaled;
namespace sf = strix::weight::scalar_float;
namespace w = strix::weight;

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        std::fflush(stdout);
        ++failures;
    }
}

float decoded(const std::vector<std::uint8_t>& codes, const std::vector<std::uint8_t>& scales, float outer, std::size_t c) {
    const auto byte = codes[c / 2U];
    const auto code = static_cast<std::uint8_t>(c % 2U == 0U ? byte & 0x0FU : byte >> 4U);
    return ms::nvfp4_value(code, sf::e4m3(scales[c / 16U]), outer);
}

void representable(std::mt19937& rng) {
    constexpr std::size_t n = 256;
    std::uniform_int_distribution<int> code(0, 15), scale(0x10, 0x60);
    std::vector<float> x(n);
    for (std::size_t b = 0; b < n / 16U; ++b) {
        const auto s = static_cast<std::uint8_t>(scale(rng));
        for (std::size_t i = 0; i < 16U; ++i) {
            // One element per block at |code| 6 makes amax / 6 the block scale exactly.
            const auto c = i == 3U ? std::uint8_t{7} : static_cast<std::uint8_t>(code(rng));
            x[b * 16U + i] = sf::e2m1(c) * sf::e4m3(s);
        }
    }
    std::vector<std::uint8_t> codes(n / 2U), scales(n / 16U);
    ms::encode_nvfp4_row(x, 1.0F, codes, scales);
    bool same = true;
    for (std::size_t c = 0; c < n; ++c) same = same && decoded(codes, scales, 1.0F, c) == x[c];
    expect(same, "representable row round-trips");
}

void mse_rule(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    double mse_total = 0.0, amax_total = 0.0;
    const float outer = 0.01F;
    for (int trial = 0; trial < 2000; ++trial) {
        float block[16];
        float amax = 0.0F;
        for (float& v : block) {
            v = nd(rng) * (trial % 7 == 0 ? 0.01F : 1.0F);
            amax = std::fmax(amax, std::fabs(v));
        }
        std::uint8_t codes[8], scale[1];
        ms::encode_nvfp4_row(std::span<const float>(block, 16), outer, codes, scale);
        const double mse = ms::nvfp4_block_error(block, outer, scale[0], nullptr);
        const auto rule = static_cast<std::uint8_t>(sf::e4m3_encode(amax / (6.0F * outer)) & 0x7F);
        const double amax_error = ms::nvfp4_block_error(block, outer, rule, nullptr);
        expect(mse <= amax_error, "mse@2 never worse than the amax rule");
        mse_total += mse;
        amax_total += amax_error;
    }
    std::printf("block SSE mse@2 %.4f amax rule %.4f (ratio %.3f)\n", mse_total, amax_total, mse_total / amax_total);
}

void fixed_mse_rule() {
    std::array<float, 16> block;
    block.fill(1.375F);
    block[0] = 6.0F;
    // The amax center is 1.0. Score its five candidates without the production encoder or decoder.
    constexpr double candidates[] = {0.875, 0.9375, 1.0, 1.125, 1.25};
    constexpr double levels[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    std::array<double, 5> errors{};
    for (std::size_t s = 0; s < errors.size(); ++s) {
        for (const float value : block) {
            double best = std::numeric_limits<double>::infinity();
            for (const double level : levels) {
                const double d = level * candidates[s] - double{value};
                best = std::min(best, d * d);
            }
            errors[s] += best;
        }
    }
    expect(std::min_element(errors.begin(), errors.end()) == errors.begin() + 1 && errors[1] < errors[2],
           "independent SSE has a strict neighbor winner");
    std::array<std::uint8_t, 8> codes{};
    std::array<std::uint8_t, 1> scale{};
    ms::encode_nvfp4_row(block, 1.0F, codes, scale);
    expect(scale[0] == 0x37 && codes == std::array<std::uint8_t, 8>{0x37, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33},
           "mse@2 selects and packs the strict neighbor winner");

    block.fill(0.0F);
    block[0] = 5.8125F;  // Exactly halfway between 6 * 0.9375 and 6 * 1.0.
    ms::encode_nvfp4_row(block, 1.0F, codes, scale);
    expect(scale[0] == 0x37 && codes == std::array<std::uint8_t, 8>{0x07}, "equal SSE selects the lower scale byte");
}

void finite_edges() {
    std::array<float, 16> block{};
    std::array<std::uint8_t, 8> codes{};
    std::array<std::uint8_t, 1> scale{};
    const float tiny = std::numeric_limits<float>::denorm_min();
    block[0] = tiny;
    block[1] = -tiny;
    expect(ms::nvfp4_outer_scale(tiny) == tiny, "positive outer scale survives underflow");
    ms::encode_nvfp4_row(block, tiny, codes, scale);
    expect(scale[0] == 0x21 && codes == std::array<std::uint8_t, 8>{0xf7} &&
               ms::nvfp4_value(7, sf::e4m3(scale[0]), tiny) == tiny &&
               ms::nvfp4_value(15, sf::e4m3(scale[0]), tiny) == -tiny,
           "decoder SSE preserves signed FP32 subnormals");

    const float huge = std::numeric_limits<float>::max();
    block[0] = huge;
    block[1] = -huge;
    ms::encode_nvfp4_row(block, huge, codes, scale);
    expect(scale[0] == 0x22 && codes == std::array<std::uint8_t, 8>{0xf7}, "finite outer scale avoids FP32 center overflow");

    block[0] = 3072.0F;
    block[1] = -3072.0F;
    ms::encode_nvfp4_row(block, 1.0F, codes, scale);
    expect(scale[0] == 0x7e && codes == std::array<std::uint8_t, 8>{0xf7}, "scale and both E2M1 signs saturate");

    // Amax / 6 is halfway between 0x37 and 0x38. Using the lower center admits 0x35,
    // which ties 0x39 in SSE and would win by scale byte. All decoded products are exact here.
    block.fill(2.34375F);
    block[0] = 5.8125F;
    const int saved = std::fegetround();
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        expect(std::fesetround(mode) == 0, "set host rounding mode");
        ms::encode_nvfp4_row(block, 1.0F, codes, scale);
        expect(scale[0] == 0x39 && codes == std::array<std::uint8_t, 8>{0x47, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44},
               "E4M3 center midpoint rounds to even in every host rounding mode");
    }
    expect(std::fesetround(saved) == 0, "restore host rounding mode");
}

void transcode(std::mt19937& rng) {
    constexpr std::size_t rows = 64, cols = 1152;
    std::vector<std::uint8_t> codes(rows * cols), scales(rows * cols / 32U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> e(118, 124);
    for (auto& s : scales) s = static_cast<std::uint8_t>(e(rng));
    for (auto& c : codes) c = sf::e4m3_encode(nd(rng) * 64.0F);
    const auto out = ms::transcode_mxfp8_to_nvfp4(codes, scales, rows, cols);
    // Recompute the error from the packed bytes.
    double sse = 0.0, energy = 0.0;
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) {
            const double src = double{sf::e4m3(codes[r * cols + c])} * double{sf::e8m0(scales[r * (cols / 32U) + c / 32U])};
            const auto byte = out.codes[(r * cols + c) / 2U];
            const auto code = static_cast<std::uint8_t>(c % 2U == 0U ? byte & 0x0FU : byte >> 4U);
            const double d = double{ms::nvfp4_value(code, sf::e4m3(out.scales[r * (cols / 16U) + c / 16U]), out.outer)} - src;
            sse += d * d;
            energy += src * src;
        }
    const double rel = std::sqrt(sse / energy);
    std::printf("MXFP8 -> NVFP4 relative RMS error %.4f (reported %.4f), outer %g\n", rel, out.error.relative(),
                double{out.outer});
    expect(std::fabs(rel - out.error.relative()) < 1e-9, "reported error matches");
    expect(rel > 0.0 && rel < 0.15, "4-bit error band");
}

void malformed() {
    bool threw = false;
    try {
        std::vector<float> x(15, 1.0F);
        std::vector<std::uint8_t> c(7), s(1);
        ms::encode_nvfp4_row(x, 1.0F, c, s);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "short row throws");
    threw = false;
    try {
        std::vector<float> x(16, 1.0F);
        x[5] = INFINITY;
        std::vector<std::uint8_t> c(8), s(1);
        ms::encode_nvfp4_row(x, 1.0F, c, s);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "non-finite value throws");
    threw = false;
    try {
        std::vector<std::uint8_t> c(64), s(2);
        c[0] = 0x7F;  // E4M3 NaN
        (void)ms::transcode_mxfp8_to_nvfp4(c, s, 1, 64);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "inconsistent or NaN MXFP8 throws");
}

// (6) A MXFP8 expert slab read from its source: every expert equals the one-matrix transcode, the bytes do not depend on
// the thread count, the device-style binding matches a checkpoint NVFP4 slab (strides, outer per expert, row order), a
// NVFP4 slab and a NaN source are refused.
void slab(std::mt19937& rng) {
    constexpr std::size_t experts = 5, rows = 6, cols = 128;
    std::vector<std::uint8_t> bytes(experts * rows * cols + experts * rows * cols / 32U);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<int> e(116, 126);
    const std::size_t code_bytes = experts * rows * cols;
    for (std::size_t i = 0; i < code_bytes; ++i) bytes[i] = sf::e4m3_encode(nd(rng) * 32.0F);
    for (std::size_t i = code_bytes; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(e(rng));
    const auto source = w::memory_source(std::as_bytes(std::span(bytes)));
    const ms::ExpertSlabView view{ms::MatrixFormat::mxfp8, ms::RowOrder::interleaved_halves, experts, rows, cols,
                                  {source, {0, code_bytes}}, {source, {code_bytes, bytes.size() - code_bytes}}, {}, {}};
    ms::Nvfp4Slab one, many;
    expect(static_cast<bool>(ms::transcode_slab_to_nvfp4(view, one, 1U)), "slab transcode, one thread");
    expect(static_cast<bool>(ms::transcode_slab_to_nvfp4(view, many, 3U)), "slab transcode, three threads");
    expect(one.codes == many.codes && one.scales == many.scales && one.outer == many.outer, "thread-count invariant");
    double sse = 0.0, energy = 0.0;
    for (std::size_t x = 0; x < experts; ++x) {
        const std::span<const std::uint8_t> codes(bytes.data() + x * rows * cols, rows * cols);
        const std::span<const std::uint8_t> scales(bytes.data() + code_bytes + x * rows * cols / 32U, rows * cols / 32U);
        const auto m = ms::transcode_mxfp8_to_nvfp4(codes, scales, rows, cols);
        expect(std::equal(m.codes.begin(), m.codes.end(), one.codes.begin() + static_cast<std::ptrdiff_t>(x * one.code_stride())),
               "expert codes equal the matrix transcode");
        expect(std::equal(m.scales.begin(), m.scales.end(),
                          one.scales.begin() + static_cast<std::ptrdiff_t>(x * one.scale_stride())),
               "expert scales equal the matrix transcode");
        expect(m.outer == one.outer[x], "outer per expert");
        sse += m.error.sse;
        energy += m.error.energy;
    }
    expect(one.error.sse == sse && one.error.energy == energy, "slab error is the expert sum in order");
    std::printf("slab relative RMS error %.4f\n", one.error.relative());

    // Binding: strides and metadata of a checkpoint NVFP4 slab of the same geometry.
    std::vector<ms::PackedMatrix> matrices(experts);
    expect(static_cast<bool>(one.bind_storage(one.codes, one.scales, matrices)), "bind");
    for (std::size_t x = 0; x < experts; ++x) {
        const auto& m = matrices[x];
        expect(m.codes == one.codes.data() + x * rows * cols / 2U && m.block_scales == one.scales.data() + x * rows * cols / 16U &&
                   m.rows == rows && m.columns == cols && m.outer_scale == one.outer[x] &&
                   m.row_order == ms::RowOrder::interleaved_halves,
               "binding layout");
    }
    expect(!one.bind_storage(std::span(one.codes).first(one.codes.size() - 1U), one.scales, matrices), "short storage refused");

    ms::Nvfp4Slab refused;
    ms::ExpertSlabView fp4 = view;
    fp4.format = ms::MatrixFormat::nvfp4;
    expect(!ms::transcode_slab_to_nvfp4(fp4, refused), "NVFP4 slab refused");
    bytes[3 * rows * cols + 7] = 0x7F;  // E4M3 NaN in expert 3 (memory_source keeps its own copy: a new source)
    ms::ExpertSlabView nan_view = view;
    nan_view.codes.source = nan_view.block_scales.source = w::memory_source(std::as_bytes(std::span(bytes)));
    const auto status = ms::transcode_slab_to_nvfp4(nan_view, refused, 2U);
    expect(!status && refused.codes.empty(), "NaN source refused, output untouched");
}

class SlabSource final : public w::Source {
public:
    explicit SlabSource(std::uint64_t bytes) : bytes_(bytes), caller_(std::this_thread::get_id()) {}

    w::Status read(w::ByteRange, std::span<w::Byte>) const override {
        throw std::runtime_error("injected source read failure");
    }
    std::uint64_t size() const noexcept override {
        if (std::this_thread::get_id() == caller_) checked_on_caller = true;
        return bytes_;
    }
    w::Digest256 identity() const override { return {}; }

    mutable std::atomic<bool> checked_on_caller{false};

private:
    std::uint64_t bytes_;
    std::thread::id caller_;
};

void slab_extents() {
    using enum w::Error;
    constexpr auto limit = std::numeric_limits<std::size_t>::max();
    ms::ExpertSlabView view{ms::MatrixFormat::mxfp8, ms::RowOrder::contiguous, limit / 2U + 1U, 1, 32, {}, {}, {}, {}};
    ms::Nvfp4Slab out;
    try {
        expect(ms::transcode_slab_to_nvfp4(view, out, 1U).code == unsupported_shape, "overflowing slab extents refused");
    } catch (const std::exception&) {
        expect(false, "overflowing slab reached allocation");
    }

    view.experts = 2;
    const auto source = std::make_shared<SlabSource>(65);
    view.codes = {std::make_shared<SlabSource>(64), {0, 64}};
    view.block_scales = {source, {64, 2}};
    expect(ms::transcode_slab_to_nvfp4(view, out, 2U).code == malformed && source->checked_on_caller,
           "short physical source range rejected during caller validation");
    view.block_scales.range.bytes = 1;
    expect(ms::transcode_slab_to_nvfp4(view, out, 2U).code == malformed, "short declared scale slice refused");
    view.rows = 0;
    expect(ms::transcode_slab_to_nvfp4(view, out, 2U).code == unsupported_shape, "zero-row source refused");

    ms::Nvfp4Slab incomplete;
    incomplete.experts = 1;
    incomplete.columns = 32;
    incomplete.outer = {1.0F};
    ms::PackedMatrix bound[1]{};
    expect(!incomplete.bind_storage({}, {}, bound), "zero-row result cannot bind");
    incomplete.rows = limit / 2U + 1U;
    expect(!incomplete.bind_storage({}, {}, bound), "wrapping result geometry cannot bind");
    incomplete.rows = 1;
    incomplete.row_order = ms::RowOrder::interleaved_halves;
    incomplete.codes.resize(16);
    incomplete.scales.resize(2);
    expect(!incomplete.bind_storage(incomplete.codes, incomplete.scales, bound), "invalid result row order cannot bind");
}

void slab_source_failure() {
    const auto source = std::make_shared<SlabSource>(66);
    const ms::ExpertSlabView view{ms::MatrixFormat::mxfp8, ms::RowOrder::contiguous, 2, 1, 32,
                                  {source, {0, 64}}, {source, {64, 2}}, {}, {}};
    ms::Nvfp4Slab out;
    out.rows = 3;
    out.codes = {0x5a};
    out.error.sse = 7.0;
    const auto status = ms::transcode_slab_to_nvfp4(view, out, 2U);
    expect(!status && status.detail.find("injected source read failure") != std::string::npos,
           "throwing source returns a diagnostic to the caller");
    expect(out.rows == 3 && out.codes == std::vector<std::uint8_t>{0x5a} && out.error.sse == 7.0,
           "worker failure preserves the previous output");
}

}  // namespace

int main() {
    std::mt19937 rng(1123U);
    representable(rng);
    mse_rule(rng);
    fixed_mse_rule();
    finite_edges();
    transcode(rng);
    malformed();
    slab(rng);
    slab_extents();
    slab_source_failure();
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
