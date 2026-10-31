#include "fixture.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

namespace df = strix::gemma4::dflash;
namespace g4 = strix::gemma4;
using fixture::check;

void silu_boundaries() {
    const auto bits = [](float x) { return std::bit_cast<std::uint32_t>(x); };
    const float below = std::nextafter(-88.0F, -std::numeric_limits<float>::infinity());
    const float above = std::nextafter(-88.0F, 0.0F);
    check(std::isfinite(g4::declared_exp(88.0F)) && std::isinf(g4::declared_exp(-below)),
          "declared exponential saturation boundary changed");
    for (float gate : {below, -88.288620F, -100.0F, -std::numeric_limits<float>::max()}) {
        check(bits(df::declared_silu(gate)) == 0x80000000U, "negative saturation must preserve signed zero");
        const float ieee = gate / (1.0F + g4::declared_exp(-gate));
        check(bits(df::declared_silu(gate)) == bits(ieee), "saturation differs from finite/+inf limit");
    }
    for (float gate : {-88.0F, above, -20.0F, -1.0F, -0.0F, 0.0F, 1.0F, 88.0F,
                       std::nextafter(103.0F, 0.0F), 103.0F,
                       std::nextafter(103.0F, std::numeric_limits<float>::infinity()),
                       std::numeric_limits<float>::max()}) {
        const float old = gate / (1.0F + g4::declared_exp(-gate));
        check(std::isfinite(df::declared_silu(gate)) && bits(df::declared_silu(gate)) == bits(old),
              "successful v1 SiLU bits changed");
    }
    check(df::declared_silu(-88.0F) < 0.0F && bits(df::declared_silu(-0.0F)) == 0x80000000U,
          "finite boundary or signed zero changed");
    for (float gate : {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()})
        check(std::isnan(df::declared_silu(gate)), "nonfinite gate must fail, never saturate");
}

void saturated_forward() {
    auto s = fixture::shape(); s.layers = 1;
    auto w = fixture::weights(s);
    auto& l = w.layer.front();
    // Constant positive residual, zero attention branch, and a finite negative gate.
    // All FFN lanes saturate; the branch must be exactly zero, leaving the residual.
    std::fill(l.o.bits.begin(), l.o.bits.end(), 0);
    std::fill(l.gate.bits.begin(), l.gate.bits.end(), g4::bf16_bits(-16.0F));
    const std::vector<float> input(df::block * s.hidden, 1.0F);
    fixture::Tables rope(df::block, s.head_dim);
    df::DraftContext context(s);
    const auto saturated = df::block_forward(w, context, rope.view(), input, 0);
    df::ForwardDiagnostics diagnostics;
    check(saturated == df::block_forward(w, context, rope.view(), input, 0, &diagnostics),
          "gate observations must preserve forward bits");
    check(diagnostics.gates.size() == 1 && diagnostics.gates[0].maximum < -88.0F &&
          diagnostics.gates[0].negative_saturated == df::block * s.inter,
          "saturation diagnostics must observe every finite negative gate");
    std::fill(l.down.bits.begin(), l.down.bits.end(), 0);
    check(saturated == df::block_forward(w, context, rope.view(), input, 0),
          "finite saturated FFN must leave only the residual");
    l.gate.bits[0] = 0x7f80U;
    fixture::rejects([&] { (void)df::block_forward(w, context, rope.view(), input, 0); });
    l.gate.bits[0] = 0x7fc0U;
    fixture::rejects([&] { (void)df::block_forward(w, context, rope.view(), input, 0); });
}

std::vector<float> taps(std::size_t rows, std::size_t width) {
    std::vector<float> out(rows * width);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<float>(static_cast<int>((i * 3) % 29) - 14) / 16.0F;
    return out;
}
std::vector<float> embedded(std::size_t width) {
    std::vector<float> out(df::block * width);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<float>(static_cast<int>((i * 5) % 23) - 11) / 8.0F;
    return out;
}
std::vector<std::uint64_t> positions(std::size_t rows, std::uint64_t start = 0) {
    std::vector<std::uint64_t> out(rows);
    std::iota(out.begin(), out.end(), start);
    return out;
}
void close(std::span<const float> got, std::span<const float> expected) {
    check(got.size() == expected.size(), "fixture length mismatch");
    for (std::size_t i = 0; i < got.size(); ++i)
        check(std::abs(got[i] - expected[i]) <= 0.00002F, "independent PyTorch semantic fixture mismatch");
}

void golden(const std::filesystem::path& dir, const df::RopeFile& rope) {
    const auto w = fixture::weights();
    df::DraftContext context(w.shape);
    const auto features = taps(3, w.shape.tap_width());
    const auto pos = positions(3);
    df::ContextAppender appender(w, context, rope.view());
    g4::DraftContext& committed_context = appender;
    committed_context.append(features, pos);
    const auto input = embedded(w.shape.hidden);
    const auto got = df::block_forward(w, context, rope.view(), input, 3);
    std::ifstream in(dir / "torch-forward.f32", std::ios::binary);
    std::vector<float> expected(128 + 2 * 3 * 8 * 2);
    in.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(expected.size() * sizeof(float)));
    check(static_cast<bool>(in), "missing semantic fixture");
    close(got, std::span<const float>(expected).first(128));
    std::size_t offset = 128;
    for (std::uint32_t l = 0; l < w.shape.layers; ++l)
        for (std::uint64_t p = 0; p < 3; ++p) { close(context.key(l, p), std::span<const float>(expected).subspan(offset, 8)); offset += 8; }
    for (std::uint32_t l = 0; l < w.shape.layers; ++l)
        for (std::uint64_t p = 0; p < 3; ++p) { close(context.value(l, p), std::span<const float>(expected).subspan(offset, 8)); offset += 8; }
    const auto repeat = df::block_forward(w, context, rope.view(), input, 3);
    check(got == repeat && context.frontier() == 3, "block published temporary K/V");
    auto bad = input;
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    fixture::rejects([&] { (void)df::block_forward(w, context, rope.view(), bad, 3); });
    bad[0] = std::numeric_limits<float>::max();
    fixture::rejects([&] { (void)df::block_forward(w, context, rope.view(), bad, 3); });
    fixture::rejects([&] { (void)df::block_forward(w, context, rope.view(), std::span<const float>(input).first(16), 3); });
    fixture::rejects([&] { context.append(w, rope.view(), features, pos); });
    check(context.frontier() == 3 && got == df::block_forward(w, context, rope.view(), input, 3), "failed append corrupted context");
}

void wrap_and_masks(const df::RopeFile& rope) {
    auto s = fixture::shape(); s.layers = 1;
    const auto w = fixture::weights(s);
    const auto features = taps(2056, s.tap_width());
    const auto pos = positions(2056);
    df::DraftContext batched(s), serial(s), tail(s);
    batched.append(w, rope.view(), features, pos);
    for (std::size_t r = 0; r < pos.size(); ++r)
        serial.append(w, rope.view(), std::span<const float>(features).subspan(r * s.tap_width(), s.tap_width()),
                      std::span<const std::uint64_t>(pos).subspan(r, 1));
    tail.append(w, rope.view(), std::span<const float>(features).subspan(1032 * s.tap_width()),
                std::span<const std::uint64_t>(pos).subspan(1032));
    check(!batched.contains(1031) && batched.contains(1032) && batched.contains(2055), "1024-entry ring wrap");
    for (std::uint64_t p = 1032; p < 2056; ++p) {
        check(std::equal(batched.key(0, p).begin(), batched.key(0, p).end(), serial.key(0, p).begin()), "append grouping changed K bytes");
        check(std::equal(batched.value(0, p).begin(), batched.value(0, p).end(), serial.value(0, p).begin()), "append grouping changed V bytes");
    }
    const auto input = embedded(s.hidden);
    const auto first = df::block_forward(w, batched, rope.view(), input, 2056);
    check(first == df::block_forward(w, serial, rope.view(), input, 2056) &&
          first == df::block_forward(w, tail, rope.view(), input, 2056), "wrapped/tail block mismatch");
    fixture::rejects([&] { (void)batched.key(0, 1031); });
    auto bad = taps(1, s.tap_width()); bad[1] = std::numeric_limits<float>::infinity();
    const std::array<std::uint64_t, 1> next{2056};
    fixture::rejects([&] { batched.append(w, rope.view(), bad, next); });
    check(first == df::block_forward(w, batched, rope.view(), input, 2056), "nonfinite append published state");

    auto context_taps = taps(1024, s.tap_width());
    const auto context_pos = positions(1024);
    df::DraftContext original(s), perturbed(s);
    original.append(w, rope.view(), context_taps, context_pos);
    for (std::size_t i = s.tap_width(); i < 2 * s.tap_width(); ++i) context_taps[i] = -context_taps[i];
    perturbed.append(w, rope.view(), context_taps, context_pos);
    const auto a = df::block_forward(w, original, rope.view(), input, 1024);
    const auto b = df::block_forward(w, perturbed, rope.view(), input, 1024);
    check(!std::equal(a.begin(), a.begin() + s.hidden, b.begin()), "row 0 did not see oldest allowed context key");
    check(std::equal(a.begin() + s.hidden, a.end(), b.begin() + s.hidden), "later query saw masked context key");
    auto future = input;
    for (std::size_t i = 7 * s.hidden; i < future.size(); ++i) future[i] = -future[i];
    const auto future_result = df::block_forward(w, original, rope.view(), future, 1024);
    check(!std::equal(a.begin(), a.begin() + s.hidden, future_result.begin()), "block attention was incorrectly causal");
    for (std::uint32_t row = 0; row < df::block; ++row) {
        const auto visible = df::visible_positions(1024, row);
        check(visible.front() == row + 1 && visible.back() == 1031 && visible.size() == 1031 - row,
              "query-relative strict context mask");
    }
    check(!df::context_visible(1024, 1024, 0) && df::context_visible(1024, 1024, 1), "strict left boundary");
    df::DraftContext missing(s);
    const auto tail_features = taps(1, s.tap_width()); const std::array<std::uint64_t, 1> last{1023};
    missing.append(w, rope.view(), tail_features, last);
    fixture::rejects([&] { (void)df::block_forward(w, missing, rope.view(), input, 1024); });
}

void target_assets() {
    auto w = fixture::weights();
    const auto width = w.shape.hidden;
    std::vector<float> table(std::size_t{8} * width, 0.0F);
    table[width] = 0.5F; table[2 * width] = -0.5F; table[4 * width] = 0.25F;
    g4::LinearView view{reinterpret_cast<const std::uint8_t*>(table.data()), 8, width,
                       std::size_t{width} * sizeof(float), g4::WeightFormat::f32};
    const auto input = df::embed_block(1, view);
    check(input[0] == 25.25F && input[width] == 12.625F, "target BF16 embedding boundary or mask ID");
    std::vector<float> hidden(std::size_t{df::block} * width, 0.0F);
    hidden[0] = -100.0F; // poison anchor row; proposals must start at row 1
    for (std::size_t r = 1; r < df::block; ++r) hidden[r * width] = r % 2 ? 2.0F : -2.0F;
    const auto tokens = df::proposals_from_hidden(w, hidden, view);
    for (std::size_t i = 0; i < tokens.size(); ++i) check(tokens[i] == (i % 2 ? 2U : 1U), "head row alignment");
    std::fill(table.begin(), table.end(), 0.0F);
    const auto ties = df::proposals_from_hidden(w, hidden, view);
    check(std::all_of(ties.begin(), ties.end(), [](auto t) { return t == 0; }), "head tie order");
    table[width] = 1000.0F; table[2 * width] = 2000.0F; w.logit_cap = 1.0F;
    const auto capped = df::proposals_from_hidden(w, hidden, view);
    check(capped[0] == 1, "draft metadata cap ties must precede argmax");
    table[width] = std::numeric_limits<float>::quiet_NaN();
    fixture::rejects([&] { (void)df::proposals_from_hidden(w, hidden, view); });

    // Nontrivial byte stride and packed target-head access, using exactly representable coefficients.
    w.shape.hidden = 32; w.logit_cap = 0.0F;
    std::vector<std::uint8_t> bf16(8 * 68, 0), q8(8 * 40, 0);
    const std::array<std::uint8_t, 8> codes{0, 2, 254, 0, 1, 0, 0, 0};
    for (std::size_t row = 0; row < 8; ++row) {
        const float value = static_cast<float>(row == 2 ? -2 : codes[row]) / 4.0F;
        const auto bits = g4::bf16_bits(value);
        bf16[row * 68] = static_cast<std::uint8_t>(bits);
        bf16[row * 68 + 1] = static_cast<std::uint8_t>(bits >> 8);
        q8[row * 40] = 0; q8[row * 40 + 1] = 0x34; // FP16 0.25
        q8[row * 40 + 2] = codes[row];
    }
    g4::LinearView bview{bf16.data(), 8, 32, 68, g4::WeightFormat::bf16};
    g4::LinearView qview{q8.data(), 8, 32, 40, g4::WeightFormat::q8_0};
    hidden.assign(8 * 32, 0.0F);
    for (std::size_t row = 1; row < 8; ++row) hidden[row * 32] = row % 2 ? 2.0F : -2.0F;
    check(df::proposals_from_hidden(w, hidden, bview) == tokens &&
          df::proposals_from_hidden(w, hidden, qview) == tokens, "packed target head or byte stride");
    check(df::embed_block(1, bview) == df::embed_block(1, qview), "packed target embedding boundary");
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "fixture directory required");
        const std::filesystem::path dir(argv[1]);
        const auto rope = df::RopeFile::open(dir / "rope8-p0-p2063.g4dr");
        silu_boundaries(); saturated_forward();
        golden(dir, rope); wrap_and_masks(rope); target_assets();
        const auto real_shape_rope = df::RopeFile::open(dir / "rope128-p0-p8.g4dr");
        check(real_shape_rope.positions == 9 && real_shape_rope.head_dim == 128 &&
              real_shape_rope.cosine[0] == 1.0F && real_shape_rope.sine[0] == 0.0F, "probe rotary table");
        std::cout << "DFlash forward semantics, mask, ring and target assets passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
