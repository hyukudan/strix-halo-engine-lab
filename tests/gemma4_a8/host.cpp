// g4-a8-v1 host contract: quantization rule, packed group decode, declared order and row invariance.
#include "gemma4/kernels/a8_host.hpp"
#include "../gemma4_gpu/fixture.hpp"
#include <cmath>

using namespace strix::gemma4;
namespace {
void quantization_tests() {
    std::int8_t codes[32];
    float x[32] = {};
    auto meta = a8::quantize_group(x, codes);
    CHECK(meta.delta == 1.0F && meta.sum == 0);
    for (auto c : codes)
        CHECK(c == 0);
    // An underflowing delta is the zero group, not a division by zero.
    x[3] = bit_cast<float>(1U);
    meta = a8::quantize_group(x, codes);
    CHECK(meta.delta == 1.0F && meta.sum == 0 && codes[3] == 0);
    // amax maps to +-127; exact half-way quotients round to even.
    for (unsigned i = 0; i < 32; ++i)
        x[i] = 0.0F;
    x[0] = -127.0F;
    x[1] = 0.5F;
    x[2] = 1.5F;
    x[3] = 2.5F;
    x[4] = -0.5F;
    x[5] = 126.5F;
    x[6] = -0.0F;
    meta = a8::quantize_group(x, codes);
    CHECK(meta.delta == 1.0F);
    CHECK(codes[0] == -127 && codes[1] == 0 && codes[2] == 2 && codes[3] == 2 && codes[4] == 0 && codes[5] == 126);
    CHECK(meta.sum == -127 + 2 + 2 + 126);
    // A subnormal delta can exceed 127 after rounding: the clamp is part of the rule.
    for (unsigned i = 0; i < 32; ++i)
        x[i] = 0.0F;
    x[7] = std::ldexp(1.0F, -140);
    x[8] = -std::ldexp(1.0F, -140);
    meta = a8::quantize_group(x, codes);
    CHECK(meta.delta == std::ldexp(4.0F, -149));
    CHECK(codes[7] == 127 && codes[8] == -127 && meta.sum == 0);
    // Ordinary data: every code within +-127 and the sum matches.
    std::mt19937 rng(7);
    for (unsigned trial = 0; trial < 2000; ++trial) {
        const int exponent = int(rng() % 40) - 20;
        for (auto& v : x)
            v = std::ldexp(float(int(rng() % 20001) - 10000), exponent - 13);
        meta = a8::quantize_group(x, codes);
        int sum = 0, peak = 0;
        for (unsigned i = 0; i < 32; ++i) {
            CHECK(codes[i] >= -127 && codes[i] <= 127);
            sum += codes[i];
            peak = std::max(peak, std::abs(int(codes[i])));
            CHECK(std::fabs(double(codes[i]) * meta.delta - x[i]) <= 0.50001 * meta.delta);
        }
        CHECK(sum == meta.sum);
        CHECK(peak == 127);
    }
}
// Every group term against delta * sum(a_i * w_i), w_i from the g4-v1 scalar decoders.
void decode_tests() {
    std::mt19937 rng(11);
    for (const auto f : {WeightFormat::q8_0, WeightFormat::q4_k, WeightFormat::q6_k}) {
        const unsigned width = 512, outputs = 9;
        const auto bytes = fixture::packed(f, outputs, width, 300U + unsigned(f));
        const LinearView w{bytes.data(), outputs, width, packed_row_bytes(f, width), f};
        std::vector<float> x(width);
        for (auto& v : x)
            v = float(int(rng() % 4001) - 2000) * 0.001F;
        std::vector<std::int8_t> codes;
        std::vector<a8::Meta> meta;
        a8::quantize_rows(x.data(), 1, width, codes, meta);
        for (unsigned o = 0; o < outputs; ++o)
            for (unsigned g = 0; g < width / 32U; ++g) {
                // Q4_K terms subtract dmin*m*S: bound the rounding by the unsubtracted magnitudes too.
                const auto* row = bytes.data() + std::size_t(o) * w.row_bytes;
                const double offset =
                    f == WeightFormat::q4_k ? 2.0 * 63.0 * std::fabs(fp16_value(load16(row + (g / 8U) * 144U + 2U))) : 0.0;
                double reference = 0.0, magnitude = 0.0;
                for (unsigned i = 0; i < 32; ++i) {
                    const double wi = weight_at(w, o, g * 32U + i);
                    reference += double(codes[g * 32U + i]) * wi;
                    magnitude += std::fabs(double(codes[g * 32U + i])) * (std::fabs(wi) + offset);
                }
                reference *= meta[g].delta;
                magnitude *= meta[g].delta;
                const float term = a8::group_term(w, o, g, codes.data() + g * 32U, meta[g]);
                CHECK(std::fabs(double(term) - reference) <= 2e-6 * magnitude + 1e-30);
            }
    }
}
void order_tests() {
    // Terms chosen so that any other grouping of the 16 slots or of their tree changes the bits.
    float terms[40];
    for (unsigned g = 0; g < 40; ++g)
        terms[g] = g == 0 ? 16777216.0F : 1.0F + float(g % 3) * 0.5F;
    float v[16] = {};
    for (unsigned p = 0; p < 16; ++p)
        for (unsigned g = p; g < 40; g += 16)
            v[p] = v[p] + terms[g];
    for (unsigned d = 8; d; d >>= 1U)
        for (unsigned l = 0; l < d; ++l)
            v[l] = v[l] + v[l + d];
    const float declared = a8::slot_dot([&](std::uint32_t g) { return terms[g]; }, 40);
    CHECK(bit_cast<std::uint32_t>(declared) == bit_cast<std::uint32_t>(v[0]));
    float serial = 0.0F;
    for (float t : terms)
        serial = serial + t;
    CHECK(bit_cast<std::uint32_t>(declared) != bit_cast<std::uint32_t>(serial));
}
template <class Stream>
ForwardResult ingest(Target& target, Stream& stream, unsigned n, unsigned token, bool permute = false) {
    auto tx = stream.begin();
    std::vector<RowPlan> plans;
    for (unsigned r = 0; r < n; ++r)
        plans.push_back({13, 7, stream.frontier() + r, (token + r) % 37, tx.get(), r, true});
    if (permute)
        std::reverse(plans.begin(), plans.end());
    const auto out = target.forward_rows(plans);
    tx->commit(n);
    return out;
}
void graph_tests() {
    const auto model = fixture::tiny(6, true);
    fixture::Tables tables;
    const auto weights = bind_host_weights(model);
    HostTarget exact(weights, tables.view(), 4), host(weights, tables.view(), 4, a8::host_recipe());
    HostStream batch(model.config), serial(model.config), reference(model.config);
    bool differs = false;
    for (unsigned n = 1; n <= 8; ++n) {
        const auto out = ingest(host, batch, n, 2 + n, true);
        const auto v1 = ingest(exact, reference, n, 2 + n);
        for (unsigned r = 0; r < n; ++r) {
            const auto single = ingest(host, serial, 1, 2 + n + r);
            bytes(single.logits, std::vector<float>(out.logits.begin() + std::ptrdiff_t(r) * model.config.vocab,
                                                    out.logits.begin() + std::ptrdiff_t(r + 1) * model.config.vocab));
            bytes(single.taps, std::vector<float>(out.taps.begin() + std::ptrdiff_t(r) * out.tap_width,
                                                  out.taps.begin() + std::ptrdiff_t(r + 1) * out.tap_width));
        }
        CHECK(batch.digest() == serial.digest());
        for (std::size_t i = 0; i < out.logits.size(); ++i)
            differs |= std::memcmp(&out.logits[i], &v1.logits[i], 4) != 0;
    }
    CHECK(differs);
    // Unquantized (BF16) models keep the g4-v1 recipe bit for bit.
    const auto plain = fixture::tiny(2);
    const auto plain_weights = bind_host_weights(plain);
    HostTarget plain_exact(plain_weights, tables.view(), 2), plain_a8(plain_weights, tables.view(), 2, a8::host_recipe());
    HostStream s1(plain.config), s2(plain.config);
    const auto a = ingest(plain_exact, s1, 3, 5), b = ingest(plain_a8, s2, 3, 5);
    bytes(a.logits, b.logits);
    bytes(a.taps, b.taps);
}
} // namespace
int main() {
    try {
        quantization_tests();
        decode_tests();
        order_tests();
        graph_tests();
        std::cout << "g4-a8-v1 host: quantization, decode, order, R=1..8 row invariance passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
