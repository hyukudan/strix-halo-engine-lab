#include "gemma4/prefill/prefill.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace strix::gemma4;
namespace pf = strix::gemma4::prefill;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template <class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, "invalid input was accepted");
}
struct Context final : DraftContext {
    std::vector<std::uint64_t> positions;
    std::vector<float> taps;
    void append(std::span<const float> values, std::span<const std::uint64_t> rows) override {
        require(!rows.empty() && rows.size() <= 8 && values.size() == rows.size() * 5, "device draft append geometry");
        positions.insert(positions.end(), rows.begin(), rows.end()); taps.insert(taps.end(), values.begin(), values.end());
    }
};
int main() try {
    require(pf::resolve("") == pf::Mode::exact && pf::resolve("auto") == pf::Mode::exact, "unqualified default");
    require(pf::resolve("exact") == pf::Mode::exact && pf::resolve(pf::identity) == pf::Mode::wmma, "profile dispatch");
    require(pf::resolve("auto", true) == pf::Mode::wmma && pf::resolve("exact", true) == pf::Mode::exact, "qualified default and exact override");
    rejects([] { (void)pf::resolve("fast"); });
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00U) == 0x7c00U) continue;
        std::uint16_t out = 0;
        require(pf::half_bits(fp16_value(std::uint16_t(h)), out) && out == h, "half roundtrip");
        if ((h & 0x8000U) || h >= 0x7bffU) continue;
        const float a = fp16_value(std::uint16_t(h)), b = fp16_value(std::uint16_t(h + 1));
        const float midpoint = float((double(a) + double(b)) * 0.5);
        require(pf::half_bits(midpoint, out) && out == h + (h & 1U), "half midpoint RNE");
        require(pf::half_bits(std::nextafter(midpoint, b), out) && out == h + 1, "half above midpoint");
        require(pf::half_bits(std::nextafter(midpoint, a), out) && out == h, "half below midpoint");
    }
    for (const float a : {65520.0F, -65520.0F, 1e30F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        std::uint16_t out = 0; require(!pf::half_bits(a, out), "half overflow or nonfinite accepted");
    }
    std::uint16_t out = 0;
    require(pf::half_bits(65519.0F, out) && out == 0x7bffU, "half finite overflow boundary");
    for (unsigned n : {0U, 1U, 15U, 16U, 17U, 511U, 512U, 513U, 2048U, 8192U, 65536U}) {
        const auto chunks = pf::plan(0, n);
        std::size_t total = 0;
        for (const auto& c : chunks) {
            require(c.position == total && c.rows <= 512 && c.rows && c.padded_rows % 16 == 0 && c.padded_rows >= c.rows, "chunk policy");
            total += c.rows;
        }
        require(total == n, "chunk coverage");
    }
    const auto boundary = pf::plan(511, 514);
    require(boundary.size() == 3 && boundary[0].rows == 1 && boundary[1].rows == 512 && boundary[2].rows == 1, "absolute chunk boundaries");
    rejects([] { (void)pf::plan(65535, 2); });
    rejects([] { (void)pf::plan(UINT64_MAX, 1); });
    for (unsigned n : {0U, 1U, 17U, 1024U}) {
        pf::Result tail; tail.frontier = n + 6;
        for (unsigned i = 0; i < n; ++i) tail.tail_positions.push_back(6 + i);
        tail.tail_taps.resize(std::size_t(n) * 5, 1.0F);
        Context context; pf::append_tail(context, tail, 1);
        require(context.positions == tail.tail_positions && context.taps == tail.tail_taps, "committed tail grouping");
        if (n) {
            Context failed; tail.tail_taps.back() = std::numeric_limits<float>::quiet_NaN();
            rejects([&] { pf::append_tail(failed, tail, 1); }); require(failed.positions.empty(), "invalid tail partially appended");
        }
    }
    pf::Fidelity missing;
    require(!missing.same_weight_bands_pass() && !missing.bf16_bands_pass(), "missing reference passed");
    pf::Fidelity same;
    const std::array<float, 4> a = {30.0F, 30.0F, -30.0F, 0.0F};
    same.observe(a, a);
    require(same.bf16_bands_pass() && same.mean_kl() == 0.0 && same.p99_kl() == 0.0 && same.rmse() == 0.0, "identical logits");
    const std::array<float, 4> b = {29.0F, 30.0F, -30.0F, 0.0F};
    pf::Fidelity changed; changed.observe(a, b);
    require(changed.top1() == 0.0 && changed.mean_kl() > 0 && changed.maximum_error == 1.0 && !changed.same_weight_bands_pass(), "tie or KL failure");
    const std::array<float, 4> uniform{}, peaked = {1.0F, 0, 0, 0};
    pf::Fidelity analytic; analytic.observe(uniform, peaked);
    require(std::abs(analytic.mean_kl() - (std::log(std::exp(1.0) + 3) - std::log(4.0) - 0.25)) < 1e-14, "full vocabulary KL direction");
    rejects([&] { same.observe(a, {}); });
    std::puts("prefill host policy/conversion/fidelity PASS");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
