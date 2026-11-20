// DFly2 drafter host reference (src/hy4/ref/hy4_dfly_ref): synthetic weights, tiny geometry, no model files.
#include "hy4/ref/hy4_dfly_ref.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strix::hy4::dfly;

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

class TinyTarget final : public Target {
public:
    TinyTarget(const Geometry& g, std::uint32_t vocab, std::uint32_t positions, std::uint64_t seed) : g_(g), vocab_(vocab) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> u(-1.0F, 1.0F);
        emb.resize(std::size_t{vocab} * g.hidden);
        for (auto& x : emb) x = u(rng);
        head.resize(std::size_t{vocab} * g.hidden);
        for (auto& x : head) x = u(rng);
        lat.resize(std::size_t{g.layers} * positions * g.latent());
        for (auto& x : lat) x = u(rng);
        positions_ = positions;
    }
    std::span<const float> embed(std::uint32_t token) override {
        if (token >= vocab_) throw std::out_of_range("tiny target: token");
        return {emb.data() + std::size_t{token} * g_.hidden, g_.hidden};
    }
    std::span<const float> latent(std::uint32_t layer, std::uint32_t pos) override {
        if (pos >= limit) throw std::out_of_range("tiny target: latent at or after the anchor read");
        return {lat.data() + (std::size_t{layer} * positions_ + pos) * g_.latent(), g_.latent()};
    }
    void logits(std::span<const float> y, std::vector<float>& out) override {
        out.assign(vocab_, 0.0F);
        for (std::size_t r = 0; r < vocab_; ++r)
            for (std::size_t i = 0; i < g_.hidden; ++i) out[r] += head[r * g_.hidden + i] * y[i];
    }
    std::vector<float> emb, head, lat;
    std::uint32_t limit = 0;  // the anchor position: target rows at or after it do not exist yet

private:
    Geometry g_;
    std::uint32_t vocab_ = 0, positions_ = 0;
};

bool same(const std::vector<float>& a, const std::vector<float>& b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

void run() {
    const auto g = Geometry::tiny();
    auto w = synthetic_weights(g, 7);
    TinyTarget t(g, 16, 12, 11);
    std::vector<float> h(g.hidden);
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = std::sin(static_cast<float>(i));
    const std::uint32_t pos = 6;
    t.limit = pos;
    const Variant causal = parse_variant("h=post,fus=eh,hc=off,mask=causal");
    check(Variant{}.fusion == FusionForm::e_plus_gate && Variant{}.mask == Mask::causal && Variant{}.correction == Correction::off &&
              Variant{}.hidden == HiddenForm::post_norm,
          "default = the measured choice");

    // shape + range; reads only target rows 0 .. pos-1 (TinyTarget throws otherwise)
    const auto b4 = draft(g, causal, *w, t, 3, pos, h, g.block);
    check(b4.proposals.size() == g.block - 1U, "proposals = rows - 1");
    check(b4.confidence.size() == b4.proposals.size(), "one confidence per proposal");
    check(b4.final_rows.size() == std::size_t{g.block} * g.hidden, "final rows");
    bool in_range = true;
    for (const auto c : b4.confidence) in_range = in_range && c > 0.0F && c < 1.0F;
    check(in_range, "confidence in (0, 1)");
    for (const auto p : b4.proposals) check(p < 16U, "proposal inside the vocabulary");

    // deterministic
    const auto again = draft(g, causal, *w, t, 3, pos, h, g.block);
    check(again.proposals == b4.proposals && same(again.final_rows, b4.final_rows, again.final_rows.size()), "bitwise repeat");

    // causal block: a shorter block is a bitwise prefix of the longer one (row j never reads rows > j)
    const auto b3 = draft(g, causal, *w, t, 3, pos, h, 3);
    check(same(b3.final_rows, b4.final_rows, b3.final_rows.size()), "causal: rows run = prefix of the full block");
    check(b3.proposals[0] == b4.proposals[0] && b3.proposals[1] == b4.proposals[1], "causal: proposals prefix");

    // bidirectional: row 0 sees the later rows, so its output depends on the block length
    const Variant bidir = parse_variant("mask=bidir");
    const auto d4 = draft(g, bidir, *w, t, 3, pos, h, g.block);
    const auto d3 = draft(g, bidir, *w, t, 3, pos, h, 3);
    check(!same(d3.final_rows, d4.final_rows, g.hidden), "bidirectional: row 0 depends on the block length");

    // KV share: a target latent row before the anchor changes the block; the anchor hidden enters through slot 0
    t.lat[std::size_t{1} * 12 * g.latent() + 2U * g.latent() + 1U] += 0.5F;  // draft layer 1, position 2
    const auto moved = draft(g, causal, *w, t, 3, pos, h, g.block);
    check(!same(moved.final_rows, b4.final_rows, moved.final_rows.size()), "shared latent rows are read");
    t.lat[std::size_t{1} * 12 * g.latent() + 2U * g.latent() + 1U] -= 0.5F;
    auto h2 = h;
    h2[0] += 1.0F;
    const auto hmoved = draft(g, causal, *w, t, 3, pos, h2, g.block);
    check(!same(hmoved.final_rows, b4.final_rows, g.hidden), "anchor hidden enters slot 0");

    // the correction / fusion switches change the block; the anchor token matters
    check(!same(draft(g, parse_variant("hc=mask+"), *w, t, 3, pos, h, g.block).final_rows, b4.final_rows, b4.final_rows.size()),
          "mask-row correction changes the fill rows");
    check(!same(draft(g, parse_variant("fus=e+g"), *w, t, 3, pos, h, g.block).final_rows, b4.final_rows, g.hidden),
          "gate fusion changes slot 0");
    check(!same(draft(g, causal, *w, t, 4, pos, h, g.block).final_rows, b4.final_rows, g.hidden), "anchor token matters");

    // invalid input
    bool threw = false;
    try { (void)draft(g, causal, *w, t, 3, 0, h, g.block); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "pos 0 refused");
    threw = false;
    try { (void)parse_variant("fus=nope"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unknown variant refused");
}

}  // namespace

int main() {
    try {
        run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception %s\n", e.what());
        return 1;
    }
    if (failures == 0) std::printf("hy4 dfly ref tests: PASS\n");
    return failures == 0 ? 0 : 1;
}
