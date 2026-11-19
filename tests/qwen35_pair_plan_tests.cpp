// Host tests: qwen35 dense placement through the shared layer-chain planner (strix/pair/chain.hpp) for any rank
// count, and exactness of the planner's bound pruning against a brute-force search over the same evaluate().
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/qwen35/pair_plan.hpp"

namespace {
namespace chain = strix::pair::chain;
namespace pp = strix::qwen35::pair_plan;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

constexpr std::uint64_t kRows = 4096U;
constexpr std::uint32_t kWindow = 512U;
constexpr std::uint64_t kContext = 4352U;  // prompt + 256 decode rows of KV admitted

struct Record { std::string name; std::uint64_t byte_size; };
pp::LayerTable table(const strix::qwen35::Config& c) {
    std::vector<Record> tensors;
    for (std::uint32_t l = 0; l < c.layers; ++l)  // ~14 GB trunk: GDN and attention layers of similar size
        tensors.push_back({"blk." + std::to_string(l) + ".ffn_up.weight", c.kind(l) == strix::qwen35::LayerKind::full_attention ? 198'000'000ULL : 212'000'000ULL});
    tensors.push_back({"blk.64.nextn.eh_proj.weight", 50'000'000ULL});  // MTP block: not a trunk layer
    tensors.push_back({"token_embd.weight", 1'300'000'000ULL});
    tensors.push_back({"output.weight", 715'000'000ULL});
    tensors.push_back({"output_norm.weight", 20'480ULL});
    return pp::layer_table(tensors, c.layers);
}
pp::Rank rank(std::string name, double capacity, double rows_per_s, double read) { return {std::move(name), capacity, rows_per_s, read, 0.0}; }
std::vector<pp::Hop> hops(std::size_t n) { return std::vector<pp::Hop>(n ? n - 1U : 0U, pp::Hop{4.58e6, 0.05}); }  // 4.58 GB/s

struct Case { std::vector<chain::Device> devices; std::vector<chain::Link> links; chain::ModelShape model; };
Case make(const std::vector<pp::Rank>& ranks) {
    const strix::qwen35::Config c;
    const auto h = hops(ranks.size());
    return {pp::devices(c, ranks), pp::links(h), pp::shape(c, table(c), kContext)};
}
chain::Choice plan(const Case& k, double decode_tokens = 0.0, const char* override_text = "") {
    const auto q = pp::request(kRows, kWindow, k.devices.size(), kContext, decode_tokens, override_text);
    return chain::choose(k.devices, k.links, k.model, q);
}
double stage_spread(const chain::Plan& p) {
    double lo = 1e300, hi = 0.0;
    for (const auto t : p.stage_block_ms) { lo = std::min(lo, t); hi = std::max(hi, t); }
    return hi / lo;
}

// Brute force over every cut set of `count` stages starting at device 0 for one decoder, in the planner's order
// (lexicographic cuts, strict improvement): the planner's run must return exactly this plan.
chain::Plan brute(const Case& k, std::uint32_t count, std::uint32_t decoder, std::uint64_t state_rows) {
    chain::Plan best;
    std::vector<std::uint32_t> cuts(count + 1U, 0U);
    cuts.back() = k.model.layers;
    const std::function<void(std::uint32_t)> walk = [&](std::uint32_t index) {
        if (index == count) {
            auto p = chain::evaluate(k.devices, k.links, k.model, kRows, 0U, decoder, cuts, kWindow, state_rows);
            if (p.valid && (!best.valid || p.wall_ms < best.wall_ms * (1.0 - 1e-12))) best = p;
            return;
        }
        for (auto cut = cuts[index - 1U] + 1U; cut + (count - index) <= k.model.layers; ++cut) { cuts[index] = cut; walk(index + 1U); }
    };
    walk(1U);
    return best;
}
void same_plan(const chain::Plan& a, const chain::Plan& b, const char* what) {
    require(a.valid == b.valid && a.cuts == b.cuts && a.block == b.block && a.decoder == b.decoder &&
            std::memcmp(&a.wall_ms, &b.wall_ms, sizeof(double)) == 0, what);
}
const chain::Plan* candidate(const chain::Choice& c, std::uint32_t stages, std::uint32_t first, std::uint32_t decoder) {
    for (const auto& p : c.candidates)
        if (p.valid && p.stages() == stages && p.first == first && p.decoder == decoder) return &p;
    return nullptr;
}

void tables_and_override() {
    const strix::qwen35::Config c;
    const auto t = table(c);
    require(t.weight_bytes.size() == 64U && t.weight_bytes[0] == 212e6 && t.weight_bytes[3] == 198e6, "blk.<l> bytes per layer");
    require(t.embed_bytes == 1.3e9 && t.head_bytes == 715e6 + 20480.0, "embedding and head bytes; MTP block excluded");
    require(pp::gdn_fixed_bytes(c) == 3227648.0 && pp::kv_bytes_per_row(c) == 4096.0 && pp::seam_bytes_per_row(c) == 20480.0,
            "engine state layout (engine.hip:419-420)");
    chain::Request q;
    pp::apply_override("cut:24,48", q);
    require(q.split == chain::Split::layer_cut && q.cuts == std::vector<std::uint32_t>{24U, 48U}, "cut override");
    pp::apply_override("single", q);
    require(q.split == chain::Split::device_only, "single override");
    bool threw = false;
    try { pp::apply_override("cut:", q); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "malformed override refused");
    // uint32 overflow is refused, not wrapped (4294967299 = 2^32 + 3 used to parse as cut 3); the maximum still parses.
    for (const char* big : {"cut:4294967299", "cut:24,4294967296", "cut:99999999999"}) {
        threw = false;
        try { pp::apply_override(big, q); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, (std::string("overflowing cut refused: ") + big).c_str());
    }
    pp::apply_override("cut:4294967295", q);
    require(q.cuts == std::vector<std::uint32_t>{4294967295U}, "UINT32_MAX cut parses exactly");
}

void one_rank() {
    const auto k = make({rank("solo", 110e9, 286.0, 176e6)});
    const auto c = plan(k, 64.0);
    require(c.best.valid && c.best.stages() == 1U && c.best.cuts == std::vector<std::uint32_t>{0U, 64U} && c.best.decoder == 0U,
            "N=1 is the single-node path");
    require(std::fabs(c.best.rows_per_s(kRows) - 286.0) < 1e-6, "N=1 prefill = the declared single-node rate");
    require(c.best.decode_ms_per_token > 75.0 && c.best.decode_ms_per_token < 90.0, "N=1 decode ~ 14 GB / 176 GB/s");
    std::printf("N=1   %s\n", pp::describe(c.best, k.devices, kRows).c_str());
}

void two_equal() {
    const auto k = make({rank("a", 110e9, 286.0, 176e6), rank("b", 110e9, 286.0, 176e6)});
    const auto c = plan(k);
    require(c.best.valid && c.best.stages() == 2U && c.best.cuts[1] == 32U, "equal pair: cut 32 (equal 4-layer groups)");
    const auto speedup = c.best.rows_per_s(kRows) / 286.0;
    require(speedup > 1.70 && speedup < 1.80, "equal pair prefill ~ 16/9 of one node (8 blocks, bubble 1/9)");
    require(stage_spread(c.best) < 1.0001, "equal pair stages balanced");
    std::printf("N=2=  %s\n", pp::describe(c.best, k.devices, kRows).c_str());
}

void two_unequal() {
    // b: half the memory, half the prefill rate and half the decode read rate of a.
    const auto k = make({rank("a", 220e9, 286.0, 176e6), rank("b", 55e9, 143.0, 88e6)});
    const auto c = plan(k, 256.0);
    require(c.best.valid && c.best.stages() == 2U && c.best.cuts[1] >= 40U && c.best.cuts[1] <= 46U,
            "unequal pair: the faster rank takes ~2/3 of the cost");
    require(stage_spread(c.best) < 1.15, "unequal pair stages balanced by cost, not by layer count");
    require(c.best.decoder == 0U, "with decode tokens the faster-read rank decodes");
    const auto* slow = candidate(c, 2U, 0U, 1U);
    require(slow && slow->decode_ms_per_token > 1.9 * c.best.decode_ms_per_token, "the slow decoder is priced, not ignored");
    std::printf("N=2!= %s\n", pp::describe(c.best, k.devices, kRows).c_str());
}

void three_and_four() {
    for (const std::uint32_t n : {3U, 4U, 6U}) {
        std::vector<pp::Rank> ranks;
        for (std::uint32_t r = 0; r < n; ++r) ranks.push_back(rank("r" + std::to_string(r), 110e9, 286.0, 176e6));
        const auto k = make(ranks);
        const auto c = plan(k);  // must not hit the search work limit
        require(c.best.valid && c.best.stages() == n, "every equal rank runs a stage");
        require(stage_spread(c.best) < (n == 4U ? 1.0001 : 1.25), "N ranks balanced (4 ranks: 16-layer ranges)");
        if (n == 4U) require(c.best.cuts == std::vector<std::uint32_t>{0U, 16U, 32U, 48U, 64U}, "4 ranks: 16 layers each");
        if (n <= 4U) same_plan(*candidate(c, n, 0U, c.best.decoder), brute(k, n, c.best.decoder, kContext),
                               "pruned search = brute force (same plan, same wall bits)");
        std::printf("N=%u   %s\n", n, pp::describe(c.best, k.devices, kRows).c_str());
    }
}

void pruning_exact_without_memory() {
    // A binder that declares no memory or decode fields (today's Qwen MoE / MiniMax use) and heterogeneous,
    // unit-scaled costs: the pruned search still returns the brute-force plan bit for bit.
    auto k = make({rank("x", 110e9, 300.0, 0.0), rank("y", 110e9, 200.0, 0.0), rank("z", 110e9, 250.0, 0.0)});
    k.model.weight_bytes.clear();
    k.model.read_bytes_per_token.clear();
    k.model.embed_bytes = k.model.head_bytes = 0.0;
    for (std::size_t d = 0; d < k.devices.size(); ++d) {
        k.devices[d].block_fixed_ms = 3.0 + double(d);
        k.devices[d].unit_scale = {{256U, 1.3}, {1024U, 0.9}};
        for (std::uint32_t l = 0; l < 64U; ++l) k.devices[d].layer_ms_per_row[l] *= 1.0 + 0.01 * double((l * 7U + d) % 5U);
    }
    const auto c = plan(k);
    require(c.best.valid, "heterogeneous plan exists");
    for (std::uint32_t decoder = 0; decoder < 3U; ++decoder)
        same_plan(*candidate(c, 3U, 0U, decoder), brute(k, 3U, decoder, 0U), "pruning exact without memory fields");
}

void too_small_rank() {
    const auto k = make({rank("big", 110e9, 286.0, 176e6), rank("tiny", 0.1e9, 286.0, 176e6), rank("other", 110e9, 286.0, 176e6)});
    const auto c = plan(k);
    require(c.best.valid, "a too-small rank never crashes the plan");
    for (std::uint32_t s = 0; s < c.best.stages(); ++s) require(c.best.first + s != 1U, "the tiny rank runs no layers");
    require(c.best.decoder != 1U, "the tiny rank does not decode");
    bool named = false;
    for (const auto& p : c.candidates) named = named || p.reason.find("tiny") != std::string::npos;
    require(named, "the refusal names the tiny rank");
    // No rank can hold the whole model for decode: refused with a reason, never a crash.
    // 14.5 GB each: every split stage fits, the whole model (15.35 GB + state) fits neither rank.
    const auto none = make({rank("p", 14.5e9, 286.0, 176e6), rank("q", 14.5e9, 286.0, 176e6)});
    const auto refused = plan(none);
    require(!refused.best.valid && !refused.best.reason.empty(), "no decoder fits: refused");
    bool decoder_reason = false;
    for (const auto& p : refused.candidates) decoder_reason = decoder_reason || p.reason.find("cannot hold the whole model") != std::string::npos;
    require(decoder_reason, "the decoder refusal is observable");
    std::printf("tiny  %s\nnone  %s\n", pp::describe(c.best, k.devices, kRows).c_str(), pp::describe(refused.best, none.devices, kRows).c_str());
}

void determinism_and_override() {
    const auto k = make({rank("a", 110e9, 286.0, 176e6), rank("b", 110e9, 286.0, 176e6)});
    same_plan(plan(k, 32.0).best, plan(k, 32.0).best, "same inputs, same plan");
    const auto forced = plan(k, 0.0, "cut:24");
    require(forced.best.valid && forced.best.cuts == std::vector<std::uint32_t>{0U, 24U, 64U}, "explicit cut honoured");
    const auto single = plan(k, 0.0, "single");
    require(single.best.valid && single.best.stages() == 1U, "explicit single-rank honoured");
}
} // namespace

int main() try {
    tables_and_override();
    one_rank();
    two_equal();
    two_unequal();
    three_and_four();
    pruning_exact_without_memory();
    too_small_rank();
    determinism_and_override();
    std::puts("PASS: qwen35 dense placement over N ranks");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
