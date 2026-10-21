#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/ds41_attention.hpp"

#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

// The prefix cache on synthetic states whose every piece is a function of the prefix it covers (as a prefill's): a
// state assembled from cached blocks and a snapshot equals the state built directly (same digest), lookups find the
// longest snapshot and the deepest shared block, a different block (or a hash-colliding key) is never a hit, the budget
// evicts the least recently used snapshot and the blocks only it needed, and the prefill piece plan.
namespace {

namespace ds = strix::deepseek41;
namespace attn = strix::ds41::attn;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    return h * 0xff51afd7ed558ccdULL;
}

// A state of tokens[0, p) with pieces depending only on the prefix they cover.
ds::AttentionState synthetic(std::span<const std::uint32_t> tokens, std::uint32_t p) {
    std::vector<std::uint64_t> prefix(p + 1U, 0x1234U);  // prefix[i]: hash of tokens[0, i)
    for (std::uint32_t i = 0; i < p; ++i) prefix[i + 1U] = mix(prefix[i], tokens[i]);
    ds::AttentionState s;
    s.position = p;
    s.tokens.assign(tokens.begin(), tokens.begin() + p);
    s.next_token = static_cast<std::uint32_t>(prefix[p] % 129280U);
    const auto plan = attn::released_plan();
    for (std::uint32_t l = 0; l < plan.size(); ++l) {
        const auto& r = plan[l];
        ds::AttentionLayerState L;
        L.owns_kv = r.owns_kv;
        L.owns_index = r.owns_index;
        L.ratio = r.ratio;
        L.ring.resize(std::size_t{ds::AttentionHost::ring_capacity} * attn::kWindowEntryBytes);
        for (std::size_t i = 0; i < L.ring.size(); ++i) L.ring[i] = static_cast<std::uint8_t>(mix(prefix[p], l * 1000003ULL + i));
        if (r.owns_kv) {
            const std::uint32_t entries = p / r.ratio;
            L.kv.resize(std::size_t{entries} * attn::kKvEntryBytes);
            L.idx_keys.resize(std::size_t{entries} * attn::kIdxEntryBytes);
            for (std::uint32_t e = 0; e < entries; ++e) {
                const auto h = prefix[(e + 1U) * r.ratio];
                for (std::uint32_t b = 0; b < attn::kKvEntryBytes; ++b) L.kv[std::size_t{e} * attn::kKvEntryBytes + b] = static_cast<std::uint8_t>(mix(h, l * 7U + b));
                for (std::uint32_t b = 0; b < attn::kIdxEntryBytes; ++b)
                    L.idx_keys[std::size_t{e} * attn::kIdxEntryBytes + b] = static_cast<std::uint8_t>(mix(h, l * 11U + b + 999U));
            }
            if (r.ratio == 2U) {
                L.carry.assign(2U * attn::kHeadDim, 0.0F);
                L.carry_valid = p % 2U == 1U;
            }
        }
        L.topk_known = false;
        s.layers.push_back(std::move(L));
    }
    return s;
}

std::vector<std::uint32_t> random_tokens(std::size_t n, std::mt19937& rng) {
    std::vector<std::uint32_t> t(n);
    for (auto& x : t) x = static_cast<std::uint32_t>(rng() % 129280U);
    return t;
}

void test_roundtrip() {
    std::mt19937 rng(7);
    ds::PrefixCache cache({256U, 0U});
    const auto a = random_tokens(1300, rng);
    expect(cache.lookup(a).resume == 0U, "empty cache misses");
    expect(cache.insert(synthetic(a, 1024U), 42U), "insert 1024");
    const auto hit = cache.lookup(a);
    expect(hit.resume == 1024U && hit.shared == 1024U && hit.logits_fnv == 42U, "hit at the last full block");
    const auto s = cache.state(a, hit.resume, hit.key);
    expect(s.digest(true) == synthetic(a, 1024U).digest(true), "assembled state == direct state (1024)");
    // a second prompt sharing 600 tokens: blocks 0..1 shared (512), no snapshot there
    auto b = a;
    b.resize(600);
    const auto tail = random_tokens(700, rng);
    b.insert(b.end(), tail.begin(), tail.end());
    const auto hb = cache.lookup(b);
    expect(hb.resume == 0U && hb.shared == 512U, "branch: no snapshot, shared 512");
    expect(cache.insert(synthetic(b, 512U), 1U), "insert at the branch point");
    expect(cache.stats().blocks == 4U, "the branch point reuses blocks 0..1");
    const auto hb2 = cache.lookup(b);
    expect(hb2.resume == 512U, "branch hit at 512");
    expect(cache.state(b, 512U, hb2.key).digest(true) == synthetic(b, 512U).digest(true), "assembled state == direct state (512)");
    expect(cache.insert(synthetic(b, 1280U), 2U), "insert b at 1280");
    expect(cache.lookup(b).resume == 1280U && cache.lookup(a).resume == 1024U, "both prompts resume at their own ends");
    expect(cache.state(b, 1280U, cache.lookup(b).key).digest(true) == synthetic(b, 1280U).digest(true), "assembled == direct (1280)");
    // a different token inside block 0 is a miss
    auto c = a;
    c[3] ^= 1U;
    expect(cache.lookup(c).resume == 0U && cache.lookup(c).shared == 0U, "a changed token misses");
    // the exact prompt length on a boundary: resume == size
    std::vector<std::uint32_t> exact(a.begin(), a.begin() + 1024);
    const auto he = cache.lookup(exact);
    expect(he.resume == 1024U && he.next_token == synthetic(a, 1024U).next_token, "full hit gives the next token");
    bool threw = false;
    try {
        (void)cache.state(c, 1024U, hit.key);
    } catch (const std::exception&) {
        threw = true;
    }
    expect(threw, "state() refuses a prompt that does not match the key");
}

void test_budget() {
    std::mt19937 rng(11);
    const auto one = ds::PrefixCache::snapshot_bytes(synthetic(random_tokens(512, rng), 512U));
    ds::PrefixCache cache({256U, one * 2U + one / 2U});  // two snapshots of 512 fit, not three
    const auto a = random_tokens(512, rng), b = random_tokens(512, rng), c = random_tokens(512, rng);
    expect(cache.insert(synthetic(a, 512U), 0U), "a fits");
    expect(cache.insert(synthetic(b, 512U), 0U), "b fits");
    (void)cache.state(a, 512U, cache.lookup(a).key);  // a used: b is the least recently used
    expect(cache.insert(synthetic(c, 512U), 0U), "c fits after an eviction");
    expect(cache.lookup(a).resume == 512U && cache.lookup(b).resume == 0U && cache.lookup(c).resume == 512U, "LRU evicted b");
    expect(cache.lookup(b).shared == 0U, "b's blocks went with its snapshot");
    expect(cache.stats().bytes <= one * 2U + one / 2U, "within the budget");
    ds::PrefixCache tiny({256U, one / 2U});
    expect(!tiny.insert(synthetic(a, 512U), 0U), "a snapshot larger than the budget is refused");
    // determinism: two caches fed the same calls hold the same entries
    ds::PrefixCache x({256U, one * 2U}), y({256U, one * 2U});
    for (int i = 0; i < 6; ++i) {
        const auto t = random_tokens(768, rng);
        for (auto* q : {&x, &y}) {
            (void)q->insert(synthetic(t, 512U), 0U);
            (void)q->insert(synthetic(t, 768U), 0U);
        }
    }
    expect(x.stats().bytes == y.stats().bytes && x.stats().snapshots == y.stats().snapshots, "same calls, same cache");
}

void test_pieces() {
    using V = std::vector<std::uint32_t>;
    expect(ds::plan_prefill_pieces(0, 0, 1000, 256, 4096) == V{768, 1000}, "miss: last full block, then the tail");
    expect(ds::plan_prefill_pieces(0, 0, 1024, 256, 4096) == V{1024}, "aligned prompt: one run");
    expect(ds::plan_prefill_pieces(768, 768, 1000, 256, 4096) == V{1000}, "hit at the last full block: the tail only");
    expect(ds::plan_prefill_pieces(0, 512, 2000, 256, 4096) == V{512, 1792, 2000}, "branch point first");
    expect(ds::plan_prefill_pieces(512, 1024, 1100, 256, 4096) == V{1024, 1100}, "branch point is the last full block");
    expect(ds::plan_prefill_pieces(0, 0, 10000, 256, 4096) == V{4096, 8192, 9984, 10000}, "max_rows pieces on block boundaries");
    expect(ds::plan_prefill_pieces(1024, 0, 1024, 256, 4096).empty(), "full hit: nothing to run");
    expect(ds::plan_prefill_pieces(0, 0, 100, 256, 4096) == V{100}, "short prompt: one run");
    // two-rank row split: no piece shorter than 9 rows
    expect(ds::plan_prefill_pieces(0, 0, 1029, 256, 4096, 9) == V{768, 1029}, "a 5-row tail moves the split one block back");
    expect(ds::plan_prefill_pieces(0, 0, 1040, 256, 4096, 9) == V{1024, 1040}, "a 16-row tail keeps the last full block");
    expect(ds::plan_prefill_pieces(0, 0, 1024, 256, 4096, 9) == V{1024}, "aligned prompt: one run");
    expect(ds::plan_prefill_pieces(0, 1792, 1800, 256, 4096, 9) == V{1536, 1800}, "a shared boundary 8 rows before the end is dropped");
    expect(ds::plan_prefill_pieces(0, 0, 4100, 256, 4096, 9) == V{3840, 4100}, "max_rows split never leaves a short piece");
    expect(ds::plan_prefill_pieces(0, 0, 8197, 256, 4096, 9) == V{4096, 7936, 8197}, "max_rows piece then the last full block");
    {
        std::mt19937 rng(3);
        ds::PrefixCache cache({256U, 0U});
        const auto t = random_tokens(1029, rng);
        (void)cache.insert(synthetic(t, 1024U), 0U);
        expect(cache.lookup(t).resume == 1024U, "min_rows 1: resume at the boundary 5 rows before the end");
        expect(cache.lookup(t, 9U).resume == 0U, "min_rows 9: that snapshot is passed over");
        std::vector<std::uint32_t> exact(t.begin(), t.begin() + 1024);
        expect(cache.lookup(exact, 9U).resume == 1024U, "min_rows 9: a full hit is still a hit");
    }
}

}  // namespace

int main() {
    test_roundtrip();
    test_budget();
    test_pieces();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("deepseek41 prefix cache tests passed\n");
    return 0;
}
