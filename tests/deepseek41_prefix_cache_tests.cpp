#include "strix/deepseek41_attention_host.hpp"
#include "strix/deepseek41_attention_state.hpp"
#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/ds41_attention.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

#ifdef _WIN32
namespace {
int setenv(const char* name, const char* value, int) { return _putenv_s(name, value); }
int setenv(const char* name, const wchar_t* value, int) {  // path::c_str() is wide on Windows
    return _wputenv_s(std::filesystem::path(name).wstring().c_str(), value);
}
int unsetenv(const char* name) { return _putenv_s(name, ""); }
}  // namespace
#endif

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

ds::PrefixCacheOptions ram_options(std::uint64_t budget) {
    ds::PrefixCacheOptions o;
    o.block = 256U;
    o.budget_bytes = budget;
    return o;
}

std::vector<std::uint32_t> random_tokens(std::size_t n, std::mt19937& rng) {
    std::vector<std::uint32_t> t(n);
    for (auto& x : t) x = static_cast<std::uint32_t>(rng() % 129280U);
    return t;
}

void test_roundtrip() {
    std::mt19937 rng(7);
    ds::PrefixCache cache(ram_options(0U));
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
    ds::PrefixCache cache(ram_options(one * 2U + one / 2U));  // two snapshots of 512 fit, not three
    const auto a = random_tokens(512, rng), b = random_tokens(512, rng), c = random_tokens(512, rng);
    expect(cache.insert(synthetic(a, 512U), 0U), "a fits");
    expect(cache.insert(synthetic(b, 512U), 0U), "b fits");
    (void)cache.state(a, 512U, cache.lookup(a).key);  // a used: b is the least recently used
    expect(cache.insert(synthetic(c, 512U), 0U), "c fits after an eviction");
    expect(cache.lookup(a).resume == 512U && cache.lookup(b).resume == 0U && cache.lookup(c).resume == 512U, "LRU evicted b");
    expect(cache.lookup(b).shared == 0U, "b's blocks went with its snapshot");
    expect(cache.stats().bytes <= one * 2U + one / 2U, "within the budget");
    ds::PrefixCache tiny(ram_options(one / 2U));
    expect(!tiny.insert(synthetic(a, 512U), 0U), "a snapshot larger than the budget is refused");
    // determinism: two caches fed the same calls hold the same entries
    ds::PrefixCache x(ram_options(one * 2U)), y(ram_options(one * 2U));
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
        ds::PrefixCache cache(ram_options(0U));
        const auto t = random_tokens(1029, rng);
        (void)cache.insert(synthetic(t, 1024U), 0U);
        expect(cache.lookup(t).resume == 1024U, "min_rows 1: resume at the boundary 5 rows before the end");
        expect(cache.lookup(t, 9U).resume == 0U, "min_rows 9: that snapshot is passed over");
        std::vector<std::uint32_t> exact(t.begin(), t.begin() + 1024);
        expect(cache.lookup(exact, 9U).resume == 1024U, "min_rows 9: a full hit is still a hit");
    }
    expect(ds::plan_prefill_pieces(0, 0, 8192, 256, 4096, 9, true) == V{4096, 7936, 8192}, "before_tail: one block before an aligned end");
    expect(ds::plan_prefill_pieces(0, 0, 8000, 256, 4096, 9, true) == V{4096, 7680, 7936, 8000}, "before_tail: one block before the last full block");
    expect(ds::plan_prefill_pieces(7936, 7936, 8192, 256, 4096, 9, true) == V{8192}, "before_tail: nothing before the resume point");
}

namespace fs = std::filesystem;

fs::path fresh_dir(const std::string& name) {
    const auto d = fs::temp_directory_path() / ("ds41-prefix-cache-test-" + name);
    fs::remove_all(d);
    return d;
}

ds::PrefixCacheOptions disk_options(const fs::path& dir, std::uint64_t ram = 0, std::uint64_t disk = 0) {
    ds::PrefixCacheOptions o;
    o.block = 256;
    o.budget_bytes = ram;
    o.dir = dir;
    o.disk_budget_bytes = disk;
    o.model_id[0] = 7;
    return o;
}

void flip_byte(const fs::path& f, std::size_t at) {
    std::fstream io(f, std::ios::in | std::ios::out | std::ios::binary);
    io.seekg(static_cast<std::streamoff>(at));
    char c = 0;
    io.read(&c, 1);
    c = static_cast<char>(c ^ 0x10);
    io.seekp(static_cast<std::streamoff>(at));
    io.write(&c, 1);
}

std::string file_name(const char* prefix, std::uint64_t key, const char* ext) {
    char name[64];
    std::snprintf(name, sizeof name, "%s-%016llx.%s", prefix, static_cast<unsigned long long>(key), ext);
    return name;
}

void test_disk() {
    std::mt19937 rng(21);
    const auto dir = fresh_dir("disk");
    const auto a = random_tokens(1300, rng);
    const auto direct1024 = synthetic(a, 1024U).digest(true), direct512 = synthetic(a, 512U).digest(true);
    {
        ds::PrefixCache cache(disk_options(dir));
        expect(cache.insert(synthetic(a, 512U), 5U), "disk: insert 512");
        expect(cache.insert(synthetic(a, 1024U), 6U), "disk: insert 1024");
        cache.flush();
        const auto st = cache.stats();
        expect(st.disk_blocks == 4U && st.disk_snapshots == 2U && st.written_files == 6U && st.write_errors == 0U, "disk: 4 block files + 2 snapshot files");
        cache.clear_ram();
        const auto hit = cache.lookup(a);
        expect(hit.resume == 1024U && hit.disk && hit.logits_fnv == 6U, "disk: hit after the RAM tier was cleared");
        const auto s = cache.state(a, hit.resume, hit.key);
        expect(s.digest(true) == direct1024, "disk: state read back == direct state");
        expect(cache.stats().disk_hits == 1U && cache.last_load().disk_files == 5U, "disk: 4 blocks + 1 snapshot read");
        const auto again = cache.lookup(a);
        expect(again.resume == 1024U && !again.disk, "disk: promoted into RAM");
    }
    {  // restart: a new cache on the directory
        ds::PrefixCache cache(disk_options(dir));
        const auto st = cache.stats();
        expect(st.scanned_files == 6U && st.ignored_files == 0U && st.disk_snapshots == 2U, "restart: 6 files, none ignored");
        const auto hit = cache.lookup(a);
        expect(hit.resume == 1024U && hit.disk && hit.next_token == synthetic(a, 1024U).next_token, "restart: hit from disk");
        expect(cache.state(a, hit.resume, hit.key).digest(true) == direct1024, "restart: state == direct");
        // a session snapshot: only the new entries (kv_from 1024)
        auto longer = a;
        longer.resize(1024);
        const auto more = random_tokens(600, rng);
        longer.insert(longer.end(), more.begin(), more.end());
        auto s1536 = synthetic(longer, 1536U);
        for (auto& L : s1536.layers)
            if (L.owns_kv) {
                L.kv.erase(L.kv.begin(), L.kv.begin() + static_cast<std::ptrdiff_t>(1024U / L.ratio * attn::kKvEntryBytes));
                L.idx_keys.erase(L.idx_keys.begin(), L.idx_keys.begin() + static_cast<std::ptrdiff_t>(1024U / L.ratio * attn::kIdxEntryBytes));
            }
        expect(cache.covered(longer, 1536U) == 1024U, "covered: 1024 in both tiers");
        expect(cache.insert(s1536, 0U, 1024U), "insert with kv_from 1024");
        cache.clear_ram();
        const auto h2 = cache.lookup(longer);
        expect(h2.resume == 1536U && cache.state(longer, 1536U, h2.key).digest(true) == synthetic(longer, 1536U).digest(true), "kv_from insert == direct");
    }
    {  // another model's files are ignored
        auto o = disk_options(fresh_dir("other"));
        {
            ds::PrefixCache x(o);
            (void)x.insert(synthetic(a, 512U), 1U);
        }
        o.model_id[0] = 8;
        ds::PrefixCache y(o);
        expect(y.stats().ignored_files == 3U && y.lookup(a).resume == 0U, "another model's files are ignored");
    }
    {  // a corrupt block file at depth 3: the snapshot at 512 survives, what hangs below block 3 is dropped
        const auto name = file_name("b", ds::prefix_key(a, 256U, 3U), "blk");
        flip_byte(dir / name, 5000U);
        ds::PrefixCache cache(disk_options(dir));
        expect(cache.stats().ignored_files >= 3U, "corrupt: the bad block and its orphans are ignored");
        const auto hit = cache.lookup(a);
        expect(hit.resume == 512U && cache.state(a, 512U, hit.key).digest(true) == direct512, "corrupt: resume at 512, state == direct");
        expect(!fs::exists(dir / name), "corrupt: the bad file is deleted");
    }
    {  // a file that goes bad after the open: load throws, forget drops the path, the cache stays usable
        ds::PrefixCache cache(disk_options(fresh_dir("late")));
        (void)cache.insert(synthetic(a, 1024U), 1U);
        cache.flush();
        cache.clear_ram();
        const auto hit = cache.lookup(a);
        flip_byte(cache.options().dir / file_name("b", ds::prefix_key(a, 256U, 2U), "blk"), 300U);
        bool threw = false;
        try {
            (void)cache.load(a, hit.resume, hit.key);
        } catch (const ds::PrefixCacheReadError&) {
            threw = true;
        }
        expect(threw, "late corruption: load throws a read error");
        cache.forget(a, hit.resume);
        expect(cache.lookup(a).resume == 0U && cache.stats().disk_blocks == 0U, "forget: the path is gone");
        expect(cache.insert(synthetic(a, 1024U), 1U), "forget: insert again");
    }
    {  // the disk budget: LRU over disk snapshots
        const auto x = random_tokens(512, rng), y = random_tokens(512, rng), z = random_tokens(512, rng);
        std::uint64_t one = 0;
        {
            ds::PrefixCache probe(disk_options(fresh_dir("probe")));
            (void)probe.insert(synthetic(x, 512U), 0U);
            one = probe.stats().disk_bytes;
        }
        ds::PrefixCache cache(disk_options(fresh_dir("budget"), 0U, one * 2U + one / 2U));
        (void)cache.insert(synthetic(x, 512U), 0U);
        (void)cache.insert(synthetic(y, 512U), 0U);
        (void)cache.state(x, 512U, cache.lookup(x).key);
        (void)cache.insert(synthetic(z, 512U), 0U);
        cache.clear_ram();
        expect(cache.lookup(x).resume == 512U && cache.lookup(y).resume == 0U && cache.lookup(z).resume == 512U, "disk LRU evicted y");
        expect(cache.stats().disk_bytes <= one * 2U + one / 2U, "disk within its budget");
        cache.flush();
        std::size_t files = 0;
        for (const auto& e : fs::directory_iterator(cache.options().dir)) files += e.is_regular_file() ? 1U : 0U;
        expect(files == 6U, "evicted files deleted");
    }
    {  // two ranks after a restart, one lost a file: retain() leaves both with the same entries
        const auto d0 = fresh_dir("r0"), d1 = fresh_dir("r1");
        const auto t = random_tokens(1100, rng);
        for (const auto& d : {d0, d1}) {
            ds::PrefixCache c(disk_options(d));
            (void)c.insert(synthetic(t, 512U), 0U);
            (void)c.insert(synthetic(t, 1024U), 0U);
        }
        fs::remove(d1 / file_name("s", ds::prefix_key(t, 256U, 4U), "snp"));
        ds::PrefixCache c0(disk_options(d0)), c1(disk_options(d1));
        const auto i0 = c0.inventory(), i1 = c1.inventory();
        expect(i0 != i1, "retain: inventories differ before");
        (void)c0.retain(i1);
        (void)c1.retain(i0);
        expect(c0.inventory() == c1.inventory(), "retain: same inventories after");
        expect(c0.lookup(t).resume == 512U && c1.lookup(t).resume == 512U, "retain: both resume at 512");
    }
    for (const char* n : {"disk", "other", "late", "probe", "budget", "r0", "r1"}) fs::remove_all(fs::temp_directory_path() / (std::string("ds41-prefix-cache-test-") + n));
}

}  // namespace

// The namespace covers the bit-relevant configuration: overlay file content, CB4, trellis pack/sites, decode identity;
// a disk tier written under one namespace is not reused under another (nor by code without a namespace).
void test_namespace() {
    const auto dir = fresh_dir("namespace");
    fs::create_directories(dir);
    const auto model = dir / "model.gguf", overlay = dir / "overlay.gguf", overlay2 = dir / "overlay2.gguf";
    std::ofstream(model, std::ios::binary) << std::string(4096, 'm');
    std::ofstream(overlay, std::ios::binary) << std::string(4096, 'o');
    std::ofstream(overlay2, std::ios::binary) << std::string(4095, 'o') << 'p';  // same size, other content
    for (const char* v : {"STRIX_DS41_OVERLAY", "STRIX_DS41_CB4", "STRIX_DS41_TRELLIS", "STRIX_DS41_TRELLIS_SITES", "STRIX_DS41_TRELLIS_MIX",
                          "STRIX_DS41_SHEXP_TP", "STRIX_DS41_EP_MAP"})
        unsetenv(v);
    const auto base = ds::prefix_namespace_from_env(false);
    expect(base.text == ds::prefix_namespace_from_env(false).text && base.tag.size() == 16U, "namespace: deterministic, 16-digit tag");
    const auto tp = ds::prefix_namespace_from_env(true);
    expect(tp.tag != base.tag, "namespace: the decode identity (expert TP) changes it");
    setenv("STRIX_DS41_OVERLAY", overlay.c_str(), 1);
    const auto ov = ds::prefix_namespace_from_env(false);
    setenv("STRIX_DS41_CB4", "1", 1);
    const auto ov_cb4 = ds::prefix_namespace_from_env(false);
    setenv("STRIX_DS41_OVERLAY", overlay2.c_str(), 1);
    const auto ov2 = ds::prefix_namespace_from_env(false);
    expect(ov.tag != base.tag && ov_cb4.tag != ov.tag && ov2.tag != ov_cb4.tag, "namespace: overlay, CB4 and the overlay's content change it");
    unsetenv("STRIX_DS41_OVERLAY");
    unsetenv("STRIX_DS41_CB4");
    setenv("STRIX_DS41_TRELLIS", dir.c_str(), 1);
    const auto pack_only = ds::prefix_namespace_from_env(false);
    expect(pack_only.tag == base.tag, "namespace: a trellis pack without sites selects nothing (same bits)");
    setenv("STRIX_DS41_TRELLIS_SITES", "head", 1);
    const auto head = ds::prefix_namespace_from_env(false);
    setenv("STRIX_DS41_TRELLIS_SITES", "shexp,head", 1);
    const auto shexp = ds::prefix_namespace_from_env(false);
    expect(head.tag != base.tag && shexp.tag != head.tag, "namespace: trellis sites change it");
    unsetenv("STRIX_DS41_TRELLIS");
    unsetenv("STRIX_DS41_TRELLIS_SITES");

    const auto id_base = ds::prefix_model_id(model, 40U, base), id_tp = ds::prefix_model_id(model, 40U, tp);
    expect(id_base == ds::prefix_model_id(model, 40U, base) && id_base != id_tp && id_base != ds::prefix_model_id(model, 4U, base),
           "model id: namespace and layer count are part of it");
    std::mt19937 rng(5);
    const auto a = random_tokens(600, rng);
    auto o = disk_options(dir / "tier");
    o.model_id = id_base;
    {
        ds::PrefixCache x(o);
        expect(x.insert(synthetic(a, 512U), 1U), "namespace: insert");
    }
    {
        auto other = o;
        other.model_id = id_tp;
        ds::PrefixCache y(other);
        expect(y.lookup(a).resume == 0U && y.stats().ignored_files == 3U, "namespace: files of another namespace are not reused");
    }
    {
        ds::PrefixCache x(o);  // (y deleted them, as for another model)
        expect(x.lookup(a).resume == 0U, "namespace: nothing left after the other namespace's open");
        expect(x.insert(synthetic(a, 512U), 1U), "namespace: insert again");
    }
    ds::PrefixCache z(o);
    expect(z.lookup(a).resume == 512U, "namespace: the same namespace reuses its files");
    fs::remove_all(dir);
}

int main() {
    test_roundtrip();
    test_budget();
    test_pieces();
    test_disk();
    test_namespace();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("deepseek41 prefix cache tests passed\n");
    return 0;
}
