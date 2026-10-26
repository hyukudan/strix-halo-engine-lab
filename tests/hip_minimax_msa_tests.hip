// Device tests of the minimax-m3 MSA kernels: every kernel output equals the host reference bit for bit.
// STRIX_MINIMAX_MSA_BENCH=1 adds kernel timings (R1/R8 at 8K, 32K, 128K, TP rank shape).
#include "strix/hip_minimax_msa.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace msa = strix::minimax::msa;
namespace mh = strix::minimax::msa::hip;

namespace {

int g_fail = 0;
void check(bool ok, const char* what, unsigned long long a = 0, unsigned long long b = 0) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s (%llu %llu)\n", what, a, b);
    }
}
void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error %s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}
template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    hip_ok(hipMalloc(&d, std::max<std::size_t>(1, h.size()) * sizeof(T)), "malloc");
    if (!h.empty()) hip_ok(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return d;
}
template <class T> std::vector<T> host(const T* d, std::size_t n) {
    std::vector<T> h(n);
    hip_ok(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
bool same_bits(const float* a, const float* b, std::size_t n, std::size_t* where) {
    for (std::size_t i = 0; i < n; ++i)
        if (std::memcmp(a + i, b + i, sizeof(float)) != 0) {
            *where = i;
            return false;
        }
    return true;
}

struct Case {
    unsigned rows, q_heads, kv_heads, idx_heads;
    std::uint64_t last_pos;
};

struct Run {
    // host inputs
    std::vector<float> q, k, v, iq, ik, w;
    std::vector<msa::RopeRow> rope;
    std::vector<std::uint64_t> pos;
    std::vector<std::uint16_t> k_cache, v_cache, ik_cache;
    std::uint32_t max_blocks;
};

Run make_run(const Case& c, std::mt19937& rng) {
    Run run;
    std::normal_distribution<float> n(0.0F, 1.0F);
    const std::uint64_t cap = c.last_pos + 1;
    run.q.resize(std::size_t(c.rows) * c.q_heads * 128);
    run.k.resize(std::size_t(c.rows) * c.kv_heads * 128);
    run.v.resize(run.k.size());
    run.iq.resize(std::size_t(c.rows) * c.idx_heads * 128);
    run.ik.resize(std::size_t(c.rows) * 128);
    for (auto* vec : {&run.q, &run.k, &run.v, &run.iq, &run.ik})
        for (auto& x : *vec) x = n(rng) * 3.0F;
    run.w.resize(4 * 128);
    for (auto& x : run.w) x = n(rng) * 0.3F;
    run.k_cache.resize(cap * c.kv_heads * 128);
    run.v_cache.resize(run.k_cache.size());
    run.ik_cache.resize(cap * 128);
    for (auto* vec : {&run.k_cache, &run.v_cache, &run.ik_cache})
        for (auto& x : *vec) x = msa::float_to_bf16(n(rng));
    for (unsigned r = 0; r < c.rows; ++r) {
        run.pos.push_back(c.last_pos + 1 - c.rows + r);
        run.rope.push_back(msa::rope_row(run.pos.back(), 1e7));
    }
    run.max_blocks = static_cast<std::uint32_t>(msa::block_count(cap));
    return run;
}

// Host pipeline of the same call (reference order).
struct HostOut {
    std::vector<float> q, iq, scores, o;
    std::vector<std::uint16_t> k_cache, v_cache, ik_cache;
    std::vector<std::uint32_t> idx, idx_count;
};
HostOut host_pipeline(const Case& c, const Run& run) {
    HostOut h;
    h.q = run.q;
    h.iq = run.iq;
    h.k_cache = run.k_cache;
    h.v_cache = run.v_cache;
    h.ik_cache = run.ik_cache;
    const float *qw = run.w.data(), *kw = qw + 128, *iqw = kw + 128, *ikw = iqw + 128;
    for (unsigned r = 0; r < c.rows; ++r) {
        const std::uint64_t p = run.pos[r];
        for (unsigned hd = 0; hd < c.q_heads; ++hd) msa::head_norm_rope(&h.q[(r * c.q_heads + hd) * 128], qw, run.rope[r]);
        for (unsigned g = 0; g < c.kv_heads; ++g) {
            float x[128];
            std::memcpy(x, &run.k[(r * c.kv_heads + g) * 128], sizeof x);
            msa::head_norm_rope(x, kw, run.rope[r]);
            for (unsigned d = 0; d < 128; ++d) {
                h.k_cache[(p * c.kv_heads + g) * 128 + d] = msa::float_to_bf16(x[d]);
                h.v_cache[(p * c.kv_heads + g) * 128 + d] = msa::float_to_bf16(run.v[(r * c.kv_heads + g) * 128 + d]);
            }
        }
        for (unsigned hd = 0; hd < c.idx_heads; ++hd) msa::head_norm_rope(&h.iq[(r * c.idx_heads + hd) * 128], iqw, run.rope[r]);
        float x[128];
        std::memcpy(x, &run.ik[r * 128], sizeof x);
        msa::head_norm_rope(x, ikw, run.rope[r]);
        for (unsigned d = 0; d < 128; ++d) h.ik_cache[p * 128 + d] = msa::float_to_bf16(x[d]);
    }
    const std::uint64_t cap = c.last_pos + 1;
    std::vector<float> ikf(cap * 128), kf(cap * c.kv_heads * 128), vf(kf.size());
    for (std::size_t i = 0; i < ikf.size(); ++i) ikf[i] = msa::bf16_to_float(h.ik_cache[i]);
    for (std::size_t i = 0; i < kf.size(); ++i) {
        kf[i] = msa::bf16_to_float(h.k_cache[i]);
        vf[i] = msa::bf16_to_float(h.v_cache[i]);
    }
    h.scores.assign(std::size_t(c.rows) * c.idx_heads * run.max_blocks, 0.0F);
    h.idx.assign(std::size_t(c.rows) * c.idx_heads * msa::kTopK, 0);
    h.idx_count.assign(std::size_t(c.rows) * c.idx_heads, 0);
    h.o.assign(h.q.size(), 0.0F);
    std::vector<float> kg, vg;
    for (unsigned r = 0; r < c.rows; ++r) {
        const std::uint64_t count = run.pos[r] + 1;
        for (unsigned hd = 0; hd < c.idx_heads; ++hd) {
            std::span<float> sc(&h.scores[(r * c.idx_heads + hd) * run.max_blocks], msa::block_count(count));
            msa::index_block_scores(&h.iq[(r * c.idx_heads + hd) * 128], ikf.data(), count, sc);
            const auto sel = msa::select_blocks(sc);
            std::copy(sel.begin(), sel.end(), &h.idx[(r * c.idx_heads + hd) * msa::kTopK]);
            h.idx_count[r * c.idx_heads + hd] = static_cast<std::uint32_t>(sel.size());
            const unsigned g = hd;  // index head g selects for KV head g
            kg.resize(count * 128);
            vg.resize(count * 128);
            for (std::uint64_t t = 0; t < count; ++t) {
                std::memcpy(&kg[t * 128], &kf[(t * c.kv_heads + g) * 128], 512);
                std::memcpy(&vg[t * 128], &vf[(t * c.kv_heads + g) * 128], 512);
            }
            msa::sparse_attention(&h.q[(r * c.q_heads + g * msa::kGroup) * 128], msa::kGroup, kg.data(), vg.data(), count,
                                  sel, &h.o[(r * c.q_heads + g * msa::kGroup) * 128]);
        }
    }
    return h;
}

struct DevBufs {
    float *q, *k, *v, *iq, *ik, *w, *scores, *partial, *o;
    msa::RopeRow* rope;
    std::uint64_t* pos;
    std::uint16_t *kc, *vc, *ikc;
    std::uint32_t *idx, *cnt;
};
DevBufs upload(const Case& c, const Run& run) {
    DevBufs d{};
    d.q = dev(run.q);
    d.k = dev(run.k);
    d.v = dev(run.v);
    d.iq = dev(run.iq);
    d.ik = dev(run.ik);
    d.w = dev(run.w);
    d.rope = dev(run.rope);
    d.pos = dev(run.pos);
    d.kc = dev(run.k_cache);
    d.vc = dev(run.v_cache);
    d.ikc = dev(run.ik_cache);
    d.scores = dev(std::vector<float>(std::size_t(c.rows) * c.idx_heads * run.max_blocks, 0.0F));
    d.idx = dev(std::vector<std::uint32_t>(std::size_t(c.rows) * c.idx_heads * msa::kTopK, 0));
    d.cnt = dev(std::vector<std::uint32_t>(std::size_t(c.rows) * c.idx_heads, 0));
    d.partial = dev(std::vector<float>(std::size_t(c.rows) * c.q_heads * msa::kMaxChunks * 130, 0.0F));
    d.o = dev(std::vector<float>(run.q.size(), 0.0F));
    return d;
}
void release(DevBufs& d) {
    for (void* p : {static_cast<void*>(d.q), static_cast<void*>(d.k), static_cast<void*>(d.v), static_cast<void*>(d.iq),
                    static_cast<void*>(d.ik), static_cast<void*>(d.w), static_cast<void*>(d.scores),
                    static_cast<void*>(d.partial), static_cast<void*>(d.o), static_cast<void*>(d.rope),
                    static_cast<void*>(d.pos), static_cast<void*>(d.kc), static_cast<void*>(d.vc),
                    static_cast<void*>(d.ikc), static_cast<void*>(d.idx), static_cast<void*>(d.cnt)})
        (void)hipFree(p);
}

struct Views {
    mh::PrepView prep;
    mh::ScoreView score;
    mh::SelectView select;
    mh::AttentionView attn;
};
Views views(const Case& c, const Run& run, const DevBufs& d) {
    Views v{};
    v.prep = {c.rows, c.q_heads, c.kv_heads, c.idx_heads, d.q, d.k, d.v, d.iq, d.ik, d.w, d.w + 128, d.w + 256,
              d.w + 384, d.rope, d.pos, d.kc, d.vc, d.ikc};
    v.score = {c.rows, c.idx_heads, run.max_blocks, d.iq, d.ikc, d.pos, d.scores};
    v.select = {c.rows, c.idx_heads, run.max_blocks, d.pos, d.scores, d.idx, d.cnt};
    v.attn = {c.rows, c.kv_heads, d.q, d.kc, d.vc, d.pos, d.idx, d.cnt, d.partial, d.o};
    return v;
}

void run_case(const Case& c, std::mt19937& rng) {
    const Run run = make_run(c, rng);
    const HostOut h = host_pipeline(c, run);
    DevBufs d = upload(c, run);
    const Views v = views(c, run, d);
    mh::enqueue_prep(v.prep, nullptr);
    mh::enqueue_scores(v.score, nullptr);
    mh::enqueue_select(v.select, nullptr);
    mh::enqueue_attention(v.attn, nullptr);
    hip_ok(hipDeviceSynchronize(), "sync");
    std::size_t at = 0;
    const auto q = host(d.q, h.q.size());
    check(same_bits(q.data(), h.q.data(), q.size(), &at), "prep q bits", at);
    const auto iq = host(d.iq, h.iq.size());
    check(same_bits(iq.data(), h.iq.data(), iq.size(), &at), "prep iq bits", at);
    check(host(d.kc, h.k_cache.size()) == h.k_cache, "prep k cache bits");
    check(host(d.vc, h.v_cache.size()) == h.v_cache, "prep v cache bits");
    check(host(d.ikc, h.ik_cache.size()) == h.ik_cache, "prep index key bits");
    const auto sc = host(d.scores, h.scores.size());
    bool scores_ok = true;
    for (unsigned r = 0; r < c.rows; ++r)
        for (unsigned hd = 0; hd < c.idx_heads; ++hd) {
            const std::size_t base = (std::size_t(r) * c.idx_heads + hd) * run.max_blocks;
            const std::size_t nb = msa::block_count(run.pos[r] + 1);
            scores_ok &= same_bits(sc.data() + base, h.scores.data() + base, nb, &at);
        }
    check(scores_ok, "index block scores bits");
    check(host(d.cnt, h.idx_count.size()) == h.idx_count, "selection counts");
    const auto idx = host(d.idx, h.idx.size());
    bool idx_ok = true;
    for (std::size_t rh = 0; rh < h.idx_count.size(); ++rh)
        for (std::uint32_t i = 0; i < h.idx_count[rh]; ++i) idx_ok &= idx[rh * msa::kTopK + i] == h.idx[rh * msa::kTopK + i];
    check(idx_ok, "selected blocks");
    const auto o = host(d.o, h.o.size());
    check(same_bits(o.data(), h.o.data(), o.size(), &at), "attention output bits", at);
    // Context-parallel indexer: two shards (round-robin blocks) score all index heads, keep their candidates, and
    // the merge must give exactly the single-shard selection.
    {
        const std::size_t n_rh = std::size_t(c.rows) * c.idx_heads;
        float* sc_cp[2];
        std::uint32_t *ck[2], *ci[2], *cc[2];
        for (int sh = 0; sh < 2; ++sh) {
            sc_cp[sh] = dev(std::vector<float>(n_rh * run.max_blocks, 0.0F));
            ck[sh] = dev(std::vector<std::uint32_t>(n_rh * msa::kTopK, 0));
            ci[sh] = dev(std::vector<std::uint32_t>(n_rh * msa::kTopK, 0));
            cc[sh] = dev(std::vector<std::uint32_t>(n_rh, 0));
            mh::ScoreView sv = v.score;
            sv.scores = sc_cp[sh];
            sv.shard_rank = static_cast<std::uint32_t>(sh);
            sv.shard_count = 2;
            mh::enqueue_scores(sv, nullptr);
            mh::enqueue_candidates({c.rows, c.idx_heads, run.max_blocks, static_cast<std::uint32_t>(sh), 2, d.pos, sc_cp[sh],
                                    ck[sh], ci[sh], cc[sh]}, nullptr);
        }
        auto* idx2 = dev(std::vector<std::uint32_t>(n_rh * msa::kTopK, 0));
        auto* cnt2 = dev(std::vector<std::uint32_t>(n_rh, 0));
        mh::MergeView mv{c.rows, c.idx_heads, d.pos, {ck[0], ck[1]}, {ci[0], ci[1]}, {cc[0], cc[1]}, idx2, cnt2};
        mh::enqueue_merge(mv, nullptr);
        hip_ok(hipDeviceSynchronize(), "cp sync");
        check(host(cnt2, n_rh) == h.idx_count, "cp selection counts");
        const auto i2 = host(idx2, n_rh * msa::kTopK);
        bool ok = true;
        for (std::size_t x = 0; x < n_rh; ++x)
            for (std::uint32_t i = 0; i < h.idx_count[x]; ++i) ok &= i2[x * msa::kTopK + i] == h.idx[x * msa::kTopK + i];
        check(ok, "cp selected blocks");
        for (int sh = 0; sh < 2; ++sh)
            for (void* ptr : {static_cast<void*>(sc_cp[sh]), static_cast<void*>(ck[sh]), static_cast<void*>(ci[sh]),
                              static_cast<void*>(cc[sh])})
                (void)hipFree(ptr);
        (void)hipFree(idx2);
        (void)hipFree(cnt2);
    }
    std::printf("case rows %u heads %u/%u/%u last_pos %llu: %s\n", c.rows, c.q_heads, c.kv_heads, c.idx_heads,
                static_cast<unsigned long long>(c.last_pos), g_fail == 0 ? "bitwise equal" : "MISMATCH");
    release(d);
}

void bench(std::mt19937& rng) {
    for (std::uint64_t ctx : {8192ULL, 32768ULL, 131072ULL})
        for (unsigned rows : {1U, 8U}) {
            const Case c{rows, 32, 2, 2, ctx - 1};
            const Run run = make_run(c, rng);
            DevBufs d = upload(c, run);
            const Views v = views(c, run, d);
            hipEvent_t e[5];
            for (auto& x : e) hip_ok(hipEventCreate(&x), "event");
            float t[4] = {0, 0, 0, 0};
            const int reps = 20;
            for (int it = 0; it < reps + 2; ++it) {
                hip_ok(hipEventRecord(e[0], nullptr), "rec");
                mh::enqueue_prep(v.prep, nullptr);
                hip_ok(hipEventRecord(e[1], nullptr), "rec");
                mh::enqueue_scores(v.score, nullptr);
                hip_ok(hipEventRecord(e[2], nullptr), "rec");
                mh::enqueue_select(v.select, nullptr);
                hip_ok(hipEventRecord(e[3], nullptr), "rec");
                mh::enqueue_attention(v.attn, nullptr);
                hip_ok(hipEventRecord(e[4], nullptr), "rec");
                hip_ok(hipEventSynchronize(e[4]), "sync");
                if (it < 2) continue;
                for (int k = 0; k < 4; ++k) {
                    float ms = 0;
                    hip_ok(hipEventElapsedTime(&ms, e[k], e[k + 1]), "elapsed");
                    t[k] += ms * 1000.0F / reps;
                }
            }
            std::printf("bench ctx %llu rows %u (rank: 32q/2kv/2idx), us per layer: prep %.1f scores %.1f select %.1f "
                        "attention %.1f total %.1f\n", static_cast<unsigned long long>(ctx), rows, t[0], t[1], t[2], t[3],
                        t[0] + t[1] + t[2] + t[3]);
            for (auto& x : e) (void)hipEventDestroy(x);
            release(d);
        }
}

}  // namespace

int main() {
    std::mt19937 rng(20261023);
    const Case cases[] = {
        {1, 32, 2, 2, 0},     {1, 32, 2, 2, 100},    {3, 32, 2, 2, 2047},  {2, 32, 2, 2, 2048},
        {8, 32, 2, 2, 4095},  {8, 64, 4, 4, 4100},   {1, 64, 4, 4, 9000},  {5, 32, 2, 2, 20000},
        {8, 32, 2, 2, 70000}, {1, 32, 2, 2, 131071}, {2, 32, 2, 2, 32767}, {2, 32, 2, 2, 32895},
        {1, 32, 2, 2, 524287},
    };
    for (const Case& c : cases) run_case(c, rng);
    if (const char* b = std::getenv("STRIX_MINIMAX_MSA_BENCH"); b != nullptr && std::strcmp(b, "1") == 0) bench(rng);
    if (g_fail != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("strix-hip-minimax-msa-tests: all passed\n");
    return 0;
}
