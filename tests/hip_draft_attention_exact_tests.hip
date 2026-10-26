// Q7 arm E qualification: the MTP draft's exact attention kernels against the production incumbent
// (attention_score_rows_wave_kernel<1, 4> + attention_tail_kernel), both launched from the engine TU
// through strix::qpersist_diagnostic. Every arm owns independently poisoned buffers with guards; the
// raw score workspace, the probability workspace (active and inactive slots), the output bytes and
// the status word must be bitwise equal.
// usage: strix-hip-draft-attention-exact-tests            (bitwise fixture)
//        strix-hip-draft-attention-exact-tests bench [reps] (incumbent vs every E variant, S = 1150 / 2051)
#include "strix/hip_qpersist_diagnostic.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace strix::qpersist_diagnostic;
void require(bool ok, const std::string& text) { if (!ok) throw std::runtime_error(text); }
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }

// Arm i = score kScores[i] + tail kTails[i]; arm 0 is the production incumbent pair. Every variant
// appears at least once; arms 5..8 are the selected pairs g4s16 + {s32, p32, v2, v4}.
constexpr std::array<const char*, 9> kScores{"w4", "w8", "w16", "g2s8", "g4s8", "g4s16", "g4s16", "g4s16", "g4s16"};
constexpr std::array<const char*, 9> kTails{"incumbent", "g32", "s8", "s16", "s64", "s32", "p32", "v2", "v4"};
constexpr const char* kDefaultScore = "g4s16";
constexpr const char* kDefaultTail = "s32";
constexpr unsigned kHeads = 24U, kKvHeads = 2U, kDim = 256U, kMaximum = 2051U;
constexpr std::size_t kGuard = 64U;

struct Owner {
    strix::HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    struct Allocation { unsigned char* base; std::size_t bytes; unsigned char poison; };
    std::vector<Allocation> allocations;
    Owner() {
        require(guard.architecture().starts_with("gfx1151"), "draft attention gate requires gfx1151");
        guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); });
    }
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner() {
        guard.recover_or_fail_stop();
        for (const auto& a : allocations) if (hipFree(a.base) != hipSuccess) std::_Exit(3);
        if (stream && hipStreamDestroy(stream) != hipSuccess) std::_Exit(3);
    }
    void finish() { check(hipStreamSynchronize(stream)); guard.require_original(); }
    // Payload between two guards of kGuard bytes; payload and guards hold `poison`.
    template <class T> T* allocate(std::size_t count, unsigned char poison) {
        void* p = nullptr;
        const auto bytes = count * sizeof(T);
        guard.run([&] { check(hipMalloc(&p, bytes + 2U * kGuard)); });
        try { allocations.push_back({static_cast<unsigned char*>(p), bytes, poison}); }
        catch (...) { (void)hipFree(p); throw; }
        guard.run([&] { check(hipMemsetAsync(p, poison, bytes + 2U * kGuard, stream)); });
        return reinterpret_cast<T*>(static_cast<unsigned char*>(p) + kGuard);
    }
    template <class T> T* upload(const std::vector<T>& values) {
        auto* p = allocate<T>(values.size(), 0xa5U); finish();
        guard.run([&] { check(hipMemcpy(p, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice)); });
        return p;
    }
    std::vector<unsigned char> read(const void* p, std::size_t bytes) {
        finish(); std::vector<unsigned char> out(bytes);
        guard.run([&] { check(hipMemcpy(out.data(), p, bytes, hipMemcpyDeviceToHost)); });
        return out;
    }
    void guards() {
        for (const auto& a : allocations) {
            const auto bytes = read(a.base, a.bytes + 2U * kGuard);
            const auto poisoned = [&](auto first, auto last) { return std::all_of(first, last, [&](auto b) { return b == a.poison; }); };
            require(poisoned(bytes.begin(), bytes.begin() + kGuard), "leading guard changed");
            require(poisoned(bytes.end() - kGuard, bytes.end()), "trailing guard changed");
        }
    }
};

std::uint32_t random_state = 0x51374541U;
std::uint32_t next() { random_state = random_state * 1664525U + 1013904223U; return random_state >> 8U; }
float number() {
    const auto r = next();
    if ((r & 63U) == 0U) return std::bit_cast<float>(0x80000000U);
    if ((r & 63U) == 1U) return std::bit_cast<float>(1U);
    return static_cast<float>(static_cast<int>((r >> 8U) % 31U) - 15) / 64.0F;
}
std::uint16_t bf16(float x) {
    const auto b = std::bit_cast<std::uint32_t>(x);
    return static_cast<std::uint16_t>((b + 0x7fffU + ((b >> 16U) & 1U)) >> 16U);
}
float pow2(int e) { return std::ldexp(1.0F, e); }

enum class Pattern : unsigned { random, cancel, range, ties, subnormal, nonfinite };
constexpr std::array<const char*, 6> kPatterns{"random", "cancel", "range", "ties", "subnormal", "nonfinite"};

struct Inputs {
    std::vector<float> query;           // [head][q dim | gate dim]
    std::vector<std::uint16_t> kv;      // [token][K kv_width | V kv_width]
    std::vector<std::uint32_t> selected; // kMaximum entries, ascending, rest UINT32_MAX
    unsigned context = 0;
};

// Ascending selections: contiguous (the code1k fill) or blocks of 4 plus the visible % 4 tail (the
// compressed top-k shape at long context).
Inputs make_inputs(unsigned count, bool sparse, Pattern pattern) {
    Inputs in;
    std::vector<std::uint32_t> tokens;
    if (!sparse) {
        for (unsigned s = 0; s < count; ++s) tokens.push_back(s);
        in.context = count;
    } else {
        const unsigned blocks = count / 4U, rem = count % 4U, total = 2U * blocks + 5U;
        std::vector<unsigned> chosen(total);
        for (unsigned b = 0; b < total; ++b) chosen[b] = b;
        for (unsigned b = total - 1U; b > 0U; --b) std::swap(chosen[b], chosen[next() % (b + 1U)]);
        chosen.resize(blocks);
        std::sort(chosen.begin(), chosen.end());
        for (const auto b : chosen) for (unsigned t = 0; t < 4U; ++t) tokens.push_back(4U * b + t);
        for (unsigned t = 0; t < rem; ++t) tokens.push_back(4U * total + t);
        in.context = 4U * total + rem;
    }
    in.selected.assign(kMaximum, UINT32_MAX);
    std::copy(tokens.begin(), tokens.end(), in.selected.begin());
    const unsigned kv_width = kKvHeads * kDim;
    in.query.resize(static_cast<std::size_t>(kHeads) * kDim * 2U);
    in.kv.resize(static_cast<std::size_t>(in.context) * kv_width * 2U);
    for (auto& x : in.query) x = number();
    for (auto& x : in.kv) x = bf16(number());
    const auto k_at = [&](unsigned token, unsigned kv_head, unsigned f) -> std::uint16_t& {
        return in.kv[static_cast<std::size_t>(token) * kv_width * 2U + kv_head * kDim + f];
    };
    const auto v_at = [&](unsigned token, unsigned kv_head, unsigned f) -> std::uint16_t& {
        return in.kv[static_cast<std::size_t>(token) * kv_width * 2U + kv_width + kv_head * kDim + f];
    };
    switch (pattern) {
    case Pattern::random: break;
    case Pattern::cancel:
        // Zero queries: every score is a signed zero (products +-0), equal probabilities; values
        // +-(1 + f/256) * 2^8 alternate by token so the chain cancels.
        for (unsigned h = 0; h < kHeads; ++h) for (unsigned f = 0; f < kDim; ++f)
            in.query[static_cast<std::size_t>(h) * kDim * 2U + f] = (f & 1U) ? -0.0F : 0.0F;
        for (unsigned t = 0; t < in.context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < kDim; ++f)
            v_at(t, k, f) = bf16((t & 1U ? -1.0F : 1.0F) * (1.0F + static_cast<float>(f) / 256.0F) * 256.0F);
        break;
    case Pattern::range:
        // Scores over many binades (most probabilities underflow), values over 2^-100..2^100.
        for (unsigned h = 0; h < kHeads; ++h) for (unsigned f = 0; f < kDim; ++f)
            in.query[static_cast<std::size_t>(h) * kDim * 2U + f] = (next() & 1U ? -1.0F : 1.0F) * pow2(static_cast<int>(f % 16U) - 6);
        for (unsigned t = 0; t < in.context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < kDim; ++f) {
            k_at(t, k, f) = bf16((next() & 1U ? -1.0F : 1.0F) * pow2(static_cast<int>(next() % 13U) - 8));
            v_at(t, k, f) = bf16((next() & 1U ? -1.0F : 1.0F) * pow2(static_cast<int>(next() % 201U) - 100));
        }
        break;
    case Pattern::ties:
        // Keys and values repeat in runs of 3 tokens: equal scores, equal probabilities, short values.
        for (unsigned t = 0; t < in.context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < kDim; ++f) {
            k_at(t, k, f) = k_at(t - t % 3U, k, f);
            v_at(t, k, f) = bf16(static_cast<float>(static_cast<int>((t / 3U + f) % 5U) - 2) * 0.5F);
        }
        break;
    case Pattern::subnormal:
        // Subnormal BF16 keys/values and signed zeros.
        for (unsigned t = 0; t < in.context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < kDim; ++f) {
            const auto r = next();
            k_at(t, k, f) = static_cast<std::uint16_t>((r & 0x8000U) | ((r & 7U) == 0U ? 0U : (r >> 4U) & 0x7fU));
            v_at(t, k, f) = static_cast<std::uint16_t>(((r << 3U) & 0x8000U) | ((r & 0x70U) == 0U ? 0U : (r >> 12U) & 0x7fU));
        }
        for (unsigned h = 0; h < kHeads; ++h) for (unsigned f = 0; f < kDim; ++f)
            in.query[static_cast<std::size_t>(h) * kDim * 2U + f] *= pow2(100);
        break;
    case Pattern::nonfinite: {
        const auto token = [&](unsigned s) { return tokens[std::min(s, count - 1U)]; };
        k_at(token(count / 2U), 0U, 3U) = 0x7f80U;               // +inf key (kv head 0)
        k_at(token(count / 3U), 1U, 5U) = 0xff80U;               // -inf key (kv head 1)
        v_at(token(count / 5U), 0U, 7U) = 0x7fc1U;               // NaN value with payload
        v_at(token(count - 1U), 1U, 11U) = 0xff80U;              // -inf value
        in.query[static_cast<std::size_t>(5U) * kDim * 2U + kDim + 9U] = -std::numeric_limits<float>::infinity(); // gate
        in.query[static_cast<std::size_t>(14U) * kDim * 2U + 2U] = std::numeric_limits<float>::quiet_NaN();       // query
        break;
    }
    }
    return in;
}

struct Arm { DraftAttention a; std::uint32_t* status = nullptr; };

std::size_t compared = 0U;
void fixture_case(unsigned count, bool sparse, Pattern pattern, bool dynamic, unsigned char poison) {
    Owner o;
    const auto in = make_inputs(count, sparse, pattern);
    const auto* query = o.upload(in.query);
    const auto* kv = o.upload(in.kv);
    const auto* selected = o.upload(in.selected);
    std::vector<std::uint64_t> position_host{static_cast<std::uint64_t>(count) - 1U};
    const auto* position = dynamic ? o.upload(position_host) : nullptr;
    const std::size_t score_bytes = static_cast<std::size_t>(kHeads) * kMaximum * sizeof(float);
    const std::size_t output_bytes = static_cast<std::size_t>(kHeads) * kDim * sizeof(float);
    std::array<Arm, kScores.size()> arms;
    for (auto& arm : arms) {
        arm.a.query = query; arm.a.selected = selected; arm.a.kv = kv;
        arm.a.scores = o.allocate<float>(static_cast<std::size_t>(kHeads) * kMaximum, poison);
        arm.a.output = o.allocate<float>(static_cast<std::size_t>(kHeads) * kDim, poison);
        arm.a.heads = kHeads; arm.a.kv_heads = kKvHeads; arm.a.dim = kDim; arm.a.maximum = kMaximum; arm.a.count = count;
        if (dynamic) {
            arm.status = o.allocate<std::uint32_t>(1U, 0x00U);
            arm.a.position = position; arm.a.context = in.context + 1U; arm.a.status = arm.status;
            arm.a.ratio = 4U; arm.a.topk = 512U;
        }
    }
    const auto label = std::string(kPatterns[static_cast<unsigned>(pattern)]) + " S=" + std::to_string(count) +
        (sparse ? " sparse" : " contiguous") + (dynamic ? " dynamic" : "");
    // Raw scores (before the in-place softmax).
    std::vector<std::vector<unsigned char>> raw(arms.size());
    for (std::size_t i = 0; i < arms.size(); ++i) {
        require(i == 0U || draft_attention_admitted(arms[i].a, kScores[i], kTails[i]), "variant not admitted: " + label);
        enqueue_draft_scores(arms[i].a, kScores[i], o.stream);
        raw[i] = o.read(arms[i].a.scores, score_bytes);
    }
    for (std::size_t h = 0; h < kHeads; ++h) {
        const auto first = raw[0].begin() + static_cast<std::ptrdiff_t>((h * kMaximum + count) * sizeof(float));
        const auto last = raw[0].begin() + static_cast<std::ptrdiff_t>((h + 1U) * kMaximum * sizeof(float));
        require(std::all_of(first, last, [&](auto b) { return b == poison; }), "incumbent wrote an inactive score slot: " + label);
    }
    for (std::size_t i = 1; i < arms.size(); ++i)
        require(raw[i] == raw[0], std::string("raw score bytes differ, score ") + kScores[i] + ": " + label);
    // Tails: tail variant i on arm i's (bitwise equal) raw scores.
    for (std::size_t i = 0; i < arms.size(); ++i) enqueue_draft_tail(arms[i].a, kTails[i], o.stream);
    if (dynamic && pattern == Pattern::nonfinite) {
        // Graph path with a failing head: which other workgroups still pass their own status check
        // depends on scheduling in every arm (incumbent included), so only the sticky status is a
        // deterministic result here.
        const auto status = o.read(arms[0].status, 4U);
        require(status != std::vector<unsigned char>(4U, 0U), "nonfinite case did not raise the status: " + label);
        for (std::size_t i = 1; i < arms.size(); ++i)
            require(o.read(arms[i].status, 4U) == status, std::string("status differs, tail ") + kTails[i] + ": " + label);
        o.guards();
        ++compared;
        return;
    }
    const auto probability = o.read(arms[0].a.scores, score_bytes);
    const auto output = o.read(arms[0].a.output, output_bytes);
    for (std::size_t h = 0; h < kHeads; ++h) {
        const auto first = probability.begin() + static_cast<std::ptrdiff_t>((h * kMaximum + count) * sizeof(float));
        const auto last = probability.begin() + static_cast<std::ptrdiff_t>((h + 1U) * kMaximum * sizeof(float));
        require(std::all_of(first, last, [&](auto b) { return b == poison; }), "incumbent wrote an inactive probability slot: " + label);
    }
    for (std::size_t i = 1; i < arms.size(); ++i) {
        require(o.read(arms[i].a.scores, score_bytes) == probability, std::string("probability bytes differ, tail ") + kTails[i] + ": " + label);
        require(o.read(arms[i].a.output, output_bytes) == output, std::string("output bytes differ, tail ") + kTails[i] + ": " + label);
        if (dynamic)
            require(o.read(arms[i].status, 4U) == o.read(arms[0].status, 4U), std::string("status differs, tail ") + kTails[i] + ": " + label);
    }
    o.guards();
    ++compared;
}

void admission() {
    DraftAttention a;
    a.heads = kHeads; a.kv_heads = kKvHeads; a.dim = kDim; a.maximum = 4096U;
    for (std::size_t i = 0; i < kScores.size(); ++i) {
        a.count = 1U; require(draft_attention_admitted(a, kScores[i], kTails[i]), "S=1 must be admitted");
        a.count = 2051U; require(draft_attention_admitted(a, kScores[i], kTails[i]), "S=2051 must be admitted");
        a.count = 2052U; require(!draft_attention_admitted(a, kScores[i], kTails[i]), "S=2052 must not be admitted");
        a.count = 4096U; require(!draft_attention_admitted(a, kScores[i], kTails[i]), "S=4096 must not be admitted");
    }
    a.count = 1150U; a.kv_heads = 4U; // 6 query heads per KV head
    require(draft_attention_admitted(a, "g2s8", "v2") && !draft_attention_admitted(a, "g4s8", "s32") &&
        !draft_attention_admitted(a, "w4", "v4"), "group admission");
    a.kv_heads = kKvHeads; a.dim = 128U;
    require(!draft_attention_admitted(a, "w4", "s32"), "head_dim 128 must not be admitted");
    for (const auto& [score, tail] : {std::pair{"w5", "s32"}, std::pair{"w4", "v3"}}) {
        bool threw = false;
        try { (void)draft_attention_admitted(a, score, tail); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "unknown variant must throw");
    }
}

// Flag resolution as the engine does it (default on with g4s16 + s32, 0 = incumbent).
void selection() {
    const auto expect = [](const char* attn, const char* score, const char* tail, const std::string& want) {
        const auto got = draft_attention_selection(attn, score, tail);
        require(got == want, "selection " + std::string(attn ? attn : "(unset)") + ": got " + got + ", want " + want);
    };
    expect(nullptr, nullptr, nullptr, std::string(kDefaultScore) + "+" + kDefaultTail + " (default)");
    expect("", nullptr, nullptr, std::string(kDefaultScore) + "+" + kDefaultTail + " (default)");
    expect("0", nullptr, nullptr, "incumbent");
    expect("0", "w8", "v2", "incumbent");
    expect("1", nullptr, nullptr, std::string(kDefaultScore) + "+" + kDefaultTail + " (explicit)");
    expect("1", "w8", "v2", "w8+v2 (explicit)");
    expect(nullptr, "w4", "p32", "w4+p32 (default)");
    for (const auto& [attn, score] : {std::pair{"2", "g4s16"}, std::pair{"yes", "g4s16"}, std::pair{"", "w5"}}) {
        bool threw = false;
        try { (void)draft_attention_selection(attn, score, nullptr); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "invalid draft attention setting must throw");
    }
}

int fixture() {
    selection();
    admission();
    const std::array<unsigned, 23> counts{1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 16U, 32U, 63U, 64U, 128U, 255U, 256U, 257U, 1023U,
        1150U, 1280U, 2048U, 2049U, 2050U, 2051U};
    const std::array<unsigned char, 3> poisons{0xa5U, 0xffU, 0x7fU};
    unsigned n = 0U;
    for (const auto count : counts) for (const bool sparse : {false, true}) for (unsigned p = 0; p < kPatterns.size(); ++p) {
        if (sparse && count < 8U) continue;
        fixture_case(count, sparse, static_cast<Pattern>(p), false, poisons[n++ % poisons.size()]);
    }
    for (const auto count : {1U, 9U, 257U, 1150U, 2051U}) for (const auto pattern : {Pattern::random, Pattern::nonfinite})
        fixture_case(count, false, pattern, true, poisons[n++ % poisons.size()]);
    std::cout << "draft attention E PASS cases=" << compared << " arms=" << kScores.size() - 1U << " (w8+g32 w16+s8 g2s8+s16 "
              << "g4s8+s64 g4s16+s32 g4s16+p32 g4s16+v2 g4s16+v4 vs w4 + incumbent)\n";
    return 0;
}

// Median per-launch times (events around each launch; one draft step per rep: score then tail, the
// score launch restores the raw scores the tail overwrites). Two-launch tails also report their
// softmax and value launches separately.
struct Times { float score = 0.0F, tail = 0.0F, softmax = 0.0F, value = 0.0F; };
struct Bench {
    Owner& o;
    DraftAttention a;
    unsigned reps;
    hipEvent_t e[4]{};
    Bench(Owner& owner, unsigned count, bool sparse, unsigned r) : o(owner), reps(r) {
        const auto in = make_inputs(count, sparse, Pattern::random);
        a.query = o.upload(in.query); a.kv = o.upload(in.kv); a.selected = o.upload(in.selected);
        a.scores = o.allocate<float>(static_cast<std::size_t>(kHeads) * kMaximum, 0U);
        a.output = o.allocate<float>(static_cast<std::size_t>(kHeads) * kDim, 0U);
        a.heads = kHeads; a.kv_heads = kKvHeads; a.dim = kDim; a.maximum = kMaximum; a.count = count;
        for (auto& x : e) check(hipEventCreate(&x));
    }
    Bench(const Bench&) = delete;
    Bench& operator=(const Bench&) = delete;
    ~Bench() { for (auto& x : e) (void)hipEventDestroy(x); }
    static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2U]; }
    Times time(const char* score, const char* tail) {
        const bool split = std::string_view(tail) == "v2" || std::string_view(tail) == "v4";
        std::vector<float> s, t, sm, va;
        for (unsigned r = 0; r < reps + 5U; ++r) {
            check(hipEventRecord(e[0], o.stream));
            enqueue_draft_scores(a, score, o.stream);
            check(hipEventRecord(e[1], o.stream));
            if (split) {
                enqueue_draft_tail_stage(a, tail, false, o.stream);
                check(hipEventRecord(e[2], o.stream));
                enqueue_draft_tail_stage(a, tail, true, o.stream);
            } else {
                enqueue_draft_tail(a, tail, o.stream);
                check(hipEventRecord(e[2], o.stream));
            }
            check(hipEventRecord(e[3], o.stream));
            check(hipEventSynchronize(e[3]));
            float ms[3]{};
            for (unsigned k = 0; k < 3U; ++k) check(hipEventElapsedTime(&ms[k], e[k], e[k + 1U]));
            if (r >= 5U) {
                s.push_back(ms[0] * 1000.0F); t.push_back((ms[1] + ms[2]) * 1000.0F);
                sm.push_back(ms[1] * 1000.0F); va.push_back(ms[2] * 1000.0F);
            }
        }
        return {median(s), median(t), split ? median(sm) : 0.0F, split ? median(va) : 0.0F};
    }
};
int bench(unsigned reps) {
    reps = std::max(reps, 1U);
    Owner o;
    std::printf("draft attention bench: heads %u kv_heads %u dim %u reps %u (median us)\n", kHeads, kKvHeads, kDim, reps);
    bool pass = true;
    for (const auto& [count, sparse, kill] : {std::tuple{1150U, false, 40.0}, std::tuple{2051U, true, 80.0}}) {
        Bench b(o, count, sparse, reps);
        const auto base = b.time("w4", "incumbent");
        std::printf("S=%u %s: incumbent score %.1f tail %.1f total %.1f\n", count, sparse ? "sparse" : "contiguous",
            base.score, base.tail, base.score + base.tail);
        float best_score = base.score, best_tail = base.tail;
        const char* best_score_name = "w4";
        const char* best_tail_name = "incumbent";
        for (std::size_t i = 1; i < kScores.size(); ++i) {
            const auto t = b.time(kScores[i], kTails[i]);
            std::printf("  score %-6s %7.1f   tail %-9s %7.1f", kScores[i], t.score, kTails[i], t.tail);
            if (t.softmax > 0.0F) std::printf("  (softmax %.1f + value %.1f)", t.softmax, t.value);
            std::printf("   pair %.1f (saved %.1f)\n", t.score + t.tail, base.score + base.tail - t.score - t.tail);
            if (t.score < best_score) { best_score = t.score; best_score_name = kScores[i]; }
            if (t.tail < best_tail) { best_tail = t.tail; best_tail_name = kTails[i]; }
        }
        const auto best = b.time(best_score_name, best_tail_name);
        const double saved = static_cast<double>(base.score + base.tail) - static_cast<double>(best.score + best.tail);
        std::printf("  best pair score %s + tail %s: %.1f + %.1f = %.1f us, saved %.1f us/step (kill line %.0f): %s\n",
            best_score_name, best_tail_name, best.score, best.tail, best.score + best.tail, saved, kill, saved >= kill ? "PASS" : "BELOW");
        pass = pass && saved >= kill;
    }
    // Short selections (admission covers S = 1..2051): incumbent vs the default pair.
    std::printf("short S (contiguous): incumbent vs default %s + %s\n", kDefaultScore, kDefaultTail);
    for (const unsigned count : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U}) {
        Bench b(o, count, false, reps);
        const auto base = b.time("w4", "incumbent");
        const auto t = b.time(kDefaultScore, kDefaultTail);
        std::printf("  S=%4u incumbent %.1f + %.1f = %.1f   E %.1f + %.1f = %.1f   delta %+.1f us/step%s\n", count,
            base.score, base.tail, base.score + base.tail, t.score, t.tail, t.score + t.tail,
            t.score + t.tail - base.score - base.tail, t.score + t.tail > base.score + base.tail ? "  (E slower)" : "");
    }
    std::printf("draft attention bench %s\n", pass ? "PASS" : "BELOW KILL LINE");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "bench")
            return bench(argc > 2 ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 200U);
        return fixture();
    } catch (const std::exception& e) { std::cerr << "draft attention E FAIL: " << e.what() << '\n'; return 1; }
}
