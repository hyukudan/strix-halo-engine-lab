// STRIX_QSA_WINDOW_ATTN_V2 qualification, from the engine TU through strix::qpersist_diagnostic: each v2 score
// against the production launch it replaces, byte for byte (NaN words included): the verify-window v2 score
// (attention_score_rows_pinned_kernel<R, 12, S, window>) against attention_score_rows_wave_kernel<R, 4>, and at R1 the
// MTP draft step's v2 score (<1, 4, 16, lds_tree>) against its attention_score_kernel, each followed by the
// production softmax and value/gate launches (probabilities, outputs, A8 codes and deltas compared too). The engine's
// switch resolution and routing (verify windows only; not prefill, cohort, int8, R > 8, explicit score selections)
// are checked as the engine decides them. The window one-block LDS-tree score against the window wave score (two
// pre-existing kernels with different operand orders) is reported, not required. Every arm owns independently
// poisoned buffers with guards.
// usage: strix-hip-qsa-window-attn-v2-tests              (bitwise fixture)
//        strix-hip-qsa-window-attn-v2-tests bench [reps]  (score launches, R = 1/2/4/8, S = 1056 / 2051)
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
#include <utility>
#include <vector>

namespace {
using namespace strix::qpersist_diagnostic;
void require(bool ok, const std::string& text) { if (!ok) throw std::runtime_error(text); }
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }

constexpr unsigned kHeads = 24U, kKvHeads = 2U, kMaximum = 2051U;
constexpr std::size_t kGuard = 64U;

struct Owner {
    strix::HipSetupQuiescenceGuard guard;
    hipStream_t stream = nullptr;
    struct Allocation { unsigned char* base; std::size_t bytes; unsigned char poison; };
    std::vector<Allocation> allocations;
    Owner() {
        require(guard.architecture().starts_with("gfx1151"), "window attention v2 gate requires gfx1151");
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

std::uint32_t random_state = 0x56395141U;
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

// One window: rows r = 0..R-1 at consecutive positions; row r sees count - (R - 1 - r) tokens (the last row
// sees `count`). Contiguous: tokens 0..n-1 (every visible token selected, as at short contexts). Sparse: the
// compressed top-k shape: 4-token blocks chosen from a larger context plus the visible % 4 tail, so the rows'
// tables differ near the block / top-k thresholds.
struct Window {
    std::vector<std::vector<float>> query;          // [row][head][q dim | gate dim]
    std::vector<std::vector<std::uint32_t>> selected; // [row] kMaximum entries, ascending, rest UINT32_MAX
    std::vector<unsigned> count;
    std::vector<std::uint16_t> kv;                  // [token][K kv_width | V kv_width]
};
Window make_window(unsigned rows, unsigned dim, unsigned count, bool sparse, Pattern pattern) {
    Window w;
    const unsigned kv_width = kKvHeads * dim;
    unsigned context = 0;
    std::vector<unsigned> base;
    if (sparse) {
        const unsigned blocks = (count + 3U) / 4U, total = 2U * blocks + 5U;
        std::vector<unsigned> chosen(total);
        for (unsigned b = 0; b < total; ++b) chosen[b] = b;
        for (unsigned b = total - 1U; b > 0U; --b) std::swap(chosen[b], chosen[next() % (b + 1U)]);
        chosen.resize(blocks);
        std::sort(chosen.begin(), chosen.end());
        for (const auto b : chosen) for (unsigned t = 0; t < 4U; ++t) base.push_back(4U * b + t);
        context = 4U * total + 8U;
    } else context = count;
    for (unsigned r = 0; r < rows; ++r) {
        const unsigned n = std::max(1U, count - std::min(count - 1U, rows - 1U - r));
        std::vector<std::uint32_t> table(kMaximum, UINT32_MAX);
        if (sparse) {
            // n - n % 4 block tokens, then the row's own visible % 4 tail after the context.
            const unsigned whole = n / 4U * 4U;
            for (unsigned s = 0; s < whole; ++s) table[s] = base[s];
            for (unsigned t = 0; t < n % 4U; ++t) table[whole + t] = context - 8U + t;
        } else for (unsigned s = 0; s < n; ++s) table[s] = s;
        w.selected.push_back(std::move(table));
        w.count.push_back(n);
        std::vector<float> q(static_cast<std::size_t>(kHeads) * dim * 2U);
        for (auto& x : q) x = number();
        w.query.push_back(std::move(q));
    }
    w.kv.resize(static_cast<std::size_t>(context) * kv_width * 2U);
    for (auto& x : w.kv) x = bf16(number());
    const auto k_at = [&](unsigned token, unsigned kv_head, unsigned f) -> std::uint16_t& {
        return w.kv[static_cast<std::size_t>(token) * kv_width * 2U + kv_head * dim + f];
    };
    const auto v_at = [&](unsigned token, unsigned kv_head, unsigned f) -> std::uint16_t& {
        return w.kv[static_cast<std::size_t>(token) * kv_width * 2U + kv_width + kv_head * dim + f];
    };
    switch (pattern) {
    case Pattern::random: break;
    case Pattern::cancel:
        for (auto& q : w.query) for (unsigned h = 0; h < kHeads; ++h) for (unsigned f = 0; f < dim; ++f)
            q[static_cast<std::size_t>(h) * dim * 2U + f] = (f & 1U) ? -0.0F : 0.0F;
        for (unsigned t = 0; t < context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < dim; ++f)
            v_at(t, k, f) = bf16((t & 1U ? -1.0F : 1.0F) * (1.0F + static_cast<float>(f) / 256.0F) * 256.0F);
        break;
    case Pattern::range:
        for (auto& q : w.query) for (unsigned h = 0; h < kHeads; ++h) for (unsigned f = 0; f < dim; ++f)
            q[static_cast<std::size_t>(h) * dim * 2U + f] = (next() & 1U ? -1.0F : 1.0F) * pow2(static_cast<int>(f % 16U) - 6);
        for (unsigned t = 0; t < context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < dim; ++f) {
            k_at(t, k, f) = bf16((next() & 1U ? -1.0F : 1.0F) * pow2(static_cast<int>(next() % 13U) - 8));
            v_at(t, k, f) = bf16((next() & 1U ? -1.0F : 1.0F) * pow2(static_cast<int>(next() % 201U) - 100));
        }
        break;
    case Pattern::ties:
        for (unsigned t = 0; t < context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < dim; ++f) {
            k_at(t, k, f) = k_at(t - t % 3U, k, f);
            v_at(t, k, f) = bf16(static_cast<float>(static_cast<int>((t / 3U + f) % 5U) - 2) * 0.5F);
        }
        break;
    case Pattern::subnormal:
        for (unsigned t = 0; t < context; ++t) for (unsigned k = 0; k < kKvHeads; ++k) for (unsigned f = 0; f < dim; ++f) {
            const auto r = next();
            k_at(t, k, f) = static_cast<std::uint16_t>((r & 0x8000U) | ((r & 7U) == 0U ? 0U : (r >> 4U) & 0x7fU));
            v_at(t, k, f) = static_cast<std::uint16_t>(((r << 3U) & 0x8000U) | ((r & 0x70U) == 0U ? 0U : (r >> 12U) & 0x7fU));
        }
        for (auto& q : w.query) for (unsigned h = 0; h < kHeads; ++h) for (unsigned f = 0; f < dim; ++f)
            q[static_cast<std::size_t>(h) * dim * 2U + f] *= pow2(100);
        break;
    case Pattern::nonfinite: {
        const auto& last = w.selected.back();
        const auto token = [&](unsigned s) { return last[std::min(s, count - 1U)]; };
        k_at(token(count / 2U), 0U, 3U) = 0x7f80U;               // +inf key (kv head 0)
        k_at(token(count / 3U), 1U, 5U) = 0xff80U;               // -inf key (kv head 1)
        v_at(token(count / 5U), 0U, 7U) = 0x7fc1U;               // NaN value with payload
        v_at(token(count - 1U), 1U, 11U) = 0xff80U;              // -inf value
        w.query.front()[static_cast<std::size_t>(5U) * dim * 2U + dim + 9U] = -std::numeric_limits<float>::infinity(); // gate
        w.query.back()[static_cast<std::size_t>(14U) * dim * 2U + 2U] = std::numeric_limits<float>::quiet_NaN();       // query
        // NaN keys with distinct payloads and signs (quiet and signalling BF16 encodings) on both sides of the lane-local
        // pairs (f, f + 128 / 64 / 32) and of the lane pairs (f, f + 16 / 8 / 4 / 2 / 1), so two different NaNs meet at
        // every reduction level of one score.
        const std::array<std::uint16_t, 9> key_nan{0x7fc1U, 0xffa3U, 0x7fe5U, 0xff81U, 0x7f82U, 0xffc2U, 0x7fc3U, 0xfff0U, 0x7fd7U};
        const std::array<unsigned, 9> key_feature{3U, 131U, 67U, 35U, 19U, 11U, 7U, 5U, 4U};
        for (unsigned i = 0; i < key_nan.size(); ++i) {
            k_at(token(count / 7U), 0U, key_feature[i]) = key_nan[i];
            k_at(token(count / 4U), 1U, key_feature[(i + 3U) % key_nan.size()]) = key_nan[(i + 5U) % key_nan.size()];
        }
        // Payload-carrying NaN queries (quiet and signalling, both signs) in several heads of every row, again on both
        // sides of the pairs; with the NaN keys and the +-inf keys times zero queries they exercise q * k and every add.
        const std::array<std::uint32_t, 6> query_nan{0x7fc00123U, 0xffc00456U, 0x7fa00001U, 0xff800789U, 0x7fc0000bU, 0xffc0000cU};
        const std::array<unsigned, 6> query_feature{10U, 138U, 74U, 42U, 26U, 9U};
        for (unsigned r = 0; r < w.query.size(); ++r)
            for (unsigned i = 0; i < query_nan.size(); ++i) {
                const unsigned head = (r * 5U + i * 7U) % kHeads;
                w.query[r][static_cast<std::size_t>(head) * dim * 2U + query_feature[i]] = std::bit_cast<float>(query_nan[i]);
                w.query[r][static_cast<std::size_t>(head) * dim * 2U + query_feature[(i + 1U) % query_feature.size()]] =
                    std::bit_cast<float>(query_nan[(i + 2U) % query_nan.size()]);
            }
        break;
    }
    }
    return w;
}

// Window arms: 0 = default window launches (attention_score_rows_wave_kernel<R, 4> + softmax + value/gate), 1 = the
// one-block LDS-tree window score (attention_score_rows_kernel) + the same tail, 2 = the window v2 score + the same tail;
// arm 2 must equal arm 0 byte for byte (NaN words included); arm 1 against arm 0 is informational. At R1 also the MTP
// draft step: its production attention_score_kernel and its v2 score, each with the production unfused tail, which
// must be byte-identical.
std::size_t compared = 0U, preexisting_words = 0U, preexisting_nan_words = 0U;
std::string preexisting_first;
void strict(const std::vector<unsigned char>& want, const std::vector<unsigned char>& got, const std::string& what) {
    if (want == got) return;
    for (std::size_t i = 0; i < want.size(); i += 4U) {
        std::uint32_t a = 0U, b = 0U;
        std::memcpy(&a, &want[i], std::min<std::size_t>(4U, want.size() - i));
        std::memcpy(&b, &got[i], std::min<std::size_t>(4U, got.size() - i));
        if (a == b) continue;
        char word[96];
        std::snprintf(word, sizeof(word), " (word %zu: 0x%08x vs 0x%08x)", i / 4U, a, b);
        throw std::runtime_error(what + word);
    }
}
void informational(const std::vector<unsigned char>& want, const std::vector<unsigned char>& got, const std::string& what) {
    for (std::size_t i = 0; i + 4U <= want.size(); i += 4U) {
        std::uint32_t a = 0U, b = 0U;
        std::memcpy(&a, &want[i], 4U); std::memcpy(&b, &got[i], 4U);
        if (a == b) continue;
        ++preexisting_words;
        if (std::isnan(std::bit_cast<float>(a)) && std::isnan(std::bit_cast<float>(b))) ++preexisting_nan_words;
        if (preexisting_first.empty()) {
            char word[96];
            std::snprintf(word, sizeof(word), " word %zu: 0x%08x vs 0x%08x", i / 4U, a, b);
            preexisting_first = what + word;
        }
    }
}

void fixture_case(unsigned rows, unsigned count, bool sparse, Pattern pattern, bool dense, unsigned char poison) {
    Owner o;
    constexpr unsigned dim = 256U;
    const auto w = make_window(rows, dim, count, sparse, pattern);
    const unsigned width = kHeads * dim;
    const auto* kv = o.upload(w.kv);
    std::vector<const float*> query;
    std::vector<const std::uint32_t*> selected;
    for (unsigned r = 0; r < rows; ++r) { query.push_back(o.upload(w.query[r])); selected.push_back(o.upload(w.selected[r])); }
    const bool draft = rows == 1U;
    constexpr std::size_t arms = 3U;
    std::vector<std::vector<AttentionRow>> table(arms);
    std::vector<Attention> a(arms);
    for (std::size_t i = 0; i < arms; ++i) {
        for (unsigned r = 0; r < rows; ++r)
            table[i].push_back({query[r], selected[r], o.allocate<float>(static_cast<std::size_t>(kHeads) * kMaximum, poison),
                o.allocate<float>(width, poison), w.count[r]});
        a[i] = {table[i], kv, kHeads, kKvHeads, dim, kMaximum};
        if (dense) { a[i].codes = o.allocate<std::int8_t>(rows * width, poison); a[i].deltas = o.allocate<float>((width / 32U) * 64U, poison); }
    }
    const auto label = std::string(kPatterns[static_cast<unsigned>(pattern)]) + " R=" + std::to_string(rows) + " S=" +
        std::to_string(count) + (sparse ? " sparse" : " contiguous") + (dense ? " dense" : "");
    const auto name = [&](std::size_t i) -> std::string { return i == 1U ? "lds-tree vs default" : "window v2 vs default"; };
    const auto reference = [](std::size_t) -> std::size_t { return 0U; };
    const auto check_arm = [&](std::size_t i, const std::vector<unsigned char>& want, const std::vector<unsigned char>& got,
        const std::string& what) {
        if (i == 1U) informational(want, got, what + ", " + name(i) + ": " + label);
        else strict(want, got, what + " differ, " + name(i) + ": " + label);
    };
    const std::size_t score_bytes = static_cast<std::size_t>(kHeads) * kMaximum * sizeof(float);
    for (std::size_t i = 0; i < arms; ++i) {
        if (i < 2U) enqueue_scores(a[i], i == 0U, o.stream);
        else {
            require(window_v2_admitted(a[i]), "window v2 not admitted: " + label);
            enqueue_window_v2_scores(a[i], o.stream);
        }
    }
    // Raw scores (inactive slots stay poisoned in the default arm).
    for (unsigned r = 0; r < rows; ++r) {
        const auto raw = o.read(table[0][r].scores, score_bytes);
        for (std::size_t h = 0; h < kHeads; ++h) {
            const auto first = raw.begin() + static_cast<std::ptrdiff_t>((h * kMaximum + w.count[r]) * sizeof(float));
            const auto last = raw.begin() + static_cast<std::ptrdiff_t>((h + 1U) * kMaximum * sizeof(float));
            require(std::all_of(first, last, [&](auto b) { return b == poison; }), "default wrote an inactive score slot: " + label);
        }
        for (std::size_t i = 1; i < arms; ++i)
            check_arm(i, o.read(table[reference(i)][r].scores, score_bytes), o.read(table[i][r].scores, score_bytes),
                "raw score bytes row " + std::to_string(r));
    }
    // The same softmax and value/gate launches in every window arm.
    for (std::size_t i = 0; i < arms; ++i) enqueue_tail(a[i], false, o.stream);
    for (unsigned r = 0; r < rows; ++r)
        for (std::size_t i = 1; i < arms; ++i) {
            const auto ref = reference(i);
            check_arm(i, o.read(table[ref][r].scores, score_bytes), o.read(table[i][r].scores, score_bytes),
                "probability bytes row " + std::to_string(r));
            check_arm(i, o.read(table[ref][r].output, width * sizeof(float)), o.read(table[i][r].output, width * sizeof(float)),
                "output bytes row " + std::to_string(r));
        }
    if (dense)
        for (std::size_t i = 1; i < arms; ++i) {
            const auto ref = reference(i);
            check_arm(i, o.read(a[ref].codes, rows * width), o.read(a[i].codes, rows * width), "A8 code bytes");
            check_arm(i, o.read(a[ref].deltas, (width / 32U) * 64U * sizeof(float)), o.read(a[i].deltas, (width / 32U) * 64U * sizeof(float)),
                "A8 delta bytes");
        }
    if (draft) {
        // MTP draft step: production attention_score_kernel (arm 0) vs its v2 score (arm 1), raw then with the tail.
        std::array<DraftAttention, 2> d;
        for (unsigned i = 0; i < 2U; ++i) {
            d[i].query = query[0]; d[i].selected = selected[0]; d[i].kv = kv;
            d[i].scores = o.allocate<float>(static_cast<std::size_t>(kHeads) * kMaximum, poison);
            d[i].output = o.allocate<float>(width, poison);
            d[i].heads = kHeads; d[i].kv_heads = kKvHeads; d[i].dim = dim; d[i].maximum = kMaximum; d[i].count = w.count[0];
            enqueue_draft_one_block(d[i], i == 1U, false, o.stream);
        }
        strict(o.read(d[0].scores, score_bytes), o.read(d[1].scores, score_bytes),
            "raw score bytes differ, draft v2 vs attention_score_kernel: " + label);
        for (unsigned i = 0; i < 2U; ++i) enqueue_draft_one_block(d[i], i == 1U, true, o.stream);
        strict(o.read(d[0].scores, score_bytes), o.read(d[1].scores, score_bytes),
            "probability bytes differ, draft v2 vs attention_score_kernel: " + label);
        strict(o.read(d[0].output, width * sizeof(float)), o.read(d[1].output, width * sizeof(float)),
            "output bytes differ, draft v2 vs attention_score_kernel: " + label);
    }
    o.guards();
    ++compared;
}

void admission() {
    AttentionRow row{};
    Attention a{std::span<const AttentionRow>(&row, 1U), nullptr, 24U, 2U, 256U, kMaximum};
    require(window_v2_admitted(a), "the 24 / 2 / 256 geometry must be admitted");
    a.kv_heads = 4U; // 6 query heads per KV head
    require(!window_v2_admitted(a), "6 query heads per KV head must not be admitted");
    a.kv_heads = 2U; a.dim = 128U;
    require(!window_v2_admitted(a), "head_dim 128 must not be admitted");
}

// Switch resolution and routing as the engine decides them.
void routing() {
    for (const char* v : {static_cast<const char*>(nullptr), "", "1", "yes"})
        require(window_v2_switch(v, nullptr, false), std::string("v2 must be on for setting ") + (v ? v : "(unset)"));
    for (const char* v : {"0", "off", "false"}) require(!window_v2_switch(v, nullptr, false), std::string("v2 must be off for ") + v);
    require(!window_v2_switch(nullptr, "0", false), "STRIX_QPF2K_DEFAULTS=0 must turn v2 off");
    require(window_v2_switch(nullptr, "1", false), "STRIX_QPF2K_DEFAULTS=1 keeps v2 on");
    require(!window_v2_switch(nullptr, nullptr, true) && !window_v2_switch("1", nullptr, true), "the baseline must turn v2 off");

    const WindowV2Route base;
    for (const unsigned rows : {2U, 4U, 8U}) { auto r = base; r.rows = rows; require(window_v2_routed(r), "verify windows of 2/4/8 rows take v2"); }
    const auto declined = [&](auto edit, const char* what) { auto r = base; edit(r); require(!window_v2_routed(r), std::string("v2 must decline: ") + what); };
    declined([](WindowV2Route& r) { r.enabled = false; }, "switch off");
    declined([](WindowV2Route& r) { r.verify_window = false; }, "prefill tiles and multi-session cohort windows");
    declined([](WindowV2Route& r) { r.verify_window = false; r.rows = 8U; }, "8-row prefill / cohort tiles");
    declined([](WindowV2Route& r) { r.rows = 1U; }, "1 row");
    declined([](WindowV2Route& r) { r.rows = 16U; }, "R > 8");
    declined([](WindowV2Route& r) { r.rows = 64U; }, "prefill groups");
    declined([](WindowV2Route& r) { r.int8 = true; }, "int8 KV");
    declined([](WindowV2Route& r) { r.fast = true; }, "fast prefill");
    declined([](WindowV2Route& r) { r.window_grouped = true; }, "STRIX_QSA_WINDOW_GROUPED=1");
    for (const unsigned slots : {1U, 2U, 8U}) declined([&](WindowV2Route& r) { r.score_slots = slots; }, "STRIX_QSA_SCORE_SLOTS != 4");
    declined([](WindowV2Route& r) { r.kv_heads = 4U; }, "6 query heads per KV head");
    declined([](WindowV2Route& r) { r.dim = 128U; }, "head_dim 128");

    require(draft_v2_routed(true, true, false, 24U, 2U, 256U), "the draft step takes v2");
    require(!draft_v2_routed(false, true, false, 24U, 2U, 256U), "draft: switch off");
    require(!draft_v2_routed(true, false, false, 24U, 2U, 256U), "draft_exact / teacher forward keep attention_score_kernel");
    require(!draft_v2_routed(true, true, true, 24U, 2U, 256U), "STRIX_QPERSIST_DRAFT_SCORE keeps the wave kernel");
    require(!draft_v2_routed(true, true, false, 24U, 8U, 256U) && !draft_v2_routed(true, true, false, 24U, 2U, 128U), "draft shapes");
}

int fixture() {
    admission();
    routing();
    const std::array<unsigned char, 3> poisons{0xa5U, 0xffU, 0x7fU};
    unsigned n = 0U;
    // Counts around the slot tiles (4/8/16), a 1K-token context and the top-k cap.
    const std::array<unsigned, 14> counts{1U, 3U, 7U, 9U, 15U, 17U, 31U, 33U, 255U, 1025U, 1056U, 2048U, 2050U, 2051U};
    for (const unsigned rows : {1U, 2U, 4U, 8U}) for (const auto count : counts) for (const bool sparse : {false, true}) {
        if (sparse && count < 8U) continue;
        for (unsigned p = 0; p < kPatterns.size(); ++p) {
            if (count > 255U && p != 0U && p != 5U && count != 1056U) continue; // long tables: random + nonfinite (+ all at 1056)
            fixture_case(rows, count, sparse, static_cast<Pattern>(p), (n & 1U) != 0U, poisons[n % poisons.size()]);
            ++n;
        }
    }
    // Non-finite windows near the top-k cap with several draws of the random inputs, sparse and contiguous.
    for (unsigned seed = 0; seed < 24U; ++seed) for (const unsigned rows : {1U, 2U, 8U})
        fixture_case(rows, 2050U - seed % 3U, (seed & 1U) != 0U, Pattern::nonfinite, (seed & 2U) != 0U, poisons[n++ % poisons.size()]);
    std::cout << "window attention v2 PASS cases=" << compared << " window v2 + draft v2 (R1) byte-identical to the production kernels they replace, routing checked; informational, pre-existing lds-tree vs "
              << "default window score: " << preexisting_words << " differing words (" << preexisting_nan_words << " NaN in both)"
              << (preexisting_first.empty() ? "" : ", first: " + preexisting_first) << '\n';
    return 0;
}

// Median per-launch score times (us): the default window score vs each v2 score, and at R1 the draft's LDS-tree
// score vs the draft v2 score. The tails are unchanged by v2.
struct Bench {
    Owner& o;
    Window w;
    std::vector<AttentionRow> table;
    Attention a;
    DraftAttention d;
    unsigned reps;
    hipEvent_t e[2]{};
    Bench(Owner& owner, unsigned rows, unsigned count, unsigned r) : o(owner), w(make_window(rows, 256U, count, false, Pattern::random)), reps(r) {
        const auto* kv = o.upload(w.kv);
        for (unsigned i = 0; i < rows; ++i)
            table.push_back({o.upload(w.query[i]), o.upload(w.selected[i]), o.allocate<float>(static_cast<std::size_t>(kHeads) * kMaximum, 0U),
                o.allocate<float>(kHeads * 256U, 0U), w.count[i]});
        a = {table, kv, kHeads, kKvHeads, 256U, kMaximum};
        d.query = table[0].query; d.selected = table[0].selected; d.scores = table[0].scores; d.output = table[0].output;
        d.kv = kv; d.heads = kHeads; d.kv_heads = kKvHeads; d.dim = 256U; d.maximum = kMaximum; d.count = w.count[0];
        for (auto& x : e) check(hipEventCreate(&x));
    }
    Bench(const Bench&) = delete;
    Bench& operator=(const Bench&) = delete;
    ~Bench() { for (auto& x : e) (void)hipEventDestroy(x); }
    template <class F> float time(F&& launch) {
        std::vector<float> t;
        for (unsigned r = 0; r < reps + 5U; ++r) {
            check(hipEventRecord(e[0], o.stream));
            launch();
            check(hipEventRecord(e[1], o.stream));
            check(hipEventSynchronize(e[1]));
            float ms = 0.0F;
            check(hipEventElapsedTime(&ms, e[0], e[1]));
            if (r >= 5U) t.push_back(ms * 1000.0F);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2U];
    }
};
int bench(unsigned reps) {
    reps = std::max(reps, 1U);
    Owner o;
    std::printf("window attention v2 score bench: heads %u kv_heads %u dim 256, reps %u (median us per launch)\n", kHeads, kKvHeads, reps);
    for (const unsigned count : {1056U, 2051U}) for (const unsigned rows : {1U, 2U, 4U, 8U}) {
        Bench b(o, rows, count, reps);
        const auto base = b.time([&] { enqueue_scores(b.a, true, o.stream); });
        const auto tail = b.time([&] { enqueue_tail(b.a, false, o.stream); });
        std::printf("S=%u R=%u default score %.1f (tail %.1f, unchanged)\n", count, rows, base, tail);
        const auto t = b.time([&] { enqueue_window_v2_scores(b.a, o.stream); });
        std::printf("  window v2 score %.1f (saved %.1f)\n", t, base - t);
        if (rows == 1U) {
            const auto lds = b.time([&] { enqueue_draft_one_block(b.d, false, false, o.stream); });
            const auto draft = b.time([&] { enqueue_draft_one_block(b.d, true, false, o.stream); });
            std::printf("  draft: attention_score_kernel %.1f, draft v2 %.1f (saved %.1f)\n", lds, draft, lds - draft);
        }
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "bench")
            return bench(argc > 2 ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 100U);
        return fixture();
    } catch (const std::exception& e) { std::cerr << "window attention v2 FAIL: " << e.what() << '\n'; return 1; }
}
