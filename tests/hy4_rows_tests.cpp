// Hy4-flash verify rows, host side (src/hy4/hip/hy4_hip_rows.hpp):
//  1. rewind replay: with the rings sized by rows::cache_rows(prompt_rows, 8) every read after a verify pass of R <= 8
//     rows and any commit k finds its own key (sliding latent ring and indexer token ring, tiny and window-512 geometry);
//     negative control: rings sized for one row (the decode-only allocation before hy4_rows) are clobbered.
//  2. position words of a row == the decode step's {p, p/4*4, p/4, 0}.
//  3. speculative decode through the shared strix::spec layer over a history-exact fake forward: ids == plain greedy,
//     speculation engages (accepted drafts > 0) and a commit that keeps rejected rows is detected (ids differ).
#include "hy4/hip/hy4_hip_rows.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace hb = strix::hy4::hipbind;
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            ++failures;                                                             \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                           \
    } while (0)

void check_rings(const strix::hy4::Config& c, const char* name) {
    const std::uint32_t context = 8192;
    std::uint32_t clobbered = 0;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        for (const std::uint32_t prompt_rows : {0U, 16U, 1024U}) {
            const auto e = hb::rows::check_rewind(c, l, context, hb::rows::cache_rows(prompt_rows, hb::rows::kMaxVerifyRows),
                                                  hb::rows::kMaxVerifyRows);
            if (!e.empty()) std::fprintf(stderr, "%s: %s\n", name, e.c_str());
            CHECK(e.empty());
        }
        const auto broken = hb::rows::check_rewind(c, l, context, hb::rows::cache_rows(0U, 0U), hb::rows::kMaxVerifyRows);
        clobbered += broken.empty() ? 0U : 1U;
    }
    CHECK(clobbered == c.layers);  // every layer keeps a ring: one-row rings fail the replay
    std::printf("%s: rewind replay R 1..8 PASS on %u layers, one-row rings clobbered on %u\n", name, c.layers, clobbered);
}

// The next-token block in the rows form (Forward::nextn_rows): chunks cover the call's rows once, in order, each within
// the verify rows (its workspace), and the block's caches (the layout allocate_nextn appends at block id c.layers, rings
// sized cache_rows(prompt, verify)) survive a chunk written ahead of its reads then redone (the flush after a rejection).
void check_nextn_rows(const strix::hy4::Config& c, const char* name) {
    for (const std::uint32_t vmax : {0U, 1U, 2U, 5U, hb::rows::kMaxVerifyRows})
        for (std::uint32_t R = 1; R <= 32U; ++R) {
            std::uint32_t off = 0, chunks = 0;
            while (const std::uint32_t n = hb::rows::nextn_chunk(off, R, vmax)) {
                CHECK(n <= std::max(1U, vmax));
                off += n;
                ++chunks;
            }
            CHECK(off == R);
            CHECK(chunks == (R + std::max(1U, vmax) - 1U) / std::max(1U, vmax));
        }
    for (const std::uint8_t indexed : {std::uint8_t{0}, std::uint8_t{1}}) {
        auto cx = c;  // allocate_nextn's form: the block's heads / indexed / dense appended at id c.layers
        cx.heads.push_back(indexed != 0U ? 32U : 64U);
        cx.indexed.push_back(indexed);
        cx.dense.push_back(0U);
        for (const std::uint32_t prompt_rows : {0U, 16U}) {
            const auto e = hb::rows::check_rewind(cx, c.layers, 8192U, hb::rows::cache_rows(prompt_rows, hb::rows::kMaxVerifyRows),
                                                  hb::rows::kMaxVerifyRows);
            if (!e.empty()) std::fprintf(stderr, "%s nextn: %s\n", name, e.c_str());
            CHECK(e.empty());
        }
    }
    std::printf("%s: next-token rows chunks 1..32 x verify 0..8 PASS, block rings (indexed / sliding) PASS\n", name);
}

void check_words() {
    for (std::uint32_t p = 0; p < 100; ++p) {
        const auto w = hb::rows::position_words(p, 4U);
        CHECK(w[0] == p && w[1] == p / 4U * 4U && w[2] == p / 4U && w[3] == 0U);
    }
    CHECK(hb::rows::cache_rows(0U, 0U) == 1U && hb::rows::cache_rows(0U, 8U) == 8U && hb::rows::cache_rows(1024U, 8U) == 1024U);
}

// A target whose greedy id is a pure function of its whole token history: next = the pattern successor of the last
// token, except at every 11th position (a function of the history length) -> copy-friendly with rejections.
struct FakeForward {
    std::vector<std::uint32_t> hist, greedy;
    std::uint32_t p0 = 0, rows = 0;
    bool broken_commit = false;
    static std::uint32_t next(const std::vector<std::uint32_t>& h) {
        const auto n = static_cast<std::uint32_t>(h.size());
        if (n % 11U == 0U) return 100U + n % 7U;
        const std::uint32_t last = h.back();
        return last < 100U ? (last + 1U) % 9U : 0U;
    }
    void forward_rows(std::span<const std::uint32_t> toks, bool) {
        if (rows != 0U) throw std::logic_error("fake: pass before commit");
        p0 = static_cast<std::uint32_t>(hist.size());
        rows = static_cast<std::uint32_t>(toks.size());
        greedy.clear();
        for (const auto t : toks) {
            hist.push_back(t);
            greedy.push_back(next(hist));
        }
    }
    [[nodiscard]] std::span<const std::uint32_t> rows_greedy() const { return greedy; }
    void commit_rows(std::uint32_t k) {
        if (k > rows) throw std::logic_error("fake: commit beyond the pass");
        if (!broken_commit) hist.resize(p0 + k);
        rows = 0;
    }
    [[nodiscard]] std::uint32_t position() const { return static_cast<std::uint32_t>(hist.size()); }
};

void check_spec() {
    const std::vector<std::uint32_t> prompt{0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2};
    const std::uint32_t n = 300;
    std::vector<std::uint32_t> plain{FakeForward::next(prompt)};
    {
        auto h = prompt;
        while (plain.size() < n) {
            h.push_back(plain.back());
            plain.push_back(FakeForward::next(h));
        }
    }
    hb::rows::SpecOptions o;
    o.max_rows = hb::rows::kMaxVerifyRows;
    for (std::uint32_t r = 1; r <= o.max_rows; ++r) o.verify_ms.push_back(57.7 + 18.0 * (r - 1U));  // the predicted curve
    FakeForward f;
    f.hist = prompt;
    const auto res = hb::rows::spec_decode(f, prompt, plain[0], n, o);
    CHECK(res.ids == plain);
    CHECK(res.accepted > 0U);
    CHECK(res.steps.size() < n - 1U);  // fewer passes than ids: rows > 1 were verified and kept
    std::uint64_t multi = 0;
    for (std::size_t r = 2; r < res.rows_hist.size(); ++r) multi += res.rows_hist[r];
    CHECK(multi > 0U);
    std::printf("spec: ids == plain over %u ids, %zu passes, %llu multi-row, accepted %llu of %llu proposed\n", n, res.steps.size(),
                static_cast<unsigned long long>(multi), static_cast<unsigned long long>(res.accepted),
                static_cast<unsigned long long>(res.proposed));
    FakeForward bad;
    bad.hist = prompt;
    bad.broken_commit = true;
    const auto wrong = hb::rows::spec_decode(bad, prompt, plain[0], n, o);
    CHECK(wrong.ids != plain);  // negative control: rejected rows left in the history change the ids
    std::printf("spec negative control (commit keeps rejected rows): ids differ = %d\n", wrong.ids != plain ? 1 : 0);
}

// The chain arm (MtpSource = the shared MtpChain over the forward's next-token block). Target: next = (5 last + 3 n) % 97
// (n = history length: hardly copyable). Its "next-token block" knows the stream only through what the chain feeds it:
// block position p gets the token at target position p + 1, so with every position fed once, in order, with the right
// successor, it predicts the target exactly; it errs on purpose at every p % 5 == 3. Hidden rows carry their source: the
// target's (verify / prompt rows) hold +p, the block's own output at p holds -(p + 1).
struct FakeMtpForward {
    struct Cfg { std::uint32_t hidden = 4; } cfg;
    static constexpr std::uint32_t kNextnRows = 32;
    std::vector<std::uint32_t> hist, greedy, mtp_tok;
    std::vector<float> rows_h, pre, post;
    std::array<std::uint32_t, kNextnRows> top{};
    std::uint32_t p0 = 0, rows = 0;
    std::uint64_t bad_hidden = 0, bad_position = 0, teacher_rows = 0, chain_rows = 0, wrong_on_purpose = 0, kv_only_rows = 0;
    std::function<void(const float*, std::uint32_t, std::uint32_t)> sink;
    static std::uint32_t next(std::uint32_t last, std::size_t n) { return static_cast<std::uint32_t>((5U * last + 3U * n) % 97U); }
    [[nodiscard]] const Cfg& config() const { return cfg; }
    [[nodiscard]] std::uint32_t position() const { return static_cast<std::uint32_t>(hist.size()); }
    void forward_rows(std::span<const std::uint32_t> toks, bool) {
        if (rows != 0U) throw std::logic_error("fake: pass before commit");
        p0 = position();
        rows = static_cast<std::uint32_t>(toks.size());
        greedy.clear();
        rows_h.assign(std::size_t{rows} * 4U, 0.0F);
        for (std::uint32_t r = 0; r < rows; ++r) {
            hist.push_back(toks[r]);
            greedy.push_back(next(toks[r], hist.size()));
            rows_h[std::size_t{r} * 4U] = static_cast<float>(p0 + r);
        }
    }
    [[nodiscard]] std::span<const std::uint32_t> rows_greedy() const { return greedy; }
    void commit_rows(std::uint32_t k) {
        hist.resize(p0 + k);
        rows = 0;
    }
    [[nodiscard]] const float* rows_hidden() const { return rows_h.data(); }
    // prompt windows of 16 rows: the sink sees the target hidden of every prompt row; returns the anchor
    std::uint32_t prefill(const std::vector<std::uint32_t>& prompt) {
        std::vector<float> win;
        for (std::size_t i = 0; i < prompt.size(); i += 16U) {
            const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(16U, prompt.size() - i));
            win.assign(std::size_t{n} * 4U, 0.0F);
            for (std::uint32_t r = 0; r < n; ++r) {
                hist.push_back(prompt[i + r]);
                win[std::size_t{r} * 4U] = static_cast<float>(hist.size() - 1U);
            }
            if (sink) sink(win.data(), static_cast<std::uint32_t>(i), n);
        }
        return next(hist.back(), hist.size());
    }
    void set_prompt_hidden(std::function<void(const float*, std::uint32_t, std::uint32_t)> s) { sink = std::move(s); }
    void set_nextn_tp(bool) {}  // the pair form (spec_decode / spec_follow with a pair exchange); one node here
    float* alloc_f32(std::size_t n) { return new float[std::max<std::size_t>(n, 1U)](); }
    void free_f32(float* p) noexcept { delete[] p; }
    void copy_f32_async(float* d, const float* s, std::size_t n) { std::copy(s, s + n, d); }
    void read_ids(const std::uint32_t* src, std::uint32_t n, std::uint32_t* host) { std::copy(src, src + n, host); }
    struct Out {
        const std::uint32_t* top;
        const float* hidden_pre;
        const float* hidden_post;
    };
    // outputs: the forward's KV-only contract: rows before the trailing `outputs` get poisoned outputs (NaN hidden): a chain
    // that read one would feed it back (bad_hidden) or draft from it.
    Out nextn_rows(const std::uint32_t* toks, const float* hidden, std::uint32_t R, std::uint32_t pos0, bool want_top,
                   std::uint32_t outputs = ~0U) {
        if (R == 0U || R > kNextnRows || pos0 > mtp_tok.size()) {
            ++bad_position;
            throw std::logic_error("fake: next-token rows / position");
        }
        mtp_tok.resize(pos0);  // a redo at the same positions replaces a rejected chain
        pre.assign(std::size_t{R} * 4U, 0.0F);
        post.assign(std::size_t{R} * 4U, 0.0F);
        for (std::uint32_t r = 0; r < R; ++r) {
            const std::uint32_t p = pos0 + r;
            const float h0 = hidden[std::size_t{r} * 4U];
            if (h0 == static_cast<float>(p)) ++teacher_rows;
            else if (h0 == -static_cast<float>(p)) ++chain_rows;  // the block's own output at p - 1
            else ++bad_hidden;
            mtp_tok.push_back(toks[r]);
            const bool kv_only = r + std::min(outputs, R) < R;
            post[std::size_t{r} * 4U] = kv_only ? std::numeric_limits<float>::quiet_NaN() : -static_cast<float>(p + 1U);
            kv_only_rows += kv_only ? 1U : 0U;
            top[r] = 0U;
        }
        if (want_top) {
            const std::uint32_t p = pos0 + R - 1U;  // block position p holds target token p + 1; predicts target token p + 2
            std::uint32_t pred = next(mtp_tok[p], std::size_t{p} + 2U);
            if (p % 5U == 3U) {
                pred = (pred + 1U) % 97U;
                ++wrong_on_purpose;
            }
            top[R - 1U] = pred;
        }
        return {top.data(), pre.data(), post.data()};
    }
};

void check_spec_mtp() {
    std::vector<std::uint32_t> prompt;
    for (std::uint32_t i = 0; i < 37; ++i) prompt.push_back((i * 13U + 5U) % 97U);
    const std::uint32_t n = 300;
    std::vector<std::uint32_t> plain;
    {
        auto h = prompt;
        plain.push_back(FakeMtpForward::next(h.back(), h.size()));
        while (plain.size() < n) {
            h.push_back(plain.back());
            plain.push_back(FakeMtpForward::next(h.back(), h.size()));
        }
    }
    hb::rows::SpecOptions o;
    o.max_rows = hb::rows::kMaxVerifyRows;
    for (std::uint32_t r = 1; r <= o.max_rows; ++r) o.verify_ms.push_back(57.7 + 11.6 * (r - 1U));
    o.chain_survival = {0.714, 0.633, 0.436, 0.675, 0.256, 0.5};
    FakeMtpForward c;  // copy only (the incumbent) on the same target
    const std::uint32_t anchor_c = c.prefill(prompt);
    const auto copy_only = hb::rows::spec_decode(c, prompt, anchor_c, n, o);
    CHECK(copy_only.ids == plain);
    for (const bool with_copy : {true, false}) {
        FakeMtpForward f;
        hb::rows::MtpSource<FakeMtpForward> source(f);
        source.install(prompt);
        const std::uint32_t anchor = f.prefill(prompt);
        source.finish_prompt(anchor);
        CHECK(anchor == plain[0]);
        auto oo = o;
        oo.copy = with_copy;
        const auto res = hb::rows::spec_decode(f, prompt, anchor, n, oo, source);
        CHECK(res.ids == plain);                            // the chain changes which rows are verified, never an id
        CHECK(f.bad_hidden == 0U && f.bad_position == 0U);  // teacher rows = target hidden at p, chain rows = own output
        CHECK(f.teacher_rows >= prompt.size());             // every prompt position fed to the block
        CHECK(f.kv_only_rows > 0U);                         // the chain used the KV-only rows (never read their outputs)
        CHECK(res.chain_drafts > 0U && f.chain_rows > 0U && f.wrong_on_purpose > 0U);
        CHECK(res.accepted > 2U * copy_only.accepted);      // a predictive block beats the copy source on a non-copy stream
        CHECK(res.steps.size() < copy_only.steps.size());
        std::printf("spec mtp (%s): ids == plain over %u ids, %zu passes (copy only %zu), accepted %llu (copy only %llu), "
                    "block calls %llu rows %llu (teacher %llu chain %llu)\n",
                    with_copy ? "chain+copy+hybrid" : "chain only", n, res.steps.size(), copy_only.steps.size(),
                    static_cast<unsigned long long>(res.accepted), static_cast<unsigned long long>(copy_only.accepted),
                    static_cast<unsigned long long>(res.chain_calls), static_cast<unsigned long long>(res.chain_rows),
                    static_cast<unsigned long long>(f.teacher_rows), static_cast<unsigned long long>(f.chain_rows));
    }
}

// The stated confidence (BlockDrafter::seed_chain_survival -> Arbiter::seed_model_survival) raises the chain arm's
// expected tokens of the first decision; with no seed the flat prior prices it.
struct NullChain final : strix::spec::ChainDrafter {
    void draft(std::span<const std::uint32_t>, std::uint32_t k, std::vector<std::uint32_t>& out) override { out.assign(k, 0U); }
    void accept(std::span<const std::uint32_t>) override {}
};
void check_survival_seed() {
    strix::spec::BlockConfig bc;
    bc.max_block = 8;
    bc.arbiter.windows = {1, 2, 3, 4, 5, 6, 7, 8};
    bc.copy = false;
    bc.hybrid = false;
    bc.chain_max_k = 7;
    bc.fixed_costs = true;
    bc.chain_seed_step_ms = 6.1;
    bc.chain_seed_flush_ms = 9.6;
    NullChain a, b;
    strix::spec::BlockDrafter flat(bc, &a), seeded(bc, &b);
    for (std::uint32_t r = 1; r <= 8; ++r) {
        flat.seed_verify_ms(r, 57.7 + 11.6 * (r - 1U));
        seeded.seed_verify_ms(r, 57.7 + 11.6 * (r - 1U));
    }
    const std::vector<double> high(7, 0.95);
    seeded.seed_chain_survival(high, 64.0);
    const std::vector<std::uint32_t> stream{1, 2, 3};
    flat.commit(stream);
    seeded.commit(stream);
    strix::spec::StepParams p;
    p.max_rows = 8;
    p.context = 3;
    const auto bf = flat.next(p);
    const auto bs = seeded.next(p);
    CHECK(bs.decision.expected_tokens > bf.decision.expected_tokens);
    CHECK(bs.decision.k >= bf.decision.k && bs.decision.k > 1U);
    std::printf("chain survival seed: first decision k %u -> %u, expected tokens %.2f -> %.2f\n", bf.decision.k, bs.decision.k,
                bf.decision.expected_tokens, bs.decision.expected_tokens);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto tiny = strix::hy4::Config::tiny();
        check_rings(tiny, "tiny");
        auto wide = tiny;
        wide.window = 512;  // the pack's sliding window: ring 1024 with 8 verify rows, 512 (clobbered) with one
        check_rings(wide, "window 512");
        if (argc > 1 && std::filesystem::exists(argv[1])) check_rings(strix::hy4::Config::from_gguf(argv[1]), "pack");
        check_words();
        check_nextn_rows(wide, "window 512");
        check_spec();
        check_spec_mtp();
        check_survival_seed();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("hy4 rows tests PASS\n");
    return 0;
}
