// e4b-eagle through the real engine::Coordinator (needs the staged-rows draft API). A row-invariant mock backend decodes the
// same prompts OFF and with e4b-eagle sessions (oracle drafter for acceptance, host reference drafter for the real
// code path), single and concurrent streams, DraftMode::on with k = 1..7; ids must equal OFF exactly.
#include "common.hpp"
#include "gemma4/eagle/session.hpp"
#include "gemma4/engine/coordinator.hpp"
#include "mock.hpp"

#if !defined(STRIX_G4_DRAFT_ROWS_API)
int main() {
    std::printf("SKIP: staged-rows draft API (DraftSession propose_window/wants_hidden/stage_rows) is not applied\n");
    return 0;
}
#else
using namespace eagle_test;
namespace eg = strix::gemma4::eagle;
namespace en = strix::gemma4::engine;

namespace {
struct State final : en::StreamState {
    MockStream s;
    std::unique_ptr<KvTxn> begin() override { return s.begin(); }
    std::uint64_t frontier() const override { return s.tokens.size(); }
    std::string digest() const override { return std::to_string(s.hashes.back()); }
};
struct Backend final : en::Backend {
    MockTarget target;
    ModelConfig c;
    Backend() { c.hidden = target.hidden; c.vocab = target.vocab; }
    State* last = nullptr; // the coordinator creates the stream right before calling the draft factory
    std::unique_ptr<en::StreamState> create_stream() override {
        auto state = std::make_unique<State>();
        last = state.get();
        return state;
    }
    ForwardResult forward_rows(std::span<const RowPlan> rows) override { return target.forward_rows(rows); }
    const ModelConfig& config() const override { return c; }
    strix::serve::Json identity() const override { return strix::serve::Json::object(); }
};
struct Fixture {
    Config c = small_config();
    Weights w = Weights::random(c, 31);
    eagle::Matrix emb = embedding_table(c, 32);
    std::vector<float> rope = eg::make_rope_table(1024);
};

#if defined(STRIX_G4_DRAFT_MULTI_API)
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
// Multi-stream: forwards every hook to an eagle::Session and declares (or not) a per-depth-cost drafter, as the MTP session
// does (candidate) and DFlash does not (default false).
struct MultiCandidate final : en::DraftSession {
    std::unique_ptr<en::DraftSession> inner;
    bool candidate;
    MultiCandidate(std::unique_ptr<en::DraftSession> s, bool c) : inner(std::move(s)), candidate(c) {}
    std::array<std::uint32_t, 7> propose(std::uint32_t a, std::uint64_t p) override { return inner->propose(a, p); }
    std::array<std::uint32_t, 7> propose_window(std::uint32_t a, std::uint64_t p, std::uint32_t k) override {
        return inner->propose_window(a, p, k);
    }
    void append(std::span<const float> t, std::span<const std::uint64_t> p) override { inner->append(t, p); }
    void attach(en::StreamState& state) override { inner->attach(state); }
    bool wants_hidden() const override { return inner->wants_hidden(); }
    void stage_rows(std::span<const float> t, std::span<const float> h, std::span<const std::uint32_t> n,
                    std::span<const std::uint64_t> p) override { inner->stage_rows(t, h, n, p); }
    void stage(std::span<const float> t, std::span<const std::uint64_t> p) override { inner->stage(t, p); }
    void publish() override { inner->publish(); }
    void discard() noexcept override { inner->discard(); }
    bool multi_stream_candidate() const override { return candidate; }
};
#endif
std::vector<std::vector<std::uint32_t>> decode(Backend& backend, en::DraftFactory drafts, const std::vector<std::vector<std::uint32_t>>& prompts,
                                               std::uint32_t k, en::DraftMode mode, dflash::Acceptance* total) {
    en::Coordinator coordinator(backend, std::move(drafts));
    std::vector<std::size_t> slots;
    for (const auto& p : prompts) {
        en::Request r;
        r.prompt = p;
        r.max_tokens = 150;
        r.dflash = mode;
        r.draft_k = k;
        slots.push_back(coordinator.submit(r));
    }
    coordinator.run();
    std::vector<std::vector<std::uint32_t>> out;
    for (const auto slot : slots) {
        const auto& r = coordinator.result(slot);
        EAGLE_CHECK(r.status == en::Status::stopped || r.status == en::Status::length, "status %s error %s", en::status_name(r.status),
                    r.error.c_str());
        out.push_back(r.ids);
        if (total) { total->attempted += r.acceptance.attempted; total->accepted += r.acceptance.accepted; }
    }
    return out;
}
} // namespace

int main() {
    try {
        Fixture fx;
        Rng r{41};
        // Prompt lengths at the prefill-window edges (the coordinator prefills prompt[0..n-2] in <= 8-row windows; the
        // last prompt token is the first decode anchor): 1, 2, 8, 9, 16, 17 plus two random lengths.
        std::vector<std::vector<std::uint32_t>> prompts;
        for (const std::size_t n : {std::size_t(1 + r.below(40)), std::size_t(1), std::size_t(2), std::size_t(8), std::size_t(9),
                                    std::size_t(16), std::size_t(17), std::size_t(1 + r.below(40))}) {
            std::vector<std::uint32_t> p(n);
            for (auto& t : p) t = 2 + r.below(1000);
            prompts.push_back(p);
        }
        Backend off_backend;
        std::vector<std::vector<std::uint32_t>> off;
        for (const auto& p : prompts) off.push_back(decode(off_backend, {}, {p}, 0, en::DraftMode::off, nullptr)[0]);
        dflash::Acceptance oracle_stats, real_stats;
        for (std::uint32_t k = 1; k <= 7; k += 3) {
            // Oracle drafter inside an eagle::Session: exercises acceptance, bonus and next-token staging.
            Backend b1;
            en::DraftFactory oracle = [&b1]() -> std::unique_ptr<en::DraftSession> {
                return std::make_unique<eg::Session>(std::make_unique<Oracle>(b1.target, b1.last->s, small_config()));
            };
            const auto exact = decode(b1, oracle, {prompts[1]}, k, en::DraftMode::on, &oracle_stats);
            EAGLE_CHECK(exact[0] == off[1], "k %u oracle ids differ from OFF", k);
            // Real code path: host reference eagle drafter (random weights; acceptance near zero but every hook runs).
            Backend b2;
            auto factory = eg::host_drafts(fx.w, {fx.rope, 1024}, fx.emb.view());
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                const auto single = decode(b2, factory, {prompts[i]}, k, en::DraftMode::on, &real_stats);
                EAGLE_CHECK(single[0] == off[i], "k %u prompt length %zu single-stream ids differ from OFF", k, prompts[i].size());
                Backend b4;
                const auto exact1 = decode(b4, [&b4]() -> std::unique_ptr<en::DraftSession> {
                    return std::make_unique<eg::Session>(std::make_unique<Oracle>(b4.target, b4.last->s, small_config()));
                }, {prompts[i]}, k, en::DraftMode::on, &oracle_stats);
                EAGLE_CHECK(exact1[0] == off[i], "k %u prompt length %zu oracle ids differ from OFF", k, prompts[i].size());
            }
            Backend b3;
            const std::vector<std::vector<std::uint32_t>> three{prompts[2], prompts[4], prompts[6]};
            const auto multi = decode(b3, factory, three, k, en::DraftMode::on, &real_stats);
            const std::size_t idx[] = {2, 4, 6};
            for (std::size_t i = 0; i < 3; ++i) EAGLE_CHECK(multi[i] == off[idx[i]], "k %u stream %zu ids differ from OFF", k, i);
        }
#if defined(STRIX_G4_DRAFT_MULTI_API)
        // multi-stream auto through the real coordinator: 4 concurrent auto streams, ids == OFF always; a
        // per-depth-cost drafter drafts at >1 stream with window <= 1 (probes at least), a DFlash-like drafter and
        // STRIX_G4_DRAFT_MULTI=0 never draft while >1 stream is live.
        for (const bool candidate : {true, false})
            for (const char* env : {static_cast<const char*>(nullptr), "0"}) {
                set_env("STRIX_G4_DRAFT_MULTI", env);
                Backend b5;
                en::DraftFactory f = [&b5, candidate]() -> std::unique_ptr<en::DraftSession> {
                    return std::make_unique<MultiCandidate>(
                        std::make_unique<eg::Session>(std::make_unique<Oracle>(b5.target, b5.last->s, small_config())), candidate);
                };
                en::Coordinator coordinator(b5, f);
                const std::size_t idx[] = {0, 2, 4, 6};
                std::vector<std::size_t> slots;
                for (const auto i : idx) {
                    en::Request req;
                    req.prompt = prompts[i];
                    req.max_tokens = 150;
                    req.dflash = en::DraftMode::automatic;
                    slots.push_back(coordinator.submit(req));
                }
                coordinator.run();
                std::uint64_t proposals = 0, cycles = 0;
                for (std::size_t j = 0; j < slots.size(); ++j) {
                    const auto& res = coordinator.result(slots[j]);
                    EAGLE_CHECK(res.ids == off[idx[j]], "multi auto (candidate %d, MULTI=%s) stream %zu ids differ from OFF",
                                int(candidate), env ? env : "unset", j);
                    EAGLE_CHECK(res.multistream_auto_max_window <= 1, "multi auto window %u above the k=1 cap",
                                unsigned(res.multistream_auto_max_window));
                    proposals += res.multistream_auto_proposals; cycles += res.multistream_auto_cycles;
                }
                EAGLE_CHECK(cycles > 0, "multi-stream cycles happened");
                if (candidate && !env) EAGLE_CHECK(proposals > 0, "per-depth drafter drafts at >1 stream");
                else EAGLE_CHECK(proposals == 0, "candidate %d MULTI=%s: auto must stay k=0 at >1 stream", int(candidate),
                                 env ? env : "unset");
                std::printf("coordinator: multi-stream auto candidate=%d STRIX_G4_DRAFT_MULTI=%s: 4 streams == OFF, %llu draft rows in %llu multi-stream cycles\n",
                            int(candidate), env ? env : "unset", (unsigned long long)proposals, (unsigned long long)cycles);
            }
        set_env("STRIX_G4_DRAFT_MULTI", nullptr);
#endif
        std::printf("coordinator: eagle sessions == OFF for k in {1,4,7}, prompt lengths 1/2/8/9/16/17/random, 1 and 3 streams; oracle accepted %llu/%llu, host drafter (1 and 3 streams) %llu/%llu\n",
                    (unsigned long long)oracle_stats.accepted, (unsigned long long)oracle_stats.attempted,
                    (unsigned long long)real_stats.accepted, (unsigned long long)real_stats.attempted);
        EAGLE_CHECK(oracle_stats.accepted > 0, "oracle acceptance through the coordinator");
        EAGLE_CHECK(real_stats.attempted > 0, "drafting happened through the coordinator");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::printf(eagle_test::failures ? "FAILED %d checks\n" : "ALL OK\n", eagle_test::failures);
    return eagle_test::failures ? 1 : 0;
}
#endif
