// Real cohort and shared drafter, deterministic host Prefill double. This verifies
// orchestration and bookkeeping; its synthetic logits/state do not qualify CUDA.
#include "strix/cuda/qwen_cohort.hpp"
#include <cstdio>
#include <map>
#include <string_view>

namespace qc = strix::cuda::qwen_cohort;
namespace qp = strix::cuda::qwen_prefill;
using V = std::vector<std::uint32_t>;
namespace {
struct Session {
    V committed, truth, staged;
    std::size_t cursor = 0;
    std::uint32_t predictor_rows = 7, predictor_calls = 0;
};
std::map<qp::SessionId, Session> sessions;
std::vector<qp::SessionId> window;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}
V range(std::uint32_t first, std::uint32_t end) {
    V out;
    for (auto i = first; i < end; ++i) out.push_back(i);
    return out;
}
V prompt() {
    V out = range(1, 41);
    out.push_back(99);
    const auto tail = range(1, 9);
    out.insert(out.end(), tail.begin(), tail.end());
    return out;
}
qc::Stream add(qp::SessionId id, qc::ContextDraft* context, const V& input, V truth) {
    auto& s = sessions[id];
    s = {};
    s.committed.assign(input.begin(), input.end() - 1);
    s.truth = std::move(truth);
    if (context) context->drafter.reset(s.committed);
    return {id, input.back(), static_cast<std::uint64_t>(s.committed.size()), 0, context};
}
qc::StepOptions options() {
    qc::StepOptions o;
    o.mtp = true; o.depth = 7; o.budget = 8; o.want_logits = true;
    o.draft_batch = 1; o.wide_rule = 0;
    return o;
}
// Compare every delivered token and synthetic logit, and the entire committed
// input state, against one-row serial execution of the same deterministic truth.
void serial_equal(qp::Prefill& p, const qc::Stream& before, const qc::Stream& after, const qc::StreamStep& step) {
    const auto tested = sessions.at(before.session);
    auto& serial = sessions[100];
    serial = tested;
    serial.cursor -= step.tokens.size();
    serial.committed.resize(static_cast<std::size_t>(before.position));
    qc::Stream stream{100, before.pending, before.position, 0, nullptr};
    auto o = options(); o.mtp = false;
    V tokens; std::vector<float> logits;
    for (std::size_t i = 0; i < step.tokens.size(); ++i) {
        std::vector<qc::Stream> streams{stream};
        const auto r = qc::step(p, streams, o).front();
        stream = streams.front();
        tokens.insert(tokens.end(), r.tokens.begin(), r.tokens.end());
        logits.insert(logits.end(), r.logits.begin(), r.logits.end());
    }
    check(tokens == step.tokens, "complete token sequence equals serial");
    check(logits == step.logits, "complete synthetic logits equal serial");
    check(serial.committed == tested.committed && stream.pending == after.pending && stream.position == after.position,
          "complete synthetic state equals serial");
    check(step.mtp_accepted <= step.mtp_proposed && step.context_accepted <= step.context_proposed,
          "acceptance never exceeds source proposals");
    check(step.mtp_proposed + step.context_proposed + 1 == step.rows &&
          step.mtp_accepted + step.context_accepted + 1 == step.tokens.size(), "source counts partition window");
}
std::vector<qc::StreamStep> run(qp::Prefill& p, std::vector<qc::Stream>& streams, const qc::StepOptions& o) {
    const auto before = streams;
    auto result = qc::step(p, streams, o);
    for (std::size_t i = 0; i < streams.size(); ++i) serial_equal(p, before[i], streams[i], result[i]);
    return result;
}
void zero_depth(qp::Prefill& p) {
    for (const auto mode : {qc::ContextMode::first, qc::ContextMode::copy, qc::ContextMode::ctx}) {
        sessions.clear();
        qc::ContextDraft context(mode);
        const auto input = prompt();
        std::vector<qc::Stream> streams{add(0, &context, input, range(9, 50))};
        for (std::uint32_t i = 1; i < 8; ++i) streams.push_back(add(i, nullptr, input, range(9, 50)));
        auto o = options();
        for (int cycle = 0; cycle < 2; ++cycle) {
            const auto r = run(p, streams, o);
            check(r[0].rows == 1 && r[0].drafted == 0, "C8 starts with zero draft capacity");
        }
        std::uint32_t suffix = 0, support = 0;
        check(context.drafter.lookup({}, 7, suffix, support) == range(11, 18) && suffix >= 8,
              "zero-depth history includes original pending exactly once");
        streams.resize(1);
        if (mode == qc::ContextMode::ctx) sessions[0].predictor_rows = 0;
        const auto r = run(p, streams, o).front();
        check(r.context_proposed == 7 && r.context_accepted == 7, "C8-to-C1 context proposal uses correct suffix");
        check(context.copy_first == (mode != qc::ContextMode::ctx), "transition takes intended context path");
    }
}
void copy(qp::Prefill& p, bool reject) {
    qc::ContextDraft context(qc::ContextMode::first);
    auto truth = range(9, 60);
    if (reject) truth[0] = 500;
    std::vector<qc::Stream> streams{add(0, &context, prompt(), truth)};
    auto r = run(p, streams, options()).front();
    check(context.copy_first && sessions[0].predictor_calls == 0, "forced copy-first skips predictor");
    check(r.drafted == 0 && r.mtp_proposed == 0 && r.mtp_accepted == 0, "copy cannot inflate MTP receipts");
    check(r.context_proposed == 7 && r.context_accepted == (reject ? 0U : 7U), "copy acceptance follows verification");
    if (reject) {
        r = run(p, streams, options()).front();
        check(!context.copy_first && sessions[0].predictor_calls == 1 && r.mtp_accepted == 7,
              "rejected copy falls back to MTP next cycle");
    } else {
        r = run(p, streams, options()).front();
        check(context.copy_first_steps == 2 && r.mtp_accepted == 0 && r.context_accepted == 7,
              "repeated copy stays separate from MTP");
    }
}
void tail(qp::Prefill& p, bool shadow) {
    qc::ContextDraft context(shadow ? qc::ContextMode::first : qc::ContextMode::ctx);
    // Short anchor prevents copy-first; a two-token predictor prefix enables extension.
    V input = range(1, 41); input.insert(input.end(), {99, 1, 2});
    std::vector<qc::Stream> streams{add(0, &context, input, range(3, 40))};
    sessions[0].predictor_rows = 2;
    auto o = options();
    o.max_positions = streams[0].position + 5;
    const auto r = run(p, streams, o).front();
    check(!context.copy_first && r.drafted == 2 && r.mtp_proposed == 2 && r.mtp_accepted == 2,
          "predictor work and accepted prefix are independent of extension");
    check(r.context_proposed == (shadow ? 0U : 2U) && r.context_accepted == r.context_proposed,
          "context tail counts exclude clipped and shadow-only rows");
    check(context.extended_steps == (shadow ? 0U : 1U), "extension delivery path is explicit");
    check(context.mtp_expected == 2.0, "context acceptance cannot inflate predictor expectation");
}
void mixed(qp::Prefill& p) {
    qc::ContextDraft context(qc::ContextMode::first);
    std::vector<qc::Stream> streams{add(0, &context, prompt(), range(9, 50)),
                                  add(1, nullptr, V{70, 71, 72}, range(73, 100))};
    const auto r = run(p, streams, options());
    check(context.copy_first && r[0].context_proposed == 3 && r[0].context_accepted == 3 && r[0].drafted == 0,
          "mixed stream copy occupies its fair share");
    check(r[1].mtp_proposed == 3 && r[1].mtp_accepted == 3 && r[1].context_proposed == 0,
          "mixed stream predictor counters remain request-local");
}
} // namespace

namespace strix::cuda::qwen_prefill {
struct Prefill::Impl {};
Prefill::Prefill(const Config&) : impl_(std::make_unique<Impl>()) {}
Prefill::~Prefill() = default;
std::uint32_t Prefill::max_positions() const { return 4096; }
std::uint64_t Prefill::length(SessionId id) const { return sessions.at(id).committed.size(); }
Draft Prefill::draft(SessionId id, std::uint32_t, std::uint32_t depth, float) {
    auto& s = sessions.at(id); ++s.predictor_calls;
    Draft result;
    for (std::uint32_t i = 0; i < std::min(depth, s.predictor_rows); ++i) {
        result.tokens.push_back(s.truth.at(s.cursor + i)); result.probability.push_back(0.9F);
    }
    return result;
}
std::vector<Draft> Prefill::draft_cohort(const std::vector<DraftRequest>& requests) {
    std::vector<Draft> result;
    for (const auto& r : requests)
        result.push_back(r.depth ? draft(r.session, r.pending, r.depth, r.min_probability) : Draft{});
    return result;
}
std::vector<StepResult> Prefill::verify_cohort(const std::vector<CohortRows>& cohort, bool want_logits) {
    window.clear(); std::vector<StepResult> result;
    for (const auto& row : cohort) {
        auto& s = sessions.at(row.session);
        s.staged.assign(row.tokens, row.tokens + row.rows); window.push_back(row.session);
        StepResult r;
        for (std::uint32_t i = 0; i < row.rows; ++i) {
            const auto t = s.truth.at(s.cursor + i); r.argmax.push_back(t);
            if (want_logits) { r.logits.push_back(static_cast<float>(t)); r.logits.push_back(-1.0F); }
        }
        result.push_back(std::move(r));
    }
    return result;
}
void Prefill::commit_cohort(const std::vector<std::uint32_t>& keep) {
    for (std::size_t i = 0; i < keep.size(); ++i) {
        auto& s = sessions.at(window.at(i));
        s.committed.insert(s.committed.end(), s.staged.begin(), s.staged.begin() + keep[i]);
        s.cursor += keep[i]; s.staged.clear();
    }
}
} // namespace strix::cuda::qwen_prefill

int main(int argc, char** argv) {
    try {
        if (argc != 2) return 2;
        qp::Prefill p(qp::Config{});
        const std::string_view name(argv[1]);
        if (name == "zero_depth") zero_depth(p);
        else if (name == "copy") copy(p, false);
        else if (name == "rejection") copy(p, true);
        else if (name == "mixed") mixed(p);
        else if (name == "tail") tail(p, false);
        else if (name == "shadow") tail(p, true);
        else return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what()); return 1;
    }
    return failures ? 1 : 0;
}
