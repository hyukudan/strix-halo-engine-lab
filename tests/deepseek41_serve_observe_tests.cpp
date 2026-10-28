#include "strix/deepseek41_serve_observe.hpp"
#include <algorithm>
#include "strix/serve/api.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// Per-request draft accounting of the serve (deepseek41_serve_observe.hpp) against the speculator's own rule: every
// window of n rows of a decoding sequence verifies n - 1 drafts and keeps accepted + 1 of them; prompt rows carry none.
namespace {

namespace so = strix::deepseek41::serve_observe;

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Seq {
    std::vector<std::uint32_t> prompt, generated;
    std::size_t fed = 0;
    [[nodiscard]] bool reading_prompt() const noexcept { return fed < prompt.size(); }
};

// One sequence's part of a step: its rows, and (decoding) how many drafts were accepted.
struct Part {
    std::uint32_t rows = 1, accepted = 0;
};

// Applies a step the way BatchSpeculator::finish does (host side): decoding owners keep accepted + 1 tokens, prompt
// owners feed their rows and emit one token when the prompt is complete. Returns what finish adds to drafts_verified
// and drafts_accepted.
so::Drafts apply(std::vector<Seq*>& seqs, const std::vector<Part>& parts) {
    so::Drafts stats;
    for (std::size_t i = 0; i < seqs.size(); ++i) {
        auto& s = *seqs[i];
        const auto& p = parts[i];
        if (!s.reading_prompt()) {
            const std::uint32_t kept = std::min(p.accepted, p.rows - 1U) + 1U;
            for (std::uint32_t k = 0; k < kept; ++k) s.generated.push_back(7U);
            if (p.rows > 1U) {
                stats.proposed += p.rows - 1U;
                stats.accepted += kept - 1U;
            }
        } else {
            s.fed += p.rows;
            if (!s.reading_prompt()) s.generated.push_back(9U);
        }
    }
    return stats;
}

void step(std::vector<Seq*>& seqs, const std::vector<Part>& parts, const std::string& what) {
    std::vector<std::uint32_t> owner;
    for (std::uint32_t i = 0; i < parts.size(); ++i) owner.insert(owner.end(), parts[i].rows, i);
    so::StepDraftTally<Seq> tally;
    tally.begin(seqs, owner);
    const auto want = apply(seqs, parts);
    so::Drafts got;
    for (std::size_t i = 0; i < seqs.size(); ++i) {
        const auto d = tally.after(i, *seqs[i]);
        got.proposed += d.proposed;
        got.accepted += d.accepted;
    }
    expect(got.proposed == want.proposed && got.accepted == want.accepted,
           what + ": tally " + std::to_string(got.proposed) + "/" + std::to_string(got.accepted) + " vs speculator " +
               std::to_string(want.proposed) + "/" + std::to_string(want.accepted));
}

// ---- prompt accounting through the serve's path: admission (prefix-cache resume or a parked imported state) ->
// completion (GenerationStats as the serve's finish builds them) -> GenerationQueue::finish -> ApiService (usage,
// status ring record, observed totals, metrics). The engine below mirrors the serve's lead loop on the CPU: it takes
// jobs, applies the admission accounting the serve applies (PromptAccount, the same calls), and finishes them.
namespace sv = strix::serve;

std::shared_ptr<const sv::BpeTokenizer> byte_tokenizer() {  // bytes, no merges: one token per byte; the chat specials
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        int cp = b;
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if (!direct) cp = 256 + extra++;
        std::string s;
        if (cp < 0x80) s.push_back(static_cast<char>(cp));
        else { s.push_back(static_cast<char>(0xC0 | (cp >> 6))); s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        tokens.push_back(s);
        types.push_back(1);
    }
    for (const char* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) { tokens.push_back(special); types.push_back(3); }
    for (const char* special : {"<tool_call>", "</tool_call>", "<think>", "</think>"}) { tokens.push_back(special); types.push_back(4); }
    return sv::BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "qwen35");
}

struct Sink final : sv::HttpResponseWriter {
    std::string body;
    int status = 0;
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string, std::string>>&) override { status = s; body = b; }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override { return false; }
    bool write(std::string_view) override { return false; }
    void end_stream() override {}
    [[nodiscard]] bool client_gone() const override { return false; }
};
sv::Json call(sv::ApiService& api, const std::string& method, const std::string& path, const std::string& body = {},
              const std::string& query = {}) {
    sv::HttpRequest r;
    r.method = method;
    r.path = path;
    r.query = query;
    r.body = body;
    Sink s;
    api.handle(r, s);
    if (path == "/metrics") {
        sv::Json text = sv::Json::object();
        text.set("text", sv::Json(s.body));
        return text;
    }
    return sv::parse_json(s.body);
}
const sv::Json& at(const sv::Json& json, std::string_view key) {
    if (const auto* value = json.find(key)) return *value;
    throw std::runtime_error("missing key " + std::string(key) + " in " + json.dump());
}
std::uint64_t num(const sv::Json& json, std::string_view key) { return static_cast<std::uint64_t>(at(json, key).as_int()); }
double metric(const std::string& text, const std::string& series) {
    const auto pos = text.find("\n" + series + " ");
    return pos == std::string::npos ? -1.0 : std::stod(text.substr(pos + series.size() + 2U));
}

// The admission a prompt gets, chosen by its first content byte (the serve decides by its parked states and its cache):
// 'I' a parked state of 8,192 rows continues it; 'F' a parked state of the whole prompt; 'P' the device prefill
// resumes 4,096 rows from the prefix cache; 'W' waits for its device prefill, then a parked state of 8,192 rows arrives
// (take_parked); 'N' nothing reused.
constexpr std::uint32_t kImported = 8192, kResumed = 4096;

void lead_like(sv::GenerationQueue& queue, const std::shared_ptr<const sv::BpeTokenizer>& tok) {
    const std::uint32_t im_end = tok->find("<|im_end|>");
    for (;;) {
        auto taken = queue.take(true);
        if (!taken) {
            if (queue.stopping()) break;
            continue;  // a cancellation-only batch returns control without shutting down
        }
        const auto& prompt = taken->job.prompt;
        // the scenario: the first byte of the user content (the rendered prompt starts with the template's tokens)
        char kind = 'N';
        for (const auto id : prompt) {
            const auto bytes = tok->token_bytes(id);
            if (bytes.size() == 1U && (bytes[0] == 'I' || bytes[0] == 'F' || bytes[0] == 'P' || bytes[0] == 'W' || bytes[0] == 'D' ||
                                       bytes[0] == 'X')) {
                kind = bytes[0];
                break;
            }
        }
        so::PromptAccount account;  // Running::account
        const auto plen = static_cast<std::uint32_t>(prompt.size());
        switch (kind) {
            case 'I': account.continued(kImported); break;  // admission: match_parked >= 0
            case 'F': account.continued(plen); break;       // the whole prompt imported: its first token is known
            case 'P': account.resumed(kResumed); break;     // prefill_done: hit.resume
            case 'W': account.continued(kImported); break;  // take_parked (its prefill was aborted before prefill_done)
            default: break;
        }
        // two decode steps with DSpark windows: the serve attributes 3 drafts (2 accepted) per armed step to an observed
        // request ('X': the armed period expires between the steps)
        std::uint64_t proposed = 0, accepted = 0;
        for (int step = 0; step < 2; ++step) {
            if (kind == 'X' && step == 1) sv::observe_disarm();
            if (sv::observe_armed() && taken->job.observed) proposed += 3U, accepted += 2U;
        }
        std::vector<std::uint32_t> out = tok->encode("ok", false);
        out.push_back(im_end);
        (void)taken->job.on_tokens(out);
        sv::GenerationStats st;  // the serve's finish()
        st.prompt_tokens = prompt.size();
        account.report(st);
        st.generated_tokens = out.size();
        st.cycles = out.size();
        st.prefill_ms = 1000.0;  // emulated: one second from admission to the first token
        st.decode_ms = 30.0;
        st.finish = "stop";
        if (taken->job.observation_complete()) {  // the serve's finish: armed-only measurements of a complete period
            st.proposed = proposed;
            st.accepted = accepted;
            st.first_token_ms = 1000.0;
        }
        queue.finish(*taken, st);
    }
}

void accounting_tests() {
    const auto tok = byte_tokenizer();
    sv::GenerationQueue queue;
    sv::ApiConfig config;
    config.model_id = "ds-accounting";
    config.stop_tokens = {tok->find("<|im_end|>")};
    sv::ApiService api(config, tok, queue, 65536, false);
    std::thread engine([&] { lead_like(queue, tok); });
    (void)call(api, "GET", "/metrics");  // armed: the ring records carry the split
    expect(sv::observe_armed(), "armed");
    struct Case {
        char kind;
        std::uint32_t content;  // user content bytes
    };
    for (const auto& c : {Case{'I', 9000}, Case{'F', 8192}, Case{'P', 9000}, Case{'W', 9000}, Case{'N', 9000}}) {
        const std::string content(c.content, c.kind);
        const auto before = queue.observed();
        // /v1/completions: the prompt is the content itself (one token per byte)
        const auto body = std::string(R"({"max_tokens":4,"prompt":")") + content + "\"}";
        const auto done = call(api, "POST", "/v1/completions", body);
        const auto& usage = at(done, "usage");
        const std::uint64_t prompt = num(usage, "prompt_tokens");
        const std::uint64_t imported = c.kind == 'I' || c.kind == 'W' ? kImported : c.kind == 'F' ? prompt : 0U;
        const std::uint64_t cached = c.kind == 'P' ? kResumed : 0U;
        const std::uint64_t computed = prompt - imported - cached;
        const std::string what = std::string("case ") + c.kind + ": ";
        expect(prompt == c.content, what + "completions prompt = the content bytes (" + std::to_string(prompt) + ")");
        // client usage: the union, as reported (cached + imported: rows not prefilled for this request)
        expect(num(at(usage, "prompt_tokens_details"), "cached_tokens") == cached + imported, what + "usage cached_tokens = cached + imported");
        // status ring record: disjoint categories, the computed rows and their rate
        const auto status = call(api, "GET", "/api/status");
        const auto& e = at(status, "requests").as_array().at(0);
        expect(num(e, "cached_tokens") == cached && num(e, "imported_tokens") == imported && num(e, "computed_tokens") == computed,
               what + "ring split " + e.dump());
        expect(num(e, "cached_tokens") + num(e, "imported_tokens") + num(e, "computed_tokens") == prompt, what + "prompt = cached + imported + computed");
        expect(at(e, "prefill_tok_s").as_double() == static_cast<double>(computed), what + "prefill rate of the computed rows (1 s) " + e.dump());
        const auto after = queue.observed();
        expect(after.computed_tokens - before.computed_tokens == computed && after.imported_tokens - before.imported_tokens == imported &&
                   after.cached_tokens - before.cached_tokens == cached,
               what + "observed totals add the split once");
        const auto metrics = at(call(api, "GET", "/metrics"), "text").as_string();
        expect(metric(metrics, "strix_prompt_tokens_computed_total") == static_cast<double>(after.computed_tokens) &&
                   metric(metrics, "strix_prompt_tokens_imported_total") == static_cast<double>(after.imported_tokens),
               what + "metrics counters = the observed totals");
        // Anthropic usage: input_tokens = prompt - (cached + imported), cache_read_input_tokens = cached + imported
        const auto msg = call(api, "POST", "/v1/messages",
                              std::string(R"({"max_tokens":4,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":")") + content + "\"}]}");
        const auto& mu = at(msg, "usage");
        const auto mrec = at(call(api, "GET", "/api/status"), "requests").as_array().at(0);
        const std::uint64_t mprompt = num(mrec, "prompt_tokens");
        const std::uint64_t mimported = c.kind == 'F' ? mprompt : imported;
        expect(num(mu, "cache_read_input_tokens") == cached + mimported && num(mu, "input_tokens") == mprompt - cached - mimported,
               what + "anthropic usage reports the union " + mu.dump());
        expect(num(mrec, "computed_tokens") == mprompt - cached - mimported, what + "anthropic request ring split " + mrec.dump());
    }
    {  // the stated case in numbers: 9,000 prompt, 8,192 imported, 808 computed at 808 tok/s
        const auto done = call(api, "POST", "/v1/completions", std::string(R"({"max_tokens":4,"prompt":")") + std::string(9000, 'I') + "\"}");
        const auto e = at(call(api, "GET", "/api/status"), "requests").as_array().at(0);
        expect(num(at(done, "usage"), "prompt_tokens") == 9000U && num(at(at(done, "usage"), "prompt_tokens_details"), "cached_tokens") == 8192U &&
                   num(e, "computed_tokens") == 808U && num(e, "imported_tokens") == 8192U && num(e, "cached_tokens") == 0U &&
                   num(e, "reused_tokens") == 0U && at(e, "prefill_tok_s").as_double() == 808.0,
               "9000 / 8192 / 808: " + e.dump());
    }
    {  // drafts: reported for a request armed to its end; a request whose armed period expired between its steps reports
       // neither its partial drafts nor armed fields (response and ring), and is not counted
        (void)call(api, "GET", "/metrics");
        const auto full = call(api, "POST", "/v1/completions", R"({"max_tokens":4,"prompt":"DDDD"})");
        const auto fe = at(call(api, "GET", "/api/status"), "requests").as_array().at(0);
        expect(num(at(full, "strix"), "mtp_proposed") == 6U && num(at(full, "strix"), "mtp_accepted") == 4U && num(fe, "mtp_proposed") == 6U &&
                   num(fe, "mtp_accepted") == 4U && fe.find("first_token_ms") != nullptr,
               "armed throughout: 6/4 drafts in the response and the ring " + fe.dump());
        const auto before = queue.observed().requests;
        const auto part = call(api, "POST", "/v1/completions", R"({"max_tokens":4,"prompt":"XXXX"})");
        expect(!sv::observe_armed(), "the period expired during the request");
        const auto pe = at(call(api, "GET", "/api/status"), "requests").as_array().at(0);
        expect(num(at(part, "strix"), "mtp_proposed") == 0U && num(at(part, "strix"), "mtp_accepted") == 0U && num(pe, "mtp_proposed") == 0U &&
                   num(pe, "mtp_accepted") == 0U && pe.find("first_token_ms") == nullptr && pe.find("computed_tokens") == nullptr,
               "expired mid-request: no partial drafts (response, ring) and no armed field " + pe.dump());
        expect(queue.observed().requests == before, "expired mid-request: not counted");
    }
    {  // the admission calls replace each other (one category per request)
        so::PromptAccount a;
        a.resumed(4096);
        a.continued(8192);
        expect(a.cached == 0U && a.imported == 8192U, "an imported state replaces a resumed prefix");
        a.resumed(100);
        expect(a.cached == 100U && a.imported == 0U, "a resumed prefix replaces an import");
    }
    queue.shutdown();
    engine.join();
}

}  // namespace

int main() {
    try {
        accounting_tests();
    } catch (const std::exception& e) {
        expect(false, std::string("accounting tests: ") + e.what());
    }
    Seq a, b, c, d;
    a.prompt.assign(4, 1U); a.fed = 4; a.generated = {5U};          // decoding
    b.prompt.assign(4, 1U); b.fed = 4; b.generated = {5U, 6U};      // decoding
    c.prompt.assign(20, 1U); c.fed = 8;                             // reading its prompt in rows
    d.prompt.assign(4, 1U); d.fed = 4; d.generated = {5U};          // decoding
    std::vector<Seq*> all{&a, &b, &c, &d};
    step(all, {{4, 2}, {1, 0}, {8, 0}, {6, 5}}, "window 4 (2 kept drafts), plain row, prompt rows, window 6 all accepted");
    step(all, {{4, 0}, {3, 2}, {4, 0}, {2, 1}}, "nothing accepted, all accepted, the prompt completes (emits one token)");
    step(all, {{1, 0}, {1, 0}, {1, 0}, {1, 0}}, "plain rows only: nothing proposed");
    // interleave: a subset of the requests steps (the owner index is into the step's own sequences)
    std::vector<Seq*> subset{&d, &a};
    step(subset, {{5, 3}, {2, 0}}, "a subset of the sequences in another order");
    {  // per request, not only in sum
        Seq e;
        e.prompt.assign(2, 1U); e.fed = 2; e.generated = {5U};
        std::vector<Seq*> one{&e};
        so::StepDraftTally<Seq> tally;
        const std::vector<std::uint32_t> owner{0, 0, 0, 0, 0};
        tally.begin(one, owner);
        for (int k = 0; k < 3; ++k) e.generated.push_back(7U);  // 2 of 4 drafts accepted
        const auto r = tally.after(0, e);
        expect(r.proposed == 4U && r.accepted == 2U, "one request: 4 proposed, 2 accepted");
        expect(tally.after(1, e).proposed == 0U, "an owner the step does not have: nothing");
        so::StepDraftTally<Seq> none;
        none.begin(one, owner);
        expect(none.after(0, e).proposed == 0U, "no growth (the step was not applied): nothing");
    }
    {  // a step of the most owners a configuration can have: every one attributed (none truncated)
        static_assert(so::StepDraftTally<Seq>::max_owners == so::kMaxStepOwners);
        std::vector<Seq> many(so::kMaxStepOwners);
        std::vector<Seq*> ptrs;
        std::vector<std::uint32_t> owner;
        for (std::size_t i = 0; i < many.size(); ++i) {
            many[i].prompt.assign(1, 1U);
            many[i].fed = 1;
            many[i].generated = {5U};
            ptrs.push_back(&many[i]);
            owner.insert(owner.end(), 3, static_cast<std::uint32_t>(i));  // a window of 3 rows each
        }
        so::StepDraftTally<Seq> tally;
        tally.begin(ptrs, owner);
        bool every = true;
        for (std::size_t i = 0; i < many.size(); ++i) {
            many[i].generated.push_back(6U);
            many[i].generated.push_back(7U);  // 1 of 2 drafts accepted
            const auto r = tally.after(i, many[i]);
            every = every && r.proposed == 2U && r.accepted == 1U;
        }
        expect(every, "all 64 owners of a step attributed");
    }
    {  // the armed per-step attribution (begin, then after per owner) plus the step's token appends it reads: the same cost
       // at the end of 200,000 steps as at the start (nothing in it grows with the steps or the tokens served); 8 decoding
       // owners in windows of 4 rows, 2 drafts accepted per window. Times: the median of 10 windows of 1,000 steps at the
       // start and at the end (robust to a descheduled window); the bound is loose, the attribution totals are exact
        constexpr std::size_t kSteps = 200000, kWindow = 1000, kWindows = 10;
        std::vector<Seq> seqs(8);
        std::vector<Seq*> ptrs;
        std::vector<std::uint32_t> owner;
        for (std::size_t i = 0; i < seqs.size(); ++i) {
            seqs[i].prompt.assign(1, 1U);
            seqs[i].fed = 1;
            seqs[i].generated.reserve(kSteps * 3U + 1U);
            seqs[i].generated = {5U};
            ptrs.push_back(&seqs[i]);
            owner.insert(owner.end(), 4, static_cast<std::uint32_t>(i));
        }
        so::StepDraftTally<Seq> tally;
        std::uint64_t proposed = 0, accepted = 0;
        const auto window_ns = [&](std::size_t steps) {
            const auto t0 = std::chrono::steady_clock::now();
            for (std::size_t k = 0; k < steps; ++k) {
                tally.begin(ptrs, owner);
                for (auto* s : ptrs) s->generated.insert(s->generated.end(), {7U, 7U, 7U});
                for (std::size_t i = 0; i < ptrs.size(); ++i) {
                    const auto got = tally.after(i, *ptrs[i]);
                    proposed += got.proposed;
                    accepted += got.accepted;
                }
            }
            return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / static_cast<double>(steps);
        };
        const auto median_of_windows = [&] {
            std::vector<double> w;
            for (std::size_t k = 0; k < kWindows; ++k) w.push_back(window_ns(kWindow));
            std::sort(w.begin(), w.end());
            return (w[kWindows / 2 - 1] + w[kWindows / 2]) / 2.0;
        };
        const double first = median_of_windows();
        (void)window_ns(kSteps - 2U * kWindow * kWindows);
        const double last = median_of_windows();
        std::printf("armed per-step draft attribution with the step's token appends (8 owners, windows of 4 rows): median of %zu windows of %zu "
                    "steps: first %.1f ns/step, last %.1f ns/step (after %zu steps)\n", kWindows, kWindow, first, last, kSteps);
        expect(proposed == kSteps * 8U * 3U && accepted == kSteps * 8U * 2U, "every step's drafts attributed");
        expect(last <= 3.0 * first + 500.0, "the per-step attribution cost does not grow with the steps served (loose bound)");
    }
    std::printf("deepseek41 serve observe tests: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
