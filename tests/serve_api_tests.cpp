// HTTP API contract tests with a scripted engine (no GPU): OpenAI chat/completions (plain and
// SSE), tool calls, stop strings, Anthropic messages (plain and events), errors and API keys,
// two concurrent clients through the FIFO queue, and cancellation when a client disconnects.
#include "strix/serve/api.hpp"
#include "strix/serve/constraint_selection.hpp"

#include <limits>
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/http_server.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <iostream>
#include <netinet/in.h>
#include <mutex>
#include <set>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace strix::serve;
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}

// Byte alphabet, a few merges and the chat specials (types: 1 normal, 3 control, 4 user-defined).
std::shared_ptr<const BpeTokenizer> tiny_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    // GPT-2 byte alphabet in byte order.
    for (int b = 0; b < 256; ++b) {
        int cp = b;
        static int extra = 0;
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if (!direct) cp = 256 + extra++;
        std::string s;
        if (cp < 0x80) s.push_back(static_cast<char>(cp));
        else { s.push_back(static_cast<char>(0xC0 | (cp >> 6))); s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        tokens.push_back(s);
        types.push_back(1);
    }
    std::vector<std::string> merges{"H e", "He l", "Hel l", "Hell o"};
    tokens.push_back("He"); tokens.push_back("Hel"); tokens.push_back("Hell"); tokens.push_back("Hello");
    for (int i = 0; i < 4; ++i) types.push_back(1);
    for (const char* special : {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}) { tokens.push_back(special); types.push_back(3); }
    for (const char* special : {"<tool_call>", "</tool_call>", "<think>", "</think>"}) { tokens.push_back(special); types.push_back(4); }
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), merges, "qwen35");
}

class ScriptedEngine final : public GenerationEngine {
public:
    explicit ScriptedEngine(std::shared_ptr<const BpeTokenizer> t) : tokenizer_(std::move(t)), trie_(std::make_shared<TokenByteTrie>(*tokenizer_)) {}
    bool supports_constraints() const override { return true; }
    std::vector<std::string> constraint_execution_routes() const override { return {"serial-host"}; }
    GenerationStats generate(GenerationJob& job) override {
        { const std::lock_guard lock(mutex_); prompts_.push_back({job.prompt, job.stable_prefix}); sampling_.push_back(job.sampling); }
        if (job.constraint) return constrained(job);
        const auto prompt = tokenizer_->decode(job.prompt);
        const bool progress = prompt.find("PROGRESS") != std::string::npos;
        if (progress) {
            const auto total = static_cast<std::uint64_t>(job.prompt.size());
            for (std::uint64_t step = 0; step <= 4; ++step) {
                if (job.on_progress) job.on_progress({GenerationPhase::prefill, (total - 1) * step / 4, 1, total});
                std::this_thread::sleep_for(std::chrono::milliseconds(240));
            }
        }
        if (prompt.find("SLOW") != std::string::npos) std::this_thread::sleep_for(std::chrono::milliseconds(750));
        if (prompt.find("ENGINE_ERROR") != std::string::npos) throw std::runtime_error("scripted failure");
        std::string script = "Let me think.\n</think>\n\nHello there, friend.<|im_end|>";
        if (prompt.find("TOOL") != std::string::npos)
            script = "Need weather.\n</think>\n\nChecking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n"
                     "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>";
        if (prompt.find("LONG") != std::string::npos) { script.clear(); for (int i = 0; i < 400; ++i) script += "word "; }
        if (prompt.find("<think>\n\n</think>") != std::string::npos) script = "Plain answer.<|im_end|>";
        const auto ids = tokenizer_->encode(script, true);
        GenerationStats stats;
        stats.prompt_tokens = job.prompt.size();
        stats.sampling = "greedy";
        std::size_t at = 0;
        while (at < ids.size()) {
            const auto n = std::min<std::size_t>(3U, ids.size() - at);
            std::vector<std::uint32_t> step(ids.begin() + static_cast<std::ptrdiff_t>(at), ids.begin() + static_cast<std::ptrdiff_t>(at + n));
            if (stats.generated_tokens + step.size() >= job.max_tokens) { step.resize(job.max_tokens - stats.generated_tokens); }
            bool stop = false;  // after the budget cut: a stop token beyond the budget is not generated
            for (std::size_t k = 0; k < step.size(); ++k)
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), step[k]) != job.stop_tokens.end()) { step.resize(k + 1U); stop = true; break; }
            stats.generated_tokens += step.size();
            ++stats.cycles;
            calls.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (!job.on_tokens(step)) { stats.finish = "cancelled"; cancelled.fetch_add(1); return stats; }
            if (progress && at == 0) std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (stop) { stats.finish = "stop"; return stats; }
            if (stats.generated_tokens >= job.max_tokens) { stats.finish = "length"; return stats; }
            at += n;
        }
        stats.finish = "length";
        return stats;
    }
    std::uint64_t context_tokens() const override { return 4096; }
    std::atomic<int> calls{0}, cancelled{0};
    using Captured = std::pair<std::vector<std::uint32_t>, std::uint32_t>;
    std::vector<SamplingParameters> sampling() { const std::lock_guard lock(mutex_); return sampling_; }
    std::vector<Captured> prompts() { const std::lock_guard lock(mutex_); return prompts_; }
private:
    GenerationStats constrained(GenerationJob& job) {
        ConstraintSelection selector(job.constraint, trie_, tokenizer_);
        GenerationStats stats; stats.prompt_tokens = job.prompt.size();
        const auto prompt = tokenizer_->decode(job.prompt);
        std::vector<std::uint32_t> desired;
        if (job.constraint->starts_in_reasoning) {
            desired = tokenizer_->encode("why", false);
            desired.push_back(*job.constraint->end_thinking_id);
        }
        const auto answer = tokenizer_->encode(" \n{\"a\":\"<tool_call> </think> caf\xC3\xA9\"}", false);
        desired.insert(desired.end(), answer.begin(), answer.end());
        desired.push_back(job.constraint->eos_ids.front());
        try {
            for (std::size_t i = 0; stats.generated_tokens < job.max_tokens; ++i) {
                if (i == 1 && prompt.find("CONSTRAINT_CONTEXT") != std::string::npos) { stats.finish = "context"; break; }
                if ((prompt.find("EXECUTION_ANSWER") != std::string::npos && selector.verdict("length", "serial-host").accepting) ||
                    (prompt.find("EXECUTION_REASONING") != std::string::npos && i == 1)) {
                    stats.finish = "execution_failure";
                    stats.constraint = selector.verdict(stats.finish, "serial-host");
                    throw GenerationRequestError(std::move(stats), "injected execution failure");
                }
                auto prepared = selector.prepare();
                std::vector<float> logits(tokenizer_->vocab_size(), -100.0F);
                logits[desired.at(i)] = 10.0F;
                if (!prepared.state.reasoning) {
                    logits[tokenizer_->find("<tool_call>")] = 100.0F; // invalid control wins without masking
                    logits[job.constraint->eos_ids.front()] = 90.0F; // blocked until accepting
                    if (desired.at(i) == static_cast<std::uint32_t>('a')) logits['b'] = 10.0F;
                }
                if (prompt.find("CONSTRAINT_FAILURE") != std::string::npos) logits[0] = std::numeric_limits<float>::infinity();
                if (prompt.find("CONSTRAINT_REASON_EOS") != std::string::npos) logits[job.constraint->eos_ids.front()] = 200;
                const auto staged = selector.select(logits, prepared);
                selector.commit(staged); // commit before publishing the pending output token
                ++stats.generated_tokens; ++stats.cycles; calls.fetch_add(1);
                if (!job.on_tokens(std::span<const std::uint32_t>(&staged.token, 1))) { stats.finish = "cancelled"; break; }
                if (selector.state().terminal) { stats.finish = "stop"; break; }
            }
            stats.constraint = selector.verdict(stats.finish, "serial-host");
        } catch (const ConstraintError&) {
            stats.constraint = selector.verdict("constraint_failure", "serial-host");
        }
        return stats;
    }
    std::mutex mutex_;
    std::vector<Captured> prompts_;
    std::vector<SamplingParameters> sampling_;
    std::shared_ptr<const BpeTokenizer> tokenizer_;
    std::shared_ptr<const TokenByteTrie> trie_;
};

// Member access that names the missing key and the document instead of dereferencing null.
const Json& at(const Json& json, std::string_view key) {
    if (const auto* value = json.find(key)) return *value;
    throw std::runtime_error("missing key " + std::string(key) + " in " + json.dump());
}
struct Reply { int status = 0; std::string headers, body; };
// Plain HTTP/1.1 client; decodes chunked bodies. `abort_after` closes the socket after that many body bytes.
Reply http(std::uint16_t port, const std::string& method, const std::string& path, const std::string& body,
    const std::string& extra = {}, std::size_t abort_after = 0) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) { ::close(fd); return {}; }
    const auto request = method + " " + path + " HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n" + extra +
        "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    (void)::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
    std::string raw;
    char buffer[4096];
    for (;;) {
        const auto n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) break;
        raw.append(buffer, static_cast<std::size_t>(n));
        if (abort_after && raw.size() > abort_after) break;
    }
    ::close(fd);
    Reply out;
    const auto end = raw.find("\r\n\r\n");
    if (end == std::string::npos) return out;
    out.headers = raw.substr(0, end);
    out.status = std::atoi(raw.c_str() + 9);
    auto rest = raw.substr(end + 4U);
    if (out.headers.find("Transfer-Encoding: chunked") == std::string::npos) { out.body = rest; return out; }
    for (std::size_t at = 0; at < rest.size();) {
        const auto line = rest.find("\r\n", at);
        if (line == std::string::npos) break;
        const auto size = std::stoul(rest.substr(at, line - at), nullptr, 16);
        if (!size) break;
        out.body += rest.substr(line + 2U, size);
        at = line + 2U + size + 2U;
    }
    return out;
}
std::vector<Json> sse_data(const std::string& body) {
    std::vector<Json> out;
    for (std::size_t at = 0; (at = body.find("data: ", at)) != std::string::npos;) {
        const auto end = body.find("\n\n", at);
        const auto payload = body.substr(at + 6U, end - at - 6U);
        if (payload != "[DONE]") out.push_back(parse_json(payload));
        at = end;
    }
    return out;
}
}  // namespace

// The pull interface of batching engines: FIFO order, jobs cancelled while queued finished and skipped,
// tokens published into the ticket, totals per finished job, nothing after shutdown.
void test_batched_queue() {
    GenerationQueue queue(8);
    const auto job = [](std::uint32_t token) { GenerationJob j; j.prompt = {token, token + 1U}; return j; };
    const auto t1 = queue.submit(job(1)), t2 = queue.submit(job(3)), t3 = queue.submit(job(5));
    check(t1 && t2 && t3, "batched queue submit");
    t2->cancel();
    auto a = queue.take(false);
    check(a && a->ticket == t1 && a->job.prompt.front() == 1U, "batched queue takes the oldest job");
    auto b = queue.take(false);
    check(b && b->ticket == t3, "batched queue skips a job cancelled while queued");
    bool done = false;
    static_cast<void>(t2->wait(std::chrono::milliseconds(0), done));
    check(done && t2->stats().finish == "cancelled", "batched queue finishes the cancelled job");
    check(!queue.take(false), "batched queue is empty");
    const std::uint32_t tokens[] = {7, 8};
    check(a->job.on_tokens(tokens), "batched job publishes");
    done = false;
    const auto got = t1->wait(std::chrono::milliseconds(0), done);
    check(got.size() == 2U && got[0] == 7U && !done, "batched job tokens reach the ticket");
    GenerationStats stats;
    stats.generated_tokens = 2;
    stats.finish = "stop";
    queue.finish(*a, stats);
    queue.finish(*b, {}, "engine failure");
    const auto snap = queue.snapshot();
    check(snap.completed == 1U && snap.failed == 1U && snap.cancelled == 1U && snap.running == 0U && snap.queued == 0U &&
          snap.generated_tokens == 2U, "batched queue totals");
    static_cast<void>(t1->wait(std::chrono::milliseconds(0), done));
    check(done && t1->stats().finish == "stop" && t3->error() == "engine failure", "batched tickets finished");
    queue.shutdown();
    check(!queue.take(true) && queue.stopping(), "batched queue after shutdown");
}

// A fake target owns the predicted frontier, just like the HIP ticket. The
// production serial adapter must stage, replace that frontier, then commit before
// publication; aborting a row must not change the committed cursor.
void test_constraint_row(const std::shared_ptr<const BpeTokenizer>& tokenizer) {
    const auto trie=std::make_shared<TokenByteTrie>(*tokenizer);
    auto spec=std::make_shared<ConstraintSpec>();
    spec->program=std::make_shared<ConstraintProgram>(compile_json_object());
    spec->eos_ids={tokenizer->find("<|im_end|>")};
    const auto selection=std::make_shared<ConstraintSelection>(spec,trie,tokenizer);
    auto adapter=std::make_shared<ConstraintRowSelection>(selection);
    std::vector<float> row(tokenizer->vocab_size(),-100.0F);
    row['x']=100;row['{']=10;
    std::uint32_t pending='x', consumed=0;
    const auto execute=[&](std::uint32_t input) {
        consumed=input;
        // Target finite first-max predicts x; the host callback replaces it.
        std::uint32_t predicted='x';
        predicted=adapter->select(row);
        return predicted;
    };
    auto first=execute(7);
    check(first=='{' && !selection->prepare().mask.contains('}'), "terminal row stages constrained ticket without commit");
    adapter->commit();pending=first;
    check(adapter->verdict("length").reason=="max_output_tokens", "budget one reports incomplete first token");
    row['{']=-100;row['}']=10;
    adapter->prepare();auto next=execute(pending);
    check(consumed=='{' && next=='}' && !selection->state().cursor.complete(), "serial target consumes constrained pending token before staging successor");
    adapter->commit();pending=next;
    check(selection->state().cursor.complete() && adapter->verdict("context").reason=="context", "serial commit advances exact output frontier");
    check(adapter->verdict("cancelled").reason=="cancelled", "row cancellation receipt");
    const auto execution = adapter->verdict("execution_failure");
    check(execution.status=="failed" && execution.reason=="execution_failure" && execution.accepting && execution.phase=="answer",
        "execution failure snapshots accepting committed row before EOS");
    row[spec->eos_ids.front()]=200;
    adapter->prepare();next=execute(pending);adapter->commit();
    check(next==spec->eos_ids.front() && adapter->verdict("stop").status=="complete", "serial EOS completes only after commit");
    // A dropped callback/state never leaks into a reused slot sharing this trie.
    std::weak_ptr<ConstraintRowSelection> retired=adapter;adapter.reset();
    check(retired.expired(), "request row adapter released at finish");
    auto fresh=std::make_shared<ConstraintSelection>(spec,trie,tokenizer);
    {
        ConstraintRowSelection aborted(fresh);
        row.assign(row.size(),-100);row['{']=10;
        check(aborted.select(row)=='{', "aborted row stages token");
    }
    check(!fresh->prepare().mask.contains('}'), "aborted row leaves next request frontier unchanged");
    ConstraintRowSelection failed(fresh);
    row['x']=std::numeric_limits<float>::quiet_NaN();
    bool refused=false;
    try {(void)failed.select(row);} catch(const ConstraintError&) {refused=true;}
    check(refused && failed.verdict("constraint_failure").status=="failed" && !fresh->prepare().mask.contains('}'),
        "row refusal preserves committed state and failed receipt");
}

void test_constraint_request_failure() {
    class FailingRequest final : public GenerationEngine {
    public:
        std::uint64_t context_tokens() const override {return 32;}
        GenerationStats generate(GenerationJob& job) override {
            GenerationStats stats;
            if(job.prompt.front()==1U) {
                stats.finish="constraint_failure";
                stats.constraint=ConstraintVerdict{"failed",false,"answer","constraint_failure","serial-host"};
                throw GenerationRequestError(std::move(stats),"empty constraint mask");
            }
            stats.finish="stop";
            return stats;
        }
    } engine;
    GenerationQueue queue;
    GenerationJob failed;failed.prompt={1};
    GenerationJob healthy;healthy.prompt={2};
    const auto first=queue.submit(std::move(failed)),second=queue.submit(std::move(healthy));
    std::thread worker([&]{queue.run(engine);});
    for(const auto& ticket:{first,second}) {
        bool done=false;
        while(!done) (void)ticket->wait(std::chrono::milliseconds(50),done);
    }
    queue.shutdown();worker.join();
    const auto stats=first->stats();
    check(first->error()=="empty constraint mask" && stats.finish=="constraint_failure" && stats.constraint &&
        stats.constraint->status=="failed" && stats.constraint->execution=="serial-host", "single queue retains failure receipt and error");
    check(second->error().empty() && second->stats().finish=="stop", "constraint refusal does not poison next queued request");
}

void test_constraint_late_cancellation() {
    for (const bool batched : {false, true}) for (const bool reasoning : {false, true}) {
        GenerationQueue queue;
        GenerationJob job; job.prompt = {1};
        auto ticket = queue.submit(std::move(job));
        GenerationStats stats; stats.finish = "stop";
        stats.constraint = ConstraintVerdict{"complete", !reasoning, reasoning ? "reasoning" : "answer", "", "serial-host"};
        if (batched) {
            auto taken = queue.take(false);
            ticket->cancel();
            queue.finish(*taken, stats);
        } else {
            class LateCancel final : public GenerationEngine {
            public:
                std::uint64_t context_tokens() const override { return 32; }
                GenerationStats generate(GenerationJob&) override {
                    ticket->cancel(); queue->shutdown(); return stats;
                }
                GenerationQueue* queue;
                std::shared_ptr<GenerationTicket> ticket;
                GenerationStats stats;
            } engine;
            engine.queue = &queue; engine.ticket = ticket; engine.stats = stats;
            queue.run(engine);
        }
        const auto result = ticket->stats();
        check(result.finish == "cancelled" && result.constraint && result.constraint->status == "incomplete" &&
            result.constraint->reason == "cancelled" && result.constraint->phase == stats.constraint->phase &&
            result.constraint->accepting == stats.constraint->accepting && result.constraint->execution == "serial-host",
            "late cancellation reconciles single/batched stats while preserving frontier");
    }
}

void test_constraint_selection(const std::shared_ptr<const BpeTokenizer>& tokenizer) {
    const auto trie = std::make_shared<TokenByteTrie>(*tokenizer);
    auto spec = std::make_shared<ConstraintSpec>();
    spec->program = std::make_shared<ConstraintProgram>(compile_json_object());
    spec->eos_ids = {tokenizer->find("<|im_end|>")}; spec->end_thinking_id = tokenizer->find("</think>");
    spec->identity = "json-object";
    const auto throws = [](const auto& operation) { try { operation(); } catch (const ConstraintError&) { return true; } return false; };
    ConstraintSelection selector(spec, trie, tokenizer);
    std::vector<float> row(tokenizer->vocab_size(), -100.0F);
    row['{'] = 1; row[spec->eos_ids.front()] = 100;
    auto first = selector.select(row, selector.prepare());
    check(first.token == '{' && !selector.state().cursor.complete(), "first position masks EOS without committing");
    check(!selector.prepare().mask.contains('}'), "staged brace has not advanced live cursor");
    auto branch = selector.prepare(first.state);
    check(branch.mask.contains('}'), "branch has staged cursor");
    row['{'] = -100; row['}'] = 1;
    auto second = selector.select(row, branch);
    check(second.state.cursor.complete() && !selector.state().cursor.complete(), "branch accepts without advancing live cursor");
    selector.commit(second);
    check(selector.state().cursor.complete(), "accepted prefix commits correction token");
    check(throws([&] { selector.commit(second); }) && throws([&] { selector.commit(first); }), "duplicate and stale commits refused");
    auto eos = selector.select(row, selector.prepare());
    check(eos.token == spec->eos_ids.front(), "accepting boundary admits EOS"); selector.commit(eos);
    check(selector.verdict("stop", "serial-host").status == "complete", "committed EOS completes constraint");
    check(throws([&] { (void)selector.prepare(); }), "no selection after EOS");
    ConstraintSelection tie(spec, trie, tokenizer);
    std::fill(row.begin(), row.end(), -100); row[' '] = -0.0F; row['{'] = 0.0F;
    check(tie.select(row, tie.prepare()).token == ' ', "signed-zero tie chooses lowest ID");
    row[tokenizer->find("<tool_call>")] = std::numeric_limits<float>::quiet_NaN();
    check(throws([&] { (void)tie.select(row, tie.prepare()); }), "nonfinite even in masked entry fails");
    auto empty = tie.prepare(); std::fill(empty.mask.words.begin(), empty.mask.words.end(), 0);
    std::fill(row.begin(), row.end(), 0);
    check(throws([&] { (void)tie.select(row, empty); }), "empty mask fails explicitly");
    auto thinking_spec = std::make_shared<ConstraintSpec>(*spec); thinking_spec->starts_in_reasoning = true;
    ConstraintSelection thinking(thinking_spec, trie, tokenizer);
    const auto execution = thinking.verdict("execution_failure", "serial-host");
    check(execution.status == "failed" && execution.reason == "execution_failure" && execution.phase == "reasoning" && !execution.accepting,
        "execution failure preserves reasoning frontier");
    row[*spec->end_thinking_id] = 10;
    auto closing = thinking.select(row, thinking.prepare());
    check(thinking.state().reasoning && !closing.state.reasoning, "end-thinking transition is staged");
    thinking.commit(closing);
    check(!thinking.state().reasoning && !thinking.prepare().mask.contains('x'), "next position uses answer mask");
    ConstraintSelection early(thinking_spec, trie, tokenizer);
    row[spec->eos_ids.front()] = 20; auto stop = early.select(row, early.prepare()); early.commit(stop);
    const auto early_verdict = early.verdict("stop", "serial-host");
    check(early_verdict.status == "incomplete" && early_verdict.reason == "reasoning_eos", "reasoning EOS stays unmasked and incomplete");
    check(thinking.verdict("context", "serial-host").reason == "context", "context verdict");
    check(thinking.verdict("cancelled", "serial-host").reason == "cancelled", "cancellation verdict");
    check(thinking.verdict("constraint_failure", "serial-host").status == "failed", "constraint failure verdict");
    ConstraintSelection accepting(spec, trie, tokenizer);
    row.assign(row.size(), -100); row['{'] = 1; accepting.commit(accepting.select(row, accepting.prepare()));
    row['{'] = -100; row['}'] = 1; accepting.commit(accepting.select(row, accepting.prepare()));
    const auto budget = accepting.verdict("length", "serial-host");
    check(budget.status == "incomplete" && budget.accepting && budget.reason == "max_output_tokens", "accepting budget boundary is still incomplete");
}

int main() try {
    {
        GenerationTicket ticket;
        check(ticket.progress().phase == GenerationPhase::queued, "ticket starts queued");
        for (std::uint64_t n = 0; n <= 100; ++n) ticket.report_progress({GenerationPhase::prefill, n, 10, 110});
        const auto progress = ticket.progress();
        check(progress.prefilled_tokens == 100 && progress.cached_tokens == 10 && progress.total_tokens == 110, "ticket coalesces completed units");
        const std::array<std::uint32_t, 1> token{42};
        ticket.publish(token);
        check(ticket.progress().phase == GenerationPhase::decode && ticket.progress().prefilled_tokens == 100, "first token changes phase only");
        bool done = false;
        check(ticket.wait(std::chrono::milliseconds(0), done) == std::vector<std::uint32_t>{42}, "progress preserves token publication");
        GenerationTicket silent; silent.publish(token);
        check(silent.progress().phase == GenerationPhase::queued, "no fabricated engine report");
    }
    test_batched_queue();
    const auto tokenizer = tiny_tokenizer();
    test_constraint_selection(tokenizer);
    test_constraint_row(tokenizer);
    test_constraint_request_failure();
    test_constraint_late_cancellation();
    ScriptedEngine engine(tokenizer);
    GenerationQueue queue;
    ApiConfig config;
    config.model_id = "test-model";
    config.stop_tokens = {tokenizer->find("<|im_end|>"), tokenizer->find("<|endoftext|>")};
    config.api_key = "secret";
    config.sse_keepalive_seconds = 0.3;
    config.engine_progress = true;
    config.sse_progress_seconds = 0.15;
    ApiService api(config, tokenizer, queue, engine.context_tokens(), false);
    auto constrained_config = config;
    constrained_config.engine_constraints = engine.supports_constraints();
    constrained_config.constraint_execution_routes = engine.constraint_execution_routes();
    ApiService constrained_api(constrained_config, tokenizer, queue, engine.context_tokens(), true);
    ApiService constrained_greedy_api(constrained_config, tokenizer, queue, engine.context_tokens(), false);
    constrained_config.stream_receipts = false;
    ApiService constrained_quiet_api(constrained_config, tokenizer, queue, engine.context_tokens(), true);
    auto off_config = config;
    off_config.capabilities = false; off_config.responses = false;
    ApiService off_api(off_config, tokenizer, queue, engine.context_tokens(), false);
    auto legacy_config = config;
    legacy_config.stream_receipts = false; legacy_config.sse_keepalive_seconds = 0;
    ApiService legacy_api(legacy_config, tokenizer, queue, engine.context_tokens(), false);
    auto quiet_config = config; quiet_config.sse_keepalive_seconds = 0;
    ApiService quiet_api(quiet_config, tokenizer, queue, engine.context_tokens(), false);
    auto progress_off_config = config; progress_off_config.sse_progress_seconds = 0;
    ApiService progress_off_api(progress_off_config, tokenizer, queue, engine.context_tokens(), false);
    auto sample_config = config; sample_config.sampling_policy = "sample";
    sample_config.capabilities_extra = [](Json& doc) { doc.set("engine_feature", Json(true)); };
    ApiService sample_api(sample_config, tokenizer, queue, engine.context_tokens(), true);
    auto defaults_config = sample_config;
    defaults_config.default_temperature = 0.7; defaults_config.default_top_k = 20;
    defaults_config.default_top_p = 0.9; defaults_config.default_min_p = 0.1;
    ApiService defaults_api(defaults_config, tokenizer, queue, engine.context_tokens(), true);
    auto reject_config = config; reject_config.sampling_policy = "reject"; reject_config.responses = false;
    ApiService reject_api(reject_config, tokenizer, queue, engine.context_tokens(), false);
    HttpServerConfig http_config;
    http_config.port = 0;
    HttpServer server(http_config, [&](const HttpRequest& r, HttpResponseWriter& w) {
        const auto mode = r.header("x-test-config");
        if (mode == "constraints") constrained_api.handle(r, w);
        else if (mode == "constraints-greedy") constrained_greedy_api.handle(r, w);
        else if (mode == "constraints-quiet") constrained_quiet_api.handle(r, w);
        else if (mode == "off") off_api.handle(r, w);
        else if (mode == "legacy") legacy_api.handle(r, w);
        else if (mode == "progress-off") progress_off_api.handle(r, w);
        else if (mode == "quiet") quiet_api.handle(r, w);
        else if (mode == "defaults") defaults_api.handle(r, w);
        else if (mode == "sample") sample_api.handle(r, w);
        else if (mode == "reject") reject_api.handle(r, w);
        else api.handle(r, w);
    });
    server.listen();
    const auto port = server.port();
    std::thread http_thread([&] { server.serve(); });
    std::thread engine_thread([&] { queue.run(engine); });
    const std::string auth = "Authorization: Bearer secret\r\n";
    const auto first_choice = [](const Json& j) -> const Json& { return at(j, "choices").as_array().at(0); };
    const auto stream_text = [](const std::string& body, std::string* reasoning, std::string* finish, bool* usage, bool* tools) {
        std::string content;
        for (const auto& event : sse_data(body)) {
            if (const auto* u = event.find("usage"); usage && u && !u->is_null()) *usage = true;
            const auto& choices = at(event, "choices").as_array();
            if (choices.empty()) continue;
            const auto& delta = at(choices[0], "delta");
            if (const auto* c = delta.find("content"); c && c->is_string()) content += c->as_string();
            if (const auto* c = delta.find("reasoning_content"); reasoning && c && c->is_string()) *reasoning += c->as_string();
            if (const auto* f = choices[0].find("finish_reason"); finish && f && f->is_string()) *finish = f->as_string();
            if (tools && delta.find("tool_calls")) *tools = true;
        }
        return content;
    };

    try {
    {
        const std::string mode = auth + "X-Test-Config: constraints\r\n";
        const std::string quiet = auth + "X-Test-Config: constraints-quiet\r\n";
        const std::string expected = " \n{\"a\":\"<tool_call> </think> caf\xC3\xA9\"}";
        const auto schema = parse_json(R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false})");
        const auto validate = [&](const std::string& answer) {
            const auto parsed = parse_json(answer);
            check(at(parsed, "a").as_string() == "<tool_call> </think> caf\xC3\xA9", "marker bytes and lowest-id tie retained");
            auto program = compile_json_schema(schema); ConstraintCursor cursor(program);
            cursor = advance_token(cursor, answer); check(cursor.complete(), "answer independently validates against schema");
        };
        const auto verdict = [&](const Json& out, const std::string& status) {
            const auto& v = at(at(out, "strix"), "constraint");
            check(at(v, "status").as_string() == status && at(v, "execution").as_string() == "serial-host", "engine constraint verdict");
            check(at(v, "selection").as_string() == "greedy-lowest-id", "selection identity");
        };
        for (bool thinking : {false, true}) for (bool schema_mode : {false, true}) {
            Json format = parse_json(R"({"type":"json_object"})");
            Json chat_format = format;
            if (schema_mode) {
                format.set("type", Json("json_schema")); format.set("name", Json("answer")); format.set("schema", schema); format.set("strict", Json(false));
                chat_format = parse_json(R"({"type":"json_schema"})");
                Json envelope = format; chat_format.set("json_schema", envelope);
            }
            auto chat = parse_json(R"({"messages":[{"role":"user","content":"JSON"}],"max_tokens":256})");
            chat.set("enable_thinking", Json(thinking)); chat.set("response_format", chat_format);
            auto responses = parse_json(R"({"input":"JSON","max_output_tokens":256})");
            responses.set("enable_thinking", Json(thinking)); Json text = Json::object(); text.set("format", format); responses.set("text", text);
            const auto chat_reply = http(port, "POST", "/v1/chat/completions", chat.dump(), mode);
            check(chat_reply.status == 200, "constrained chat admitted");
            const auto c = parse_json(chat_reply.body); verdict(c, "complete");
            const auto answer = at(at(first_choice(c), "message"), "content").as_string();
            check(answer == expected, "exact constrained chat bytes incl leading whitespace"); validate(answer);
            const auto response_reply = http(port, "POST", "/v1/responses", responses.dump(), mode);
            check(response_reply.status == 200, "constrained Responses admitted");
            const auto r = parse_json(response_reply.body); verdict(r, "complete");
            check(at(r, "status").as_string() == "completed", "completed constrained Responses");
            const auto& output = at(r, "output").as_array();
            check(at(at(output.back(), "content").as_array().front(), "text").as_string() == answer, "chat and Responses bytes match");
            const auto prompts = engine.prompts(); check(prompts[prompts.size()-1].first == prompts[prompts.size()-2].first, "constrained endpoint prompt IDs equal");
            chat.set("stream", Json(true)); responses.set("stream", Json(true));
            for (const auto& headers : {mode, quiet}) {
                const auto cs = http(port, "POST", "/v1/chat/completions", chat.dump(), headers);
                check(stream_text(cs.body, nullptr, nullptr, nullptr, nullptr) == answer, "chat SSE bytes equal nonstream bytes");
                const auto events = sse_data(cs.body); verdict(events.front(), "in_progress"); verdict(events.back(), "complete");
                if (headers == quiet) check(at(events.front(), "strix").size() == 1 && at(events.back(), "strix").size() == 1, "receipts off retains only mandatory constraint metadata");
                const auto rs = sse_data(http(port, "POST", "/v1/responses", responses.dump(), headers).body);
                std::string joined;
                for (const auto& event : rs) if (at(event, "type").as_string() == "response.output_text.delta") joined += at(event, "delta").as_string();
                check(joined == answer, "Responses SSE bytes equal nonstream bytes");
                verdict(at(rs.front(), "response"), "in_progress"); verdict(at(rs.back(), "response"), "complete");
            }
            chat.set("stream", Json(false)); chat.set("max_tokens", Json(1));
            const auto cut = parse_json(http(port, "POST", "/v1/chat/completions", chat.dump(), quiet).body); verdict(cut, "incomplete");
            check(at(first_choice(cut), "finish_reason").as_string() == "length", "chat budget keeps finish vocabulary");
            responses.set("stream", Json(false)); responses.set("max_output_tokens", Json(1));
            const auto cut_r = parse_json(http(port, "POST", "/v1/responses", responses.dump(), quiet).body); verdict(cut_r, "incomplete");
            check(at(cut_r, "status").as_string() == "incomplete" && at(at(cut_r, "incomplete_details"), "reason").as_string() == "max_output_tokens", "Responses budget verdict");
        }
        for (const bool response_endpoint : {false, true}) {
            const std::string path = response_endpoint ? "/v1/responses" : "/v1/chat/completions";
            auto body = response_endpoint ? parse_json(R"({"input":"JSON"})") : parse_json(R"({"messages":[{"role":"user","content":"JSON"}]})");
            const auto set_format = [&](const Json& f) {
                if (response_endpoint) { Json text = Json::object(); text.set("format", f); body.set("text", text); }
                else body.set("response_format", f);
            };
            const auto before = engine.prompts().size();
            auto bad_schema = parse_json(R"({"type":"json_schema","name":"answer","schema":{"type":"object","properties":{"age":{"type":"integer","minimum":0}}},"strict":false})");
            Json envelope = bad_schema;
            if (!response_endpoint) { envelope = parse_json(R"({"type":"json_schema"})"); envelope.set("json_schema", bad_schema); }
            set_format(envelope);
            auto error = parse_json(http(port, "POST", path, body.dump(), mode).body);
            const auto& e = at(error, "error");
            check(at(e, "param").as_string() == (response_endpoint ? "text.format.schema.properties.age.minimum" : "response_format.json_schema.schema.properties.age.minimum"), "full schema error path");
            check(at(e, "keyword").as_string() == "minimum" && at(e, "code").as_string() == "unsupported_schema_keyword", "schema error code and keyword");
            bad_schema.set("strict", Json("yes"));
            if (!response_endpoint) envelope.set("json_schema", bad_schema); else envelope = bad_schema;
            set_format(envelope); check(http(port, "POST", path, body.dump(), mode).status == 400, "strict requires boolean");
            set_format(parse_json(R"({"type":"json_object"})"));
            check(http(port, "POST", path, body.dump(), auth).status == 400, "unsupported engine refuses constraints");
            for (const auto& extra : {parse_json(R"({"stop":["x"]})"), parse_json(R"({"temperature":0.5})"), parse_json(R"({"tools":[{"type":"function","name":"f","function":{"name":"f","parameters":{}}}]})")}) {
                auto request = parse_json(body.dump()); for (const auto& [key, value] : extra.as_object()) request.set(key, value);
                check(http(port, "POST", path, request.dump(), mode).status == 400, "unsupported constraint combination refused");
            }
            check(engine.prompts().size() == before, "constraint refusals never submitted");
            for (const auto& tools : {Json(), Json::array()}) {
                auto request = parse_json(body.dump()); request.set("tools", tools);
                check(http(port, "POST", path, request.dump(), mode).status == 200, "null and empty tools admitted with constraints");
            }
            for (const auto& headers : {mode, quiet}) for (const auto& scenario : {"CONSTRAINT_FAILURE", "EXECUTION_ANSWER", "EXECUTION_REASONING"}) {
                auto request = parse_json(body.dump());
                const bool execution = std::string(scenario) != "CONSTRAINT_FAILURE";
                const bool reasoning = std::string(scenario) == "EXECUTION_REASONING";
                request.set("enable_thinking", Json(reasoning));
                if (response_endpoint) request.set("input", Json(scenario));
                else request.set("messages", parse_json(std::string("[{\"role\":\"user\",\"content\":\"") + scenario + "\"}]"));
                const auto reply = http(port, "POST", path, request.dump(), headers);
                check(reply.status == 500, "terminal constrained failure retains HTTP 500");
                const auto failure_body = parse_json(reply.body); verdict(failure_body, "failed");
                const auto& v = at(at(failure_body, "strix"), "constraint");
                check(at(v, "reason").as_string() == (execution ? "execution_failure" : "constraint_failure"), "HTTP preserves engine failure reason");
                check(at(v, "phase").as_string() == (reasoning ? "reasoning" : "answer") &&
                    at(v, "accepting").as_bool() == (execution && !reasoning), "HTTP preserves engine failure frontier");
                check(at(failure_body, "error").is_object(), "failure envelope retains error detail");
                if (headers == quiet) check(at(failure_body, "strix").size() == 1, "mandatory failure verdict with receipts off");
            }
        }
        {
            auto body = parse_json(R"({"messages":[{"role":"user","content":"JSON"}],"response_format":{"type":"json_object"},"enable_thinking":false,"stop":"","temperature":0.7,"tools":[],"tool_choice":"none"})");
            const auto reply = http(port, "POST", "/v1/chat/completions", body.dump(), auth + "X-Test-Config: constraints-greedy\r\n");
            check(reply.status == 200, "effective greedy downgrade, empty stop and disabled tools admitted"); verdict(parse_json(reply.body), "complete");
        }
        for (const auto& scenario : {std::string("CONSTRAINT_FAILURE"), std::string("CONSTRAINT_CONTEXT"), std::string("CONSTRAINT_REASON_EOS")}) {
            auto body = parse_json(R"({"input":"JSON","text":{"format":{"type":"json_object"}},"max_output_tokens":128})");
            body.set("input", Json(scenario));
            const bool failed = scenario == "CONSTRAINT_FAILURE";
            const std::string status = failed ? "failed" : "incomplete";
            const std::string reason = failed ? "constraint_failure" : scenario == "CONSTRAINT_CONTEXT" ? "context" : "reasoning_eos";
            for (const bool stream : {false, true}) {
                body.set("stream", Json(stream));
                const auto reply = http(port, "POST", "/v1/responses", body.dump(), quiet);
                if (failed && !stream) {  // a failure is not a reply: 500, as for unconstrained requests
                    check(reply.status == 500, "non-streamed constraint failure answers 500");
                    continue;
                }
                const auto events = stream ? sse_data(reply.body) : std::vector<Json>{};
                const auto terminal = stream ? at(events.back(), "response") : parse_json(reply.body);
                verdict(terminal, status);
                check(at(terminal, "status").as_string() == status, "Responses terminal failure/context/reasoning status");
                check(at(at(at(terminal, "strix"), "constraint"), "reason").as_string() == reason, "terminal constraint reason retained");
                if (!failed) check(at(at(terminal, "incomplete_details"), "reason").as_string() == reason, "Responses incomplete reason retained");
            }
            auto chat = parse_json(R"({"messages":[{"role":"user","content":"CONSTRAINT_FAILURE"}],"response_format":{"type":"json_object"},"stream":true})");
            if (failed) {
                const auto events = sse_data(http(port, "POST", "/v1/chat/completions", chat.dump(), quiet).body);
                verdict(events.back(), "failed");
            }
        }
        for (const auto& headers : {auth, mode}) {
            const auto caps = parse_json(http(port, "GET", "/v1/capabilities", "", headers).body);
            const bool enabled = headers == mode;
            check(at(caps, "json_object").as_bool() == enabled && at(caps, "json_schema").as_bool() == enabled, "capabilities follow engine execution");
            check(at(at(caps, "constraints"), "identities").size() == (enabled ? 2U : 0U), "constraint identity inventory");
        }
    }

    {
        const auto request = [&](const std::string& fields, const std::string& mode = "sample") {
            return http(port, "POST", "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1" + fields + "}",
                auth + "X-Test-Config: " + mode + "\r\n");
        };
        const auto filters_equal = [&](const Json& receipt, const std::string& source, double temperature, int top_k, double top_p, double min_p) {
            const auto& filters = at(receipt, "filters");
            for (const auto& [key, value] : std::vector<std::pair<std::string, double>>{{"temperature", temperature}, {"top_k", top_k}, {"top_p", top_p}, {"min_p", min_p}}) {
                check(at(at(filters, key), "value").as_double() == value, key + " resolved value");
                check(at(at(filters, key), "source").as_string() == source, key + " resolved source");
            }
        };
        const auto builtin = parse_json(request("").body);
        filters_equal(at(builtin, "strix"), "builtin", 0, 0, 1, 0);
        check(!at(builtin, "strix").find("seed"), "greedy receipt has no seed");
        const auto defaults = parse_json(request("", "defaults").body);
        const auto& receipt = at(defaults, "strix");
        filters_equal(receipt, "server", 0.7, 20, 0.9, 0.1);
        const auto captured = engine.sampling().back();
        check(captured.temperature == 0.7 && captured.top_k == 20 && captured.top_p == 0.9 && captured.min_p == 0.1, "engine receives server filters");
        check(captured.seeded && captured.seed == static_cast<std::uint64_t>(at(receipt, "seed").as_int()), "API draws and echoes effective seed");
        const auto explicit_values = parse_json(request(",\"temperature\":0.5,\"top_k\":5,\"top_p\":0.8,\"min_p\":0.2,\"seed\":42", "defaults").body);
        filters_equal(at(explicit_values, "strix"), "request", 0.5, 5, 0.8, 0.2);
        check(at(at(explicit_values, "strix"), "seed").as_int() == 42 && engine.sampling().back().seed == 42, "explicit seed echoed and submitted");
        for (const auto* key : {"temperature", "top_k", "top_p", "min_p"}) {
            const auto mixed = parse_json(request(std::string(",\"") + key + "\":null", "defaults").body);
            filters_equal(at(mixed, "strix"), "server", 0.7, 20, 0.9, 0.1);
        }
        for (const auto& [key, value] : std::vector<std::pair<std::string, std::string>>{{"temperature", "0"}, {"top_k", "0"}, {"top_p", "1"}, {"min_p", "0"}}) {
            for (const auto* mode : {"sample", "defaults"}) {
                const auto mixed = parse_json(request(",\"" + key + "\":" + value, mode).body);
                const auto& filters = at(at(mixed, "strix"), "filters");
                for (const auto* field : {"temperature", "top_k", "top_p", "min_p"})
                    check(at(at(filters, field), "source").as_string() == (key == field ? "request" : std::string(mode) == "defaults" ? "server" : "builtin"), "independent filter precedence");
            }
        }
        const auto ignored = parse_json(request(",\"temperature\":0,\"top_k\":10").body);
        check(at(at(ignored, "strix"), "note").as_string() == "filters ignored at temperature 0 (greedy)", "greedy filter note");
        check(!at(ignored, "strix").find("seed"), "greedy filters do not create seed");
        const auto downgraded = parse_json(request(",\"temperature\":1", "quiet").body);
        check(at(at(downgraded, "strix"), "note").as_string() == "temperature ignored: this engine decodes greedy (exact)", "downgrade note unchanged");
        for (const auto& [key, values] : std::vector<std::pair<std::string, std::vector<std::string>>>{
                {"temperature", {"-1", "2.1", "1e999", "\"1\"", "true", "[]", "{}"}},
                {"top_p", {"0", "-0.1", "1.1", "1e999", "\"1\"", "false", "[]", "{}"}},
                {"min_p", {"-0.1", "1", "1e999", "\"0\"", "false", "[]", "{}"}},
                {"top_k", {"1.5", "-1", "4294967296", "1e999", "\"1\"", "true", "[]", "{}"}},
                {"seed", {"-1", "42.5", "9223372036854775808", "18446744073709551616", "\"1\"", "true", "[]", "{}"}}}) {
            for (const auto& value : values) {
                const auto before = engine.prompts().size();
                const auto bad = request(",\"" + key + "\":" + value);
                check(bad.status == 400 && at(at(parse_json(bad.body), "error"), "param").as_string() == key, key + " rejects " + value);
                check(engine.prompts().size() == before, "invalid " + key + " never submitted");
            }
        }
        // integral values in any spelling are integers
        for (const auto* spelling : {"10.0", "1e1", "10"}) {
            const auto ok = request(std::string(",\"temperature\":1,\"top_k\":") + spelling + ",\"seed\":" + spelling);
            check(ok.status == 200 && engine.sampling().back().top_k == 10U && engine.sampling().back().seed == 10U, std::string("integral spelling ") + spelling);
        }
        check(request(",\"temperature\":1,\"top_k\":-0").status == 200 && engine.sampling().back().top_k == 0U, "top_k -0 is 0");
        // integer spellings are read exactly (never through a double)
        for (const auto& [spelling, expected] : std::vector<std::pair<std::string, std::uint64_t>>{
                 {"9007199254740993.0", 9007199254740993ULL}, {"9007199254740994.0", 9007199254740994ULL},
                 {"9.223372036854775807e18", 9223372036854775807ULL}, {"100e-2", 1ULL}, {"0.0e5", 0ULL}}) {
            const auto ok = request(",\"temperature\":1,\"seed\":" + spelling);
            check(ok.status == 200 && engine.sampling().back().seed == expected, "exact integer spelling " + spelling);
        }
        for (const char* spelling : {"9007199254740992.1", "9.223372036854775808e18", "1e-1", "100e-3", "-1e0"}) {
            const auto bad = request(std::string(",\"temperature\":1,\"seed\":") + spelling);
            check(bad.status == 400, std::string("non-integer or out-of-range seed refused ") + spelling);
        }
        check(request(",\"temperature\":1,\"top_k\":1.0000000000000001").status == 400, "top_k fraction below double precision refused");
        // the engine samples with float parameters: values that narrow to an invalid float are refused up front
        for (const auto& [extra, param] : std::vector<std::pair<std::string, std::string>>{
                 {",\"temperature\":1e-50", "temperature"}, {",\"temperature\":1,\"top_p\":1e-50", "top_p"},
                 {",\"temperature\":1,\"min_p\":0.99999999", "min_p"}}) {
            const auto bad = request(extra);
            check(bad.status == 400 && at(at(parse_json(bad.body), "error"), "param").as_string() == param, "float narrowing refused " + extra);
        }
        for (const auto* seed : {"0", "9223372036854775807"}) {
            const auto accepted = parse_json(request(std::string(",\"temperature\":1,\"seed\":") + seed).body);
            check(at(at(accepted, "strix"), "seed").number_source() == seed, "seed boundary accepted");
        }
        const auto caps = parse_json(http(port, "GET", "/v1/capabilities", "", auth + "X-Test-Config: defaults\r\n").body);
        Json cap_receipt = Json::object(); cap_receipt.set("filters", at(at(caps, "sampling"), "defaults"));
        filters_equal(cap_receipt, "server", 0.7, 20, 0.9, 0.1);
        check(at(at(caps, "sampling"), "seed_echo").as_bool(), "capabilities seed echo");
        for (const auto* route : {"/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses"}) {
            std::string body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"prompt\":\"Hi\",\"input\":\"Hi\",\"max_tokens\":1,\"max_output_tokens\":1,\"stream\":true}";
            const auto events = sse_data(http(port, "POST", route, body, auth + "X-Test-Config: defaults\r\n").body);
            std::vector<Json> receipts;
            for (const auto& event : events) {
                const Json* object = &event;
                if (const auto* r = event.find("response")) object = r;
                if (const auto* m = event.find("message")) object = m;
                if (const auto* r = object->find("strix")) receipts.push_back(*r);
            }
            check(receipts.size() >= 2, std::string(route) + " stream end receipts");
            if (receipts.size() >= 2) {
                check(at(receipts.front(), "seed").as_int() == at(receipts.back(), "seed").as_int(), "stream seed is stable");
                filters_equal(receipts.front(), "server", 0.7, 20, 0.9, 0.1);
                filters_equal(receipts.back(), "server", 0.7, 20, 0.9, 0.1);
            }
            const auto plain = parse_json(body);
            Json nonstream = Json::object();
            for (const auto& [key, value] : plain.as_object()) if (key != "stream") nonstream.set(key, value);
            const auto legacy = parse_json(http(port, "POST", route, nonstream.dump(), auth + "X-Test-Config: legacy\r\n").body);
            check(!at(legacy, "strix").find("filters") && !at(legacy, "strix").find("seed"), "receipt off omits new policy fields");
        }
    }
    {
        const auto r = http(port, "GET", "/v1/capabilities", "", auth);
        const auto doc = parse_json(r.body);
        const auto id = at(doc, "capabilities_id").as_string();
        check(r.status == 200 && at(doc, "object").as_string() == "strix.capabilities" && at(doc, "version").as_int() == 1, "capabilities default");
        check(http(port, "GET", "/v1/capabilities", "").status == 401, "capabilities authenticated");
        check(at(parse_json(http(port, "GET", "/v1/capabilities", "", auth).body), "capabilities_id").as_string() == id, "capabilities stable id");
        Json canonical = Json::object();
        for (const auto& [key, value] : doc.as_object()) if (key != "capabilities_id") canonical.set(key, value);
        std::uint64_t hash = 14695981039346656037ULL;
        for (const unsigned char c : canonical.dump()) { hash ^= c; hash *= 1099511628211ULL; }
        char hex[17]; std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
        check(id == hex, "capabilities FNV-1a canonical dump");
        for (const char* path : {"/api/status", "/health"})
            check(at(parse_json(http(port, "GET", path, "", auth).body), "capabilities_id").as_string() == id, "capabilities reference " + std::string(path));
        const auto models = parse_json(http(port, "GET", "/v1/models", "", auth).body);
        check(at(at(models, "data").as_array()[0], "capabilities_id").as_string() == id, "models capabilities reference");
        for (const char* key : {"json_schema", "strict_tools", "penalties", "logit_bias", "logprobs", "retry", "thinking_budget"})
            check(!at(doc, key).as_bool(), "unavailable capability " + std::string(key));
        check(!at(at(doc, "sampling"), "sample").as_bool(), "greedy-only capability");
        for (const auto* field : {"temperature", "top_k", "top_p", "min_p"}) {
            const auto& entry = at(at(at(doc, "sampling"), "defaults"), field);
            check(at(entry, "source").as_string() == "builtin", "capabilities builtin source");
            check(at(entry, "value").as_double() == (std::string(field) == "top_p" ? 1.0 : 0.0), "capabilities builtin value");
        }
        check(ApiConfig{}.sse_keepalive_seconds == 10.0, "default keepalive interval");
        check(ApiConfig{}.sse_progress_seconds == 1.0 && !ApiConfig{}.engine_progress, "default progress policy");
        const auto& progress = at(at(doc, "stream"), "progress");
        check(at(progress, "interval_seconds").as_double() == config.sse_progress_seconds && at(progress, "engine_reports").as_bool() && at(doc, "prefill_progress").as_bool(), "progress capability");
        const auto sampled = parse_json(http(port, "GET", "/v1/capabilities", "", auth + "X-Test-Config: sample\r\n").body);
        check(at(at(sampled, "sampling"), "sample").as_bool() && at(sampled, "engine_feature").as_bool() && at(sampled, "capabilities_id").as_string() != id, "sampling capabilities and engine hook");
        const auto rejected = parse_json(http(port, "GET", "/v1/capabilities", "", auth + "X-Test-Config: reject\r\n").body);
        check(at(at(rejected, "sampling"), "positive_temperature").as_string() == "reject" && !at(at(rejected, "endpoints"), "responses").as_bool(), "reject policy and disabled endpoint capabilities");
        const auto before = engine.prompts().size();
        check(http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"temperature":1})", auth + "X-Test-Config: reject\r\n").status == 400 && engine.prompts().size() == before, "sampling reject semantics unchanged");
        const auto off = auth + "X-Test-Config: off\r\n";
        check(http(port, "GET", "/v1/capabilities", "", off).status == 404, "capabilities off");
        check(http(port, "POST", "/v1/responses", "{}", off).status == 404, "responses off");
        for (const char* path : {"/health", "/api/status"})
            check(!parse_json(http(port, "GET", path, "", off).body).find("capabilities_id"), "off omits id " + std::string(path));
        const auto off_models = parse_json(http(port, "GET", "/v1/models", "", off).body);
        check(!at(off_models, "data").as_array()[0].find("capabilities_id"), "off models omit id");
    }
    {
        // The real routes submit the same complete prompt and stable prefix, including literal argument bytes.
        std::vector<std::pair<Json, Json>> pairs;
        const auto pair = [&](const char* chat, const char* responses) { pairs.emplace_back(parse_json(chat), parse_json(responses)); };
        pair(R"({"messages":[{"role":"user","content":"Hi"}]})", R"({"input":"Hi"})");
        pair(R"({"messages":[{"role":"system","content":"Rules"},{"role":"user","content":"Hi"}]})", R"({"instructions":"Rules","input":"Hi"})");
        pair(R"({"messages":[{"role":"user","content":[{"type":"text","text":"  a\n"},{"type":"text","text":"\u03bb <think> <tool_call>  "}]}]})",
             R"({"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"  a\n"},{"type":"output_text","text":"\u03bb <think> <tool_call>  "}]}]})");
        pair(R"({"messages":[{"role":"developer","content":"Rules"},{"role":"user","content":"Hi"}]})", R"({"input":[{"role":"developer","content":"Rules"},{"role":"user","content":"Hi"}]})");
        pair(R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"description":"desc","name":"f","parameters":{"type":"object"},"strict":false}}],"tool_choice":"none"})",
             R"({"input":"Hi","tools":[{"type":"function","description":"desc","name":"f","parameters":{"type":"object"},"strict":false}],"tool_choice":"none","n":1,"previous_response_id":null,"store":false,"background":false})");
        pair(R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"description":"desc","name":"f","parameters":{"type":"object"},"strict":false}}],"tool_choice":"auto"})",
             R"({"input":"Hi","tools":[{"type":"function","description":"desc","name":"f","parameters":{"type":"object"},"strict":false}],"tool_choice":"auto"})");
        pair(R"({"messages":[{"role":"user","content":"First"},{"role":"assistant","content":"Answer","reasoning_content":"Reason\n"},{"role":"user","content":"Next"}],"preserve_thinking":true})",
             R"({"input":[{"role":"user","content":"First"},{"type":"reasoning","content":[{"type":"reasoning_text","text":"Reason\n"}]},{"role":"assistant","content":"Answer"},{"role":"user","content":"Next"}],"preserve_thinking":true})");
        pair(R"({"messages":[{"role":"user","content":"First"},{"role":"assistant","content":"Answer","reasoning_content":"Summary"},{"role":"user","content":"Next"}]})",
             R"({"input":[{"role":"user","content":"First"},{"type":"reasoning","summary":[{"type":"summary_text","text":"Summary"}]},{"role":"assistant","content":"Answer"},{"role":"user","content":"Next"}]})");
        for (const char* effort : {"none", "low", "high"}) {
            auto c = parse_json(R"({"messages":[{"role":"user","content":"Hi"}]})"); c.set("reasoning_effort", Json(effort));
            auto r = parse_json(R"({"input":"Hi"})"); Json reasoning = Json::object(); reasoning.set("effort", Json(effort)); r.set("reasoning", std::move(reasoning));
            pairs.emplace_back(std::move(c), std::move(r));
        }
        auto c = parse_json(R"({"messages":[{"role":"user","content":"Call"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":""}}]},{"role":"tool","tool_call_id":"c1","content":"result"}],"tools":[{"type":"function","function":{"name":"f","description":"function","parameters":{"type":"object","properties":{"s":{"type":"string"}}}}}]})");
        auto r = parse_json(R"({"input":[{"role":"user","content":"Call"},{"type":"function_call","call_id":"c1","name":"f","arguments":""},{"type":"function_call_output","call_id":"c1","output":"result"}],"tools":[{"type":"function","name":"f","description":"function","parameters":{"type":"object","properties":{"s":{"type":"string"}}}}]})");
        const std::string args = " {\n\"s\": \"quote \\\" backslash \\\\ newline \\n </tool_call> </parameter>\"\n} ";
        auto call = parse_json(R"({"id":"c1","type":"function","function":{"name":"f"}})");
        Json function = Json::object(); function.set("name", Json("f")); function.set("arguments", Json(args)); call.set("function", std::move(function));
        auto& messages = c.as_object()[0].second.as_array(); Json calls = Json::array(); calls.push(std::move(call)); messages[1].set("tool_calls", std::move(calls));
        r.as_object()[0].second.as_array()[1].set("arguments", Json(args));
        pairs.emplace_back(std::move(c), std::move(r));
        // an assistant message followed by two calls is one assistant turn with two tool calls
        pair(R"({"messages":[{"role":"user","content":"Both"},{"role":"assistant","content":"Calling","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{\"s\":\"1\"}"}},{"id":"b","type":"function","function":{"name":"f","arguments":"{\"s\":\"2\"}"}}]},{"role":"tool","tool_call_id":"a","content":"one"},{"role":"tool","tool_call_id":"b","content":"two"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]})",
             R"({"input":[{"role":"user","content":"Both"},{"role":"assistant","content":"Calling"},{"type":"function_call","call_id":"a","name":"f","arguments":"{\"s\":\"1\"}"},{"type":"function_call","call_id":"b","name":"f","arguments":"{\"s\":\"2\"}"},{"type":"function_call_output","call_id":"a","output":"one"},{"type":"function_call_output","call_id":"b","output":"two"}],"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]})");
        pair(R"({"messages":[{"role":"user","content":"Both"},{"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"a","content":"one"},{"role":"tool","tool_call_id":"b","content":"two"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]})",
             R"({"input":[{"role":"user","content":"Both"},{"type":"function_call","call_id":"a","name":"f","arguments":"{}"},{"type":"function_call","call_id":"b","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"a","output":"one"},{"type":"function_call_output","call_id":"b","output":"two"}],"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]})");
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            const auto before = engine.prompts().size();
            const auto a = http(port, "POST", "/v1/chat/completions", pairs[i].first.dump(), auth);
            const auto b = http(port, "POST", "/v1/responses", pairs[i].second.dump(), auth);
            const auto captured = engine.prompts();
            check(a.status == 200 && b.status == 200, "responses prompt pair accepted " + std::to_string(i) + " " + b.body);
            check(captured.size() == before + 2U && captured[before] == captured[before + 1U], "responses_prompt_ids_equal_chat " + std::to_string(i));
        }
    }
    {
        const std::vector<std::pair<std::string, std::string>> invalid{
            {R"({"previous_response_id":"resp_x"})", "previous_response_id"}, {R"({"store":true})", "store"},
            {R"({"conversation":"conv_1"})", "conversation"}, {R"({"background":true})", "background"},
            {R"({"input":[{"type":"unknown"}]})", "input[0].type"},
            {R"({"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]})", "input[0].content"},
            {R"({"input":[{"role":"user","content":[{"type":"input_file","file_id":"x"}]}]})", "input[0].content"},
            {R"({"tools":[{"type":"web_search"}]})", "tools[0].type"},
            {R"({"input":[{"type":"reasoning","encrypted_content":"x"}]})", "input[0]"},
            {R"({"input":[{"type":"reasoning","summary":[],"encrypted_content":"x"}]})", "input[0]"},
            {R"({"input":[{"type":"function_call_output","call_id":"missing","output":"x"}]})", "input[0].call_id"},
            {R"({"input":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]})", "input[1].call_id"},
            {R"({"input":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"x"},{"type":"function_call_output","call_id":"c","output":"x"}]})", "input[2].call_id"},
            {R"({"text":{"format":{"type":"json_schema"}}})", "text.format"}, {R"({"text":{"format":{"type":"json_object"}}})", "text.format"},
            {R"({"text":{"format":{"type":"other"}}})", "text.format"},
            {R"({"tools":[{"type":"function","name":"f","strict":true}]})", "tools[0].strict"},
            {R"({"tool_choice":"required"})", "tool_choice"}, {R"({"tool_choice":{"type":"function","name":"f"}})", "tool_choice"},
            {R"({"n":"1"})", "n"}, {R"({"n":2})", "n"}, {R"({"instructions":[]})", "instructions"}
        };
        const auto before = engine.prompts().size();
        for (const auto& [body, param] : invalid) {
            auto request = parse_json(body); if (!request.find("input")) request.set("input", Json("Hi"));
            const auto reply = http(port, "POST", "/v1/responses", request.dump(), auth);
            check(reply.status == 400 && at(at(parse_json(reply.body), "error"), "param").as_string() == param, "responses_validation " + param + " " + reply.body);
        }
        check(engine.prompts().size() == before, "responses_validation never invokes engine");
    }
    {
        const auto plain = http(port, "POST", "/v1/responses", R"({"input":"Hi","temperature":1,"text":{"format":{"type":"text"}}})", auth);
        const auto full = parse_json(plain.body);
        check(plain.status == 200 && at(full, "object").as_string() == "response" && at(full, "status").as_string() == "completed", "responses non-stream shape");
        check(at(full, "id").as_string().starts_with("resp_") && at(full, "created_at").is_number(), "responses id and time");
        const auto& output = at(full, "output").as_array();
        check(output.size() == 2U && at(output[0], "type").as_string() == "reasoning" && at(output[0], "summary").size() == 0U, "responses reasoning item");
        check(at(output[1], "role").as_string() == "assistant" && at(output[1], "status").as_string() == "completed", "responses message item");
        const auto expected_text = at(at(output[1], "content").as_array()[0], "text").as_string();
        const auto expected_reasoning = at(at(output[0], "content").as_array()[0], "text").as_string();
        check(at(at(output[1], "content").as_array()[0], "annotations").size() == 0U, "responses annotations");
        const auto& usage = at(full, "usage");
        check(at(usage, "input_tokens").as_int() + at(usage, "output_tokens").as_int() == at(usage, "total_tokens").as_int() &&
            at(at(usage, "input_tokens_details"), "cached_tokens").as_int() == 0, "responses usage");
        check(at(at(full, "strix"), "sampling_requested").as_string() == "sample" && at(at(full, "strix"), "sampling_effective").as_string() == "greedy" &&
            !at(at(full, "strix"), "note").as_string().empty(), "responses downgrade receipt");
        const auto incomplete = parse_json(http(port, "POST", "/v1/responses", R"({"input":"Hi","max_output_tokens":4})", auth).body);
        check(at(incomplete, "status").as_string() == "incomplete" && at(at(incomplete, "incomplete_details"), "reason").as_string() == "max_output_tokens", "responses non-stream incomplete");
        const auto status = parse_json(http(port, "GET", "/api/status", "").body);
        check(at(at(status, "requests").as_array()[0], "route").as_string() == "responses", "responses recent request route");
        for (const char* input : {"Hi", "TOOL", "ENGINE_ERROR", "length"}) {
            Json request = Json::object(); request.set("input", Json(input)); request.set("stream", Json(true));
            if (std::string(input) == "TOOL") request.set("tools", parse_json(R"([{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"days":{"type":"integer"}}}}])"));
            if (std::string(input) == "length") request.set("max_output_tokens", Json(4));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            check(r.status == 200 && r.body.find("[DONE]") == std::string::npos, "responses SSE transport");
            const auto events = sse_data(r.body);
            check(at(events[0], "type").as_string() == "response.created" && at(events[1], "type").as_string() == "response.in_progress", "responses created/progress order");
            check(at(events[0], "response").find("strix"), "responses first receipt");
            std::set<std::int64_t> added, parts, texts_done, parts_done, items_done;
            std::string text, reasoning, arguments;
            std::int64_t sequence = 0;
            int terminal = 0;
            for (const auto& e : events) {
                const auto type = at(e, "type").as_string();
                check(at(e, "sequence_number").as_int() == sequence++, "responses sequence_number");
                check(r.body.find("event: " + type + "\ndata: " + e.dump() + "\n\n") != std::string::npos, "responses event == data.type");
                if (type == "response.output_item.added") added.insert(at(e, "output_index").as_int());
                if (type == "response.content_part.added") {
                    const auto i = at(e, "output_index").as_int(); check(added.count(i) && !items_done.count(i), "part follows item added"); parts.insert(i);
                }
                if (type.ends_with(".delta")) {
                    const auto i = at(e, "output_index").as_int(); check(added.count(i) && !items_done.count(i) && !texts_done.count(i), "delta ordering");
                    if (type == "response.output_text.delta") { check(parts.count(i), "text part added before delta"); text += at(e, "delta").as_string(); }
                    if (type == "response.reasoning_text.delta") reasoning += at(e, "delta").as_string();
                    if (type == "response.function_call_arguments.delta") arguments += at(e, "delta").as_string();
                }
                if (type == "response.output_text.done" || type == "response.reasoning_text.done" || type == "response.function_call_arguments.done") texts_done.insert(at(e, "output_index").as_int());
                if (type == "response.content_part.done") {
                    const auto i = at(e, "output_index").as_int(); check(texts_done.count(i), "part done after text done"); parts_done.insert(i);
                }
                if (type == "response.output_item.done") {
                    const auto i = at(e, "output_index").as_int(); check(added.count(i) && texts_done.count(i) && (!parts.count(i) || parts_done.count(i)), "item done ordering"); items_done.insert(i);
                }
                if (type == "response.completed" || type == "response.incomplete" || type == "response.failed") {
                    ++terminal; check(items_done == added && at(e, "response").find("strix"), "terminal complete output and receipt");
                }
            }
            check(terminal == 1, "exactly one responses terminal");
            if (std::string(input) == "Hi") check(text == expected_text && reasoning == expected_reasoning, "responses deltas equal non-stream text");
            if (std::string(input) == "TOOL") {
                const auto& result = at(events.back(), "response"); const auto& calls = at(result, "output").as_array();
                check(at(calls.back(), "type").as_string() == "function_call" && !at(calls.back(), "call_id").as_string().empty() && at(calls.back(), "arguments").as_string() == arguments, "responses tool item and arguments");
                request.set("stream", Json(false)); const auto nonstream = parse_json(http(port, "POST", "/v1/responses", request.dump(), auth).body);
                check(at(at(nonstream, "output").as_array().back(), "arguments").as_string() == arguments, "responses tool arguments equal non-stream");
            }
            if (std::string(input) == "length") check(at(at(at(events.back(), "response"), "incomplete_details"), "reason").as_string() == "max_output_tokens", "responses incomplete reason");
            if (std::string(input) == "ENGINE_ERROR") check(at(events.back(), "type").as_string() == "response.failed", "responses engine failure terminal");
        }
        check(http(port, "POST", "/v1/responses", R"({"input":"ENGINE_ERROR"})", auth).status == 500, "responses non-stream engine failure");
    }
    {
        const auto body = R"({"messages":[{"role":"user","content":"Hi"}],"stream":true,"temperature":1,"max_tokens":1000})";
        const auto events = sse_data(http(port, "POST", "/v1/chat/completions", body, auth).body);
        const auto& receipt = at(events.front(), "strix");
        check(at(receipt, "sampling_requested").as_string() == "sample" && at(receipt, "sampling_effective").as_string() == "greedy", "first chat receipt without usage");
        check(at(receipt, "max_tokens_requested").as_int() == 1000 && at(receipt, "max_tokens_effective").as_int() == 1000 && receipt.find("note"), "first chat policy limit and note");
        check(events.back().find("strix"), "final chat receipt without usage");
        const auto old = sse_data(http(port, "POST", "/v1/chat/completions", body, auth + "X-Test-Config: legacy\r\n").body);
        check(!old.front().find("strix") && !old.back().find("strix"), "receipts off");
        // Build every old chunk with the original ordered fields, normalizing only the random id and timestamp.
        const auto normalize = [](const Json& value) {
            Json out = parse_json(value.dump()); out.set("id", Json("ID")); out.set("created", Json(0)); return out.dump();
        };
        for (const auto& e : old) {
            Json golden = Json::object(); golden.set("id", Json("ID")); golden.set("object", Json("chat.completion.chunk")); golden.set("created", Json(0));
            golden.set("model", Json("test-model")); golden.set("choices", at(e, "choices"));
            check(normalize(e) == golden.dump(), "old chat chunk golden bytes");
        }
        check(events.size() == old.size(), "receipt does not add chat chunks");
        for (std::size_t i = 0; i < events.size() && i < old.size(); ++i)
            check(at(events[i], "choices").dump() == at(old[i], "choices").dump(), "receipt leaves chat deltas unchanged");
    }
    {
        for (const char* route : {"/v1/chat/completions", "/v1/messages", "/v1/completions", "/v1/responses"}) {
            auto request = parse_json(R"({"stream":true,"max_tokens":1000,"return_token_ids":true})");
            const std::string path = route;
            if (path == "/v1/responses") request.set("input", Json("PROGRESS"));
            else if (path == "/v1/completions") request.set("prompt", Json("PROGRESS"));
            else request.set("messages", parse_json(R"([{"role":"user","content":"PROGRESS"}])"));
            const auto active = http(port, "POST", path, request.dump(), auth);
            const auto off = http(port, "POST", path, request.dump(), auth + "X-Test-Config: progress-off\r\n");
            request.set("strix", parse_json(R"({"progress":false})"));
            const auto disabled = http(port, "POST", path, request.dump(), auth);
            check(off.body.find(": progress ") == std::string::npos && disabled.body.find(": progress ") == std::string::npos, "progress off switches " + path);
            std::int64_t prefilled = 0, cached = 0;
            bool saw_prefill = false, saw_decode = false;
            for (auto pos = active.body.find(": progress "); pos != std::string::npos; pos = active.body.find(": progress ", pos + 1)) {
                const auto value = parse_json(active.body.substr(pos + 11, active.body.find("\n\n", pos) - pos - 11));
                const auto n = at(value, "prefilled_tokens").as_int(), c = at(value, "cached_tokens").as_int(), t = at(value, "total_tokens").as_int();
                check(n >= prefilled && c >= cached && n + c <= t, "progress counters " + path);
                check(value.dump() == "{\"phase\":\"" + at(value, "phase").as_string() + "\",\"prefilled_tokens\":" + std::to_string(n) + ",\"cached_tokens\":" + std::to_string(c) + ",\"total_tokens\":" + std::to_string(t) + "}", "compact ordered progress JSON " + path);
                prefilled = n; cached = c;
                if (at(value, "phase").as_string() == "prefill") { check(!saw_decode, "progress phase order " + path); saw_prefill = true; }
                else { check(saw_prefill, "prefill before decode " + path); saw_decode = true; }
            }
            check(saw_prefill && saw_decode, "progress phases " + path);
            check(active.body.find(": keepalive") == std::string::npos, "progress resets keepalive " + path);
            const auto a = sse_data(active.body), b = sse_data(off.body), c = sse_data(disabled.body);
            const std::function<Json(const Json&)> payload = [&](const Json& value) -> Json {
                if (value.is_array()) { Json out = Json::array(); for (const auto& v : value.as_array()) out.push(payload(v)); return out; }
                if (!value.is_object()) return value;
                Json out = Json::object();
                for (const auto& [key, v] : value.as_object())
                    if (key != "id" && key != "item_id" && key != "created" && key != "created_at" && key != "strix") out.set(key, payload(v));
                return out;
            };
            check(a.size() == b.size() && a.size() == c.size(), "progress data count " + path);
            for (std::size_t i = 0; i < a.size() && i < b.size() && i < c.size(); ++i) {
                check(payload(a[i]).dump() == payload(b[i]).dump() && payload(a[i]).dump() == payload(c[i]).dump(), "progress data events identical " + path);
                // IDs and timestamps are request-local; compare the generation payloads and final token IDs.
                for (const char* key : {"choices", "delta", "token_ids"})
                    if (const auto* v = a[i].find(key)) check(b[i].find(key) && c[i].find(key) && v->dump() == at(b[i], key).dump() && v->dump() == at(c[i], key).dump(), "progress data unchanged " + path);
            }
            const auto ids = [&](const std::vector<Json>& values) {
                const auto& last = values.back();
                return at(path == "/v1/responses" ? at(at(last, "response"), "strix") : at(last, "strix"), "token_ids").dump();
            };
            check(ids(a) == ids(b) && ids(a) == ids(c), "progress token IDs identical " + path);
            request.set("stream", Json(false)); request.set("strix", Json::object());
            const auto plain = http(port, "POST", path, request.dump(), auth);
            check(plain.body.find(": progress ") == std::string::npos, "non-stream has no progress " + path);
        }
    }
    {
        for (const char* route : {"/v1/chat/completions", "/v1/messages", "/v1/completions", "/v1/responses"}) {
            auto request = parse_json(R"({"stream":true,"max_tokens":1000,"return_token_ids":true})");
            const std::string path = route;
            if (path == "/v1/responses") request.set("input", Json("SLOW"));
            else if (path == "/v1/completions") request.set("prompt", Json("SLOW"));
            else request.set("messages", parse_json(R"([{"role":"user","content":"SLOW"}])"));
            const auto active = http(port, "POST", path, request.dump(), auth);
            // the opening events (role chunk, message_start, response.created + in_progress) are written at once; the
            // slow engine's first text comes after keepalive comments
            const auto keepalive = active.body.find(": keepalive\n\n");
            std::size_t opening = 0;
            for (auto at_ = active.body.find("data: "); at_ < keepalive; at_ = active.body.find("data: ", at_ + 1)) ++opening;
            const std::size_t expected = path == "/v1/responses" ? 2U : 1U;
            check(active.body.find(": progress ") == std::string::npos, "no fabricated progress " + path);
            check(keepalive != std::string::npos && opening == expected, "keepalive before the first generated text " + path);
            const auto quiet = http(port, "POST", path, request.dump(), auth + "X-Test-Config: quiet\r\n");
            request.set("strix", parse_json(R"({"keepalive":false})"));
            const auto disabled = http(port, "POST", path, request.dump(), auth);
            check(quiet.body.find(": keepalive") == std::string::npos && disabled.body.find(": keepalive") == std::string::npos, "keepalive off switches " + path);
            const auto a = sse_data(active.body), b = sse_data(quiet.body), c = sse_data(disabled.body);
            check(a.size() == b.size() && a.size() == c.size(), "keepalive event count unchanged " + path);
            const auto ids = [&](const std::vector<Json>& values) {
                const auto& last = values.back();
                return at(path == "/v1/responses" ? at(at(last, "response"), "strix") : at(last, "strix"), "token_ids").dump();
            };
            check(ids(a) == ids(b) && ids(a) == ids(c), "keepalive generated ids identical " + path);
            for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
                if (path == "/v1/chat/completions" || path == "/v1/completions") check(at(a[i], "choices").dump() == at(b[i], "choices").dump(), "keepalive deltas unchanged " + path);
                if (const auto* delta = a[i].find("delta")) check(delta->dump() == at(b[i], "delta").dump(), "keepalive delta unchanged " + path);
            }
            if (path == "/v1/messages") check(at(a.front(), "message").find("strix") && a.back().find("strix"), "messages first and final receipt");
            if (path == "/v1/completions") check(a.front().find("strix") && a.back().find("strix"), "completions first and final receipt");
        }
    }
    {
        // A call is completed only when its frame closed: as the budget grows, every call status goes incomplete ->
        // completed exactly once (streamed or not), even while the response itself is still incomplete.
        const auto tools = R"([{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}])";
        for (const bool stream : {false, true}) {
            std::string history;  // per limit: '-' no call, 'i' incomplete call, 'c' completed call
            bool completed_in_incomplete_response = false;
            for (int limit = 20; limit <= 140; ++limit) {
                auto request = parse_json(R"({"input":"TOOL"})"); request.set("tools", parse_json(tools));
                request.set("max_output_tokens", Json(limit)); request.set("stream", Json(stream));
                const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
                const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
                char state = '-';
                for (const auto& item : at(response, "output").as_array())
                    if (at(item, "type").as_string() == "function_call") state = at(item, "status").as_string() == "completed" ? 'c' : 'i';
                if (state == 'c' && at(response, "status").as_string() == "incomplete") completed_in_incomplete_response = true;
                history += state;
            }
            const auto first_i = history.find('i'), first_c = history.find('c');
            check(first_i != std::string::npos && first_c != std::string::npos && first_i < first_c &&
                  history.find('-', first_i) == std::string::npos && history.find('i', first_c) == std::string::npos,
                  std::string("call status monotone in the budget ") + (stream ? "stream " : "") + history);
            check(completed_in_incomplete_response, "a closed call stays completed when the budget then ends the response");
        }
        // A stop string inside the arguments cuts the frame: the call is incomplete although the response stopped.
        for (const bool stream : {false, true}) {
            auto request = parse_json(R"({"input":"TOOL","stop":["slo"]})"); request.set("tools", parse_json(tools)); request.set("stream", Json(stream));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
            int calls = 0;
            for (const auto& item : at(response, "output").as_array())
                if (at(item, "type").as_string() == "function_call") { ++calls; check(at(item, "status").as_string() == "incomplete", "stop-cut call incomplete"); }
            check(calls == 1, "stop-cut call reported " + r.body.substr(0, 200));
        }
        const auto cut = parse_assistant_output("x</think>\n\n<tool_call>\n<function=f>\n<parameter=a>\nv", true, Json());
        const auto closed = parse_assistant_output("<tool_call>\n<function=f>\n</function>\n</tool_call><tool_call>\n<function=g>", false, Json());
        check(cut.call_complete == std::vector<bool>{false} && closed.call_complete == std::vector<bool>{true, false}, "parser call_complete");
        check(parse_assistant_output("only reasoning", true, Json()).tool_calls.is_array(), "tool_calls array inside thinking");
    }
    {
        // Generation-affecting fields the server does not execute are refused before the engine runs; neutral values pass.
        const auto before = engine.prompts().size();
        const std::vector<std::pair<std::string, std::string>> refused = {
            {"presence_penalty", R"(0.5)"}, {"frequency_penalty", R"(-1)"}, {"repetition_penalty", R"(1.1)"},
            {"logit_bias", R"({"5":-100})"}, {"logprobs", R"(true)"}, {"top_logprobs", R"(3)"},
            {"response_format", R"({"type":"json_object"})"}, {"parallel_tool_calls", R"(false)"}, {"tool_choice", R"("required")"},
            {"tool_choice", R"({"type":"function","function":{"name":"f"}})"},
        };
        for (const auto& [key, value] : refused)
            for (const char* route : {"/v1/chat/completions", "/v1/responses"}) {
                if (std::string(route) == "/v1/responses" && key == "tool_choice") continue;  // the Responses translation refuses it itself
                auto request = parse_json(std::string(route) == "/v1/responses" ? R"({"input":"Hi"})" : R"({"messages":[{"role":"user","content":"Hi"}]})");
                request.set(key, parse_json(value));
                const auto r = http(port, "POST", route, request.dump(), auth);
                check(r.status == 400 && at(at(parse_json(r.body), "error"), "param").as_string() == key, "refused " + key + " on " + route + ": " + r.body);
            }
        auto strict = parse_json(R"({"messages":[{"role":"user","content":"Hi"}],"tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"}}}]})");
        check(http(port, "POST", "/v1/chat/completions", strict.dump(), auth).status == 400, "strict chat tool refused");
        const auto any = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"any"}})", auth);
        check(any.status == 400, "messages forced tool_choice refused");
        for (const char* body : {R"({"input":"Hi","include":["reasoning.encrypted_content"]})", R"({"input":"Hi","truncation":"auto"})", R"({"input":"Hi","max_tool_calls":2})"})
            check(http(port, "POST", "/v1/responses", body, auth).status == 400, std::string("responses refuses ") + body);
        check(engine.prompts().size() == before, "refused requests never reach the engine");
        const auto neutral = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"presence_penalty":0,"frequency_penalty":0,"logit_bias":{},"logprobs":false,"top_logprobs":0,"response_format":{"type":"text"},"parallel_tool_calls":true,"tool_choice":"auto"})", auth);
        check(neutral.status == 200, "neutral values pass " + neutral.body);
        const auto neutral_messages = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"auto"}})", auth);
        check(neutral_messages.status == 200, "messages auto tool_choice passes " + neutral_messages.body);
    }
    {
        // A tool-call header cut before any call parses: no call, but the response is incomplete (unfinished_tool_call).
        const auto header = parse_assistant_output("<tool_call>\n<function=nex", false, Json());
        check(header.tool_calls.size() == 0U && header.unfinished_call, "parser keeps the unfinished header");
        for (const bool stream : {false, true}) {
            auto request = parse_json(R"({"input":"TOOL","stop":["_weather"],"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]})");
            request.set("stream", Json(stream));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
            check(at(response, "status").as_string() == "incomplete" &&
                  at(at(response, "incomplete_details"), "reason").as_string() == "unfinished_tool_call", "cut header => incomplete " + r.body.substr(0, 300));
        }
        // Without tools an unclosed marker is plain text: the reply stays completed.
        for (const bool stream : {false, true}) {
            auto request = parse_json(R"({"input":"TOOL","stop":["_weather"]})"); request.set("stream", Json(stream));
            const auto r = http(port, "POST", "/v1/responses", request.dump(), auth);
            const Json response = stream ? at(sse_data(r.body).back(), "response") : parse_json(r.body);
            check(at(response, "status").as_string() == "completed", "literal marker without tools stays completed");
        }
        // DeepSeek: a calls block that never closed is an unfinished call; a closed malformed one is not.
        const std::string dsml = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";
        const auto ds_cut = parse_deepseek_v41_output("Answer.\n\n<" + dsml + " calls>\n<" + dsml + " invoke name=\"f\">\n", false, Json());
        check(ds_cut.unfinished_call && ds_cut.tool_calls.size() == 0U, "deepseek unfinished calls block");
        check(!parse_deepseek_v41_output("Answer.", false, Json()).unfinished_call, "deepseek plain answer finished");
        check(http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"output_config":{"effort":null,"format":null}})", auth).status == 200,
              "messages output_config null members pass");
        // Anthropic: tool_choice none renders no tools; unexecuted constraints are refused; the budget is stated.
        const auto before = engine.prompts().size();
        const auto plain = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}]})", auth);
        const auto none = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"none"}})", auth);
        const auto prompts = engine.prompts();
        check(plain.status == 200 && none.status == 200 && prompts.size() == before + 2U && prompts[before] == prompts[before + 1U], "messages tool_choice none renders no tools");
        for (const char* body : {
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"auto","disable_parallel_tool_use":true}})",
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tools":[{"name":"f","strict":true,"input_schema":{"type":"object"}}]})",
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"adaptive"}})",
                 R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"output_config":{"effort":"low"}})"})
            check(http(port, "POST", "/v1/messages", body, auth).status == 400, std::string("messages refuses ") + body);
        const auto budget = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"enabled","budget_tokens":1024}})", auth);
        check(budget.status == 200 && budget.body.find("budget_tokens not enforced") != std::string::npos, "thinking budget stated as not enforced");
        const auto budget_stream = http(port, "POST", "/v1/messages", R"({"max_tokens":8,"stream":true,"messages":[{"role":"user","content":"Hi"}],"thinking":{"type":"enabled","budget_tokens":1024}})",
            auth + "X-Test-Config: legacy\r\n");
        check(budget_stream.body.find("budget_tokens not enforced") != std::string::npos, "budget note kept in a stream without receipts");
        check(http(port, "POST", "/v1/messages", R"({"max_tokens":8,"messages":[{"role":"user","content":"Hi"}],"tool_choice":{"type":"auto","disable_parallel_tool_use":false}})", auth).status == 200,
              "disable_parallel_tool_use false passes");
        // Completions: one candidate, no suffix, no scores; chat: no legacy functions.
        for (const char* body : {R"({"prompt":"Hi","max_tokens":4,"n":2})", R"({"prompt":"Hi","max_tokens":4,"best_of":2})",
                 R"({"prompt":"Hi","max_tokens":4,"suffix":"end"})", R"({"prompt":"Hi","max_tokens":4,"logprobs":0})"})
            check(http(port, "POST", "/v1/completions", body, auth).status == 400, std::string("completions refuses ") + body);
        check(http(port, "POST", "/v1/completions", R"({"prompt":"Hi","max_tokens":4,"n":1,"best_of":1,"suffix":null,"logprobs":null})", auth).status == 200, "completions neutral values pass");
        for (const char* body : {R"({"messages":[{"role":"user","content":"Hi"}],"functions":[{"name":"f","parameters":{"type":"object"}}]})",
                 R"({"messages":[{"role":"user","content":"Hi"}],"function_call":{"name":"f"}})"})
            check(http(port, "POST", "/v1/chat/completions", body, auth).status == 400, std::string("chat refuses ") + body);
        // Responses: null is absent; verbosity other than medium is refused.
        const auto nulls = http(port, "POST", "/v1/responses", R"({"input":"Hi","conversation":null,"instructions":null,"reasoning":{"effort":null},"text":{"verbosity":null,"format":null},"tools":[{"type":"function","name":"f","strict":null,"parameters":{"type":"object"}}],"tool_choice":null,"previous_response_id":null})", auth);
        check(nulls.status == 200, "responses nullable fields " + nulls.body.substr(0, 200));
        check(http(port, "POST", "/v1/responses", R"({"input":"Hi","reasoning":null})", auth).status == 200, "responses reasoning null");
        check(http(port, "POST", "/v1/responses", R"({"input":"Hi","text":{"verbosity":"low"}})", auth).status == 400, "responses refuses verbosity low");
    }
    {
        // A failed stream: response.failed, and no item carries a failed status.
        const auto r = http(port, "POST", "/v1/responses", R"({"input":"ENGINE_ERROR","stream":true})", auth);
        const auto events = sse_data(r.body);
        check(at(events.back(), "type").as_string() == "response.failed", "failed stream ends response.failed");
        for (const auto& e : events)
            if (const auto* item = e.find("item"); item && item->find("status"))
                check(at(*item, "status").as_string() != "failed", "items never carry a failed status");
        // Output text carries (empty) logprobs arrays while token scores are unavailable.
        const auto ok = sse_data(http(port, "POST", "/v1/responses", R"({"input":"Hi","stream":true})", auth).body);
        int text_events = 0;
        for (const auto& e : ok) {
            const auto& type = at(e, "type").as_string();
            if (type == "response.output_text.delta" || type == "response.output_text.done") { ++text_events; check(at(e, "logprobs").is_array(), "text event logprobs " + type); }
            if (type == "response.content_part.done" && at(at(e, "part"), "type").as_string() == "output_text") check(at(at(e, "part"), "logprobs").is_array(), "output_text part logprobs");
        }
        check(text_events >= 2, "output text events seen");
        // Reasoning summaries are refused before the engine runs.
        const auto before = engine.prompts().size();
        for (const char* body : {R"({"input":"Hi","reasoning":{"summary":"detailed"}})", R"({"input":"Hi","reasoning":{"generate_summary":"auto"}})"}) {
            const auto rejected = http(port, "POST", "/v1/responses", body, auth);
            check(rejected.status == 400 && rejected.body.find("reasoning.") != std::string::npos, "reasoning summary refused");
        }
        check(engine.prompts().size() == before, "refused summaries never reach the engine");
    }
    {
        const auto r = http(port, "GET", "/health", "");
        check(r.status == 200 && at(parse_json(r.body), "status").as_string() == "ok", "health");
        check(http(port, "GET", "/v1/models", "").status == 401, "models needs the API key");
        const auto models = http(port, "GET", "/v1/models", "", auth);
        check(models.status == 200 && at(at(parse_json(models.body), "data").as_array().at(0), "id").as_string() == "test-model", "models");
    }
    {
        const auto r = http(port, "POST", "/v1/chat/completions", R"({"model":"x","messages":[{"role":"user","content":"Hi"}]})", auth);
        check(r.status == 200, "chat status " + std::to_string(r.status) + " " + r.body);
        const auto j = parse_json(r.body);
        const auto& message = at(first_choice(j), "message");
        check(at(message, "content").as_string() == "Hello there, friend.", "chat content");
        check(at(message, "reasoning_content").as_string() == "Let me think.", "chat reasoning");
        check(at(first_choice(j), "finish_reason").as_string() == "stop", "chat finish stop");
        check(at(at(j, "usage"), "prompt_tokens").as_int() > 0, "chat usage");
    }
    {
        const auto r = http(port, "POST", "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true}})", auth);
        std::string reasoning, finish;
        bool usage = false;
        const auto content = stream_text(r.body, &reasoning, &finish, &usage, nullptr);
        check(content == "Hello there, friend." && reasoning == "Let me think.", "chat stream deltas [" + content + "] [" + reasoning + "]");
        check(finish == "stop" && usage && r.body.ends_with("data: [DONE]\n\n"), "chat stream finish, usage and [DONE]");
    }
    {
        const std::string body = R"({"messages":[{"role":"user","content":"TOOL please"}],"tools":[{"type":"function","function":{"name":"get_weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]})";
        const auto j = parse_json(http(port, "POST", "/v1/chat/completions", body, auth).body);
        const auto& choice = first_choice(j);
        const auto& message = at(choice, "message");
        const auto* calls = message.find("tool_calls");
        check(at(choice, "finish_reason").as_string() == "tool_calls" && calls && calls->size() == 1U, "tool call finish " + j.dump());
        if (calls && calls->size() == 1U) {
            const auto args = parse_json(at(at(calls->as_array()[0], "function"), "arguments").as_string());
            check(args.find("days") && at(args, "city").as_string() == "Oslo" && at(args, "days").as_int() == 2, "tool call arguments " + j.dump());
        }
        check(at(message, "content").is_string() && at(message, "content").as_string() == "Checking.", "content before the tool call");
        std::string stream_body = body;
        stream_body.insert(1, R"("stream":true,)");
        bool tools = false;
        const auto streamed = stream_text(http(port, "POST", "/v1/chat/completions", stream_body, auth).body, nullptr, nullptr, nullptr, &tools);
        check(streamed == "Checking." && tools, "streamed tool call [" + streamed + "]");
    }
    {
        const auto j = parse_json(http(port, "POST", "/v1/chat/completions",
            R"({"messages":[{"role":"user","content":"Hi"}],"enable_thinking":false})", auth).body);
        check(at(at(first_choice(j), "message"), "content").as_string() == "Plain answer.", "thinking off answer");
        const auto s = parse_json(http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"stop":"there"})", auth).body);
        check(at(at(first_choice(s), "message"), "content").as_string() == "Hello" && at(first_choice(s), "finish_reason").as_string() == "stop",
            "stop string truncates " + s.dump());
    }
    {
        const auto j = parse_json(http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"LONG"}],"max_tokens":20})", auth).body);
        check(at(first_choice(j), "finish_reason").as_string() == "length" && at(at(j, "usage"), "completion_tokens").as_int() == 20,
            "max_tokens finish length " + j.dump());
    }
    {
        const auto j = parse_json(http(port, "POST", "/v1/completions", R"({"prompt":"Hello","max_tokens":50})", auth).body);
        check(at(j, "object").as_string() == "text_completion" &&
            at(first_choice(j), "text").as_string().find("Hello there") != std::string::npos, "completions " + j.dump());
        const auto ids = parse_json(http(port, "POST", "/v1/completions", R"({"prompt":[72,101],"max_tokens":5,"return_token_ids":true})", auth).body);
        check(at(at(ids, "usage"), "prompt_tokens").as_int() == 2, "completions with token IDs " + ids.dump());
        check(at(at(ids, "strix"), "token_ids").size() == 5U, "returned token IDs " + ids.dump());
    }
    {
        const std::string body = R"({"model":"x","max_tokens":100,"system":"Be brief.","thinking":{"type":"enabled","budget_tokens":100},
            "messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]})";
        const auto j = parse_json(http(port, "POST", "/v1/messages", body, "x-api-key: secret\r\n").body);
        const auto& content = at(j, "content").as_array();
        check(at(j, "type").as_string() == "message" && content.size() == 2U && at(content[0], "type").as_string() == "thinking" &&
            at(content[1], "text").as_string() == "Hello there, friend." && at(j, "stop_reason").as_string() == "end_turn", "anthropic message " + j.dump());
        std::string stream_body = body;
        stream_body.insert(1, R"("stream":true,)");
        const auto events = http(port, "POST", "/v1/messages", stream_body, "x-api-key: secret\r\n").body;
        check(events.find("event: message_start") == 0 && events.find("event: content_block_start") != std::string::npos &&
            events.find("thinking_delta") != std::string::npos && events.find("text_delta") != std::string::npos &&
            events.rfind("event: message_stop") != std::string::npos, "anthropic event stream");
        const auto tool = parse_json(http(port, "POST", "/v1/messages", R"({"max_tokens":1000,"messages":[{"role":"user","content":"TOOL"}],
            "tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"days":{"type":"integer"}}}}]})", "x-api-key: secret\r\n").body);
        bool used = false;
        for (const auto& block : at(tool, "content").as_array())
            if (at(block, "type").as_string() == "tool_use" && at(at(block, "input"), "days").as_int() == 2) used = true;
        check(used && at(tool, "stop_reason").as_string() == "tool_use", "anthropic tool_use " + tool.dump());
    }
    {
        const auto bad = http(port, "POST", "/v1/chat/completions", "{nope", auth);
        check(bad.status == 400 && at(at(parse_json(bad.body), "error"), "type").as_string() == "invalid_request_error", "bad JSON");
        check(http(port, "POST", "/v1/nowhere", "{}", auth).status == 404, "unknown route");
        const auto anthropic = http(port, "POST", "/v1/messages", R"({"messages":[]})", auth);
        check(anthropic.status == 400 && at(parse_json(anthropic.body), "type").as_string() == "error", "anthropic error format");
    }
    {
        // Two concurrent clients: both complete with their own outputs (FIFO on one engine).
        Reply a, b;
        std::thread ta([&] { a = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"LONG"}],"max_tokens":60})", auth); });
        std::thread tb([&] { b = http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"Hi"}],"stream":true})", auth); });
        ta.join(); tb.join();
        check(a.status == 200 && at(at(parse_json(a.body), "usage"), "completion_tokens").as_int() == 60, "concurrent client A");
        check(b.status == 200 && stream_text(b.body, nullptr, nullptr, nullptr, nullptr) == "Hello there, friend.", "concurrent client B");
    }
    {
        const auto before = engine.cancelled.load();
        (void)http(port, "POST", "/v1/chat/completions", R"({"messages":[{"role":"user","content":"LONG"}],"stream":true,"max_tokens":400})", auth, 300);
        for (int i = 0; i < 200 && engine.cancelled.load() == before; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        check(engine.cancelled.load() > before, "client disconnect cancels the generation");
        const auto status = parse_json(http(port, "GET", "/api/status", "").body);
        check(at(status, "schema").as_string() == "strix-serve-status-v1" && at(status, "requests").size() > 0U, "status");
        const auto metrics = http(port, "GET", "/metrics", "").body;
        check(metrics.find("strix_generated_tokens_total") != std::string::npos, "metrics");
    }
    } catch (const std::exception& error) {
        check(false, std::string("exception: ") + error.what());
    }
    server.stop();
    queue.shutdown();
    http_thread.join();
    engine_thread.join();
    std::cout << "serve API tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "serve API tests: " << error.what() << '\n';
    return 1;
}
