// Portable, in-memory HTTP/API regression; --dump is compared against 05fedd80.
#include "strix/serve/api.hpp"
#include "strix/serve/session_api.hpp"
#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/deepseek41_serve_sched.hpp"
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace strix::serve;
#if defined(_WIN32)
// The socket server is POSIX; this test needs only header lookup on Windows.
std::string HttpRequest::header(std::string_view name) const {
    for (const auto& h : headers) if (h.first == name) return h.second;
    return {};
}
#endif
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


struct Reply : HttpResponseWriter {
    int status = 0;
    std::string bytes;
    void send(int s, std::string_view type, std::string_view body,
              const std::vector<std::pair<std::string,std::string>>& headers = {}) override {
        status = s; bytes += std::to_string(s) + "\n" + std::string(type) + "\n";
        for (const auto& h : headers) bytes += h.first + ": " + h.second + "\n";
        bytes += body;
    }
    bool begin_stream(int, std::string_view, const std::vector<std::pair<std::string,std::string>>&) override { throw std::runtime_error("unexpected stream"); }
    bool write(std::string_view) override { return true; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
    void add_response_header(std::string k, std::string v) override { bytes += k + ": " + v + "\n"; }
};
struct IdentityEngine : GenerationEngine {
    GenerationJob seen;
    std::vector<std::uint32_t> emitted{259}; // Hello
    GenerationStats generate(GenerationJob& job) override {
        seen = job;
        if (!job.session_id.empty()) throw std::runtime_error("unnamed request entered session path");
        job.on_tokens(emitted);
        GenerationStats stats; stats.prompt_tokens = job.prompt.size();
        stats.generated_tokens = emitted.size(); stats.cycles = 1;
        return stats;
    }
    std::uint64_t context_tokens() const override { return 4096; }
};
void unnamed_generation_identity(const std::shared_ptr<const BpeTokenizer>& tokenizer) {
    std::vector<GenerationJob> jobs;
    for (const bool enabled : {false,true}) {
        ApiConfig config; config.model_id = "baseline-model"; config.sessions_enabled = enabled;
        GenerationQueue queue; ApiService api(config,tokenizer,queue,4096,false);
        IdentityEngine engine;
        std::thread worker([&] { queue.run(engine); });
        HttpRequest request; request.method = "POST"; request.path = "/v1/chat/completions";
        request.body = R"({"messages":[{"role":"user","content":"Hello"}],"max_tokens":1,"temperature":0})";
        Reply response;
        try { api.handle(request,response); }
        catch (...) { queue.shutdown(); worker.join(); throw; }
        queue.shutdown(); worker.join();
        if (response.status != 200 || engine.seen.prompt.empty()) throw std::runtime_error("unnamed generation failed");
        jobs.push_back(std::move(engine.seen));
    }
    const auto& a = jobs[0]; const auto& b = jobs[1];
    if (a.prompt != b.prompt || a.max_tokens != b.max_tokens || a.stop_tokens != b.stop_tokens ||
        a.stable_prefix != b.stable_prefix || a.sampling.temperature != b.sampling.temperature ||
        a.ngram_draft != b.ngram_draft || a.draft_depth_max != b.draft_depth_max ||
        a.tool_draft != b.tool_draft || a.ngram_peers != b.ngram_peers)
        throw std::runtime_error("defaults changed unnamed generation scheduling inputs");
}
int main(int argc, char** argv) try {
    bool enabled = argc > 1 && std::string_view(argv[1]) == "--enabled";
    const bool default_on = argc > 1 && std::string_view(argv[1]) == "--defaults-on-no-session";
    if (default_on || (argc > 1 && std::string_view(argv[1]) == "--session-cache")) {
        namespace pd = strix::deepseek41::pair_defaults;
        pd::Probe p;
        p.file = p.dir = [](const std::string&) { return false; };
        p.list = [](const std::string&) { return std::vector<std::string>{}; };
        p.available_memory = [] { return 64ULL << 30U; };
        std::vector<std::string> args{"model.gguf", "--no-dspark"};
        if (!default_on) args.insert(args.end(), argv + 1, argv + argc);
        const auto r = pd::resolve(pd::App::serve,args,[](const char*) -> const char* { return nullptr; },p,{});
        enabled = r.session.enabled;
        if (default_on && enabled) throw std::runtime_error("unqualified defaults enabled cache");
        if (!default_on && enabled) throw std::runtime_error("OFF switch did not disable cache");
    }
    {
        std::map<std::string, Json> rows;
        session_inventory_live(rows, "first", 2, {}, 0, true);
        if (rows.size() != 1 || !rows.at("first").find("pending")->is_null()) throw std::runtime_error("first active inventory");
        session_inventory_live(rows, "first", 8, 17, 4, true);
        if (rows.size() != 1 || rows.at("first").find("consumed")->as_int() != 8 ||
            rows.at("first").find("pending")->as_int() != 17 || rows.at("first").find("state")->as_string() != "active" ||
            rows.at("first").find("durability")->as_string() != "none") throw std::runtime_error("live inventory overrides stale frontier");
        session_inventory_live(rows, "restored", 3, 17, 2, false);
        if (rows.at("restored").find("state")->as_string() != "parked") throw std::runtime_error("parked inventory");
    }
    ApiConfig config;
    config.model_id = "baseline-model";
    config.sessions_enabled = enabled;
    GenerationQueue queue;
    const auto tokenizer = tiny_tokenizer();
    ApiService api(config, tokenizer, queue, 4096, false);
    if (default_on) unnamed_generation_identity(tokenizer);
    for (const auto& path : {"/health", "/v1/models", "/tokenize", "/v1/completions", "/v1/messages", "/admin/session-cache"}) {
        HttpRequest req;
        req.method = std::string_view(path) == "/health" || std::string_view(path) == "/v1/models" ? "GET" : "POST";
        req.path = path; req.body = "{}";
        req.headers = {{"x-request-id", "baseline-fixed"}};
        Reply plain; api.handle(req, plain);
        for (const auto& id : {"valid_session", "../invalid session", ""}) {
            req.headers = {{"x-request-id", "baseline-fixed"}, {"x-session-id", id}};
            Reply named; api.handle(req, named);
            if (!enabled && named.bytes != plain.bytes) throw std::runtime_error("disabled header changed response");
            std::cout << path << " " << id << "\n" << named.bytes << "\n";
        }
    }
    namespace sc = strix::deepseek41::sched;
    for (const auto bytes : {0ULL, 1024ULL}) {
        const auto baseline = sc::serve_identity_text("baseline-ns", bytes, 2, "controller", "full");
        if (sc::session_serve_identity_text(baseline, 0, false) != baseline ||
            sc::session_serve_identity_text(baseline, 0, true) != baseline ||
            sc::session_serve_identity_text(baseline, 512, false) == baseline)
            throw std::runtime_error("hello identity regression");
        std::cout << "hello " << baseline << "\n";
    }
    if (sc::session_admin_route(false, "/v1/sessions") || !sc::session_admin_route(true, "/v1/sessions") || sc::session_admin_route(true, "/admin/session-cache"))
        throw std::runtime_error("session route enablement regression");
    // Keep the original dump unchanged: additional assertions never print response bytes.
    for (const auto& path : {"/v1/chat/completions", "/v1/completions"}) {
        HttpRequest req; req.method = "POST"; req.path = path; req.body = "{}";
        req.headers = {{"x-request-id", "baseline-fixed"}};
        Reply plain; api.handle(req, plain);
        for (const auto& field : {"123", "\"../bad\"", "\"\"", "null"}) {
            req.body = std::string("{\"session\":") + field + "}";
            Reply named; api.handle(req, named);
            if (!enabled && plain.bytes != named.bytes) throw std::runtime_error("disabled session field changed response");
        }
    }
    for (const auto& path : {"/v1/sessions", "/v1/cache/stats", "/v1/sessions/id/checkpoint", "/v1/sessions/id/park", "/v1/sessions/id/resume", "/v1/sessions/id"})
        if (sc::session_admin_route(false, path) || !sc::session_admin_route(true, path)) throw std::runtime_error("route gating");
    if (enabled) {
        for (const auto& op : {"checkpoint", "park", "resume"}) {
            const auto command = session_command("POST", std::string("/v1/sessions/test_id/") + op, "{\"durability\":\"disk\"}");
            if (command.find("id")->as_string() != "test_id" || command.find("op")->as_string() != op) throw std::runtime_error("session route translation");
        }
        if (session_command("POST", "/v1/sessions/id/checkpoint", "{\"durability\":\"ram\"}").find("op")->as_string() != "ram_checkpoint" ||
            session_command("GET", "/v1/sessions", "").find("op")->as_string() != "list" ||
            session_command("GET", "/v1/cache/stats", "").find("op")->as_string() != "stats" ||
            session_command("DELETE", "/v1/sessions/id", "").find("op")->as_string() != "delete") throw std::runtime_error("session commands");
        for (const auto& path : {"/admin/session-cache", "/v1/sessions/id/restore", "/v1/sessions/id/park/extra"}) {
            try { (void)session_command("POST", path, "{}"); throw std::runtime_error("accepted legacy/unknown route"); }
            catch (const SessionApiError& e) { if (e.status != 404) throw; }
        }
        for (const auto& body : {"{}", "{", "{\"durability\":1}", "{\"durability\":\"other\"}"}) {
            try { (void)session_command("POST", "/v1/sessions/id/checkpoint", body); throw std::runtime_error("accepted bad durability"); }
            catch (const SessionApiError& e) { if (e.status != 400) throw; }
        }
        for (const auto& item : {std::pair{"session_incompatible",409}, {"session_corrupt",422}, {"session_not_found",404}, {"session_budget",507}}) {
            const auto e = session_failure(item.first);
            if (e.status != item.second || e.code != item.first || !session_error(e.code,e.what()).find("error")->find("message")) throw std::runtime_error("session error contract");
        }
        HttpRequest req; req.method = "POST"; req.path = "/v1/chat/completions";
        req.body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"session\":\"../bad\"}";
        Reply reply; api.handle(req, reply); if (reply.status != 400) throw std::runtime_error("invalid session body id");
    }
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
