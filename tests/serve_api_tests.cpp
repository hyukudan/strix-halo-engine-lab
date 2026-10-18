// HTTP API contract tests with a scripted engine (no GPU): OpenAI chat/completions (plain and
// SSE), tool calls, stop strings, Anthropic messages (plain and events), errors and API keys,
// two concurrent clients through the FIFO queue, and cancellation when a client disconnects.
#include "strix/serve/api.hpp"
#include "strix/serve/http_server.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <iostream>
#include <netinet/in.h>
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
    explicit ScriptedEngine(std::shared_ptr<const BpeTokenizer> t) : tokenizer_(std::move(t)) {}
    GenerationStats generate(GenerationJob& job) override {
        const auto prompt = tokenizer_->decode(job.prompt);
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
            bool stop = false;
            for (std::size_t k = 0; k < step.size(); ++k)
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), step[k]) != job.stop_tokens.end()) { step.resize(k + 1U); stop = true; break; }
            if (stats.generated_tokens + step.size() >= job.max_tokens) { step.resize(job.max_tokens - stats.generated_tokens); }
            stats.generated_tokens += step.size();
            ++stats.cycles;
            calls.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (!job.on_tokens(step)) { stats.finish = "cancelled"; cancelled.fetch_add(1); return stats; }
            if (stop) { stats.finish = "stop"; return stats; }
            if (stats.generated_tokens >= job.max_tokens) { stats.finish = "length"; return stats; }
            at += n;
        }
        stats.finish = "length";
        return stats;
    }
    std::uint64_t context_tokens() const override { return 4096; }
    std::atomic<int> calls{0}, cancelled{0};
private:
    std::shared_ptr<const BpeTokenizer> tokenizer_;
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

int main() try {
    const auto tokenizer = tiny_tokenizer();
    ScriptedEngine engine(tokenizer);
    GenerationQueue queue;
    ApiConfig config;
    config.model_id = "test-model";
    config.stop_tokens = {tokenizer->find("<|im_end|>"), tokenizer->find("<|endoftext|>")};
    config.api_key = "secret";
    ApiService api(config, tokenizer, queue, engine.context_tokens(), false);
    HttpServerConfig http_config;
    http_config.port = 0;
    HttpServer server(http_config, [&](const HttpRequest& r, HttpResponseWriter& w) { api.handle(r, w); });
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
        const auto ids = parse_json(http(port, "POST", "/v1/completions", R"({"prompt":[72,101],"max_tokens":5})", auth).body);
        check(at(at(ids, "usage"), "prompt_tokens").as_int() == 2, "completions with token IDs " + ids.dump());
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
