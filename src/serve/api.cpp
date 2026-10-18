#include "strix/serve/api.hpp"

#include "strix/serve/chat_template.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <sstream>

namespace strix::serve {
namespace {

using Headers = std::vector<std::pair<std::string, std::string>>;

class ApiError : public std::runtime_error {
public:
    ApiError(int status, std::string type, const std::string& message, std::string param = {})
        : std::runtime_error(message), status_(status), type_(std::move(type)), param_(std::move(param)) {}
    int status() const { return status_; }
    const std::string& type() const { return type_; }
    const std::string& param() const { return param_; }
private:
    int status_;
    std::string type_, param_;
};

std::string random_id(std::string_view prefix) {
    static std::mutex mutex;
    static std::mt19937_64 rng{std::random_device{}()};
    const std::lock_guard lock(mutex);
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%016llx%08llx", static_cast<unsigned long long>(rng()),
        static_cast<unsigned long long>(rng() & 0xffffffffULL));
    return std::string(prefix) + buffer;
}
std::int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
// Valid UTF-8 for JSON output: invalid bytes become U+FFFD.
std::string sanitize_utf8(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::size_t n = b0 < 0x80U ? 1U : (b0 >= 0xC2U && b0 <= 0xDFU) ? 2U : (b0 >= 0xE0U && b0 <= 0xEFU) ? 3U : (b0 >= 0xF0U && b0 <= 0xF4U) ? 4U : 0U;
        bool valid = n && i + n <= text.size();
        for (std::size_t k = 1; valid && k < n; ++k) valid = (static_cast<unsigned char>(text[i + k]) & 0xC0U) == 0x80U;
        if (valid && n == 3U) {
            const auto b1 = static_cast<unsigned char>(text[i + 1U]);
            if ((b0 == 0xE0U && b1 < 0xA0U) || (b0 == 0xEDU && b1 >= 0xA0U)) valid = false;
        }
        if (valid && n == 4U) {
            const auto b1 = static_cast<unsigned char>(text[i + 1U]);
            if ((b0 == 0xF0U && b1 < 0x90U) || (b0 == 0xF4U && b1 >= 0x90U)) valid = false;
        }
        if (valid) { out.append(text.substr(i, n)); i += n; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

// Holds back the tail that could still become a stop string; reports the first match.
class StopFilter {
public:
    explicit StopFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {
        for (const auto& s : stops_) hold_ = std::max(hold_, s.empty() ? 0U : s.size() - 1U);
    }
    std::string push(std::string_view text) {
        if (matched_) return {};
        buffer_.append(text);
        std::size_t first = std::string::npos;
        for (const auto& s : stops_) {
            if (s.empty()) continue;
            const auto at = buffer_.find(s);
            if (at != std::string::npos && at < first) { first = at; match_ = s; }
        }
        if (first != std::string::npos) {
            matched_ = true;
            auto out = buffer_.substr(0, first);
            buffer_.clear();
            return out;
        }
        if (buffer_.size() <= hold_) return {};
        // Never cut a UTF-8 sequence: move the cut back to a character start.
        auto cut = buffer_.size() - hold_;
        while (cut > 0 && cut < buffer_.size() && (static_cast<unsigned char>(buffer_[cut]) & 0xC0U) == 0x80U) --cut;
        auto out = buffer_.substr(0, cut);
        buffer_.erase(0, cut);
        return out;
    }
    std::string finish() { auto out = matched_ ? std::string{} : buffer_; buffer_.clear(); return out; }
    bool matched() const { return matched_; }
    const std::string& match() const { return match_; }
private:
    std::vector<std::string> stops_;
    std::size_t hold_ = 0;
    std::string buffer_, match_;
    bool matched_ = false;
};

// Splits the assistant text stream into reasoning and content deltas; everything from the first
// <tool_call> on is kept for the final tool-call parse.
class ChatSplitter {
public:
    explicit ChatSplitter(bool thinking) : phase_(thinking ? Phase::reasoning : Phase::content) {}
    struct Delta { bool reasoning; std::string text; };
    std::vector<Delta> push(std::string_view text) {
        std::vector<Delta> out;
        buffer_.append(text);
        for (bool progress = true; progress;) {
            progress = false;
            if (phase_ == Phase::reasoning) {
                if (reasoning_leading_) {
                    const auto first = buffer_.find_first_not_of(" \t\r\n");
                    if (first == std::string::npos) { buffer_.clear(); break; }
                    buffer_.erase(0, first);
                    reasoning_leading_ = false;
                }
                const auto end = buffer_.find("</think>");
                if (end != std::string::npos) {
                    auto before = buffer_.substr(0, end);
                    const auto last = before.find_last_not_of(" \t\r\n");
                    before.erase(last == std::string::npos ? 0U : last + 1U);
                    emit(out, true, before);
                    buffer_.erase(0, end + 8U);
                    phase_ = Phase::content;
                    leading_ = true;
                    progress = true;
                } else hold_tail(out, true, 7U);
            } else if (phase_ == Phase::content) {
                if (leading_) {
                    const auto first = buffer_.find_first_not_of(" \t\r\n");
                    if (first == std::string::npos) { buffer_.clear(); break; }
                    buffer_.erase(0, first);
                    leading_ = false;
                }
                const auto call = buffer_.find("<tool_call>");
                if (call != std::string::npos) {
                    auto before = buffer_.substr(0, call);
                    const auto last = before.find_last_not_of(" \t\r\n");
                    before.erase(last == std::string::npos ? 0U : last + 1U);
                    emit(out, false, before);
                    tools_ = buffer_.substr(call);
                    buffer_.clear();
                    phase_ = Phase::tools;
                } else hold_tail(out, false, 10U);
            } else {
                tools_.append(buffer_);
                buffer_.clear();
            }
        }
        return out;
    }
    std::vector<Delta> finish() {
        std::vector<Delta> out;
        if (phase_ == Phase::reasoning) {
            const auto last = buffer_.find_last_not_of(" \t\r\n");
            emit(out, true, last == std::string::npos ? std::string{} : buffer_.substr(0, last + 1U));
            buffer_.clear();
        }
        else if (phase_ == Phase::content) {
            const auto last = buffer_.find_last_not_of(" \t\r\n");
            emit(out, false, last == std::string::npos ? std::string{} : buffer_.substr(0, last + 1U));
            buffer_.clear();
        }
        return out;
    }
    bool saw_tools() const { return phase_ == Phase::tools; }
private:
    enum class Phase { reasoning, content, tools };
    void emit(std::vector<Delta>& out, bool reasoning, const std::string& text) {
        if (!text.empty()) out.push_back({reasoning, text});
    }
    void emit_cut(std::vector<Delta>& out, bool reasoning, std::size_t cut) {
        while (cut > 0 && cut < buffer_.size() && (static_cast<unsigned char>(buffer_[cut]) & 0xC0U) == 0x80U) --cut;
        if (!cut) return;
        emit(out, reasoning, buffer_.substr(0, cut));
        buffer_.erase(0, cut);
    }
    // Emits all but a possible partial marker (`hold` bytes) and the trailing whitespace, which is
    // stripped if the block ends there (as the final parse strips it).
    void hold_tail(std::vector<Delta>& out, bool reasoning, std::size_t hold) {
        auto keep = std::min<std::size_t>(buffer_.size(), hold);
        const auto last = buffer_.find_last_not_of(" \t\r\n");
        if (last == std::string::npos) keep = buffer_.size();
        else keep = std::max(keep, buffer_.size() - last - 1U);
        emit_cut(out, reasoning, buffer_.size() - keep);
    }
    Phase phase_;
    std::string buffer_, tools_;
    bool leading_ = false, reasoning_leading_ = true;
};

struct RecentRequest {
    std::string id, route, status;
    std::int64_t time = 0;
    std::uint64_t prompt = 0, cached = 0, output = 0, cycles = 0, accepted = 0, proposed = 0;
    double queue_ms = 0, prefill_ms = 0, decode_ms = 0;
};

std::vector<std::string> stop_strings(const Json* stop) {
    std::vector<std::string> out;
    if (!stop || stop->is_null()) return out;
    if (stop->is_string()) { out.push_back(stop->as_string()); return out; }
    if (!stop->is_array()) throw ApiError(400, "invalid_request_error", "stop must be a string or an array of strings", "stop");
    for (const auto& s : stop->as_array()) {
        if (!s.is_string()) throw ApiError(400, "invalid_request_error", "stop must be a string or an array of strings", "stop");
        if (!s.as_string().empty()) out.push_back(s.as_string());
    }
    if (out.size() > 16U) throw ApiError(400, "invalid_request_error", "at most 16 stop strings", "stop");
    return out;
}

double number_or(const Json& body, std::string_view key, double fallback) {
    const auto* value = body.find(key);
    if (!value || value->is_null()) return fallback;
    if (!value->is_number()) throw ApiError(400, "invalid_request_error", std::string(key) + " must be a number", std::string(key));
    return value->as_double();
}
bool bool_or(const Json& body, std::string_view key, bool fallback) {
    const auto* value = body.find(key);
    if (!value || value->is_null()) return fallback;
    if (!value->is_bool()) throw ApiError(400, "invalid_request_error", std::string(key) + " must be a boolean", std::string(key));
    return value->as_bool();
}

}  // namespace

struct ApiService::Impl {
    ApiConfig config;
    std::shared_ptr<const BpeTokenizer> tokenizer;
    GenerationQueue& queue;
    std::uint64_t context_tokens;
    bool engine_samples;
    std::mutex mutex;
    std::deque<RecentRequest> recent;
    std::string state = "idle";
    std::uint64_t requests_total = 0, requests_failed = 0;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

    Impl(ApiConfig c, std::shared_ptr<const BpeTokenizer> t, GenerationQueue& q, std::uint64_t context, bool samples)
        : config(std::move(c)), tokenizer(std::move(t)), queue(q), context_tokens(context), engine_samples(samples) {}

    bool is_stop_token(std::uint32_t id) const {
        return std::find(config.stop_tokens.begin(), config.stop_tokens.end(), id) != config.stop_tokens.end();
    }
    void record(RecentRequest entry) {
        const std::lock_guard lock(mutex);
        ++requests_total;
        if (entry.status == "error") ++requests_failed;
        recent.push_front(std::move(entry));
        while (recent.size() > config.recent_requests) recent.pop_back();
    }

    SamplingParameters sampling(const Json& body, std::string& note) const {
        SamplingParameters s;
        s.temperature = number_or(body, "temperature", 0.0);
        s.top_p = number_or(body, "top_p", 1.0);
        const auto top_k = number_or(body, "top_k", 0.0);
        s.top_k = top_k > 0.0 ? static_cast<std::uint32_t>(top_k) : 0U;
        if (const auto* seed = body.find("seed"); seed && seed->is_number()) { s.seed = static_cast<std::uint64_t>(seed->as_int()); s.seeded = true; }
        if (s.temperature < 0.0 || s.temperature > 2.0) throw ApiError(400, "invalid_request_error", "temperature must be in [0, 2]", "temperature");
        if (s.temperature > 0.0 && !engine_samples) {
            if (config.sampling_policy == "reject")
                throw ApiError(400, "invalid_request_error", "this server decodes greedy only; send temperature 0", "temperature");
            note = "temperature ignored: this engine decodes greedy (exact)";
            s = SamplingParameters{};
        }
        return s;
    }
    std::uint32_t max_tokens(const Json& body, std::size_t prompt) const {
        double requested = config.default_max_tokens;
        for (const char* key : {"max_completion_tokens", "max_tokens"})
            if (const auto* v = body.find(key); v && !v->is_null()) {
                if (!v->is_number() || v->as_double() < 1.0) throw ApiError(400, "invalid_request_error", std::string(key) + " must be a positive integer", key);
                requested = v->as_double();
                break;
            }
        if (prompt >= context_tokens)
            throw ApiError(400, "invalid_request_error", "prompt of " + std::to_string(prompt) + " tokens does not fit the context of " +
                std::to_string(context_tokens) + " tokens", "messages");
        const auto room = static_cast<double>(context_tokens - prompt);
        return static_cast<std::uint32_t>(std::max(1.0, std::min(requested, room)));
    }
    void authorize(const HttpRequest& request) const {
        if (config.api_key.empty()) return;
        const auto bearer = request.header("authorization");
        if (bearer == "Bearer " + config.api_key || request.header("x-api-key") == config.api_key) return;
        throw ApiError(401, "authentication_error", "invalid or missing API key");
    }

    // ---- chat ----
    struct ChatPrompt { std::vector<std::uint32_t> tokens; bool thinking; Json tools; };
    ChatPrompt chat_prompt(const Json& body) const {
        const auto* messages = body.find("messages");
        if (!messages || !messages->is_array() || messages->size() == 0U)
            throw ApiError(400, "invalid_request_error", "messages must be a non-empty array", "messages");
        ChatTemplateInput input;
        Json normalized = Json::array();
        for (const auto& message : messages->as_array()) {
            if (!message.is_object()) throw ApiError(400, "invalid_request_error", "every message must be an object", "messages");
            Json copy = message;
            if (!copy.find("reasoning_content")) if (const auto* r = copy.find("reasoning"); r && r->is_string()) copy.set("reasoning_content", *r);
            normalized.push(std::move(copy));
        }
        input.messages = normalized;
        const auto* tool_choice = body.find("tool_choice");
        const bool tools_off = tool_choice && tool_choice->is_string() && tool_choice->as_string() == "none";
        if (const auto* tools = body.find("tools"); tools && tools->is_array() && !tools_off) input.tools = *tools;
        const auto* kwargs = body.find("chat_template_kwargs");
        const auto kwarg = [&](std::string_view key) -> const Json* {
            if (kwargs && kwargs->is_object()) if (const auto* v = kwargs->find(key)) return v;
            return body.find(key);
        };
        if (const auto* v = kwarg("enable_thinking"); v && v->is_bool()) input.enable_thinking = v->as_bool();
        else if (!config.default_thinking) input.enable_thinking = false;
        if (const auto* v = kwarg("preserve_thinking"); v && v->is_bool()) input.preserve_thinking = v->as_bool();
        if (const auto* v = kwarg("reasoning_effort"); v && v->is_string()) {
            auto effort = v->as_string();
            if (effort == "minimal") effort = "low";
            if (effort == "none") input.enable_thinking = false;
            else input.reasoning_effort = effort;
        }
        std::string text;
        try { text = render_chat_template(input); }
        catch (const ChatTemplateError& error) { throw ApiError(400, "invalid_request_error", error.what(), "messages"); }
        ChatPrompt out;
        out.tokens = tokenizer->encode(text, true);
        out.thinking = !input.enable_thinking || *input.enable_thinking;
        out.tools = input.tools;
        return out;
    }

    // Drives one ticket: decodes tokens to text, applies stop strings, hands text to `on_text`.
    struct Outcome { std::string text; GenerationStats stats; std::string finish; std::string matched; std::string error; std::uint64_t output_tokens = 0; };
    Outcome drive(const std::shared_ptr<GenerationTicket>& ticket, const std::vector<std::string>& stops,
        const std::function<bool(const std::string&)>& on_text, const std::function<bool()>& client_alive) {
        Outcome out;
        StopFilter filter(stops);
        Utf8StreamDecoder utf8;
        bool stopped = false;
        for (bool done = false; !done;) {
            const auto tokens = ticket->wait(std::chrono::milliseconds(200), done);
            for (const auto id : tokens) {
                if (stopped) break;
                ++out.output_tokens;
                if (is_stop_token(id)) { stopped = true; break; }
                const auto text = filter.push(utf8.push(tokenizer->token_bytes(id)));
                if (!text.empty()) {
                    out.text += text;
                    if (on_text && !on_text(text)) { ticket->cancel(); stopped = true; }
                }
                if (filter.matched()) { out.matched = filter.match(); ticket->cancel(); stopped = true; }
            }
            if (client_alive && !client_alive()) ticket->cancel();
        }
        if (!out.matched.size()) {
            const auto tail = filter.push(utf8.flush()) + filter.finish();
            if (!tail.empty()) { out.text += tail; if (on_text) on_text(tail); }
        }
        out.stats = ticket->stats();
        out.error = ticket->error();
        out.finish = !out.matched.empty() ? "stop" : out.stats.finish == "stop" ? "stop" : out.stats.finish == "cancelled" ? "cancelled" : "length";
        return out;
    }

    std::shared_ptr<GenerationTicket> submit(std::vector<std::uint32_t> prompt, std::uint32_t max_tokens, const SamplingParameters& sampling) {
        GenerationJob job;
        job.prompt = std::move(prompt);
        job.max_tokens = max_tokens;
        job.stop_tokens = config.stop_tokens;
        job.sampling = sampling;
        auto ticket = queue.submit(std::move(job));
        if (!ticket) throw ApiError(429, "rate_limit_error", "the request queue is full");
        return ticket;
    }

    static Json usage_json(const Outcome& o) {
        Json usage = Json::object();
        usage.set("prompt_tokens", Json(o.stats.prompt_tokens));
        usage.set("completion_tokens", Json(o.output_tokens));
        usage.set("total_tokens", Json(o.stats.prompt_tokens + o.output_tokens));
        Json details = Json::object();
        details.set("cached_tokens", Json(o.stats.cached_tokens));
        usage.set("prompt_tokens_details", std::move(details));
        return usage;
    }
    static Json engine_json(const Outcome& o, const std::string& note) {
        Json e = Json::object();
        e.set("sampling", Json(o.stats.sampling));
        if (!note.empty()) e.set("note", Json(note));
        e.set("queue_ms", Json(o.stats.queue_ms));
        e.set("prefill_ms", Json(o.stats.prefill_ms));
        e.set("decode_ms", Json(o.stats.decode_ms));
        e.set("decode_tokens_per_second", Json(o.stats.decode_ms > 0 ? 1000.0 * static_cast<double>(o.stats.generated_tokens) / o.stats.decode_ms : 0.0));
        e.set("mtp_cycles", Json(o.stats.cycles));
        e.set("mtp_accepted", Json(o.stats.accepted));
        e.set("mtp_proposed", Json(o.stats.proposed));
        return e;
    }
    static Json tool_calls_json(const Json& calls) {
        Json out = Json::array();
        for (const auto& call : calls.as_array()) {
            Json function = Json::object();
            function.set("name", *call.find("name"));
            function.set("arguments", Json(call.find("arguments")->dump()));
            Json entry = Json::object();
            entry.set("id", Json(random_id("call_")));
            entry.set("type", Json("function"));
            entry.set("function", std::move(function));
            out.push(std::move(entry));
        }
        return out;
    }

    void chat_completions(const HttpRequest&, HttpResponseWriter& response, const Json& body) {
        const auto prompt = chat_prompt(body);
        if (const auto* n = body.find("n"); n && n->is_number() && n->as_int() != 1)
            throw ApiError(400, "invalid_request_error", "only n = 1 is supported", "n");
        std::string note;
        const auto sampling_params = sampling(body, note);
        const auto stops = stop_strings(body.find("stop"));
        const bool stream = bool_or(body, "stream", false);
        bool include_usage = false;
        if (const auto* options = body.find("stream_options"); options && options->is_object()) include_usage = bool_or(*options, "include_usage", false);
        const auto limit = max_tokens(body, prompt.tokens.size());
        auto ticket = submit(prompt.tokens, limit, sampling_params);
        const auto id = random_id("chatcmpl-");
        const auto created = unix_now();
        RecentRequest entry{id, "chat.completions", "ok", created};
        if (!stream) {
            auto outcome = drive(ticket, stops, {}, [&] { return !response.client_gone(); });
            if (!outcome.error.empty()) throw ApiError(500, "server_error", outcome.error);
            const auto parsed = parse_assistant_output(outcome.text, prompt.thinking, prompt.tools);
            Json message = Json::object();
            message.set("role", Json("assistant"));
            message.set("content", parsed.content.empty() && parsed.tool_calls.size() ? Json() : Json(sanitize_utf8(parsed.content)));
            if (prompt.thinking) message.set("reasoning_content", Json(sanitize_utf8(parsed.reasoning)));
            std::string finish = outcome.finish == "cancelled" ? "stop" : outcome.finish;
            if (parsed.tool_calls.size()) { message.set("tool_calls", tool_calls_json(parsed.tool_calls)); if (finish == "stop") finish = "tool_calls"; }
            Json choice = Json::object();
            choice.set("index", Json(0));
            choice.set("message", std::move(message));
            choice.set("logprobs", Json());
            choice.set("finish_reason", Json(finish));
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("chat.completion")); out.set("created", Json(created));
            out.set("model", Json(config.model_id));
            Json choices = Json::array(); choices.push(std::move(choice));
            out.set("choices", std::move(choices));
            out.set("usage", usage_json(outcome));
            out.set("system_fingerprint", Json("strix-" + outcome.stats.sampling));
            out.set("strix", engine_json(outcome, note));
            response.send(200, "application/json", out.dump());
            finish_entry(entry, outcome);
            return;
        }
        if (!response.begin_stream(200, "text/event-stream")) { ticket->cancel(); return; }
        const auto chunk = [&](Json delta, const Json& finish) {
            Json choice = Json::object();
            choice.set("index", Json(0));
            choice.set("delta", std::move(delta));
            choice.set("logprobs", Json());
            choice.set("finish_reason", finish);
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("chat.completion.chunk")); out.set("created", Json(created));
            out.set("model", Json(config.model_id));
            Json choices = Json::array(); choices.push(std::move(choice));
            out.set("choices", std::move(choices));
            return response.write("data: " + out.dump() + "\n\n");
        };
        Json first = Json::object();
        first.set("role", Json("assistant"));
        first.set("content", Json(""));
        chunk(std::move(first), Json());
        ChatSplitter splitter(prompt.thinking);
        const auto send_deltas = [&](const std::vector<ChatSplitter::Delta>& deltas) {
            for (const auto& d : deltas) {
                Json delta = Json::object();
                delta.set(d.reasoning ? "reasoning_content" : "content", Json(sanitize_utf8(d.text)));
                if (!chunk(std::move(delta), Json())) return false;
            }
            return true;
        };
        auto outcome = drive(ticket, stops, [&](const std::string& text) { return send_deltas(splitter.push(text)); },
            [&] { return !response.client_gone(); });
        send_deltas(splitter.finish());
        std::string finish = outcome.finish == "cancelled" ? "stop" : outcome.finish;
        if (!outcome.error.empty()) {
            Json error = Json::object();
            Json detail = Json::object(); detail.set("message", Json(outcome.error)); detail.set("type", Json("server_error"));
            error.set("error", std::move(detail));
            response.write("data: " + error.dump() + "\n\n");
        } else if (splitter.saw_tools()) {
            const auto parsed = parse_assistant_output(outcome.text, prompt.thinking, prompt.tools);
            if (parsed.tool_calls.size()) {
                auto calls = tool_calls_json(parsed.tool_calls);
                std::int64_t index = 0;
                for (auto& call : calls.as_array()) call.set("index", Json(index++));
                Json delta = Json::object();
                delta.set("tool_calls", std::move(calls));
                chunk(std::move(delta), Json());
                if (finish == "stop") finish = "tool_calls";
            }
        }
        chunk(Json::object(), Json(finish));
        if (include_usage) {
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("chat.completion.chunk")); out.set("created", Json(created));
            out.set("model", Json(config.model_id)); out.set("choices", Json::array());
            out.set("usage", usage_json(outcome));
            out.set("strix", engine_json(outcome, note));
            response.write("data: " + out.dump() + "\n\n");
        }
        response.write("data: [DONE]\n\n");
        response.end_stream();
        finish_entry(entry, outcome);
    }

    // ---- completions ----
    void completions(const HttpRequest&, HttpResponseWriter& response, const Json& body) {
        const auto* prompt_value = body.find("prompt");
        if (!prompt_value) throw ApiError(400, "invalid_request_error", "prompt is required", "prompt");
        std::vector<std::uint32_t> prompt;
        const auto take_ids = [&](const Json& array) {
            for (const auto& id : array.as_array()) {
                if (!id.is_number() || id.as_int() < 0 || static_cast<std::uint64_t>(id.as_int()) >= tokenizer->vocab_size())
                    throw ApiError(400, "invalid_request_error", "prompt token IDs must be in the vocabulary", "prompt");
                prompt.push_back(static_cast<std::uint32_t>(id.as_int()));
            }
        };
        if (prompt_value->is_string()) prompt = tokenizer->encode(prompt_value->as_string(), true);
        else if (prompt_value->is_array() && prompt_value->size() == 1U && prompt_value->as_array()[0].is_string())
            prompt = tokenizer->encode(prompt_value->as_array()[0].as_string(), true);
        else if (prompt_value->is_array() && prompt_value->size() && prompt_value->as_array()[0].is_number()) take_ids(*prompt_value);
        else throw ApiError(400, "invalid_request_error", "prompt must be one string or one array of token IDs", "prompt");
        if (prompt.empty()) throw ApiError(400, "invalid_request_error", "prompt is empty", "prompt");
        if (bool_or(body, "echo", false)) throw ApiError(400, "invalid_request_error", "echo is not supported", "echo");
        std::string note;
        const auto sampling_params = sampling(body, note);
        const auto stops = stop_strings(body.find("stop"));
        const bool stream = bool_or(body, "stream", false);
        const auto limit = max_tokens(body, prompt.size());
        auto ticket = submit(prompt, limit, sampling_params);
        const auto id = random_id("cmpl-");
        const auto created = unix_now();
        RecentRequest entry{id, "completions", "ok", created};
        const auto chunk = [&](const std::string& text, const Json& finish, bool final_usage, const Outcome* outcome) {
            Json choice = Json::object();
            choice.set("index", Json(0)); choice.set("text", Json(sanitize_utf8(text)));
            choice.set("logprobs", Json()); choice.set("finish_reason", finish);
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("text_completion")); out.set("created", Json(created));
            out.set("model", Json(config.model_id));
            Json choices = Json::array(); choices.push(std::move(choice));
            out.set("choices", std::move(choices));
            if (final_usage && outcome) { out.set("usage", usage_json(*outcome)); out.set("strix", engine_json(*outcome, note)); }
            return out;
        };
        if (!stream) {
            auto outcome = drive(ticket, stops, {}, [&] { return !response.client_gone(); });
            if (!outcome.error.empty()) throw ApiError(500, "server_error", outcome.error);
            response.send(200, "application/json", chunk(outcome.text, Json(outcome.finish == "cancelled" ? "stop" : outcome.finish), true, &outcome).dump());
            finish_entry(entry, outcome);
            return;
        }
        if (!response.begin_stream(200, "text/event-stream")) { ticket->cancel(); return; }
        auto outcome = drive(ticket, stops, [&](const std::string& text) {
            return response.write("data: " + chunk(text, Json(), false, nullptr).dump() + "\n\n"); }, [&] { return !response.client_gone(); });
        response.write("data: " + chunk("", Json(outcome.finish == "cancelled" ? "stop" : outcome.finish), true, &outcome).dump() + "\n\n");
        response.write("data: [DONE]\n\n");
        response.end_stream();
        finish_entry(entry, outcome);
    }

    // ---- Anthropic messages ----
    Json anthropic_to_chat(const Json& body, bool& thinking) const {
        Json messages = Json::array();
        if (const auto* system = body.find("system"); system && !system->is_null()) {
            std::string text;
            if (system->is_string()) text = system->as_string();
            else if (system->is_array()) for (const auto& block : system->as_array()) if (const auto* t = block.find("text"); t && t->is_string()) text += (text.empty() ? "" : "\n") + t->as_string();
            Json m = Json::object(); m.set("role", Json("system")); m.set("content", Json(text));
            messages.push(std::move(m));
        }
        const auto* input = body.find("messages");
        if (!input || !input->is_array() || input->size() == 0U) throw ApiError(400, "invalid_request_error", "messages must be a non-empty array", "messages");
        for (const auto& message : input->as_array()) {
            const auto* role = message.find("role");
            if (!role || !role->is_string()) throw ApiError(400, "invalid_request_error", "every message needs a role", "messages");
            const auto* content = message.find("content");
            if (content && content->is_string()) {
                Json m = Json::object(); m.set("role", *role); m.set("content", *content); messages.push(std::move(m));
                continue;
            }
            if (!content || !content->is_array()) throw ApiError(400, "invalid_request_error", "message content must be a string or an array", "messages");
            std::string text, reasoning;
            Json calls = Json::array();
            for (const auto& block : content->as_array()) {
                const auto* type = block.find("type");
                const auto kind = type && type->is_string() ? type->as_string() : std::string{};
                if (kind == "text") { if (const auto* t = block.find("text"); t && t->is_string()) text += t->as_string(); }
                else if (kind == "thinking") { if (const auto* t = block.find("thinking"); t && t->is_string()) reasoning += t->as_string(); }
                else if (kind == "tool_use") {
                    Json function = Json::object();
                    function.set("name", block.find("name") ? *block.find("name") : Json(""));
                    function.set("arguments", block.find("input") ? *block.find("input") : Json::object());
                    Json call = Json::object(); call.set("type", Json("function")); call.set("function", std::move(function));
                    calls.push(std::move(call));
                } else if (kind == "tool_result") {
                    std::string result;
                    if (const auto* c = block.find("content")) {
                        if (c->is_string()) result = c->as_string();
                        else if (c->is_array()) for (const auto& part : c->as_array()) if (const auto* t = part.find("text"); t && t->is_string()) result += t->as_string();
                    }
                    if (!text.empty()) { Json m = Json::object(); m.set("role", *role); m.set("content", Json(text)); messages.push(std::move(m)); text.clear(); }
                    Json m = Json::object(); m.set("role", Json("tool")); m.set("content", Json(result));
                    messages.push(std::move(m));
                } else if (kind == "redacted_thinking") {
                } else throw ApiError(400, "invalid_request_error", "unsupported content block type " + kind, "messages");
            }
            if (!text.empty() || calls.size() || !reasoning.empty() || role->as_string() == "assistant") {
                Json m = Json::object(); m.set("role", *role); m.set("content", Json(text));
                if (!reasoning.empty()) m.set("reasoning_content", Json(reasoning));
                if (calls.size()) m.set("tool_calls", std::move(calls));
                messages.push(std::move(m));
            }
        }
        Json chat = Json::object();
        chat.set("messages", std::move(messages));
        if (const auto* tools = body.find("tools"); tools && tools->is_array() && tools->size()) {
            Json converted = Json::array();
            for (const auto& tool : tools->as_array()) {
                Json function = Json::object();
                if (const auto* n = tool.find("name")) function.set("name", *n);
                if (const auto* d = tool.find("description")) function.set("description", *d);
                if (const auto* s = tool.find("input_schema")) function.set("parameters", *s);
                Json t = Json::object(); t.set("type", Json("function")); t.set("function", std::move(function));
                converted.push(std::move(t));
            }
            chat.set("tools", std::move(converted));
        }
        thinking = config.default_thinking;
        if (const auto* t = body.find("thinking"); t && t->is_object())
            if (const auto* type = t->find("type"); type && type->is_string()) thinking = type->as_string() == "enabled";
        chat.set("enable_thinking", Json(thinking));
        return chat;
    }

    void messages(const HttpRequest&, HttpResponseWriter& response, const Json& body) {
        bool thinking = false;
        const auto chat = anthropic_to_chat(body, thinking);
        const auto prompt = chat_prompt(chat);
        std::string note;
        const auto sampling_params = sampling(body, note);
        const auto stops = stop_strings(body.find("stop_sequences"));
        const bool stream = bool_or(body, "stream", false);
        const auto* max = body.find("max_tokens");
        if (!max || !max->is_number()) throw ApiError(400, "invalid_request_error", "max_tokens is required", "max_tokens");
        const auto limit = max_tokens(body, prompt.tokens.size());
        auto ticket = submit(prompt.tokens, limit, sampling_params);
        const auto id = random_id("msg_");
        RecentRequest entry{id, "messages", "ok", unix_now()};
        const auto stop_reason = [&](const Outcome& o, bool tools) {
            if (!o.matched.empty()) return std::string("stop_sequence");
            if (tools) return std::string("tool_use");
            return o.finish == "length" ? std::string("max_tokens") : std::string("end_turn");
        };
        const auto usage = [&](const Outcome& o) {
            Json u = Json::object();
            u.set("input_tokens", Json(o.stats.prompt_tokens - std::min(o.stats.prompt_tokens, o.stats.cached_tokens)));
            u.set("cache_read_input_tokens", Json(o.stats.cached_tokens));
            u.set("cache_creation_input_tokens", Json(0));
            u.set("output_tokens", Json(o.output_tokens));
            return u;
        };
        if (!stream) {
            auto outcome = drive(ticket, stops, {}, [&] { return !response.client_gone(); });
            if (!outcome.error.empty()) throw ApiError(500, "api_error", outcome.error);
            const auto parsed = parse_assistant_output(outcome.text, prompt.thinking, prompt.tools);
            Json content = Json::array();
            if (prompt.thinking && !parsed.reasoning.empty()) {
                Json b = Json::object(); b.set("type", Json("thinking")); b.set("thinking", Json(sanitize_utf8(parsed.reasoning))); b.set("signature", Json(""));
                content.push(std::move(b));
            }
            if (!parsed.content.empty()) {
                Json b = Json::object(); b.set("type", Json("text")); b.set("text", Json(sanitize_utf8(parsed.content)));
                content.push(std::move(b));
            }
            for (const auto& call : parsed.tool_calls.as_array()) {
                Json b = Json::object(); b.set("type", Json("tool_use")); b.set("id", Json(random_id("toolu_")));
                b.set("name", *call.find("name")); b.set("input", *call.find("arguments"));
                content.push(std::move(b));
            }
            Json out = Json::object();
            out.set("id", Json(id)); out.set("type", Json("message")); out.set("role", Json("assistant"));
            out.set("model", Json(config.model_id)); out.set("content", std::move(content));
            out.set("stop_reason", Json(stop_reason(outcome, parsed.tool_calls.size() > 0U)));
            out.set("stop_sequence", outcome.matched.empty() ? Json() : Json(outcome.matched));
            out.set("usage", usage(outcome));
            out.set("strix", engine_json(outcome, note));
            response.send(200, "application/json", out.dump());
            finish_entry(entry, outcome);
            return;
        }
        if (!response.begin_stream(200, "text/event-stream")) { ticket->cancel(); return; }
        const auto event = [&](const char* name, const Json& data) {
            return response.write(std::string("event: ") + name + "\ndata: " + data.dump() + "\n\n");
        };
        {
            Json message = Json::object();
            message.set("id", Json(id)); message.set("type", Json("message")); message.set("role", Json("assistant"));
            message.set("model", Json(config.model_id)); message.set("content", Json::array());
            message.set("stop_reason", Json()); message.set("stop_sequence", Json());
            Json u = Json::object(); u.set("input_tokens", Json(prompt.tokens.size())); u.set("output_tokens", Json(0));
            message.set("usage", std::move(u));
            Json start = Json::object(); start.set("type", Json("message_start")); start.set("message", std::move(message));
            event("message_start", start);
        }
        std::int64_t index = -1;
        std::string open_kind;
        const auto close_block = [&] {
            if (open_kind.empty()) return;
            Json stop = Json::object(); stop.set("type", Json("content_block_stop")); stop.set("index", Json(index));
            event("content_block_stop", stop);
            open_kind.clear();
        };
        const auto open_block = [&](const std::string& kind, Json block) {
            close_block();
            ++index;
            open_kind = kind;
            Json start = Json::object(); start.set("type", Json("content_block_start")); start.set("index", Json(index));
            start.set("content_block", std::move(block));
            event("content_block_start", start);
        };
        ChatSplitter splitter(prompt.thinking);
        const auto send_deltas = [&](const std::vector<ChatSplitter::Delta>& deltas) {
            for (const auto& d : deltas) {
                const std::string kind = d.reasoning ? "thinking" : "text";
                if (open_kind != kind) {
                    Json block = Json::object(); block.set("type", Json(kind));
                    if (d.reasoning) { block.set("thinking", Json("")); block.set("signature", Json("")); } else block.set("text", Json(""));
                    open_block(kind, std::move(block));
                }
                Json delta = Json::object();
                delta.set("type", Json(d.reasoning ? "thinking_delta" : "text_delta"));
                delta.set(d.reasoning ? "thinking" : "text", Json(sanitize_utf8(d.text)));
                Json e = Json::object(); e.set("type", Json("content_block_delta")); e.set("index", Json(index)); e.set("delta", std::move(delta));
                if (!event("content_block_delta", e)) return false;
            }
            return true;
        };
        auto outcome = drive(ticket, stops, [&](const std::string& text) { return send_deltas(splitter.push(text)); },
            [&] { return !response.client_gone(); });
        send_deltas(splitter.finish());
        bool tools = false;
        if (splitter.saw_tools()) {
            const auto parsed = parse_assistant_output(outcome.text, prompt.thinking, prompt.tools);
            for (const auto& call : parsed.tool_calls.as_array()) {
                tools = true;
                Json block = Json::object(); block.set("type", Json("tool_use")); block.set("id", Json(random_id("toolu_")));
                block.set("name", *call.find("name")); block.set("input", Json::object());
                open_block("tool_use", std::move(block));
                Json delta = Json::object(); delta.set("type", Json("input_json_delta")); delta.set("partial_json", Json(call.find("arguments")->dump()));
                Json e = Json::object(); e.set("type", Json("content_block_delta")); e.set("index", Json(index)); e.set("delta", std::move(delta));
                event("content_block_delta", e);
            }
        }
        close_block();
        Json delta = Json::object();
        delta.set("stop_reason", Json(stop_reason(outcome, tools)));
        delta.set("stop_sequence", outcome.matched.empty() ? Json() : Json(outcome.matched));
        Json m = Json::object(); m.set("type", Json("message_delta")); m.set("delta", std::move(delta)); m.set("usage", usage(outcome));
        event("message_delta", m);
        Json stop = Json::object(); stop.set("type", Json("message_stop"));
        event("message_stop", stop);
        response.end_stream();
        finish_entry(entry, outcome);
    }

    void finish_entry(RecentRequest& entry, const Outcome& o) {
        entry.status = !o.error.empty() ? "error" : o.finish;
        entry.prompt = o.stats.prompt_tokens; entry.cached = o.stats.cached_tokens; entry.output = o.output_tokens;
        entry.cycles = o.stats.cycles; entry.accepted = o.stats.accepted; entry.proposed = o.stats.proposed;
        entry.queue_ms = o.stats.queue_ms; entry.prefill_ms = o.stats.prefill_ms; entry.decode_ms = o.stats.decode_ms;
        record(entry);
    }

    // ---- service endpoints ----
    Json status() {
        const auto q = queue.snapshot();
        Json out = Json::object();
        out.set("schema", Json("strix-serve-status-v1"));
        out.set("model", Json(config.model_id));
        out.set("engine", Json(config.engine_description));
        out.set("context_tokens", Json(context_tokens));
        out.set("uptime_seconds", Json(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()));
        std::lock_guard lock(mutex);
        out.set("state", Json(q.running ? std::string("generating") : q.queued ? std::string("queued") : state));
        Json queue_json = Json::object();
        queue_json.set("queued", Json(q.queued)); queue_json.set("running", Json(q.running));
        queue_json.set("completed", Json(q.completed)); queue_json.set("failed", Json(q.failed)); queue_json.set("cancelled", Json(q.cancelled));
        out.set("queue", std::move(queue_json));
        Json totals = Json::object();
        totals.set("requests", Json(requests_total)); totals.set("prompt_tokens", Json(q.prompt_tokens));
        totals.set("cached_tokens", Json(q.cached_tokens)); totals.set("generated_tokens", Json(q.generated_tokens));
        totals.set("busy_seconds", Json(q.busy_seconds));
        out.set("totals", std::move(totals));
        Json list = Json::array();
        for (const auto& r : recent) {
            Json e = Json::object();
            e.set("id", Json(r.id)); e.set("route", Json(r.route)); e.set("time", Json(r.time)); e.set("status", Json(r.status));
            e.set("prompt_tokens", Json(r.prompt)); e.set("reused_tokens", Json(r.cached)); e.set("output_tokens", Json(r.output));
            e.set("queue_ms", Json(r.queue_ms)); e.set("prefill_ms", Json(r.prefill_ms)); e.set("decode_ms", Json(r.decode_ms));
            e.set("prefill_tokens_per_second", Json(r.prefill_ms > 0 ? 1000.0 * static_cast<double>(r.prompt - std::min(r.prompt, r.cached)) / r.prefill_ms : 0.0));
            e.set("decode_tokens_per_second", Json(r.decode_ms > 0 ? 1000.0 * static_cast<double>(r.output) / r.decode_ms : 0.0));
            e.set("mtp_cycles", Json(r.cycles)); e.set("mtp_accepted", Json(r.accepted)); e.set("mtp_proposed", Json(r.proposed));
            list.push(std::move(e));
        }
        out.set("requests", std::move(list));
        return out;
    }
    std::string metrics() {
        const auto q = queue.snapshot();
        std::ostringstream m;
        const auto counter = [&](const char* name, const char* help, double value, const char* type = "counter") {
            m << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type << '\n' << name << ' ' << value << '\n';
        };
        counter("strix_requests_completed_total", "Generations completed.", static_cast<double>(q.completed));
        counter("strix_requests_failed_total", "Generations that failed.", static_cast<double>(q.failed));
        counter("strix_requests_cancelled_total", "Generations cancelled (client gone or stop string).", static_cast<double>(q.cancelled));
        counter("strix_prompt_tokens_total", "Prompt tokens of finished generations.", static_cast<double>(q.prompt_tokens));
        counter("strix_prompt_tokens_cached_total", "Prompt tokens restored from the prefix state cache.", static_cast<double>(q.cached_tokens));
        counter("strix_generated_tokens_total", "Generated tokens.", static_cast<double>(q.generated_tokens));
        counter("strix_engine_busy_seconds_total", "Seconds the engine spent on generations.", q.busy_seconds);
        counter("strix_queue_waiting", "Generations waiting in the queue.", static_cast<double>(q.queued), "gauge");
        counter("strix_queue_running", "Generations running.", static_cast<double>(q.running), "gauge");
        return m.str();
    }
};

ApiService::ApiService(ApiConfig config, std::shared_ptr<const BpeTokenizer> tokenizer, GenerationQueue& queue,
    std::uint64_t context_tokens, bool engine_samples)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(tokenizer), queue, context_tokens, engine_samples)) {}
ApiService::~ApiService() = default;

std::vector<std::uint32_t> ApiService::chat_prompt(const Json& body) const { return impl_->chat_prompt(body).tokens; }
void ApiService::set_state(std::string state) { const std::lock_guard lock(impl_->mutex); impl_->state = std::move(state); }

void ApiService::handle(const HttpRequest& request, HttpResponseWriter& response) {
    auto& impl = *impl_;
    const bool anthropic = request.path == "/v1/messages";
    const auto error_body = [&](const ApiError& e) {
        Json detail = Json::object();
        if (anthropic) {
            detail.set("type", Json(e.type())); detail.set("message", Json(std::string(e.what())));
            Json out = Json::object(); out.set("type", Json("error")); out.set("error", std::move(detail));
            return out.dump();
        }
        detail.set("message", Json(std::string(e.what()))); detail.set("type", Json(e.type()));
        detail.set("param", e.param().empty() ? Json() : Json(e.param())); detail.set("code", Json());
        Json out = Json::object(); out.set("error", std::move(detail));
        return out.dump();
    };
    try {
        if (request.method == "GET" && (request.path == "/health" || request.path == "/v1/health")) {
            Json out = Json::object(); out.set("status", Json("ok")); out.set("model", Json(impl.config.model_id));
            response.send(200, "application/json", out.dump());
            return;
        }
        if (request.method == "GET" && request.path == "/metrics") { response.send(200, "text/plain; version=0.0.4", impl.metrics()); return; }
        if (request.method == "GET" && request.path == "/api/status") { response.send(200, "application/json", impl.status().dump()); return; }
        impl.authorize(request);
        if (request.method == "GET" && request.path == "/v1/models") {
            Json model = Json::object();
            model.set("id", Json(impl.config.model_id)); model.set("object", Json("model")); model.set("created", Json(unix_now()));
            model.set("owned_by", Json("strix")); model.set("context_length", Json(impl.context_tokens));
            Json data = Json::array(); data.push(std::move(model));
            Json out = Json::object(); out.set("object", Json("list")); out.set("data", std::move(data));
            response.send(200, "application/json", out.dump());
            return;
        }
        if (request.method != "POST") throw ApiError(404, "not_found_error", "unknown route " + request.method + " " + request.path);
        Json body;
        try { body = parse_json(request.body); }
        catch (const JsonError& e) { throw ApiError(400, "invalid_request_error", std::string("invalid JSON body: ") + e.what()); }
        if (!body.is_object()) throw ApiError(400, "invalid_request_error", "the request body must be a JSON object");
        if (request.path == "/v1/chat/completions") impl.chat_completions(request, response, body);
        else if (request.path == "/v1/completions") impl.completions(request, response, body);
        else if (request.path == "/v1/messages") impl.messages(request, response, body);
        else if (request.path == "/tokenize") {
            std::vector<std::uint32_t> ids;
            if (const auto* messages = body.find("messages")) { static_cast<void>(messages); ids = impl.chat_prompt(body).tokens; }
            else if (const auto* text = body.find("prompt"); text && text->is_string()) ids = impl.tokenizer->encode(text->as_string(), true);
            else throw ApiError(400, "invalid_request_error", "tokenize takes a prompt string or chat messages");
            Json tokens = Json::array();
            for (const auto id : ids) tokens.push(Json(static_cast<std::uint64_t>(id)));
            Json out = Json::object(); out.set("tokens", std::move(tokens)); out.set("count", Json(static_cast<std::uint64_t>(ids.size())));
            response.send(200, "application/json", out.dump());
        } else if (request.path == "/detokenize") {
            const auto* tokens = body.find("tokens");
            if (!tokens || !tokens->is_array()) throw ApiError(400, "invalid_request_error", "detokenize takes a tokens array");
            std::vector<std::uint32_t> ids;
            for (const auto& id : tokens->as_array()) {
                if (!id.is_number() || id.as_int() < 0 || static_cast<std::uint64_t>(id.as_int()) >= impl.tokenizer->vocab_size())
                    throw ApiError(400, "invalid_request_error", "token IDs must be in the vocabulary");
                ids.push_back(static_cast<std::uint32_t>(id.as_int()));
            }
            Json out = Json::object(); out.set("text", Json(sanitize_utf8(impl.tokenizer->decode(ids))));
            response.send(200, "application/json", out.dump());
        } else throw ApiError(404, "not_found_error", "unknown route " + request.path);
    } catch (const ApiError& e) {
        RecentRequest entry{"", request.path, "error", unix_now()};
        if (e.status() >= 500) impl.record(entry);
        response.send(e.status(), "application/json", error_body(e));
    } catch (const std::exception& e) {
        impl.record(RecentRequest{"", request.path, "error", unix_now()});
        response.send(500, "application/json", error_body(ApiError(500, anthropic ? "api_error" : "server_error", e.what())));
    }
}

}  // namespace strix::serve
