#include "strix/serve/ernie_chat.hpp"

#include "strix/serve/api.hpp"
#include "strix/serve/chat_template_kit.hpp"
#include "strix/serve/json_constraint.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <ranges>

namespace strix::serve {
namespace {

using namespace std::string_view_literals;
using chat_kit::raise;
using chat_kit::truthy;

constexpr auto mask = "<|ERNIE_MASK|>"sv;
constexpr auto start = "<|ernie:im_start|>"sv;
constexpr auto end = "<|ernie:im_end|>"sv;
constexpr auto think = "<ernie:think>"sv;
constexpr auto think_end = "</ernie:think>"sv;
constexpr auto call = "<ernie:tool_call>"sv;
constexpr auto call_end = "</ernie:tool_call>"sv;
constexpr auto tool_out = "<ernie:tool_output>"sv;
constexpr auto tool_out_end = "</ernie:tool_output>"sv;
constexpr auto parameter_end = "</parameter>"sv;
constexpr auto invoke_end = "</invoke>"sv;
constexpr std::array completion_markers{mask, start, end, think, think_end, call, call_end};

const Json* string_field(const Json& object, std::string_view key) {
    const auto* value = object.find(key);
    return value && value->is_string() ? value : nullptr;
}

bool off(const Json* value) {
    return value && ((value->is_bool() && !value->as_bool()) ||
        (value->is_number() && value->as_double() == 0.0) ||
        (value->is_string() && value->as_string() == "false"));
}

std::vector<std::string_view> roles(const Json& messages) {
    if (!messages.is_array() || !messages.size()) raise("ERNIE requires a non-empty message array.");
    std::vector<std::string_view> result;
    result.reserve(messages.size());
    for (const auto& message : messages.as_array()) {
        const auto* role = string_field(message, "role");
        if (!role || (role->as_string() != "system" && role->as_string() != "user" &&
            role->as_string() != "assistant" && role->as_string() != "tool"))
            raise("Unsupported ERNIE message role.");
        result.push_back(role->as_string());
    }
    return result;
}

struct PromptBuilder {
    ErniePrompt prompt;
    bool assistant_body = false;

    void append(std::string_view bytes, bool control) {
        if (bytes.empty()) return;
        auto& spans = prompt.spans;
        if (!spans.empty() && spans.back().control == control && spans.back().assistant == assistant_body)
            spans.back().text += bytes;
        else spans.push_back({std::string(bytes), control, assistant_body});
    }

    void control(std::string_view bytes) { append(bytes, true); }
    void plain(std::string_view bytes) { append(bytes, false); }
    void append(const PromptBuilder& other) {
        for (const auto& span : other.prompt.spans) append(span.text, span.control);
    }

    void range(const PromptBuilder& other, std::size_t first, std::size_t last) {
        std::size_t at = 0;
        for (const auto& span : other.prompt.spans) {
            const auto past = at + span.text.size();
            if (past > first && at < last) {
                const auto lo = std::max(first, at);
                append(std::string_view(span.text).substr(lo - at, std::min(last, past) - lo), span.control);
            }
            if (past >= last) break;
            at = past;
        }
    }

    void history(const PromptBuilder& other, bool example = false) {
        // Each rendered turn ends with the end marker and a newline.
        if (other.prompt.spans.empty()) return;
        const auto& spans = other.prompt.spans;
        for (std::size_t i = 0; i < spans.size(); ++i) {
            const auto& span = spans[i];
            auto text = std::string_view(span.text);
            if (i + 1U == spans.size()) text = text.substr(0, text.find_last_not_of('\n') + 1U);
            assistant_body = !example && span.assistant;
            append(text, span.control);
        }
        assistant_body = false;
    }

    void header(std::string_view role) {
        control(start);
        control(role);
    }
};

std::string subtitle_time(const Json& value) {
    std::array<char, 350> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(),
        value.as_double(), std::chars_format::fixed, 1);
    if (result.ec != std::errc{}) raise("Invalid ERNIE subtitle time.");
    return {buffer.data(), result.ptr};
}

class Renderer {
public:
    explicit Renderer(bool thinking) : thinking_(thinking) {}

    PromptBuilder body(const Json* value, std::string_view role, bool has_reasoning = false) {
        PromptBuilder result;
        if (value && value->is_string()) result.plain(value->as_string());
        if (!value || !value->is_array()) return result;
        for (const auto& item : value->as_array()) {
            if (!item.is_object()) raise("ERNIE content parts must be objects.");
            const auto* type = string_field(item, "type");
            if (!type) continue;
            const auto& kind = type->as_string();
            if (kind == "text") result.plain(chat_kit::py_str(item.find("text"), "ERNIE text"));
            else if (kind == "image_url") {
                ++pictures_;
                result.control(role == "assistant" && has_reasoning ? " " : " Picture " + std::to_string(pictures_) + ":");
                result.control("<image_start><|image@placeholder|><image_end>");
            } else if (kind == "audio_url") {
                ++audios_;
                if (role != "user" && role != "assistant") continue;
                result.control(role == "user" ? " Audio " + std::to_string(audios_) + ":" : " ");
                result.control("<audio_start><|audio@placeholder|><audio_end>");
            } else if (kind == "video_url") {
                result.control(" Video " + std::to_string(++videos_) + ":<video_start><|video@placeholder|><video_end>");
                const auto* video = item.find("video_url");
                const auto* subtitles = video ? video->find("subtitles") : nullptr;
                if (!truthy(subtitles)) continue;
                if (!subtitles->is_array()) raise("ERNIE subtitles must be an array.");
                result.control(" <Start of Audio Content>:");
                for (const auto& cue : subtitles->as_array()) {
                    if (!cue.is_array() || cue.size() != 3U) raise("ERNIE subtitles require [text, start, end].");
                    const auto& fields = cue.as_array();
                    result.control("[" + subtitle_time(fields[1]) + "," + subtitle_time(fields[2]) + "]");
                    result.plain(chat_kit::py_str(&fields[0], "ERNIE subtitle"));
                }
                result.control("<End of Audio Content> ");
            }
        }
        return result;
    }

    PromptBuilder turns(const Json& messages, const std::vector<std::string_view>& names) {
        PromptBuilder output;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto& message = messages.as_array()[i];
            const auto role = names[i];
            auto content = body(message.find("content"), role, message.find("reasoning_content") != nullptr);
            if (role == "system" && i == 0U) continue;
            if (role == "tool") {
                if (i == 0U || names[i - 1] != role) output.header(role);
                output.control("\n");
                output.control(tool_out);
                output.append(content);
                output.control(tool_out_end);
                if (i + 1 < names.size() && names[i + 1] == role) continue;
            } else {
                output.header(role);
                output.control("\n");
                output.assistant_body = role == "assistant";
                if (output.assistant_body) assistant(output, message, content);
                else output.append(content);
            }
            output.control(end);
            output.assistant_body = false;
            output.control("\n");
        }
        return output;
    }

private:
    void assistant(PromptBuilder& output, const Json& message, const PromptBuilder& content) const {
        output.control(think);
        const auto* reasoning = string_field(message, "thoughts");
        if (!reasoning) reasoning = string_field(message, "reasoning_content");
        std::size_t answer = 0;
        if (reasoning) {
            if (thinking_ && !strip(reasoning->as_string()).empty()) output.plain(reasoning->as_string());
        } else {
            const auto text = content.prompt.text();
            if (const auto close = text.find(think_end); close != std::string::npos) {
                const auto open = text.rfind(think, close);
                const auto first = open == std::string::npos ? 0U : open + think.size();
                if (thinking_ && !strip(text.substr(first, close - first)).empty()) output.range(content, first, close);
                answer = text.rfind(think_end) + think_end.size();
            }
        }
        output.control(think_end);
        output.range(content, answer, std::string::npos);
        const auto* calls = message.find("tool_calls");
        if (!truthy(calls)) return;
        if (!calls->is_array()) raise("ERNIE tool_calls must be an array.");
        output.control(call);
        output.control("\n");
        for (const auto& entry : calls->as_array()) {
            const auto* function = entry.find("function");
            const auto& spec = truthy(function) ? *function : entry;
            const auto* name = string_field(spec, "name");
            if (!name) raise("An ERNIE tool call requires a function name.");
            const auto* field = spec.find("arguments");
            if (field && field->is_string() && strip(field->as_string()).empty()) raise("ERNIE arguments must be an object.");
            auto arguments = chat_kit::call_arguments(spec, name->as_string());
            std::ranges::sort(arguments.as_object(), {}, &Json::Member::first);
            output.control("<invoke name=\"");
            output.plain(name->as_string());
            output.control("\">");
            for (const auto& [key, value] : arguments.as_object()) {
                output.control("<parameter name=\"");
                output.plain(key);
                output.control("\">");
                output.plain(value.is_string() ? value.as_string() : chat_kit::spaced_json(value, chat_kit::JsonKeyOrder::recursive));
                output.control(parameter_end);
            }
            output.control(invoke_end);
            output.control("\n");
        }
        output.control(call_end);
    }

    bool thinking_;
    std::size_t pictures_ = 0, audios_ = 0, videos_ = 0;
};

ErniePrompt render(const ChatTemplateInput& input, const ErnieChatOptions& extra) {
    const auto names = roles(input.messages);
    const bool tools = truthy(&input.tools);
    if (!input.tools.is_null() && !input.tools.is_array()) raise("ERNIE tools must be an array.");
    if (!extra.options.is_null() && !extra.options.is_object()) raise("ERNIE options must be an object.");
    const auto* mode = string_field(extra.options, "thinking_mode");
    const bool thinking = input.enable_thinking.value_or(true) && !(mode &&
        (mode->as_string() == "close" || mode->as_string() == "false"));
    Renderer renderer(thinking);
    PromptBuilder output;
    output.prompt.thinking = thinking;
    output.control(mask);
    output.header("system\n");
    if (names.front() == "system" || truthy(&extra.system_settings) || truthy(&extra.agent_storage) || truthy(&extra.examples)) {
        output.control("<ernie:system_setting>");
        if (truthy(&extra.system_settings)) {
            output.control("\n");
            output.plain(chat_kit::py_str(&extra.system_settings, "ERNIE system_settings"));
        }
        if (names.front() == "system") {
            const auto* content = input.messages.as_array().front().find("content");
            if (content && (content->is_string() || content->is_array())) {
                output.control("\n");
                output.append(renderer.body(content, "user"));
            }
        }
        if (truthy(&extra.agent_storage)) {
            output.control("\n【agent storage】记录额外的信息，包含但不限于用户画像、长期记忆内容等。\n");
            output.plain(chat_kit::py_str(&extra.agent_storage, "ERNIE agent_storage"));
        }
        if (truthy(&extra.examples)) {
            if (!extra.examples.is_array()) raise("ERNIE examples must be an array.");
            const auto& examples = extra.examples.as_array();
            if (truthy(&examples.front())) {
                output.control("\n【行为例子】列举了一些情况下的行为逻辑供你参考。\n");
                std::size_t index = 0;
                for (const auto& sample : examples | std::views::reverse) {
                    if (index) output.control("\n");
                    output.control("[示例" + std::to_string(++index) + "]\n");
                    output.history(renderer.turns(sample, roles(sample)), true);
                }
            }
        }
        output.control("\n</ernie:system_setting>\n");
    }
    output.control("<ernie:global_setting>");
    if (thinking) {
        const auto effort = input.reasoning_effort.value_or("high");
        if (effort != "low" && effort != "medium" && effort != "high")
            raise("ERNIE reasoning_effort must be low, medium or high.");
        output.control("\nreasoning_effort=");
        output.plain(effort);
    }
    if (tools && !off(extra.options.find("parallel_tool_calls"))) output.control("\nparallel_tool_calls=True");
    output.control("\n</ernie:global_setting>");
    if (tools) {
        for (const auto& spec : input.tools.as_array())
            if (!spec.is_object()) raise("Every ERNIE tool must be an object.");
        output.control("\n<ernie:tool_list>\n");
        output.plain(chat_kit::spaced_json(input.tools, chat_kit::JsonKeyOrder::recursive));
        output.control("\n</ernie:tool_list>");
    }
    output.control(end);
    output.control("\n");
    output.history(renderer.turns(input.messages, names));
    if (input.add_generation_prompt) {
        output.control("\n");
        output.header("assistant\n");
        output.control(think);
        if (!thinking) output.control(think_end);
    }
    return std::move(output.prompt);
}

}  // namespace

std::string ErniePrompt::text() const {
    std::string result;
    for (const auto& span : spans) result += span.text;
    return result;
}

std::vector<std::uint32_t> ErniePrompt::encode(const BpeTokenizer& tokenizer) const {
    std::vector<std::uint32_t> result;
    for (const auto& span : spans) {
        const auto ids = span.control ? tokenizer.encode(span.text) : tokenizer.encode_ordinary(span.text);
        result.insert(result.end(), ids.begin(), ids.end());
    }
    return result;
}

ErniePrompt render_ernie_chat(const ChatTemplateInput& input, const ErnieChatOptions& options) {
    return render(input, options);
}

namespace {

struct MarkerIds {
    std::array<std::uint32_t, completion_markers.size()> ids{};

    explicit MarkerIds(const BpeTokenizer& tokenizer) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            ids[i] = tokenizer.find(completion_markers[i]);
            if (ids[i] == UINT32_MAX ||
                tokenizer.encode(completion_markers[i]) != std::vector<std::uint32_t>{ids[i]} ||
                (tokenizer.type(ids[i]) != BpeTokenizer::TokenType::control &&
                 tokenizer.type(ids[i]) != BpeTokenizer::TokenType::user_defined))
                throw std::invalid_argument("ERNIE tokenizer lacks an atomic marker: " + std::string(completion_markers[i]));
        }
        auto stops = stop_tokens();
        std::ranges::sort(stops);
        if (stops != std::vector<std::uint32_t>{2, 8})
            throw std::invalid_argument("ERNIE tokenizer stop IDs must match the checkpoint generation settings");
    }

    std::string_view marker(std::uint32_t id) const {
        const auto it = std::ranges::find(ids, id);
        return it == ids.end() ? std::string_view{} : completion_markers[static_cast<std::size_t>(it - ids.begin())];
    }

    std::vector<std::uint32_t> stop_tokens() const { return {ids[0], ids[2]}; }
};

bool accepts(const ConstraintProgram& program, const Json& value) {
    ConstraintCursor cursor(program);
    for (const char byte : value.dump())
        if (!cursor.advance(static_cast<unsigned char>(byte))) return false;
    return cursor.complete();
}

std::optional<Json> typed_arguments(const std::string& name, const Json::Object& raw, const Json& tools) {
    const Json* schema = nullptr;
    if (tools.is_array()) {
        const auto match = std::ranges::find_if(tools.as_array(), [&](const Json& entry) {
            const auto* function = entry.find("function");
            const auto* tool_name = string_field(function ? *function : entry, "name");
            return tool_name && tool_name->as_string() == name;
        });
        if (match == tools.as_array().end()) return {};
        const auto* function = match->find("function");
        schema = (function ? *function : *match).find("parameters");
    }
    std::optional<ConstraintProgram> program;
    if (schema) program = compile_json_schema(*schema);
    const auto* properties = schema ? schema->find("properties") : nullptr;
    Json::Object candidate;
    std::vector<std::pair<std::size_t, Json>> alternatives;
    for (const auto& [key, value] : raw) {
        std::optional<Json> parsed;
        try {
            auto json = parse_json_unique(value.as_string());
            if (!json.is_string()) parsed = std::move(json);
        } catch (const JsonError&) {}
        bool string_allowed = true;
        if (const auto* property = properties ? properties->find(key) : nullptr) {
            // Retain the root definitions when checking an individual property.
            auto local = *property;
            if (const auto* defs = schema->find("$defs"); defs && local.is_object() && !local.find("$defs"))
                local.set("$defs", *defs);
            const auto filter = compile_json_schema(local);
            string_allowed = accepts(filter, value);
            if (parsed && !accepts(filter, *parsed)) parsed.reset();
        }
        if (!string_allowed && !parsed) return {};
        if (string_allowed && parsed) alternatives.emplace_back(candidate.size(), *parsed);
        candidate.emplace_back(key, string_allowed ? value : *parsed);
    }
    if (alternatives.size() > 8U || (!program && !alternatives.empty())) return {};
    const std::size_t count = std::size_t{1} << alternatives.size();
    std::optional<Json> unique;
    for (std::size_t selection = 0; selection < count; ++selection) {
        auto object = candidate;
        for (std::size_t bit = 0; bit < alternatives.size(); ++bit) {
            const auto& [index, value] = alternatives[bit];
            if (selection & (std::size_t{1} << bit)) object[index].second = value;
        }
        Json args(std::move(object));
        if (program && !accepts(*program, args)) continue;
        if (unique) return {};
        unique = std::move(args);
    }
    return unique;
}

void skip_space(std::string_view& input) {
    const auto first = input.find_first_not_of(" \t\n\r\v\f");
    input.remove_prefix(first == std::string_view::npos ? input.size() : first);
}

bool take(std::string_view& input, std::string_view prefix) {
    if (!input.starts_with(prefix)) return false;
    input.remove_prefix(prefix.size());
    return true;
}

bool attribute(std::string_view& input, std::string_view tag, std::string& name) {
    if (!take(input, tag)) return false;
    const auto before = input.size();
    skip_space(input);
    if (before == input.size() || !take(input, "name=\"")) return false;
    const auto quote = input.find('"');
    if (quote == 0 || quote == std::string_view::npos) return false;
    name = input.substr(0, quote);
    input.remove_prefix(quote);
    return take(input, "\">");
}

std::optional<Json> decode_calls(std::string_view body, const Json& tools) {
    Json calls = Json::array();
    skip_space(body);
    while (!body.empty()) {
        std::string name;
        if (!attribute(body, "<invoke", name)) return {};
        Json::Object raw;
        for (;;) {
            skip_space(body);
            if (take(body, invoke_end)) break;
            std::string key;
            if (!attribute(body, "<parameter", key)) return {};
            if (std::ranges::any_of(raw, [&](const auto& entry) { return entry.first == key; })) return {};
            const auto close = body.find(parameter_end);
            if (close == std::string_view::npos) return {};
            raw.emplace_back(std::move(key), Json(body.substr(0, close)));
            body.remove_prefix(close + parameter_end.size());
        }
        auto arguments = typed_arguments(name, raw, tools);
        if (!arguments) return {};
        Json entry = Json::object();
        entry.set("name", Json(std::move(name)));
        entry.set("arguments", std::move(*arguments));
        calls.push(std::move(entry));
        skip_space(body);
    }
    if (!calls.size()) return {};
    return calls;
}

class ErnieCompletion {
public:
    ErnieCompletion(std::shared_ptr<const BpeTokenizer> tokenizer, const MarkerIds& markers,
                    bool thinking, Json tools, bool tools_off)
        : tokenizer_(std::move(tokenizer)), markers_(markers), tools_(std::move(tools)), tools_off_(tools_off),
          phase_(thinking ? Phase::reasoning : Phase::content) {}

    void push(std::span<const std::uint32_t> ids, GenerationOutput& output) {
        GenerationOutput::Delta delta;
        for (const auto id : ids) {
            if (phase_ == Phase::done || phase_ == Phase::invalid) break;
            const auto marker = markers_.marker(id);
            if (marker.empty()) {
                append(utf8_.push(tokenizer_->token_bytes(id)), delta);
                continue;
            }
            append(utf8_.flush(), delta);
            if (phase_ == Phase::invalid) break;
            if (marker == think_end && phase_ == Phase::reasoning) {
                phase_ = Phase::content;
                output.parsed.reasoning_complete = true;
            } else if (marker == call && phase_ == Phase::content) {
                phase_ = Phase::tools;
                output.parsed.unfinished_call = true;
            } else if (marker == call_end && phase_ == Phase::tools) {
                if (!tools_off_) {
                    try { calls_ = decode_calls(body_, tools_); }
                    catch (const ConstraintError&) {}
                }
                phase_ = Phase::after_tools;
                body_.clear();
            } else if ((marker == end || marker == mask) &&
                (phase_ == Phase::content || phase_ == Phase::after_tools)) {
                phase_ = Phase::done;
            } else {
                phase_ = Phase::invalid;
            }
        }
        output.publish({{}, std::move(delta.reasoning)});
        output.publish({std::move(delta.content), {}});
    }

    void finish(GenerationOutput& output) {
        GenerationOutput::Delta delta;
        append(utf8_.flush(), delta);
        output.publish(std::move(delta));
        const bool closed = phase_ == Phase::after_tools || phase_ == Phase::done;
        if (closed && calls_) {
            output.parsed.tool_calls = std::move(*calls_);
            output.parsed.unfinished_call = false;
        } else output.parsed.tool_calls = Json::array();
    }

private:
    enum class Phase { reasoning, content, tools, after_tools, done, invalid };

    void append(std::string bytes, GenerationOutput::Delta& delta) {
        switch (phase_) {
        case Phase::reasoning: delta.reasoning += bytes; break;
        case Phase::content: delta.content += bytes; break;
        case Phase::tools: if (!tools_off_) body_ += bytes; break;
        case Phase::after_tools: if (!strip(bytes).empty()) phase_ = Phase::invalid; break;
        case Phase::done:
        case Phase::invalid: break;
        }
    }

    std::shared_ptr<const BpeTokenizer> tokenizer_;
    MarkerIds markers_;
    Json tools_;
    bool tools_off_;
    Phase phase_;
    Utf8StreamDecoder utf8_;
    std::string body_;
    std::optional<Json> calls_;
};

struct ErnieRequest final : GenerationContext {
    bool thinking = true, tools_off = false;
    Json tools;
};

ApiChatPrompt prepare(const Json& body, const BpeTokenizer& tokenizer) {
    const auto* kwargs = body.find("chat_template_kwargs");
    const auto field = [&](std::string_view key) {
        const auto* value = kwargs ? kwargs->find(key) : nullptr;
        return value ? value : body.find(key);
    };
    const auto copy = [&](std::string_view key) { const auto* v = field(key); return v ? *v : Json{}; };
    auto context = std::make_shared<ErnieRequest>();
    context->tools_off = tool_choice_is_none(body);
    ChatTemplateInput input;
    if (const auto* messages = body.find("messages"); messages && messages->is_array()) {
        input.messages = Json::array();
        for (auto message : messages->as_array()) {
            if (!message.find("reasoning_content"))
                if (const auto* reasoning = string_field(message, "reasoning")) message.set("reasoning_content", *reasoning);
            input.messages.push(std::move(message));
        }
    }
    if (const auto* tools = body.find("tools"); tools && !context->tools_off) input.tools = *tools;
    if (off(field("enable_thinking"))) input.enable_thinking = false;
    if (const auto* effort = field("reasoning_effort")) {
        if (effort->is_string()) {
            const auto& value = effort->as_string();
            if (value == "none") input.enable_thinking = false;
            else input.reasoning_effort = value == "minimal" ? "low" : value;
        } else input.reasoning_effort = "";
    }
    const auto rendered = render_ernie_chat(input, {copy("system_settings"), copy("agent_storage"), copy("examples"), copy("options")});
    context->thinking = rendered.thinking;
    context->tools = input.tools;
    ApiChatPrompt result;
    result.tokens = rendered.encode(tokenizer);
    result.thinking = context->thinking;
    result.tools = context->tools;
    result.context = std::move(context);
    return result;
}

}  // namespace

std::shared_ptr<const BpeTokenizer> load_ernie_tokenizer(const std::string& path) {
    auto tokenizer = BpeTokenizer::load_hf_byte_bpe(path, end);
    (void)MarkerIds(*tokenizer);
    return tokenizer;
}

void bind_ernie_chat(ApiConfig& config, std::shared_ptr<const BpeTokenizer> tokenizer) {
    if (!tokenizer) throw std::invalid_argument("ERNIE tokenizer is required");
    const MarkerIds markers(*tokenizer);
    config.stop_tokens = markers.stop_tokens();
    config.model_id = "ERNIE-5.5-Flash-Preview";
    config.default_thinking = true;
    config.prepare_chat = [tokenizer](const Json& body) { return prepare(body, *tokenizer); };
    for (const char* field : {"chat_template_kwargs", "system_settings", "agent_storage", "examples", "options"})
        config.chat_extension_fields.emplace_back(field);
    config.prepare_job = [tokenizer, markers, previous = std::move(config.prepare_job)](
        const HttpRequest& request, const Json& body, GenerationJob& job) {
        if (previous) previous(request, body, job);
        const auto context = std::dynamic_pointer_cast<const ErnieRequest>(job.family_context);
        if (!context) return;
        auto output = std::make_shared<GenerationOutput>();
        output->tool_frames_parsed = true;
        auto parser = std::make_shared<ErnieCompletion>(tokenizer, markers, context->thinking, context->tools, context->tools_off);
        const std::weak_ptr<GenerationOutput> weak = output;
        output->consume_tokens = [parser, weak](std::span<const std::uint32_t> ids) {
            if (const auto presentation = weak.lock()) parser->push(ids, *presentation);
            return true;
        };
        output->finish_tokens = [parser, weak] {
            if (const auto presentation = weak.lock()) parser->finish(*presentation);
        };
        job.output = std::move(output);
    };
}

}  // namespace strix::serve
