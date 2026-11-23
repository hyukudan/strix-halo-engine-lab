#include "strix/serve/inkling2_chat.hpp"

#include "inkling2/inkling2_mini_config.hpp"
#include "strix/serve/chat_template_kit.hpp"

#include <array>
#include <cmath>
#include <string_view>
#include <utility>

// Native text binding of chat_template.jinja from the Inkling 2 Mini checkpoint.
// The role and content markers, effort aliases and tool envelopes belong to the checkpoint.
namespace strix::serve {

const BpeTokenizer::HfIdentity inkling2_mini_tokenizer_identity{
    "o200k", inkling2::kOutputVocabulary, inkling2::kMiniEosToken, "<|begin_of_text|>", 200028U};

namespace {

using chat_kit::compact_json;
using chat_kit::JsonKeyOrder;
using chat_kit::py_str;
using chat_kit::raise;
using chat_kit::truthy;

constexpr auto kModel = "<|message_model|>";
constexpr auto kSystem = "<|message_system|>";
constexpr auto kText = "<|content_text|>";
constexpr auto kEnd = "<|end_message|>";

void frame(std::string& out, std::string_view header, std::string_view kind, std::string_view body) {
    out.append(header).append(kind).append(body).append(kEnd);
}

std::string_view role_marker(std::string_view role) {
    if (role == "user") return "<|message_user|>";
    if (role == "assistant") return kModel;
    if (role == "system") return kSystem;
    if (role == "tool") return "<|message_tool|>";
    raise("Unknown message role: " + std::string(role));
}

std::string effort_text(const Json& effort) {
    double number = 0.9;
    if (effort.is_string()) {
        using Alias = std::pair<std::string_view, double>;
        constexpr std::array aliases{Alias{"none", 0.0}, Alias{"minimal", 0.1}, Alias{"low", 0.2},
                                     Alias{"medium", 0.7}, Alias{"high", 0.9}, Alias{"xhigh", 0.99}, Alias{"max", 0.99}};
        const auto key = strip(effort.as_string());
        bool found = false;
        for (const auto& [name, value] : aliases) {
            if (key != name) continue;
            number = value;
            found = true;
            break;
        }
        if (!found) raise("Unknown reasoning_effort: " + effort.as_string());
    } else if (effort.is_number()) {
        number = effort.as_double();
    } else if (effort.is_bool()) {
        number = effort.as_bool() ? 1.0 : 0.0;
    } else if (!effort.is_null()) {
        raise("reasoning_effort must be an alias or a number");
    }
    if (!std::isfinite(number) || number < 0.0 || number > 0.99)
        raise("reasoning_effort must be in [0.0, 0.99]");
    return number == 0.0 ? "0" : chat_kit::python_float(number);
}

void tool_declarations(std::string& out, const Json& tools) {
    if (!truthy(&tools)) return;
    if (!tools.is_array()) raise("tools must be an array");
    Json specs = Json::array();
    for (const auto& tool : tools.as_array()) {
        const auto& fn = chat_kit::tool_spec(tool);
        const auto* name = fn.find("name");
        if (!name || !name->is_string()) raise("tool declaration function name must be a string");
        const auto* description = fn.find("description");
        const auto* parameters = fn.find("parameters");
        const auto* type = tool.find("type");
        specs.push(Json(Json::Object{{"description", truthy(description) ? *description : Json("")},
                                    {"name", *name}, {"parameters", truthy(parameters) ? *parameters : Json::object()},
                                    {"type", truthy(type) ? *type : Json("function")}}));
    }
    frame(out, "<|message_system|>tool_declare", "<|content_xml|>", compact_json(specs, JsonKeyOrder::recursive));
}

void append_tool_name(std::string& header, const Json* id, const Json::Array& messages) {
    if (!id || !id->is_string() || id->as_string().empty()) return;
    for (const auto& message : messages) {
        const auto* role = message.find("role");
        const auto* calls = message.find("tool_calls");
        if (!role || !role->is_string() || role->as_string() != "assistant" || !calls || !calls->is_array()) continue;
        for (const auto& call : calls->as_array()) {
            const auto* call_id = call.find("id");
            const auto* fn = call.find("function");
            if (fn && call_id && call_id->is_string() && call_id->as_string() == id->as_string())
                header += py_str(fn->find("name"), "tool call name");
        }
    }
}

// images: the checkpoint template's image part, one <|content_image|> frame holding the placeholder the vision path
// expands to the tower rows (user and tool messages only). Off: image parts are refused as before.
void content_parts(std::string& out, std::string_view header, const Json& content, std::string_view role, bool images) {
    if (!content.is_array()) raise("message content must be text or an array of parts");
    for (const auto& part : content.as_array()) {
        if (part.is_string()) {
            frame(out, header, kText, part.as_string());
            continue;
        }
        const auto* type = part.find("type");
        const auto kind = py_str(type, "content part type");
        if (images && type && (kind == "image" || kind == "input_image" || kind == "image_url")) {
            if (role != "user" && role != "tool") raise("Image content is only supported for user and tool messages");
            frame(out, header, "<|content_image|>", kInkling2ImagePlaceholder);
            continue;
        }
        if (images && type && (kind == "audio" || kind == "input_audio" || kind == "audio_url"))
            raise("Inkling 2 does not accept audio input");
        if (type && kind != "text" && kind != "input_text")
            raise("Unsupported Inkling 2 Mini text content part type: " + kind);
        const auto* text = part.find("text");
        frame(out, header, kText, text && text->is_string() ? std::string_view(text->as_string()) : std::string_view{});
    }
}

void tool_calls(std::string& out, const Json& calls, const Json* assistant_name) {
    if (!calls.is_array()) raise("tool_calls must be an array");
    for (const auto& call : calls.as_array()) {
        const auto* fn = call.find("function");
        const auto* name = fn ? fn->find("name") : nullptr;
        if (!name || !name->is_string()) raise("tool call function name must be a string");
        if (truthy(assistant_name) && (!assistant_name->is_string() || assistant_name->as_string() != name->as_string()))
            raise("assistant name does not match its tool call name");
        const auto* args = fn->find("arguments");
        if (truthy(args) && !args->is_object()) raise("tool call arguments must be a parsed object");
        const auto args_text = truthy(args) ? compact_json(*args, JsonKeyOrder::top_level) : "{}";
        const auto body = "{\"name\":" + name->dump() + ",\"args\":" + args_text + "}";
        frame(out, std::string(kModel) + name->as_string(), "<|content_invoke_tool_json|>", body);
    }
}

}  // namespace

std::string render_inkling2_chat(const Json& messages, const Json& tools, const Json& reasoning_effort, bool images) {
    if (!messages.is_array()) raise("messages must be an array");
    std::string out;
    tool_declarations(out, tools);
    const auto effort = "Thinking effort level: " + effort_text(reasoning_effort);
    bool effort_emitted = false;
    for (const auto& message : messages.as_array()) {
        const auto role = py_str(message.find("role"), "message role");
        std::string header(role_marker(role));
        if (!effort_emitted && role != "system") {
            frame(out, kSystem, kText, effort);
            effort_emitted = true;
        }

        const bool assistant = role == "assistant";
        const auto* name = message.find("name");
        if ((assistant || role == "tool") && truthy(name)) header += py_str(name, "message name");
        else if (role == "tool") append_tool_name(header, message.find("tool_call_id"), messages.as_array());

        const auto* reasoning = message.find("reasoning_content");
        const auto* calls = message.find("tool_calls");
        const auto* content = message.find("content");
        if (assistant && truthy(reasoning)) frame(out, header, "<|content_thinking|>", py_str(reasoning, "reasoning content"));
        if (content && content->is_string()) {
            if (truthy(content) || !(assistant && (truthy(calls) || truthy(reasoning))))
                frame(out, header, kText, content->as_string());
        } else if (truthy(content)) {
            content_parts(out, header, *content, role, images);
        }
        if (assistant) {
            if (truthy(calls)) tool_calls(out, *calls, name);
            out += "<|content_model_end_sampling|>";
        }
    }
    if (!effort_emitted) frame(out, kSystem, kText, effort);
    return out;
}

namespace {
std::string render_request(std::string_view json, bool images) {
    const auto request = parse_request_json(json);
    const Json absent;
    if (request.is_array()) return render_inkling2_chat(request, absent, absent, images);
    const auto* messages = request.find("messages");
    if (!messages) throw std::invalid_argument("expected a messages array or an object with messages");
    const auto* tools = request.find("tools");
    const auto* effort = request.find("reasoning_effort");
    return render_inkling2_chat(*messages, tools ? *tools : absent, effort ? *effort : absent, images);
}
}  // namespace

std::string render_inkling2_chat_request(std::string_view json) { return render_request(json, false); }

std::string render_inkling2_chat_media_request(std::string_view json) { return render_request(json, true); }

}  // namespace strix::serve
