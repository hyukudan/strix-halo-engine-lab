#include "strix/serve/glm_chat.hpp"
#include "strix/serve/chat_template_kit.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

// Port of the GLM-5 chat_template.jinja (macros visible_text, tool_to_json, render_tool_response,
// tool_references_to_response, id_of, is_list_of_outputs) as HF transformers renders it (trim_blocks, lstrip_blocks).
// tojson(ensure_ascii=False) is Json::dump_spaced; Jinja's {{ value }} of a scalar is Python str() ("None", "True").
namespace strix::serve {
namespace {

const std::string kThinkStart = "<think>", kThinkEnd = "</think>";
const std::string kCallStart = "<tool_call>", kCallEnd = "</tool_call>";
const std::string kKeyStart = "<arg_key>", kKeyEnd = "</arg_key>", kValueStart = "<arg_value>", kValueEnd = "</arg_value>";

using chat_kit::call_arguments;
using chat_kit::py_str;
using chat_kit::raise;
using chat_kit::tool_spec;
using chat_kit::tool_to_json;
using chat_kit::truthy;

bool is_type(const Json* item, std::string_view type) {
    const auto* t = item ? item->find("type") : nullptr;
    return t && t->is_string() && t->as_string() == type;
}
// A non-empty array whose first element is an object (content.0 in the template).
const Json* first_item(const Json* v) {
    return v && v->is_array() && v->size() > 0U && v->as_array()[0].is_object() ? &v->as_array()[0] : nullptr;
}

// visible_text: a string, or the text parts of a list (media parts become a reminder), or str() of anything else.
std::string visible_text(const Json* content) {
    if (content && content->is_array()) {
        std::string out;
        for (const auto& item : content->as_array()) {
            if (item.is_string()) { out += item.as_string(); continue; }
            if (!item.is_object()) continue;
            if (is_type(&item, "text")) { out += py_str(item.find("text"), "text part"); continue; }
            const auto* type = item.find("type");
            if (!type || !type->is_string()) continue;
            const auto& t = type->as_string();
            if (t == "image" || t == "image_url" || t == "video" || t == "video_url" || t == "audio" || t == "audio_url" || t == "input_audio") {
                const std::string media = t == "input_audio" ? "audio" : t.substr(0, t.find("_url"));
                out += "<reminder>You are unable to process this " + media +
                       " because you don't have multi-modal input ability. Try different methods.</reminder>";
            }
        }
        return out;
    }
    return py_str(content, "content");
}

std::string role_of(const Json& message) {
    const auto* role = message.find("role");
    return role && role->is_string() ? role->as_string() : std::string{};
}

std::string tool_response(const std::string& text) { return "<tool_response>" + text + "</tool_response>"; }

// tool_references_to_response: the request's tools named by the references (deferred ones too).
std::string tool_references(const Json& refs, const Json& tools) {
    if (!truthy(&tools)) raise("A tool_reference result needs the request's tools.");
    std::string out = "<tool_response><tools>\n";
    for (const auto& ref : refs.as_array()) {
        const auto* want = ref.find("name");
        for (const auto& entry : tools.as_array()) {
            const auto& tool = tool_spec(entry);
            const auto* name = tool.find("name");
            // Jinja: undefined == undefined
            const bool same = (!want && !name) || (want && name && want->dump() == name->dump());
            if (same) out += tool_to_json(tool) + "\n";
        }
    }
    return out + "</tools></tool_response>";
}

bool is_tool_reference_list(const Json* v) { return is_type(first_item(v), "tool_reference"); }
bool is_list_of_outputs(const Json& message) {
    const auto* first = first_item(message.find("content"));
    return first && first->find("output");
}

// One tool output (an entry of a list-of-outputs content).
std::string render_output(const Json& entry, const Json& tools) {
    const auto* output = entry.find("output");
    return is_tool_reference_list(output) ? tool_references(*output, tools) : tool_response(visible_text(output));
}

// render_tool_response
std::string render_tool_response(const Json& message, const Json& tools) {
    const auto* content = message.find("content");
    if (content && content->is_string()) return tool_response(content->as_string());
    if (is_tool_reference_list(content)) return tool_references(*content, tools);
    if (is_list_of_outputs(message)) {
        std::string out;
        for (const auto& entry : content->as_array()) out += render_output(entry, tools);
        return out;
    }
    return tool_response(visible_text(content));
}

// id_of: tool_call_id, else id, else empty.
std::string id_of(const Json& v) {
    if (const auto* id = v.find("tool_call_id"); truthy(id)) return py_str(id, "tool_call_id");
    if (const auto* id = v.find("id"); truthy(id)) return py_str(id, "id");
    return {};
}

// The <|observation|> block of the tool messages [begin, end]: in the order of the preceding assistant's tool calls
// when every call and every result has an id (results matching no call are dropped), else in message order.
std::string observation_block(const Json::Array& messages, std::size_t begin, std::size_t end, const Json& tools) {
    const Json* calls = nullptr;
    if (begin > 0U && role_of(messages[begin - 1U]) == "assistant" && truthy(messages[begin - 1U].find("tool_calls")))
        calls = messages[begin - 1U].find("tool_calls");
    bool can_sort = calls != nullptr;
    if (can_sort && !calls->is_array()) raise("tool_calls must be an array.");
    for (std::size_t k = begin; can_sort && k <= end; ++k) {
        if (is_list_of_outputs(messages[k])) {
            for (const auto& entry : messages[k].find("content")->as_array())
                if (id_of(entry).empty()) can_sort = false;
        } else if (id_of(messages[k]).empty()) can_sort = false;
    }
    if (can_sort)
        for (const auto& call : calls->as_array())
            if (id_of(call).empty()) can_sort = false;
    std::string out = "<|observation|>";
    if (!can_sort) {
        for (std::size_t k = begin; k <= end; ++k) out += render_tool_response(messages[k], tools);
        return out;
    }
    for (const auto& call : calls->as_array()) {
        const auto id = id_of(call);
        for (std::size_t k = begin; k <= end; ++k) {
            if (is_list_of_outputs(messages[k])) {
                for (const auto& entry : messages[k].find("content")->as_array())
                    if (id_of(entry) == id) out += render_output(entry, tools);
            } else if (id_of(messages[k]) == id) {
                out += render_tool_response(messages[k], tools);
            }
        }
    }
    return out;
}

// The arguments mapping of a tool call: an object as given, or a JSON string parsed into one.
std::string render_assistant(const Json& message, bool keep_reasoning) {
    std::string out = "<|assistant|>";
    auto content = visible_text(message.find("content"));
    std::optional<std::string> reasoning;
    if (const auto* r = message.find("reasoning_content"); r && r->is_string()) {
        reasoning = r->as_string();
    } else if (const auto end = content.find(kThinkEnd); end != std::string::npos) {
        const auto before = content.substr(0, end);
        const auto start = before.rfind(kThinkStart);
        reasoning = start == std::string::npos ? before : before.substr(start + kThinkStart.size());
        content = content.substr(content.rfind(kThinkEnd) + kThinkEnd.size());
    }
    out += keep_reasoning && reasoning ? kThinkStart + *reasoning + kThinkEnd : kThinkStart + kThinkEnd;
    out += strip(content);
    const auto* calls = message.find("tool_calls");
    if (!truthy(calls)) return out;
    if (!calls->is_array()) raise("tool_calls must be an array.");
    for (const auto& entry : calls->as_array()) {
        const auto* call = &entry;
        if (const auto* function = entry.find("function"); truthy(function)) call = function;
        const auto* name = call->find("name");
        if (!name || !name->is_string()) raise("Tool call is missing a function name.");
        out += kCallStart + name->as_string();
        const auto arguments = call_arguments(*call, name->as_string());
        for (const auto& [key, value] : arguments.as_object())
            out += kKeyStart + key + kKeyEnd + kValueStart + (value.is_string() ? value.as_string() : value.dump_spaced()) + kValueEnd;
        out += kCallEnd;
    }
    return out;
}

}  // namespace

std::string render_glm_chat(const ChatTemplateInput& input) {
    if (!input.messages.is_array()) raise("messages must be an array.");
    const auto& messages = input.messages.as_array();
    const auto& tools = input.tools;
    if (truthy(&tools) && !tools.is_array()) raise("tools must be an array.");
    const auto effort = input.reasoning_effort.value_or("max");
    std::string out = "[gMASK]<sop><|system|>Reasoning Effort: ";
    out += effort == "low" ? "Low" : effort == "high" ? "High" : "Max";
    if (truthy(&tools)) {
        out += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
               "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
        for (const auto& entry : tools.as_array()) {
            const auto& tool = tool_spec(entry);
            if (!truthy(tool.find("defer_loading"))) out += tool_to_json(tool) + "\n";
        }
        out += "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
               "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}"
               "</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
    }
    std::int64_t last_user = -1;
    for (std::size_t i = 0; i < messages.size(); ++i)
        if (role_of(messages[i]) == "user") last_user = static_cast<std::int64_t>(i);
    const bool keep_all = input.preserve_thinking && *input.preserve_thinking;  // clear_thinking=false
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message = messages[i];
        if (!message.is_object()) raise("Every message must be an object.");
        const auto role = role_of(message);
        if (role == "user") {
            out += "<|user|>" + visible_text(message.find("content"));
        } else if (role == "assistant") {
            out += render_assistant(message, keep_all || static_cast<std::int64_t>(i) > last_user);
        } else if (role == "tool") {
            if (i > 0U && role_of(messages[i - 1U]) == "tool") continue;  // rendered with the first of the run
            auto end = i;
            while (end + 1U < messages.size() && role_of(messages[end + 1U]) == "tool") ++end;
            out += observation_block(messages, i, end, tools);
        } else if (role == "system") {
            out += "<|system|>" + visible_text(message.find("content"));
        }  // other roles render nothing
    }
    if (input.add_generation_prompt) out += "<|assistant|>" + kThinkStart;
    return out;
}

AssistantOutput parse_glm_output(const std::string& text, bool thinking, const Json& tools) {
    (void)thinking;  // the generation prompt always opens the reasoning block
    static constexpr chat_kit::KvToolGrammar grammar{"</think>", "", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>",
                                                     "<arg_value>", "</arg_value>"};
    return chat_kit::parse_kv_tool_output(text, true, grammar, tools);
}

}  // namespace strix::serve
