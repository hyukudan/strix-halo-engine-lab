#include "strix/serve/inkling2_chat.hpp"
#include "strix/serve/chat_template_kit.hpp"

#include <iostream>
#include <string_view>

namespace {

using namespace strix::serve;

bool matches(std::string_view label, std::string_view rendered, std::string_view expected) {
    if (rendered == expected) return true;
    std::cerr << label << " differs\nExpected: " << expected << "\nRendered: " << rendered << '\n';
    return false;
}

bool check_template() {
    // Rendered with the checkpoint's chat_template.jinja.
    const auto messages = parse_json(R"([
        {"role":"system","content":"Be brief."},
        {"role":"user","content":["Find ",{"type":"input_text","text":"this"},{"text":"."}]},
        {"role":"assistant","name":"lookup","reasoning_content":"Checking","content":"","tool_calls":[
            {"id":"call-1","function":{"name":"lookup","arguments":{"z":0.00,"a":{"b":1e-5,"a":true}}}}]},
        {"role":"tool","tool_call_id":"call-1","content":"Found"},
        {"role":"tool","name":"override","tool_call_id":"call-1","content":""},
        {"role":"assistant","content":"Done"}
    ])");
    const auto tools = parse_json("[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"description\":\"caf\\u00e9 <&>\","
                                  "\"parameters\":{\"z\":1.00,\"a\":{\"b\":true,\"a\":false}}}}]");
    constexpr std::string_view expected =
        "<|message_system|>tool_declare<|content_xml|>"
        "[{\"description\":\"caf\xC3\xA9 <&>\",\"name\":\"lookup\",\"parameters\":{\"a\":{\"a\":false,\"b\":true},\"z\":1.0},"
        "\"type\":\"function\"}]<|end_message|>"
        "<|message_system|><|content_text|>Be brief.<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
        "<|message_user|><|content_text|>Find <|end_message|><|message_user|><|content_text|>this<|end_message|>"
        "<|message_user|><|content_text|>.<|end_message|>"
        "<|message_model|>lookup<|content_thinking|>Checking<|end_message|>"
        "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"a\":{\"b\":1e-05,\"a\":true},\"z\":0.0}}"
        "<|end_message|><|content_model_end_sampling|>"
        "<|message_tool|>lookup<|content_text|>Found<|end_message|>"
        "<|message_tool|>override<|content_text|><|end_message|>"
        "<|message_model|><|content_text|>Done<|end_message|><|content_model_end_sampling|>";
    return matches("tool conversation", render_inkling2_chat(messages, tools, " low "), expected) &&
           matches("bare request array", render_inkling2_chat_request("[]"),
                   "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>") &&
           matches("request object", render_inkling2_chat_request(R"({"messages":[],"reasoning_effort":"none"})"),
                   "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>") &&
           matches("empty chat", render_inkling2_chat(Json::array()),
                   "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>") &&
           matches("system only", render_inkling2_chat(parse_json(R"([{"role":"system","content":""}])"), {}, "none"),
                   "<|message_system|><|content_text|><|end_message|>"
                   "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>") &&
           matches("numeric effort", render_inkling2_chat(Json::array(), {}, parse_json("1e-4")),
                   "<|message_system|><|content_text|>Thinking effort level: 0.0001<|end_message|>");
}

bool check_errors() {
    for (const auto input : {R"([{"role":"unknown"}])", R"([{"role":"user","content":[{"type":"image_url"}]}])",
                             R"([{"role":"assistant","tool_calls":[{"function":{"name":"lookup","arguments":"{}"}}]}])"}) {
        try {
            (void)render_inkling2_chat(parse_json(input));
            std::cerr << "accepted an unsupported message: " << input << '\n';
            return false;
        } catch (const ChatTemplateError&) {}
    }
    for (const auto& effort : {Json("unknown"), Json(1), Json(-0.1)}) {
        try {
            (void)render_inkling2_chat(Json::array(), {}, effort);
            std::cerr << "accepted invalid reasoning effort\n";
            return false;
        } catch (const ChatTemplateError&) {}
    }
    return true;
}

// With images: the checkpoint template's image frame, user and tool messages only; audio refused.
bool check_images() {
    const auto msgs = parse_json(
        R"([{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}])");
    const std::string got = render_inkling2_chat(msgs, {}, {}, true);
    const std::string frame = "<|message_user|><|content_image|><|unused_200054|><|end_message|>";
    if (got.find(frame) == std::string::npos || got.find("What is this?") > got.find(frame)) {
        std::cerr << "image frame missing or out of order: " << got << '
';
        return false;
    }
    for (const auto input : {R"([{"role":"assistant","content":[{"type":"image"}]}])",
                             R"([{"role":"user","content":[{"type":"input_audio"}]}])"}) {
        try {
            (void)render_inkling2_chat(parse_json(input), {}, {}, true);
            std::cerr << "accepted: " << input << '
';
            return false;
        } catch (const ChatTemplateError&) {}
    }
    return true;
}

}  // namespace

int main() {
    return check_template() && check_errors() && check_images() &&
           matches("shared compact JSON", chat_kit::compact_json(parse_json(R"({"z":[1e-4,1e-5,1e15,1e16,-0.0,-0],"a":0})"),
                                                                 chat_kit::JsonKeyOrder::recursive),
                   R"({"a":0,"z":[0.0001,1e-05,1000000000000000.0,1e+16,-0.0,0]})") ? 0 : 1;
}
