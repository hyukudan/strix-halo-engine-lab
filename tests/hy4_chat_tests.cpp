// Hy4 chat template rendered natively (render_hy4_chat) and its output parser (parse_hy4_output, the shared key/value
// tool-call grammar). Expected texts are written from the model's chat_template.jinja: the model card's own example
// (one user turn, reasoning_effort no_think), the default high effort with a system turn, reasoning kept only after the
// last user turn, tools with calls and responses reordered to the call order, the fallback order, and the errors.
#include "strix/serve/hy4_chat.hpp"
#include "strix/serve/json.hpp"

#include <exception>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <string>

namespace {

using namespace strix::serve;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::cout << "FAIL " << what << "\n";
}
void check_text(const std::string& got, const std::string& want, const std::string& what) {
    if (got == want) return;
    std::size_t at = 0;
    while (at < got.size() && at < want.size() && got[at] == want[at]) ++at;
    check(false, what + ": differs at byte " + std::to_string(at) + "\n  got  ..." + got.substr(at, 120) + "\n  want ..." +
                     want.substr(at, 120));
}

const std::string S = "<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>", M = "<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>",
                  E = "<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>", RM = "<\xef\xbd\x9creasoning_mode:opensource\xef\xbd\x9c>";
const std::string TB = "<think:opensource>", TE = "</think:opensource>";
const std::string CSB = "<tool_calls:opensource>", CSE = "</tool_calls:opensource>", CB = "<tool_call:opensource>",
                  CE = "</tool_call:opensource>", KB = "<arg_key:opensource>", KE = "</arg_key:opensource>",
                  VB = "<arg_value:opensource>", VE = "</arg_value:opensource>", RB = "<tool_response:opensource>",
                  RE = "</tool_response:opensource>";

ChatTemplateInput input_of(const std::string& messages, const std::string& tools = "null") {
    ChatTemplateInput in;
    in.messages = parse_json(messages);
    in.tools = parse_json(tools);
    return in;
}

const std::string weather_tools =
    R"([{"type": "function", "function": {"name": "get_weather", "description": "Get weather",)"
    R"( "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}])";

std::string tools_prompt() {
    return "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function "
           "signatures within <tools></tools> XML tags:\n<tools>\n"
           R"({"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": )"
           R"({"city": {"type": "string"}, "days": {"type": "integer"}}}})"
           "\n</tools>\n\nFor function call returns, you should first print " + CSB +
           "\nFor each function call, you should return object like:\n" + CB + "{function-name}" + KB + "{arg-key-1}" + KE + VB +
           "{arg-value-1}" + VE + KB + "{arg-key-2}" + KE + VB + "{arg-value-2}" + VE + "..." + CE +
           "\nAt the end of function call returns, you should print " + CSE;
}

void render_tests() {
    {  // the model card's example: reasoning_effort no_think, generation prompt
        auto in = input_of(R"([{"role": "user", "content": "Hello! Can you briefly introduce yourself?"}])");
        in.reasoning_effort = "no_think";
        check_text(render_hy4_chat(in),
                   S + "system" + M + RM + "reasoning_effort:no_think" + E + S + "user" + M +
                       "Hello! Can you briefly introduce yourself?" + E + S + "assistant" + M + TB + TE,
                   "card example no_think");
        in.reasoning_effort.reset();
        in.enable_thinking = false;  // the API's "thinking off" maps to the same prompt
        check_text(render_hy4_chat(in),
                   S + "system" + M + RM + "reasoning_effort:no_think" + E + S + "user" + M +
                       "Hello! Can you briefly introduce yourself?" + E + S + "assistant" + M + TB + TE,
                   "enable_thinking=false");
    }
    {  // default effort high, leading system turn (content then the reasoning mode), list content
        auto in = input_of(R"([{"role": "system", "content": "You are helpful."},)"
                           R"({"role": "user", "content": [{"type": "text", "text": "Hi "}, "there"]}])");
        check_text(render_hy4_chat(in),
                   S + "system" + M + "You are helpful." + RM + "reasoning_effort:high" + E + S + "user" + M + "Hi there" + E +
                       S + "assistant" + M + TB,
                   "system + high");
    }
    {  // reasoning kept only after the last user turn without tools; no generation prompt after an assistant turn
        auto in = input_of(R"([{"role": "user", "content": "A"}, {"role": "assistant", "content": "B", "reasoning_content": "r1"},)"
                           R"({"role": "user", "content": "C"}, {"role": "assistant", "content": "D", "reasoning": "r2"}])");
        in.reasoning_effort = "max";
        const auto want = S + "system" + M + RM + "reasoning_effort:max" + E + S + "user" + M + "A" + E + S + "assistant" + M +
                          TB + TE + "B" + E + S + "user" + M + "C" + E + S + "assistant" + M + TB + "r2" + TE + "D" + E;
        check_text(render_hy4_chat(in), want, "reasoning after last user");
        in.preserve_thinking = true;
        auto kept = want;
        kept.replace(kept.find(TB + TE + "B"), (TB + TE + "B").size(), TB + "r1" + TE + "B");
        check_text(render_hy4_chat(in), kept, "preserved thinking");
    }
    {  // tools: synthesized system, calls, responses reordered to the call order
        const auto messages =
            R"([{"role": "user", "content": "Weather?"},)"
            R"({"role": "assistant", "content": "", "tool_calls": [)"
            R"({"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}},)"
            R"({"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]},)"
            R"({"role": "tool", "tool_call_id": "c2", "content": "sunny"}, {"role": "tool", "tool_call_id": "c1", "content": "rain"}])";
        const auto in = input_of(messages, weather_tools);
        const auto calls = CSB + CB + "get_weather" + KB + "city" + KE + VB + "Paris" + VE + KB + "days" + KE + VB + "2" + VE + CE +
                           CB + "get_weather" + KB + "city" + KE + VB + "Rome" + VE + CE + CSE;
        check_text(render_hy4_chat(in),
                   S + "system" + M + tools_prompt() + RM + "reasoning_effort:high" + E + S + "user" + M + "Weather?" + E + S +
                       "assistant" + M + TB + TE + calls + E + S + "tool" + M + RB + "rain" + RE + RB + "sunny" + RE + E + S +
                       "assistant" + M + TB,
                   "tools + reorder");
        // a system turn with tools: tools prompt, blank line, content, reasoning mode
        const auto with_system = input_of(R"([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}])",
                                          weather_tools);
        check_text(render_hy4_chat(with_system),
                   S + "system" + M + tools_prompt() + "\n\nBe brief." + RM + "reasoning_effort:high" + E + S + "user" + M + "x" +
                       E + S + "assistant" + M + TB,
                   "system + tools");
    }
    {  // an id set that does not match one-to-one keeps the message order; tool_search_output parts share <tools>
        const auto messages =
            R"([{"role": "user", "content": "q"},)"
            R"({"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": {}}}]},)"
            R"({"role": "tool", "tool_call_id": "b", "content": "one"},)"
            R"({"role": "tool", "content": [{"type": "tool_search_output", "tools": [{"name": "g"}]}, {"type": "text", "text": "t"}]}])";
        const auto in = input_of(messages);
        check_text(render_hy4_chat(in),
                   S + "system" + M + RM + "reasoning_effort:high" + E + S + "user" + M + "q" + E + S + "assistant" + M + TB + TE +
                       CSB + CB + "f" + CE + CSE + E + S + "tool" + M + RB + "one" + RE + RB + "<tools>\n{\"name\": \"g\"}\n</tools>t" +
                       RE + E + S + "assistant" + M + TB,
                   "fallback order + search output");
    }
    {  // errors of the template (and the engine's text-only input)
        auto bad_effort = input_of(R"([{"role": "user", "content": "x"}])");
        bad_effort.reasoning_effort = "low";
        bool raised = false;
        try { (void)render_hy4_chat(bad_effort); } catch (const ChatTemplateError&) { raised = true; }
        check(raised, "reasoning_effort low must raise");
        raised = false;
        try { (void)render_hy4_chat(input_of(R"([{"role": "user", "content": [{"type": "image_url", "image_url": "u"}]}])")); }
        catch (const ChatTemplateError&) { raised = true; }
        check(raised, "image part must raise");
    }
}

void dialect_tests() {
    const auto official = hy4_chat_dialect("{%- set HYTK = ':opensource' %}\n{%- elif reasoning_effort not in ['max', 'high', 'no_think'] %}");
    const Hy4ChatDialect defaults;
    check(official.suffix == defaults.suffix && official.efforts == defaults.efforts, "official dialect = defaults");
    // the Q4_K_M GGUF's tokenizer.chat_template: its own suffix and one more effort
    const auto pack =
        hy4_chat_dialect("{%- set HYTK = ':6124c78e' %}\n{%- elif reasoning_effort not in ['max', 'high', 'low', 'no_think'] %}");
    check(pack.suffix == ":6124c78e" && pack.efforts == std::vector<std::string>{"max", "high", "low", "no_think"}, "pack dialect");
    bool refused = false;
    try { (void)hy4_chat_dialect("{{ messages }}"); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "a template without HYTK is refused");
    auto in = input_of(R"([{"role": "user", "content": "Hi"}])");
    in.reasoning_effort = "low";
    const std::string s = "<\xef\xbd\x9chy_start:6124c78e\xef\xbd\x9c>", m = "<\xef\xbd\x9chy_middle:6124c78e\xef\xbd\x9c>",
                      e = "<\xef\xbd\x9chy_end:6124c78e\xef\xbd\x9c>", rm = "<\xef\xbd\x9creasoning_mode:6124c78e\xef\xbd\x9c>";
    check_text(render_hy4_chat(in, pack),
               s + "system" + m + rm + "reasoning_effort:low" + e + s + "user" + m + "Hi" + e + s + "assistant" + m + "<think:6124c78e>",
               "pack dialect render");
    in.reasoning_effort = "medium";
    std::string message;
    try { (void)render_hy4_chat(in, pack); } catch (const ChatTemplateError& error) { message = error.what(); }
    check(message == "reasoning_effort error : medium, should be no_think/low/high/max", "pack effort error: " + message);
    const auto out = parse_hy4_output("r</think:6124c78e>c<tool_calls:6124c78e><tool_call:6124c78e>f</tool_call:6124c78e>", true,
                                      Json(), pack);
    check(out.reasoning == "r" && out.content == "c" && out.tool_calls.size() == 1U, "pack dialect parse");
}

void parse_tests() {
    const auto tools = parse_json(weather_tools);
    {
        const auto out = parse_hy4_output("let me think" + TE + "Answer." + CSB + CB + "get_weather" + KB + "city" + KE + VB +
                                              "Paris" + VE + KB + "days" + KE + VB + "2" + VE + CE + CSE,
                                          true, tools);
        check(out.reasoning == "let me think" && out.reasoning_complete, "reasoning");
        check(out.content == "Answer.", "content before the call block: " + out.content);
        check(out.tool_calls.size() == 1U && out.call_complete.size() == 1U && out.call_complete[0], "one closed call");
        if (out.tool_calls.size() == 1U) {
            const auto& call = out.tool_calls.as_array()[0];
            check(call.find("name")->as_string() == "get_weather", "call name");
            const auto* args = call.find("arguments");
            check(args->find("city")->is_string() && args->find("city")->as_string() == "Paris", "string argument");
            check(args->find("days")->is_number() && args->find("days")->as_double() == 2.0, "integer argument typed");
        }
    }
    {
        const auto out = parse_hy4_output("Hello there.", false, tools);
        check(out.content == "Hello there." && out.reasoning.empty() && out.tool_calls.size() == 0U, "no_think content");
        const auto cut = parse_hy4_output("partial reasoning", true, tools);
        check(cut.reasoning == "partial reasoning" && !cut.reasoning_complete && cut.content.empty(), "reasoning cut by budget");
        const auto open = parse_hy4_output(TE + "x" + CSB + CB + "get_weather" + KB + "city" + KE + VB + "Ro", true, tools);
        check(open.unfinished_call && open.call_complete.size() == 1U && !open.call_complete[0], "unfinished call");
    }
}

}  // namespace

int main() {
    try {
        render_tests();
        dialect_tests();
        parse_tests();
    } catch (const std::exception& error) {
        std::cout << "FAIL exception: " << error.what() << "\n";
        ++failures;
    }
    std::cout << (failures ? "FAILED " + std::to_string(failures) : std::string("OK")) << "\n";
    return failures ? 1 : 0;
}
