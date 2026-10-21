// DeepSeek V4.1 chat encoding: the C++ renderer equals the model release's reference encoder (encoding.py) on the
// recorded cases, and the output parser splits reasoning, content and DSML tool calls.
//   strix-serve-deepseek-chat-tests tests/data/serve/deepseek_v41_chat_cases.json
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/json.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) try {
    using namespace strix::serve;
    if (argc != 2) {
        std::cerr << "usage: strix-serve-deepseek-chat-tests CASES.json\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto cases = parse_json(text);
    std::size_t index = 0, equal = 0;
    for (const auto& c : cases.as_array()) {
        ChatTemplateInput input;
        input.messages = *c.find("messages");
        if (const auto* t = c.find("tools"); t && t->is_array()) input.tools = *t;
        input.enable_thinking = c.find("thinking")->as_bool();
        if (const auto* e = c.find("reasoning_effort"); e && e->is_string()) input.reasoning_effort = e->as_string();
        if (c.find("preserve_thinking")->as_bool()) input.preserve_thinking = true;
        const auto got = render_deepseek_v41_chat(input);
        const auto& want = c.find("expected")->as_string();
        if (got == want) ++equal;
        else {
            std::size_t k = 0;
            while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
            check(false, "case " + std::to_string(index) + " differs at byte " + std::to_string(k) + ": got [" + got.substr(k, 80) + "] want [" +
                             want.substr(k, 80) + "]");
        }
        ++index;
    }
    std::cout << "deepseek chat: " << equal << " of " << index << " cases equal the reference encoder\n";

    // output parsing
    const std::string dsml = "\xEF\xBD\x9C" "DSML\xEF\xBD\x9C";
    {
        const auto out = parse_deepseek_v41_output("Madrid.", false, Json());
        check(out.content == "Madrid." && out.reasoning.empty() && out.tool_calls.size() == 0U, "chat content");
    }
    {
        const auto out = parse_deepseek_v41_output("think hard</think>The answer is 4.", true, Json());
        check(out.reasoning == "think hard" && out.reasoning_complete && out.content == "The answer is 4.", "thinking split");
        const auto open = parse_deepseek_v41_output("still thinking", true, Json());
        check(open.reasoning == "still thinking" && !open.reasoning_complete && open.content.empty(), "unfinished reasoning");
    }
    {
        const std::string completion = "Let me check.\n\n<" + dsml + " calls>\n<" + dsml + " invoke name=\"get_weather\">\n<" + dsml +
                                       " parameter name=\"city\" string=\"true\">Paris</" + dsml + " parameter>\n<" + dsml +
                                       " parameter name=\"days\" string=\"false\">3</" + dsml + " parameter>\n</" + dsml + " invoke>\n</" + dsml +
                                       " calls>";
        const auto out = parse_deepseek_v41_output(completion, false, Json());
        check(out.content == "Let me check.", "content before the tool calls");
        check(out.tool_calls.size() == 1U, "one tool call");
        if (out.tool_calls.size() == 1U) {
            const auto& call = out.tool_calls.as_array()[0];
            check(call.find("name")->as_string() == "get_weather", "tool name");
            check(call.find("arguments")->find("city")->as_string() == "Paris" && call.find("arguments")->find("days")->as_int() == 3,
                  "typed arguments");
        }
        const auto broken = parse_deepseek_v41_output("x\n\n<" + dsml + " calls>\ngarbage", false, Json());
        check(broken.tool_calls.size() == 0U && broken.content == "x\n\n<" + dsml + " calls>\ngarbage", "malformed block stays content");
    }
    std::cout << "deepseek chat tests: failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
