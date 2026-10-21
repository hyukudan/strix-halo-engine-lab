// Host tests of the serving front end: JSON, chat template (against Jinja renderings of the
// model's template), assistant output parsing, pre-tokenizer and streaming UTF-8 decoding.
// usage: strix-serve-frontend-tests CASES.json [MODEL.gguf]
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/chat_template.hpp"
#include "strix/serve/json.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace strix::serve;
int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}
std::string read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot read ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void json_tests() {
    const auto value = parse_json(R"({"b":1,"a":[true,false,null,-2.5e3,"x\"\\\n\u00e9\ud83d\ude00"],"b":2})");
    check(value.as_object().size() == 2U && value.as_object()[0].first == "b", "JSON keeps first key position");
    check(value.find("b")->as_int() == 2, "JSON duplicate key keeps the last value");
    check(value.dump() == "{\"b\":2,\"a\":[true,false,null,-2.5e3,\"x\\\"\\\\\\n\xC3\xA9\xF0\x9F\x98\x80\"]}", "JSON compact dump");
    check(value.dump_spaced() == "{\"b\": 2, \"a\": [true, false, null, -2.5e3, \"x\\\"\\\\\\n\xC3\xA9\xF0\x9F\x98\x80\"]}",
        "JSON spaced dump");
    for (const char* bad : {"", "{", "[1,]", "{\"a\" 1}", "01", "\"\\x\"", "tru", "1 2", "\"a\nb\""}) {
        bool thrown = false;
        try { (void)parse_json(bad); } catch (const JsonError&) { thrown = true; }
        check(thrown, std::string("JSON rejects ") + bad);
    }
    check(parse_json("123456789012").as_int() == 123456789012LL, "JSON integer");
    check(Json(std::string("\x01")).dump() == "\"\\u0001\"", "JSON control escape");
}

void template_tests(const char* cases_path) {
    const auto cases = parse_json(read_file(cases_path));
    unsigned passed = 0;
    for (const auto& c : cases.as_array()) {
        ChatTemplateInput input;
        input.messages = *c.find("messages");
        if (const auto* tools = c.find("tools")) input.tools = *tools;
        if (const auto* v = c.find("add_generation_prompt")) input.add_generation_prompt = v->as_bool();
        if (const auto* v = c.find("enable_thinking")) input.enable_thinking = v->as_bool();
        if (const auto* v = c.find("preserve_thinking")) input.preserve_thinking = v->as_bool();
        if (const auto* v = c.find("reasoning_effort")) input.reasoning_effort = v->as_string();
        const auto rendered = render_chat_template(input);
        const auto& expected = c.find("expected")->as_string();
        if (rendered == expected) { ++passed; continue; }
        std::size_t at = 0;
        while (at < rendered.size() && at < expected.size() && rendered[at] == expected[at]) ++at;
        check(false, "chat template case " + c.find("name")->as_string() + " differs at byte " + std::to_string(at) +
            ": ours [" + rendered.substr(at, 60) + "] expected [" + expected.substr(at, 60) + "]");
    }
    std::cout << "chat template: " << passed << " of " << cases.size() << " cases equal the Jinja renderings\n";
    // A JSON-string arguments field is parsed into the same object.
    ChatTemplateInput a, b;
    a.messages = parse_json(R"([{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{"k":"v","n":2}}}]}])");
    b.messages = parse_json(R"([{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"{\"k\":\"v\",\"n\":2}"}}]}])");
    check(render_chat_template(a) == render_chat_template(b), "string tool arguments render as the parsed object");
    bool thrown = false;
    try {
        ChatTemplateInput image;
        image.messages = parse_json(R"([{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}])");
        (void)render_chat_template(image);
    } catch (const ChatTemplateError&) { thrown = true; }
    check(thrown, "image content is rejected");
    thrown = false;
    try { ChatTemplateInput late; late.messages = parse_json(R"([{"role":"user","content":"a"},{"role":"system","content":"b"}])");
          (void)render_chat_template(late); } catch (const ChatTemplateError&) { thrown = true; }
    check(thrown, "system message after the start is rejected");
}

void output_tests() {
    const auto tools = parse_json(R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object",
        "properties":{"city":{"type":"string"},"days":{"type":"integer"},"opts":{"type":"object"}}}}}])");
    const auto out = parse_assistant_output("Let me check.\n</think>\n\nSure.\n\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\n42\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"u\": \"c\"}\n</parameter>\n"
        "</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\nNorth\n</parameter>\n</function>\n</tool_call>",
        true, tools);
    check(out.reasoning == "Let me check." && out.reasoning_complete, "reasoning before </think>");
    check(out.content == "Sure.", "content before the first tool call");
    check(out.tool_calls.size() == 2U, "two tool calls");
    if (out.tool_calls.size() == 2U) {
        const auto& args = *out.tool_calls.as_array()[0].find("arguments");
        check(args.find("city")->is_string() && args.find("city")->as_string() == "42", "string parameter stays text");
        check(args.find("days")->is_number() && args.find("days")->as_int() == 3, "integer parameter is typed");
        check(args.find("opts")->is_object(), "object parameter is parsed");
        check(out.tool_calls.as_array()[1].find("arguments")->find("city")->as_string() == "Oslo\nNorth", "multi-line value");
    }
    const auto plain = parse_assistant_output("Just text.\n", false, Json());
    check(plain.content == "Just text." && plain.reasoning.empty() && plain.tool_calls.size() == 0U, "no thinking, no tools");
    const auto partial = parse_assistant_output("still thinking", true, Json());
    check(!partial.reasoning_complete && partial.reasoning == "still thinking" && partial.content.empty(), "unterminated reasoning");
}

void tokenizer_tests(const char* model) {
    const auto pieces = BpeTokenizer::pre_tokenize("Hello world's 123  \n\n x\tdon'T 😀!!\n");
    const std::vector<std::string> expected{"Hello", " world", "'s", " ", "1", "2", "3", "  \n\n", " x", "\tdon", "'T", " 😀!!\n"};
    std::string got;
    for (const auto& piece : pieces) got += "[" + piece + "]";
    check(pieces == expected, "pre-tokenizer pieces " + got);
    // DeepSeek V3/V4 pre-tokenizer: pieces of the reference implementation (tokenizer.json of DeepSeek V4.1)
    const std::vector<std::pair<std::string, std::vector<std::string>>> deepseek{
        {"Hello world's 123456  \n\n x\tdon'T 😀!!\n", {"Hello", " world", "'s", " ", "123", "456", "  \n\n", " x", "\tdon", "'T", " 😀!!\n"}},
        {"x=a.b(c,d); // ünïcödé ÄÖÜ\r\n\r\n  end", {"x", "=a", ".b", "(c", ",d", ");", " //", " ünïcödé", " ÄÖÜ", "\r\n\r\n", " ", " end"}},
        {"日本語のテキストabc中文12漢字 カタカナ", {"日本語のテキスト", "abc", "中文", "12", "漢字", " ", "カタカナ"}},
        {"€100 and $5.99, (test) [x] {y} @user #tag",
         {"€", "100", " and", " $", "5", ".", "99", ",", " (", "test", ")", " [", "x", "]", " {", "y", "}", " @", "user", " #", "tag"}},
        {"naïve café \xCC\x81" "e \xC2\xA0nbsp", {"naïve", " café", " \xCC\x81" "e", " ", "\xC2\xA0nbsp"}},
        {"\x01\x02" "abc\xE2\x80\x8B" "def  ", {"\x01", "\x02" "abc", "\xE2\x80\x8B" "def", "  "}},
    };
    for (const auto& [text, want] : deepseek) {
        const auto ds = BpeTokenizer::pre_tokenize(text, "joyai-llm");
        std::string seen;
        for (const auto& piece : ds) seen += "[" + piece + "]";
        check(ds == want, "DeepSeek pre-tokenizer pieces " + seen);
    }
    Utf8StreamDecoder decoder;
    std::string text;
    for (const char* part : {"a\xF0\x9F", "\x98", "\x80" "b", "\xC3"}) text += decoder.push(part);
    check(text == "a\xF0\x9F\x98\x80" "b" && decoder.pending() == 1U, "streaming UTF-8 holds incomplete sequences");
    check(decoder.flush() == "\xEF\xBF\xBD", "flush replaces an incomplete tail");
    if (!model) return;
    const auto tokenizer = BpeTokenizer::load_gguf(model);
    for (const std::string& sample : {std::string("<|im_start|>user\nHola, ¿qué tal? 😀<|im_end|>\n"), std::string("def f(x):\n\treturn x**2\r\n")}) {
        const auto ids = tokenizer->encode(sample);
        check(tokenizer->decode(ids) == sample, "decode(encode(text)) == text");
    }
    check(tokenizer->find("<|im_end|>") == tokenizer->eos_token(), "EOS is <|im_end|>");
    std::cout << "tokenizer: vocabulary " << tokenizer->vocab_size() << ", pre " << tokenizer->pre() << '\n';
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) { std::cerr << "usage: strix-serve-frontend-tests CASES.json [MODEL.gguf]\n"; return 2; }
    json_tests();
    template_tests(argv[1]);
    output_tests();
    tokenizer_tests(argc > 2 ? argv[2] : nullptr);
    std::cout << "serve front-end tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "serve front-end tests: " << error.what() << '\n';
    return 1;
}
