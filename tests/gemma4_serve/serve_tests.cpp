#include "gemma4/serve/tokenizer.hpp"
#include "gemma4/serve/chat.hpp"
#include "gemma4/serve/parse.hpp"
#include "gemma4/serve/adapter.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace g4 = strix::gemma4::serve;
using strix::serve::Json;
namespace {
void check(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f, const std::string& message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    check(threw, message);
}
const Json& get(const Json& j, std::string_view k) {
    const auto* p = j.find(k); if (!p) throw std::runtime_error("missing fixture key " + std::string(k)); return *p;
}
Json read(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    check(static_cast<bool>(f), "missing fixture");
    return strix::serve::parse_json(std::string(std::istreambuf_iterator<char>(f), {}));
}
std::vector<std::uint32_t> ids(const Json& j) {
    std::vector<std::uint32_t> v;
    for (const auto& i : j.as_array()) v.push_back(static_cast<std::uint32_t>(i.as_int()));
    return v;
}
void fixture(const Json& root, const g4::Tokenizer* tok) {
    if (tok) for (const auto& c : get(root, "encode").as_array()) {
        const auto expected = ids(get(c, "ids"));
        const auto actual = tok->encode(get(c, "text").as_string());
        check(actual == expected, "encode IDs: " + get(c, "text").as_string().substr(0, 100));
        check(tok->decode(expected) == get(c, "decoded").as_string(), "decode round trip");
    }
    for (const auto& c : get(root, "chat").as_array()) {
        check(!c.find("error"), "reference error is NO-REFERENCE");
        const auto rendered = g4::render_chat(g4::chat_input(get(c, "messages"), get(c, "kwargs")));
        check(rendered == get(c, "rendered").as_string(), "chat bytes " + get(c, "name").as_string() +
            "\nactual: " + rendered + "\nexpected: " + get(c, "rendered").as_string());
        if (tok) check(tok->encode(rendered) == ids(get(c, "ids")), "chat IDs " + get(c, "name").as_string());
    }
    if (tok && root.find("decode")) for (const auto& c : get(root, "decode").as_array())
        check(tok->decode(ids(get(c, "ids"))) == get(c, "decoded").as_string(), "arbitrary byte fallback decode");
    if (tok && root.find("encode_no_special")) for (const auto& c : get(root, "encode_no_special").as_array())
        check(tok->encode(get(c, "text").as_string(), false) == ids(get(c, "ids")), "encode with added-token matching off");
}
void streaming(const std::string& s, const std::string& reasoning, const std::string& content,
    const Json& calls, bool starts = false, bool incomplete = false) {
    for (std::size_t split = 0; split <= s.size() + 1; ++split) {
        g4::OutputParser parser(starts);
        std::string rd, cd; Json tc = Json::array();
        const auto consume = [&](const g4::OutputDelta& d) {
            rd += d.reasoning; cd += d.content;
            for (const auto& c : d.tool_calls.as_array()) tc.push(c);
            // Every published delta is complete UTF-8, validated through the shared asset parser.
            (void)strix::serve::parse_json_unique(Json(d.reasoning + d.content).dump());
        };
        if (split <= s.size()) {
            consume(parser.push(std::string_view(s).substr(0, split)));
            consume(parser.push(std::string_view(s).substr(split)));
        } else for (const auto c : s) consume(parser.push(std::string_view(&c, 1)));
        consume(parser.finish());
        const auto& out = parser.output();
        check(out.reasoning == reasoning && rd == reasoning, "stream reasoning at split " + std::to_string(split));
        check(out.content == content && cd == content, "stream content at split " + std::to_string(split));
        check(out.tool_calls.dump() == calls.dump() && tc.dump() == calls.dump(), "stream tool calls");
        check(out.unfinished_call == incomplete, "incomplete tool frame");
    }
}
void parser_tests() {
    const auto args = strix::serve::parse_json_unique(R"({"a":[null,true,false,-12,1.25],"nest":{"x":"a\\b\"c\n🙂<turn|><tool_call|>"}})");
    const auto argument_text = g4::format_argument(args, false);
    check(g4::parse_arguments(argument_text).dump() == args.dump(), "argument syntax inverse");
    const auto quoted = g4::format_argument(args, true);
    check(g4::parse_arguments(quoted).dump() == args.dump(), "quoted key syntax inverse");
    Json tc(Json::Object{{"id", "call_0"}, {"type", "function"}, {"function", Json(Json::Object{{"name", "inspect"}, {"arguments", args}})}});
    const Json calls(Json::Array{tc});
    g4::Output parsed; parsed.tool_calls = calls;
    const auto shared = g4::shared_output(parsed);
    check(shared.tool_calls.as_array()[0].find("arguments")->dump() == args.dump(), "shared output arguments remain an object");
    check(shared.call_complete == std::vector<bool>{true}, "shared complete call contract");
    streaming("<|channel>thought\nThink 🙂\n<channel|>Answer 東京<turn|>\n", "Think 🙂\n", "Answer 東京", Json::array());
    streaming("<|tool_call>call:inspect" + argument_text + "<tool_call|><|tool_response>", "", "", calls);
    streaming("Before<|tool_call>call:inspect" + argument_text + "<tool_call|>After<eos>", "", "BeforeAfter", calls);
    auto two = Json(calls.as_array());
    auto second = Json(tc.as_object()); second.set("id", "call_1"); two.push(second);
    streaming("<|tool_call>call:inspect" + argument_text + "<tool_call|><|tool_call>call:inspect" +
        argument_text + "<tool_call|>", "", "", two);
    streaming("More thought<channel|>Done", "More thought", "Done", Json::array(), true);
    streaming("<|channel>thought\nUnfinished", "Unfinished", "", Json::array());
    streaming("Text <|tool_call>call:inspect{a:<|\"|>unfinished", "", "Text ", Json::array(), false, true);
    streaming("Literal <to and <|tool_cal", "", "Literal <to and <|tool_cal", Json::array());
    streaming("<turn|>ignored", "", "", Json::array());
    rejects([] { g4::OutputParser p; p.push("<|channel>analysis\n"); p.finish(); }, "unknown channel rejected");
    rejects([] { g4::OutputParser p; p.push("<|tool_call>call:f{a:1,a:2}<tool_call|>"); }, "duplicate arguments rejected");
    rejects([] { g4::OutputParser p; p.push("<|tool_call>call:f{a:wat}<tool_call|>"); }, "malformed arguments rejected");
    rejects([] { g4::OutputParser p(false, 4); p.push("<|tool_call>call:f{}"); }, "frame limit enforced");
    rejects([] { g4::OutputParser p; p.finish(); p.push("x"); }, "push after finish rejected");
    rejects([] { (void)g4::parse_arguments("{}trailing"); }, "trailing arguments rejected");
    rejects([] { (void)g4::format_argument(Json("x<|\"|>y")); }, "ambiguous quote rejected");
    g4::OutputParser partial_channel; partial_channel.push("<|channel>thou"); partial_channel.finish();
    check(partial_channel.output().unfinished_control && partial_channel.output().content.empty(), "cut channel header is explicit");
}
void chat_tests() {
    check(g4::format_argument(Json::number_text("1e3")) == "1000.0", "Python float rendering");
    check(g4::format_argument(Json::number_text("-0")) == "0", "Python integer negative zero");
    const auto messages = strix::serve::parse_json(R"([{"role":"user","content":"Hi"}])");
    rejects([&] { (void)g4::chat_input(messages, strix::serve::parse_json(R"({"thinking_budget":2})")); }, "unknown option rejected");
    rejects([] { (void)g4::render_chat(g4::chat_input(strix::serve::parse_json(R"([{"role":"user","content":[{"type":"image_url","image_url":"image"}]}])"))); }, "multimodal rejected");
    rejects([] { (void)g4::render_chat(g4::chat_input(strix::serve::parse_json(R"([{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{}"}}]}])"))); }, "string arguments rejected");
    auto input = g4::chat_input(messages); input.enable_thinking = true;
    g4::Output partial; partial.reasoning = "Reason";
    const auto continuation = g4::continue_after_budget(input, partial, 32, 32, 128);
    check(continuation.remaining_tokens == 96, "budget remaining token count");
    check(continuation.prompt.ends_with("<|turn>model\n<|channel>thought\nReason\n<channel|>"), "budget closes thought and continues assistant");
    check(input.messages.size() == 1, "budget does not mutate caller");
    rejects([&] { (void)g4::continue_after_budget(input, partial, 32, 32, 32); }, "budget requires answer room");
    auto spoken = input; spoken.conversational_mode = true;
    const auto before = spoken.messages.dump(); (void)g4::render_chat(spoken);
    check(spoken.messages.dump() == before, "renderer does not mutate caller");
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc >= 3, "usage: serve-tests host|tokenizer FIXTURE_DIR [TOKENIZER_JSON]");
        const std::filesystem::path dir(argv[2]);
        const bool tokenizer = std::string(argv[1]) == "tokenizer";
        std::shared_ptr<const g4::Tokenizer> tok;
        if (tokenizer) {
            if (argc < 4 || !*argv[3]) { std::cout << "NO-REFERENCE: official tokenizer asset not configured\n"; return 77; }
            const auto start = std::chrono::steady_clock::now();
            tok = g4::Tokenizer::load(argv[3]);
            std::cout << "tokenizer_load_ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() << '\n';
            check(tok->vocab_size() == 262144 && tok->find("<bos>") == 2 && tok->find("<eos>") == 1, "vocab contract");
            check(tok->decode(std::vector<std::uint32_t>{2, 1}, true).empty(), "skip special decode");
            rejects([&] { (void)tok->encode(std::string("\xFF")); }, "invalid UTF-8 rejected");
            rejects([&] { (void)tok->token_bytes(262144); }, "invalid ID rejected");
            g4::Adapter adapter(tok);
            const auto prompt = adapter.prepare(g4::chat_input(strix::serve::parse_json(R"([{"role":"user","content":"Hi"}])")));
            const auto job = prompt.job(64);
            check(job.prompt == prompt.ids && job.stop_tokens == std::vector<std::uint32_t>{1, 106} && job.max_tokens == 64,
                "shared job preserves IDs and model stop tokens");
            check(!prompt.starts_in_reasoning && prompt.rendered.starts_with("<bos>"), "adapter prompt state");
        } else { parser_tests(); chat_tests(); }
        fixture(read(dir / "tok_goldens.json"), tok.get());
        fixture(read(dir / "extended.json"), tok.get());
        std::cout << "PASS gemma4 serve " << argv[1] << '\n'; return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
