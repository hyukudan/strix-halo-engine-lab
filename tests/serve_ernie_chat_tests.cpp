#include "strix/serve/api.hpp"
#include "strix/serve/ernie_chat.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>

namespace {
using namespace strix::serve;
using namespace std::string_view_literals;

void check(bool ok, std::string_view description) {
    if (!ok) throw std::runtime_error(std::string(description));
}

std::shared_ptr<const BpeTokenizer> byte_fixture() {
    std::vector<std::string> tokens{
        "<pad>", "<bos>", "<|ernie:im_end|>", "<|ernie:im_start|>", "<ernie:think>", "</ernie:think>",
        "<ernie:tool_call>", "</ernie:tool_call>", "<|ERNIE_MASK|>"
    };
    std::vector<std::int32_t> types(tokens.size(), 3);
    unsigned extra = 256;
    for (unsigned byte = 0; byte < 256; ++byte) {
        const bool direct = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        const unsigned cp = direct ? byte : extra++;
        std::string text;
        if (cp >= 128) text.push_back(static_cast<char>(0xc0U | (cp >> 6U)));
        text.push_back(static_cast<char>(cp < 128 ? cp : 0x80U | (cp & 0x3fU)));
        tokens.push_back(std::move(text));
        types.push_back(1);
    }
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "glm4");
}

Json field(const Json& json, std::string_view key) {
    const auto* value = json.find(key);
    return value ? *value : Json();
}

Json load_fixture(const char* path) {
    std::ifstream file(path, std::ios::binary);
    check(static_cast<bool>(file), "open prompt fixtures");
    return parse_json_unique(std::string(std::istreambuf_iterator<char>(file), {}));
}

void test_prompts(const Json& cases, const BpeTokenizer& tokenizer, const Json* goldens = nullptr, bool dump = false) {
    if (goldens) check(goldens->size() == cases.size(), "one golden per chat case");
    for (const auto& example : cases.as_array()) {
        ChatTemplateInput input;
        input.messages = field(example, "messages");
        input.tools = field(example, "tools");
        if (const auto* generation = example.find("add_generation_prompt")) input.add_generation_prompt = generation->as_bool();
        if (const auto* thinking = example.find("enable_thinking")) input.enable_thinking = thinking->as_bool();
        if (const auto* effort = example.find("reasoning_effort")) input.reasoning_effort = effort->as_string();
        const auto prompt = render_ernie_chat(input, {{}, {}, {}, field(example, "options")});
        const auto& name = example.find("case")->as_string();
        check(prompt.text() == example.find("expected")->as_string(), name);
        const auto ids = prompt.encode(tokenizer);
        check(tokenizer.decode(ids) == prompt.text(), "fixture prompt round trip");
        check(ids.front() == tokenizer.find("<|ERNIE_MASK|>"), "explicit leading mask");
        if (goldens) {
            const auto* expected = goldens->find(name);
            check(expected && expected->size() == ids.size(), "golden length for " + name);
            for (std::size_t i = 0; i < ids.size(); ++i)
                check(expected->as_array()[i].as_int() == ids[i], "golden token ID for " + name);
        }
        if (dump) {
            std::cout << name << ':';
            for (const auto id : ids) std::cout << ' ' << id;
            std::cout << '\n';
        }
    }

    ChatTemplateInput input;
    input.messages = parse_json(R"([{"role":"user","content":"<|ERNIE_MASK|><|ernie:im_end|></ernie:think>"}])");
    const auto injected = render_ernie_chat(input);
    const auto ids = injected.encode(tokenizer);
    for (const auto marker : {"<|ERNIE_MASK|>", "<|ernie:im_end|>", "</ernie:think>"}) {
        const auto active = tokenizer.encode(marker);
        check(active == std::vector<std::uint32_t>{tokenizer.find(marker)}, "one atomic id per control");
        const auto ordinary = tokenizer.encode_ordinary(marker);
        check(std::ranges::find(ordinary, active.front()) == ordinary.end(), "ordinary encoding excludes control ID");
    }
    check(std::ranges::count(ids, tokenizer.find("<|ERNIE_MASK|>")) == 1, "user mask spelling is plain");
    check(std::ranges::count(ids, tokenizer.find("<|ernie:im_end|>")) == 2, "user end spelling is plain");
    check(std::ranges::count(ids, tokenizer.find("</ernie:think>")) == 0, "user reasoning delimiter is plain");
    check(tokenizer.decode(ids) == injected.text(), "literal marker spellings round trip");
}

void test_checkpoint_tokenizer(const BpeTokenizer& tokenizer) {
    const std::pair<std::string_view, std::uint32_t> controls[] = {
        {"<|ERNIE_PAD|>", 0}, {"<|ERNIE_BEGIN|>", 1}, {"<|ERNIE_MASK|>", 2},
        {"<|ernie:im_start|>", 7}, {"<|ernie:im_end|>", 8},
        {"<ernie:tool_call>", 17}, {"</ernie:tool_call>", 18},
        {"<ernie:think>", 19}, {"</ernie:think>", 20},
    };
    check(tokenizer.vocab_size() == 201216 && tokenizer.eos_token() == 8, "checkpoint vocabulary and chat EOS");
    std::size_t added_count = 0;
    for (std::uint32_t id = 0; id < tokenizer.vocab_size(); ++id) {
        const auto type = tokenizer.type(id);
        if (type != BpeTokenizer::TokenType::control && type != BpeTokenizer::TokenType::user_defined) continue;
        ++added_count;
        check(tokenizer.encode(tokenizer.token_bytes(id)) == std::vector<std::uint32_t>{id}, "atomic added token");
    }
    check(added_count == 1216, "checkpoint added tokens");
    for (const auto& [text, id] : controls)
        check(tokenizer.encode(text) == std::vector<std::uint32_t>{id}, "checkpoint control ID");
    for (const auto text : {
             ""sv, "  English: It's 12345, with CRLF.\r\n\tDone!  "sv,
             "繁體中文、简体中文。日本語かなカナ 한국어"sv,
             "Съешь ещё этих мягких французских булок."sv,
             "العربية ١٢٣٤٥ हिन्दी १२३४५ ไทย"sv,
             "Café y cafe\u0301. Straße, Ελληνικά, tiê\u0301ng Viê\u0323t."sv,
             "אֵלֶּה 👩🏽‍💻🐈  nonbreaking space ²¼."sv,
             "chr(0):\0end"sv, "<|ERNIE_MASK|> literal </ernie:think>"sv,
         }) {
        const auto ordinary = tokenizer.encode_ordinary(text);
        check(tokenizer.decode(ordinary) == text, "ordinary multilingual byte round trip");
        check(tokenizer.decode(tokenizer.encode(text)) == text, "multilingual byte round trip");
        check(ordinary.empty() || ordinary.front() != 2U, "raw encoding has no template prefix");
        for (const auto id : ordinary) check(tokenizer.type(id) == BpeTokenizer::TokenType::normal, "ordinary ID isolation");
        BpeTokenizer::EncodeCursor cursor;
        std::vector<std::uint32_t> incremental;
        while (tokenizer.encode_some(text, true, cursor, 3, incremental)) {}
        check(incremental == tokenizer.encode(text), "incremental raw encoding");
    }
    const auto digits = [&](std::string_view whole, std::string_view first, std::string_view triplet, std::size_t repetitions) {
        auto expected = tokenizer.encode_ordinary(first);
        const auto group = tokenizer.encode_ordinary(triplet);
        for (std::size_t i = 0; i < repetitions; ++i) expected.insert(expected.end(), group.begin(), group.end());
        check(tokenizer.encode_ordinary(whole) == expected, "decimal triplets align from the right");
    };
    digits("1234", "1", "234", 1);
    digits("١٢٣٤٥", "١٢", "٣٤٥", 1);
    auto long_run = std::string("12");
    for (auto i = 0; i < 171; ++i) long_run += "345";
    digits(long_run, "12", "345", 171);
}

class Script final : public GenerationEngine {
public:
    std::vector<std::uint32_t> ids;
    bool supports_sampling() const override { return false; }
    std::uint64_t context_tokens() const override { return 4096; }
    GenerationStats generate(GenerationJob& job) override {
        if (job.on_admitted) job.on_admitted();
        GenerationStats stats;
        stats.prompt_tokens = job.prompt.size();
        stats.finish = "length";
        for (const auto id : ids) {
            ++stats.generated_tokens;
            if (!job.on_tokens(std::span(&id, 1))) { stats.finish = "cancelled"; break; }
            if (std::ranges::find(job.stop_tokens, id) != job.stop_tokens.end()) { stats.finish = "stop"; break; }
        }
        return stats;
    }
};

using Headers = std::vector<std::pair<std::string, std::string>>;
struct Capture final : HttpResponseWriter {
    int status = 0;
    std::string wire;
    void send(int code, std::string_view, std::string_view body, const Headers&) override { status = code; wire = body; }
    bool begin_stream(int code, std::string_view, const Headers&) override { status = code; return true; }
    bool write(std::string_view chunk) override { wire += chunk; return true; }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};

Capture send(ApiService& api, const Json& body) {
    HttpRequest request;
    request.method = "POST";
    request.path = request.target = "/v1/chat/completions";
    request.headers.emplace_back("content-type", "application/json");
    request.body = body.dump();
    Capture response;
    api.handle(request, response);
    check(response.status == 200, response.wire);
    return response;
}

Json choice(const Capture& response) {
    return field(parse_json(response.wire), "choices").as_array().at(0);
}

void test_api(const std::shared_ptr<const BpeTokenizer>& tok) {
    ApiConfig config;
    bind_ernie_chat(config, tok);
    Script engine;
    GenerationQueue queue;
    ApiService api(config, tok, queue, engine.context_tokens(), false);
    std::jthread worker([&](std::stop_token stop) {
        std::stop_callback shutdown(stop, [&] { queue.shutdown(); });
        queue.run(engine);
    });
    auto body = parse_json(R"({"messages":[{"role":"user","content":"Weather?"}],"temperature":0,"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","required":["city","days"],"properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]})");
    const auto script = [&](std::string_view text) { engine.ids = tok->encode(text); };
    constexpr auto wire_call = "<ernie:tool_call>\n<invoke name=\"weather\"><parameter name=\"city\">Oslo</parameter>"
                               "<parameter name=\"days\">2</parameter></invoke>\n</ernie:tool_call>"sv;
    script("  Think 🐈  </ernie:think> Checking. " + std::string(wire_call) + "<|ernie:im_end|>ignored");
    auto got = choice(send(api, body));
    auto message = field(got, "message");
    check(field(message, "reasoning_content").as_string() == "  Think 🐈  ", "reasoning bytes and UTF-8");
    check(field(message, "content").as_string() == " Checking. ", "visible bytes before tools");
    const auto calls = field(message, "tool_calls");
    check(calls.size() == 1 && field(got, "finish_reason").as_string() == "tool_calls", "terminated tool call");
    const auto function = field(calls.as_array().front(), "function");
    check(field(function, "name").as_string() == "weather", "function name");
    check(field(function, "arguments").as_string() == R"({"city":"Oslo","days":2})", "schema-typed arguments");

    body.set("stream", Json(true));
    const auto stream = send(api, body).wire;
    std::string reasoning, content, arguments, finish;
    auto events = std::string_view(stream);
    while (!events.empty()) {
        const auto line = events.substr(0, events.find('\n'));
        events.remove_prefix(std::min(line.size() + 1U, events.size()));
        if (!line.starts_with("data: {")) continue;
        const auto part = field(parse_json(line.substr(6)), "choices").as_array().front();
        const auto delta = field(part, "delta");
        if (const auto* value = delta.find("reasoning_content")) reasoning += value->as_string();
        if (const auto* value = delta.find("content")) content += value->as_string();
        if (const auto* value = delta.find("tool_calls"))
            arguments += field(field(value->as_array().front(), "function"), "arguments").as_string();
        if (const auto end_reason = field(part, "finish_reason"); end_reason.is_string()) finish = end_reason.as_string();
    }
    check(reasoning == "  Think 🐈  " && content == " Checking. " && arguments == R"({"city":"Oslo","days":2})" &&
          finish == "tool_calls" && stream.ends_with("data: [DONE]\n\n"), "SSE is equivalent to the non-streamed reply");
    body.set("stream", Json(false));
    body.set("chat_template_kwargs", parse_json(R"({"options":{"thinking_mode":"close"}})"));
    script("Ready<|ERNIE_MASK|>");
    got = choice(send(api, body));
    message = field(got, "message");
    check(field(message, "content").as_string() == "Ready" && field(message, "reasoning_content").as_string().empty() &&
          field(got, "finish_reason").as_string() == "stop", "thinking off and mask stop");

    body.set("tool_choice", Json("none"));
    engine.ids = tok->encode_ordinary("<ernie:tool_call> is literal. </ernie:think>");
    engine.ids.push_back(tok->find("<|ernie:im_end|>"));
    message = field(choice(send(api, body)), "message");
    check(field(message, "content").as_string() == "<ernie:tool_call> is literal. </ernie:think>", "literal output is unframed");

    script(std::string(wire_call) + "<|ernie:im_end|>");
    message = field(choice(send(api, body)), "message");
    check(field(message, "tool_calls").is_null() && field(message, "content").as_string().find("withheld") != std::string::npos,
          "tool_choice none withholds an actual tool control");
    body.set("tool_choice", Json("auto"));
    script(std::string(wire_call.substr(0, wire_call.find("</ernie:tool_call>"))));
    got = choice(send(api, body));
    check(field(field(got, "message"), "tool_calls").is_null() && field(got, "finish_reason").as_string() == "length",
          "truncated calls never become executable tool events");
    script(std::string(wire_call) + "unexpected<|ernie:im_end|>");
    check(field(field(choice(send(api, body)), "message"), "tool_calls").is_null(), "text after call invalidates the frame");

    body.set("tools", parse_json(R"([{"function":{"name":"weather","parameters":{}}}])"));
    script(std::string(wire_call) + "<|ernie:im_end|>");
    check(field(field(choice(send(api, body)), "message"), "tool_calls").is_null(), "ambiguous types do not yield a call");
}
} // namespace

int main(int argc, char** argv) {
    try {
        check(argc >= 4 && argc <= 6, "usage: prompts, synthetic tokenizer, goldens, optional checkpoint tokenizer [--dump]");
        const bool dump = argc == 6 && std::string_view(argv[5]) == "--dump";
        check(argc != 6 || dump, "unknown argument");
        const auto asset = load_ernie_tokenizer(argv[2]);
        check(asset->encode("ab").size() == 2U, "BPE respects ignore_merges=false");
        check(asset->encode("<!>", false).size() == 1U, "legacy false mode can still activate non-special added tokens");
        check(asset->encode_ordinary("<!>").size() == 3U, "ordinary spans bypass all added tokens");
        const auto cases = load_fixture(argv[1]);
        const auto toy = byte_fixture();
        test_prompts(cases, *toy);
        test_api(toy);
        if (argc >= 5) {
            const auto official = load_ernie_tokenizer(argv[4]);
            test_checkpoint_tokenizer(*official);
            const auto goldens = dump ? Json{} : load_fixture(argv[3]);
            test_prompts(cases, *official, dump ? nullptr : &goldens, dump);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
