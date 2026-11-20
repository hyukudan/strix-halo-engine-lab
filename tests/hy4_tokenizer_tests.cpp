// Hy4 tokenizer: the GGUF pre name `hy_v4` binds the shared byte-level BPE with the DeepSeek V3/V4 split rule
// (the model's tokenizer.json lists the same three Split regexes and a regex-free ByteLevel step; no normalizer).
//   strix-hy4-tokenizer-tests [HY4_GGUF]
// Always: hy_v4 pieces equal the DeepSeek rule's pieces on a mixed corpus, and hand-derived pieces of the rule.
// With the GGUF (only its header is read): vocabulary size, special-token ids and types, ids of short texts whose
// pieces are whole vocabulary entries (ids read from the official tokenizer.json vocabulary), the special-token
// split on and off, and decode(encode(text)) == text.
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/hy4_chat.hpp"
#include "strix/serve/json.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using strix::serve::BpeTokenizer;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    if (failures <= 40) std::cout << "FAIL " << what << "\n";
}

std::string ids_text(const std::vector<std::uint32_t>& ids) {
    std::ostringstream os;
    os << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) os << (i ? "," : "") << ids[i];
    os << ']';
    return os.str();
}

std::string pieces_text(const std::vector<std::string>& pieces) {
    std::string out = "[";
    for (std::size_t i = 0; i < pieces.size(); ++i) out += (i ? "|" : "") + pieces[i];
    return out + "]";
}

// English, Chinese, Japanese, code, digits, whitespace runs and the template's markers as plain text.
const std::vector<std::string> corpus = {
    "Hello, world! The quick brown fox jumps over 12345 lazy dogs.",
    "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81\xe4\xb8\xad\xe5\x9b\xbd\xe7\x9a\x84" "abc123",
    "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf \xe3\x82\xab\xe3\x82\xbf\xe3\x82\xab\xe3\x83\x8a",
    "def f(x):\n    return x+1  # comment\n\n\tif (a<=b) { y = a->c; }\r\n",
    "std::vector<int> v{1,2,3}; // 3.14159e-10\n",
    "  leading and trailing  \n\n\n  ",
    "<think:opensource>reasoning</think:opensource>answer<tool_calls:opensource>",
    "MixedCASE words, don't stop... 'quoted' \"double\" ($100) [list] 7x7=49",
};

// Pieces of the rule, derived by hand from the three regexes (each Split isolates its matches; the text between
// matches is a piece of its own).
const std::vector<std::pair<std::string, std::vector<std::string>>> rule_cases = {
    {"The world is 12", {"The", " world", " is", " ", "12"}},
    {"the \xe4\xb8\xad\xe5\x9b\xbd", {"the", " ", "\xe4\xb8\xad\xe5\x9b\xbd"}},
    {"x = 1", {"x", " =", " ", "1"}},
    {"The\n\nThe", {"The", "\n\n", "The"}},
    {"    return", {"   ", " return"}},
    {"12345", {"123", "45"}},
    {"import x\n", {"import", " x", "\n"}},
};

// Texts whose pieces are whole vocabulary entries; ids from the official tokenizer.json vocabulary.
const std::vector<std::pair<std::string, std::vector<std::uint32_t>>> id_cases = {
    {"The world is 12", {802, 3048, 341, 220, 757}},
    {"the \xe4\xb8\xad\xe5\x9b\xbd", {1728, 220, 2218}},
    {"x = 1", {87, 416, 220, 16}},
    {"The\n\nThe", {802, 268, 802}},
    {"    return", {326, 1315}},
    {"import x\n", {2012, 1142, 198}},
    {"\xe7\x9a\x84", {295}},
    {"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>user<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>The world"
     "<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>",
     {120000, 3717, 120001, 802, 3048, 120025}},
    {"<think:opensource>The</think:opensource>", {120029, 802, 120030}},
    {"<tool_calls:opensource><tool_call:opensource>x<arg_key:opensource>x</arg_key:opensource>",
     {120035, 120033, 87, 120040, 87, 120041}},
};

struct Named {
    const char* text;
    std::uint32_t id;
};
const std::vector<Named> specials = {
    {"<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>", 120000}, {"<\xef\xbd\x9chy_middle:opensource\xef\xbd\x9c>", 120001},
    {"<\xef\xbd\x9chy_pad:opensource\xef\xbd\x9c>", 120002},   {"<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>", 120025},
    {"<think:opensource>", 120029},  {"</think:opensource>", 120030},  {"<tool_call:opensource>", 120033},
    {"</tool_call:opensource>", 120034}, {"<tool_calls:opensource>", 120035}, {"</tool_calls:opensource>", 120036},
    {"<tool_response:opensource>", 120037}, {"</tool_response:opensource>", 120038},
    {"<\xef\xbd\x9creasoning_mode:opensource\xef\xbd\x9c>", 120039}, {"<arg_key:opensource>", 120040},
    {"</arg_key:opensource>", 120041}, {"<arg_value:opensource>", 120042}, {"</arg_value:opensource>", 120043},
};

void rule_checks() {
    for (const auto& text : corpus)
        check(BpeTokenizer::pre_tokenize(text, "hy_v4") == BpeTokenizer::pre_tokenize(text, "deepseek-v3"),
              "hy_v4 pieces differ from the DeepSeek rule: " + pieces_text(BpeTokenizer::pre_tokenize(text, "hy_v4")));
    for (const auto& [text, want] : rule_cases) {
        const auto got = BpeTokenizer::pre_tokenize(text, "hy_v4");
        check(got == want, "pieces " + pieces_text(got) + " want " + pieces_text(want));
    }
}

std::string respell(std::string text, const std::string& suffix) {
    const std::string official = ":opensource";
    for (auto at = text.find(official); at != std::string::npos; at = text.find(official, at + suffix.size()))
        text.replace(at, official.size(), suffix);
    return text;
}

void gguf_checks(const std::string& path) {
    const auto tok = BpeTokenizer::load_gguf(path);
    // The pack spells the special tokens with its own HYTK suffix (Q4_K_M GGUF ":6124c78e", official ":opensource"):
    // read it from token 120000 and respell the official texts of the cases with it.
    const std::string open = "<\xef\xbd\x9chy_start", close = "\xef\xbd\x9c>";
    const auto first = tok->token_bytes(120000U);
    check(first.size() > open.size() + close.size() && first.starts_with(open) && first.ends_with(close), "token 120000: " + first);
    const auto suffix = first.substr(open.size(), first.size() - open.size() - close.size());
    std::cout << "pack suffix " << suffix << "\n";
    check(tok->pre() == "hy_v4", "pre " + tok->pre());
    check(tok->vocab_size() == 120832U, "vocab " + std::to_string(tok->vocab_size()));
    check(tok->eos_token() == 120025U, "eos " + std::to_string(tok->eos_token()));
    for (const auto& s : specials) {
        const auto text = respell(s.text, suffix);
        check(tok->find(text) == s.id, "find " + text + " = " + std::to_string(tok->find(text)) +
                                             " (GGUF text of " + std::to_string(s.id) + ": " + tok->token_bytes(s.id) + ", type " +
                                             std::to_string(static_cast<int>(tok->type(s.id))) + ")");
        const auto t = tok->type(s.id);
        check(t == BpeTokenizer::TokenType::control || t == BpeTokenizer::TokenType::user_defined,
              "special type of " + text);
    }
    // Every added token (120000..120831) is matched whole; every base entry below them is an ordinary token.
    std::size_t base_special = 0;
    for (std::uint32_t id = 0; id < 120000U; ++id) {
        const auto t = tok->type(id);
        if (t == BpeTokenizer::TokenType::control || t == BpeTokenizer::TokenType::user_defined) ++base_special;
    }
    check(base_special == 0U, "special tokens below 120000: " + std::to_string(base_special));
    for (const auto& [official, want] : id_cases) {
        const auto got = tok->encode(respell(official, suffix), true);
        check(got == want, "encode " + ids_text(got) + " want " + ids_text(want));
    }
    const auto end_text = respell("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>", suffix);
    const auto plain = tok->encode(end_text, false);
    check(plain.size() > 1U, "parse_special=false still produced " + ids_text(plain));
    check(tok->decode(plain) == end_text, "plain decode of the end marker");
    for (const auto& text : corpus) {
        const auto ids = tok->encode(text, true);
        check(tok->decode(ids) == text, "round trip " + ids_text(ids));
    }
    // The model card's example rendered in the pack's dialect: its markers are exactly these special ids.
    strix::serve::ChatTemplateInput in;
    in.messages = strix::serve::parse_json(R"([{"role": "user", "content": "Hello! Can you briefly introduce yourself?"}])");
    in.reasoning_effort = "no_think";
    strix::serve::Hy4ChatDialect dialect;
    dialect.suffix = suffix;
    std::vector<std::uint32_t> markers;
    for (const auto id : tok->encode(strix::serve::render_hy4_chat(in, dialect), true))
        if (id >= 120000U) markers.push_back(id);
    const std::vector<std::uint32_t> want = {120000, 120001, 120039, 120025, 120000, 120001, 120025, 120000, 120001, 120029, 120030};
    check(markers == want, "rendered prompt markers " + ids_text(markers) + " want " + ids_text(want));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        rule_checks();
        if (argc > 1) gguf_checks(argv[1]);
        else std::cout << "Hy4 GGUF not given: pre-tokenizer rule checks only\n";
    } catch (const std::exception& error) {
        std::cout << "FAIL exception: " << error.what() << "\n";
        ++failures;
    }
    std::cout << (failures ? "FAILED " + std::to_string(failures) : std::string("OK")) << "\n";
    return failures ? 1 : 0;
}
