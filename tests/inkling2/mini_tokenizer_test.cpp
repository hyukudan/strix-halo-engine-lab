#include "strix/serve/inkling2_chat.hpp"
#include "serve/o200k_pretokenizer.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strix::serve::BpeTokenizer;

bool check_malformed_assets() {
    struct Scratch {
        const std::filesystem::path directory{"inkling2-mini-malformed-tokenizer"};
        Scratch() {
            if (!std::filesystem::create_directory(directory))
                throw std::runtime_error("tokenizer fixture directory already exists");
        }
        ~Scratch() {
            std::error_code error;
            std::filesystem::remove(directory / "tokenizer.json", error);
            std::filesystem::remove(directory, error);
        }
    } scratch;
    const auto path = scratch.directory / "tokenizer.json";
    const auto pattern = strix::serve::o200k::kPattern;
    for (const std::string_view model_fields : {
             R"("merges":[])", R"("vocab":{"a":0})", R"("vocab":[],"merges":[])",
             R"("vocab":{"a":0},"merges":{})", R"("vocab":{"a":0},"merges":[])"}) {
        std::ofstream file(path, std::ios::binary);
        file << R"({"model":{"type":"BPE","ignore_merges":true,)" << model_fields
             << R"(},"decoder":{"type":"ByteLevel"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[)"
             << R"({"type":"Split","behavior":"Isolated","invert":false,"pattern":{"Regex":")";
        for (const char c : pattern) {
            if (c == '\\' || c == '"') file << '\\';
            file << c;
        }
        file << R"("}},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]}})";
        file.close();
        if (!file) throw std::runtime_error("could not write tokenizer fixture");

        const bool valid = model_fields == R"("vocab":{"a":0},"merges":[])";
        try {
            const auto tokenizer = BpeTokenizer::load_hf_json(path.string());
            if (!valid || tokenizer->encode("a") != std::vector<std::uint32_t>{0}) return false;
        } catch (const std::invalid_argument& error) {
            if (valid || std::string_view(error.what()).find("model.vocab") == std::string_view::npos) return false;
        }
    }
    constexpr BpeTokenizer::HfIdentity identity{"o200k", 1U, UINT32_MAX, "a", 0U};
    if (BpeTokenizer::load_hf_json(path.string(), identity)->find("a") != 0U) return false;
    for (const auto& mismatch : {BpeTokenizer::HfIdentity{"glm4", 1U, UINT32_MAX, "a", 0U},
                                 BpeTokenizer::HfIdentity{"o200k", 2U, UINT32_MAX, "a", 0U},
                                 BpeTokenizer::HfIdentity{"o200k", 1U, 0U, "a", 0U},
                                 BpeTokenizer::HfIdentity{"o200k", 1U, UINT32_MAX, "a", 1U}}) {
        try {
            (void)BpeTokenizer::load_hf_json(path.string(), mismatch);
            return false;
        } catch (const std::invalid_argument& error) {
            if (std::string_view(error.what()).find("checkpoint tokenizer identity") == std::string_view::npos) return false;
        }
    }
    return true;
}

bool check_splitter() {
    const std::vector<std::string> camel{"XMLParser", " camel", "Case", " HTTP", "2", " API", "123", "456", "7"};
    const std::vector<std::string> mixed{"You've", " got", " '", "EM", "\t", "\tX's", " "};
    if (BpeTokenizer::pre_tokenize("XMLParser camelCase HTTP2 API1234567", "o200k") != camel ||
        BpeTokenizer::pre_tokenize("You've got 'EM\t\tX's ", "o200k") != mixed) return false;

    using Type = BpeTokenizer::TokenType;
    const auto tokenizer = BpeTokenizer::from_parts({"a", "b", "ab", "«", "<|marker|>"},
        {static_cast<int>(Type::normal), static_cast<int>(Type::normal), static_cast<int>(Type::normal),
         static_cast<int>(Type::normal), static_cast<int>(Type::control)}, {}, "o200k");
    const std::array<std::uint32_t, 1> invalid_utf8{3};  // GPT-2 byte alphabet decodes this as raw 0xAB.
    return tokenizer->encode("ab") == std::vector<std::uint32_t>{2} &&
           tokenizer->encode("ab<|marker|>a") == std::vector<std::uint32_t>{2, 4, 0} &&
           tokenizer->token_bytes(3) == "\xAB" && tokenizer->decode(invalid_utf8) == "\uFFFD";
}

bool check_official(const std::string& path) {
    using namespace strix::serve;
    const auto tokenizer = BpeTokenizer::load_hf_json(path, inkling2_mini_tokenizer_identity);
    const auto fixture = [&](std::string_view text, bool special, std::span<const std::uint32_t> expected) {
        const auto encoded = tokenizer->encode(text, special);
        if (!std::ranges::equal(encoded, expected) || tokenizer->decode(encoded) != text) return false;
        for (const std::size_t chunk : {1U, 3U, 7U, 16U}) {
            std::vector<std::uint32_t> incremental;
            BpeTokenizer::EncodeCursor cursor;
            while (tokenizer->encode_some(text, special, cursor, chunk, incremental)) {}
            if (encoded != incremental) return false;
        }
        return true;
    };
    const std::array<std::uint32_t, 6> es{66776, 11, 150873, 69592, 150666, 30};
    const std::array<std::uint32_t, 4> accents{85593, 220, 34681, 69195};
    const std::array<std::uint32_t, 12> unicode{220, 74269, 69389, 70363, 111037, 220, 11739, 23643, 151945,
                                                68247, 71273, 71511};
    const std::array<std::uint32_t, 12> contractions{1735, 6, 290, 151325, 196984, 146039, 197, 197, 55, 6, 82, 220};
    const std::array<std::uint32_t, 5> rendered{200028, 200000, 80377, 200010, 200001};
    const std::array<std::uint32_t, 8> raw{27, 91, 65752, 62, 2986, 91, 29, 4896};
    // Captured with load_mini_tokenizer at 9b3b6227c, before replacing the private adapter.
    const std::array<std::uint32_t, 8> mixed_scripts{88223, 152000, 57629, 114903, 220, 70904, 81554, 22};
    const std::array<std::uint32_t, 8> whitespace{220, 152376, 82, 197, 197, 406, 117762, 149859};
    const std::array<std::uint32_t, 15> literal{27, 91, 65752, 62, 2986, 91, 29, 65825, 27, 91, 406, 62, 65752, 91, 29};
    const std::array<std::uint32_t, 16> chat{200002, 200004, 34182, 279, 154554, 150628, 25, 220,
                                           15, 13, 24, 200010, 200000, 200004, 80377, 200010};
    const auto chat_text = render_inkling2_chat(parse_json(R"([{"role":"user","content":"hola"}])"));
    return fixture("Hola, ¿qué tal?", false, es) && fixture("café cafe\u0301", false, accents) &&
           fixture(" 👩🏽‍💻 你好 World\r\n🇪🇸", false, unicode) &&
           fixture("You've got 'EM\t\tX's ", false, contractions) &&
           fixture("<|begin_of_text|><|message_user|>hola<|end_message|><|message_model|>", true, rendered) &&
           fixture("Straße Δelta中文 1234567", false, mixed_scripts) && fixture("  tabs\t\tend\r\n\n  ", false, whitespace) &&
           fixture("<|message_user|>literal<|end_message|>", false, literal) && fixture(chat_text, true, chat) &&
           fixture("<|message_user|>Hi", false, raw) && tokenizer->decode(rendered, true) == "hola" &&
           tokenizer->decode(std::array<std::uint32_t, 1>{127}) == "\uFFFD";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (check_splitter() && check_malformed_assets() &&
            (argc == 1 || (argc == 2 && check_official(argv[1])))) return 0;
        std::cerr << "Inkling 2 Mini tokenizer fixture differs from the reference\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
    return 1;
}
