#include "inkling2/inkling2_mini_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using strix::serve::BpeTokenizer;

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
    const auto tokenizer = strix::inkling2::load_mini_tokenizer(path);
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
    return fixture("Hola, ¿qué tal?", false, es) && fixture("café cafe\u0301", false, accents) &&
           fixture(" 👩🏽‍💻 你好 World\r\n🇪🇸", false, unicode) &&
           fixture("You've got 'EM\t\tX's ", false, contractions) &&
           fixture("<|begin_of_text|><|message_user|>hola<|end_message|><|message_model|>", true, rendered) &&
           fixture("<|message_user|>Hi", false, raw) && tokenizer->decode(rendered, true) == "hola" &&
           tokenizer->decode(std::array<std::uint32_t, 1>{127}) == "\uFFFD";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (check_splitter() && (argc == 1 || (argc == 2 && check_official(argv[1])))) return 0;
        std::cerr << "Inkling 2 Mini tokenizer fixture differs from the reference\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
    return 1;
}
