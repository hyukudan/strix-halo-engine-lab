// Pre-tokenizer rules: every accepted `pre` key splits the probe texts exactly as recorded from the previous
// per-name code paths (golden pieces, '|' joined), and an unknown key is refused with the same message.
//   strix-pre-split-rule-tests          check
//   strix-pre-split-rule-tests --print  emit the golden table
#include "strix/serve/bpe_tokenizer.hpp"

#include <cstring>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using strix::serve::BpeTokenizer;

constexpr const char* kKeys[] = {"qwen35", "joyai-llm", "deepseek-v3", "hy_v4", "glm4", "o200k"};
constexpr const char* kProbes[] = {
    "Hello, world! It's 12345 items.", "12345678", "e\xCC\x81x cafe\xCC\x81", "don'T 'LL x'VE 'ré", "a  \n\n b   c\r\n",
    "  trailing   ", "price $12.50 (approx)", "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E\xE3\x81\xAE\xE3\x83\x86 \xE6\xBC\xA2\xE5\xAD\x97" "123",
    "x=3+4;\tfoo_bar()", "\xC3\x9Cn\xC3\xAF" "code \xC7\x85 \xE1\xBE\xBC", "...!!!\n\n?",
    "\xD9\xA3\xD9\xA4\xD9\xA5\xD9\xA6\xD9\xA7 \xE2\x91\xA0 \xC2\xBD", "invalid \xFF\xFE bytes", "CamelCaseWord HTTPServer don'tStop",
    "'", "", "1 22 333 4444 55555", "\xC2\xA0nb\xC2\xA0sp \xE2\x80\xA8ls"};

std::string pieces(std::string_view text, std::string_view key) {
    std::string out;
    for (const auto& piece : BpeTokenizer::pre_tokenize(text, key)) { out += piece; out += '|'; }
    return out;
}
std::string escaped(const std::string& s) {
    std::string out;
    static constexpr char hex[] = "0123456789ABCDEF";
    for (unsigned char c : s) {
        if (c == 0x5CU || c == 0x22U) { out += '\\'; out += static_cast<char>(c); }
        else if (c == 0x0AU) out += "\\n";
        else if (c == 0x0DU) out += "\\r";
        else if (c == 0x09U) out += "\\t";
        else if (c >= 0x20U && c < 0x7FU) out += static_cast<char>(c);
        else { out += "\\x"; out += hex[c >> 4U]; out += hex[c & 15U]; out += "\"\""; }
    }
    return out;
}

#include "pre_split_rule_golden.inc"
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--print") == 0) {
        for (const char* key : kKeys)
            for (const char* probe : kProbes) std::cout << "\"" << escaped(pieces(probe, key)) << "\",\n";
        return 0;
    }
    int failures = 0;
    std::size_t at = 0;
    for (const char* key : kKeys)
        for (const char* probe : kProbes) {
            const auto got = pieces(probe, key);
            if (at >= std::size(kGolden) || got != kGolden[at]) { ++failures; std::cerr << "mismatch key=" << key << " probe#" << (at % std::size(kProbes)) << '\n'; }
            ++at;
        }
    if (at != std::size(kGolden)) { ++failures; std::cerr << "golden size differs\n"; }
    try { (void)BpeTokenizer::pre_tokenize("x", "no-such-rule"); ++failures; std::cerr << "unknown key accepted\n"; }
    catch (const std::invalid_argument& e) { if (std::string(e.what()) != "tokenizer: unknown pre-tokenizer") { ++failures; std::cerr << "unknown-key text changed\n"; } }
    try { (void)BpeTokenizer::from_parts({"a"}, {1}, {}, "no-such-rule"); ++failures; std::cerr << "from_parts accepted unknown key\n"; }
    catch (const std::invalid_argument& e) {
        if (std::string(e.what()) != "tokenizer: unsupported byte-level BPE pre-tokenizer no-such-rule") { ++failures; std::cerr << "from_parts text changed\n"; }
    }
    std::cout << (failures ? "FAIL" : "PASS") << " pre-split rules (" << at << " cases)\n";
    return failures ? 1 : 0;
}
