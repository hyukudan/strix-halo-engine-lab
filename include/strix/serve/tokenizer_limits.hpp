#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

// Bounds shared by the tokenizer asset loaders (BPE JSON / GGUF, MiniMax, H3 text encoder).
namespace strix::serve {

// Largest tokenizer text asset (tokenizer.json, vocab.json, merges.txt, tokenizer_config.json) the loaders read.
// Real files are tens of MiB at most.
inline constexpr std::uint64_t kMaxTokenizerFileBytes = std::uint64_t{128} << 20U;
// Largest token ID a loader accepts. The largest vocabulary the engine serves is well under 2^20 entries; IDs size
// the vocabulary tables, so an unbounded ID is an allocation request from the file.
inline constexpr std::int64_t kMaxTokenizerVocabId = (std::int64_t{1} << 22U) - 1;

// Reads a whole file of at most `limit` bytes; throws std::runtime_error (prefixed by `what`) otherwise.
[[nodiscard]] inline std::string read_tokenizer_file(const std::string& path, const char* what,
                                                     std::uint64_t limit = kMaxTokenizerFileBytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string(what) + ": cannot open " + path);
    std::string text;
    char buf[1 << 16];
    for (;;) {
        in.read(buf, sizeof buf);
        const auto n = static_cast<std::size_t>(in.gcount());
        if (n > limit - text.size()) throw std::runtime_error(std::string(what) + ": " + path + " exceeds the size limit");
        text.append(buf, n);
        if (!in) break;
    }
    if (!in.eof()) throw std::runtime_error(std::string(what) + ": cannot read " + path);
    return text;
}

}  // namespace strix::serve
