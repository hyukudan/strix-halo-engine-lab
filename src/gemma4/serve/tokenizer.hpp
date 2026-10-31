#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::gemma4::serve {

// Official tokenizer.json: Unicode BPE, space replacement and byte fallback.
// Encoding never inserts BOS/EOS; the chat renderer owns those boundaries.
// STRIX_G4_TOKENIZER_HEAP=0 selects the literal ranked-pair reference at load time.
class Tokenizer {
public:
    static std::shared_ptr<const Tokenizer> load(const std::filesystem::path& path);
    ~Tokenizer();
    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text, bool parse_special = true) const;
    [[nodiscard]] std::string decode(std::span<const std::uint32_t> ids, bool skip_special = false) const;
    // Byte-fallback pieces may end inside a UTF-8 scalar. Feed these to the shared Utf8StreamDecoder.
    [[nodiscard]] std::string token_bytes(std::uint32_t id) const;
    [[nodiscard]] std::uint32_t find(std::string_view text) const;
    [[nodiscard]] std::size_t vocab_size() const noexcept;
private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix::gemma4::serve
