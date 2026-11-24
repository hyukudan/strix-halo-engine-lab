#pragma once
// Low-bit pack sidecar: pre-encoded q4_g64 / q6_g64 resident rows for named checkpoint tensors, loaded instead of
// the load-time BF16 conversion. Layout: text header ("strix-lowbit-pack/v1 COUNT", COUNT "tensor ..." lines, "end"),
// padded with '\n' to a multiple of 4096 bytes, then payloads at 4096-aligned absolute offsets.
#include "strix/weight/admission.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight {

inline constexpr std::string_view kLowbitPackTag = "strix-lowbit-pack/v1";

struct LowbitPackEntry {
    std::string tensor;
    FormatId storage;
    std::uint64_t rows{}, columns{};
    std::uint64_t offset{}, bytes{};
    Digest256 source{};
};

// SHA-256 over the little-endian 8-byte slice length, its first and its last min(65536, bytes) bytes.
Status lowbit_source_fingerprint(const Slice& source, Digest256& out);

class LowbitPackWriter {
public:
    void add(std::string tensor, FormatId storage, std::uint64_t rows, std::uint64_t columns,
             const Digest256& source, std::vector<std::uint8_t> bytes);
    Status write(const std::string& path) const;

private:
    struct Item {
        LowbitPackEntry entry;
        std::vector<std::uint8_t> payload;
    };
    std::vector<Item> items_;
};

class LowbitPack {
public:
    static Status open(const std::string& path, LowbitPack& out);
    [[nodiscard]] const std::vector<LowbitPackEntry>& entries() const { return entries_; }
    // SHA-256 of the header text (through the "end" line, without padding).
    [[nodiscard]] const Digest256& identity() const { return identity_; }
    // No entry for the tensor: ok with nullopt. An entry that disagrees with the request or the source: error.
    Status find(std::string_view tensor, FormatId storage, const RawTensor& source, std::optional<Slice>& out) const;

private:
    std::string path_;
    std::shared_ptr<const Source> file_;
    std::vector<LowbitPackEntry> entries_;
    Digest256 identity_{};
};

}  // namespace strix::weight
