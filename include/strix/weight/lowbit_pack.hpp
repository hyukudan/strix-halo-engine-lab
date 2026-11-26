#pragma once
// Low-bit pack sidecar: pre-encoded q4_g64 / q6_g64 resident rows for named checkpoint tensors, loaded instead of
// the load-time BF16 conversion. Layout: text header ("strix-lowbit-pack/v2 COUNT", "checkpoint HEX", COUNT
// "tensor ..." lines, "end"), padded with '\n' to a multiple of 4096 bytes, then payloads at 4096-aligned absolute
// offsets. A pack is bound to the exact checkpoint bytes it was coded from (raw SHA-256 of the checkpoint file: the
// calibration statistics depend on every tensor, packed or not) and each entry to the exact BF16 payload of its tensor
// (SHA-256 of every byte); its identity is the SHA-256 of the header and every payload byte, read through the same
// open file the resident slices then read.
// A Strix pack entry (strix.sp4t) adds a tenth field, the 16-byte tensor header (strix/weight/host_lowbit.hpp
// sp::Header) in hex: it is part of the identity, it must carry the storage's fixed table, and its base reaches the
// consumers as the matrix's outer exponent (find()).
#include "strix/weight/admission.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight {

inline constexpr std::string_view kLowbitPackTag = "strix-lowbit-pack/v2";

struct LowbitPackEntry {
    std::string tensor;
    FormatId storage;
    std::uint64_t rows{}, columns{};
    std::uint64_t offset{}, bytes{};
    Digest256 source{};  // SHA-256 of the tensor's complete BF16 payload (digest_slice)
    std::optional<sp::Header> sp_header{};  // Strix pack storages only
};

// A Strix pack storage carries a tensor header (strix.sp4t: 4 bits, the fixed 4-bit table); q4_g64 / q6_g64 carry none.
// lowbit_header_defect: empty when `header` is admissible for `storage`, otherwise the first violated rule.
[[nodiscard]] bool lowbit_storage_has_header(FormatId storage) noexcept;
[[nodiscard]] std::string lowbit_header_defect(FormatId storage, const std::optional<sp::Header>& header);

class LowbitPackWriter {
public:
    // `checkpoint`: raw SHA-256 of the checkpoint file the rows and the statistics come from (required).
    explicit LowbitPackWriter(const Digest256& checkpoint) : checkpoint_(checkpoint) {}
    void add(std::string tensor, FormatId storage, std::uint64_t rows, std::uint64_t columns,
             const Digest256& source, std::vector<std::uint8_t> bytes,
             std::optional<sp::Header> sp_header = std::nullopt);
    Status write(const std::string& path) const;

private:
    struct Item {
        LowbitPackEntry entry;
        std::vector<std::uint8_t> payload;
    };
    Digest256 checkpoint_{};
    std::vector<Item> items_;
};

class LowbitPack {
public:
    // `checkpoint` is the raw SHA-256 of the checkpoint the engine loads; a pack coded from other bytes (or without a
    // checkpoint identity) is refused with stale_source. Framing, bounds and the identity are read from one open file.
    static Status open(const std::string& path, const Digest256& checkpoint, LowbitPack& out);
    [[nodiscard]] const std::vector<LowbitPackEntry>& entries() const { return entries_; }
    // SHA-256 of the header text and payloads in entry order (without padding).
    [[nodiscard]] const Digest256& identity() const { return identity_; }
    [[nodiscard]] const Digest256& checkpoint() const { return checkpoint_; }
    // Re-read the same opened source after staging to confirm the admitted content.
    [[nodiscard]] Status verify_identity() const;
    // No entry for the tensor: ok with nullopt. An entry that disagrees with the request or the complete source
    // payload: error (stale_source for a payload whose digest differs). `outer_exponent` (optional): the power of two
    // every output of the matrix carries outside its group scales (a Strix pack entry's base; 0 for q4_g64 / q6_g64).
    Status find(std::string_view tensor, FormatId storage, const RawTensor& source, std::optional<Slice>& out,
                std::int32_t* outer_exponent = nullptr) const;

private:
    std::string path_;
    std::shared_ptr<const Source> file_;
    std::vector<LowbitPackEntry> entries_;
    std::size_t header_bytes_{};
    Digest256 identity_{}, checkpoint_{};
};

}  // namespace strix::weight
