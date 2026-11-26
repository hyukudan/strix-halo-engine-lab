#pragma once
// Pre-encoded resident rows for named checkpoint tensors. The v2 text header
// ("strix-lowbit-pack/v2 COUNT", "checkpoint HEX", COUNT "tensor ..." lines,
// "end") precedes payloads at 4096-aligned absolute offsets. The pack binds to
// the original checkpoint and source-byte fingerprints from pack_source.hpp.
// Its identity hashes the header text and complete payloads in entry order,
// read through the same open file as the resident slices.
// A Strix entry appends a tenth field in hex: its 16-byte sp::Header, including
// the table and base. The native fixed-table SP4 load exposes the base as the
// matrix outer exponent. Host grouped loads use the complete header.
#include "strix/weight/admission.hpp"
#include "strix/weight/host_lowbit.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"

#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace strix::weight {

struct PackMatrixSource;

inline constexpr std::string_view kLowbitPackTag = "strix-lowbit-pack/v2";

inline constexpr std::string_view kCheckedLowbitPackTag = "strix-lowbit-pack/v3";

enum class PackVersion { legacy, checked };

// Stable single-token labels. A zero calibration digest denotes no calibration.
struct PackCoding {
    std::string role, coder;
    Digest256 calibration{};
};

struct LowbitPackEntry {
    std::string tensor;
    FormatId storage;
    std::uint64_t rows{}, columns{};
    std::uint64_t offset{}, bytes{};
    Digest256 source{};  // PackMatrixSource fingerprint (raw SHA-256 for BF16)
    std::optional<sp::Header> sp_header{};  // Strix pack storages only
    PackCoding coding{};
    Digest256 payload{};
    unsigned group_size{};
    std::string_view scale_form{};
};

// Strix storages carry a header, and strix.sp4t restricts it to the fixed SP4 table.
// lowbit_header_defect: empty when `header` is admissible for `storage`, otherwise the first violated rule.
[[nodiscard]] bool lowbit_storage_has_header(FormatId storage) noexcept;
[[nodiscard]] std::string lowbit_header_defect(FormatId storage, const std::optional<sp::Header>& header);

class LowbitPackWriter {
public:
    // `checkpoint`: raw SHA-256 of the checkpoint file the rows and the statistics come from (required).
    explicit LowbitPackWriter(const Digest256& checkpoint);
    // Streaming variant: write payloads into an adjacent staging file and keep
    // only the entry metadata. finish() publishes a new destination.
    LowbitPackWriter(const Digest256& checkpoint, const std::string& path, std::span<const std::string_view> names,
                     PackVersion version = PackVersion::legacy);
    ~LowbitPackWriter();
    void add(std::string tensor, FormatId storage, std::uint64_t rows, std::uint64_t columns,
             const Digest256& source, std::vector<std::uint8_t> bytes,
             std::optional<sp::Header> sp_header = std::nullopt, PackCoding coding = {});
    Status write(const std::string& path) const;
    Status finish();

private:
    struct Item {
        LowbitPackEntry entry;
        std::vector<std::uint8_t> payload;
    };
    Digest256 checkpoint_{};
    std::vector<Item> items_;
    struct Staging;
    std::unique_ptr<Staging> staging_;
};

class LowbitPack {
public:
    // `checkpoint` is the raw SHA-256 of the checkpoint the engine loads; a pack coded from other bytes (or without a
    // checkpoint identity) is refused with stale_source. Framing, bounds and the identity are read from one open file.
    static Status open(const std::string& path, const Digest256& checkpoint, LowbitPack& out);
    // Host reference admission includes SP2/SP3/SP4 entries, with fixed or fitted tables.
    static Status open_host(const std::string& path, const Digest256& checkpoint, LowbitPack& out);
    // Standalone inspection, or when the caller has a separately trusted pack identity.
    static Status inspect(const std::string& path, LowbitPack& out);
    [[nodiscard]] PackVersion version() const { return version_; }
    [[nodiscard]] const LowbitPackEntry* entry(std::string_view tensor) const;
    // Index-only role selection. Map only bindings selected by the active profile.
    [[nodiscard]] std::span<const std::size_t> role_entries(std::string_view role) const;
    // V3 verifies the named tensor on first access, once across concurrent consumers.
    [[nodiscard]] Status map_tensor(std::string_view tensor, MappedBytes& out) const;
    [[nodiscard]] const std::vector<LowbitPackEntry>& entries() const { return entries_; }
    // V2 header/payload hash, or the V3 declared whole-file digest (verify_identity checks it).
    [[nodiscard]] const Digest256& identity() const { return identity_; }
    [[nodiscard]] const Digest256& checkpoint() const { return checkpoint_; }
    // Re-read the same opened source after staging to confirm the admitted content.
    [[nodiscard]] Status verify_identity() const;
    // Buffered integrity scan keeps the process working set independent of pack size.
    [[nodiscard]] Status verify_payloads() const;
    // No entry for the tensor: ok with nullopt. An entry that disagrees with the request or the complete source
    // payload: error (stale_source for a payload whose digest differs). `outer_exponent` (optional): the power of two
    // every output of the matrix carries outside its group scales (a Strix pack entry's base; 0 for q4_g64 / q6_g64).
    Status find(std::string_view tensor, FormatId storage, const RawTensor& source, std::optional<Slice>& out,
                std::int32_t* outer_exponent = nullptr) const;

    Status find(std::string_view tensor, FormatId storage, const PackMatrixSource& source,
                std::optional<Slice>& out) const;

private:
    static Status open_impl(const std::string&, const Digest256*, bool host, LowbitPack&);
    [[nodiscard]] Status verify_tensor(std::size_t index, MappedBytes& bytes) const;
    Status find_impl(std::string_view, FormatId, const PackMatrixSource&, std::optional<Slice>&, std::int32_t*) const;
    std::string path_;
    std::shared_ptr<const Source> file_;
    std::vector<LowbitPackEntry> entries_;
    std::map<std::string, std::size_t, std::less<>> entry_index_;
    std::map<std::string, std::vector<std::size_t>, std::less<>> roles_;
    struct Check {
        std::once_flag once;
        Status result{};
    };
    std::vector<std::shared_ptr<Check>> checks_;
    PackVersion version_ = PackVersion::legacy;
    std::size_t header_bytes_{};
    Digest256 identity_{}, checkpoint_{};
};

}  // namespace strix::weight
