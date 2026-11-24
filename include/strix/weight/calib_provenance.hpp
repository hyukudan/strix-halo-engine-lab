#pragma once
// Calibration / held-out separation, host side, any model. A window is one run of token ids fed to the model as one
// sequence (a calibration chunk, or a gate case). Provenance records, for a set of windows, the identity of each window
// (SHA-256 of its ids as little-endian uint32) and the hash of every run of kShingleTokens consecutive ids at every
// offset. Two provenances overlap when they share a window identity or one run hash: any common stretch of at least
// kShingleTokens ids, and any identical window of any length, is found. The calibrated packer refuses statistics whose
// provenance overlaps the gate's held-out windows (its prompt manifest) or its proxy evaluation statistics.
#include "strix/weight/types.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace strix::weight::calib {

inline constexpr std::size_t kShingleTokens = 32;

struct WindowId {
    Digest256 digest{};
    std::uint64_t tokens = 0;
};

class Provenance {
public:
    void add_window(std::span<const std::uint32_t> ids);
    void merge(const Provenance& other);
    // Adds every case of a gate prompt manifest (rows "NAME TOKENS IDS_PATH [ARMS]", '#' comments; a relative IDS_PATH
    // is next to the manifest; the ids file is whitespace-separated). Throws std::runtime_error on a malformed row.
    void add_manifest(const std::string& path);

    [[nodiscard]] bool empty() const noexcept { return windows_.empty(); }
    [[nodiscard]] const std::vector<WindowId>& windows() const noexcept { return windows_; }
    [[nodiscard]] const std::vector<std::uint64_t>& shingles() const noexcept { return shingles_; }  // sorted, unique
    // Rebuilds from stored parts (file readers); the run hashes are sorted and deduplicated.
    static Provenance from_parts(std::vector<WindowId> windows, std::vector<std::uint64_t> shingles);

private:
    std::vector<WindowId> windows_;
    std::vector<std::uint64_t> shingles_;
};

[[nodiscard]] Digest256 window_identity(std::span<const std::uint32_t> ids);
// First 8 bytes (little-endian) of the SHA-256 of kShingleTokens ids as little-endian uint32.
[[nodiscard]] std::uint64_t shingle_hash(std::span<const std::uint32_t, kShingleTokens> ids);
// Empty when the two share no window and no run; otherwise what they share (for the refusal message).
[[nodiscard]] std::string overlap(const Provenance& a, const Provenance& b);

}  // namespace strix::weight::calib
