#pragma once

#include "strix/model_manifest.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>

// Weight overlay for requantization candidates: a small GGUF whose tensors replace tensors of the admitted
// model by name. Every overlay tensor must exist in the model with the same dimensions, storage type and
// byte size, so the plan, the kernels and the declared numerics are unchanged; only the bytes differ (e.g. a
// 4-bit codebook quantization of an attention projection stored as exact Q8_0 blocks). The overlay's identity
// is its `ds41.requant.identity` string plus the FNV-1a of its tensor bytes, printed by the runners.
namespace strix::ds41 {

class WeightOverlay {
public:
    // Opens and maps `path` and checks every tensor against `base` (throws std::invalid_argument on mismatch).
    WeightOverlay(const std::filesystem::path& path, const ModelManifest& base);
    ~WeightOverlay();
    WeightOverlay(const WeightOverlay&) = delete;
    WeightOverlay& operator=(const WeightOverlay&) = delete;

    struct Entry {
        std::uint64_t absolute_offset = 0, byte_size = 0;
    };
    [[nodiscard]] const Entry* find(const std::string& name) const;
    [[nodiscard]] std::span<const std::uint8_t> bytes(const Entry& e) const;
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] const std::string& identity() const noexcept { return identity_; }
    [[nodiscard]] std::uint64_t content_fnv() const noexcept { return fnv_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    // One line: path, identity, tensor count, overlay bytes and content FNV.
    [[nodiscard]] std::string describe() const;

private:
    std::filesystem::path path_;
    std::map<std::string, Entry> entries_;
    std::string identity_;
    std::uint64_t fnv_ = 0, bytes_ = 0, size_ = 0;
    int fd_ = -1;
    const std::uint8_t* base_ = nullptr;
};

}  // namespace strix::ds41
