#pragma once

#include "strix/model_manifest.hpp"
#include "strix/model_runtime_binding.hpp"
#include "strix/runtime_allocator.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

namespace strix {

class MappedModelFile {
public:
    MappedModelFile() = default;
    MappedModelFile(
        const MappedModelFile&) = delete;
    MappedModelFile& operator=(
        const MappedModelFile&) = delete;

    MappedModelFile(
        MappedModelFile&& other) noexcept;
    MappedModelFile& operator=(
        MappedModelFile&& other) noexcept;

    ~MappedModelFile();

    [[nodiscard]] static MappedModelFile open_read_only(
        const std::filesystem::path& path,
        std::string logical_name,
        AllocationGenerationSource& generations);

    [[nodiscard]] bool valid() const noexcept {
        return mapping_base_ != 0;
    }

    [[nodiscard]] std::uintptr_t base_address() const noexcept {
        return mapping_base_;
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return bytes_;
    }

    [[nodiscard]] std::uint64_t allocation_generation() const noexcept {
        return allocation_generation_;
    }

    [[nodiscard]] const ModelContentIdentity&
    content_identity() const noexcept {
        return content_identity_;
    }

    // Actual raw digest from the SAME full-file pass/FD used to establish the
    // framed content identity at open; invalid on empty/moved-from owners.
    [[nodiscard]] const ModelContentIdentity& file_sha256() const noexcept {return file_sha256_;}
    // Setup-only FD/stat witness against the original hash+map observation.
    // No rehash and no claim of protection against arbitrary privileged writers.
    void validate_unchanged() const;
    [[nodiscard]] const std::string&
    logical_name() const noexcept {
        return logical_name_;
    }

    [[nodiscard]] ModelRuntimeBinding
    build_runtime_binding(
        const ModelManifest& manifest) const;

    void read_at(
        std::uint64_t byte_offset,
        std::span<std::byte> destination) const;

private:
    void release() noexcept;

    int fd_ = -1;
    std::uintptr_t mapping_base_ = 0;
    std::uint64_t bytes_ = 0;
    std::uint64_t allocation_generation_ = 0;
    std::string logical_name_;
    ModelContentIdentity content_identity_;
    ModelContentIdentity file_sha256_;
    std::array<std::uint64_t,7> file_state_{};
};

}  // namespace strix
