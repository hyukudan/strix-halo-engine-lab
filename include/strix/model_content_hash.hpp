#pragma once
#include "strix/file_snapshot.hpp"

#include "strix/model_runtime_binding.hpp"

#include <filesystem>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strix {

class ModelContentChanged : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class FileHashPolicy { cached, full };
struct FileHashCacheStats { std::uint64_t hits = 0, misses = 0; };
[[nodiscard]] FileHashCacheStats file_hash_cache_stats() noexcept;

struct FileContentHashStats { bool cache_hit=false; };

struct ModelContentFile {
    std::string logical_name;
    std::filesystem::path path;
};

[[nodiscard]] ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path);

// Explicit full reads bypass the persistent metadata cache without changing process settings.
[[nodiscard]] ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path, FileHashPolicy policy);
[[nodiscard]] ModelContentIdentity sha256_file_content(const std::filesystem::path&, const FileIdentity&);

// Completed bytes during uncached file hashing; a cache hit reports completion.
using FileHashProgress=std::function<void(std::uint64_t,std::uint64_t)>;
[[nodiscard]] ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path, FileContentHashStats* stats);
[[nodiscard]] ModelContentIdentity sha256_file_content(
    const std::filesystem::path& path, FileContentHashStats* stats,
    const FileHashProgress& progress);

[[nodiscard]] ModelContentIdentity sha256_bytes(
    std::span<const std::byte> bytes);

// The same digest as sha256_bytes from the fastest provider for in-memory bytes: the CPU's SHA extensions (x86 SHA-NI,
// detected at run time), else the portable implementation. sha256_fast_provider() names the one in use.
[[nodiscard]] ModelContentIdentity sha256_bytes_fast(
    std::span<const std::byte> bytes);
[[nodiscard]] const char* sha256_fast_provider() noexcept;
// Incremental form of sha256_bytes_fast: the digest of the concatenation of every update().
class Sha256Fast {
public:
    Sha256Fast();
    ~Sha256Fast();
    Sha256Fast(const Sha256Fast&) = delete;
    Sha256Fast& operator=(const Sha256Fast&) = delete;
    void update(std::span<const std::byte> bytes);
    [[nodiscard]] ModelContentIdentity finalize();

private:
    struct Impl;
    Impl* impl_;
};

[[nodiscard]] ModelContentIdentity build_model_content_identity(
    std::span<const ModelContentFile> files);

// One stable full-file pass yields both the raw file digest and its existing
// single-file STRIX-MODEL-CONTENT-V1 identity. Neither digest aliases the other.
struct SingleModelContentIdentity {
    ModelContentIdentity file_sha256,content_identity;
    std::uint64_t file_bytes=0;
};
[[nodiscard]] SingleModelContentIdentity hash_single_model_content_file(const ModelContentFile&);
[[nodiscard]] SingleModelContentIdentity hash_single_model_content_file(
    const ModelContentFile&, FileHashPolicy policy);
[[nodiscard]] std::string model_content_identity_hex(
    const ModelContentIdentity& identity);

// The content identity of a weight pack (the "pack=" scope of a promotion key, profile::LadderScope): the
// STRIX-MODEL-CONTENT-V1 identity of its files. A regular file is one file named by its file name; a directory is
// every regular file below it named by its relative generic path, skipping any path component that starts with '.'
// (download caches and editor state are not the pack). Uses the persistent file digest cache (a warm start reads
// file metadata only). std::invalid_argument when the path is neither, or the directory holds no file.
[[nodiscard]] std::vector<ModelContentFile> pack_content_files(const std::filesystem::path& pack);
[[nodiscard]] ModelContentIdentity pack_content_identity(const std::filesystem::path& pack);

}  // namespace strix
