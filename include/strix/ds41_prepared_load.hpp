#pragma once

#include "strix/deepseek41_model.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace strix::ds41::prepared {
inline constexpr std::size_t chunk_bytes = 16U << 20U;
using Bytes = std::span<const std::uint8_t>;
// Fifteen full-chunk checksums can overlap the consumed upload (256 MiB bounded staging).
inline constexpr std::size_t staging_slots = 16;
using Staging = std::array<std::span<std::uint8_t>, staging_slots>;
using Sink = std::function<void(Bytes, std::uint64_t)>;
// SHA-256, accelerated by OpenSSL when available; identical v1 digests on either backend.
const char* digest_backend() noexcept;
std::string digest(Bytes bytes);
// Diagnostic SHA256(concatenated ASCII chunk SHA256s), in byte order. The producer
// fills chunks serially on the caller's thread; bounded parallel SHA work overlaps it.
std::string chunked_digest(std::uint64_t bytes,
    const std::function<void(std::span<std::uint8_t>, std::uint64_t)>& read, Staging staging = {});
struct DenseBinding { std::string trellis, mix, sites, headout; };
DenseBinding dense_from_env();
// Content namespace: verified/cached model SHA, overlay SHA, trellis pack files and mapping,
// rank/world, EP vs TP identity, and packed vs duplicated TP down storage.
std::string name_space(const std::filesystem::path& model, const std::filesystem::path& overlay,
                       const deepseek41::TargetModel& target, unsigned rank, unsigned world, bool tp, bool packed, DenseBinding dense = dense_from_env());
// Snapshot detects source changes during preparation/loading, including restored mtime.
std::string source_stamp(const std::filesystem::path& path);

// Empty environment values are treated as unset. No home means no default root.
std::filesystem::path default_root(const std::string& root, const std::string& xdg, const std::string& home);
std::filesystem::path default_root();
std::string directory_name(const std::string& ns, unsigned rank, bool tp, bool packed);
// "0" disables, a nonempty explicit directory wins, otherwise scan immediate children.
// Reuses Reader validation (manifest + payload sizes only); never reads payload contents.
// Multiple valid copies are resolved in lexical path order; incomplete packs are ignored.
std::filesystem::path select_directory(const std::filesystem::path& root, const char* setting,
                                       const std::string& expected_namespace);
// Lazy namespace construction avoids source identity work when the root has no candidates.
std::filesystem::path select_directory(const std::filesystem::path& root, const char* setting,
                                       const std::function<std::string()>& expected_namespace);

class Writer {
public:
    Writer(const std::filesystem::path& destination, std::string ns);
    ~Writer();
    void tensor(const std::string& name, std::uint64_t bytes,
                const std::function<void(const Sink&)>& produce);
    void commit();
private:
    std::filesystem::path destination_, temporary_;
    std::string ns_, entries_;
    bool committed_ = false;
};
struct LoadStats { double read_s = 0, verify_s = 0; bool direct = true; };
class Reader {
public:
    Reader(const std::filesystem::path& directory, const std::string& expected_namespace);
    // Sixteen 16 MiB read/verify slots; all chunks are verified before consumption; any failure throws. The caller MUST overwrite
    // the entire destination with the legacy loader if a later chunk fails.
    void load(const std::string& name, std::uint64_t expected_bytes, const Sink& consume, LoadStats* stats = nullptr, Staging staging = {}) const;
private:
    struct Entry { std::uint64_t bytes = 0; std::vector<std::string> hashes; };
    std::filesystem::path directory_;
    std::map<std::string, Entry> entries_;
};
// Offline generation: only routed experts, final rank-local bytes; <= two source chunks +
// two output chunks regardless of model size. No HIP/device/communicator required.
void prepare(const std::filesystem::path& model, const std::filesystem::path& overlay,
             const std::filesystem::path& destination, unsigned rank, unsigned world, bool tp, bool packed,
             unsigned layers);
} // namespace strix::ds41::prepared
