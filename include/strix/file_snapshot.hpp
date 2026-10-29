#pragma once
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#ifdef __linux__
#include <sys/stat.h>
#endif
namespace strix {
struct FileIdentity {
    std::uint64_t dev = 0, ino = 0, size = 0;
    std::int64_t mtime_sec = 0, mtime_ns = 0, ctime_sec = 0, ctime_ns = 0;
    bool operator==(const FileIdentity&) const = default;
};
#ifdef __linux__
inline FileIdentity file_identity(const struct stat& s) {
    return {s.st_dev, s.st_ino, static_cast<std::uint64_t>(s.st_size),
            s.st_mtim.tv_sec, s.st_mtim.tv_nsec, s.st_ctim.tv_sec, s.st_ctim.tv_nsec};
}
#endif
inline FileIdentity fd_identity([[maybe_unused]] int fd) {
#ifdef __linux__
    struct stat s{};
    if (::fstat(fd, &s) == 0 && S_ISREG(s.st_mode)) return file_identity(s);
#endif
    throw std::runtime_error("model files changed during startup");
}
struct LoadedFile {
    std::filesystem::path path, resolved_path; // resolved path is diagnostic only
    FileIdentity identity;
    int fd = -1; // borrowed; -1 means the loader closed its descriptor
};
using LoaderSnapshot = std::vector<LoadedFile>;
inline LoadedFile loaded_file(const std::filesystem::path& path, int fd, bool retained = false) {
    const auto identity = fd_identity(fd);
    std::error_code ec;
    auto resolved = std::filesystem::canonical(path, ec);
    return {std::filesystem::absolute(path), resolved, identity, retained ? fd : -1};
}
inline void verify_loader_snapshot(const LoaderSnapshot& files) {
    for ([[maybe_unused]] const auto& file : files) {
#ifdef __linux__
        struct stat s{};
        const int result = file.fd >= 0 ? ::fstat(file.fd, &s) : ::stat(file.path.c_str(), &s);
        if (result == 0 && S_ISREG(s.st_mode) && file_identity(s) == file.identity) continue;
#endif
        throw std::runtime_error("model files changed during startup");
    }
}
} // namespace strix

namespace strix {
// Shared by startup loaders, including their worker threads. No descriptor is
// borrowed: admission conservatively rechecks original paths even for live fds.
class LoaderCapture {
    inline static std::mutex mutex_;
    inline static std::map<std::filesystem::path, std::vector<FileIdentity>> files_;
    inline static std::atomic<bool> enabled_{false};
public:
    explicit LoaderCapture(bool enable = true) {
        std::lock_guard lock(mutex_);
        files_.clear();
        enabled_ = enable;
    }
    ~LoaderCapture() { enabled_ = false; }
    LoaderCapture(const LoaderCapture&) = delete;
    static bool active() { return enabled_; }
    static void record(const std::filesystem::path& path, int fd) {
#ifdef __linux__
        if (!enabled_) return;
        const auto file = loaded_file(path, fd);
        std::lock_guard lock(mutex_);
        auto& identities = files_[file.path];
        for (const auto& old : identities) if (old == file.identity) return;
        identities.push_back(file.identity);
#else
        (void)path; (void)fd;
#endif
    }
    static LoaderSnapshot snapshot() {
        std::lock_guard lock(mutex_);
        LoaderSnapshot out;
        for (const auto& [path, identities] : files_)
            for (const auto& identity : identities) out.push_back({path, {}, identity, -1});
        return out;
    }
};
} // namespace strix
