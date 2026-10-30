#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#if defined(__unix__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::deepseek41 {

// Nonblocking, process-owned ceiling reservations. Kernel locks are released on crash.
// One service user shares 1024 MiB across instances; no host memory is preallocated.
class PrefixHostLease {
public:
    explicit PrefixHostLease(std::uint64_t requested_mb, std::filesystem::path root = {}) {
#if defined(__unix__)
        if (root.empty()) root = std::filesystem::path("/tmp") / ("strix-ds41-cache-" + std::to_string(::getuid()));
        std::error_code ec;
        if (::mkdir(root.c_str(), 0700) != 0 && !std::filesystem::is_directory(root, ec)) return;
        struct stat st{};
        if (::lstat(root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != ::getuid() || (st.st_mode & 0077) != 0) return;
        const auto count = (std::min<std::uint64_t>(requested_mb, 1024) + 63U) / 64U;
        for (unsigned i = 0; i < 16 && fds_.size() < count; ++i) {
            const auto file = root / ("host-" + std::to_string(i));
            const int fd = ::open(file.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
            if (fd < 0) continue;
            if (::flock(fd, LOCK_EX | LOCK_NB) == 0) fds_.push_back(fd);
            else ::close(fd);
        }
        // Refuse rather than vary a service's capacity with a partial reservation.
        if (fds_.size() != count) release();
        else budget_mb_ = std::min<std::uint64_t>(requested_mb, 1024);
#else
        (void)root;
        (void)requested_mb;  // unsupported reservation mechanism: retention fails closed
#endif
    }
    ~PrefixHostLease() { release(); }
    PrefixHostLease(const PrefixHostLease&) = delete;
    PrefixHostLease& operator=(const PrefixHostLease&) = delete;
    [[nodiscard]] std::uint64_t budget_mb() const noexcept { return budget_mb_; }

private:
    void release() {
#if defined(__unix__)
        for (const auto fd : fds_) ::close(fd);
#endif
        fds_.clear();
        budget_mb_ = 0;
    }
    std::vector<int> fds_;
    std::uint64_t budget_mb_ = 0;
};

}  // namespace strix::deepseek41
