#include "strix/direct_stream_control.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix {
namespace {
#if defined(__linux__) || defined(__APPLE__)
DirectStreamDeviceIdentity identity(const std::string& path, const struct stat& state) {
    return {path, static_cast<std::uint64_t>(state.st_dev), static_cast<std::uint64_t>(state.st_ino),
            static_cast<std::uint64_t>(state.st_rdev), S_ISCHR(state.st_mode)};
}
bool same_stat(const DirectStreamDeviceIdentity& expected, const struct stat& state) noexcept {
    return expected.filesystem_device == static_cast<std::uint64_t>(state.st_dev) &&
           expected.inode == static_cast<std::uint64_t>(state.st_ino) &&
           expected.character_device_number == static_cast<std::uint64_t>(state.st_rdev) &&
           expected.character_device == static_cast<bool>(S_ISCHR(state.st_mode));
}
#endif
void validate(const DirectStreamDeviceIdentity& expected, const DirectStreamControlPolicy& policy) {
    if (!expected.character_device || expected.canonical_path.empty() || expected.canonical_path.size() > 4096 ||
        !std::filesystem::path(expected.canonical_path).is_absolute() ||
        policy.maximum_frame_bytes == 0 || policy.maximum_frame_bytes > std::numeric_limits<std::uint32_t>::max() ||
        policy.maximum_io_calls == 0 || policy.maximum_io_calls > 1048576 ||
        policy.timeout.count() <= 0 || policy.timeout.count() > 30000) {
        throw std::invalid_argument("direct control requires an explicit character route and bounded frames/progress/deadline");
    }
}
#if defined(__linux__) || defined(__APPLE__)
std::uint64_t next_generation() {
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("direct control FD generations exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
}
DirectStreamIoResult io_result(ssize_t result) noexcept {
    if (result > 0) return {DirectStreamIoStatus::transferred, static_cast<std::uint64_t>(result)};
    if (result == 0) return {DirectStreamIoStatus::end_of_stream, 0};
    if (errno == EINTR) return {DirectStreamIoStatus::interrupted, 0};
    if (errno == EAGAIN || errno == EWOULDBLOCK) return {DirectStreamIoStatus::would_block, 0};
    return {DirectStreamIoStatus::failed, 0};
}
#endif
}  // namespace

DirectStreamDeviceIdentity inspect_direct_stream_device(const std::string& path) {
#if defined(__linux__) || defined(__APPLE__)
    const auto canonical = std::filesystem::canonical(path).string();
    struct stat state{};
    if (::lstat(canonical.c_str(), &state) != 0 || !S_ISCHR(state.st_mode)) {
        throw std::invalid_argument("direct control path is not an observable character device");
    }
    return identity(canonical, state);
#else
    static_cast<void>(path);
    throw std::runtime_error("POSIX direct control character-device discovery is unsupported on this platform");
#endif
}

#if defined(__linux__) || defined(__APPLE__)
struct PosixDirectStreamControlIo::Impl {
    struct Open { std::uint64_t generation; DirectStreamDeviceIdentity device; };
    std::unordered_map<int, Open> opened;
    const Open& require(DirectStreamFd fd) const {
        const auto found = opened.find(fd.descriptor);
        if (found == opened.end() || found->second.generation != fd.open_generation) {
            throw std::logic_error("direct control FD instance is stale or unknown");
        }
        return found->second;
    }
};
PosixDirectStreamControlIo::PosixDirectStreamControlIo() : impl_(std::make_unique<Impl>()) {}
PosixDirectStreamControlIo::~PosixDirectStreamControlIo() {
    while (!impl_->opened.empty()) close({impl_->opened.begin()->first, impl_->opened.begin()->second.generation});
}
DirectStreamFd PosixDirectStreamControlIo::open(const DirectStreamDeviceIdentity& expected) {
    validate(expected, {});
    const int fd = ::open(expected.canonical_path.c_str(), O_RDWR | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW);
    if (fd < 0) throw std::runtime_error("direct control character-device open failed");
    try {
        struct stat state{};
        if (::fstat(fd, &state) != 0 || identity(expected.canonical_path, state) != expected) {
            throw std::runtime_error("direct control opened device differs from explicit route observation");
        }
        const auto generation = next_generation();
        const auto inserted = impl_->opened.emplace(fd, Impl::Open{generation, expected});
        if (!inserted.second) throw std::logic_error("direct control FD ownership collision");
        return {fd, generation};
    } catch (...) { static_cast<void>(::close(fd)); throw; }
}
DirectStreamFdIdentity PosixDirectStreamControlIo::inspect(DirectStreamFd fd) {
    const auto& opened = impl_->require(fd);
    struct stat state{};
    if (::fstat(fd.descriptor, &state) != 0) throw std::runtime_error("direct control FD identity is inaccessible");
    return {identity(opened.device.canonical_path, state), opened.generation};
}
DirectStreamIoResult PosixDirectStreamControlIo::read(DirectStreamFd fd, std::span<std::byte> bytes) {
    static_cast<void>(impl_->require(fd));
    return io_result(::read(fd.descriptor, bytes.data(), bytes.size()));
}
DirectStreamIoResult PosixDirectStreamControlIo::write(DirectStreamFd fd, std::span<const std::byte> bytes) {
    static_cast<void>(impl_->require(fd));
    return io_result(::write(fd.descriptor, bytes.data(), bytes.size()));
}
DirectStreamWaitStatus PosixDirectStreamControlIo::wait(DirectStreamFd fd, bool writing, DirectStreamClock::time_point limit) {
    static_cast<void>(impl_->require(fd));
    const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(limit - now()).count();
    if (remaining <= 0) return DirectStreamWaitStatus::timed_out;
    const int timeout = static_cast<int>(std::min<std::int64_t>(remaining, 30000));
    struct pollfd state{fd.descriptor, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
    const int result = ::poll(&state, 1, timeout);
    if (result == 0) return DirectStreamWaitStatus::timed_out;
    if (result < 0) return errno == EINTR ? DirectStreamWaitStatus::interrupted : DirectStreamWaitStatus::failed;
    if ((state.revents & (POLLERR | POLLNVAL)) != 0) return DirectStreamWaitStatus::failed;
    if ((state.revents & state.events) != 0) return DirectStreamWaitStatus::ready;
    if ((state.revents & POLLHUP) != 0) return DirectStreamWaitStatus::end_of_stream;
    return DirectStreamWaitStatus::failed;
}
DirectStreamClock::time_point PosixDirectStreamControlIo::now() const { return DirectStreamClock::now(); }
void PosixDirectStreamControlIo::close(DirectStreamFd fd) noexcept {
    const auto found = impl_->opened.find(fd.descriptor);
    if (found == impl_->opened.end() || found->second.generation != fd.open_generation) return;
    struct stat state{};
    if (::fstat(fd.descriptor, &state) == 0 && same_stat(found->second.device, state)) {
        static_cast<void>(::close(fd.descriptor));
    }
    impl_->opened.erase(found);
}
#else
struct PosixDirectStreamControlIo::Impl {};
PosixDirectStreamControlIo::PosixDirectStreamControlIo() : impl_(std::make_unique<Impl>()) {}
PosixDirectStreamControlIo::~PosixDirectStreamControlIo() = default;
DirectStreamFd PosixDirectStreamControlIo::open(const DirectStreamDeviceIdentity&) {
    throw std::runtime_error("POSIX direct control opener is unsupported on this platform");
}
DirectStreamFdIdentity PosixDirectStreamControlIo::inspect(DirectStreamFd) {
    throw std::runtime_error("POSIX direct control FD inspection is unsupported on this platform");
}
DirectStreamIoResult PosixDirectStreamControlIo::read(DirectStreamFd, std::span<std::byte>) {
    throw std::runtime_error("POSIX direct control read is unsupported on this platform");
}
DirectStreamIoResult PosixDirectStreamControlIo::write(DirectStreamFd, std::span<const std::byte>) {
    throw std::runtime_error("POSIX direct control write is unsupported on this platform");
}
DirectStreamWaitStatus PosixDirectStreamControlIo::wait(DirectStreamFd, bool, DirectStreamClock::time_point) {
    throw std::runtime_error("POSIX direct control wait is unsupported on this platform");
}
DirectStreamClock::time_point PosixDirectStreamControlIo::now() const { return DirectStreamClock::now(); }
void PosixDirectStreamControlIo::close(DirectStreamFd) noexcept {}
#endif

DirectStreamControlByteStream::DirectStreamControlByteStream(DirectStreamControlIo& io,
    DirectStreamDeviceIdentity expected, DirectStreamControlPolicy policy)
    : io_(&io), expected_(std::move(expected)), policy_(policy) {
    validate(expected_, policy_);
    fd_ = io_->open(expected_);
    try {
        if (fd_.descriptor < 0 || fd_.open_generation == 0) throw std::runtime_error("direct control returned invalid owned FD");
        require_identity();
    } catch (...) { close_noexcept(); throw; }
}
DirectStreamControlByteStream DirectStreamControlByteStream::open_device(
    const DirectStreamDeviceIdentity& expected, DirectStreamControlPolicy policy) {
    auto owner = std::make_unique<PosixDirectStreamControlIo>();
    auto stream = open_with_io(*owner, expected, policy);
    stream.owned_io_ = std::move(owner);
    return stream;
}
DirectStreamControlByteStream DirectStreamControlByteStream::open_with_io(DirectStreamControlIo& io,
    const DirectStreamDeviceIdentity& expected, DirectStreamControlPolicy policy) {
    return DirectStreamControlByteStream(io, expected, policy);
}
DirectStreamControlByteStream::DirectStreamControlByteStream(DirectStreamControlByteStream&& other) noexcept
    : owned_io_(std::move(other.owned_io_)), io_(other.io_), expected_(std::move(other.expected_)),
      policy_(other.policy_), fd_(other.fd_), failed_(other.failed_) {
    other.io_ = nullptr; other.fd_ = {};
}
DirectStreamControlByteStream& DirectStreamControlByteStream::operator=(DirectStreamControlByteStream&& other) noexcept {
    if (this != &other) {
        close_noexcept();
        owned_io_ = std::move(other.owned_io_); io_ = other.io_; expected_ = std::move(other.expected_);
        policy_ = other.policy_; fd_ = other.fd_; failed_ = other.failed_;
        other.io_ = nullptr; other.fd_ = {};
    }
    return *this;
}
DirectStreamControlByteStream::~DirectStreamControlByteStream() { close_noexcept(); }
void DirectStreamControlByteStream::close_noexcept() noexcept {
    if (io_ != nullptr && fd_.descriptor >= 0) io_->close(fd_);
    io_ = nullptr; fd_ = {};
}
void DirectStreamControlByteStream::require_identity() {
    if (!valid() || failed_) throw std::logic_error("direct control stream is closed or failed");
    const auto current = io_->inspect(fd_);
    if (current.open_generation != fd_.open_generation || current.device != expected_) {
        throw std::runtime_error("direct control FD instance or device route changed");
    }
}
DirectStreamClock::time_point DirectStreamControlByteStream::deadline(DirectStreamClock::time_point& last_time) {
    last_time = io_->now();
    const auto duration = std::chrono::duration_cast<DirectStreamClock::duration>(policy_.timeout);
    if (last_time > DirectStreamClock::time_point::max() - duration) throw std::overflow_error("direct control deadline overflow");
    return last_time + duration;
}
void DirectStreamControlByteStream::check_time(DirectStreamClock::time_point limit, DirectStreamClock::time_point& last_time) {
    const auto current = io_->now();
    if (current < last_time || current >= limit) throw std::runtime_error("direct control frame deadline expired or clock changed");
    last_time = current;
}
void DirectStreamControlByteStream::transfer(std::span<std::byte> destination, std::span<const std::byte> source,
    bool writing, DirectStreamClock::time_point limit, DirectStreamClock::time_point& last_time, std::uint32_t& calls) {
    const auto bytes = writing ? source.size() : destination.size();
    std::size_t completed = 0;
    const auto budget = [&] {
        if (calls == policy_.maximum_io_calls) throw std::runtime_error("direct control frame IO budget exhausted");
        ++calls;
    };
    while (completed < bytes) {
        check_time(limit, last_time); require_identity(); check_time(limit, last_time); budget();
        const auto result = writing ? io_->write(fd_, source.subspan(completed)) : io_->read(fd_, destination.subspan(completed));
        require_identity(); check_time(limit, last_time);
        if (result.status == DirectStreamIoStatus::transferred) {
            if (result.bytes == 0 || result.bytes > bytes - completed) throw std::runtime_error("direct control IO returned impossible byte count");
            completed += static_cast<std::size_t>(result.bytes);
        } else if (result.bytes != 0) throw std::runtime_error("direct control failed IO carried a byte count");
        else if (result.status == DirectStreamIoStatus::interrupted) continue;
        else if (result.status == DirectStreamIoStatus::would_block) {
            budget();
            const auto ready = io_->wait(fd_, writing, limit);
            // A readiness result from a replaced FD or after expiry is not a
            // license to issue another read/write against the old operation.
            require_identity(); check_time(limit, last_time);
            if (ready == DirectStreamWaitStatus::ready || ready == DirectStreamWaitStatus::interrupted) continue;
            throw std::runtime_error("direct control readiness timed out, closed or failed");
        } else throw std::runtime_error("direct control stream closed or failed during frame");
    }
}
void DirectStreamControlByteStream::send(std::span<const std::byte> frame) {
    if (!valid() || failed_ || frame.empty() || frame.size() > policy_.maximum_frame_bytes) {
        throw std::invalid_argument("direct control outgoing frame exceeds bounds or stream is failed");
    }
    try {
        std::array<std::byte, 4> prefix{};
        const auto bytes = static_cast<std::uint32_t>(frame.size());
        for (unsigned shift = 0; shift < 32; shift += 8) prefix[shift / 8U] = static_cast<std::byte>((bytes >> shift) & 0xffU);
        DirectStreamClock::time_point last_time; const auto limit = deadline(last_time); std::uint32_t calls = 0;
        transfer({}, prefix, true, limit, last_time, calls);
        transfer({}, frame, true, limit, last_time, calls);
    } catch (...) { failed_ = true; throw; }
}
std::vector<std::byte> DirectStreamControlByteStream::receive() {
    if (!valid() || failed_) throw std::logic_error("direct control incoming stream is closed or failed");
    try {
        std::array<std::byte, 4> prefix{};
        DirectStreamClock::time_point last_time; const auto limit = deadline(last_time); std::uint32_t calls = 0;
        transfer(prefix, {}, false, limit, last_time, calls);
        std::uint32_t bytes = 0;
        for (unsigned shift = 0; shift < 32; shift += 8) bytes |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(prefix[shift / 8U])) << shift;
        if (bytes == 0 || bytes > policy_.maximum_frame_bytes) throw std::invalid_argument("direct control incoming length exceeds bounds");
        std::vector<std::byte> frame(bytes);
        transfer(frame, {}, false, limit, last_time, calls);
        return frame;
    } catch (...) { failed_ = true; throw; }
}

}  // namespace strix
