#include "strix/tcp_copied_payload_stream.hpp"

#if !defined(__linux__) && !defined(__APPLE__)
#error "TcpCopiedPayloadByteStream currently requires POSIX sockets."
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <utility>

namespace strix {
namespace {
using Clock = std::chrono::steady_clock;
void validate_policy(const TcpCopiedPayloadPolicy& policy) {
    if (!policy.maximum_frame_bytes || policy.maximum_frame_bytes > 64ULL * 1024ULL * 1024ULL ||
        !policy.maximum_io_calls || policy.maximum_io_calls > 1'000'000U ||
        policy.timeout.count() <= 0 || policy.timeout > std::chrono::minutes(1))
        throw std::invalid_argument("TCP payload requires bounded frame/call/deadline limits");
}
bool loopback(const sockaddr_storage& address) {
    if (address.ss_family == AF_INET)
        return (ntohl(reinterpret_cast<const sockaddr_in*>(&address)->sin_addr.s_addr) >> 24U) == 127U;
    if (address.ss_family == AF_INET6) {
        const auto& value = reinterpret_cast<const sockaddr_in6*>(&address)->sin6_addr;
        if (IN6_IS_ADDR_LOOPBACK(&value)) return true;
        return IN6_IS_ADDR_V4MAPPED(&value) && value.s6_addr[12] == 127U;
    }
    return false;
}
void check_loopback_tcp(int fd) {
    int type = 0;
    socklen_t length = sizeof(type);
    if (::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &length) != 0 || type != SOCK_STREAM)
        throw std::invalid_argument("TCP payload requires a connected stream socket");
#if defined(__linux__) && defined(SO_PROTOCOL)
    int protocol = 0;
    length = sizeof(protocol);
    if (::getsockopt(fd, SOL_SOCKET, SO_PROTOCOL, &protocol, &length) != 0 || protocol != IPPROTO_TCP)
        throw std::invalid_argument("TCP payload requires TCP socket protocol");
#endif
    sockaddr_storage local{}, peer{};
    length = sizeof(local);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &length) != 0 || !loopback(local))
        throw std::invalid_argument("TCP payload local endpoint must be loopback");
    length = sizeof(peer);
    if (::getpeername(fd, reinterpret_cast<sockaddr*>(&peer), &length) != 0 || !loopback(peer))
        throw std::invalid_argument("TCP payload peer endpoint must be connected loopback");
    const int enabled = 1;
    if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0)
        throw std::runtime_error("TCP payload NODELAY failed");
#if defined(SO_NOSIGPIPE)
    if (::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0)
        throw std::runtime_error("TCP payload NOSIGPIPE failed");
#endif
    const auto flags = ::fcntl(fd, F_GETFL);
    const auto descriptor_flags = ::fcntl(fd, F_GETFD);
    if (flags < 0 || descriptor_flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0 ||
        ::fcntl(fd, F_SETFD, descriptor_flags | FD_CLOEXEC) != 0)
        throw std::runtime_error("TCP payload nonblocking ownership setup failed");
}
void budget(std::uint32_t& calls, const TcpCopiedPayloadPolicy& policy, Clock::time_point deadline) {
    if (Clock::now() >= deadline) throw std::runtime_error("TCP payload absolute deadline expired");
    if (calls >= policy.maximum_io_calls) throw std::runtime_error("TCP payload IO call budget exhausted");
    ++calls;
}
} // namespace

TcpCopiedPayloadByteStream::TcpCopiedPayloadByteStream(int fd, TcpCopiedPayloadPolicy policy,
    std::uint64_t device, std::uint64_t inode) : fd_(fd), policy_(policy), device_(device), inode_(inode) {}
TcpCopiedPayloadByteStream TcpCopiedPayloadByteStream::adopt_connected_loopback(int fd, TcpCopiedPayloadPolicy policy) {
    try {
        validate_policy(policy);
        check_loopback_tcp(fd);
        struct stat observed{};
        if (::fstat(fd, &observed) != 0 || !S_ISSOCK(observed.st_mode))
            throw std::invalid_argument("TCP payload socket identity unavailable");
        return TcpCopiedPayloadByteStream(fd, policy, static_cast<std::uint64_t>(observed.st_dev),
                                         static_cast<std::uint64_t>(observed.st_ino));
    } catch (...) { if (fd >= 0) ::close(fd); throw; }
}
TcpCopiedPayloadByteStream::TcpCopiedPayloadByteStream(TcpCopiedPayloadByteStream&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), policy_(other.policy_), device_(other.device_), inode_(other.inode_), failed_(other.failed_) {}
TcpCopiedPayloadByteStream& TcpCopiedPayloadByteStream::operator=(TcpCopiedPayloadByteStream&& other) noexcept {
    if (this != &other) {
        close_noexcept(); fd_ = std::exchange(other.fd_, -1); policy_ = other.policy_;
        device_ = other.device_; inode_ = other.inode_; failed_ = other.failed_;
    }
    return *this;
}
TcpCopiedPayloadByteStream::~TcpCopiedPayloadByteStream() { close_noexcept(); }
void TcpCopiedPayloadByteStream::close_noexcept() noexcept { if (fd_ >= 0) ::close(std::exchange(fd_, -1)); }
void TcpCopiedPayloadByteStream::fail() noexcept { failed_ = true; close_noexcept(); }
void TcpCopiedPayloadByteStream::require_healthy() const {
    if (!valid()) throw std::logic_error("TCP payload connection is unavailable or poisoned");
}
void TcpCopiedPayloadByteStream::check_fd() const {
    struct stat observed{};
    if (::fstat(fd_, &observed) != 0 || !S_ISSOCK(observed.st_mode) ||
        static_cast<std::uint64_t>(observed.st_dev) != device_ || static_cast<std::uint64_t>(observed.st_ino) != inode_)
        throw std::runtime_error("TCP payload owned socket identity changed");
}
void TcpCopiedPayloadByteStream::wait(bool writing, Clock::time_point deadline, std::uint32_t& calls) {
    for (;;) {
        budget(calls, policy_, deadline); check_fd();
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (remaining < 0) throw std::runtime_error("TCP payload absolute deadline expired");
        pollfd descriptor{fd_, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
        const auto result = ::poll(&descriptor, 1, static_cast<int>(remaining + 1));
        if (result < 0 && errno == EINTR) continue;
        if (result < 0 || (descriptor.revents & (POLLERR | POLLNVAL)))
            throw std::runtime_error("TCP payload poll failed");
        if (result == 0) throw std::runtime_error("TCP payload absolute deadline expired");
        // POLLIN|POLLHUP can still contain a complete final frame. recv decides EOF.
        if (descriptor.revents & (writing ? POLLOUT : POLLIN)) return;
        if (descriptor.revents & POLLHUP) throw std::runtime_error("TCP payload peer closed");
    }
}
void TcpCopiedPayloadByteStream::read_all(std::span<std::byte> output, Clock::time_point deadline, std::uint32_t& calls) {
    std::size_t offset = 0;
    while (offset < output.size()) {
        wait(false, deadline, calls); budget(calls, policy_, deadline); check_fd();
        int flags = 0;
#if defined(MSG_DONTWAIT)
        flags |= MSG_DONTWAIT;
#endif
        const auto result = ::recv(fd_, output.data() + offset, output.size() - offset, flags);
        if (result < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (result <= 0) throw std::runtime_error("TCP payload receive failed or truncated");
        offset += static_cast<std::size_t>(result);
    }
    if (Clock::now() >= deadline) throw std::runtime_error("TCP payload absolute deadline expired");
}
void TcpCopiedPayloadByteStream::send(std::span<const std::byte> frame) {
    require_healthy();
    try {
        if (frame.empty() || frame.size() > policy_.maximum_frame_bytes)
            throw std::invalid_argument("TCP payload outbound frame exceeds bounded contract");
        std::array<std::byte, 4> prefix{};
        const auto length = static_cast<std::uint32_t>(frame.size());
        for (unsigned shift = 0; shift < 32; shift += 8)
            prefix[shift / 8U] = static_cast<std::byte>((length >> shift) & 255U);
        const auto deadline = Clock::now() + policy_.timeout;
        std::uint32_t calls = 0;
        std::size_t offset = 0;
        const auto total = prefix.size() + frame.size();
        while (offset < total) {
            wait(true, deadline, calls); budget(calls, policy_, deadline); check_fd();
            std::array<iovec, 2> slices{};
            std::size_t count = 0;
            if (offset < prefix.size()) slices[count++] = {prefix.data() + offset, prefix.size() - offset};
            const auto body_offset = offset > prefix.size() ? offset - prefix.size() : 0;
            slices[count++] = {const_cast<std::byte*>(frame.data() + body_offset), frame.size() - body_offset};
            msghdr message{}; message.msg_iov = slices.data();
            message.msg_iovlen = static_cast<decltype(message.msg_iovlen)>(count);
            int flags = 0;
#if defined(MSG_DONTWAIT)
            flags |= MSG_DONTWAIT;
#endif
#if defined(MSG_NOSIGNAL)
            flags |= MSG_NOSIGNAL;
#endif
            const auto result = ::sendmsg(fd_, &message, flags);
            if (result < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            if (result <= 0) throw std::runtime_error("TCP payload send failed");
            offset += static_cast<std::size_t>(result);
        }
        if (Clock::now() >= deadline) throw std::runtime_error("TCP payload absolute deadline expired");
    } catch (...) { fail(); throw; }
}
std::vector<std::byte> TcpCopiedPayloadByteStream::receive() {
    require_healthy();
    try {
        const auto deadline = Clock::now() + policy_.timeout;
        std::uint32_t calls = 0;
        std::array<std::byte, 4> prefix{};
        read_all(prefix, deadline, calls);
        std::uint32_t length = 0;
        for (unsigned shift = 0; shift < 32; shift += 8)
            length |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(prefix[shift / 8U])) << shift;
        if (!length || length > policy_.maximum_frame_bytes)
            throw std::invalid_argument("TCP payload inbound length rejected before allocation");
        std::vector<std::byte> frame(length);
        read_all(frame, deadline, calls);
        return frame;
    } catch (...) { fail(); throw; }
}
} // namespace strix
