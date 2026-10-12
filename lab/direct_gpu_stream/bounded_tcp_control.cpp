#include "bounded_tcp_control.hpp"
#include <array>
#include <algorithm>
#include <cerrno>
#include <limits>
#include <stdexcept>

#if defined(__linux__)
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::transport {
namespace {
constexpr std::uint32_t loopback = 0x7f000001U;
void validate(const CommissioningTcpPolicy& policy) {
    if (!policy.port || !policy.maximum_frame_bytes || policy.maximum_frame_bytes > 2048 || !policy.maximum_io_calls || policy.maximum_io_calls > 100000 ||
        policy.timeout.count() <= 0 || policy.timeout > std::chrono::minutes{1}) throw std::invalid_argument("commissioning loopback TCP requires explicit bounded port/frame/IO/deadline");
}
DirectStreamClock::time_point deadline(DirectStreamClock::time_point now, std::chrono::milliseconds timeout) {
    const auto duration = std::chrono::duration_cast<DirectStreamClock::duration>(timeout);
    if (now > DirectStreamClock::time_point::max() - duration) throw std::overflow_error("commissioning control deadline overflow");
    return now + duration;
}
#if defined(__linux__)
DirectStreamWaitStatus native_wait(int fd, bool writing, DirectStreamClock::time_point limit) {
    const auto now = DirectStreamClock::now(); if (now >= limit) return DirectStreamWaitStatus::timed_out;
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(limit - now).count();
    pollfd poll_fd{fd, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
    const auto result = ::poll(&poll_fd, 1, static_cast<int>(std::max<std::int64_t>(1, std::min<std::int64_t>(remaining, 60000))));
    if (result < 0) return errno == EINTR ? DirectStreamWaitStatus::interrupted : DirectStreamWaitStatus::failed;
    if (!result) return DirectStreamWaitStatus::timed_out;
    if (poll_fd.revents & POLLNVAL) return DirectStreamWaitStatus::failed;
    if (poll_fd.revents & (writing ? POLLOUT : POLLIN)) return DirectStreamWaitStatus::ready;
    return poll_fd.revents & POLLHUP ? DirectStreamWaitStatus::end_of_stream : DirectStreamWaitStatus::failed;
}
CommissioningTcpIdentity native_inspect(int fd) {
    struct stat stat_value{}; sockaddr_in local{}, peer{}; socklen_t local_size = sizeof(local), peer_size = sizeof(peer);
    if (::fstat(fd, &stat_value) || !S_ISSOCK(stat_value.st_mode) || ::getsockname(fd, reinterpret_cast<sockaddr*>(&local), &local_size) ||
        ::getpeername(fd, reinterpret_cast<sockaddr*>(&peer), &peer_size) || local_size != sizeof(local) || peer_size != sizeof(peer) ||
        local.sin_family != AF_INET || peer.sin_family != AF_INET) throw std::runtime_error("commissioning socket identity is unavailable");
    return {static_cast<std::uint64_t>(stat_value.st_dev), static_cast<std::uint64_t>(stat_value.st_ino), ntohl(local.sin_addr.s_addr), ntohl(peer.sin_addr.s_addr), ntohs(local.sin_port), ntohs(peer.sin_port)};
}
int native_open(const CommissioningTcpPolicy& policy, DirectStreamClock::time_point limit) {
    int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0); if (fd < 0) throw std::runtime_error("commissioning socket open failed");
    int connected = -1;
    try {
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(loopback); address.sin_port = htons(policy.port);
        std::uint32_t calls = 0;
        const auto ready = [&](bool writing) {
            for (;;) {
                if (calls++ >= policy.maximum_io_calls || DirectStreamClock::now() >= limit) throw std::runtime_error("commissioning TCP setup budget expired");
                const auto status = native_wait(fd, writing, limit); if (status == DirectStreamWaitStatus::ready) return;
                if (status != DirectStreamWaitStatus::interrupted) throw std::runtime_error("commissioning TCP setup wait failed");
            }
        };
        if (policy.listener) {
            if (::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) || ::listen(fd, 1)) throw std::runtime_error("commissioning loopback bind/listen failed");
            for (;;) {
                ready(false); connected = ::accept4(fd, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
                if (connected >= 0) break;
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) throw std::runtime_error("commissioning accept failed");
            }
            ::close(fd); fd = connected; connected = -1;
        } else {
            const auto result = ::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
            if (result && errno != EINPROGRESS && errno != EINTR) throw std::runtime_error("commissioning loopback connect failed");
            if (result) ready(true);
            int error = 0; socklen_t size = sizeof(error); if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) || error || size != sizeof(error)) throw std::runtime_error("commissioning TCP connect result failed");
        }
        const int enabled = 1; if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled))) throw std::runtime_error("commissioning TCP nodelay failed");
        const auto identity = native_inspect(fd); if (identity.local_address != loopback || identity.peer_address != loopback || DirectStreamClock::now() >= limit) throw std::runtime_error("commissioning TCP must remain loopback within setup deadline");
        return fd;
    } catch (...) { if (connected >= 0) ::close(connected); ::close(fd); throw; }
}
DirectStreamIoResult native_result(ssize_t bytes) {
    if (bytes > 0) return {DirectStreamIoStatus::transferred, static_cast<std::uint64_t>(bytes)};
    if (!bytes) return {DirectStreamIoStatus::end_of_stream, 0};
    return {errno == EINTR ? DirectStreamIoStatus::interrupted : (errno == EAGAIN || errno == EWOULDBLOCK) ? DirectStreamIoStatus::would_block : DirectStreamIoStatus::failed, 0};
}
#endif
}
CommissioningTcpCalls CommissioningTcpCalls::native() {
    CommissioningTcpCalls calls; calls.now = [] { return DirectStreamClock::now(); };
#if defined(__linux__)
    calls.open = native_open; calls.inspect = native_inspect; calls.wait = native_wait;
    calls.read = [](int fd, std::span<std::byte> bytes) { return native_result(::recv(fd, bytes.data(), bytes.size(), 0)); };
    calls.write = [](int fd, std::span<const std::byte> bytes) { return native_result(::send(fd, bytes.data(), bytes.size(), MSG_NOSIGNAL)); };
    calls.close = [](int fd) { static_cast<void>(::close(fd)); };
#else
    calls.open = [](const CommissioningTcpPolicy&, DirectStreamClock::time_point) -> int { throw std::runtime_error("commissioning TCP requires Linux"); };
#endif
    return calls;
}
BoundedCommissioningTcpControl::BoundedCommissioningTcpControl(CommissioningTcpPolicy policy, CommissioningTcpCalls calls) : policy_(policy), calls_(std::move(calls)) {
    validate(policy_);
    if (!calls_.open || !calls_.inspect || !calls_.read || !calls_.write || !calls_.wait || !calls_.now || !calls_.close) throw std::invalid_argument("commissioning TCP call boundary is incomplete");
    const auto limit = deadline(calls_.now(), policy_.timeout); fd_ = calls_.open(policy_, limit);
    try {
        if (fd_ < 0) throw std::runtime_error("commissioning TCP owned descriptor is unavailable");
        identity_ = calls_.inspect(fd_); identity_observed_ = true;
        if (identity_.local_address != loopback || identity_.peer_address != loopback || calls_.now() >= limit) throw std::runtime_error("commissioning TCP route/setup is unavailable");
    } catch (...) { close_noexcept(); throw; }
}
BoundedCommissioningTcpControl::~BoundedCommissioningTcpControl() { close_noexcept(); }
void BoundedCommissioningTcpControl::close_noexcept() noexcept {
    const int descriptor = fd_; fd_ = -1;
    if (descriptor < 0 || !identity_observed_) return;
    try {
        if (calls_.inspect(descriptor) != identity_) { failed_ = true; return; }
        // No retry after an ambiguous close callback. The trusted native owner
        // is serialized and never exposes this FD to external close/dup2.
        calls_.close(descriptor);
    } catch (...) { failed_ = true; }
}
void BoundedCommissioningTcpControl::require_current() {
    if (fd_ < 0 || failed_ || calls_.inspect(fd_) != identity_) throw std::runtime_error("commissioning TCP FD/peer identity changed or stream failed");
}
void BoundedCommissioningTcpControl::transfer(std::span<std::byte> destination, std::span<const std::byte> source, bool writing,
    DirectStreamClock::time_point limit, DirectStreamClock::time_point& last, std::uint32_t& calls) {
    const auto check = [&] { const auto now = calls_.now(); if (now < last || now >= limit) throw std::runtime_error("commissioning TCP deadline expired or clock changed"); last = now; require_current(); };
    const auto budget = [&] { if (calls++ >= policy_.maximum_io_calls) throw std::runtime_error("commissioning TCP IO budget exhausted"); };
    const auto size = writing ? source.size() : destination.size(); std::size_t offset = 0;
    while (offset < size) {
        check(); budget(); const auto result = writing ? calls_.write(fd_, source.subspan(offset)) : calls_.read(fd_, destination.subspan(offset)); check();
        if (result.status == DirectStreamIoStatus::transferred) { if (!result.bytes || result.bytes > size - offset) throw std::runtime_error("commissioning TCP impossible transfer count"); offset += static_cast<std::size_t>(result.bytes); }
        else if (result.bytes) throw std::runtime_error("commissioning TCP failed transfer has bytes");
        else if (result.status == DirectStreamIoStatus::interrupted) continue;
        else if (result.status == DirectStreamIoStatus::would_block) { budget(); const auto status = calls_.wait(fd_, writing, limit); check(); if (status != DirectStreamWaitStatus::ready && status != DirectStreamWaitStatus::interrupted) throw std::runtime_error("commissioning TCP wait failed/expired"); }
        else throw std::runtime_error("commissioning TCP partial frame EOF/failure");
    }
}
void BoundedCommissioningTcpControl::send(std::span<const std::byte> bytes) {
    if (bytes.empty() || bytes.size() > policy_.maximum_frame_bytes) throw std::invalid_argument("commissioning TCP frame exceeds bound");
    try {
        std::array<std::byte, 4> prefix{}; for (unsigned i = 0; i < 4; ++i) prefix[i] = static_cast<std::byte>((bytes.size() >> (8U * i)) & 255U);
        auto last = calls_.now(); const auto limit = deadline(last, policy_.timeout); std::uint32_t calls = 0;
        transfer({}, prefix, true, limit, last, calls); transfer({}, bytes, true, limit, last, calls);
    } catch (...) { failed_ = true; throw; }
}
std::vector<std::byte> BoundedCommissioningTcpControl::receive() {
    try {
        std::array<std::byte, 4> prefix{}; auto last = calls_.now(); const auto limit = deadline(last, policy_.timeout); std::uint32_t calls = 0;
        transfer(prefix, {}, false, limit, last, calls); std::uint32_t size = 0;
        for (unsigned i = 0; i < 4; ++i) size |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(prefix[i])) << (8U * i);
        if (!size || size > policy_.maximum_frame_bytes) throw std::runtime_error("commissioning TCP incoming length exceeds bound");
        std::vector<std::byte> bytes(size); transfer(bytes, {}, false, limit, last, calls); return bytes;
    } catch (...) { failed_ = true; throw; }
}
} // namespace strix::transport
