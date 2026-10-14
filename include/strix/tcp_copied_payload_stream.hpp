#pragma once

#include "strix/distributed_control_wire.hpp"
#include <chrono>
#include <cstdint>

namespace strix {

struct TcpCopiedPayloadPolicy {
    std::uint64_t maximum_frame_bytes = 2ULL * 1024ULL * 1024ULL + 4096ULL;
    std::uint32_t maximum_io_calls = 4096;
    std::chrono::milliseconds timeout{1000};
};

// Bounded CPU framing over an exclusively owned, already-connected loopback
// TCP socket. No listener/connect API, peer authentication or GPU semantics.
// An authenticated tunnel/session owner must separately reconcile admission;
// loopback origin is not proof of an authenticated remote peer.
// One serialized progress owner. Partial/error/expired frames poison/close.
class TcpCopiedPayloadByteStream final : public DistributedControlByteStream {
public:
    // Consumes fd even on validation failure. The caller must not retain any
    // alias that can close/reuse/reconfigure it after this ownership transfer.
    [[nodiscard]] static TcpCopiedPayloadByteStream adopt_connected_loopback(
        int fd, TcpCopiedPayloadPolicy policy = {});
    TcpCopiedPayloadByteStream(const TcpCopiedPayloadByteStream&) = delete;
    TcpCopiedPayloadByteStream& operator=(const TcpCopiedPayloadByteStream&) = delete;
    TcpCopiedPayloadByteStream(TcpCopiedPayloadByteStream&&) noexcept;
    TcpCopiedPayloadByteStream& operator=(TcpCopiedPayloadByteStream&&) noexcept;
    ~TcpCopiedPayloadByteStream() override;
    void send(std::span<const std::byte>) override;
    [[nodiscard]] std::vector<std::byte> receive() override;
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] bool valid() const noexcept { return fd_ >= 0 && !failed_; }
    void poison() noexcept { fail(); }
    [[nodiscard]] std::uint64_t maximum_frame_bytes() const noexcept { return policy_.maximum_frame_bytes; }
private:
    TcpCopiedPayloadByteStream(int, TcpCopiedPayloadPolicy, std::uint64_t, std::uint64_t);
    void require_healthy() const;
    void check_fd() const;
    void close_noexcept() noexcept;
    void fail() noexcept;
    void wait(bool writing, std::chrono::steady_clock::time_point deadline, std::uint32_t& calls);
    void read_all(std::span<std::byte>, std::chrono::steady_clock::time_point, std::uint32_t& calls);
    int fd_ = -1;
    TcpCopiedPayloadPolicy policy_;
    std::uint64_t device_ = 0, inode_ = 0;
    bool failed_ = false;
};

} // namespace strix
