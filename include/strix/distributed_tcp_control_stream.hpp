#pragma once

#include "strix/distributed_control_wire.hpp"

#include <cstdint>
#include <string>

namespace strix {

class TcpDistributedControlByteStream final
    : public DistributedControlByteStream {
public:
    TcpDistributedControlByteStream() = default;
    TcpDistributedControlByteStream(
        const TcpDistributedControlByteStream&) = delete;
    TcpDistributedControlByteStream& operator=(
        const TcpDistributedControlByteStream&) = delete;

    TcpDistributedControlByteStream(
        TcpDistributedControlByteStream&& other) noexcept;
    TcpDistributedControlByteStream& operator=(
        TcpDistributedControlByteStream&& other) noexcept;

    ~TcpDistributedControlByteStream();

    [[nodiscard]] static TcpDistributedControlByteStream
    adopt_connected(
        int fd,
        std::uint64_t maximum_frame_bytes =
            64ULL * 1024ULL);

    [[nodiscard]] static TcpDistributedControlByteStream
    connect_to(
        const std::string& host,
        std::uint16_t port,
        std::uint64_t maximum_frame_bytes =
            64ULL * 1024ULL);

    [[nodiscard]] static TcpDistributedControlByteStream
    listen_once(
        const std::string& bind_address,
        std::uint16_t port,
        std::uint64_t maximum_frame_bytes =
            64ULL * 1024ULL);

    void send(
        std::span<const std::byte> frame) override;

    [[nodiscard]] std::vector<std::byte>
    receive() override;

    [[nodiscard]] bool valid() const noexcept {
        return fd_ >= 0;
    }

    [[nodiscard]] int native_handle() const noexcept {
        return fd_;
    }

private:
    explicit TcpDistributedControlByteStream(
        int fd,
        std::uint64_t maximum_frame_bytes);

    void close_noexcept() noexcept;

    int fd_ = -1;
    std::uint64_t maximum_frame_bytes_ = 0;
};

}  // namespace strix
