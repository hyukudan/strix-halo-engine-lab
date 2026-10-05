#include "strix/distributed_tcp_control_stream.hpp"

#if !defined(__linux__) && !defined(__APPLE__)
#error "TcpDistributedControlByteStream currently requires POSIX sockets."
#endif

#include <cerrno>
#include <cstdint>
#include <limits>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace strix {
namespace {

void validate_limit(
    std::uint64_t maximum_frame_bytes) {
    if (maximum_frame_bytes == 0 ||
        maximum_frame_bytes >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "TCP control maximum frame bytes must fit uint32 and be non-zero");
    }
}

void send_all(
    int fd,
    const void* data,
    std::size_t bytes) {
    const auto* cursor =
        static_cast<const std::byte*>(
            data);
    std::size_t sent = 0;
    while (sent < bytes) {
        const auto result =
            ::send(
                fd,
                cursor + sent,
                bytes - sent,
                0);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(
                "TCP control send failed");
        }
        if (result == 0) {
            throw std::runtime_error(
                "TCP control socket closed during send");
        }
        sent +=
            static_cast<std::size_t>(
                result);
    }
}

void receive_all(
    int fd,
    void* data,
    std::size_t bytes) {
    auto* cursor =
        static_cast<std::byte*>(
            data);
    std::size_t received = 0;
    while (received < bytes) {
        const auto result =
            ::recv(
                fd,
                cursor + received,
                bytes - received,
                0);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(
                "TCP control receive failed");
        }
        if (result == 0) {
            throw std::runtime_error(
                "TCP control socket closed during receive");
        }
        received +=
            static_cast<std::size_t>(
                result);
    }
}

void set_nodelay(
    int fd) {
    const int enabled = 1;
    if (::setsockopt(
            fd,
            IPPROTO_TCP,
            TCP_NODELAY,
            &enabled,
            sizeof(enabled)) != 0) {
        throw std::runtime_error(
            "TCP control TCP_NODELAY failed");
    }
}

std::uint32_t decode_length(
    const std::byte prefix[4]) noexcept {
    return
        static_cast<std::uint32_t>(
            std::to_integer<std::uint8_t>(
                prefix[0])) |
        (static_cast<std::uint32_t>(
             std::to_integer<std::uint8_t>(
                 prefix[1])) << 8U) |
        (static_cast<std::uint32_t>(
             std::to_integer<std::uint8_t>(
                 prefix[2])) << 16U) |
        (static_cast<std::uint32_t>(
             std::to_integer<std::uint8_t>(
                 prefix[3])) << 24U);
}

void encode_length(
    std::uint32_t value,
    std::byte prefix[4]) noexcept {
    for (unsigned shift = 0;
         shift < 32;
         shift += 8) {
        prefix[shift / 8U] =
            static_cast<std::byte>(
                (value >> shift) &
                0xffU);
    }
}

int connect_socket(
    const std::string& host,
    std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* raw = nullptr;
    const auto port_text =
        std::to_string(port);
    const auto status =
        ::getaddrinfo(
            host.c_str(),
            port_text.c_str(),
            &hints,
            &raw);
    if (status != 0) {
        throw std::runtime_error(
            "TCP control getaddrinfo failed");
    }

    int connected = -1;
    for (auto* candidate = raw;
         candidate != nullptr;
         candidate = candidate->ai_next) {
        const int fd =
            ::socket(
                candidate->ai_family,
                candidate->ai_socktype,
                candidate->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (::connect(
                fd,
                candidate->ai_addr,
                candidate->ai_addrlen) == 0) {
            connected = fd;
            break;
        }
        ::close(fd);
    }
    ::freeaddrinfo(raw);

    if (connected < 0) {
        throw std::runtime_error(
            "TCP control unable to connect");
    }

    try {
        set_nodelay(connected);
    } catch (...) {
        ::close(connected);
        throw;
    }
    return connected;
}

int accept_once(
    const std::string& bind_address,
    std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* raw = nullptr;
    const auto port_text =
        std::to_string(port);
    const char* node =
        bind_address == "*"
            ? nullptr
            : bind_address.c_str();
    const auto status =
        ::getaddrinfo(
            node,
            port_text.c_str(),
            &hints,
            &raw);
    if (status != 0) {
        throw std::runtime_error(
            "TCP control getaddrinfo(bind) failed");
    }

    int listener = -1;
    for (auto* candidate = raw;
         candidate != nullptr;
         candidate = candidate->ai_next) {
        const int fd =
            ::socket(
                candidate->ai_family,
                candidate->ai_socktype,
                candidate->ai_protocol);
        if (fd < 0) {
            continue;
        }
        const int reuse = 1;
        static_cast<void>(
            ::setsockopt(
                fd,
                SOL_SOCKET,
                SO_REUSEADDR,
                &reuse,
                sizeof(reuse)));
        if (::bind(
                fd,
                candidate->ai_addr,
                candidate->ai_addrlen) == 0 &&
            ::listen(fd, 1) == 0) {
            listener = fd;
            break;
        }
        ::close(fd);
    }
    ::freeaddrinfo(raw);

    if (listener < 0) {
        throw std::runtime_error(
            "TCP control unable to bind/listen");
    }

    const int accepted =
        ::accept(
            listener,
            nullptr,
            nullptr);
    ::close(listener);
    if (accepted < 0) {
        throw std::runtime_error(
            "TCP control accept failed");
    }

    try {
        set_nodelay(accepted);
    } catch (...) {
        ::close(accepted);
        throw;
    }
    return accepted;
}

}  // namespace

TcpDistributedControlByteStream::
TcpDistributedControlByteStream(
    int fd,
    std::uint64_t maximum_frame_bytes)
    : fd_(fd),
      maximum_frame_bytes_(
          maximum_frame_bytes) {
    validate_limit(
        maximum_frame_bytes_);
    if (fd_ < 0) {
        throw std::invalid_argument(
            "TCP control stream requires connected file descriptor");
    }
}

TcpDistributedControlByteStream::
TcpDistributedControlByteStream(
    TcpDistributedControlByteStream&& other) noexcept
    : fd_(other.fd_),
      maximum_frame_bytes_(
          other.maximum_frame_bytes_) {
    other.fd_ = -1;
    other.maximum_frame_bytes_ = 0;
}

TcpDistributedControlByteStream&
TcpDistributedControlByteStream::operator=(
    TcpDistributedControlByteStream&& other) noexcept {
    if (this != &other) {
        close_noexcept();
        fd_ = other.fd_;
        maximum_frame_bytes_ =
            other.maximum_frame_bytes_;
        other.fd_ = -1;
        other.maximum_frame_bytes_ = 0;
    }
    return *this;
}

TcpDistributedControlByteStream::
~TcpDistributedControlByteStream() {
    close_noexcept();
}

TcpDistributedControlByteStream
TcpDistributedControlByteStream::adopt_connected(
    int fd,
    std::uint64_t maximum_frame_bytes) {
    return TcpDistributedControlByteStream(
        fd,
        maximum_frame_bytes);
}

TcpDistributedControlByteStream
TcpDistributedControlByteStream::connect_to(
    const std::string& host,
    std::uint16_t port,
    std::uint64_t maximum_frame_bytes) {
    validate_limit(
        maximum_frame_bytes);
    return TcpDistributedControlByteStream(
        connect_socket(
            host,
            port),
        maximum_frame_bytes);
}

TcpDistributedControlByteStream
TcpDistributedControlByteStream::listen_once(
    const std::string& bind_address,
    std::uint16_t port,
    std::uint64_t maximum_frame_bytes) {
    validate_limit(
        maximum_frame_bytes);
    return TcpDistributedControlByteStream(
        accept_once(
            bind_address,
            port),
        maximum_frame_bytes);
}

void TcpDistributedControlByteStream::send(
    std::span<const std::byte> frame) {
    if (!valid() ||
        frame.empty() ||
        frame.size() >
            maximum_frame_bytes_ ||
        frame.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "TCP control frame exceeds configured bounds");
    }

    std::byte prefix[4]{};
    encode_length(
        static_cast<std::uint32_t>(
            frame.size()),
        prefix);
    send_all(
        fd_,
        prefix,
        sizeof(prefix));
    send_all(
        fd_,
        frame.data(),
        frame.size());
}

std::vector<std::byte>
TcpDistributedControlByteStream::receive() {
    if (!valid()) {
        throw std::logic_error(
            "TCP control stream is not connected");
    }

    std::byte prefix[4]{};
    receive_all(
        fd_,
        prefix,
        sizeof(prefix));
    const auto bytes =
        decode_length(prefix);
    if (bytes == 0 ||
        static_cast<std::uint64_t>(
            bytes) >
            maximum_frame_bytes_) {
        throw std::invalid_argument(
            "TCP control received frame length exceeds configured bounds");
    }

    std::vector<std::byte> frame(
        bytes);
    receive_all(
        fd_,
        frame.data(),
        frame.size());
    return frame;
}

void TcpDistributedControlByteStream::close_noexcept() noexcept {
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_ = -1;
    maximum_frame_bytes_ = 0;
}

}  // namespace strix
