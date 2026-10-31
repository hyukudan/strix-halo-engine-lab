#pragma once
#include <fstream>
#include <random>

#include "glm_import.hpp"
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
namespace strix::glm::import {
#ifdef _WIN32
using Socket = SOCKET;
inline constexpr Socket bad_socket = INVALID_SOCKET;
inline void close_socket(Socket s) { closesocket(s); }
#else
using Socket = int;
inline constexpr Socket bad_socket = -1;
inline void close_socket(Socket s) { close(s); }
#endif
struct SocketOwner {
    Socket fd = bad_socket;
    ~SocketOwner() {
        if (fd != bad_socket) close_socket(fd);
    }
    SocketOwner() = default;
    SocketOwner(const SocketOwner&) = delete;
};
inline void io(Socket fd, std::span<std::byte> b, bool write) {
    while (!b.empty()) {
        const auto n = static_cast<int>(std::min<std::size_t>(b.size(), 1048576));
#ifdef _WIN32
        int rc = write ? send(fd, reinterpret_cast<const char*>(b.data()), n, 0)
                       : recv(fd, reinterpret_cast<char*>(b.data()), n, 0);
#else
        auto rc = write ? send(fd, b.data(), static_cast<std::size_t>(n), MSG_NOSIGNAL)
                        : recv(fd, b.data(), static_cast<std::size_t>(n), 0);
#endif
        require(rc > 0, "import socket disconnected/timeout");
        b = b.subspan(static_cast<std::size_t>(rc));
    }
}
inline Bytes read_key(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    require(bool(f), "open import key");
    std::string s((std::istreambuf_iterator<char>(f)), {});
    require(s.size() == 32, "key file must contain 32 raw bytes");
    return Bytes(reinterpret_cast<const std::byte*>(s.data()), reinterpret_cast<const std::byte*>(s.data() + s.size()));
}
inline void validate(const Begin& got, const Begin& expected) {
    require(got.version == expected.version && got.world == expected.world && got.rank == expected.rank &&
                got.layers == expected.layers && got.total == expected.total && got.frontier == expected.frontier &&
                got.model == expected.model && got.prompt == expected.prompt && got.node == expected.node,
            "import identity/model/frontier/layout/rank mismatch");
    require(got.producer == "glm-pf-rtx-native-v1" || got.producer == "glm-pf-node-replay-v1",
            "unknown import producer");
}
// One authenticated transaction, one connection. ACK 0 = accepted/staged, 1 = locally sealed.
// No ACK implies rejection. A local seal is NOT permission to decode: peer barriers follow.
inline std::unique_ptr<Staging> receive(const std::string& endpoint, const Bytes& key, const Begin& expected) {
#ifdef _WIN32
    WSADATA wsa{};
    require(WSAStartup(MAKEWORD(2, 2), &wsa) == 0, "WSAStartup");
    struct Cleanup {
        ~Cleanup() { WSACleanup(); }
    } cleanup;
#endif
    auto colon = endpoint.rfind(':');
    require(colon != std::string::npos, "import endpoint needs HOST:PORT");
    const auto host = endpoint.substr(0, colon), port = endpoint.substr(colon + 1);
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICHOST | AI_NUMERICSERV;
    addrinfo* addr = nullptr;
    require(getaddrinfo(host.c_str(), port.c_str(), &hints, &addr) == 0, "import endpoint must be numeric IPv4:port");
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> address(addr, freeaddrinfo);
    SocketOwner listener;
    listener.fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    require(listener.fd != bad_socket, "import socket");
    require(bind(listener.fd, addr->ai_addr, static_cast<int>(addr->ai_addrlen)) == 0 && listen(listener.fd, 1) == 0,
            "import bind/listen");
    fd_set ready;
    FD_ZERO(&ready);
    FD_SET(listener.fd, &ready);
    timeval accept_timeout{120, 0};
    require(select(static_cast<int>(listener.fd + 1), &ready, nullptr, nullptr, &accept_timeout) > 0,
            "import accept timeout");
    SocketOwner client;
    client.fd = accept(listener.fd, nullptr, nullptr);
    require(client.fd != bad_socket, "import accept");
#ifdef _WIN32
    DWORD timeout = 120000;
#else
    timeval timeout{120, 0};
#endif
    require(
        setsockopt(client.fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout), sizeof timeout) == 0 &&
            setsockopt(client.fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout), sizeof timeout) ==
                0,
        "import socket timeout");
    Bytes nonce(32);
    std::random_device random;
    for (auto& x : nonce) x = std::byte(random() & 255);
    io(client.fd, nonce, true);
    std::uint64_t sequence = 0, wire = 32;
    std::unique_ptr<Staging> staged;
    while (true) {
        Bytes size(4);
        io(client.fd, size, false);
        Reader length{size};
        auto n = length.number(4);
        require(n >= 4 && n <= 8388608, "import packet length exceeds limit");
        Bytes body(static_cast<std::size_t>(n));
        io(client.fd, body, false);
        Bytes mac(32);
        io(client.fd, mac, false);
        Bytes authenticated = nonce;
        number(authenticated, sequence++);
        append(authenticated, size);
        append(authenticated, body);
        Reader mr{mac};
        require(secure_equal(hmac(key, authenticated), mr.digest()), "import authentication failed");
        wire += n + 36;
        Reader r{body};
        auto op = r.number(4);
        if (op == 4) {
            r.end();
            throw std::runtime_error("import aborted by sender");
        }
        if (op == 1) {
            require(!staged, "duplicate begin");
            auto b = Begin::parse(r);
            validate(b, expected);
            staged = std::make_unique<Staging>(b);
        } else {
            require(bool(staged), "import frame/commit before begin");
            if (op == 2)
                staged->frame(r, body);
            else if (op == 3)
                staged->commit(r);
            else
                require(false, "unknown import operation");
        }
        Bytes ack{staged->committed ? std::byte{1} : std::byte{0}};
        io(client.fd, ack, true);
        ++wire;
        if (staged->committed) {
            staged->wire_bytes = wire;
            return staged;
        }
    }
}
}  // namespace strix::glm::import
