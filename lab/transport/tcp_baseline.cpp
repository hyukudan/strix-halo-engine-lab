#if !defined(__linux__) && !defined(__APPLE__)
#error "The transport baseline currently requires POSIX sockets."
#endif

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <netdb.h>
#include <netinet/tcp.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::uint32_t kMagic = 0x53545258U;  // STRX
constexpr std::uint16_t kVersion = 1;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

enum class WireKind : std::uint16_t {
    ping = 1,
    stream = 2,
    ack = 3,
    stop = 4,
    error = 5,
};

struct WireHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t kind;
    std::uint64_t sequence;
    std::uint64_t bytes;
    std::uint64_t checksum;
};

static_assert(sizeof(WireHeader) == 32);

struct Socket {
    int fd = -1;

    Socket() = default;
    explicit Socket(int value) : fd(value) {}
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    Socket(Socket&& other) noexcept : fd(other.fd) {
        other.fd = -1;
    }

    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            if (fd >= 0) {
                close(fd);
            }
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    ~Socket() {
        if (fd >= 0) {
            close(fd);
        }
    }
};

[[noreturn]] void fail(const std::string& operation) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
}

std::uint64_t host_to_network64(std::uint64_t value) noexcept {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (static_cast<std::uint64_t>(htonl(static_cast<std::uint32_t>(value))) << 32U) |
           htonl(static_cast<std::uint32_t>(value >> 32U));
#else
    return value;
#endif
}

std::uint64_t network_to_host64(std::uint64_t value) noexcept {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (static_cast<std::uint64_t>(ntohl(static_cast<std::uint32_t>(value))) << 32U) |
           ntohl(static_cast<std::uint32_t>(value >> 32U));
#else
    return value;
#endif
}

WireHeader encode(WireKind kind, std::uint64_t sequence, std::uint64_t bytes,
                  std::uint64_t checksum) noexcept {
    WireHeader header{};
    header.magic = htonl(kMagic);
    header.version = htons(kVersion);
    header.kind = htons(static_cast<std::uint16_t>(kind));
    header.sequence = host_to_network64(sequence);
    header.bytes = host_to_network64(bytes);
    header.checksum = host_to_network64(checksum);
    return header;
}

struct DecodedHeader {
    WireKind kind = WireKind::error;
    std::uint64_t sequence = 0;
    std::uint64_t bytes = 0;
    std::uint64_t checksum = 0;
};

DecodedHeader decode(const WireHeader& header) {
    if (ntohl(header.magic) != kMagic || ntohs(header.version) != kVersion) {
        throw std::runtime_error("invalid transport header");
    }
    const auto raw_kind = ntohs(header.kind);
    if (raw_kind < static_cast<std::uint16_t>(WireKind::ping) ||
        raw_kind > static_cast<std::uint16_t>(WireKind::error)) {
        throw std::runtime_error("invalid transport message kind");
    }
    return DecodedHeader{
        static_cast<WireKind>(raw_kind),
        network_to_host64(header.sequence),
        network_to_host64(header.bytes),
        network_to_host64(header.checksum),
    };
}

void send_all(int fd, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t sent = 0;
    while (sent < bytes) {
        const auto result = send(fd, cursor + sent, bytes - sent, MSG_NOSIGNAL);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("send");
        }
        if (result == 0) {
            throw std::runtime_error("socket closed during send");
        }
        sent += static_cast<std::size_t>(result);
    }
}

void recv_all(int fd, void* data, std::size_t bytes) {
    auto* cursor = static_cast<std::uint8_t*>(data);
    std::size_t received = 0;
    while (received < bytes) {
        const auto result = recv(fd, cursor + received, bytes - received, 0);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("recv");
        }
        if (result == 0) {
            throw std::runtime_error("socket closed during receive");
        }
        received += static_cast<std::size_t>(result);
    }
}

std::uint64_t checksum(const std::vector<std::uint8_t>& payload) noexcept {
    std::uint64_t value = kFnvOffset;
    for (const auto byte : payload) {
        value ^= byte;
        value *= kFnvPrime;
    }
    return value;
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("size overflow");
    }
    return left * right;
}

std::uint64_t parse_size(std::string_view text) {
    std::uint64_t factor = 1;
    auto number = text;
    const struct Suffix {
        std::string_view text;
        std::uint64_t factor;
    } suffixes[] = {
        {"GiB", 1024ULL * 1024ULL * 1024ULL},
        {"MiB", 1024ULL * 1024ULL},
        {"KiB", 1024ULL},
        {"GB", 1000ULL * 1000ULL * 1000ULL},
        {"MB", 1000ULL * 1000ULL},
        {"KB", 1000ULL},
    };
    for (const auto& suffix : suffixes) {
        if (text.size() >= suffix.text.size() &&
            text.substr(text.size() - suffix.text.size()) == suffix.text) {
            factor = suffix.factor;
            number = text.substr(0, text.size() - suffix.text.size());
            break;
        }
    }
    if (number.empty()) {
        throw std::invalid_argument("invalid size");
    }
    std::uint64_t value = 0;
    for (const char character : number) {
        if (character < '0' || character > '9') {
            throw std::invalid_argument("invalid size");
        }
        value = checked_mul(value, 10);
        value += static_cast<std::uint64_t>(character - '0');
    }
    return checked_mul(value, factor);
}

std::uint64_t parse_integer(std::string_view text) {
    return parse_size(text);
}

std::uint16_t parse_port(std::string_view text) {
    const auto value = parse_integer(text);
    if (value == 0 || value > 65535) {
        throw std::invalid_argument("port is out of range");
    }
    return static_cast<std::uint16_t>(value);
}

void set_nodelay(int fd) {
    const int enabled = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0) {
        fail("setsockopt(TCP_NODELAY)");
    }
}

Socket listen_socket(const std::string& bind_address, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* raw = nullptr;
    const auto port_text = std::to_string(port);
    const char* node = bind_address == "*" ? nullptr : bind_address.c_str();
    const auto status = getaddrinfo(node, port_text.c_str(), &hints, &raw);
    if (status != 0) {
        throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(status));
    }

    Socket listener;
    for (auto* candidate = raw; candidate != nullptr; candidate = candidate->ai_next) {
        Socket socket_fd(socket(candidate->ai_family, candidate->ai_socktype,
                                candidate->ai_protocol));
        if (socket_fd.fd < 0) {
            continue;
        }
        const int reuse = 1;
        static_cast<void>(setsockopt(socket_fd.fd, SOL_SOCKET, SO_REUSEADDR,
                                     &reuse, sizeof(reuse)));
        if (bind(socket_fd.fd, candidate->ai_addr, candidate->ai_addrlen) == 0 &&
            listen(socket_fd.fd, 1) == 0) {
            listener = std::move(socket_fd);
            break;
        }
    }
    freeaddrinfo(raw);

    if (listener.fd < 0) {
        throw std::runtime_error("unable to bind/listen");
    }
    return listener;
}

Socket connect_socket(const std::string& host, std::uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* raw = nullptr;
    const auto port_text = std::to_string(port);
    const auto status = getaddrinfo(host.c_str(), port_text.c_str(), &hints, &raw);
    if (status != 0) {
        throw std::runtime_error(std::string("getaddrinfo: ") + gai_strerror(status));
    }

    Socket connected;
    for (auto* candidate = raw; candidate != nullptr; candidate = candidate->ai_next) {
        Socket socket_fd(socket(candidate->ai_family, candidate->ai_socktype,
                                candidate->ai_protocol));
        if (socket_fd.fd < 0) {
            continue;
        }
        if (connect(socket_fd.fd, candidate->ai_addr, candidate->ai_addrlen) == 0) {
            connected = std::move(socket_fd);
            break;
        }
    }
    freeaddrinfo(raw);

    if (connected.fd < 0) {
        throw std::runtime_error("unable to connect");
    }
    set_nodelay(connected.fd);
    return connected;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto position = fraction * static_cast<double>(values.size() - 1);
    const auto index = static_cast<std::size_t>(std::ceil(position));
    return values[std::min(index, values.size() - 1)];
}

struct CommonOptions {
    std::uint16_t port = 19090;
    bool verify = false;
};

struct ServerOptions : CommonOptions {
    std::string bind_address = "*";
    std::uint64_t maximum_bytes = 64ULL * 1024ULL * 1024ULL;
};

struct ClientOptions : CommonOptions {
    std::string host;
    WireKind mode = WireKind::ping;
    std::uint64_t bytes = 64ULL * 1024ULL;
    std::uint64_t warmup = 10;
    std::uint64_t iterations = 100;
};

std::string_view next_value(int argc, char** argv, int& index, const char* name) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
    }
    ++index;
    return argv[index];
}

ServerOptions parse_server(int argc, char** argv) {
    ServerOptions options;
    for (int index = 2; index < argc; ++index) {
        const std::string_view arg = argv[index];
        if (arg == "--bind") {
            options.bind_address = next_value(argc, argv, index, "--bind");
        } else if (arg == "--port") {
            options.port = parse_port(next_value(argc, argv, index, "--port"));
        } else if (arg == "--max-bytes") {
            options.maximum_bytes = parse_size(
                next_value(argc, argv, index, "--max-bytes"));
        } else if (arg == "--verify") {
            options.verify = true;
        } else {
            throw std::invalid_argument("unknown server argument: " + std::string(arg));
        }
    }
    if (options.maximum_bytes == 0) {
        throw std::invalid_argument("max-bytes must be non-zero");
    }
    return options;
}

ClientOptions parse_client(int argc, char** argv) {
    ClientOptions options;
    for (int index = 2; index < argc; ++index) {
        const std::string_view arg = argv[index];
        if (arg == "--host") {
            options.host = next_value(argc, argv, index, "--host");
        } else if (arg == "--port") {
            options.port = parse_port(next_value(argc, argv, index, "--port"));
        } else if (arg == "--bytes") {
            options.bytes = parse_size(next_value(argc, argv, index, "--bytes"));
        } else if (arg == "--warmup") {
            options.warmup = parse_integer(next_value(argc, argv, index, "--warmup"));
        } else if (arg == "--iterations") {
            options.iterations = parse_integer(
                next_value(argc, argv, index, "--iterations"));
        } else if (arg == "--mode") {
            const auto mode = next_value(argc, argv, index, "--mode");
            if (mode == "pingpong") {
                options.mode = WireKind::ping;
            } else if (mode == "stream") {
                options.mode = WireKind::stream;
            } else {
                throw std::invalid_argument("mode must be pingpong or stream");
            }
        } else if (arg == "--verify") {
            options.verify = true;
        } else {
            throw std::invalid_argument("unknown client argument: " + std::string(arg));
        }
    }
    if (options.host.empty()) {
        throw std::invalid_argument("--host is required");
    }
    if (options.bytes == 0 || options.iterations == 0) {
        throw std::invalid_argument("bytes and iterations must be non-zero");
    }
    if (options.bytes >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("payload does not fit this host");
    }
    return options;
}

void run_server(const ServerOptions& options) {
    auto listener = listen_socket(options.bind_address, options.port);
    Socket peer(accept(listener.fd, nullptr, nullptr));
    if (peer.fd < 0) {
        fail("accept");
    }
    set_nodelay(peer.fd);

    std::vector<std::uint8_t> payload;
    std::uint64_t messages = 0;
    std::uint64_t payload_bytes = 0;

    while (true) {
        WireHeader wire{};
        recv_all(peer.fd, &wire, sizeof(wire));
        const auto header = decode(wire);

        if (header.kind == WireKind::stop) {
            break;
        }
        if (header.kind != WireKind::ping && header.kind != WireKind::stream) {
            throw std::runtime_error("server received unsupported message kind");
        }
        if (header.bytes == 0 || header.bytes > options.maximum_bytes ||
            header.bytes > static_cast<std::uint64_t>(
                               std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("server rejected payload size");
        }

        payload.resize(static_cast<std::size_t>(header.bytes));
        recv_all(peer.fd, payload.data(), payload.size());

        if ((options.verify || header.checksum != 0) &&
            checksum(payload) != header.checksum) {
            const auto error = encode(WireKind::error, header.sequence, 0, 0);
            send_all(peer.fd, &error, sizeof(error));
            throw std::runtime_error("payload checksum mismatch");
        }

        if (header.kind == WireKind::ping) {
            send_all(peer.fd, &wire, sizeof(wire));
            send_all(peer.fd, payload.data(), payload.size());
        } else {
            const auto ack = encode(WireKind::ack, header.sequence, 0, header.checksum);
            send_all(peer.fd, &ack, sizeof(ack));
        }

        ++messages;
        payload_bytes += header.bytes;
    }

    std::cout << '{' << '\n'
              << "  \"role\": \"server\"," << '\n'
              << "  \"messages\": " << messages << ',' << '\n'
              << "  \"payload_bytes_received\": " << payload_bytes << '\n'
              << '}' << '\n';
}

void run_client(const ClientOptions& options) {
    auto peer = connect_socket(options.host, options.port);
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(options.bytes));
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<std::uint8_t>((index * 131U + 17U) & 0xffU);
    }
    std::vector<std::uint8_t> returned;
    if (options.mode == WireKind::ping) {
        returned.resize(payload.size());
    }

    const auto payload_checksum = options.verify ? checksum(payload) : 0;
    std::vector<double> measured;
    measured.reserve(static_cast<std::size_t>(options.iterations));

    const auto total_iterations = options.warmup + options.iterations;

    for (std::uint64_t iteration = 0; iteration < total_iterations; ++iteration) {
        const auto sequence = iteration + 1;
        const auto header = encode(options.mode, sequence, options.bytes,
                                   payload_checksum);
        const auto start = std::chrono::steady_clock::now();

        send_all(peer.fd, &header, sizeof(header));
        send_all(peer.fd, payload.data(), payload.size());

        WireHeader response_wire{};
        recv_all(peer.fd, &response_wire, sizeof(response_wire));
        const auto response = decode(response_wire);
        if (response.sequence != sequence || response.kind == WireKind::error) {
            throw std::runtime_error("invalid benchmark response");
        }

        if (options.mode == WireKind::ping) {
            if (response.kind != WireKind::ping || response.bytes != options.bytes) {
                throw std::runtime_error("invalid ping response");
            }
            recv_all(peer.fd, returned.data(), returned.size());
            if (options.verify && checksum(returned) != payload_checksum) {
                throw std::runtime_error("returned payload checksum mismatch");
            }
        } else if (response.kind != WireKind::ack || response.bytes != 0) {
            throw std::runtime_error("invalid stream acknowledgement");
        }

        const auto stop = std::chrono::steady_clock::now();
        if (iteration >= options.warmup) {
            measured.push_back(std::chrono::duration<double>(stop - start).count());
        }
    }

    const auto stop = encode(WireKind::stop, total_iterations + 1, 0, 0);
    send_all(peer.fd, &stop, sizeof(stop));

    const auto total_seconds =
        std::accumulate(measured.begin(), measured.end(), 0.0);
    const auto direction_multiplier =
        options.mode == WireKind::ping ? std::uint64_t{2} : std::uint64_t{1};
    const long double total_payload =
        static_cast<long double>(options.bytes) *
        static_cast<long double>(options.iterations) *
        static_cast<long double>(direction_multiplier);
    const auto payload_gib_s = total_seconds > 0.0
        ? static_cast<double>(
              total_payload / (1024.0L * 1024.0L * 1024.0L) / total_seconds)
        : 0.0;

    const auto p50 = percentile(measured, 0.50) * 1e6;
    const auto p95 = percentile(measured, 0.95) * 1e6;
    const auto p99 = percentile(measured, 0.99) * 1e6;
    const auto maximum = measured.empty()
        ? 0.0
        : *std::max_element(measured.begin(), measured.end()) * 1e6;

    std::cout << '{' << '\n'
              << "  \"role\": \"client\"," << '\n'
              << "  \"mode\": \""
              << (options.mode == WireKind::ping ? "pingpong" : "stream")
              << "\"," << '\n'
              << "  \"bytes\": " << options.bytes << ',' << '\n'
              << "  \"warmup\": " << options.warmup << ',' << '\n'
              << "  \"iterations\": " << options.iterations << ',' << '\n'
              << "  \"verify\": " << (options.verify ? "true" : "false")
              << ',' << '\n'
              << "  \"measured_seconds\": " << total_seconds << ',' << '\n'
              << "  \"effective_payload_gib_s\": " << payload_gib_s << ',' << '\n'
              << "  \"transaction_p50_us\": " << p50 << ',' << '\n'
              << "  \"transaction_p95_us\": " << p95 << ',' << '\n'
              << "  \"transaction_p99_us\": " << p99 << ',' << '\n'
              << "  \"transaction_max_us\": " << maximum << '\n'
              << '}' << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    std::signal(SIGPIPE, SIG_IGN);
    if (argc < 2) {
        throw std::invalid_argument("usage: strix-transport-tcp server|client ...");
    }
    const std::string_view role = argv[1];
    if (role == "server") {
        run_server(parse_server(argc, argv));
    } else if (role == "client") {
        run_client(parse_client(argc, argv));
    } else {
        throw std::invalid_argument("first argument must be server or client");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-transport-tcp: " << error.what() << '\n';
    return 1;
}
