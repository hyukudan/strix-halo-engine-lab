#if !defined(__linux__)
#error "The USB4STREAM transport baseline requires Linux."
#endif

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::uint32_t kMagic = 0x53545258U;
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

struct FileDescriptor {
    int fd = -1;

    FileDescriptor() = default;
    explicit FileDescriptor(int value) : fd(value) {}
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    FileDescriptor(FileDescriptor&& other) noexcept : fd(other.fd) {
        other.fd = -1;
    }

    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            if (fd >= 0) {
                close(fd);
            }
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    ~FileDescriptor() {
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

void write_all(int fd, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t written = 0;
    while (written < bytes) {
        const auto result = write(fd, cursor + written, bytes - written);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("write");
        }
        if (result == 0) {
            throw std::runtime_error("tbstream returned zero during write");
        }
        written += static_cast<std::size_t>(result);
    }
}

void read_all(int fd, void* data, std::size_t bytes) {
    auto* cursor = static_cast<std::uint8_t*>(data);
    std::size_t received = 0;
    while (received < bytes) {
        const auto result = read(fd, cursor + received, bytes - received);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("read");
        }
        if (result == 0) {
            throw std::runtime_error("tbstream reached EOF during read");
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

std::optional<std::string> read_text_file(const std::string& path) {
    std::ifstream input(path);
    std::string value;
    if (!input || !std::getline(input, value)) {
        return std::nullopt;
    }
    return value;
}

std::string json_string_or_null(const std::optional<std::string>& value) {
    if (!value.has_value()) {
        return "null";
    }
    std::string escaped = "\"";
    for (const char character : *value) {
        if (character == '\\' || character == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

FileDescriptor open_device(const std::string& path) {
    const int fd = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        fail("open(" + path + ")");
    }
    return FileDescriptor(fd);
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
    std::string device;
    std::string config_path;
    bool verify = false;
};

struct ServerOptions : CommonOptions {
    std::uint64_t maximum_bytes = 64ULL * 1024ULL * 1024ULL;
};

struct ClientOptions : CommonOptions {
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

void parse_common(std::string_view arg, int argc, char** argv, int& index,
                  CommonOptions& options, bool& handled) {
    handled = true;
    if (arg == "--device") {
        options.device = next_value(argc, argv, index, "--device");
    } else if (arg == "--config") {
        options.config_path = next_value(argc, argv, index, "--config");
    } else if (arg == "--verify") {
        options.verify = true;
    } else {
        handled = false;
    }
}

ServerOptions parse_server(int argc, char** argv) {
    ServerOptions options;
    for (int index = 2; index < argc; ++index) {
        const std::string_view arg = argv[index];
        bool handled = false;
        parse_common(arg, argc, argv, index, options, handled);
        if (handled) {
            continue;
        }
        if (arg == "--max-bytes") {
            options.maximum_bytes = parse_size(
                next_value(argc, argv, index, "--max-bytes"));
        } else {
            throw std::invalid_argument("unknown server argument: " + std::string(arg));
        }
    }
    if (options.device.empty()) {
        throw std::invalid_argument("--device is required");
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
        bool handled = false;
        parse_common(arg, argc, argv, index, options, handled);
        if (handled) {
            continue;
        }
        if (arg == "--bytes") {
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
        } else {
            throw std::invalid_argument("unknown client argument: " + std::string(arg));
        }
    }
    if (options.device.empty()) {
        throw std::invalid_argument("--device is required");
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

struct ConfigSnapshot {
    std::optional<std::string> index;
    std::optional<std::string> busy_poll;
    std::optional<std::string> ring_size;
    std::optional<std::string> throttling;
    std::optional<std::string> in_hopid;
    std::optional<std::string> out_hopid;
};

ConfigSnapshot config_snapshot(const std::string& base) {
    if (base.empty()) {
        return {};
    }
    return ConfigSnapshot{
        read_text_file(base + "/index"),
        read_text_file(base + "/busy_poll"),
        read_text_file(base + "/ring_size"),
        read_text_file(base + "/throttling"),
        read_text_file(base + "/in_hopid"),
        read_text_file(base + "/out_hopid"),
    };
}

void print_config_json(const ConfigSnapshot& config) {
    std::cout << "  \"config\": {" << '\n'
              << "    \"index\": " << json_string_or_null(config.index) << ',' << '\n'
              << "    \"busy_poll\": " << json_string_or_null(config.busy_poll) << ',' << '\n'
              << "    \"ring_size\": " << json_string_or_null(config.ring_size) << ',' << '\n'
              << "    \"throttling\": " << json_string_or_null(config.throttling) << ',' << '\n'
              << "    \"in_hopid\": " << json_string_or_null(config.in_hopid) << ',' << '\n'
              << "    \"out_hopid\": " << json_string_or_null(config.out_hopid) << '\n'
              << "  }";
}

void run_server(const ServerOptions& options) {
    auto stream = open_device(options.device);
    const auto config = config_snapshot(options.config_path);

    std::vector<std::uint8_t> payload;
    std::uint64_t messages = 0;
    std::uint64_t payload_bytes = 0;

    while (true) {
        WireHeader wire{};
        read_all(stream.fd, &wire, sizeof(wire));
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
        read_all(stream.fd, payload.data(), payload.size());

        if ((options.verify || header.checksum != 0) &&
            checksum(payload) != header.checksum) {
            const auto error = encode(WireKind::error, header.sequence, 0, 0);
            write_all(stream.fd, &error, sizeof(error));
            throw std::runtime_error("payload checksum mismatch");
        }

        if (header.kind == WireKind::ping) {
            write_all(stream.fd, &wire, sizeof(wire));
            write_all(stream.fd, payload.data(), payload.size());
        } else {
            const auto ack = encode(WireKind::ack, header.sequence, 0, header.checksum);
            write_all(stream.fd, &ack, sizeof(ack));
        }

        ++messages;
        payload_bytes += header.bytes;
    }

    std::cout << '{' << '\n'
              << "  \"role\": \"server\"," << '\n'
              << "  \"transport\": \"usb4stream\"," << '\n'
              << "  \"device\": \"" << options.device << "\"," << '\n'
              << "  \"messages\": " << messages << ',' << '\n'
              << "  \"payload_bytes_received\": " << payload_bytes << ',' << '\n';
    print_config_json(config);
    std::cout << '\n' << '}' << '\n';
}

void run_client(const ClientOptions& options) {
    auto stream = open_device(options.device);
    const auto config = config_snapshot(options.config_path);

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

        write_all(stream.fd, &header, sizeof(header));
        write_all(stream.fd, payload.data(), payload.size());

        WireHeader response_wire{};
        read_all(stream.fd, &response_wire, sizeof(response_wire));
        const auto response = decode(response_wire);
        if (response.sequence != sequence || response.kind == WireKind::error) {
            throw std::runtime_error("invalid benchmark response");
        }

        if (options.mode == WireKind::ping) {
            if (response.kind != WireKind::ping || response.bytes != options.bytes) {
                throw std::runtime_error("invalid ping response");
            }
            read_all(stream.fd, returned.data(), returned.size());
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
    write_all(stream.fd, &stop, sizeof(stop));

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
              << "  \"transport\": \"usb4stream\"," << '\n'
              << "  \"device\": \"" << options.device << "\"," << '\n'
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
              << "  \"transaction_max_us\": " << maximum << ',' << '\n';
    print_config_json(config);
    std::cout << '\n' << '}' << '\n';
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: strix-transport-tbstream server|client ...");
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
    std::cerr << "strix-transport-tbstream: " << error.what() << '\n';
    return 1;
}
