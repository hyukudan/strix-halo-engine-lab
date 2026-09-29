#if !defined(__linux__)
#error "The UMA residency lab is intended for native Linux."
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

struct Mapping {
    void* address = MAP_FAILED;
    std::uint64_t bytes = 0;
    int fd = -1;

    ~Mapping() {
        if (address != MAP_FAILED) {
            munmap(address, static_cast<std::size_t>(bytes));
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Mapping() = default;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message + ": " + std::strerror(errno));
}

std::uint64_t multiply_checked(std::uint64_t value, std::uint64_t factor) {
    if (value > std::numeric_limits<std::uint64_t>::max() / factor) {
        throw std::overflow_error("size overflow");
    }
    return value * factor;
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
    for (const char c : number) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument("invalid size");
        }
        value = multiply_checked(value, 10);
        value += static_cast<std::uint64_t>(c - '0');
    }
    return multiply_checked(value, factor);
}

std::uint64_t parse_u64(std::string_view text) {
    return parse_size(text);
}

struct Options {
    std::string mode = "anon";
    std::string file;
    std::string pattern = "scan";
    std::string advice = "normal";
    std::uint64_t bytes = 256ULL * 1024ULL * 1024ULL;
    std::uint64_t iterations = 3;
    bool prefault = false;
    bool drop_pages = false;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        const auto value = [&](const char* name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (arg == "--mode") {
            options.mode = value("--mode");
        } else if (arg == "--file") {
            options.file = value("--file");
        } else if (arg == "--bytes") {
            options.bytes = parse_size(value("--bytes"));
        } else if (arg == "--iterations") {
            options.iterations = parse_u64(value("--iterations"));
        } else if (arg == "--pattern") {
            options.pattern = value("--pattern");
        } else if (arg == "--advice") {
            options.advice = value("--advice");
        } else if (arg == "--prefault") {
            options.prefault = true;
        } else if (arg == "--drop-pages") {
            options.drop_pages = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.bytes == 0 || options.iterations == 0) {
        throw std::invalid_argument("bytes and iterations must be non-zero");
    }
    if (options.mode != "anon" && options.mode != "mmap") {
        throw std::invalid_argument("mode must be anon or mmap");
    }
    if (options.mode == "mmap" && options.file.empty()) {
        throw std::invalid_argument("--file is required for mmap mode");
    }
    if (options.pattern != "scan" && options.pattern != "random-pages" &&
        options.pattern != "copy") {
        throw std::invalid_argument("pattern must be scan, random-pages or copy");
    }
    return options;
}

Mapping create_mapping(const Options& options) {
    Mapping mapping;
    mapping.bytes = options.bytes;
    if (options.mode == "anon") {
        mapping.address = mmap(nullptr, static_cast<std::size_t>(options.bytes),
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping.address == MAP_FAILED) {
            fail("anonymous mmap failed");
        }
        return mapping;
    }

    mapping.fd = open(options.file.c_str(), O_RDWR | O_CREAT, 0644);
    if (mapping.fd < 0) {
        fail("open failed");
    }
    if (ftruncate(mapping.fd, static_cast<off_t>(options.bytes)) != 0) {
        fail("ftruncate failed");
    }
    mapping.address = mmap(nullptr, static_cast<std::size_t>(options.bytes),
                           PROT_READ | PROT_WRITE, MAP_SHARED, mapping.fd, 0);
    if (mapping.address == MAP_FAILED) {
        fail("file mmap failed");
    }
    return mapping;
}

int advice_value(const std::string& advice) {
    if (advice == "normal") return MADV_NORMAL;
    if (advice == "random") return MADV_RANDOM;
    if (advice == "sequential") return MADV_SEQUENTIAL;
    throw std::invalid_argument("advice must be normal, random or sequential");
}

void page_touch(const Mapping& mapping, std::size_t page_size) {
    volatile std::uint8_t sink = 0;
    auto* bytes = static_cast<std::uint8_t*>(mapping.address);
    for (std::uint64_t offset = 0; offset < mapping.bytes; offset += page_size) {
        sink ^= bytes[offset];
    }
    (void)sink;
}

struct Faults {
    long minor = 0;
    long major = 0;
};

Faults faults() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        fail("getrusage failed");
    }
    return Faults{usage.ru_minflt, usage.ru_majflt};
}

std::uint64_t scan(const Mapping& mapping) {
    const auto* data = static_cast<const std::uint8_t*>(mapping.address);
    std::uint64_t sum = 0;
    for (std::uint64_t offset = 0; offset < mapping.bytes; offset += 64) {
        sum += data[offset];
    }
    return sum;
}

std::uint64_t random_pages(const Mapping& mapping, std::size_t page_size,
                           std::vector<std::uint64_t>& order) {
    const auto pages = (mapping.bytes + page_size - 1) / page_size;
    if (order.size() != pages) {
        order.resize(static_cast<std::size_t>(pages));
        std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 generator(0x5354524958ULL);
        std::shuffle(order.begin(), order.end(), generator);
    }
    const auto* data = static_cast<const std::uint8_t*>(mapping.address);
    std::uint64_t sum = 0;
    for (const auto page : order) {
        const auto offset = page * page_size;
        if (offset < mapping.bytes) {
            sum += data[offset];
        }
    }
    return sum;
}

std::uint64_t copy_all(const Mapping& mapping, std::vector<std::uint8_t>& destination) {
    if (destination.size() != mapping.bytes) {
        destination.resize(static_cast<std::size_t>(mapping.bytes));
    }
    std::memcpy(destination.data(), mapping.address, static_cast<std::size_t>(mapping.bytes));
    return destination.empty() ? 0 : destination.front();
}

}  // namespace

int main(int argc, char** argv) try {
    const auto options = parse_args(argc, argv);
    const auto page_size_raw = sysconf(_SC_PAGESIZE);
    if (page_size_raw <= 0) {
        throw std::runtime_error("cannot determine page size");
    }
    const auto page_size = static_cast<std::size_t>(page_size_raw);
    auto mapping = create_mapping(options);

    if (madvise(mapping.address, static_cast<std::size_t>(mapping.bytes),
                advice_value(options.advice)) != 0) {
        fail("madvise failed");
    }

    if (options.drop_pages) {
        if (options.mode == "mmap" && mapping.fd >= 0) {
            if (posix_fadvise(mapping.fd, 0, static_cast<off_t>(mapping.bytes),
                              POSIX_FADV_DONTNEED) != 0) {
                throw std::runtime_error("posix_fadvise(DONTNEED) failed");
            }
        }
        if (madvise(mapping.address, static_cast<std::size_t>(mapping.bytes),
                    MADV_DONTNEED) != 0) {
            fail("madvise(DONTNEED) failed");
        }
    }

    if (options.prefault) {
        page_touch(mapping, page_size);
    }

    std::vector<std::uint64_t> order;
    std::vector<std::uint8_t> destination;
    const auto before = faults();
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t checksum = 0;

    for (std::uint64_t iteration = 0; iteration < options.iterations; ++iteration) {
        if (options.pattern == "scan") {
            checksum ^= scan(mapping);
        } else if (options.pattern == "random-pages") {
            checksum ^= random_pages(mapping, page_size, order);
        } else {
            checksum ^= copy_all(mapping, destination);
        }
    }

    const auto stop = std::chrono::steady_clock::now();
    const auto after = faults();
    const auto seconds = std::chrono::duration<double>(stop - start).count();
    const auto total_bytes = static_cast<long double>(mapping.bytes) *
                             static_cast<long double>(options.iterations);
    const auto gib_s = seconds > 0.0
        ? static_cast<double>(total_bytes / (1024.0L * 1024.0L * 1024.0L) / seconds)
        : 0.0;

    std::cout << "{
"
              << "  \"mode\": \"" << options.mode << "\",
"
              << "  \"pattern\": \"" << options.pattern << "\",
"
              << "  \"advice\": \"" << options.advice << "\",
"
              << "  \"bytes\": " << options.bytes << ",
"
              << "  \"iterations\": " << options.iterations << ",
"
              << "  \"prefault\": " << (options.prefault ? "true" : "false") << ",
"
              << "  \"drop_pages\": " << (options.drop_pages ? "true" : "false") << ",
"
              << "  \"seconds\": " << seconds << ",
"
              << "  \"effective_gib_s\": " << gib_s << ",
"
              << "  \"minor_faults\": " << (after.minor - before.minor) << ",
"
              << "  \"major_faults\": " << (after.major - before.major) << ",
"
              << "  \"checksum\": " << checksum << "
"
              << "}
";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-uma-host: " << error.what() << '\n';
    return 1;
}
