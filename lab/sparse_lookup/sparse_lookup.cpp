#if !defined(__linux__)
#error "The sparse lookup lab currently requires Linux."
#endif

#include "strix/sparse_lookup.hpp"
#include "strix/sparse_lookup_executor.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

[[noreturn]] void fail(const std::string& operation) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
}

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error("size overflow");
    }
    return left * right;
}

std::uint64_t parse_u64(std::string_view text, const char* name) {
    if (text.empty()) {
        throw std::invalid_argument(std::string(name) + " is empty");
    }
    std::uint64_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(std::string(name) + " must be integer");
        }
        value = checked_mul(value, 10);
        value += static_cast<std::uint64_t>(c - '0');
    }
    return value;
}

struct Options {
    std::string command = "bench";
    std::string file;
    std::string mode = "pread";
    std::string pattern = "ple";
    std::uint64_t table_rows = 16'384;
    std::uint64_t row_bytes = 320;
    std::uint64_t requests = 16'000;
    std::uint64_t window = 16;
    std::uint64_t cache_rows = 256;
    std::uint64_t max_rows_per_io = 4;
    std::uint64_t max_inflight_ios = 4;
    std::uint64_t seed = 1234;
    bool drop_cache = false;
};

std::string_view next_value(int argc, char** argv, int& index, const char* name) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
    }
    ++index;
    return argv[index];
}

Options parse_args(int argc, char** argv) {
    Options options;
    if (argc >= 2) {
        options.command = argv[1];
    }

    for (int index = 2; index < argc; ++index) {
        const std::string_view arg = argv[index];
        if (arg == "--file") {
            options.file = next_value(argc, argv, index, "--file");
        } else if (arg == "--mode") {
            options.mode = next_value(argc, argv, index, "--mode");
        } else if (arg == "--pattern") {
            options.pattern = next_value(argc, argv, index, "--pattern");
        } else if (arg == "--table-rows") {
            options.table_rows =
                parse_u64(next_value(argc, argv, index, "--table-rows"), "--table-rows");
        } else if (arg == "--row-bytes") {
            options.row_bytes =
                parse_u64(next_value(argc, argv, index, "--row-bytes"), "--row-bytes");
        } else if (arg == "--requests") {
            options.requests =
                parse_u64(next_value(argc, argv, index, "--requests"), "--requests");
        } else if (arg == "--window") {
            options.window =
                parse_u64(next_value(argc, argv, index, "--window"), "--window");
        } else if (arg == "--cache-rows") {
            options.cache_rows =
                parse_u64(next_value(argc, argv, index, "--cache-rows"), "--cache-rows");
        } else if (arg == "--max-rows-per-io") {
            options.max_rows_per_io =
                parse_u64(
                    next_value(argc, argv, index, "--max-rows-per-io"),
                    "--max-rows-per-io");
        } else if (arg == "--max-inflight-ios") {
            options.max_inflight_ios =
                parse_u64(
                    next_value(argc, argv, index, "--max-inflight-ios"),
                    "--max-inflight-ios");
        } else if (arg == "--seed") {
            options.seed =
                parse_u64(next_value(argc, argv, index, "--seed"), "--seed");
        } else if (arg == "--drop-cache") {
            options.drop_cache = true;
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (options.file.empty()) {
        throw std::invalid_argument("--file is required");
    }
    if (options.table_rows == 0 || options.row_bytes == 0) {
        throw std::invalid_argument("table rows and row bytes must be non-zero");
    }
    if (options.command == "bench" &&
        (options.requests == 0 || options.window == 0)) {
        throw std::invalid_argument("requests and window must be non-zero");
    }
    if (options.max_inflight_ios == 0) {
        throw std::invalid_argument("--max-inflight-ios must be non-zero");
    }
    return options;
}

struct File {
    int fd = -1;
    ~File() {
        if (fd >= 0) {
            close(fd);
        }
    }
    File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&& other) noexcept : fd(other.fd) {
        other.fd = -1;
    }
    File& operator=(File&&) = delete;
};

File open_file(const std::string& path, int flags) {
    File file;
    file.fd = open(path.c_str(), flags, 0644);
    if (file.fd < 0) {
        fail("open");
    }
    return file;
}

void pwrite_all(int fd, const void* data, std::size_t bytes, off_t offset) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t written = 0;
    while (written < bytes) {
        const auto result =
            pwrite(fd, cursor + written, bytes - written,
                   offset + static_cast<off_t>(written));
        if (result < 0) {
            if (errno == EINTR) continue;
            fail("pwrite");
        }
        if (result == 0) {
            throw std::runtime_error("pwrite returned zero");
        }
        written += static_cast<std::size_t>(result);
    }
}

void pread_all(int fd, void* data, std::size_t bytes, off_t offset) {
    auto* cursor = static_cast<std::uint8_t*>(data);
    std::size_t read_bytes = 0;
    while (read_bytes < bytes) {
        const auto result =
            pread(fd, cursor + read_bytes, bytes - read_bytes,
                  offset + static_cast<off_t>(read_bytes));
        if (result < 0) {
            if (errno == EINTR) continue;
            fail("pread");
        }
        if (result == 0) {
            throw std::runtime_error("pread reached EOF");
        }
        read_bytes += static_cast<std::size_t>(result);
    }
}

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

void prepare_file(const Options& options) {
    const auto file_bytes =
        checked_mul(options.table_rows, options.row_bytes);
    if (file_bytes >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::overflow_error("table file does not fit off_t");
    }

    auto file = open_file(options.file, O_RDWR | O_CREAT | O_TRUNC);
    if (ftruncate(file.fd, static_cast<off_t>(file_bytes)) != 0) {
        fail("ftruncate");
    }

    constexpr std::size_t chunk_bytes = 1U << 20U;
    std::vector<std::uint8_t> chunk(chunk_bytes);
    std::uint64_t offset = 0;
    while (offset < file_bytes) {
        const auto remaining = file_bytes - offset;
        const auto current =
            static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, chunk_bytes));
        for (std::size_t index = 0; index < current; ++index) {
            chunk[index] =
                static_cast<std::uint8_t>(
                    splitmix64(offset + index + options.seed) & 0xffU);
        }
        pwrite_all(file.fd, chunk.data(), current,
                   static_cast<off_t>(offset));
        offset += current;
    }

    if (fsync(file.fd) != 0) {
        fail("fsync");
    }

    std::cout << "{\n"
              << "  \"command\": \"prepare\",\n"
              << "  \"file_bytes\": " << file_bytes << ",\n"
              << "  \"table_rows\": " << options.table_rows << ",\n"
              << "  \"row_bytes\": " << options.row_bytes << "\n"
              << "}\n";
}

std::vector<std::uint64_t> make_requests(const Options& options) {
    std::vector<std::uint64_t> rows(
        static_cast<std::size_t>(options.requests));

    if (options.pattern == "random") {
        for (std::uint64_t i = 0; i < options.requests; ++i) {
            rows[static_cast<std::size_t>(i)] =
                splitmix64(options.seed + i) % options.table_rows;
        }
        return rows;
    }

    if (options.pattern == "hotset") {
        const auto hot_rows =
            std::max<std::uint64_t>(
                1, std::min<std::uint64_t>(1024, options.table_rows / 8U));
        for (std::uint64_t i = 0; i < options.requests; ++i) {
            const auto random = splitmix64(options.seed + i);
            const bool hot = (random % 10U) < 8U;
            rows[static_cast<std::size_t>(i)] =
                hot ? (random % hot_rows)
                    : (random % options.table_rows);
        }
        return rows;
    }

    if (options.pattern == "runs") {
        for (std::uint64_t begin = 0;
             begin < options.requests;
             begin += options.window) {
            const auto count =
                std::min(options.window, options.requests - begin);
            const auto span =
                options.table_rows > count
                    ? options.table_rows - count
                    : 1;
            const auto base =
                splitmix64(options.seed + begin) % span;
            for (std::uint64_t i = 0; i < count; ++i) {
                rows[static_cast<std::size_t>(begin + i)] =
                    std::min(base + i, options.table_rows - 1U);
            }
        }
        return rows;
    }

    if (options.pattern == "ple") {
        constexpr std::uint64_t heads = 16;
        if (options.table_rows < heads) {
            throw std::invalid_argument(
                "PLE pattern requires at least 16 table rows");
        }
        const auto rows_per_head =
            std::max<std::uint64_t>(1, options.table_rows / heads);
        for (std::uint64_t i = 0; i < options.requests; ++i) {
            const auto head = i % heads;
            const auto base = head * rows_per_head;
            const auto available =
                std::min(rows_per_head, options.table_rows - base);
            rows[static_cast<std::size_t>(i)] =
                base +
                (splitmix64(options.seed + i * 17U) % available);
        }
        return rows;
    }

    throw std::invalid_argument(
        "pattern must be random, hotset, runs or ple");
}

std::uint64_t consume_row(const std::uint8_t* row, std::size_t bytes) noexcept {
    std::uint64_t hash = kFnvOffset;
    for (std::size_t offset = 0; offset < bytes; offset += 64U) {
        hash ^= row[offset];
        hash *= kFnvPrime;
    }
    if (bytes != 0) {
        hash ^= row[bytes - 1U];
        hash *= kFnvPrime;
    }
    return hash;
}

struct Faults {
    long minor = 0;
    long major = 0;
};

Faults faults() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        fail("getrusage");
    }
    return Faults{usage.ru_minflt, usage.ru_majflt};
}

struct Result {
    std::uint64_t checksum = 0;
    std::uint64_t physical_read_bytes = 0;
    std::uint64_t read_syscalls = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
};

off_t row_offset(std::uint64_t row, std::uint64_t row_bytes) {
    const auto offset = checked_mul(row, row_bytes);
    if (offset >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::overflow_error("row offset does not fit off_t");
    }
    return static_cast<off_t>(offset);
}

off_t byte_offset(std::uint64_t offset) {
    if (offset >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw std::overflow_error("byte offset does not fit off_t");
    }
    return static_cast<off_t>(offset);
}

class PreadIoSource final
    : public strix::SparseLookupIoSource {
public:
    explicit PreadIoSource(int fd)
        : fd_(fd) {
        if (fd_ < 0) {
            throw std::invalid_argument(
                "pread I/O source requires a valid file descriptor");
        }
    }

    void read(
        std::uint64_t offset,
        std::span<std::byte> destination) override {
        pread_all(
            fd_,
            destination.data(),
            destination.size(),
            byte_offset(offset));
    }

private:
    int fd_ = -1;
};

Result bench_pread(
    int fd,
    const Options& options,
    const std::vector<std::uint64_t>& rows) {
    Result result;
    std::vector<std::uint8_t> buffer(
        static_cast<std::size_t>(options.row_bytes));
    for (const auto row : rows) {
        pread_all(fd, buffer.data(), buffer.size(),
                  row_offset(row, options.row_bytes));
        ++result.read_syscalls;
        result.physical_read_bytes += options.row_bytes;
        result.checksum ^= consume_row(buffer.data(), buffer.size());
    }
    return result;
}

Result bench_mmap(
    int fd,
    const Options& options,
    const std::vector<std::uint64_t>& rows) {
    const auto file_bytes =
        checked_mul(options.table_rows, options.row_bytes);
    if (file_bytes >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("mapped sparse table does not fit size_t");
    }
    void* mapping =
        mmap(nullptr, static_cast<std::size_t>(file_bytes),
             PROT_READ, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        fail("mmap");
    }

    if (madvise(mapping, static_cast<std::size_t>(file_bytes),
                MADV_RANDOM) != 0) {
        munmap(mapping, static_cast<std::size_t>(file_bytes));
        fail("madvise(MADV_RANDOM)");
    }
    if (options.drop_cache) {
        static_cast<void>(
            madvise(mapping, static_cast<std::size_t>(file_bytes),
                    MADV_DONTNEED));
    }

    Result result;
    std::vector<std::uint8_t> scratch(
        static_cast<std::size_t>(options.row_bytes));
    const auto* bytes = static_cast<const std::uint8_t*>(mapping);

    for (const auto row : rows) {
        const auto offset =
            checked_mul(row, options.row_bytes);
        std::memcpy(
            scratch.data(),
            bytes + static_cast<std::size_t>(offset),
            scratch.size());
        result.checksum ^= consume_row(scratch.data(), scratch.size());
    }

    if (munmap(mapping, static_cast<std::size_t>(file_bytes)) != 0) {
        fail("munmap");
    }
    return result;
}

struct CacheEntry {
    std::vector<std::uint8_t> bytes;
    std::list<std::uint64_t>::iterator lru;
};

Result bench_cache(
    int fd,
    const Options& options,
    const std::vector<std::uint64_t>& rows) {
    if (options.cache_rows == 0) {
        return bench_pread(fd, options, rows);
    }

    Result result;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t, CacheEntry> cache;
    cache.reserve(
        static_cast<std::size_t>(
            std::min<std::uint64_t>(options.cache_rows, options.table_rows)));

    for (const auto row : rows) {
        const auto found = cache.find(row);
        if (found != cache.end()) {
            ++result.cache_hits;
            lru.splice(lru.begin(), lru, found->second.lru);
            found->second.lru = lru.begin();
            result.checksum ^=
                consume_row(found->second.bytes.data(),
                            found->second.bytes.size());
            continue;
        }

        ++result.cache_misses;
        if (cache.size() >= options.cache_rows) {
            const auto victim = lru.back();
            cache.erase(victim);
            lru.pop_back();
        }

        CacheEntry entry;
        entry.bytes.resize(static_cast<std::size_t>(options.row_bytes));
        pread_all(fd, entry.bytes.data(), entry.bytes.size(),
                  row_offset(row, options.row_bytes));
        ++result.read_syscalls;
        result.physical_read_bytes += options.row_bytes;
        lru.push_front(row);
        entry.lru = lru.begin();
        const auto checksum =
            consume_row(entry.bytes.data(), entry.bytes.size());
        cache.emplace(row, std::move(entry));
        result.checksum ^= checksum;
    }

    return result;
}

Result bench_coalesced(
    int fd,
    const Options& options,
    const std::vector<std::uint64_t>& rows) {
    Result result;
    std::vector<std::uint8_t> scratch(
        static_cast<std::size_t>(options.row_bytes));

    for (std::size_t begin = 0;
         begin < rows.size();
         begin += static_cast<std::size_t>(options.window)) {
        const auto end =
            std::min(rows.size(),
                     begin + static_cast<std::size_t>(options.window));
        std::vector<std::uint64_t> unique(
            rows.begin() + static_cast<std::ptrdiff_t>(begin),
            rows.begin() + static_cast<std::ptrdiff_t>(end));
        std::sort(unique.begin(), unique.end());
        unique.erase(
            std::unique(unique.begin(), unique.end()),
            unique.end());

        struct Run {
            std::uint64_t first = 0;
            std::uint64_t count = 0;
            std::vector<std::uint8_t> bytes;
        };
        std::vector<Run> runs;

        for (const auto row : unique) {
            if (runs.empty() ||
                row != runs.back().first + runs.back().count) {
                runs.push_back(Run{row, 1, {}});
            } else {
                ++runs.back().count;
            }
        }

        for (auto& run : runs) {
            const auto bytes =
                checked_mul(run.count, options.row_bytes);
            run.bytes.resize(static_cast<std::size_t>(bytes));
            pread_all(fd, run.bytes.data(), run.bytes.size(),
                      row_offset(run.first, options.row_bytes));
            ++result.read_syscalls;
            result.physical_read_bytes += bytes;
        }

        for (std::size_t index = begin; index < end; ++index) {
            const auto row = rows[index];
            const Run* owner = nullptr;
            for (const auto& run : runs) {
                if (row >= run.first &&
                    row < run.first + run.count) {
                    owner = &run;
                    break;
                }
            }
            if (owner == nullptr) {
                throw std::logic_error("coalesced row has no owning run");
            }
            const auto within = row - owner->first;
            const auto byte_offset =
                checked_mul(within, options.row_bytes);
            std::memcpy(
                scratch.data(),
                owner->bytes.data() + static_cast<std::size_t>(byte_offset),
                scratch.size());
            result.checksum ^= consume_row(scratch.data(), scratch.size());
        }
    }

    return result;
}


Result bench_bounded_async(
    int fd,
    const Options& options,
    const std::vector<std::uint64_t>& rows) {
    Result result;
    PreadIoSource source(fd);

    for (std::size_t begin = 0;
         begin < rows.size();
         begin += static_cast<std::size_t>(options.window)) {
        const auto end =
            std::min(
                rows.size(),
                begin +
                    static_cast<std::size_t>(
                        options.window));
        const std::span<const std::uint64_t> window_rows(
            rows.data() + begin,
            end - begin);

        strix::SparseLookupRequest request;
        request.table_rows =
            options.table_rows;
        request.row_bytes =
            options.row_bytes;
        request.cache_capacity_rows = 0;
        request.max_rows_per_io =
            options.max_rows_per_io;
        request.max_inflight_ios =
            options.max_inflight_ios;
        request.row_ids =
            window_rows;

        const auto plan =
            strix::build_sparse_lookup_plan(
                request);
        const auto execution =
            strix::execute_sparse_lookup_io_plan(
                plan,
                source);

        result.read_syscalls +=
            execution.operations_completed;
        result.physical_read_bytes +=
            execution.bytes_read;

        for (const auto row : window_rows) {
            const std::byte* row_data = nullptr;
            for (std::size_t operation_index = 0;
                 operation_index < plan.io_plan.size();
                 ++operation_index) {
                const auto& operation =
                    plan.io_plan[operation_index];
                if (row < operation.first_row ||
                    row >=
                        operation.first_row +
                            operation.row_count) {
                    continue;
                }

                const auto within_rows =
                    row - operation.first_row;
                const auto within_bytes =
                    checked_mul(
                        within_rows,
                        options.row_bytes);
                const auto& payload =
                    execution.operation_payloads[
                        operation_index];
                if (within_bytes >
                        payload.size() ||
                    options.row_bytes >
                        payload.size() -
                            static_cast<std::size_t>(
                                within_bytes)) {
                    throw std::logic_error(
                        "bounded async payload does not contain requested row");
                }
                row_data =
                    payload.data() +
                    static_cast<std::size_t>(
                        within_bytes);
                break;
            }

            if (row_data == nullptr) {
                throw std::logic_error(
                    "bounded async row has no owning I/O operation");
            }

            result.checksum ^=
                consume_row(
                    reinterpret_cast<
                        const std::uint8_t*>(
                            row_data),
                    static_cast<std::size_t>(
                        options.row_bytes));
        }
    }

    return result;
}

void verify_file_size(int fd, const Options& options) {
    struct stat info {};
    if (fstat(fd, &info) != 0) {
        fail("fstat");
    }
    const auto expected =
        checked_mul(options.table_rows, options.row_bytes);
    if (info.st_size < 0 ||
        static_cast<std::uint64_t>(info.st_size) < expected) {
        throw std::runtime_error(
            "file is smaller than configured sparse table");
    }
}

void run_benchmark(const Options& options) {
    auto file = open_file(options.file, O_RDONLY);
    verify_file_size(file.fd, options);

    if (options.drop_cache) {
        static_cast<void>(
            posix_fadvise(file.fd, 0, 0, POSIX_FADV_DONTNEED));
    }

    const auto rows = make_requests(options);
    const auto before = faults();
    const auto start = std::chrono::steady_clock::now();

    Result result;
    if (options.mode == "pread") {
        result = bench_pread(file.fd, options, rows);
    } else if (options.mode == "mmap") {
        result = bench_mmap(file.fd, options, rows);
    } else if (options.mode == "pread-cache") {
        result = bench_cache(file.fd, options, rows);
    } else if (options.mode == "pread-coalesced") {
        result = bench_coalesced(file.fd, options, rows);
    } else if (options.mode == "pread-bounded-async") {
        result = bench_bounded_async(file.fd, options, rows);
    } else {
        throw std::invalid_argument(
            "mode must be pread, mmap, pread-cache, pread-coalesced or pread-bounded-async");
    }

    const auto stop = std::chrono::steady_clock::now();
    const auto after = faults();
    const auto seconds =
        std::chrono::duration<double>(stop - start).count();
    const auto requested_bytes =
        checked_mul(options.requests, options.row_bytes);

    std::cout << "{\n"
              << "  \"command\": \"bench\",\n"
              << "  \"mode\": \"" << options.mode << "\",\n"
              << "  \"pattern\": \"" << options.pattern << "\",\n"
              << "  \"table_rows\": " << options.table_rows << ",\n"
              << "  \"row_bytes\": " << options.row_bytes << ",\n"
              << "  \"requests\": " << options.requests << ",\n"
              << "  \"window\": " << options.window << ",\n"
              << "  \"cache_rows\": " << options.cache_rows << ",\n"
              << "  \"max_rows_per_io\": " << options.max_rows_per_io << ",\n"
              << "  \"max_inflight_ios\": " << options.max_inflight_ios << ",\n"
              << "  \"seconds\": " << seconds << ",\n"
              << "  \"requested_bytes\": " << requested_bytes << ",\n"
              << "  \"physical_read_bytes\": "
              << result.physical_read_bytes << ",\n"
              << "  \"read_syscalls\": " << result.read_syscalls << ",\n"
              << "  \"cache_hits\": " << result.cache_hits << ",\n"
              << "  \"cache_misses\": " << result.cache_misses << ",\n"
              << "  \"minor_faults\": " << (after.minor - before.minor) << ",\n"
              << "  \"major_faults\": " << (after.major - before.major) << ",\n"
              << "  \"requested_gib_s\": "
              << (seconds > 0.0
                      ? static_cast<double>(requested_bytes) /
                            (1024.0 * 1024.0 * 1024.0) / seconds
                      : 0.0)
              << ",\n"
              << "  \"checksum\": " << result.checksum << "\n"
              << "}\n";
}

}  // namespace

int main(int argc, char** argv) try {
    const auto options = parse_args(argc, argv);
    if (options.command == "prepare") {
        prepare_file(options);
    } else if (options.command == "bench") {
        run_benchmark(options);
    } else {
        throw std::invalid_argument(
            "first argument must be prepare or bench");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-sparse-lookup: " << error.what() << '\n';
    return 1;
}
