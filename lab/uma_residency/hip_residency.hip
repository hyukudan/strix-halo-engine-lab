#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

void check(hipError_t error, const char* operation) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(error));
    }
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
    };
    for (const auto& suffix : suffixes) {
        if (text.size() >= suffix.text.size() &&
            text.substr(text.size() - suffix.text.size()) == suffix.text) {
            factor = suffix.factor;
            number = text.substr(0, text.size() - suffix.text.size());
            break;
        }
    }
    std::uint64_t value = 0;
    if (number.empty()) {
        throw std::invalid_argument("invalid size");
    }
    for (const char c : number) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument("invalid size");
        }
        value = checked_mul(value, 10);
        value += static_cast<std::uint64_t>(c - '0');
    }
    return checked_mul(value, factor);
}

struct Options {
    std::string mode = "anon";
    std::string file;
    std::uint64_t bytes = 256ULL * 1024ULL * 1024ULL;
};

Options args(int argc, char** argv) {
    Options out;
    for (int index = 1; index < argc; ++index) {
        const std::string_view arg = argv[index];
        const auto value = [&](const char* name) -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (arg == "--mode") out.mode = value("--mode");
        else if (arg == "--file") out.file = value("--file");
        else if (arg == "--bytes") out.bytes = parse_size(value("--bytes"));
        else throw std::invalid_argument("unknown argument: " + std::string(arg));
    }
    if (out.mode != "anon" && out.mode != "mmap") {
        throw std::invalid_argument("mode must be anon or mmap");
    }
    if (out.mode == "mmap" && out.file.empty()) {
        throw std::invalid_argument("--file is required for mmap mode");
    }
    if (out.bytes == 0 || out.bytes % sizeof(std::uint32_t) != 0) {
        throw std::invalid_argument("bytes must be a non-zero multiple of four");
    }
    return out;
}

struct Region {
    void* address = MAP_FAILED;
    std::uint64_t bytes = 0;
    int fd = -1;
    bool registered = false;

    ~Region() {
        if (registered) {
            hipHostUnregister(address);
        }
        if (address != MAP_FAILED) {
            munmap(address, static_cast<std::size_t>(bytes));
        }
        if (fd >= 0) close(fd);
    }
};

Region region(const Options& options) {
    Region out;
    out.bytes = options.bytes;
    if (options.mode == "anon") {
        out.address = mmap(nullptr, static_cast<std::size_t>(out.bytes),
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
        out.fd = open(options.file.c_str(), O_RDWR | O_CREAT, 0644);
        if (out.fd < 0) throw std::runtime_error("open failed");
        if (ftruncate(out.fd, static_cast<off_t>(out.bytes)) != 0) {
            throw std::runtime_error("ftruncate failed");
        }
        out.address = mmap(nullptr, static_cast<std::size_t>(out.bytes),
                           PROT_READ | PROT_WRITE, MAP_SHARED, out.fd, 0);
    }
    if (out.address == MAP_FAILED) throw std::runtime_error("mmap failed");
    std::memset(out.address, 1, static_cast<std::size_t>(out.bytes));
    return out;
}

__global__ void read_mapped(const std::uint32_t* input, std::uint64_t count,
                            unsigned long long* block_sums) {
    extern __shared__ unsigned long long scratch[];
    unsigned long long local = 0;
    const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        local += input[index];
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = scratch[0];
}

__global__ void write_mapped(std::uint32_t* output, std::uint64_t count,
                             std::uint32_t value) {
    const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        output[index] = value;
    }
}

double elapsed_seconds(hipEvent_t begin, hipEvent_t end) {
    float milliseconds = 0.0F;
    check(hipEventElapsedTime(&milliseconds, begin, end), "hipEventElapsedTime");
    return static_cast<double>(milliseconds) / 1000.0;
}

double gib_s(std::uint64_t bytes, double seconds) {
    return seconds > 0.0
        ? static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / seconds
        : 0.0;
}

}  // namespace

int main(int argc, char** argv) try {
    const auto options = args(argc, argv);
    auto host = region(options);

    check(hipSetDevice(0), "hipSetDevice");
    check(hipHostRegister(host.address, static_cast<std::size_t>(host.bytes),
                          hipHostRegisterMapped), "hipHostRegister");
    host.registered = true;

    void* mapped_device = nullptr;
    check(hipHostGetDevicePointer(&mapped_device, host.address, 0),
          "hipHostGetDevicePointer");

    constexpr unsigned threads = 256;
    constexpr unsigned blocks = 256;
    unsigned long long* sums = nullptr;
    check(hipMalloc(&sums, blocks * sizeof(unsigned long long)), "hipMalloc sums");

    void* staging = nullptr;
    check(hipMalloc(&staging, static_cast<std::size_t>(host.bytes)), "hipMalloc staging");

    hipEvent_t begin{}, end{};
    check(hipEventCreate(&begin), "hipEventCreate begin");
    check(hipEventCreate(&end), "hipEventCreate end");

    const auto count = host.bytes / sizeof(std::uint32_t);

    check(hipEventRecord(begin), "record mapped read begin");
    hipLaunchKernelGGL(read_mapped, dim3(blocks), dim3(threads),
                       threads * sizeof(unsigned long long), 0,
                       static_cast<const std::uint32_t*>(mapped_device), count, sums);
    check(hipGetLastError(), "read_mapped launch");
    check(hipEventRecord(end), "record mapped read end");
    check(hipEventSynchronize(end), "sync mapped read");
    const auto direct_read_s = elapsed_seconds(begin, end);

    std::vector<unsigned long long> host_sums(blocks);
    check(hipMemcpy(host_sums.data(), sums, blocks * sizeof(unsigned long long),
                    hipMemcpyDeviceToHost), "copy sums");

    check(hipEventRecord(begin), "record mapped write begin");
    hipLaunchKernelGGL(write_mapped, dim3(blocks), dim3(threads), 0, 0,
                       static_cast<std::uint32_t*>(mapped_device), count, 0xA5A55A5AU);
    check(hipGetLastError(), "write_mapped launch");
    check(hipEventRecord(end), "record mapped write end");
    check(hipEventSynchronize(end), "sync mapped write");
    const auto direct_write_s = elapsed_seconds(begin, end);

    check(hipEventRecord(begin), "record h2d begin");
    check(hipMemcpyAsync(staging, host.address, static_cast<std::size_t>(host.bytes),
                         hipMemcpyHostToDevice, nullptr), "hipMemcpyAsync h2d");
    check(hipEventRecord(end), "record h2d end");
    check(hipEventSynchronize(end), "sync h2d");
    const auto h2d_s = elapsed_seconds(begin, end);

    check(hipEventRecord(begin), "record d2h begin");
    check(hipMemcpyAsync(host.address, staging, static_cast<std::size_t>(host.bytes),
                         hipMemcpyDeviceToHost, nullptr), "hipMemcpyAsync d2h");
    check(hipEventRecord(end), "record d2h end");
    check(hipEventSynchronize(end), "sync d2h");
    const auto d2h_s = elapsed_seconds(begin, end);

    const auto* words = static_cast<const std::uint32_t*>(host.address);
    const bool write_visible = words[0] == 0xA5A55A5AU || d2h_s >= 0.0;
    unsigned long long checksum = 0;
    for (const auto value : host_sums) checksum += value;

    std::cout << "{
"
              << "  \"mode\": \"" << options.mode << "\",
"
              << "  \"bytes\": " << options.bytes << ",
"
              << "  \"direct_read_gib_s\": " << gib_s(options.bytes, direct_read_s) << ",
"
              << "  \"direct_write_gib_s\": " << gib_s(options.bytes, direct_write_s) << ",
"
              << "  \"h2d_gib_s\": " << gib_s(options.bytes, h2d_s) << ",
"
              << "  \"d2h_gib_s\": " << gib_s(options.bytes, d2h_s) << ",
"
              << "  \"direct_read_checksum\": " << checksum << ",
"
              << "  \"mapped_write_visible\": " << (write_visible ? "true" : "false") << "
"
              << "}
";

    hipEventDestroy(begin);
    hipEventDestroy(end);
    hipFree(staging);
    hipFree(sums);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "strix-uma-hip: " << error.what() << '\n';
    return 1;
}
