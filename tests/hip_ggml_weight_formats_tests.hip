// Device element decode (strix::hip_ggml::element, used by every launcher's generic load_weight for the
// GSQ-RCO formats) == host strix::ggml_dequant (bitwise llama.cpp dequantize_row_*), every element.
// Random blocks with finite fp16 scales for each type; with GGUF paths as arguments, also the first,
// middle and last row of every tensor of a supported type.
#include "strix/ggml_dequant.hpp"
#include "strix/hip_ggml_weight_formats.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace dq = strix::ggml_dequant;
namespace hg = strix::hip_ggml;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template <std::uint32_t Type>
__global__ void decode_kernel(const void* weights, std::uint32_t row_width, std::uint32_t rows, float* out) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::uint64_t>(row_width) * rows) return;
    out[i] = hg::element<Type>(weights, row_width, i / row_width, static_cast<std::uint32_t>(i % row_width));
}

void launch(std::uint32_t type, const void* w, std::uint32_t width, std::uint32_t rows, float* out) {
    const std::uint64_t n = static_cast<std::uint64_t>(width) * rows;
    const dim3 grid(static_cast<std::uint32_t>((n + 255) / 256)), block(256);
    switch (type) {
#define CASE(T) case T: hipLaunchKernelGGL(decode_kernel<T>, grid, block, 0, nullptr, w, width, rows, out); break;
    CASE(42) CASE(2) CASE(6) CASE(8) CASE(11) CASE(12) CASE(13) CASE(14) CASE(18) CASE(20) CASE(21) CASE(22) CASE(23)
#undef CASE
    default: throw std::invalid_argument("type");
    }
    check(hipGetLastError(), "launch");
}

// Device decode of `rows` rows of `width` elements == host decode, bit for bit. Returns mismatches.
long long compare(std::uint32_t type, const std::vector<std::byte>& bytes, std::uint32_t width, std::uint32_t rows) {
    const std::size_t n = static_cast<std::size_t>(width) * rows;
    std::vector<float> host(n), device(n);
    const std::size_t rb = dq::row_bytes(type, width);
    for (std::uint32_t r = 0; r < rows; ++r)
        dq::dequantize_row(type, std::span<const std::byte>(bytes.data() + r * rb, rb), std::span<float>(host.data() + static_cast<std::size_t>(r) * width, width));
    void* dw = nullptr; float* dout = nullptr;
    check(hipMalloc(&dw, bytes.size()), "malloc w");
    check(hipMalloc(reinterpret_cast<void**>(&dout), n * 4), "malloc out");
    check(hipMemcpy(dw, bytes.data(), bytes.size(), hipMemcpyHostToDevice), "copy");
    launch(type, dw, width, rows, dout);
    check(hipMemcpy(device.data(), dout, n * 4, hipMemcpyDeviceToHost), "copy back");
    check(hipFree(dw), "free"); check(hipFree(dout), "free");
    long long bad = 0;
    for (std::size_t i = 0; i < n; ++i) bad += std::memcmp(&host[i], &device[i], 4) != 0;
    return bad;
}

void sanitize(std::uint32_t t, std::byte* blk, std::mt19937& rng) {
    std::vector<std::size_t> offs;
    switch (t) {
    case 11: offs = {108}; break;
    case 14: offs = {208}; break;
    case 12: case 13: offs = {0, 2}; break;
    default: offs = {0};
    }
    std::uniform_int_distribution<int> e(1, 29), m(0, 1023), s(0, 1);
    for (std::size_t o : offs) {
        const auto h = static_cast<std::uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
        std::memcpy(blk + o, &h, 2);
    }
}

long long file_rows(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("cannot open ") + path);
    auto rd = [&](void* p, std::size_t n) { in.read(static_cast<char*>(p), static_cast<std::streamsize>(n)); };
    auto u32 = [&] { std::uint32_t v; rd(&v, 4); return v; };
    auto u64 = [&] { std::uint64_t v; rd(&v, 8); return v; };
    auto str = [&] { std::string s(u64(), '\0'); rd(s.data(), s.size()); return s; };
    char magic[4]; rd(magic, 4); (void)u32();
    const std::uint64_t nt = u64(), nkv = u64();
    std::uint64_t align = 32;
    std::function<void(std::uint32_t)> skip = [&](std::uint32_t ty) {
        static const int sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (ty == 8) { (void)str(); return; }
        if (ty == 9) { const std::uint32_t et = u32(); const std::uint64_t n = u64(); for (std::uint64_t i = 0; i < n; ++i) skip(et); return; }
        in.seekg(sz[ty], std::ios::cur);
    };
    for (std::uint64_t i = 0; i < nkv; ++i) {
        const std::string k = str(); const std::uint32_t ty = u32();
        if (k == "general.alignment" && ty == 4) align = u32(); else skip(ty);
    }
    struct T { std::string name; std::vector<std::uint64_t> dims; std::uint32_t type; std::uint64_t off; };
    std::vector<T> ts;
    for (std::uint64_t i = 0; i < nt; ++i) {
        T t; t.name = str(); const std::uint32_t nd = u32();
        for (std::uint32_t d = 0; d < nd; ++d) t.dims.push_back(u64());
        t.type = u32(); t.off = u64(); ts.push_back(t);
    }
    const std::uint64_t data = (static_cast<std::uint64_t>(in.tellg()) + align - 1) / align * align;
    long long bad = 0, rows = 0, tensors = 0;
    for (const auto& t : ts) {
        const auto g = dq::geometry(t.type);
        if (!g || g->block_elements == 1) continue;
        const auto width = static_cast<std::uint32_t>(t.dims[0]);
        std::uint64_t nrows = 1;
        for (std::size_t d = 1; d < t.dims.size(); ++d) nrows *= t.dims[d];
        const std::size_t rb = dq::row_bytes(t.type, width);
        for (std::uint64_t r : {std::uint64_t{0}, nrows / 2, nrows - 1}) {
            std::vector<std::byte> bytes(rb);
            in.seekg(static_cast<std::streamoff>(data + t.off + r * rb)); rd(bytes.data(), rb);
            bad += compare(t.type, bytes, width, 1); ++rows;
        }
        ++tensors;
    }
    std::printf("file %s tensors %lld rows %lld mismatches %lld\n", path, tensors, rows, bad);
    return bad;
}

}  // namespace

int main(int argc, char** argv) try {
    long long total = 0;
    std::mt19937 rng(20261024);
    for (std::uint32_t t : {42U, 2U, 6U, 8U, 11U, 12U, 13U, 14U, 18U, 20U, 21U, 22U, 23U}) {
        const auto g = *dq::geometry(t);
        const std::uint32_t width = static_cast<std::uint32_t>(g.block_elements * 10), rows = 400;  // 640 for Q2_0 (down width)
        std::vector<std::byte> bytes(dq::row_bytes(t, width) * rows);
        for (auto& x : bytes) x = static_cast<std::byte>(rng() & 0xff);
        for (std::size_t b = 0; b < bytes.size() / g.block_bytes; ++b) sanitize(t, bytes.data() + b * g.block_bytes, rng);
        const long long bad = compare(t, bytes, width, rows);
        std::printf("random %-8s width %u rows %u mismatches %lld\n", std::string(dq::name(t)).c_str(), width, rows, bad);
        total += bad;
    }
    for (int i = 1; i < argc; ++i) total += file_rows(argv[i]);
    std::printf("%s\n", total == 0 ? "PASS" : "FAIL");
    return total == 0 ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
