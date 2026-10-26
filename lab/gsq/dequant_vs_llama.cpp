// Bitwise check of strix::ggml_dequant against llama.cpp's own dequantize_row_* (ggml-quants.c of
// llama.cpp@9d1878a, compiled alongside with -ffp-contract=off). Inputs: random blocks with finite
// fp16 scales (seeded), plus optional real tensors: `dequant_vs_llama FILE.gguf` samples rows of every
// tensor whose type both sides support.
#include "strix/ggml_dequant.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

extern "C" {
void dequantize_row_q2_0(const void*, float*, int64_t);
void dequantize_row_q4_0(const void*, float*, int64_t);
void dequantize_row_q5_0(const void*, float*, int64_t);
void dequantize_row_q5_1(const void*, float*, int64_t);
void dequantize_row_q8_0(const void*, float*, int64_t);
void dequantize_row_q3_K(const void*, float*, int64_t);
void dequantize_row_q4_K(const void*, float*, int64_t);
void dequantize_row_q5_K(const void*, float*, int64_t);
void dequantize_row_q6_K(const void*, float*, int64_t);
void dequantize_row_iq4_nl(const void*, float*, int64_t);
void dequantize_row_iq4_xs(const void*, float*, int64_t);
void dequantize_row_iq2_s(const void*, float*, int64_t);
void dequantize_row_iq3_xxs(const void*, float*, int64_t);
void dequantize_row_iq3_s(const void*, float*, int64_t);
}

namespace dq = strix::ggml_dequant;
using Fn = void (*)(const void*, float*, int64_t);

static Fn llama_fn(uint32_t t) {
    switch (t) {
    case 42: return dequantize_row_q2_0; case 2: return dequantize_row_q4_0; case 6: return dequantize_row_q5_0;
    case 7: return dequantize_row_q5_1; case 8: return dequantize_row_q8_0; case 11: return dequantize_row_q3_K;
    case 12: return dequantize_row_q4_K; case 13: return dequantize_row_q5_K; case 14: return dequantize_row_q6_K;
    case 20: return dequantize_row_iq4_nl; case 23: return dequantize_row_iq4_xs; case 22: return dequantize_row_iq2_s;
    case 18: return dequantize_row_iq3_xxs; case 21: return dequantize_row_iq3_s; default: return nullptr;
    }
}

static long long compare(uint32_t t, const std::vector<std::byte>& bytes, size_t n) {
    std::vector<float> a(n), b(n);
    dq::dequantize_row(t, bytes, a);
    llama_fn(t)(bytes.data(), b.data(), static_cast<int64_t>(n));
    long long bad = 0;
    for (size_t i = 0; i < n; ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;
    return bad;
}

// fp16 scale fields per type (byte offsets inside the block) are overwritten with finite values.
static void sanitize(uint32_t t, std::byte* blk, std::mt19937& rng) {
    std::vector<size_t> offs;
    switch (t) {
    case 11: offs = {108}; break; case 14: offs = {208}; break;
    case 12: case 13: offs = {0, 2}; break; case 7: offs = {0, 2}; break;
    default: offs = {0};
    }
    std::uniform_int_distribution<int> e(1, 29), m(0, 1023), s(0, 1);
    for (size_t o : offs) {
        const uint16_t h = static_cast<uint16_t>((s(rng) << 15) | (e(rng) << 10) | m(rng));
        std::memcpy(blk + o, &h, 2);
    }
}

int main(int argc, char** argv) {
    long long total_bad = 0;
    std::mt19937 rng(20261024);
    const uint32_t types[] = {42, 2, 6, 7, 8, 11, 12, 13, 14, 20, 23, 22, 18, 21};
    for (uint32_t t : types) {
        const auto g = *dq::geometry(t);
        const size_t blocks = 4096;
        std::vector<std::byte> bytes(blocks * g.block_bytes);
        for (auto& x : bytes) x = static_cast<std::byte>(rng() & 0xff);
        for (size_t b = 0; b < blocks; ++b) sanitize(t, bytes.data() + b * g.block_bytes, rng);
        const long long bad = compare(t, bytes, blocks * g.block_elements);
        std::printf("random %-8s blocks %zu values %zu mismatches %lld\n", std::string(dq::name(t)).c_str(), blocks,
                    blocks * g.block_elements, bad);
        total_bad += bad;
    }
    // Real tensors: argv = GGUF files; tensor table read by a minimal parser.
    for (int f = 1; f < argc; ++f) {
        std::ifstream in(argv[f], std::ios::binary);
        auto rd = [&](void* p, size_t n) { in.read(static_cast<char*>(p), static_cast<std::streamsize>(n)); };
        auto u32 = [&] { uint32_t v; rd(&v, 4); return v; };
        auto u64 = [&] { uint64_t v; rd(&v, 8); return v; };
        auto str = [&] { std::string s(u64(), '\0'); rd(s.data(), s.size()); return s; };
        char magic[4]; rd(magic, 4); u32();
        const uint64_t nt = u64(), nkv = u64();
        uint64_t align = 32;
        std::function<void(uint32_t)> skip = [&](uint32_t ty) {
            static const int sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
            if (ty == 8) { str(); return; }
            if (ty == 9) { const uint32_t et = u32(); const uint64_t n = u64(); for (uint64_t i = 0; i < n; ++i) skip(et); return; }
            in.seekg(sz[ty], std::ios::cur);
        };
        for (uint64_t i = 0; i < nkv; ++i) {
            const std::string k = str(); const uint32_t ty = u32();
            if (k == "general.alignment" && ty == 4) align = u32(); else skip(ty);
        }
        struct T { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t off; };
        std::vector<T> ts;
        for (uint64_t i = 0; i < nt; ++i) {
            T t; t.name = str(); const uint32_t nd = u32(); for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(u64());
            t.type = u32(); t.off = u64(); ts.push_back(t);
        }
        const uint64_t data = (static_cast<uint64_t>(in.tellg()) + align - 1) / align * align;
        long long file_bad = 0, rows = 0;
        for (const auto& t : ts) {
            if (!llama_fn(t.type)) continue;
            const size_t cols = t.dims[0];
            uint64_t nrows = 1; for (size_t d = 1; d < t.dims.size(); ++d) nrows *= t.dims[d];
            const size_t rb = dq::row_bytes(t.type, cols);
            for (uint64_t r : {uint64_t{0}, nrows / 2, nrows - 1}) {
                std::vector<std::byte> bytes(rb);
                in.seekg(static_cast<std::streamoff>(data + t.off + r * rb)); rd(bytes.data(), rb);
                file_bad += compare(t.type, bytes, cols); ++rows;
            }
        }
        std::printf("file %s rows %lld mismatches %lld\n", argv[f], rows, file_bad);
        total_bad += file_bad;
    }
    std::printf(total_bad == 0 ? "PASS\n" : "FAIL\n");
    return total_bad == 0 ? 0 : 1;
}
