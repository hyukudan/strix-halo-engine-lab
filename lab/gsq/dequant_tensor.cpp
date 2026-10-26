// dequant_tensor FILE NAME : writes tensor NAME of GGUF FILE as raw little-endian float32 to stdout, row by row,
// using strix::ggml_dequant (bitwise llama.cpp's dequantize_row_*). GSQ tooling (variant files).
#include "strix/ggml_dequant.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: dequant_tensor FILE NAME");
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) throw std::runtime_error("cannot open file");
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
    std::vector<std::uint64_t> dims; std::uint32_t type = 0; std::uint64_t off = 0; bool found = false;
    for (std::uint64_t i = 0; i < nt; ++i) {
        const std::string name = str(); const std::uint32_t nd = u32();
        std::vector<std::uint64_t> d; for (std::uint32_t j = 0; j < nd; ++j) d.push_back(u64());
        const std::uint32_t t = u32(); const std::uint64_t o = u64();
        if (name == argv[2]) { dims = d; type = t; off = o; found = true; }
    }
    if (!found) throw std::runtime_error("tensor not found");
    const std::uint64_t data = (static_cast<std::uint64_t>(in.tellg()) + align - 1) / align * align;
    const std::size_t width = dims[0];
    std::uint64_t rows = 1; for (std::size_t j = 1; j < dims.size(); ++j) rows *= dims[j];
    const std::size_t rb = strix::ggml_dequant::row_bytes(type, width);
    std::vector<std::byte> row(rb); std::vector<float> out(width);
    in.seekg(static_cast<std::streamoff>(data + off));
    for (std::uint64_t r = 0; r < rows; ++r) {
        rd(row.data(), rb);
        strix::ggml_dequant::dequantize_row(type, row, out);
        if (std::fwrite(out.data(), 4, width, stdout) != width) throw std::runtime_error("write failed");
    }
    std::fprintf(stderr, "dequant %s type %u width %zu rows %llu\n", argv[2], type, width, static_cast<unsigned long long>(rows));
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
