// RoPE table of the MiniMax engine (strix::minimax::msa::rope_row, theta 1e7, cos/sin in double on the node's libm) for
// the RTX PRO prefill: positions [0, N) as N x (32 cos + 32 sin) FP32, little endian.
// build: c++ -O2 -std=c++20 -Iinclude tools/minimax/rtxpf_rope_dump.cpp src/minimax_msa.cpp -o rope_dump
#include "strix/minimax_msa.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: rope_dump N OUT.bin\n"); return 2; }
    const unsigned long long n = std::strtoull(argv[1], nullptr, 10);
    std::FILE* f = std::fopen(argv[2], "wb");
    if (!f) return 1;
    for (unsigned long long p = 0; p < n; ++p) {
        const auto r = strix::minimax::msa::rope_row(p, 1e7);
        std::fwrite(r.cos.data(), 4, r.cos.size(), f);
        std::fwrite(r.sin.data(), 4, r.sin.size(), f);
    }
    std::fclose(f);
    return 0;
}
