// strix-ds41-engram-pages: the model-file byte ranges of the Engram rows a prompt reads (every token's n-gram rows of
// every Engram layer), as sorted unique 4 KiB page offsets, one per line. A partial copy of the model (the Engram tables
// left sparse) gets exactly these pages from a full copy to run that prompt with the model's bits.
//   strix-ds41-engram-pages MODEL.gguf PROMPT (comma-separated ids) [ROWS]
#include "strix/deepseek41_model.hpp"
#include "strix/ds41_engram.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: strix-ds41-engram-pages MODEL.gguf PROMPT [ROWS]\n");
        return 2;
    }
    try {
        std::ifstream in(argv[2]);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        for (auto& c : text)
            if (c == ',' || c == '\n' || c == '\r') c = ' ';
        std::istringstream s(text);
        std::vector<std::uint32_t> tokens;
        std::uint64_t v = 0;
        while (s >> v) tokens.push_back(static_cast<std::uint32_t>(v));
        if (argc > 3) tokens.resize(std::min<std::size_t>(tokens.size(), std::stoul(argv[3])));
        const auto info = strix::deepseek41::read_file(argv[1]);
        const auto model = strix::deepseek41::validate_target(info);
        const auto tables = strix::ds41::engram_tables(model);
        strix::ds41::EngramHasher hasher(model.engram);
        const auto ids = hasher.hash_sequence(tokens);  // [token][layer][column]
        const std::uint32_t layers = hasher.layers(), cols = hasher.columns();
        std::vector<std::uint64_t> pages;
        for (std::size_t t = 0; t < tokens.size(); ++t)
            for (std::uint32_t l = 0; l < layers; ++l)
                for (std::uint32_t k = 0; k < cols; ++k) {
                    const std::uint64_t off = tables[l].file_offset + ids[(t * layers + l) * cols + k] * 264U;
                    pages.push_back(off / 4096U * 4096U);
                    pages.push_back((off + 263U) / 4096U * 4096U);
                }
        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        for (const auto p : pages) std::printf("%llu\n", static_cast<unsigned long long>(p));
        std::fprintf(stderr, "%zu tokens, %zu pages (%.1f MB)\n", tokens.size(), pages.size(), pages.size() * 4096.0 / 1e6);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strix-ds41-engram-pages: %s\n", e.what());
        return 1;
    }
    return 0;
}
