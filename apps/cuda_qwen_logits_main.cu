// Diagnostic teacher forcing over the production CUDA engine; no sampling or draft tokens.
#include "strix/cuda_qwen_prefill.hpp"
#include "strix/device/cuda_enumerator.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qp = strix::cuda::qwen_prefill;
static std::vector<std::uint32_t> ids(const char* path) {
    std::ifstream f(path);
    std::vector<std::uint32_t> result;
    std::uint64_t value;
    while (f >> value) {
        if (value >= 248320) throw std::invalid_argument("token outside vocabulary");
        result.push_back(static_cast<std::uint32_t>(value));
    }
    if (!f.eof() || result.size() < 2) throw std::invalid_argument("invalid token IDs file");
    return result;
}
template<class T> static void put(std::ofstream& f, T value) {
    f.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
int main(int argc, char** argv) {
    try {
        strix::device::bind_cuda_main(argc, argv);
        if (argc != 9) throw std::invalid_argument("usage: strix-cuda-qwen-logits MODEL TABLES ROPE IDS POSITIONS CAP OUT BLOCK_ROWS (text uint IDs/positions; positions predict IDS[p+1])");
        const auto input = ids(argv[4]);
        std::ifstream pf(argv[5]);
        std::vector<std::uint64_t> positions;
        std::uint64_t pos;
        while (pf >> pos) {
            if (pos >= input.size() - 1 || (!positions.empty() && pos <= positions.back()))
                throw std::invalid_argument("positions must ascend and have a next-token label");
            positions.push_back(pos);
        }
        if (!pf.eof() || positions.empty()) throw std::invalid_argument("invalid positions");
        const auto cap = std::stoull(argv[6]), block = std::stoull(argv[8]);
        if (cap < input.size() || cap > UINT32_MAX || !block || block > UINT32_MAX)
            throw std::invalid_argument("invalid capacity/block rows");
        if (std::filesystem::exists(argv[7])) throw std::invalid_argument("output already exists");
        qp::Config config;
        config.model_gguf = argv[1]; config.tables_dir = argv[2]; config.rope_table = argv[3];
        config.max_positions = static_cast<std::uint32_t>(cap);
        config.block_schedule = {static_cast<std::uint32_t>(block)};
        config.decode = true; config.export_state = false; config.release_by_layer = false;
        qp::Prefill engine(config);
        std::cerr << "teacher engine: " << engine.describe() << '\n';
        const auto identity = engine.numerical_identity();
        std::cerr << "teacher route: prefill=" << identity.prefill << " decode=" << identity.decode
                  << " storage=" << identity.storage << " hash=" << identity.hash << '\n';
        std::ofstream output(argv[7], std::ios::binary);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write("TMLLOG02", 8);
        put<std::uint64_t>(output, 248320); put<std::uint64_t>(output, positions.size());
        put<std::uint64_t>(output, input.size());
        output.write(reinterpret_cast<const char*>(input.data()), input.size() * sizeof(input[0]));
        output.write(reinterpret_cast<const char*>(positions.data()), positions.size() * sizeof(positions[0]));
        // Prefix rows before the first selection need no LM head. Every later input is still consumed.
        if (positions.front()) engine.run(input, positions.front(), [](const qp::Piece&) {});
        std::size_t selected = 0;
        for (std::uint64_t p = positions.front(); p <= positions.back(); ++p) {
            const bool save = p == positions[selected];
            const auto row = engine.step(&input[p], 1, p, save);
            if (!save) continue;
            if (row.logits.size() != 248320) throw std::runtime_error("unexpected logit extent");
            for (float v : row.logits) if (!std::isfinite(v)) throw std::runtime_error("nonfinite logits");
            put(output, input[p]); put(output, p);
            output.write(reinterpret_cast<const char*>(row.logits.data()), row.logits.size() * sizeof(float));
            ++selected;
        }
        output.write("TMLEOF01", 8); put<std::uint64_t>(output, positions.size());
        output.close();
        std::cout << "TEACHER LOGITS PASS rows=" << positions.size() << " conditioned=" << positions.back()+1
                  << " scope=scalar-decode-full-vocabulary\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
