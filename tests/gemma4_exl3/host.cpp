#include "gemma4/exl3/format.hpp"
#include "gemma4/model/model.hpp"
#include "gemma4/dflash/dflash.hpp"
#include <cstring>
#include <fstream>
#include <iostream>

namespace g4 = strix::gemma4;
namespace ex = g4::exl3;
namespace {
template <class T> std::vector<T> read(const std::string& name) {
    std::ifstream f(std::filesystem::path(G4_EXL3_FIXTURES) / name, std::ios::binary | std::ios::ate);
    if (!f || std::size_t(f.tellg()) % sizeof(T)) throw std::runtime_error("fixture read");
    std::vector<T> data(std::size_t(f.tellg()) / sizeof(T));
    f.seekg(0); f.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(T)));
    if (!f) throw std::runtime_error("truncated fixture");
    return data;
}
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
}
int main() {
    try {
        // Identity auto-selection table: fixed per pack and session maximum concurrency.
        static_assert(ex::auto_version("M4", 8) == 3 && ex::auto_version("M4", 4) == 3 && ex::auto_version("M4", 1) == 3);
        static_assert(ex::auto_version("M5", 8) == 3 && ex::auto_version("M5", 4) == 1 && ex::auto_version("M5", 1) == 1);
        static_assert(ex::auto_version("M6", 1) == 3 && ex::auto_version("M6", 8) == 3 && ex::auto_version("U5", 8) == 1);
        static_assert(ex::auto_version("", 8) == 1);
        if (ex::forced_version() == 0) {
            // Unconfigured processes use v1; a session fixes it once and a different request is refused.
            if (ex::configure_identity("U5", 8) != 1 || ex::core_version() != 1) throw std::runtime_error("auto identity");
            bool refused = false;
            try { (void)ex::configure_identity("M4", 8); } catch (const std::logic_error&) { refused = true; }
            if (!refused || ex::core_version() != 1) throw std::runtime_error("identity must stay fixed per session");
        }
        const auto table = read<std::uint16_t>("codebook.f16");
        require(table.size() == 65536, "codebook shape");
        for (unsigned i = 0; i < table.size(); ++i)
            require(g4::bit_cast<unsigned>(ex::codebook(i)) == g4::bit_cast<unsigned>(g4::fp16_value(table[i])), "codebook bytes");
        const auto x = read<float>("input.f32");
        for (const auto k : {4U, 5U, 6U, 8U}) {
            const auto prefix = "k" + std::to_string(k);
            auto data = read<std::uint8_t>(prefix + ".bin");
            g4::LinearView w{data.data(), 128, 256, 256ULL * k / 8, g4::WeightFormat::exl3_mul1};
            require(g4::weight_storage_bytes(w) == data.size(), "storage geometry");
            const auto rotated = read<std::uint16_t>(prefix + "-rotated.f16");
            for (unsigned i = 0; i < w.inputs; ++i)
                for (unsigned o = 0; o < w.outputs; ++o)
                    require(ex::rotated_at(w, i, o) == g4::fp16_value(rotated[std::size_t(i) * w.outputs + o]), "tile decode bytes");
            const auto expected = read<float>(prefix + "-output.f32");
            for (unsigned rows = 1; rows <= 8; ++rows) {
                std::vector<float> out(std::size_t(rows) * w.outputs);
                ex::linear(w, x.data(), rows, out.data());
                require(std::memcmp(out.data(), expected.data(), out.size() * sizeof(float)) == 0, "linear golden/row invariance");
            }
            g4::dflash::Weights draft; draft.shape.hidden = w.inputs;
            const auto proposals = g4::dflash::proposals_from_hidden(draft, x, w);
            for (unsigned r = 0; r < 7; ++r) {
                unsigned best = 0;
                for (unsigned o = 1; o < w.outputs; ++o)
                    if (g4::better(expected[(r + 1) * w.outputs + o], o, expected[(r + 1) * w.outputs + best], best)) best = o;
                require(proposals[r] == best, "separate EXL3 draft head/mask row selection");
            }
            std::vector<float> overflow(w.inputs, g4::bit_cast<float>(0x7f7fffffU)), result(w.outputs);
            bool overflow_refused = false;
            try { ex::linear(w, overflow.data(), 1, result.data()); } catch (const std::exception&) { overflow_refused = true; }
            require(overflow_refused, "FP16 input overflow accepted");
            data[0] = 0; data[1] = 0x7c;
            bool rejected = false;
            try { ex::validate(w); } catch (const std::exception&) { rejected = true; }
            require(rejected, "nonfinite scale accepted");
            auto invalid = w; invalid.inputs = 127;
            require(!ex::geometry(invalid), "unaligned geometry accepted");
        }
        std::cout << "EXL3 codebook, circular tiles, transforms, R1..8 EXACT; invalid scales rejected\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
