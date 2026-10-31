#include "dflash.hpp"

#include <chrono>
#include <iostream>

namespace df = strix::gemma4::dflash;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: strix-gemma4-dflash-probe DRAFTER.gguf ROPE.g4dr\n"
                  << "Host diagnostic: synthetic committed taps, embedding and 16-token head.\n"
                  << "RoPE table must cover positions 0..8, head width 128. No target qualification.\n";
        return 2;
    }
    try {
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        const auto file = df::DraftFile::open(argv[1]);
        const auto& w = file.weights();
        const auto loaded = Clock::now();
        const auto rope = df::RopeFile::open(argv[2]);
        df::DraftContext context;
        std::vector<float> taps(w.shape.tap_width());
        for (std::size_t i = 0; i < taps.size(); ++i)
            taps[i] = static_cast<float>(static_cast<int>((i * 3) % 29) - 14) / 16.0F;
        const std::array<std::uint64_t, 1> positions{0};
        context.append(w, rope.view(), taps, positions);
        const auto appended = Clock::now();
        std::vector<float> synthetic(std::size_t{16} * w.shape.hidden);
        for (std::size_t i = 0; i < synthetic.size(); ++i)
            synthetic[i] = static_cast<float>(static_cast<int>((i * 7 + 1) % 17) - 8) / 64.0F;
        strix::gemma4::LinearView assets{reinterpret_cast<const std::uint8_t*>(synthetic.data()), 16,
            w.shape.hidden, std::size_t{w.shape.hidden} * sizeof(float), strix::gemma4::WeightFormat::f32};
        const auto embedded = df::embed_block(3, assets);
        const auto hidden = df::block_forward(w, context, rope.view(), embedded, 1);
        const auto forwarded = Clock::now();
        const auto tokens = df::proposals_from_hidden(w, hidden, assets);
        const auto finished = Clock::now();
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cout << "identity " << df::arithmetic_id << " synthetic_assets true tensor_bytes " << w.tensor_bytes << '\n';
        std::cout << "proposals";
        for (auto t : tokens) std::cout << ' ' << t;
        std::cout << "\nload_ms " << ms(start, loaded) << " context_ms " << ms(loaded, appended)
                  << " block_ms " << ms(appended, forwarded) << " head_ms " << ms(forwarded, finished) << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
