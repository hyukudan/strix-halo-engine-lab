#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_host.hpp"
#include "strix/serial_request.hpp"
#include "strix/session/prompt_windows.hpp"

#include <algorithm>
#include <iostream>

namespace {

class CpuCliDecoder final : public strix::serve::SerialGenerationDecoder {
public:
    CpuCliDecoder(const std::string& checkpoint, unsigned max_positions) : decoder_(checkpoint, max_positions) {}

    unsigned prefill(std::span<const unsigned> prompt, std::span<float> logits) override {
        unsigned next{};
        strix::session::prompt_windows::append_chunks(prompt, 1U, [&](auto chunk, bool last) {
            const auto row = decoder_.step(chunk.front(), last);
            if (last) next = select(row, logits);
        });
        return next;
    }

    unsigned decode(unsigned token, std::span<float> logits) override {
        return select(decoder_.step(token, true), logits);
    }

private:
    static unsigned select(std::span<const float> row, std::span<float> logits) {
        if (!logits.empty()) std::ranges::copy(row, logits.begin());
        return strix::select_greedy_token(row, strix::inkling2::kOutputVocabulary);
    }

    strix::inkling2::MiniHostDecoder decoder_;
};

}  // namespace

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    namespace serve = strix::serve;
    const serve::SerialGenerationFamily family{
        "strix-inkling2-mini-cpu", "Inkling 2 Mini CPU", "MODEL.safetensors",
        kVocabulary, kOutputVocabulary, kMaxContext, kMiniEosToken,
        [](const serve::SerialGenerationLoad& load) -> std::unique_ptr<serve::SerialGenerationDecoder> {
            return std::make_unique<CpuCliDecoder>(load.checkpoint, load.positions);
        }};
    if (argc < 3) {
        std::cerr << "usage: strix-inkling2-mini-cpu MODEL.safetensors TOKEN [TOKEN ...] [--profile exact|fast|turbo] [--generate N] [--logits FILE]"
                     " [--temperature T] [--top-k K] [--seed N]\n";
        return 2;
    }
    try {
        auto options = cli::parse(argc, argv, family);
        serve::run_serial_generation(argv[1], std::move(options.generation), family, std::cout,
            [&] { cli::write_profile(options.tier, std::cerr, false); });
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini CPU: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
