#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_host.hpp"
#include "strix/head_argmax.hpp"

#include <algorithm>

namespace {

class CpuCliDecoder {
public:
    CpuCliDecoder(const std::string& checkpoint, unsigned max_positions) : decoder_(checkpoint, max_positions) {}

    unsigned prefill(std::span<const unsigned> prompt, std::span<float> logits) {
        for (const auto token : prompt.first(prompt.size() - 1U)) (void)decoder_.step(token, false);
        const auto row = decoder_.step(prompt.back(), true);
        if (!logits.empty()) std::ranges::copy(row, logits.begin());
        return pick(row);
    }

    unsigned decode(unsigned token) { return pick(decoder_.step(token, true)); }

private:
    static unsigned pick(std::span<const float> logits) {
        const auto pick = strix::head_argmax::reference(logits);
        if (pick.nonfinite != 0U || pick.index == strix::head_argmax::kNone)
            throw std::runtime_error("non-finite output logits");
        return pick.index;
    }

    strix::inkling2::MiniHostDecoder decoder_;
};

}  // namespace

int main(int argc, char** argv) {
    namespace cli = strix::inkling2::cli;
    if (argc < 3) {
        std::cerr << "usage: strix-inkling2-mini-cpu MODEL.safetensors TOKEN [TOKEN ...] [--generate N] [--logits FILE]\n";
        return 2;
    }
    try {
        const auto options = cli::parse(argc, argv);
        CpuCliDecoder decoder(argv[1], cli::context(options));
        cli::run(decoder, options);
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini CPU: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
