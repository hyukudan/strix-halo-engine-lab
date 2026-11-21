#pragma once

#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "inkling2/inkling2_mini_tokenizer.hpp"

#include <algorithm>
#include <memory>

int main(int argc, char** argv) {
    using namespace strix::inkling2;
#if defined(__HIPCC__)
    constexpr auto backend = "HIP";
    constexpr auto executable = "strix-inkling2-mini-hip";
#else
    constexpr auto backend = "CUDA";
    constexpr auto executable = "strix-inkling2-mini-cuda";
#endif
    if (argc < 3) {
        std::cerr << "usage: " << executable
                  << " MODEL.safetensors (TOKEN [TOKEN ...] | --prompt TEXT [--tokenizer FILE])"
                     " [--prefill-chunk N] [--packed-decode] [--generate N] [--logits FILE]\n";
        return 2;
    }
    try {
        auto options = cli::parse(argc, argv, 128U);
        std::shared_ptr<const strix::serve::BpeTokenizer> tokenizer;
        if (options.text) {
            namespace fs = std::filesystem;
            const fs::path asset = options.tokenizer ? fs::path(options.tokenizer) :
                fs::path(argv[1]).parent_path() / "tokenizer.json";
            if (options.logits_path && fs::exists(options.logits_path) && fs::equivalent(asset, options.logits_path))
                throw std::invalid_argument("logits output must differ from the tokenizer");
            tokenizer = load_mini_tokenizer(asset.string());
            options.prompt = tokenizer->encode(options.text, false);
        }
        const auto positions = cli::context(options);
        const auto max_rows = std::min(options.prefill_chunk, static_cast<unsigned>(options.prompt.size()));
        MiniResidentDecoder decoder(argv[1], positions, max_rows, options.packed_decode);
        std::vector<unsigned> generated;
        cli::run(decoder, options, tokenizer ? &generated : nullptr);
        if (tokenizer && !generated.empty()) std::cout << "text " << tokenizer->decode(generated, true) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini " << backend << ": " << error.what() << '\n';
        return 1;
    }
    return 0;
}
