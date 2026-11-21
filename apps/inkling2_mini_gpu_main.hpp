#pragma once

#include "inkling2_mini_cli.hpp"

#include "inkling2/inkling2_mini_resident_decoder.hpp"
#include "inkling2/inkling2_mini_tokenizer.hpp"

#include <algorithm>
#include <memory>

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    namespace session = strix::session;
#if defined(__HIPCC__)
    constexpr auto backend = "HIP";
    constexpr auto executable = "strix-inkling2-mini-hip";
    // Single-token decode runs the experts on the one-row packed GEMV (gfx1151, 128 tokens:
    // 19.7 tok/s against 16.0 tok/s on the grouped tile kernel); --grouped-decode keeps the tile kernel.
    constexpr bool packed_decode_default = true;
#else
    constexpr auto backend = "CUDA";
    constexpr auto executable = "strix-inkling2-mini-cuda";
    constexpr bool packed_decode_default = false;  // grouped until a CUDA receipt says otherwise
#endif
    if (argc < 3) {
        std::cerr << "usage: " << executable
                  << " MODEL.safetensors (TOKEN [TOKEN ...] | --prompt TEXT [--tokenizer FILE])"
                     " [--prefill-chunk N] [--packed-decode | --grouped-decode] [--generate N] [--logits FILE]\n";
        return 2;
    }
    try {
        // Larger chunks give each expert more rows per grouped tile; 2048 was the fastest
        // 4096-token prefill on gfx1151 (128: 12.8 s, 512: 6.7 s, 2048: 4.8 s, 4096: 6.2 s).
        auto options = cli::parse(argc, argv, 2048U);
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
        const auto load_start = session::CliClock::now();
        if (options.packed_decode && options.grouped_decode)
            throw std::invalid_argument("--packed-decode and --grouped-decode are exclusive");
        const bool packed_decode = options.packed_decode || (packed_decode_default && !options.grouped_decode);
        MiniResidentDecoder decoder(argv[1], positions, max_rows, packed_decode);
        session::write_load_timing(std::cout, load_start);
        std::vector<unsigned> generated;
        cli::run(decoder, options, tokenizer ? &generated : nullptr);
        if (tokenizer && !generated.empty())
            session::write_generated_text(std::cout, tokenizer->decode(generated, true));
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini " << backend << ": " << error.what() << '\n';
        return 1;
    }
    return 0;
}
