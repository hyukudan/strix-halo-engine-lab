#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_host.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

unsigned parse_number(std::string_view value) {
    unsigned number{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("expected an unsigned integer");
    }
    return number;
}

unsigned next_token(const std::vector<float>& logits) {
    return static_cast<unsigned>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void write_logits(const char* path, const std::vector<float>& logits) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("could not open logits output");
    for (float value : logits) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        unsigned char encoded[4];
        for (unsigned i = 0; i < 4; ++i) encoded[i] = static_cast<unsigned char>(bits >> (8U * i));
        output.write(reinterpret_cast<const char*>(encoded), sizeof(encoded));
    }
    output.close();
    if (!output) throw std::runtime_error("could not write logits output");
}

}  // namespace

int main(int argc, char** argv) {
    using namespace strix::inkling2;
    if (argc < 3) {
        std::cerr << "usage: strix-inkling2-mini-cpu MODEL.safetensors TOKEN [TOKEN ...] [--generate N] [--logits FILE]\n";
        return 2;
    }
    try {
        std::vector<unsigned> prompt;
        unsigned generate = 0;
        const char* logits_path = nullptr;
        for (int i = 2; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--generate" || argument == "--logits") {
                if (++i == argc) throw std::invalid_argument("missing option value");
                if (argument == "--generate") generate = parse_number(argv[i]);
                else logits_path = argv[i];
            } else {
                const auto token = parse_number(argument);
                if (token >= kVocabulary) throw std::out_of_range("token id exceeds the Mini vocabulary");
                prompt.push_back(token);
            }
        }
        if (prompt.empty() || prompt.size() > kMaxContext || generate > kMaxContext - prompt.size()) {
            throw std::invalid_argument("prompt and generation must fit the Mini context");
        }
        if (logits_path != nullptr && std::filesystem::equivalent(argv[1], logits_path)) {
            throw std::invalid_argument("logits output must differ from the checkpoint");
        }
        MiniHostDecoder decoder(argv[1], static_cast<unsigned>(prompt.size()) + generate);
        std::vector<float> logits;
        for (std::size_t i = 0; i < prompt.size(); ++i) logits = decoder.step(prompt[i], i + 1U == prompt.size());
        if (logits_path != nullptr) write_logits(logits_path, logits);

        unsigned token = next_token(logits);
        std::cout << "next_token " << token << '\n';
        if (generate != 0) {
            std::cout << "generated" << std::flush;
            for (unsigned i = 0; i < generate; ++i) {
                std::cout << ' ' << token << std::flush;
                if (token == kMiniEosToken || i + 1U == generate) break;
                token = next_token(decoder.step(token, true));
            }
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Inkling 2 Mini CPU: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
