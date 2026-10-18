// strix-tokenize: the server's tokenizer from a GGUF file, for prompts and checks.
//   strix-tokenize MODEL.gguf encode FILE   token IDs of FILE's bytes (special tokens parsed), comma-separated
//   strix-tokenize MODEL.gguf plain FILE    same without special-token parsing
//   strix-tokenize MODEL.gguf decode FILE   text of the comma/space-separated token IDs in FILE
#include "strix/serve/bpe_tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

int main(int argc, char** argv) try {
    if (argc != 4) {
        std::cerr << "usage: strix-tokenize MODEL.gguf encode|plain|decode FILE\n";
        return 2;
    }
    const std::string mode = argv[2];
    std::ifstream file(argv[3], std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot read ") + argv[3]);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const auto tokenizer = strix::serve::BpeTokenizer::load_gguf(argv[1]);
    if (mode == "encode" || mode == "plain") {
        const auto ids = tokenizer->encode(text, mode == "encode");
        for (std::size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
        std::cout << '\n';
        return 0;
    }
    if (mode == "decode") {
        std::string spaced = text;
        for (auto& c : spaced) if (c == ',') c = ' ';
        std::istringstream words(spaced);
        std::vector<std::uint32_t> ids;
        for (std::uint64_t id = 0; words >> id;) ids.push_back(static_cast<std::uint32_t>(id));
        std::cout << tokenizer->decode(ids);
        return 0;
    }
    std::cerr << "unknown mode " << mode << '\n';
    return 2;
} catch (const std::exception& error) {
    std::cerr << "strix-tokenize: " << error.what() << '\n';
    return 1;
}
