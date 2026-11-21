#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/serve/bpe_tokenizer.hpp"

#include <stdexcept>

namespace strix::inkling2 {

// Uses the official text tokenizer without inserting BOS or formatting chat.
// Rendered prompts enable special-token parsing explicitly at the call site.
inline std::shared_ptr<const serve::BpeTokenizer> load_mini_tokenizer(const std::string& tokenizer_json) {
    auto tokenizer = serve::BpeTokenizer::load_hf_json(tokenizer_json);
    if (tokenizer->pre() != "o200k" || tokenizer->vocab_size() != kOutputVocabulary ||
        tokenizer->eos_token() != kMiniEosToken || tokenizer->find("<|begin_of_text|>") != 200028U) {
        throw std::invalid_argument("tokenizer.json does not match the Inkling 2 Mini text checkpoint");
    }
    return tokenizer;
}

}  // namespace strix::inkling2
