#pragma once
// Loads every tensor the DSpark drafter reads (the draft file's three stages and the target's token
// embedding and output head) into memory and builds the Weights views of them.

#include "strix/ds41_dspark.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strix::ds41::dspark {

struct LoadedWeights {
    std::map<std::string, std::vector<std::uint8_t>> bytes;
    Weights weights;
};
// target: the deepseek41 model file (token_embd, output); draft: the deepseek41-dspark file.
[[nodiscard]] std::unique_ptr<LoadedWeights> load_weights(const std::string& target, const std::string& draft);

} // namespace strix::ds41::dspark
