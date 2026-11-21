#pragma once

#include "inkling2_mini_config.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strix::inkling2 {

// Serial CPU reference for one unpadded text sequence. After an I/O or
// arithmetic failure during a step, the session is invalid and must be reopened.
class MiniHostDecoder {
public:
    MiniHostDecoder(const std::string& checkpoint_path, unsigned max_positions);
    ~MiniHostDecoder();

    MiniHostDecoder(const MiniHostDecoder&) = delete;
    MiniHostDecoder& operator=(const MiniHostDecoder&) = delete;

    // Skip the vocabulary projection on prompt rows whose logits are discarded.
    std::vector<float> step(unsigned token, bool want_logits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::inkling2
