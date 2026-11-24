#pragma once

#include "inkling2_mini_config.hpp"
#include "strix/spec/block_drafter.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strix::inkling2 {

// Serial CPU reference for one unpadded text sequence. After an I/O or
// arithmetic failure during a step, the session is invalid and must be reopened.
class MiniHostDecoder final : public spec::VerifyTarget {
public:
    MiniHostDecoder(const std::string& checkpoint_path, unsigned max_positions, const char* dspark = nullptr);
    ~MiniHostDecoder() override;

    MiniHostDecoder(const MiniHostDecoder&) = delete;
    MiniHostDecoder& operator=(const MiniHostDecoder&) = delete;

    // Skip the vocabulary projection on prompt rows whose logits are discarded.
    std::vector<float> step(unsigned token, bool want_logits);
    // A provided BF16 row of kHiddenWidth (an image row) in place of a token: it passes embed_norm like a token row.
    std::vector<float> step_row(std::span<const std::uint16_t> row, bool want_logits);

    void verify(std::span<const std::uint32_t> inputs, std::vector<std::uint32_t>& greedy) override;
    void commit(std::uint32_t rows) override;

    [[nodiscard]] bool has_dspark() const noexcept;
    spec::ChainDrafter* start_dspark(unsigned anchor, unsigned position);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::inkling2
