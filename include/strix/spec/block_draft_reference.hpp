#pragma once

#include "strix/weight/block_draft.hpp"
#include <functional>

namespace strix::spec::reference {
// Scalar oracle, FP64 intermediates, row-major matrices; deliberately not a
// device-bitwise reference. Device qualification must declare rounding policy.
struct Matrix {
    std::size_t rows{}, cols{};
    std::vector<double> data;
    Matrix() = default;
    Matrix(std::size_t r, std::size_t c);
    double& operator()(std::size_t r, std::size_t c) { return data.at(r * cols + c); }
    double operator()(std::size_t r, std::size_t c) const { return data.at(r * cols + c); }
};
// Borrowed weights in checkpoint order. Caller owns storage for the whole call.
using Weights = std::function<std::span<const float>(const std::string&)>;

Matrix linear(const Matrix&, std::span<const float> weights, std::size_t outputs);
Matrix rms(const Matrix&, std::span<const float> gain, double epsilon);
// dynamic = [row, phase=2, tap, group]; base = [phase=2, tap, channel].
Matrix convolve(const Matrix&, const Matrix& dynamic, std::span<const float> base,
                std::uint32_t group_size, std::uint32_t taps, std::uint32_t phase);

// ONE pass over [anchor embedding, mask embeddings...]. Features contain only
// committed target rows, concatenated in config.target_layers order. The slow
// oracle reconstructs draft KV from those features instead of persisting it.
// first_position is the absolute position of features[0]. Supply enough history
// for the sliding window. No target KV is shared or modified.
Matrix forward(const weight::BlockDraftConfig&, const Weights&, const Matrix& features,
               const Matrix& noise, std::uint64_t first_position = 0);

struct Candidates {
    std::vector<std::vector<std::uint32_t>> ids;
    std::vector<std::vector<double>> probabilities;
    std::vector<std::uint32_t> path;
};
// logits = RAW target-head projection of the last block-1 normalized hidden
// rows, restricted to real target vocabulary. Padding ids are never candidates.
// temperature==0: greedy; otherwise one caller-provided U[0,1) per position.
// Sparse conditional proposal q is returned for a future rejection sampler.
Candidates select(const weight::BlockDraftConfig&, const Weights&, const Matrix& hidden,
                  const Matrix& logits, std::uint32_t anchor, double temperature = 0,
                  std::span<const double> uniforms = {});
} // namespace strix::spec::reference
