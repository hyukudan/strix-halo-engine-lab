#include "strix/engram/hasher.hpp"

#include <array>
#include <stdexcept>

namespace strix::engram {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

} // namespace

void hash_layer(const Spec& s, std::span<const std::uint64_t> offsets, std::uint32_t e,
                std::span<const std::uint64_t> ngram, std::span<std::uint64_t> rows) {
    const std::size_t cols = s.columns(), n = s.max_ngram_size;
    require(e < s.layer_ids.size() && ngram.size() == n && rows.size() == cols &&
                offsets.size() == s.layer_ids.size() * cols,
            "Engram hash sizes");
    std::uint64_t rolling = ngram[0] * s.multipliers[e * n];
    for (std::size_t o = 1; o < n; ++o) {
        rolling ^= ngram[o] * s.multipliers[e * n + o];
        for (std::uint32_t h = 0; h < s.heads_per_order; ++h) {
            const std::size_t k = (o - 1) * s.heads_per_order + h;
            rows[k] = rolling % s.moduli[e * cols + k] + offsets[e * cols + k];
        }
    }
}

void hash_position(const Spec& s, std::span<const std::uint64_t> offsets, std::span<const std::uint64_t> ngram,
                   std::span<std::uint64_t> rows) {
    const std::size_t layers = s.layer_ids.size(), cols = s.columns();
    require(rows.size() == layers * cols, "Engram hash sizes");
    for (std::size_t e = 0; e < layers; ++e)
        hash_layer(s, offsets, static_cast<std::uint32_t>(e), ngram, rows.subspan(e * cols, cols));
}

Hasher::Hasher(Spec spec) : spec_(std::move(spec)) {
    validate(spec_);
    pad_ = spec_.compress[spec_.pad_token];
    offsets_ = column_offsets(spec_);
    ngram_.resize(spec_.max_ngram_size);
}

std::uint32_t Hasher::compressed(std::uint32_t token) const {
    const auto& s = spec_;
    const bool dead = (token & kDeadBit) != 0U;
    token &= ~kDeadBit;
    const bool mapped = token < s.valid_ids();
    require(mapped || s.unmapped == Unmapped::pad, "Engram hash request");
    return dead ? kDead : mapped ? s.compress[token] : pad_;
}

void Hasher::push(std::uint32_t token, std::span<std::uint64_t> rows) {
    require(rows.size() == std::size_t{spec_.layers()} * spec_.columns(), "Engram hash request");
    history_.push_back(compressed(token));
    fill_ngram(history_.size() - 1, [this](std::size_t i) { return history_[i]; }, ngram_);
    hash_position(spec_, offsets_, ngram_, rows);
}

void Hasher::rows_at(std::span<const std::uint32_t> tokens, std::size_t t, std::uint32_t e,
                     std::span<std::uint64_t> rows) const {
    const std::size_t n = spec_.max_ngram_size;
    require(t < tokens.size() && n <= kMaxStatelessNgram, "Engram rows_at request");
    std::array<std::uint64_t, kMaxStatelessNgram> buf{};
    const std::span<std::uint64_t> ngram(buf.data(), n);
    fill_ngram(t, [&](std::size_t i) { return compressed(tokens[i]); }, ngram);
    hash_layer(spec_, offsets_, e, ngram, rows);
}

std::vector<std::uint64_t> Hasher::hash_sequence(std::span<const std::uint32_t> tokens) {
    reset();
    const std::size_t per = static_cast<std::size_t>(layers()) * columns();
    std::vector<std::uint64_t> out(tokens.size() * per);
    for (std::size_t i = 0; i < tokens.size(); ++i) push(tokens[i], std::span<std::uint64_t>(out.data() + i * per, per));
    return out;
}

void Hasher::truncate(std::uint64_t positions) {
    require(positions <= history_.size(), "Engram history cannot grow by truncation");
    history_.resize(static_cast<std::size_t>(positions));
}

} // namespace strix::engram
