#include "strix/engram/hasher.hpp"

#include <stdexcept>

namespace strix::engram {
namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}

} // namespace

void hash_position(const Spec& s, std::span<const std::uint64_t> offsets, std::span<const std::uint64_t> ngram,
                   std::span<std::uint64_t> rows) {
    const std::size_t layers = s.layer_ids.size(), cols = s.columns(), n = s.max_ngram_size;
    require(ngram.size() == n && rows.size() == layers * cols && offsets.size() == layers * cols, "Engram hash sizes");
    for (std::size_t e = 0; e < layers; ++e) {
        std::uint64_t rolling = ngram[0] * s.multipliers[e * n];
        for (std::size_t o = 1; o < n; ++o) {
            rolling ^= ngram[o] * s.multipliers[e * n + o];
            for (std::uint32_t h = 0; h < s.heads_per_order; ++h) {
                const std::size_t k = (o - 1) * s.heads_per_order + h;
                rows[e * cols + k] = rolling % s.moduli[e * cols + k] + offsets[e * cols + k];
            }
        }
    }
}

Hasher::Hasher(Spec spec) : spec_(std::move(spec)) {
    validate(spec_);
    pad_ = spec_.compress[spec_.pad_token];
    offsets_ = column_offsets(spec_);
    ngram_.resize(spec_.max_ngram_size);
}

void Hasher::push(std::uint32_t token, std::span<std::uint64_t> rows) {
    const auto& s = spec_;
    const std::size_t n = s.max_ngram_size;
    const bool dead = (token & kDeadBit) != 0U;
    token &= ~kDeadBit;
    const bool mapped = token < s.valid_ids();
    require((mapped || s.unmapped == Unmapped::pad) && rows.size() == std::size_t{s.layers()} * s.columns(),
            "Engram hash request");
    history_.push_back(dead ? kDead : mapped ? s.compress[token] : pad_);
    const std::size_t t = history_.size() - 1;
    bool blocked = false;
    for (std::size_t o = 0; o < n; ++o) {
        blocked = blocked || t < o || history_[t - o] == kDead;  // nothing reads through a dead position
        ngram_[o] = blocked ? pad_ : history_[t - o];
    }
    hash_position(s, offsets_, ngram_, rows);
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
