#pragma once

#include "strix/attn/hybrid_rope.hpp"

#include <vector>

namespace strix::attn_core {

class RotaryReferenceRow {
public:
    RotaryReferenceRow(unsigned dim, std::uint64_t position, double base, const attn::HybridYarn* yarn = nullptr)
        : cos_(dim / 2U), sin_(dim / 2U) {
        if (dim % 2U != 0) throw std::invalid_argument("rotary row: odd rotary width");
        attn::hybrid_rope_row(position, base, yarn, cos_, sin_);
    }

    void apply(std::span<float> rows, unsigned width, bool inverse = false) const {
        attn::hybrid_rope_tail(rows, width, cos_, sin_, inverse);
    }

    void apply_leading(std::span<float> rows, unsigned width) const {
        const auto dim = static_cast<unsigned>(2U * cos_.size());
        if (width == 0 || rows.size() % width != 0 || dim > width)
            throw std::invalid_argument("rotary row: invalid leading rotary shape");
        if (dim == 0) return;
        for (std::size_t offset = 0; offset < rows.size(); offset += width)
            apply(rows.subspan(offset, dim), dim);
    }

private:
    std::vector<float> cos_, sin_;
};

} // namespace strix::attn_core
