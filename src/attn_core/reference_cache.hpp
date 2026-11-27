#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strix::attn_core::detail {

template<class T>
void reserve_append(std::vector<T>& values, std::size_t elements) {
    const auto required = values.size() + elements;
    if (required > values.capacity())
        values.reserve(std::max(required, values.capacity() + values.capacity() / 2U));
}

} // namespace strix::attn_core::detail
