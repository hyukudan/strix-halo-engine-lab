#pragma once

#include "strix/exec/catalog.hpp"

#include <algorithm>

namespace strix::cuda::mix {

// Compatibility view of the generated CUDA rows. These are already in the
// builtin catalogue. Callers building a custom catalogue can add this span.
[[nodiscard]] inline std::span<const exec::KernelEntry> catalog_entries() noexcept {
    const auto entries = exec::Catalog::builtin_entries(exec::Backend::cuda);
    const auto is_readout = [](const exec::KernelEntry& entry) {
        return entry.key.op == graph::Op::mix && entry.key.form == "residual.bank";
    };
    const auto first = std::find_if(entries.begin(), entries.end(), is_readout);
    return {first, std::find_if_not(first, entries.end(), is_readout)};
}

} // namespace strix::cuda::mix
