#pragma once

#include "strix/exec/profiler.hpp"

namespace strix::device { struct Bound; }

namespace strix::exec::bench {

// Synthetic R1 entry cases for the X6 dispatcher. Null means another family owns the ID.
[[nodiscard]] const KernelEntry* find_moer_entry(const Options& options);
[[nodiscard]] Output run_moer_entry(const Options& options, const KernelEntry& entry, const device::Bound& bound);

}  // namespace strix::exec::bench
