#pragma once

#include "strix/exec/program.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>

namespace strix::exec {

// Bind twice against the same stable fake device allocations, varying the replay inputs only.
// Everything inside an enqueued ABI packet (including a device descriptor's address, copy
// endpoints and dimensions) is baked. Its pointee is read when the graph executes. Builders
// using this check must serialize deterministic argument packets, including any padding.
// This observes the physical binding; a kernel's use of the pointed-to data still needs a
// replay comparison. It does not launch production thunks on the host.
inline void verify_dry_capture(const Program& before, const Program& after) {
    if (before.key != after.key || before.launches.empty() || before.launches.size() != after.launches.size())
        throw std::logic_error("step inputs changed captured graph shape");
    const auto arguments = [](const Program& program, const BoundLaunch& launch) {
        const auto bytes = std::span(program.args);
        if (launch.args_offset > bytes.size() || launch.args_bytes > bytes.size() - launch.args_offset)
            throw std::logic_error("invalid captured argument range at " + launch.entry);
        return bytes.subspan(launch.args_offset, launch.args_bytes);
    };
    for (std::size_t i = 0; i < before.launches.size(); ++i) {
        const auto& a = before.launches[i];
        const auto& b = after.launches[i];
        if (a.kernel != b.kernel || a.fn != b.fn || a.identity != b.identity || a.entry != b.entry ||
            a.chain_identity != b.chain_identity || a.rows != b.rows || a.layer != b.layer ||
            a.after != b.after || a.stream != b.stream || a.kernels != b.kernels || a.capturable != b.capturable ||
            a.grid.blocks != b.grid.blocks || a.grid.threads != b.grid.threads ||
            a.grid.shared_bytes != b.grid.shared_bytes || !std::ranges::equal(arguments(before, a), arguments(after, b)))
            throw std::logic_error("step inputs changed captured leaf " + std::to_string(i) + " " + a.entry);
    }
}

}  // namespace strix::exec
