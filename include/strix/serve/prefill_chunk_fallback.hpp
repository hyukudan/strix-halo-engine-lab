#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "strix/device_allocation_error.hpp"

namespace strix::serve {

// Loads a decoder at `chunk` rows. When the chunk is the family's implicit default (not set by the user) and the load
// fails with a typed device allocation failure (strix::DeviceAllocationError, thrown only for an out-of-memory status of
// a device-memory allocation, device::native::check_device_allocation: the row-dependent device buffers; a pinned host
// staging buffer refused for lack of memory is a generic runtime error), the load is retried once at
// `fallback` rows and one line says so. Nothing else is retried: not a message that merely contains "out of memory",
// and not std::bad_alloc, a host allocation failure that fewer device rows do not address.
// An explicit chunk, or a chunk already at or below the fallback, is never changed: the error propagates.
template <class Load>
auto load_with_chunk_fallback(std::uint32_t chunk, bool explicit_chunk, std::uint32_t fallback, std::ostream& log,
                              Load&& load) {
    if (explicit_chunk || fallback == 0U || chunk <= fallback) return load(chunk);
    try {
        return load(chunk);
    } catch (const DeviceAllocationError&) {
        log << "strix-prefill-chunk " << chunk << " -> " << fallback << " (device memory)\n";
    }
    return load(fallback);
}

// A service loaded at one candidate prefill chunk with the session slots it admitted (what load_with_chunk_ladder's
// load returns), and the ladder's pick: the kept service, its slots and its chunk.
template <class Service>
struct ChunkAdmission {
    Service service{};
    std::uint32_t slots = 0;
};

template <class Service>
struct ChunkLadderPick {
    Service service{};
    std::uint32_t slots = 0;
    std::uint32_t chunk = 0;
};

// Served chunk admission of an implicit server chunk: the largest candidate whose WHOLE load (row-scaled workspaces and
// session slots, allocated) keeps the slot capacity of the incumbent chunk. `chunks`: the candidates, largest first,
// the incumbent last. `cap`: the most slots one service uses (its lanes). load(chunk, last) builds the service at
// `chunk` rows with its slots allocated and returns ChunkAdmission{service, slots}; a non-last load declines with zero
// slots or the typed device allocation failure (strix::DeviceAllocationError); the last load (the incumbent) applies
// the caller's own refusals. The pick (ChunkLadderPick) adds the kept chunk. bound(chunk): an upper bound of the slots
// a candidate can admit, called only after the incumbent loaded (host arithmetic from the incumbent's measured budget),
// so a candidate that cannot keep the capacity is never loaded. Loads, in order:
//   1. the largest candidate: kept when it admits `cap` slots (no smaller chunk admits more than the cap);
//   2. the incumbent: its slots are the capacity to keep; kept when no other candidate can reach them;
//   3. the other candidates, largest first: the first that reaches the capacity is kept; else the incumbent again.
// The returned service is always the last one loaded, so side outputs of `load` describe it. One line names a change.
template <class Load, class Bound>
auto load_with_chunk_ladder(std::span<const std::uint32_t> chunks, std::uint32_t cap, std::ostream& log, Load&& load,
                            Bound&& bound) {
    using Service = decltype(load(std::uint32_t{}, true).service);
    using Admission = ChunkLadderPick<Service>;
    if (chunks.empty()) throw std::invalid_argument("chunk ladder: no candidate chunk");
    const std::size_t last = chunks.size() - 1U;
    const auto admit = [&](std::size_t i, bool final_load) {
        auto loaded = load(chunks[i], final_load);
        return Admission{std::move(loaded.service), loaded.slots, chunks[i]};
    };
    if (last == 0U) return admit(0U, true);
    constexpr std::uint32_t kUnmeasured = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> slots(chunks.size(), kUnmeasured);
    const auto attempt = [&](std::size_t i) -> std::optional<Admission> {
        try {
            auto admitted = admit(i, false);
            slots[i] = admitted.slots;
            if (admitted.slots != 0U) return admitted;
        } catch (const DeviceAllocationError&) {
            slots[i] = 0U;
        }
        return std::nullopt;
    };
    const auto changed = [&](const Admission& kept, std::uint32_t target) {
        log << "strix-prefill-chunk " << chunks[0] << " -> " << kept.chunk << " (slots " << slots[0] << " < "
            << target << " of the " << chunks[last] << "-row chunk)\n";
    };
    if (auto first = attempt(0U); first && first->slots >= cap) return std::move(*first);
    std::uint32_t target = 0;
    std::vector<bool> reachable(last, false);
    {
        auto incumbent = admit(last, true);
        target = incumbent.slots;
        bool any = false;
        for (std::size_t i = 0; i < last; ++i) {
            reachable[i] = slots[i] == kUnmeasured ? bound(chunks[i]) >= target : slots[i] >= target;
            any = any || reachable[i];
        }
        if (!any) {
            changed(incumbent, target);
            return incumbent;
        }
    }  // the incumbent is released before a larger candidate loads
    for (std::size_t i = 0; i < last; ++i) {
        if (!reachable[i]) continue;
        if (auto admitted = attempt(i); admitted && admitted->slots >= target) {
            if (i != 0U) changed(*admitted, target);
            return std::move(*admitted);
        }
    }
    auto incumbent = admit(last, true);
    changed(incumbent, target);
    return incumbent;
}

} // namespace strix::serve
