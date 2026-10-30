#pragma once

#include "strix/minimax/prefill_types.hpp"
#include <array>
#include <cstring>
#include <stdexcept>

namespace strix::minimax::serve {
struct Control {
    std::uint32_t magic = 0x43534d4dU;  // "MMSC"
    std::uint32_t version = 3;
    std::uint32_t command = 0;
    std::uint64_t task_id = 0, parent_task_id = 0, slot_generation = 0;
    strix::minimax::engine::PfProgress active{}, parked{};
    std::uint32_t rows = 0;             // step: rows used in row[]
    std::uint32_t pslot = 0, plen = 0;  // prefill: slot and prompt length
    std::uint32_t pflags = 0, bank = 0;
    std::uint64_t round = 0, ack_round = ~std::uint64_t{0};
    strix::minimax::engine::Row row[8]{};
    std::uint32_t layers = 0;  // unit
    std::uint32_t append = 0;  // step
    std::uint32_t dslots = 0;  // draft
    std::uint32_t dslot[8]{}, danchor[8]{};
    std::uint64_t dpos[8]{};
};
static_assert(sizeof(Control) <= 512);

template<class Engine>
Control exchange(Engine& eng, const Control& out) try {
    std::array<std::byte, 512> o{}, i{};
    Control local = out;
    local.active = eng.prefill_progress();
    local.parked = eng.prefill_progress(true);
    std::memcpy(o.data(), &local, sizeof(Control));
    eng.exchange_bytes(o.data(), i.data(), o.size());
    Control in;
    std::memcpy(&in, i.data(), sizeof(Control));
    if (in.magic != out.magic) {
        eng.abort("control channel: bad magic");
        throw std::runtime_error("control channel: bad magic from the peer");
    }
    if (in.version != local.version || in.active != local.active || in.parked != local.parked) {
        eng.abort("control channel: protocol / task checkpoint mismatch");
        throw std::runtime_error("control channel: protocol / task checkpoint mismatch");
    }
    if (local.command != 0 && in.command == 0 &&
        in.ack_round != (out.round == 0 ? ~std::uint64_t{0} : out.round - 1)) {
        eng.abort("control channel: preceding round not acknowledged");
        throw std::runtime_error("control channel: preceding round not acknowledged");
    }
    return in;
} catch (...) {
    eng.abort("control channel: protocol failure");
    throw;
}
}  // namespace strix::minimax::serve
