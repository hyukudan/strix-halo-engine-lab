#pragma once

#include "inkling2_mini_weight_options.hpp"
#include "strix/serve/cohort_decode.hpp"
#include "strix/serve/row_budget.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace strix::inkling2 {

struct MiniServeOptions {
    std::string checkpoint;
    std::string tokenizer;
    std::string devices_path;
    std::string device_id;
    std::string device_profile;
    unsigned context = 8192;
    unsigned prefill_rows = 128;
    unsigned slots = 0;
    std::uint64_t kv_bytes = 0;
    MiniWeightOptions weight_options;
    profile::Name tier = profile::Name::standard;
    // Resolved operand class of the FAST W4A4 expert lever (mini_moe_w4a4_class; bf16 = off): the served prefill runs
    // the class the profile line reports, as the CLI and the gate do.
    moe_core::grouped_w4a4::Activations moe_w4a4 = moe_core::grouped_w4a4::Activations::bf16;
    // Served identity, read only from the resolution (profile::Resolution::selected): packed = identity variant
    // inkling2-mini-moe-packed (HIP), screened_head = inkling2-mini-cuda-screened-head (CUDA). Both off: the incumbent
    // grouped serving identity of the backend, as on main.
    bool packed_decode = false;
    bool screened_head = false;
    std::string identity = "identity incumbent";  // profile::Resolution::identity(), printed on the route line
};

// Served KV slots, and whether the screened greedy head plane stays resident. The plane is an EXACT decode speed lever
// (same picks as the full head). The HIP plane is built with the decoder (as on main), before the slot budget is read:
// it stays only when it costs no KV slot, so screening never lowers the slot capacity the full head admits.
struct MiniServeSlots {
    std::uint32_t lanes;
    bool keep_screen;
};

inline MiniServeSlots mini_serve_slots(std::uint64_t free, std::uint64_t total, std::uint64_t device_limit,
                                       std::uint64_t kv_limit, std::uint64_t slot_bytes, std::uint32_t rows,
                                       std::uint64_t screen_bytes) {
    const auto with_screen = serve::resident_kv_slots(free, total, device_limit, kv_limit, slot_bytes, rows);
    if (screen_bytes == 0U) return {with_screen, false};
    const auto freed = free + std::min(screen_bytes, total - std::min(free, total));
    const auto full_head = serve::resident_kv_slots(freed, total, device_limit, kv_limit, slot_bytes, rows);
    return full_head > with_screen ? MiniServeSlots{full_head, false} : MiniServeSlots{with_screen, true};
}

// The CUDA plane of the admitted screened-head variant is built only AFTER the KV slots the full-head budget admits are
// allocated (`committed`, slot 0 included), so it can never take a committed slot. It stays when the shared budget,
// re-read with the plane resident and the committed extra slots counted as free, still admits every committed slot, and
// the plane leaves the runtime headroom that budget keeps (one eighth of the bytes it was given). Else the plane is
// released and the incumbent full-head server runs (same tokens).
inline bool mini_keep_screen(std::uint64_t free_before_plane, std::uint64_t free_after_plane, std::uint64_t total,
                             std::uint64_t device_limit, std::uint64_t kv_limit, std::uint64_t slot_bytes,
                             std::uint32_t rows, std::uint32_t committed) {
    if (committed == 0U || free_after_plane >= free_before_plane) return false;  // nothing committed / no plane built
    const auto extra = static_cast<std::uint64_t>(committed - 1U) * slot_bytes;
    const auto budget_free = free_before_plane + extra;  // what the slot budget was given, before the plane
    if (free_after_plane < budget_free / 8U) return false;
    return serve::resident_kv_slots(free_after_plane + extra, total, device_limit, kv_limit, slot_bytes, rows) >= committed;
}

// The order of the served device commitments, one body for both backends (`Decoder` is the resident decoder; the host
// test drives it with a fake). The decoder already holds slot 0. `memory()` returns {free, total}; `commit(lanes)`
// allocates every admitted slot (the cohort) and returns it; `screen_variant`: the resolution admitted the CUDA
// screened-head variant. Returns the cohort; `lanes` receives the slot count.
template <class Decoder, class Memory, class Commit>
auto mini_commit_slots_then_screen(Decoder& decoder, const Memory& memory, std::uint64_t device_limit,
                                   std::uint64_t kv_limit, std::uint64_t slot_bytes, std::uint32_t rows,
                                   bool screen_variant, std::uint32_t& lanes, Commit&& commit) {
    const auto before = memory();
    const auto resident_plane = decoder.screen_device_bytes();  // HIP: built at load; CUDA: 0 here
    const auto plan = mini_serve_slots(before.free, before.total, device_limit, kv_limit, slot_bytes, rows,
                                       resident_plane);
    if (resident_plane != 0U && !plan.keep_screen) decoder.release_screen("the screened plane would cost a KV slot");
    if (!plan.lanes) throw std::runtime_error("Inkling KV budget holds no session slots");
    lanes = plan.lanes;
    auto cohort = commit(plan.lanes);
    if (screen_variant) {
        const auto committed = memory();
        decoder.build_screen();
        const auto after = memory();
        if (decoder.screen_device_bytes() == 0U)
            decoder.release_screen("screened head variant: plane not built");
        else if (!mini_keep_screen(committed.free, after.free, after.total, device_limit, kv_limit, slot_bytes, rows,
                                   plan.lanes))
            decoder.release_screen("screened head variant: the plane would cost a committed KV slot or headroom");
    }
    return cohort;
}

struct MiniServeCohort {
    std::unique_ptr<serve::CohortBackend> backend;
    weight::WeightLoadPlan weights;
};

MiniServeCohort make_mini_hip_cohort(const MiniServeOptions&);
MiniServeCohort make_mini_cuda_cohort(const MiniServeOptions&);

} // namespace strix::inkling2
