#pragma once

#include "strix/gsq_a8_groups.hpp"
#include "strix/model_manifest.hpp"

#include <array>
#include <cstdint>

namespace strix::qpf2k {

enum class Item : unsigned {
    sched, dense, gdn, gdn_gates_batch, gdn_core16, qsa, qsa_mask,
    hc, hc_mask, hc_output3, hc_output3_band, moe_da16, moe_route, ple_reclaim, dense_rows, pipe, attend_batch, score_park,
    norm_group, index_rows, cohort_lean, rows_lean, dispatch_fuse,
};
struct Default {
    Item item;
    const char* name;
    const char* value;
};
inline constexpr std::array defaults{
    Default{Item::sched, "STRIX_QPF2K_SCHED", "12"},
    Default{Item::dense, "STRIX_QPF2K_DENSE", "1"},
    Default{Item::gdn, "STRIX_QPF2K_GDN", "1"},
    Default{Item::gdn_gates_batch, "STRIX_QPF2K_GDN_GATES_BATCH", "1"},
    Default{Item::gdn_core16, "STRIX_QPF2K_GDN_CORE16", "1"},
    Default{Item::qsa, "STRIX_QPF2K_QSA", "1"},
    Default{Item::qsa_mask, "STRIX_QPF2K_QSA_MASK", "0x257"},
    Default{Item::hc, "STRIX_QPF2K_HC", "1"},
    Default{Item::hc_mask, "STRIX_QPF2K_HC_MASK", "14"},
    Default{Item::hc_output3, "STRIX_QPF2K_HC_OUTPUT3", "1"},
    Default{Item::hc_output3_band, "STRIX_QPF2K_HC_OUTPUT3_BAND", "16"},
    Default{Item::moe_da16, "STRIX_QPF2K_MOE_DA16", "1"},
    Default{Item::moe_route, "STRIX_QPF2K_MOE_ROUTE", "14"},
    Default{Item::ple_reclaim, "STRIX_PLE_RECLAIMABLE", "auto"},
    Default{Item::dense_rows, "STRIX_INTPREFILL_DENSE_ROWS", "512"},
    Default{Item::pipe, "STRIX_PREFILL_PIPE", "255"},
    Default{Item::attend_batch, "STRIX_QSA_ATTEND_BATCH", "1"},
    Default{Item::score_park, "STRIX_QSA_SCORE_PARK", "1"},
    Default{Item::norm_group, "STRIX_QSA_NORM_GROUP", "1"},
    Default{Item::index_rows, "STRIX_QSA_INDEX_ROWS", "1"},
    Default{Item::cohort_lean, "STRIX_LM_HEAD_COHORT_LEAN", "1"},
    Default{Item::rows_lean, "STRIX_LM_HEAD_ROWS_LEAN", "2"},
    Default{Item::dispatch_fuse, "STRIX_DISPATCH_FUSE", "1"},
};
static_assert(defaults.size() < 32U);
constexpr unsigned bit(Item item) noexcept { return 1U << static_cast<unsigned>(item); }

// Only injected values are excluded from GSQ execution. Keep parsed environment values
// immutable, and filter AFTER their lazy cache: owners/threads may alternate models.
inline unsigned defaulted_items = 0U;
inline unsigned pre_injected = 0U;
inline bool pre_injected_moe_marker = false;
inline bool baseline = false;
inline thread_local bool gsq_model = false;
inline bool suppressed(Item item, bool gsq = gsq_model) noexcept {
    return baseline || (gsq && (defaulted_items & bit(item)) != 0U);
}
class ModelScope {
public:
    explicit ModelScope(bool gsq) noexcept : previous_(gsq_model) { gsq_model = gsq; }
    ~ModelScope() { gsq_model = previous_; }
    ModelScope(const ModelScope&) = delete;
    ModelScope& operator=(const ModelScope&) = delete;
private:
    bool previous_;
};
inline bool has_gsq_gate(const ModelManifest& manifest) noexcept {
    for (const auto& tensor : manifest.tensors)
        if (tensor.name.ends_with(".ffn_gate_exps.weight") && gsq_a8::gate_type(tensor.ggml_type)) return true;
    return false;
}

// Shared by startup and the host regression test; put returns true only for an
// actually injected value. QSA/HC masks retain the parent-dependent old defaults.
template<class Get, class Put>
unsigned initialize(Get get, Put put, unsigned allowed = ~0U) {
    if (baseline) return 0U;
    if (const char* b = get("STRIX_BASELINE"); b && b[0] == '1' && b[1] == '\0') return 0U;
    if (const char* d = get("STRIX_QPF2K_DEFAULTS"); d && d[0] == '0') return 0U;
    unsigned injected = 0U;
    for (const auto& entry : defaults) {
        if (!(allowed & bit(entry.item))) continue;
        if (entry.item == Item::qsa_mask && !(injected & bit(Item::qsa))) continue;
        if (entry.item == Item::hc_mask && !(injected & bit(Item::hc))) continue;
        if (!get(entry.name) && put(entry.name, entry.value)) injected |= bit(entry.item);
    }
    const unsigned moe = ((injected & bit(Item::moe_da16)) ? 1U : 0U) |
        ((injected & bit(Item::moe_route)) ? 2U : 0U);
    if (moe && !get("STRIX_QPF2K_MOE_DEFAULTED"))
        put("STRIX_QPF2K_MOE_DEFAULTED", moe == 3U ? "3" : moe == 1U ? "1" : "2");
    return injected;
}
// The historical set is restricted to known-input tiled kernels/scheduling:
// SCHED selects tile/tail owners; DENSE batches tiled projections; GDN,
// GATES_BATCH and CORE16 group tiled recurrence/gates; QSA/MASK select tiled
// attention; HC/MASK/OUTPUT3/BAND select tiled hyper-mix; MOE_DA16/ROUTE
// select tiled expert work. Each reader filters injected values per GSQ owner.
inline constexpr unsigned historical_items = bit(Item::moe_route) * 2U - 1U;
// L1 has its own registered-mapping, single-session tiled admission (including
// serving/MTP/cohort exclusions). Other new wrapper items are decoder-only:
// PIPE also affects MTP priming; COHORT_LEAN affects active serving heads;
// DISPATCH_FUSE affects verification windows. Do not inject those at startup.
// DENSE_ROWS, ATTEND_BATCH, SCORE_PARK, NORM_GROUP, INDEX_ROWS and ROWS_LEAN
// are also deferred to decoder admission so non-decoder clients keep old defaults.
inline constexpr unsigned startup_items = historical_items | bit(Item::ple_reclaim);
template<class Get, class Put>
void initialize_before_main(Get get, Put put) {
    const bool had_marker = get("STRIX_QPF2K_MOE_DEFAULTED") != nullptr;
    pre_injected = initialize(get, put, startup_items);
    defaulted_items |= pre_injected;
    pre_injected_moe_marker = !had_marker && get("STRIX_QPF2K_MOE_DEFAULTED");
}
template<class Erase>
void undo_pre_injected(Erase erase) {
    for (const auto& entry : defaults)
        if (pre_injected & bit(entry.item)) erase(entry.name);
    if (pre_injected_moe_marker) erase("STRIX_QPF2K_MOE_DEFAULTED");
    pre_injected = defaulted_items = 0U;
    pre_injected_moe_marker = false;
    baseline = true;
}
} // namespace strix::qpf2k
