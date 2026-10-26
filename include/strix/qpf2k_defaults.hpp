#pragma once

#include "strix/gsq_a8_groups.hpp"
#include "strix/model_manifest.hpp"

#include <array>
#include <cstdint>

namespace strix::qpf2k {

enum class Item : unsigned {
    sched, dense, gdn, gdn_gates_batch, gdn_core16, qsa, qsa_mask,
    hc, hc_mask, hc_output3, hc_output3_band, moe_da16, moe_route,
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
};
constexpr unsigned bit(Item item) noexcept { return 1U << static_cast<unsigned>(item); }

// Only injected values are excluded from GSQ execution. Keep parsed environment values
// immutable, and filter AFTER their lazy cache: owners/threads may alternate models.
inline unsigned defaulted_items = 0U;
inline thread_local bool gsq_model = false;
inline bool suppressed(Item item, bool gsq = gsq_model) noexcept {
    return gsq && (defaulted_items & bit(item)) != 0U;
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
unsigned initialize(Get get, Put put) {
    if (const char* d = get("STRIX_QPF2K_DEFAULTS"); d && d[0] == '0') return 0U;
    unsigned injected = 0U;
    for (const auto& entry : defaults) {
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
} // namespace strix::qpf2k
