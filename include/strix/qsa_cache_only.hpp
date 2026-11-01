#pragma once

#include "strix/qpf2k_defaults.hpp"

namespace strix::qsa_cache_only {

// STRIX_QSA_CACHE_ONLY (default on; =0, the baseline, STRIX_QPF2K_DEFAULTS=0 or a GSQ model keep the full step): the known
// prefill runtime marks the last layer's indexed-attention step of a headless block that stops after it (STRIX_PREFILL_PIPE
// bit 64, no step observer). Nothing reads that step's rows, so the QSA prefill launcher writes only the layer's persistent
// state: the input projections (unchanged: their launch routes depend on the whole matrix set), the k norm and
// rotation, and the raw index-key / KV append.
inline bool enabled() noexcept { return qpf2k::long_context_item_enabled("STRIX_QSA_CACHE_ONLY"); }

// Set on the launching thread for the duration of the marked step's launch only.
inline thread_local bool active = false;

class Scope {
public:
    explicit Scope(bool on) noexcept : previous_(active) { active = on; }
    ~Scope() { active = previous_; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    bool previous_;
};

}  // namespace strix::qsa_cache_only
