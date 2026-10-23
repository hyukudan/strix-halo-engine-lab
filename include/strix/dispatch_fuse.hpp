#pragma once

#include <cstdlib>

namespace strix {
// STRIX_DISPATCH_FUSE=1: the decode cycle's dependent-dispatch reductions as one switch (default off).
// It turns on, unless the individual switch is set explicitly to 0:
// - STRIX_HC_PERSISTENT, STRIX_HC_PERSISTENT_INJECT: each HC step of a 2- or 4-row verify window (and the inject
//   before it) as one persistent launch instead of four (STRIX_HC_PERSISTENT_ROWS selects the sizes);
// - STRIX_WINDOW_INJECT_NORM: inject folded into the HC normalisation where the persistent fold declines (8 rows);
// - STRIX_WINDOW_COMMIT_COMBINE: MoE combine folded into the routed rows commit.
// Not included (opt-in only, measured neutral): STRIX_MTP_HC_PERSISTENT=1, the draft chain's three HC mixes per
// step as one-row persistent launches.
// Every fused form keeps the expressions and operation order of the separate launches (bit-identical).
inline bool strix_dispatch_fuse_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_DISPATCH_FUSE"); return v && v[0] == '1'; }();
    return on;
}
// An individual switch: "1" on, any other non-empty value off, unset or empty follows STRIX_DISPATCH_FUSE.
inline bool strix_fuse_switch(const char* name) noexcept {
    const char* v = std::getenv(name);
    if (v && *v) return v[0] == '1';
    return strix_dispatch_fuse_enabled();
}
} // namespace strix
