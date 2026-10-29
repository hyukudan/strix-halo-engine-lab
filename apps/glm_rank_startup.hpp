#pragma once
// Host-only adapter for the existing two-round decode configuration handshake.
#include "glm_rank_defaults.hpp"
#include "glm/spec/glm_spec_merge.hpp"
#include <array>

namespace strix::glm::rank_defaults {
inline bool has_startup_handshake(unsigned world, const std::string& xch, bool decode, bool reference_gate) {
    return world == 2 && xch == "comm" && decode && !reference_gate;
}
// No new control exchange is allowed in modes without the decode handshake. Both ranks
// independently retain the fixed 4096-row startup before constructing the engine.
inline void fallback_without_handshake(Effective& e, bool comm, bool handshake) {
    if (comm && (max_rows_auto(e) || e.max_rows.source == "auto-4096-ref") && !handshake)
        e.max_rows = {"4096", "auto-4096-no-handshake"};
}
// Shared by inspection and execution, before either environment changes or peer/device work.
inline void fallback_without_handshake(Effective& e, const Request& r, const EnvGet& env) {
    const bool comm = r.world == 2 && r.xch == "comm" && !r.mem_plan && !env_is(env, "STRIX_GLM_LOAD_TEST", "1");
    fallback_without_handshake(e, comm, has_startup_handshake(r.world, r.xch, r.decode, r.reference_gate));
}
inline constexpr std::uint32_t kAutoRowsConfig = 1U << 11;

struct StartupResult { unsigned rows; bool eligible; };

// Fixed mode preserves every payload byte and both merge calls. Auto uses a spare config bit and
// negates only the row-count value: R18 then chooses the minimum proposed rows. The existing OR vote
// rejects a mode/config mismatch on both ranks before either can enter another collective.
template<class Merge>
StartupResult startup_agreement(std::array<std::uint32_t, 8> fields, bool automatic, bool enabled, Merge&& merge) {
    const bool eligible = fields[0] != 0;
    if (!enabled) return {fields[2], eligible};
    if (automatic) fields[1] |= kAutoRowsConfig;
    spec::MergeRow rows[8];
    for (unsigned i = 0; i < 8; ++i) {
        rows[i].value = static_cast<float>(fields[i]);
        rows[i].id = fields[i];
        rows[i].flag = eligible ? 0U : 1U;
    }
    if (automatic) rows[2].value = -rows[2].value;
    spec::MergeCtl ctl;
    ctl.kind = spec::kEligibilityMerge;
    merge(rows, 8, ctl);
    bool mismatch = false;
    const char* names[] = {"eligible", "config", "max_rows", "decode_n", "capacity-lo", "capacity-hi", "slots", "prompt-fnv1a"};
    std::string local;
    for (unsigned i = 0; i < 8; ++i) {
        mismatch = mismatch || (i != 0 && !(automatic && i == 2) && rows[i].id != fields[i]);
        if (i != 0) local += ", ";
        local += std::string(names[i]) + "=" + std::to_string(fields[i]);
    }
    spec::MergeRow vote;
    vote.flag = mismatch ? 1U : 0U;
    spec::MergeCtl vote_ctl;
    vote_ctl.kind = spec::kConfigurationMismatchMerge;
    merge(&vote, 1, vote_ctl);
    if (vote.flag != 0)
        throw std::runtime_error("TP2 ranks resolved different decode configurations or max-rows modes: " + local);
    return {rows[2].id, rows[0].flag == 0};
}
// Used only after both peers have passed the mode vote. A fresh admission can therefore request a
// coordinated fallback without ever pairing this extra round with a fixed peer's first forward.
template<class Merge>
unsigned recheck_rows(unsigned rows, bool enabled, Merge&& merge) {
    if (!enabled) return rows;
    spec::MergeRow proposal;
    proposal.id = rows;
    proposal.value = -static_cast<float>(rows);
    spec::MergeCtl ctl;
    ctl.kind = 0x474c4d52U;
    merge(&proposal, 1, ctl);
    return proposal.id;
}
}  // namespace strix::glm::rank_defaults
