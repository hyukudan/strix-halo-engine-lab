#pragma once
#include "strix/model_content_hash.hpp"
#include "strix/serial_session.hpp"
#include <cstdlib>
#include <optional>
#include <span>
#include <vector>

namespace strix {
inline bool prompt_terminal_digest_enabled() {
    static const bool enabled=[] {const char* v=std::getenv("STRIX_PREFIX_CACHE_DIGEST");return v && v[0]=='1';}();
    return enabled;
}
inline std::optional<ModelContentIdentity> raw_fp32_logits_digest(std::span<const float> row) {
    if(row.empty()) return std::nullopt;
    return sha256_bytes(std::as_bytes(row));
}
// Canonical LE values, never structure padding or allocation addresses. The frontier
// and committed count are emitted separately beside this generations/lineage hash.
inline ModelContentIdentity terminal_generations_lineage_digest(const SerialSessionContext& session) {
    std::vector<std::byte> frame;
    const auto u64=[&](std::uint64_t value) {
        for(unsigned shift=0;shift<64;shift+=8) frame.push_back(static_cast<std::byte>((value>>shift)&255U));
    };
    u64(static_cast<std::uint64_t>(session.state.valid));
    for(const auto generation:session.state.generations) u64(generation);
    const auto& lineage=session.arithmetic_lineage;
    u64(lineage.history_identity);u64(lineage.tainted?1U:0U);u64(lineage.intervals.size());
    for(const auto& interval:lineage.intervals) {
        u64(interval.begin);u64(interval.end);u64(static_cast<std::uint64_t>(interval.producer));
        u64(interval.numerical_plan_identity);u64(interval.reduction_plan_identity);u64(interval.state_update_identity);
    }
    return sha256_bytes(frame);
}
} // namespace strix
