#pragma once
#include "qwen_tiled_prompt_plan.hpp"
#include "strix/known_prompt_chunks.hpp"
namespace strix::decode_cli {
// Versioned route tag: serial receipts cannot authorize MTP or batch producers.
inline constexpr std::uint64_t qwen_serial_certificate_route=0x53455249414c0001ULL;
inline std::vector<std::uint64_t> qwen_serial_certificate(const QwenTiledPlanConfig& c,
    std::span<const KnownPromptChunkUnit> trace,std::uint64_t n,std::uint64_t p) {
    if(p>=n || !known_prompt_full_t_prefix(trace,p,c.rows)) return {};
    std::vector<std::uint64_t> proof{0x514650554c4c5431ULL,c.rows,c.priming,qwen_serial_certificate_route,
        c.schedule,c.tiled_start,c.suppressed,c.force_sync_priming,
        c.decode_execution,c.decode_kernel,c.prefill_execution,c.prefill_kernel,
        c.prefill_profile,c.decode_profile,c.profile_pair,c.valid_state};
    for(const auto& u:trace) {
        if(u.begin>=p) break;
        proof.insert(proof.end(),{u.begin,u.end,u.owner_rows,static_cast<std::uint64_t>(c.block.policy),
            c.block.execution,c.block.kernel,u.t_owner,u.balanced,u.merged,u.deferred,u.headless,u.state_only});
    }
    for(std::size_t f=0;f<c.initial_generations.size();++f) {
        if(c.generations_per_row[f] && p>(UINT64_MAX-c.initial_generations[f])/c.generations_per_row[f]) return {};
        proof.push_back(c.initial_generations[f]+p*c.generations_per_row[f]);
    }
    return proof;
}
} // namespace strix::decode_cli
