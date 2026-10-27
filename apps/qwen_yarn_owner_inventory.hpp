#pragma once
#include "qwen_effective_config.hpp"
#include "strix/memory_plan.hpp"
#include "strix/q51_known_rows.hpp"
namespace strix::decode_cli {
// Closed plain-CLI owner capacities. This is a conservative payload upper
// bound, not RSS, GTT usage, or allocator metadata (the separate reserve).
inline void append_yarn_owner_requirements(const Options& options, const ModelContract& model,
    const SerialSessionStateLayout& state, std::uint64_t model_bytes, PrefillRequirements& result) {
    const auto add = prefill_checked_add;
    const auto mul = prefill_checked_mul;
    result.owners.clear();
    for (const auto& allocation : state.allocations)
        result.owners[std::array{"target_kv","gdn_matrix","gdn_convolution","ple_convolution","target_raw_index","ngram_history","position_history"}.at(static_cast<unsigned>(allocation.component))] = allocation.bytes;
    std::uint64_t state_payload = 0;
    for (const auto& [name, bytes] : result.owners) { (void)name; state_payload = add(state_payload,bytes); }
    if (state_payload > result.state_d1) throw std::invalid_argument("YaRN state owner payload exceeds layout reservation");
    result.owners["state_alignment_and_d1_shadow"] = result.state_d1 - state_payload;
    result.owners["model_backing"] = result.model_backing;
    result.owners["serial_scratch"] = result.serial_scratch;
    result.owners["prefill_workspace_and_status"] = result.workspace;
    result.owners["tail_workspaces_and_status"] = result.tails;
    result.owners["plan_metadata"] = result.metadata;
    result.owners["runtime_payload"] = result.runtime_payload;
    const auto base_owners = result.owners;
    const auto blocks = add(options.context_tokens,model.indexer_compress_ratio-1U) / model.indexer_compress_ratio;
    // One shared prefill pool, reused by all twelve layers. Window/slab
    // routes and their enabling environment controls are refused.
    const auto pool = mul(mul(blocks,model.indexer_head_dim),sizeof(float));
    result.owners["qsa_shared_prefill_pool"] = pool;
    result.owners["qsa_window_pool_and_slab_bound"] = 0;
    // Group rotary rows, deferred maxima arena (at least 16M floats),
    // per-group maxima and key-range descriptors are launcher-owned globals.
    result.owners["qsa_group_rotary_rows"] = 5U*1024U*64U*8U;
    result.owners["qsa_selection_maxima"] = add(mul(std::max(UINT64_C(16)<<20U,
        mul(1024U,add(blocks,3U)/4U)),sizeof(float)),mul(3U*64U*16384U,sizeof(float)));
    result.owners["rotary_upload_host_staging"] = 256U*256U;
    result.owners["qsa_key_ranges"] = mul(blocks,sizeof(std::uint64_t));
    // PLE prompt hash IDs and issued-page bitmap coexist with the caller's
    // prompt and output IDs. One MiB covers mincore's bounded staging.
    result.owners["ple_prompt_and_token_ids"] = add(mul(options.tokens.size(),
        add(mul(model.ple_head_count(),sizeof(std::uint32_t)),4U*sizeof(std::uint32_t))),
        add(mul(options.generate_tokens,sizeof(std::uint32_t)),add(model_bytes/4096U/8U,UINT64_C(1)<<20U)));
    const auto t = static_cast<std::uint64_t>(options.known_prefix_prefill_rows);
    const auto groups = (t+63U)/64U;
    // Two persistent stage slots plus one replacement maxima allocation
    // coexist during growth; tables have the launcher's fixed 128-group extent.
    result.owners["qsa_batch_tables"] = 2U*128U*64U*64U*8U;
    result.owners["qsa_batch_maxima_growth_peak"] = mul(mul(mul(3U,groups),64U),mul((blocks+3U)/4U,4U));
    result.owners["qsa_batch_host_descriptors_bound"] = 2U*128U*20480U;
    // Each retained HipKnownPrefixWindow constructs an independent PLE owner.
    // share_deferred shares GDN state, never these mapped/device/host buffers.
    // CSR geometry is bounded by the production layout helper before arithmetic
    // or narrowing. These allocations are external to the compiled arenas and
    // to the separate eight-row union allowance in each plan.
    const auto csr_bytes = [&](std::uint64_t rows) {
        if (rows > q51_known_rows::prefill_max_rows || model.num_experts > 512U ||
            model.num_experts_per_token > 10U)
            throw std::invalid_argument("YaRN external routed CSR geometry exceeds production bounds");
        return q51_known_rows::prefill_scratch_bytes(static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(model.num_experts),static_cast<std::uint32_t>(model.num_experts_per_token));
    };
    const auto append_window = [&](const std::string& prefix, std::uint64_t rows) {
        result.owners[prefix + "ple_staging"] = mul(mul(rows,PleHashContract::maximum_heads),1024U);
        result.owners[prefix + "ple_projections_and_gates"] = mul(mul(rows,add(mul(model.hidden_size,add(model.hc_count,1U)),model.hc_count)),4U);
        result.owners[prefix + "ple_snapshot_and_hashes"] = add(ple_conv_state_bytes_per_session(model),mul(rows,sizeof(PleHashResult)));
        result.owners[prefix + "routed_csr"] = csr_bytes(rows);
    };
    append_window("",t);
    for (const auto rows : options.known_prefix_tail_rows)
        append_window("tail_" + std::to_string(rows) + "_",rows);
    // The admitted plain CLI runs one prompt, with fixed SCHED=12 (merge,
    // prune; no balanced blocks, cache, serving or replay). Its leading tail
    // units merge once in the primary owner, retaining at most this one extra
    // row-count CSR in prefill_union_workspace. Use the original ladder, since
    // pruning removes merged tail owners but does not change the schedule.
    const auto& ladder = options.known_prefix_ladder_tail_rows.empty() ?
        options.known_prefix_tail_rows : options.known_prefix_ladder_tail_rows;
    if (!t) throw std::invalid_argument("YaRN primary window has zero rows");
    const auto merged = known_prefill_merged_tail_rows(options.tokens.size()%t,t,ladder);
    result.owners["merged_routed_csr"] = merged ? csr_bytes(merged) : 0U;
    // Every tile8 backend retains all 36 GDN layers independently. Row
    // storage has 256-byte alignment and both banks of eight saved rows.
    const auto key = mul(model.gdn_key_heads,model.gdn_key_head_dim);
    const auto value = mul(model.gdn_value_heads,model.gdn_value_head_dim);
    const auto conv = add(mul(key,2U),value);
    const auto align = [&](std::uint64_t n) { return mul(add(n,255U)/256U,256U); };
    const auto saved_row = align(add(align(mul(conv,2U)),mul(add(add(key,value),mul(model.gdn_value_heads,2U)),4U)));
    result.owners["gdn_retained_rows_and_controls"] = mul(add(options.known_prefix_tail_rows.size(),1U),
        add(mul(mul(saved_row,16U),model.gdn_layers),256U));
    result.owners["gdn_fused_a8"] = 2048U*8192U*9U/8U;
    result.owners["gdn_core_exchange"] = add(mul(model.gdn_value_heads,2U*4U*8U*32U*4U+2U*8U),
        add(16U,mul(8U*3U*2U,key)));
    result.owners["gdn_core16"] = mul(t,6208U*2U);
    // Header-inline dense pools are single program owners across all sites.
    result.owners["dense_a8_slots"] = (64U+256U)*8192U*9U;
    result.owners["dense_f16_slots_and_counter"] = 256U*8192U*2U*2U*4U+1024U*8192U*4U+256U;
    result.owners["dense_tiled_f16"] = mul(t,8256U*2U);
    result.owners["hc_wide_pack_low_and_up"] = 80U*40U*(32U*32U+4U*8U*4U)+320U*8192U*4U+320U*2560U*16U+4U*2560U*320U*4U;
    result.owners["hc_persistent_barrier"] = 256U+8U*4U*16U*4U+40U*24U*8U+16U*8U+8U*(10240U+320U)*4U;
    result.owners["hc_fold_workspaces"] = mul(mul(t,10240U+320U),8U); // folded and qpf2k owners
    result.owners["hc_f16_weights_low_norm_inverse"] = add((320U*10240U+4U*2560U*320U+8192U*320U)*2U+256U,
        8192U*(10304U*2U+4U*4U));
    // Routed-expert images are fixed maximum-T owners even after a T fallback.
    result.owners["moe_f16_input_and_da16"] = UINT64_C(8192)*(2560U+10U*640U)*2U;
    result.owners["moe_counters_and_route"] = ((8192U*10U/64U+1024U)*2U+(8192U*16U+64U)+514U+40U*512U)*4U;
    std::uint64_t extra = 0;
    for (const auto& [name,bytes] : result.owners)
        if (!base_owners.contains(name)) extra = add(extra,bytes);
    result.device_base = add(result.device_base,extra); // conservative GTT upper bound
}
} // namespace strix::decode_cli
