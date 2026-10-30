#pragma once
#include "hip_decode_options.hpp"
#include "qwen_tiled_prompt_plan.hpp"
#include "strix/ple_lookup.hpp"
#include "strix/q51_known_rows.hpp"
#include "strix/session_state_layout.hpp"

namespace strix::decode_cli {
// Payloads owned by each native prefill HipKnownPrefixWindow, outside its
// borrowed compiled arena. Primary and tail owners do not share these buffers;
// add_tiled constructs another complete set for each additional slot.
struct PrefillExternalMemory {
    std::uint64_t host = 0, device = 0;
    std::uint64_t csr_fixed = 0, csr_variants = 0;
    std::uint64_t staging = 0, projections = 0, gates = 0, snapshots = 0, hashes = 0, gdn = 0;
};
inline PrefillExternalMemory qwen_prefill_external_memory(const Options& o, const ModelContract& model) {
    const auto add=[](std::uint64_t a,std::uint64_t b) {
        if(b>UINT64_MAX-a) throw std::overflow_error("prefill owner reservation overflow");
        return a+b;
    };
    const auto mul=[](std::uint64_t a,std::uint64_t b) {
        if(b && a>UINT64_MAX/b) throw std::overflow_error("prefill owner reservation overflow");
        return a*b;
    };
    const auto align=[&](std::uint64_t n) { return mul(add(n,255U)/256U,256U); };
    PrefillExternalMemory r;
    if(!o.known_prefix_prefill_rows || !o.known_prefix_window || !o.known_prefix_full_rows) return r;
    const auto key=mul(model.gdn_key_heads,model.gdn_key_head_dim);
    const auto value=mul(model.gdn_value_heads,model.gdn_value_head_dim);
    // gdn_window_saved: aligned BF16 convolution, FP32 k/v/beta/gate;
    // two banks of eight rows and one mapped GdnWindowControl per owner.
    const auto saved=align(add(align(mul(add(mul(2U,key),value),2U)),
        mul(add(add(key,value),mul(2U,model.gdn_value_heads)),4U)));
    const auto owner=[&](std::uint64_t rows) {
        // Conservative full-MoE/A8 bound even if a runtime off switch disables
        // this allocation. Include every retained partial-row CSR variant.
        if(q51_known_rows::prefill_csr_enabled(true,o.known_prefix_expert_union,true,false,
                static_cast<std::uint32_t>(rows))) {
            const auto csr=q51_known_rows::prefill_csr_inventory(static_cast<std::uint32_t>(rows),
                static_cast<std::uint32_t>(model.num_experts),static_cast<std::uint32_t>(model.num_experts_per_token));
            r.csr_fixed=add(r.csr_fixed,csr.fixed); r.csr_variants=add(r.csr_variants,csr.variants);
        }
        r.staging=add(r.staging,mul(mul(rows,PleHashContract::maximum_heads),1024U));
        r.projections=add(r.projections,mul(mul(mul(rows,model.hidden_size),model.hc_count+1U),4U));
        r.gates=add(r.gates,mul(mul(rows,model.hc_count),4U));
        r.snapshots=add(r.snapshots,mul(mul(mul(model.hidden_size,model.hc_count),
            model.ple_conv_kernel_size-1U),mul(model.ngram_size,model.bytes_bf16)));
        r.hashes=add(r.hashes,mul(rows,sizeof(PleHashResult)));
        r.gdn=add(r.gdn,add(mul(mul(saved,16U),model.gdn_layers),8U));
    };
    owner(o.known_prefix_prefill_rows);
    for(const auto rows:o.known_prefix_tail_rows) owner(rows);
    r.device=add(r.staging,add(r.projections,add(r.gates,add(r.snapshots,r.gdn))));
    r.device=add(r.device,add(r.csr_fixed,r.csr_variants));
    r.host=add(r.device,r.hashes); // mapped/device payloads share physical UMA RAM
    return r;
}

// Same target registry geometry as HipPrefixStateCacheSession, before any HIP
// setup. Cache staging is shared by capture, never multiplied by slot count.
inline std::uint64_t qwen_http_batch_staging_reserve(const Options& o,const ModelContract& model,
    const SerialSessionStateLayout& layout) {
    if(!o.http_serve || o.baseline || !ordinary_http_batch_slots(o) ||
        !o.prefix_batch_cert || !o.prefix_cache || !o.prefix_cache_host_bytes ||
        !mtp_tiled_prompt(o) || o.session_management_requested || !o.session_save.empty() ||
        !o.session_load.empty() || o.session_live_control) return 0;
    const auto add=[](std::uint64_t a,std::uint64_t b) {
        if(b>UINT64_MAX-a) throw std::overflow_error("batch staging reservation overflow");
        return a+b;
    };
    std::uint64_t row=0,fixed=0;
    for(const auto& a:layout.allocations) {
        if(a.component==SessionStateComponent::attention_kv || a.component==SessionStateComponent::sparse_index) {
            if(!layout.context_tokens || a.bytes%layout.context_tokens)
                throw std::invalid_argument("cache registry has invalid row geometry");
            row=add(row,a.bytes/layout.context_tokens);
        } else fixed=add(fixed,add(a.bytes,255U)&~UINT64_C(255));
    }
    const auto mul=[](std::uint64_t a,std::uint64_t b) {
        if(b && a>UINT64_MAX/b) throw std::overflow_error("prefill owner reservation overflow");
        return a*b;
    };
    const auto predictor=add(mul(mul(model.kv_heads,model.head_dim),4U),mul(model.indexer_head_dim,2U));
    const auto held=mul(mul(model.hidden_size,model.hc_count),2U);
    return qwen_batch_staging_cap(o.prefix_cache_staging_bytes,row,fixed,predictor,held,o.known_prefix_prefill_rows);
}
} // namespace strix::decode_cli
