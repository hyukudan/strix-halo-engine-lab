#pragma once
// ONE resolver of the DeepSeek V4.1 FAST prefill levers, read by the device prefill (server and CLI alike) and by the
// prefix-cache identity, so both always agree.
//   STRIX_DS41_FAST_PREFILL=1 (default off) = profile ds41-fast-pf-v1:
//     attention  STRIX_DS41_FAST_ATTN=2   ds41-attn-bf16flash-v1 (BF16 one-pass sparse attention, shared flash core)
//     MoE        STRIX_DS41_FAST_MOE=1    ds41-moe-relaxed-v1
//     dense      STRIX_DS41_FAST_DENSE=1  ds41-dense-f16-v1 (shared policy dense-q8a8-f16-v1)
// A lever variable that is set (any value, "0" = off) wins over the profile. Changes logits; see docs/toggles.md.
#include <cstdlib>

namespace strix::ds41::fast_profile {
inline const char* lever(const char* name, const char* profile_value) {
    if (const char* e = std::getenv(name); e != nullptr && *e != 0) return e;
    const char* p = std::getenv("STRIX_DS41_FAST_PREFILL");
    return p != nullptr && p[0] == '1' && p[1] == 0 ? profile_value : "";
}
inline const char* attention() { return lever("STRIX_DS41_FAST_ATTN", "2"); }  // "1" f16wmma-v1, "2" bf16flash-v1
inline bool attention_on() { const char* v = attention(); return (v[0] == '1' || v[0] == '2') && v[1] == 0; }
inline bool attention_flash() { const char* v = attention(); return v[0] == '2' && v[1] == 0; }
inline bool moe_on() { const char* v = lever("STRIX_DS41_FAST_MOE", "1"); return v[0] == '1' && v[1] == 0; }
inline bool dense_on() { const char* v = lever("STRIX_DS41_FAST_DENSE", "1"); return v[0] == '1' && v[1] == 0; }
}  // namespace strix::ds41::fast_profile
