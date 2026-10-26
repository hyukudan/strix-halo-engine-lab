#pragma once

#include <cstdlib>

namespace strix {
// Independent experiments: none inherits STRIX_DISPATCH_FUSE. Default off.
inline bool qpersist_draft_score_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPERSIST_DRAFT_SCORE"); return v && v[0] == '1'; }();
    return on;
}
// STRIX_QSA_FAST_INT8=1 (opt-in): the fast prefill QSA kernels read kv-int8-tok-v1 (dequantized, rounded once to FP16).
inline bool qsa_fast_int8_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QSA_FAST_INT8"); return v && v[0] == '1'; }();
    return on;
}
// STRIX_QSA_TAIL_INT8=1 (opt-in): the fused QSA window tail also with kv-int8-tok-v1.
inline bool qsa_tail_int8_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QSA_TAIL_INT8"); return v && v[0] == '1'; }();
    return on;
}
inline bool qpersist_qsa_tail_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPERSIST_QSA_TAIL"); return v && v[0] == '1'; }();
    return on;
}
// STRIX_QPERSIST_DRAFT_ATTN (Q7 arm E, default ON): the MTP draft's score + tail use the exact
// draft-attention kernels whenever their admission holds. Unset or empty = on (default), 1 = on
// (explicit), 0 = incumbent kernels; variants STRIX_QPERSIST_DRAFT_ATTN_SCORE / _TAIL. The value is
// resolved (and rejected if unknown) by the launcher; this returns the raw setting, read once.
inline const char* qpersist_draft_attention_setting() noexcept {
    static const char* const value = std::getenv("STRIX_QPERSIST_DRAFT_ATTN");
    return value;
}
inline bool qpersist_mtp_cache_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPERSIST_MTP_CACHE"); return v && v[0] == '1'; }();
    return on;
}
inline bool qpersist_gdn_tail_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPERSIST_GDN_TAIL"); return v && v[0] == '1'; }();
    return on;
}
} // namespace strix
