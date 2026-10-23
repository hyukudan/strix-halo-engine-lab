#pragma once

#include <cstdlib>

namespace strix {
// Independent experiments: none inherits STRIX_DISPATCH_FUSE. Default off.
inline bool qpersist_draft_score_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPERSIST_DRAFT_SCORE"); return v && v[0] == '1'; }();
    return on;
}
inline bool qpersist_qsa_tail_enabled() noexcept {
    static const bool on = [] { const char* v = std::getenv("STRIX_QPERSIST_QSA_TAIL"); return v && v[0] == '1'; }();
    return on;
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
