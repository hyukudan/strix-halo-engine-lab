#pragma once
#include "strix/qpf2k_defaults.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace strix {
// Compact tiled-prefill memory items follow the qpf2k defaults policy: off under --baseline / STRIX_BASELINE=1,
// STRIX_QPF2K_DEFAULTS=0 and for GSQ checkpoints (qpf2k::gsq_model of the compiling thread), then STRIX_CTX_EXACT_MEMORY=0
// (all items) or the item's own variable =0.
template<class Get> bool ctx_exact_memory_enabled(Get get, const char* item) {
    if (qpf2k::baseline || qpf2k::gsq_model) return false;
    if (const char* b=get("STRIX_BASELINE"); b && b[0]=='1' && b[1]=='\0') return false;
    if (const char* d=get("STRIX_QPF2K_DEFAULTS"); d && d[0]=='0') return false;
    const auto on = [&](const char* name) { const char* v=get(name);return !v || v[0]!='0'; };
    return on("STRIX_CTX_EXACT_MEMORY") && on(item);
}
inline bool ctx_exact_memory_enabled(const char* item) {
    return ctx_exact_memory_enabled([](const char* name){return std::getenv(name);},item);
}
inline bool qsa_score_ring_enabled() {
    if (!ctx_exact_memory_enabled("STRIX_QSA_SCORE_RING")) return false;
    // These diagnostic routes replay or inspect the complete score image.
    for (const char* name : {"STRIX_QSA_PREFILL_SELFCHECK", "STRIX_QSA_PREFILL_TILES", "STRIX_QSA_POOLED_CHECK"})
        if (const char* v=std::getenv(name);v && v[0]!='0') return false;
    if(const char* v=std::getenv("STRIX_QRTX_STEP_DUMP");v && *v) return false;
    if(const char* v=std::getenv("STRIX_PREFILL_FULL_QSA");v && v[0]=='0') return false;
    return true;
}
inline bool qsa_score_ring_rows_valid(std::uint32_t rows) {
    return rows >= 64U && rows <= 1024U && (rows & (rows - 1U)) == 0U;
}
inline std::uint32_t qsa_score_ring_rows_override(const char* value) {
    if (!value) return 0U;
    for (const auto rows : {64U,128U,256U,512U,1024U}) {
        // Accept only the documented decimal spelling, without signs or suffixes.
        const char* text = rows == 64U ? "64" : rows == 128U ? "128" : rows == 256U ? "256" : rows == 512U ? "512" : "1024";
        const char* a=value; const char* b=text;
        while (*a && *a == *b) { ++a; ++b; }
        if (!*a && !*b) return rows;
    }
    throw std::invalid_argument("STRIX_QSA_SCORE_RING_ROWS must be 64, 128, 256, 512 or 1024");
}
struct QsaScoreRingGeometry {
    static constexpr std::uint32_t slots=2;
    static constexpr std::uint64_t default_budget=1ULL << 30U;
    std::uint32_t rows=512U;
    std::uint64_t blocks=0,maxima_stride=0;
    explicit QsaScoreRingGeometry(std::uint64_t maximum, std::uint64_t ratio=4,
        std::uint32_t requested_rows=0) {
        if (!maximum || !ratio || maximum>UINT64_MAX-(ratio-1))
            throw std::invalid_argument("QSA score ring geometry overflow");
        if (requested_rows && !qsa_score_ring_rows_valid(requested_rows))
            throw std::invalid_argument("QSA score ring rows");
        blocks=(maximum+ratio-1)/ratio;
        maxima_stride=blocks/4U+(blocks%4U != 0U);
        // Check the per-row sum before multiplying by rows/slots.
        if (blocks>UINT64_MAX/4U || maxima_stride>UINT64_MAX/4U-blocks)
            throw std::overflow_error("QSA score ring storage overflow");
        const auto row_bytes=(blocks+maxima_stride)*4U;
        if (requested_rows) rows=requested_rows;
        else while (rows>64U && row_bytes>default_budget/(slots*rows)) rows/=2U;
        if (row_bytes>UINT64_MAX/(slots*rows))
            throw std::overflow_error("QSA score ring storage overflow");
    }
    std::uint64_t score_slot_bytes() const {return rows*blocks*4;}
    std::uint64_t maxima_slot_bytes() const {return rows*maxima_stride*4;}
    std::uint64_t total_bytes() const {return slots*(score_slot_bytes()+maxima_slot_bytes());}
};
// The wait callback must wait for the slot's last consumer event, not enqueue return.
// Both native HIP use and the CPU event oracle use this same slot state machine.
class QsaScoreRingLifetime {
    std::array<bool,2> pending_{};
    std::uint32_t next_=0;
public:
    template<class Wait> std::uint32_t acquire(Wait wait) {
        const auto slot=next_;
        if(pending_[slot]) wait(slot);
        pending_[slot]=false;next_=(slot+1)%2;return slot;
    }
    template<class Record> void consumed(std::uint32_t slot,Record record) {
        if(slot>=2 || pending_[slot]) throw std::logic_error("QSA score slot has no active lease");
        record(slot);pending_[slot]=true;
    }
};
} // namespace strix
