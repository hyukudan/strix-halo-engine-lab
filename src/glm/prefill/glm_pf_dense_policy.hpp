#pragma once
// Dense prefill linears: which core a call of `rows` rows takes, and what a serving prefill chunk does to that choice.
// Host only (no HIP). One definition for the kernel launcher (glm_pf_exact.hip), the route levers of the rank CLI and
// the server (apps/glm_route_levers.hpp) and the host policy test, so they cannot disagree.
//
// Measurement (whole call, four dense shapes, rows per call): the exact X3 core against the TM16 core is 0.99-1.19 at
// 32 rows, 0.52-0.80 at 64, 0.48-0.62 at 256, 0.44-0.58 at 512 and 0.49-0.55 at 2048 rows; the FP16 WMMA core against
// TM16 is 0.54-0.75 at 32 rows, 0.34-0.49 at 256 and 0.33-0.38 at 2048. So auto takes the large-call core from 64 rows
// (exact, same bits as TM16) and from 32 rows when the FAST WMMA identity is selected.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::glm::pf::dense_policy {
inline constexpr unsigned kExactMinRows = 64, kWmmaMinRows = 32;
// The values of exact::Route (glm_pf_exact.hpp), checked there.
inline constexpr unsigned kTm = 0, kX = 1, kX2 = 2, kX3 = 3, kX3r8 = 4;

template<class Env> bool wmma_selected(Env&& env) {
    const char* p = env("STRIX_GLM_PF_WMMA");
    return p != nullptr && std::strcmp(p, "1") == 0;
}
// STRIX_GLM_PF_EXACT_X_MIN_ROWS overrides the auto threshold (rows per linear call); =2048 restores the former one.
template<class Env> unsigned auto_min_rows(Env&& env) {
    const char* p = env("STRIX_GLM_PF_EXACT_X_MIN_ROWS");
    if (!p) return wmma_selected(env) ? kWmmaMinRows : kExactMinRows;
    char* end = nullptr;
    const unsigned long v = std::strtoul(p, &end, 10);
    if (!*p || *end || v == 0 || v > 65536) throw std::invalid_argument("STRIX_GLM_PF_EXACT_X_MIN_ROWS must be 1..65536");
    return static_cast<unsigned>(v);
}
struct Choice {
    unsigned route = kTm;     // an exact::Route value
    bool automatic = false;   // STRIX_GLM_PF_EXACT_X=auto decided by rows
    unsigned min_rows = 0;    // the auto threshold (0 when the route is forced)
};
template<class Env> Choice choose(unsigned rows, Env&& env) {
    const char* p = env("STRIX_GLM_PF_EXACT_X");
    if (!p || std::strcmp(p, "0") == 0) return {kTm, false, 0};
    if (std::strcmp(p, "1") == 0) return {kX, false, 0};
    if (std::strcmp(p, "2") == 0) return {kX2, false, 0};
    if (std::strcmp(p, "3") == 0) return {kX3, false, 0};
    if (std::strcmp(p, "4") == 0) return {kX3r8, false, 0};
    if (std::strcmp(p, "auto") == 0) {
        const unsigned floor = auto_min_rows(env);
        return {rows >= floor ? kX3r8 : kTm, true, floor};
    }
    throw std::invalid_argument("STRIX_GLM_PF_EXACT_X must be 0, 1, 2, 3, 4 or auto");
}

// Rows of one dense linear call when the engine prefills `tokens` rows in one chunk: the TP2 sequence-parallel
// pipeline 2 issues two half-chunk calls (xfer::seqpar_tiles), everything else one call of the whole chunk.
inline unsigned call_rows(unsigned tokens, unsigned world, bool seqpar, unsigned pipeline) {
    return world == 2 && seqpar && pipeline == 2 && tokens >= 4 && tokens % 4 == 0 ? tokens / 2U : tokens;
}

// The serving prefill interleave chunk: tokens of a new prompt per decode step while another request decodes (the
// whole prompt when nothing decodes, and always in the rank CLI). 0 = whole prompt (the off switch).
inline constexpr unsigned kServeChunk = 512, kServeChunkCeiling = 1024;
struct Chunk {
    bool explicit_request = false;
    unsigned requested = 0;   // the value of the flag when explicit
    unsigned tokens = 0;      // effective chunk
    unsigned rows = 0;        // rows per dense call of one chunk (0: whole prompt)
    unsigned route = kTm;     // dense core of those calls
    bool wmma = false;        // FAST WMMA identity selected for those calls (a matrix still needs input % 64 == 0)
    bool declined = false;    // chunked calls do not take the large-call core
    std::string reason;
};
template<class Env> Chunk resolve_chunk(bool explicit_request, unsigned requested, unsigned world, unsigned max_rows, Env&& env) {
    Chunk c;
    c.explicit_request = explicit_request;
    c.requested = requested;
    c.tokens = explicit_request ? requested : kServeChunk;
    if (c.tokens == 0) {
        c.route = choose(max_rows, env).route;
        c.wmma = c.route == kX3r8 && wmma_selected(env);
        c.reason = "chunking off: every prompt is prefilled whole, as in the CLI";
        return c;
    }
    const char* sp = env("STRIX_GLM_PF_SEQPAR");
    const char* pl = env("STRIX_GLM_PF_PIPELINE");
    const bool seqpar = sp == nullptr || std::strcmp(sp, "0") != 0;
    const unsigned pipeline = pl != nullptr && std::strcmp(pl, "2") == 0 ? 2U : 1U;
    const auto rows_of = [&](unsigned tokens) {
        return call_rows(max_rows ? std::min(tokens, max_rows) : tokens, world, seqpar, pipeline);
    };
    c.rows = rows_of(c.tokens);
    Choice pick = choose(c.rows, env);
    if (pick.automatic && pick.route == kTm && !explicit_request) {
        // Default chunk: grow it until its calls reach the auto threshold, within the decode-stall ceiling.
        for (unsigned t = c.tokens * 2U; t <= kServeChunkCeiling && (max_rows == 0 || t <= max_rows); t *= 2U)
            if (choose(rows_of(t), env).route != kTm) {
                c.tokens = t;
                c.rows = rows_of(t);
                pick = choose(c.rows, env);
                break;
            }
    }
    c.route = pick.route;
    const bool fast = wmma_selected(env);
    c.wmma = pick.route == kX3r8 && fast;
    const std::string rows = std::to_string(c.rows) + "-row dense calls";
    if (!pick.automatic) {
        c.declined = pick.route != kX3r8 && fast;
        c.reason = rows + ", route forced by STRIX_GLM_PF_EXACT_X" + (c.declined ? " (FAST WMMA core declined: wrong_route)" : "");
    } else if (pick.route == kTm) {
        c.declined = true;
        c.reason = rows + " below the auto threshold of " + std::to_string(pick.min_rows) + " rows: large-call core" +
                   (fast ? " and FAST WMMA identity" : "") + " declined (short_call), TM16 core; " +
                   (explicit_request ? std::string("raise --prefill-chunk")
                                     : "the default chunk is not grown past " + std::to_string(kServeChunkCeiling) + " tokens") +
                   " or lower STRIX_GLM_PF_EXACT_X_MIN_ROWS";
    } else {
        c.reason = rows + " at or above the auto threshold of " + std::to_string(pick.min_rows) + " rows: " +
                   (c.wmma ? "X3 route with the FAST WMMA core" : "exact X3 core") + ", as whole-prompt calls" +
                   (!explicit_request && c.tokens != kServeChunk ? " (default chunk grown from " + std::to_string(kServeChunk) + ")"
                                                                 : std::string());
    }
    return c;
}
inline std::string chunk_line(const Chunk& c) {
    return "effective serve-prefill-chunk=" + std::to_string(c.tokens) + " (" + (c.explicit_request ? "explicit" : "default") +
           "; serving only: one chunk of a new prompt per decode step while another request decodes; " + c.reason + ")\n";
}
}  // namespace strix::glm::pf::dense_policy
