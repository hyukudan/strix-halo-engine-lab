#pragma once

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

// packed_decode picks the expert body of DECODE rows (one-row decode, verify windows). A prefill chunk takes the
// prefill body at every size, a 1-row chunk included (a 1-token prompt, the 1-row remainder of chunk + 1 rows): the
// prefill rows of a prompt then do not depend on the decode route, so the server (grouped decode) and the CLI (packed
// or grouped decode) prefill a prompt with the same bits at every length.
constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && kind != MiniAppendKind::prefill && (rows == 1U || verify),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

}  // namespace strix::inkling2
