#pragma once

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

// packed_decode picks the expert body of one-row appends (a one-row decode, a verify window, and a one-row prefill
// chunk: a 1-token prompt or the 1-row remainder of chunk + 1 rows), as the packed CLI always did. The server decodes
// grouped (packed_decode false), so its prefill rows take the body of the CLI under --grouped-decode at every length.
constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || verify),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

}  // namespace strix::inkling2
