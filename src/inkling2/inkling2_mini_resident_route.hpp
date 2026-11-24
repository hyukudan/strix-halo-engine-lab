#pragma once

#include <string_view>

namespace strix::inkling2 {

inline constexpr unsigned kMiniMaxVerifyRows = 8U;

enum class MiniAppendKind { prefill, decode, verify };

struct MiniResidentRoute {
    unsigned attention_rows;
    bool packed_moe;
    bool step_graph;
};

constexpr MiniResidentRoute mini_resident_route(MiniAppendKind kind, unsigned rows,
                                                bool packed_decode, bool allow_graph) {
    const bool verify = kind == MiniAppendKind::verify;
    return {
        verify ? 1U : rows,
        packed_decode && (rows == 1U || verify),
        allow_graph && rows == 1U && kind != MiniAppendKind::prefill,
    };
}

// HIP targets the exact down-image lever (inkling2-mini-combine-down-image@1) is qualified on: the NaN identity fixture and
// the model receipts cover gfx1151 only; the resident device policy admits more targets for the rest of the model.
// `arch` is the gcnArchName up to the first ':'.
constexpr bool mini_combine_image_arch(std::string_view arch) { return arch == "gfx1151"; }

}  // namespace strix::inkling2
