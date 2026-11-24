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

// Where the prepared-attention round fold (inkling2-mini-prepared-round-fold@1) runs: the HIP prepared attention on the
// qualified target (the fold fixture and the model receipts cover gfx1151 only). The CUDA preparation has no fold, so a
// selection there is declined; `arch` is the gcnArchName up to the first ':' (ignored off HIP).
constexpr bool mini_prepared_round_fold_admitted(bool hip_backend, std::string_view arch) {
    return hip_backend && arch == "gfx1151";
}

}  // namespace strix::inkling2
