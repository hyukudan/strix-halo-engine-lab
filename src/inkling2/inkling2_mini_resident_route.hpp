#pragma once

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

// Dense arithmetic class of one dense call: the FAST mma1 class only for a multi-row call of a prompt-prefill
// chunk with the lever on; decode, verify (MTP / speculation) and every one-row call keep the exact mma3 class.
enum class MiniDenseClass { exact_mma3, bf16_mma1 };

constexpr MiniDenseClass mini_dense_class(bool prefill_mma1, MiniAppendKind kind, unsigned rows) {
    return prefill_mma1 && kind == MiniAppendKind::prefill && rows > 1U ? MiniDenseClass::bf16_mma1
                                                                        : MiniDenseClass::exact_mma3;
}

}  // namespace strix::inkling2
