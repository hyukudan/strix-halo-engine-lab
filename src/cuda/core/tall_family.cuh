#pragma once

#include <cstdint>

namespace strix::cuda::core::tall {

enum class Operand { bf16, f16, fp8, fp4 };
enum class Prologue { bf16_planes, f16_permuted, e4m3_row, e2m1_row_tile };
enum class WeightPrep { bf16, q8_bf16, nvfp4_f16, nvfp4_e4m3, nvfp4_tile };
enum class Reduction { warp_chain, q8_group_fold, nvfp4_split, sequential };
enum class Scale { none, tensor, row_tensor };
enum class Store { plain, residual, silu_mul, round_bf16, glu, glu_bf16 };

struct Tile {
    int rows, outputs, k, threads, stages;
};

// Scale the accumulator first, then apply the store operation. Runtime
// choices are still checked by the incumbent launcher's argument contract.
template <Operand Op, Prologue Prep, Reduction Reduce, Scale ScaleBy, Store... Stores>
struct Chain {
    static constexpr Operand operand = Op;
    static constexpr Prologue prologue = Prep;
    static constexpr Reduction reduction = Reduce;
    static constexpr Scale scale = ScaleBy;
    static constexpr bool admits(Store store) { return ((store == Stores) || ...); }
};

template <int Terms, int NT, int BlocksK, int MT, int OutWarps>
struct Dense16Tile {
    static_assert(Terms >= 1 && Terms <= 3 && (BlocksK == 1 || BlocksK == 2));
    static constexpr int Threads = 4 * OutWarps * 32;
    static constexpr int Outputs = MT * OutWarps * 16;
    static constexpr int BN = 4 * NT * 8;
    static constexpr int UK = 32 * BlocksK, RU = 8 * BlocksK;
    static constexpr int A_ELEMS = Outputs * UK, B_ELEMS = Terms * BN * UK;
    static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
    static constexpr int STAGES = (STAGE_ELEMS * 2 * 4 <= 99 * 1024) ? 4 : (STAGE_ELEMS * 2 * 3 <= 99 * 1024) ? 3 : 2;
    static constexpr int SMEM = STAGES * STAGE_ELEMS * 2;
    static constexpr Tile tile{BN, Outputs, UK, Threads, STAGES};
};

// Preserve the warp-partial reduction, term count and K permutation of the
// incumbent. Tile K measures the pipeline stage, while the instruction is K16.
template <bool F16, int Terms, int NT, Reduction Reduce, int BlocksK,
          bool RoundBf16 = false, int MT = 4, int OutWarps = 2>
struct Dense16 : Dense16Tile<Terms, NT, BlocksK, MT, OutWarps>,
                 Chain<F16 ? Operand::f16 : Operand::bf16,
                       F16 ? Prologue::f16_permuted : Prologue::bf16_planes,
                       Reduce, F16 ? Scale::tensor : Scale::none> {
    static_assert(F16 ? (Terms == 1 && Reduce == Reduction::nvfp4_split)
                      : (Reduce == Reduction::warp_chain || Reduce == Reduction::q8_group_fold));
    static_assert(!RoundBf16 || (Terms == 3 && Reduce == Reduction::warp_chain));

    static constexpr int terms = Terms;
    static constexpr WeightPrep weight = F16 ? WeightPrep::nvfp4_f16
        : Reduce == Reduction::q8_group_fold ? WeightPrep::q8_bf16 : WeightPrep::bf16;

    static constexpr bool admits(Store store) {
        if constexpr (RoundBf16) return store == Store::round_bf16;
        if constexpr (OutWarps == 1) return store == Store::plain;
        return store == Store::plain || store == Store::residual || store == Store::silu_mul;
    }
};

template <int NT, int BlocksK, int Stages>
struct Dense8 : Chain<Operand::fp8, Prologue::e4m3_row, Reduction::sequential, Scale::row_tensor,
                      Store::plain, Store::residual, Store::silu_mul> {
    static constexpr WeightPrep weight = WeightPrep::nvfp4_e4m3;
    static constexpr int Threads = 256, Outputs = 128;
    static constexpr int BN = 4 * NT * 8;
    static constexpr int UK = 32 * BlocksK, RU = 8 * BlocksK;
    static constexpr int A_ELEMS = Outputs * UK, B_ELEMS = BN * UK;
    static constexpr int STAGE_ELEMS = A_ELEMS + B_ELEMS;
    static constexpr int SMEM = Stages * STAGE_ELEMS * 2;
    static constexpr Tile tile{BN, Outputs, UK, Threads, Stages};
};

// The feed is a scheduling parameter. The BF16 store exists only on GLU.
template <bool Bulk, bool Bf16Store>
struct Dense4 : Chain<Operand::fp4, Prologue::e2m1_row_tile, Reduction::sequential, Scale::row_tensor> {
    static constexpr WeightPrep weight = WeightPrep::nvfp4_tile;
    static constexpr Tile tile{256, 128, 128, Bulk ? 384 : 256, 3};
    static constexpr int glu_outputs = tile.outputs / 2;

    static constexpr bool admits(Store store) {
        if constexpr (Bf16Store) return store == Store::glu_bf16;
        return store == Store::plain || store == Store::residual || store == Store::silu_mul || store == Store::glu;
    }
};

}  // namespace strix::cuda::core::tall
