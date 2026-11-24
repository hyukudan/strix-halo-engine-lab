// Shared small-row GEMV sweep, lookahead schedules (see gemv_rows.hpp). Textual body: include it inside a
// kernel, like gemv_rows_direct.inc and gemv_rows_staged.inc (the same reason: callable policies compiled to
// different register allocation on gfx1151). One output per wave (T = 1); the activations are read per row
// element from x, so x needs no alignment. The schedule only chooses when the weight chunks are loaded:
//   Lookahead 0  one lane chunk per step (inputs % 16 may be nonzero: the last chunk then reads its in-range
//                weights through GEMV_ROWS_TAIL_WEIGHT and adds an explicit +0 for every padding element)
//   Lookahead 1  rolling: chunk c + 32 is loaded before chunk c is accumulated (inputs % 16 == 0)
//   Lookahead 2  paired: chunks c and c + 32 are loaded, then both accumulated, c += 64 (inputs % 16 == 0)
// In scope at the include:
//   constants  Rows, Streams (1 = one matrix, 2 = two matrices sharing every activation read), Lookahead (0, 1, 2)
//   values     const float* x; unsigned inputs, outputs, first (this wave's output), lane
//   macros     GEMV_ROWS_WEIGHT_ROW(stream, output)      a weight row handle
//              GEMV_ROWS_CHUNK                           the type that decodes 16 weights, with at(e)
//              GEMV_ROWS_LOAD_CHUNK(row, start)          a GEMV_ROWS_CHUNK of that row at column `start`
//              GEMV_ROWS_LANE_SUM(partial)               the family's lane sum
//              GEMV_ROWS_STORE(row, output, sum0, sum1)  the family's epilogue (every lane; it stores once)
//              GEMV_ROWS_TAIL_WEIGHT(stream, output, i)  optional, Lookahead 0: weight i of a row as float; without it
//                                                        the caller guarantees inputs % 16 == 0
// Every (stream, row, lane) accumulator starts at +0 and visits its chunks and their elements in ascending order
// as acc = acc + weight * activation: the bits of the direct and staged schedules for inputs % 16 == 0.
#ifndef STRIX_GEMV_ROWS_UNROLL
#if defined(__clang__)
#define STRIX_GEMV_ROWS_UNROLL _Pragma("unroll")
#else
#define STRIX_GEMV_ROWS_UNROLL
#endif
#endif
{
    static_assert((Streams == 1U || Streams == 2U) && Lookahead <= 2U);
    if (first < outputs) {
        const auto* wp = GEMV_ROWS_WEIGHT_ROW(0U, first);
        const auto* uptr = Streams == 2U ? GEMV_ROWS_WEIGHT_ROW(1U, first) : wp;
        float acc[Rows] = {}, uacc[Rows] = {};
        if constexpr (Lookahead == 0U) {
            for (unsigned c = lane; c < (inputs + 15U) / 16U; c += 32U) {
                const auto start = c * 16U;
                if (start + 16U <= inputs) {
                    const auto a = GEMV_ROWS_LOAD_CHUNK(wp, start);
                    const auto b = GEMV_ROWS_LOAD_CHUNK(uptr, start);
                    STRIX_GEMV_ROWS_UNROLL
                    for (unsigned e = 0; e < 16; ++e) {
                        const float v = a.at(e);
                        float u = 0.0F;
                        if constexpr (Streams == 2U)
                            u = b.at(e);
                        STRIX_GEMV_ROWS_UNROLL
                        for (unsigned r = 0; r < Rows; ++r) {
                            const float input = x[std::size_t(r) * inputs + start + e];
                            acc[r] = acc[r] + v * input;
                            if constexpr (Streams == 2U)
                                uacc[r] = uacc[r] + u * input;
                        }
                    }
                } else {
#ifdef GEMV_ROWS_TAIL_WEIGHT
                    // No chunk load may read beyond the last stored element.
                    for (unsigned e = 0; e < 16; ++e) {
                        const auto i = start + e;
                        const float a = i < inputs ? GEMV_ROWS_TAIL_WEIGHT(0U, first, i) : 0.0F;
                        const float b = Streams == 2U && i < inputs ? GEMV_ROWS_TAIL_WEIGHT(1U, first, i) : 0.0F;
                        for (unsigned r = 0; r < Rows; ++r) {
                            acc[r] = acc[r] + (i < inputs ? a * x[std::size_t(r) * inputs + i] : 0.0F);
                            if constexpr (Streams == 2U)
                                uacc[r] = uacc[r] + (i < inputs ? b * x[std::size_t(r) * inputs + i] : 0.0F);
                        }
                    }
#endif
                }
            }
        } else if constexpr (Lookahead == 1U) {
            const auto chunks = inputs / 16U;
            GEMV_ROWS_CHUNK a, b;
            if (lane < chunks) {
                a = GEMV_ROWS_LOAD_CHUNK(wp, lane * 16U);
                if constexpr (Streams == 2U)
                    b = GEMV_ROWS_LOAD_CHUNK(uptr, lane * 16U);
            }
            for (unsigned c = lane; c < chunks; c += 32U) {
                GEMV_ROWS_CHUNK na, nb;
                if (c + 32U < chunks) {
                    na = GEMV_ROWS_LOAD_CHUNK(wp, (c + 32U) * 16U);
                    if constexpr (Streams == 2U)
                        nb = GEMV_ROWS_LOAD_CHUNK(uptr, (c + 32U) * 16U);
                }
                const auto start = c * 16U;
                STRIX_GEMV_ROWS_UNROLL
                for (unsigned e = 0; e < 16; ++e) {
                    const float v = a.at(e);
                    float u = 0.0F;
                    if constexpr (Streams == 2U)
                        u = b.at(e);
                    STRIX_GEMV_ROWS_UNROLL
                    for (unsigned r = 0; r < Rows; ++r) {
                        const float input = x[std::size_t(r) * inputs + start + e];
                        acc[r] = acc[r] + v * input;
                        if constexpr (Streams == 2U)
                            uacc[r] = uacc[r] + u * input;
                    }
                }
                a = na;
                if constexpr (Streams == 2U)
                    b = nb;
            }
        } else {
            const auto chunks = inputs / 16U;
            for (unsigned c = lane; c < chunks; c += 64U) {
                GEMV_ROWS_CHUNK a[2], b[2];
                STRIX_GEMV_ROWS_UNROLL
                for (unsigned t = 0; t < 2; ++t)
                    if (c + 32U * t < chunks) {
                        a[t] = GEMV_ROWS_LOAD_CHUNK(wp, (c + 32U * t) * 16U);
                        if constexpr (Streams == 2U)
                            b[t] = GEMV_ROWS_LOAD_CHUNK(uptr, (c + 32U * t) * 16U);
                    }
                STRIX_GEMV_ROWS_UNROLL
                for (unsigned t = 0; t < 2; ++t)
                    if (c + 32U * t < chunks) {
                        const auto start = (c + 32U * t) * 16U;
                        STRIX_GEMV_ROWS_UNROLL
                        for (unsigned e = 0; e < 16; ++e) {
                            const float v = a[t].at(e);
                            float u = 0.0F;
                            if constexpr (Streams == 2U)
                                u = b[t].at(e);
                            STRIX_GEMV_ROWS_UNROLL
                            for (unsigned r = 0; r < Rows; ++r) {
                                const float input = x[std::size_t(r) * inputs + start + e];
                                acc[r] = acc[r] + v * input;
                                if constexpr (Streams == 2U)
                                    uacc[r] = uacc[r] + u * input;
                            }
                        }
                    }
            }
        }
        // The epilogue loop: unrolled for the lookahead schedules; schedule 0 leaves the unrolling to the compiler (a
        // large epilogue such as a fused activation with checks, fully unrolled, doubles the kernel at 7 and 8 rows).
        if constexpr (Lookahead == 0U) {
            for (unsigned r = 0; r < Rows; ++r) {
                const float sa = GEMV_ROWS_LANE_SUM(acc[r]);
                float sb = 0.0F;
                if constexpr (Streams == 2U)
                    sb = GEMV_ROWS_LANE_SUM(uacc[r]);
                GEMV_ROWS_STORE(r, first, sa, sb);
            }
        } else {
            STRIX_GEMV_ROWS_UNROLL
            for (unsigned r = 0; r < Rows; ++r) {
                const float sa = GEMV_ROWS_LANE_SUM(acc[r]);
                float sb = 0.0F;
                if constexpr (Streams == 2U)
                    sb = GEMV_ROWS_LANE_SUM(uacc[r]);
                GEMV_ROWS_STORE(r, first, sa, sb);
            }
        }
    }
}
