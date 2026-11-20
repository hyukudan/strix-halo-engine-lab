#pragma once
// Shared dense FP16 GEMM for prefill row batches on v_wmma_f32_16x16x16_f16: y[r][o] = sum_k fp16(x[r][k]) * w[o][k].
//
// Class: FAST (not the exact FP32 chains). Arithmetic contract: x is rounded to FP16 by the _Float16 cast (RNE, finite
// values beyond the FP16 range become inf, NaN stays NaN); the weights are FP16 as stored; per (row tile, column tile)
// one WMMA chain from +0 over the 16-wide k units in ascending order (FP32 accumulation in hardware order).
//
// Geometry (core<TC>): one CTA = 128 rows x 16 TC columns, 8 waves; wave w owns row tile w and all TC column tiles,
// so a row fragment (the only in-loop VALU work: 16 conversions) feeds TC WMMAs.
//   rows    : A fragment = 16 FP32 values of the lane's row, converted in registers (lane = row); never staged.
//   weights : per stage of four k units, thread i of the CTA copies the 64 FP16 weights of column i into LDS (once per
//             CTA, so a weight byte is read from memory once per 128 rows); every wave reads its B fragments from
//             there (lane = output column). The next stage's weights are requested before the current WMMAs.
// The k loop holds two stages per iteration: the accumulators are loop-carried and a back edge costs register moves.
// Cols policy: tile(i) -> ColumnTile for column tile i of the launch (16 consecutive outputs). Dead tiles (padding of
// the last CTA column) must still name readable weights; they are multiplied and never stored. plan() picks the column
// width of the remainder launch (TC 4, 2 or 1) so few such tiles exist.
// Store policy: where a finished sum goes (default RawStore: y[r][c] = sum). An adapter's policy can apply its epilogue
// (scale, bias, residual, a sum with an earlier pass) in the same store, so no second kernel re-reads the raw sums.
//   open(i, ct, c) -> per-lane state for live column tile i of the launch (c = lane's column inside the tile)
//   put(state, r, v): row r < rows, v = the WMMA chain's sum;  close(): once per thread after its last put.
//   Pair mode (the store has Tile and put_pair(state, r, first, second)): the CTA's TC tiles are TC / 2 output tiles,
//   tile t paired with tile t + TC / 2 (two products of the same outputs, e.g. gate and up); open() is called for
//   tile t only and put_pair gets both chains' sums, so a combining epilogue (act(gate) * up) needs no second pass.
//   run(f): calls f(s) with the object that has open/put/close. A policy with several modes switches on its
//   (wave-uniform) mode here and passes a store whose mode is a compile-time constant: no per-element mode branches.
// Row waves with no live row (the CTA past the last row) keep their staging and barrier duties and skip their row
// loads and WMMAs (skip_dead_waves; a wave with at least one live row is unchanged).
// Limits (callers check): in % 128 == 0, rows * in * 4 < 2^32.
#include "strix/env_flags.hpp"

#include <hip/hip_runtime.h>
#include "../hip/weight/arch/gfx1151_sources.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace strix::core::f16_dense_tall {
// Optional ownership for matrix weights / row flags. Constructor unwinding must
// also drain outstanding conversion or product work before releasing storage.
struct DeviceStorageDeleter {
    void operator()(void* p) const noexcept {
        if (p == nullptr) return;
        (void)hipDeviceSynchronize();
        (void)hipFree(p);
    }
};
template <class T> using DeviceStorage = std::unique_ptr<T, DeviceStorageDeleter>;

using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using Float4 = float __attribute__((ext_vector_type(4)));
using U4 = std::uint32_t __attribute__((ext_vector_type(4)));

inline constexpr unsigned kRows = 128;  // rows per CTA

struct ColumnTile {
    const _Float16* w;  // weight rows of the tile's 16 outputs, [16][in] row-major
    float* y;           // output of row 0 for the tile's first column
    unsigned stride;    // floats between consecutive rows of y
    bool live;          // false: computed, never stored
};

namespace detail {
using GlobalF4 = const Float4 __attribute__((address_space(1), aligned(4)));
using GlobalU4 = const U4 __attribute__((address_space(1), aligned(2)));

__device__ __forceinline__ unsigned uni(unsigned v) {  // wave-uniform value -> scalar register
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}
template <class T> __device__ __forceinline__ T* uniform(T* p) {  // wave-uniform pointer -> scalar registers (saddr loads)
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    const auto lo = uni(static_cast<std::uint32_t>(v)), hi = uni(static_cast<std::uint32_t>(v >> 32U));
    return reinterpret_cast<T*>((static_cast<std::uintptr_t>(hi) << 32U) | lo);
}
// The lane's 16 FP32 x values of one A fragment (loads only) and their FP16 form (the core's RNE cast). Split so the
// core can issue the loads one k unit ahead and convert after the WMMAs in between.
struct RowRaw {
    Float4 f[4];
};
__device__ __forceinline__ RowRaw row_raw(const unsigned char* x, std::uint32_t offset) {
    asm("" : "+v"(offset));
    RowRaw r;
#pragma unroll
    for (unsigned i = 0; i < 4U; ++i) r.f[i] = *(GlobalF4*)(x + offset + 16U * i);
    return r;
}
__device__ __forceinline__ Half16 row_half(const RowRaw& r) {
    Half16 h;
#pragma unroll
    for (unsigned i = 0; i < 16U; ++i) h[i] = static_cast<_Float16>(r.f[i / 4U][i % 4U]);
    return h;
}
__device__ __forceinline__ Half16 row_fragment(const unsigned char* x, std::uint32_t offset) { return row_half(row_raw(x, offset)); }
// Scheduling fence: no instruction moves across it (no code is emitted). The core uses it to keep loads ahead of the
// WMMAs that hide their latency; it changes no operand and no order of arithmetic.
__device__ __forceinline__ void sched_fence() { __builtin_amdgcn_sched_barrier(0); }
// B-fragment ring of the core (ahead + 1 slots): slot t % (ahead + 1) must hold fragment t when WMMA t reads it, and
// no later load may overwrite it before. Replays the core's load/read order (compile-time check per width).
constexpr bool b_ring_ok(unsigned tc, unsigned ahead) {
    int slot[17] = {};
    for (unsigned i = 0; i <= ahead && i < 17U; ++i) slot[i] = -1;
    for (unsigned t = 0; t < ahead; ++t) slot[t % (ahead + 1U)] = static_cast<int>(t);
    for (unsigned t = 0; t < tc; ++t) {
        if (t + ahead < tc) {
            const unsigned w = (t + ahead) % (ahead + 1U);
            if (slot[w] >= static_cast<int>(t)) return false;  // would overwrite a fragment not yet read (or this one)
            slot[w] = static_cast<int>(t + ahead);
        }
        if (slot[t % (ahead + 1U)] != static_cast<int>(t)) return false;
    }
    return true;
}
static_assert(b_ring_ok(1, 1) && b_ring_ok(2, 2) && b_ring_ok(4, 2) && b_ring_ok(16, 2) && b_ring_ok(16, 3));
// B fragment as 8 dwords (two 16-byte LDS loads), bit-cast to 16 halves only as the WMMA operand: a <16 x half> value
// held across the ring makes gfx11 true16 codegen split and re-pack every dword (v_lshrrev_b32 16 + v_mov_b16 .h per
// dword: 16 no-op VALU per WMMA, measured 1,026 of 1,458 VALU per 64 WMMA in the q8a8 loop). Bits identical either way
// (the same fix as tiled_core.hpp / f16_tile_gemm.hpp / hip_moe_prefill_core2.inc).
typedef std::uint32_t Frag __attribute__((ext_vector_type(8), aligned(16)));
__device__ __forceinline__ Frag staged_fragment(const std::uint32_t* p) {
    struct { U4 l, h; } v{*reinterpret_cast<const U4*>(p), *reinterpret_cast<const U4*>(p + 4)};
    return __builtin_bit_cast(Frag, v);
}
}  // namespace detail

// Optional Cols hooks (a policy without them compiles to the code it had):
//   stage(tile, col, unit, nb): the stager's 64 FP16 weights of column `col` of column tile `tile` for the k units
//     [unit, unit + 4) into nb[8], instead of raw FP16 loads from tile(i).w: weights of another stored form are decoded
//     once per (CTA, stage) and every row wave reads the decoded FP16 from LDS.
//   fragment(row, unit) -> Half16: the lane's A fragment of `row` at k unit `unit`, instead of 16 FP32 loads of x.
//   fragment_load(row, unit) -> raw, fragment_half(raw) -> Half16 (optional, with fragment() = their composition):
//     fragment() split in its loads and its arithmetic, like the core's own FP32 x path: the loads of unit u + 1 are
//     issued before the WMMAs of unit u and decoded after them (without the split the decode waits for its loads).
//   kWaves: row waves per CTA (default 8 = 128 rows per weight read); launch with 32 * kWaves threads and
//     grid.x = ceil(rows / (16 * kWaves)). Stagers stay the first 16 * TC threads.
//   kColumnFast = true: grid.x = column block, grid.y = row block (launch dim3(column blocks, row blocks)), so the
//     CTAs of one row block are dispatched together and share its x through L2: x is read from memory about once,
//     not once per column block. Same bits. For launches with several column blocks whose x exceeds the caches
//     (measured on the MiniMax index projection, 6 column blocks, 4096 rows: x re-streamed per block at rows-fast).
template <class C> concept ColsStage = requires(const C& c, U4* nb) { c.stage(0U, 0U, 0U, nb); };
template <class C> concept ColsFragment = requires(const C& c) { { c.fragment(0U, 0U) } -> std::same_as<Half16>; };
template <class C> concept ColsFragmentSplit =
    ColsFragment<C> && requires(const C& c) { { c.fragment_half(c.fragment_load(0U, 0U)) } -> std::same_as<Half16>; };
//   stage_load(tile, col, unit) -> raw, stage_decode(raw, nb) (optional, both or neither): stage() split in its
//     global loads and its arithmetic. The core issues the loads after the stage's second barrier and decodes after
//     the stage's WMMAs, so the load latency hides behind them (stage() must equal stage_decode(stage_load())).
template <class C> concept ColsStageSplit = requires(const C& c, U4* nb) { c.stage_decode(c.stage_load(0U, 0U, 0U), nb); };
template <class C> struct StageRaw { using type = int; };
template <ColsStageSplit C> struct StageRaw<C> { using type = decltype(std::declval<const C&>().stage_load(0U, 0U, 0U)); };
template <class C> inline constexpr unsigned kCtaWaves = 8U;
template <class C> requires requires { C::kWaves; } inline constexpr unsigned kCtaWaves<C> = C::kWaves;
template <class C> inline constexpr bool kColumnFast = false;
template <class C> requires requires { C::kColumnFast; } inline constexpr bool kColumnFast<C> = C::kColumnFast;

// A store mode with the pair epilogue (see the Store policy notes at the top).
template <class S> concept PairStoreMode = requires(S& s, const typename S::Tile& t) { s.put_pair(t, 0U, 0.0F, 0.0F); };

// Pitch 36 dwords per column: b128 accesses stay aligned and columns rotate banks. 2,304 bytes per column tile.
template <unsigned TC> struct alignas(16) Storage {
    std::uint32_t w[16U * TC][36];
};
static_assert(sizeof(Storage<16>) == 36864 && sizeof(Storage<4>) == 9216 && sizeof(Storage<1>) == 2304);

struct RawStore {
    struct Tile {
        float* y;
        unsigned stride, c;
    };
    __device__ __forceinline__ Tile open(unsigned, const ColumnTile& ct, unsigned c) const {
        return {detail::uniform(ct.y), detail::uni(ct.stride), c};
    }
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float v) const { t.y[static_cast<std::size_t>(r) * t.stride + t.c] = v; }
    __device__ __forceinline__ void close() const {}
    template <class F> __device__ __forceinline__ void run(F&& f) { f(*this); }
};

// Host side: launches that cover `tiles` column tiles of `rows` rows. 16-tile CTA columns first; the remainder
// (rest < 16 tiles) in one launch of `tc`-tile columns, `gy` of them. small_tail = false is the previous rule (always
// 4-tile columns). The rule is the measured one (in = 1024): a 1- or 2-tile remainder runs fastest in 1-tile columns
// at every row count; up to 256 rows the launch is latency-bound and 2-tile columns win for every larger remainder;
// with more rows 2-tile columns win up to 5 tiles and 4-tile columns from 6.
struct Plan {
    unsigned full = 0;        // grid.y of the core<16> launch (0: none), tile0 = 0
    unsigned tc = 0, gy = 0;  // remainder launch core<tc>, grid.y = gy, tile0 = 16 * full (tc = 0: none)
};
inline Plan plan(unsigned tiles, unsigned rows, bool small_tail) {
    Plan p;
    p.full = tiles / 16U;
    const unsigned rest = tiles % 16U;
    if (!rest) return p;
    p.tc = 4U;
    p.gy = (rest + 3U) / 4U;
    if (!small_tail) return p;
    p.tc = rest <= 2U ? 1U : (rows <= 2U * kRows || rest <= 5U) ? 2U : 4U;
    p.gy = (rest + p.tc - 1U) / p.tc;
    return p;
}
// STRIX_F16_DENSE_TAIL (default on; 0 = previous behaviour): remainder launch sized by plan(), idle row waves skip
// their WMMAs. Results are bit-identical either way.
inline bool tail_policy() {
    static const bool on = envcfg::flag("STRIX_F16_DENSE_TAIL", true);
    return on;
}

template <unsigned TC, class Cols, class Store = RawStore>
__device__ __forceinline__ void core(const float* x, unsigned in, unsigned nr, Cols cols, Storage<TC>& lds, Store store = {},
                                     bool skip_dead_waves = true) {
    // No 8-tile width: its WMMAs sink below the fragment reads of the whole stage and the kernel spills (124+ VGPRs).
    static_assert(TC == 1 || TC == 2 || TC == 4 || TC == 16);
    const unsigned tid = threadIdx.x, lane = tid & 31U, wave = tid >> 5U, c = lane & 15U;
    const unsigned rblock = kColumnFast<Cols> ? blockIdx.y : blockIdx.x, cblock = kColumnFast<Cols> ? blockIdx.x : blockIdx.y;
    const unsigned row0 = rblock * (16U * kCtaWaves<Cols>) + wave * 16U;
    const unsigned units = detail::uni(in) / 16U;
    const auto* xb = reinterpret_cast<const unsigned char*>(detail::uniform(x));
    // Rows past nr read row 0 (their sums are never stored).
    const std::uint32_t xr = row0 + c < nr ? row0 + c : 0U, xo = xr * in * 4U;
    // A fragment in two steps: frag_load issues the loads (the policy's split load, its fragment, or the 16 FP32 x
    // values), frag_half gives the WMMA operand (the policy's decode, identity, or the core's RNE cast). The loads of
    // unit u + 1 are issued before the WMMAs of unit u and converted after them.
    const auto frag_load = [&](unsigned unit) __attribute__((always_inline)) {
        if constexpr (ColsFragmentSplit<Cols>) return cols.fragment_load(xr, unit);
        else if constexpr (ColsFragment<Cols>) return cols.fragment(xr, unit);
        else return detail::row_raw(xb, xo + unit * 64U);
    };
    const auto frag_half = [&](const auto& raw) __attribute__((always_inline)) {
        if constexpr (ColsFragmentSplit<Cols>) return cols.fragment_half(raw);
        else if constexpr (ColsFragment<Cols>) return raw;
        else return detail::row_half(raw);
    };
    // Stager role: thread tid owns column tid of the CTA's column block.
    const bool stager = tid < 16U * TC;
    const unsigned stile = cblock * TC + (stager ? tid / 16U : 0U);
    const auto* wp = ColsStage<Cols> ? nullptr : reinterpret_cast<const unsigned char*>(cols.tile(stile).w) +
                                                     static_cast<std::size_t>(tid & 15U) * in * 2U;
    U4 nb[8];
    const auto fetch = [&](unsigned unit) __attribute__((always_inline)) {
        if constexpr (ColsStage<Cols>) {
            cols.stage(stile, tid & 15U, unit, nb);
        } else {
            // Every existing raw-F16 Columns policy receives this default source.
            ::strix::weight::hip::RawF16Source::stage(wp,unit,nb);
        }
    };
    // A wave with no live row keeps its staging and barrier duties and skips its row loads and WMMAs.
    const bool idle = detail::uni(skip_dead_waves && row0 >= nr ? 1U : 0U) != 0U;
    Float8 acc[TC] = {};
    auto ar = frag_load(0U);
    if (stager) {
        if constexpr (ColsStageSplit<Cols>) cols.stage_decode(cols.stage_load(stile, tid & 15U, 0U), nb);
        else fetch(0U);
    }
    // Schedule (no operand or order of arithmetic changes; measured, wave4 hip_dense2): every load is issued ahead of
    // the WMMAs that hide its latency and waited for after them. The fences keep the compiler from sinking the waits
    // back next to the loads (it did: each B fragment, each A fragment and the stagers' weights were waited for
    // immediately, 52 % of the wave time in waits). B fragments run two WMMAs ahead.
    constexpr unsigned kAhead = TC < 2U ? TC : 2U;
    for (unsigned base = 0; base < units; base += 8U) {
#pragma unroll
        for (unsigned s = 0; s < 2U; ++s) {
            const unsigned u0 = base + 4U * s;
            const bool more = s == 0U || u0 + 4U < units;
            __syncthreads();  // every wave finished reading the previous stage
            if (stager) {
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) *reinterpret_cast<U4*>(&lds.w[tid][4U * i]) = nb[i];
            }
            __syncthreads();
            [[maybe_unused]] typename StageRaw<Cols>::type sraw{};
            if (stager && more) {
                if constexpr (ColsStageSplit<Cols>) sraw = cols.stage_load(stile, tid & 15U, u0 + 4U);
                else fetch(u0 + 4U);
            }
            asm volatile("" ::: "memory");  // the requests stay ahead of this stage's WMMAs
            detail::sched_fence();
            if (!idle) {
#pragma unroll
                for (unsigned kk = 0; kk < 4U; ++kk) {
                    const Half16 a = frag_half(ar);
                    if (kk + 1U < 4U || more) ar = frag_load(u0 + kk + 1U);
                    detail::sched_fence();
                    detail::Frag b[kAhead + 1U];  // 8 dwords each; Half16 only at the WMMA (see staged_fragment)
#pragma unroll
                    for (unsigned t = 0; t < kAhead; ++t) b[t] = detail::staged_fragment(&lds.w[16U * t + c][kk * 8U]);
                    detail::sched_fence();
#pragma unroll
                    for (unsigned t = 0; t < TC; ++t) {
                        if (t + kAhead < TC) b[(t + kAhead) % (kAhead + 1U)] = detail::staged_fragment(&lds.w[16U * (t + kAhead) + c][kk * 8U]);
                        acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, __builtin_bit_cast(Half16, b[t % (kAhead + 1U)]), acc[t]);
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
                        if constexpr (requires { cols.capture_partial(acc[t]); }) cols.capture_partial(acc[t]);
#endif
                        detail::sched_fence();
                    }
                }
            }
            if constexpr (ColsStageSplit<Cols>) {
                if (stager && more) cols.stage_decode(sraw, nb);
            }
        }
    }
    // acc[t][j] -> row row0 + 2 j + lane / 16, column 16 t + lane % 16. The policy's run() instantiates this loop once
    // per store mode it has, so a mode is chosen once per wave and never per element.
    store.run([&](auto& s) __attribute__((always_inline)) {
        // Pair store (the mode has put_pair): tile t and tile t + TC / 2 hold two products of the same outputs (for
        // example gate and up); open() sees tile t, put_pair gets both sums of the same lane slot. Same WMMA chains.
        if constexpr (PairStoreMode<std::remove_cvref_t<decltype(s)>>) {
            static_assert(TC % 2U == 0U, "a pair store needs an even tile count");
#pragma unroll
            for (unsigned t = 0; t < TC / 2U; ++t) {
                const ColumnTile ct = cols.tile(cblock * TC + t);
                if (!detail::uni(ct.live ? 1U : 0U)) continue;
                const auto st = s.open(cblock * TC + t, ct, c);
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) {
                    const unsigned r = row0 + 2U * j + (lane >> 4U);
                    if (r < nr) s.put_pair(st, r, acc[t][j], acc[t + TC / 2U][j]);
                }
            }
        } else {
#pragma unroll
            for (unsigned t = 0; t < TC; ++t) {
                const ColumnTile ct = cols.tile(cblock * TC + t);
                if (!detail::uni(ct.live ? 1U : 0U)) continue;
                const auto st = s.open(cblock * TC + t, ct, c);
#pragma unroll
                for (unsigned j = 0; j < 8U; ++j) {
                    const unsigned r = row0 + 2U * j + (lane >> 4U);
                    if (r < nr) s.put(st, r, acc[t][j]);
                }
            }
        }
        s.close();
    });
}

// x16 tile form (shared; same bits as the FP32 x path): x rounded to FP16 ONCE by the core's own cast (RNE) into
// k-tile-major tiles, element (g, r, j) at (g * rows + r) * 64 + j for input k = 64 g + j. A CTA stage then reads 128
// rows x 128 contiguous bytes instead of 128 FP32 rows `in` floats apart, and the core's per-fragment conversions go
// away. Measured (MiniMax index projection, 4096 x 6144 -> 384): rows-major FP32 x streamed at 93-114 GB/s with
// 16-32 CTAs, the tiles at ~214 GB/s; WMMA alone 0.54 ms = ~73 % of the WMMA peak. Pays from ~2048 rows (below, x fits
// the MALL and the direct form wins). Workspace: x16_bytes(rows, in). The launch is column-fast (kColumnFast), so the
// column blocks of one row block share its tiles through L2.
inline constexpr unsigned kX16Rows = 2048;  // measured crossover (MiniMax index projection ABAB: 1024 direct, 2048 x16)
inline std::size_t x16_bytes(unsigned rows, unsigned in) { return static_cast<std::size_t>(rows) * in * 2U; }
// Tile quantiser of the producers (Q::apply on the lane's 8 FP32 values before the FP16 cast). TileRne = the core's
// cast alone (the x16 contract above); a4_tiles::Emu<G, Rotate> = the A4 groups of the W4A4 class (gate emulation).
struct TileRne {
    template <class V8> __device__ __forceinline__ static void apply(V8&) {}
};
// One thread per 8-input chunk (in % 64 == 0; a quantiser with lane groups needs in % 128 == 0).
template <class Q = TileRne> __global__ __launch_bounds__(256) void x16_tiles_kernel(const float* __restrict__ x, _Float16* __restrict__ xt,
                                                                                    unsigned rows, unsigned in) {
    const unsigned chunks = in / 8U;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * chunks) return;
    const unsigned r = static_cast<unsigned>(i / chunks), c = static_cast<unsigned>(i % chunks);
    const float4 a = *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + c * 8U);
    const float4 b = *reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + c * 8U + 4U);
    using H8 = _Float16 __attribute__((ext_vector_type(8)));
    using F8 = float __attribute__((ext_vector_type(8)));
    F8 v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    Q::apply(v);
    const H8 h = {static_cast<_Float16>(v[0]), static_cast<_Float16>(v[1]), static_cast<_Float16>(v[2]), static_cast<_Float16>(v[3]),
                  static_cast<_Float16>(v[4]), static_cast<_Float16>(v[5]), static_cast<_Float16>(v[6]), static_cast<_Float16>(v[7])};
    *reinterpret_cast<H8*>(xt + (static_cast<std::size_t>(c / 8U) * rows + r) * 64U + (c % 8U) * 8U) = h;
}
template <class Q = TileRne> inline void enqueue_x16_tiles(const float* x, _Float16* xt, unsigned rows, unsigned in, hipStream_t s) {
    const std::size_t n = static_cast<std::size_t>(rows) * (in / 8U);
    if (n) hipLaunchKernelGGL(x16_tiles_kernel<Q>, dim3(static_cast<unsigned>((n + 255U) / 256U)), dim3(256), 0, s, x, xt, rows, in);
}
// x16 tiles of an elementwise producer: element (r, k) = fp16_rne(F::apply(a[r][k], b[r][k])) (e.g. a gated linear unit's
// SiLU(g) * u, F supplied by the engine). Same bits as storing the FP32 values and running x16_tiles_kernel: the FP32
// result is pinned in a register (asm) so the compiler cannot fold its last multiply into the FP16 conversion
// (v_fma_mix: one rounding instead of two). Saves the FP32 store and the tile pass's read of it.
template <class F, class Q = TileRne> __global__ __launch_bounds__(256) void x16_tiles_map2_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                                                              _Float16* __restrict__ xt, unsigned rows, unsigned in) {
    const unsigned chunks = in / 8U;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * chunks) return;
    const unsigned r = static_cast<unsigned>(i / chunks), c = static_cast<unsigned>(i % chunks);
    const std::size_t at = static_cast<std::size_t>(r) * in + c * 8U;
    using F8 = float __attribute__((ext_vector_type(8)));
    using H8 = _Float16 __attribute__((ext_vector_type(8)));
    const F8 va = *reinterpret_cast<const F8*>(a + at), vb = *reinterpret_cast<const F8*>(b + at);  // two b128 loads each
    F8 v;
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) {
        float t = F::apply(va[j], vb[j]);
        asm volatile("" : "+v"(t));
        v[j] = t;
    }
    Q::apply(v);
    H8 h;
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) h[j] = static_cast<_Float16>(v[j]);
    *reinterpret_cast<H8*>(xt + (static_cast<std::size_t>(c / 8U) * rows + r) * 64U + (c % 8U) * 8U) = h;
}
template <class F, class Q = TileRne> inline void enqueue_x16_tiles_map2(const float* a, const float* b, _Float16* xt, unsigned rows, unsigned in, hipStream_t s) {
    const std::size_t n = static_cast<std::size_t>(rows) * (in / 8U);
    if (n) x16_tiles_map2_kernel<F, Q><<<dim3(static_cast<unsigned>((n + 255U) / 256U)), dim3(256), 0, s>>>(a, b, xt, rows, in);
}
// Cols mixin: Base's tile / stage hooks, A fragments from the x16 tiles, column-fast CTA order
// (launch dim3(column blocks, row blocks)).
template <class Base> struct X16Cols : Base {
    static constexpr bool kColumnFast = true;
    const _Float16* xt;
    unsigned rows;
    __device__ __forceinline__ Half16 fragment(unsigned row, unsigned unit) const {
        using GH16 = const Half16 __attribute__((address_space(1)));
        const _Float16* p = detail::uniform(xt) + (static_cast<std::size_t>(unit / 4U) * detail::uni(rows) + row) * 64U + (unit % 4U) * 16U;
        return *(GH16*)p;
    }
};
// Column width of a column-fast x16 launch: the widest of 4 / 2 / 1 tiles that divides `tiles` (no dead tiles).
inline unsigned x16_tc(unsigned tiles) { return tiles % 4U == 0 ? 4U : tiles % 2U == 0 ? 2U : 1U; }
// ---------------------------------------------------------------- one matrix, guarded (FAST products that feed a selection)
// Guarded<Store>: the same stores, plus flags[r] = 1 for a row with a non-finite sum. x is rounded to FP16 in the A
// fragments, so a finite |x| > 65504 becomes inf and every sum of that row is inf or NaN (inf * 0 = NaN): the flag is
// complete for overflow. The adapter's fallback kernel then calls exact_rows(), which recomputes the flagged rows with
// the family's exact chain and clears the flags (zeroed once at allocation, left zero by every fallback).
template <class Inner> struct Guarded {
    Inner inner;
    unsigned* flags;  // [rows]
    template <class S> struct View {
        S& s;
        unsigned* flags;
        __device__ __forceinline__ auto open(unsigned i, const ColumnTile& ct, unsigned c) const { return s.open(i, ct, c); }
        template <class T> __device__ __forceinline__ void put(const T& t, unsigned r, float v) const {
            if (!__builtin_isfinite(v)) atomicOr(flags + r, 1U);
            s.put(t, r, v);
        }
        __device__ __forceinline__ void close() const { s.close(); }
    };
    template <class F> __device__ __forceinline__ void run(F&& f) {
        inner.run([&](auto& s) __attribute__((always_inline)) {
            View<std::remove_reference_t<decltype(s)>> v{s, flags};
            f(v);
        });
    }
};

// The fallback walk, run by every thread of a CTA: CTA b covers rows [b * span, min((b + 1) * span, rows)) in steps of
// blockDim.x rows; a step lists its flagged rows in LDS (list[blockDim.x], *count), calls row(r) once per listed row
// with every thread (barrier after each) and clears their flags. Nothing flagged: one flag read per row.
template <class Row>
__device__ __forceinline__ void exact_rows(unsigned* flags, unsigned rows, unsigned span, unsigned* list, unsigned* count, Row&& row) {
    const unsigned first = blockIdx.x * span, last = first + span < rows ? first + span : rows;
    for (unsigned base = first; base < last; base += blockDim.x) {
        if (threadIdx.x == 0U) *count = 0U;
        __syncthreads();
        const unsigned r = base + threadIdx.x;
        const bool mine = r < last && flags[r] != 0U;
        if (mine) list[atomicAdd(count, 1U)] = r;
        __syncthreads();
        const unsigned n = *count;
        for (unsigned i = 0; i < n; ++i) {
            row(list[i]);
            __syncthreads();
        }
        if (mine) flags[r] = 0U;
        __syncthreads();  // every thread read *count before the next step resets it
    }
}

// Columns of one [n][in] FP16 matrix (n % 16 == 0): tile t = outputs [16 t, 16 t + 16); tiles past n / 16 are dead
// (they read tile 0). The store receives y + 16 t and stride (RawStore: y[r * stride + column]).
template <bool ColumnsFastest> struct MatrixCols {
    static constexpr bool kColumnFast = ColumnsFastest;
    const _Float16* w;
    float* y;
    unsigned in, stride, tiles, tile0;
    __device__ __forceinline__ ColumnTile tile(unsigned i) const {
        const unsigned t = tile0 + i;
        if (t < tiles) return {w + static_cast<std::size_t>(t) * 16U * in, y + t * 16U, stride, true};
        return {w, y, stride, false};
    }
};

template <unsigned TC, bool ColumnsFastest, class Store>
__global__ __launch_bounds__(256) void matrix_kernel(const float* x, unsigned rows, MatrixCols<ColumnsFastest> cols, Store store) {
    __shared__ Storage<TC> lds;
    core<TC>(x, cols.in, rows, cols, lds, store, true);
}

// Launch forms of a one-matrix product, all with the same bits (each tile's chain does not depend on the form):
//   plan   : 16-tile CTA columns, the remainder in plan()'s width (one x read per launch: two when n % 256 != 0)
//   narrow4: 4-tile CTA columns, column-fastest grid, one launch (the column blocks of a row block share x in L2)
//   wide16 : 16-tile CTA columns, column-fastest grid, one launch, dead tiles up to a multiple of 16
// STRIX_F16_MATRIX_FORM = auto (adapter's choice) | plan | narrow4 | wide16.
enum class MatrixForm : unsigned { plan = 0, narrow4 = 1, wide16 = 2 };
inline MatrixForm matrix_form(MatrixForm automatic) {
    static const int f = envcfg::choice("STRIX_F16_MATRIX_FORM", {"auto", "plan", "narrow4", "wide16"}, 0);
    return f <= 0 ? automatic : static_cast<MatrixForm>(f - 1);
}
inline const char* matrix_form_name(MatrixForm f) {
    return f == MatrixForm::narrow4 ? "narrow4" : f == MatrixForm::wide16 ? "wide16" : "plan";
}

// y-side contract as core (in % 128 == 0, rows * in * 4 < 2^32, n % 16 == 0); the caller checks.
template <class Store>
inline void launch_matrix(MatrixForm form, const float* x, unsigned rows, const _Float16* w, unsigned in, unsigned n, float* y,
                          unsigned stride, const Store& store, hipStream_t s) {
    const unsigned tiles = n / 16U, gr = (rows + kRows - 1U) / kRows;
    if (form == MatrixForm::narrow4) {
        matrix_kernel<4, true, Store><<<dim3((tiles + 3U) / 4U, gr), 256, 0, s>>>(x, rows, MatrixCols<true>{w, y, in, stride, tiles, 0U}, store);
        return;
    }
    if (form == MatrixForm::wide16) {
        matrix_kernel<16, true, Store><<<dim3((tiles + 15U) / 16U, gr), 256, 0, s>>>(x, rows, MatrixCols<true>{w, y, in, stride, tiles, 0U}, store);
        return;
    }
    const Plan pl = plan(tiles, rows, true);
    if (pl.full) matrix_kernel<16, false, Store><<<dim3(gr, pl.full), 256, 0, s>>>(x, rows, MatrixCols<false>{w, y, in, stride, tiles, 0U}, store);
    const MatrixCols<false> rest{w, y, in, stride, tiles, pl.full * 16U};
    switch (pl.tc) {
    case 4: matrix_kernel<4, false, Store><<<dim3(gr, pl.gy), 256, 0, s>>>(x, rows, rest, store); break;
    case 2: matrix_kernel<2, false, Store><<<dim3(gr, pl.gy), 256, 0, s>>>(x, rows, rest, store); break;
    case 1: matrix_kernel<1, false, Store><<<dim3(gr, pl.gy), 256, 0, s>>>(x, rows, rest, store); break;
    default: break;
    }
}

}  // namespace strix::core::f16_dense_tall
