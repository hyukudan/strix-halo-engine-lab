#pragma once
// NVFP4 x NVFP4 large-M GEMM operands for prefill (W4A4, FAST class). Home: shared (backend core CUDA, arch specialisation sm_120:
// block-scaled mma.sync kind::mxf4nvf4 m16n8k64, see src/cuda/core/arch/sm120/nvfp4_tile.cuh). Model-agnostic: any engine whose
// dense weights are NVFP4 (e2m1 values, one e4m3 scale per 16 inputs, one FP32 scale per tensor) binds it through
// strix/cuda_core/dense_big.hpp (FAST4 operand class).
//
// Arithmetic identity "nvfp4-tile-mma-f32-v1": products of exact E2M1 x E4M3-scaled operands summed in FP32 inside the tensor core
// (order not specified by the ISA, deterministic for one plan), then y = acc * (gscale * row_scale[r]) with IEEE single rounding.
// Same operand semantics as nvfp4-mma-f32-v1 (strix/cuda/core/nvfp4.hpp); only the staging layout and the tiling differ.
//
// Activation quantisation (one FP32 scale per row, no global scale): rs = amax(row) / 2688 (= 6 * 448, so a block scale never
// saturates); per 16 consecutive inputs s = e4m3_rn_satfinite(min(448, amax16 / 6 / rs)); code = e2m1_rn(clamp(x / (s * rs), +-6))
// (s == 0 -> codes 0). Exactly the per-row form of nvfp4::quantize with tensor_scale = rs. A zero row has rs = 0 and zero codes.
//
// STAGING LAYOUTS (both are byte permutations of the canonical [rows][K/2] codes + [rows][K/16] scales; K % 128 == 0):
//   W tile record (16 weight rows x 128 inputs, 1,152 B), records [row16][K/128]:
//     [kb 0..1][lane 0..31][4 x u32]  register i of lane (g = lane/4, t = lane%4) = codes of row g + 8 (i & 1), inputs
//                                     64 kb + 32 (i >> 1) + 8 t .. + 7 (low nibble = even input): the mma A fragment, one LDS.128.
//     [1024 + kb * 64][row 0..15][u32] four e4m3 scales of the row's 64-input block (byte j = inputs 16 j ..).
//   X tile record (8 activation rows x 128 inputs, 576 B), records [row8][K/128]:
//     [kb][lane][2 x u32]             register i = codes of row g, inputs 64 kb + 32 i + 8 t .. + 7: the mma B fragment (LDS.64).
//     [512 + kb * 32][row 0..7][u32]  four e4m3 scales.
// The fragment and scale-lane maps are those of the GPU-validated operator (arch/sm120/nvfp4.cuh): A scale from row
// g + 8 (lane & 1), B scale from row g.
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define STRIX_NVFP4_TILE_HD __host__ __device__ __forceinline__
#else
#define STRIX_NVFP4_TILE_HD inline
#endif

namespace strix::cuda::core::nvfp4::tile {
inline constexpr const char* kArithmeticId = "nvfp4-tile-mma-f32-v1";
inline constexpr std::uint32_t kWRec = 1152, kXRec = 576, kK = 128;
inline constexpr std::uint32_t kCtaOut = 128, kCtaRows = 256;  // CTA tile: 128 weight rows (64 gate + 64 up channels in GLU) x 256 rows
inline constexpr float kRowScaleDiv = 2688.0f;                   // 6 * 448

STRIX_NVFP4_TILE_HD std::size_t w_tile_bytes(std::uint32_t out, std::uint32_t in) {
    return static_cast<std::size_t>(out / 16) * (in / kK) * kWRec;
}
STRIX_NVFP4_TILE_HD std::uint32_t rows_padded(std::uint32_t rows) { return (rows + kCtaRows - 1) / kCtaRows * kCtaRows; }
// The RMSNorm-fused quantiser (norm_quantize_enqueue) keeps a row in 256 threads x <= 32 registers and in * 4 bytes of shared memory
// (<= 32 KiB, under the 48 KiB static launch limit): rows of width kK * j up to 8192. Wider rows use rmsnorm + quantize_enqueue.
inline constexpr std::uint32_t kNormQuantMaxIn = 8192;
inline constexpr std::uint32_t kGatedNormHead = 128;  // head width of the gated (GDN) normaliser = cuda_core kGatedHead
static_assert(kK % kGatedNormHead == 0, "norm_quantize_fits(in) implies whole gated-norm heads");
STRIX_NVFP4_TILE_HD bool norm_quantize_fits(std::uint32_t in) { return in != 0 && in % kK == 0 && in <= kNormQuantMaxIn; }
STRIX_NVFP4_TILE_HD std::size_t x_tile_bytes(std::uint32_t rows_pad, std::uint32_t in) {
    return static_cast<std::size_t>(rows_pad / 8) * (in / kK) * kXRec;
}

// ---- readers (offsets inside one record; used by the kernel and by the host fragment simulation)
STRIX_NVFP4_TILE_HD std::uint32_t a_frag(std::uint32_t kb, std::uint32_t lane) { return kb * 512 + lane * 16; }
STRIX_NVFP4_TILE_HD std::uint32_t a_scale(std::uint32_t kb, std::uint32_t lane) { return 1024 + kb * 64 + ((lane >> 2) + 8 * (lane & 1)) * 4; }
STRIX_NVFP4_TILE_HD std::uint32_t b_frag(std::uint32_t kb, std::uint32_t lane) { return kb * 256 + lane * 8; }
STRIX_NVFP4_TILE_HD std::uint32_t b_scale(std::uint32_t kb, std::uint32_t lane) { return 512 + kb * 32 + (lane >> 2) * 4; }

// ---- writers: byte holding inputs (k, k + 1) of row o (k even; low nibble = k), and the scale byte of (row, 16-group)
STRIX_NVFP4_TILE_HD std::size_t w_code_byte(std::uint32_t o, std::uint32_t k, std::uint32_t in) {
    const std::uint32_t gr = o & 15, kk = k & 63;
    const std::uint32_t lane = 4 * (gr & 7) + ((kk & 31) >> 3), reg = (gr >> 3) + 2 * (kk >> 5);
    return (static_cast<std::size_t>(o >> 4) * (in / kK) + k / kK) * kWRec + ((k & 127) >> 6) * 512 + lane * 16 + reg * 4 + ((kk & 7) >> 1);
}
STRIX_NVFP4_TILE_HD std::size_t w_scale_byte(std::uint32_t o, std::uint32_t group, std::uint32_t in) {
    const std::uint32_t k = group * 16;
    return (static_cast<std::size_t>(o >> 4) * (in / kK) + k / kK) * kWRec + 1024 + ((k & 127) >> 6) * 64 + (o & 15) * 4 + ((k & 63) >> 4);
}
STRIX_NVFP4_TILE_HD std::size_t x_code_byte(std::uint32_t r, std::uint32_t k, std::uint32_t in) {
    const std::uint32_t kk = k & 63, lane = 4 * (r & 7) + ((kk & 31) >> 3);
    return (static_cast<std::size_t>(r >> 3) * (in / kK) + k / kK) * kXRec + ((k & 127) >> 6) * 256 + lane * 8 + (kk >> 5) * 4 + ((kk & 7) >> 1);
}
STRIX_NVFP4_TILE_HD std::size_t x_scale_byte(std::uint32_t r, std::uint32_t group, std::uint32_t in) {
    const std::uint32_t k = group * 16;
    return (static_cast<std::size_t>(r >> 3) * (in / kK) + k / kK) * kXRec + 512 + ((k & 127) >> 6) * 32 + (r & 7) * 4 + ((k & 63) >> 4);
}

// ---- source: the NVFP4 decode record layout of strix/cuda_core/nvfp4.hpp (16 rows x 128 inputs, 1,152 B; data [h][lane][16 B],
// word i byte b = group 2i + (b >> 1), pair b & 1 -> inputs 16 j + 2t (+8 for pair 1); scales byte 16 g + 8 h + j).
struct RecSlot {
    bool scale = false;
    std::uint32_t row = 0, k = 0;  // k = first (even) input of the byte, or the 16-group index for a scale byte
};
STRIX_NVFP4_TILE_HD RecSlot rec_slot(std::size_t off, std::uint32_t in) {
    const std::size_t rec = off / kWRec;
    const std::uint32_t within = static_cast<std::uint32_t>(off % kWRec), nc = in / kK;
    const std::uint32_t m = static_cast<std::uint32_t>(rec / nc), c = static_cast<std::uint32_t>(rec % nc);
    RecSlot s;
    if (within < 1024) {
        const std::uint32_t h = within >> 9, lane = (within & 511) >> 4, i = (within & 15) >> 2, b = within & 3;
        const std::uint32_t g = lane >> 2, t = lane & 3, j = 2 * i + (b >> 1), p = b & 1;
        s.row = 16 * m + 8 * h + g;
        s.k = kK * c + 16 * j + 2 * t + 8 * p;
    } else {
        const std::uint32_t sb = within - 1024, g = sb >> 4, h = (sb & 15) >> 3, j = sb & 7;
        s.scale = true;
        s.row = 16 * m + 8 * h + g;
        s.k = 8 * c + j;
    }
    return s;
}
// destination of one source record byte in the W tile layout (a pure byte permutation: both nibbles of a source byte are
// consecutive inputs k, k + 1 of one row and stay together)
STRIX_NVFP4_TILE_HD std::size_t repack_dst(std::size_t src_off, std::uint32_t in) {
    const RecSlot s = rec_slot(src_off, in);
    return s.scale ? w_scale_byte(s.row, s.k, in) : w_code_byte(s.row, s.k, in);
}

// ---- GLU staging: in the fused gate/up GEMM a CTA covers 64 channels; its 8 A slots alternate gate / up 16-row tiles of the same
// channels (slot s: weight s & 1, channel tile 4 * tile_m + s / 2), so warp (wm) holds gate in acc[2p] and up in acc[2p + 1] for
// channel tile 4 tile_m + 2 wm + p and SiLU(gate) * up is formed in registers.
STRIX_NVFP4_TILE_HD std::uint32_t glu_slot_tile(std::uint32_t tile_m, std::uint32_t slot) { return 4 * tile_m + slot / 2; }

// ---- CTA / warp / accumulator coordinates (kernel epilogue and host simulation share them). Warp w: wm = w & 1 (weight half),
// wn = w >> 1 (row quarter); accumulator acc[mt][nt][z] of lane (g, t) holds m16n8 element (g + 8 (z >> 1), 2 t + (z & 1)).
STRIX_NVFP4_TILE_HD std::uint32_t a_slot_tile(std::uint32_t tile_m, std::uint32_t slot) { return tile_m * 8 + slot; }
STRIX_NVFP4_TILE_HD std::uint32_t out_row(std::uint32_t tile_m, std::uint32_t wm, std::uint32_t mt, std::uint32_t g, std::uint32_t z) {
    return a_slot_tile(tile_m, wm * 4 + mt) * 16 + g + 8 * (z >> 1);
}
STRIX_NVFP4_TILE_HD std::uint32_t glu_channel(std::uint32_t tile_m, std::uint32_t wm, std::uint32_t p, std::uint32_t g, std::uint32_t z) {
    return glu_slot_tile(tile_m, wm * 4 + 2 * p) * 16 + g + 8 * (z >> 1);
}

// ---- producer of one 128-K stage: kStageWRecs weight records then kStageXRecs activation records, copied verbatim in 16-byte
// chunks c = 0 .. kStageChunks - 1. Chunk c of stage ks reads operand `which` (0 = w, 1 = w2 (GLU up), 2 = x) at byte
// base + ks * step (zero-filled when !ok) and lands at stage offset dst. Nothing but ks depends on the stage, so the kernel resolves
// a chunk once per thread and pays one multiply-add per stage.
inline constexpr std::uint32_t kStageWRecs = 8, kStageXRecs = 32;
inline constexpr std::uint32_t kStageChunksW = kStageWRecs * kWRec / 16, kStageChunks = kStageChunksW + kStageXRecs * kXRec / 16;
struct StageChunk {
    std::size_t base;
    std::uint32_t dst, step, which;
    bool ok;
};
STRIX_NVFP4_TILE_HD StageChunk stage_chunk(std::uint32_t c, std::uint32_t tile_m, std::uint32_t tile_n, bool glu, std::uint32_t ks_count,
                                           std::uint32_t w_tiles, std::uint32_t x_tiles) {
    StageChunk r{};
    if (c < kStageChunksW) {
        const std::uint32_t s = c / (kWRec / 16), off = (c % (kWRec / 16)) * 16;
        const std::uint32_t t = glu ? glu_slot_tile(tile_m, s) : a_slot_tile(tile_m, s);
        r.ok = t < w_tiles;
        r.which = glu && (s & 1) ? 1U : 0U;
        r.base = static_cast<std::size_t>(r.ok ? t : 0) * ks_count * kWRec + off;
        r.dst = s * kWRec + off;
        r.step = kWRec;
    } else {
        const std::uint32_t c2 = c - kStageChunksW, j = c2 / (kXRec / 16), off = (c2 % (kXRec / 16)) * 16;
        const std::uint32_t t = tile_n * kStageXRecs + j;
        r.ok = t < x_tiles;
        r.which = 2;
        r.base = static_cast<std::size_t>(r.ok ? t : 0) * ks_count * kXRec + off;
        r.dst = kStageWRecs * kWRec + j * kXRec + off;
        r.step = kXRec;
    }
    return r;
}
STRIX_NVFP4_TILE_HD std::uint32_t act_row(std::uint32_t tile_n, std::uint32_t wn, std::uint32_t nt, std::uint32_t t, std::uint32_t z) {
    return (tile_n * 32 + wn * 8 + nt) * 8 + 2 * t + (z & 1);
}

enum class Epi : std::uint32_t {
    store,     // y = acc * (gs * rs)
    residual,  // y = aux + acc * (gs * rs)         (aux may alias y)
    silu_mul,  // y = silu(aux) * (acc * (gs * rs))
    glu,       // y[ch] = silu(gate * (gs * rs)) * (up * (gs2 * rs)); w = gate tile, w2 = up tile, out = channels
};
struct GemmArgs {
    const std::uint8_t* w = nullptr;   // W tile layout, out rows (glu: gate, channels rows)
    const std::uint8_t* w2 = nullptr;  // glu: up, same shape
    const std::uint8_t* x = nullptr;   // X tile layout, rows_pad rows
    const float* rs = nullptr;         // [rows_pad] row scales
    const float* aux = nullptr;        // residual / gate input, [rows][out]
    float* y = nullptr;                // [rows][out]
    float gs = 1.f, gs2 = 1.f;         // FP32 tensor scales of w, w2
    std::uint32_t out = 0, in = 0, rows = 0, rows_pad = 0;
    Epi epi = Epi::store;
};
STRIX_NVFP4_TILE_HD bool bytes_disjoint(const void* p, std::size_t pb, const void* q, std::size_t qb) {
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p), b = reinterpret_cast<std::uintptr_t>(q);
    return a + pb <= b || b + qb <= a;
}
// host validation of the shape contract (no device work). Aliasing contract of the epilogue (it issues all of a thread's rs / aux
// loads before that thread's y stores): each y element is read only by the thread that writes it, so aux is y itself (in place)
// or disjoint from y, and rs never overlaps y.
STRIX_NVFP4_TILE_HD bool valid(const GemmArgs& a) {
    if (!a.w || !a.x || !a.rs || !a.y || a.out == 0 || a.out % 16 || a.in == 0 || a.in % kK || a.rows == 0) return false;
    if (a.rows_pad != rows_padded(a.rows)) return false;
    if ((a.epi == Epi::residual || a.epi == Epi::silu_mul) && !a.aux) return false;
    if (a.epi == Epi::glu && !a.w2) return false;
    const std::size_t ybytes = static_cast<std::size_t>(a.rows) * a.out * sizeof(float);
    if (!bytes_disjoint(a.rs, static_cast<std::size_t>(a.rows_pad) * sizeof(float), a.y, ybytes)) return false;
    if (a.aux && a.aux != a.y && !bytes_disjoint(a.aux, ybytes, a.y, ybytes)) return false;
    return true;
}
}  // namespace strix::cuda::core::nvfp4::tile
