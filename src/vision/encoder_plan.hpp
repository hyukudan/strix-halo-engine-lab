#pragma once
// Backend-neutral plan of the vision tower: which steps run, in which order, for which EncoderConfig, and how the
// MLP is chunked. A backend supplies one method per step (instruction bodies, launches, memory) and no ordering.
// Bound today by the CUDA tower (CudaEncoder); the HIP runner (encoder.hip) keeps its own driver until it adopts this.
//
// Backend contract (all row counts are host values; a step that does not apply to the config is never called):
//   embed()                 patch rows -> x            (the patch linear, bias when the config has one)
//   add_positions()         x += position addend       (only when the input carries an addend)
//   norm_embed()            x = norm(x)                (EncoderConfig::embed_norm)
//   attn_norm(l)            nrm = norm1(x)
//   qkv(l)                  qkv = nrm * Wqkv + b
//   rope(l)                 rotate q and k in qkv      (only when the config has rotary positions)
//   attend(l, row0, n)      ao[row0..row0+n) = softmax(q k^T / sqrt(d)) v over one tile, per head
//   out_proj(l)             x += ao * Wout + b
//   mlp_norm(l)             nrm = norm2(x)
//   mlp(l, r0, nr)          x[r0..r0+nr) += mlp2(act(mlp1(nrm[r0..r0+nr)))) for one row chunk
//   tap(l, taps)            append x after block l to the deepstack taps
//   final_norm()            hidden = norm(x)           (EncoderConfig::final_norm)
#include "strix/vision/encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace strix::vision::tower {

// Exclusive row ends of the attention tiles (one tile over all rows when the grid declares none).
inline std::vector<unsigned> tile_ends(const PatchGrid& g) {
    std::vector<unsigned> t = g.tile_end;
    if (t.empty()) t.push_back(g.rows);
    unsigned prev = 0;
    for (const unsigned e : t) {
        if (e <= prev || e > g.rows) throw std::invalid_argument("vision tower: tile ends");
        prev = e;
    }
    if (prev != g.rows) throw std::invalid_argument("vision tower: tiles do not cover the rows");
    return t;
}

// Rows per MLP chunk: the wide middle (mlp_out floats) plus the activated rows (mlp_width floats) stay under the cap.
inline unsigned mlp_chunk_rows(const EncoderConfig& c, unsigned rows, std::size_t workspace_bytes) {
    const std::size_t per_row = std::size_t{4} * (c.mlp_out() + c.mlp_width);
    const std::size_t fit = std::max<std::size_t>(1, workspace_bytes / per_row);
    return static_cast<unsigned>(std::min<std::size_t>(fit, std::max(1U, rows)));
}

// Floats of the device workspace (all FP32 rows): x, nrm, qkv, ao, then the MLP chunk (mid, act).
struct Buffers {
    std::size_t x = 0, nrm = 0, qkv = 0, ao = 0, mid = 0, act = 0, patches = 0, pos = 0, trig = 0;
    std::size_t total() const { return x + nrm + qkv + ao + mid + act + patches + pos + 2U * trig; }
};
inline Buffers buffers(const EncoderConfig& c, unsigned rows, unsigned chunk, bool has_addend) {
    Buffers b;
    const std::size_t n = rows, w = c.width;
    b.x = n * w;
    b.nrm = n * w;
    b.qkv = n * 3U * w;
    b.ao = n * w;
    b.mid = std::size_t{chunk} * c.mlp_out();
    b.act = std::size_t{chunk} * c.mlp_width;
    b.patches = n * c.patch_dim();
    b.pos = has_addend ? n * w : 0U;
    b.trig = c.has_rope() ? n * (c.head_dim() / 2U) : 0U;
    return b;
}

// The step order of one encode. `be` holds the buffers; the plan only sequences it.
template <class Backend>
void drive(const EncoderConfig& c, const std::vector<unsigned>& tiles, unsigned rows, unsigned chunk, bool has_addend, Backend& be,
           EncoderOutput& out) {
    be.embed();
    if (has_addend) be.add_positions();
    if (c.embed_norm) be.norm_embed();
    std::size_t tap = 0;
    for (unsigned l = 0; l < c.layers; ++l) {
        be.attn_norm(l);
        be.qkv(l);
        if (c.has_rope()) be.rope(l);
        unsigned row0 = 0;
        for (const unsigned end : tiles) {
            be.attend(l, row0, end - row0);
            row0 = end;
        }
        be.out_proj(l);
        be.mlp_norm(l);
        for (unsigned r0 = 0; r0 < rows; r0 += chunk) be.mlp(l, r0, std::min(chunk, rows - r0));
        if (tap < c.deepstack_taps.size() && c.deepstack_taps[tap] == l) {
            be.tap(l, out.taps);
            ++tap;
        }
    }
    if (c.final_norm) be.final_norm();
}

}  // namespace strix::vision::tower
