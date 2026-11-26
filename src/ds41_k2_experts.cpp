#include "strix/ds41_k2_experts.hpp"

#include "strix/ds41_trellis.hpp"

#include "moe_core/exl3_lane_window.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace strix::ds41::k2 {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("ds41 k2 experts: " + what); }

void append_u16(std::vector<std::uint16_t>& out, const std::vector<std::uint8_t>& blob, std::uint64_t offset, std::uint64_t bytes) {
    const std::size_t n = static_cast<std::size_t>(bytes / 2U);
    const std::size_t at = out.size();
    out.resize(at + n);
    std::memcpy(out.data() + at, blob.data() + offset, n * 2U);
}

}  // namespace

const char* mat_suffix(Mat m) noexcept {
    switch (m) {
    case Mat::gate: return "w1";
    case Mat::up: return "w3";
    case Mat::down: return "w2";
    }
    return "w1";
}

std::string pack_name(std::uint32_t layer, std::uint32_t expert, Mat m) {
    return "layers." + std::to_string(layer) + ".ffn.experts." + std::to_string(expert) + "." + mat_suffix(m);
}

RankSlice rank_slice(const Geometry& g, std::uint32_t rank, std::uint32_t ranks) {
    if (ranks == 0U || rank >= ranks) fail("rank " + std::to_string(rank) + " of " + std::to_string(ranks));
    if (g.inter % ranks != 0U || (g.inter / ranks) % 128U != 0U || g.hidden % 128U != 0U)
        fail("I = " + std::to_string(g.inter) + " over " + std::to_string(ranks) + " ranks is not 128-aligned");
    RankSlice s;
    s.rank = rank;
    s.ranks = ranks;
    s.count = g.inter / ranks;
    s.first = rank * s.count;
    return s;
}

SplitPlan split_plan(const Geometry& g, const RankSlice& s) {
    SplitPlan p;
    p.gate_splits = moe_core::split_policy::vector_splits(g.hidden, g.inter);
    p.down_splits = moe_core::split_policy::vector_splits(g.inter, g.hidden);
    const std::uint32_t kt = g.inter / 16U;
    // The split boundaries are kt * j / splits (stream_column_lds_u); the rank's range must be a union of whole splits.
    // With more splits than k tiles several j share a boundary (empty splits): both ends take the first j, so the
    // empty splits at a rank boundary belong to the next rank and no split is run by two ranks.
    bool first_found = false, end_found = false;
    for (std::uint32_t j = 0; j <= p.down_splits; ++j) {
        const std::uint32_t at = kt * j / p.down_splits * 16U;
        if (at == s.first && !first_found) { p.down_first = j; first_found = true; }
        if (at == s.first + s.count && !end_found) { p.down_end = j; end_found = true; }
    }
    if (!first_found || !end_found || p.down_first >= p.down_end)
        fail("rank range [" + std::to_string(s.first) + ", " + std::to_string(s.first + s.count) + ") is not a union of the " +
             std::to_string(p.down_splits) + " down splits of I = " + std::to_string(g.inter));
    // Gate/up: the rank's output columns must be whole 16-column tiles; its 16 splits over H are the single-node ones.
    if (s.first % 16U != 0U || s.count % 16U != 0U) fail("rank range not tile-aligned");
    p.down_k_tile0 = s.first / 16U;
    return p;
}

std::uint64_t layer_bytes(const Geometry& g, const RankSlice& s, std::uint32_t experts) {
    const std::uint64_t h = g.hidden, n = s.count;
    const std::uint64_t per = 3U * h * n * g.bits / 8U + 2U * (2U * (h + n) + (n + h));
    return per * experts;
}

std::uint64_t LayerImage::bytes() const noexcept {
    return gate.size() + up.size() + down.size() +
           2U * (gate_suh.size() + up_suh.size() + down_suh.size() + gate_svh.size() + up_svh.size() + down_svh.size());
}

Pack::Pack(std::filesystem::path root, Geometry g) : root_(std::move(root)), geometry_(g) {
    if (!std::filesystem::is_directory(root_)) fail("pack directory " + root_.string() + " not found");
    if (g.bits < 1U || g.bits > 8U) fail("K " + std::to_string(g.bits));
}

Pack::~Pack() = default;

TrellisPack& Pack::layer_pack(std::uint32_t layer) {
    auto dir = root_ / ("L" + std::to_string(layer));
    if (!std::filesystem::is_directory(dir)) dir = root_;
    auto& p = packs_[dir];
    if (!p) p = std::make_unique<TrellisPack>(dir);
    return *p;
}

LayerImage Pack::load_layer(std::uint32_t layer, const RankSlice& s, std::uint32_t first, std::uint32_t count) {
    const Geometry& g = geometry_;
    if (first >= g.experts) fail("expert " + std::to_string(first) + " of " + std::to_string(g.experts));
    if (count == 0U) count = g.experts - first;
    if (count > g.experts - first) fail("experts past " + std::to_string(g.experts));
    (void)split_plan(g, s);  // the rank range must keep the single-node split chains
    auto& pack = layer_pack(layer);
    LayerImage out;
    out.layer = layer;
    out.expert_first = first;
    out.expert_count = count;
    out.geometry = g;
    out.slice = s;
    out.gate_tile_bytes = std::uint64_t{g.hidden} * s.count * g.bits / 8U;
    out.down_tile_bytes = out.gate_tile_bytes;
    for (auto* v : {&out.gate, &out.up, &out.down}) v->reserve(static_cast<std::size_t>(out.gate_tile_bytes * count));
    for (std::uint32_t e = first; e < first + count; ++e) {
        for (const Mat m : {Mat::gate, Mat::up, Mat::down}) {
            const std::string name = pack_name(layer, e, m);
            const auto info = pack.info(name);
            if (!info) fail("no " + name + " in " + pack.dir().string());
            const bool gu = m != Mat::down;
            const std::uint32_t rows = gu ? g.inter : g.hidden, columns = gu ? g.hidden : g.inter;
            if (info->bits != g.bits || info->rows != rows || info->columns != columns)
                fail(name + ": K" + std::to_string(info->bits) + " " + std::to_string(info->rows) + " x " + std::to_string(info->columns) +
                     ", expected K" + std::to_string(g.bits) + " " + std::to_string(rows) + " x " + std::to_string(columns));
            TrellisSlice slice;
            if (s.ranks > 1U) {
                slice.kind = gu ? TrellisSlice::Kind::rows : TrellisSlice::Kind::columns;
                slice.first = s.first;
                slice.count = s.count;
            }
            const TrellisHostBlob b = pack.load(name, slice);  // checks the mul1 marker, suh / svh types and sizes
            auto& tiles = m == Mat::gate ? out.gate : m == Mat::up ? out.up : out.down;
            if (b.trellis_bytes != out.gate_tile_bytes) fail(name + ": slice bytes");
            tiles.insert(tiles.end(), b.blob.begin(), b.blob.begin() + static_cast<std::ptrdiff_t>(b.trellis_bytes));
            append_u16(m == Mat::gate ? out.gate_suh : m == Mat::up ? out.up_suh : out.down_suh, b.blob, b.suh_offset, b.suh_bytes);
            append_u16(m == Mat::gate ? out.gate_svh : m == Mat::up ? out.up_svh : out.down_svh, b.blob, b.svh_offset, b.svh_bytes);
        }
    }
    return out;
}

std::string step_identity(std::uint32_t ranks) {
    return std::string(identity) + (ranks > 1U ? ":tp" + std::to_string(ranks) : ":single");
}

}  // namespace strix::ds41::k2
