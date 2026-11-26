// Host tests of the deepseek41 K2 routed-expert loader (ds41_k2_experts.hpp) and of the shared EXL3 reader traits it
// relies on (moe_core/exl3_lane_window.hpp): lane window words for K 1..8 against a bit-level count, the split plan of
// the single-node and two-rank forms, a synthetic K2 pack written as safetensors and read back per rank (tile bytes,
// suh / svh slices bitwise against an independent extraction), the refusals (K3 tensor, wrong codebook, missing expert),
// byte accounting and the lever row.
#include "strix/ds41_k2_experts.hpp"

#include "moe_core/exl3_lane_window.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

namespace k2 = strix::ds41::k2;
namespace lw = strix::moe_core::lane_window;

// Words a lane touches, counted bit by bit from the window start (independent of the trait's closed form).
unsigned words_by_bits(int bits, unsigned lane) {
    const unsigned start = (lane * 8U + 1U) * static_cast<unsigned>(bits) + 256U * static_cast<unsigned>(bits) - 16U;
    unsigned last = 0;
    for (unsigned j = 0; j < 8U; ++j)
        for (unsigned b = 0; b < 16U; ++b) {
            const unsigned word = (start + j * static_cast<unsigned>(bits) + b) / 32U - start / 32U;
            if (word > last) last = word;
        }
    return last + 1U;
}

void test_lane_window() {
    for (int bits = 1; bits <= 8; ++bits) {
        bool third = false;
        for (unsigned lane = 0; lane < 32U; ++lane) {
            CHECK(lw::words(bits, lane) == words_by_bits(bits, lane));
            third = third || words_by_bits(bits, lane) == 3U;
        }
        CHECK(lw::needs_third_word(bits) == third);
        CHECK(lw::needs_third_word(bits) == (bits >= 5));
    }
    // K3 shifts named in exl3_trellis.hpp LdsRingU, K2 shifts of the K2 reader.
    CHECK(lw::shift(3, 0) == 19U && lw::shift(3, 1) == 11U && lw::shift(3, 2) == 3U && lw::shift(3, 3) == 27U);
    CHECK(lw::shift(2, 0) == 18U && lw::shift(2, 1) == 2U);
    std::puts("PASS lane window: K1..K4 two words, K5..K8 three (all 32 lanes, bit count)");
}

void test_split_plan() {
    const k2::Geometry g{};  // DS41: H 5120, I 2304, 384 experts, K2
    const auto one = k2::split_plan(g, k2::rank_slice(g, 0, 1));
    CHECK(one.gate_splits == 16U && one.down_splits == 4U && one.down_first == 0U && one.down_end == 4U && one.down_k_tile0 == 0U);
    const auto r0 = k2::split_plan(g, k2::rank_slice(g, 0, 2));
    const auto r1 = k2::split_plan(g, k2::rank_slice(g, 1, 2));
    CHECK(r0.down_first == 0U && r0.down_end == 2U && r1.down_first == 2U && r1.down_end == 4U && r1.down_k_tile0 == 72U);
    CHECK(r1.gate_splits == one.gate_splits && r1.down_splits == one.down_splits);
    bool threw = false;
    try {
        (void)k2::rank_slice(g, 0, 3);  // 768 per rank: 128-aligned, but 768 is not a down split boundary
        (void)k2::split_plan(g, k2::rank_slice(g, 0, 3));
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    // Bytes: one layer, all experts.
    CHECK(k2::layer_bytes(g, k2::rank_slice(g, 0, 1), 384) == 3414491136ULL);
    CHECK(k2::layer_bytes(g, k2::rank_slice(g, 1, 2), 384) == 1713143808ULL);
    std::puts("PASS split plan: 16 gate / 4 down splits, rank 1 = splits [2, 4) from k tile 72; 3.414 GB/layer, 1.713 GB/layer/rank");
}

// ---- synthetic pack
struct Tensor {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> data;
};

void write_safetensors(const std::filesystem::path& file, const std::vector<Tensor>& ts) {
    std::string header = "{";
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < ts.size(); ++i) {
        header += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[";
        for (std::size_t d = 0; d < ts[i].shape.size(); ++d) header += (d ? "," : "") + std::to_string(ts[i].shape[d]);
        header += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + ts[i].data.size()) + "]}";
        off += ts[i].data.size();
    }
    header += "}";
    while (header.size() % 8U != 0U) header += ' ';
    std::ofstream f(file, std::ios::binary);
    const std::uint64_t n = header.size();
    unsigned char len[8];
    for (int b = 0; b < 8; ++b) len[b] = static_cast<unsigned char>((n >> (8 * b)) & 255U);
    f.write(reinterpret_cast<const char*>(len), 8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& t : ts) f.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size()));
}

std::vector<std::uint8_t> random_bytes(std::mt19937& rng, std::size_t n) {
    std::vector<std::uint8_t> v(n);
    for (auto& b : v) b = static_cast<std::uint8_t>(rng() & 255U);
    return v;
}

struct Mat {
    std::vector<std::uint8_t> trellis, suh, svh;
};

void add_linear(std::vector<Tensor>& ts, const std::string& name, std::uint32_t in, std::uint32_t out, std::uint32_t bits, const Mat& m,
                std::uint32_t marker = 0x83DCD12DU) {
    ts.push_back({name + ".trellis", "I16", {in / 16U, out / 16U, 16U * bits}, m.trellis});
    ts.push_back({name + ".suh", "F16", {in}, m.suh});
    ts.push_back({name + ".svh", "F16", {out}, m.svh});
    std::vector<std::uint8_t> mk(4);
    for (int b = 0; b < 4; ++b) mk[static_cast<std::size_t>(b)] = static_cast<std::uint8_t>((marker >> (8 * b)) & 255U);
    ts.push_back({name + ".mul1", "I32", {}, mk});
}

// Independent extraction: tiles (k, n) of a [in/16][out/16] tile grid, k in [k0, k1), n in [n0, n1), in k-major order.
std::vector<std::uint8_t> tiles(const std::vector<std::uint8_t>& t, std::uint32_t out, std::uint32_t bits, std::uint32_t k0, std::uint32_t k1,
                                std::uint32_t n0, std::uint32_t n1) {
    const std::size_t tile = 32U * bits, per_k = out / 16U;
    std::vector<std::uint8_t> v;
    for (std::uint32_t k = k0; k < k1; ++k)
        for (std::uint32_t n = n0; n < n1; ++n) v.insert(v.end(), t.begin() + static_cast<std::ptrdiff_t>((k * per_k + n) * tile),
                                                          t.begin() + static_cast<std::ptrdiff_t>((k * per_k + n + 1) * tile));
    return v;
}

std::vector<std::uint16_t> halves(const std::vector<std::uint8_t>& b, std::uint32_t first, std::uint32_t count) {
    std::vector<std::uint16_t> v(count);
    std::memcpy(v.data(), b.data() + 2U * first, 2U * count);
    return v;
}

void test_pack() {
    const k2::Geometry g{256, 256, 3, 2};
    const auto root = std::filesystem::temp_directory_path() / ("strix-k2-pack-test-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(root / "L0");
    std::filesystem::create_directories(root / "L1");
    std::mt19937 rng(1234);
    Mat m[3][3];  // [expert][gate, up, down]
    std::vector<Tensor> ts;
    for (std::uint32_t e = 0; e < g.experts; ++e)
        for (int w = 0; w < 3; ++w) {
            const std::uint32_t in = w < 2 ? g.hidden : g.inter, out = w < 2 ? g.inter : g.hidden;
            m[e][w] = {random_bytes(rng, std::size_t{in} * out * g.bits / 8U), random_bytes(rng, 2U * in), random_bytes(rng, 2U * out)};
            add_linear(ts, k2::pack_name(0, e, static_cast<k2::Mat>(w)), in, out, g.bits, m[e][w]);
        }
    write_safetensors(root / "L0" / "x.L0.c00.safetensors", ts);
    // L1: expert 0 complete, expert 1 gate stored as K3, expert 2 up with a wrong codebook marker.
    std::vector<Tensor> bad;
    for (int w = 0; w < 3; ++w) add_linear(bad, k2::pack_name(1, 0, static_cast<k2::Mat>(w)), w < 2 ? g.hidden : g.inter, w < 2 ? g.inter : g.hidden, 2, m[0][w]);
    add_linear(bad, k2::pack_name(1, 1, k2::Mat::gate), g.hidden, g.inter, 3,
               {random_bytes(rng, std::size_t{g.hidden} * g.inter * 3U / 8U), m[1][0].suh, m[1][0].svh});
    add_linear(bad, k2::pack_name(1, 2, k2::Mat::gate), g.hidden, g.inter, 2, m[2][0]);
    add_linear(bad, k2::pack_name(1, 2, k2::Mat::up), g.hidden, g.inter, 2, m[2][1], 0x12345678U);
    add_linear(bad, k2::pack_name(1, 2, k2::Mat::down), g.inter, g.hidden, 2, m[2][2]);
    write_safetensors(root / "L1" / "x.L1.c00.safetensors", bad);
    {
        k2::Pack pack(root, g);
        for (std::uint32_t ranks : {1U, 2U})
            for (std::uint32_t rank = 0; rank < ranks; ++rank) {
                const auto s = k2::rank_slice(g, rank, ranks);
                const auto img = pack.load_layer(0, s);
                CHECK(img.expert_count == 3U && img.gate_tile_bytes == std::uint64_t{g.hidden} * s.count * g.bits / 8U);
                CHECK(img.bytes() == k2::layer_bytes(g, s, 3));
                std::vector<std::uint8_t> gate, up, down;
                std::vector<std::uint16_t> gsuh, usuh, dsuh, gsvh, usvh, dsvh;
                for (std::uint32_t e = 0; e < 3; ++e) {
                    const auto a = tiles(m[e][0].trellis, g.inter, g.bits, 0, g.hidden / 16U, s.first / 16U, (s.first + s.count) / 16U);
                    const auto b = tiles(m[e][1].trellis, g.inter, g.bits, 0, g.hidden / 16U, s.first / 16U, (s.first + s.count) / 16U);
                    const auto c = tiles(m[e][2].trellis, g.hidden, g.bits, s.first / 16U, (s.first + s.count) / 16U, 0, g.hidden / 16U);
                    gate.insert(gate.end(), a.begin(), a.end());
                    up.insert(up.end(), b.begin(), b.end());
                    down.insert(down.end(), c.begin(), c.end());
                    for (auto [dst, src, f, n] : {std::tuple{&gsuh, &m[e][0].suh, 0U, g.hidden}, std::tuple{&usuh, &m[e][1].suh, 0U, g.hidden},
                                                  std::tuple{&dsuh, &m[e][2].suh, s.first, s.count}, std::tuple{&gsvh, &m[e][0].svh, s.first, s.count},
                                                  std::tuple{&usvh, &m[e][1].svh, s.first, s.count}, std::tuple{&dsvh, &m[e][2].svh, 0U, g.hidden}}) {
                        const auto h = halves(*src, f, n);
                        dst->insert(dst->end(), h.begin(), h.end());
                    }
                }
                CHECK(img.gate == gate && img.up == up && img.down == down);
                CHECK(img.gate_suh == gsuh && img.up_suh == usuh && img.down_suh == dsuh);
                CHECK(img.gate_svh == gsvh && img.up_svh == usvh && img.down_svh == dsvh);
            }
        const auto sub = pack.load_layer(0, k2::rank_slice(g, 1, 2), 1, 1);  // one expert
        CHECK(sub.expert_first == 1U && sub.expert_count == 1U && sub.gate.size() == sub.gate_tile_bytes);
        const auto one = pack.load_layer(1, k2::rank_slice(g, 0, 2), 0, 1);  // L1 expert 0 is valid
        CHECK(one.expert_count == 1U);
        auto refused = [&](std::uint32_t layer, std::uint32_t first, const char* what) {
            try {
                (void)pack.load_layer(layer, k2::rank_slice(g, 0, 1), first, 1);
            } catch (const std::exception& ex) {
                return std::string(ex.what()).find(what) != std::string::npos;
            }
            return false;
        };
        CHECK(refused(1, 1, "K3"));       // a K3 tensor in a K2 pack
        CHECK(refused(1, 2, "mul1"));     // wrong codebook marker
        CHECK(refused(0, 3, "expert 3"));  // past the experts
        k2::Pack wide(root, k2::Geometry{256, 256, 4, 2});
        bool missing = false;
        try {
            (void)wide.load_layer(0, k2::rank_slice(g, 0, 1), 3, 1);
        } catch (const std::exception& ex) { missing = std::string(ex.what()).find("no layers.0.ffn.experts.3.w1") != std::string::npos; }
        CHECK(missing);
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);  // the test's own temporary pack
    std::puts("PASS synthetic K2 pack: per-rank tiles / suh / svh bitwise, refusals (K3, codebook, missing, range)");
}

void test_lever() {
    const auto& row = k2::decode_k2_experts_lever;
    CHECK(row.name == std::string_view("decode_k2_experts") && row.exactness == strix::lever::Exactness::bit_changing && !row.default_on);
    CHECK(!strix::lever::select(row, {}, nullptr).on);
    CHECK(strix::lever::select(row, {}, nullptr).origin == strix::lever::Origin::gate_closed);
    CHECK(strix::lever::select(row, {}, "1").on);
    CHECK(!strix::lever::select(row, {}, nullptr, true).on);  // gate open: still default off until timing promotes it
    static constexpr strix::lever::Decl table[] = {k2::decode_k2_experts_lever};
    static_assert(strix::lever::valid(table));
    CHECK(k2::step_identity(2) == "ds41-expert-k2-v1:tp2" && k2::step_identity(1) == "ds41-expert-k2-v1:single");
    std::puts("PASS lever row decode_k2_experts: bit_changing, default off, explicit switch selects it");
}

}  // namespace

int main() {
    test_lane_window();
    test_split_plan();
    test_pack();
    test_lever();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("ds41 k2 experts tests: PASS");
    return 0;
}
