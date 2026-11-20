// Host test of the Hy4-flash tensor-parallel layout (strix/hy4/tp_layout.hpp), the invariant the exact TP2 decode of
// src/hy4/hip/hy4_hip_forward.hip rests on: after each rank writes its share where the layout says and the shared
// pair gather swaps the two slots, both ranks hold the one-node buffer word for word, with no join pass:
//   - half(): the two shares tile every split dimension exactly once (heads, rows, vocabulary); odd sizes refuse;
//   - moe(): [act0 | raw0] / [raw1 | act1] leave the 16 routed slot rows contiguous in slot order and the two shared
//     activation halves one pitch apart, each rank writing only inside its own slot;
//   - decode_stages(): the planner's joins are the step's gathers (128 on the pack's shape), every join exact, and
//     the plan prefers the pair at a fast trip and one node at a slow one.
#include "strix/hy4/tp_layout.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

namespace htp = strix::hy4::tp;

void halves_tile() {
    for (const std::size_t n : {std::size_t{2}, std::size_t{32}, std::size_t{64}, std::size_t{3072}, std::size_t{13824},
                                std::size_t{120832}}) {
        std::vector<int> owner(n, 0);
        for (std::uint32_t r = 0; r < 2U; ++r) {
            const auto h = htp::half(n, r);
            for (std::size_t i = h.first; i < h.first + h.count; ++i) owner[i] += 1;
        }
        bool once = true;
        for (const int o : owner) once = once && o == 1;
        expect(once, "half(" + std::to_string(n) + ") tiles the dimension exactly once");
        expect(htp::half(n, 0).first == 0U && htp::half(n, 1).first == n / 2U, "rank 0 owns the low half, rank 1 the high half");
    }
    bool threw = false;
    try { (void)htp::half(31, 0); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "odd dimension refused");
    threw = false;
    try { (void)htp::half(32, 2); } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "rank 2 refused");
}

// One MoE gather simulated on the host: each rank fills its own buffer at its layout offsets, then the symmetric swap
// of GatherLink (slot[1 - rank] <- the peer's slot[1 - rank]).
void moe_join(std::size_t K, std::size_t L, std::size_t S) {
    const std::string tag = "moe K " + std::to_string(K) + " L " + std::to_string(L) + " shared " + std::to_string(S);
    std::vector<float> raw(K * L), act(S);  // the one-node rawdots [K][L] and shared activations [S]
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = 1.0F + static_cast<float>(i);
    for (std::size_t i = 0; i < act.size(); ++i) act[i] = -1.0F - static_cast<float>(i);
    const auto m0 = htp::moe(0, K, L, S);
    const auto m1 = htp::moe(1, K, L, S);
    expect(m0.slot == m1.slot && m0.raw_all == m1.raw_all && m0.act_pitch == m1.act_pitch, tag + ": both ranks agree on the layout");
    std::vector<std::vector<float>> buf(2, std::vector<float>(2U * m0.slot, 0.0F));
    for (std::uint32_t r = 0; r < 2U; ++r) {
        const auto m = htp::moe(r, K, L, S);
        const std::size_t lo = r * m.slot, hi = lo + m.slot;
        const auto sr = htp::half(S, r);
        expect(m.raw_mine >= lo && m.raw_mine + m.slots * L <= hi, tag + ": slot rows inside the rank's slot");
        expect(m.act_mine >= lo && m.act_mine + sr.count <= hi, tag + ": activations inside the rank's slot");
        expect(m.slots == K / 2U && m.slot_first == r * (K / 2U), tag + ": slots 8 + 8 in slot order");
        for (std::size_t i = 0; i < m.slots * L; ++i) buf[r][m.raw_mine + i] = raw[m.slot_first * L + i];
        for (std::size_t i = 0; i < sr.count; ++i) buf[r][m.act_mine + i] = act[sr.first + i];
    }
    for (std::uint32_t r = 0; r < 2U; ++r) {  // the swap
        const std::uint32_t q = 1U - r;
        for (std::size_t i = 0; i < m0.slot; ++i) buf[r][q * m0.slot + i] = buf[q][q * m0.slot + i];
    }
    expect(buf[0] == buf[1], tag + ": both ranks hold the same joined buffer");
    bool rows = true, acts = true;
    for (std::size_t i = 0; i < K * L; ++i) rows = rows && buf[0][m0.raw_all + i] == raw[i];
    for (std::size_t h = 0; h < 2U; ++h)
        for (std::size_t i = 0; i < S / 2U; ++i) acts = acts && buf[0][m0.act_first + h * m0.act_pitch + i] == act[h * (S / 2U) + i];
    expect(rows, tag + ": the 16 slot rows are contiguous, in slot order, word for word");
    expect(acts, tag + ": the activation halves sit one pitch apart (one 2-row copy gives the one-node activations)");
}

struct FakeConfig {  // the Config fields the layout reads (src/hy4/ref Config)
    std::uint32_t layers = 42, hidden = 3072, vocab = 120832, v_head = 256, dense_ff = 13824, top_k = 16, latent = 1536,
                  shared_ff = 1536, experts = 512;
    std::vector<std::uint32_t> heads;
    std::vector<std::uint8_t> dense, indexed;
};

void stages_match_step() {
    FakeConfig c;
    for (std::uint32_t l = 0; l < c.layers; ++l) {
        c.indexed.push_back(static_cast<std::uint8_t>(l % 2U));
        c.heads.push_back(l % 2U ? 32U : 64U);
        c.dense.push_back(static_cast<std::uint8_t>(l == 0U));
    }
    const auto n = htp::step_count(htp::dims_of(c));
    expect(n.gathers == 128U, "the pack's shape: 128 gathers per step (got " + std::to_string(n.gathers) + ")");
    // synthetic tensor bytes: 1 MB per named tensor, experts 512 MB per layer (1 MB per expert)
    const auto bytes_of = [](const std::string& name) -> std::uint64_t {
        return name.find("_exps") != std::string::npos ? (512ULL << 20) / 3ULL : (1ULL << 20);
    };
    const auto stages = htp::decode_stages(c, bytes_of);
    namespace ps = strix::pair::split;
    const ps::Costs fast{{200e6, 200e6}, {0.00001, 0.0}, 0.0, false};
    const auto pf = ps::plan(stages, ps::Pass{}, fast);
    expect(pf.tensor.trips == n.gathers, "planner trips = step gathers (" + std::to_string(pf.tensor.trips) + ")");
    expect(pf.tensor.exact && pf.tensor.admissible, "every join exact (gathers and slot joins, no reduce)");
    expect(pf.tensor.ms < pf.single.ms, "a fast trip: the pair wins");
    double total = 0.0;  // one node reads every stage (a slot stage: its top_k slots per row)
    for (const auto& s : stages) total += s.axis == ps::Axis::slots ? s.bytes * s.units : s.bytes;
    expect(pf.tensor.rank_bytes.size() == 2U && pf.tensor.rank_bytes[0] < 0.75 * total && pf.tensor.rank_bytes[1] < 0.75 * total,
           "each rank reads well under the one-node bytes");
    const ps::Costs slow{{200e6, 200e6}, {5.0, 0.0}, 0.0, false};  // 5 ms per trip
    const auto psl = ps::plan(stages, ps::Pass{}, slow);
    expect(!(psl.tensor.ms < psl.single.ms), "a slow trip: one node wins");
}

}  // namespace

int main() {
    halves_tile();
    moe_join(16, 1536, 1536);
    moe_join(16, 8, 12);
    moe_join(4, 3, 2);
    stages_match_step();
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS: hy4 tensor-parallel layout (halves tile, MoE join contiguous, planner joins = step gathers)\n");
    return 0;
}
