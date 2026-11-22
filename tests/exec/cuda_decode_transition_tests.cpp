// Host test of the residual transition fuser of the shared CUDA decode recipe (B3 row owner): which leaves a program
// emits for add -> bank readout -> input norms, with the form off and on, and the legality rules of TransitionArgs.
// No GPU is used: bind() only records launches.
#include "exec/cuda_decode_transition.hpp"
#include "cuda/core/mix_cuda_latency_plan.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace decode = strix::exec::cuda_decode;
namespace residual = strix::core::residual_mix;

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

struct Node {
    std::string identity;
    strix::exec::Grid grid;
    std::vector<std::byte> args;
    bool pdl;
};

constexpr unsigned kWidth = 2304U;
constexpr const char* kAdd = "cuda-sm120-residual-add-r1-v1";
constexpr const char* kK8c = "cuda-sm120-residual-bank-mix-r1-v1";
constexpr const char* kTiled = "cuda-sm120-residual-bank-softmax-tiled-r1-v1";
constexpr const char* kRms = "cuda-sm120-rms-bf16-image-r1-v1";
constexpr const char* kTransition = "cuda-sm120-residual-transition-r1-v1";

alignas(256) float blocks[4][kWidth];
alignas(256) float hidden[kWidth];
alignas(256) float residual_row[kWidth];
alignas(256) float readout[3][kWidth];
alignas(256) float inputs[3][kWidth];
alignas(256) float queries[3 * kWidth];
alignas(256) std::uint16_t gains[3][kWidth];

struct Recorder {
    std::vector<Node> nodes;
    auto emit() {
        return [this](const auto& leaf, bool pdl) {
            auto bound = decode::bind(leaf);
            nodes.push_back({bound.identity, bound.grid, bound.args, pdl});
        };
    }
};

residual::MixArgs read(unsigned prior_blocks, float* current, unsigned outputs) {
    residual::MixArgs mix{};
    for (unsigned b = 0; b < prior_blocks; ++b) mix.banks[b] = blocks[b];
    mix.banks[prior_blocks] = current;
    mix.bank_count = prior_blocks + 1U;
    mix.queries = queries;
    mix.rows = 1U;
    mix.width = kWidth;
    mix.query_count = outputs;
    mix.epsilon = 1e-6F;
    for (unsigned q = 0; q < outputs; ++q) mix.output[q] = readout[q];
    return mix;
}

decode::RmsBf16 rms(unsigned q) { return {readout[q], gains[q], inputs[q], 1U, kWidth, 1e-6F}; }

// One visit-shaped chain: end-of-visit add (rounded) into hidden, attention read with three outputs and three input
// norms, then an unrelated leaf; then the in-place mid-visit add, the one-query mlp read and its norm.
std::vector<Node> chain(decode::R1Forms forms) {
    Recorder r;
    decode::TransitionChain fuser(forms);
    const auto emit = r.emit();
    fuser.add({residual_row, readout[0], hidden, kWidth, true}, emit);
    fuser.readout(read(3, hidden, 3), emit);
    for (unsigned q = 0; q < 3; ++q) fuser.norm(rms(q), emit);
    fuser.flush(emit);
    emit(decode::Bf16Round{inputs[0], kWidth}, false);  // stands for the attention leaves
    fuser.add({residual_row, hidden, residual_row, kWidth, false}, emit);
    fuser.readout(read(3, residual_row, 1), emit);
    fuser.norm(rms(0), emit);
    fuser.flush(emit);
    return r.nodes;
}

void program_chain() {
    const auto base = chain({.bank_tiled = true});
    const auto fused = chain({.bank_tiled = true, .transition = true});
    // Incumbent: add, tiled read, 3 norms, round, add, tiled read, norm.
    const std::vector<std::string> want_base{kAdd, kTiled, kRms, kRms, kRms, "cuda-sm120-pointwise-bf16-round-v1",
                                             kAdd, kTiled, kRms};
    check(base.size() == want_base.size(), "incumbent chain node count");
    for (std::size_t i = 0; i < base.size() && i < want_base.size(); ++i) check(base[i].identity == want_base[i], "incumbent chain leaf");
    check(fused.size() == 3U, "transition chain = transition, round, transition");
    if (fused.size() != 3U) return;
    check(fused[0].identity == kTransition && fused[2].identity == kTransition, "transition leaves");
    check(fused[1].identity == base[5].identity && fused[1].args == base[5].args, "the unrelated leaf keeps its place");
    check(fused[0].pdl && fused[2].pdl, "transition waits on its producer through PDL");
    residual::TransitionArgs first{}, second{};
    std::memcpy(&first, fused[0].args.data(), sizeof(first));
    std::memcpy(&second, fused[2].args.data(), sizeof(second));
    residual::MixArgs base_read{};
    std::memcpy(&base_read, base[1].args.data(), sizeof(base_read));
    bool same_read = first.mix.queries == base_read.queries && first.mix.rows == base_read.rows &&
                     first.mix.width == base_read.width && first.mix.bank_count == base_read.bank_count &&
                     first.mix.query_count == base_read.query_count && first.mix.epsilon == base_read.epsilon &&
                     first.mix.normalized == base_read.normalized;
    for (unsigned b = 0; b < residual::kMaxBanks; ++b) same_read = same_read && first.mix.banks[b] == base_read.banks[b];
    for (unsigned q = 0; q < residual::kMaxQueries; ++q) same_read = same_read && first.mix.output[q] == base_read.output[q];
    check(same_read, "transition keeps the incumbent readout operands");
    residual::AddArgs base_add{};
    std::memcpy(&base_add, base[0].args.data(), sizeof(base_add));
    check(first.add.lhs == base_add.lhs && first.add.rhs == base_add.rhs && first.add.output == base_add.output &&
              first.add.elements == base_add.elements && first.add.round_bf16 == base_add.round_bf16,
          "transition keeps the incumbent add operands");
    for (unsigned q = 0; q < 3; ++q) {
        decode::RmsBf16 norm{};
        std::memcpy(&norm, base[2U + q].args.data(), sizeof(norm));
        check(first.norm[q].output == norm.output && first.norm[q].gain == norm.gain && norm.input == first.mix.output[q],
              "transition norm q = incumbent norm of raw row q");
    }
    check(first.norm_epsilon == 1e-6F && residual::norm_count(first) == 3U, "three norms attached");
    check(second.add.lhs == residual_row && second.add.output == residual_row && !second.add.round_bf16 &&
              second.mix.query_count == 1U && residual::norm_count(second) == 1U, "in-place mid-visit add + mlp read + norm");
    check(fused[0].grid.blocks[0] == 5U && fused[0].grid.blocks[1] == 1U && fused[0].grid.threads[0] == 128U &&
              fused[0].grid.threads[1] == 4U,
          "one cluster of 5 tiles (width 2304), blocks of 128 x banks threads");
}

// The cluster plan of the transition: tiles cover the row in the portable cluster size, each tile starts on an
// a8g32 group, and the owner / offset map of the RMS lanes visits every value of the row once, in the rms.bf16 order.
void cluster_plan() {
    namespace mix = strix::cuda::mix;
    check(mix::transition_plan(0, 3).tiles == 0U && mix::transition_plan(4097, 3).tiles == 0U &&
              mix::transition_plan(2304, 0).tiles == 0U && mix::transition_plan(2304, 9).tiles == 0U,
          "plan refuses empty, too wide and bank counts outside 1..8");
    check(mix::transition_plan(1, 1).slices == 2U && mix::transition_plan(2304, 5).slices == 5U,
          "slices = banks, at least the 256 norm lanes");
    for (unsigned width = 1; width <= mix::kLatencyMaxWidth; ++width) {
        const auto plan = mix::transition_plan(width, 3);
        const bool covers = plan.tiles >= 1U && plan.tiles <= mix::kTransitionMaxClusterTiles &&
                            (plan.tiles - 1U) * mix::kLatencyTile < width && plan.tiles * mix::kLatencyTile >= width;
        if (!covers) {
            check(false, "tiles cover the row within the portable cluster size");
            break;
        }
    }
    // Lane t of 256 visits t, t + 256, ... ascending; every index once, owned by a live tile at an in-tile offset.
    for (unsigned width : {96U, 513U, 1000U, 2304U, 4096U}) {
        const auto tiles = mix::transition_plan(width, 3).tiles;
        std::vector<unsigned> seen(width, 0U);
        bool ok = true;
        for (unsigned t = 0; t < mix::kTransitionNormLanes; ++t)
            for (unsigned idx = t; idx < width; idx += mix::kTransitionNormLanes) {
                const unsigned owner = mix::transition_owner(idx), offset = mix::transition_offset(idx);
                ok = ok && owner < tiles && offset < mix::kLatencyTile && owner * mix::kLatencyTile + offset == idx &&
                     (owner * mix::kLatencyTile) % 32U == 0U;
                ++seen[idx];
            }
        for (unsigned v : seen) ok = ok && v == 1U;
        check(ok, "RMS lanes read every row value once from its owner tile");
    }
}

// The shared-memory staging of the transition: raw tiles + (prologue) lhs and rhs rows + a ring of query rows, within
// the device budget; queries beyond the ring are staged in batches, and a budget below one query row refuses.
void stage_plan() {
    namespace mix = strix::cuda::mix;
    constexpr std::size_t row = 2304U * sizeof(float), tile = mix::kLatencyTile * sizeof(float);
    const auto decoder = mix::transition_stage(2304, 3, true, 96U * 1024U);
    check(decoder.query_rows == 3U && decoder.add_rows == 2U && decoder.bytes == 3U * tile + 5U * row,
          "decoder shape stages every query row at entry (one round trip)");
    const auto raw_only = mix::transition_stage(2304, 1, false, 96U * 1024U);
    check(raw_only.add_rows == 0U && raw_only.bytes == tile + row, "no prologue: no add rows");
    check(mix::transition_stage(0, 3, true, std::size_t{1} << 20U).query_rows == 0U &&
              mix::transition_stage(4097, 3, true, std::size_t{1} << 20U).query_rows == 0U &&
              mix::transition_stage(2304, 0, true, std::size_t{1} << 20U).query_rows == 0U &&
              mix::transition_stage(2304, 9, true, std::size_t{1} << 20U).query_rows == 0U &&
              mix::transition_stage(2304, 3, true, 3U * tile + 3U * row - 1U).query_rows == 0U,
          "stage refuses empty or oversized shapes and a budget below one query row");
    check(mix::transition_stage(2304, 3, true, 3U * tile + 3U * row).query_rows == 1U &&
              mix::transition_stage(2304, 8, false, 8U * tile + 2U * row + 7U).query_rows == 2U,
          "a tight budget stages the queries in batches");
    bool ok = true;
    constexpr std::size_t budget = 94U * 1024U;
    for (unsigned width = 1; width <= mix::kLatencyMaxWidth; width += 7U)
        for (unsigned queries = 1; queries <= 8U; ++queries)
            for (bool prologue : {false, true}) {
                const auto s = mix::transition_stage(width, queries, prologue, budget);
                const std::size_t w = std::size_t{width} * sizeof(float);
                ok = ok && s.query_rows >= 1U && s.query_rows <= queries && s.bytes <= budget &&
                     s.bytes == std::size_t{queries} * tile + std::size_t{s.add_rows + s.query_rows} * w &&
                     (s.query_rows == queries || s.bytes + w > budget);
            }
    check(ok, "every admitted shape fits the budget, with the largest ring that fits");
}

void flush_rules() {
    const decode::R1Forms on{.transition = true};
    {
        // An add whose output is not the readout's current bank stays a separate leaf, in order.
        Recorder r;
        decode::TransitionChain fuser(on);
        fuser.add({residual_row, hidden, readout[2], kWidth, false}, r.emit());
        fuser.readout(read(2, hidden, 1), r.emit());
        fuser.flush(r.emit());
        check(r.nodes.size() == 2U && r.nodes[0].identity == kAdd && r.nodes[1].identity == kK8c,
              "unrelated add: add then incumbent readout");
    }
    {
        // A readout alone gains nothing: the incumbent (tiled when selected) leaf.
        Recorder r;
        decode::TransitionChain fuser({.bank_tiled = true, .transition = true});
        fuser.readout(read(2, hidden, 1), r.emit());
        fuser.flush(r.emit());
        check(r.nodes.size() == 1U && r.nodes[0].identity == kTiled, "bare readout keeps the tiled leaf");
    }
    {
        // A norm of another row flushes the held readout first.
        Recorder r;
        decode::TransitionChain fuser(on);
        fuser.readout(read(2, hidden, 1), r.emit());
        fuser.norm({hidden, gains[0], inputs[0], 1U, kWidth, 1e-6F}, r.emit());
        check(r.nodes.size() == 2U && r.nodes[0].identity == kK8c && r.nodes[1].identity == kRms, "unrelated norm");
    }
    {
        // In-place norm of a raw row (output overlaps the raw row): refused by fits, incumbent leaves.
        Recorder r;
        decode::TransitionChain fuser(on);
        fuser.add({residual_row, hidden, hidden, kWidth, false}, r.emit());
        fuser.readout(read(2, hidden, 1), r.emit());
        fuser.norm({readout[0], gains[0], readout[0], 1U, kWidth, 1e-6F}, r.emit());
        fuser.flush(r.emit());
        check(r.nodes.size() == 2U && r.nodes[0].identity == kTransition && r.nodes[1].identity == kRms,
              "in-place norm stays a separate leaf after the fused add + readout");
    }
    {
        // Norms with different epsilons cannot share one transition.
        Recorder r;
        decode::TransitionChain fuser(on);
        fuser.readout(read(2, hidden, 2), r.emit());
        fuser.norm(rms(0), r.emit());
        auto other = rms(1);
        other.epsilon = 1e-5F;
        fuser.norm(other, r.emit());
        check(r.nodes.size() == 2U && r.nodes[0].identity == kTransition && r.nodes[1].identity == kRms,
              "second epsilon flushes");
    }
    {
        // Off: every call emits its incumbent leaf at once.
        Recorder r;
        decode::TransitionChain fuser{decode::R1Forms{}};
        fuser.add({residual_row, readout[0], hidden, kWidth, true}, r.emit());
        check(r.nodes.size() == 1U && fuser.idle(), "off: add emitted at once");
        fuser.readout(read(1, hidden, 1), r.emit());
        fuser.norm(rms(0), r.emit());
        check(r.nodes.size() == 3U && r.nodes[1].identity == kK8c && r.nodes[2].identity == kRms, "off: incumbents");
    }
}

void legality() {
    residual::TransitionArgs t{};
    t.mix = read(3, hidden, 2);
    t.add = {residual_row, readout[0], hidden, kWidth, true};
    t.norm_epsilon = 1e-6F;
    t.norm[0] = {gains[0], inputs[0]};
    check(residual::fits(t), "decoder transition fits");
    auto wrong = t;
    wrong.add.output = residual_row;
    check(!residual::fits(wrong), "add output must be the current bank");
    wrong = t;
    wrong.add.elements = kWidth - 1U;
    check(!residual::fits(wrong), "add covers the row");
    wrong = t;
    wrong.norm[0].output = readout[1];
    check(!residual::fits(wrong), "norm row disjoint from raw rows");
    wrong = t;
    wrong.norm[0].output = blocks[1];
    check(!residual::fits(wrong), "norm row disjoint from banks");
    wrong = t;
    wrong.norm[2] = {gains[2], inputs[2]};
    check(!residual::fits(wrong), "no norm past the query count");
    wrong = t;
    wrong.norm_epsilon = 0.0F;
    check(!residual::fits(wrong), "norm epsilon positive");
    wrong = t;
    static std::int8_t codes[kWidth];
    static float deltas[kWidth / 32U];
    wrong.norm[0].codes = codes;
    check(!residual::fits(wrong), "a8g32 image needs deltas");
    wrong.norm[0].deltas = deltas;
    check(residual::fits(wrong), "a8g32 image with codes and deltas");
    wrong.mix.width = 1000U;
    wrong.add.elements = 1000U;
    check(!residual::fits(wrong), "a8g32 image needs width % 32");
    wrong = t;
    wrong.mix.width = 4097U;
    wrong.add.elements = 4097U;
    check(!residual::fits(wrong), "row owner width <= 4096");
    wrong = t;
    wrong.mix.output[1] = residual_row;
    wrong.add.lhs = residual_row;
    check(residual::fits(wrong), "add operands may alias written rows (read before any store)");
    wrong = t;
    wrong.mix.banks[0] = hidden;
    check(!residual::fits(wrong), "no other bank may overlap the add output");
}

}  // namespace

int main() {
    program_chain();
    flush_rules();
    legality();
    cluster_plan();
    stage_plan();
    if (failures != 0) {
        std::fprintf(stderr, "FAIL transition host: %d checks\n", failures);
        return 1;
    }
    std::puts("PASS transition host: fuser emission order, incumbent operands, flush rules and TransitionArgs legality");
    return 0;
}
