// Host test of the exact one-row latency forms in the shared CUDA decode recipe: R1Forms{} keeps every
// incumbent leaf, a selected form replaces exactly its incumbent node (same operands, same order) and
// out-of-shape operands fall back to the incumbent. No GPU is used: bind() only records launches.
#include "exec/cuda_decode_lowering.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace decode = strix::exec::cuda_decode;
namespace mix = strix::core::residual_mix;

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
constexpr unsigned kRank = 64U;

alignas(256) float scratch[6U * kWidth];
alignas(256) float raw[kWidth];
alignas(256) float query[kWidth];
alignas(256) float output[kWidth];
alignas(256) float banks[4][kWidth];
alignas(256) std::uint16_t weights[4][16];

std::vector<Node> controller(decode::R1Forms forms) {
    std::vector<Node> nodes;
    const auto add = [&](const auto& leaf, bool pdl) {
        auto bound = decode::bind(leaf);
        nodes.push_back({bound.identity, bound.grid, bound.args, pdl});
    };
    const std::array<std::span<const float>, 3> bank{std::span<const float>(banks[0]), std::span<const float>(banks[1]),
                                                     std::span<const float>(banks[2])};
    decode::lower_low_rank_controller(kWidth, kRank, 1e-6F, query, raw, bank, output, scratch, weights[0], weights[1],
                                      weights[2], weights[3], add, forms);
    return nodes;
}

bool same_grid(const strix::exec::Grid& a, const strix::exec::Grid& b) {
    return a.blocks == b.blocks && a.threads == b.threads && a.shared_bytes == b.shared_bytes;
}

void controller_forms() {
    const auto base = controller({});
    const auto fast = controller({.bank_tiled = true, .output_split = true, .moe_pipe = false});
    check(base.size() == fast.size(), "forms keep the controller node count");
    if (base.size() != fast.size() || base.size() < 3U) return;
    check(base[0].identity == "cuda-sm120-residual-bank-mix-r1-v1", "incumbent bank readout leaf");
    check(base[2].identity == "cuda-sm120-bf16-r1-mma3-group-round-v1", "incumbent controller down projection leaf");
    check(fast[0].identity == "cuda-sm120-residual-bank-softmax-tiled-r1-v1", "tiled bank readout leaf");
    check(fast[0].args == base[0].args, "tiled readout keeps the incumbent MixArgs bytes");
    check(fast[0].grid.blocks[0] == 5U && fast[0].grid.blocks[1] == 1U && fast[0].grid.threads[0] == 128U &&
              fast[0].grid.threads[1] == 3U, "tiled readout grid = 5 tiles x 1 query, 128 x 3 banks");
    check(fast[0].pdl && !base[0].pdl, "tiled readout waits on its producer through PDL");
    check(fast[2].identity == "cuda-sm120-bf16-r1-mma3-output-split-round-v1", "output split controller leaf");
    check(fast[2].grid.blocks[0] == 8U && fast[2].grid.threads[0] == 256U, "output split = 8 CTAs x 256 threads");
    decode::Bf16OutputSplitRound split{};
    std::memcpy(&split, fast[2].args.data(), sizeof(split));
    check(split.input_width == 2U * kWidth && split.outputs == kRank && split.weight == weights[0] &&
              split.output == scratch + 4U * kWidth && split.input == scratch + kWidth,
          "output split keeps the incumbent operands");
    for (std::size_t i = 0; i < base.size(); ++i) {
        if (i == 0U || i == 2U) continue;
        // NormalizeArgs carries padding bytes from its temporaries: compare that node by members.
        bool operands = base[i].args == fast[i].args;
        if (i == 1U && base[i].args.size() == sizeof(mix::NormalizePairArgs) && fast[i].args.size() == base[i].args.size()) {
            mix::NormalizePairArgs a{}, b{};
            std::memcpy(&a, base[i].args.data(), sizeof(a));
            std::memcpy(&b, fast[i].args.data(), sizeof(b));
            const auto eq = [](const mix::NormalizeArgs& x, const mix::NormalizeArgs& y) {
                return x.input == y.input && x.output == y.output && x.rows == y.rows && x.width == y.width &&
                       x.epsilon == y.epsilon && x.input_stride == y.input_stride &&
                       x.output_stride == y.output_stride && x.round_bf16 == y.round_bf16;
            };
            operands = eq(a.first, b.first) && eq(a.second, b.second);
        }
        const bool same = base[i].identity == fast[i].identity && operands &&
                          same_grid(base[i].grid, fast[i].grid) && base[i].pdl == fast[i].pdl;
        if (!same)
            std::fprintf(stderr, "node %zu: %s / %s args %d grid %d pdl %d\n", i, base[i].identity.c_str(),
                         fast[i].identity.c_str(), operands, same_grid(base[i].grid, fast[i].grid),
                         base[i].pdl == fast[i].pdl);
        check(same, "every other controller node is unchanged");
    }
}

void fallbacks() {
    std::vector<std::string> seen;
    const auto add = [&](const auto& leaf, bool) { seen.push_back(decode::bind(leaf).identity); };
    mix::MixArgs wide{};
    wide.queries = query;
    wide.output[0] = output;
    wide.rows = wide.query_count = wide.bank_count = 1U;
    wide.banks[0] = banks[0];
    wide.width = 4608U;  // above the tiled 4096-feature register bound
    wide.normalized = true;
    decode::lower_bank_readout(wide, {.bank_tiled = true}, add);
    check(seen.size() == 1U && seen[0] == "cuda-sm120-residual-bank-mix-r1-v1", "wide readout keeps the incumbent");
    seen.clear();
    const strix::core::Bf16RowProjection unrounded{weights[0], kRank, output, false};  // the split form rounds
    decode::lower_rounded_projection(raw, kWidth, unrounded, {.output_split = true}, add);
    check(seen.size() == 1U && seen[0] == "cuda-sm120-bf16-r1-mma3-group-v1",
          "an unrounded projection keeps lower_dense");
    bool threw = false;
    try {
        (void)decode::bind(decode::BankReadoutTiled{wide});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "bind rejects an out-of-shape tiled readout");
}

void moe_pipe() {
    alignas(256) static strix::moe_core::device_grouped::ExpertWeights matrices[128]{};
    alignas(256) static unsigned ids[8]{};
    alignas(256) static float route[8]{};
    alignas(256) static unsigned failure[1]{};
    alignas(256) static std::uint16_t activation[8U * 1152U]{};
    const strix::moe_core::microscaled_r1::Args args{matrices, ids, route, failure, raw, activation, output,
                                                     128U, kWidth, 1152U, 30.0F, 30.0F};
    using Format = strix::weight::microscaled::MatrixFormat;
    for (const auto format : {Format::nvfp4, Format::mxfp8}) {
        const auto gate = decode::bind(decode::PackedGateUpPipe{args, format});
        const auto gate_base = decode::bind(decode::PackedGateUp{args, format});
        check(gate.grid.blocks[0] == 288U && gate.grid.blocks[1] == 8U && gate.grid.threads[0] == 128U,
              "pipe gate-up grid = 288 x 8 CTAs x 128 threads");
        check(gate.identity.find("-pipe-v1") != std::string::npos && gate.args.size() == gate_base.args.size(),
              "pipe gate-up identity and operand bytes");
        check(gate.entry_cost.cost.weight_bytes == gate_base.entry_cost.cost.weight_bytes,
              "pipe gate-up prices the incumbent weight bytes");
        const auto down = decode::bind(decode::PackedDownCombinePipe{args, format});
        check(down.grid.blocks[0] == 1152U && down.grid.threads[0] == 256U, "pipe down grid = 1152 CTAs x 256 threads");
        check(down.identity.find("down-combine-r1-k8-pipe-v1") != std::string::npos, "pipe down identity");
    }
}

}  // namespace

int main() {
    controller_forms();
    fallbacks();
    moe_pipe();
    if (failures != 0) return 1;
    std::puts("cuda decode R1 forms: incumbent default, tiled bank, output split, MoE pipe bindings PASS");
    return 0;
}
