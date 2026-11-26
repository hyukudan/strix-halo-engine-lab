#include "strix/ernie/pair_plan.hpp"

#include <charconv>
#include <iostream>

namespace {

using namespace strix::ernie;

template<class Integer>
Integer number(std::string_view value) {
    Integer result{};
    const auto* const end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument("Expected a decimal byte or row count: " + std::string(value));
    return result;
}

void print_rank(const PairRank& rank) {
    const auto& m = rank.memory;
    const auto& p = rank.placement;
    std::cout << "rank=" << p.stage << " first=" << p.first() << " last=" << p.last()
              << " embedding=" << p.embeds() << " head=" << p.head()
              << " tensors=" << rank.tensors.size()
              << " weight_payload=" << m.weight_payload << " weight_resident=" << m.weight_resident
              << " kv_payload=" << m.kv_payload << " kv_resident=" << m.kv_resident
              << " carry=" << m.carry << " logits=" << m.logits
              << " exchange=" << m.exchange << " load_staging=" << m.load_staging
              << " scratch=" << m.scratch << " reserve=" << m.reserve
              << " required=" << m.required << " usable=" << m.usable
              << " admitted=" << m.admitted() << '\n';
}

void print_ranges(const PairPlan& plan) {
    std::cout << "rank\ttensor\tshard\tdtype\tshape\tsource_offset\tbytes\tresident_offset\n";
    for (const auto& rank : plan.ranks) {
        for (const auto& tensor : rank.tensors) {
            std::cout << rank.placement.stage << '\t' << tensor.name << '\t' << tensor.shard
                      << '\t' << tensor.wire_dtype << '\t';
            for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
                if (axis) std::cout << ',';
                std::cout << tensor.shape[axis];
            }
            std::cout << '\t' << tensor.source.range.offset << '\t' << tensor.source.range.bytes
                      << '\t' << tensor.resident_offset << '\n';
        }
    }
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 11 && argc != 12) {
        std::cerr << "usage: strix-ernie-pair-plan CHECKPOINT_DIR TOPOLOGY_TOML CONTEXT PREFILL_ROWS "
                     "ALIGNMENT LOAD_CHUNK_BYTES SCRATCH0 SCRATCH1 RESERVE0 RESERVE1 [--ranges]\n";
        return 2;
    }
    const bool ranges = argc == 12 && std::string_view(argv[11]) == "--ranges";
    if (argc == 12 && !ranges) throw std::invalid_argument("Unknown ERNIE pair argument");
    PairLimits limits;
    limits.context = number<std::uint32_t>(argv[3]);
    limits.prefill_rows = number<std::uint32_t>(argv[4]);
    limits.alignment = number<std::uint64_t>(argv[5]);
    limits.load_chunk_bytes = number<std::uint64_t>(argv[6]);
    limits.scratch = {number<std::uint64_t>(argv[7]), number<std::uint64_t>(argv[8])};
    limits.reserve = {number<std::uint64_t>(argv[9]), number<std::uint64_t>(argv[10])};

    const auto topology = strix::device::load_topology(argv[2]);
    Inventory inventory;
    if (const auto status = inspect_weights(argv[1], inventory); !status)
        throw std::runtime_error(status.detail);
    const auto plan = make_pair_plan(inventory, topology, limits, "usb4auto");
    std::cout << "scope=host-planning transport=" << plan.transport << " context=" << limits.context
              << " prefill_rows=" << limits.prefill_rows << " seam_row_bytes=" << kPairRowBytes
              << " draft_deferred=" << plan.draft_bytes << '\n';
    for (const auto& rank : plan.ranks) print_rank(rank);
    if (ranges) print_ranges(plan);
    return plan.admitted() ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
