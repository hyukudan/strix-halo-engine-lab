#include "strix/ernie/checkpoint.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>
#include <map>

using namespace strix::ernie;

namespace {

void print_total(std::string_view label, const PayloadTotal& total) {
    std::cerr << label << " tensors=" << total.tensors << " bytes=" << total.bytes << '\n';
}

} // namespace

int main(int argc, char** argv) try {
    bool list = false;
    unsigned cut = (kLayerCount + 1U) / 2U;
    if (argc < 2) {
        std::cerr << "usage: strix-ernie-engine-plan CHECKPOINT_DIR [--list] [--cut N]\n";
        return 2;
    }
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--list") {
            list = true;
        } else if (arg == "--cut" && i + 1 < argc) {
            const std::string_view value(argv[++i]);
            const auto end = value.data() + value.size();
            const auto parsed = std::from_chars(value.data(), end, cut);
            if (parsed.ec != std::errc{} || parsed.ptr != end || cut == 0 || cut >= kLayerCount)
                throw std::invalid_argument("invalid layer cut");
        } else {
            throw std::invalid_argument("unknown or incomplete engine-plan argument");
        }
    }

    Inventory inventory;
    if (const auto status = inspect_weights(argv[1], inventory); !status) throw std::runtime_error(status.detail);
    const CheckpointWeights weights(std::move(inventory));
    const auto& found = weights.inventory();
    std::map<TensorConsumer, PayloadTotal> totals;
    std::cout << "tensor\tconsumer_or_pending_reason\tshard\tdtype\tshape\toffset\tbytes\n";
    for (const auto& [name, tensor] : found.tensors) {
        const auto consumer = tensor_consumer(name, tensor);
        auto& total = totals[consumer];
        ++total.tensors;
        total.bytes += tensor.payload.range.bytes;
        if (!list && consumer == TensorConsumer::first_row_reference) continue;
        std::cout << name << '\t' << consumer_name(consumer) << '\t' << tensor.payload.source->shard_name()
                  << '\t' << tensor.wire_dtype << '\t';
        for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
            if (axis != 0) std::cout << ',';
            std::cout << tensor.shape[axis];
        }
        std::cout << '\t' << tensor.payload.range.offset << '\t' << tensor.payload.range.bytes << '\n';
    }
    for (const auto& [consumer, total] : totals) print_total(consumer_name(consumer), total);
    for (const auto& shard : found.missing_shards) std::cerr << "missing_shard=" << shard << '\n';
    std::cerr << "tensors=" << found.tensors.size() << " manifest=" << found.has_manifest
              << " missing_known_tensors=" << found.missing_known_tensors << '\n';
    if (!found.has_manifest || !found.missing_shards.empty() || found.missing_known_tensors != 0 ||
        totals.contains(TensorConsumer::unknown)) return 1;

    const auto placement = pair_payload(found, cut);
    std::cerr << "layer_cut=" << cut << '\n';
    print_total("target_rank0", placement.rank0);
    print_total("target_rank1", placement.rank1);
    print_total("draft_deferred", placement.draft);

    const auto probe = target_layer_prefix(0) + "self_attn.core_attention.attn_sink";
    const auto& tensor = found.tensors.at(probe);
    if (tensor.wire_dtype != "F32" || tensor.payload.range.bytes > 4096U)
        throw std::invalid_argument("engine-plan probe must be a small F32 tensor");
    std::vector<float> sink(static_cast<std::size_t>(tensor.payload.range.bytes / sizeof(float)));
    weights.read(probe, sink);
    if (!std::ranges::all_of(sink, [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("engine-plan probe has nonfinite values");
    const auto bounds = std::ranges::minmax_element(sink);
    std::cerr << "probe=" << probe << " bytes=" << tensor.payload.range.bytes
              << " min=" << *bounds.min << " max=" << *bounds.max << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
