#include "strix/ernie/inventory.hpp"
#include "strix/weight/formats/fp8_block.hpp"
#include "strix/weight/formats/microscaled.hpp"
#include "strix/weight/typed_slice.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace strix::ernie;
using namespace strix::weight;

void require(Status status) {
    if (!status) throw std::runtime_error(status.detail);
}

std::vector<std::uint8_t> read_codes(const Tensor& tensor) {
    if (tensor.payload.range.bytes > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("probe tensor exceeds addressable storage");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.payload.range.bytes));
    require(read_slice(tensor.payload, std::as_writable_bytes(std::span(bytes))));
    return bytes;
}

std::vector<float> read_plain(const Tensor& tensor) {
    RawTensor raw;
    raw.wire_dtype = tensor.wire_dtype;
    raw.payload = tensor.payload;
    std::uint64_t count = 1U;
    for (const auto dim : tensor.shape) count *= dim;
    if (count > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("probe tensor exceeds addressable storage");
    std::vector<float> values(static_cast<std::size_t>(count));
    require(read_float_slice(raw, values));
    return values;
}

struct Stats {
    std::uint64_t count{}, zeros{}, positive{}, negative{};
    float min{std::numeric_limits<float>::infinity()}, max{-std::numeric_limits<float>::infinity()};
    double sum{}, squares{};

    void add(std::span<const float> values) {
        for (const auto x : values) {
            if (!std::isfinite(x)) throw std::runtime_error("probe contains a nonfinite decoded value");
            ++count;
            zeros += x == 0.0F;
            positive += x > 0.0F;
            negative += x < 0.0F;
            min = std::min(min, x);
            max = std::max(max, x);
            sum += x;
            squares += static_cast<double>(x) * x;
        }
    }

    void print(std::string_view name, std::uint64_t endpoints = 0) const {
        const auto n = static_cast<double>(count);
        std::cout << name << " n=" << count << " min=" << min << " max=" << max
                  << " mean=" << sum / n << " rms=" << std::sqrt(squares / n)
                  << " zero=" << zeros << " pos=" << positive << " neg=" << negative
                  << " endpoint_codes=" << endpoints << '\n';
    }
};

void scale_report(std::vector<float> scales) {
    if (scales.empty() || !std::ranges::all_of(scales, [](float s) { return std::isfinite(s) && s > 0.0F; }))
        throw std::runtime_error("probe contains an invalid block scale");
    std::ranges::sort(scales);
    std::cout << "scales n=" << scales.size() << " min=" << scales.front()
              << " p10=" << scales[scales.size() / 10U]
              << " median=" << scales[scales.size() / 2U]
              << " p90=" << scales[scales.size() * 9U / 10U]
              << " max=" << scales.back() << '\n';
}

void probe_expert(const Inventory& inventory) {
    constexpr auto name = "model.layers.1.block_sparse_moe.experts.0.w1.weight";
    const auto& weight = inventory.tensors.at(name);
    const auto& scale = inventory.tensors.at("model.layers.1.block_sparse_moe.experts.0.w1.weight_scale_inv");
    const auto codes = read_codes(weight);
    const auto scales = read_codes(scale);
    const auto rows = static_cast<std::size_t>(weight.shape.at(0));
    const auto packed_columns = static_cast<std::size_t>(weight.shape.at(1));
    const auto block_count = packed_columns * 2U / microscaled::kMxfp4BlockColumns;
    if (scale.shape != std::vector<std::uint64_t>{rows, block_count})
        throw std::runtime_error("MXFP4 probe has unexpected scale geometry");
    std::vector<float> decoded(packed_columns * 2U), expanded;
    expanded.reserve(scales.size());
    for (const auto s : scales) expanded.push_back(scalar_float::e8m0(s));
    Stats stats;
    unsigned max_endpoints_in_block = 0U;
    std::uint64_t endpoints = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto packed = std::span(codes).subspan(row * packed_columns, packed_columns);
        microscaled::decode_mxfp4_row(packed, std::span(scales).subspan(row * block_count, block_count), decoded);
        stats.add(decoded);
        for (std::size_t first = 0; first < packed.size(); first += microscaled::kMxfp4BlockColumns / 2U) {
            unsigned hits = 0;
            for (const auto byte : packed.subspan(first, microscaled::kMxfp4BlockColumns / 2U)) {
                hits += (byte & 7U) == 7U;
                hits += ((byte >> 4U) & 7U) == 7U;
            }
            endpoints += hits;
            max_endpoints_in_block = std::max(max_endpoints_in_block, hits);
        }
    }
    stats.print(name, endpoints);
    scale_report(std::move(expanded));
    std::cout << "maximum_endpoint_codes_per_32_weights=" << max_endpoints_in_block << '\n';
}

void probe_fp8(const Inventory& inventory, const char* name, const char* scale_name) {
    using Decoder = BlockScaledE4M3<128, 128, float>;
    const auto& weight = inventory.tensors.at(name);
    const auto& scale = inventory.tensors.at(scale_name);
    const auto rows = static_cast<std::size_t>(weight.shape.at(0));
    const auto columns = static_cast<std::size_t>(weight.shape.at(1));
    if (scale.shape != std::vector<std::uint64_t>{Decoder::scale_rows(rows), Decoder::scale_columns(columns)})
        throw std::runtime_error("FP8 probe has unexpected scale geometry");
    const auto codes = read_codes(weight);
    auto scales = read_plain(scale);
    const Decoder matrix(codes, scales, rows, columns);
    std::vector<float> decoded(columns);
    Stats stats;
    for (std::size_t row = 0; row < rows; ++row) {
        matrix.decode_row(row, decoded);
        stats.add(decoded);
    }
    const auto endpoints = std::ranges::count_if(codes, [](std::uint8_t c) { return (c & 127U) == 126U; });
    stats.print(name, static_cast<std::uint64_t>(endpoints));
    scale_report(std::move(scales));
}

void probe(const Inventory& inventory) {
    probe_expert(inventory);
    probe_fp8(inventory, "model.layers.7.self_attn.kv_a_proj_with_mqa.weight",
              "model.layers.7.self_attn.kv_a_proj_with_mqa.weight_scale_inv");
    probe_fp8(inventory, "model.layers.7.self_attn.core_attention.indexer.wq_b.weight",
              "model.layers.7.self_attn.core_attention.indexer.wq_b.weight_scale_inv");
    for (const auto& [name, tensor] : inventory.tensors) {
        if (tensor.module != Module::flash_indexer) continue;
        Stats stats;
        stats.add(read_plain(tensor));
        stats.print(name);
    }
}

struct Total {
    std::size_t tensors{};
    std::uint64_t bytes{};
    void add(const Tensor& tensor) { ++tensors; bytes += tensor.payload.range.bytes; }
};

void print_location(std::string_view name, const Tensor& tensor) {
    std::cout << name << '\t' << module_name(tensor.module) << '\t' << (tensor.drafter ? "draft" : "target") << '\t';
    if (tensor.layer) std::cout << *tensor.layer;
    std::cout << '\t' << tensor.payload.source->shard_name() << '\t' << tensor.wire_dtype << '\t';
    for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
        if (axis != 0) std::cout << ',';
        std::cout << tensor.shape[axis];
    }
    std::cout << '\t' << tensor.payload.range.offset << '\t' << tensor.payload.range.bytes << '\n';
}
} // namespace

int main(int argc, char** argv) try {
    const auto option = argc == 3 ? std::string_view(argv[2]) : std::string_view{};
    const bool list = option == "--list";
    const bool run_probe = option == "--probe";
    if (argc != 2 && !list && !run_probe) {
        std::cerr << "usage: strix-ernie-inventory CHECKPOINT_DIR [--list|--probe]\n";
        return 2;
    }
    Inventory inventory;
    require(inspect_weights(argv[1], inventory));
    using Key = std::pair<std::string_view, std::string_view>;
    std::map<Key, Total> modules, dtypes;
    Total all;
    std::size_t unclassified = 0;
    if (list) std::cout << "tensor\tmodule\tstack\tlayer\tshard\tdtype\tshape\tbyte_offset\tbytes\n";
    for (const auto& [name, tensor] : inventory.tensors) {
        if (tensor.module == Module::unclassified) {
            std::cerr << "unclassified_tensor=" << name << '\n';
            ++unclassified;
        }
        const auto stack = tensor.drafter ? "draft" : "target";
        modules[{stack, module_name(tensor.module)}].add(tensor);
        dtypes[{stack, tensor.wire_dtype}].add(tensor);
        all.add(tensor);
        if (list) print_location(name, tensor);
    }
    for (const auto& [key, total] : modules)
        std::cerr << "module " << key.first << ' ' << key.second << ' ' << total.tensors << ' ' << total.bytes << '\n';
    for (const auto& [key, total] : dtypes)
        std::cerr << "dtype " << key.first << ' ' << key.second << ' ' << total.tensors << ' ' << total.bytes << '\n';
    for (const auto& name : inventory.missing_shards) std::cerr << "missing_shard=" << name << '\n';
    std::cerr << "tensors=" << all.tensors << " payload_bytes=" << all.bytes
              << " manifest=" << inventory.has_manifest << " missing_shards=" << inventory.missing_shards.size()
              << " missing_known_tensors=" << inventory.missing_known_tensors
              << " unclassified=" << unclassified << '\n';
    const bool complete = inventory.has_manifest && inventory.missing_shards.empty() &&
                          inventory.missing_known_tensors == 0 && unclassified == 0;
    if (run_probe && complete) probe(inventory);
    return complete || (!run_probe && !inventory.missing_shards.empty()) ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
