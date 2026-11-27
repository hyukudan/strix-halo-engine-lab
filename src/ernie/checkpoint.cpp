#include "strix/ernie/checkpoint.hpp"

#include "strix/weight/formats/fp8_block.hpp"
#include "strix/weight/formats/microscaled_matrix.hpp"
#include "strix/weight/typed_slice.hpp"

#include <stdexcept>
#include <utility>

namespace strix::ernie {
namespace {

using Fp8 = weight::BlockScaledE4M3<
    model::narrow<std::size_t>(data::weights::checkpoint::weight_block_size[0]),
    model::narrow<std::size_t>(data::weights::checkpoint::weight_block_size[1]), float>;
constexpr auto kMxfp4Columns = weight::microscaled::kMxfp4BlockColumns;
static_assert(kMxfp4Columns == data::weights::checkpoint::expert_block_columns);

void require(weight::Status status) {
    if (!status) throw std::runtime_error(status.detail);
}

void read_plain(std::string_view name, const Tensor& tensor, std::span<float> out, std::size_t first = 0) {
    weight::RawTensor raw{};
    raw.name = name;
    raw.wire_dtype = tensor.wire_dtype;
    raw.payload = tensor.payload;
    require(weight::read_float_slice(raw, out, first));
}

std::vector<std::uint8_t> read_codes(const Tensor& tensor) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(tensor.payload.range.bytes));
    require(weight::read_slice(tensor.payload, std::as_writable_bytes(std::span(bytes))));
    return bytes;
}

} // namespace

std::string_view consumer_name(TensorConsumer consumer) noexcept {
    using enum TensorConsumer;
    switch (consumer) {
    case first_row_reference: return "first_row_reference";
    case hca_history_candidate: return "hca_history_candidate";
    case mla_layer_candidate: return "mla_layer_candidate";
    case vocabulary_activation: return "vocabulary_activation";
    case draft_pending: return "draft_pending";
    case unknown: return "unknown";
    }
    return "unknown";
}

TensorConsumer tensor_consumer(std::string_view name, const Tensor& tensor) noexcept {
    using enum TensorConsumer;
    if (tensor.module == Module::unclassified) return unknown;
    if (tensor.drafter) return draft_pending;
    if (tensor.layer && layer_is_mla(*tensor.layer)) return mla_layer_candidate;
    switch (tensor.module) {
    case Module::embedding:
    case Module::layer_norm:
    case Module::hyper_connection:
    case Module::hca:
    case Module::attention_gate:
    case Module::value_head_mix:
    case Module::dense_mlp:
    case Module::router:
    case Module::latent_projection:
    case Module::shared_expert:
    case Module::routed_expert:
    case Module::final_norm:
    case Module::hc_contract: return first_row_reference;
    case Module::compressor: return hca_history_candidate;
    case Module::mla:
    case Module::sparse_indexer:
    case Module::flash_indexer: return mla_layer_candidate;
    case Module::lm_head: return name == "lm_head.weight" ? first_row_reference : vocabulary_activation;
    case Module::draft_input:
    case Module::draft_markov:
    case Module::draft_confidence: return draft_pending;
    case Module::unclassified: return unknown;
    }
    return unknown;
}

PairPayload pair_payload(const Inventory& inventory, unsigned cut) {
    if (cut == 0 || cut >= kLayerCount) throw std::invalid_argument("ERNIE: layer cut must leave both ranks nonempty");
    PairPayload plan;
    for (const auto& [name, tensor] : inventory.tensors) {
        if (tensor.module == Module::unclassified)
            throw std::invalid_argument("ERNIE: cannot place unknown tensor " + name);
        const bool first = tensor.layer ? *tensor.layer < cut : tensor.module == Module::embedding;
        auto& total = tensor.drafter ? plan.draft : first ? plan.rank0 : plan.rank1;
        ++total.tensors;
        total.bytes += tensor.payload.range.bytes;
    }
    return plan;
}

CheckpointWeights::CheckpointWeights(Inventory inventory, CheckpointReadPolicy policy)
    : inventory_(std::move(inventory)), policy_(policy) {}

void CheckpointWeights::read(std::string_view name, std::span<float> out, std::size_t first_element,
                             const RowExecutor& execute) const {
    const auto it = inventory_.tensors.find(name);
    if (it == inventory_.tensors.end()) throw std::invalid_argument("ERNIE: missing tensor " + std::string(name));
    const auto& tensor = it->second;
    if (tensor.drafter || tensor.module == Module::unclassified)
        throw std::invalid_argument("ERNIE: no target consumer for " + std::string(name));
    if (weight::slice_dtype(tensor.wire_dtype) != weight::SliceDtype::bytes) {
        read_plain(name, tensor, out, first_element);
        return;
    }
    const bool fp8 = tensor.wire_dtype == "F8_E4M3";
    if (!fp8 && tensor.wire_dtype != "I8")
        throw std::invalid_argument("ERNIE: expected a matrix code plane, got " + std::string(name));
    if (first_element != 0 || tensor.shape.size() != 2 || tensor.shape[0] == 0 || tensor.shape[1] == 0)
        throw std::invalid_argument("ERNIE: expected a complete quantized matrix");
    const auto rows = static_cast<std::size_t>(tensor.shape[0]);
    const auto packed_width = static_cast<std::size_t>(tensor.shape[1]);
    const auto columns = fp8 ? packed_width : packed_width * 2U;
    if (out.size() / columns != rows || out.size() % columns != 0)
        throw std::invalid_argument("ERNIE: output does not match logical matrix shape");
    const auto scale_name = inverse_scale_name(name);
    const auto& scale = inventory_.tensors.at(scale_name);
    const auto scale_rows = fp8 ? Fp8::scale_rows(rows) : rows;
    const auto scale_columns = fp8 ? Fp8::scale_columns(columns) : columns / kMxfp4Columns;
    if (scale.shape != std::vector<std::uint64_t>{scale_rows, scale_columns} ||
        scale.wire_dtype != (fp8 ? "F32" : "F8_E8M0") || (!fp8 && columns % kMxfp4Columns != 0))
        throw std::invalid_argument("ERNIE: invalid scale plane for " + std::string(name));

    const auto codes = read_codes(tensor);
    if (codes.size() / packed_width != rows || codes.size() % packed_width != 0)
        throw std::invalid_argument("ERNIE: code slice does not match matrix shape");
    const auto schedule = [&](std::size_t count, const auto& work) {
        if (execute) execute(count, work);
        else work(std::size_t{0}, count);
    };
    if (fp8) {
        std::vector<float> scales(scale_rows * scale_columns);
        read_plain(scale_name, scale, scales);
        weight::microscaled::dequantize_fp8_block_matrix<Fp8>(codes, scales, rows, columns, out, schedule);
    } else {
        const auto scales = read_codes(scale);
        if (scales.size() / scale_columns != rows || scales.size() % scale_columns != 0)
            throw std::invalid_argument("ERNIE: scale slice does not match MXFP4 matrix shape");
        weight::microscaled::dequantize_mxfp4_matrix(
            codes, scales, rows, columns, out, {policy_.table_mxfp4, policy_.high_nibble_first}, schedule);
    }
}

} // namespace strix::ernie
