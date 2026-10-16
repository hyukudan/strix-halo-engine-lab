#include "strix/moe_q4k_a8_runtime.hpp"
#include <array>
#include <limits>
#include <stdexcept>

namespace strix::moe_q4k_a8 {
namespace {
std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::overflow_error("Q4K A8 workspace multiplication overflow");
    return a * b;
}
std::uint64_t align256(std::uint64_t a) {
    if (a > std::numeric_limits<std::uint64_t>::max() - 255)
        throw std::overflow_error("Q4K A8 workspace alignment overflow");
    return (a + 255) & ~std::uint64_t{255};
}
std::uint64_t weight_bytes(std::uint32_t type, std::uint64_t input,
    std::uint64_t output, std::uint64_t experts) {
    std::uint64_t block = 1, bytes = 0;
    switch (type) {
    case 0: bytes = 4; break;
    case 1: case 30: bytes = 2; break;
    case 7: block = 32; bytes = 24; break;
    case 8: block = 32; bytes = 34; break;
    case 12: block = 256; bytes = 144; break;
    case 13: block = 256; bytes = 176; break;
    case 20: block = 32; bytes = 18; break;
    case 23: block = 256; bytes = 136; break;
    default: throw std::invalid_argument("Q4K A8 runtime unsupported down type");
    }
    if (input % block != 0) throw std::invalid_argument("Q4K A8 weight block width mismatch");
    return mul(mul(mul(input / block, bytes), output), experts);
}
void check_tensor(const ManifestTensor& t, std::uint64_t input,
    std::uint64_t output, std::uint64_t experts) {
    if (t.dimensions != std::vector<std::uint64_t>{input, output, experts} ||
        t.byte_size != weight_bytes(t.ggml_type, input, output, experts) ||
        t.absolute_offset % 16 != 0 ||
        t.byte_size > std::numeric_limits<std::uint64_t>::max() - t.absolute_offset)
        throw std::invalid_argument("Q4K A8 runtime tensor shape, bytes or alignment mismatch");
}
}  // namespace
RuntimeWorkspace build_runtime_workspace(std::uint64_t hidden, std::uint64_t rows,
    std::uint64_t experts, std::uint64_t top_k) {
    return build_runtime_workspace(hidden, rows, experts, top_k, false);
}
RuntimeWorkspace build_runtime_workspace(std::uint64_t hidden, std::uint64_t rows,
    std::uint64_t experts, std::uint64_t top_k, bool q51_slot_rawdots) {
    if (hidden < 256 || hidden > 2560 || hidden % 256 != 0 ||
        rows < 32 || rows > 640 || rows % 32 != 0 ||
        experts == 0 || experts > 512 || top_k == 0 || top_k > 10 || top_k > experts)
        throw std::invalid_argument("Q4K A8 runtime unsupported H/I/E/K geometry");
    RuntimeWorkspace l;
    l.gate_up_bytes = mul(mul(mul(top_k, 2), rows), 4);
    l.accumulation_offset = align256(l.gate_up_bytes);
    l.accumulation_bytes = mul(hidden, 4);
    l.status_offset = l.accumulation_offset + l.accumulation_bytes;
    l.codes_offset = align256(l.status_offset + 4);
    l.codes_bytes = hidden;
    l.deltas_offset = align256(l.codes_offset + l.codes_bytes);
    l.deltas_bytes = mul(hidden / 32, 4);
    l.sums_offset = align256(l.deltas_offset + l.deltas_bytes);
    l.sums_bytes = mul(hidden / 32, 4);
    l.total_bytes = align256(l.sums_offset + l.sums_bytes);
    if (q51_slot_rawdots) {
        l.rawdots_offset = l.total_bytes;
        l.rawdots_bytes = mul(mul(top_k, hidden), 4U);
        if (l.rawdots_bytes > std::numeric_limits<std::uint64_t>::max() - l.rawdots_offset)
            throw std::overflow_error("Q5_1 raw-dot workspace addition overflow");
        l.total_bytes = align256(l.rawdots_offset + l.rawdots_bytes);
    }
    return l;
}
RuntimeLayerClassification classify_runtime_manifest(
    const ModelContract& model, const ModelManifest& manifest) {
    if (const auto error = model.validate(); !error.empty()) throw std::invalid_argument(error);
    (void)build_runtime_workspace(model.hidden_size, model.moe_intermediate_size,
        model.num_experts, model.num_experts_per_token);
    std::vector<std::array<const ManifestTensor*, 3>> layers(model.hidden_layers);
    for (const auto& tensor : manifest.tensors) {
        if (tensor.role != TensorRole::moe_expert) continue;
        if (!tensor.layer || *tensor.layer >= model.hidden_layers)
            throw std::invalid_argument("Q4K A8 routed tensor has invalid layer");
        const auto& name = tensor.name;
        unsigned slot = 3;
        if (name.find("gate_up") != std::string::npos)
            throw std::invalid_argument("Q4K A8 runtime rejects fused gate/up");
        if (name.find("gate") != std::string::npos) slot = 0;
        else if (name.find("up") != std::string::npos) slot = 1;
        else if (name.find("down") != std::string::npos) slot = 2;
        if (slot == 3 || layers[*tensor.layer][slot] != nullptr)
            throw std::invalid_argument("Q4K A8 routed tensor family is ambiguous");
        layers[*tensor.layer][slot] = &tensor;
    }
    RuntimeLayerClassification result;
    for (std::uint32_t layer = 0; layer < model.hidden_layers; ++layer) {
        const auto& ts = layers[layer];
        if (!ts[0] || !ts[1] || !ts[2])
            throw std::invalid_argument("Q4K A8 routed tensor family is missing");
        if (ts[0]->ggml_type != ts[1]->ggml_type ||
            (ts[0]->ggml_type != 12 && ts[0]->ggml_type != 13))
            throw std::invalid_argument("Q4K A8 requires paired Q4K or Q5K gate/up");
        check_tensor(*ts[0], model.hidden_size, model.moe_intermediate_size, model.num_experts);
        check_tensor(*ts[1], model.hidden_size, model.moe_intermediate_size, model.num_experts);
        check_tensor(*ts[2], model.moe_intermediate_size, model.hidden_size, model.num_experts);
        (ts[0]->ggml_type == 12 ? result.quantized_layers : result.legacy_q5k_layers).push_back(layer);
        if(ts[0]->ggml_type==12 && ts[2]->ggml_type==7)
            result.q4k_with_q51_down_layers.push_back(layer);
    }
    if (result.quantized_layers.empty()) throw std::invalid_argument("Q4K A8 requires a Q4K routed layer");
    return result;
}
void validate_slot_parallel_manifest(const ModelContract& model, const RuntimeLayerClassification& classification) {
    if (!model.validate().empty() || model.hidden_layers != 48U || model.hidden_size != 2560U ||
        model.moe_intermediate_size != 640U || model.num_experts != 512U ||
        model.num_experts_per_token != 10U || model.bytes_fp32 != 4U)
        throw std::invalid_argument("Q5_1 slot-parallel down requires canonical 48-layer geometry");
    std::vector<std::uint32_t> expected;
    for (std::uint32_t layer = 0; layer < 48U; ++layer)
        if (layer != 2U && layer != 4U && layer != 30U && layer != 46U && layer != 47U)
            expected.push_back(layer);
    if (classification.q4k_with_q51_down_layers != expected)
        throw std::invalid_argument("Q5_1 slot-parallel down requires the complete canonical eligible layer set");
}
}  // namespace strix::moe_q4k_a8
