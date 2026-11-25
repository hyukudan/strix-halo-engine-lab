// Host only (no GPU, no model file): the generated staged checkpoint of the CUDA operand-class fixture passes everything
// the resident upload validates on the host: the safetensors index and text schema bind, the resident plan, and for every
// physical layer and expert projection the sidecar checks of the expert bind (finite positive NVFP4 denominator, finite
// non-negative per-expert amax); and every attention tensor and layer-norm gain reads finite and bounded.
#include "mini_staged_source.hpp"

#include "core/weight_certificate.hpp"
#include "inkling2/inkling2_mini_attention_operands.hpp"
#include "inkling2/inkling2_mini_resident_weights.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

namespace wei = strix::weight;
namespace mini = strix::inkling2;

bool run() {
    const auto source = std::make_shared<mini_staged::StagedSource>();
    wei::TensorIndex index;
    if (const auto status = wei::index_safetensors(source, index); !status) {
        std::cerr << "index: " << status.detail << '\n';
        return false;
    }
    mini::MiniTextWeights text;
    if (const auto status = text.bind(std::move(index)); !status) {
        std::cerr << "bind: " << status.detail << '\n';
        return false;
    }
    (void)mini::MiniResidentWeightPlan::make(text);
    std::size_t slabs = 0;
    for (unsigned layer = 0; layer < mini::kPhysicalLayers; ++layer)
        for (const auto projection : {mini::MiniExpertProjection::gate_up, mini::MiniExpertProjection::down}) {
            mini::MiniExpertLayerView view;
            if (const auto status = text.expert_layer(layer, projection, view); !status) {
                std::cerr << "expert layer " << layer << ": " << status.detail << '\n';
                return false;
            }
            // The resident bind reads the sidecars from the source and checks them; it forms one pointer per expert into
            // the packed spans, so they are backed by real storage of the complete slab (one slab at a time, never read:
            // left uninitialised so the pages are not touched).
            const auto code_bytes = static_cast<std::size_t>(view.codes.range.bytes);
            const auto scale_bytes = static_cast<std::size_t>(view.block_scales.range.bytes);
            const auto codes = std::make_unique_for_overwrite<std::uint8_t[]>(code_bytes);
            const auto scales = std::make_unique_for_overwrite<std::uint8_t[]>(scale_bytes);
            std::vector<wei::microscaled::PackedMatrix> out(view.experts);
            const auto status = view.bind_storage({codes.get(), code_bytes}, {scales.get(), scale_bytes}, out);
            if (!status) {
                std::cerr << "expert bind layer " << layer << ": " << status.detail << '\n';
                return false;
            }
            // Every expert's pointers are the slab base plus its own stride: the last expert ends at the slab end.
            const auto& last = out.back();
            if (last.codes != codes.get() + code_bytes - code_bytes / view.experts ||
                last.block_scales != scales.get() + scale_bytes - scale_bytes / view.experts) {
                std::cerr << "expert bind layer " << layer << ": expert pointers leave the slab\n";
                return false;
            }
            ++slabs;
        }
    // Attention tensors and gains: scanned through the shared bounded scan exactly as the staged certificate does, on the
    // source's own bytes.
    strix::core::Bf16StagedScan scan;
    const auto certificate = mini::certify_mini_attention(
        scan,
        [&](std::string_view name) -> std::span<const std::uint16_t> {
            const auto* tensor = text.find(std::string(mini::kMiniTextTensorPrefix) + std::string(name));
            static std::vector<std::vector<std::uint16_t>> keep;
            keep.emplace_back(static_cast<std::size_t>(tensor->payload.range.bytes / 2U));
            const auto status = source->read(tensor->payload.range, std::as_writable_bytes(std::span(keep.back())));
            if (!status) throw std::runtime_error(status.detail);
            return keep.back();
        },
        [](std::span<const std::uint16_t> staged, std::span<std::uint16_t> host) {
            std::copy(staged.begin(), staged.end(), host.begin());
        });
    const bool certified =
        certificate.route(strix::core::Bf16Operand::bf16_exact) == strix::core::Bf16Operand::bf16_exact;
    std::cout << "staged source: " << slabs << " expert slabs bound, certificate one plane " << certified << std::endl;
    return slabs == 2U * mini::kPhysicalLayers && certified;
}

}  // namespace

int main() {
    try {
        if (!run()) return 1;
    } catch (const std::exception& error) {
        std::cerr << "staged source: exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << "staged source: PASSED" << std::endl;
    return 0;
}
