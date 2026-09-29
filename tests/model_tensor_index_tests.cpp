#include "strix/model_tensor_index.hpp"

#include <iostream>
#include <stdexcept>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

strix::ModelManifest fixture() {
    strix::ModelManifest manifest;
    manifest.architecture = "fixture";

    auto add =
        [&](std::string name,
            strix::TensorRole role,
            std::optional<std::uint32_t> layer,
            strix::PlacementKind placement) mutable {
            strix::ManifestTensor tensor;
            tensor.name = std::move(name);
            tensor.dimensions = {64, 64};
            tensor.ggml_type = 1;
            tensor.byte_size = 8192;
            tensor.absolute_offset =
                4096 +
                manifest.tensors.size() * 8192;
            tensor.role = role;
            tensor.layer = layer;
            tensor.placement.kind = placement;
            tensor.placement.node =
                placement ==
                        strix::PlacementKind::local
                    ? 0
                    : 1;
            manifest.total_tensor_bytes +=
                tensor.byte_size;
            manifest.tensors.push_back(
                std::move(tensor));
        };

    add(
        "token_embd.weight",
        strix::TensorRole::embedding,
        std::nullopt,
        strix::PlacementKind::local);
    add(
        "blk.0.attn_q.weight",
        strix::TensorRole::attention_projection,
        0,
        strix::PlacementKind::local);
    add(
        "blk.0.attn_k.weight",
        strix::TensorRole::attention_projection,
        0,
        strix::PlacementKind::local);
    add(
        "blk.1.attn_q.weight",
        strix::TensorRole::attention_projection,
        1,
        strix::PlacementKind::remote);

    manifest.layout_identity =
        strix::model_manifest_layout_identity(
            manifest);
    return manifest;
}

}  // namespace

int main() {
    const auto manifest = fixture();
    const auto index =
        strix::build_model_tensor_index(
            manifest);

    const auto* embedding =
        strix::find_manifest_tensor(
            manifest,
            index,
            "token_embd.weight");
    expect(embedding != nullptr &&
               embedding->role ==
                   strix::TensorRole::embedding,
           "exact-name lookup resolves tensor without hot-path string scan");

    strix::ModelTensorSelector layer_zero;
    layer_zero.role =
        strix::TensorRole::attention_projection;
    layer_zero.layer = 0;
    const auto attention =
        strix::select_manifest_tensors(
            manifest,
            index,
            layer_zero);
    expect(attention.size() == 2,
           "role/layer selector returns all matching projections");

    auto q_selector = layer_zero;
    q_selector.name_contains = "attn_q";
    const auto q0 =
        strix::select_unique_manifest_tensor(
            manifest,
            index,
            q_selector);
    expect(manifest.tensors[q0].name ==
               "blk.0.attn_q.weight",
           "name fragment disambiguates a layer projection");

    strix::ModelTensorSelector remote_q;
    remote_q.role =
        strix::TensorRole::attention_projection;
    remote_q.layer = 1;
    remote_q.placement =
        strix::PlacementKind::remote;
    const auto q1 =
        strix::select_unique_manifest_tensor(
            manifest,
            index,
            remote_q);
    expect(manifest.tensors[q1].placement.node == 1,
           "selector can resolve remote placement explicitly");

    bool ambiguous = false;
    try {
        static_cast<void>(
            strix::select_unique_manifest_tensor(
                manifest,
                index,
                layer_zero));
    } catch (const std::invalid_argument&) {
        ambiguous = true;
    }
    expect(ambiguous,
           "unique selector rejects ambiguous tensor role");

    auto changed = manifest;
    ++changed.tensors[0].absolute_offset;
    bool stale_index = false;
    try {
        static_cast<void>(
            strix::select_manifest_tensors(
                changed,
                index,
                q_selector));
    } catch (const std::invalid_argument&) {
        stale_index = true;
    }
    expect(stale_index,
           "tensor index rejects changed manifest identity");


    strix::ModelTensorSelector alias_selector;
    alias_selector.role =
        strix::TensorRole::dense_projection;
    alias_selector.name_contains_any = {
        "_shexp",
        "shared_expert",
    };
    const auto aliases =
        strix::select_manifest_tensors(
            manifest,
            index,
            alias_selector);
    expect(
        !aliases.empty(),
        "contains-any selector accepts naming aliases");

    if (failures == 0) {
        std::cout << "model tensor index tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
