// Hy4-flash tensor naming table (src/weight/adapters/hy4.cpp) on the shared checker: the metadata resolver on a small
// synthetic pack (4 blocks: 0 dense, 1-2 MoE, 3 next-token; sparse attention on 0 and 3, Engram on block 1), every tensor
// claimed exactly once, and unclaimed / missing / shape / type / wrong-class errors.
#include "gguf_test_builder.hpp"

#include <exception>

namespace {

// Small Hy4-shaped pack: 4 blocks (0 dense, 1-2 MoE, 3 next-token), sparse attention on 0 and 3, Engram on block 1.
Builder hy4_metadata() {
    Builder b;
    const std::string p = "hy_v4_flash.";
    b.str("general.architecture", "hy_v4_flash");
    b.u32(p + "block_count", 4);
    b.u32(p + "nextn_predict_layers", 1);
    b.u32(p + "leading_dense_block_count", 1);
    b.u32(p + "embedding_length", 32);
    b.u32(p + "vocab_size", 64);
    b.u32(p + "attention.q_lora_rank", 16);
    b.u32(p + "attention.kv_lora_rank", 8);
    b.u32(p + "rope.dimension_count", 4);
    b.u32(p + "attention.key_length_mla", 12);
    b.u32(p + "attention.value_length_mla", 6);
    b.u32(p + "feed_forward_length", 48);
    b.u32(p + "expert_count", 8);
    b.u32(p + "expert_feed_forward_length", 10);
    b.u32(p + "expert_shared_count", 1);
    b.u32(p + "expert_latent_length", 14);
    b.u32(p + "attention.indexer.head_count", 2);
    b.u32(p + "attention.indexer.key_length", 4);
    b.u32(p + "attention.indexer.kpool", 4);
    b.u32(p + "hyper_connection.count", 4);
    b.u32(p + "ple.ngram_size", 3);
    b.u32(p + "ple.heads_per_ngram", 2);
    b.u32(p + "embedding_length_per_layer_input", 6);
    b.i32_array(p + "attention.head_count", {2, 4, 4, 2});
    b.bool_array(p + "attention.indexer.types", {true, false, false});
    b.i32_array(p + "ple.layers", {1});
    b.u64_array(p + "ple.head_offsets", {0, 5, 12, 20});
    b.u64_array(p + "ple.head_vocab_sizes", {5, 7, 8, 9});
    return b;
}

// Adds every tensor the table expands to (F32 everywhere, F8_E4M3 lookup table) with the resolved shapes.
void add_pack_tensors(Builder& b, const tm::Model& model) {
    const auto& map = tm::hy4_flash_map();
    for (const auto& entry : map.table) {
        std::vector<std::int64_t> layers;
        if (entry.layers == tm::Layers::global) layers.push_back(-1);
        for (std::uint32_t layer = 0; layer < model.blocks; ++layer)
            if (model.in(entry.layers, layer)) layers.push_back(layer);
        for (const auto layer : layers) {
            std::vector<std::uint64_t> dims;
            std::uint64_t elements = 1;
            for (const auto& extent : entry.dims) {
                if (extent.a == tm::Sym::none) break;
                const auto at = layer < 0 ? 0U : static_cast<std::uint32_t>(layer);
                dims.push_back(model.resolve(extent.a, at) * model.resolve(extent.b, at));
                elements *= dims.back();
            }
            const bool lookup = entry.component == "engram.table";
            b.tensor(tm::expand(entry.name, layer), dims, lookup ? 43U : 0U, lookup ? elements : elements * 4);
        }
    }
}

void hy4_map() {
    const auto metadata_only = parse(hy4_metadata().build());
    tm::Model model;
    std::string error;
    const auto& map = tm::hy4_flash_map();
    check(map.model(metadata_only, model, error), "Hy4 metadata resolves");
    if (!error.empty()) std::printf("  %s\n", error.c_str());
    check(model.blocks == 4, "block count");
    check(model.resolve(tm::Sym::lookup_rows, 0) == 29, "lookup rows = sum of head vocab sizes");
    check(model.resolve(tm::Sym::heads, 1) == 4 && model.resolve(tm::Sym::heads, 3) == 2, "per-layer heads");
    check(model.resolve(tm::Sym::qk_nope, 0) == 8 && model.resolve(tm::Sym::kv_a, 0) == 12, "derived MLA extents");
    check(model.in(tm::Layers::sparse_index, 0) && !model.in(tm::Layers::sparse_index, 1) && model.in(tm::Layers::sparse_index, 3),
          "sparse blocks: indexer.types plus the next-token block");
    check(model.in(tm::Layers::dense_ffn, 0) && model.in(tm::Layers::moe, 3) && !model.in(tm::Layers::main, 3), "block classes");

    Builder good = hy4_metadata();
    add_pack_tensors(good, model);
    const auto pack = parse(good.build());
    const auto result = tm::check(pack.tensors, map.table, model);
    for (const auto& message : result.errors) std::printf("  %s\n", message.c_str());
    check(result.ok(), "synthetic pack: every tensor claimed exactly once");
    // Independent count: 7 global + 14 x 4 every-block + 6 x 3 main + 8 x 2 sparse + 3 dense + 10 x 3 MoE + 6 Engram + 4 next-token.
    check(result.claims.size() == 7 + 56 + 18 + 16 + 3 + 30 + 6 + 4, "claim count");
    check(pack.tensors.size() == result.claims.size(), "one claim per tensor");
    for (const auto& t : pack.tensors) {
        if (t.name == "blk.1.attn_q_b.weight") check(t.dimensions == std::vector<std::uint64_t>{16, 48}, "q_b = q_lora x heads*qk_head");
        if (t.name == "blk.1.engram_kv.weight") check(t.dimensions == std::vector<std::uint64_t>{24, 160}, "engram_kv = (4 heads*6) x 32*5");
    }

    const auto expect_error = [&](Builder b, const char* needle, const char* what) {
        const auto file = parse(b.build());
        const auto r = tm::check(file.tensors, map.table, model);
        bool hit = false;
        for (const auto& message : r.errors) hit = hit || message.find(needle) != std::string::npos;
        check(!r.ok() && hit && r.claims.empty(), what);
    };
    Builder extra = good;
    extra.tensor("blk.1.unknown.weight", {32}, 0, 128);
    expect_error(extra, "unclaimed tensor: blk.1.unknown.weight", "unknown tensor is reported unclaimed");
    Builder missing = good;
    missing.tensors.erase(missing.tensors.begin() + 3);
    expect_error(missing, "missing tensor:", "absent tensor is reported missing");
    Builder shape = good;
    for (auto& t : shape.tensors) if (t.name == "blk.2.ffn_gate_exps.weight") t.dims[2] = 7;
    expect_error(shape, "shape [14,10,7] != expected [14,10,8]", "wrong expert count is a shape error");
    Builder type = good;
    for (auto& t : type.tensors) if (t.name == "blk.0.attn_norm.weight") { t.type = 8; t.dims = {32}; t.bytes = 34; }
    expect_error(type, "type 8 not admitted for blk.0.attn_norm.weight", "quantized norm is a type error");
    Builder stray = good;
    for (auto& t : stray.tensors) if (t.name == "blk.1.engram_gate.weight") t.name = "blk.2.engram_gate.weight";
    expect_error(stray, "unclaimed tensor: blk.2.engram_gate.weight", "Engram tensor on a non-Engram block is unclaimed");
}

}  // namespace

int main() {
    try {
        hy4_map();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("FAIL %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS hy4 tensor map tests\n");
    return 0;
}
