// Host tests: (1) GGUF wire id 43 is upstream F8_E4M3 (1 byte / element) unless the file carries the Strix NVFP4 tile-pack
// marker; (2) the shared tensor naming map checker claims every tensor of a pack exactly once with shapes resolved from a
// model's symbols (a synthetic table with global, every-block, per-class and per-layer entries), and reports unclaimed,
// missing, mis-shaped and mis-typed tensors. A family's own table is tested with its adapter.
#include "gguf_test_builder.hpp"

#include <exception>

namespace {

void type43_by_file() {
    Builder plain;
    plain.str("general.architecture", "test");
    plain.tensor("lookup.weight", {96, 3}, 43, 96 * 3);
    const auto upstream = parse(plain.build());
    check(!upstream.strix_nvfp4_tile_pack, "plain file is not a tile pack");
    check(upstream.tensors[0].byte_size == 288, "upstream F8_E4M3 is one byte per element");
    check(upstream.is_f8_e4m3(upstream.tensors[0]), "43 outside a tile pack reads as F8_E4M3");

    Builder pack;
    pack.str("general.architecture", "test");
    pack.tensor("w.weight", {128, 16}, 43, 72 * 16);
    pack.tensor(gguf::kNvfp4TilePackMarker, {1}, 0, 4);
    const auto tiles = parse(pack.build());
    check(tiles.strix_nvfp4_tile_pack, "marker tensor makes a tile pack");
    check(tiles.tensors[0].byte_size == 72 * 16, "tile pack keeps the NVFP4 tile geometry");
    check(!tiles.is_f8_e4m3(tiles.tensors[0]), "43 inside a tile pack is not F8_E4M3");

    Builder wrong;
    wrong.str("general.architecture", "test");
    wrong.tensor("lookup.weight", {96, 3}, 43, 96 * 3);
    wrong.tensor(gguf::kNvfp4TilePackMarker, {1}, 0, 4);
    bool threw = false;
    try { (void)parse(wrong.build()); } catch (const std::exception&) { threw = true; }
    check(threw, "a 96-wide 43 tensor in a tile pack is rejected (tile geometry)");
}

// Synthetic family: 3 blocks (0 dense, 1-2 MoE), expert count per layer (4 on block 1, 5 on block 2), one product extent.
constexpr std::array<tm::Entry, 4> kTable{{
    {"token_embd.weight", tm::Layers::global, "embed", {{{tm::Sym::hidden}, {tm::Sym::vocab}, {}, {}}}, tm::kLinearTypes},
    {"blk.{L}.attn_norm.weight", tm::Layers::all, "norm", {{{tm::Sym::hidden}, {}, {}, {}}}, tm::kF32},
    {"blk.{L}.attn_v.weight", tm::Layers::all, "attn.v", {{{tm::Sym::hidden}, {tm::Sym::heads, tm::Sym::v_head}, {}, {}}},
     tm::kLinearTypes},
    {"blk.{L}.ffn_gate_exps.weight", tm::Layers::moe, "experts.gate",
     {{{tm::Sym::hidden}, {tm::Sym::expert_ff}, {tm::Sym::experts}, {}}}, tm::kLinearTypes},
}};

tm::Model synthetic_model() {
    tm::Model m;
    m.blocks = 3;
    const auto set = [&](tm::Sym s, std::uint64_t v) { m.value[static_cast<std::size_t>(s)] = v; };
    set(tm::Sym::hidden, 32);
    set(tm::Sym::vocab, 10);
    set(tm::Sym::heads, 2);
    set(tm::Sym::v_head, 3);
    set(tm::Sym::expert_ff, 6);
    m.per_layer[static_cast<std::size_t>(tm::Sym::experts)] = {0, 4, 5};
    const auto bit = [](tm::Layers l) { return 1U << static_cast<std::uint32_t>(l); };
    const std::uint32_t every = bit(tm::Layers::all) | bit(tm::Layers::main);
    m.layer_class = {every | bit(tm::Layers::dense_ffn), every | bit(tm::Layers::moe), every | bit(tm::Layers::moe)};
    return m;
}

Builder synthetic_pack(const tm::Model& model) {
    Builder b;
    b.str("general.architecture", "test");
    for (const auto& entry : kTable) {
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
            b.tensor(tm::expand(entry.name, layer), dims, 0U, elements * 4);
        }
    }
    return b;
}

void synthetic_map() {
    const auto model = synthetic_model();
    check(model.resolve(tm::Sym::experts, 1) == 4 && model.resolve(tm::Sym::experts, 2) == 5, "per-layer symbol");
    check(model.in(tm::Layers::moe, 1) && !model.in(tm::Layers::moe, 0) && !model.in(tm::Layers::global, 0), "layer classes");
    const Builder good = synthetic_pack(model);
    const auto pack = parse(good.build());
    const auto result = tm::check(pack.tensors, kTable, model);
    for (const auto& message : result.errors) std::printf("  %s\n", message.c_str());
    check(result.ok(), "synthetic pack: every tensor claimed exactly once");
    // Independent count: 1 global + 2 x 3 every-block + 1 x 2 MoE.
    check(result.claims.size() == 9 && pack.tensors.size() == result.claims.size(), "claim count");
    for (const auto& t : pack.tensors) {
        if (t.name == "blk.0.attn_v.weight") check(t.dimensions == std::vector<std::uint64_t>{32, 6}, "product extent heads*v_head");
        if (t.name == "blk.2.ffn_gate_exps.weight") check(t.dimensions == std::vector<std::uint64_t>{32, 6, 5}, "per-layer extent");
    }

    const auto expect_error = [&](const Builder& b, const char* needle, const char* what) {
        const auto file = parse(b.build());
        const auto r = tm::check(file.tensors, kTable, model);
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
    for (auto& t : shape.tensors) if (t.name == "blk.2.ffn_gate_exps.weight") t.dims[2] = 4;
    expect_error(shape, "shape [32,6,4] != expected [32,6,5]", "wrong per-layer expert count is a shape error");
    Builder type = good;
    for (auto& t : type.tensors) if (t.name == "blk.0.attn_norm.weight") { t.type = 8; t.dims = {32}; t.bytes = 34; }
    expect_error(type, "type 8 not admitted for blk.0.attn_norm.weight", "quantized norm is a type error");
    Builder stray = good;
    for (auto& t : stray.tensors) if (t.name == "blk.1.ffn_gate_exps.weight") t.name = "blk.0.ffn_gate_exps.weight";
    expect_error(stray, "unclaimed tensor: blk.0.ffn_gate_exps.weight", "MoE tensor on a dense block is unclaimed");
}

}  // namespace

int main() {
    try {
        type43_by_file();
        synthetic_map();
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("FAIL %d check(s)\n", failures);
        return 1;
    }
    std::printf("PASS gguf tensor map tests\n");
    return 0;
}
