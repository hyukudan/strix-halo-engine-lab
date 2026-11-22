#include "strix/graph/value.hpp"
#include "strix/graph/split_view.hpp"
#include "strix/hy4/tp_layout.hpp"
#include "strix/pair/stage_map.hpp"
#include "strix/pair/tp_layout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace strix::graph;
namespace md = strix::model_description;
namespace ps = strix::pair::split;
namespace tp = strix::pair::tp;
namespace hy = strix::hy4::tp;

void check(bool pass, std::string_view detail) {
    if (!pass) throw std::runtime_error(std::string(detail));
}

template<class F>
void rejects(F run) {
    try {
        run();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected invalid split data or value reference");
}

void value_references() {
    constexpr ValueDecl values[] = {
        {"x", 4, 2560, ValueScope::visit}, {"sel", 1, 64, ValueScope::step},
        {"bank", 1, 2560, ValueScope::pass}, {"ple", 1, 256, ValueScope::model}
    };
    static constexpr LayerVisit schedule[] = {{20,0,0}, {21,0,1}, {20,1,2}, {21,1,3}, {22,0,4}};
    const auto resolve = [&](std::string_view spelling, const LayerVisit* current = &schedule[3]) {
        return resolve_value_ref(parse_value_ref(spelling), values, schedule, current);
    };
    check(parse_value_scope("step") == ValueScope::step && parse_value_scope("visit") == ValueScope::visit &&
          parse_value_scope("pass") == ValueScope::pass && parse_value_scope("model") == ValueScope::model,
          "value scopes");
    check(resolve("x").logical_visit == 3 && resolve("x").value.streams == 4 &&
          resolve("x").value.width == 2560 && resolve("x").value.scope == ValueScope::visit, "local value");
    check(resolve("sel@20").logical_visit == 2 && resolve("sel@20", nullptr).logical_visit == 0 &&
          resolve("sel@20").value.scope == ValueScope::step, "absolute physical layer");
    check(resolve("bank@L-1").logical_visit == 2 && resolve("bank@L-1").value.scope == ValueScope::pass &&
          resolve("bank@L+1").logical_visit == 4 && resolve("bank@L").logical_visit == 3, "relative visit selection");
    check(resolve("ple[L]").model_slice == 21 && !resolve("ple[L]").logical_visit &&
          resolve("ple[L]").value.scope == ValueScope::model && !resolve("x", nullptr).logical_visit,
          "model slice and edge value");
    rejects([] { (void)parse_value_scope("layer"); });
    for (const auto bad : {"@20", "x@", "x@-1", "x@L-+1", "x@L+", "x@9223372036854775808", "ple[20]",
                           "ghost", "x[L]", "ple@0", "sel@L-22", "sel@L+9223372036854775807"})
        rejects([&] { (void)resolve(bad); });
    rejects([&] { (void)resolve("x@L", nullptr); });
    rejects([&] { (void)resolve("ple[L]", nullptr); });
    std::cout << "PASS value scopes and references\n";
}

void declare_stage(md::Value::Array& stages, std::string_view id, std::string_view split,
                   std::string_view table = "layer", std::string_view when = {}) {
    md::Value::Table row{{"id", {std::string(id)}}, {"op", {std::string("elementwise")}},
        {"out", {std::string("planned")}}, {"table", {std::string(table)}}, {"split", {std::string(split)}}};
    if (!when.empty()) row.emplace("when", md::Value{std::string(when)});
    stages.push_back({std::move(row)});
}

std::string_view split_word(ps::Axis axis) {
    switch (axis) {
    case ps::Axis::replicated: return "replicate";
    case ps::Axis::heads: return "by_heads";
    case ps::Axis::columns: return "by_k";
    case ps::Axis::slots: return "by_experts";
    case ps::Axis::vocab: return "by_vocab";
    }
    throw std::logic_error("unknown test axis");
}

std::vector<TensorPartition> half_partitions(std::span<const SplitStage> stages) {
    std::vector<TensorPartition> partitions;
    for (std::size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].axis == ps::Axis::replicated) continue;
        const auto half = stages[i].cost.units / 2U;
        check(2U * half == stages[i].cost.units, "even test axis");
        partitions.push_back({i, {{{{0, half}}, {}}, {{{half, half}}, {}}}});
    }
    return partitions;
}

std::vector<tp::Segment> legacy_segments(const RankStage& stage, std::uint64_t width = 1) {
    std::vector<tp::Segment> segments;
    for (const auto range : stage.shard.units)
        segments.push_back({static_cast<std::uint32_t>(range.first * width),
                            static_cast<std::uint32_t>(range.count * width)});
    return segments;
}

void check_segments(std::span<const tp::Segment> actual, std::span<const tp::Segment> expected) {
    check(std::ranges::equal(actual, expected, [](const auto& a, const auto& b) {
        return a.global == b.global && a.count == b.count;
    }), "Qwen TP segments");
}

void qwen_identity() {
    enum StageIndex : std::size_t { norm, qkv, beta, core, from_gdn, q, k, ctx, from_attn, mlp, head };
    constexpr std::string_view ids[] = {"attn_norm","qkv","beta","gdn","gdn_out","q","k","attend","o","silu_mul","vocab"};
    constexpr ps::Axis axes[] = {ps::Axis::replicated, ps::Axis::columns, ps::Axis::heads, ps::Axis::columns,
        ps::Axis::columns, ps::Axis::heads, ps::Axis::heads, ps::Axis::columns, ps::Axis::columns,
        ps::Axis::columns, ps::Axis::vocab};
    static_assert(std::size(ids) == std::size(axes));
    auto desc = md::parse(R"(
schema = "strix-model/v2"
name = "qwen-27b-tp-view"
[source]
container = "gguf"
architecture = "qwen35"
[dims]
layers = 64
[classes]
full = "(L+1)%4==0"
)");
    md::Value::Array declarations;
    for (std::size_t i = 0; i < std::size(ids); ++i) {
        const auto guard = i >= qkv && i <= from_gdn ? "!full" : i >= q && i <= from_attn ? "full" : "";
        declare_stage(declarations, ids[i], split_word(axes[i]), i == head ? "after" : "layer", guard);
        if (i == core)
            std::get<md::Value::Table>(declarations.back().data).emplace("join_order", md::Value{std::string("rank")});
    }
    desc.fields.emplace("stage", md::Value{std::move(declarations)});
    const auto model = inspect_model_decl(desc);
    constexpr std::uint64_t units[] = {1,10240,48,6144,80,24,4,6144,80,272,3880};
    constexpr std::uint64_t widths[] = {5120,1,1,1,64,256,256,1,64,64,64};
    constexpr std::uint32_t visits[] = {0,0,0,0,0,3,3,3,3,3,0};
    std::vector<PricedSplitStage> program;
    for (std::size_t i = 0; i < std::size(ids); ++i) {
        const SplitStageKey key{ids[i], i == head ? StageTable::after : StageTable::layer,
                                i == head ? std::nullopt : std::optional{visits[i]}};
        SplitStageCost cost;
        cost.units = units[i];
        cost.unit_width = widths[i];
        if (i == core || i == ctx || i == mlp) {
            cost.join = ps::Join::gather;
            cost.wire_format = Format::f16;
            cost.join_bytes_per_row = units[i] * widths[i] * 2;
        }
        if (i == from_gdn || i == from_attn) {
            cost.join = ps::Join::gather;
            cost.wire_format = Format::f32;
            cost.join_bytes_per_row = units[i] * widths[i] * 4;
        }
        if (i == head) {
            cost.join = ps::Join::pick;
            cost.wire_format = Format::f32;
            cost.join_bytes_per_row = 8;
        }
        program.push_back({key, cost});
    }
    const auto view = split_view(model, program);
    check(view[core].join_in_rank_order && view[core].cost.wire_format == Format::f16 &&
          view[head].layer() == -1 && view[k].layer() == 3, "Qwen stage metadata");
    auto partitions = half_partitions(view);
    const auto shards = [&](StageIndex i) -> std::vector<RankShard>& {
        const auto found = std::ranges::find(partitions, static_cast<std::size_t>(i), &TensorPartition::stage);
        check(found != partitions.end(), "missing Qwen partition");
        return found->ranks;
    };
    for (std::uint32_t r = 0; r < 2; ++r) {
        const std::uint64_t h = r * 8U;
        shards(beta)[r].units = {{h,8}, {16+h,8}, {32+h,8}};
        shards(core)[r].units = {{h*128,1024}, {(16+h)*128,1024}, {(32+h)*128,1024}};
        shards(qkv)[r].units = {{h*128,1024}, {(16+h)*128,1024}, {(32+h)*128,1024},
                                {(48+h)*128,1024}, {(64+h)*128,1024}};
    }
    constexpr std::size_t slot_stride = 17408;
    for (const auto i : {core, ctx, from_gdn, from_attn, mlp}) {
        for (auto& shard : shards(i)) {
            shard.join_layout = RankJoinLayout{slot_stride, units[i] * widths[i], {}};
            std::uint64_t local = 0;
            for (const auto range : shard.units) {
                const auto count = range.count * widths[i];
                shard.join_layout->slices.push_back({{local, count}, range.first * widths[i]});
                local += count;
            }
        }
    }
    const SplitPlan plan{2, TensorSplit{partitions}};
    constexpr double rates[] = {1.0, 1.0};
    const auto legacy_head = tp::head_columns(true, 248320, rates);
    std::array<std::vector<tp::Segment>, 2> legacy_core;
    std::array<RankView, 2> ranks{split_view(view, plan, 0), split_view(view, plan, 1)};
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        const auto& stages = ranks[rank].stages;
        const auto keys = tp::split_units(16, rates)[rank];
        const auto vh = tp::mod_value_heads(16, 48, keys);
        legacy_core[rank] = tp::expand(vh, 128);
        auto qkv_columns = tp::expand(std::array{tp::Segment{keys.begin, keys.size()}}, 128);
        qkv_columns.push_back({2048U + keys.begin * 128U, keys.size() * 128U});
        const auto v_columns = tp::expand(vh, 128, 4096);
        qkv_columns.insert(qkv_columns.end(), v_columns.begin(), v_columns.end());
        check_segments(legacy_segments(stages[qkv]), qkv_columns);
        check(tp::select_map(legacy_segments(stages[beta])) == tp::select_map(vh), "Qwen beta/alpha head map");
        check_segments(legacy_segments(stages[core]), legacy_core[rank]);
        for (const auto i : {q, k, ctx, from_gdn, from_attn, mlp}) {
            const auto share = tp::split_units(static_cast<std::uint32_t>(units[i]), rates)[rank];
            const auto w = static_cast<std::uint32_t>(i == q || i == k ? 1 : widths[i]);
            check_segments(legacy_segments(stages[i], w),
                           std::array{tp::Segment{share.begin * w, share.size() * w}});
        }
        check(legacy_head.split && stages[head].partitioned && view[head].cost.join == ps::Join::pick,
              "Qwen split head and argmax policy");
        check_segments(legacy_segments(stages[head], 64), std::span{legacy_head.cols}.subspan(rank, 1));
        check(!stages[norm].partitioned && stages[norm].shard.units == std::vector<SplitRange>{{0,1}},
              "Qwen replicated norm");
    }
    // The layout view specifies the same mapping as the legacy base/stride arrays.
    const auto legacy_join = tp::join_map(legacy_core, 6144, slot_stride);
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        const auto& layout = *ranks[rank].stages[core].shard.join_layout;
        check(layout.slot_elements == legacy_join.slot_stride, "Qwen gather slot stride");
        for (const auto& slice : layout.slices) {
            for (std::uint64_t i = 0; i < slice.local.count; ++i) {
                const auto c = static_cast<std::size_t>(slice.destination + i);
                check(legacy_join.base[c] == static_cast<std::int32_t>(rank * layout.slot_elements + slice.local.first + i) &&
                      legacy_join.stride[c] == static_cast<std::int32_t>(legacy_join.max_width), "Qwen gather base/stride");
            }
        }
    }
    auto wrong_visit = program[q];
    wrong_visit.key.logical_visit = 0;
    rejects([&] { (void)split_view(model, std::span{&wrong_visit, 1}); });
    const std::array duplicate{program[q], program[q]};
    rejects([&] { (void)split_view(model, duplicate); });
    rejects([&] { (void)split_view(model, program, "missing"); });
    std::cout << "PASS planner_view_identity Qwen 27B TP2\n";
}

struct Hy4Config {
    std::uint32_t layers = 42, hidden = 3072, vocab = 120832, v_head = 256, dense_ff = 13824,
                  top_k = 16, latent = 1536, shared_ff = 1536, experts = 512;
    std::vector<std::uint32_t> heads;
    std::vector<std::uint8_t> dense, indexed;
};

void hy4_identity() {
    Hy4Config shape;
    auto desc = md::parse(R"(
schema = "strix-model/v2"
name = "hy4-pair-view"
[source]
container = "gguf"
architecture = "hy4"
[dims]
layers = 42
)");
    md::Value::Array declarations;
    std::vector<PricedSplitStage> program;
    constexpr std::uint64_t mib = 1ULL << 20U;
    const auto add = [&](std::string_view id, ps::Axis axis, std::uint64_t units, std::uint64_t width,
                         std::uint64_t bytes, ps::Join join, std::uint64_t join_bytes,
                         std::uint32_t layer, std::string_view when = {}) {
        const bool edge = layer == shape.layers;
        if (layer <= 1 || edge) {
            const bool already = std::ranges::any_of(declarations, [&](const md::Value& row) {
                return std::get<std::string>(std::get<md::Value::Table>(row.data).at("id").data) == id;
            });
            if (!already) declare_stage(declarations, id, split_word(axis), edge ? "after" : "layer", when);
        }
        SplitStageCost cost;
        cost.units = units;
        cost.unit_width = width;
        cost.bytes = bytes;
        cost.join = join;
        cost.join_bytes_per_row = join_bytes;
        if (join != ps::Join::none) cost.wire_format = edge ? Format::f32 : Format::bf16;
        if (axis == ps::Axis::slots) {
            cost.pool = shape.experts;
            cost.mean_slot_bytes = static_cast<double>(3 * ((512 * mib) / 3)) / shape.experts;
        }
        program.push_back({{id, edge ? StageTable::after : StageTable::layer,
                            edge ? std::nullopt : std::optional{layer}}, cost});
    };
    using enum ps::Axis;
    constexpr auto none = ps::Join::none;
    constexpr auto gather = ps::Join::gather;
    for (std::uint32_t l = 0; l < shape.layers; ++l) {
        const auto n_head = l % 2 == 0 ? 64U : 32U;
        shape.heads.push_back(n_head);
        shape.indexed.push_back(static_cast<std::uint8_t>(l % 2));
        shape.dense.push_back(static_cast<std::uint8_t>(l == 0));
        add("attn inputs", replicated, 1, 1, 2*mib, none, 0, l);
        add("attn heads", heads, n_head, shape.v_head, 4*mib, gather, 4ULL*n_head*shape.v_head, l);
        add("o_proj rows", columns, shape.hidden/64, 64, mib, gather, 4ULL*shape.hidden, l);
        if (l == 0) {
            add("ffn gate/up rows", columns, shape.dense_ff/64, 64, 2*mib, gather, 4ULL*shape.dense_ff, l, "L == 0");
            add("ffn down rows", columns, shape.hidden/64, 64, mib, gather, 4ULL*shape.hidden, l, "L == 0");
        } else {
            add("router + latent down", replicated, 1, 1, 2*mib, none, 0, l, "L > 0");
            add("shared gate/up rows", columns, shape.shared_ff/64, 64, 2*mib, none, 0, l, "L > 0");
            add("routed slots", ps::Axis::slots, shape.top_k, shape.latent, 0, ps::Join::slots, 4ULL*shape.latent, l, "L > 0");
            add("latent up + shared down", replicated, 1, 1, 2*mib, none, 0, l, "L > 0");
        }
    }
    add("head rows", vocab, shape.vocab/64, 64, mib, gather, 4ULL*shape.vocab, shape.layers);
    desc.fields.emplace("stage", md::Value{std::move(declarations)});
    const auto model = inspect_model_decl(desc);
    const auto view = split_view(model, program);
    const auto expected = hy::decode_stages(shape, [=](const std::string& tensor) -> std::uint64_t {
        return tensor.find("_exps") == std::string::npos ? mib : (512 * mib) / 3;
    });
    check(view.size() == expected.size(), "Hy4 stage count");
    const auto same_bits = [](double a, double b) { return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b); };
    std::size_t moe_stage = 0;
    std::uint32_t gathers = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto& a = view[i];
        const auto& b = expected[i];
        const auto legacy_layer = a.visit ? a.visit->physical_layer : shape.layers;
        const auto& ac = a.cost;
        check(a.id == b.name && legacy_layer == b.layer && a.axis == b.axis && ac.units == b.units &&
              same_bits(ac.mean_slot_bytes.value_or(static_cast<double>(ac.bytes)), b.bytes) &&
              same_bits(static_cast<double>(ac.bytes_per_row), b.bytes_per_row) && same_bits(ac.ms_per_row, b.ms_per_row) &&
              same_bits(ac.extra_slot_bytes, b.extra_slot_bytes) && ac.pool == b.pool &&
              same_bits(ac.efficiency, b.efficiency) && same_bits(ac.fixed_ms, b.fixed_ms) && ac.join == b.join &&
              same_bits(static_cast<double>(ac.join_bytes_per_row), b.join_bytes_per_row) &&
              same_bits(ac.extra_join_bytes, b.extra_join_bytes), b.name);
        if (a.id == "routed slots") moe_stage = i;
        if (ac.join != none) ++gathers;
    }
    check(gathers == hy::step_count(hy::dims_of(shape)).gathers, "Hy4 gather count");

    // The rank-local stage units and both destination regions of the fused MoE join.
    auto partitions = half_partitions(view);
    auto moe = std::ranges::find(partitions, moe_stage, &TensorPartition::stage);
    check(moe != partitions.end(), "Hy4 MoE partition");
    moe->ranks[0].join_layout = RankJoinLayout{13056, 26112, {{{0,768},0}, {{768,12288},768}}};
    moe->ranks[1].join_layout = RankJoinLayout{13056, 26112, {{{0,768},25344}, {{768,12288},13056}}};
    const SplitPlan tensor{2, TensorSplit{std::move(partitions)}};
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        const auto out = split_view(view, tensor, rank);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const auto& s = out.stages[i];
            const auto& legacy = expected[i];
            const auto range = legacy.axis == replicated ? hy::Range{0,legacy.units} : hy::half(legacy.units, rank);
            check(s.stage == i && s.partitioned == (legacy.axis != replicated) &&
                  s.shard.units == std::vector<SplitRange>{{range.first, range.count}}, legacy.name);
        }
        const auto legacy_moe = hy::moe(rank, shape.top_k, shape.latent, shape.shared_ff);
        const auto& shard = out.stages[moe_stage].shard;
        const auto& join = *shard.join_layout;
        check(shard.units.front() == SplitRange{legacy_moe.slot_first, legacy_moe.slots} &&
              join.slot_elements == legacy_moe.slot && join.slices[1].local.first == legacy_moe.raw_all &&
              join.slices[1].destination == legacy_moe.raw_mine && join.slices[0].destination == legacy_moe.act_mine &&
              join.joined_elements - join.slices[0].local.count == legacy_moe.act_pitch &&
              legacy_moe.act_first == 0 && view[moe_stage].cost.wire_format == Format::bf16, "Hy4 MoE layout");
    }

    const auto map = strix::pair::StageMap::parse("0,10.5,21,31,42", shape.layers);
    constexpr std::size_t boundaries[] = {0,71,145,215,293};
    LayersSplit layers;
    for (std::uint32_t segment = 0; segment < map.stages(); ++segment)
        layers.segments.push_back({segment % 2, {boundaries[segment], boundaries[segment + 1] - boundaries[segment]}});
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        const auto out = split_view(view, {2, layers}, rank);
        for (std::size_t i = 0; i < view.size(); ++i) {
            const auto& s = view[i];
            const bool ffn = s.id != "attn inputs" && s.id != "attn heads" && s.id != "o_proj rows";
            const auto half_layer = s.visit ? 2ULL * s.visit->physical_layer + ffn : 2ULL * shape.layers - 1;
            std::uint32_t segment = 0;
            while (half_layer >= map.cuts[segment + 1].units()) ++segment;
            check((std::ranges::find(out.stages, i, &RankStage::stage) != out.stages.end()) ==
                  (map.owner(segment) == rank), "Hy4 interleaved layer partition");
        }
    }
    std::cout << "PASS planner_view_identity Hy4 pair\n";
}

void partition_validation() {
    const SplitStage stage{"test", StageTable::layer, LayerVisit{0,0,0}, ps::Axis::heads, Split::by_heads,
                           false, SplitStageCost{}};
    const std::span stages{&stage, 1};
    const SplitPlan context{2, ContextSplit{256, {{0,128}, {128,128}}}};
    const auto second = split_view(stages, context, 1);
    check(second.context == SplitRange{128,128} && second.stages.size() == 1 && !second.stages[0].partitioned,
          "context positions and replicated stages");
    rejects([&] { (void)split_view(stages, {2, ContextSplit{256, {{0,128}, {127,129}}}}, 0); });
    rejects([&] { (void)split_view(stages, {2, ContextSplit{256, {{0,128}, {129,127}}}}, 0); });
    rejects([&] { (void)split_view(stages, context, 2); });
    rejects([&] { (void)split_view(stages, {2, LayersSplit{{{0, {0,2}}}}}, 0); });
    TensorPartition bad{0, {{{{0,1}}, {}}, {{{0,1}}, {}}}};
    rejects([&] { (void)split_view(stages, {2, TensorSplit{{bad}}}, 0); });
    bad.ranks[1].units.clear();
    check(split_view(stages, {2, TensorSplit{{bad}}}, 1).stages.front().shard.units.empty(), "empty tensor rank");
    auto gather_stage = stage;
    gather_stage.cost.join = ps::Join::gather;
    bad.ranks[0].join_layout = RankJoinLayout{1, 1, {{{0,1},0}}};
    bad.ranks[1].join_layout = RankJoinLayout{1, 1, {{{0,1},0}}};
    rejects([&] { (void)split_view(std::span{&gather_stage, 1}, {2, TensorSplit{{bad}}}, 0); });
}
}  // namespace

int main() {
    try {
        value_references();
        qwen_identity();
        hy4_identity();
        partition_validation();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL split view: " << error.what() << '\n';
        return 1;
    }
}
