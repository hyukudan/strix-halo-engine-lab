#include "exec/cuda_bound_transfers.hpp"
#include "exec/cuda_decode_epilogue.hpp"
#include "exec/executor_cuda_binding.hpp"
#include "exec/step_replay_row.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_expert_layout.hpp"
#include "inkling2/inkling2_mini_program_cuda.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#include "moe_core/microscaled_r1.hpp"
#include "strix/exec/compiler.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace {

namespace exec = strix::exec;
namespace graph = strix::graph;
namespace md = strix::model_description;
namespace mini = strix::inkling2;
namespace session = strix::session;
using namespace std::literals;

void check(bool condition, std::string_view what) {
    if (!condition) throw std::runtime_error(std::string(what));
}

md::ModelDescription load_fixture(const char* path) {
    std::ifstream input(path, std::ios::binary);
    check(input.good(), "could not open the Inkling declaration fixture");
    const std::string file{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    check(!input.bad(), "could not read the Inkling declaration fixture");
    const auto graph = file.find("\n[graph]");
    const auto tail = file.find("\n[[value]]", graph);
    check(graph != std::string::npos && tail != std::string::npos, "missing Inkling fixture sections");
    const std::string_view text(file);
    constexpr auto syntax = md::TableSyntax::root_array_tables;
    // Scope the MSVC parser's inline-table tracking to the graph block. Later
    // root table allocations can otherwise reuse addresses from its stage array.
    auto metadata = md::parse(file.substr(0, graph) + file.substr(tail), syntax);
    auto stages = md::parse(text.substr(graph, tail - graph), syntax);
    metadata.fields.merge(stages.fields);
    check(stages.fields.empty(), "duplicate graph in Inkling declaration fixture");
    return metadata;
}

void check_plan(const graph::LaunchPlan& plan) {
    graph::validate_plan(plan);
    std::size_t index = 0;
    const auto stage = [&](std::string_view id, std::int32_t logical, bool live = true) {
        if (!live) return;
        const auto& instance = plan.instances.at(index);
        const auto& launch = plan.launches.at(index);
        check(instance.decl->id == id && instance.layer == logical, "declared stage or visit out of order");
        check(launch.entry == instance.decl->entry && launch.layer == logical && launch.rows == 1U &&
              launch.stages.size() == 1U && launch.stages.front() == index,
              "today plan changed the semantic stage order");
        ++index;
    };
    stage("embed", -1);
    stage("embed_norm", -1);
    for (const auto visit : mini::mini_schedule()) {
        const auto layer = visit.physical_layer;
        const auto logical = static_cast<std::int32_t>(visit.logical_layer);
        const bool separate = mini::is_full_attention(layer) || visit.pass != 0U;
        stage("save_core_entry", logical, layer == mini::kCoreFirst && visit.pass == 0U);
        stage("restore_core_entry", logical, layer == mini::kCoreFirst && visit.pass != 0U);
        stage("capture_boundary", logical, mini::captures_message_boundary(visit));
        stage("attn_read", logical, layer != 0U);
        stage("query_input", logical);
        stage("key_input", logical, separate);
        stage("value_input", logical, separate);
        for (const auto id : {"qkv"sv, "prepare_kv"sv, "attend"sv, "attn_out"sv, "attn_conv"sv})
            stage(id, logical);
        stage("save_block", logical, mini::starts_residual_block(visit));
        stage("add_residual", logical, !mini::starts_residual_block(visit));
        for (const auto id : {"moe_read"sv, "moe_input"sv, "router"sv, "route"sv, "expert_gate_up"sv,
                             "expert_down"sv, "expert_merge"sv, "moe_conv"sv, "layer_out"sv})
            stage(id, logical);
        stage("capture_pass", logical, layer + 1U == mini::kCoreLast && visit.pass + 1U < mini::kCorePasses);
    }
    for (const auto id : {"readout"sv, "final_norm"sv, "head"sv, "softcap"sv}) stage(id, -1);
    check(index == plan.instances.size() && index == plan.launches.size(), "unexpected semantic stages");
}

void unexecuted_binding(std::span<const std::byte>, const exec::Grid&, void*) {
    throw std::logic_error("the semantic compiler fixture must remain device-free");
}

void check_compiler_admission(const graph::StageGraph& stages, const graph::LaunchPlan& plan) {
    exec::Compiler compiler(stages,
        [](const auto&, const auto&) { return exec::BufferLayout{exec::BufferStorage::external}; },
        [](const auto& request) {
            exec::KernelBinding binding;
            binding.fn = unexecuted_binding;
            binding.identity = request.plan.instances[request.launch.stages.front().instance].decl->id;
            return binding;
        });
    const auto program = compiler.compile({}, plan);
    check(program.launches.size() == plan.launches.size(), "X1 changed the semantic launch count");
    check(program.stages.size() == plan.instances.size(), "X1 changed the semantic stage count");
    std::size_t nvfp4 = 0;
    std::size_t mxfp8 = 0;
    const auto schedule = mini::mini_schedule();
    for (std::size_t i = 0; i < plan.launches.size(); ++i) {
        const auto& bound = program.launches[i];
        const auto& source = plan.launches[i];
        check(bound.entry == source.entry && bound.layer == source.layer &&
              bound.stream == source.stream && bound.after == source.after,
              "X1 changed the D3 launch identity or edges");
        const auto& stage = program.stages[i];
        if (stage.id != "expert_gate_up" && stage.id != "expert_down") continue;
        check(stage.layer >= 0, "missing logical expert visit");
        const auto visit = schedule[static_cast<std::size_t>(stage.layer)];
        const bool packed_fp4 = mini::mini_expert_format(visit.physical_layer) ==
            strix::weight::microscaled::MatrixFormat::nvfp4;
        check(stage.format == (packed_fp4 ? graph::Format::nvfp4 : graph::Format::mxfp8),
              "X1 changed the format of a physical expert on a logical visit");
        if (packed_fp4) ++nvfp4;
        else ++mxfp8;
    }
    check(nvfp4 != 0 && mxfp8 != 0 && nvfp4 + mxfp8 == 2U * schedule.size(),
          "expected both microscaled expert formats on the complete loop");
}

void check_transfer_nodes() {
    exec::Program program;
    exec::CudaProgramBinding binding;
    binding.readback = exec::CudaReadback::bound;
    std::array<unsigned, 2U * mini::kLogicalVisits> flags{};
    std::array<float, mini::kHiddenWidth> hidden{}, residual{};
    std::uint32_t picked{}, reply{};
    exec::append_cuda_memset(program, binding, {flags.data(), sizeof(flags), 0});
    exec::append_cuda_copy(program, binding, exec::CudaNodeKind::copy_d2d,
                           {residual.data(), hidden.data(), sizeof(hidden)});
    exec::append_cuda_copy(program, binding, exec::CudaNodeKind::copy_d2h_pinned,
                           {&reply, &picked, sizeof(reply)});
    const auto nodes = exec::cuda_detail::capture_plan(program, binding, true);
    check(nodes == std::vector<exec::CudaNodeBinding>{{exec::CudaNodeKind::memset, {}},
              {exec::CudaNodeKind::copy_d2d, {{0}}}, {exec::CudaNodeKind::copy_d2h_pinned, {{1}}}},
          "bound transfers changed native kind, order or completion edges");
    const auto read = [&]<class Args>(std::size_t index) {
        const auto& launch = program.launches.at(index);
        check(launch.fn == nullptr && launch.kernels == 0U && launch.args_bytes == sizeof(Args) &&
              launch.args_offset % alignof(Args) == 0,
              "transfer has an unexpected thunk or argument layout");
        Args args{};
        std::memcpy(&args, program.args.data() + launch.args_offset, sizeof(Args));
        return args;
    };
    const auto clear = read.template operator()<exec::CudaMemsetArgs>(0);
    const auto copy = read.template operator()<exec::CudaCopyArgs>(1);
    const auto readback = read.template operator()<exec::CudaCopyArgs>(2);
    check(clear.destination == flags.data() && clear.bytes == sizeof(flags) && clear.value == 0,
          "flag clear changed its binding");
    check(copy.destination == residual.data() && copy.source == hidden.data() && copy.bytes == sizeof(hidden),
          "D2D copy changed its binding");
    check(readback.destination == &reply && readback.source == &picked && readback.bytes == sizeof(reply),
          "greedy reply changed its binding");
}

namespace decode = strix::exec::cuda_decode;
namespace residual = strix::core::residual_mix;
namespace grouped_rows = strix::moe_core::grouped_rows;

struct FixtureWeights {
    // All fake tensors borrow host storage. Weight lookups still provide the
    // checkpoint's per-projection shape and stable, distinct aligned addresses.
    std::vector<std::uint16_t> bf = std::vector<std::uint16_t>(mini::kQueryWidth * mini::kHiddenWidth + 1024U);
    std::vector<float> fp = std::vector<float>(2U * mini::MiniLayers::core_width * mini::kHiddenWidth + 1024U);

    std::span<const std::uint16_t> bf16(std::string_view name) const {
        const auto offset = (std::hash<std::string_view>{}(name) % 64U) * 8U;
        auto size = bf.size() - offset;
        if (name.starts_with("layers.")) {
            const auto suffix = name.substr(name.find('.', 7) + 1);
            for (unsigned i = 0; i < 10U; ++i) {
                const auto spec = mini::mini_attention_projection_spec(static_cast<mini::MiniAttentionProjection>(i));
                if (suffix == spec.tensor_suffix) size = std::size_t{spec.input_width} * spec.output_width;
            }
            if (suffix == "attn.q_l2_clip.scale") size = mini::kQueryWidth;
            if (suffix == "attn.rel_l2_clip.scale") size = mini::kQueryHeads * mini::kRelativeWidth;
        }
        return std::span(bf).subspan(offset, size);
    }
    std::span<const float> f32(std::string_view name) const {
        return std::span(fp).subspan((std::hash<std::string_view>{}(name) % 64U) * 4U);
    }
};

namespace r1_geometry = strix::moe_core::microscaled_r1;

constexpr exec::Grid grid(unsigned x, unsigned threads = 256U, unsigned y = 1U, unsigned ty = 1U,
                          unsigned shared = 0U) {
    return {{x, y, 1U}, {threads, ty, 1U}, shared};
}

// When built with CUDA, use the actual catalogue's host binders. The
// SDK-free build supplies source-recorded geometry for the same typed leaves.
template <class A>
exec::KernelBinding bind_physical_fixture(const A& a) {
    using namespace decode;
    if constexpr (std::is_same_v<A, CheckedWaveRoute> || std::is_same_v<A, GroupedPlanLeaf> ||
                  std::is_same_v<A, RouteTapExchange>) {
        exec::KernelBinding wave;
        wave.fn = unexecuted_binding;
        if constexpr (std::is_same_v<A, CheckedWaveRoute>) {
            wave.identity = "cuda-sm120-checked-wave-mini-route-r1-v1";
            wave.grid = grid(1U, 128U);
        } else if constexpr (std::is_same_v<A, GroupedPlanLeaf>) {
            constexpr std::array names{"cuda-grouped-csr-count-v1", "cuda-grouped-csr-scan-v1",
                "cuda-grouped-csr-scatter-v1", "cuda-grouped-materialize-v1"};
            wave.identity = names.at(static_cast<std::size_t>(a.phase));
            wave.grid = grid(1U);
        } else {
            wave.identity = "cuda-route-tap-exchange-v1";
        }
        const auto bytes = std::as_bytes(std::span(&a, 1U));
        wave.args.assign(bytes.begin(), bytes.end());
        return wave;
    } else {
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
        return decode::bind(a);
#else
        auto describe = [&]() -> std::pair<std::string_view, exec::Grid> {
            const auto pointwise = [](std::size_t n) { return grid(static_cast<unsigned>((n + 255U) / 256U)); };
            if constexpr (std::is_same_v<A, Embedding>) return {"lookup-bf16-f32", grid(1U)};
            if constexpr (std::is_same_v<A, RmsBf16>) return {"rms-bf16-image-r1", grid(a.groups)};
            if constexpr (std::is_same_v<A, AppendKvBf16>) return {"prepared-bf16-kv-step", grid(a.kv_heads, a.width)};
            if constexpr (std::is_same_v<A, PreparedFlashBf16>) return {"prepared-flash-pad2-d128", grid(a.heads, 256U, 1, 1, 35840U)};
            if constexpr (std::is_same_v<A, AppendKvHostBf16>) return {"prepared-bf16-kv-host-r1", grid(a.args.kv_heads, a.args.width)};
            if constexpr (std::is_same_v<A, PreparedFlashHostBf16>)
                return {"prepared-flash-pad2-d128-host-r1", grid(a.args.heads, 256U, 1, 1, 35840U)};
            if constexpr (std::is_same_v<A, PreparedFinite>) return {"prepared-finite-check", pointwise(a.heads * a.relative_extent)};
            if constexpr (std::is_same_v<A, CausalConvolution>) return {"causal-convolution-r1", pointwise(a.plan.channels)};
            if constexpr (std::is_same_v<A, RoundedCausalConvolution>)
                return {"causal-convolution-round-r1", pointwise(a.convolution.plan.channels)};
            if constexpr (std::is_same_v<A, HeadClipBf16>) return {"head-clip-bf16-r1", grid(a.heads, a.width <= 32U ? 32U : 128U)};
            if constexpr (std::is_same_v<A, residual::NormalizeArgs>) return {"residual-bank-norm-r1", grid(1U)};
            if constexpr (std::is_same_v<A, residual::MixArgs>) return {"residual-bank-mix-r1", grid(1U, a.bank_count == 1 ? 256U : 128U, 1U, a.bank_count)};
            if constexpr (std::is_same_v<A, residual::AddArgs>) return {"residual-add-r1", pointwise(a.elements)};
            if constexpr (std::is_same_v<A, residual::ControllerArgs>) return {"residual-controller-r1", pointwise(a.elements)};
            if constexpr (std::is_same_v<A, residual::NormalizePairArgs>) return {"controller-pair-norm-r1", grid(1U, 256U, 2U)};
            if constexpr (std::is_same_v<A, Bf16Round>) return {"pointwise-bf16-round", pointwise(a.elements)};
            if constexpr (std::is_same_v<A, Bf16SigmoidProduct>) return {"pointwise-bf16-sigmoid-product", pointwise(a.elements)};
            if constexpr (std::is_same_v<A, Bf16Softcap>) return {"pointwise-bf16-softcap", pointwise(a.elements)};
            if constexpr (std::is_same_v<A, Bf16RelativeProjection16>) return {"relative-bf16-d16", pointwise(a.heads * a.relative_extent)};
            if constexpr (std::is_same_v<A, Bf16DenseGroup>) {
                unsigned outputs = 0;
                for (unsigned i = 0; i < a.count; ++i) outputs += a.projections[i].outputs;
                if (a.mode == Bf16DenseMode::fp32) return {"bf16-r1-fp32-group", grid((outputs + 7U) / 8U)};
                const auto name = a.mode == Bf16DenseMode::mma3 ? "bf16-r1-mma3-group" : "bf16-r1-mma3-group-round";
                return {name, grid(outputs / 16U)};
            }
            if constexpr (std::is_same_v<A, Bf16Fp32Epilogue>) {
                const auto name = a.form == graph::Form::softcap ? "bf16-r1-fp32-softcap" : "bf16-r1-fp32-round";
                return {name, grid((a.dense.projections[0].outputs + 7U) / 8U)};
            }
            if constexpr (std::is_same_v<A, PackedGateUp>) {
                const auto name = a.format == strix::weight::microscaled::MatrixFormat::nvfp4
                    ? "packed-nvfp4-gate-up-r1-k8-pdl" : "packed-mxfp8-gate-up-r1-k8-pdl";
                return {name, grid(r1_geometry::gate_blocks(a.args.intermediate, r1_geometry::kCudaGeometry.gate_columns), 256, 8)};
            }
            if constexpr (std::is_same_v<A, PackedDownCombine>) {
                const auto name = a.format == strix::weight::microscaled::MatrixFormat::nvfp4
                    ? "packed-nvfp4-down-combine-r1-k8-pdl" : "packed-mxfp8-down-combine-r1-k8-pdl";
                return {name, grid(r1_geometry::down_blocks(a.args.hidden, r1_geometry::kCudaGeometry.down_columns))};
            }
            if constexpr (std::is_same_v<A, GreedyPartial>) return {"greedy-partial-r1", grid(a.chunks)};
            if constexpr (std::is_same_v<A, GreedyFinal>) return {"greedy-final-r1", grid(1U)};
            if constexpr (std::is_same_v<A, grouped_rows::GatherArgs>)
                return {"grouped-gather-r1", grid((a.width + 255U) / 256U, 256U, a.assignments)};
            if constexpr (std::is_same_v<A, GroupedProjection>) {
                const auto name = a.format == strix::weight::microscaled::MatrixFormat::nvfp4
                    ? "grouped-nvfp4-projection-r1" : "grouped-mxfp8-projection-r1";
                return {name, {{a.args.outputs / 32U, 1U, a.args.count}, {128U, 1U, 1U}, 0U}};
            }
            if constexpr (std::is_same_v<A, GroupedActivation>)
                return {"grouped-swiglu-r1", grid(static_cast<unsigned>((a.args.width + 127U) / 128U),
                                                 128U, static_cast<unsigned>(a.args.rows))};
            if constexpr (std::is_same_v<A, grouped_rows::CombineArgs>)
                return {"grouped-combine-r1", grid((a.width + 255U) / 256U, 256U, a.rows)};
            throw std::logic_error("unknown fixture kernel binding");
        };
        const auto [symbol, shape] = describe();
        exec::KernelBinding binding;
        binding.identity = "cuda-sm120-" + std::string(symbol) + "-v1";
        binding.fn = unexecuted_binding;
        binding.grid = shape;
        const auto bytes = std::as_bytes(std::span(&a, 1U));
        binding.args.assign(bytes.begin(), bytes.end());
        return binding;
#endif
    }
}

template <class A>
A launch_args(const exec::CudaRecipeProgram& recipe, const exec::BoundLaunch& launch) {
    check(launch.args_bytes == sizeof(A), "unexpected kernel argument size");
    A args{};
    std::memcpy(&args, recipe.program.args.data() + launch.args_offset, sizeof(args));
    return args;
}

struct PhysicalTrace {
    const exec::CudaRecipeProgram& result;
    std::size_t index = 0;
    std::size_t count_kernels = 0;
    bool previous_kernel = false;

    template <class A>
    A read(std::string_view identity, exec::Grid shape, exec::CudaNodeKind kind, bool pdl = false,
           bool host_action = false) {
        const auto& launch = result.program.launches.at(index);
        const auto& node = result.binding.nodes.at(index);
        const bool thunk = kind == exec::CudaNodeKind::kernel;
        const bool kernel = thunk && !host_action;
        if (launch.identity != identity || node.kind != kind || launch.grid.blocks != shape.blocks ||
            launch.grid.threads != shape.threads || launch.grid.shared_bytes != shape.shared_bytes ||
            launch.kernels != (kernel ? 1U : 0U) || (launch.fn != nullptr) != thunk || launch.args_bytes != sizeof(A))
            throw std::runtime_error("physical node " + std::to_string(index) + " expected " + std::string(identity) +
                                     ", got " + launch.identity);
        if (index == 0) check(node.after.empty() && launch.after == exec::no_launch, "unexpected first node edge");
        else {
            const auto edge = pdl && previous_kernel && !result.program.key.tap
                ? exec::CudaEdgeKind::programmatic : exec::CudaEdgeKind::completion;
            check(node.after == std::vector<exec::CudaDependency>{{index - 1U, edge}} && launch.after == index - 1U,
                  "native node changed its stream predecessor or PDL edge");
        }
        ++index;
        count_kernels += kernel;
        previous_kernel = kernel;
        return launch_args<A>(result, launch);
    }

    template <class A>
    A kernel(std::string_view symbol, exec::Grid shape, bool pdl = false) {
        return read<A>("cuda-sm120-" + std::string(symbol) + "-v1", shape, exec::CudaNodeKind::kernel, pdl);
    }
    void copy(float* to, const float* from) {
        const auto args = read<exec::CudaCopyArgs>("cuda-native-copy-d2d-v1", {}, exec::CudaNodeKind::copy_d2d);
        check(args.destination == to && args.source == from && args.bytes == mini::kHiddenWidth * sizeof(float),
              "resident image transfer binding");
    }
};

struct EpilogueTrace {
    const exec::CudaRecipeProgram& original;
    struct Pair {
        std::size_t head;
        std::vector<std::size_t> tails;
        std::string_view entry;
        std::function<bool(exec::CudaProgramRecipe&)> emit;
    };
    std::vector<Pair> pairs;

    template <class Head, class Tail>
    void add(std::size_t head_index, std::size_t tail_index, Head head, Tail tail, graph::Form form,
             std::span<const std::size_t> other_tails = {}) {
        const auto& producer = original.program.launches.at(head_index);
        const auto& consumer = original.program.launches.at(tail_index);
        const auto& edges = original.binding.nodes.at(head_index).after;
        const bool pdl = !edges.empty() && edges.front().kind == exec::CudaEdgeKind::programmatic;
        const auto& catalog = exec::Catalog::builtin(exec::Backend::cuda);
        const auto op = std::is_same_v<Head, decode::Bf16DenseGroup> ? graph::Op::linear : graph::Op::elementwise;
        std::array declaration{
            graph::StageDecl{.id = "producer", .op = op, .entry = producer.identity, .out = "raw"},
            graph::StageDecl{.id = "neighbour", .op = graph::Op::argmax, .entry = "neighbour", .out = "independent"},
            graph::StageDecl{.id = "tail", .entry = consumer.identity, .in = {"raw"}, .out = "result"},
        };
        const graph::StageGraph stage_graph{.before = declaration};
        graph::PlanStage binding{.stage = &declaration.back()};
        const auto plan = [&](bool enabled, const exec::Catalog& lookup, unsigned rows = 1) {
            decode::project_exact_epilogue(binding, lookup, producer.identity, form, enabled);
            return graph::launch_plan(stage_graph, graph::PlanRequest{.rows = rows, .stages = std::span(&binding, 1)});
        };
        const auto baseline = graph::launch_plan(stage_graph, 0, 1, graph::FusionPolicy{});
        const exec::Catalog empty;
        check(baseline.kernel_count() == 3U && plan(false, catalog).kernel_count() == 3U &&
              plan(true, empty).kernel_count() == 3U && plan(true, catalog, 2).kernel_count() == 3U,
              "unavailable epilogue removed a launch");
        declaration[1].in[0] = "raw";
        check(plan(true, catalog).kernel_count() == 3U, "live unrounded image permits fusion");
        declaration[1].in[0] = {};
        auto fused = plan(true, catalog);
        check(fused.kernel_count() == 2U && fused.launches[0].epilogue == std::vector<std::uint32_t>{2U},
              "exact pair did not fuse");
        const auto entry = fused.launches[0].entry;
        fused.instances.clear();  // The recipe lowerer needs only the chosen launch and chain.
        std::vector<std::size_t> tails{tail_index};
        tails.insert(tails.end(), other_tails.begin(), other_tails.end());
        pairs.push_back({head_index, std::move(tails), entry,
            [head, tail, pdl, fused = std::move(fused)](auto& recipe) {
                const auto bind_leaf = [](const auto& args) { return bind_physical_fixture(args); };
                if constexpr (std::is_same_v<Tail, std::array<decode::Bf16Round, 2>>) {
                    const std::span<const decode::Bf16Round> bundle = tail;
                    check(!decode::lower_exact_epilogue(recipe, fused, fused.launches.front(), head,
                                                        bundle.first(1), pdl, bind_leaf), "partial round bundle fused");
                    return decode::lower_exact_epilogue(recipe, fused, fused.launches.front(), head,
                                                        bundle, pdl, bind_leaf);
                } else {
                    return decode::lower_exact_epilogue(recipe, fused, fused.launches.front(), head,
                                                        tail, pdl, bind_leaf);
                }
            }});
    }

    static void check_dense(const decode::Bf16DenseGroup& actual, const decode::Bf16DenseGroup& expected) {
        check(actual.input == expected.input && actual.input_width == expected.input_width &&
              actual.count == expected.count && actual.mode == expected.mode, "fused dense binding");
        for (unsigned i = 0; i < expected.count; ++i) {
            const auto& a = actual.projections[i];
            const auto& e = expected.projections[i];
            check(a.weights == e.weights && a.outputs == e.outputs && a.output == e.output &&
                  a.round_bf16 == e.round_bf16, "fused projection image binding");
        }
    }

    void check_lowering(const exec::CudaRecipeProgram& actual, bool reject_fp32 = false) const {
        exec::CudaProgramRecipe recipe;
        const auto enabled = [reject_fp32](const Pair& pair) {
            return !reject_fp32 || !pair.entry.starts_with("cuda-sm120-bf16-r1-fp32-");
        };
        std::vector<std::size_t> replaced;
        std::size_t removed = 0;
        for (std::size_t i = 0; i < original.program.launches.size(); ++i) {
            const auto head = std::ranges::find(pairs, i, &Pair::head);
            if (std::ranges::any_of(pairs, [&](const auto& pair) {
                    return enabled(pair) && std::ranges::find(pair.tails, i) != pair.tails.end();
                })) {
                ++removed;
                continue;
            }
            const auto& launch = original.program.launches[i];
            recipe.visit(launch.layer);
            recipe.stage(launch.entry);
            if (head != pairs.end() && enabled(*head)) {
                check(head->emit(recipe), "planned CUDA epilogue failed to lower");
                replaced.push_back(i - removed);
                continue;
            }
            const auto& node = original.binding.nodes[i];
            const auto bytes = std::span(original.program.args).subspan(launch.args_offset, launch.args_bytes);
            if (node.kind == exec::CudaNodeKind::kernel) {
                exec::KernelBinding unmodified;
                unmodified.identity = launch.identity;
                unmodified.grid = launch.grid;
                unmodified.fn = launch.fn;
                unmodified.args.assign(bytes.begin(), bytes.end());
                recipe.kernel(std::move(unmodified), !node.after.empty() &&
                              node.after.front().kind == exec::CudaEdgeKind::programmatic);
            } else if (node.kind == exec::CudaNodeKind::memset) {
                exec::CudaMemsetArgs args{};
                std::memcpy(&args, bytes.data(), sizeof(args));
                recipe.clear(args);
            } else {
                exec::CudaCopyArgs args{};
                std::memcpy(&args, bytes.data(), sizeof(args));
                recipe.copy(node.kind, args);
            }
        }
        const auto expected = recipe.compile({});
        check(removed == (reject_fp32 ? 208U : 257U) &&
              actual.program.launches.size() == original.program.launches.size() - removed &&
              actual.program.predicted.launches == original.program.predicted.launches - removed &&
              actual.binding.nodes.size() == actual.program.launches.size(),
              "fused decode node or launch count");
        for (std::size_t i = 0; i < expected.program.launches.size(); ++i) {
            const auto& a = actual.program.launches[i];
            const auto& e = expected.program.launches[i];
            const auto& actual_node = actual.binding.nodes[i];
            const auto& expected_node = expected.binding.nodes[i];
            check(a.identity == e.identity && a.layer == e.layer && a.grid.blocks == e.grid.blocks &&
                  a.grid.threads == e.grid.threads && a.grid.shared_bytes == e.grid.shared_bytes &&
                  a.args_bytes == e.args_bytes && a.kernels == e.kernels && a.fn == e.fn &&
                  actual_node.kind == expected_node.kind && actual_node.after == expected_node.after,
                  "lowered launch order, geometry or PDL edge");
        }
        for (const auto i : replaced) {
            const auto& a = actual.program.launches[i];
            const auto& e = expected.program.launches[i];
            check(a.chain_identity == e.chain_identity, "fused chain identity");
            if (e.identity.starts_with("cuda-sm120-bf16-r1-fp32-")) {
                const auto x = launch_args<decode::Bf16Fp32Epilogue>(actual, a);
                const auto y = launch_args<decode::Bf16Fp32Epilogue>(expected, e);
                check(x.form == y.form && x.cap == y.cap, "FP32 epilogue form or softcap");
                check_dense(x.dense, y.dense);
            } else if (e.identity == "cuda-sm120-causal-convolution-round-r1-v1") {
                const auto x = launch_args<decode::RoundedCausalConvolution>(actual, a).convolution;
                const auto y = launch_args<decode::RoundedCausalConvolution>(expected, e).convolution;
                check(x.plan.channels == y.plan.channels && x.plan.taps == y.plan.taps &&
                      x.plan.dynamic_groups == y.plan.dynamic_groups && x.plan.residual_weight == y.plan.residual_weight &&
                      x.plan.output_divisor == y.plan.output_divisor && x.input == y.input &&
                      x.static_weights == y.static_weights && x.dynamic_weights == y.dynamic_weights &&
                      x.history == y.history && x.output == y.output && x.round_bf16 == y.round_bf16,
                      "fused convolution images or history");
            } else {
                check_dense(launch_args<decode::Bf16DenseGroup>(actual, a),
                            launch_args<decode::Bf16DenseGroup>(expected, e));
            }
        }
    }
};

#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
// The exact R1 latency forms replace their incumbent node one for one: same node list, same edges,
// every other node unchanged; each incumbent bank readout, controller down projection and (packed
// recipe) expert pair moves to its form.
void check_latency_forms(const exec::CudaRecipeProgram& base, const exec::CudaRecipeProgram& forms, bool grouped) {
    const auto& a = base.program.launches;
    const auto& b = forms.program.launches;
    check(a.size() == b.size() && base.binding.nodes.size() == forms.binding.nodes.size(), "forms keep the node list");
    if (a.size() != b.size()) return;
    const std::pair<std::string_view, std::string_view> map[]{
        {"cuda-sm120-residual-bank-mix-r1-v1", "cuda-sm120-residual-bank-softmax-tiled-r1-v1"},
        {"cuda-sm120-bf16-r1-mma3-group-round-v1", "cuda-sm120-bf16-r1-mma3-output-split-round-v1"},
        {"cuda-sm120-packed-nvfp4-gate-up-r1-k8-pdl-v1", "cuda-sm120-packed-nvfp4-gate-up-r1-k8-pipe-v1"},
        {"cuda-sm120-packed-mxfp8-gate-up-r1-k8-pdl-v1", "cuda-sm120-packed-mxfp8-gate-up-r1-k8-pipe-v1"},
        {"cuda-sm120-packed-nvfp4-down-combine-r1-k8-pdl-v1", "cuda-sm120-packed-nvfp4-down-combine-r1-k8-pipe-v1"},
        {"cuda-sm120-packed-mxfp8-down-combine-r1-k8-pdl-v1", "cuda-sm120-packed-mxfp8-down-combine-r1-k8-pipe-v1"},
    };
    std::array<unsigned, std::size(map)> moved{}, incumbent{};
    for (std::size_t i = 0; i < a.size(); ++i) {
        check(a[i].after == b[i].after, "forms keep every stream edge");
        std::size_t hit = std::size(map);
        for (std::size_t m = 0; m < std::size(map); ++m)
            if (a[i].identity == map[m].first) hit = m;
        // Only the controller down projection (kControllerRank outputs = 4 MMA3 tiles) is narrow; the other
        // rounded groups (fused attention projections) keep the incumbent tile.
        if (hit == 1U && a[i].grid.blocks[0] != mini::kControllerRank / 16U) hit = std::size(map);
        if (hit == std::size(map)) {
            check(a[i].identity == b[i].identity && a[i].grid.blocks == b[i].grid.blocks &&
                      a[i].grid.threads == b[i].grid.threads && a[i].args_bytes == b[i].args_bytes,
                  "forms leave every other node unchanged");
            continue;
        }
        ++incumbent[hit];
        moved[hit] += b[i].identity == map[hit].second;
    }
    for (std::size_t m = 0; m < std::size(map); ++m)
        if (moved[m] != incumbent[m])
            std::cout << "inkling2_program_host forms " << map[m].first << " moved " << moved[m] << " of "
                      << incumbent[m] << '\n';
    for (std::size_t m = 0; m < std::size(map); ++m)
        check(moved[m] == incumbent[m] && (m >= 2U || incumbent[m] != 0U), "every incumbent moves to its form");
    check(grouped || incumbent[2] + incumbent[3] == 48U, "packed recipe moves all 48 gate-up nodes");
    std::cout << "inkling2_program_host latency_forms nodes=" << b.size() << " bank_tiled=" << moved[0]
              << " output_split=" << moved[1] << " moe_pipe=" << (moved[2] + moved[3] + moved[4] + moved[5])
              << " grouped=" << grouped << '\n';
}
#endif

struct VerifyFixtureBinding {
    bool reject_convolution = false;

    template <class A>
    bool operator()(exec::CudaProgramRecipe& recipe, const A& args, const exec::StepBlock* descriptor,
                    unsigned rows) const {
        static const auto thunks = [] {
            decode::wr1::Thunks table{};
            for (auto& width : table) width.fill(unexecuted_binding);
            return table;
        }();
        // WR2's state/route launch thunks are declared only for CUDA. Record
        // their typed operands and native names here without enqueuing them.
        const auto other = [this](exec::CudaProgramRecipe& leaves, const auto& a, unsigned width) {
            using Leaf = std::remove_cvref_t<decltype(a)>;
            const auto suffix = "-x3-r" + std::to_string(width) + "-v1";
            exec::KernelBinding binding;
            binding.fn = unexecuted_binding;
            binding.predicted.launches = 1U;
            if constexpr (std::is_same_v<Leaf, mini::MiniCudaVerifyEmbedding>) {
                binding.identity = "cuda-inkling2-step-embedding-bf16-v1";
                binding.grid = grid(width);
            } else if constexpr (std::is_same_v<Leaf, strix::attn_core::PreparedStepKv>) {
                binding.identity = "cuda-sm120-kv-append-bf16" + suffix;
                binding.grid = grid(a.kv_heads, a.width, width);
            } else if constexpr (std::is_same_v<Leaf, strix::attn_core::PreparedStepFlash>) {
                binding.identity = "cuda-sm120-prepared-attention-bf16-pad2" + suffix;
                binding.grid = grid(a.heads, 256U, width, 1U, 35840U);
            } else if constexpr (std::is_same_v<Leaf, strix::attn_core::PreparedStepFinite>) {
                binding.identity = "cuda-sm120-prepared-finite" + suffix;
                binding.grid = grid(width * a.heads * a.relative_extent / 256U);
            } else if constexpr (std::is_same_v<Leaf, strix::core::causal_conv::StepArgs>) {
                if (reject_convolution) return false;
                binding.identity = std::string("cuda-sm120-causal-convolution-") +
                    (a.round_bf16 ? "bf16" : "f32") + suffix;
                binding.grid = grid(static_cast<unsigned>((a.plan.channels + 255U) / 256U));
            } else if constexpr (std::is_same_v<Leaf, decode::StepCheckedWaveRoute>) {
                binding.identity = "cuda-sm120-checked-wave-mini-route" + suffix;
                binding.grid = grid((width + 3U) / 4U, 128U);
            } else if constexpr (std::is_same_v<Leaf, decode::StepPackedGateUp> ||
                                 std::is_same_v<Leaf, decode::StepPackedDownCombine>) {
                constexpr bool gate = std::is_same_v<Leaf, decode::StepPackedGateUp>;
                const auto format = a.format == strix::weight::microscaled::MatrixFormat::nvfp4 ? "nvfp4" : "mxfp8";
                binding.identity = std::string("cuda-sm120-packed-") + format +
                    (gate ? "-gate-up-k8" : "-down-combine-k8") + "-x3-r" + std::to_string(width) + "-v2";
                binding.grid = gate ? grid(r1_geometry::gate_blocks(a.args.batch.intermediate,
                                               r1_geometry::kCudaGeometry.gate_columns), 256U, 8U * width)
                                    : grid(r1_geometry::down_blocks(a.args.batch.hidden,
                                               r1_geometry::kCudaGeometry.down_columns), 256U, width);
            } else if constexpr (std::is_same_v<Leaf, strix::cuda::core::StepRowStoreBf16>) {
                binding.identity = "cuda-sm120-step-row-store-bf16" + suffix;
                binding.grid = grid(strix::cuda::core::step_row_store_blocks(a.width), 256U, width);
            } else {
                throw std::logic_error("unknown verify fixture kernel binding");
            }
            const auto bytes = std::as_bytes(std::span(&a, 1U));
            binding.args.assign(bytes.begin(), bytes.end());
            leaves.kernel(std::move(binding), false, width);
            return true;
        };
        return mini::append_mini_verify_leaf(recipe, args, descriptor, rows, thunks, other);
    }
};

std::string verify_identity(std::string_view scalar, unsigned rows) {
    constexpr std::pair<std::string_view, std::string_view> forms[]{
        {"bf16-r1-fp32-group", "dense-fp32"}, {"bf16-r1-mma3-group", "dense-mma3"},
        {"bf16-r1-mma3-group-round", "dense-mma3-round"}, {"rms-bf16-image-r1", "rms-bf16"},
        {"residual-bank-mix-r1", "residual-mix"}, {"residual-bank-norm-r1", "bank-norm"},
        {"controller-pair-norm-r1", "pair-norm"}, {"residual-controller-r1", "controller"},
        {"residual-add-r1", "add"}, {"head-clip-bf16-r1", "head-clip"}, {"relative-bf16-d16", "relative-d16"},
        {"pointwise-bf16-round", "round"}, {"pointwise-bf16-sigmoid-product", "sigmoid-product"},
        {"pointwise-bf16-softcap", "softcap"}, {"greedy-partial-r1", "greedy-partial"},
        {"greedy-final-r1", "greedy-final"},
    };
    constexpr std::pair<std::string_view, std::string_view> stateful[]{
        {"prepared-bf16-kv-step", "kv-append-bf16"}, {"prepared-flash-pad2-d128", "prepared-attention-bf16-pad2"},
        {"prepared-finite-check", "prepared-finite"}, {"causal-convolution-r1", "causal-convolution-f32"},
        {"causal-convolution-round-r1", "causal-convolution-bf16"},
        {"checked-wave-mini-route-r1", "checked-wave-mini-route"},
        {"packed-nvfp4-gate-up-r1-k8", "packed-nvfp4-gate-up-k8"},
        {"packed-mxfp8-gate-up-r1-k8", "packed-mxfp8-gate-up-k8"},
        {"packed-nvfp4-down-combine-r1-k8", "packed-nvfp4-down-combine-k8"},
        {"packed-mxfp8-down-combine-r1-k8", "packed-mxfp8-down-combine-k8"},
    };
    if (scalar.starts_with("cuda-native-")) return std::string(scalar);
    constexpr std::string_view prefix = "cuda-sm120-", version = "-v1";
    check(scalar.starts_with(prefix) && scalar.ends_with(version), "unknown scalar kernel recipe");
    const auto family = scalar.substr(prefix.size(), scalar.size() - prefix.size() - version.size());
    if (family == "embedding-bf16-r1" || family == "lookup-bf16-f32") return "cuda-inkling2-step-embedding-bf16-v1";
    const auto suffix = "-r" + std::to_string(rows) + "-v1";
    for (const auto& [from, to] : forms) {
        if (family != from) continue;
        // The fp32 dense step leaf reads each weight once per row tile (v2); the other WR1 forms keep v1.
        if (to == "dense-fp32") return "cuda-wr1-dense-fp32-r" + std::to_string(rows) + "-v2";
        return "cuda-wr1-" + std::string(to) + suffix;
    }
    for (const auto& [from, to] : stateful) {
        if (family != from) continue;
        // The packed MoE step leaves are the v2 rows launch of the decode body.
        if (to.starts_with("packed-")) return "cuda-sm120-" + std::string(to) + "-x3-r" + std::to_string(rows) + "-v2";
        return "cuda-sm120-" + std::string(to) + "-x3" + suffix;
    }
    throw std::logic_error("unknown scalar leaf " + std::string(scalar));
}

// A verify window of `rows` rows lowers every scalar leaf of the incumbent R1 recipe on its width-bound
// form, in the same order (plus the row-major logit readback). With drafter taps on, every width (R1 too)
// adds the four descriptor-positioned tap stores (visits 5, 23, 41 and the final normed row).
void check_verify_plan(unsigned position, FixtureWeights& weights, unsigned rows, std::span<const std::string> scalar,
                       bool reject = false, bool taps = false) {
    using namespace mini;
    check(rows != 1U || taps, "R1 lowering is checked by check_physical_plan");
    const MiniSessionStatePlan layout(position + rows, rows);
    const auto kv_elements = static_cast<std::size_t>(layout.kv_elements_per_plane());
    const auto keys = std::make_unique_for_overwrite<std::uint16_t[]>(kv_elements);
    const auto values = std::make_unique_for_overwrite<std::uint16_t[]>(kv_elements);
    std::vector<float> history(static_cast<std::size_t>(layout.convolution_elements())), history_backup(history.size());
    std::vector<std::uint16_t> kv_backup(mini_resident_max_kv_checkpoint_elements());
    MiniResidentSessionState state(layout, {{keys.get(), kv_elements}, {values.get(), kv_elements}}, history);
    struct PositionOnly {
        bool try_step_graph(MiniResidentChunk, const std::function<void()>&) { return true; }
        void finish() {}
        void copy(std::span<const float>, std::span<float>) {}
        void copy(std::span<const std::uint16_t>, std::span<std::uint16_t>) {}
    };
    // The frontier follows successful graph-style appends without touching buffers.
    for (unsigned row = 0; row < position; ++row) {
        PositionOnly skip;
        state.append(skip, {row, 1U}, history_backup, kv_backup, [] {}, [] {});
    }

    std::vector<float> activations(mini_resident_activation_elements(rows));
    std::vector<float> controller(decode::controller_scratch_floats(kHiddenWidth, kControllerRank, rows));
    std::vector<float> attention(mini_resident_attention_elements(rows));
    std::vector<float> lag(kMiniConvLayerElements * kPhysicalLayers);
    std::vector<strix::moe_core::device_grouped::ExpertWeights> matrices(kPhysicalLayers * kExperts);
    std::vector<unsigned> ids(rows * kExpertsPerToken);
    std::vector<float> scores(rows * kExperts), route_weights(rows * kExpertsPerToken);
    std::vector<std::uint16_t> expert_activation(rows * kExpertsPerToken * kExpertWidth);
    std::vector<unsigned> flags(2U * kLogicalVisits * rows);
    std::vector<std::uint16_t> tap_rows(taps ? MiniDSparkTapPlan::elements(position + rows) : 0U);
    MiniCudaStepStorage scratch{rows, controller, attention, lag.data(), matrices, ids.data(),
        scores.data(), route_weights.data(), expert_activation.data(), flags};
    if (taps) {
        scratch.draft_taps = tap_rows.data();
        scratch.draft_tap_positions = position + rows;
    }
    std::vector<float> embeddings(kHiddenWidth * rows), hidden(embeddings.size()), logits(kOutputVocabulary * rows);
    std::vector<float> logits_reply(logits.size());
    std::array<strix::head_argmax::Pick, 128U> partials{};
    std::array<std::uint32_t, kMiniVerifyMaxRows> picked{}, reply{};
    const MiniCudaStepRows buffers{embeddings, hidden, logits, partials.data(), picked.data(), reply.data()};
    std::vector<strix::head_argmax::Pick2> verify_partials(strix::head_argmax::workspace2_bytes(rows) /
                                                          sizeof(strix::head_argmax::Pick2));
    std::vector<exec::RowOut> verify_picked(rows), verify_reply(rows);
    const MiniCudaVerifyRows replies{verify_partials.data(), verify_picked, verify_reply, logits_reply};
    std::array<strix::state::SlotEntry, MiniCudaVerifyState::entries> directory{};
    std::vector<std::uint16_t> pending_keys(kLogicalVisits * rows * kKvWidth), pending_values(pending_keys.size());
    std::vector<float> pending_history(history.size() * rows);
    const MiniCudaVerifyState verify_state{{directory.data(), MiniCudaVerifyState::entries, 1U}, rows,
        pending_keys.data(), pending_values.data(), pending_history.data(), layout};
    exec::StepBlock descriptor{};
    exec::CudaRecipeProgram result;
    const auto bind = [](const auto& args) { return bind_physical_fixture(args); };
    const auto dispatch = [&](MiniResidentChunk chunk, const auto& build) {
        check(rows == chunk.rows, "verify lowering width");
        result = build().compile(mini_cuda_step_key(chunk.rows));
        return true;
    };
    const auto finish = [] {};
    const VerifyFixtureBinding bind_verify{reject};
    MiniCudaStepLowering lower(weights, scratch, &descriptor, bind, dispatch, finish, bind_verify, MiniCudaStepMode{},
                               &verify_state);
    MiniResidentForward forward(lower, state, {activations, history_backup, kv_backup, tap_rows});
    try {
        forward.append({position, rows}, buffers.embeddings, buffers.hidden,
            [&](MiniResidentChunk, MiniConstRows h) { lower.head(h, buffers, replies); }, [] {},
            [&] { lower.embed(buffers.embeddings); });
        check(!reject, "unbound verify convolution compiled");
    } catch (const std::invalid_argument& error) {
        check(reject && std::string_view(error.what()) ==
              "Inkling verify R=4 needs shared CUDA leaves: causal_convolution (slot- and parent-indexed history)",
              error.what());
        return;
    }

    // Drafter taps: four stores per step, at the positions of the step descriptor (never a fixed position).
    unsigned tap_nodes = 0;
    for (std::size_t i = 0; i < result.program.launches.size(); ++i) {
        const auto& launch = result.program.launches[i];
        if (!launch.identity.starts_with("cuda-sm120-step-row-store-bf16")) continue;
        check(launch.identity == "cuda-sm120-step-row-store-bf16-x3-r" + std::to_string(rows) + "-v1" &&
              launch.rows == rows, "tap store width");
        strix::cuda::core::StepRowStoreBf16 store{};
        std::memcpy(&store, result.program.args.data() + launch.args_offset, sizeof(store));
        check(store.step == &descriptor && store.width == kHiddenWidth && store.pitch == MiniDSparkTapPlan::width &&
              store.positions == position + rows && store.output == tap_rows.data() + tap_nodes * kHiddenWidth,
              "tap store binds the descriptor, its tap slice and the position-major pitch");
        ++tap_nodes;
    }
    check(tap_nodes == (taps ? MiniDSparkTapPlan::count : 0U), "one tap store per drafter tap");
    if (taps) {
        std::cout << "inkling2_program_host draft taps R=" << rows << " position=" << position
                  << " stores=" << tap_nodes << " nodes=" << result.program.launches.size() << '\n';
        return;
    }

    PhysicalTrace trace{result};
    const auto clear = trace.read<exec::CudaMemsetArgs>("cuda-native-memset-v1", {}, exec::CudaNodeKind::memset);
    check(clear.destination == flags.data() && clear.bytes == std::span(flags).size_bytes() && clear.value == 0,
          "native flag binding");
    {
        const auto suffix = "-x3-r" + std::to_string(rows) + "-v1";
        check(result.program.key == exec::ProgramKey{.phase = exec::Phase::verify, .rows = rows} &&
              result.program.descriptor.rows == rows && result.program.capturable &&
              result.program.launches.size() == scalar.size() + 1U && result.program.predicted.launches == 2119U,
              "verify graph width or launch count");
        for (std::size_t i = 0; i < scalar.size(); ++i) {
            const auto& launch = result.program.launches[i];
            const auto expected = verify_identity(scalar[i], rows);
            if (launch.identity != expected || launch.rows != rows)
                throw std::runtime_error("verify node " + std::to_string(i) + " expected " + expected +
                                         ", got " + launch.identity);
            if (i != 0U)
                check(result.binding.nodes[i].after == std::vector<exec::CudaDependency>{{i - 1U}},
                      "verify graph requires completion-ordered leaves");
        }
        const auto embed = trace.read<MiniCudaVerifyEmbedding>("cuda-inkling2-step-embedding-bf16-v1", grid(rows),
                                                             exec::CudaNodeKind::kernel);
        check(embed.step == &descriptor && embed.table == weights.bf16("embed.weight").data() &&
              embed.output == embeddings.data() && embed.rows == rows, "verify embedding descriptor binding");
        const auto norm = trace.read<decode::wr1::LaunchArgs<decode::RmsBf16>>(
            "cuda-wr1-rms-bf16-r" + std::to_string(rows) + "-v1", grid(1U, 256U, rows), exec::CudaNodeKind::kernel);
        check(norm.descriptor == &descriptor && norm.scalar.input == embed.output && norm.scalar.output == embed.output,
              "verify RMS binding");
        const auto locate = [&](std::string_view identity, std::size_t visit) {
            const auto matched = std::ranges::find_if(result.program.launches, [&](const auto& launch) {
                return launch.identity == identity && launch.layer == static_cast<std::int32_t>(visit);
            });
            check(matched != result.program.launches.end(), "missing verify stateful leaf");
            trace.index = static_cast<std::size_t>(matched - result.program.launches.begin());
        };
        for (const auto visit : mini_schedule()) {
            const auto index = visit.logical_layer;
            const auto cache = verify_state.cache(visit, state.visit(visit).kv);
            const auto kv_name = "cuda-sm120-kv-append-bf16" + suffix;
            locate(kv_name, index);
            const auto kv = trace.read<strix::attn_core::PreparedStepKv>(kv_name, grid(kKvHeads, kHeadWidth, rows),
                                                                       exec::CudaNodeKind::kernel);
            const auto attn_name = "cuda-sm120-prepared-attention-bf16-pad2" + suffix;
            const auto attention_args = trace.read<strix::attn_core::PreparedStepFlash>(
                attn_name, grid(kQueryHeads, 256U, rows, 1U, 35840U), exec::CudaNodeKind::kernel);
            check(kv.step == &descriptor && attention_args.step == &descriptor &&
                  kv.cache.keys.slots.table.entries == directory.data() &&
                  kv.cache.keys.slots.instance == 6U * index && kv.cache.values.slots.instance == 6U * index + 1U &&
                  kv.cache.keys.ring == cache.keys.ring && kv.cache.pending_keys == cache.pending_keys &&
                  attention_args.cache.pending_values == cache.pending_values &&
                  attention_args.window == (uses_sliding_attention(visit) ? kSlidingWindow : 0U),
                  "verify KV directory or ancestor-aware attention binding");
            const auto conv_name = "cuda-sm120-causal-convolution-f32" + suffix;
            locate(conv_name, index);
            unsigned stream = 0;
            for (std::size_t i = trace.index; i < result.program.launches.size() && stream < 4U; ++i) {
                const auto& launch = result.program.launches[i];
                if (launch.layer != static_cast<std::int32_t>(index)) break;
                if (!launch.identity.starts_with("cuda-sm120-causal-convolution-")) continue;
                trace.index = i;
                const auto convolution = trace.read<strix::core::causal_conv::StepArgs>(launch.identity, launch.grid,
                                                                                        exec::CudaNodeKind::kernel);
                check(convolution.step == &descriptor && convolution.history.table.entries == directory.data() &&
                      convolution.history.instance == 6U * index + 2U + stream &&
                      convolution.pending == verify_state.history(visit, static_cast<MiniConvolutionStream>(stream)),
                      "verify convolution slot or pending image");
                ++stream;
            }
            check(stream == 4U, "verify convolution stream count");
        }
        trace.index = result.program.launches.size() - 2U;
        const auto pick_copy = trace.read<exec::CudaCopyArgs>("cuda-native-copy-d2h-v1", {}, exec::CudaNodeKind::copy_d2h_pinned);
        const auto logit_copy = trace.read<exec::CudaCopyArgs>("cuda-native-copy-d2h-v1", {}, exec::CudaNodeKind::copy_d2h_pinned);
        check(pick_copy.source == verify_picked.data() && pick_copy.destination == verify_reply.data() &&
              pick_copy.bytes == rows * sizeof(verify_picked[0]), "verify pick order");
        check(logit_copy.source == logits.data() && logit_copy.destination == logits_reply.data() &&
              logit_copy.bytes == logits.size() * sizeof(float), "verify row-major logit readback");
        std::cout << "inkling2_program_host verify R=" << rows << " position=" << position
                  << " nodes=" << trace.index << " launches=" << result.program.predicted.launches << " missing=0\n";
    }
}

// R1 recipe checks; `scalar` (optional) receives the leaf identities of the incumbent R1 recipe (no fusion,
// no latency forms), the reference list of the width-bound verify recipes.
std::uint64_t check_physical_plan(unsigned position, FixtureWeights& weights, decode::DecodeRecipeKey key = {},
                                  std::vector<std::string>* scalar_out = nullptr) {
    using namespace mini;
    constexpr unsigned rows = 1U;
    const MiniSessionStatePlan layout(position + 1U, rows);
    const auto kv_elements = static_cast<std::size_t>(layout.kv_elements_per_plane());
    const auto keys = std::make_unique_for_overwrite<std::uint16_t[]>(kv_elements);
    const auto values = std::make_unique_for_overwrite<std::uint16_t[]>(kv_elements);
    std::vector<float> history(static_cast<std::size_t>(layout.convolution_elements())), history_backup(history.size());
    std::vector<std::uint16_t> kv_backup(mini_resident_max_kv_checkpoint_elements());
    const auto make_state = [&] {
        return MiniResidentSessionState(layout, {{keys.get(), kv_elements}, {values.get(), kv_elements}}, history);
    };
    struct PositionOnly {
        bool try_step_graph(MiniResidentChunk, const std::function<void()>&) { return true; }
        void finish() {}
        void copy(std::span<const float>, std::span<float>) {}
        void copy(std::span<const std::uint16_t>, std::span<std::uint16_t>) {}
    };

    std::vector<float> activations(mini_resident_activation_elements(rows));
    std::vector<float> controller(4U * kHiddenWidth + kControllerRank), attention(50000U);
    std::vector<float> lag(kMiniConvLayerElements * kPhysicalLayers);
    std::vector<strix::moe_core::device_grouped::ExpertWeights> matrices(kPhysicalLayers * kExperts);
    std::array<unsigned, kExpertsPerToken> ids{};
    std::array<float, kExperts> scores{};
    std::array<float, kExpertsPerToken> route_weights{};
    std::array<std::uint16_t, kExpertsPerToken * kHiddenWidth> expert_activation{};
    std::array<unsigned, 2U * kLogicalVisits> flags{};
    const MiniCudaStepStorage scratch{rows, controller, attention, lag.data(), matrices, ids.data(),
        scores.data(), route_weights.data(), expert_activation.data(), flags};
    std::vector<float> embeddings(kHiddenWidth), hidden(kHiddenWidth), logits(kOutputVocabulary);
    std::array<strix::head_argmax::Pick, 128U> partials{};
    std::uint32_t picked{}, reply{};
    const MiniCudaStepRows buffers{embeddings, hidden, logits, partials.data(), &picked, &reply};
    exec::StepBlock descriptor{};
    std::vector<unsigned> words(decode::GroupedDecodeScratch::word_count(kExperts, kExpertsPerToken));
    std::vector<float> values_scratch(decode::GroupedDecodeScratch::value_count(kHiddenWidth, kExpertWidth, kExpertsPerToken));
    std::vector<strix::moe_core::device_grouped::TileRows> descriptors(2U * kExpertsPerToken);
    const auto build_recipe = [&](bool fusion, bool reject_fp32 = false, bool grouped_projections = false,
                                   decode::R1Forms forms = {}) {
        auto state = make_state();
        // Advance the frontier without touching buffers.
        PositionOnly skip;
        for (unsigned row = 0; row < position; ++row)
            state.append(skip, {row, rows}, history_backup, kv_backup, [] {}, [] {});
        unsigned checkpoints = 0;
        MiniCudaStepMode mode{key, {words, values_scratch, descriptors}, nullptr, &checkpoints,
            +[](void*, const void*, std::size_t, void* count) { ++*static_cast<unsigned*>(count); }};
        if (!fusion) mode.epilogue_fusion = false;
        mode.key.forms = forms;
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
        strix::device::Profile profile{};
        profile.units = 1;
        if (grouped_projections) {
            mode.projection_profile = &profile;
            mode.bind_projections = decode::bind_projection_batch;
        }
#else
        (void)grouped_projections;
#endif
        const auto bind = [reject_fp32](const auto& args) {
            auto leaf = bind_physical_fixture(args);
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(args)>, decode::Bf16Fp32Epilogue>) {
                if (reject_fp32) leaf.identity = "rejected-fp32-epilogue";
            }
            return leaf;
        };
        exec::CudaRecipeProgram result;
        const auto dispatch = [&](MiniResidentChunk, const auto& build) {
            result = build().compile(key.program);
            return true;
        };
        const auto finish = [] {};
        const VerifyFixtureBinding bind_verify{};
        MiniCudaStepLowering lower(weights, scratch, &descriptor, bind, dispatch, finish, bind_verify, mode);
        MiniResidentForward forward(lower, state, {activations, history_backup, kv_backup});
        forward.append({position, rows}, buffers.embeddings, buffers.hidden,
            [&](MiniResidentChunk chunk, MiniConstRows h) { lower.head(h, buffers); lower.submit_eager(chunk); }, [] {},
            [&] { lower.begin_eager(); lower.embed(buffers.embeddings); });
        check(result.program.capturable == !key.program.tap && (checkpoints != 0U) == key.program.tap,
              "tapped recipe must use eager checkpointing");
        return result;
    };

    const auto state = make_state();
    const auto result = build_recipe(key.program.tap);  // Tapped recipes use the default policy.
    if (auto* scalar = scalar_out) {
        scalar->clear();
        for (const auto& launch : result.program.launches) {
            if (launch.identity == "cuda-sm120-causal-convolution-r1-v1") {
                decode::CausalConvolution convolution{};
                std::memcpy(&convolution, result.program.args.data() + launch.args_offset, sizeof(convolution));
                if (convolution.round_bf16) {
                    scalar->emplace_back("cuda-sm120-causal-convolution-round-r1-v1");
                    continue;
                }
            }
            scalar->push_back(launch.identity);
        }
    }

    PhysicalTrace trace{result};
    EpilogueTrace epilogues{result, {}};
    const auto clear = trace.read<exec::CudaMemsetArgs>("cuda-native-memset-v1", {}, exec::CudaNodeKind::memset);
    check(clear.destination == flags.data() && clear.bytes == sizeof(flags) && clear.value == 0, "native flag binding");
    const auto embed = trace.kernel<decode::Embedding>("lookup-bf16-f32", grid(1U), true);
    check(embed.token == &descriptor.row[0].token && embed.output == embeddings.data() &&
          embed.table == weights.bf16("embed.weight").data(), "X3 embedding binding");
    const auto rms = [&](unsigned groups = 1U) {
        return trace.kernel<decode::RmsBf16>("rms-bf16-image-r1", grid(groups), true);
    };
    check(rms().output == embeddings.data(), "embedding RMS binding");
    const auto image = [&](unsigned i) { return activations.data() + std::size_t{i} * kHiddenWidth; };
    trace.copy(image(0), embeddings.data());
    unsigned blocks = 0;
    const auto mix = [&](unsigned banks, unsigned queries, bool normalized = false) {
        const auto args = trace.kernel<residual::MixArgs>("residual-bank-mix-r1",
            grid(1U, banks == 1U ? 256U : 128U, 1U, banks));
        check(args.bank_count == banks && args.query_count == queries && args.normalized == normalized,
              "residual bank geometry or prescaling");
        return args;
    };
    const auto dense = [&](unsigned count, unsigned launch_blocks, bool scalar = false, bool round = false) {
        const auto id = scalar ? "bf16-r1-fp32-group" : (round ? "bf16-r1-mma3-group-round" : "bf16-r1-mma3-group");
        const auto args = trace.kernel<decode::Bf16DenseGroup>(id, grid(launch_blocks), true);
        check(args.count == count, "BF16 projection group size");
        return args;
    };
    const auto row_input = [&](unsigned pass, unsigned output) {
        if (pass != 0U) {
            const auto bank = mix(2U, 1U, true);
            check(bank.banks[0] == image(6) && bank.banks[1] == image(7) &&
                  bank.output[0] == controller.data(), "repeat controller message bank");
            const auto pair = trace.kernel<residual::NormalizePairArgs>("controller-pair-norm-r1", grid(1U, 256U, 2U));
            check(pair.first.output == controller.data() + kHiddenWidth &&
                  pair.second.output == controller.data() + 2U * kHiddenWidth && pair.second.input == bank.output[0],
                  "controller pair scratch binding");
            check(dense(1U, 4U, false, true).input == pair.first.output, "controller down binding");
            check(dense(2U, 288U).input == controller.data() + 4U * kHiddenWidth, "controller gate binding");
            const auto blend = trace.kernel<residual::ControllerArgs>("residual-controller-r1", grid(9U));
            check(blend.output == controller.data() + 3U * kHiddenWidth, "controller blend binding");
        }
        check(rms().output == image(output), "layer input binding");
    };
    const auto round = [&](unsigned n) { return trace.kernel<decode::Bf16Round>("pointwise-bf16-round", grid(n)); };
    const auto conv = [&](unsigned n) { return trace.kernel<decode::CausalConvolution>("causal-convolution-r1", grid(n)); };
    const auto add = [&](bool bf16) {
        check(trace.kernel<residual::AddArgs>("residual-add-r1", grid(9U)).round_bf16 == bf16, "residual add store");
    };
    for (const auto visit : mini_schedule()) {
        const auto physical = visit.physical_layer;
        const auto cached = state.visit(visit);
        const bool full = is_full_attention(physical);
        if (physical == kCoreFirst) {
            if (visit.pass == 0U) trace.copy(image(1), image(0));
            else {
                const auto bank0 = trace.kernel<residual::NormalizeArgs>("residual-bank-norm-r1", grid(1U));
                const auto bank1 = trace.kernel<residual::NormalizeArgs>("residual-bank-norm-r1", grid(1U));
                check(bank0.input == image(2U + blocks - 1U) && bank0.output == image(6) &&
                      bank1.input == image(0) && bank1.output == image(7), "previous pass bank binding");
                trace.copy(image(0), image(1));
                blocks = 1U;
            }
        }
        if (physical != 0U) static_cast<void>(mix(blocks + 1U, full ? 3U : 1U));
        row_input(visit.pass, 11U);
        if (full || visit.pass != 0U) {
            row_input(visit.pass, 12U);
            row_input(visit.pass, 13U);
        }
        const auto sink_head = trace.index;
        const auto sink_projection = dense(1U, 3U, true);
        check(sink_projection.input == image(11), "attention sink projection input");
        check(dense(3U, 408U).input == image(11), "attention query projection input");
        std::array<std::size_t, 2> kv_heads{};
        std::array<decode::Bf16DenseGroup, 2> kv_projections{};
        if (full || visit.pass != 0U) {
            for (unsigned i = 0; i < kv_heads.size(); ++i) {
                kv_heads[i] = trace.index;
                kv_projections[i] = dense(full ? 2U : 1U, full ? 66U : 64U);
                check(kv_projections[i].input == image(12U + i), "key/value projection input");
            }
        } else {
            kv_heads[0] = trace.index;
            kv_projections[0] = dense(2U, 128U);
            check(kv_projections[0].input == image(11), "shared local KV projection input");
        }
        std::size_t value_head = 0;
        decode::CausalConvolution value_convolution{};
        std::array<std::size_t, 2> raw_tails{};
        std::array<decode::Bf16Round, 2> raw_rounds{};
        for (unsigned i = 0; i < 2U; ++i) {
            const auto kv_tail = trace.index;
            const auto rounding = round(4U);
            if (full) {
                const std::array tap_tail{trace.index};
                const std::array bundle{rounding, round(1U)};
                epilogues.add(kv_heads[i], kv_tail, kv_projections[i], bundle, graph::Form::store_bf16, tap_tail);
            } else if (visit.pass != 0U) {
                epilogues.add(kv_heads[i], kv_tail, kv_projections[i], rounding, graph::Form::store_bf16);
            } else {
                raw_tails[i] = kv_tail;
                raw_rounds[i] = rounding;
            }
            const auto conv_index = trace.index;
            const auto convolution = conv(4U);
            if (i == 1U) {
                value_head = conv_index;
                value_convolution = convolution;
            }
            check(convolution.input == rounding.values && convolution.history == cached.history[i].data() &&
                  convolution.static_weights == lag.data() + physical * kMiniConvLayerElements + i * kKvWidth * kConvTaps &&
                  (convolution.dynamic_weights != nullptr) == full, "visit KV convolution binding");
        }
        if (!full && visit.pass == 0U) {
            epilogues.add(kv_heads[0], raw_tails[0], kv_projections[0], raw_rounds, graph::Form::store_bf16,
                          std::span(raw_tails).last(1));
        }
        const auto clip = [&](unsigned heads, unsigned threads, bool from_x3) {
            const auto args = trace.kernel<decode::HeadClipBf16>("head-clip-bf16-r1", grid(heads, threads));
            check(args.position == (from_x3 && !key.program.tap ? &descriptor.row[0].position : nullptr) &&
                  args.first_position == (from_x3 ? position : 0U), "visit clip position binding");
        };
        clip(24U, 128U, full);
        clip(8U, 128U, false);
        const auto value_tail = trace.index;
        const auto value_round = round(4U);
        epilogues.add(value_head, value_tail, value_convolution, value_round, graph::Form::store_bf16);
        check(rms(8U).input == value_round.values, "value RMS binding");
        clip(24U, 32U, full);
        const auto relative = trace.kernel<decode::Bf16RelativeProjection16>("relative-bf16-d16", grid(96U));
        const auto sink_tail = trace.index;
        const auto sinks = round(1U);
        epilogues.add(sink_head, sink_tail, sink_projection, sinks, graph::Form::store_bf16);
        const auto append = key.program.tap
            ? trace.kernel<decode::AppendKvHostBf16>("prepared-bf16-kv-host-r1", grid(8U, 128U)).args
            : trace.kernel<decode::AppendKvBf16>("prepared-bf16-kv-step", grid(8U, 128U));
        check(append.key_cache == cached.kv.keys && append.value_cache == cached.kv.values &&
              append.position == (key.program.tap ? nullptr : &descriptor.row[0].position) && append.first_position == position &&
              append.capacity == cached.kv.slots && append.wrapped == cached.kv.wrapped, "visit KV append binding");
        const auto flash_grid = grid(24U, 256U, 1U, 1U, 35840U);
        const auto flash = key.program.tap
            ? trace.kernel<decode::PreparedFlashHostBf16>("prepared-flash-pad2-d128-host-r1", flash_grid).args
            : trace.kernel<decode::PreparedFlashBf16>("prepared-flash-pad2-d128", flash_grid);
        check(flash.key_cache == append.key_cache && flash.value_cache == append.value_cache &&
              flash.position == append.position && flash.first_position == position && flash.window == (full ? 0U : kSlidingWindow) &&
              flash.relative_bias == relative.output && flash.sinks == sinks.values, "visit flash binding");
        const auto finite = trace.kernel<decode::PreparedFinite>("prepared-finite-check", grid(96U));
        check(finite.failure == flags.data() + visit.logical_layer && finite.rows.output == flash.output,
              "visit attention validation binding");
        const auto gate = trace.kernel<decode::Bf16SigmoidProduct>("pointwise-bf16-sigmoid-product", grid(12U));
        const auto output_head = trace.index;
        const auto out_projection = dense(1U, 144U);
        check(gate.input == flash.output && out_projection.input == gate.output, "attention output projection binding");
        const auto output_tail = trace.index;
        epilogues.add(output_head, output_tail, out_projection, round(9U), graph::Form::store_bf16);
        check(conv(9U).history == cached.history[2].data(), "attention post-convolution state");
        if (starts_residual_block(visit)) trace.copy(image(2U + blocks++), image(0));
        else add(false);
        static_cast<void>(mix(blocks + 1U, 1U));
        row_input(visit.pass, 11U);
        check(dense(1U, 16U, true).projections[0].output == scores.data(), "router logits binding");
        const auto wave = trace.kernel<decode::CheckedWaveRoute>("checked-wave-mini-route-r1", grid(1U, 128U));
        auto* failure = flags.data() + kLogicalVisits + visit.logical_layer;
        check(wave.logits == scores.data() && wave.weights == route_weights.data() &&
              wave.ids == ids.data() && wave.failure == (key.grouped ? words.data() : failure), "route status binding");
        if (key.program.tap) {
            const auto exchange = trace.read<decode::RouteTapExchange>(
                "cuda-route-tap-exchange-v1", {}, exec::CudaNodeKind::kernel, false, true);
            check(exchange.layer == physical && exchange.rows == 1U && exchange.top_k == kExpertsPerToken &&
                  exchange.experts == kExperts && exchange.ids == wave.ids && exchange.weights == wave.weights,
                  "tapped route must exchange selected routes before expert work");
        }
        const auto format = mini_expert_format(physical);
        const bool nvfp4 = format == strix::weight::microscaled::MatrixFormat::nvfp4;
        if (key.grouped) {
            // Incumbent order: device_plan.hpp PackedPlan::enqueue, then
            // device_grouped_plan.hpp materialize, then grouped_dispatch.hpp enqueue.
            const auto reset = trace.read<exec::CudaMemsetArgs>("cuda-native-memset-v1", {}, exec::CudaNodeKind::memset);
            check(reset.destination == words.data() + 2U * kExperts + 5U * kExpertsPerToken + 4U &&
                  reset.bytes == 130U * sizeof(unsigned) && reset.value == 0, "grouped CSR header binding");
            constexpr std::array phases{"cuda-grouped-csr-count-v1", "cuda-grouped-csr-scan-v1",
                "cuda-grouped-csr-scatter-v1", "cuda-grouped-materialize-v1"};
            strix::moe_core::device_grouped::Args plan{};
            for (std::size_t i = 0; i < phases.size(); ++i) {
                const auto leaf = trace.read<decode::GroupedPlanLeaf>(phases[i], grid(1U), exec::CudaNodeKind::kernel);
                plan = leaf.args;
                check(static_cast<std::size_t>(leaf.phase) == i && leaf.scratch == reset.destination &&
                      plan.rows == 1U && plan.experts == kExperts && plan.top_k == kExpertsPerToken &&
                      plan.ids == wave.ids && plan.weights == wave.weights && plan.route_status == wave.failure &&
                      plan.matrices == matrices.data() + physical * kExperts && plan.out.failure == failure &&
                      plan.out.gate_up == descriptors.data() && plan.out.down == descriptors.data() + 8U &&
                      plan.out.tile_capacity == 8U && plan.workspace.bf16 == expert_activation.data() &&
                      plan.workspace.fp32 == values_scratch.data() + 8U, "grouped plan binding");
            }
            const auto gather = trace.kernel<grouped_rows::GatherArgs>("grouped-gather-r1", grid(9U, 256U, 8U));
            const auto projection = nvfp4 ? "grouped-nvfp4-projection-r1" : "grouped-mxfp8-projection-r1";
            const exec::Grid projection_grid{{72U, 1U, 8U}, {128U, 1U, 1U}, 0U};
            const auto up = trace.kernel<decode::GroupedProjection>(projection, projection_grid);
            const auto activation = trace.kernel<decode::GroupedActivation>("grouped-swiglu-r1", grid(9U, 128U, 8U));
            const auto down = trace.kernel<decode::GroupedProjection>(projection, projection_grid);
            const auto combine = trace.kernel<grouped_rows::CombineArgs>("grouped-combine-r1", grid(9U));
            check(gather.assignment_rows == plan.csr.rows && gather.grouped == plan.workspace.bf16 &&
                  up.format == format && down.format == format &&
                  up.args.tiles == plan.out.gate_up && down.args.tiles == plan.out.down &&
                  up.args.inputs == kHiddenWidth && down.args.inputs == kExpertWidth &&
                  up.args.tile_count == plan.out.tile_count && down.args.tile_count == up.args.tile_count &&
                  activation.args.gate_up == plan.workspace.fp32 && activation.args.output == gather.grouped &&
                  activation.gate_cap == kMlpGateSoftcap && activation.up_cap == kMlpUpSoftcap &&
                  combine.combine_rows == plan.out.combine_rows && combine.assignment_weights == plan.out.assignment_weights &&
                  combine.grouped == plan.workspace.fp32 && combine.output == image(14), "grouped expert binding");
        } else {
            const auto up = trace.kernel<decode::PackedGateUp>(
                nvfp4 ? "packed-nvfp4-gate-up-r1-k8-pdl" : "packed-mxfp8-gate-up-r1-k8-pdl", grid(r1_geometry::gate_blocks(1152U, r1_geometry::kCudaGeometry.gate_columns), 256U, 8U));
            const auto down = trace.kernel<decode::PackedDownCombine>(
                nvfp4 ? "packed-nvfp4-down-combine-r1-k8-pdl" : "packed-mxfp8-down-combine-r1-k8-pdl", grid(r1_geometry::down_blocks(2304U, r1_geometry::kCudaGeometry.down_columns)));
            check(up.format == format && down.format == format && up.args.matrices == matrices.data() + physical * kExperts &&
                  up.args.ids == wave.ids && up.args.weights == wave.weights && up.args.failure == failure &&
                  up.args.activation == expert_activation.data() && down.args.activation == up.args.activation &&
                  down.args.matrices == up.args.matrices && down.args.output == image(14), "packed expert binding");
        }
        check(conv(9U).history == cached.history[3].data(), "MoE post-convolution state");
        add(true);
    }
    static_cast<void>(mix(blocks + 1U, 1U));
    check(rms().output == hidden.data(), "final RMS binding");
    const auto logit_head = trace.index;
    const auto logit_projection = dense(1U, 25008U, true);
    check(logit_projection.input == hidden.data(), "LM head binding");
    const auto logit_tail = trace.index;
    const auto softcap = trace.kernel<decode::Bf16Softcap>("pointwise-bf16-softcap", grid(782U));
    check(softcap.values == logits.data(), "logit softcap binding");
    epilogues.add(logit_head, logit_tail, logit_projection, softcap, graph::Form::softcap);
    const auto partial = trace.kernel<decode::GreedyPartial>("greedy-partial-r1", grid(128U));
    const auto final = trace.kernel<decode::GreedyFinal>("greedy-final-r1", grid(1U));
    check(partial.logits == logits.data() && partial.partial == partials.data() &&
          final.reduction.partial == partials.data() && final.output == &picked, "greedy reduction binding");
    const auto readback = trace.read<exec::CudaCopyArgs>("cuda-native-copy-d2h-v1", {}, exec::CudaNodeKind::copy_d2h_pinned);
    check(readback.source == &picked && readback.destination == &reply && readback.bytes == sizeof(reply), "pinned reply binding");
    const auto expected_nodes = (key.grouped ? 2517U : 2133U) + (key.program.tap ? 48U : 0U);
    const auto expected_launches = key.grouped ? 2455U : 2119U;
    check(trace.index == expected_nodes && trace.index == result.program.launches.size() &&
          trace.index == result.binding.nodes.size() && trace.count_kernels == expected_launches &&
          trace.count_kernels == result.program.predicted.launches, "full native list count");
    const auto report = [&](const exec::CudaRecipeProgram& plan, bool fusion) {
        std::cout << "inkling2_program_host native position=" << position << " nodes=" << plan.program.launches.size()
                  << " launches=" << plan.program.predicted.launches << " grouped=" << key.grouped
                  << " tap=" << key.program.tap << " epilogue_fusion=" << fusion << '\n';
    };
    report(result, false);
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
    const auto check_projection_recipe = [&](const exec::CudaRecipeProgram& old, bool fusion, bool reject_fp32 = false) {
        const auto candidate = build_recipe(fusion, reject_fp32, true);
        const auto& program = candidate.program;
        unsigned groups = 0;
        for (std::size_t i = 0; i < program.launches.size(); ++i) {
            const auto& leaf = program.launches[i];
            if (leaf.entry != "cuda-sm120-bf16-r1-mma3-independent-v1") continue;
            check(i != 0 && leaf.layer >= 0 && static_cast<unsigned>(leaf.layer) < mini::kLogicalVisits &&
                  leaf.kernels == 1, "invalid independent MMA leaf");
            const auto visit = mini::mini_schedule()[static_cast<unsigned>(leaf.layer)];
            const bool full = mini::is_full_attention(visit.physical_layer);
            const std::string_view mask = fusion ? (full ? "fffrrrr" : "fffrr") : (full ? "fffffff" : "fffff");
            check(leaf.chain_identity.ends_with(mask), "incorrect independent output rounding");
            const auto& sink = program.launches[i - 1];
            check(sink.entry == (fusion && !reject_fp32 ? "cuda-sm120-bf16-r1-fp32-round-v1" :
                                                          "cuda-sm120-bf16-r1-fp32-group-v1"),
                  "scalar sink fallback lost its epilogue choice");
            ++groups;
        }
        const unsigned saving = key.program.tap ? 0U : 76U;
        check(groups == (key.program.tap ? 0U : mini::kLogicalVisits) &&
              program.launches.size() + saving == old.program.launches.size() &&
              program.predicted.launches + saving == old.program.predicted.launches, "independent preparation counts");
        check(key.program.tap ? program.identity == old.program.identity : program.identity != old.program.identity,
              "independent preparation physical identity");
        std::cout << "inkling2_program_host independent position=" << position << " grouped=" << key.grouped
                  << " tap=" << key.program.tap << " epilogue_fusion=" << fusion << " reject_fp32=" << reject_fp32
                  << " independent_leaves=" << groups << " nodes=" << program.launches.size()
                  << " launches=" << program.predicted.launches << '\n';
    };
    check_projection_recipe(result, false);
#endif
    if (key.program.tap) return result.program.identity;

    const auto fused = build_recipe(true);
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
    check_projection_recipe(fused, true);
#endif
    check(fused.program.launches.size() == (key.grouped ? 2260U : 1876U) &&
          fused.program.predicted.launches == (key.grouped ? 2198U : 1862U), "default fused recipe counts");
    check(fused.program.identity != result.program.identity, "fusion must change the physical recipe identity");
    epilogues.check_lowering(fused);
    if (position == 7U && !key.grouped) {
        const auto rejected = build_recipe(true, true);
        epilogues.check_lowering(rejected, true);
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
        check_projection_recipe(rejected, true, true);
#endif
    }
    report(fused, true);
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
    check_latency_forms(fused, build_recipe(true, false, false, kMiniLatencyForms), key.grouped);
#endif
    return fused.program.identity;
}

void check_descriptor(std::uint32_t position) {
    exec::StepBlock block{};
    block.h.rows = block.h.valid_mask = block.h.feed_mask = 1U;
    block.row[0] = {3711U, position, 0U, exec::kStepNoParent, 1U, exec::step_valid, 0U, 0U};
    session::StepDescriptor incumbent{position, 3711U};
    for (std::uint32_t i = 0; i < 1000U; ++i) {
        const auto replay = exec::to_step_replay_row(exec::step_row(block, 0U));
        check(std::bit_cast<std::uint64_t>(replay.input) == std::bit_cast<std::uint64_t>(incumbent) &&
              replay.slot == 0U && replay.parent == exec::kStepNoParent && replay.ancestors == 1U &&
              replay.flags == session::step_replay_valid && block.h.step == i,
              "X3 row differs from incumbent transport");
        const exec::RowOut pick{150011U + i, 0.0F, 0.0F, exec::RowStatus::picked};
        const auto advanced = exec::advance_step(block, std::span(&pick, 1), {});
        check(advanced.commit_mask == 1U && exec::step_row_valid(advanced.next, 0U),
              "descriptor advance did not commit and feed the one row");
        block = advanced.next;
        incumbent = {incumbent.position + 1U, pick.id};
    }
    const auto replay = exec::to_step_replay_row(exec::step_row(block, 0U));
    check(std::bit_cast<std::uint64_t>(replay.input) == std::bit_cast<std::uint64_t>(incumbent),
          "X3 row differs after the last advance");
}

#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
// R1 expert leaves run the EntryPdl schedule: every CTA waits before its first read, so the catalog declares the
// wait contract. The legacy per-call schedule keeps a completion edge; an explicit programmatic call admits it.
void check_r1_entry_contract() {
    using strix::weight::microscaled::MatrixFormat;
    namespace r1 = strix::moe_core::microscaled_r1;
    std::vector<strix::moe_core::device_grouped::ExpertWeights> matrices(128U);
    std::vector<unsigned> ids(8U);
    std::vector<float> route(8U), input(2304U), output(2304U);
    std::vector<std::uint16_t> activation(8U * 1152U);
    for (const auto format : {MatrixFormat::nvfp4, MatrixFormat::mxfp8}) {
        r1::Args args{};
        args.matrices = matrices.data();
        args.ids = ids.data();
        args.weights = route.data();
        args.input = input.data();
        args.activation = activation.data();
        args.output = output.data();
        args.experts = 128U;
        args.hidden = 2304U;
        args.intermediate = 1152U;
        args.gate_cap = args.up_cap = 30.0F;
        const auto up = decode::bind(decode::PackedGateUp{args, format});
        const auto down = decode::bind(decode::PackedDownCombine{args, format});
        check(up.pdl.wait == exec::PdlWait::before_step_io && up.pdl.opt_out == exec::PdlOptOut::none &&
              down.pdl.wait == exec::PdlWait::before_step_io && down.pdl.opt_out == exec::PdlOptOut::none,
              "R1 expert leaves must declare the EntryPdl wait contract");
        check(up.identity.ends_with("-r1-k8-pdl-v1") && down.identity.ends_with("-r1-k8-pdl-v1"),
              "R1 EntryPdl leaves carry their own identity");
    }
}
#endif

}  // namespace

int main(int argc, const char** argv) try {
    check(argc == 3, "expected decoder template path and Inkling declaration path");
    const auto declaration = graph::inspect_model_decl(load_fixture(argv[2]), {argv[1]});
    const mini::MiniCudaDecodePlan decode(declaration);
    const auto short_plan = decode.at(7U);
    const auto long_plan = decode.at(4096U);
    check_plan(short_plan);
    check_plan(long_plan);
    check(graph::plan_identity(short_plan) == graph::plan_identity(long_plan),
          "the declared step plan changed with position");
    check_compiler_admission(decode.stages(), short_plan);
    check_compiler_admission(decode.stages(), long_plan);
    check_transfer_nodes();
    FixtureWeights weights;
    std::vector<std::string> scalar, scalar_long;
    for (const auto position : {7U, 4096U}) {
        constexpr std::array modes{decode::DecodeRecipeKey{}, decode::DecodeRecipeKey{true, {}},
            decode::DecodeRecipeKey{false, {.tap = true}}, decode::DecodeRecipeKey{true, {.tap = true}}};
        std::array<std::uint64_t, modes.size()> identities{};
        for (std::size_t i = 0; i < modes.size(); ++i)
            identities[i] = check_physical_plan(position, weights, modes[i],
                                                i == 0U ? (position == 7U ? &scalar : &scalar_long) : nullptr);
        std::ranges::sort(identities);
        check(std::ranges::adjacent_find(identities) == identities.end(), "decode modes share a physical program identity");
    }
    check(scalar == scalar_long, "scalar launch list changed at long context");
    for (const unsigned rows : {2U, 4U, 8U, 16U})
        check_verify_plan(7U, weights, rows, scalar);
    check_verify_plan(4096U, weights, 4U, scalar);
    check_verify_plan(7U, weights, 4U, scalar, true);
    for (const unsigned rows : {1U, 2U, 8U, 16U})
        check_verify_plan(4096U, weights, rows, scalar, false, true);
    check_descriptor(7U);
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
    check_r1_entry_contract();
#endif
    check_descriptor(4096U);
    std::cout << "inkling2_program_host PASS: 48 visits at 7 and 4096, "
                 "X1 microscaled formats and edges, native lists and buffer bindings, "
                 "1000 descriptor advances per position, verify R=2/4/8/16 on the scalar leaf order, "
                 "drafter tap stores at R=1/2/8/16\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "inkling2_program_host FAIL: " << error.what() << '\n';
    return 1;
}
