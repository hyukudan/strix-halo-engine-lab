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
    if constexpr (std::is_same_v<A, CheckedWaveRoute>) {
        exec::KernelBinding wave;
        wave.identity = "cuda-sm120-checked-wave-mini-route-r1-v1";
        wave.fn = unexecuted_binding;
        wave.grid = grid(1U, 128U);
        const auto bytes = std::as_bytes(std::span(&a, 1U));
        wave.args.assign(bytes.begin(), bytes.end());
        return wave;
    } else if constexpr (std::is_same_v<A, RoutedPackedGateUp>) {
        // The policy binder (bind_routed_gate_up) is instantiated by the CUDA program TU; the fixture records its
        // identity and grid (the gate-up grid).
        exec::KernelBinding routed;
        routed.identity = routed_gate_up_identity(a.format, mini::kMiniRouteIdentity);
        routed.fn = unexecuted_binding;
        routed.grid = grid(r1_geometry::gate_blocks(a.args.intermediate, r1_geometry::kCudaGeometry.gate_columns), 256, 8);
        routed.pdl.wait = exec::PdlWait::before_step_io;
        const auto bytes = std::as_bytes(std::span(&a, 1U));
        routed.args.assign(bytes.begin(), bytes.end());
        return routed;
    } else {
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
        return decode::bind(a);
#else
        auto describe = [&]() -> std::pair<std::string_view, exec::Grid> {
            const auto pointwise = [](std::size_t n) { return grid(static_cast<unsigned>((n + 255U) / 256U)); };
            if constexpr (std::is_same_v<A, Embedding>) return {"embedding-bf16-r1", grid(1U)};
            if constexpr (std::is_same_v<A, RmsBf16>) return {"rms-bf16-image-r1", grid(a.groups)};
            if constexpr (std::is_same_v<A, AppendKvBf16>) return {"prepared-bf16-kv-step", grid(a.kv_heads, a.width)};
            if constexpr (std::is_same_v<A, PreparedFlashBf16>) return {"prepared-flash-pad2-d128", grid(a.heads, 256U, 1, 1, 35840U)};
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
            if constexpr (std::is_same_v<A, PackedDownCombineTail>) {
                const auto name = a.format == strix::weight::microscaled::MatrixFormat::nvfp4
                    ? "packed-nvfp4-down-combine-conv4-residual-r1-k8-pdl" : "packed-mxfp8-down-combine-conv4-residual-r1-k8-pdl";
                return {name, grid(r1_geometry::down_blocks(a.args.hidden, r1_geometry::kCudaGeometry.down_columns))};
            }
            if constexpr (std::is_same_v<A, GreedyPartial>) return {"greedy-partial-r1", grid(a.chunks)};
            if constexpr (std::is_same_v<A, GreedyFinal>) return {"greedy-final-r1", grid(1U)};
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

struct PhysicalTrace {
    const exec::CudaRecipeProgram& result;
    std::size_t index = 0;
    std::size_t count_kernels = 0;
    bool previous_kernel = false;

    template <class A>
    A read(std::string_view identity, exec::Grid shape, exec::CudaNodeKind kind, bool pdl = false) {
        const auto& launch = result.program.launches.at(index);
        const auto& node = result.binding.nodes.at(index);
        const bool kernel = kind == exec::CudaNodeKind::kernel;
        if (launch.identity != identity || node.kind != kind || launch.grid.blocks != shape.blocks ||
            launch.grid.threads != shape.threads || launch.grid.shared_bytes != shape.shared_bytes ||
            launch.kernels != (kernel ? 1U : 0U) || (launch.fn != nullptr) != kernel || launch.args_bytes != sizeof(A))
            throw std::runtime_error("physical node " + std::to_string(index) + " expected " + std::string(identity) +
                                     ", got " + launch.identity);
        if (index == 0) check(node.after.empty() && launch.after == exec::no_launch, "unexpected first node edge");
        else {
            const auto edge = pdl && previous_kernel ? exec::CudaEdgeKind::programmatic : exec::CudaEdgeKind::completion;
            check(node.after == std::vector<exec::CudaDependency>{{index - 1U, edge}} && launch.after == index - 1U,
                  "native node changed its stream predecessor or PDL edge");
        }
        ++index;
        count_kernels += kernel;
        previous_kernel = kernel;
        A data{};
        std::memcpy(&data, result.program.args.data() + launch.args_offset, sizeof(data));
        return data;
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

    void check_lowering() const {
        exec::CudaProgramRecipe recipe;
        std::vector<std::string_view> expected;
        std::size_t removed = 0;
        for (std::size_t i = 0; i < original.program.launches.size(); ++i) {
            const auto head = std::ranges::find(pairs, i, &Pair::head);
            if (std::ranges::any_of(pairs, [i](const auto& pair) {
                    return std::ranges::find(pair.tails, i) != pair.tails.end();
                })) {
                ++removed;
                continue;
            }
            const auto& launch = original.program.launches[i];
            recipe.visit(launch.layer);
            recipe.stage(launch.entry);
            if (head != pairs.end()) {
                check(head->emit(recipe), "planned CUDA epilogue failed to lower");
                expected.push_back(head->entry);
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
            expected.push_back(launch.identity);
        }
        const auto result = recipe.compile({});
        check(removed == 257U && result.program.launches.size() == original.program.launches.size() - removed &&
              result.program.predicted.launches == original.program.predicted.launches - removed,
              "fused decode node or launch count");
        for (std::size_t i = 0; i < expected.size(); ++i)
            check(result.program.launches[i].identity == expected[i], "lowered launch order");
        std::cout << "inkling2_epilogue_host nodes=" << result.program.launches.size()
                  << " launches=" << result.program.predicted.launches << " fused=" << removed << '\n';
    }
};

// moe_tail: the MoE post-convolution and residual add run inside the down-combine launch (two nodes fewer per
// visit); otherwise the incumbent separate convolution and add nodes. moe_route: the checked route runs inside the
// gate-up launch (one node fewer per visit); otherwise the route launch precedes the gate-up launch.
void check_physical_plan(unsigned position, FixtureWeights& weights, bool moe_tail, bool moe_route) {
    using namespace mini;
    constexpr unsigned rows = 1U;
    const MiniSessionStatePlan layout(position + 1U, rows);
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
        state.append(skip, {row, rows}, history_backup, kv_backup, [] {}, [] {});
    }

    std::vector<float> activations(mini_resident_activation_elements(rows));
    std::vector<float> controller(4U * kHiddenWidth + kControllerRank), attention(50000U);
    std::vector<float> lag(kMiniConvLayerElements * kPhysicalLayers);
    std::vector<strix::moe_core::device_grouped::ExpertWeights> matrices(kPhysicalLayers * kExperts);
    std::array<unsigned, kExpertsPerToken> ids{};
    std::array<float, kExperts> scores{};
    std::array<float, kExpertsPerToken> route_weights{};
    std::array<std::uint16_t, kExpertsPerToken * kExpertWidth> expert_activation{};
    std::array<unsigned, 2U * kLogicalVisits> flags{};
    const MiniCudaStepStorage scratch{rows, controller, attention, lag.data(), matrices, ids.data(),
        scores.data(), route_weights.data(), expert_activation.data(), flags};
    std::vector<float> embeddings(kHiddenWidth), hidden(kHiddenWidth), logits(kOutputVocabulary);
    std::array<strix::head_argmax::Pick, 128U> partials{};
    std::uint32_t picked{}, reply{};
    const MiniCudaStepRows buffers{embeddings, hidden, logits, partials.data(), &picked, &reply};
    exec::StepBlock descriptor{};
    exec::CudaRecipeProgram result;
    const auto bind = [](const auto& args) { return bind_physical_fixture(args); };
    const auto dispatch = [&](MiniResidentChunk, const auto& build) {
        result = build().compile({});
        return true;
    };
    const auto finish = [] {};
    MiniCudaStepLowering lower(weights, scratch, &descriptor, bind, dispatch, finish, MiniMoeForms{moe_tail, moe_route});
    check(lower.moe_tail_enabled() == moe_tail && lower.moe_route_inline() == moe_route, "MoE form switches");
    MiniResidentForward forward(lower, state, {activations, history_backup, kv_backup});
    forward.append({position, rows}, buffers.embeddings, buffers.hidden,
        [&](MiniResidentChunk, MiniConstRows h) { lower.head(h, buffers); }, [] {},
        [&] { lower.embed(buffers.embeddings); });

    PhysicalTrace trace{result};
    EpilogueTrace epilogues{result, {}};
    const auto clear = trace.read<exec::CudaMemsetArgs>("cuda-native-memset-v1", {}, exec::CudaNodeKind::memset);
    check(clear.destination == flags.data() && clear.bytes == sizeof(flags) && clear.value == 0, "native flag binding");
    const auto embed = trace.kernel<decode::Embedding>("embedding-bf16-r1", grid(1U), true);
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
            check(args.position == (from_x3 ? &descriptor.row[0].position : nullptr), "visit clip position binding");
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
        const auto append = trace.kernel<decode::AppendKvBf16>("prepared-bf16-kv-step", grid(8U, 128U));
        check(append.key_cache == cached.kv.keys && append.value_cache == cached.kv.values &&
              append.position == &descriptor.row[0].position && append.first_position == position &&
              append.capacity == cached.kv.slots && append.wrapped == cached.kv.wrapped, "visit KV append binding");
        const auto flash = trace.kernel<decode::PreparedFlashBf16>("prepared-flash-pad2-d128", grid(24U, 256U, 1U, 1U, 35840U));
        check(flash.key_cache == append.key_cache && flash.value_cache == append.value_cache &&
              flash.position == append.position && flash.window == (full ? 0U : kSlidingWindow) &&
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
        auto* failure = flags.data() + kLogicalVisits + visit.logical_layer;
        const auto format = mini_expert_format(physical);
        const bool nvfp4 = format == strix::weight::microscaled::MatrixFormat::nvfp4;
        const auto up_grid = grid(r1_geometry::gate_blocks(1152U, r1_geometry::kCudaGeometry.gate_columns), 256U, 8U);
        decode::CheckedWaveRoute wave{};
        decode::PackedGateUp up{};
        if (moe_route) {
            // one launch: the route (same body, same buffers) runs inside gate-up, which publishes it for down
            const auto routed = trace.kernel<decode::RoutedPackedGateUp>(
                nvfp4 ? "packed-nvfp4-gate-up-routed-mini-route-r1-k8-pdl" : "packed-mxfp8-gate-up-routed-mini-route-r1-k8-pdl",
                up_grid, true);
            wave = {routed.logits, routed.bias, routed.ids, routed.weights, routed.failure};
            up = {routed.args, routed.format};
            check(routed.bias == weights.f32("layers." + std::to_string(physical) + ".mlp.gate.bias").data(),
                  "routed gate-up selection bias");
        } else {
            wave = trace.kernel<decode::CheckedWaveRoute>("checked-wave-mini-route-r1", grid(1U, 128U));
            up = trace.kernel<decode::PackedGateUp>(nvfp4 ? "packed-nvfp4-gate-up-r1-k8-pdl" : "packed-mxfp8-gate-up-r1-k8-pdl",
                                                    up_grid, true);
        }
        check(wave.logits == scores.data() && wave.weights == route_weights.data() &&
              wave.ids == ids.data() && wave.failure == failure, "route status binding");
        const auto down_grid = grid(r1_geometry::down_blocks(2304U, r1_geometry::kCudaGeometry.down_columns));
        // The incumbent MoE post-convolution (moe stream) and the BF16-rounded residual add, as the tail must
        // reproduce them: input image(14) = MoE output, conv image readout_[0] = image(8), residual image(15),
        // sum hidden image(0).
        constexpr auto post_plan = strix::attn_core::prepared_gqa_post_convolution_plan<MiniPreparedGqaShape>();
        const auto* post_weights = mini_post_convolution_weights(lag.data(), visit, true);
        const auto same_plan = [&](const strix::core::ShortConvolution& plan) {
            return plan.channels == kHiddenWidth && plan.taps == kConvTaps && plan.taps == 4U &&
                   plan.dynamic_groups == 0U && plan.residual_weight == 1.0F &&
                   plan.output_divisor == MiniPreparedGqaShape::kLocalConvDivisor &&
                   plan.channels == post_plan.channels && plan.taps == post_plan.taps &&
                   plan.residual_weight == post_plan.residual_weight && plan.output_divisor == post_plan.output_divisor;
        };
        decode::PackedDownCombine down{};
        if (moe_tail) {
            const auto fused = trace.kernel<decode::PackedDownCombineTail>(
                nvfp4 ? "packed-nvfp4-down-combine-conv4-residual-r1-k8-pdl"
                      : "packed-mxfp8-down-combine-conv4-residual-r1-k8-pdl", down_grid, true);
            down = {fused.args, fused.format};
            const auto& tail = fused.tail;
            check(same_plan(tail.plan) && tail.round_conv && tail.round_sum && tail.weights == post_weights &&
                  tail.history == cached.history[3].data() && tail.conv_output == image(8) &&
                  tail.residual == image(15) && tail.sum == image(0), "MoE tail plan and buffer binding");
        } else {
            down = trace.kernel<decode::PackedDownCombine>(
                nvfp4 ? "packed-nvfp4-down-combine-r1-k8-pdl" : "packed-mxfp8-down-combine-r1-k8-pdl", down_grid, true);
        }
        check(up.format == format && down.format == format && up.args.matrices == matrices.data() + physical * kExperts &&
              up.args.ids == wave.ids && up.args.weights == wave.weights && up.args.failure == failure &&
              up.args.activation == expert_activation.data() && down.args.activation == up.args.activation &&
              down.args.matrices == up.args.matrices && down.args.output == image(14), "packed expert binding");
        if (!moe_tail) {
            const auto post = conv(9U);
            check(same_plan(post.plan) && post.round_bf16 && post.static_weights == post_weights &&
                  post.dynamic_weights == nullptr && post.input == image(14) && post.output == image(8) &&
                  post.history == cached.history[3].data(), "MoE post-convolution state");
            const auto sum = trace.kernel<residual::AddArgs>("residual-add-r1", grid(9U));
            check(sum.round_bf16 && sum.lhs == image(15) && sum.rhs == image(8) && sum.output == image(0) &&
                  sum.elements == kHiddenWidth, "MoE residual add binding");
        }
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
    // 48 logical visits: the tail removes the convolution and add nodes of each MoE visit, the inline route the
    // route launch of each MoE visit.
    const std::size_t fused_nodes = (moe_tail ? 2U * kLogicalVisits : 0U) + (moe_route ? kLogicalVisits : 0U);
    check(trace.index == 2133U - fused_nodes && trace.index == result.program.launches.size() &&
          trace.index == result.binding.nodes.size() && trace.count_kernels == 2119U - fused_nodes &&
          trace.count_kernels == result.program.predicted.launches, "full native list count");
    epilogues.check_lowering();
    std::cout << "inkling2_program_host native position=" << position << " moe_tail=" << (moe_tail ? "on" : "off")
              << " moe_route=" << (moe_route ? "inline" : "launch") << " nodes=" << trace.index
              << " launches=" << trace.count_kernels << '\n';
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
    for (const bool moe_tail : {false, true}) {
        for (const bool moe_route : {false, true}) {
            check_physical_plan(7U, weights, moe_tail, moe_route);
            check_physical_plan(4096U, weights, moe_tail, moe_route);
        }
    }
    check_descriptor(7U);
#if defined(STRIX_INKLING2_TEST_CUDA_BINDINGS)
    check_r1_entry_contract();
#endif
    check_descriptor(4096U);
    std::cout << "inkling2_program_host PASS: 48 visits at 7 and 4096, "
                 "X1 microscaled formats and edges, native lists and buffer bindings, "
                 "1000 descriptor advances per position\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "inkling2_program_host FAIL: " << error.what() << '\n';
    return 1;
}
