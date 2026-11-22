#include "strix/device/profiles.hpp"
#include "strix/engine_profile.hpp"
#include "strix/exec/compiler.hpp"
#include "strix/execution_arena.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace {

namespace exec = strix::exec;
namespace graph = strix::graph;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Error = std::invalid_argument, class Work>
void rejects(Work work, const char* message) {
    try {
        work();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

using graph::Op;
using graph::StageDecl;
using graph::weights;

// Small declaration slices exercise the proving models' launch forms.
constexpr StageDecl kBefore[] = {
    {.id = "embed", .op = Op::gather, .entry = "embed", .in = {"token"}, .out = "x"},
};
constexpr StageDecl kQwenLayer[] = {
    {.id = "norm", .op = Op::norm, .entry = "norm", .in = {"x"}, .out = "n"},
    {.id = "q", .op = Op::linear, .entry = "iu4", .site = "attn", .in = {"n"}, .out = "q",
     .operand = weights("q", 5120, 5120)},
    {.id = "k", .op = Op::linear, .entry = "iu4", .site = "attn", .in = {"n"}, .out = "k",
     .operand = weights("k", 1024, 5120)},
    {.id = "attend", .op = Op::attend, .entry = "gqa", .in = {"q", "k", "kv"}, .out = "a"},
    {.id = "residual", .op = Op::combine, .entry = "add", .in = {"x", "a"}, .out = "x"},
};
constexpr StageDecl kInklingLayer[] = {
    {.id = "norm", .op = Op::norm, .entry = "norm", .in = {"x"}, .out = "n"},
    {.id = "query", .op = Op::linear, .entry = "proj_q", .in = {"n"}, .out = "q",
     .operand = weights("q", 2048, 2048)},
    {.id = "key", .op = Op::linear, .entry = "proj_k", .in = {"n"}, .out = "k",
     .operand = weights("k", 512, 2048)},
    {.id = "attend", .op = Op::attend, .entry = "gqa", .in = {"q", "k", "kv"}, .out = "a"},
    {.id = "residual", .op = Op::combine, .entry = "add", .in = {"x", "a"}, .out = "x"},
};
constexpr StageDecl kAfter[] = {
    {.id = "head", .op = Op::head, .entry = "head", .in = {"x"}, .out = "logits",
     .operand = weights("head", 64, 2048)},
    {.id = "pick", .op = Op::argmax, .entry = "pick", .in = {"logits"}, .out = "output"},
};
constexpr graph::FormatRule kQwenFormats[] = {
    {.format = graph::Format::iu4},
};
constexpr graph::FormatRule kInklingFormats[] = {
    {.format = graph::Format::bf16},
    {.tensor = "k", .format = graph::Format::iu4},
};
constexpr std::array kGroupFormats{graph::Format::iu4};

void host_stub(std::span<const std::byte> args, const exec::Grid& grid, void* stream) {
    check(args.size() == 1 && grid.blocks[0] > 0, "thunk sees the bound argument block and grid");
    ++*static_cast<unsigned*>(stream);
}

exec::BufferLayout resolve_buffer(const exec::ProgramKey&, const graph::Instance& instance) {
    if (instance.decl->out == "output") return {exec::BufferStorage::external, 4, 4};
    return {exec::BufferStorage::arena, 16, 32};
}

exec::KernelBinding resolve_kernel(const exec::LaunchBindingRequest& request) {
    const auto& launch = request.launch;
    check(launch.entry == request.plan.launches[request.launch_index].entry, "resolver receives the ordered launch");
    if (launch.entry == "add") {
        const auto id = launch.stages.front().inputs[0];
        check(request.buffers[id].generation != 0, "resolver sees the bound input generation");
    }
    exec::KernelBinding binding;
    binding.kernel = exec::KernelId{request.launch_index};
    binding.fn = host_stub;
    binding.identity = launch.entry + "/fixture/v1";
    binding.chain_identity = "fixture-chain/v1";
    binding.exact_identity = binding.identity;
    binding.grid.blocks[0] = request.key.rows;
    binding.args = {std::byte{1}};
    binding.predicted = {1.0, 2, 3, 4, launch.kernels};
    return binding;
}

exec::CompilerOptions fixture_options() {
    exec::CompilerOptions options;
    options.states = {{9, 1, 0x03}};
    options.weight_format_identity = 11;
    options.state_format_identity = 12;
    return options;
}

const exec::ArenaAllocation& allocation_for(const exec::Program& program, exec::BufferId id) {
    for (const auto& allocation : program.arena.allocations)
        if (allocation.lifetime.id == id) return allocation;
    throw std::runtime_error("fixture buffer has no arena allocation");
}

void check_edges(const exec::Compiler& compiler, const graph::LaunchPlan& plan) {
    const auto program = compiler.compile({}, plan);
    check(program.launches.size() == plan.launches.size(), "bound launch count matches the D3 plan");
    bool has_edge = false;
    for (std::size_t i = 0; i < plan.launches.size(); ++i) {
        const auto& want = plan.launches[i];
        const auto& got = program.launches[i];
        check(got.after == want.after && got.stream == want.stream, "bound launch preserves the D3 predecessor and stream");
        has_edge |= want.after != graph::no_launch;
    }
    check(has_edge, "D3 fixture must supply a predecessor");
}

void compile_qwen() {
    const graph::StageGraph graph{kBefore, kQwenLayer, kAfter, {0, 0}, kQwenFormats};
    const graph::FusionPolicy policy{graph::Fusion::today, kGroupFormats};
    unsigned plans = 0;
    const exec::PlanProvider provider = [&](const auto& declaration, const auto& key) {
        ++plans;
        return graph::launch_plan(declaration, 0, key.rows, policy);
    };
    exec::Compiler compiler(graph, resolve_buffer, resolve_kernel, fixture_options(), provider);
    const exec::ProgramKey one;
    const auto& serial = compiler.get(one);
    const std::array keys{one, exec::ProgramKey{.phase = exec::Phase::verify, .rows = 8},
                         exec::ProgramKey{.phase = exec::Phase::prefill, .rows = 4096}};
    compiler.warm(keys);
    check(plans == keys.size() && &serial == &compiler.get(one), "warm caches each key once");

    unsigned calls = 0;
    for (const auto& launch : serial.launches) {
        check(launch.args_offset % alignof(std::max_align_t) == 0, "argument block alignment");
        const auto args = std::span(serial.args).subspan(launch.args_offset, launch.args_bytes);
        launch.fn(args, launch.grid, &calls);
    }
    const auto plan = graph::launch_plan(graph, 0, 1, policy);
    check_edges(compiler, graph::launch_plan(graph, graph::PlanRequest{}));
    check(calls == serial.launches.size() && calls == plan.launches.size() &&
          serial.predicted.us == static_cast<double>(calls) &&
          serial.predicted.launches == plan.kernel_count() && serial.predicted.macs == 4U * calls,
          "bound launch thunks and cost totals preserve the plan");
    const auto& grouped = serial.launches.at(2);
    check(grouped.stages.size() == 2 && grouped.stages[0].inputs[0] == grouped.stages[1].inputs[0],
          "Qwen 27B grouped projection shares its input generation");
    const auto& live_in = serial.buffers[serial.launches.front().stages.front().inputs[0]];
    check(live_in.generation == 0 && live_in.symbol == "token" && live_in.layout.storage == exec::BufferStorage::external,
          "unproduced ports are external live-ins");
    const auto& add = serial.launches.at(4).stages.front();
    check(serial.buffers[add.inputs[0]].generation == 1 && serial.buffers[add.output].generation == 2,
          "same-name write advances the consumed generation");
    const auto& old = allocation_for(serial, add.inputs[0]);
    const auto& next = allocation_for(serial, add.output);
    check(old.lifetime.last_launch == next.lifetime.first_launch && old.offset != next.offset,
          "generations live in the same launch cannot alias");
    const auto& wide = compiler.get(keys[1]);
    const auto& tall = compiler.get(keys[2]);
    check(wide.descriptor.rows == 8 && tall.descriptor.rows == 4096 &&
          allocation_for(wide, add.output).lifetime.bytes == 7U * 32U + 16U &&
          allocation_for(tall, add.output).lifetime.bytes == 4095U * 32U + 16U,
          "row variants use padded physical geometry and support tall prefill");
    check(serial.identity != wide.identity && wide.identity != tall.identity &&
          serial.numerical_identity && serial.numerical_identity == wide.numerical_identity &&
          serial.numerical_identity == tall.numerical_identity,
          "physical identity varies by row shape and exact arithmetic identity is shared");
    check(serial.identity == compiler.compile(one, plan).identity, "direct and cached compilation agree");
    const auto fast = compiler.compile({.tier = strix::profile::Name::fast}, plan);
    check(fast.identity != serial.identity && fast.numerical_identity == serial.numerical_identity,
          "profile belongs to the execution key even when arithmetic agrees");

    auto bad = plan;
    std::swap(bad.launches[0], bad.launches[1]);
    rejects([&] { static_cast<void>(compiler.compile(one, bad)); }, "use before production must fail");
    rejects([&] { static_cast<void>(compiler.compile({.rows = 3}, plan)); }, "invalid decode width must fail");
    std::cout << "exec_program_identity: Qwen 27B host launch binding PASS\n";
}

void compile_inkling() {
    const graph::StageGraph graph{kBefore, kInklingLayer, kAfter, {0}, kInklingFormats};
    const auto plan = graph::launch_plan(graph, 0, 1, {graph::Fusion::fused});
    exec::Compiler compiler(graph, resolve_buffer, resolve_kernel, fixture_options());
    const auto program = compiler.compile({}, plan);
    const strix::device::Profile device{};
    const graph::PlanOverlap overlap[] = {{"proj_q", "proj_k", 10, 6, 2}};
    const auto scheduled = graph::launch_plan(graph, graph::PlanRequest{.device = &device, .overlap = overlap});
    check(scheduled.launches.at(2).stream == 2 && scheduled.launches.at(3).after == 1,
          "Inkling D3 fixture assigns a nonzero stream and a nonadjacent predecessor");
    check_edges(compiler, scheduled);
    const auto& query = program.launches.at(1);
    const auto& key = program.launches.at(2);
    check(query.prologue.size() == 1 && key.prologue.size() == 1 &&
          query.prologue.front().instance == key.prologue.front().instance &&
          query.stages.front().inputs[0] == query.prologue.front().output &&
          key.stages.front().inputs[0] == key.prologue.front().output &&
          program.buffers[key.prologue.front().output].layout.storage == exec::BufferStorage::launch_local,
          "Inkling2 projection fanout binds a recomputed prologue on each consumer");
    check(program.stages[query.prologue.front().instance].id == "norm", "program owns the profiling stage label");
    bool alternate_chain = false;
    bool noncapturable = false;
    bool alternate_address = false;
    auto binding = [&](const exec::LaunchBindingRequest& request) {
        auto entry = resolve_kernel(request);
        if (alternate_address) {
            entry.args.front() = std::byte{2};
            entry.kernel = exec::KernelId{42};
        }
        if (request.launch.entry == "head") {
            if (alternate_chain) entry.chain_identity = "head-chain/v2";
            if (noncapturable) {
                entry.capturable = false;
                entry.exact_identity.clear();
            }
        }
        return entry;
    };
    exec::Compiler rebinding(graph, resolve_buffer, binding, fixture_options());
    alternate_address = true;
    check(rebinding.compile({}, plan).identity == program.identity, "runtime argument bytes and handles do not key layout");
    alternate_chain = true;
    check(rebinding.compile({}, plan).identity != program.identity, "chain identity participates in the program hash");
    noncapturable = true;
    const auto eager = rebinding.compile({}, plan);
    check(!eager.capturable && eager.launches.size() == program.launches.size() && !eager.numerical_identity,
          "a noncapturable entry keeps its eager binding and withholds an undeclared exact identity");

    auto changed_states = fixture_options();
    changed_states.states.front().mask = 0x07;
    exec::Compiler state_compiler(graph, resolve_buffer, resolve_kernel, changed_states);
    check(state_compiler.compile({}, plan).identity != program.identity, "state masks are execution identity fields");
    changed_states = fixture_options();
    ++changed_states.state_format_identity;
    exec::Compiler format_compiler(graph, resolve_buffer, resolve_kernel, changed_states);
    check(format_compiler.compile({}, plan).numerical_identity != program.numerical_identity,
          "state formats are numerical identity fields");
    auto changed_arena = fixture_options();
    changed_arena.arena_alignment = 512;
    exec::Compiler arena_compiler(graph, resolve_buffer, resolve_kernel, changed_arena);
    const auto arena_variant = arena_compiler.compile({}, plan);
    check(arena_variant.identity != program.identity && arena_variant.numerical_identity == program.numerical_identity,
          "arena layout changes physical identity while preserving arithmetic");

    const exec::KernelResolver context_binding = [](const auto& request) {
        auto entry = resolve_kernel(request);
        entry.grid_by_context = request.launch.entry == "gqa";
        return entry;
    };
    exec::Compiler context_compiler(graph, resolve_buffer, context_binding);
    rejects([&] { static_cast<void>(context_compiler.compile({}, plan)); }, "context-dependent grid needs a bucket");
    check(context_compiler.compile({.ctx_bucket = 2}, plan).identity !=
          context_compiler.compile({.ctx_bucket = 3}, plan).identity, "context grids have distinct cache keys");
    rejects([&] { static_cast<void>(compiler.compile({.ctx_bucket = 2}, plan)); }, "fixed grid has no bucket");

    unsigned attempts = 0;
    const exec::PlanProvider retry = [&](const auto&, const auto&) {
        if (++attempts == 1) return graph::LaunchPlan{plan.instances, {}};
        return plan;
    };
    exec::Compiler retry_compiler(graph, resolve_buffer, resolve_kernel, fixture_options(), retry);
    rejects([&] { static_cast<void>(retry_compiler.get({})); }, "a failed first compile propagates");
    check(retry_compiler.get({}).identity == program.identity && attempts == 2, "a failed compile leaves no cache entry");
    std::cout << "exec_program_identity: Inkling2 host launch binding PASS\n";
}

void qwen_serial_arena() {
    const auto model = strix::ModelContract::qwen38_flash_next();
    const auto plan = strix::build_serial_decode_execution_plan(model);
    const auto io = strix::build_serial_execution_io_plan(plan);
    const strix::ExecutionBufferRequest request;
    const auto sizes = strix::build_serial_execution_buffer_plan(model, request);
    const std::array bytes_by_class{
        sizes.expanded_hidden_bytes, sizes.hidden_bytes, sizes.injection_weight_bytes,
        sizes.route_index_bytes, sizes.route_weight_bytes, sizes.logits_bytes, sizes.mlp_intermediate_bytes,
        sizes.gdn_workspace_bytes, sizes.ple_workspace_bytes, sizes.indexed_attention_workspace_bytes,
        sizes.hyper_workspace_bytes, sizes.router_workspace_bytes,
    };
    using Use = std::tuple<strix::ExecutionBufferClass, std::uint32_t, std::uint64_t>;
    const auto key = [](strix::ExecutionBufferUse use) {
        return Use{use.buffer.buffer_class, use.buffer.slot, use.generation};
    };
    // Preserve the legacy tie-break order when adapting its buffer-use ids.
    std::map<Use, exec::BufferLifetime> uses;
    for (std::uint32_t index = 0; index < io.steps.size(); ++index) {
        const auto& step = io.steps[index];
        const auto write = [&](strix::ExecutionBufferUse use) {
            const auto bytes = bytes_by_class.at(static_cast<std::size_t>(use.buffer.buffer_class));
            check(uses.emplace(key(use), exec::BufferLifetime{0, index, index, bytes}).second,
                  "serial fixture allocates each generation once");
        };
        for (const auto use : step.writes) write(use);
        for (const auto use : step.reads) uses.at(key(use)).last_launch = index;
        for (const auto use : step.temporaries) write(use);
    }
    std::vector<exec::BufferLifetime> lifetimes;
    std::vector<Use> ids;
    for (auto& [use, lifetime] : uses) {
        lifetime.id = static_cast<exec::BufferId>(ids.size());
        ids.push_back(use);
        lifetimes.push_back(lifetime);
    }
    for (const auto alignment : {1U, 256U}) {
        const auto expected = strix::build_serial_execution_arena_plan(model, request, plan, io, alignment);
        const auto actual = exec::Arena::plan(lifetimes, alignment);
        check(actual.alignment == expected.alignment && actual.total_bytes == expected.total_bytes &&
              actual.naive_nonaliased_bytes == expected.naive_nonaliased_bytes &&
              actual.allocations.size() == expected.allocations.size(), "Qwen serial arena totals match");
        for (std::size_t index = 0; index < actual.allocations.size(); ++index) {
            const auto& got = actual.allocations[index];
            const auto& want = expected.allocations[index];
            check(ids.at(got.lifetime.id) == key(want.use) &&
                  got.lifetime.first_launch == want.first_step && got.lifetime.last_launch == want.last_step &&
                  got.lifetime.bytes == want.bytes && got.offset == want.offset, "Qwen serial arena allocation matches");
        }
        std::cout << "exec_program_identity: Qwen serial arena alignment=" << alignment
                  << " allocations=" << actual.allocations.size() << " bytes=" << actual.total_bytes << " PASS\n";
    }
    const exec::BufferLifetime duplicate[] = {{0, 0, 1, 3}, {0, 2, 2, 3}};
    rejects([&] { static_cast<void>(exec::Arena::plan(duplicate)); }, "duplicate arena IDs must fail");
    const exec::BufferLifetime huge[] = {{0, 0, 0, std::numeric_limits<std::uint64_t>::max()}};
    rejects<std::overflow_error>([&] { static_cast<void>(exec::Arena::plan(huge)); }, "arena overflow must fail");
}

}  // namespace

int main() {
    try {
        compile_qwen();
        compile_inkling();
        qwen_serial_arena();
    } catch (const std::exception& error) {
        std::cerr << "exec_program_identity: FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "exec_program_identity: PASS\n";
    return 0;
}
