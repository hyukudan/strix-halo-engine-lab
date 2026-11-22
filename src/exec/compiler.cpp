#include "strix/exec/compiler.hpp"

#include "native_launch_contract.hpp"
#include "strix/fnv.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace strix::exec {
namespace {

class Identity {
public:
    explicit Identity(std::string_view domain) : value(host_fnv::field(host_fnv::offset64, domain)) {}

    void integer(std::uint64_t number) {
        for (unsigned i = 0; i < 8; ++i) {
            value = host_fnv::byte(value, static_cast<unsigned char>(number & 255U));
            number >>= 8;
        }
    }
    template <class E> requires std::is_enum_v<E>
    void enumerator(E item) { integer(static_cast<std::uint64_t>(item)); }
    void field(std::string_view text) { value = host_fnv::field(value, text); }

    std::uint64_t value;
};

std::uint64_t declaration_identity(const graph::StageGraph& graph, std::string_view model_identity) {
    Identity hash("strix.exec.declaration/v1");
    hash.field(model_identity);
    for (const auto table : {graph.before, graph.layer, graph.after}) {
        hash.integer(table.size());
        for (const auto& stage : table) {
            hash.field(stage.id);
            hash.enumerator(stage.op);
            hash.field(stage.form);
            hash.field(stage.entry);
            hash.field(stage.site);
            for (const auto input : stage.in) hash.field(input);
            hash.field(stage.out);
            hash.field(stage.operand.tensor);
            hash.integer(stage.operand.rows);
            hash.integer(stage.operand.cols);
            hash.integer(stage.operand.items);
            hash.integer(stage.operand.read);
            hash.field(stage.numeric);
            hash.enumerator(stage.rows);
            hash.enumerator(stage.split);
            hash.integer(stage.leaves);
            hash.integer(stage.reduce);
            hash.integer(stage.kernels);
            hash.field(stage.call);
            hash.integer(stage.when_all);
            hash.integer(stage.when_none);
            hash.enumerator(stage.pos);
        }
    }
    hash.integer(graph.layers.size());
    for (const auto traits : graph.layers) hash.integer(traits);
    hash.integer(graph.formats.size());
    for (const auto& format : graph.formats) {
        hash.field(format.tensor);
        hash.enumerator(format.format);
        hash.integer(format.layers.size());
        for (const auto layer : format.layers) hash.integer(layer);
    }
    hash.integer(graph.select_after);
    hash.integer(graph.group);
    return hash.value;
}

std::uint64_t execution_identity(const Program& program, std::uint64_t declaration, const CompilerOptions& options) {
    Identity hash("strix.exec.program/v1");
    hash.integer(declaration);
    hash.integer(options.weight_format_identity);
    hash.integer(options.state_format_identity);
    const auto& key = program.key;
    hash.enumerator(key.phase);
    hash.integer(key.rows);
    hash.enumerator(key.pos);
    hash.integer(key.tree);
    hash.integer(key.sampled);
    hash.integer(key.tap);
    hash.integer(key.graph_id);
    hash.enumerator(key.tier);
    hash.integer(key.ctx_bucket);
    hash.integer(program.stages.size());
    for (const auto& stage : program.stages) {
        hash.field(stage.id);
        hash.integer(static_cast<std::uint32_t>(stage.layer));
        hash.enumerator(stage.format);
    }
    hash.integer(program.launches.size());
    for (const auto& launch : program.launches) {
        hash.field(launch.entry);
        hash.field(launch.identity);
        hash.field(launch.chain_identity);
        hash.integer(static_cast<std::uint32_t>(launch.layer));
        hash.integer(launch.rows);
        hash.integer(launch.kernels);
        hash.integer(launch.stream);
        hash.integer(launch.after);
        hash.integer(launch.args_offset);
        hash.integer(launch.args_bytes);
        for (const auto blocks : launch.grid.blocks) hash.integer(blocks);
        for (const auto threads : launch.grid.threads) hash.integer(threads);
        hash.integer(launch.grid.shared_bytes);
        hash.integer(launch.capturable);
        for (const auto* part : {&launch.prologue, &launch.stages, &launch.epilogue}) {
            hash.integer(part->size());
            for (const auto& stage : *part) {
                hash.integer(stage.instance);
                for (const auto input : stage.inputs) hash.integer(input);
                hash.integer(stage.output);
            }
        }
    }
    hash.integer(program.buffers.size());
    for (const auto& buffer : program.buffers) {
        hash.field(buffer.symbol);
        hash.integer(buffer.generation);
        hash.enumerator(buffer.layout.storage);
        hash.integer(buffer.layout.row_bytes);
        hash.integer(buffer.layout.row_stride);
    }
    const auto& arena = program.arena;
    hash.integer(arena.alignment);
    hash.integer(arena.total_bytes);
    hash.integer(arena.naive_nonaliased_bytes);
    hash.integer(arena.allocations.size());
    for (const auto& allocation : arena.allocations) {
        const auto& lifetime = allocation.lifetime;
        hash.integer(lifetime.id);
        hash.integer(lifetime.first_launch);
        hash.integer(lifetime.last_launch);
        hash.integer(lifetime.bytes);
        hash.integer(allocation.offset);
    }
    hash.integer(program.states.size());
    for (const auto& state : program.states) {
        hash.integer(state.declaration);
        hash.integer(state.table_index);
        hash.integer(state.mask);
    }
    return hash.value;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("execution program size or cost overflow");
    return left + right;
}

std::uint32_t index_of(std::size_t size) {
    if (size >= no_buffer) throw std::overflow_error("execution program index overflow");
    return static_cast<std::uint32_t>(size);
}

bool in_table(std::span<const graph::StageDecl> table, const graph::StageDecl* declaration) {
    return std::any_of(table.begin(), table.end(), [=](const auto& stage) { return &stage == declaration; });
}

bool belongs_to(const graph::StageGraph& graph, const graph::Instance& instance) {
    if (instance.layer == -1)
        return in_table(graph.before, instance.decl) || in_table(graph.after, instance.decl);
    return instance.layer >= 0 && static_cast<std::size_t>(instance.layer) < graph.layers.size() &&
           in_table(graph.layer, instance.decl);
}

enum class StageUse { absent, material, prologue };

std::vector<StageUse> stage_uses(const graph::LaunchPlan& plan, std::uint32_t rows) {
    std::vector<StageUse> uses(plan.instances.size(), StageUse::absent);
    for (const auto& launch : plan.launches) {
        if (launch.rows != rows || launch.stages.empty() || launch.kernels == 0)
            throw std::invalid_argument("launch plan has an invalid shape");
        for (const auto* part : {&launch.prologue, &launch.stages, &launch.epilogue}) {
            const bool prologue = part == &launch.prologue;
            for (const auto stage : *part) {
                if (stage >= uses.size() ||
                    (uses[stage] != StageUse::absent && (!prologue || uses[stage] != StageUse::prologue)))
                    throw std::invalid_argument("launch plan has an invalid stage use");
                uses[stage] = prologue ? StageUse::prologue : StageUse::material;
            }
        }
    }
    if (std::find(uses.begin(), uses.end(), StageUse::absent) != uses.end())
        throw std::invalid_argument("launch plan omits a stage");
    return uses;
}

std::vector<StageBinding> bind_stages(const graph::StageGraph& declaration, const graph::LaunchPlan& plan,
                                      const BufferResolver& resolve, Program& program) {
    const auto uses = stage_uses(plan, program.key.rows);
    std::vector<StageBinding> bindings;
    std::map<std::string_view, BufferId> latest;
    for (const auto& instance : plan.instances) {
        if (!belongs_to(declaration, instance)) throw std::invalid_argument("launch stage is outside the declaration");
        const auto& stage = *instance.decl;
        const auto format = stage.operand.tensor.empty() ? graph::Format::none :
            graph::format_of(declaration, stage.operand.tensor, instance.layer);
        if (format != instance.format) throw std::invalid_argument("launch stage has a mismatched format");
        const auto stage_index = index_of(bindings.size());
        StageBinding binding{stage_index, {no_buffer, no_buffer, no_buffer, no_buffer}, no_buffer};
        // Resolve inputs before writes so in-place names keep both generations.
        for (std::size_t port = 0; port < binding.inputs.size(); ++port) {
            if (stage.in[port].empty()) continue;
            const auto [input, inserted] = latest.try_emplace(stage.in[port], no_buffer);
            if (inserted) {
                input->second = index_of(program.buffers.size());
                program.buffers.push_back({std::string(input->first), 0, {BufferStorage::external}});
            }
            binding.inputs[port] = input->second;
        }
        if (!stage.out.empty()) {
            const auto [output, inserted] = latest.try_emplace(stage.out, no_buffer);
            const auto generation = inserted ? 1U : program.buffers[output->second].generation + 1U;
            auto layout = uses[stage_index] == StageUse::prologue ?
                BufferLayout{BufferStorage::launch_local} : resolve(program.key, instance);
            if (layout.row_stride == 0) layout.row_stride = layout.row_bytes;
            if (layout.row_stride < layout.row_bytes ||
                (layout.storage == BufferStorage::arena && layout.row_bytes == 0))
                throw std::invalid_argument("launch buffer has an invalid row layout");
            binding.output = index_of(program.buffers.size());
            output->second = binding.output;
            program.buffers.push_back({std::string(stage.out), generation, layout});
        }
        bindings.push_back(binding);
        program.stages.push_back({std::string(stage.id), instance.layer, instance.format});
    }
    return bindings;
}

void bind_launches(const graph::LaunchPlan& plan, std::span<const StageBinding> stages, Program& program) {
    std::vector<std::uint32_t> producer(program.buffers.size(), no_launch);
    std::vector<BufferLifetime> lifetimes(program.buffers.size());
    for (const auto& launch : plan.launches) {
        const auto launch_index = index_of(program.launches.size());
        auto& bound = program.launches.emplace_back();
        bound.entry = launch.entry;
        bound.layer = launch.layer;
        bound.rows = launch.rows;
        bound.kernels = launch.kernels;
        bound.stream = launch.stream;
        bound.after = launch.after;
        const auto bind = [&](std::span<const std::uint32_t> part, std::vector<StageBinding>& destination) {
            for (const auto index : part) {
                const auto& stage = stages[index];
                for (const auto id : stage.inputs) {
                    if (id == no_buffer) continue;
                    const auto& buffer = program.buffers[id];
                    if (buffer.generation != 0 &&
                        (producer[id] == no_launch ||
                         (buffer.layout.storage == BufferStorage::launch_local && producer[id] != launch_index)))
                        throw std::invalid_argument("launch reads a buffer outside its producing schedule");
                    if (buffer.layout.storage == BufferStorage::arena) lifetimes[id].last_launch = launch_index;
                }
                if (stage.output != no_buffer) {
                    const auto id = stage.output;
                    if (producer[id] == launch_index)
                        throw std::invalid_argument("launch repeats a stage");
                    producer[id] = launch_index;
                    const auto& layout = program.buffers[id].layout;
                    if (layout.storage == BufferStorage::arena) {
                        const std::uint64_t preceding_rows = bound.rows - 1U;
                        if (preceding_rows > (std::numeric_limits<std::uint64_t>::max() - layout.row_bytes) /
                                             layout.row_stride)
                            throw std::overflow_error("launch buffer extent overflow");
                        const auto bytes = preceding_rows * layout.row_stride + layout.row_bytes;
                        lifetimes[id] = {id, launch_index, launch_index, bytes};
                    }
                }
                destination.push_back(stage);
            }
        };
        bind(launch.prologue, bound.prologue);
        bind(launch.stages, bound.stages);
        bind(launch.epilogue, bound.epilogue);
    }
    std::erase_if(lifetimes, [](const auto& lifetime) { return lifetime.bytes == 0; });
    program.arena = Arena::plan(lifetimes, program.arena.alignment);
    for (const auto& allocation : program.arena.allocations)
        program.buffers[allocation.lifetime.id].arena_offset = allocation.offset;
}

void append_args(Program& program, BoundLaunch& launch, std::span<const std::byte> args) {
    if (args.empty()) return;
    constexpr auto alignment = alignof(std::max_align_t);
    const auto padding = (alignment - program.args.size() % alignment) % alignment;
    const auto offset = checked_add(program.args.size(), padding);
    const auto end = checked_add(offset, args.size());
    if (end >= no_buffer) throw std::overflow_error("execution program argument block overflow");
    launch.args_offset = static_cast<std::uint32_t>(offset);
    launch.args_bytes = static_cast<std::uint32_t>(args.size());
    program.args.resize(launch.args_offset);
    program.args.insert(program.args.end(), args.begin(), args.end());
}

void bind_kernels(const graph::LaunchPlan& plan, const KernelResolver& resolve, const CompilerOptions& options,
                  Program& program) {
    Identity numerical("strix.exec.numerical/v1");
    numerical.integer(options.weight_format_identity);
    numerical.integer(options.state_format_identity);
    numerical.integer(program.launches.size());
    bool bitwise = true;
    bool grid_by_context = false;
    for (std::uint32_t index = 0; index < program.launches.size(); ++index) {
        auto& launch = program.launches[index];
        auto entry = resolve({program.key, plan, index, launch, program.buffers});
        const auto native_op = [&] {
            if (launch.stages.size() != 1 || launch.kernels != 1 || !launch.prologue.empty() ||
                !launch.epilogue.empty() || entry.args.empty() || entry.identity != launch.entry)
                return false;
            const auto& stage = *plan.instances[launch.stages.front().instance].decl;
            return stage.form == detail::native_transfer_form &&
                (stage.op == graph::Op::copy || stage.op == graph::Op::download || stage.op == graph::Op::clear);
        };
        const bool native = !entry.fn && native_op();
        if ((!entry.fn && !native) || entry.identity.empty())
            throw std::invalid_argument("unresolved kernel entry: " + launch.entry);
        if (native) launch.kernels = 0;
        if (entry.predicted.launches != 0 && entry.predicted.launches != launch.kernels)
            throw std::invalid_argument("kernel launch count differs from plan: " + launch.entry);
        launch.kernel = entry.kernel;
        launch.fn = entry.fn;
        launch.identity = std::move(entry.identity);
        launch.chain_identity = std::move(entry.chain_identity);
        launch.grid = entry.grid;
        launch.capturable = entry.capturable;
        append_args(program, launch, entry.args);
        program.capturable &= launch.capturable;
        grid_by_context |= entry.grid_by_context;
        bitwise &= !entry.exact_identity.empty();
        numerical.field(entry.exact_identity);
        auto& cost = program.predicted;
        cost.us += entry.predicted.us;
        cost.weight_bytes = checked_add(cost.weight_bytes, entry.predicted.weight_bytes);
        cost.state_bytes = checked_add(cost.state_bytes, entry.predicted.state_bytes);
        cost.macs = checked_add(cost.macs, entry.predicted.macs);
        cost.launches = checked_add(cost.launches, launch.kernels);
    }
    if (grid_by_context != (program.key.ctx_bucket != 0))
        throw std::invalid_argument("program context bucket must match the resolved grid policy");
    if (bitwise) program.numerical_identity = numerical.value;
}

void validate_key(const ProgramKey& key) {
    if (key.rows == 0 || key.ctx_bucket > 8 ||
        ((key.phase == Phase::decode || key.phase == Phase::verify) &&
         (key.rows > 32 || (key.rows & (key.rows - 1U)) != 0)))
        throw std::invalid_argument("invalid execution program key shape");
}

}  // namespace

Compiler::Compiler(const graph::StageGraph& declaration, BufferResolver buffers, KernelResolver kernels,
                   CompilerOptions options, PlanProvider provider)
    : declaration_(declaration), buffers_(std::move(buffers)), kernels_(std::move(kernels)),
      options_(std::move(options)), provider_(std::move(provider)),
      declaration_identity_(declaration_identity(declaration, options_.declaration_identity)) {
    if (!buffers_ || !kernels_ || options_.arena_alignment == 0)
        throw std::invalid_argument("execution compiler requires resolvers and nonzero arena alignment");
    graph::validate(declaration_);
}

Program Compiler::compile(const ProgramKey& key, const graph::LaunchPlan& plan) const {
    validate_key(key);
    static_cast<void>(index_of(plan.launches.size()));
    Program program;
    program.key = key;
    program.states = options_.states;
    program.descriptor.rows = key.rows;
    program.arena.alignment = options_.arena_alignment;
    const auto bindings = bind_stages(declaration_, plan, buffers_, program);
    bind_launches(plan, bindings, program);
    bind_kernels(plan, kernels_, options_, program);
    program.identity = execution_identity(program, declaration_identity_, options_);
    return program;
}

const Program& Compiler::get(const ProgramKey& key) {
    const auto found = programs_.find(key);
    if (found != programs_.end()) return found->second;
    validate_key(key);
    if (!provider_) throw std::logic_error("cached compilation requires a launch plan provider");
    auto program = compile(key, provider_(declaration_, key));
    return programs_.emplace(key, std::move(program)).first->second;
}

void Compiler::warm(std::span<const ProgramKey> keys) {
    for (const auto& key : keys) static_cast<void>(get(key));
}

}  // namespace strix::exec
