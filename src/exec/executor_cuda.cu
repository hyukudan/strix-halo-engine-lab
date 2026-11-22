#include "executor_cuda.hpp"
#include "executor_cuda_binding.hpp"
#include "strix/device/runtime.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace strix::exec {
namespace {

namespace native = device::native;
using session::StepReplayBlock;
using cuda_detail::GraphAction;
using cuda_detail::empty_binding;
using cuda_detail::node_kind;

constexpr std::uint32_t max_steps = 8;
constexpr std::size_t max_tickets = 8;
constexpr auto max_rows = kStepMaxRows;
static_assert(max_rows == session::step_replay_max_rows);

struct Staging {
    StepReplayBlock input;
    StepBlock x3_input;
    CudaStepOut output[max_steps * max_rows];
};

struct Slot {
    Staging* staging = nullptr;
    cudaEvent_t done = nullptr;
    cudaEvent_t resident_ready = nullptr;
    std::uint64_t serial = 0;
    std::size_t results = 0;

    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

    void allocate() {
        void* address = nullptr;
        native::check(cudaMallocHost(&address, sizeof(Staging)), "allocate pinned program staging");
        staging = std::construct_at(static_cast<Staging*>(address));
        native::check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "allocate program ticket event");
        native::check(cudaEventCreateWithFlags(&resident_ready, cudaEventDisableTiming), "allocate resident handoff event");
    }
    ~Slot() {
        if (done) (void)cudaEventDestroy(done);
        if (resident_ready) (void)cudaEventDestroy(resident_ready);
        if (staging) (void)cudaFreeHost(staging);
    }
};

class TimingEvent {
public:
    TimingEvent() { native::check(cudaEventCreate(&event_), "allocate program timing event"); }
    ~TimingEvent() { (void)cudaEventDestroy(event_); }
    TimingEvent(const TimingEvent&) = delete;
    TimingEvent& operator=(const TimingEvent&) = delete;
    [[nodiscard]] cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

template <class Args>
Args read_args(const Program& program, const BoundLaunch& launch) {
    Args args{};
    std::memcpy(&args, program.args.data() + launch.args_offset, sizeof(args));
    return args;
}

void validate_program(const Program& program, const CudaProgramBinding& binding) {
    cuda_detail::validate_plan(program, binding);
    for (std::size_t i = 0; i < program.launches.size(); ++i) {
        const auto& launch = program.launches[i];
        if (launch.args_offset > program.args.size() || launch.args_bytes > program.args.size() - launch.args_offset)
            throw std::invalid_argument("invalid CUDA program launch binding");
        const auto kind = node_kind(binding, i);
        if (kind == CudaNodeKind::kernel) {
            if (!launch.fn) throw std::invalid_argument("missing CUDA kernel thunk");
        } else if (kind == CudaNodeKind::memset) {
            if (launch.fn || launch.args_bytes != sizeof(CudaMemsetArgs))
                throw std::invalid_argument("invalid CUDA memset binding");
            const auto args = read_args<CudaMemsetArgs>(program, launch);
            if (!args.destination || args.bytes == 0) throw std::invalid_argument("invalid CUDA memset range");
        } else {
            if (launch.fn || launch.args_bytes != sizeof(CudaCopyArgs))
                throw std::invalid_argument("invalid CUDA copy binding");
            const auto args = read_args<CudaCopyArgs>(program, launch);
            if (!args.destination || !args.source || args.bytes == 0)
                throw std::invalid_argument("invalid CUDA copy range");
            if (kind == CudaNodeKind::copy_d2h_pinned) {
                cudaPointerAttributes attributes{};
                native::check(cudaPointerGetAttributes(&attributes, args.destination), "inspect CUDA D2H destination");
                if (attributes.type != cudaMemoryTypeHost)
                    throw std::invalid_argument("CUDA D2H destination must be pinned host memory");
            }
        }
    }
}

bool capturable(const Program& program) {
    return std::all_of(program.launches.begin(), program.launches.end(),
                       [](const BoundLaunch& launch) { return launch.capturable; });
}

void launch_bound(const Program& program, std::size_t index, CudaNodeKind kind, cudaStream_t stream) {
    const auto& launch = program.launches[index];
    switch (kind) {
    case CudaNodeKind::kernel:
        launch.fn(std::span(program.args).subspan(launch.args_offset, launch.args_bytes), launch.grid, stream);
        native::check(cudaGetLastError(), "enqueue bound CUDA launch");
        break;
    case CudaNodeKind::copy_d2d:
    case CudaNodeKind::copy_d2h_pinned: {
        const auto args = read_args<CudaCopyArgs>(program, launch);
        const auto direction = kind == CudaNodeKind::copy_d2d ? native::device_to_device : native::device_to_host;
        native::copy(args.destination, args.source, args.bytes, direction, stream);
        break;
    }
    case CudaNodeKind::memset: {
        const auto args = read_args<CudaMemsetArgs>(program, launch);
        native::check(cudaMemsetAsync(args.destination, args.value, args.bytes, stream), "enqueue bound CUDA memset");
        break;
    }
    }
}

cudaGraphNode_t captured_node(cudaStream_t stream, cudaGraphNodeType expected_type,
                              std::span<const cudaGraphNode_t> parents,
                              std::span<const cudaGraphEdgeData> expected_edges) {
    cudaStreamCaptureStatus status{};
    const cudaGraphNode_t* frontier = nullptr;
    const cudaGraphEdgeData* frontier_edges = nullptr;
    std::size_t count = 0;
    native::check(cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &frontier, &frontier_edges, &count),
                  "query program capture frontier");
    if (status != cudaStreamCaptureStatusActive || count != 1 ||
        std::find(parents.begin(), parents.end(), *frontier) != parents.end())
        throw std::runtime_error("expected one bound launch on the capture stream");
    const auto node = *frontier;
    cudaGraphNodeType type{};
    native::check(cudaGraphNodeGetType(node, &type), "query bound graph node type");
    // CUDA truncates to the supplied capacity. Include room to catch an extra edge.
    std::vector<cudaGraphNode_t> from(parents.size() + 1);
    std::vector<cudaGraphEdgeData> edges(from.size());
    count = from.size();
    native::check(cudaGraphNodeGetDependencies(node, from.data(), edges.data(), &count), "inspect bound graph edge");
    if (type != expected_type || count != parents.size())
        throw std::runtime_error("captured CUDA node differs from the bound plan");
    const auto actual_parents = std::span(from).first(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto match = std::find(actual_parents.begin(), actual_parents.end(), parents[i]);
        if (match == actual_parents.end()) throw std::runtime_error("missing bound graph predecessor");
        const auto& edge = edges[static_cast<std::size_t>(match - actual_parents.begin())];
        const auto& expected = expected_edges[i];
        if (edge.type != expected.type || edge.from_port != expected.from_port || edge.to_port != expected.to_port)
            throw std::runtime_error("captured edge violates the bound launch order");
    }
    return node;
}

void capture_bound(const Program& program, std::span<const CudaNodeBinding> plan, cudaStream_t stream,
                   CudaGraphStats& captured) {
    std::vector<cudaGraphNode_t> nodes(plan.size());
    std::vector<cudaGraphNode_t> parents;
    std::vector<cudaGraphEdgeData> edges;
    for (std::size_t i = 0; i < plan.size(); ++i) {
        const auto& node = plan[i];
        parents.clear();
        edges.clear();
        for (const auto& dependency : node.after) {
            parents.push_back(nodes[dependency.predecessor]);
            cudaGraphEdgeData edge{};
            if (dependency.kind != CudaEdgeKind::completion) {
                edge.type = cudaGraphDependencyTypeProgrammatic;
                edge.from_port = dependency.kind == CudaEdgeKind::programmatic
                    ? cudaGraphKernelNodePortProgrammatic : cudaGraphKernelNodePortLaunchCompletion;
                ++captured.pdl_edges;
            }
            edges.push_back(edge);
        }
        if (i != 0) {
            native::check(cudaStreamUpdateCaptureDependencies(stream, parents.empty() ? nullptr : parents.data(),
                                                               edges.empty() ? nullptr : edges.data(), parents.size(),
                                                               cudaStreamSetCaptureDependencies), "bind CUDA graph predecessors");
        }
        launch_bound(program, i, node.kind, stream);
        auto type = cudaGraphNodeTypeKernel;
        switch (node.kind) {
        case CudaNodeKind::kernel: ++captured.kernels; break;
        case CudaNodeKind::copy_d2d:
        case CudaNodeKind::copy_d2h_pinned: type = cudaGraphNodeTypeMemcpy; ++captured.copies; break;
        case CudaNodeKind::memset: type = cudaGraphNodeTypeMemset; ++captured.memsets; break;
        }
        nodes[i] = captured_node(stream, type, parents, edges);
        ++captured.nodes;
        captured.edges += parents.size();
    }
}

void validate_dimensions(const Program& program, std::uint32_t steps) {
    if (steps == 0 || steps > max_steps || program.key.rows == 0 || program.key.rows > max_rows ||
        program.descriptor.rows != program.key.rows)
        throw std::invalid_argument("invalid CUDA step dimensions");
}

void validate_feed(std::uint32_t valid, std::uint32_t feed, std::uint32_t steps) {
    if ((feed & ~valid) != 0 || (steps > 1 && feed == 0))
        throw std::invalid_argument("invalid CUDA step feed mask");
}

const StepReplayBlock& stage_input(Staging& staging, const Program& program, const CudaStepIn& input,
                                  std::uint32_t steps) {
    validate_dimensions(program, steps);
    if (input.rows.size() > program.key.rows) throw std::invalid_argument("invalid CUDA step row count");
    auto& block = staging.input;
    block = {};
    block.header.rows = program.key.rows;
    block.header.step = input.step;
    block.header.sampling_policy = input.sampling_policy;
    for (std::size_t i = 0; i < input.rows.size(); ++i) {
        block.rows[i] = input.rows[i];
        if (input.rows[i].flags & session::step_replay_valid) block.header.valid_mask |= std::uint32_t{1} << i;
    }
    validate_feed(block.header.valid_mask, input.feed, steps);
    block.header.feed_mask = steps > 1 ? input.feed : 0;
    return block;
}

const StepBlock& stage_input(Staging& staging, const Program& program, const StepBlock& input, std::uint32_t steps) {
    validate_dimensions(program, steps);
    if (input.h.rows != program.key.rows || !std::has_single_bit(input.h.rows) || input.h.stop_count > kStepMaxStops ||
        (input.h.stop_count != 0 && step_stop_address(input.h) == 0))
        throw std::invalid_argument("invalid CUDA X3 step header");
    std::uint32_t valid = 0;
    for (std::uint32_t i = 0; i < input.h.rows; ++i) {
        if ((step_row(input, i).flags & step_valid) != 0) valid |= std::uint32_t{1} << i;
    }
    if (valid != input.h.valid_mask) throw std::invalid_argument("invalid CUDA X3 step valid mask");
    validate_feed(valid, input.h.feed_mask, steps);
    staging.x3_input = input;
    return staging.x3_input;
}

}  // namespace

struct CudaExecutor::Impl {
    struct Cached {
        cuda_detail::GraphBinding binding;
        native::StepGraph graph;
        CudaGraphStats stats;
    };

    Impl(bool enable_pdl, bool enable_replay, CudaDescriptorLayout layout)
        : block(layout == CudaDescriptorLayout::legacy ? 1U : 0U),
          x3_block(layout == CudaDescriptorLayout::x3 ? 1U : 0U), pdl(enable_pdl), replay(enable_replay) {
        for (auto& slot : slots) slot.allocate();
    }

    ~Impl() { (void)cudaStreamSynchronize(stream.get()); }

    Cached* graph_for(const Program& program, const CudaProgramBinding& binding) {
        if (!replay || !program.capturable || program.key.tap) {
            validate_program(program, binding);
            return nullptr;
        }
        auto& cached = graphs[program.key];
        const auto action = cached.binding.select(program, epoch, binding);
        const bool same_binding = cached.binding.matches(program, epoch, binding);
        if (!same_binding) validate_program(program, binding);
        if (action == GraphAction::eager) return nullptr;
        if (action == GraphAction::replay) {
            if (!same_binding) cached.binding.remember(program, epoch, true, binding);
            return &cached;
        }
        const bool rebind = action == GraphAction::rebind;
        cached.binding.remember(program, epoch, false, binding);
        if (!rebind) cached.graph.reset();
        CudaGraphStats captured{cached.stats.captures, cached.stats.instantiations, cached.stats.rebinds};
        cached.stats = captured;
        if (!capturable(program)) {
            cached.graph.invalidate();
            return nullptr;
        }

        const auto plan = cuda_detail::capture_plan(program, binding, pdl);
        ++cached.stats.captures;
        const bool installed = cached.graph.prepare_program(capture_stream.get(), rebind,
            [&] { capture_bound(program, plan, capture_stream.get(), captured); },
            [&](cudaGraph_t graph) noexcept {
                std::size_t count = 0;
                return cudaGraphGetNodes(graph, nullptr, &count) == cudaSuccess && count == program.launches.size();
            });
        if (!installed) return nullptr;

        cached.binding.remember(program, epoch, true, binding);
        captured.captures = cached.stats.captures;
        captured.instantiations = cached.graph.instantiations();
        captured.rebinds += rebind;
        cached.stats = captured;
        return &cached;
    }

    template <class Block, class Input>
    CudaTicket submit(const Program& program, const Input& input, Block* descriptor, std::uint32_t steps,
                      CudaSubmitOptions options) {
        if (!descriptor) throw std::invalid_argument("CUDA input does not match the descriptor layout");
        if (serial == std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("CUDA ticket serial exhausted");
        const auto& binding = options.binding ? *options.binding : empty_binding;
        const bool readback = binding.readback == CudaReadback::executor;
        if (!readback && steps != 1) throw std::invalid_argument("bound CUDA readback is single-step");
        auto& slot = acquire();
        auto& staging = *slot.staging;
        const auto& staged = stage_input(staging, program, input, steps);
        auto* graph = graph_for(program, binding);  // Capture/update cannot execute the program.
        const auto queue = stream.get();
        const auto resident = static_cast<cudaStream_t>(options.resident_stream);
        const auto rows = program.key.rows;
        try {
            if (resident) {
                native::check(cudaEventRecord(slot.resident_ready, resident), "record resident work before CUDA program");
                native::check(cudaStreamWaitEvent(queue, slot.resident_ready, 0), "wait for resident work");
            }
            native::copy(descriptor, &staged, sizeof(staged), native::host_to_device, queue);
            for (std::uint32_t step = 0; step < steps; ++step) {
                if (graph) graph->graph.launch_program(queue);
                else for (std::size_t i = 0; i < program.launches.size(); ++i)
                    launch_bound(program, i, node_kind(binding, i), queue);
                if (readback) native::copy(staging.output + step * rows, results.data(), rows * sizeof(CudaStepOut),
                                           native::device_to_host, queue);
            }
            native::check(cudaEventRecord(slot.done, queue), "record CUDA program ticket");
            if (resident) native::check(cudaStreamWaitEvent(resident, slot.done, 0), "wait for CUDA program on resident stream");
        } catch (...) {
            failed = true;
            // If submission failed before the completion dependency could be
            // queued, the resident owner may retire its buffers on unwind.
            if (resident) (void)cudaStreamSynchronize(queue);
            throw;
        }
        slot.results = readback ? rows * steps : 0;
        slot.serial = ++serial;
        return {slot.serial};
    }

    template <class Block, class Input>
    CudaStepProfile profile(const Program& program, const Input& input, Block* descriptor,
                            const CudaProgramBinding& binding) {
        if (!descriptor) throw std::invalid_argument("CUDA input does not match the descriptor layout");
        const auto& staged = stage_input(*acquire().staging, program, input, 1);
        validate_program(program, binding);
        const auto count = program.launches.size();
        std::vector<TimingEvent> events(count + 1);
        CudaStepProfile result{std::vector<float>(count), 0};
        const auto queue = stream.get();
        try {
            native::copy(descriptor, &staged, sizeof(staged), native::host_to_device, queue);
            native::check(cudaEventRecord(events.front().get(), queue), "record program timing start");
            for (std::size_t i = 0; i < count; ++i) {
                launch_bound(program, i, node_kind(binding, i), queue);
                native::check(cudaEventRecord(events[i + 1].get(), queue), "record bound launch timing");
            }
            native::check(cudaEventSynchronize(events.back().get()), "wait for program timing");
            for (std::size_t i = 0; i < count; ++i) {
                float ms = 0;
                native::check(cudaEventElapsedTime(&ms, events[i].get(), events[i + 1].get()), "read bound launch timing");
                result.launch_us[i] = ms * 1000.0F;
                result.total_us += result.launch_us[i];
            }
        } catch (...) {
            failed = true;
            throw;
        }
        return result;
    }

    Slot& acquire() {
        if (failed) throw std::runtime_error("CUDA executor has an unresolved submission error");
        const auto it = std::find_if(slots.begin(), slots.end(), [](const Slot& slot) { return slot.serial == 0; });
        if (it == slots.end()) throw std::runtime_error("no CUDA ticket available");
        return *it;
    }

    Slot& lookup(CudaTicket ticket, std::size_t result_count) {
        const auto it = std::find_if(slots.begin(), slots.end(), [&](const Slot& slot) {
            return ticket.serial != 0 && slot.serial == ticket.serial;
        });
        if (it == slots.end() || it->results != result_count)
            throw std::invalid_argument("invalid CUDA ticket or result span");
        return *it;
    }

    static void retire(Slot& slot, std::span<CudaStepOut> out) {
        if (slot.results != 0) std::copy_n(slot.staging->output, slot.results, out.begin());
        slot.serial = 0;
    }

    native::StreamOwner stream;
    native::StreamOwner capture_stream;
    native::Buffer<StepReplayBlock> block;
    native::Buffer<StepBlock> x3_block;
    native::Buffer<CudaStepOut> results{max_rows};
    std::array<Slot, max_tickets> slots;
    std::map<ProgramKey, Cached> graphs;
    std::uint64_t epoch = 0;
    std::uint64_t serial = 0;
    bool pdl;
    bool replay;
    bool failed = false;
};

CudaExecutor::CudaExecutor(bool pdl, bool replay, CudaDescriptorLayout layout)
    : impl_(std::make_unique<Impl>(pdl, replay, layout)) {}
CudaExecutor::~CudaExecutor() = default;

StepReplayBlock* CudaExecutor::descriptor() const noexcept { return impl_->block.data(); }
StepBlock* CudaExecutor::step_descriptor() const noexcept { return impl_->x3_block.data(); }
CudaStepOut* CudaExecutor::output() const noexcept { return impl_->results.data(); }

CudaTicket CudaExecutor::submit(const Program& program, const CudaStepIn& input, CudaSubmitOptions options) {
    return submit_chain(program, input, 1, options);
}

CudaTicket CudaExecutor::submit_chain(const Program& program, const CudaStepIn& first, std::uint32_t steps,
                                      CudaSubmitOptions options) {
    return impl_->submit(program, first, descriptor(), steps, options);
}

CudaTicket CudaExecutor::submit_x3(const Program& program, const StepBlock& input, CudaSubmitOptions options) {
    return submit_x3_chain(program, input, 1, options);
}

CudaTicket CudaExecutor::submit_x3_chain(const Program& program, const StepBlock& first, std::uint32_t steps,
                                         CudaSubmitOptions options) {
    return impl_->submit(program, first, step_descriptor(), steps, options);
}

bool CudaExecutor::poll(CudaTicket ticket, std::span<CudaStepOut> out) {
    auto& slot = impl_->lookup(ticket, out.size());
    const auto status = cudaEventQuery(slot.done);
    if (status == cudaErrorNotReady) return false;
    if (status != cudaSuccess) impl_->failed = true;
    native::check(status, "poll CUDA program ticket");
    Impl::retire(slot, out);
    return true;
}

void CudaExecutor::wait(CudaTicket ticket, std::span<CudaStepOut> out) {
    auto& slot = impl_->lookup(ticket, out.size());
    const auto status = cudaEventSynchronize(slot.done);
    if (status != cudaSuccess) impl_->failed = true;
    native::check(status, "wait for CUDA program ticket");
    Impl::retire(slot, out);
}

void CudaExecutor::invalidate(std::uint64_t layout_epoch) noexcept { impl_->epoch = layout_epoch; }

CudaGraphStats CudaExecutor::graph_stats(const ProgramKey& key) const noexcept {
    const auto it = impl_->graphs.find(key);
    return it == impl_->graphs.end() ? CudaGraphStats{} : it->second.stats;
}

CudaStepProfile CudaExecutor::profile(const Program& program, const CudaStepIn& input,
                                      const CudaProgramBinding* binding) {
    return impl_->profile(program, input, descriptor(), binding ? *binding : empty_binding);
}

CudaStepProfile CudaExecutor::profile_x3(const Program& program, const StepBlock& input,
                                         const CudaProgramBinding* binding) {
    return impl_->profile(program, input, step_descriptor(), binding ? *binding : empty_binding);
}

}  // namespace strix::exec
