#include "executor_cuda.hpp"
#include "executor_cuda_binding.hpp"
#include "strix/device/runtime.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>

namespace strix::exec {
namespace {

namespace native = device::native;
using session::StepReplayBlock;
using cuda_detail::GraphAction;

constexpr std::uint32_t max_steps = 8;
constexpr std::size_t max_tickets = 8;
constexpr auto max_rows = session::step_replay_max_rows;

struct Staging {
    StepReplayBlock input;
    CudaStepOut output[max_steps * max_rows];
};

struct Slot {
    Staging* staging = nullptr;
    cudaEvent_t done = nullptr;
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
    }
    ~Slot() {
        if (done) (void)cudaEventDestroy(done);
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

void validate_program(const Program& program) {
    if (program.launches.empty()) throw std::invalid_argument("empty CUDA program");
    for (std::size_t i = 0; i < program.launches.size(); ++i) {
        const auto& launch = program.launches[i];
        if (!launch.fn || launch.args_offset > program.args.size() ||
            launch.args_bytes > program.args.size() - launch.args_offset ||
            (launch.after != no_launch && launch.after >= i))
            throw std::invalid_argument("invalid CUDA program launch binding");
    }
}

bool capturable(const Program& program) {
    return std::all_of(program.launches.begin(), program.launches.end(),
                       [](const BoundLaunch& launch) { return launch.capturable; });
}

void launch_bound(const Program& program, const BoundLaunch& launch, cudaStream_t stream) {
    launch.fn(std::span(program.args).subspan(launch.args_offset, launch.args_bytes), launch.grid, stream);
    native::check(cudaGetLastError(), "enqueue bound CUDA launch");
}

cudaGraphNode_t captured_kernel(cudaStream_t stream, std::span<const cudaGraphNode_t> parents,
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
    std::array<cudaGraphNode_t, 3> from{};
    std::array<cudaGraphEdgeData, 3> edges{};
    count = from.size();
    native::check(cudaGraphNodeGetDependencies(node, from.data(), edges.data(), &count), "inspect bound graph edge");
    if (type != cudaGraphNodeTypeKernel || count != parents.size())
        throw std::runtime_error("expected a linear graph of bound kernels");
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

void capture_bound(const Program& program, cudaStream_t stream, bool pdl, std::size_t& pdl_edges) {
    std::vector<cudaGraphNode_t> nodes(program.launches.size());
    for (std::size_t i = 0; i < program.launches.size(); ++i) {
        const auto& launch = program.launches[i];
        const bool dependent = pdl && launch.after != no_launch;
        std::array<cudaGraphNode_t, 2> parents{};
        std::array<cudaGraphEdgeData, 2> edges{};
        std::size_t count = 0;
        if (i != 0) parents[count++] = nodes[i - 1];
        if (dependent) {
            // Preserve the serial arena order for a non-adjacent dependency.
            if (launch.after != i - 1) parents[count++] = nodes[launch.after];
            edges[count - 1].type = cudaGraphDependencyTypeProgrammatic;
            edges[count - 1].from_port = cudaGraphKernelNodePortProgrammatic;
            native::check(cudaStreamUpdateCaptureDependencies(stream, parents.data(), edges.data(), count,
                                                               cudaStreamSetCaptureDependencies),
                          "bind programmatic graph edge");
        }
        launch_bound(program, launch, stream);
        nodes[i] = captured_kernel(stream, std::span(parents).first(count), std::span(edges).first(count));
        pdl_edges += dependent;
    }
}

void stage_input(StepReplayBlock& block, const Program& program, const CudaStepIn& input, std::uint32_t steps) {
    if (steps == 0 || steps > max_steps || program.key.rows == 0 || program.key.rows > max_rows ||
        program.descriptor.rows != program.key.rows || input.rows.size() > program.key.rows)
        throw std::invalid_argument("invalid CUDA step dimensions");
    block = {};
    block.header.rows = program.key.rows;
    block.header.step = input.step;
    block.header.sampling_policy = input.sampling_policy;
    for (std::size_t i = 0; i < input.rows.size(); ++i) {
        block.rows[i] = input.rows[i];
        if (input.rows[i].flags & session::step_replay_valid) block.header.valid_mask |= std::uint32_t{1} << i;
    }
    if ((input.feed & ~block.header.valid_mask) != 0 || (steps > 1 && input.feed == 0))
        throw std::invalid_argument("invalid CUDA step feed mask");
    block.header.feed_mask = steps > 1 ? input.feed : 0;
}

}  // namespace

struct CudaExecutor::Impl {
    struct Cached {
        cuda_detail::GraphBinding binding;
        native::StepGraph graph;
        CudaGraphStats stats;
    };

    Impl(bool enable_pdl, bool enable_replay) : pdl(enable_pdl), replay(enable_replay) {
        for (auto& slot : slots) slot.allocate();
    }

    ~Impl() { (void)cudaStreamSynchronize(stream.get()); }

    Cached* graph_for(const Program& program) {
        if (!replay || !program.capturable || program.key.tap) {
            validate_program(program);
            return nullptr;
        }
        auto& cached = graphs[program.key];
        const auto action = cached.binding.select(program, epoch);
        const bool same_binding = cached.binding.matches(program, epoch);
        if (!same_binding) validate_program(program);
        if (action == GraphAction::eager) return nullptr;
        if (action == GraphAction::replay) {
            if (!same_binding) cached.binding.remember(program, epoch, true);
            return &cached;
        }
        const bool rebind = action == GraphAction::rebind;
        cached.binding.remember(program, epoch, false);
        if (!rebind) cached.graph.reset();
        cached.stats.nodes = 0;
        cached.stats.pdl_edges = 0;
        if (!capturable(program)) {
            cached.graph.invalidate();
            return nullptr;
        }

        std::size_t edges = 0;
        ++cached.stats.captures;
        const bool installed = cached.graph.prepare_program(capture_stream.get(), rebind,
            [&] { capture_bound(program, capture_stream.get(), pdl, edges); },
            [&](cudaGraph_t graph) noexcept {
                std::size_t count = 0;
                return cudaGraphGetNodes(graph, nullptr, &count) == cudaSuccess && count == program.launches.size();
            });
        if (!installed) return nullptr;

        cached.binding.remember(program, epoch, true);
        cached.stats.instantiations = cached.graph.instantiations();
        cached.stats.rebinds += rebind;
        cached.stats.nodes = program.launches.size();
        cached.stats.pdl_edges = edges;
        return &cached;
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
        std::copy_n(slot.staging->output, slot.results, out.begin());
        slot.serial = 0;
    }

    native::StreamOwner stream;
    native::StreamOwner capture_stream;
    native::Buffer<StepReplayBlock> block{1};
    native::Buffer<CudaStepOut> results{max_rows};
    std::array<Slot, max_tickets> slots;
    std::map<ProgramKey, Cached> graphs;
    std::uint64_t epoch = 0;
    std::uint64_t serial = 0;
    bool pdl;
    bool replay;
    bool failed = false;
};

CudaExecutor::CudaExecutor(bool pdl, bool replay) : impl_(std::make_unique<Impl>(pdl, replay)) {}
CudaExecutor::~CudaExecutor() = default;

StepReplayBlock* CudaExecutor::descriptor() const noexcept { return impl_->block.data(); }
CudaStepOut* CudaExecutor::output() const noexcept { return impl_->results.data(); }

CudaTicket CudaExecutor::submit(const Program& program, const CudaStepIn& input) {
    return submit_chain(program, input, 1);
}

CudaTicket CudaExecutor::submit_chain(const Program& program, const CudaStepIn& first, std::uint32_t steps) {
    auto& state = *impl_;
    if (state.serial == std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("CUDA ticket serial exhausted");
    auto& slot = state.acquire();
    auto& staging = *slot.staging;
    stage_input(staging.input, program, first, steps);
    auto* graph = state.graph_for(program);  // Capture/update cannot execute the program.
    const auto stream = state.stream.get();
    const auto rows = program.key.rows;
    try {
        native::copy(descriptor(), &staging.input, sizeof(staging.input), native::host_to_device, stream);
        for (std::uint32_t step = 0; step < steps; ++step) {
            if (graph) graph->graph.launch_program(stream);
            else for (const auto& launch : program.launches) launch_bound(program, launch, stream);
            native::copy(staging.output + step * rows, output(), rows * sizeof(CudaStepOut), native::device_to_host, stream);
        }
        native::check(cudaEventRecord(slot.done, stream), "record CUDA program ticket");
    } catch (...) {
        state.failed = true;
        throw;
    }

    slot.results = rows * steps;
    slot.serial = ++state.serial;
    return {slot.serial};
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

CudaStepProfile CudaExecutor::profile(const Program& program, const CudaStepIn& input) {
    auto& state = *impl_;
    auto& staging = state.acquire().staging->input;
    stage_input(staging, program, input, 1);
    validate_program(program);
    const auto count = program.launches.size();
    std::vector<TimingEvent> events(count + 1);
    CudaStepProfile result{std::vector<float>(count), 0};
    const auto stream = state.stream.get();
    try {
        native::copy(descriptor(), &staging, sizeof(staging), native::host_to_device, stream);
        native::check(cudaEventRecord(events.front().get(), stream), "record program timing start");
        for (std::size_t i = 0; i < count; ++i) {
            launch_bound(program, program.launches[i], stream);
            native::check(cudaEventRecord(events[i + 1].get(), stream), "record bound launch timing");
        }
        native::check(cudaEventSynchronize(events.back().get()), "wait for program timing");
        for (std::size_t i = 0; i < count; ++i) {
            float ms = 0;
            native::check(cudaEventElapsedTime(&ms, events[i].get(), events[i + 1].get()), "read bound launch timing");
            result.launch_us[i] = ms * 1000.0F;
            result.total_us += result.launch_us[i];
        }
    } catch (...) {
        state.failed = true;
        throw;
    }
    return result;
}

}  // namespace strix::exec
