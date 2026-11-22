#include "strix/exec/profiler.hpp"

#include "executor_cuda.hpp"
#include "strix/device/cuda_enumerator.hpp"
#include "strix/device/runtime.hpp"
#include "strix/fnv.hpp"
#include "strix/receipt/writer.hpp"

#ifdef STRIX_X6_CUDA_CATALOGUE
#include "strix/cuda_core/norms.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace strix::exec {
namespace {

using Clock = std::chrono::steady_clock;
using device::native::check;
using receipt::detail::nullable;
using receipt::detail::object;
using serve::Json;
namespace native = device::native;

double elapsed(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template<class Input>
CudaTicket submit(CudaExecutor& executor, const Program& program, const Input& input,
                  const CudaProgramBinding* binding = nullptr) {
    if constexpr (std::is_same_v<Input, StepBlock>) return executor.submit_x3(program, input, {binding});
    else return executor.submit(program, input, {binding});
}

template<class Input>
StepProfile profile_impl(CudaExecutor& executor, const Program& program, const Input& input,
                         const ProfileOptions& options, const CudaProgramBinding* binding) {
    (void)make_step_profile(program, options);  // Reject malformed plan facts before submitting.
    if (!program.capturable || program.key.tap ||
        !std::ranges::all_of(program.launches, &BoundLaunch::capturable))
        throw std::invalid_argument("step profiling requires a capturable CUDA program");
    std::vector<CudaStepOut> out(binding && binding->readback == CudaReadback::bound ? 0 : program.key.rows);
    std::vector<double> intervals(program.launches.size());
    const auto before = executor.graph_stats(program.key);
    auto start = Clock::now();
    executor.wait(submit(executor, program, input, binding), out);
    const double first_wall = elapsed(start, Clock::now());
    const auto installed = executor.graph_stats(program.key);
    if (installed.nodes != program.launches.size() || installed.instantiations == 0)
        throw std::runtime_error("step profiling requires an installed CUDA graph");

    ProfileTimings timings;
    if (installed.instantiations != before.instantiations) timings.cold_wall_ms = first_wall;
    for (std::uint32_t i = 0; i < options.repeats; ++i) {
        start = Clock::now();
        const auto ticket = submit(executor, program, input, binding);
        const auto queued = Clock::now();
        executor.wait(ticket, out);
        timings.wall_ms += elapsed(start, Clock::now());
        timings.host_ms += elapsed(start, queued);
        for (const auto& row : out) timings.device_status |= row.status;

        const auto eager = [&] {
            if constexpr (std::is_same_v<Input, StepBlock>) return executor.profile_x3(program, input, binding);
            else return executor.profile(program, input, binding);
        }();
        for (std::size_t j = 0; j < intervals.size(); ++j) intervals[j] += eager.launch_us[j] / 1000.0;
    }
    if (executor.graph_stats(program.key).captures != installed.captures)
        throw std::runtime_error("CUDA program recaptured during step profiling");
    for (auto& ms : intervals) ms /= options.repeats;
    timings.launch_ms = intervals;
    timings.wall_ms /= options.repeats;
    timings.host_ms /= options.repeats;
    return make_step_profile(program, options, timings);
}

template<class Args>
void bind_args(Program& program, BoundLaunch& launch, const Args& args) {
    static_assert(std::is_trivially_copyable_v<Args>);
    launch.args_offset = static_cast<std::uint32_t>(program.args.size());
    launch.args_bytes = sizeof(Args);
    program.args.resize(program.args.size() + sizeof(Args));
    std::memcpy(program.args.data() + launch.args_offset, &args, sizeof(Args));
}

template<class Args>
Args read_args(std::span<const std::byte> bytes) {
    Args args{};
    std::memcpy(&args, bytes.data(), sizeof(args));
    return args;
}

struct Stamp {
    std::uint64_t start, end;
};

__device__ std::uint64_t timestamp() {
    std::uint64_t ticks;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ticks) :: "memory");
    return ticks;
}

struct StampArgs {
    Stamp* stamps;
    std::uint32_t index;
    bool pdl;
};

__global__ void stamp_node(StampArgs args) {
    const auto start = timestamp();
    if (args.pdl) {
        // Trigger after waiting so at most one successor occupies the pre-wait region.
        cudaGridDependencySynchronize();
        cudaTriggerProgrammaticLaunchCompletion();
    }
    const auto end = timestamp();
    args.stamps[args.index] = {start, end};
}

void enqueue_stamp(std::span<const std::byte> bytes, const Grid&, void* stream) {
    const auto args = read_args<StampArgs>(bytes);
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(1);
    config.stream = static_cast<cudaStream_t>(stream);
    config.attrs = &attr;
    config.numAttrs = args.pdl ? 1U : 0U;
    check(cudaLaunchKernelEx(&config, stamp_node, args), "enqueue launch gap probe");
}

// X2c's default ticket copies the output slot after the graph. The probes
// have no token output, so give that slot a defined value before the first run.
void clear_reply(CudaExecutor& executor) {
    check(cudaMemset(executor.output(), 0, sizeof(CudaStepOut) * kStepMaxRows), "initialize probe reply");
    check(cudaStreamSynchronize(nullptr), "wait for probe reply initialization");
}

Json launch_case(const bench::Options& options, const char* mode, bool replay, bool pdl) {
    native::Buffer<Stamp> buffer(options.nodes);
    std::vector<Stamp> stamps(options.nodes);
    CudaExecutor executor(pdl, replay);
    clear_reply(executor);
    Program program{};
    program.descriptor.rows = 1;
    for (std::uint32_t i = 0; i < options.nodes; ++i) {
        auto& launch = program.launches.emplace_back();
        launch.fn = enqueue_stamp;
        launch.after = i ? i - 1 : no_launch;
        bind_args(program, launch, StampArgs{buffer.data(), i, pdl});
    }
    std::array<CudaStepOut, 1> reply{};
    const auto cold_start = Clock::now();
    executor.wait(submit(executor, program, CudaStepIn{}), reply);
    const auto cold_ms = elapsed(cold_start, Clock::now());
    const auto stats = executor.graph_stats(program.key);
    const auto edges = options.nodes - 1;
    if (replay && (stats.nodes != options.nodes || stats.instantiations != 1 || stats.pdl_edges != (pdl ? edges : 0)))
        throw std::runtime_error("launch probe graph was not installed with the requested edges");

    double submit_ms = 0, span_ns = 0, gap_ns = 0;
    std::uint64_t early = 0;
    for (std::uint32_t repeat = 0; repeat < options.repeats; ++repeat) {
        const auto start = Clock::now();
        const auto ticket = submit(executor, program, CudaStepIn{});
        submit_ms += elapsed(start, Clock::now());
        executor.wait(ticket, reply);
        check(cudaMemcpy(stamps.data(), buffer.data(), stamps.size() * sizeof(Stamp), cudaMemcpyDeviceToHost),
              "read launch gap timestamps");
        for (std::size_t i = 0; i < stamps.size(); ++i) {
            const auto& stamp = stamps[i];
            if (stamp.start == 0 || stamp.end < stamp.start || (i && stamp.end < stamps[i - 1].end))
                throw std::runtime_error("invalid launch probe timestamps");
            if (!i) continue;
            const auto predecessor_end = stamps[i - 1].end;
            if (stamp.start < predecessor_end) {
                if (!pdl) throw std::runtime_error("completion edge started before its predecessor ended");
                gap_ns -= static_cast<double>(predecessor_end - stamp.start);
                ++early;
            } else {
                gap_ns += static_cast<double>(stamp.start - predecessor_end);
            }
        }
        span_ns += static_cast<double>(stamps.back().end - stamps.front().start);
    }
    if (executor.graph_stats(program.key).captures != stats.captures)
        throw std::runtime_error("launch probe recaptured during replay");
    return object({{"mode", mode}, {"status", "ok"}, {"nodes", std::uint64_t{options.nodes}},
        {"graph_nodes", std::uint64_t{stats.nodes}}, {"pdl_edges", std::uint64_t{stats.pdl_edges}},
        {"cold_wall_ms", cold_ms}, {"host_submit_us_per_step", submit_ms * 1000 / options.repeats},
        {"gpu_span_us", span_ns / 1000 / options.repeats},
        {"edge_gap_ns", gap_ns / options.repeats / edges}, {"early_edges", early}});
}

__global__ void read_probe(const uint4* data, std::size_t vectors, unsigned* sink) {
    const auto first = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const auto stride = std::size_t{gridDim.x} * blockDim.x;
    unsigned sum = 0;
    for (auto i = first; i < vectors; i += stride) {
        const uint4 loaded = data[i];
        sum += loaded.x + loaded.y + loaded.z + loaded.w;
    }
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffffU, sum, offset);
    if (threadIdx.x % 32U == 0) sink[first / 32U] = sum;
}

__device__ unsigned mix_bits(unsigned value) {
    value = (value ^ (value >> 16U)) * 0x7feb352dU;
    value = (value ^ (value >> 15U)) * 0x846ca68bU;
    return value ^ (value >> 16U);
}

__global__ void seed_probe(uint4* data, std::size_t vectors) {
    const auto first = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    const auto stride = std::size_t{gridDim.x} * blockDim.x;
    for (auto i = first; i < vectors; i += stride) {
        const auto word = static_cast<unsigned>(i * 4);
        data[i] = make_uint4(mix_bits(word), mix_bits(word + 1), mix_bits(word + 2), mix_bits(word + 3));
    }
}

class Event {
public:
    Event() { check(cudaEventCreate(&event_), "allocate probe timing event"); }
    ~Event() { (void)cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    [[nodiscard]] cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

template<class Work>
Json byte_case(const char* mode, std::uint64_t bytes, std::uint64_t traffic, std::uint32_t repeats,
               cudaStream_t stream, Work work) {
    Event begin, end;
    const auto measure = [&](std::uint32_t count) {
        check(cudaEventRecord(begin.get(), stream), "record byte probe start");
        for (std::uint32_t i = 0; i < count; ++i) work();
        check(cudaEventRecord(end.get(), stream), "record byte probe end");
        check(cudaEventSynchronize(end.get()), "wait for byte probe");
        float ms = 0;
        check(cudaEventElapsedTime(&ms, begin.get(), end.get()), "read byte probe timing");
        return static_cast<double>(ms) / count;
    };
    const double first_ms = measure(1);
    const double ms = measure(repeats);
    return object({{"mode", mode}, {"buffer_bytes", bytes}, {"traffic_bytes", traffic}, {"first_gpu_ms", first_ms},
        {"gpu_ms", ms}, {"gbytes_per_second", ms > 0 ? Json(static_cast<double>(traffic) / 1e6 / ms) : Json()}});
}

Json byte_cases(const bench::Options& options, const device::Bound& bound, const cudaDeviceProp& properties) {
    Json::Array cases;
    constexpr unsigned threads = 256;
    for (const auto bytes : options.bytes) {
        const auto vectors = static_cast<std::size_t>(bytes / sizeof(uint4));
        const auto blocks = static_cast<unsigned>(std::min((vectors + threads - 1) / threads,
            std::size_t{8} * static_cast<unsigned>(properties.multiProcessorCount)));
        const auto sink_elements = std::size_t{blocks} * (threads / 32U);
        if (bound.budget.device_capped() && 2 * bytes + sink_elements * sizeof(unsigned) > bound.budget.device_bytes)
            throw std::runtime_error("byte-rate buffers exceed the declared device budget");
        native::Buffer<uint4> input(vectors), output(vectors);
        native::Buffer<unsigned> sink(sink_elements);
        native::StreamOwner stream;
        const auto queue = stream.get();
        seed_probe<<<blocks, threads, 0, queue>>>(input.data(), vectors);
        check(cudaGetLastError(), "initialize byte probe buffer");
        native::synchronize(queue);
        cases.push_back(byte_case("copy", bytes, 2 * bytes, options.repeats, queue, [&] {
            native::copy(output.data(), input.data(), static_cast<std::size_t>(bytes), native::device_to_device, queue);
        }));
        cases.push_back(byte_case("read", bytes, bytes, options.repeats, queue, [&] {
            read_probe<<<blocks, threads, 0, queue>>>(input.data(), vectors, sink.data());
            check(cudaGetLastError(), "enqueue read bandwidth probe");
        }));
    }
    return Json(std::move(cases));
}

#ifdef STRIX_X6_CUDA_CATALOGUE

const KernelEntry& rms_entry(const bench::Options& options) {
    const auto rows = options.rows == 1 ? RowsClass::r1 : options.rows <= 8 ? RowsClass::r2_8 : RowsClass::r9_32;
    const auto catalogue = Catalog::builtin_entries(Backend::cuda);
    const auto entry = std::ranges::find_if(catalogue, [&](const KernelEntry& candidate) {
        return candidate.identity == options.entry && candidate.key.rows == rows && candidate.key.op == graph::Op::norm &&
            (candidate.key.form == "norm.rms" || candidate.key.form == "norm.rms.bf16_store");
    });
    if (entry == catalogue.end()) throw std::invalid_argument("no synthetic binder for entry " + options.entry);
    return *entry;
}

struct RmsArgs {
    const float* input;
    const void* gain;
    float* output;
    cuda_core::GroupedView view;
};

template<bool Bf16>
int enqueue_rms(void* payload, void* queue) {
    const auto& args = *static_cast<const RmsArgs*>(payload);
    const auto stream = static_cast<cudaStream_t>(queue);
    constexpr float eps = 1e-5F;
    if constexpr (Bf16)
        return cuda_core::rmsnorm_bf16(args.input, args.view, static_cast<const std::uint16_t*>(args.gain),
                                      args.output, args.view, eps, stream);
    else
        return cuda_core::rmsnorm(args.input, args.view, static_cast<const float*>(args.gain),
                                 args.output, args.view, eps, stream);
}

EntryLaunch resolve_rms(void* context, const LauncherRef& ref, const LaunchSite&) {
    if (ref.symbol == "strix::cuda_core::rmsnorm" && ref.mode == "fp32_gain") return {enqueue_rms<false>, context};
    if (ref.symbol == "strix::cuda_core::rmsnorm_bf16" && ref.mode == "bf16_gain.bf16_store")
        return {enqueue_rms<true>, context};
    return {};
}

void enqueue_d4(std::span<const std::byte> bytes, const Grid&, void* queue) {
    const auto call = read_args<EntryLaunch>(bytes);
    check(static_cast<cudaError_t>(call.enqueue(call.binding, queue)), "enqueue synthetic D4 entry");
}

template<class T>
void seed_rms(T* device, std::size_t elements, std::uint32_t seed) {
    std::vector<T> host(elements);
    for (auto& value : host) {
        seed = seed * 1664525U + 1013904223U;
        if constexpr (std::is_same_v<T, std::uint16_t>) value = static_cast<T>(0x3f00U + (seed >> 25U));
        else value = 0.5F + static_cast<float>(seed >> 16U) / 65536.F;
    }
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "seed synthetic RMS operands");
}

bench::Output run_rms(const bench::Options& options, const KernelEntry& entry, const device::Bound& bound) {
    if ((entry.arch & device::arch_mask(bound.visible.arch)) == device::ArchMask::none)
        throw std::invalid_argument("requested D4 entry does not support the selected CUDA architecture");
    const auto elements = std::size_t{options.rows} * options.width;
    const bool bf16 = entry.key.weight.name == "plain.bf16";
    const auto gain_bytes = std::uint64_t{options.width} * (bf16 ? sizeof(std::uint16_t) : sizeof(float));
    const auto tensor_bytes = std::uint64_t{elements} * sizeof(float);
    if (bound.budget.device_capped() && 2 * tensor_bytes + gain_bytes > bound.budget.device_bytes)
        throw std::runtime_error("synthetic operands exceed the declared device budget");

    native::Buffer<float> input(elements), output(elements), f32_gain(bf16 ? 0 : options.width);
    native::Buffer<std::uint16_t> bf16_gain(bf16 ? options.width : 0);
    seed_rms(input.data(), elements, 1);
    if (bf16) seed_rms(bf16_gain.data(), options.width, 2);
    else seed_rms(f32_gain.data(), options.width, 2);
    RmsArgs args{input.data(), bf16 ? static_cast<const void*>(bf16_gain.data()) : f32_gain.data(),
                 output.data(), {options.rows, 1, options.width}};

    LaunchSite site;
    site.weight_bytes = gain_bytes * options.rows;
    site.state_bytes = tensor_bytes;
    site.rows = options.rows;
    site.backend_args = &args;
    EntryBinder binder{resolve_rms, &args};
    const auto call = entry.bind(site, binder);
    if (!call) throw std::runtime_error("D4 entry declined the synthetic site");

    Program program{};
    program.key.rows = program.descriptor.rows = options.rows;
    auto identity = host_fnv::field(host_fnv::offset64, entry.identity);
    identity = host_fnv::field(identity, std::to_string(options.rows));
    program.identity = host_fnv::field(identity, std::to_string(options.width));
    auto& launch = program.launches.emplace_back();
    launch.fn = enqueue_d4;
    launch.identity = entry.identity;
    launch.entry = entry.key.form;
    launch.grid.blocks = {1, options.rows, 1};
    launch.grid.threads = {256, 1, 1};
    launch.capturable = entry.capturable;
    launch.kernels = 1;
    program.stages.push_back({"synthetic.rms", -1, graph::Format::f32});
    launch.stages.emplace_back();
    bind_args(program, launch, call);

    // Two multiplies on output plus a square and accumulate on input. The
    // nominal element count excludes padded lanes, reduction and inverse.
    const LaunchWork work{site.weight_bytes, site.state_bytes, 4 * std::uint64_t{elements}, &entry, options.peak_tops};
    CudaExecutor executor(false);
    clear_reply(executor);
    const auto result = profile(executor, program, CudaStepIn{}, {std::span(&work, 1), options.byte_gbps, options.repeats});

    receipt::Receipt receipt;
    receipt.identity.engine = "strix-bench";
    receipt.identity.implementation = "X2c CUDA graph and eager profiler";
    receipt.identity.kernel = options.entry;
    receipt.identity.build = options.build;
    receipt.identity.device = bound.visible.name;
    receipt.identity.command = options.command;
    receipt.identity.topology = "single-device";
    receipt.resolution.emplace();
    receipt.resolution->tier = receipt::Tier::base;
    receipt.resolution->description = "synthetic CUDA catalogue reference";
    auto& device = receipt.devices.emplace().emplace_back();
    device.profile_id = bound.device.profile;
    device.power_state = options.power_state;
    if (options.byte_gbps > 0) device.bytes_per_second = options.byte_gbps * 1e9;

    auto json = serve::parse_json_unique(write_step_profile(result, std::move(receipt)));
    json.set("width", std::uint64_t{options.width});
    json.set("device_arch", bound.visible.arch);
    json.set("operand_peak_tops", options.peak_tops > 0 ? Json(options.peak_tops) : Json());
    return {json.dump(), write_step_table(result)};
}

#endif

}  // namespace

StepProfile profile(CudaExecutor& executor, const Program& program, const CudaStepIn& input,
                    const ProfileOptions& options, const CudaProgramBinding* binding) {
    return profile_impl(executor, program, input, options, binding);
}

StepProfile profile(CudaExecutor& executor, const Program& program, const StepBlock& input,
                    const ProfileOptions& options, const CudaProgramBinding* binding) {
    return profile_impl(executor, program, input, options, binding);
}

bench::Output bench::run_cuda(const Options& options) {
    if (options.kind == Kind::exchange) throw std::invalid_argument("single-device runner has no exchange channel");
#ifdef STRIX_X6_CUDA_CATALOGUE
    const auto* entry = options.kind == Kind::entry ? &rms_entry(options) : nullptr;
#else
    if (options.kind == Kind::entry) throw std::runtime_error("entry requires a build with STRIX_CUDA_CORE=ON");
#endif
    device::Request request;
    request.select = options.device_id;
    request.profile = options.device_profile;
    const auto& bound = device::bind_cuda(std::move(request), options.devices, nullptr);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, bound.visible.ordinal), "read benchmark device properties");
#ifdef STRIX_X6_CUDA_CATALOGUE
    if (entry) return run_rms(options, *entry, bound);
#endif

    Json results;
    const bool launch = options.kind == Kind::launch;
    if (launch) {
        results = Json::array();
        results.push(launch_case(options, "eager", false, false));
        results.push(launch_case(options, "graph", true, false));
        results.push(properties.major >= 9 ? launch_case(options, "pdl-graph", true, true) :
            object({{"mode", "pdl-graph"}, {"status", "unsupported"}, {"reason", "requires CUDA capability 9.0 or newer"}}));
    } else {
        results = byte_cases(options, bound, properties);
    }
    return {object({{"schema", "strix-bench/v1"}, {"bench", launch ? "launch" : "byte-rate"}, {"status", "ok"},
        {"device", bound.visible.name}, {"device_arch", bound.visible.arch}, {"device_profile", bound.device.profile},
        {"power_state", options.power_state}, {"build", options.build}, {"command", options.command},
        {"repeats", std::uint64_t{options.repeats}}, {"results", std::move(results)}}).dump(), {}};
}

}  // namespace strix::exec
