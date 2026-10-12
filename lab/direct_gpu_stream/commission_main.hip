#include "bounded_tcp_control.hpp"
#include "strix/direct_gpu_stream_commissioning.hpp"
#include "strix/hip_direct_gpu_pool_allocator.hpp"
#include "strix/hip_direct_gpu_stream_gpu.hpp"
#include "strix/model_content_hash.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
using namespace strix;
using namespace strix::transport;
void hip_check(hipError_t result, const char* operation) { if (result != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result)); }
std::string json_quote(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef"; std::string result = "\"";
    for (const char character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte == '"' || byte == '\\') { result += '\\'; result += static_cast<char>(byte); }
        else if (byte < 32) { result += "\\u00"; result += hex[byte >> 4U]; result += hex[byte & 15U]; }
        else result += static_cast<char>(byte);
    }
    return result + '"';
}
std::string uuid(const std::array<std::uint8_t, 16>& value) {
    constexpr char digits[] = "0123456789abcdef"; std::string text; for (const auto byte : value) { text += digits[byte >> 4U]; text += digits[byte & 15U]; } return text;
}
bool hex_string(std::string_view value, std::size_t minimum, std::size_t maximum) {
    return value.size() >= minimum && value.size() <= maximum && value.size() % 2 == 0 && std::all_of(value.begin(), value.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
std::uint64_t number(const std::string& value, const char* name, std::uint64_t minimum, std::uint64_t maximum) {
    std::uint64_t parsed = 0; const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < minimum || parsed > maximum) throw std::invalid_argument(std::string("invalid bounded number: ") + name);
    return parsed;
}
struct Options {
    std::string campaign, run, dma_device, local, peer, gpu_uuid, kernel_release, build_id, physical_link;
    std::uint64_t epoch = 0, bytes = 0, seed = 0, request = 0, sequence = 1, plan = 1, generation = 1, boundary = 1;
    std::uint32_t dtype = 1, ring_frames = 32, polls = 10000;
    int gpu_device = 0, runtime_version = 0, driver_version = 0;
    std::uint16_t port = 0;
    bool receiver = false, listener = false, check = false;
    std::chrono::milliseconds timeout{10000};
    std::uint64_t shape = 0;
};
Options parse(int argc, char** argv) {
    std::map<std::string, std::string> values; bool opt_in = false; Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--experimental-commissioning") { if (opt_in) throw std::invalid_argument("duplicate opt-in"); opt_in = true; continue; }
        if (key == "--check-config") { if (options.check) throw std::invalid_argument("duplicate check-config"); options.check = true; continue; }
        if (index + 1 >= argc || !values.emplace(key, argv[++index]).second) throw std::invalid_argument("unknown/missing/duplicate commissioning option");
    }
    const auto take = [&](const char* key, const char* fallback = nullptr) {
        const auto found = values.find(key); if (found == values.end()) { if (fallback) return std::string(fallback); throw std::invalid_argument(std::string("required option: ") + key); }
        auto value = found->second; values.erase(found); if (value.empty() || value.size() > 4096) throw std::invalid_argument("option is empty or excessive"); return value;
    };
    options.campaign = take("--campaign-id"); options.run = take("--run-id"); options.dma_device = take("--dma-device");
    options.local = take("--local-endpoint"); options.peer = take("--peer-endpoint"); options.gpu_uuid = take("--gpu-uuid");
    options.kernel_release = take("--kernel-release"); options.build_id = take("--driver-build-id"); options.physical_link = take("--physical-link");
    const auto role = take("--role"), control = take("--control-mode");
    if ((role != "sender" && role != "receiver") || (control != "listen" && control != "connect")) throw std::invalid_argument("role/control mode is unsupported");
    options.receiver = role == "receiver"; options.listener = control == "listen";
    options.epoch = number(take("--epoch"), "epoch", 1, UINT64_MAX);
    options.request = number(take("--request-id"), "request-id", 1, UINT64_MAX);
    options.bytes = number(take("--payload-bytes"), "payload-bytes", 1, 4095ULL * 4096ULL);
    options.seed = number(take("--seed", "0"), "seed", 0, UINT64_MAX);
    options.sequence = number(take("--sequence", "1"), "sequence", 1, 1);
    options.plan = number(take("--model-plan-id", "1"), "model-plan-id", 1, UINT64_MAX);
    options.generation = number(take("--generation", "1"), "generation", 1, UINT64_MAX);
    options.boundary = number(take("--boundary-id", "1"), "boundary-id", 1, UINT64_MAX);
    options.dtype = static_cast<std::uint32_t>(number(take("--dtype-id", "1"), "dtype-id", 1, UINT32_MAX));
    options.port = static_cast<std::uint16_t>(number(take("--control-port"), "control-port", 1, UINT16_MAX));
    options.gpu_device = static_cast<int>(number(take("--gpu-device", "0"), "gpu-device", 0, 15));
    options.runtime_version = static_cast<int>(number(take("--gpu-runtime-version"), "gpu-runtime-version", 1, INT32_MAX));
    options.driver_version = static_cast<int>(number(take("--gpu-driver-version"), "gpu-driver-version", 1, INT32_MAX));
    options.ring_frames = static_cast<std::uint32_t>(number(take("--ring-frames", "32"), "ring-frames", 32, 4096));
    options.timeout = std::chrono::milliseconds{static_cast<std::int64_t>(number(take("--timeout-ms", "10000"), "timeout-ms", 1, 60000))};
    options.polls = static_cast<std::uint32_t>(number(take("--maximum-polls", "10000"), "maximum-polls", 1, 100000));
    if (!values.empty() || !opt_in || options.dma_device.front() != '/' || !hex_string(options.local, 32, 32) || !hex_string(options.peer, 32, 32) ||
        !hex_string(options.gpu_uuid, 32, 32) || options.local == options.peer || options.local == std::string(32, '0') || options.peer == std::string(32, '0') || options.gpu_uuid == std::string(32, '0') ||
        !hex_string(options.build_id, 2, 128) || options.bytes > static_cast<std::uint64_t>(options.ring_frames - 1) * 4096) throw std::invalid_argument("commissioning requires opt-in, exact identities and bounded contiguous ring geometry");
    const auto canonical = options.campaign + "\n" + options.run + "\n1\n" + std::to_string(options.seed) + "\n" + std::to_string(options.bytes) + "\n" + std::to_string(options.sequence);
    const auto hash = sha256_bytes({reinterpret_cast<const std::byte*>(canonical.data()), canonical.size()});
    for (unsigned index = 0; index < 8; ++index) options.shape |= static_cast<std::uint64_t>(hash.bytes[index]) << (8U * index);
    if (!options.shape) options.shape = 1;
    return options;
}
TransferDescriptor descriptor(const Options& options) { return {1, options.epoch, options.plan, options.request, options.generation, options.boundary, options.sequence, options.shape, options.dtype, options.bytes}; }
DirectDriverBinding discovery(const Options& options) {
    DirectDriverBinding binding; binding.interface.physical_link_id = options.physical_link; binding.interface.connection_epoch = options.epoch; return binding;
}
DirectGpuCommissioningConfig config(const Options& options, const DirectDriverBinding& binding) {
    return {options.campaign, options.run, binding, descriptor(options), options.seed, options.receiver, options.timeout, options.polls};
}
std::vector<std::byte> expected_pattern(const Options& options) {
    std::vector<std::byte> pattern(static_cast<std::size_t>(options.bytes));
    for (std::size_t index = 0; index < pattern.size(); ++index) pattern[index] = static_cast<std::byte>(direct_gpu_commissioning_pattern_byte(index, options.seed, options.sequence));
    return pattern;
}
void config_json(const Options& options, const std::string& expected_hash) {
    std::cout << "{\"schema\":\"strix-direct-gpu-commissioning-config-v1\",\"executed\":false,\"production_qualified\":false,\"campaign_id\":" << json_quote(options.campaign)
        << ",\"run_id\":" << json_quote(options.run) << ",\"role\":" << json_quote(options.receiver ? "receiver" : "sender") << ",\"local_endpoint_identity\":" << json_quote(options.local)
        << ",\"peer_endpoint_identity\":" << json_quote(options.peer) << ",\"physical_link_id\":" << json_quote(options.physical_link) << ",\"connection_epoch\":" << options.epoch
        << ",\"sequence\":" << options.sequence << ",\"payload_bytes\":" << options.bytes << ",\"pattern_seed\":" << options.seed << ",\"pattern_version\":1,\"expected_payload_sha256\":" << json_quote(expected_hash)
        << ",\"model_plan_id\":" << options.plan << ",\"request_id\":" << options.request << ",\"generation\":" << options.generation << ",\"boundary_id\":" << options.boundary
        << ",\"shape_fingerprint\":" << options.shape << ",\"dtype_id\":" << options.dtype << ",\"gpu_uuid\":" << json_quote(options.gpu_uuid) << ",\"gpu_architecture\":\"gfx1151\",\"gpu_runtime_version\":" << options.runtime_version
        << ",\"gpu_driver_version\":" << options.driver_version << ",\"kernel_identity\":" << json_quote(options.kernel_release) << ",\"driver_build_id\":" << json_quote(options.build_id)
        << ",\"driver_identity\":" << json_quote("thunderbolt_stream:gnu-build-id:" + options.build_id) << ",\"abi_identity\":\"sgu-1.0-5354475055303031\"}\n";
}
__global__ void fill_pattern(std::uint8_t* destination, std::uint64_t bytes, std::uint64_t seed, std::uint64_t sequence) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < bytes) destination[index] = direct_gpu_commissioning_pattern_byte(index, seed, sequence);
}
struct ExecutionOwner {
    DirectGpuExportDevice device;
    std::array<void*, 3> allocations{};
    void* readback = nullptr;
    hipEvent_t readback_event = nullptr;
    std::vector<TransportBufferView> views;
    std::uint64_t bytes = 0;
    bool quiescent = true, release_unknown = false;
    void require_device() {
        HipDirectGpuExportIo observer; if (observer.inspect_device() != device) throw std::runtime_error("commissioning execution GPU identity changed");
    }
    void allocate(AllocationGenerationSource& generations) {
        require_device();
        for (auto& allocation : allocations) {
            hip_check(hipMalloc(&allocation, static_cast<std::size_t>(bytes)), "commissioning retained GPU allocation"); if (!allocation) throw std::runtime_error("commissioning GPU allocation missing");
            views.push_back({reinterpret_cast<std::uintptr_t>(allocation), bytes, generations.next(), TransportMemoryDomain::gpu_device, TransportBufferAccess::read_write}); require_device();
        }
        hip_check(hipHostMalloc(&readback, static_cast<std::size_t>(bytes), hipHostMallocDefault), "commissioning full readback pinned allocation"); require_device();
        hip_check(hipEventCreateWithFlags(&readback_event, hipEventDisableTiming), "commissioning readback system-fence event"); require_device();
    }
    void produce(hipStream_t stream, const Options& options) {
        require_device(); if (!stream) throw std::runtime_error("commissioning producer stream unavailable"); quiescent = false;
        hipLaunchKernelGGL(fill_pattern, dim3(static_cast<unsigned>((bytes + 255) / 256)), dim3(256), 0, stream, static_cast<std::uint8_t*>(allocations[0]), bytes, options.seed, options.sequence);
        hip_check(hipGetLastError(), "commissioning actual GPU pattern production"); require_device();
    }
    bool verify(hipStream_t stream, const Options& options, std::string& observed_hash) {
        require_device(); if (!stream || !readback || !readback_event) throw std::runtime_error("commissioning retained readback resources unavailable"); quiescent = false;
        hip_check(hipMemcpyAsync(readback, allocations[2], static_cast<std::size_t>(bytes), hipMemcpyDeviceToHost, stream), "commissioning complete unpacked payload readback");
        require_device(); hip_check(hipEventRecord(readback_event, stream), "commissioning full payload readback event"); require_device();
        const auto limit = std::chrono::steady_clock::now() + options.timeout;
        for (std::uint32_t poll = 0; poll < options.polls; ++poll) {
            require_device(); const auto result = hipEventQuery(readback_event); require_device();
            if (std::chrono::steady_clock::now() >= limit) throw std::runtime_error("commissioning readback deadline expired");
            if (result == hipSuccess) {
                const auto data = std::span{static_cast<const std::byte*>(readback), static_cast<std::size_t>(bytes)};
                observed_hash = model_content_identity_hex(sha256_bytes(data)); return verify_direct_gpu_commissioning_pattern(data, options.seed, options.sequence);
            }
            if (result != hipErrorNotReady) hip_check(result, "commissioning readback event query");
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        throw std::runtime_error("commissioning bounded readback completion exhausted");
    }
    bool release() noexcept {
        if (!quiescent || release_unknown) return false;
        try {
            require_device();
            if (readback_event) { release_unknown = true; hip_check(hipEventDestroy(readback_event), "commissioning readback event release"); readback_event = nullptr; release_unknown = false; require_device(); }
            if (readback) { release_unknown = true; hip_check(hipHostFree(readback), "commissioning pinned readback release"); readback = nullptr; release_unknown = false; require_device(); }
            for (auto& allocation : allocations) if (allocation) { release_unknown = true; hip_check(hipFree(allocation), "commissioning execution allocation release"); allocation = nullptr; release_unknown = false; require_device(); }
            return true;
        } catch (...) { return false; }
    }
    ~ExecutionOwner() { static_cast<void>(release()); } // Unknown native ownership remains allocated for process life.
};
void receipt_json(const Options& options, const DirectDriverBinding& binding, const DirectGpuExportDevice& gpu,
    const std::string& architecture, const std::string& expected_hash, const std::string& observed_hash, const DirectGpuCommissioningReport& report) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    const auto& identity = binding.interface;
    std::cout << "{\"schema\":\"strix-direct-gpu-commissioning-v1\",\"campaign_id\":" << json_quote(options.campaign) << ",\"run_id\":" << json_quote(options.run)
        << ",\"role\":" << json_quote(options.receiver ? "receiver" : "sender") << ",\"executed\":" << boolean(report.executed) << ",\"production_qualified\":false,\"iterations\":1,\"completed_count\":" << report.completed_count
        << ",\"local_endpoint_identity\":" << json_quote(identity.local_endpoint_identity) << ",\"peer_endpoint_identity\":" << json_quote(identity.peer_endpoint_identity)
        << ",\"device_identity\":" << json_quote(binding.device_identity) << ",\"device_instance_generation\":" << binding.device_instance_generation
        << ",\"kernel_identity\":" << json_quote(identity.kernel_release) << ",\"driver_identity\":" << json_quote(identity.driver_identity) << ",\"driver_build_id\":" << json_quote(options.build_id)
        << ",\"abi_identity\":" << json_quote(identity.abi_identity) << ",\"physical_link_id\":" << json_quote(identity.physical_link_id) << ",\"connection_epoch\":" << identity.connection_epoch
        << ",\"sequence\":" << options.sequence << ",\"payload_bytes\":" << options.bytes << ",\"pattern_seed\":" << options.seed << ",\"pattern_version\":1,\"model_plan_id\":" << options.plan
        << ",\"request_id\":" << options.request << ",\"generation\":" << options.generation << ",\"boundary_id\":" << options.boundary << ",\"shape_fingerprint\":" << options.shape << ",\"dtype_id\":" << options.dtype
        << ",\"gpu_uuid\":" << json_quote(uuid(gpu.uuid)) << ",\"gpu_architecture\":" << json_quote(architecture) << ",\"gpu_runtime_version\":" << gpu.runtime_version << ",\"gpu_driver_version\":" << gpu.driver_version
        << ",\"pool_binding_identity\":" << report.pool_binding_identity << ",\"import_generation\":" << report.import_generation << ",\"expected_payload_sha256\":" << json_quote(expected_hash)
        << ",\"observed_payload_sha256\":" << (observed_hash.empty() ? "null" : json_quote(observed_hash)) << ",\"full_payload_verified\":" << boolean(report.full_payload_verified)
        << ",\"controller_frames_complete\":" << boolean(report.controller_frames_complete) << ",\"gpu_acquire_complete\":" << boolean(report.gpu_acquire_complete)
        << ",\"consumer_complete\":" << boolean(report.consumer_complete) << ",\"receive_reposted\":" << boolean(report.receive_reposted) << ",\"local_tx_complete\":" << boolean(report.local_tx_complete)
        << ",\"transport_ack\":" << boolean(report.transport_ack) << ",\"consumer_credit\":" << boolean(report.consumer_credit) << ",\"cleanup\":" << json_quote(report.cleanup == DirectPoolReleaseStatus::released ? "released" : "retained")
        << ",\"error\":" << json_quote(report.error.substr(0, 4096)) << "}\n";
}
int run(const Options& options, const std::string& expected_hash) {
    const auto driver_calls = LinuxDirectGpuStreamCalls::native();
    const auto actual_build = driver_calls.driver_build_identity();
    if (!actual_build || *actual_build != options.build_id || driver_calls.kernel_release() != options.kernel_release) throw std::runtime_error("expected provisioned experimental kernel/module is unavailable before opening any stream route");
    hip_check(hipSetDevice(options.gpu_device), "commissioning explicit GPU selection");
    auto exporter = std::make_shared<HipDirectGpuExportIo>(); const auto device = exporter->inspect_device();
    hipDeviceProp_t properties{}; hip_check(hipGetDeviceProperties(&properties, device.index), "commissioning GPU architecture observation");
    std::string architecture(properties.gcnArchName); const auto suffix = architecture.find(':'); if (suffix != std::string::npos) architecture.resize(suffix);
    if (uuid(device.uuid) != options.gpu_uuid || device.runtime_version != options.runtime_version || device.driver_version != options.driver_version || architecture != "gfx1151") throw std::runtime_error("commissioning observed GPU UUID/runtime/driver/architecture differs from operator expected identity");
    AllocationGenerationSource generations;
    auto allocator = std::make_shared<HipDirectGpuPoolAllocator>(device, HipDirectGpuPoolPolicy{true, 128ULL * 1024ULL * 1024ULL, 4096});
    const auto geometry = plan_multi_link_transport_pools(std::array{DirectFrameGeometry{4096, options.ring_frames, true}});
    auto exported = DirectGpuExportOwner::allocate_and_export(exporter, allocator, geometry, generations, device); auto lease = exported->retain_for_import();
    auto execution = std::make_shared<ExecutionOwner>(); execution->device = device; execution->bytes = options.bytes; execution->allocate(generations);
    HipDirectGpuStreamExecutionOwner owner{execution, execution->views, execution->views[1], [execution] { execution->require_device(); return execution->views; }};
    auto gpu = std::make_shared<HipDirectGpuStreamGpu>(lease, owner, HipDirectGpuStreamGpuPolicy{true, options.polls, 64ULL * 1024ULL * 1024ULL});
    auto control = std::make_shared<BoundedCommissioningTcpControl>(CommissioningTcpPolicy{options.listener, options.port, 2048, options.polls, options.timeout});
    auto commissioning = std::make_unique<LinuxDirectGpuStreamCommissioning>(inspect_direct_stream_device(options.dma_device), discovery(options), gpu, control, LinuxDirectGpuStreamPolicy{true, options.bytes, 128ULL * 1024ULL * 1024ULL, 4096});
    const auto observed = commissioning->observe(); const auto& identity = observed.binding.interface;
    if (observed.native.query_status != DirectNativeQueryStatus::unavailable || !observed.backend.maximum_payload_bytes || identity.local_endpoint_identity != options.local || identity.peer_endpoint_identity != options.peer ||
        identity.kernel_release != options.kernel_release || identity.driver_identity != "thunderbolt_stream:gnu-build-id:" + options.build_id || identity.abi_identity != "sgu-1.0-5354475055303031") throw std::runtime_error("commissioning exact native QUERY/module/kernel/peer identity unavailable or differs from operator expectation");
    commissioning->bind_exact_observed(observed.binding); const auto run_config = config(options, observed.binding); validate_direct_gpu_commissioning_config(run_config);
    std::string observed_hash;
    DirectGpuCommissioningPayload payload{{execution->views[0]}, {execution->views[2]}, [execution, gpu, &options] { execution->produce(gpu->stream(), options); }, [execution, gpu, &options, &observed_hash] { return execution->verify(gpu->stream(), options, observed_hash); }};
    auto report = commissioning->run_once(run_config, payload);
    if (report.cleanup == DirectPoolReleaseStatus::released) {
        execution->quiescent = true;
        if (!execution->release()) { report.cleanup = DirectPoolReleaseStatus::failed_retained; report.error += " execution cleanup retained"; }
    }
    commissioning.reset(); payload = {}; owner = {}; gpu.reset(); lease.reset();
    if (report.cleanup == DirectPoolReleaseStatus::released) {
        const auto released = exported->release(); const auto drained = allocator->drain();
        if (released != DirectPoolReleaseStatus::released || drained != DirectPoolReleaseStatus::released) { report.cleanup = DirectPoolReleaseStatus::failed_retained; report.error += " export/allocator cleanup retained"; }
    }
    receipt_json(options, observed.binding, device, architecture, expected_hash, observed_hash, report);
    return report.completed_count == 1 && report.cleanup == DirectPoolReleaseStatus::released && report.error.empty() ? 0 : 1;
}
void help() {
    std::cout << "Experimental Linux/HIP two-host GPU USB4 commissioning; production qualification remains false.\n"
        "--observe-gpu INDEX queries actual HIP UUID/architecture/runtime/driver only, without payload allocation/export/compute or sockets.\n"
        "--help and --check-config perform no GPU/DMA/socket work. One transfer per fresh epoch, sequence1.\n"
        "Required: --experimental-commissioning --campaign-id ID --run-id ID --role sender|receiver\n"
        "--dma-device /absolute/char/device --local-endpoint HEX32 --peer-endpoint HEX32 --physical-link ID\n"
        "--kernel-release RELEASE --driver-build-id HEX --gpu-uuid HEX32 --gpu-runtime-version N --gpu-driver-version N\n"
        "--epoch N --request-id N --payload-bytes N --control-mode listen|connect --control-port N\n"
        "Optional: --check-config --gpu-device N --ring-frames N --seed N --timeout-ms N --maximum-polls N\n"
        "--model-plan-id N --generation N --boundary-id N --dtype-id N --sequence 1\n"
        "TCP is loopback127.0.0.1 only. Operator must establish authenticated SSH tunnel externally.\n"
        "Keep JSON receipts and hardware identities outside Git. No model/state execution or performance qualification.\n";
}
void observe_gpu(const char* argument) {
    const auto index = static_cast<int>(number(argument, "observe-gpu index", 0, 15));
    if (const char* override_arch = std::getenv("HSA_OVERRIDE_GFX_VERSION"); override_arch && *override_arch) throw std::invalid_argument("GPU metadata observation rejects architecture override");
    hip_check(hipSetDevice(index), "GPU metadata explicit device selection");
    HipDirectGpuExportIo observer; const auto device = observer.inspect_device();
    hipDeviceProp_t properties{}; hip_check(hipGetDeviceProperties(&properties, device.index), "GPU metadata architecture observation");
    const auto identity = uuid(device.uuid);
    if (device.index != index || identity == std::string(32, '0') || device.runtime_version <= 0 || device.driver_version <= 0 || !properties.gcnArchName[0]) throw std::runtime_error("actual GPU metadata is unavailable");
    // Preserve the complete observed architecture feature string separately;
    // the first commissioning profile consumes its gfx1151 base identity.
    const std::string architecture_features(properties.gcnArchName); auto architecture = architecture_features;
    const auto suffix = architecture.find(':'); if (suffix != std::string::npos) architecture.resize(suffix);
    std::cout << "{\"schema\":\"strix-direct-gpu-device-observation-v1\",\"qualified\":false,\"production_qualified\":false,\"native_ready\":false,\"executed_payload\":false,\"allocation_performed\":false,\"export_performed\":false,\"gpu_index\":"
        << device.index << ",\"gpu_uuid\":" << json_quote(identity) << ",\"gpu_architecture\":" << json_quote(architecture) << ",\"gpu_architecture_features\":" << json_quote(architecture_features)
        << ",\"gpu_runtime_version\":" << device.runtime_version << ",\"gpu_driver_version\":" << device.driver_version << "}\n";
}
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") { help(); return 0; }
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--observe-gpu") { observe_gpu(argv[2]); return 0; }
        const auto options = parse(argc, argv); const auto pattern = expected_pattern(options); const auto hash = model_content_identity_hex(sha256_bytes(pattern));
        auto expected = discovery(options); expected.device_identity = "operator-config-only"; expected.device_instance_generation = 1;
        expected.interface = {1, options.kernel_release, "thunderbolt_stream:gnu-build-id:" + options.build_id, "sgu-1.0-5354475055303031", options.local, options.peer, options.physical_link, options.epoch};
        validate_direct_gpu_commissioning_config(config(options, expected));
        if (options.check) { config_json(options, hash); return 0; }
        return run(options, hash);
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"strix-direct-gpu-commissioning-error-v1\",\"production_qualified\":false,\"error\":" << json_quote(error.what()) << "}\n"; return 1;
    }
}
