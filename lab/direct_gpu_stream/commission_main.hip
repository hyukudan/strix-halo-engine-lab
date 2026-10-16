#include "bounded_tcp_control.hpp"
#include "strix/direct_gpu_stream_commissioning.hpp"
#include "strix/hip_direct_gpu_pool_allocator.hpp"
#include "strix/hip_direct_gpu_stream_gpu.hpp"
#include "strix/model_content_hash.hpp"

#include <hip/hip_runtime.h>
#include "native_visibility_probe.hpp"
#include "observed_gpu.hpp"
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
    std::uint32_t dtype = 1, ring_frames = 32, polls = 10000, iterations = 1;
    std::uint64_t gpu_polls = 100001;
    std::vector<std::uint64_t> payload_sizes;
    int gpu_device = 0, runtime_version = 0, driver_version = 0;
    std::uint16_t port = 0;
    bool receiver = false, listener = false, check = false, alternate = false, duplex = false;
    bool gpu_visibility = false;
    std::chrono::milliseconds producer_delay{0}, consumer_delay{0};
    std::chrono::milliseconds timeout{10000};
    std::chrono::milliseconds gpu_timeout{10000}, gpu_pause{1};
    std::uint64_t shape = 0;
};
Options parse(int argc, char** argv) {
    std::map<std::string, std::string> values; bool opt_in = false; Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--experimental-commissioning") { if (opt_in) throw std::invalid_argument("duplicate opt-in"); opt_in = true; continue; }
        if (key == "--gpu-visibility") { if (options.gpu_visibility) throw std::invalid_argument("duplicate gpu-visibility"); options.gpu_visibility = true; continue; }
        if (key == "--check-config") { if (options.check) throw std::invalid_argument("duplicate check-config"); options.check = true; continue; }
        if (key == "--alternate-directions") { if (options.alternate) throw std::invalid_argument("duplicate alternate-directions"); options.alternate = true; continue; }
        if (key == "--duplex-pairs") { if (options.duplex) throw std::invalid_argument("duplicate duplex-pairs"); options.duplex = true; continue; }
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
    options.gpu_polls = number(take("--gpu-maximum-polls", "100001"), "gpu-maximum-polls", 1, 1000000);
    options.gpu_timeout = std::chrono::milliseconds{static_cast<std::int64_t>(number(take("--gpu-timeout-ms", "10000"), "gpu-timeout-ms", 1, 60000))};
    options.gpu_pause = std::chrono::milliseconds{static_cast<std::int64_t>(number(take("--gpu-pause-ms", "1"), "gpu-pause-ms", 1, 1000))};
    if (options.gpu_pause > options.gpu_timeout) throw std::invalid_argument("GPU pause exceeds independent GPU completion deadline");
    options.producer_delay = std::chrono::milliseconds{number(take("--producer-delay-ms", "0"), "producer-delay-ms", 0, 5000)};
    options.consumer_delay = std::chrono::milliseconds{number(take("--consumer-delay-ms", "0"), "consumer-delay-ms", 0, 5000)};
    if ((!options.gpu_visibility && (options.producer_delay.count() || options.consumer_delay.count())) ||
        options.producer_delay + options.consumer_delay >= options.timeout)
        throw std::invalid_argument("visibility delays require opt-in and room in the operation deadline");
    options.iterations = static_cast<std::uint32_t>(number(take("--iterations", "1"), "iterations", 1, 128));
    const auto sizes = take("--payload-sizes", "");
    if (!sizes.empty()) {
        std::size_t start = 0;
        while (start < sizes.size()) {
            const auto end = sizes.find(',', start);
            options.payload_sizes.push_back(number(sizes.substr(start, end == std::string::npos ? end : end - start), "payload-sizes", 1, 4095ULL * 4096ULL));
            if (end == std::string::npos) break;
            start = end + 1; if (start == sizes.size()) throw std::invalid_argument("trailing payload size delimiter");
        }
        if (options.payload_sizes.size() != options.iterations || options.payload_sizes.front() != options.bytes) throw std::invalid_argument("payload-sizes must name every transfer and match initial payload-bytes");
    } else options.payload_sizes.assign(options.iterations, options.bytes);
    if (!values.empty() || !opt_in || options.dma_device.front() != '/' || !hex_string(options.local, 32, 32) || !hex_string(options.peer, 32, 32) ||
        !hex_string(options.gpu_uuid, 32, 32) || options.local == options.peer || options.local == std::string(32, '0') || options.peer == std::string(32, '0') || options.gpu_uuid == std::string(32, '0') ||
        !hex_string(options.build_id, 2, 128) || (options.alternate && options.iterations < 2) || (options.duplex && (options.alternate || options.iterations < 2 || options.iterations % 2)) || (options.iterations > 1 && options.run.size() > 122)) throw std::invalid_argument("commissioning requires opt-in, exact identities and bounded contiguous ring geometry");
    if (options.gpu_visibility) {
        std::array<std::uint64_t, 2> direction_frames{};
        for (std::size_t index = 0; index < options.payload_sizes.size(); ++index) {
            const auto direction = ((options.alternate || options.duplex) && index % 2) ? 1U : 0U;
            const auto frames = (options.payload_sizes[index] + 4095) / 4096;
            if (direction_frames[direction] && direction_frames[direction] != frames)
                throw std::invalid_argument("legacy RX reuse requires constant frame count per direction; use a new session for other geometry");
            direction_frames[direction] = frames;
        }
    }
    std::uint64_t total_bytes = 0;
    for (const auto bytes : options.payload_sizes) {
        if (bytes > static_cast<std::uint64_t>(options.ring_frames - 1) * 4096 || bytes > 128ULL * 1024ULL * 1024ULL - total_bytes) throw std::invalid_argument("session ring or aggregate payload budget exceeded");
        total_bytes += bytes;
    }
    return options;
}
void set_shape(Options& options) {
    auto canonical = options.campaign + "\n" + options.run + "\n1\n" + std::to_string(options.seed) + "\n" + std::to_string(options.bytes) + "\n" + std::to_string(options.sequence);
    if (options.gpu_visibility) canonical += "\ngpu-normal-loads-v1\n" + std::to_string(options.producer_delay.count()) + "\n" + std::to_string(options.consumer_delay.count());
    const auto hash = sha256_bytes({reinterpret_cast<const std::byte*>(canonical.data()), canonical.size()});
    for (unsigned index = 0; index < 8; ++index) options.shape |= static_cast<std::uint64_t>(hash.bytes[index]) << (8U * index);
    if (!options.shape) options.shape = 1;
}
std::vector<Options> schedule(const Options& options) {
    std::vector<Options> steps; steps.reserve(options.iterations);
    for (std::uint32_t index = 0; index < options.iterations; ++index) {
        auto step = options; step.bytes = options.payload_sizes[index]; step.sequence = index + 1;
        if (options.iterations > 1) step.run += ".s" + std::to_string(index);
        if ((options.alternate || options.duplex) && index % 2) step.receiver = !step.receiver;
        set_shape(step); steps.push_back(std::move(step));
    }
    return steps;
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
void config_json(const Options& options, const std::string& expected_hash, bool newline = true) {
    std::cout << "{\"schema\":\"strix-direct-gpu-commissioning-config-v1\",\"executed\":false,\"production_qualified\":false,\"campaign_id\":" << json_quote(options.campaign)
        << ",\"run_id\":" << json_quote(options.run) << ",\"role\":" << json_quote(options.receiver ? "receiver" : "sender") << ",\"local_endpoint_identity\":" << json_quote(options.local)
        << ",\"peer_endpoint_identity\":" << json_quote(options.peer) << ",\"physical_link_id\":" << json_quote(options.physical_link) << ",\"connection_epoch\":" << options.epoch
        << ",\"sequence\":" << options.sequence << ",\"payload_bytes\":" << options.bytes << ",\"pattern_seed\":" << options.seed << ",\"pattern_version\":1,\"expected_payload_sha256\":" << json_quote(expected_hash)
        << ",\"model_plan_id\":" << options.plan << ",\"request_id\":" << options.request << ",\"generation\":" << options.generation << ",\"boundary_id\":" << options.boundary
        << ",\"shape_fingerprint\":" << options.shape << ",\"dtype_id\":" << options.dtype << ",\"gpu_uuid\":" << json_quote(options.gpu_uuid) << ",\"gpu_architecture\":\"gfx1151\",\"gpu_runtime_version\":" << options.runtime_version
        << ",\"gpu_driver_version\":" << options.driver_version << ",\"kernel_identity\":" << json_quote(options.kernel_release) << ",\"driver_build_id\":" << json_quote(options.build_id)
        << ",\"gpu_visibility_profile\":" << (options.gpu_visibility ? "\"gpu-normal-loads-v1\"" : "null")
        << ",\"producer_delay_ms\":" << options.producer_delay.count() << ",\"consumer_delay_ms\":" << options.consumer_delay.count()
        << ",\"driver_identity\":" << json_quote("thunderbolt_stream:gnu-build-id:" + options.build_id) << ",\"abi_identity\":\"sgu-1.0-5354475055303031\"}" << (newline ? "\n" : "");
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
    std::shared_ptr<visibility::NativeProbe> visibility_probe;
    bool visibility_self_test = false;
    std::vector<TransportBufferView> views;
    std::uint64_t bytes = 0;
    bool quiescent = true, release_unknown = false;
    void require_device() {
        HipDirectGpuExportIo observer; if (observer.inspect_device() != device) throw std::runtime_error("commissioning execution GPU identity changed");
    }
    void allocate(AllocationGenerationSource& generations, bool gpu_visibility) {
        require_device();
        for (auto& allocation : allocations) {
            hip_check(hipMalloc(&allocation, static_cast<std::size_t>(bytes)), "commissioning retained GPU allocation"); if (!allocation) throw std::runtime_error("commissioning GPU allocation missing");
            views.push_back({reinterpret_cast<std::uintptr_t>(allocation), bytes, generations.next(), TransportMemoryDomain::gpu_device, TransportBufferAccess::read_write}); require_device();
        }
        if (gpu_visibility) {
            visibility_probe = std::make_shared<visibility::NativeProbe>([this] { require_device(); }, generations.next());
            views.push_back(visibility_probe->view());
            return;
        }
        hip_check(hipHostMalloc(&readback, static_cast<std::size_t>(bytes), hipHostMallocDefault), "commissioning full readback pinned allocation"); require_device();
        hip_check(hipEventCreateWithFlags(&readback_event, hipEventDisableTiming), "commissioning readback system-fence event"); require_device();
    }
    void produce(hipStream_t stream, const Options& options) {
        require_device(); if (!stream || options.bytes > bytes) throw std::runtime_error("commissioning producer stream/extent unavailable"); quiescent = false;
        if (options.producer_delay.count()) std::this_thread::sleep_for(options.producer_delay);
        hipLaunchKernelGGL(fill_pattern, dim3(static_cast<unsigned>((options.bytes + 255) / 256)), dim3(256), 0, stream, static_cast<std::uint8_t*>(allocations[0]), options.bytes, options.seed, options.sequence);
        hip_check(hipGetLastError(), "commissioning actual GPU pattern production"); require_device();
    }
    bool verify(hipStream_t stream, const Options& options, std::string& observed_hash) {
        require_device(); if (!stream || !readback || !readback_event || options.bytes > bytes) throw std::runtime_error("commissioning retained readback resources unavailable"); quiescent = false;
        auto last = std::chrono::steady_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(options.gpu_timeout);
        if (last > std::chrono::steady_clock::time_point::max() - duration) throw std::overflow_error("commissioning readback deadline overflow");
        const auto limit = last + duration;
        const auto check_deadline = [&] { const auto now = std::chrono::steady_clock::now(); if (now < last || now >= limit) throw std::runtime_error("commissioning readback deadline expired or clock regressed"); last = now; };
        hip_check(hipMemcpyAsync(readback, allocations[2], static_cast<std::size_t>(options.bytes), hipMemcpyDeviceToHost, stream), "commissioning complete unpacked payload readback");
        require_device(); hip_check(hipEventRecord(readback_event, stream), "commissioning full payload readback event"); require_device();
        for (std::uint64_t poll = 0; poll < options.gpu_polls; ++poll) {
            check_deadline();
            require_device(); const auto result = hipEventQuery(readback_event); require_device();
            check_deadline();
            if (result == hipSuccess) {
                const auto data = std::span{static_cast<const std::byte*>(readback), static_cast<std::size_t>(options.bytes)};
                observed_hash = model_content_identity_hex(sha256_bytes(data)); return verify_direct_gpu_commissioning_pattern(data, options.seed, options.sequence);
            }
            if (result != hipErrorNotReady) hip_check(result, "commissioning readback event query");
            if (poll + 1 < options.gpu_polls) { std::this_thread::sleep_for(options.gpu_pause); check_deadline(); }
        }
        throw std::runtime_error("commissioning bounded readback completion exhausted");
    }
    visibility::Observation verify_gpu(hipStream_t stream, TransportBufferView source,
        const Options& options, visibility::Stage stage) {
        require_device();
        if (!visibility_probe || source.bytes < options.bytes) throw std::runtime_error("GPU verification resources unavailable");
        source.bytes = options.bytes; quiescent = false;
        return visibility_probe->verify(stream, source, options.seed, options.epoch, options.sequence,
            stage, options.gpu_timeout, options.gpu_pause, options.gpu_polls);
    }
    void warm_and_self_test(hipStream_t stream, TransportBufferView raw, const Options& options) {
        if (!visibility_probe || !stream || raw.bytes < bytes) throw std::runtime_error("invalid unimported RX warmup");
        // Called once before the commissioning facade can IMPORT/ARM. Never
        // write a legacy RX_REPOST buffer or overwrite it between transfers.
        auto warm = options; warm.bytes = bytes; warm.sequence = 0; warm.seed ^= 0x73584e49454c4455ULL;
        raw.bytes = bytes; quiescent = false; require_device();
        hipLaunchKernelGGL(fill_pattern, dim3(static_cast<unsigned>((bytes + 255) / 256)), dim3(256), 0, stream,
            reinterpret_cast<std::uint8_t*>(raw.address), bytes, warm.seed, warm.sequence);
        hip_check(hipGetLastError(), "fill unimported RX with an old pattern");
        if (!verify_gpu(stream, raw, warm, visibility::Stage::warm).passed()) throw std::runtime_error("GPU warmup positive control failed");
        const auto bad_index = bytes / 2;
        hipLaunchKernelGGL(visibility::flip_byte, dim3(1), dim3(1), 0, stream, reinterpret_cast<std::uint8_t*>(raw.address), bad_index);
        hip_check(hipGetLastError(), "inject one-byte GPU negative control before IMPORT");
        const auto negative = verify_gpu(stream, raw, warm, visibility::Stage::negative_control);
        if (!visibility::complete(negative.result, negative.expected) || negative.result.mismatches != 1 || negative.result.first_bad != bad_index)
            throw std::runtime_error("GPU verifier did not detect the deliberate one-byte error");
        hipLaunchKernelGGL(visibility::flip_byte, dim3(1), dim3(1), 0, stream, reinterpret_cast<std::uint8_t*>(raw.address), bad_index);
        hip_check(hipGetLastError(), "restore unimported RX after negative control");
        if (!verify_gpu(stream, raw, warm, visibility::Stage::warm).passed()) throw std::runtime_error("GPU verifier restore control failed");
        visibility_self_test = true;
    }
    bool release() noexcept {
        if (!quiescent || release_unknown) return false;
        try {
            require_device();
            if (visibility_probe) {
                visibility_probe->mark_stream_quiescent();
                if (!visibility_probe->release()) return false;
                visibility_probe.reset();
            }
            if (readback_event) { release_unknown = true; hip_check(hipEventDestroy(readback_event), "commissioning readback event release"); readback_event = nullptr; release_unknown = false; require_device(); }
            if (readback) { release_unknown = true; hip_check(hipHostFree(readback), "commissioning pinned readback release"); readback = nullptr; release_unknown = false; require_device(); }
            for (auto& allocation : allocations) if (allocation) { release_unknown = true; hip_check(hipFree(allocation), "commissioning execution allocation release"); allocation = nullptr; release_unknown = false; require_device(); }
            return true;
        } catch (...) { return false; }
    }
    ~ExecutionOwner() { static_cast<void>(release()); } // Unknown native ownership remains allocated for process life.
};
void visibility_json(const visibility::Observation& o) {
    const auto& r = o.result;
    std::cout << "{\"executed\":" << (o.executed ? "true" : "false") << ",\"passed\":" << (o.passed() ? "true" : "false")
        << ",\"nonce\":" << o.expected.nonce << ",\"started\":" << r.started << ",\"finished\":" << r.finished
        << ",\"epoch\":" << r.epoch << ",\"sequence\":" << r.sequence << ",\"stage\":" << r.stage
        << ",\"checked_bytes\":" << r.checked << ",\"mismatches\":" << r.mismatches << ",\"first_bad\":";
    if (r.first_bad == ~0ULL) std::cout << "null"; else std::cout << r.first_bad;
    std::cout << '}';
}
void receipt_json(const Options& options, const DirectDriverBinding& binding, const DirectGpuExportDevice& gpu,
    const std::string& architecture, const std::string& expected_hash, const std::string& observed_hash, const DirectGpuCommissioningReport& report, const visibility::TransferObservation& visibility_result,
    bool self_test, bool session_transfer = false) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    const auto& identity = binding.interface;
    std::cout << "{\"schema\":" << json_quote(session_transfer ? "strix-direct-gpu-commissioning-transfer-v2" : "strix-direct-gpu-commissioning-v1") << ",\"campaign_id\":" << json_quote(options.campaign) << ",\"run_id\":" << json_quote(options.run)
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
        << ",\"transport_ack\":" << boolean(report.transport_ack) << ",\"consumer_credit\":" << boolean(report.consumer_credit) << ",\"cleanup\":" << json_quote(session_transfer ? "import_held" : report.cleanup == DirectPoolReleaseStatus::released ? "released" : "retained")
        << ",\"error\":" << json_quote(report.error.substr(0, 4096));
    std::cout << ",\"gpu_visibility_profile\":" << (options.gpu_visibility ? "\"gpu-normal-loads-v1\"" : "null")
        << ",\"gpu_verifier_self_test\":" << boolean(self_test)
        << ",\"producer_delay_ms\":" << options.producer_delay.count() << ",\"consumer_delay_ms\":" << options.consumer_delay.count()
        << ",\"raw_rx_gpu\":"; visibility_json(visibility_result.raw_rx);
    std::cout << ",\"consumer_gpu\":"; visibility_json(visibility_result.consumer);
    std::cout << '}' << (session_transfer ? "" : "\n");
}
int run(const Options& options, const std::vector<Options>& steps, const std::vector<std::string>& expected_hashes) {
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
    const auto maximum_bytes = *std::max_element(options.payload_sizes.begin(), options.payload_sizes.end());
    auto execution = std::make_shared<ExecutionOwner>(); execution->device = device; execution->bytes = maximum_bytes; execution->allocate(generations, options.gpu_visibility);
    HipDirectGpuStreamExecutionOwner owner{execution, execution->views, execution->views[1], [execution] { execution->require_device(); return execution->views; }};
    auto gpu_calls = HipDirectGpuStreamGpuCalls::native();
    if (options.gpu_visibility) gpu_calls.create_event = [](hipEvent_t* event, unsigned flags) {
        return hipEventCreateWithFlags(event, flags | hipEventReleaseToSystem);
    };
    auto gpu = std::make_shared<HipDirectGpuStreamGpu>(lease, owner, HipDirectGpuStreamGpuPolicy{true, options.gpu_polls, 64ULL * 1024ULL * 1024ULL, options.gpu_timeout, options.gpu_pause}, gpu_calls);
    auto visibility_results = std::make_shared<std::vector<visibility::TransferObservation>>(steps.size());
    std::shared_ptr<DirectGpuStreamGpu> transport_gpu = gpu;
    if (options.gpu_visibility) {
        execution->warm_and_self_test(gpu->stream(), gpu->exported_binding().pool.rx, options);
        transport_gpu = std::make_shared<ObservedCommissioningGpu>(gpu,
            [execution, gpu, steps, visibility_results](const DirectDriverOperation& op) {
                const auto sequence = op.descriptor.sequence;
                if (!op.receive_path || !sequence || sequence > steps.size() || !steps[sequence - 1].receiver ||
                    op.descriptor.byte_length != steps[sequence - 1].bytes || op.descriptor.connection_epoch != steps[sequence - 1].epoch)
                    throw std::runtime_error("raw GPU verifier received a foreign operation");
                const auto& pool = gpu->exported_binding().pool.rx;
                if (pool.bytes < 4096 || op.buffer.address != pool.address ||
                    op.buffer.allocation_generation != pool.allocation_generation || op.buffer.domain != pool.domain ||
                    op.buffer.bytes != steps[sequence - 1].bytes || op.buffer.bytes > pool.bytes - 4096)
                    throw std::runtime_error("raw GPU verifier buffer is outside the exact retained RX payload");
                auto& result = (*visibility_results)[sequence - 1].raw_rx;
                if (result.executed) throw std::runtime_error("duplicate raw GPU verification");
                result.executed = true; // Launch may fail or complete after our deadline.
                result = execution->verify_gpu(gpu->stream(), op.buffer, steps[sequence - 1], visibility::Stage::raw_rx);
                if (!result.passed()) throw std::runtime_error("GPU normal loads observed incorrect raw USB4 RX bytes");
            });
    }
    auto control = std::make_shared<BoundedCommissioningTcpControl>(CommissioningTcpPolicy{options.listener, options.port, 2048, options.polls, options.timeout});
    auto commissioning = std::make_unique<LinuxDirectGpuStreamCommissioning>(inspect_direct_stream_device(options.dma_device), discovery(options), transport_gpu, control, LinuxDirectGpuStreamPolicy{true, maximum_bytes, 128ULL * 1024ULL * 1024ULL, 4096});
    const auto observed = commissioning->observe(); const auto& identity = observed.binding.interface;
    if (observed.native.query_status != DirectNativeQueryStatus::unavailable || !observed.backend.maximum_payload_bytes || identity.local_endpoint_identity != options.local || identity.peer_endpoint_identity != options.peer ||
        identity.kernel_release != options.kernel_release || identity.driver_identity != "thunderbolt_stream:gnu-build-id:" + options.build_id || identity.abi_identity != "sgu-1.0-5354475055303031") throw std::runtime_error("commissioning exact native QUERY/module/kernel/peer identity unavailable or differs from operator expectation");
    commissioning->bind_exact_observed(observed.binding);
    std::vector<DirectGpuCommissioningConfig> configs; std::vector<DirectGpuCommissioningPayload> payloads;
    auto observed_hashes = std::make_shared<std::vector<std::string>>(steps.size());
    for (std::size_t index = 0; index < steps.size(); ++index) {
        configs.push_back(config(steps[index], observed.binding));
        auto source = execution->views[0], destination = execution->views[2]; source.bytes = destination.bytes = steps[index].bytes;
        const auto step = steps[index];
        payloads.push_back({{source}, {destination}, [execution, gpu, step] { execution->produce(gpu->stream(), step); }, [execution, gpu, step, observed_hashes, visibility_results, index, destination] {
            if (!step.gpu_visibility) return execution->verify(gpu->stream(), step, (*observed_hashes)[index]);
            if (step.consumer_delay.count()) std::this_thread::sleep_for(step.consumer_delay);
            auto& result = (*visibility_results)[index].consumer;
            if (result.executed) throw std::runtime_error("duplicate GPU consumer verification");
            result.executed = true;
            result = execution->verify_gpu(gpu->stream(), destination, step, visibility::Stage::consumer);
            return result.passed();
        }});
    }
    DirectGpuCommissioningSessionReport report;
    if (steps.size() == 1) {
        auto transfer = commissioning->run_once(configs.front(), payloads.front()); report.completed_count = transfer.completed_count; report.cleanup = transfer.cleanup; report.error = transfer.error;
        report.transfers.push_back(std::move(transfer));
    } else report = options.duplex ? commissioning->run_duplex_session(configs, payloads) : commissioning->run_session(configs, payloads);
    const bool visibility_self_test = execution->visibility_self_test;
    if (report.cleanup == DirectPoolReleaseStatus::released) {
        execution->quiescent = true;
        if (!execution->release()) { report.cleanup = DirectPoolReleaseStatus::failed_retained; report.error += " execution cleanup retained"; }
    }
    commissioning.reset(); payloads.clear(); owner = {}; transport_gpu.reset(); gpu.reset(); lease.reset();
    if (report.cleanup == DirectPoolReleaseStatus::released) {
        const auto released = exported->release(); const auto drained = allocator->drain();
        if (released != DirectPoolReleaseStatus::released || drained != DirectPoolReleaseStatus::released) { report.cleanup = DirectPoolReleaseStatus::failed_retained; report.error += " export/allocator cleanup retained"; }
    }
    if (steps.size() == 1) {
        report.transfers.front().cleanup = report.cleanup; report.transfers.front().error = report.error;
        receipt_json(steps.front(), observed.binding, device, architecture, expected_hashes.front(), observed_hashes->front(), report.transfers.front(), visibility_results->front(), visibility_self_test);
    } else {
        std::cout << "{\"schema\":\"strix-direct-gpu-commissioning-session-v2\",\"campaign_id\":" << json_quote(options.campaign) << ",\"session_id\":" << json_quote(options.run)
            << ",\"schedule\":" << json_quote(options.duplex ? "duplex_pairs" : options.alternate ? "alternating" : "sequential") << ",\"concurrency\":" << (options.duplex ? 2 : 1) << ",\"requested_transfers\":" << steps.size() << ",\"completed_count\":" << report.completed_count
            << ",\"pool_binding_identity\":" << report.pool_binding_identity << ",\"import_generation\":" << report.import_generation << ",\"cleanup\":" << json_quote(report.cleanup == DirectPoolReleaseStatus::released ? "released" : "retained")
            << ",\"error\":" << json_quote(report.error.substr(0, 4096)) << ",\"production_qualified\":false,\"transfers\":[";
        for (std::size_t index = 0; index < report.transfers.size(); ++index) { if (index) std::cout << ','; receipt_json(steps[index], observed.binding, device, architecture, expected_hashes[index], (*observed_hashes)[index], report.transfers[index], (*visibility_results)[index], visibility_self_test, true); }
        std::cout << "]}\n";
    }
    return report.completed_count == steps.size() && report.cleanup == DirectPoolReleaseStatus::released && report.error.empty() ? 0 : 1;
}
void help() {
    std::cout << "Experimental Linux/HIP two-host GPU USB4 commissioning; production qualification remains false.\n"
        "--observe-gpu INDEX queries actual HIP UUID/architecture/runtime/driver only, without payload allocation/export/compute or sockets.\n"
        "--help and --check-config perform no GPU/DMA/socket work. Fresh epoch per session; complete sequential reuse.\n"
        "Required: --experimental-commissioning --campaign-id ID --run-id ID --role sender|receiver\n"
        "--dma-device /absolute/char/device --local-endpoint HEX32 --peer-endpoint HEX32 --physical-link ID\n"
        "--kernel-release RELEASE --driver-build-id HEX --gpu-uuid HEX32 --gpu-runtime-version N --gpu-driver-version N\n"
        "--epoch N --request-id N --payload-bytes N --control-mode listen|connect --control-port N\n"
        "Optional: --check-config --gpu-device N --ring-frames N --seed N --timeout-ms N --maximum-polls N\n"
        "Separate GPU waits: --gpu-timeout-ms 1..60000 --gpu-maximum-polls 1..1000000 --gpu-pause-ms 1..1000\n"
        "--model-plan-id N --generation N --boundary-id N --dtype-id N --sequence 1\n"
        "--iterations 1..128 --alternate-directions --payload-sizes N,N,... (exact per-transfer sizes; first matches payload-bytes)\n"
        "--gpu-visibility verifies raw RX before copies and the consumer payload with normal GPU loads; only 64-byte summaries return to CPU.\n"
        "--producer-delay-ms 0..5000 --consumer-delay-ms 0..5000 are explicit bounded visibility stress delays.\n"
        "--duplex-pairs selects even 2..128 transfers, one TX+RX active per pair; no production duplex qualification.\n"
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
        const auto options = parse(argc, argv); const auto steps = schedule(options); std::vector<std::string> hashes;
        auto expected = discovery(options); expected.device_identity = "operator-config-only"; expected.device_instance_generation = 1;
        expected.interface = {1, options.kernel_release, "thunderbolt_stream:gnu-build-id:" + options.build_id, "sgu-1.0-5354475055303031", options.local, options.peer, options.physical_link, options.epoch};
        for (const auto& step : steps) { validate_direct_gpu_commissioning_config(config(step, expected)); hashes.push_back(model_content_identity_hex(sha256_bytes(expected_pattern(step)))); }
        if (options.check) {
            if (steps.size() == 1) config_json(steps.front(), hashes.front());
            else {
                std::cout << "{\"schema\":\"strix-direct-gpu-commissioning-session-config-v2\",\"campaign_id\":" << json_quote(options.campaign) << ",\"session_id\":" << json_quote(options.run) << ",\"schedule\":" << json_quote(options.duplex ? "duplex_pairs" : options.alternate ? "alternating" : "sequential") << ",\"concurrency\":" << (options.duplex ? 2 : 1) << ",\"executed\":false,\"production_qualified\":false,\"transfers\":[";
                for (std::size_t index = 0; index < steps.size(); ++index) { if (index) std::cout << ','; config_json(steps[index], hashes[index], false); } std::cout << "]}\n";
            }
            return 0;
        }
        return run(options, steps, hashes);
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"strix-direct-gpu-commissioning-error-v1\",\"production_qualified\":false,\"error\":" << json_quote(error.what()) << "}\n"; return 1;
    }
}
