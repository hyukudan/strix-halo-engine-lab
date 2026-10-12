#include "strix/linux_direct_transport_driver.hpp"
#include "strix/direct_gpu_stream_commissioning.hpp"
#include "direct_gpu_stream_fixture.hpp"
#include "direct_native_fixture.hpp"

#include <deque>
#include <functional>
#include <iostream>
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>

#if defined(__linux__)
#include "strix/linux_direct_gpu_stream_uapi.h"
#endif

namespace {
std::atomic<int> failures{0};
void expect(bool condition, const char* message) { if (!condition) { std::cerr << message << '\n'; ++failures; } }
template<class F> bool rejected(F function) { try { function(); } catch (const std::exception&) { return true; } return false; }
#if defined(__linux__)
using namespace direct_gpu_stream_fixture;
DirectStreamDeviceIdentity route() { return {"/synthetic/direct-gpu", 11, 12, 77, true}; }
class Fd final : public DirectStreamControlIo {
public:
    DirectStreamFd open(const DirectStreamDeviceIdentity& value) override { device = value; return {20, 7}; }
    DirectStreamFdIdentity inspect(DirectStreamFd) override { return {device, generation}; }
    DirectStreamIoResult read(DirectStreamFd, std::span<std::byte> bytes) override {
        if (incoming.empty()) return {DirectStreamIoStatus::would_block, 0};
        const auto count = std::min<std::size_t>({bytes.size(), incoming.size(), 7});
        for (std::size_t i = 0; i < count; ++i) { bytes[i] = incoming.front(); incoming.pop_front(); } return {DirectStreamIoStatus::transferred, count};
    }
    DirectStreamIoResult write(DirectStreamFd, std::span<const std::byte> bytes) override {
        const auto count = std::min<std::size_t>(bytes.size(), 11); written.insert(written.end(), bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(count)); return {DirectStreamIoStatus::transferred, count};
    }
    DirectStreamWaitStatus wait(DirectStreamFd, bool, DirectStreamClock::time_point deadline) override {
        if (incoming.empty()) { tick = deadline; return DirectStreamWaitStatus::timed_out; } return DirectStreamWaitStatus::ready;
    }
    DirectStreamClock::time_point now() const override { tick += std::chrono::microseconds{1}; return tick; }
    void close(DirectStreamFd) noexcept override { ++closed; }
    void frame(const DistributedDirectGpuControlFrame& message) {
        const auto bytes = encode_distributed_direct_gpu_control(message, {2, 2048, 1});
        for (unsigned i = 0; i < 4; ++i) incoming.push_back(static_cast<std::byte>((bytes.size() >> (i * 8U)) & 255U));
        incoming.insert(incoming.end(), bytes.begin(), bytes.end());
    }
    DirectStreamDeviceIdentity device;
    std::uint64_t generation = 7;
    mutable DirectStreamClock::time_point tick{};
    int closed = 0;
    std::deque<std::byte> incoming;
    std::vector<std::byte> written;
};
DirectDriverBinding expected() {
    return {1, "/synthetic/direct-gpu#77:08000000000000000000000000000000:99:3:4", 8,
        {1, "synthetic-kernel", "thunderbolt_stream:gnu-build-id:01020304", "sgu-1.0-5354475055303031",
         "01000000000000000000000000000000", "02000000000000000000000000000000", "synthetic-physical-link", 77}};
}
class Qualification final : public DirectGpuStreamQualification {
public:
    // Synthetic proof declaration only; not a native qualifier or factory.
    DirectNativeCapabilities proof(const DirectDriverBinding& binding, const DirectGpuExportBinding& pool) override {
        expected_pool = pool.pool_binding_identity; return binding == approved_binding ? synthetic_native_capabilities(binding.interface) : DirectNativeCapabilities{};
    }
    bool current(const DirectDriverBinding& binding, const DirectGpuExportBinding& pool) const override { return valid && binding == approved_binding && pool.pool_binding_identity == expected_pool; }
    DirectDriverBinding approved_binding = expected();
    bool valid = true;
    std::uint64_t expected_pool = 0;
};
struct Kernel {
    sgu_query query{};
    sgu_prepare prepared{};
    sgu_import imported{};
    std::deque<sgu_event> events;
    int imports = 0, arms = 0, submits = 0, stops = 0, releases = 0, reposts = 0, retires = 0, cancels = 0;
    bool import_error = false, bad_import_ack = false, bad_stop_proof = false;
    std::function<void(sgu_event&)> event_mutation;
    std::function<void()> after_import;
    std::function<void(const sgu_prepare&)> after_prepare;
    std::vector<sgu_prepare> preparations;
    Kernel() {
        query.fence.abi_major = 1; query.fence.abi_cookie = SGU_ABI_COOKIE; query.fence.device_instance[0] = 8;
        query.fence.local_uuid[0] = 1; query.fence.peer_uuid[0] = 2; query.fence.route = 99; query.fence.in_hopid = 3; query.fence.out_hopid = 4;
        query.frame_bytes = 4096; query.ring_frames = 32; query.maximum_sg_entries = 65536; query.maximum_window_bytes = 32 * 4096;
        query.maximum_imported_bytes = SGU_MAX_IMPORTED_BYTES; query.maximum_event_batch = 64;
    }
    void completed() {
        sgu_event event{}; event.fence = query.fence; event.handle = prepared.handle; event.sequence = prepared.sequence;
        event.payload_bytes = prepared.payload_bytes; event.monotonic_ns = 1000000000; event.direction = prepared.direction;
        event.total_frames = static_cast<std::uint32_t>((prepared.payload_bytes + 4095) / 4096 + 1); event.count = event.total_frames;
        for (std::uint32_t i = 0; i < event.count; ++i) {
            auto& frame = event.frames[i]; const auto terminal = i + 1 == event.count;
            frame.pool_offset = terminal ? query.maximum_window_bytes - 4096 : prepared.pool_offset + static_cast<std::uint64_t>(i) * 4096;
            frame.capacity = 4096; frame.expected_bytes = terminal ? 32 : static_cast<std::uint32_t>(std::min<std::uint64_t>(4096, prepared.payload_bytes - static_cast<std::uint64_t>(i) * 4096));
            frame.actual_bytes = frame.expected_bytes; frame.physical_slot = static_cast<std::uint32_t>(frame.pool_offset / 4096);
        }
        events.push_back(event);
    }
    LinuxDirectGpuStreamSyscallResult call(unsigned long command, void* argument) {
        if (command == SGU_IOC_QUERY) { *static_cast<sgu_query*>(argument) = query; return {0, 0}; }
        if (command == SGU_IOC_IMPORT) {
            ++imports; auto& request = *static_cast<sgu_import*>(argument); imported = request; query.fence = request.fence; query.fence.import_generation = 11;
            query.state = SGU_STATE_IMPORTED; request.fence = query.fence;
            if (bad_import_ack) ++request.rx.window_offset;
            if (after_import) after_import();
            return import_error ? LinuxDirectGpuStreamSyscallResult{-1, EFAULT} : LinuxDirectGpuStreamSyscallResult{0, 0};
        }
        if (command == SGU_IOC_PREPARE) { prepared = *static_cast<sgu_prepare*>(argument); preparations.push_back(prepared); if (after_prepare) after_prepare(prepared); return {0, 0}; }
        if (command == SGU_IOC_EVENT) {
            if (events.empty()) return {-1, EAGAIN};
            auto event = events.front(); events.pop_front(); if (event_mutation) event_mutation(event); *static_cast<sgu_event*>(argument) = event; return {0, 0};
        }
        auto& action = *static_cast<sgu_action*>(argument);
        if (command == SGU_IOC_ARM) { ++arms; query.state = SGU_STATE_ARMED; }
        else if (command == SGU_IOC_SUBMIT) { ++submits; completed(); action.queued_frames = events.back().total_frames; }
        else if (command == SGU_IOC_STOP) { ++stops; query.state = SGU_STATE_STOPPED; action.result = bad_stop_proof ? SGU_RESULT_CANCEL_REQUESTED : SGU_RESULT_CONTROLLER_STOPPED; }
        else if (command == SGU_IOC_RELEASE) { ++releases; query.state = SGU_STATE_OPEN; query.fence.connection_epoch = query.fence.pool_binding_identity = query.fence.import_generation = 0; }
        else if (command == SGU_IOC_RX_REPOST) { ++reposts; action.result = SGU_RESULT_RX_REPOSTED; action.queued_frames = static_cast<std::uint32_t>((prepared.payload_bytes + 4095) / 4096 + 1); }
        else if (command == SGU_IOC_RETIRE) ++retires;
        else if (command == SGU_IOC_CANCEL) { ++cancels; action.result = SGU_RESULT_CANCEL_REQUESTED; }
        else return {-1, ENOTTY};
        return {0, 0};
    }
};
struct Fixture {
    Pool pool;
    std::shared_ptr<Gpu> gpu = std::make_shared<Gpu>(pool.lease);
    std::shared_ptr<Fd> native_fd = std::make_shared<Fd>(), control_fd = std::make_shared<Fd>();
    std::shared_ptr<Kernel> kernel = std::make_shared<Kernel>();
    std::shared_ptr<Qualification> qualification = std::make_shared<Qualification>();
    std::shared_ptr<std::optional<std::string>> build_id = std::make_shared<std::optional<std::string>>("01020304");
    std::shared_ptr<DirectStreamControlByteStream> control = std::make_shared<DirectStreamControlByteStream>(DirectStreamControlByteStream::open_with_io(*control_fd, route()));
    std::unique_ptr<LinuxDirectTransportDriverIo> provider;
    std::shared_ptr<DirectDriverImport> imported;
    Fixture(bool qualified = true, bool discover = false, bool create_provider = true) {
        LinuxDirectGpuStreamCalls calls; calls.fd_owner = native_fd; calls.kernel_release = [] { return "synthetic-kernel"; };
        const auto observed_build = build_id; calls.driver_build_identity = [observed_build] { return *observed_build; };
        const auto retained_kernel = kernel; calls.ioctl_call = [retained_kernel](DirectStreamFd, unsigned long command, void* argument) { return retained_kernel->call(command, argument); };
        auto requested = expected(); if (discover) { requested.device_identity.clear(); requested.device_instance_generation = 0; }
        if (create_provider) provider = std::make_unique<LinuxDirectTransportDriverIo>(route(), requested, gpu, control, LinuxDirectGpuStreamPolicy{true, 65536, 1048576, 64}, qualified ? qualification : nullptr, calls);
    }
    void open() {
        expect(provider->query().native.query_status == DirectNativeQueryStatus::verified, "synthetic qualification explicitly gates test provider");
        const auto& pool_binding = pool.exported->binding();
        imported = provider->prepare_import(expected(), {"synthetic-physical-link", 77, pool_binding.pool_binding_identity, 0, 1}, pool_binding.pool); provider->activate_import(imported);
    }
    TransferDescriptor descriptor(std::uint64_t sequence = 1) const { return {1, 77, 2, 3, 4, 5, sequence, 6, 1, 5120}; }
    DirectDriverOperation operation(bool receive, std::uint64_t sequence = 1) const {
        auto view = receive ? imported->binding().pool.rx : imported->binding().pool.tx; view.bytes = 5120;
        return {imported->binding(), receive ? 21ULL : 11ULL, descriptor(sequence), view, receive};
    }
    GpuCommitRecord commit(std::uint64_t sequence = 1) const { return {1, gpu_commit_word, 77, sequence, 5120, 0, 1}; }
    void incoming(DistributedDirectGpuControlKind kind, TransferDescriptor descriptor_value) {
        DistributedDirectGpuControlFrame frame; frame.kind = kind; frame.physical_link_id = expected().interface.physical_link_id;
        frame.local_endpoint_identity = expected().interface.peer_endpoint_identity; frame.peer_endpoint_identity = expected().interface.local_endpoint_identity;
        frame.descriptor = descriptor_value; if (kind == DistributedDirectGpuControlKind::producer_released) { frame.commit = commit(descriptor_value.sequence); frame.commit.payload_bytes = static_cast<std::uint32_t>(descriptor_value.byte_length); }
        control_fd->frame(frame);
    }
    void send_ready() { incoming(DistributedDirectGpuControlKind::receive_ready, descriptor()); }
};
// Two entirely synthetic peers exchange actual bounded kind4 byte frames and
// CPU byte-vector payloads. No socket, GPU or controller is used by this fixture.
struct PeerControlQueues {
    std::mutex mutex; std::condition_variable changed;
    std::array<std::deque<std::vector<std::byte>>, 2> incoming;
};
class PeerControl final : public DistributedControlByteStream {
public:
    PeerControl(std::shared_ptr<PeerControlQueues> queues, std::size_t side) : queues_(std::move(queues)), side_(side) {}
    void send(std::span<const std::byte> value) override {
        if (value.empty() || value.size() > 2048) throw std::runtime_error("synthetic peer frame budget");
        { std::lock_guard lock(queues_->mutex); auto& target = queues_->incoming[1 - side_]; if (target.size() >= 4) throw std::runtime_error("synthetic peer queue budget"); target.emplace_back(value.begin(), value.end()); }
        queues_->changed.notify_all();
    }
    std::vector<std::byte> receive() override {
        std::unique_lock lock(queues_->mutex);
        if (!queues_->changed.wait_for(lock, std::chrono::milliseconds{1000}, [&] { return !queues_->incoming[side_].empty(); })) throw std::runtime_error("synthetic peer bounded control timeout");
        auto value = std::move(queues_->incoming[side_].front()); queues_->incoming[side_].pop_front(); lock.unlock();
        if (received) received(decode_distributed_direct_gpu_control(value, {2, 2048, 1}).kind);
        return value;
    }
    std::function<void(DistributedDirectGpuControlKind)> received;
private:
    std::shared_ptr<PeerControlQueues> queues_; std::size_t side_;
};
#endif
}
int main() {
#if defined(__linux__)
    const auto commission = [](Fixture& fixture, int delay = 0) {
        LinuxDirectGpuStreamCalls calls; calls.fd_owner = fixture.native_fd; calls.kernel_release = [] { return "synthetic-kernel"; };
        const auto build = fixture.build_id; calls.driver_build_identity = [build] { return *build; };
        const auto kernel = fixture.kernel; calls.ioctl_call = [kernel, delay](DirectStreamFd, unsigned long command, void* argument) mutable {
            if (command == SGU_IOC_EVENT && delay-- > 0) return LinuxDirectGpuStreamSyscallResult{-1, EAGAIN};
            return kernel->call(command, argument);
        };
        auto discovery = expected(); discovery.device_identity.clear(); discovery.device_instance_generation = 0;
        return std::make_unique<LinuxDirectGpuStreamCommissioning>(route(), discovery, fixture.gpu, fixture.control, LinuxDirectGpuStreamPolicy{true, 65536, 1048576, 64}, calls);
    };
    const auto config = [](const DirectDriverBinding& binding, bool receiver) {
        return DirectGpuCommissioningConfig{"synthetic-campaign", "synthetic-run", binding, {1, 77, 2, 3, 4, 5, 1, 6, 1, 5120}, 123, receiver, std::chrono::milliseconds{1000}, 16};
    };
    for (const int mode : {0, 1, 2}) {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        std::vector<DirectGpuCommissioningConfig> configs; std::vector<DirectGpuCommissioningPayload> payloads;
        int produced = 0, verified = 0;
        fixture.kernel->after_prepare = [&](const sgu_prepare& value) { fixture.gpu->terminal = {1, gpu_commit_word, 77, value.sequence, static_cast<std::uint32_t>(value.payload_bytes), 0, 1}; };
        for (std::size_t index = 0; index < 3; ++index) {
            auto step = config(observed.binding, mode == 1 || (mode == 2 && index == 1));
            step.run_id += "-" + std::to_string(index); step.descriptor.sequence = index + 1; step.descriptor.byte_length = std::array<std::uint64_t, 3>{1, 4097, 5120}[index];
            configs.push_back(step); auto view = fixture.pool.external(step.descriptor.byte_length);
            if (step.receiver) {
                fixture.incoming(DistributedDirectGpuControlKind::producer_released, step.descriptor);
                payloads.push_back({{}, {view}, {}, [&, index] { expect(fixture.kernel->retires == static_cast<int>(index), "RX verification occurs before next retirement"); ++verified; return true; }});
            } else {
                fixture.incoming(DistributedDirectGpuControlKind::receive_ready, step.descriptor); fixture.incoming(DistributedDirectGpuControlKind::received, step.descriptor); fixture.incoming(DistributedDirectGpuControlKind::consumed, step.descriptor);
                payloads.push_back({{view}, {}, [&, index] { expect(fixture.kernel->retires == static_cast<int>(index), "next producer cannot run before preceding ACK/credit retirement"); ++produced; }, {}});
            }
        }
        const auto report = harness->run_session(configs, payloads);
        expect(report.completed_count == 3 && report.transfers.size() == 3 && report.cleanup == DirectPoolReleaseStatus::released && report.error.empty(), "one bounded session completes repeated TX/RX and alternating directions");
        expect(fixture.kernel->imports == 1 && fixture.kernel->arms == 1 && fixture.kernel->releases == 1 && fixture.kernel->retires == 3 && produced + verified == 3, "sequential reuse has exactly one import/ARM/release and full per-step retirement");
        for (std::size_t index = 0; index < report.transfers.size(); ++index) expect(report.transfers[index].import_generation == report.import_generation && report.transfers[index].pool_binding_identity == report.pool_binding_identity && report.transfers[index].cleanup == DirectPoolReleaseStatus::pending && fixture.kernel->preparations[index].handle == index + 1 && fixture.kernel->preparations[index].sequence == index + 1, "session binds same physical import, increasing handles/sequences and truthful held-import stage");
    }
    for (const bool fail_receive : {false, true}) {
        std::array<std::unique_ptr<Fixture>, 2> peers{std::make_unique<Fixture>(false, true, false), std::make_unique<Fixture>(false, true, false)};
        peers[1]->kernel->query.fence.local_uuid[0] = 2; peers[1]->kernel->query.fence.peer_uuid[0] = 1;
        auto queues = std::make_shared<PeerControlQueues>();
        std::array<std::shared_ptr<PeerControl>, 2> controls{std::make_shared<PeerControl>(queues, 0), std::make_shared<PeerControl>(queues, 1)};
        std::array<std::unique_ptr<LinuxDirectGpuStreamCommissioning>, 2> harnesses;
        std::array<std::array<sgu_prepare, 2>, 2> prepared{};
        std::array<GpuCommitRecord, 2> tx_commits{};
        std::array<std::atomic<unsigned>, 2> delivered{};
        std::array<unsigned, 2> delivery_consumed{}, rx_delays{}, tx_first{}, producer_calls{}, verify_calls{};
        std::array<std::vector<DirectGpuCommissioningConfig>, 2> configs;
        std::array<std::vector<DirectGpuCommissioningPayload>, 2> payloads;
        for (std::size_t side = 0; side < 2; ++side) {
            auto& peer = *peers[side]; auto discovery = expected(); discovery.device_identity.clear(); discovery.device_instance_generation = 0;
            LinuxDirectGpuStreamCalls calls; calls.fd_owner = peer.native_fd; calls.kernel_release = [] { return "synthetic-kernel"; }; calls.driver_build_identity = [] { return std::optional<std::string>{"01020304"}; };
            calls.ioctl_call = [&, side](DirectStreamFd, unsigned long command, void* argument) {
                auto& local = *peers[side]; auto& kernel = *local.kernel;
                if (command == SGU_IOC_PREPARE) { const auto& request = *static_cast<sgu_prepare*>(argument); prepared[side][request.direction] = request; }
                if (command == SGU_IOC_SUBMIT || command == SGU_IOC_RX_REPOST || command == SGU_IOC_RETIRE) kernel.prepared = prepared[side][static_cast<sgu_action*>(argument)->direction];
                if (command == SGU_IOC_SUBMIT && static_cast<sgu_action*>(argument)->direction == SGU_DIRECTION_RX) tx_commits[side] = local.gpu->terminal;
                if (command == SGU_IOC_EVENT) {
                    const auto tx = std::find_if(kernel.events.begin(), kernel.events.end(), [](const auto& event) { return event.direction == SGU_DIRECTION_TX; });
                    if (tx != kernel.events.end()) { std::rotate(kernel.events.begin(), tx, tx + 1); ++tx_first[side]; }
                    else if (!kernel.events.empty() && kernel.events.front().direction == SGU_DIRECTION_RX) {
                        if (delivered[side].load(std::memory_order_acquire) <= delivery_consumed[side] || rx_delays[side]++ % 4 != 3) return LinuxDirectGpuStreamSyscallResult{-1, EAGAIN};
                        ++delivery_consumed[side];
                    }
                }
                const auto result = kernel.call(command, argument);
                if (command == SGU_IOC_SUBMIT && static_cast<sgu_action*>(argument)->direction == SGU_DIRECTION_TX) {
                    auto& remote = *peers[1 - side]; const auto bytes = kernel.prepared.payload_bytes;
                    std::memcpy(remote.pool.state->pointer(remote.pool.exported->binding().pool.rx.address, static_cast<std::size_t>(bytes)), local.pool.state->pointer(local.pool.exported->binding().pool.tx.address, static_cast<std::size_t>(bytes)), static_cast<std::size_t>(bytes));
                    remote.gpu->terminal = tx_commits[side]; delivered[1 - side].fetch_add(1, std::memory_order_release);
                }
                return result;
            };
            controls[side]->received = [&, side](DistributedDirectGpuControlKind kind) {
                if (kind == DistributedDirectGpuControlKind::received || kind == DistributedDirectGpuControlKind::consumed) expect(peers[side]->kernel->reposts > 0, "paired TX ACK consumption cannot block local RX acquire/consumer/repost");
            };
            harnesses[side] = std::make_unique<LinuxDirectGpuStreamCommissioning>(route(), discovery, peer.gpu, controls[side], LinuxDirectGpuStreamPolicy{true, 65536, 1048576, 64}, calls);
            const auto observed = harnesses[side]->observe(); harnesses[side]->bind_exact_observed(observed.binding);
            for (std::size_t index = 0; index < 4; ++index) {
                auto step = config(observed.binding, (index + side) % 2 != 0); step.run_id += ".s" + std::to_string(index); step.descriptor.sequence = index + 1;
                step.descriptor.byte_length = index % 2 ? 4097 : 5120; step.maximum_progress_polls = 10000; step.timeout = std::chrono::milliseconds{5000}; configs[side].push_back(step);
                auto external = peer.pool.external(step.descriptor.byte_length);
                payloads[side].push_back({{external}, {external}, [&, side, step] {
                    ++producer_calls[side]; auto& local = *peers[side]; auto* data = local.pool.state->pointer(local.pool.exported->binding().pool.tx.address, static_cast<std::size_t>(step.descriptor.byte_length));
                    for (std::uint64_t offset = 0; offset < step.descriptor.byte_length; ++offset) data[offset] = static_cast<std::byte>(direct_gpu_commissioning_pattern_byte(offset, step.pattern_seed, step.descriptor.sequence));
                }, [&, side, step, external] {
                    ++verify_calls[side]; auto& local = *peers[side];
                    if (fail_receive && side == 0) { local.gpu->gpu_idle = false; throw std::runtime_error("synthetic delayed duplex GPU/readback error"); }
                    const auto bytes = static_cast<std::size_t>(step.descriptor.byte_length); auto* destination = local.pool.state->pointer(external.address, bytes);
                    std::memcpy(destination, local.pool.state->pointer(local.pool.exported->binding().pool.rx.address, bytes), bytes);
                    return verify_direct_gpu_commissioning_pattern({destination, bytes}, step.pattern_seed, step.descriptor.sequence);
                }});
            }
        }
        auto first = std::async(std::launch::async, [&] { return harnesses[0]->run_duplex_session(configs[0], payloads[0]); });
        auto second = std::async(std::launch::async, [&] { return harnesses[1]->run_duplex_session(configs[1], payloads[1]); });
        const std::array reports{first.get(), second.get()};
        for (std::size_t side = 0; side < 2; ++side) {
            if (!fail_receive && !reports[side].error.empty()) std::cerr << "synthetic duplex side " << side << ": " << reports[side].error << '\n';
            expect(tx_first[side] > 0 && rx_delays[side] > 0 && peers[side]->kernel->imports == 1 && peers[side]->kernel->arms == 1, "two-peer bounded duplex exercises TX-first delayed RX with one stable import");
            if (!fail_receive) expect(reports[side].completed_count == 4 && reports[side].cleanup == DirectPoolReleaseStatus::released && reports[side].error.empty() && producer_calls[side] == 2 && verify_calls[side] == 2 && peers[side]->kernel->retires == 4, "two actual synthetic control peers complete two overlapping TX/RX pairs and full byte verification");
            else expect(reports[side].completed_count == 0 && reports[side].transfers.size() == 2 && producer_calls[side] == 1 && !reports[side].error.empty(), "duplex failure aborts later pairs on both exact control peers");
        }
        if (fail_receive) { expect(reports[0].cleanup == DirectPoolReleaseStatus::failed_retained && peers[0]->kernel->reposts == 0 && peers[0]->kernel->releases == 0, "unknown duplex GPU completion cannot credit/repost/free pending import"); peers[0]->gpu->gpu_idle = true; expect(harnesses[0]->close() == DirectPoolReleaseStatus::released, "late typed duplex quiescence cleans retained old import"); }
    }
    for (const int invalid : {0, 1, 2, 3}) {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        std::array configs{config(observed.binding, false), config(observed.binding, false)}; configs[1].run_id += "-2"; configs[1].descriptor.sequence = 2;
        auto view = fixture.pool.external(5120); std::array<DirectGpuCommissioningPayload, 2> payloads{{{{view}, {}, [] {}, {}}, {{view}, {}, [] {}, {}}}};
        if (invalid == 0) configs[1].descriptor.sequence = 1;
        if (invalid == 1) ++configs[1].expected.device_instance_generation;
        if (invalid == 2) configs[1].run_id = configs[0].run_id;
        if (invalid == 3) payloads[1].send_segments[0].bytes = 5119;
        expect(rejected([&] { static_cast<void>(harness->run_session(configs, payloads)); }) && fixture.kernel->imports == 0, "all schedule identities/coverage are checked before import or any producer exposure");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        auto step = config(observed.binding, false); step.descriptor.sequence = 2; auto view = fixture.pool.external(5120);
        expect(rejected([&] { static_cast<void>(harness->run_once(step, {{view}, {}, [] {}, {}})); }) && fixture.kernel->imports == 0, "fresh standalone import rejects non-one first sequence before controller exposure");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        std::array configs{config(observed.binding, false), config(observed.binding, false)}; configs[1].run_id += "-2"; configs[1].descriptor.sequence = 2;
        auto view = fixture.pool.external(5120); std::array<DirectGpuCommissioningPayload, 2> payloads{{{{view}, {}, [] {}, {}}, {{view}, {}, [] {}, {}}}};
        expect(rejected([&] { static_cast<void>(harness->run_duplex_session(configs, payloads)); }) && fixture.kernel->imports == 0, "duplex requires opposite directions before importing or exposing any producer");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        auto step = config(observed.binding, false); step.timeout = std::chrono::milliseconds{2}; auto view = fixture.pool.external(5120);
        std::chrono::steady_clock::time_point time{}; int produced = 0; fixture.kernel->after_import = [&] { time += std::chrono::milliseconds{3}; };
        const auto report = harness->run_once(step, {{view}, {}, [&] { ++produced; }, {}}, {[&] { return time; }, [] {}});
        expect(!report.executed && report.completed_count == 0 && !report.error.empty() && produced == 0 && fixture.kernel->arms == 0 && report.cleanup == DirectPoolReleaseStatus::released, "initial import delay is charged to first total deadline before ARM or GPU producer exposure");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        std::array configs{config(observed.binding, false), config(observed.binding, false), config(observed.binding, false)};
        for (std::size_t index = 0; index < configs.size(); ++index) { configs[index].run_id += std::to_string(index); configs[index].descriptor.sequence = index + 1; }
        fixture.incoming(DistributedDirectGpuControlKind::receive_ready, configs[0].descriptor); fixture.incoming(DistributedDirectGpuControlKind::received, configs[0].descriptor); fixture.incoming(DistributedDirectGpuControlKind::consumed, configs[0].descriptor);
        auto view = fixture.pool.external(5120); int later = 0;
        std::array<DirectGpuCommissioningPayload, 3> payloads{{{{view}, {}, [] {}, {}}, {{view}, {}, [&] { fixture.gpu->gpu_idle = false; throw std::runtime_error("delayed second producer failed"); }, {}}, {{view}, {}, [&] { ++later; }, {}}}};
        const auto report = harness->run_session(configs, payloads);
        expect(report.completed_count == 1 && report.transfers.size() == 2 && report.transfers.back().executed && later == 0 && fixture.kernel->imports == 1 && fixture.kernel->releases == 0 && report.cleanup == DirectPoolReleaseStatus::failed_retained, "second-transfer unknown callback completion retains same import and aborts whole remaining schedule");
        fixture.gpu->gpu_idle = true; expect(harness->close() == DirectPoolReleaseStatus::released, "late exact GPU/controller quiescence permits failed session cleanup only");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture, 3); const auto observed = harness->observe();
        expect(observed.native.query_status != DirectNativeQueryStatus::verified && !assess_direct_native_readiness(observed.backend, observed.native).ready, "commissioning observation cannot grant native production readiness");
        auto foreign = observed.binding; ++foreign.interface.connection_epoch;
        expect(rejected([&] { harness->bind_exact_observed(foreign); }), "commissioning rejects operator binding foreign to same FD");
        harness->bind_exact_observed(observed.binding);
        expect(harness->observe().native.query_status == DirectNativeQueryStatus::unavailable, "bound experimental facade remains unavailable to production");
        fixture.incoming(DistributedDirectGpuControlKind::receive_ready, fixture.descriptor());
        fixture.incoming(DistributedDirectGpuControlKind::received, fixture.descriptor()); fixture.incoming(DistributedDirectGpuControlKind::consumed, fixture.descriptor());
        auto source = fixture.pool.external(5120); int produced = 0, pauses = 0;
        const auto result = harness->run_once(config(observed.binding, false), {{source}, {}, [&] { ++produced; }, {}}, {[] { return std::chrono::steady_clock::now(); }, [&] { ++pauses; }});
        expect(result.executed && result.completed_count == 1 && produced == 1 && pauses == 3 && result.local_tx_complete && result.transport_ack && result.consumer_credit && result.cleanup == DirectPoolReleaseStatus::released,
               "synthetic commissioning sender waits delayed controller plus independent ACK/credit before retire/cleanup");
        expect(!result.full_payload_verified && !result.gpu_acquire_complete && !result.consumer_complete && !result.receive_reposted, "sender never self-attests receiver payload or consumer stages");
        expect(rejected([&] { static_cast<void>(harness->run_once(config(observed.binding, false), {{source}, {}, [] {}, {}})); }), "commissioning cannot silently repeat same epoch/sequence");
    }
    for (const bool matching : {true, false}) {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        fixture.incoming(DistributedDirectGpuControlKind::producer_released, fixture.descriptor()); fixture.gpu->terminal = fixture.commit();
        auto destination = fixture.pool.external(5120); bool verified_after_gpu = false;
        const auto result = harness->run_once(config(observed.binding, true), {{}, {destination}, {}, [&] { verified_after_gpu = fixture.gpu->acquired == 1 && fixture.gpu->unpacked == 1 && fixture.gpu->consumers == 1; return matching; }});
        expect(verified_after_gpu && result.gpu_acquire_complete && result.consumer_complete && result.full_payload_verified == matching && result.completed_count == static_cast<std::uint64_t>(matching), "full-byte verification follows actual acquire/unpack/GPU completion stage");
        expect(fixture.kernel->reposts == static_cast<int>(matching) && result.consumer_credit == matching && result.receive_reposted == matching, "payload mismatch cannot authorize consumed credit or RX repost");
        expect(result.cleanup == DirectPoolReleaseStatus::released && (matching || !result.error.empty()), "mismatch stops/drains exact old import rather than promotes production");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        auto source = fixture.pool.external(5120); fixture.gpu->gpu_idle = false;
        const auto result = harness->run_once(config(observed.binding, false), {{source}, {}, [] { throw std::runtime_error("synthetic asynchronous producer callback failure"); }, {}});
        expect(result.executed && result.cleanup == DirectPoolReleaseStatus::failed_retained && fixture.kernel->stops == 1 && fixture.kernel->releases == 0 && fixture.native_fd->closed == 0,
               "commissioning producer callback error retains controller/GPU/import owners without credit");
        fixture.gpu->gpu_idle = true; expect(harness->close() == DirectPoolReleaseStatus::released, "late typed quiescence permits exact experimental cleanup only");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture, 100); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        fixture.incoming(DistributedDirectGpuControlKind::producer_released, fixture.descriptor()); fixture.gpu->terminal = fixture.commit(); auto destination = fixture.pool.external(5120);
        auto bounded = config(observed.binding, true); bounded.maximum_progress_polls = 2;
        const auto result = harness->run_once(bounded, {{}, {destination}, {}, [] { return true; }});
        expect(result.executed && result.completed_count == 0 && !result.gpu_acquire_complete && !result.consumer_credit && fixture.kernel->reposts == 0 && result.cleanup == DirectPoolReleaseStatus::released, "exhausted progress cannot verify/acquire/repost unseen payload");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture, 1); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        fixture.incoming(DistributedDirectGpuControlKind::producer_released, fixture.descriptor()); fixture.gpu->terminal = fixture.commit(); auto destination = fixture.pool.external(5120);
        auto bounded = config(observed.binding, true); bounded.timeout = std::chrono::milliseconds{2}; std::chrono::steady_clock::time_point time{};
        const auto result = harness->run_once(bounded, {{}, {destination}, {}, [] { return true; }}, {[&] { time += std::chrono::microseconds{1}; return time; }, [&] { time += std::chrono::milliseconds{3}; }});
        expect(result.executed && result.completed_count == 0 && !result.gpu_acquire_complete && !result.consumer_credit && !result.error.empty() && fixture.kernel->reposts == 0, "total deadline after delayed controller readiness fails before acquire or credit");
    }
    {
        Fixture fixture(false, true, false); auto harness = commission(fixture); const auto observed = harness->observe(); harness->bind_exact_observed(observed.binding);
        fixture.incoming(DistributedDirectGpuControlKind::producer_released, fixture.descriptor()); fixture.gpu->terminal = fixture.commit(); auto destination = fixture.pool.external(5120);
        const auto result = harness->run_once(config(observed.binding, true), {{}, {destination}, {}, [&]() -> bool { fixture.gpu->gpu_idle = false; throw std::runtime_error("synthetic D2H event error after enqueue"); }});
        expect(result.executed && result.consumer_complete && !result.full_payload_verified && !result.consumer_credit && fixture.kernel->reposts == 0 && fixture.kernel->releases == 0 && result.cleanup == DirectPoolReleaseStatus::failed_retained,
               "verification D2H callback failure retains owner/import after prior consumer event without credit/repost");
        fixture.gpu->gpu_idle = true; expect(harness->close() == DirectPoolReleaseStatus::released, "late full-stream quiescence cleans failed verification without qualifying data");
    }
    {
        std::vector<std::byte> note(24); note[0] = std::byte{4}; note[4] = std::byte{8}; note[8] = std::byte{3};
        note[12] = std::byte{'G'}; note[13] = std::byte{'N'}; note[14] = std::byte{'U'};
        for (unsigned i = 0; i < 8; ++i) note[16 + i] = static_cast<std::byte>(i + 1);
        expect(parse_linux_direct_gpu_stream_module_build_id(note) == "0102030405060708", "bounded actual GNU note parser records loaded module build identity independently of ABI name");
        note[8] = std::byte{2}; expect(rejected([&] { static_cast<void>(parse_linux_direct_gpu_stream_module_build_id(note)); }), "wrong ELF note type is unavailable build evidence");
        expect(rejected([&] { static_cast<void>(parse_linux_direct_gpu_stream_module_build_id({})); }), "missing module build note cannot become guessed positive identity");
    }
    {
        Fixture fixture; fixture.build_id->reset();
        expect(fixture.provider->query().native.query_status == DirectNativeQueryStatus::unavailable && !fixture.provider->current_binding(), "compatible ABI without observed loaded driver build remains unavailable");
    }
    {
        Fixture fixture(false); const auto query = fixture.provider->query();
        expect(query.binding == expected() && query.native.query_status == DirectNativeQueryStatus::unavailable && !query.native.uncached_gpu_pool && !query.native.external_dma_visibility_qualified,
               "real ABI/query/code/uncached request cannot self-authorize native readiness without independent qualification");
        expect(!fixture.provider->current_binding() && fixture.kernel->imports == 0, "unqualified provider cannot expose productive binding or import");
    }
    {
        Fixture fixture(false, true); fixture.kernel->query.fence.device_instance[0] = 55;
        const auto observed = fixture.provider->query();
        expect(observed.binding.device_instance_generation == 55 && observed.native.query_status == DirectNativeQueryStatus::unavailable && !fixture.provider->current_binding(),
               "unpredictable open UUID is observed on the same owned FD without positive readiness");
        auto foreign = observed.binding; ++foreign.device_instance_generation;
        expect(rejected([&] { fixture.provider->bind_observed(foreign, fixture.qualification); }), "foreign observed binding cannot be adopted");
        fixture.qualification->approved_binding = observed.binding; fixture.provider->bind_observed(observed.binding, fixture.qualification);
        expect(!fixture.provider->current_binding() && fixture.provider->query().native.query_status == DirectNativeQueryStatus::verified,
               "explicit same-FD binding still requires separate synthetic qualification query before readiness");
        expect(rejected([&] { fixture.provider->bind_observed(observed.binding, fixture.qualification); }), "observed endpoint cannot be rebound or authority silently replaced");
        const auto pool_binding = fixture.pool.exported->binding();
        fixture.imported = fixture.provider->prepare_import(observed.binding, {"synthetic-physical-link", 77, pool_binding.pool_binding_identity, 0, 1}, pool_binding.pool);
        fixture.provider->activate_import(fixture.imported);
        expect(rejected([&] { fixture.provider->bind_observed(observed.binding, fixture.qualification); }), "imported epoch/peer identity cannot be changed through discovery adoption");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "same-FD discovery/bind/import lifecycle cleans exact resources");
    }
    {
        Fixture fixture(false, true); const auto observed = fixture.provider->query(); fixture.provider->bind_observed(observed.binding, {});
        expect(fixture.provider->query().native.query_status == DirectNativeQueryStatus::unavailable && !fixture.provider->current_binding(), "adopting observed identity alone cannot authorize unproven native import");
    }
    for (const auto state : {SGU_STATE_FAILED, SGU_STATE_DISCONNECTED, SGU_STATE_STOPPED, 99U}) {
        Fixture fixture(false, true); const auto observed = fixture.provider->query();
        fixture.qualification->approved_binding = observed.binding; fixture.provider->bind_observed(observed.binding, fixture.qualification);
        fixture.kernel->query.state = state;
        expect(fixture.provider->query().native.query_status != DirectNativeQueryStatus::verified && !fixture.provider->current_binding() && fixture.kernel->imports == 0,
               "observe/bind cannot qualify failed/disconnected/stopped/unknown raw state with unchanged fence");
        fixture.kernel->query.state = SGU_STATE_OPEN;
        expect(fixture.provider->query().native.query_status != DirectNativeQueryStatus::verified && !fixture.provider->current_binding() && fixture.provider->failed(),
               "restoring a raw OPEN observation cannot recover a failed provider or qualify the same FD");
    }
    {
        Fixture fixture;
        expect(fixture.provider->query().native.query_status == DirectNativeQueryStatus::verified, "qualified unimported OPEN phase remains available");
        const auto pool_binding = fixture.pool.exported->binding();
        fixture.imported = fixture.provider->prepare_import(expected(), {"synthetic-physical-link", 77, pool_binding.pool_binding_identity, 0, 1}, pool_binding.pool);
        expect(fixture.provider->query().native.query_status == DirectNativeQueryStatus::verified && fixture.provider->current_binding(), "qualified pre-arm IMPORTED phase remains available");
        fixture.provider->activate_import(fixture.imported);
        expect(fixture.provider->query().native.query_status == DirectNativeQueryStatus::verified && fixture.provider->current_binding(), "qualified ARMED phase remains available");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released && fixture.provider->query().native.query_status != DirectNativeQueryStatus::verified && !fixture.provider->current_binding(), "released provider cannot republish readiness");
    }
    for (const auto state : {SGU_STATE_FAILED, SGU_STATE_DISCONNECTED, SGU_STATE_STOPPED, SGU_STATE_OPEN, SGU_STATE_IMPORTED, 99U}) {
        Fixture fixture; fixture.open(); fixture.kernel->query.state = state;
        expect(fixture.provider->query().native.query_status != DirectNativeQueryStatus::verified && !fixture.provider->current_binding() && fixture.provider->failed(),
               "live armed import cannot regain readiness from failed/disconnected or incompatible phase query");
        fixture.gpu->gpu_idle = false;
        expect(fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->releases == 0 && fixture.native_fd->closed == 0,
               "known failed raw state retains old import until controller AND GPU quiescence");
        fixture.gpu->gpu_idle = true;
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "exact old-import cleanup remains possible after query-state failure");
    }
    {
        Fixture fixture; fixture.open(); auto send = fixture.operation(false); fixture.provider->prepare(send);
        const auto proof = fixture.provider->publish_gpu_commit(send, fixture.commit()); fixture.send_ready(); fixture.provider->submit(send);
        expect(proof.visibility == DirectDriverVisibility::producer_release_to_external_dma && fixture.kernel->prepared.payload_bytes == 5120 && fixture.kernel->events.front().frames[1].expected_bytes == 1024,
               "exact partial final frame is submitted with separate terminal frame after producer proof and peer-ready");
        expect(fixture.provider->poll_event()->event.completion.kind == CompletionKind::local_buffer_reusable, "controller completion supplies only local transport ownership event");
        fixture.incoming(DistributedDirectGpuControlKind::received, fixture.descriptor());
        fixture.incoming(DistributedDirectGpuControlKind::consumed, fixture.descriptor());
        expect(fixture.provider->poll_event()->event.completion.kind == CompletionKind::remote_transport_ack && fixture.provider->poll_event()->event.completion.kind == CompletionKind::remote_consumer_complete,
               "transport ACK and peer consumer credit are separately descriptor-fenced");
        fixture.provider->retire(send);
        expect(fixture.kernel->retires == 1 && fixture.gpu->retired == 1 && rejected([&] { fixture.provider->prepare(send); }), "terminal retire retains monotonic physical history and rejects replay");
        fixture.gpu->gpu_idle = false;
        expect(fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->stops == 1 && fixture.kernel->releases == 0 && fixture.native_fd->closed == 0,
               "controller STOP alone cannot release GPU import or exclusive FD");
        fixture.gpu->gpu_idle = true;
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released && fixture.kernel->releases == 1 && fixture.native_fd->closed == 1, "later explicit GPU quiescence permits real typed kernel release and FD close");
    }
    {
        Fixture fixture; fixture.open(); const auto receive = fixture.operation(true); fixture.provider->prepare(receive); fixture.gpu->terminal = fixture.commit();
        fixture.incoming(DistributedDirectGpuControlKind::producer_released, fixture.descriptor()); fixture.provider->submit(receive);
        expect(fixture.kernel->submits == 1 && fixture.provider->poll_event()->event.completion.kind == CompletionKind::receive_complete && fixture.gpu->acquired == 0,
               "RX controller completion does not fabricate GPU acquire or consumer proof");
        static_cast<void>(fixture.provider->read_gpu_commit(receive));
        fixture.gpu->consumer_complete = false;
        expect(rejected([&] { fixture.provider->publish_consumer_credit(receive, 5120); }) && fixture.kernel->reposts == 0, "credit before actual GPU consumer proof is rejected");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "rejected early credit still requires explicit controller/GPU cleanup");
    }
    {
        Fixture fixture; fixture.open(); const auto receive = fixture.operation(true); fixture.provider->prepare(receive); fixture.gpu->terminal = fixture.commit();
        fixture.incoming(DistributedDirectGpuControlKind::producer_released, fixture.descriptor()); fixture.provider->submit(receive); static_cast<void>(fixture.provider->poll_event());
        static_cast<void>(fixture.provider->read_gpu_commit(receive));
        auto destination = fixture.pool.external(5120); fixture.provider->consume_received_segments(fixture.descriptor(), std::array{destination});
        expect(fixture.provider->complete_receive_consumer(fixture.descriptor()).visibility == DirectDriverVisibility::consumer_release_complete, "descriptor-keyed boundary seam obtains only real normalized GPU completion proof");
        fixture.provider->publish_consumer_credit(receive, 5120); fixture.provider->repost_receive(receive);
        expect(fixture.kernel->reposts == 1 && fixture.provider->poll_event()->event.completion.kind == CompletionKind::receive_reposted, "physical RX repost follows completed consumer credit");
        fixture.provider->retire(receive); expect(fixture.kernel->retires == 1, "RX retirement follows actual repost and keeps imported standby pool ownership");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "successful receive drains standby DMA before import retirement");
    }
    for (int failure = 0; failure < 5; ++failure) {
        Fixture fixture; fixture.open(); const auto send = fixture.operation(false); fixture.provider->prepare(send); static_cast<void>(fixture.provider->publish_gpu_commit(send, fixture.commit()));
        fixture.send_ready(); fixture.provider->submit(send);
        fixture.kernel->event_mutation = [failure](sgu_event& event) {
            if (failure == 0) ++event.fence.connection_epoch;
            if (failure == 1) ++event.sequence;
            if (failure == 2) --event.frames[1].actual_bytes;
            if (failure == 3) ++event.frames[2].physical_slot;
            if (failure == 4) event.first_frame = 2;
        };
        expect(rejected([&] { static_cast<void>(fixture.provider->poll_event()); }) && fixture.provider->failed() && fixture.kernel->releases == 0, "stale/malformed physical completions fail sticky and retain ownership");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "malformed event cannot skip controller/GPU cleanup but matching typed stop can drain");
    }
    {
        Fixture fixture; fixture.open(); const auto receive = fixture.operation(true); fixture.provider->prepare(receive);
        auto stale = fixture.descriptor(); ++stale.request_id; fixture.incoming(DistributedDirectGpuControlKind::producer_released, stale);
        expect(rejected([&] { fixture.provider->submit(receive); }) && fixture.kernel->submits == 0 && fixture.gpu->acquired == 0, "foreign logical request rejected before RX submission/exposure");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "stale control failure retains until matching typed cleanup");
    }
    for (bool bad_ack : {false, true}) {
        Fixture fixture; static_cast<void>(fixture.provider->query()); fixture.kernel->import_error = !bad_ack; fixture.kernel->bad_import_ack = bad_ack;
        const auto binding = fixture.pool.exported->binding();
        expect(rejected([&] { fixture.imported = fixture.provider->prepare_import(expected(), {"synthetic-physical-link", 77, binding.pool_binding_identity, 0, 1}, binding.pool); }) && fixture.kernel->arms == 0,
               "partial import/copyout or changed full-extent acknowledgement is retained before activation");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released && fixture.kernel->stops == 1 && fixture.kernel->releases == 1, "exact exclusive QUERY recovers hidden import generation for controller/GPU cleanup only");
    }
    {
        Fixture fixture; fixture.open(); fixture.kernel->bad_stop_proof = true;
        expect(fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->releases == 0 && fixture.native_fd->closed == 0, "cancel request result cannot forge controller-quiescence release");
        fixture.kernel->bad_stop_proof = false; expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "only actual matching STOP acknowledgement permits cleanup retry");
    }
    {
        Fixture fixture; fixture.open(); fixture.native_fd->generation = 8;
        expect(!fixture.provider->current_binding() && fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->releases == 0, "same path/device FD replacement invalidates cached binding and retains imported buffers");
        fixture.native_fd->generation = 7; expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "restored exact FD identity allows old-import cleanup without restoring readiness");
    }
    {
        Fixture fixture; fixture.open(); fixture.qualification->valid = false;
        expect(!fixture.provider->current_binding() && fixture.provider->failed(), "qualification loss invalidates cached binding independently of unchanged device/ABI");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "lost qualification does not preclude exact old-import typed cleanup");
    }
    {
        Fixture fixture; fixture.open(); *fixture.build_id = "05060708";
        expect(!fixture.provider->current_binding() && fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->releases == 0,
               "same path/device with replaced module build invalidates binding and retains old import");
        *fixture.build_id = "01020304";
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released && fixture.provider->failed(), "restored exact old driver build allows cleanup only");
    }
    {
        Fixture fixture; fixture.open(); *fixture.build_id = "05060708";
        static_cast<void>(fixture.provider->query());
        expect(fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->releases == 0,
               "failed query cannot erase immutable loaded-driver baseline and authorize old-import cleanup on replacement");
        *fixture.build_id = "01020304";
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "restored immutable driver baseline permits cleanup after failed diagnostic query");
    }
    {
        Fixture fixture; DirectTransportDriverBackend backend(*fixture.provider, expected());
        const auto pool_binding = fixture.pool.exported->binding();
        const DirectTransportLease lease{"synthetic-physical-link", 77, pool_binding.pool_binding_identity, 0, 1};
        backend.import_link_pool(lease, pool_binding.pool); auto payload = pool_binding.pool.tx; payload.bytes = 5120;
        const auto handle = backend.prepare_send(fixture.descriptor(), payload); backend.publish_gpu_commit(handle, fixture.commit());
        fixture.send_ready(); backend.submit(handle);
        expect(backend.poll_direct_event()->completion.kind == CompletionKind::local_buffer_reusable && backend.active_operations() == 1,
               "existing native adapter retains concrete provider operation after only local completion");
        fixture.incoming(DistributedDirectGpuControlKind::received, fixture.descriptor()); fixture.incoming(DistributedDirectGpuControlKind::consumed, fixture.descriptor());
        static_cast<void>(backend.poll_direct_event()); static_cast<void>(backend.poll_direct_event());
        expect(backend.active_operations() == 0 && fixture.kernel->retires == 1 && fixture.gpu->retired == 1,
               "actual normalized native adapter retires concrete provider after all three ownership events");
        expect(backend.release_link_pool(lease) == DirectPoolReleaseStatus::released && fixture.kernel->stops == 1 && fixture.kernel->releases == 1,
               "concrete provider integrates typed controller/GPU release with existing native adapter");
    }
    {
        Fixture fixture; fixture.open(); const auto send = fixture.operation(false); fixture.provider->prepare(send);
        static_cast<void>(fixture.provider->publish_gpu_commit(send, fixture.commit()));
        expect(rejected([&] { fixture.provider->submit(send); }) && fixture.control->failed() && fixture.kernel->submits == 0,
               "bounded peer-ready timeout fails before any TX enqueue and retains producer source");
        fixture.gpu->gpu_idle = false;
        expect(fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->releases == 0, "deadline cannot stand in for late GPU completion or release");
        fixture.gpu->gpu_idle = true; expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "late explicit completion permits timeout cleanup only");
    }
    {
        Fixture fixture; fixture.open(); const auto send = fixture.operation(false); fixture.provider->prepare(send);
        static_cast<void>(fixture.provider->publish_gpu_commit(send, fixture.commit())); fixture.send_ready(); fixture.provider->submit(send);
        fixture.gpu->gpu_idle = false; fixture.provider->cancel(send);
        expect(fixture.provider->failed() && fixture.kernel->cancels == 1 && rejected([&] { static_cast<void>(fixture.provider->poll_event()); }) &&
               fixture.provider->close() == DirectPoolReleaseStatus::failed_retained && fixture.kernel->releases == 0,
               "cancel/late controller events cannot fabricate local reuse or GPU quiescence");
        fixture.gpu->gpu_idle = true; expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "cancelled work requires explicit controller STOP plus actual GPU drain");
    }
    {
        Fixture fixture; fixture.open(); const auto send = fixture.operation(false); fixture.provider->prepare(send);
        static_cast<void>(fixture.provider->publish_gpu_commit(send, fixture.commit())); fixture.send_ready(); fixture.provider->submit(send);
        fixture.kernel->events.push_back(fixture.kernel->events.front()); static_cast<void>(fixture.provider->poll_event());
        fixture.incoming(DistributedDirectGpuControlKind::received, fixture.descriptor()); fixture.incoming(DistributedDirectGpuControlKind::consumed, fixture.descriptor());
        static_cast<void>(fixture.provider->poll_event()); static_cast<void>(fixture.provider->poll_event()); fixture.provider->retire(send);
        expect(rejected([&] { static_cast<void>(fixture.provider->poll_event()); }) && fixture.provider->failed(), "duplicate controller completion after terminal retire is rejected before new ownership can advance");
        expect(fixture.provider->close() == DirectPoolReleaseStatus::released, "duplicate event failure retains import until explicit cleanup");
    }
    {
        Fixture fixture(false); auto actual_route = inspect_direct_stream_device("/dev/null");
        LinuxDirectTransportDriverIo stock(actual_route, expected(), fixture.gpu, fixture.control, {true, 65536, 1048576, 64});
        expect(stock.query().native.query_status == DirectNativeQueryStatus::unsupported && !stock.current_binding(), "actual stock character device ENOTTY is unavailable without guessed or positive ABI");
        expect(stock.close() == DirectPoolReleaseStatus::released, "unsupported route does not import or retain DMA resources");
    }
    if (!failures) std::cout << "synthetic experimental ABI/GPU/control ownership tests passed; no GPU/controller DMA executed\n";
#else
    std::cout << "experimental Linux direct GPU stream unavailable on this platform\n";
#endif
    return failures ? 1 : 0;
}
