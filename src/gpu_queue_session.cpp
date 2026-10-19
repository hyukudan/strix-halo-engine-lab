#include "strix/gpu_queue_session.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <thread>

namespace strix::transport {
namespace {
constexpr std::uint64_t magic = 0x5354524958513031ULL;
void append(std::vector<std::byte>& bytes, std::uint64_t value, unsigned count) {
    for (unsigned n = count; n; --n) bytes.push_back(static_cast<std::byte>((value >> ((n - 1) * 8)) & 255));
}
std::uint64_t take(std::span<const std::byte> bytes, std::size_t& cursor, unsigned count) {
    if (cursor > bytes.size() || count > bytes.size() - cursor) throw std::invalid_argument("truncated GPU queue peer frame");
    std::uint64_t value = 0; for (unsigned n = 0; n < count; ++n) value = (value << 8) | std::to_integer<unsigned>(bytes[cursor++]); return value;
}
template<class Range> bool nonzero(const Range& bytes) { return std::any_of(bytes.begin(), bytes.end(), [](auto b) { return b != 0; }); }
void validate(const GpuQueuePeerFrame& f) {
    if (!nonzero(f.campaign) || !nonzero(f.local) || !nonzero(f.peer) || !nonzero(f.device_instance) || f.local == f.peer ||
        !f.epoch || !f.import_generation || !f.pool_identity || !f.first_sequence || !f.payload_bytes ||
        (f.recovery_flags & ~(gpu_queue_reset_controller | gpu_queue_reset_network | gpu_queue_terminal_kernel)) ||
        ((f.recovery_flags & gpu_queue_reset_network) && !(f.recovery_flags & gpu_queue_reset_controller)) ||
        (f.kind >= gpu_queue_reset_done && !(f.recovery_flags & gpu_queue_reset_controller)) ||
        f.depth < 2 || f.depth > SGU_MAX_QUEUE_DEPTH || f.ring_frames < 32 || f.ring_frames > SGU_MAX_FRAMES ||
        f.payload_bytes > static_cast<std::uint64_t>((f.ring_frames - f.depth) / f.depth) * SGU_FRAME_BYTES ||
        f.kind < gpu_queue_hello || f.kind > gpu_queue_reset_failed || f.directions < 1 || f.directions > 3 ||
        f.completed_mask >= (1U << f.depth) || (f.kind != gpu_queue_consumed && f.completed_mask))
        throw std::invalid_argument("invalid bounded GPU queue peer contract");
}
}
std::vector<std::byte> encode_gpu_queue_peer_frame(const GpuQueuePeerFrame& f) {
    validate(f); std::vector<std::byte> out; out.reserve(160);
    append(out, magic, 8);
    for (auto b : f.campaign) append(out, b, 1);
    for (const auto& a : {f.local, f.peer, f.device_instance}) for (auto b : a) append(out, b, 1);
    for (auto v : {f.epoch, f.import_generation, f.pool_identity, f.window, f.first_sequence, f.payload_bytes}) append(out, v, 8);
    for (auto v : {f.kind, f.depth, f.ring_frames, f.directions, f.completed_mask}) append(out, v, 4);
    append(out, f.recovery_flags, 4); return out;
}
GpuQueuePeerFrame decode_gpu_queue_peer_frame(std::span<const std::byte> bytes) {
    if (bytes.size() != 160) throw std::invalid_argument("GPU queue peer frame has wrong exact size");
    std::size_t cursor = 0; if (take(bytes, cursor, 8) != magic) throw std::invalid_argument("foreign GPU queue protocol");
    GpuQueuePeerFrame f;
    for (auto& b : f.campaign) b = static_cast<std::uint8_t>(take(bytes, cursor, 1));
    for (auto* a : {&f.local, &f.peer, &f.device_instance}) for (auto& b : *a) b = static_cast<std::uint8_t>(take(bytes, cursor, 1));
    for (auto* v : {&f.epoch, &f.import_generation, &f.pool_identity, &f.window, &f.first_sequence, &f.payload_bytes}) *v = take(bytes, cursor, 8);
    for (auto* v : {&f.kind, &f.depth, &f.ring_frames, &f.directions, &f.completed_mask}) *v = static_cast<std::uint32_t>(take(bytes, cursor, 4));
    f.recovery_flags = static_cast<std::uint32_t>(take(bytes, cursor, 4));
    if (cursor != bytes.size()) throw std::invalid_argument("GPU queue peer reserved/trailing bytes are nonzero");
    validate(f); return f;
}
void require_gpu_queue_peer(const GpuQueuePeerFrame& local, const GpuQueuePeerFrame& peer, const GpuQueuePeerFrame* established) {
    validate(local); validate(peer);
    const auto opposite = ((local.directions & 1U) << 1) | ((local.directions & 2U) >> 1);
    if ((peer.recovery_flags ^ local.recovery_flags) & gpu_queue_terminal_kernel)
        throw std::runtime_error("GPU queue peers disagree on the terminal format (kernel-validated header vs application record)");
    if (peer.campaign != local.campaign || peer.local != local.peer || peer.peer != local.local ||
        peer.epoch != local.epoch || peer.window != local.window || peer.first_sequence != local.first_sequence ||
        peer.payload_bytes != local.payload_bytes || peer.kind != local.kind || peer.depth != local.depth ||
        peer.ring_frames != local.ring_frames || peer.directions != opposite || peer.recovery_flags != local.recovery_flags ||
        (local.kind == gpu_queue_consumed && peer.completed_mask != ((local.directions & 1U) ? (1U << local.depth) - 1 : 0)))
        throw std::runtime_error("GPU queue peer readiness/credit differs from exact planned window");
    if (established && (peer.device_instance != established->device_instance || peer.import_generation != established->import_generation ||
        peer.pool_identity != established->pool_identity || peer.epoch != established->epoch))
        throw std::runtime_error("GPU queue peer import/device was replaced during the window");
}
GpuQueueProgress GpuQueueProgress::native() {
    return {[] { return std::chrono::steady_clock::now(); }, [] { std::this_thread::sleep_for(std::chrono::milliseconds{1}); }};
}
GpuQueueSessionReport run_gpu_queue_session(LinuxGpuQueue& queue, const GpuQueuePool& pool, GpuQueueSessionPolicy policy,
    GpuQueueSessionIo control, GpuQueueSessionGpu gpu, GpuQueueProgress progress) {
    if (!policy.windows || policy.windows > 128 || policy.directions < 1 || policy.directions > 3 ||
        !policy.payload_bytes || policy.payload_bytes > queue.slot_capacity() || !policy.terminal_bytes || policy.terminal_bytes > SGU_FRAME_BYTES ||
        !nonzero(policy.campaign) || !control.send || !control.receive || !progress.now || !progress.pause ||
        ((policy.directions & 1U) && !gpu.produce) || ((policy.directions & 2U) && !gpu.consume) ||
        policy.window_timeout.count() <= 0 || policy.window_timeout > std::chrono::minutes{1} ||
        !policy.maximum_progress_calls || policy.maximum_progress_calls > 1000000 ||
        (queue.required_terminal_bytes() && policy.terminal_bytes != queue.required_terminal_bytes()) ||
        policy.payload_bytes * queue.depth() * policy.windows * (policy.directions == 3 ? 2U : 1U) > 128ULL * 1024 * 1024)
        throw std::invalid_argument("GPU queue session exceeds bounded explicit policy");
    if ((policy.recovery_flags & ~(gpu_queue_reset_controller | gpu_queue_reset_network)) ||
        ((policy.recovery_flags & gpu_queue_reset_network) && (!(policy.recovery_flags & gpu_queue_reset_controller) || !policy.independent_control)) ||
        (policy.recovery_flags && (!policy.recovery_every_windows || policy.recovery_every_windows >= policy.windows)))
        throw std::invalid_argument("controller reset requires explicit schedule and independent control for NETWORK");
    GpuQueueSessionReport report;
    try {
        GpuQueuePeerFrame local; local.campaign = policy.campaign; local.depth = queue.depth(); local.ring_frames = queue.observation().ring_frames;
        local.payload_bytes = policy.payload_bytes; local.directions = policy.directions; local.recovery_flags = policy.recovery_flags |
            (queue.terminal() == GpuQueueTerminal::kernel_header ? gpu_queue_terminal_kernel : 0U);
        std::copy_n(queue.binding().local_uuid, 16, local.local.begin()); std::copy_n(queue.binding().peer_uuid, 16, local.peer.begin());
        std::copy_n(queue.binding().device_instance, 16, local.device_instance.begin());
        std::optional<GpuQueuePeerFrame> peer;
        auto last = progress.now(); auto deadline = last;
        const auto check = [&] { const auto now = progress.now(); if (now < last || now >= deadline) throw std::runtime_error("GPU queue total window deadline expired or clock regressed"); last = now; };
        const auto reset_deadline = [&] {
            const auto now = progress.now(); if (now < last) throw std::runtime_error("GPU queue clock regressed between windows"); last = now;
            const auto duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(policy.window_timeout);
            if (last > std::chrono::steady_clock::time_point::max() - duration) throw std::overflow_error("GPU queue deadline overflow");
            deadline = last + duration;
        };
        const auto exchange = [&](std::uint32_t kind, std::uint32_t completed = 0) {
            check(); local.kind = kind; local.completed_mask = completed; const auto wire = encode_gpu_queue_peer_frame(local); std::vector<std::byte> answer;
            if (control.send_first) { control.send(wire); check(); answer = control.receive(); }
            else {
                answer = control.receive(); check();
                // Do not publish this phase's response to an invalid/foreign
                // peer frame, even if our own local completion already passed.
                require_gpu_queue_peer(local, decode_gpu_queue_peer_frame(answer), peer ? &*peer : nullptr);
                control.send(wire);
            }
            check(); auto remote = decode_gpu_queue_peer_frame(answer); require_gpu_queue_peer(local, remote, peer ? &*peer : nullptr);
            if (!peer) peer = remote;
        };
        std::uint64_t sequence = 1;
        for (std::uint32_t window = 0; window < policy.windows; ++window) {
            reset_deadline(); local.window = window; local.first_sequence = sequence;
            local.epoch = queue.binding().connection_epoch; local.import_generation = queue.binding().import_generation; local.pool_identity = queue.binding().pool_binding_identity;
            report.final_epoch = local.epoch;
            if (!peer) exchange(gpu_queue_hello);
            std::array<std::vector<GpuQueueOperation>, 2> operations;
            for (unsigned direction = 0; direction < 2; ++direction) if (policy.directions & (1U << direction)) {
                for (std::uint32_t slot = 0; slot < queue.depth(); ++slot) {
                    check(); const auto op = queue.prepare(direction, slot, sequence + slot, policy.payload_bytes, policy.terminal_bytes); check();
                    operations[direction].push_back(op);
                    if (!direction) { gpu.produce(op); check(); queue.gpu_produced(op); }
                }
            }
            const auto submit_all = [&](unsigned direction) {
                for (const auto& op : operations[direction]) {
                    bool submitted = false;
                    for (std::uint32_t call = 0; call < policy.maximum_progress_calls; ++call) {
                        check(); submitted = queue.submit(op); check(); if (submitted) break; progress.pause();
                    }
                    if (!submitted) throw std::runtime_error("GPU queue bounded nonblocking SUBMIT budget exhausted");
                }
            };
            submit_all(SGU_DIRECTION_RX); exchange(gpu_queue_ready); submit_all(SGU_DIRECTION_TX);
            report.peak_inflight_tx = std::max(report.peak_inflight_tx, static_cast<std::uint32_t>(operations[0].size()));
            report.peak_inflight_rx = std::max(report.peak_inflight_rx, static_cast<std::uint32_t>(operations[1].size()));
            std::array<std::uint32_t, 2> completed{}; const auto full = (1U << queue.depth()) - 1;
            const std::array<std::uint32_t, 2> wanted{(policy.directions & 1U) ? full : 0, (policy.directions & 2U) ? full : 0};
            for (std::uint32_t call = 0; completed != wanted && call < policy.maximum_progress_calls; ++call) {
                check(); const auto event = queue.poll(); check();
                if (!event) { progress.pause(); continue; }
                const auto& op = event->operation;
                if (op.direction > 1 || op.slot >= operations[op.direction].size() || operations[op.direction][op.slot] != op ||
                    (completed[op.direction] & (1U << op.slot))) throw std::runtime_error("GPU queue completion escaped the owned batch window");
                completed[op.direction] |= 1U << op.slot;
            }
            if (completed != wanted) throw std::runtime_error("GPU queue bounded EVENT progress exhausted");
            // Both pools have drained controller access before GPU readers or
            // new producers touch them. Do not wait on a whole DMA-BUF's implicit
            // fences while a not-yet-submitted operation needs that same GPU.
            for (const auto& op : operations[SGU_DIRECTION_RX]) {
                if (!gpu.consume(op)) throw std::runtime_error("GPU queue terminal/payload/consumer verification failed");
                check(); queue.gpu_consumed(op);
            }
            exchange(gpu_queue_consumed, wanted[SGU_DIRECTION_RX]);
            for (const auto& op : operations[SGU_DIRECTION_TX]) queue.peer_consumed(op);
            for (const auto& direction : operations) for (const auto& op : direction) { check(); queue.retire(op); check(); }
            report.completed_tx += operations[0].size(); report.completed_rx += operations[1].size(); ++report.completed_windows;
            sequence += queue.depth();
            if (policy.recovery_every_windows && (window + 1) % policy.recovery_every_windows == 0 && window + 1 < policy.windows) {
                const auto epoch = local.epoch;
                if (policy.recovery_flags & gpu_queue_reset_controller) {
                    if (epoch == UINT64_MAX || !queue.stop() || !pool.gpu_quiescent())
                        throw std::runtime_error("controller recovery cannot prove local DMA/GPU quiescence");
                    exchange(gpu_queue_recovery_ready);
                    std::array<LinuxGpuQueue*, 1> members{&queue};
                    const auto reset = LinuxGpuQueue::reset_controller_group(members,
                        policy.recovery_flags & gpu_queue_reset_network);
                    exchange(reset.success ? gpu_queue_reset_done : gpu_queue_reset_failed);
                    if (!reset.success) throw std::runtime_error("local or peer controller reset failed; no window replay");
                    ++report.controller_resets;
                } else {
                    exchange(gpu_queue_recovery_ready);
                }
                if (epoch == UINT64_MAX || !queue.recover()) throw std::runtime_error("GPU queue peer-agreed session recovery could not prove quiescence");
                check(); queue.import_pool(pool, epoch + 1, local.pool_identity); queue.arm(); check();
                ++report.recoveries; peer.reset(); sequence = 1;
            }
        }
    } catch (const std::exception& error) { report.error = error.what(); queue.invalidate(); }
    catch (...) { report.error = "unknown GPU queue callback failure"; queue.invalidate(); }
    report.cleanup_released = queue.release();
    if (!report.cleanup_released && report.error.empty()) report.error = "GPU queue controller/GPU ownership retained";
    return report;
}
} // namespace strix::transport
