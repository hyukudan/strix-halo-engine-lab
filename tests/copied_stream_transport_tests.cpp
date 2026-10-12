#include "strix/copied_stream_transport.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>

#if defined(__linux__) || defined(__APPLE__)
#include <cerrno>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {
using namespace strix;
using namespace strix::transport;
int failures = 0;
void expect(bool condition, const char* message) { if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; } }
template<class Function> bool rejected(Function&& function) {
    try { function(); } catch (const std::exception&) { return true; } return false;
}
DirectStreamDeviceIdentity device() {
#if defined(_WIN32)
    return {"C:/synthetic/copied-stream", 1, 2, 3, true};
#else
    return {"/synthetic/copied-stream", 1, 2, 3, true};
#endif
}
struct Pipe { std::deque<std::byte> inbound[2]; };
class Io : public DirectStreamControlIo {
public:
    Io(Pipe& pipe, unsigned endpoint) : pipe_(&pipe), endpoint_(endpoint) {}
    DirectStreamFd open(const DirectStreamDeviceIdentity&) override { return {static_cast<int>(endpoint_) + 9, 1}; }
    DirectStreamFdIdentity inspect(DirectStreamFd) override { return {device(), generation}; }
    DirectStreamIoResult read(DirectStreamFd, std::span<std::byte> output) override {
        ++reads;
        auto& input = pipe_->inbound[endpoint_];
        if (input.empty()) return {eof ? DirectStreamIoStatus::end_of_stream : DirectStreamIoStatus::would_block, 0};
        const auto count = std::min({output.size(), input.size(), chunk});
        for (std::size_t index = 0; index < count; ++index) { output[index] = input.front(); input.pop_front(); }
        return {DirectStreamIoStatus::transferred, count};
    }
    DirectStreamIoResult write(DirectStreamFd, std::span<const std::byte> input) override {
        ++writes;
        if (fail_after >= 0 && written.size() >= static_cast<std::size_t>(fail_after)) return {DirectStreamIoStatus::failed, 0};
        const auto count = std::min(input.size(), chunk);
        for (std::size_t index = 0; index < count; ++index) { pipe_->inbound[1U - endpoint_].push_back(input[index]); written.push_back(input[index]); }
        return {DirectStreamIoStatus::transferred, count};
    }
    DirectStreamWaitStatus wait(DirectStreamFd, bool, DirectStreamClock::time_point) override {
        time += std::chrono::milliseconds(20); return DirectStreamWaitStatus::ready;
    }
    DirectStreamClock::time_point now() const override { return time; }
    void close(DirectStreamFd) noexcept override { ++closes; }
    std::vector<std::byte> written;
    std::uint64_t generation = 1;
    std::size_t chunk = 7;
    int reads = 0, writes = 0, closes = 0, fail_after = -1;
    bool eof = false;
    DirectStreamClock::time_point time{};
protected:
    Pipe* pipe_;
    unsigned endpoint_;
};
DirectStreamControlPolicy fd_policy() { return {4096, 4096, std::chrono::milliseconds(20)}; }
CopiedStreamTransportPolicy policy(std::uint64_t polls = 4) { return {128, 4096, polls}; }
CopiedStreamConnection connection(std::uint32_t local = 0) { return {"synthetic-physical-cpu-link", 51, local, 1U - local, 0}; }
DistributedPlacementPlan placement(std::uint32_t local = 0) {
    DistributedPlacementPlan plan;
    plan.local_node = local; plan.plan_identity = 31; plan.executable = true; plan.has_remote_work = true;
    plan.steps = {{1, SerialOp::token_embedding, 0, local != 0, false},
                  {2, SerialOp::token_embedding, 1, local != 1, false}};
    plan.boundaries.push_back({91, 1, 2, 0, 1, 16, {}});
    return plan;
}
DistributedBoundaryTransfer transfer(std::uint64_t request = 11, std::uint64_t sequence = 1) {
    const auto plan = placement(); const auto& boundary = plan.boundaries[0];
    DistributedBoundaryTransfer value;
    value.flow = classify_distributed_boundary_flow(plan, boundary);
    value.descriptor = make_distributed_boundary_descriptor(plan, boundary, 9, request, 2, sequence, 7);
    value.link_plan.kind = PlanKind::single_link; value.link_plan.slices = {{0, 0, 16}};
    return value;
}
CopiedStreamOwnedBuffer buffer(AllocationGenerationSource& generations) {
    auto allocation = std::make_shared<CopiedStreamHostBuffer>(16, generations);
    return {allocation, allocation->view()};
}
CopiedStreamPreparedSource synchronous_source(const DistributedBoundaryTransfer& message, CopiedStreamOwnedBuffer source) {
    auto proof = make_copied_stream_synchronous_proof({message.descriptor, source.view, CopiedStreamCompletionScope::producer_released});
    return {std::move(source), std::move(proof)};
}
std::unique_ptr<CopiedStreamCompletionProof> synchronous_consumer(const DistributedBoundaryTransfer& message, TransportBufferView view) {
    return make_copied_stream_synchronous_proof({message.descriptor, view, CopiedStreamCompletionScope::consumer_complete});
}
struct ProofState { CopiedStreamCompletionStatus status = CopiedStreamCompletionStatus::pending; int polls = 0; bool throws = false; };
class DelayedProof final : public CopiedStreamCompletionProof {
public:
    DelayedProof(CopiedStreamCompletionBinding value, std::shared_ptr<ProofState> state)
        : binding_(std::move(value)), state_(std::move(state)) {}
    const CopiedStreamCompletionBinding& binding() const noexcept override { return binding_; }
    CopiedStreamCompletionStatus poll() override {
        ++state_->polls; if (state_->throws) throw std::runtime_error("injected completion error"); return state_->status;
    }
private:
    CopiedStreamCompletionBinding binding_;
    std::shared_ptr<ProofState> state_;
};
void fill(const CopiedStreamOwnedBuffer& value, std::byte seed = std::byte{17}) {
    std::fill(value.allocation->writable_bytes().begin(), value.allocation->writable_bytes().end(), seed);
}
void finish_receive(CopiedStreamTransportBackend& receiver, TransferHandle handle,
                    const DistributedBoundaryTransfer& message, TransportBufferView view) {
    receiver.finish_consumer(handle, synchronous_consumer(message, view));
    const auto reusable = receiver.poll();
    expect(reusable && reusable->kind == CompletionKind::local_buffer_reusable, "completed RX proof retires copied host buffer");
}
struct Pair {
    Pipe pipe; Io left_io{pipe, 0}, right_io{pipe, 1};
    DirectStreamControlByteStream left = DirectStreamControlByteStream::open_with_io(left_io, device(), fd_policy());
    DirectStreamControlByteStream right = DirectStreamControlByteStream::open_with_io(right_io, device(), fd_policy());
    CopiedStreamConnectionState left_state{connection()}, right_state{connection(1)};
    CopiedStreamTransportBackend sender{left, left_state, policy()}, receiver{right, right_state, policy()};
    AllocationGenerationSource generations;
};

#if defined(__linux__) || defined(__APPLE__)
class SocketIo final : public Io {
public:
    SocketIo(Pipe& unused, unsigned endpoint, int fd) : Io(unused, endpoint), socket_(fd) {}
    DirectStreamFd open(const DirectStreamDeviceIdentity&) override { return {socket_, 1}; }
    DirectStreamIoResult read(DirectStreamFd fd, std::span<std::byte> output) override {
        return result(::recv(fd.descriptor, output.data(), std::min(output.size(), chunk), MSG_DONTWAIT));
    }
    DirectStreamIoResult write(DirectStreamFd fd, std::span<const std::byte> input) override {
        int flags = MSG_DONTWAIT;
#if defined(MSG_NOSIGNAL)
        flags |= MSG_NOSIGNAL;
#endif
        return result(::send(fd.descriptor, input.data(), std::min(input.size(), chunk), flags));
    }
    DirectStreamWaitStatus wait(DirectStreamFd fd, bool writing, DirectStreamClock::time_point deadline) override {
        const auto millis = std::chrono::ceil<std::chrono::milliseconds>(deadline - now()).count();
        if (millis <= 0) return DirectStreamWaitStatus::timed_out;
        struct pollfd state{fd.descriptor, static_cast<short>(writing ? POLLOUT : POLLIN), 0};
        const auto status = ::poll(&state, 1, static_cast<int>(millis));
        if (status == 0) return DirectStreamWaitStatus::timed_out;
        if (status < 0) return errno == EINTR ? DirectStreamWaitStatus::interrupted : DirectStreamWaitStatus::failed;
        if ((state.revents & (POLLERR | POLLNVAL)) != 0) return DirectStreamWaitStatus::failed;
        return (state.revents & state.events) != 0 ? DirectStreamWaitStatus::ready : DirectStreamWaitStatus::end_of_stream;
    }
    DirectStreamClock::time_point now() const override { return DirectStreamClock::now(); }
    void close(DirectStreamFd fd) noexcept override { static_cast<void>(::close(fd.descriptor)); }
private:
    static DirectStreamIoResult result(ssize_t count) {
        if (count > 0) return {DirectStreamIoStatus::transferred, static_cast<std::uint64_t>(count)};
        if (count == 0) return {DirectStreamIoStatus::end_of_stream, 0};
        if (errno == EINTR) return {DirectStreamIoStatus::interrupted, 0};
        if (errno == EAGAIN || errno == EWOULDBLOCK) return {DirectStreamIoStatus::would_block, 0};
        return {DirectStreamIoStatus::failed, 0};
    }
    int socket_;
};
#endif
class Local final : public DistributedNativeLocalExecutor {
public:
    void execute_local_step(const DistributedStepPlacement&) override { ++calls; }
    int calls = 0;
};
class Control final : public DistributedNativeControlPlane {
public:
    void execute_remote_step(const DistributedStepPlacement&) override { ++steps; }
    void publish_boundary_transfer(const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override { ++published; }
    DistributedBoundaryTransfer await_boundary_transfer(const DistributedExecutionBoundary&) override { throw std::logic_error("copied receiver uses verified queue"); }
    int steps = 0, published = 0;
};
class Source final : public CopiedStreamBoundarySourceProvider {
public:
    explicit Source(CopiedStreamOwnedBuffer value) : buffer_(std::move(value)) {}
    CopiedStreamPreparedSource stage_outbound_owned(const DistributedExecutionBoundary&, const DistributedBoundaryTransfer& message) override {
        ++calls; return synchronous_source(message, buffer_);
    }
    int calls = 0;
private:
    CopiedStreamOwnedBuffer buffer_;
};
class CompleteInbound final : public DistributedInboundCompletion {
public:
    explicit CompleteInbound(std::shared_ptr<ProofState> state = {}) : state_(std::move(state)) {}
    DistributedInboundCompletionStatus poll() override {
        if (!state_) return DistributedInboundCompletionStatus::complete;
        ++state_->polls;
        if (state_->throws) throw std::runtime_error("injected inbound completion error");
        if (state_->status == CopiedStreamCompletionStatus::complete) return DistributedInboundCompletionStatus::complete;
        if (state_->status == CopiedStreamCompletionStatus::pending) return DistributedInboundCompletionStatus::pending;
        return DistributedInboundCompletionStatus::failed;
    }
private:
    std::shared_ptr<ProofState> state_;
};
class Consumer final : public DistributedBoundaryStagingProvider {
public:
    TransportBufferView stage_outbound(const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override { throw std::logic_error("owned copied source required"); }
    TransportBufferView stage_inbound(const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override { throw std::logic_error("copied transport owns CPU staging"); }
    void consume_inbound(const DistributedExecutionBoundary&, const TransportBufferView& staged) override {
        ++calls; last_view = staged; actual.resize(static_cast<std::size_t>(staged.bytes));
        std::memcpy(actual.data(), reinterpret_cast<const void*>(staged.address), actual.size());
        if (consume_throws) throw std::runtime_error("injected consume error");
    }
    std::unique_ptr<DistributedInboundCompletion> finish_inbound(const DistributedExecutionBoundary&, const DistributedBoundaryTransfer&) override {
        if (finish_throws) throw std::runtime_error("injected finish error");
        return std::make_unique<CompleteInbound>(proof_state);
    }
    int calls = 0; std::vector<std::byte> actual;
    TransportBufferView last_view;
    bool consume_throws = false, finish_throws = false;
    std::shared_ptr<ProofState> proof_state;
};
}  // namespace

int main() {
    {
        Pair pair; auto source = buffer(pair.generations), destination = buffer(pair.generations); fill(source); fill(destination, std::byte{99});
        for (const std::uint64_t request : {11ULL, 12ULL}) {
            const auto message = transfer(request);
            const auto tx = pair.sender.prepare_send(message, synchronous_source(message, source));
            const auto rx = pair.receiver.prepare_receive(message, destination);
            pair.sender.submit(tx); expect(pair.sender.poll()->kind == CompletionKind::local_buffer_reusable, "owned CPU snapshot proves local source reusable");
            pair.receiver.submit(rx); expect(pair.receiver.poll()->kind == CompletionKind::receive_complete, "exact message copied before RX ready event");
            const auto view = pair.receiver.receive_buffer(rx);
            expect(std::equal(source.allocation->bytes().begin(), source.allocation->bytes().end(), destination.allocation->bytes().begin()),
                   "logical epoch 9 payload is copied exactly over independent physical epoch 51");
            expect(pair.sender.poll()->kind == CompletionKind::remote_transport_ack, "validated receive ACK differs from consumer completion");
            auto state = std::make_shared<ProofState>();
            pair.receiver.finish_consumer(rx, std::make_unique<DelayedProof>(CopiedStreamCompletionBinding{
                message.descriptor, view, CopiedStreamCompletionScope::consumer_complete}, state));
            expect(!pair.receiver.poll() && pair.receiver.retains_buffer() && pair.pipe.inbound[0].empty(),
                   "CPU return and delayed consumer cannot publish consumer ACK or reuse staging");
            state->status = CopiedStreamCompletionStatus::complete;
            expect(pair.receiver.poll()->kind == CompletionKind::local_buffer_reusable &&
                   pair.sender.poll()->kind == CompletionKind::remote_consumer_complete && !pair.sender.retains_buffer(),
                   "exact completed-consumer proof authorizes ACK and both C1 retirements");
        }
        expect(!pair.sender.capabilities().external_dma && !pair.sender.capabilities().supports_gpu_transport_pool &&
               pair.sender.capabilities().buffer_mode == TransportBufferMode::copy_stream,
               "host copied tier cannot advertise native GPU DMA capabilities");
    }
    {
        Pair pair; auto source = buffer(pair.generations);
        expect(rejected([&] { static_cast<void>(pair.sender.prepare_send(transfer().descriptor, source.view)); }),
               "generic raw pointer preparation cannot bypass ownership/completion contract");
        expect(rejected([&] { CopiedStreamTransportBackend duplicate(pair.left, pair.left_state, policy()); }),
               "C1 ownership is exclusive per physical connection, across different adapters");
        const auto message = transfer();
        const auto tx = pair.sender.prepare_send(message, synchronous_source(message, source));
        expect(rejected([&] { static_cast<void>(pair.sender.prepare_send(message, synchronous_source(message, source))); }),
               "one active operation bounds metadata and owned staging allocations");
        pair.sender.cancel(tx);
        expect(pair.sender.drain_retained() == CopiedStreamCleanupStatus::released && pair.left_state.failed(),
               "local cancellation may clean proven host resources but cannot recover its physical epoch");
    }
    for (int failure = 0; failure < 3; ++failure) {
        Pair pair; auto source = buffer(pair.generations); const auto message = transfer();
        auto state = std::make_shared<ProofState>(); state->throws = failure == 1;
        if (failure == 2) state->status = CopiedStreamCompletionStatus::failed_retained;
        std::weak_ptr<HostUploadSource> retained = source.allocation;
        const auto tx = pair.sender.prepare_send(message, {source, std::make_unique<DelayedProof>(
            CopiedStreamCompletionBinding{message.descriptor, source.view, CopiedStreamCompletionScope::producer_released}, state)});
        source.allocation.reset();
        expect(rejected([&] { pair.sender.submit(tx); }) && pair.sender.failed() && pair.sender.retains_buffer() &&
               !retained.expired() && pair.right_io.reads == 0 && pair.pipe.inbound[1].empty(),
               "producer pending/callback failure/failed proof retains source and exposes no payload");
        state->throws = false; state->status = CopiedStreamCompletionStatus::complete;
        expect(pair.sender.drain_retained() == CopiedStreamCleanupStatus::released && retained.expired(),
               "later producer quiescence permits cleanup without ACK or epoch recovery");
    }
    for (int field = 0; field < 9; ++field) {
        Pair pair; auto source = buffer(pair.generations), destination = buffer(pair.generations); fill(destination, std::byte{99});
        auto sent = transfer(); const auto expected = sent;
        if (field == 0) ++sent.descriptor.connection_epoch;
        if (field == 1) ++sent.descriptor.model_plan_id;
        if (field == 2) ++sent.descriptor.request_id;
        if (field == 3) ++sent.descriptor.generation;
        if (field == 4) ++sent.descriptor.boundary_id;
        if (field == 5) ++sent.descriptor.sequence;
        if (field == 6) ++sent.descriptor.shape_fingerprint;
        if (field == 7) ++sent.descriptor.dtype_id;
        if (field == 8) ++sent.flow.flow_id;
        const auto tx = pair.sender.prepare_send(sent, synchronous_source(sent, source));
        const auto rx = pair.receiver.prepare_receive(expected, destination); pair.sender.submit(tx);
        expect(rejected([&] { pair.receiver.submit(rx); }) && pair.receiver.failed() &&
               destination.allocation->bytes()[0] == std::byte{99} && pair.pipe.inbound[0].empty(),
               "full queued descriptor/flow mismatch cannot copy destination or publish any ACK");
        pair.sender.cancel(tx); static_cast<void>(pair.sender.drain_retained()); static_cast<void>(pair.receiver.drain_retained());
    }
    for (const std::size_t offset : {8U, 12U, 14U, 16U, 20U, 24U, 32U, 40U, 44U, 48U, 52U}) {
        Pair pair; auto source = buffer(pair.generations), destination = buffer(pair.generations); fill(destination, std::byte{99});
        const auto message = transfer(); const auto tx = pair.sender.prepare_send(message, synchronous_source(message, source));
        const auto rx = pair.receiver.prepare_receive(message, destination); pair.sender.submit(tx);
        pair.pipe.inbound[1][offset] ^= std::byte{1};
        expect(rejected([&] { pair.receiver.submit(rx); }) && destination.allocation->bytes()[0] == std::byte{99} &&
               pair.pipe.inbound[0].empty(), "version/kind/reserved/node/physical epoch/sequence/length/route tamper fails before destination acceptance");
        pair.sender.cancel(tx); static_cast<void>(pair.sender.drain_retained()); static_cast<void>(pair.receiver.drain_retained());
    }
    {
        Pair pair; auto source = buffer(pair.generations), destination = buffer(pair.generations); const auto message = transfer();
        const auto tx = pair.sender.prepare_send(message, synchronous_source(message, source));
        const auto rx = pair.receiver.prepare_receive(message, destination); pair.sender.submit(tx); pair.receiver.submit(rx);
        static_cast<void>(pair.sender.poll()); static_cast<void>(pair.receiver.poll());
        const auto view = pair.receiver.receive_buffer(rx);
        auto foreign = message.descriptor; ++foreign.request_id;
        expect(rejected([&] { pair.receiver.finish_consumer(rx, make_copied_stream_synchronous_proof(
            {foreign, view, CopiedStreamCompletionScope::consumer_complete})); }) && pair.receiver.failed(),
            "foreign consumer proof cannot authorize reuse or consumer ACK");
        pair.receiver.attach_cleanup_proof(rx, synchronous_consumer(message, view));
        expect(pair.receiver.drain_retained() == CopiedStreamCleanupStatus::released,
               "cleanup proof retires exposed staging while preserving failed request authority");
        pair.sender.cancel(tx); static_cast<void>(pair.sender.drain_retained());
    }
    {
        Pair pair; auto source = buffer(pair.generations), destination = buffer(pair.generations); const auto message = transfer();
        const auto tx = pair.sender.prepare_send(message, synchronous_source(message, source));
        const auto rx = pair.receiver.prepare_receive(message, destination); pair.sender.submit(tx); pair.receiver.submit(rx);
        static_cast<void>(pair.sender.poll()); static_cast<void>(pair.receiver.poll());
        const auto view = pair.receiver.receive_buffer(rx); std::weak_ptr<HostUploadSource> retained = destination.allocation;
        destination.allocation.reset(); pair.receiver.cancel(rx);
        expect(pair.receiver.drain_retained() == CopiedStreamCleanupStatus::pending && !retained.expired(),
               "cancel/consumer callback error without proof cannot free exposed inbound memory");
        pair.receiver.attach_cleanup_proof(rx, synchronous_consumer(message, view));
        expect(pair.receiver.drain_retained() == CopiedStreamCleanupStatus::released && retained.expired(),
               "later exact proof cleans unknown consumer access without publishing credit");
        pair.sender.cancel(tx); static_cast<void>(pair.sender.drain_retained());
    }
    for (int fault = 0; fault < 3; ++fault) {
        Pair pair; auto source = buffer(pair.generations), destination = buffer(pair.generations); const auto message = transfer(); fill(destination, std::byte{99});
        const auto tx = pair.sender.prepare_send(message, synchronous_source(message, source));
        const auto rx = pair.receiver.prepare_receive(message, destination);
        if (fault == 0) {
            pair.left_io.fail_after = 14;
            expect(rejected([&] { pair.sender.submit(tx); }) && pair.sender.retains_buffer(), "partial FD send failure retains operation ownership");
            pair.right_io.eof = true;
        } else if (fault == 1) {
            pair.sender.submit(tx); pair.pipe.inbound[1].resize(2); pair.right_io.eof = true;
        }
        expect(rejected([&] { pair.receiver.submit(rx); }) && pair.receiver.failed() &&
               destination.allocation->bytes()[0] == std::byte{99}, "partial frame EOF/deadline preserves destination and fails correspondence");
        if (!pair.sender.failed()) pair.sender.cancel(tx);
        static_cast<void>(pair.sender.drain_retained()); static_cast<void>(pair.receiver.drain_retained());
    }

    {
        Pipe pipe; Io left_io(pipe, 0), right_io(pipe, 1);
        auto left = DirectStreamControlByteStream::open_with_io(left_io, device(), fd_policy());
        auto right = DirectStreamControlByteStream::open_with_io(right_io, device(), fd_policy());
        CopiedStreamConnectionState left_state(connection()), right_state(connection(1));
        AllocationGenerationSource generations; auto source = buffer(generations), destination = buffer(generations);
        std::deque<std::byte> old_payload;
        for (const std::uint64_t request : {11ULL, 12ULL}) {
            CopiedStreamTransportBackend sender(left, left_state, policy()), receiver(right, right_state, policy());
            const auto message = transfer(request);
            const auto tx = sender.prepare_send(message, synchronous_source(message, source));
            const auto rx = receiver.prepare_receive(message, destination); sender.submit(tx);
            if (request == 11) old_payload = pipe.inbound[1];
            receiver.submit(rx); static_cast<void>(sender.poll()); static_cast<void>(receiver.poll());
            static_cast<void>(sender.poll()); finish_receive(receiver, rx, message, receiver.receive_buffer(rx));
            expect(sender.poll()->kind == CompletionKind::remote_consumer_complete,
                   "healthy backend reopen retains physical sequence while independent logical request starts at one");
        }
        CopiedStreamTransportBackend receiver(right, right_state, policy());
        fill(destination, std::byte{99});
        const auto rx = receiver.prepare_receive(transfer(), destination); pipe.inbound[1] = old_payload;
        expect(rejected([&] { receiver.submit(rx); }) && right_state.failed() &&
               destination.allocation->bytes()[0] == std::byte{99} && pipe.inbound[0].empty(),
               "same-epoch reopen cannot accept a replay with otherwise exact logical descriptor");
        expect(receiver.drain_retained() == CopiedStreamCleanupStatus::released,
               "rejected replay retains no exposed consumer and can clean owned staging");
    }
    {
        Pair pair; auto source = buffer(pair.generations), destination = buffer(pair.generations);
        const auto first = transfer(); const auto tx = pair.sender.prepare_send(first, synchronous_source(first, source));
        const auto rx = pair.receiver.prepare_receive(first, destination); pair.sender.submit(tx); pair.receiver.submit(rx);
        auto old_ack = pair.pipe.inbound[0];
        static_cast<void>(pair.sender.poll()); static_cast<void>(pair.receiver.poll()); static_cast<void>(pair.sender.poll());
        finish_receive(pair.receiver, rx, first, pair.receiver.receive_buffer(rx)); static_cast<void>(pair.sender.poll());
        const auto next = transfer(12); const auto next_tx = pair.sender.prepare_send(next, synchronous_source(next, source));
        pair.sender.submit(next_tx); static_cast<void>(pair.sender.poll()); pair.pipe.inbound[0] = std::move(old_ack);
        expect(rejected([&] { static_cast<void>(pair.sender.poll()); }) && pair.sender.failed() && pair.sender.retains_buffer(),
               "duplicate prior-request ACK cannot retire a later physical/logical operation");
        expect(pair.sender.drain_retained() == CopiedStreamCleanupStatus::released,
               "completed producer permits failed ACK cleanup without epoch reuse");
    }
    for (int fault = 0; fault < 5; ++fault) {
        Pair pair; const auto plan = placement(1), sender_plan = placement();
        const DistributedRuntimeRequest request{9, 11, 2, 7};
        Local local; Control control; Consumer consumer; auto source_buffer = buffer(pair.generations); Source source(source_buffer);
        consumer.consume_throws = fault == 0; consumer.finish_throws = fault == 1;
        if (fault >= 2) {
            consumer.proof_state = std::make_shared<ProofState>();
            consumer.proof_state->throws = fault == 3;
            if (fault == 4) consumer.proof_state->status = CopiedStreamCompletionStatus::failed_retained;
        }
        DistributedControlReceiver queue(sender_plan, request, 1, local); const auto message = transfer();
        queue.receive_boundary_transfer(message);
        CopiedStreamDistributedBackend receiving(pair.receiver, plan, request, control, queue, local, source, consumer, pair.generations, 2);
        const auto tx = pair.sender.prepare_send(message, synchronous_source(message, source_buffer)); pair.sender.submit(tx);
        expect(rejected([&] { static_cast<void>(receiving.receive_boundary(plan.boundaries[0])); }) &&
               receiving.failed() && pair.receiver.failed() && pair.receiver.retains_buffer() && consumer.calls == 1,
               "consume/finish/proof failure and bounded delayed completion retain concrete adapter staging");
        static_cast<void>(pair.sender.poll());
        expect(pair.sender.poll()->kind == CompletionKind::remote_transport_ack && pair.pipe.inbound[0].empty(),
               "consumer callback return/failure never emits completion ACK");
        const auto retained = *pair.receiver.retained_handle();
        if (fault <= 1) {
            expect(pair.receiver.drain_retained() == CopiedStreamCleanupStatus::pending,
                   "unknown consumer access after callback exception requires later exact proof");
            pair.receiver.attach_cleanup_proof(retained, synchronous_consumer(message, consumer.last_view));
        } else {
            consumer.proof_state->throws = false; consumer.proof_state->status = CopiedStreamCompletionStatus::complete;
        }
        expect(pair.receiver.drain_retained() == CopiedStreamCleanupStatus::released && pair.pipe.inbound[0].empty(),
               "later explicit consumer quiescence cleans adapter buffer without completion ACK or recovery");
        pair.sender.cancel(tx); static_cast<void>(pair.sender.drain_retained());
    }
    {
        Pair pair; auto plan = placement(1), sender_plan = placement();
        plan.boundaries[0].payload_bytes = 129; sender_plan.boundaries[0].payload_bytes = 129;
        const DistributedRuntimeRequest request{9, 11, 2, 7};
        Local local; Control control; Consumer consumer; Source source(buffer(pair.generations));
        AllocationGenerationSource receiver_generations;
        DistributedControlReceiver queue(sender_plan, request, 1, local);
        auto message = transfer();
        message.descriptor = make_distributed_boundary_descriptor(sender_plan, sender_plan.boundaries[0], 9, 11, 2, 1, 7);
        message.flow = classify_distributed_boundary_flow(sender_plan, sender_plan.boundaries[0]);
        message.link_plan.slices[0].byte_length = 129; queue.receive_boundary_transfer(message);
        CopiedStreamDistributedBackend receiving(pair.receiver, plan, request, control, queue, local, source, consumer, receiver_generations);
        expect(rejected([&] { static_cast<void>(receiving.receive_boundary(plan.boundaries[0])); }) &&
               receiving.failed() && !pair.receiver.retains_buffer() && pair.right_io.reads == 0 &&
               consumer.calls == 0 && receiver_generations.next() == 1,
               "oversized trusted boundary is rejected before CPU allocation generation, payload IO or consume");
    }

#if defined(__linux__) || defined(__APPLE__)
    {
        // Real socketpair read/write/poll, synthetic local route declarations.
        int fds[2]{}; if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) throw std::runtime_error("socketpair failed");
        Pipe unused; SocketIo left_io(unused, 0, fds[0]), right_io(unused, 1, fds[1]);
        const DirectStreamControlPolicy fd_limits{4096, 4096, std::chrono::milliseconds(1000)};
        auto left = DirectStreamControlByteStream::open_with_io(left_io, device(), fd_limits);
        auto right = DirectStreamControlByteStream::open_with_io(right_io, device(), fd_limits);
        CopiedStreamConnectionState left_state(connection()), right_state(connection(1));
        CopiedStreamTransportBackend sender(left, left_state, policy()), receiver(right, right_state, policy());
        AllocationGenerationSource generations; auto source = buffer(generations), destination = buffer(generations);
        for (std::size_t index = 0; index < 16; ++index) source.allocation->writable_bytes()[index] = static_cast<std::byte>(index * 13U);
        const auto message = transfer();
        const auto tx = sender.prepare_send(message, synchronous_source(message, source));
        const auto rx = receiver.prepare_receive(message, destination); sender.submit(tx); receiver.submit(rx);
        static_cast<void>(sender.poll()); static_cast<void>(receiver.poll()); static_cast<void>(sender.poll());
        finish_receive(receiver, rx, message, receiver.receive_buffer(rx));
        expect(sender.poll()->kind == CompletionKind::remote_consumer_complete &&
               std::equal(source.allocation->bytes().begin(), source.allocation->bytes().end(), destination.allocation->bytes().begin()),
               "real socketpair copies every opaque tensor byte and completes descriptor-bound receive/consumer ACK lifecycle");
    }
    {
        int fds[2]{}; if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) throw std::runtime_error("socketpair failed");
        Pipe unused; SocketIo left_io(unused, 0, fds[0]), right_io(unused, 1, fds[1]);
        const DirectStreamControlPolicy fd_limits{4096, 4096, std::chrono::milliseconds(1000)};
        auto left = DirectStreamControlByteStream::open_with_io(left_io, device(), fd_limits);
        auto right = DirectStreamControlByteStream::open_with_io(right_io, device(), fd_limits);
        CopiedStreamConnectionState left_state(connection()), right_state(connection(1));
        CopiedStreamTransportBackend sender(left, left_state, policy()), receiver(right, right_state, policy());
        AllocationGenerationSource generations; auto source_buffer = buffer(generations); fill(source_buffer, std::byte{41});
        Local local; Control control; Consumer consumer; Source source(source_buffer);
        const auto left_plan = placement(), right_plan = placement(1);
        const DistributedRuntimeRequest request{9, 11, 2, 7};
        // Existing control receivers use the sender's global placement view.
        DistributedControlReceiver left_queue(right_plan, request, 0, local), right_queue(left_plan, request, 1, local);
        const auto message = transfer(); right_queue.receive_boundary_transfer(message); // Already validated v2 metadata queue.
        CopiedStreamDistributedBackend sending(sender, left_plan, request, control, left_queue, local, source, consumer, generations);
        CopiedStreamDistributedBackend receiving(receiver, right_plan, request, control, right_queue, local, source, consumer, generations);
        std::exception_ptr receive_error; std::optional<DistributedBoundaryTransfer> observed;
        std::thread peer([&] { try { observed = receiving.receive_boundary(right_plan.boundaries[0]); } catch (...) { receive_error = std::current_exception(); } });
        std::exception_ptr send_error;
        try { sending.send_boundary(left_plan.boundaries[0], message); } catch (...) { send_error = std::current_exception(); }
        peer.join();
        if (receive_error) std::rethrow_exception(receive_error);
        if (send_error) std::rethrow_exception(send_error);
        expect(observed && observed->flow.direction == LinkDirection::inbound && consumer.actual ==
               std::vector<std::byte>(source_buffer.allocation->bytes().begin(), source_buffer.allocation->bytes().end()) &&
               control.published == 1 && consumer.calls == 1 && !right_queue.has_boundary(91),
               "queued v2 metadata integrates into existing distributed boundary SPI with exact copy and receiver flow normalization");
        DistributedRequestState state{31, 11, 2, 1, false};
        validate_distributed_boundary_transfer_receive(state, right_plan, right_plan.boundaries[0], *observed, 1);
        expect(state.next_receive_sequence == 2 && !state.failed, "existing runtime retains logical sequence/state validation authority");
    }
#endif
    if (failures == 0) std::cout << "copied stream transport tests passed\n";
    return failures == 0 ? 0 : 1;
}
