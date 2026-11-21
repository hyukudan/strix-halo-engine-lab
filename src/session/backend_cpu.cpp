#include "strix/session/backend_cpu.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <type_traits>

namespace strix::session {
namespace {
[[noreturn]] void unsupported() { throw UnsupportedOperation("backend resource operation unsupported"); }
bool power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }
void no_overlap(const detail::BoundBuffer& a, const detail::BoundBuffer& b) {
    if (a.state == b.state && a.bytes && b.bytes &&
        a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes)
        throw std::invalid_argument("overlapping resource ranges");
}
void validate_copy(const detail::BoundBuffer& dst, const detail::BoundBuffer& src) {
    if (dst.bytes != src.bytes) throw std::invalid_argument("copy size mismatch");
    no_overlap(dst, src);
}
}

Buffer Backend::allocate(std::size_t bytes, std::size_t alignment) {
    registry_sealed_ = true;
    if (!bytes || !power_of_two(alignment)) throw std::invalid_argument("invalid buffer size/alignment");
    if (bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1))
        throw std::overflow_error("buffer alignment reservation overflow");
    Buffer result;
    result.state_ = allocate_buffer(bytes, alignment);
    if (!result.state_) throw std::runtime_error("backend returned no buffer");
    result.state_->owner = owner_;
    result.state_->bytes = bytes;
    result.state_->alignment = alignment;
    return result;
}
Queue Backend::create_queue() {
    registry_sealed_ = true;
    Queue result;
    result.state_ = make_queue();
    if (!result.state_) throw std::runtime_error("backend returned no queue");
    result.state_->owner = owner_;
    return result;
}
detail::QueueState& Backend::checked(const Queue& queue) const {
    if (!queue.state_ || queue.state_->released || queue.state_->owner != owner_)
        throw std::invalid_argument("invalid, released or foreign queue");
    if (queue.state_->failed) throw std::runtime_error("queue is poisoned by an execution error");
    return *queue.state_;
}
detail::BoundBuffer Backend::checked(const BufferSlice& view) const {
    auto state = view.state_.lock();
    if (!state || state->released || state->owner != owner_)
        throw std::invalid_argument("invalid, released or foreign buffer");
    if (view.offset > state->bytes || view.bytes > state->bytes - view.offset)
        throw std::out_of_range("buffer byte range");
    return {std::move(state), view.offset, view.bytes};
}
void Backend::run(detail::QueueState& queue, const std::function<void()>& call) {
    try { call(); finish(queue); }
    catch (...) { queue.failed = true; drain(queue); throw; }
}
Event Backend::record(Queue& queue) {
    auto& q = checked(queue);
    Event result;
    run(q, [&] { result.state_ = record_event(q); });
    if (!result.state_) throw std::runtime_error("backend returned no event");
    result.state_->owner = owner_;
    return result;
}
void Backend::wait(Queue& queue, const Event& event) {
    auto& q = checked(queue);
    if (!event.state_ || event.state_->released || event.state_->owner != owner_)
        throw std::invalid_argument("invalid, released or foreign event");
    run(q, [&] { wait_event(q, *event.state_); });
}
void Backend::synchronize(Queue& queue) { auto& q = checked(queue); run(q, [] {}); }
void Backend::upload(Queue& queue, BufferSlice destination, std::span<const std::byte> source) {
    auto& q = checked(queue);
    const auto dst = checked(destination);
    if (dst.bytes != source.size() || (!source.empty() && !source.data()))
        throw std::invalid_argument("upload span size/data");
    run(q, [&] { transfer(q, &dst, nullptr, nullptr, source.data(), dst.bytes); });
}
void Backend::download(Queue& queue, BufferSlice source, std::span<std::byte> destination) {
    auto& q = checked(queue);
    const auto src = checked(source);
    if (src.bytes != destination.size() || (!destination.empty() && !destination.data()))
        throw std::invalid_argument("download span size/data");
    run(q, [&] { transfer(q, nullptr, &src, destination.data(), nullptr, src.bytes); });
}
void Backend::copy(Queue& queue, BufferSlice destination, BufferSlice source) {
    auto& q = checked(queue);
    const auto dst = checked(destination), src = checked(source);
    validate_copy(dst, src);
    run(q, [&] { transfer(q, &dst, &src, nullptr, nullptr, dst.bytes); });
}
bool Backend::supports(const OperationKey& key) const noexcept {
    registry_sealed_ = true;
    return std::any_of(registry_.begin(), registry_.end(), [&](const auto& entry) { return entry.key == key; });
}
std::optional<OperationSchema> Backend::query(const OperationKey& key) const noexcept {
    registry_sealed_ = true;
    for (const auto& entry : registry_) if (entry.key == key) return entry.schema;
    return std::nullopt;
}
void Backend::register_operation(OperationKey key, OperationSchema schema, Executor executor) {
    if (registry_sealed_) throw std::logic_error("operation registry is sealed after first use/query");
    const bool duplicate = std::any_of(registry_.begin(), registry_.end(), [&](const auto& entry) { return entry.key == key; });
    if (key.backend != backend_name() || key.backend.empty() || key.operation.empty() || key.dtype.empty() ||
        key.quantization.empty() || !schema.element_bytes || !power_of_two(schema.alignment) || !executor || duplicate)
        throw std::invalid_argument("invalid or duplicate operation registration");
    registry_.push_back({std::move(key), schema, std::move(executor)});
}
const Backend::Registration& Backend::registered(const OperationKey& key) {
    registry_sealed_ = true;
    const auto found = std::find_if(registry_.begin(), registry_.end(), [&](const auto& entry) { return entry.key == key; });
    if (found == registry_.end()) throw UnsupportedOperation("operation/dtype/quantization/backend key unsupported");
    return *found;
}
detail::BoundOperation Backend::bind_operation(const Operation& op, const OperationSchema& schema) const {
    if (op.inputs.size() != schema.inputs) throw std::invalid_argument("operation input count");
    if (op.elements > schema.max_elements || op.elements > std::numeric_limits<std::size_t>::max() / schema.element_bytes)
        throw std::overflow_error("operation element count");
    const auto bytes = op.elements * schema.element_bytes;
    auto bind = [&](const BufferSlice& view) {
        auto bound = checked(view);
        if (bound.bytes != bytes || bound.offset % schema.alignment || bound.state->alignment < schema.alignment)
            throw std::invalid_argument("operation byte size/alignment");
        return bound;
    };
    detail::BoundOperation bound;
    bound.elements = op.elements;
    bound.output = bind(op.output);
    for (const auto& view : op.inputs) {
        auto input = bind(view);
        no_overlap(bound.output, input);
        bound.inputs.push_back(std::move(input));
    }
    return bound;
}
void Backend::dispatch(Queue& queue, const Operation& op) {
    const auto& entry = registered(op.key);
    auto& q = checked(queue);
    const auto bound = bind_operation(op, entry.schema);
    // Every resource is pinned through completion/drain, with no dispatch in
    // the executor's element loop. Output/input aliasing is conservatively refused.
    run(q, [&] { entry.execute(q, bound); });
}
Backend::BoundCommand Backend::bind_step(const SubmissionStep& step) {
    return std::visit([this](const auto& command) -> BoundCommand {
        using Step = std::decay_t<decltype(command)>;
        if constexpr (std::is_same_v<Step, SubmissionUpload>) {
            auto dst = checked(command.destination);
            const auto& source = command.source;
            if (!source || dst.bytes != source->size()) throw std::invalid_argument("upload storage size");
            return [this, dst = std::move(dst), source](detail::QueueState& q) {
                transfer(q, &dst, nullptr, nullptr, source->data(), dst.bytes);
            };
        } else if constexpr (std::is_same_v<Step, SubmissionDownload>) {
            auto src = checked(command.source);
            const auto& destination = command.destination;
            if (!destination || src.bytes != destination->size()) throw std::invalid_argument("download storage size");
            return [this, src = std::move(src), destination](detail::QueueState& q) {
                transfer(q, nullptr, &src, destination->data(), nullptr, src.bytes);
            };
        } else if constexpr (std::is_same_v<Step, SubmissionCopy>) {
            auto dst = checked(command.destination), src = checked(command.source);
            validate_copy(dst, src);
            return [this, dst = std::move(dst), src = std::move(src)](detail::QueueState& q) {
                transfer(q, &dst, &src, nullptr, nullptr, dst.bytes);
            };
        } else {
            const auto& entry = registered(command.key);
            auto bound = bind_operation(command, entry.schema);
            return [executor = &entry.execute, bound = std::move(bound)](detail::QueueState& q) {
                (*executor)(q, bound);
            };
        }
    }, step);
}
Submission Backend::submit(Queue& queue, std::span<const SubmissionStep> steps) {
    if (!supports_async_submission()) throw UnsupportedOperation("backend asynchronous submission unsupported");
    (void)checked(queue);
    std::vector<BoundCommand> commands;
    commands.reserve(steps.size());
    for (const auto& step : steps) commands.push_back(bind_step(step));
    auto fence = enqueue_batch(queue.state_, std::move(commands));
    if (!fence) throw std::runtime_error("backend returned no submission fence");
    return Submission(std::move(fence));
}
std::shared_ptr<detail::BufferState> Backend::allocate_buffer(std::size_t, std::size_t) { unsupported(); }
std::shared_ptr<detail::QueueState> Backend::make_queue() { unsupported(); }
std::shared_ptr<detail::EventState> Backend::record_event(detail::QueueState&) { unsupported(); }
void Backend::wait_event(detail::QueueState&, detail::EventState&) { unsupported(); }
void Backend::finish(detail::QueueState&) { unsupported(); }
void Backend::transfer(detail::QueueState&, const detail::BoundBuffer*, const detail::BoundBuffer*,
                       void*, const void*, std::size_t) { unsupported(); }
std::unique_ptr<detail::SubmissionFence> Backend::enqueue_batch(
    std::shared_ptr<detail::QueueState>, std::vector<BoundCommand>) { unsupported(); }

struct CpuBackend::State {
    std::size_t limit, reserved = 0;
    explicit State(std::size_t limit_bytes) : limit(limit_bytes) {}
};
namespace {
struct CpuSubmissionFence final : detail::SubmissionFence {
    std::exception_ptr error;
    SubmissionStatus poll() const override { return error ? SubmissionStatus::failed : SubmissionStatus::complete; }
    SubmissionStatus wait() const override {
        if (error) std::rethrow_exception(error);
        return SubmissionStatus::complete;
    }
    bool cancel() override { return false; }
};
struct CpuBuffer final : detail::BufferState {
    std::byte* data = nullptr;
    std::size_t physical_alignment = 0;
    std::function<void()> release;
    ~CpuBuffer() override {
        if (data) ::operator delete(data, std::align_val_t(physical_alignment));
        if (release) release();
    }
};
std::byte* address(const detail::BoundBuffer& bound) {
    return static_cast<CpuBuffer&>(*bound.state).data + bound.offset;
}
}
CpuBackend::CpuBackend(std::size_t budget_bytes) : state_(std::make_shared<State>(budget_bytes)) {
    register_operation({"add", "f32", "none", "cpu"}, {2, sizeof(float), alignof(float)},
        [](detail::QueueState&, const detail::BoundOperation& op) {
            const auto* a = address(op.inputs[0]);
            const auto* b = address(op.inputs[1]);
            auto* out = address(op.output);
            for (std::size_t i = 0; i < op.elements; ++i) {
                float x, y;
                std::memcpy(&x, a + i * sizeof(float), sizeof(float));
                std::memcpy(&y, b + i * sizeof(float), sizeof(float));
                const float sum = x + y;
                std::memcpy(out + i * sizeof(float), &sum, sizeof(float));
            }
        });
}
CpuBackend::~CpuBackend() = default;
MemoryBudget CpuBackend::memory_budget() const {
    return {state_->limit, state_->reserved, state_->limit - state_->reserved, 0, 0, false};
}
std::shared_ptr<detail::BufferState> CpuBackend::allocate_buffer(std::size_t bytes, std::size_t alignment) {
    const auto physical_alignment = std::max(alignment, alignof(std::max_align_t));
    if (bytes > std::numeric_limits<std::size_t>::max() - (physical_alignment - 1))
        throw std::overflow_error("CPU alignment reservation overflow");
    const auto reservation = bytes + physical_alignment - 1;
    if (reservation > state_->limit - state_->reserved) throw std::length_error("CPU buffer budget exhausted");
    auto buffer = std::make_shared<CpuBuffer>();
    buffer->physical_alignment = physical_alignment;
    buffer->data = static_cast<std::byte*>(::operator new(bytes, std::align_val_t(buffer->physical_alignment)));
    buffer->release = [state = state_, reservation] { state->reserved -= reservation; };
    state_->reserved += reservation;
    return buffer;
}
std::shared_ptr<detail::QueueState> CpuBackend::make_queue() { return std::make_shared<detail::QueueState>(); }
std::shared_ptr<detail::EventState> CpuBackend::record_event(detail::QueueState&) { return std::make_shared<detail::EventState>(); }
void CpuBackend::wait_event(detail::QueueState&, detail::EventState&) {}
void CpuBackend::finish(detail::QueueState&) {}
std::unique_ptr<detail::SubmissionFence> CpuBackend::enqueue_batch(
    std::shared_ptr<detail::QueueState> queue, std::vector<BoundCommand> commands) {
    auto fence = std::make_unique<CpuSubmissionFence>();
    try {
        run(*queue, [&] {
            for (const auto& command : commands) command(*queue);
        });
    } catch (...) {
        fence->error = std::current_exception();
    }
    return fence;
}
void CpuBackend::transfer(detail::QueueState&, const detail::BoundBuffer* dst, const detail::BoundBuffer* src,
                          void* host_dst, const void* host_src, std::size_t bytes) {
    if (bytes) std::memcpy(dst ? address(*dst) : host_dst, src ? address(*src) : host_src, bytes);
}
void CpuBackend::execute_plan(ExecutionPlan&) { throw UnsupportedOperation("CPU family plan unsupported"); }
}
