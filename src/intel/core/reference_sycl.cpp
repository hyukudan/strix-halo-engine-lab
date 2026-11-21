#include "reference.hpp"

#include <stdexcept>
#include <string>

#if defined(STRIX_HAVE_SYCL) && STRIX_HAVE_SYCL
#include "backend_sycl.hpp"
#include "strix/core/chunked_dot.hpp"
#include "strix/session/backend.hpp"

#if defined(STRIX_HAVE_SYCL_MATRIX)
#include "xmx_sycl.hpp"
#include <vector>
#endif

#include <limits>
#include <map>
#include <utility>
#endif

namespace strix::intel {
namespace {
#if defined(STRIX_HAVE_SYCL) && STRIX_HAVE_SYCL
void require_ok(Status status, const char* action) {
    if (status != Status::ok)
        throw std::runtime_error(std::string("Intel reference ") + action + ": " + status_name(status));
}

struct QueueOwner {
    Queue handle;
    ~QueueOwner() { close_queue(handle); }
};

class Transfers final : public session::Backend {
    struct Storage final : session::detail::BufferState {
        Queue queue;
        void* data = nullptr;
        ~Storage() override { if (data) (void)release(queue, data); }
    };
public:
    Transfers() {
        const auto status = open_queue(0, &owner_.handle, &caps_);
        const auto refusal = reference_refusal(status, caps_);
        if (!refusal.empty()) throw std::runtime_error(std::string(refusal));
        queue_ = create_queue();
    }
    session::Queue& staging() { return queue_; }
    void* address(session::BufferSlice range) { return address(checked(range)); }
    const Capabilities& caps() const { return caps_; }
    template<class Range, class Kernel>
    void launch(Range range, Kernel kernel) {
        run(checked(queue_), [&] { native().parallel_for(range, kernel); });
    }
private:
    QueueOwner owner_;
    Capabilities caps_;
    session::Queue queue_;
    sycl::queue& native() const { return detail::native_queue(owner_.handle); }
    static void* address(const session::detail::BoundBuffer& range) {
        return static_cast<std::byte*>(static_cast<const Storage&>(*range.state).data) + range.offset;
    }
    std::shared_ptr<session::detail::BufferState> allocate_buffer(std::size_t bytes, std::size_t alignment) override {
        auto buffer = std::make_shared<Storage>();
        buffer->queue = owner_.handle;
        buffer->data = sycl::aligned_alloc_device(alignment, bytes, native());
        if (!buffer->data) throw std::runtime_error("Intel reference device allocation failed");
        return buffer;
    }
    std::shared_ptr<session::detail::QueueState> make_queue() override {
        return std::make_shared<session::detail::QueueState>();
    }
    void transfer(session::detail::QueueState&, const session::detail::BoundBuffer* destination,
                  const session::detail::BoundBuffer* source, void* host_destination,
                  const void* host_source, std::size_t bytes) override {
        if (bytes)
            require_ok(strix::intel::copy(owner_.handle, destination ? address(*destination) : host_destination,
                                         source ? address(*source) : host_source, bytes), "copy submission");
    }
    void finish(session::detail::QueueState&) override {
        require_ok(strix::intel::synchronize(owner_.handle), "completion");
    }
    void drain(session::detail::QueueState&) noexcept override { (void)strix::intel::synchronize(owner_.handle); }
    void execute_plan(session::ExecutionPlan&) override {
        throw session::UnsupportedOperation("unknown Intel reference execution plan");
    }
};

class Allocation {
public:
    Allocation(Transfers& transfers, std::size_t bytes)
        : transfers_(transfers), bytes_(bytes), buffer_(bytes ? transfers_.allocate(bytes) : session::Buffer{}) {}
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    template<class T> T* as() const { return bytes_ ? static_cast<T*>(transfers_.address(range())) : nullptr; }
    void upload(const void* data, std::size_t bytes) const {
        if (bytes_) transfers_.upload(transfers_.staging(), range(), {static_cast<const std::byte*>(data), bytes});
    }
    void download(void* data, std::size_t bytes) const {
        if (bytes) transfers_.download(transfers_.staging(), {buffer_, 0, bytes}, {static_cast<std::byte*>(data), bytes});
    }
private:
    session::BufferSlice range() const { return {buffer_, 0, bytes_}; }
    Transfers& transfers_;
    std::size_t bytes_;
    session::Buffer buffer_;
};

class Reference final : public core::DenseReference, public core::RoutedReference {
public:
    explicit Reference(LinearPolicy policy) : policy_(policy) {}
    void gather(core::Q8Rows matrix, std::uint32_t row, std::span<float> output) override {
        if (row >= matrix.rows || output.size() != matrix.width)
            throw std::invalid_argument("invalid Intel embedding row");
        const auto* source = weights(matrix) + static_cast<std::size_t>(row) * matrix.row_bytes;
        Allocation result(transfers_, output.size_bytes());
        auto* destination = result.as<float>();
        transfers_.launch(sycl::range<1>(output.size()), [=](sycl::id<1> item) {
            const auto i = static_cast<std::uint32_t>(item[0]);
            const auto* block = source + (i / 32U) * std::size_t(34);
            destination[i] = dense::q8_value(block, i % 32U, dense::q8_scale(block));
        });
        result.download(output.data(), output.size_bytes());
    }

    void linear(core::Q8Rows matrix, std::span<const float> input, std::span<float> output) override {
        if (!matrix.width || !matrix.rows || input.empty() || input.size() % matrix.width ||
            output.size() % matrix.rows || output.size() / matrix.rows != input.size() / matrix.width)
            throw std::invalid_argument("invalid Intel dense linear extents");
        const auto* coefficients = weights(matrix);
        const auto width = matrix.width;
        const auto pitch = matrix.row_bytes;
        scalar_linear(matrix.rows, width, input, output, [=](std::size_t row, const float* x) {
            return dense::q8_row_dot<false>(
                coefficients + row * pitch, width, [=](std::uint32_t block, std::uint32_t lane) {
                    return x[block * 32U + lane];
                });
        });
    }

    void linear(core::F32Rows matrix, std::span<const float> input, std::span<float> output) override {
        const auto batch = linear_batch(matrix, input.size(), output.size());
        const auto plan = plan_linear(Status::ok, transfers_.caps(), policy_, batch, matrix.rows, matrix.width);
#if defined(STRIX_HAVE_SYCL_MATRIX)
        if (plan.path == LinearPath::xmx && linear_xmx(plan, matrix, input, output)) return;
#else
        (void)plan;
#endif
        Allocation coefficients(transfers_, matrix.values.size_bytes());
        coefficients.upload(matrix.values.data(), matrix.values.size_bytes());
        const auto* w = coefficients.as<float>();
        const auto width = matrix.width;
        scalar_linear(matrix.rows, width, input, output, [=](std::size_t row, const float* x) {
            const auto* wr = w + row * width;
            return core::chunked_dot_of([=](std::size_t i) { return wr[i]; },
                                        [=](std::size_t i) { return x[i]; }, width);
        });
    }

    void route(const moe::RoutePolicy& policy, std::span<const float> scores, std::span<const float> bias,
               std::span<std::uint32_t> ids, std::span<float> selected_weights) override {
        moe::validate_route_row(policy, scores.size(), bias.size(), ids.size(), selected_weights.size());
        scores = scores.first(policy.experts);
        bias = bias.first(policy.selection_bias ? policy.experts : 0U);
        ids = ids.first(policy.top_k);
        selected_weights = selected_weights.first(policy.top_k);
        Allocation scores_in(transfers_, scores.size_bytes()), bias_in(transfers_, bias.size_bytes());
        Allocation ids_out(transfers_, ids.size_bytes()), weights_out(transfers_, selected_weights.size_bytes());
        Allocation scratch(transfers_, policy.experts);
        scores_in.upload(scores.data(), scores.size_bytes());
        bias_in.upload(bias.data(), bias.size_bytes());
        const auto* s = scores_in.as<float>();
        const auto* b = bias_in.as<float>();
        auto* i = ids_out.as<std::uint32_t>();
        auto* w = weights_out.as<float>();
        auto* taken = scratch.as<std::uint8_t>();
        transfers_.launch(sycl::range<1>(1), [=](sycl::id<1>) {
            moe::route_select_row(policy, s, b, i, w, taken);
        });
        ids_out.download(ids.data(), ids.size_bytes());
        weights_out.download(selected_weights.data(), selected_weights.size_bytes());
    }

    void rms_norm(std::span<const float> input, std::span<const float> scale, std::span<float> output,
                  std::uint32_t width, float epsilon) override {
        if (!width || input.empty() || input.size() % width || output.size() != input.size() ||
            (!scale.empty() && scale.size() != width) || !(epsilon > 0.0F))
            throw std::invalid_argument("invalid Intel RMSNorm extents");
        Allocation source(transfers_, input.size_bytes()), multiplier(transfers_, scale.size_bytes()),
                   result(transfers_, output.size_bytes());
        source.upload(input.data(), input.size_bytes());
        multiplier.upload(scale.data(), scale.size_bytes());
        const auto* x = source.as<float>();
        const auto* w = multiplier.as<float>();
        auto* y = result.as<float>();
        transfers_.launch(sycl::range<1>(input.size() / width), [=](sycl::id<1> item) {
            const auto base = item[0] * width;
            float sum = 0.0F;
            for (std::uint32_t i = 0; i < width; ++i) sum += x[base + i] * x[base + i];
            const float mean = sum / static_cast<float>(width) + epsilon;
            const float inv = sycl::isfinite(mean) ? 1.0F / sycl::sqrt(mean) : std::numeric_limits<float>::quiet_NaN();
            for (std::uint32_t i = 0; i < width; ++i) y[base + i] = (x[base + i] * inv) * (w ? w[i] : 1.0F);
        });
        result.download(output.data(), output.size_bytes());
    }

    void rope(std::span<float> heads, std::span<const float> interleaved, std::uint32_t width) override {
        if (!width || width % 2U || heads.empty() || heads.size() % width ||
            interleaved.size() % 2U || interleaved.size() > width)
            throw std::invalid_argument("invalid Intel RoPE extents");
        if (interleaved.empty()) return;
        Allocation vectors(transfers_, heads.size_bytes()), table(transfers_, interleaved.size_bytes());
        vectors.upload(heads.data(), heads.size_bytes());
        table.upload(interleaved.data(), interleaved.size_bytes());
        auto* x = vectors.as<float>();
        const auto* angles = table.as<float>();
        transfers_.launch(sycl::range<2>(heads.size() / width, interleaved.size() / 2U), [=](sycl::id<2> item) {
            const auto i = item[1];
            auto* head = x + item[0] * width;
            const float a = head[i], b = head[i + width / 2U];
            const float cs = angles[2U * i], sn = angles[2U * i + 1U];
            head[i] = a * cs - b * sn;
            head[i + width / 2U] = b * cs + a * sn;
        });
        vectors.download(heads.data(), heads.size_bytes());
    }

    void attention(const core::DenseAttentionRow& row, std::span<float> output) override {
        const auto count = attention_score_elements(row, output.size());
        Allocation queries(transfers_, row.query.size_bytes()), keys(transfers_, row.keys.size_bytes()),
                   values(transfers_, row.values.size_bytes()), result(transfers_, output.size_bytes()),
                   scores(transfers_, count * sizeof(float));
        queries.upload(row.query.data(), row.query.size_bytes());
        keys.upload(row.keys.data(), row.keys.size_bytes());
        values.upload(row.values.data(), row.values.size_bytes());
        const auto width = row.width;
        const auto width_kv = std::size_t(row.kv_heads) * width;
        const auto group_size = row.heads / row.kv_heads;
        const auto positions = row.positions;
        const auto* q = queries.as<float>();
        const auto* k = keys.as<float>();
        const auto* v = values.as<float>();
        auto* s = scores.as<float>();
        auto* y = result.as<float>();
        transfers_.launch(sycl::range<2>(row.heads, positions), [=](sycl::id<2> item) {
            const auto head = item[0], position = item[1];
            const auto* query = q + head * width;
            const auto* key = k + position * width_kv + (head / group_size) * width;
            float score = 0.0F;
            for (std::uint32_t i = 0; i < width; ++i) score += query[i] * key[i];
            s[head * positions + position] = score;
        });
        transfers_.launch(sycl::range<1>(row.heads), [=](sycl::id<1> item) {
            const auto head = item[0];
            auto* scores_row = s + head * positions;
            float maximum = scores_row[0];
            for (std::size_t p = 0; p < positions; ++p)
                maximum = scores_row[p] > maximum ? scores_row[p] : maximum;
            float total = 0.0F;
            for (std::size_t p = 0; p < positions; ++p) {
                const float delta = scores_row[p] - maximum;
                scores_row[p] = sycl::isfinite(delta) ? sycl::exp(delta) : std::numeric_limits<float>::quiet_NaN();
                total += scores_row[p];
            }
            const auto group = head / group_size;
            for (std::uint32_t i = 0; i < width; ++i) {
                float sum = 0.0F;
                for (std::size_t p = 0; p < positions; ++p)
                    sum += scores_row[p] * v[p * width_kv + group * width + i];
                y[head * width + i] = sum / total;
            }
        });
        result.download(output.data(), output.size_bytes());
    }

private:
    template<class Dot>
    void scalar_linear(std::size_t rows, std::size_t width, std::span<const float> input,
                       std::span<float> output, Dot dot) {
        Allocation source(transfers_, input.size_bytes()), result(transfers_, output.size_bytes());
        source.upload(input.data(), input.size_bytes());
        const auto* x = source.as<float>();
        auto* y = result.as<float>();
        transfers_.launch(sycl::range<2>(input.size() / width, rows), [=](sycl::id<2> item) {
            y[item[0] * rows + item[1]] = dot(item[1], x + item[0] * width);
        });
        result.download(output.data(), output.size_bytes());
    }

#if defined(STRIX_HAVE_SYCL_MATRIX)
    bool linear_xmx(const LinearPlan& plan, core::F32Rows matrix,
                    std::span<const float> input, std::span<float> output) {
        using namespace arch::xe2;
        const auto in_range = [](float v) { return v >= -65504.0F && v <= 65504.0F; };
        if (!std::all_of(input.begin(), input.end(), in_range) ||
            !std::all_of(matrix.values.begin(), matrix.values.end(), in_range)) return false;
        std::vector<sycl::half> a(plan.padded_batch * matrix.width, sycl::half(0.0F));
        for (std::size_t i = 0; i < input.size(); ++i) a[i] = static_cast<sycl::half>(input[i]);
        std::vector<sycl::half> b(matrix.values.size());
        for (std::size_t n = 0; n < matrix.rows; ++n)
            for (std::size_t k = 0; k < matrix.width; ++k)
                b[packed_b_offset(k, n, matrix.rows)] = static_cast<sycl::half>(matrix.values[n * matrix.width + k]);
        const auto a_bytes = a.size() * sizeof(sycl::half), b_bytes = b.size() * sizeof(sycl::half);
        Allocation a_device(transfers_, a_bytes), b_device(transfers_, b_bytes);
        Allocation result(transfers_, plan.padded_batch * matrix.rows * sizeof(float));
        a_device.upload(a.data(), a_bytes);
        b_device.upload(b.data(), b_bytes);
        transfers_.launch(sycl::nd_range<2>({plan.padded_batch / kMatM, matrix.rows / kMatN * kSubgroup}, {1, kSubgroup}),
                          detail::XmxLinear{a_device.as<sycl::half>(), b_device.as<sycl::half>(),
                                            result.as<float>(), matrix.rows, matrix.width});
        result.download(output.data(), output.size_bytes());
        return true;
    }
#endif

    const std::uint8_t* weights(core::Q8Rows matrix) {
        const auto bytes = q8_matrix_bytes(matrix);
        if (const auto found = weights_.find(matrix.data); found != weights_.end()) {
            if (found->second.first != bytes) throw std::invalid_argument("inconsistent Intel matrix size");
            return found->second.second->as<std::uint8_t>();
        }
        auto allocation = std::make_unique<Allocation>(transfers_, bytes);
        allocation->upload(matrix.data, bytes);
        const auto* pointer = allocation->as<std::uint8_t>();
        weights_.emplace(matrix.data, std::make_pair(bytes, std::move(allocation)));
        return pointer;
    }
    LinearPolicy policy_;
    Transfers transfers_; // released after all device allocations
    std::map<const std::uint8_t*, std::pair<std::size_t, std::unique_ptr<Allocation>>> weights_;
};
#endif
} // namespace

std::unique_ptr<core::DenseReference> open_reference() {
#if defined(STRIX_HAVE_SYCL) && STRIX_HAVE_SYCL
    return std::make_unique<Reference>(LinearPolicy::reference);
#else
    throw std::runtime_error(std::string(reference_refusal(Status::not_built, {})));
#endif
}

std::unique_ptr<core::RoutedReference> open_routed_reference(LinearPolicy policy) {
#if defined(STRIX_HAVE_SYCL) && STRIX_HAVE_SYCL
    return std::make_unique<Reference>(policy);
#else
    (void)policy;
    throw std::runtime_error(std::string(reference_refusal(Status::not_built, {})));
#endif
}

} // namespace strix::intel
