#include "exec/catalog_cuda_wr1.hpp"
#include "exec/cuda_decode_lowering.hpp"
#include "cuda/core/head_argmax.cuh"
#include "strix/cuda_core/dense.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace strix::exec;
namespace decode = strix::exec::cuda_decode;
namespace wr1 = decode::wr1;
namespace mix = strix::core::residual_mix;
namespace argmax = strix::cuda::core::head_argmax;
namespace head = strix::head_argmax;
using strix::core::Bf16RowProjection;

constexpr float canary = -317.5F;
constexpr float epsilon = 0.0001F;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <class T>
void equal(std::span<const T> got, std::span<const T> want, const char* name) {
    if (got.size() != want.size() || std::memcmp(got.data(), want.data(), got.size_bytes()) != 0)
        throw std::runtime_error(name);
}

template <class T>
class Device {
public:
    explicit Device(const std::vector<T>& initial) : elements_(initial.size()) {
        check(cudaMalloc(&data, elements_ * sizeof(T)));
        const auto status = cudaMemcpy(data, initial.data(), elements_ * sizeof(T), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            cudaFree(data);
            check(status);
        }
    }
    Device(std::size_t elements, T value) : Device(std::vector<T>(elements, value)) {}
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] std::vector<T> read() const {
        std::vector<T> result(elements_);
        check(cudaMemcpy(result.data(), data, elements_ * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
    T* data{};
private:
    std::size_t elements_;
};

struct Mirror {
    explicit Mirror(std::size_t size) : wide(size, canary), scalar(size, canary) {}
    explicit Mirror(const std::vector<float>& value) : wide(value), scalar(value) {}
    void compare(const char* name) const { equal<float>(wide.read(), scalar.read(), name); }
    Device<float> wide, scalar;
};

std::vector<float> input(std::size_t elements, unsigned salt = 0) {
    std::vector<float> result(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        const auto j = i + 13U * salt;
        result[i] = static_cast<float>(static_cast<int>((j * 367U + j / 19U) % 1009U) - 503) * 0.00171F;
    }
    return result;
}

std::vector<std::uint16_t> weights(std::size_t elements, unsigned salt = 0) {
    const auto values = input(elements, salt);
    std::vector<std::uint16_t> result(elements);
    for (std::size_t i = 0; i < elements; ++i)
        result[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(values[i]) >> 16U);
    return result;
}

StepBlock descriptor(unsigned width, bool padded) {
    StepBlock result{};
    result.h.rows = width;
    for (unsigned r = 0; r < width; ++r) {
        const bool valid = !padded || (r % 3U != 1U && r + 1U != width);
        if (valid) result.h.valid_mask |= 1U << r;
        // Distinct from a single consecutive position base and slot zero.
        result.row[r] = {r, padded ? 47U + r * 101U : 257U + r * 37U, 72U - r,
                         kStepNoParent, 1U << r, valid ? std::uint32_t{step_valid} : 0U, 0, 0};
    }
    return result;
}

void launch(const KernelBinding& binding, cudaStream_t stream = nullptr) {
    binding.fn(binding.args, binding.grid, stream);
}

struct Captured {
    ~Captured() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
    }
    void record(const KernelBinding& binding) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(binding, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
    }
    void replay() const {
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
    }
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
};

struct Case {
    Case(unsigned width, bool padded)
        : rows(width), step(descriptor(width, padded)), device(1, step), bind(device.data, rows, wr1::native_thunks()) {}

    template <class F>
    void active(const F& scalar) const {
        for (unsigned r = 0; r < rows; ++r)
            if (step_row_valid(step, r)) scalar(r);
    }

    const unsigned rows;
    const StepBlock step;
    Device<StepBlock> device;
    wr1::Bindings bind;
};

void dense(const Case& test) {
    constexpr unsigned width = 544, vocabulary = 4100, next = 36, tiled = 48, short_tile = 32;
    const Device<float> source(input(std::size_t{test.rows} * width));
    for (const auto mode : {decode::Bf16DenseMode::fp32, decode::Bf16DenseMode::mma3,
                            decode::Bf16DenseMode::mma3_round}) {
        const bool fp32 = mode == decode::Bf16DenseMode::fp32;
        const bool round = mode == decode::Bf16DenseMode::mma3_round;
        const unsigned dim0 = fp32 ? vocabulary : tiled, dim1 = fp32 ? next : short_tile;
        const Device<std::uint16_t> w0(weights(std::size_t{width} * dim0, 2)),
                                    w1(weights(std::size_t{width} * dim1, 3));
        Mirror y0(std::size_t{test.rows + 1U} * dim0), y1(std::size_t{test.rows + 1U} * dim1);
        const auto args = [&](const float* x, float* o0, float* o1) {
            return decode::Bf16DenseGroup{x, width, {Bf16RowProjection{w0.data, dim0, o0, round},
                                                  Bf16RowProjection{w1.data, dim1, o1, round}}, 2, mode};
        };
        launch(test.bind.bind(args(source.data, y0.wide.data, y1.wide.data)));
        if (fp32) launch(test.bind.bind(decode::Bf16Softcap{y0.wide.data, dim0, 4.5F}));
        test.active([&](unsigned r) {
            auto* const logits = y0.scalar.data + r * dim0;
            launch(decode::bind(args(source.data + r * width, logits, y1.scalar.data + r * dim1)));
            if (fp32) launch(decode::bind(decode::Bf16Softcap{logits, dim0, 4.5F}));
        });
        y0.compare("dense projection/softcap");
        y1.compare("dense grouped projection");
    }
}

void bank_and_mix(const Case& test) {
    constexpr unsigned width = 521, stride = width + 7U;
    const Device<float> a(input(test.rows * width, 4)), b(input(test.rows * width, 6)),
        queries(input(2U * width, 9));
    for (bool round : {false, true}) {
        Mirror norm((test.rows + 1U) * stride);
        const auto args = [&](const float* x, float* y, unsigned rows) {
            return mix::NormalizeArgs{x, y, rows, width, epsilon, 0, stride, round};
        };
        launch(test.bind.bind(args(a.data, norm.wide.data, test.rows)));
        test.active([&](unsigned r) {
            launch(decode::bind(args(a.data + r * width, norm.scalar.data + r * stride, 1)));
        });
        norm.compare("message bank RMS");
    }

    for (const unsigned banks : {1U, 2U}) {
        Mirror y0((test.rows + 1U) * width), y1((test.rows + 1U) * width);
        const auto args = [&](unsigned rows, unsigned row, float* first, float* second) {
            return mix::MixArgs{{a.data + row * width, b.data + row * width}, queries.data,
                                {first + row * width, second + row * width}, rows, width, banks, 2, epsilon, false};
        };
        launch(test.bind.bind(args(test.rows, 0, y0.wide.data, y1.wide.data)));
        test.active([&](unsigned r) { launch(decode::bind(args(1, r, y0.scalar.data, y1.scalar.data))); });
        y0.compare("residual mix query 0");
        y1.compare("residual mix query 1");
    }
}

void controller(const Case& test) {
    constexpr unsigned width = 512, rank = 32, scratch_size = 4U * width + rank;
    const Device<float> raw(input(test.rows * width, 7)), previous(input(test.rows * width, 9)),
        previous2(input(test.rows * width, 12)), query(input(width, 13));
    const Device<std::uint16_t> down(weights(2U * width * rank, 14)), forget(weights(width * rank, 15)),
        gain(weights(width * rank, 16)), norm(weights(width, 17));
    Device<float> scratch(decode::controller_scratch_floats(width, rank, test.rows), canary);
    Device<float> ref_scratch(test.rows * scratch_size, canary);
    Mirror y((test.rows + 1U) * width);
    const auto lower = [&](unsigned rows, unsigned row, float* result, float* work, const auto& add) {
        const auto first = row * width, size = rows * width;
        const std::array bank{std::span<const float>(previous.data + first, size),
                              std::span<const float>(previous2.data + first, size)};
        decode::lower_low_rank_controller(width, rank, epsilon, {query.data, width}, {raw.data + first, size},
            bank, {result + first, size}, {work, decode::controller_scratch_floats(width, rank, rows)},
            down.data, forget.data, gain.data,
            norm.data, add, rows);
    };
    lower(test.rows, 0, y.wide.data, scratch.data,
          [&](const auto& a, bool) { launch(test.bind.bind(a)); });
    test.active([&](unsigned r) {
        lower(1, r, y.scalar.data, ref_scratch.data + r * scratch_size,
              [&](const auto& a, bool) { launch(decode::bind(a)); });
    });
    y.compare("controller result");
    const auto actual = scratch.read(), reference = ref_scratch.read();
    for (unsigned r = 0; r < test.rows; ++r) {
        const auto expected = std::span(reference).subspan(r * scratch_size, scratch_size);
        for (unsigned i = 0; i < 4; ++i)
            equal<float>(std::span(actual).subspan((i * test.rows + r) * width, width),
                         expected.subspan(i * width, width), "controller scratch plane");
        equal<float>(std::span(actual).subspan(4U * test.rows * width + r * rank, rank),
                     expected.last(rank), "controller latent");
    }
    if (test.rows > 1U) {
        Device<float> paired(2U * width * test.rows, canary);
        test.active([&](unsigned r) {
            auto* const output = paired.data + r * 2U * width;
            launch(decode::bind(mix::NormalizePairArgs{
                {raw.data + r * width, output, 1, width, epsilon, width, 2U * width, true},
                {ref_scratch.data + r * scratch_size, output + width, 1, width, epsilon, width, 2U * width, true}}));
        });
        equal<float>(std::span(actual).subspan(test.rows * scratch_size), paired.read(), "controller normalized pair");
    }
}

void relative(const Case& test) {
    constexpr unsigned heads = 3, extent = 91;
    const Device<float> x(input(test.rows * heads * 16U, 18));
    const Device<std::uint16_t> w(weights(16U * extent, 19));
    Mirror y((test.rows + 1U) * heads * extent);
    launch(test.bind.bind(decode::Bf16RelativeProjection16{x.data, w.data, y.wide.data, heads, extent}));
    test.active([&](unsigned r) {
        launch(decode::bind(decode::Bf16RelativeProjection16{
            x.data + r * heads * 16U, w.data, y.scalar.data + r * heads * extent, heads, extent}));
    });
    y.compare("relative logits");
}

void clip(const Case& test) {
    constexpr unsigned heads = 3;
    for (unsigned width : {16U, 80U, 159U}) {
        const auto stride = width * heads;
        const Device<float> x(input(test.rows * stride, 20));
        const Device<std::uint16_t> gain(weights(stride, 21));
        Mirror y((test.rows + 1U) * stride);
        const auto args = [&](const float* in, float* out, const unsigned* pos) {
            return decode::HeadClipBf16{in, width == 80U ? nullptr : gain.data, out,
                                      pos, heads, width, 0, 128, epsilon, 0.75F, 0.17F};
        };
        const auto scalar = [&](unsigned r) {
            launch(decode::bind(args(x.data + r * stride, y.scalar.data + r * stride,
                                     &test.device.data->row[r].position)));
        };
        Captured captured;
        captured.record(test.bind.bind(args(x.data, y.wide.data, nullptr)));
        captured.replay();
        test.active(scalar);
        y.compare("descriptor head clip");

        auto changed = test.step;
        for (unsigned r = 0; r < test.rows; ++r) changed.row[r].position += 12345U;
        check(cudaMemcpy(test.device.data, &changed, sizeof(changed), cudaMemcpyHostToDevice));
        captured.replay();
        test.active(scalar);
        y.compare("descriptor head clip replay");
        check(cudaMemcpy(test.device.data, &test.step, sizeof(test.step), cudaMemcpyHostToDevice));
    }
}

struct ScalarEmit {
    RowOut* output;
    __device__ void operator()(unsigned, const head::Pick2& p) const {
        *output = {p.first.nonfinite ? head::kNone : p.first.index, p.first.value, p.value2, RowStatus::picked};
    }
};

void greedy(const Case& test) {
    constexpr unsigned vocabulary = 4100;
    constexpr unsigned chunks = head::default_chunks(vocabulary);
    auto logits = input(test.rows * vocabulary, 22);
    for (unsigned r = 0; r < test.rows; ++r) {
        logits[r * vocabulary + 35U] = logits[r * vocabulary + 3701U] = 17.0F + static_cast<float>(r);
        if (r % 4U == 2U) logits[r * vocabulary + 51U] = std::numeric_limits<float>::infinity();
    }
    const Device<float> x(logits);
    const head::Pick2 sentinel{{canary, 15U, 777U}, canary, 27U};
    Device<head::Pick2> part((test.rows + 1U) * chunks, sentinel), ref_part((test.rows + 1U) * chunks, sentinel);
    const RowOut initial{19, canary, canary, RowStatus::stop};
    Device<RowOut> y(test.rows + 1U, initial), ref(test.rows + 1U, initial);
    Device<std::uint32_t> legacy(test.rows, 0U);
    Device<head::Pick> legacy_part(test.rows * chunks, head::Pick{});
    const wr1::Greedy args{x.data, part.data, y.data, vocabulary, 0};
    launch(test.bind.bind_partial(args));
    launch(test.bind.bind(args));
    test.active([&](unsigned r) {
        const auto* logits_row = x.data + r * vocabulary;
        const auto view = argmax::View{logits_row, 0U, vocabulary, 1, 0, chunks, nullptr};
        check(argmax::pick2_rows(view, ref_part.data + r * chunks, ScalarEmit{ref.data + r}, nullptr));
        const decode::GreedyPartial scalar{logits_row, legacy_part.data + r * chunks, vocabulary, chunks};
        launch(decode::bind(scalar));
        launch(decode::bind(decode::GreedyFinal{scalar, legacy.data + r}));
    });
    equal<head::Pick2>(part.read(), ref_part.read(), "greedy partial/masked canary");
    const auto ids = legacy.read();
    auto expected = ref.read();
    for (unsigned r = 0; r < test.rows; ++r) {
        if (!step_row_valid(test.step, r)) expected[r].status = RowStatus::masked;
        else if (expected[r].id != ids[r]) throw std::runtime_error("greedy scalar reference ID");
    }
    equal<RowOut>(y.read(), expected, "greedy result/masked status");
}

}  // namespace

int main() try {
    strix::cuda_core::set_programmatic_launch(false);
    for (unsigned width : wr1::widths) {
        for (bool padded : {false, true}) {
            const Case test(width, padded);
            dense(test);
            bank_and_mix(test);
            controller(test);
            relative(test);
            clip(test);
            greedy(test);
        }
        std::printf("wr1_cuda PASS R=%u dense controller residual_read message_bank relative_logits head_clip greedy_head padding\n",
                    width);
    }
    std::puts("wr1_cuda PASS widths=1,2,4,8,16 bitwise");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "wr1_cuda FAIL %s\n", error.what());
    return 1;
}
