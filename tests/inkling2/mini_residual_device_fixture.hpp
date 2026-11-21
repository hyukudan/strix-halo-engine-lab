#pragma once

#include "core/residual_mix.hpp"
#include "core/residual_reference.hpp"
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_mini_controller_device.hpp"
#include "inkling2/inkling2_mini_schedule.hpp"
#if defined(__HIPCC__)
#include "inkling2/inkling2_mini_dense_hip.hpp"
#else
#include "inkling2/inkling2_mini_dense_cuda.hpp"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace strix::tests::mini_residual {
namespace {

namespace device = core::residual_mix;
namespace ref = core::residual_reference;
namespace gi = attn::group_indexer;

#if defined(__HIPCC__)
device::Error allocate_managed(void** ptr, std::size_t bytes) {
    return hipMallocManaged(ptr, bytes, hipMemAttachGlobal);
}
constexpr auto free_device = hipFree;
constexpr auto create_stream = hipStreamCreate;
constexpr auto sync_stream = hipStreamSynchronize;
constexpr auto destroy_stream = hipStreamDestroy;
constexpr auto error_string = hipGetErrorString;
#else
device::Error allocate_managed(void** ptr, std::size_t bytes) {
    return cudaMallocManaged(ptr, bytes, cudaMemAttachGlobal);
}
constexpr auto free_device = cudaFree;
constexpr auto create_stream = cudaStreamCreate;
constexpr auto sync_stream = cudaStreamSynchronize;
constexpr auto destroy_stream = cudaStreamDestroy;
constexpr auto error_string = cudaGetErrorString;
#endif

void require(device::Error error) {
    if (error == device::success) return;
    std::cerr << error_string(error) << '\n';
    std::exit(1);
}

template <class T>
struct ManagedBuffer {
    T* data;
    explicit ManagedBuffer(std::size_t count) : data(nullptr) {
        void* memory = nullptr;
        require(allocate_managed(&memory, count * sizeof(T)));
        data = static_cast<T*>(memory);
    }
    ~ManagedBuffer() { require(free_device(data)); }
    ManagedBuffer(const ManagedBuffer&) = delete;
    ManagedBuffer& operator=(const ManagedBuffer&) = delete;
};
using ManagedRows = ManagedBuffer<float>;

constexpr unsigned kRows = 3;
constexpr unsigned kWidth = 513;
constexpr float kEpsilon = 1e-6F;

bool check_mix(unsigned bank_count, unsigned query_count, device::Stream stream) {
    const std::size_t image = kRows * kWidth;
    ManagedRows raw(bank_count * image), normalized(bank_count * image);
    ManagedRows query_values(query_count * kWidth), output_raw(query_count * image), output_ready(query_count * image);
    for (std::size_t i = 0; i < bank_count * image; ++i)
        raw.data[i] = static_cast<float>(static_cast<int>((i * 11U + i / kWidth) % 31U) - 15) * 0.125F;
    for (std::size_t i = 0; i < query_count * kWidth; ++i)
        query_values.data[i] = static_cast<float>(static_cast<int>((i * 7U + i / kWidth) % 17U) - 8) * 0.0625F;

    device::MixArgs args{};
    args.queries = query_values.data;
    args.rows = kRows;
    args.width = kWidth;
    args.bank_count = bank_count;
    args.query_count = query_count;
    args.epsilon = kEpsilon;
    for (unsigned bank = 0; bank < bank_count; ++bank) args.banks[bank] = raw.data + bank * image;
    for (unsigned query = 0; query < query_count; ++query) args.output[query] = output_raw.data + query * image;
    auto aliased = args;
    aliased.output[0] = raw.data;
    if (device::launch_mix(aliased, stream) != device::invalid_value) return false;
    std::array<device::ConstRows, device::kMaxBanks> banks{};
    std::array<device::MutableRows, device::kMaxBanks> captured{};
    std::array<device::MutableRows, device::kMaxQueries> outputs{};
    for (unsigned bank = 0; bank < bank_count; ++bank) {
        banks[bank] = {args.banks[bank], image};
        captured[bank] = {normalized.data + bank * image, image};
    }
    for (unsigned query = 0; query < query_count; ++query) outputs[query] = {args.output[query], image};
    require(device::launch_residual_read(std::span{banks}.first(bank_count - 1U), banks[bank_count - 1U],
                                        {query_values.data, query_count * kWidth},
                                        std::span{outputs}.first(query_count), kWidth, kRows, kEpsilon, stream));
    require(device::launch_capture_message_bank(std::span{banks}.first(bank_count),
                                               std::span{captured}.first(bank_count), kWidth, kRows, kEpsilon, stream));

    args.normalized = true;
    for (unsigned bank = 0; bank < bank_count; ++bank) args.banks[bank] = captured[bank].data();
    for (unsigned query = 0; query < query_count; ++query) args.output[query] = output_ready.data + query * image;
    require(device::launch_mix(args, stream));
    require(sync_stream(stream));

    std::vector<float> bank_rows(bank_count * kWidth);
    for (unsigned row = 0; row < kRows; ++row) {
        for (unsigned bank = 0; bank < bank_count; ++bank) {
            const auto* source = raw.data + bank * image + row * kWidth;
            std::copy_n(source, kWidth, bank_rows.data() + bank * kWidth);
        }
        const auto bank = ref::normalize_bank(bank_rows, kWidth, kEpsilon);
        const auto expected = ref::mix(bank, {query_values.data, query_count * kWidth}, kWidth);
        for (unsigned query = 0; query < query_count; ++query) {
            for (unsigned i = 0; i < kWidth; ++i) {
                const auto index = query * image + row * kWidth + i;
                if (std::abs(output_raw.data[index] - expected[query * kWidth + i]) > 3e-5F ||
                    output_raw.data[index] != output_ready.data[index]) return false;
            }
        }
    }
    return true;
}

bool check_mix_schedules(unsigned rows, unsigned bank_count, unsigned query_count, bool exceptional,
                         device::Stream stream) {
    constexpr unsigned width = inkling2::kHiddenWidth;
    const auto image = std::size_t{rows} * width;
    const auto output_count = query_count * image;
    ManagedRows bank_values(bank_count * image), queries(query_count * width);
    ManagedRows serial(output_count), parallel(output_count), dispatched(output_count);
    for (std::size_t i = 0; i < bank_count * image; ++i) {
        bank_values.data[i] = static_cast<float>(static_cast<int>((i * 11U + i / width) % 31U) - 15) * 0.137F;
    }
    for (std::size_t i = 0; i < query_count * width; ++i) {
        queries.data[i] = static_cast<float>(static_cast<int>((i * 7U + i / width) % 17U) - 8) * 0.073F;
    }
    if (exceptional) {
        bank_values.data[0] = std::numeric_limits<float>::quiet_NaN();
        bank_values.data[image + width + 128U] = std::numeric_limits<float>::infinity();
    }

    device::MixArgs args{};
    args.queries = queries.data;
    args.rows = rows;
    args.width = width;
    args.bank_count = bank_count;
    args.query_count = query_count;
    args.epsilon = kEpsilon;
    for (unsigned bank = 0; bank < bank_count; ++bank) args.banks[bank] = bank_values.data + bank * image;
    for (const bool normalized : {false, true}) {
        args.normalized = normalized;
        std::fill_n(serial.data, output_count, -123.0F);
        std::fill_n(parallel.data, output_count, -124.0F);
        std::fill_n(dispatched.data, output_count, -125.0F);
        for (unsigned query = 0; query < query_count; ++query) args.output[query] = serial.data + query * image;
        require(device::detail::launch_mix_schedule(args, stream));
        for (unsigned query = 0; query < query_count; ++query) args.output[query] = parallel.data + query * image;
        require(device::detail::launch_mix_schedule<true>(args, stream));
        for (unsigned query = 0; query < query_count; ++query) args.output[query] = dispatched.data + query * image;
        require(device::launch_mix(args, stream));
        require(sync_stream(stream));
        if (std::memcmp(serial.data, parallel.data, output_count * sizeof(float)) != 0 ||
            std::memcmp(serial.data, dispatched.data, output_count * sizeof(float)) != 0) {
            std::cerr << "Mix schedule mismatch for rows=" << rows << " banks=" << bank_count
                      << " queries=" << query_count << " normalized=" << normalized
                      << " exceptional=" << exceptional << '\n';
            return false;
        }
    }
    return true;
}

bool check_mix_schedules(device::Stream stream) {
    constexpr unsigned max_banks = inkling2::kPhysicalLayers / inkling2::kResidualBlockLayers + 1U;
    for (const unsigned rows : {1U, 2U, 17U, 300U}) {
        for (const unsigned banks : {1U, 2U, max_banks}) {
            for (const unsigned queries : {1U, 3U}) {
                if (!check_mix_schedules(rows, banks, queries, false, stream)) return false;
            }
        }
    }
    return check_mix_schedules(2U, max_banks, 3U, true, stream);
}

bool check_pointwise(device::Stream stream) {
    constexpr std::size_t input_stride = kWidth + 7U;
    constexpr std::size_t output_stride = 2U * kWidth;
    constexpr std::size_t n = kRows * kWidth;
    ManagedRows original(kRows * input_stride), strided(kRows * output_stride);
    ManagedRows lhs(n), rhs(n), forget(n), gain(n), output(n);
    std::fill_n(strided.data, kRows * output_stride, -123.0F);
    for (unsigned row = 0; row < kRows; ++row) {
        for (unsigned i = 0; i < kWidth; ++i) {
            original.data[row * input_stride + i] = i % 2U == 0U ? 1.0F : -1.0F;
        }
    }
    require(device::launch_normalize({original.data, strided.data + kWidth, kRows, kWidth, kEpsilon,
                                      input_stride, output_stride, true}, stream));

    for (std::size_t i = 0; i < n; ++i) {
        lhs.data[i] = static_cast<float>(static_cast<int>(i % 17U) - 8) * 0.03125F;
        rhs.data[i] = 0.00454321F;
        forget.data[i] = 0.0037F * static_cast<float>(i % 13U);
        gain.data[i] = -0.0079F * static_cast<float>(i % 21U);
    }
    require(device::launch_controller({lhs.data, rhs.data, forget.data, gain.data, output.data, n}, stream));
    require(sync_stream(stream));
    for (unsigned row = 0; row < kRows; ++row) {
        for (unsigned i = 0; i < kWidth; ++i) {
            if (strided.data[row * output_stride + i] != -123.0F ||
                strided.data[row * output_stride + kWidth + i] != (i % 2U == 0U ? 1.0F : -1.0F))
                return false;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = (1.0F - std::tanh(forget.data[i])) * lhs.data[i] +
                               std::tanh(gain.data[i]) * rhs.data[i];
        if (!(std::abs(output.data[i] - expected) <= 1e-6F)) return false;
    }

    require(device::launch_add({lhs.data, rhs.data, output.data, n, false}, stream));
    require(device::launch_round({output.data, output.data, n}, stream));
    require(sync_stream(stream));
    for (std::size_t i = 0; i < n; ++i)
        if (output.data[i] != gi::bf16_round(lhs.data[i] + rhs.data[i])) return false;
    require(device::launch_add({lhs.data, rhs.data, output.data, n, true}, stream));
    require(sync_stream(stream));
    for (std::size_t i = 0; i < n; ++i)
        if (output.data[i] != gi::bf16_round(lhs.data[i] + rhs.data[i])) return false;
    return true;
}

struct ControllerOperations {
    device::Stream queue;
#if !defined(__HIPCC__)
    std::unique_ptr<cuda_core::BigCtx, decltype(&cuda_core::dense_big_ctx_destroy)> context{
        cuda_core::dense_big_ctx_create(), cuda_core::dense_big_ctx_destroy};
#endif

    device::Stream stream() const { return queue; }
    void check(device::Error status) const { require(status); }
    void dense(const float* input, unsigned width, unsigned rows, std::span<const core::Bf16RowProjection> projections) {
#if defined(__HIPCC__)
        require(inkling2::enqueue_mini_hip_dense_rows(input, width, rows, projections, queue));
#else
        require(inkling2::enqueue_mini_cuda_dense_rows(input, width, rows, projections, queue,
                                                      cuda_core::DenseArith::mma3, context.get()));
#endif
    }
    void rms(const float* input, const std::uint16_t* scale, float* output, unsigned rows) const {
#if defined(__HIPCC__)
        constexpr auto width = inkling2::kHiddenWidth;
        require(inkling2::enqueue_mini_hip_rms({input, scale, output, rows, width, width, width}, queue));
#else
        require(inkling2::enqueue_mini_cuda_rmsnorm(input, scale, output, rows, queue));
#endif
    }
};

bool check_controller(unsigned rows, ControllerOperations& operations) {
    constexpr unsigned width = inkling2::kHiddenWidth;
    constexpr unsigned rank = inkling2::kControllerRank;
    constexpr unsigned stride = 2U * width;
    constexpr auto plan = inkling2::kMiniController;
    const auto elements = std::size_t{rows} * width;
    const auto latent_elements = std::size_t{rows} * rank;
    ManagedRows raw(elements), first(elements), second(elements), query(width);
    ManagedRows scratch(plan.scratch_elements(rows)), actual(elements), expected(elements);
    ManagedRows message(elements), operands(2U * elements), latent(latent_elements);
    ManagedRows forget(elements), gain(elements), conditioned(elements);
    ManagedBuffer<std::uint16_t> down_weights(width * rank * 2U), forget_weights(rank * width),
        gain_weights(rank * width), norm_weights(width);
    const auto pattern = [](std::size_t i, unsigned seed) {
        return static_cast<float>(static_cast<int>((i * seed + i / 13U) % 37U) - 18) * 0.00391F;
    };
    const auto fill_weights = [&](std::uint16_t* weights, std::size_t count, unsigned seed) {
        for (std::size_t i = 0; i < count; ++i)
            weights[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(gi::bf16_round(pattern(i, seed))) >> 16U);
    };
    fill_weights(down_weights.data, width * rank * 2U, 3U);
    fill_weights(forget_weights.data, rank * width, 5U);
    fill_weights(gain_weights.data, rank * width, 7U);
    fill_weights(norm_weights.data, width, 11U);
    for (std::size_t i = 0; i < elements; ++i) {
        raw.data[i] = pattern(i, 17U);
        first.data[i] = pattern(i, 19U);
        second.data[i] = pattern(i, 23U);
    }
    for (unsigned i = 0; i < width; ++i) query.data[i] = pattern(i, 29U);

    const device::ConstRows bank[] = {{first.data, elements}, {second.data, elements}};
    const device::ControllerWeights weights{
        {query.data, width}, down_weights.data, forget_weights.data, gain_weights.data, norm_weights.data,
    };
    device::enqueue_low_rank_controller(plan, weights, rows, {raw.data, elements}, bank, {actual.data, elements},
                                        {scratch.data, plan.scratch_elements(rows)}, operations);

    // Original launch sequence, with the same backend's unrounded projection.
    const auto stream = operations.stream();
    require(device::launch_mix(
        {{first.data, second.data}, query.data, {message.data}, rows, width, 2U, 1U, kEpsilon, true}, stream));
    require(device::launch_normalize({raw.data, operands.data, rows, width, kEpsilon, width, stride, true}, stream));
    require(device::launch_normalize(
        {message.data, operands.data + width, rows, width, kEpsilon, width, stride, true}, stream));
    const core::Bf16RowProjection down[] = {{weights.down, rank, latent.data}};
    operations.dense(operands.data, stride, rows, down);
    require(device::launch_round({latent.data, latent.data, latent_elements}, stream));
    const core::Bf16RowProjection gates[] = {{weights.forget, width, forget.data}, {weights.gain, width, gain.data}};
    operations.dense(latent.data, rank, rows, gates);
    require(device::launch_controller(
        {raw.data, message.data, forget.data, gain.data, conditioned.data, elements}, stream));
    operations.rms(conditioned.data, weights.norm, expected.data, rows);
    require(sync_stream(stream));

    const auto same = [](const float* a, const float* b, std::size_t count) {
        return std::memcmp(a, b, count * sizeof(float)) == 0;
    };
    return same(actual.data, expected.data, elements) && same(scratch.data, message.data, elements) &&
           same(scratch.data + elements, forget.data, elements) && same(scratch.data + 2U * elements, gain.data, elements) &&
           same(scratch.data + 3U * elements, conditioned.data, elements) &&
           same(scratch.data + 4U * elements, latent.data, latent_elements);
}

}  // namespace

inline int run() {
    device::Stream stream{};
    require(create_stream(&stream));
    ControllerOperations operations{stream};
#if !defined(__HIPCC__)
    if (!operations.context) return 1;
#endif
    const bool ok = check_mix(1, 1, stream) && check_mix(2, 1, stream) && check_mix(5, 3, stream) &&
                    check_mix_schedules(stream) && check_pointwise(stream) && check_controller(1, operations) &&
                    check_controller(3, operations) && check_controller(33, operations);
    require(destroy_stream(stream));
    if (!ok) std::cerr << "Mini residual device mismatch\n";
    return ok ? 0 : 1;
}

}  // namespace strix::tests::mini_residual
