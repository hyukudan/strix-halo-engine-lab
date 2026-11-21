#pragma once

#include "core/residual_device.hpp"
#include "core/residual_reference.hpp"
#include "inkling2/inkling2_mini_residual_device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace strix::tests::mini_residual {
namespace {

namespace device = core::residual_device;
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

struct ManagedRows {
    float* data;
    explicit ManagedRows(std::size_t count) : data(nullptr) {
        void* memory = nullptr;
        require(allocate_managed(&memory, count * sizeof(float)));
        data = static_cast<float*>(memory);
    }
    ~ManagedRows() { require(free_device(data)); }
    ManagedRows(const ManagedRows&) = delete;
    ManagedRows& operator=(const ManagedRows&) = delete;
};

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
    require(device::launch_mix(args, stream));

    args.normalized = true;
    for (unsigned bank = 0; bank < bank_count; ++bank) {
        auto* ready = normalized.data + bank * image;
        require(device::launch_normalize({args.banks[bank], ready, kRows, kWidth, kEpsilon}, stream));
        args.banks[bank] = ready;
    }
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

}  // namespace

inline int run() {
    device::Stream stream{};
    require(create_stream(&stream));
    const bool ok = check_mix(1, 1, stream) && check_mix(2, 1, stream) && check_mix(5, 3, stream) &&
                    check_pointwise(stream);
    require(destroy_stream(stream));
    if (!ok) std::cerr << "Mini residual device mismatch\n";
    return ok ? 0 : 1;
}

}  // namespace strix::tests::mini_residual
