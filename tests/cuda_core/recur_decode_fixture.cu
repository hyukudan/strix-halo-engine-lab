#include "cuda/core/recur_cuda.cuh"
#include "recur_decode_frozen.cuh"
#include "strix/cuda/amd_math_tables.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace {

namespace rec = strix::cuda::recur;
namespace frozen = rec::test::frozen;
namespace am = strix::cuda::amdmath;
using strix::RecurrentStateType;
constexpr auto kShape = strix::ops::recur::shape::gdn_2560_16x48x128;
using L = strix::ops::recur::GdnDecodeLayout<kShape>;

static_assert(L::channels == frozen::kConvWidth && L::value_width == frozen::kValueWidth &&
              L::matrix_elements == frozen::kMatrixFloats && L::history_elements == frozen::kConvStateHalfs);

constexpr float kSentinel = -912.0F;
constexpr std::int8_t kCodeSentinel = -128;

template <class T> struct Result {
    alignas(16) float qkv[frozen::kConvWidth];
    alignas(16) float prepared[frozen::kConvWidth];
    float raw[frozen::kConvWidth], beta[frozen::kValueHeads], gate[frozen::kValueHeads], out[frozen::kValueWidth];
    std::uint16_t history[frozen::kConvStateHalfs];
    T matrix_in[frozen::kMatrixFloats], matrix_out[frozen::kMatrixFloats];
    std::int8_t codes[frozen::kValueWidth];
    float deltas[frozen::kValueWidth / 32U * frozen::kA8Pitch];
    float decay[frozen::kValueHeads];
};

template <class T> struct Data {
    std::uint16_t input[frozen::kHidden];
    float z[frozen::kValueWidth];
    float beta[frozen::kValueHeads * frozen::kHidden], alpha[frozen::kValueHeads * frozen::kHidden];
    float dt[frozen::kValueHeads], a[frozen::kValueHeads], conv[frozen::kConvWidth * 4U], gain[frozen::kHeadDim];
    Result<T> ref, work;
};

struct Random {
    std::uint32_t seed;
    float next(float scale) {
        seed ^= seed << 13U;
        seed ^= seed >> 17U;
        seed ^= seed << 5U;
        return static_cast<float>(static_cast<int>(seed & 0xffffU) - 32768) * (scale * 0x1p-15F);
    }
};

std::uint16_t bf16(float x) {
    const auto bits = std::bit_cast<std::uint32_t>(x);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

template <RecurrentStateType Type, class T>
void initialize(Data<T>& data, std::uint32_t seed, bool a8) {
    Random rng{seed};
    for (auto& v : data.input) v = bf16(rng.next(1.0F));
    for (auto& v : data.z) v = std::bit_cast<float>(std::uint32_t{bf16(rng.next(8.0F))} << 16U);
    for (auto& v : data.beta) v = rng.next(0.125F);
    for (auto& v : data.alpha) v = rng.next(0.125F);
    for (auto& v : data.dt) v = rng.next(1.0F);
    data.dt[0] = -24.0F;
    data.dt[1] = 24.0F;
    for (auto& v : data.a) v = -0.125F - std::abs(rng.next(0.75F));
    for (auto& v : data.conv) v = rng.next(1.0F);
    for (auto& v : data.gain) v = 0.75F + rng.next(0.25F);
    if (a8) {
        std::fill_n(data.gain, 32, 0x1p-125F);
        std::fill_n(data.gain + 32, 32, 0.0F);
        data.gain[64] = std::numeric_limits<float>::infinity();
    }
    for (auto& v : data.ref.qkv) v = rng.next(1.0F);
    data.ref.qkv[0] = 0x1.01p-2F;   // exact BF16 midpoint, even rounds down
    data.ref.qkv[1] = -0.0F;
    for (auto& v : data.ref.history) v = bf16(rng.next(1.0F));
    for (auto& v : data.ref.matrix_in) {
        const float x = rng.next(0.0625F);
        if constexpr (Type == RecurrentStateType::f32) v = x;
        else if constexpr (Type == RecurrentStateType::bf16) v = bf16(x);
        else v = static_cast<__half_raw>(__float2half_rn(x)).x;
    }
    std::fill_n(data.ref.prepared, frozen::kConvWidth, kSentinel);
    std::fill_n(data.ref.raw, frozen::kConvWidth, kSentinel);
    std::fill_n(data.ref.codes, frozen::kValueWidth, kCodeSentinel);
    std::fill_n(data.ref.deltas, L::a8_deltas, kSentinel);
    std::memcpy(&data.work, &data.ref, sizeof(data.ref));
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T, std::size_t N>
void compare(const char* label, const T (&actual)[N], const T (&expected)[N]) {
    if (std::memcmp(actual, expected, sizeof(actual)) == 0) return;
    for (std::size_t i = 0; i < N; ++i) {
        if (std::memcmp(actual + i, expected + i, sizeof(T)) == 0) continue;
        std::fprintf(stderr, "%s[%zu] differs\n", label, i);
        throw std::runtime_error("bitwise mismatch");
    }
}

template <RecurrentStateType Type, class T>
void reference(Data<T>& data, bool a8, const am::Tables& tables) {
    auto& ref = data.ref;
    const frozen::Rows rows{ref.qkv, data.z, ref.beta, ref.gate, ref.out, frozen::kConvWidth + 16U};
    frozen::decode_front_kernel<1><<<frozen::kDecConvBlocks + frozen::kDecGateBlocks, 256>>>(
        data.input, frozen::kHidden, rows, data.beta, data.alpha, data.dt, data.a, data.conv, 1e-6F, tables,
        {ref.history}, a8 ? ref.raw : nullptr);
    check(cudaGetLastError());

    const frozen::DevRef<float> in{reinterpret_cast<float*>(ref.matrix_in)};
    const frozen::DevRef<float> out{reinterpret_cast<float*>(a8 ? ref.matrix_in : ref.matrix_out)};
    constexpr unsigned blocks = frozen::kValueHeads * (frozen::kHeadDim / 32U);
    constexpr unsigned threads = frozen::RecShape<4>::Threads;
    if constexpr (Type == RecurrentStateType::f32) {
        frozen::recurrence_dec_kernel<4><<<blocks, threads>>>(rows, 1U, in, out, tables, nullptr, nullptr, nullptr);
    } else {
        frozen::decay_kernel<<<1, 256>>>(rows, 1U, ref.decay, tables);
        check(cudaGetLastError());
        frozen::recurrence_kernel<4, Type, true><<<blocks, threads>>>(rows, 1U, in, out, ref.decay);
    }
    check(cudaGetLastError());
    if (a8) frozen::norm_kernel<true><<<6, 256>>>(rows, 1U, data.gain, 1e-6F, tables, ref.codes, ref.deltas);
    else frozen::norm_kernel<false><<<6, 256>>>(rows, 1U, data.gain, 1e-6F, tables, nullptr, nullptr);
    check(cudaGetLastError());
}

template <RecurrentStateType Type>
void run(const am::Tables& tables, bool a8) {
    using T = std::conditional_t<Type == RecurrentStateType::f32, float, std::uint16_t>;
    using Storage = Data<T>;
    Storage* allocated = nullptr;
    check(cudaMallocManaged(&allocated, sizeof(Storage)));
    const auto free = [](Storage* p) { (void)cudaFree(p); };
    const std::unique_ptr<Storage, decltype(free)> memory(allocated, free);
    auto& data = *memory;
    // Initialize also the storage that must keep its original bytes.
    std::memset(allocated, 0, sizeof(Storage));
    initialize<Type>(data, a8 ? 0x17edbc4aU : 0x90c673abU, a8);
    reference<Type>(data, a8, tables);

    auto& work = data.work;
    const auto& ref = data.ref;
    const rec::DecodeWeights w{data.beta, data.alpha, data.dt, data.a, data.conv, data.gain};
    const rec::DecodeRow row{data.input, work.qkv, data.z, a8 ? work.qkv : work.prepared,
                             work.beta, work.gate, work.out, a8 ? work.raw : nullptr};
    const rec::DecodeState state{work.history, work.matrix_in, a8 ? work.matrix_in : work.matrix_out, Type};
    const rec::DecodeA8 quant = a8 ? rec::DecodeA8{work.codes, work.deltas} : rec::DecodeA8{};
    check(rec::enqueue_decode(kShape, w, row, state, tables, quant, nullptr));
    check(cudaDeviceSynchronize());

    compare("prepared QKV", a8 ? work.qkv : work.prepared, ref.qkv);
    compare("beta", work.beta, ref.beta);
    compare("log decay", work.gate, ref.gate);
    compare("raw capture", work.raw, ref.raw);
    compare("gated RMS", work.out, ref.out);
    compare("BF16 tap image", work.history, ref.history);
    compare("matrix input image", work.matrix_in, ref.matrix_in);
    compare("matrix output image", work.matrix_out, ref.matrix_out);
    compare("A8 codes", work.codes, ref.codes);
    compare("A8 pitched deltas", work.deltas, ref.deltas);
    if (a8 && (work.deltas[0] != 1.0F || work.deltas[L::a8_pitch] != 1.0F ||
               std::bit_cast<std::uint32_t>(work.deltas[2U * L::a8_pitch]) != 0x7fc00000U ||
               work.deltas[1] != kSentinel))
        throw std::runtime_error("A8 tiny, zero, nonfinite or pitch case did not match");
}

template <RecurrentStateType Type>
void test(const am::Tables& tables) {
    run<Type>(tables, false);
    run<Type>(tables, true);
    std::printf("K9G gfx1151 decode %s EXACT (FP32 output, A8, taps, matrix)\n",
                strix::recurrent_state_type_name(Type));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: strix-k9g-decode-fixture TABLES_DIR\n");
        return 2;
    }
    try {
        auto tables = am::load_tables(argv[1]);
        try {
            test<RecurrentStateType::f32>(tables);
            test<RecurrentStateType::f16>(tables);
            test<RecurrentStateType::bf16>(tables);
        } catch (...) {
            am::free_tables(tables);
            throw;
        }
        am::free_tables(tables);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "K9G gfx1151 decode FAIL: %s\n", error.what());
        return 1;
    }
}
