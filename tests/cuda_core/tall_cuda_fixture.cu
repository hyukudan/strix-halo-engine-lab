// Run the incumbent with --write and the family with --compare on the same node.
#include "strix/cuda_core/dense_big.hpp"
#include "strix/cuda_core/nvfp4.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string_view>
#include <vector>

namespace cc = strix::cuda_core;
namespace ft = strix::cuda::core::nvfp4::tile;

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "FAIL K2c CUDA tall: %s\n", message);
    std::exit(1);
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL K2c CUDA tall: %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

#define CK(operation) check((operation), #operation)

template <class T>
class DeviceBuffer {
    T* data_ = nullptr;

public:
    explicit DeviceBuffer(std::size_t count) { CK(cudaMalloc(&data_, count * sizeof(T))); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { CK(cudaFree(data_)); }
    T* data() const { return data_; }

    void upload(const std::vector<T>& source) {
        CK(cudaMemcpy(data_, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

// Use integer-generated finite operands to keep the same bits across tools.
float sample(std::size_t index, unsigned seed) {
    std::uint32_t x = static_cast<std::uint32_t>(index) + seed;
    x = (x ^ (x >> 16)) * 0x7feb352dU;
    x = (x ^ (x >> 15)) * 0x846ca68bU;
    x ^= x >> 16;
    return static_cast<float>(static_cast<int>(x & 65535U) - 32768) * 0x1p-15F;
}

enum class Form { bf16, q8, f16, fp8, fp4, fp4_glu, fp4_glu_bf16, bf16_compact };

struct Case {
    const char* name;
    Form form;
    std::uint32_t rows, in, out;
    int terms = 3;
    cc::BigEpi epi = cc::BigEpi::none;
};

constexpr Case cases[] = {
    {"Qwen27B BF16 gate", Form::bf16, 129, 5120, 17408},
    {"Qwen27B BF16 down residual", Form::bf16, 65, 17408, 5120, 2, cc::BigEpi::residual},
    {"Qwen27B Q8 gate", Form::q8, 65, 5120, 17408},
    {"Qwen27B Q8 down SiLU", Form::q8, 65, 17408, 5120, 2, cc::BigEpi::silu_mul},
    {"Qwen27B FP16 NVFP4 gate", Form::f16, 129, 5120, 17408},
    {"Qwen27B FP8 NVFP4 down residual", Form::fp8, 129, 17408, 5120, 1, cc::BigEpi::residual},
    {"Qwen27B FP4 NVFP4 down residual", Form::fp4, 257, 17408, 5120, 1, cc::BigEpi::residual},
    {"Qwen27B FP4 NVFP4 gate-up LSU", Form::fp4_glu, 65, 5120, 17408},
    {"Qwen27B FP4 NVFP4 gate-up bulk BF16", Form::fp4_glu_bf16, 65, 5120, 17408},
    {"Inkling BF16 Q", Form::bf16, 129, 2304, 3072},
    {"Inkling BF16 K rounded", Form::bf16, 65, 2304, 1024, 3, cc::BigEpi::round_bf16},
    {"BF16 compact output tail", Form::bf16_compact, 65, 5120, 16, 3, cc::BigEpi::round_bf16},
};

cc::DenseWeight weight(const Case& c, cc::WeightFormat format, unsigned seed, DeviceBuffer<std::uint8_t>& packed) {
    cc::DenseWeight result{packed.data(), format, c.out, c.in};
    const auto elements = static_cast<std::size_t>(c.out) * c.in;
    if (format == cc::WeightFormat::q8_0) {
        std::vector<std::uint8_t> bytes((elements / 32) * 34);
        for (std::size_t b = 0; b < elements / 32; ++b) {
            bytes[b * 34] = static_cast<std::uint8_t>(b & 255U);
            bytes[b * 34 + 1] = 0x24;
            for (std::size_t k = 0; k < 32; ++k)
                bytes[b * 34 + 2 + k] = static_cast<std::uint8_t>(static_cast<int>(sample(b * 32 + k, seed) * 127.0F));
        }
        packed.upload(bytes);
        return result;
    }

    std::vector<std::uint16_t> bf16(elements);
    for (std::size_t i = 0; i < elements; ++i)
        bf16[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(sample(i, seed) * 0x1p-5F) >> 16);
    if (format == cc::WeightFormat::bf16) {
        CK(cudaMemcpy(packed.data(), bf16.data(), elements * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    } else {
        DeviceBuffer<std::uint16_t> source(elements);
        source.upload(bf16);
        CK(cc::nvfp4_quantize_bf16(source.data(), c.out, c.in, packed.data(), &result.gscale, nullptr));
        CK(cudaDeviceSynchronize());
    }
    return result;
}

template <class T>
void compare_output(const Case& c, const T* device, std::fstream& file, bool write) {
    const auto count = static_cast<std::size_t>(c.rows) * c.out;
    const auto bytes = static_cast<std::streamsize>(count * sizeof(T));
    std::vector<T> output(count);
    CK(cudaMemcpy(output.data(), device, static_cast<std::size_t>(bytes), cudaMemcpyDeviceToHost));
    bool nonzero = false;
    for (T value : output) {
        float widened;
        if constexpr (sizeof(T) == 2) widened = std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
        else widened = value;
        if (!std::isfinite(widened)) fail(c.name);
        nonzero = nonzero || widened != 0.0F;
    }
    if (!nonzero) fail(c.name);
    const std::string_view name(c.name);
    const auto name_length = static_cast<std::streamsize>(name.size());
    if (write) {
        file.write(name.data(), name_length);
        file.write(reinterpret_cast<const char*>(output.data()), bytes);
        if (!file) fail("writing incumbent outputs");
        return;
    }

    std::vector<char> label(name.size());
    std::vector<T> baseline(count);
    file.read(label.data(), name_length);
    file.read(reinterpret_cast<char*>(baseline.data()), bytes);
    if (!file || std::string_view(label.data(), label.size()) != name) fail("invalid incumbent data or case order");
    for (std::size_t i = 0; i < count; ++i) {
        if (std::memcmp(&output[i], &baseline[i], sizeof(T)) != 0) {
            std::fprintf(stderr, "FAIL K2c CUDA tall: %s at row %zu, output %zu\n", c.name, i / c.out, i % c.out);
            std::exit(1);
        }
    }
    std::printf("PASS K2c %s bitwise\n", c.name);
}

void run(const Case& c, std::fstream& file, bool write) {
    const auto elements = static_cast<std::size_t>(c.out) * c.in;
    const auto inputs = static_cast<std::size_t>(c.rows) * c.in;
    const auto outputs = static_cast<std::size_t>(c.rows) * c.out;
    const bool fp4 = c.form == Form::fp4 || c.form == Form::fp4_glu || c.form == Form::fp4_glu_bf16;
    const bool glu = c.form == Form::fp4_glu || c.form == Form::fp4_glu_bf16;
    const auto format = c.form == Form::q8 ? cc::WeightFormat::q8_0
        : c.form == Form::bf16 || c.form == Form::bf16_compact ? cc::WeightFormat::bf16 : cc::WeightFormat::nvfp4;
    const auto packed_bytes = format == cc::WeightFormat::bf16 ? elements * 2
        : format == cc::WeightFormat::q8_0 ? elements / 32 * 34 : cc::nvfp4_bytes(c.out, c.in);
    DeviceBuffer<std::uint8_t> packed(packed_bytes);
    const auto w = weight(c, format, 133, packed);
    DeviceBuffer<float> x(inputs);
    std::vector<float> hx(inputs);
    for (std::size_t i = 0; i < inputs; ++i) hx[i] = sample(i, 29) * (i % 97 == 0 ? 8.0F : 1.0F);
    x.upload(hx);
    std::unique_ptr<cc::BigCtx, decltype(&cc::dense_big_ctx_destroy)> context(cc::dense_big_ctx_create(), cc::dense_big_ctx_destroy);
    if (!context) fail("creating prefill context");
    if (c.form == Form::fp4_glu_bf16) cc::dense_big_ctx_set_tile_feed(context.get(), ft::Feed::bulk);

    cc::BigAct act{};
    if (fp4) {
        CK(cc::dense_big_split4(x.data(), c.rows, c.in, &act, nullptr, context.get()));
    } else {
        CK(cc::dense_big_split(x.data(), c.rows, c.in, format, c.terms, &act, nullptr, context.get(), c.form == Form::fp8));
    }
    DeviceBuffer<float> y(outputs);
    CK(cudaMemset(y.data(), 0xff, outputs * sizeof(float)));
    if (glu) {
        DeviceBuffer<std::uint8_t> up_packed(packed_bytes);
        const auto up = weight(c, format, 881, up_packed);
        if (c.form == Form::fp4_glu_bf16) {
            DeviceBuffer<std::uint16_t> y_bf16(outputs);
            CK(cudaMemset(y_bf16.data(), 0xff, outputs * sizeof(std::uint16_t)));
            CK(cc::dense_big_glu4(w, up, act, y_bf16.data(), nullptr, context.get()));
            compare_output(c, y_bf16.data(), file, write);
        } else {
            CK(cc::dense_big_glu4(w, up, act, y.data(), nullptr, context.get()));
            compare_output(c, y.data(), file, write);
        }
        return;
    }
    if (c.form == Form::bf16_compact) {
        CK(cc::dense_big_bf16_compact(w, act, y.data(), c.epi, nullptr, context.get()));
    } else if (c.epi == cc::BigEpi::residual || c.epi == cc::BigEpi::silu_mul) {
        std::vector<float> aux(outputs);
        for (std::size_t i = 0; i < outputs; ++i) aux[i] = sample(i, 397);
        y.upload(aux);
        CK(cc::dense_big_gemm(w, act, y.data(), y.data(), c.epi, nullptr, context.get()));
    } else {
        CK(cc::dense_big_gemm(w, act, y.data(), nullptr, c.epi, nullptr, context.get()));
    }
    compare_output(c, y.data(), file, write);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) fail("usage: fixture --write|--compare INCUMBENT.bin");
    const std::string_view mode(argv[1]);
    if (mode != "--write" && mode != "--compare") fail("expected --write or --compare");
    const bool write = mode == "--write";
    std::fstream file(argv[2], std::ios::binary | (write ? (std::ios::out | std::ios::trunc) : std::ios::in));
    if (!file) fail("opening incumbent output file");
    if (!cc::dense_big_fast8_supported() || !cc::dense_big_fast4_supported()) fail("sm_120a FAST8 and FAST4 support required");
    for (const Case& c : cases) run(c, file, write);
    if (write) {
        file.close();
        if (!file) fail("closing incumbent output file");
        std::puts("PASS K2c CUDA tall incumbent capture");
    } else {
        if (file.peek() != std::char_traits<char>::eof() || file.bad()) fail("invalid incumbent output file end");
        std::puts("PASS K2c CUDA tall fixture (12 cases)");
    }
}
