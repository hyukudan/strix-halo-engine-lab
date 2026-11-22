#include "lookup_cases.hpp"

#include "cuda/models/minimax/mm_rtx_kernels.cuh"
#include "cuda/weight/arch/sm120_sources.cuh"
#include "cuda_core/common.cuh"
#include "strix/cuda/qwen_hc.hpp"
#include "strix/cuda/qwen_plemtp.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/nvfp4.hpp"
#include "strix/ds41_dspark.hpp"
#include "strix/ds41_layer.hpp"

#include "k10c_lookup_incumbents.cuh"

#include <algorithm>
#include <cstdio>
#include <string>

namespace {

using namespace lookup_test;
namespace cc = strix::cuda_core;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check(cudaMalloc(data_.put(), count_ * sizeof(T)), "cudaMalloc");
    }

    T* data() const { return data_.get(); }

    void upload(const T* source) const {
        check(cudaMemcpy(data(), source, count_ * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }

    std::vector<T> read() const {
        std::vector<T> result(count_);
        check(cudaMemcpy(result.data(), data(), count_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return result;
    }

private:
    cc::UniqueHandle<T*, cudaFree> data_;
    std::size_t count_;
};

template <class T>
void mask_rows(std::vector<T>& output, std::size_t row_size) {
    for (std::uint32_t row = 0; row < kRows; ++row)
        if (!active(row)) std::fill_n(output.data() + row * row_size, row_size, sentinel<T>());
}

template <class T, class W, class Launch>
void check_incumbent(Form form, std::uint32_t width, std::uint32_t copies, const std::vector<W>& weights, Launch launch,
                     const char* label, const std::array<std::uint32_t, kRows>& tokens = kIds,
                     bool carry = false, bool programmatic = false) {
    constexpr std::uint32_t initial_status = 0x40U;
    const auto row_size = std::size_t{width} * copies;
    const auto count = row_size * kRows;
    const auto carry_count = std::size_t{copies} * kRows;
    // Prefix and suffix guards catch stray writes even on the unmasked launch.
    const std::vector<T> blank(count + 2U, sentinel<T>());
    const std::vector<float> blank_carry(carry_count, sentinel<float>());
    DeviceBuffer<W> table(weights.size());
    DeviceBuffer<std::uint32_t> ids(kRows), status(1U);
    DeviceBuffer<exec::StepBlock> step(1U);
    DeviceBuffer<T> output(blank.size());
    DeviceBuffer<float> pre_carry(carry_count);
    T* const out = output.data() + 1U;
    auto* const carry_out = carry ? pre_carry.data() : nullptr;
    auto* const status_out = form == Form::q8_0_to_bf16_row_zero ? status.data() : nullptr;
    table.upload(weights.data());
    ids.upload(tokens.data());
    const auto descriptor = make_step(tokens);
    step.upload(&descriptor);

    const auto reset = [&] {
        output.upload(blank.data());
        pre_carry.upload(blank_carry.data());
        status.upload(&initial_status);
    };
    const auto read = [&] {
        const auto guarded = output.read();
        require(std::memcmp(guarded.data(), blank.data(), sizeof(T)) == 0 &&
                std::memcmp(&guarded.back(), &blank.back(), sizeof(T)) == 0, "output guard bytes");
        return std::vector<T>(guarded.begin() + 1, guarded.end() - 1);
    };

    reset();
    launch(table.data(), ids.data(), out, carry_out, status_out);
    check(cudaGetLastError(), label);
    auto expected = read();
    auto expected_carry = pre_carry.read();
    const auto expected_status = status.read()[0];
    const bool invalid = status_out && std::any_of(tokens.begin(), tokens.end(), [](std::uint32_t id) {
        return id >= kVocabulary;
    });
    require(expected_status == (initial_status | static_cast<std::uint32_t>(invalid)), "incumbent invalid-ID status");
    const std::vector<std::uint32_t> host_ids(tokens.begin(), tokens.end());
    equal_bytes(expected, reference<T>(form, weights.data(), width, copies, host_ids), label);
    if (carry) {
        std::vector<float> host_carry(carry_count, 0.0F);
        for (std::uint32_t row = 0; row < kRows; ++row) host_carry[std::size_t{row} * copies] = 1.0F;
        equal_bytes(expected_carry, host_carry, "incumbent pre-carry initializer");
    }

    const lookup::View view{{table.data(), kVocabulary, width}, form, out, kRows, copies, carry_out, status_out, programmatic};
    reset();
    check(lookup::enqueue(view, lookup::TokenIds{ids.data()}, nullptr), label);
    equal_bytes(read(), expected, label);
    equal_bytes(pre_carry.read(), expected_carry, "token ID pre-carry");
    require(status.read()[0] == expected_status, "token ID status");

    reset();
    check(lookup::enqueue(view, lookup::StepTokens{step.data()}, nullptr), label);
    mask_rows(expected, row_size);
    mask_rows(expected_carry, copies);
    equal_bytes(read(), expected, label);
    equal_bytes(pre_carry.read(), expected_carry, "X3 pre-carry");
    auto step_status = initial_status;
    if (status_out) {
        for (std::uint32_t row = 0; row < kRows; ++row)
            if (active(row) && tokens[row] >= kVocabulary) step_status |= 1U;
    }
    require(status.read()[0] == step_status, "X3 status respects mask");
}

template <class W>
void dense(Form form, cc::WeightFormat format, std::uint32_t width, const std::vector<W>& weights, const char* label) {
    check_incumbent<float>(form, width, 1U, weights,
        [=](const W* table, const std::uint32_t* ids, float* out, float*, std::uint32_t*) {
            check(cc::embedding_rows({table, format, kVocabulary, width}, ids, out, kRows, nullptr), label);
        }, label, kIds, false, true);
}

void dense_cases() {
    dense(Form::bf16_to_f32, cc::WeightFormat::bf16, 257U, plain16(257U, false), "dense BF16");
    auto f32 = plain16(259U, false);
    std::vector<float> weights;
    for (auto bits : f32) weights.push_back(std::bit_cast<float>((std::uint32_t{bits} << 16U) | (bits & 15U)));
    dense(Form::f32, cc::WeightFormat::f32, 259U, weights, "dense F32");
    dense(Form::q8_0_to_f32, cc::WeightFormat::q8_0, 288U, q8(288U), "dense Q8_0");
}

void qwen_cases() {
    constexpr unsigned width = strix::cuda::qwen_hc::kHidden;
    auto weights = q8(width);
    // A NaN scale in row 31 distinguishes canonical BF16 from raw MTP RNE.
    const auto block = std::size_t{31} * (width / 32U) * 34U;
    weights[block] = 0x45U;
    weights[block + 1U] = 0x7dU;
    check_incumbent<std::uint16_t>(Form::q8_0_to_bf16, width, 1U, weights,
        [](const std::uint8_t* table, const std::uint32_t* ids, std::uint16_t* out, float*, std::uint32_t*) {
            incumbent::qwen_hc::embedding_kernel<<<kRows, 256>>>(table, ids, out);
        }, "Qwen prefill embedding");

    const auto mtp = [](const std::uint8_t* table, const std::uint32_t* ids, std::uint16_t* out, float*, std::uint32_t* status) {
        incumbent::qwen_mtp::mtp_embedding_kernel<<<dim3(width / 256U, kRows), 256>>>(table, kVocabulary, ids, out, status);
    };
    auto tokens = kIds;
    tokens[3] = kVocabulary;
    check_incumbent<std::uint16_t>(Form::q8_0_to_bf16_row_zero, width, 1U, weights, mtp, "Qwen MTP masked invalid ID", tokens);
    tokens[4] = 0xffffffffU;
    check_incumbent<std::uint16_t>(Form::q8_0_to_bf16_row_zero, width, 1U, weights, mtp, "Qwen MTP active invalid ID", tokens);
}

void deepseek_cases() {
    constexpr unsigned width = 515U;
    check_incumbent<float>(Form::f16_payload_to_f32, width, 4U, plain16(width, true),
        [](const std::uint16_t* table, const std::uint32_t* ids, float* out, float* carry, std::uint32_t*) {
            incumbent::ds41::embed_kernel<<<(kRows * width + 255U) / 256U, 256>>>(table, width, ids, kRows, out, carry);
        }, "DeepSeek target embedding", kIds, true);

    constexpr unsigned draft_width = strix::ds41::dspark::hidden;
    check_incumbent<float>(Form::f16_to_f32, draft_width, 4U, plain16(draft_width, true),
        [](const std::uint16_t* table, const std::uint32_t* ids, float* out, float*, std::uint32_t*) {
            incumbent::dspark::embed_block_kernel<<<dim3((draft_width + 255U) / 256U, kRows), 256>>>(table, ids, out);
        }, "DeepSeek drafter embedding");
}

void bf16_cases() {
    constexpr unsigned glm_width = 4096U;
    const auto weights = plain16(glm_width, false);
    for (unsigned copies : {1U, 4U}) {
        check_incumbent<float>(Form::bf16_to_f32, glm_width, copies, weights,
            [=](const std::uint16_t* table, const std::uint32_t* ids, float* out, float*, std::uint32_t*) {
                incumbent::glm::embed<<<kRows * copies * glm_width / 256U, 256>>>(table, ids, out, kRows, copies);
            }, "GLM embedding");
    }
    constexpr unsigned mm_width = strix::minimax::rtxpf::cuda::kHidden;
    check_incumbent<float>(Form::bf16_to_f32, mm_width, 1U, plain16(mm_width, false),
        [](const std::uint16_t* table, const std::uint32_t* ids, float* out, float*, std::uint32_t*) {
            incumbent::minimax::embed_kernel<<<kRows, 256>>>(table, ids, out);
        }, "MiniMax embedding");
}

void nvfp4_case() {
    constexpr unsigned width = 256U;
    const auto tile = tile_data(width);
    const std::vector<float> blank(std::size_t{kRows} * width, sentinel<float>());
    DeviceBuffer<std::uint8_t> table(tile.bytes.size());
    DeviceBuffer<std::uint32_t> ids(kRows);
    DeviceBuffer<exec::StepBlock> step(1U);
    DeviceBuffer<std::uint16_t> expanded(tile.values.size());
    DeviceBuffer<float> output(blank.size());
    table.upload(tile.bytes.data());
    ids.upload(kIds.data());
    const auto descriptor = make_step(kIds);
    step.upload(&descriptor);
    check(cc::nvfp4_expand_f16({table.data(), cc::WeightFormat::nvfp4, kVocabulary, width}, expanded.data(), nullptr),
          "incumbent NVFP4 expand");
    const auto fp16 = expanded.read();
    std::vector<float> row_major(fp16.size());
    for (std::uint32_t row = 0; row < kVocabulary; ++row)
        for (std::uint32_t col = 0; col < width; ++col) {
            // Convert the documented F16 GEMM fragment order back to input order.
            const auto fragment = (col / 32U) * 32U + ((col % 8U) / 2U) * 8U +
                ((col % 32U) / 16U) * 4U + ((col % 16U) / 8U) * 2U + col % 2U;
            row_major[std::size_t{row} * width + col] = lookup_test::half(fp16[std::size_t{row} * width + fragment]) * kTensorScale;
        }
    equal_bytes(row_major, tile.values, "incumbent NVFP4 expansion vs CPU layout oracle");

    const std::vector<std::uint32_t> host_ids(kIds.begin(), kIds.end());
    auto expected = reference<float>(Form::f32, row_major.data(), width, 1U, host_ids);
    const lookup::View view{{table.data(), kVocabulary, width, kTensorScale}, Form::nvfp4_tile_to_f32, output.data(), kRows};
    output.upload(blank.data());
    check(lookup::enqueue(view, lookup::TokenIds{ids.data()}, nullptr), "NVFP4 IDs");
    equal_bytes(output.read(), expected, "NVFP4 IDs vs expansion");
    output.upload(blank.data());
    check(lookup::enqueue(view, lookup::StepTokens{step.data()}, nullptr), "NVFP4 X3");
    mask_rows(expected, width);
    equal_bytes(output.read(), expected, "NVFP4 X3 vs expansion");
}

}  // namespace

int main() {
    try {
        dense_cases();
        std::puts("PASS K10c dense BF16/F32/Q8_0 device IDs and X3");
        qwen_cases();
        std::puts("PASS K10c Qwen Q8_0-to-BF16 and MTP fallback/status");
        deepseek_cases();
        std::puts("PASS K10c DeepSeek F16 broadcast/carry and drafter");
        bf16_cases();
        std::puts("PASS K10c GLM and MiniMax BF16 embedding");
        nvfp4_case();
        std::puts("PASS K10c NVFP4 tile expansion and scale");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL K10c lookup CUDA: %s\n", error.what());
        return 1;
    }
}
