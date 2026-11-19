// Raw embedding row gather (launch_hip_embedding_rows / launch_hip_embedding_device_row) on a Q8_0 table with the
// model-B row width (3072) vs the shared host dequantizer: FP32 output bitwise, BF16 output = round-to-nearest-even of
// the host value, device id out of the vocabulary -> row 0, host id out of the vocabulary -> throws, nothing enqueued.
// Exit 77 (skip) without a HIP device.
#include "strix/ggml_dequant.hpp"
#include "strix/hip_embedding_launcher.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what);
}
void hip_ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
std::uint16_t bf16_rne(float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

}  // namespace

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::printf("SKIP no HIP device\n");
        return 77;
    }
    try {
        constexpr std::uint32_t kWidth = 3072, kVocab = 300, kQ8 = 8;
        const auto row_bytes = strix::ggml_dequant::row_bytes(kQ8, kWidth);
        std::vector<std::byte> table(row_bytes * kVocab);
        std::mt19937 rng(11);
        for (std::size_t b = 0; b < table.size(); b += 34) {  // Q8_0 block: fp16 scale + 32 int8
            const std::uint16_t scale = static_cast<std::uint16_t>(0x2000U + (rng() & 0x0FFFU));  // small finite fp16
            std::memcpy(&table[b], &scale, 2);
            for (std::size_t i = 2; i < 34; ++i) table[b + i] = static_cast<std::byte>(rng() & 0xFFU);
        }
        const std::vector<std::uint32_t> ids{0, 7, 299, 7, 120, 1, 298, 64};
        std::vector<float> want(ids.size() * kWidth);
        for (std::size_t t = 0; t < ids.size(); ++t)
            strix::ggml_dequant::dequantize_row(kQ8, std::span<const std::byte>(table).subspan(ids[t] * row_bytes, row_bytes),
                                                std::span<float>(want).subspan(t * kWidth, kWidth));
        void* d_table = nullptr;
        void* d_out = nullptr;
        std::uint32_t* d_id = nullptr;
        constexpr std::uint64_t kPitch = kWidth + 64U;  // pitch > width: the gap must stay untouched
        hip_ok(hipMalloc(&d_table, table.size()), "malloc table");
        hip_ok(hipMalloc(&d_out, ids.size() * kPitch * sizeof(float)), "malloc out");
        hip_ok(hipMalloc(reinterpret_cast<void**>(&d_id), sizeof(std::uint32_t)), "malloc id");
        hip_ok(hipMemcpy(d_table, table.data(), table.size(), hipMemcpyHostToDevice), "copy table");
        const strix::HipEmbeddingTable t{d_table, kQ8, kWidth, kVocab};

        // FP32 rows, host ids, pitched.
        hip_ok(hipMemset(d_out, 0xFF, ids.size() * kPitch * sizeof(float)), "fill");
        strix::launch_hip_embedding_rows(t, strix::HipActivationStorage::fp32, ids, d_out, kPitch, nullptr);
        std::vector<float> got(ids.size() * kPitch);
        hip_ok(hipMemcpy(got.data(), d_out, got.size() * sizeof(float), hipMemcpyDeviceToHost), "read fp32");
        bool rows_ok = true, gap_ok = true;
        for (std::size_t r = 0; r < ids.size(); ++r) {
            rows_ok = rows_ok && std::memcmp(&got[r * kPitch], &want[r * kWidth], kWidth * sizeof(float)) == 0;
            std::uint32_t gap = 0;
            std::memcpy(&gap, &got[r * kPitch + kWidth], 4);
            gap_ok = gap_ok && gap == 0xFFFFFFFFU;
        }
        check(rows_ok, "fp32 rows == host dequantize_row (bitwise)");
        check(gap_ok, "pitch gap untouched");

        // BF16 rows: round to nearest even of the host value.
        strix::launch_hip_embedding_rows(t, strix::HipActivationStorage::bf16, ids, d_out, kWidth, nullptr);
        std::vector<std::uint16_t> half(ids.size() * kWidth);
        hip_ok(hipMemcpy(half.data(), d_out, half.size() * 2U, hipMemcpyDeviceToHost), "read bf16");
        bool bf_ok = true;
        for (std::size_t i = 0; i < half.size(); ++i) bf_ok = bf_ok && half[i] == bf16_rne(want[i]);
        check(bf_ok, "bf16 rows == RNE(host)");

        // Device id: in range, then out of range -> row 0.
        const std::uint32_t words[2] = {299, kVocab + 5U};
        std::vector<float> row0(kWidth), row299(kWidth);
        strix::ggml_dequant::dequantize_row(kQ8, std::span<const std::byte>(table).first(row_bytes), row0);
        strix::ggml_dequant::dequantize_row(kQ8, std::span<const std::byte>(table).subspan(299U * row_bytes, row_bytes), row299);
        for (int w = 0; w < 2; ++w) {
            hip_ok(hipMemcpy(d_id, &words[w], 4, hipMemcpyHostToDevice), "copy id");
            strix::launch_hip_embedding_device_row(t, strix::HipActivationStorage::fp32, d_id, d_out, nullptr);
            hip_ok(hipMemcpy(got.data(), d_out, kWidth * sizeof(float), hipMemcpyDeviceToHost), "read device row");
            check(std::memcmp(got.data(), (w == 0 ? row299 : row0).data(), kWidth * sizeof(float)) == 0,
                  w == 0 ? "device id 299" : "device id out of vocabulary -> row 0");
        }

        // Host id out of the vocabulary: throws before enqueueing.
        bool threw = false;
        try {
            const std::vector<std::uint32_t> bad{1, kVocab};
            strix::launch_hip_embedding_rows(t, strix::HipActivationStorage::fp32, bad, d_out, kWidth, nullptr);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        check(threw, "host id out of vocabulary throws");
        hip_ok(hipDeviceSynchronize(), "sync");
        (void)hipFree(d_table);
        (void)hipFree(d_out);
        (void)hipFree(d_id);
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        ++failures;
    }
    if (failures) std::printf("FAILED %d\n", failures);
    else std::printf("PASS embedding rows (fp32 bitwise, bf16 RNE, device id, bounds)\n");
    return failures ? 1 : 0;
}
