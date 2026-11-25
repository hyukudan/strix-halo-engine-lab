// The CUDA dense family has no decoder for the q4_g64 / q6_g64 resident row storages (strix/weight/host_lowbit.hpp):
// the shared resident rows entry refuses them at every row count, one row included, before any device work, and
// leaves the output untouched. The matrix is valid for the HIP rows family, so only the backend gate refuses it.
#include "core/resident_q8_rows.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

template <class Storage>
void check_refused(strix::weight::FormatId storage, const char* name) {
    constexpr unsigned outputs = 2U, columns = 128U;
    alignas(16) static std::uint8_t weights[outputs * Storage::bytes];
    alignas(16) static float input[strix::core::gemv_rows::max_rows * columns];
    alignas(16) static float output[strix::core::gemv_rows::max_rows * outputs];
    const strix::weight::ResidentMatrixView matrix{{weights, sizeof(weights)}, storage, outputs, columns,
                                                   Storage::row_bytes(columns)};
    for (const unsigned rows : {1U, 2U, strix::core::gemv_rows::max_rows}) {
        check(strix::core::valid_resident_lowbit_rows(matrix, outputs, input, rows, output), "matrix is valid");
        std::memset(output, 0xff, sizeof(output));
        const auto status = strix::core::enqueue_resident_q8_rows(matrix, outputs, input, rows, output, nullptr);
        check(status == strix::core::row_transforms::invalid, "low-bit rows refused on CUDA");
        bool untouched = true;
        for (const float v : output) {
            std::uint32_t b;
            std::memcpy(&b, &v, sizeof(b));
            untouched = untouched && b == 0xffffffffU;
        }
        check(untouched, "output untouched");
    }
    std::cout << name << " refused\n";
}
}  // namespace

int main() {
    check_refused<strix::weight::Q4G64Storage>(strix::weight::kQ4G64WeightStorage, "q4_g64");
    check_refused<strix::weight::Q6G64Storage>(strix::weight::kQ6G64WeightStorage, "q6_g64");
    if (failures) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "cuda resident lowbit refusal tests passed\n";
    return 0;
}
