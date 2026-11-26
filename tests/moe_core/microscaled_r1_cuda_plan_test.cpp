#include "moe_core/microscaled_r1_cuda_plan.hpp"
#include "moe_core/microscaled_r1_pipe_plan.hpp"
#include "moe_core/microscaled_r1.hpp"

#include <array>
#include <cstdint>
#include <iostream>

namespace {

namespace r1 = strix::moe_core::microscaled_r1;

template <class Run, unsigned PairBytes, unsigned ScaleColumns>
constexpr bool coverage(unsigned width) {
    const unsigned full = width / Run::kColumns * Run::kColumns;
    for (unsigned lane = 0; lane < Run::kLanes; ++lane) {
        unsigned next = 2U * lane;
        for (unsigned base = 0; base < full; base += Run::kColumns) {
            for (unsigned stripe = 0; stripe < Run::kPairsPerWord; ++stripe) {
                const unsigned byte = Run::code_lane(stripe, lane) * Run::kWordBytes + Run::code_shift(lane) / 8U;
                const unsigned k = base + 2U * byte / PairBytes;
                const unsigned local_scale = Run::scale_lane(stripe, lane);
                const unsigned word_lane = r1::pipe::ScaleWord::lane(local_scale);
                const unsigned shift = r1::pipe::ScaleWord::shift(local_scale);
                const unsigned scale = base / ScaleColumns + word_lane * r1::pipe::ScaleWord::kBytes + shift / 8U;
                if (k != next || k + 1U >= width || scale != k / ScaleColumns || byte + PairBytes > Run::kRunBytes)
                    return false;
                if (word_lane >= Run::kScales / r1::pipe::ScaleWord::kBytes || shift >= 32U) return false;
                next += Run::kStripeColumns;
            }
        }
        if (next != full + 2U * lane) return false;
        for (unsigned k = full + 2U * lane; k < width; k += Run::kStripeColumns) {
            if (k != next || k + 1U >= width) return false;
            next += Run::kStripeColumns;
        }
        if (next < width) return false;
    }
    return true;
}

constexpr std::uint64_t matrix_bytes(unsigned rows, unsigned width, unsigned values_per_byte, unsigned scale_columns) {
    return std::uint64_t{rows} * (width / values_per_byte + width / scale_columns);
}

} // namespace

int main() {
    using Geometry = strix::device::LaunchGeometry;
    static_assert(r1::gate_up_launch(1152U, 8U, 8U) == Geometry{{72, 64, 1}, {256, 1, 1}, 0});
    static_assert(r1::down_combine_launch(2304U, 8U) == Geometry{{576, 8, 1}, {256, 1, 1}, 0});
    static_assert(r1::pipe::gate_up_launch(1152U) == Geometry{{288, 8, 1}, {128, 1, 1}, 0});
    static_assert(r1::pipe::down_combine_launch(2304U) == Geometry{{1152, 1, 1}, {256, 1, 1}, 0});
    static_assert(strix::moe_core::grouped_rows::row_launch(257, 65536) == Geometry{{2, 65535, 1}, {256, 1, 1}, 0});
    using Fp4 = r1::CudaRun<1U, 16U>;
    using Fp8 = r1::CudaRun<2U, 32U>;
    static_assert(Fp4::kRunBytes == 128U && Fp8::kRunBytes == 128U);
    static_assert(Fp4::kColumns == 256U && Fp8::kColumns == 128U);
    static_assert(r1::pipe::gate_blocks(1152U) == 288U && r1::pipe::down_blocks(2304U) == 1152U);
    static_assert(r1::pipe::gate_blocks(17U) == 5U && r1::pipe::down_blocks(17U) == 9U);
    static_assert(matrix_bytes(16U, 2304U, 2U, 16U) == 20736U);
    static_assert(matrix_bytes(8U, 1152U, 2U, 16U) == 5184U);
    static_assert(matrix_bytes(16U, 2304U, 1U, 32U) == 38016U);
    static_assert(matrix_bytes(8U, 1152U, 1U, 32U) == 9504U);
    for (const unsigned width : std::array{16U, 48U, 64U, 112U, 128U, 240U, 256U, 272U, 1152U, 2304U}) {
        if (!coverage<Fp4, 1U, 16U>(width)) return 1;
    }
    for (const unsigned width : std::array{32U, 96U, 128U, 160U, 224U, 256U, 1152U, 2304U}) {
        if (!coverage<Fp8, 2U, 32U>(width)) return 1;
    }
    std::cout << "K3c R1 CUDA lane map and weight bytes: PASS\n";
}
