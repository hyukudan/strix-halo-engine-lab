#include "strix/ds41_fast_profile.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using hipStream_t = void*;
namespace strix::core::f16_dense_tall::q8a8 {
struct View {
    const std::uint8_t* weight{};
    const std::int8_t* codes{};
    const float* deltas{};
    float* out{};
    unsigned row_bytes{}, k_bytes{}, in{}, count{}, rows{}, x_stride{}, out_stride{}, accumulate{};
};
std::vector<View> calls;
bool supported(const View& v) { return v.in && v.in % 128U == 0U && v.in <= v.x_stride; }
void enqueue(const View& v, hipStream_t) {
    calls.push_back(v);
    float sum = 0.0F;
    for (unsigned k = 0; k < v.in; ++k) sum = std::fma(static_cast<float>(v.codes[k]), v.deltas[k / 32U], sum);
    *v.out = v.accumulate ? *v.out + sum : sum;
}
#include "parts.inc"
}
namespace fast_profile = strix::ds41::fast_profile;
struct Q8A8DensePrefillView {
    const std::uint8_t* weight{};
    const std::int8_t* codes{};
    const float* deltas{};
    float* out{};
    unsigned row_width{}, width{}, k_offset{}, k_parts{}, count{}, rows{}, x_stride{}, out_stride{};
    bool accumulate{};
};
void* as_stream(void* stream) { return stream; }
int hipGetLastError() { return 0; }
void hip_check(int, const char*) {}
#include "dense.inc"
int main() {
    try {
#ifdef _WIN32
        require(_putenv_s("STRIX_DS41_FAST_DENSE_MIN_ROWS", "128") == 0, "set threshold");
#else
        require(setenv("STRIX_DS41_FAST_DENSE_MIN_ROWS", "128", 1) == 0, "set threshold");
#endif
        namespace q = strix::core::f16_dense_tall::q8a8;
        std::array<std::int8_t, 256> codes{};
        std::array<float, 8> deltas{};
        codes[0] = codes[128] = codes[160] = 1;
        deltas[0] = 16777216.0F; deltas[4] = 1.0F; deltas[5] = -16777216.0F;
        float out = -9.0F;
        Q8A8DensePrefillView v{};
        v.codes = codes.data(); v.deltas = deltas.data(); v.out = &out;
        v.width = 128; v.row_width = 512; v.k_offset = 128; v.k_parts = 2;
        v.rows = 4096; v.count = v.out_stride = 1; v.x_stride = 256;
        require(enqueue_fast_dense(v, nullptr), "FAST parts rejected");
        require(q::calls.size() == 2 && out == 1.0F, "part chain was fused across boundary");
        const auto first = q::calls[0], second = q::calls[1];
        require(first.in == 128 && second.in == 128 && first.accumulate == 0 && second.accumulate == 1,
                "independent chain or output order lost");
        require(first.k_bytes == 136 && second.k_bytes == 272 && first.row_bytes == 544 && second.row_bytes == 544 &&
                second.codes == codes.data() + 128 && second.deltas == deltas.data() + 4, "part offsets/strides wrong");
        q::calls.clear(); out = -9.0F;
        v.k_parts = 1;
        require(enqueue_fast_dense(v, nullptr), "first unfused call rejected");
        v.k_offset += v.width; v.codes += v.width; v.deltas += v.width / 32U; v.accumulate = true;
        require(enqueue_fast_dense(v, nullptr) && out == 1.0F, "unfused arithmetic differs");
        q::calls.clear(); v.width = 64; v.k_parts = 2;
        require(!enqueue_fast_dense(v, nullptr) && q::calls.empty(), "unsupported part partially launched");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
