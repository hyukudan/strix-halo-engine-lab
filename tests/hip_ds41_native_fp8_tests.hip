// Native FP8 x A8 site kernel and the shared-expert SwiGLU + A8 epilogue against their host references, bitwise:
// product_fp8 (ds41_native_fp8.hpp) for plain, row-offset (TP), grouped (attn_output_a) and K-split (attn_output_b)
// views at 1..8 activation rows and every outputs-per-wave choice, and quantize_a8(swiglu_clamped(gate, up)).
// Synthetic data only; runs in seconds.

#include "strix/ds41_formats.hpp"
#include "strix/ds41_moe.hpp"
#include "strix/ds41_native_fp8.hpp"
#include "strix/hip_ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strix::ds41;
int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::cerr << what << ": " << hipGetErrorString(e) << '\n'; std::exit(2); }
}

template <class T>
struct Device {
    T* p = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(1, count) * sizeof(T)), "hipMalloc"); }
    explicit Device(const std::vector<T>& h) : Device(h.size()) { up(h); }
    ~Device() { (void)hipFree(p); }
    Device(const Device&) = delete;
    void up(const std::vector<T>& h) { check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        check(hipDeviceSynchronize(), "sync");
        check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
        return h;
    }
};
template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

std::vector<std::uint8_t> random_fp8(std::mt19937& rng, std::size_t rows, std::size_t width) {
    std::vector<std::uint8_t> t(fp8_tensor_bytes(rows, width));
    std::normal_distribution<float> d(0.0F, 60.0F);
    for (std::size_t i = 0; i < rows * width; ++i) t[i] = float_to_e4m3(d(rng));
    for (std::size_t c = 0; c < 256U && c < rows * width; ++c)  // every finite code (subnormals, +-0, +-448) in row 0
        t[c] = (c & 0x7FU) == 0x7FU ? static_cast<std::uint8_t>(0x80U) : static_cast<std::uint8_t>(c);
    std::uniform_int_distribution<int> e(114, 119);
    for (std::size_t i = rows * width; i < t.size(); ++i) t[i] = static_cast<std::uint8_t>(e(rng));
    return t;
}
std::vector<float> random_values(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// One site case: a tensor of tensor_rows x row_width, outputs [first, first + count) in groups of per_group (activation
// group g = o / per_group), columns [k_offset, k_offset + width); optionally accumulated onto a prior output.
struct Case {
    const char* name;
    std::uint32_t tensor_rows, row_width, first, count, per_group, width, k_offset;
    bool accumulate;
};

void run_case(std::mt19937& rng, const Case& k) {
    const auto t = random_fp8(rng, k.tensor_rows, k.row_width);
    const std::uint32_t groups = k.count / k.per_group, x_stride = groups * k.width;
    Device<std::uint8_t> dw(t);
    for (const std::uint32_t rows : {1U, 2U, 3U, 4U, 5U, 8U}) {
        std::vector<std::int8_t> codes(std::size_t{rows} * x_stride);
        std::vector<float> deltas(codes.size() / 32U);
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto a8 = quantize_a8(random_values(rng, x_stride, 2.0F));
            std::copy(a8.codes.begin(), a8.codes.end(), codes.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * x_stride));
            std::copy(a8.deltas.begin(), a8.deltas.end(), deltas.begin() + static_cast<std::ptrdiff_t>(std::size_t{r} * x_stride / 32U));
        }
        const auto prior = random_values(rng, std::size_t{rows} * k.count, 3.0F);
        std::vector<float> ref(prior.size());
        for (std::uint32_t r = 0; r < rows; ++r)
            for (std::uint32_t o = 0; o < k.count; ++o) {
                const std::uint32_t row = k.first + o, g = o / k.per_group;
                const std::uint8_t* wc = t.data() + std::size_t{row} * k.row_width + k.k_offset;
                const std::uint8_t* ws = t.data() + std::size_t{k.tensor_rows} * k.row_width + std::size_t{row / 32U} * (k.row_width / 32U) + k.k_offset / 32U;
                const std::int8_t* xc = codes.data() + std::size_t{r} * x_stride + std::size_t{g} * k.width;
                const float* xd = deltas.data() + (std::size_t{r} * x_stride + std::size_t{g} * k.width) / 32U;
                const float y = product_fp8(wc, ws, xc, xd, k.width);
                const std::size_t at = std::size_t{r} * k.count + o;
                ref[at] = k.accumulate ? prior[at] + y : y;
            }
        Device<std::int8_t> dc(codes);
        Device<float> dd(deltas);
        for (const std::uint32_t w : {0U, 1U, 2U, 4U}) {  // 0: the default choice
            set_native_fp8_outputs_per_wave(w);
            Device<float> out(prior);
            Q8A8View v;
            v.weight = dw.p;
            v.scales = dw.p + std::size_t{k.tensor_rows} * k.row_width;
            v.packed = packed_native_fp8;
            v.width = k.width;
            v.first = k.first;
            v.count = k.count;
            v.per_group = k.per_group;
            v.codes = dc.p;
            v.deltas = dd.p;
            v.rows = rows;
            v.x_stride = x_stride;
            v.out_stride = k.count;
            v.out = out.p;
            v.row_width = k.row_width == k.width ? 0U : k.row_width;
            v.k_offset = k.k_offset;
            v.accumulate = k.accumulate;
            enqueue_q8_0_a8(v, nullptr);
            const auto got = out.down();
            expect(same_bits(got, ref), std::string("native FP8 site ") + k.name + " rows=" + std::to_string(rows) + " W=" + std::to_string(w));
            if (rows == 1U && w == 1U) {
                const auto access = launch_access(v);
                bool scales = false;
                for (std::uint32_t i = 0; i < access.count; ++i) scales = scales || access.ranges[i].pointer == v.scales;
                expect(scales, std::string("launch access lists the exponents, ") + k.name);
            }
        }
    }
    set_native_fp8_outputs_per_wave(0);
}

}  // namespace

int main() {
    std::mt19937 rng(1017);
    // Geometries of the decode sites, reduced in rows: q_a / kv (5120 wide), q_b with a TP row offset that is not a
    // multiple of 32, the grouped attn_output_a (2 groups), both K-halves of attn_output_b, the shared down (2304 wide).
    const Case cases[] = {
        {"q_a", 96, 5120, 0, 96, 96, 5120, 0, false},
        {"q_b tp", 160, 1280, 40, 96, 96, 1280, 0, false},
        {"o_a grouped", 128, 4096, 64, 64, 32, 4096, 0, false},
        {"o_b h0", 64, 8192, 0, 64, 64, 4096, 0, false},
        {"o_b h1 accumulate", 64, 8192, 0, 64, 64, 4096, 4096, true},
        {"shared down", 64, 2304, 32, 32, 32, 2304, 0, false},
    };
    for (const auto& k : cases) run_case(rng, k);

    // Shared-expert epilogue: act = swiglu_clamped(gate, up, 10), A8 per row (quantize_a8), against the host.
    for (const std::uint32_t rows : {1U, 3U, 8U}) {
        const std::uint32_t width = 2304;
        auto gate = random_values(rng, std::size_t{rows} * width, 6.0F), up = random_values(rng, std::size_t{rows} * width, 6.0F);
        gate[5] = 0.0F;  // a zero group: every value of the first group of row 0
        for (std::uint32_t j = 0; j < 32U; ++j) up[j] = 0.0F;
        std::vector<std::int8_t> ref_codes;
        std::vector<float> ref_deltas;
        std::vector<std::int32_t> ref_sums;
        for (std::uint32_t r = 0; r < rows; ++r) {
            std::vector<float> act(width);
            for (std::uint32_t i = 0; i < width; ++i) act[i] = swiglu_clamped(gate[std::size_t{r} * width + i], up[std::size_t{r} * width + i], 10.0F);
            const auto a8 = quantize_a8(act);
            ref_codes.insert(ref_codes.end(), a8.codes.begin(), a8.codes.end());
            ref_deltas.insert(ref_deltas.end(), a8.deltas.begin(), a8.deltas.end());
            ref_sums.insert(ref_sums.end(), a8.sums.begin(), a8.sums.end());
        }
        Device<float> dg(gate), du(up);
        Device<std::int8_t> codes(ref_codes.size());
        Device<float> deltas(ref_deltas.size());
        Device<std::int32_t> sums(ref_sums.size());
        enqueue_swiglu_a8(dg.p, du.p, rows, width, 10.0F, codes.p, deltas.p, sums.p, nullptr);
        expect(same_bits(codes.down(), ref_codes) && same_bits(deltas.down(), ref_deltas) && same_bits(sums.down(), ref_sums),
               "SwiGLU + A8 bitwise, rows=" + std::to_string(rows));
    }

    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "hip ds41 native fp8 tests passed\n";
    return 0;
}
