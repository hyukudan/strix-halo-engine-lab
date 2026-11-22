// Decode tight BF16 GEMV: the rows-1 bucket (one output per wave, chunk loop unrolled by two) stores the same bits as
// the two-outputs-per-wave tile it replaced, for single launches and same-input groups, FP32 and BF16-rounded stores,
// K with an odd chunk count per lane, partial blocks, non-finite / signed-zero / subnormal activations.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "core/dense_linear.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace dl = strix::core::dense_linear;
using strix::core::weight_decode::Format;

bool ok(hipError_t status) { return status == hipSuccess; }

template <class T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (!ok(hipMalloc(&device, host.size() * sizeof(T)))) return nullptr;
    if (!ok(hipMemcpy(device, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice))) return nullptr;
    return device;
}

std::vector<float> download(const float* device, std::size_t count) {
    std::vector<float> host(count);
    if (!ok(hipMemcpy(host.data(), device, count * sizeof(float), hipMemcpyDeviceToHost))) host.clear();
    return host;
}

template <bool RoundBf16>
bool run(unsigned K, const std::vector<unsigned>& widths, std::mt19937& random, std::size_t& compared) {
    constexpr auto epilogue = RoundBf16 ? dl::Epilogue::store_tight_rows_bf16 : dl::Epilogue::store_tight_rows;
    std::uniform_real_distribution<float> value(-2.0F, 2.0F);
    std::vector<float> x(K);
    for (auto& v : x) v = value(random);
    const std::uint32_t specials[] = {0x7F800000U, 0xFF800000U, 0x7FC00000U, 0x80000000U, 0x00000001U, 0x7F7FFFFFU};
    for (std::size_t i = 3; i < x.size(); i += 211U) std::memcpy(&x[i], &specials[random() % 6U], sizeof(float));
    float* d_x = upload(x);
    const unsigned count = static_cast<unsigned>(widths.size());
    std::vector<std::uint8_t*> weights(count, nullptr);
    std::vector<float*> group_out(count, nullptr), single_out(count, nullptr), reference(count, nullptr);
    std::vector<dl::View> views(count);
    bool pass = d_x != nullptr;
    for (unsigned p = 0; p < count && pass; ++p) {
        std::vector<std::uint16_t> w(std::size_t{widths[p]} * K);
        for (auto& v : w) {
            const float f = value(random);
            std::uint32_t bits;
            std::memcpy(&bits, &f, sizeof bits);
            v = static_cast<std::uint16_t>(bits >> 16U);
        }
        auto* d_w = upload(w);
        weights[p] = reinterpret_cast<std::uint8_t*>(d_w);
        views[p] = {weights[p], K, widths[p], std::size_t{K} * 2U};
        const std::size_t bytes = std::size_t{widths[p]} * sizeof(float);
        pass = d_w && ok(hipMalloc(&group_out[p], bytes)) && ok(hipMalloc(&single_out[p], bytes)) &&
               ok(hipMalloc(&reference[p], bytes)) && ok(hipMemset(group_out[p], 0x5A, bytes)) &&
               ok(hipMemset(single_out[p], 0x5A, bytes));
    }
    if (pass) {
        for (unsigned p = 0; p < count && pass; ++p) {
            // The replaced rows-1 tile: two outputs per wave, the compiler's chunk loop.
            dl::kernel<1, 2, Format::bf16, epilogue>
                <<<(widths[p] + 7U) / 8U, 128>>>(views[p], views[p], d_x, reference[p], 1U);
            pass = ok(dl::enqueue_bf16_tight_rows<RoundBf16>(views[p], d_x, single_out[p], 1U, nullptr));
        }
        pass = pass &&
               ok(dl::enqueue_bf16_tight_rows_group<RoundBf16>(views.data(), group_out.data(), count, d_x, 1U, nullptr)) &&
               ok(hipDeviceSynchronize());
    }
    for (unsigned p = 0; p < count && pass; ++p) {
        const auto want = download(reference[p], widths[p]), single = download(single_out[p], widths[p]),
                   grouped = download(group_out[p], widths[p]);
        pass = want.size() == widths[p] && single.size() == widths[p] && grouped.size() == widths[p] &&
               std::memcmp(want.data(), single.data(), widths[p] * sizeof(float)) == 0 &&
               std::memcmp(want.data(), grouped.data(), widths[p] * sizeof(float)) == 0;
        compared += 2U * widths[p];
    }
    (void)hipFree(d_x);
    for (unsigned p = 0; p < count; ++p)
        for (void* q : {static_cast<void*>(weights[p]), static_cast<void*>(group_out[p]),
                        static_cast<void*>(single_out[p]), static_cast<void*>(reference[p])})
            if (q) (void)hipFree(q);
    return pass;
}

}  // namespace

int main() {
    std::mt19937 random(20261122U);
    std::size_t compared = 0;
    // K 3072 (6 chunks per lane), K 2304 (4-5, an odd count on half the lanes), K 48 (lanes past the chunk count idle),
    // K 1040 (65 chunks: one lane runs a third iteration); widths with a partial last block and a group of four.
    const struct { unsigned K; std::vector<unsigned> widths; bool round; } cases[] = {
        {3072U, {2304U}, false}, {2304U, {3072U, 384U, 37U}, false}, {48U, {200U, 9U}, true},
        {1040U, {64U, 65U, 1U, 129U}, true}, {2304U, {2304U, 2304U}, true},
    };
    for (const auto& c : cases) {
        const bool pass = c.round ? run<true>(c.K, c.widths, random, compared) : run<false>(c.K, c.widths, random, compared);
        if (!pass) {
            std::printf("dense-linear-tight-rows: FAIL K %u projections %zu\n", c.K, c.widths.size());
            return 1;
        }
    }
    std::printf("dense-linear-tight-rows: %zu outputs bitwise equal (rows-1 tile vs two outputs per wave)\n", compared);
    return 0;
}
