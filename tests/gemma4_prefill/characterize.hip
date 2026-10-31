// Instruction characterization, intentionally separate from model qualification.
#include <hip/hip_runtime.h>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Half2 = _Float16 __attribute__((ext_vector_type(2)));
using Float8 = float __attribute__((ext_vector_type(8)));
struct Case {
    std::uint16_t a[16], b[16];
    std::uint32_t c, result;
};
static_assert(sizeof(Case) == 72);
void checked(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
__global__ void probe(Case* cases, unsigned* orders) {
    auto& t = cases[blockIdx.x];
    Half16 a{}, b{};
    Float8 c{};
    for (unsigned i = 0; i < 16; ++i) {
        a[i] = __builtin_bit_cast(_Float16, t.a[i]);
        b[i] = __builtin_bit_cast(_Float16, t.b[i]);
    }
    for (unsigned i = 0; i < 8; ++i) c[i] = __builtin_bit_cast(float, t.c);
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    float forward = __builtin_bit_cast(float, t.c), reverse = forward;
    Float8 paired{};
    for (unsigned i = 0; i < 8; ++i) paired[i] = __builtin_bit_cast(float, t.c);
    for (unsigned j = 0; j < 8; ++j) {
        Half2 x = {a[2*j], a[2*j+1]}, y = {b[2*j], b[2*j+1]};
        forward = __builtin_amdgcn_fdot2(x, y, forward, false);
        Half2 rx = {a[14-2*j], a[15-2*j]}, ry = {b[14-2*j], b[15-2*j]};
        reverse = __builtin_amdgcn_fdot2(rx, ry, reverse, false);
        Half16 pa{}, pb{};
        pa[0] = a[2*j]; pa[1] = a[2*j+1]; pb[0] = b[2*j]; pb[1] = b[2*j+1];
        paired = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, pb, paired);
    }
    if (!threadIdx.x) {
        orders[std::size_t(blockIdx.x) * 3] = __builtin_bit_cast(unsigned, forward);
        orders[std::size_t(blockIdx.x) * 3 + 1] = __builtin_bit_cast(unsigned, reverse);
        orders[std::size_t(blockIdx.x) * 3 + 2] = __builtin_bit_cast(unsigned, paired[0]);
    }
    if (!threadIdx.x) t.result = __builtin_bit_cast(unsigned, c[0]);
}
unsigned random_bits(unsigned& s) {
    s ^= s << 13U; s ^= s >> 17U; s ^= s << 5U;
    return s;
}
int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("usage: characterize OUTPUT.bin");
    hipDeviceProp_t p{};
    checked(hipGetDeviceProperties(&p, 0));
    if (std::string(p.gcnArchName).find("gfx1151") != 0 || p.warpSize != 32)
        throw std::runtime_error("requires gfx1151 wave32");
    std::vector<Case> cases;
    unsigned seed = 0x472abcd1U;
    for (unsigned mode = 0; mode < 9; ++mode)
        for (unsigned j = 0; j < 32768; ++j) {
            Case t{};
            for (unsigned k = 0; k < 16; ++k) {
                if ((mode < 2 || mode == 7) && k != j % 16) continue;
                if (mode == 7 && j % 4 == 0) continue;
                if (mode == 2 && k >= 2) continue;
                const auto u = random_bits(seed), v = random_bits(seed);
                t.a[k] = std::uint16_t((u & 0x83ffU) | (((u >> 16U) % 31U) << 10U));
                t.b[k] = std::uint16_t((v & 0x83ffU) | (((v >> 16U) % 31U) << 10U));
                if (mode == 4) {
                    t.a[k] = std::uint16_t((u & 0x83ffU) | 0x3c00U);
                    t.b[k] = std::uint16_t((v & 0x83ffU) | 0x3c00U);
                }
            }
            const auto u = random_bits(seed);
            t.c = mode == 0 || mode == 3 ? 0U : (u & 0x807fffffU) | ((90U + ((u >> 23U) % 70U)) << 23U);
            if (mode >= 6) {
                t.c = (u & 0x7f800000U) == 0x7f800000U ? u ^ 0x00800000U : u;
                if (j < 64) {
                    const unsigned edges[] = {0, 1, 0x007fffff, 0x00800000, 0x00800001, 0x3f7fffff, 0x3f800000, 0x3f800001,
                                              0x4affffff, 0x4b000000, 0x4b000001, 0x7f7fffff, 0x80000000, 0x80000001, 0x807fffff, 0x80800000};
                    t.c = edges[j % 16];
                }
            }
            if (mode == 8) {
                const unsigned i = j % 16, k = (i + 1 + (j / 16) % 15) % 16;
                const auto av = t.a[0], bv = t.b[0];
                for (unsigned h = 0; h < 16; ++h) { t.a[h] = 0; t.b[h] = 0; }
                t.a[i] = av; t.b[i] = bv; t.a[k] = av ^ 0x8000U; t.b[k] = bv;
                const float product = float(__builtin_bit_cast(_Float16, av)) * float(__builtin_bit_cast(_Float16, bv));
                if (j % 4 == 1) t.c = __builtin_bit_cast(unsigned, product);
                if (j % 4 == 2) t.c = __builtin_bit_cast(unsigned, -product);
                if (j % 4 == 3) {
                    const unsigned extra = (k + 1) % 16 == i ? (k + 2) % 16 : (k + 1) % 16;
                    t.a[extra] = 1; t.b[extra] = 1;
                }
            }
            cases.push_back(t);
        }
    Case* device = nullptr;
    unsigned* device_orders = nullptr;
    std::vector<unsigned> orders(cases.size() * 3);
    checked(hipMalloc(&device, cases.size() * sizeof(Case)));
    checked(hipMalloc(&device_orders, orders.size() * sizeof(unsigned)));
    checked(hipMemcpy(device, cases.data(), cases.size() * sizeof(Case), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe, dim3(unsigned(cases.size())), dim3(32), 0, nullptr, device, device_orders);
    checked(hipGetLastError());
    checked(hipMemcpy(cases.data(), device, cases.size() * sizeof(Case), hipMemcpyDeviceToHost));
    checked(hipMemcpy(orders.data(), device_orders, orders.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
    checked(hipFree(device));
    checked(hipFree(device_orders));
    auto* f = std::fopen(argv[1], "wb");
    if (!f || std::fwrite(cases.data(), sizeof(Case), cases.size(), f) != cases.size())
        throw std::runtime_error("cannot write trace");
    std::fclose(f);
    const auto order_file = std::string(argv[1]) + ".orders";
    f = std::fopen(order_file.c_str(), "wb");
    if (!f || std::fwrite(orders.data(), sizeof(unsigned), orders.size(), f) != orders.size())
        throw std::runtime_error("cannot write order trace");
    std::fclose(f);
    std::array<std::size_t, 3> matches{};
    for (std::size_t i = 0; i < cases.size(); ++i)
        for (unsigned j = 0; j < 3; ++j) matches[j] += unsigned(cases[i].result == orders[i * 3 + j]);
    std::printf("order equality: forward dot2 %zu reverse dot2 %zu paired WMMA %zu\n", matches[0], matches[1], matches[2]);
    std::printf("gfx1151 instruction cases %zu\n", cases.size());
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
