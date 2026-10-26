// GSQ-RCO Q2_0 A8 gate/up (q20_numerical_identity): every output of enqueue_probe == the original-byte CPU
// oracle evaluate_q20_row, bit for bit, on random Q2_0 rows (all codes, signed/zero/subnormal/large fp16 d),
// BF16 and FP32 inputs, distinct and duplicate slots; activation image == the contract quantizer. The type is
// rejected without STRIX_GSQ_Q20_A8=1.
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/gsq_a8_groups.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using namespace strix::moe_q4k_a8;
void expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }

struct Buffer {
    const strix::HipSetupQuiescenceGuard& guard; void* owner = nullptr; std::size_t size;
    Buffer(std::size_t n, const strix::HipSetupQuiescenceGuard& g) : guard(g), size(n) {
        guard.run([&] { check(hipMalloc(&owner, n + 32U)); check(hipMemset(owner, 0xa5, n + 32U)); });
    }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (owner) guard.run([&] { check(hipFree(owner)); }); }
    void* data() const { return static_cast<unsigned char*>(owner) + 16U; }
    DeviceBytes view() const { return {reinterpret_cast<std::uintptr_t>(data()), size}; }
    void upload(const void* p) { guard.run([&] { check(hipMemcpy(data(), p, size, hipMemcpyHostToDevice)); }); }
    template <class T> std::vector<T> read() const {
        std::vector<T> out(size / sizeof(T));
        guard.run([&] { check(hipMemcpy(out.data(), data(), size, hipMemcpyDeviceToHost)); });
        return out;
    }
};

void fixture(std::uint32_t type, unsigned hidden, unsigned rows, unsigned top_k, bool bf16, unsigned seed,
             const strix::HipSetupQuiescenceGuard& guard, hipStream_t stream) {
    const unsigned experts = std::max(top_k, 3U);
    Request request; request.enabled = true; request.approved_numerical_identity = numerical_identity_for_gate_type(type);
    request.gate_type = request.up_type = type; request.input = bf16 ? InputKind::bf16 : InputKind::fp32;
    request.shape = {hidden, rows, experts, top_k, 1};
    const std::size_t row_bytes = strix::gsq_a8::row_bytes(type, hidden), block = strix::gsq_a8::block_bytes(type), weight_bytes = static_cast<std::size_t>(experts) * rows * row_bytes;
    std::mt19937 random(0x20a8U + seed + hidden + rows + top_k + type);
    std::vector<float> activation(hidden); std::vector<std::uint16_t> bf(hidden);
    for (unsigned i = 0; i < hidden; ++i) {
        activation[i] = std::ldexp(static_cast<float>(static_cast<int>(random() % 255U) - 127), -6);
        if (seed == 1U && i % 64U < 32U) activation[i] = (i & 1U) ? -0.0F : 0.0F;  // all-zero groups
        bf[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(activation[i]) >> 16U);
        if (bf16) activation[i] = std::bit_cast<float>(static_cast<std::uint32_t>(bf[i]) << 16U);
    }
    std::vector<std::byte> gw(weight_bytes), uw(weight_bytes);
    const std::array<unsigned, 10> headers{0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x3c00, 0xbc00, 0x5bff, 0xdbff};
    for (auto* w : {&gw, &uw}) {
        for (auto& v : *w) v = static_cast<std::byte>(random() & 255U);
        for (std::size_t b = 0; b < weight_bytes / block; ++b) {
            const auto d = headers[random() % headers.size()];
            (*w)[b * block] = static_cast<std::byte>(d & 255U); (*w)[b * block + 1U] = static_cast<std::byte>(d >> 8U);
        }
    }
    std::vector<unsigned> rt(top_k);
    for (unsigned s = 0; s < top_k; ++s) rt[s] = (experts - 1U - s % experts + (s >= experts ? 1U : 0U)) % experts;
    Buffer input(hidden * (bf16 ? 2U : 4U), guard), gate(weight_bytes, guard), up(weight_bytes, guard),
        routes(top_k * 4U, guard), codes(hidden, guard), deltas(hidden / 8U, guard), sums(hidden / 8U, guard),
        status(4, guard), out(top_k * 2U * rows * 4U, guard);
    input.upload(bf16 ? static_cast<const void*>(bf.data()) : static_cast<const void*>(activation.data()));
    gate.upload(gw.data()); up.upload(uw.data()); routes.upload(rt.data());
    const unsigned zero = 0; status.upload(&zero);
    const Bindings binding{input.view(), gate.view(), up.view(), routes.view(), codes.view(), deltas.view(), sums.view(),
                           status.view(), out.view()};
    guard.run([&] { enqueue_probe(request, binding, stream); check(hipStreamSynchronize(stream)); });
    expect(status.read<unsigned>()[0] == 0, "Q2_0 finite status");
    const auto image = quantize_f32(activation);
    expect(codes.read<std::int8_t>() == image.codes && sums.read<int>() == image.sums, "Q2_0 activation codes/sums");
    const auto ds = deltas.read<unsigned>();
    for (unsigned g = 0; g < ds.size(); ++g) expect(ds[g] == std::bit_cast<unsigned>(image.deltas[g]), "Q2_0 delta bits");
    const auto actual = out.read<unsigned>();
    unsigned negative = 0;
    for (unsigned slot = 0; slot < top_k; ++slot) for (unsigned which = 0; which < 2; ++which) for (unsigned row = 0; row < rows; ++row) {
        const auto& w = which == 0 ? gw : uw;
        const auto raw = std::span<const std::byte>(w).subspan((static_cast<std::size_t>(rt[slot]) * rows + row) * row_bytes, row_bytes);
        const float expected = evaluate_gsq_row(type, raw, image);
        negative += expected < 0.0F;
        expect(actual[(slot * 2U + which) * rows + row] == std::bit_cast<unsigned>(expected),
               "Q2_0 gate/up output differs from the original-byte CPU oracle");
    }
    expect(negative > 0, "Q2_0 fixture has no negative outputs");
    expect(gate.read<std::byte>() == gw && up.read<std::byte>() == uw, "Q2_0 weights changed");
}
}  // namespace

int main() try {
    strix::HipSetupQuiescenceGuard guard; expect(guard.architecture().starts_with("gfx1151"), "actual gfx1151 required");
    require_f32_environment();
    bool rejected = false;
    try { (void)numerical_identity_for_gate_type(42); } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "Q2_0 admitted without STRIX_GSQ_Q20_A8=1");
    setenv("STRIX_GSQ_Q20_A8", "1", 1);
    expect(numerical_identity_for_gate_type(42) == q20_numerical_identity, "Q2_0 identity");
    hipStream_t stream = nullptr; guard.run([&] { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); });
    for (std::uint32_t type : {42U, 22U, 18U, 21U, 23U}) {
        for (bool bf16 : {false, true}) for (unsigned seed : {0U, 1U}) {
            fixture(type, 256U, 32U, 2U, bf16, seed, guard, stream);
            fixture(type, 2560U, 640U, 10U, bf16, seed, guard, stream);
        }
        std::cout << "type " << type << " PASS" << std::endl;
    }
    guard.run([&] { check(hipStreamDestroy(stream)); });
    std::cout << "Q2_0 A8 gate/up probe fixtures PASS\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
