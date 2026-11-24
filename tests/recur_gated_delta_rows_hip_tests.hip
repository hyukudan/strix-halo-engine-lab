// F9 recur gated_delta rows launch vs step launches (src/ops/recur/gated_delta_rows.hpp, gated_delta_step.hpp).
// One rows launch over 5 rows with padded qkv / core strides and distinct per-row beta and gate planes must equal five
// single-row step launches on the same operands: every output byte (padding included) and the final state bytes.
// A rows body that reuses row 0 gate (or beta, or qkv) changes the bytes and fails. Both policies.
#include <hip/hip_runtime.h>

#include "ops/recur/gated_delta_rows.hpp"
#include "ops/recur/gated_delta_step.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace recur = ::strix::ops::recur;
using strix::RecurrentStateType;
namespace lds = ::strix::gdn_recurrent_lds;

constexpr std::uint32_t kD = recur::kStepThreads;
constexpr std::uint32_t kKeyHeads = 1, kValueHeads = 2, kRows = 5;
int failures = 0;

struct Rng {
    std::uint64_t s;
    float next(float lo, float hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + (hi - lo) * static_cast<float>(static_cast<std::uint32_t>(s >> 40) & 0xFFFFFFU) / 16777216.0F;
    }
};

template <class T> struct Dev {
    T* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    bool put(const std::vector<T>& h) {
        n = h.size();
        return hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)) == hipSuccess &&
               hipMemcpy(p, h.data(), n * sizeof(T), hipMemcpyHostToDevice) == hipSuccess;
    }
    bool get(std::vector<T>& h) const { return hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess; }
    ~Dev() {
        if (p != nullptr) (void)hipFree(p);
    }
};

bool bytes_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

template <class Policy> bool run(const Policy& policy, std::uint64_t seed, const char* name) {
    const std::uint64_t width = (2ULL * kKeyHeads + kValueHeads) * kD;
    const std::uint64_t qkv_stride = width + 9U;                                          // padded
    const std::uint64_t core_stride = static_cast<std::uint64_t>(kValueHeads) * kD + 5U;  // padded
    const std::uint64_t gate_stride = policy.gate_row_stride(kValueHeads, kD);
    Rng rng{seed};
    std::vector<float> qkv(kRows * qkv_stride), beta(static_cast<std::size_t>(kRows) * kValueHeads),
        gate(static_cast<std::size_t>(kRows * gate_stride)), state(static_cast<std::size_t>(kValueHeads) * kD * kD);
    for (auto& x : qkv) x = rng.next(-1.0F, 1.0F);
    for (auto& x : beta) x = rng.next(0.1F, 1.0F);
    for (auto& x : gate) x = rng.next(-2.0F, -0.01F);  // distinct gate per row (and per key channel)
    for (auto& x : state) x = rng.next(-0.5F, 0.5F);
    const std::vector<float> sentinel(static_cast<std::size_t>(kRows * core_stride), 12345.0F);

    Dev<float> dq, db, dg, ds_in, ds_out, dcore_rows, dstate_steps, dcore_steps;
    if (!dq.put(qkv) || !db.put(beta) || !dg.put(gate) || !ds_in.put(state) || !ds_out.put(state) || !dcore_rows.put(sentinel) ||
        !dstate_steps.put(state) || !dcore_steps.put(sentinel)) {
        std::printf("FAIL %s: device setup\n", name);
        return false;
    }
    const dim3 grid(kValueHeads * (kD / recur::kStepValueTile)), block(recur::kStepThreads);
    recur::gated_delta_rows_kernel<lds::Form::lane_major, Policy><<<grid, block>>>(
        dq.p, db.p, dg.p, ds_in.p, ds_out.p, dcore_rows.p, kKeyHeads, kValueHeads, kD, kRows, qkv_stride, core_stride, policy);
    if (hipGetLastError() != hipSuccess) return false;
    for (std::uint32_t r = 0; r < kRows; ++r) {
        recur::gated_delta_step_kernel<RecurrentStateType::f32, Policy><<<grid, block>>>(
            dq.p + r * qkv_stride, db.p + static_cast<std::size_t>(r) * kValueHeads,
            dg.p + static_cast<std::size_t>(r * gate_stride), dstate_steps.p, dcore_steps.p + r * core_stride, kKeyHeads,
            kValueHeads, kD, nullptr, policy);
        if (hipGetLastError() != hipSuccess) return false;
    }
    if (hipDeviceSynchronize() != hipSuccess) return false;
    std::vector<float> core_rows(sentinel.size()), core_steps(sentinel.size()), state_rows(state.size()),
        state_steps(state.size()), state_in(state.size());
    if (!dcore_rows.get(core_rows) || !dcore_steps.get(core_steps) || !ds_out.get(state_rows) ||
        !dstate_steps.get(state_steps) || !ds_in.get(state_in))
        return false;
    const bool out_ok = bytes_equal(core_rows, core_steps);
    const bool state_ok = bytes_equal(state_rows, state_steps);
    const bool in_ok = bytes_equal(state_in, state);       // rows kernel left its input state alone
    const bool moved = !bytes_equal(state_rows, state);    // the test is not vacuous
    const bool written = !bytes_equal(core_rows, sentinel);
    const bool ok = out_ok && state_ok && in_ok && moved && written;
    std::printf("%s %s: outputs+padding %s, final state %s, input state %s\n", ok ? "PASS" : "FAIL", name,
                out_ok ? "equal" : "DIFFER", state_ok ? "equal" : "DIFFER", in_ok ? "untouched" : "MODIFIED");
    if (!ok) ++failures;
    return true;
}

}  // namespace

int main() {
    if (!run(recur::ScalarGate{}, 21, "rows==steps ScalarGate")) return 2;
    if (!run(recur::PerKeyChannelGate<recur::DeclaredExp>{}, 22, "rows==steps PerKeyChannelGate")) return 2;
    if (failures != 0) return 1;
    std::printf("recur gated_delta rows hip tests: OK\n");
    return 0;
}
