#include <hip/hip_runtime.h>
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"

#include "mini_nvfp4_table_cases.hpp"

namespace {

using namespace mini_nvfp4_table_cases;

__global__ void check_tables(const std::uint32_t* outer, Mismatch* results) {
    const unsigned scale = threadIdx.x;
    results[blockIdx.x * scale_count + scale] =
        check_block(strix::weight::scalar_float::u2f(outer[blockIdx.x]), static_cast<std::uint8_t>(scale));
}

bool hip_ok(hipError_t status) {
    if (status == hipSuccess) return true;
    std::fprintf(stderr, "%s\n", hipGetErrorString(status));
    return false;
}

}  // namespace

int main() {
    std::uint32_t* outer = nullptr;
    Mismatch* results = nullptr;
    if (!hip_ok(hipMallocManaged(&outer, outer_bits.size() * sizeof(*outer)))) return 1;
    if (!hip_ok(hipMallocManaged(&results, outer_bits.size() * scale_count * sizeof(*results)))) {
        static_cast<void>(hipFree(outer));
        return 1;
    }
    for (unsigned i = 0; i < outer_bits.size(); ++i) outer[i] = outer_bits[i];
    check_tables<<<static_cast<unsigned>(outer_bits.size()), scale_count>>>(outer, results);
    bool ok = hip_ok(hipGetLastError()) && hip_ok(hipDeviceSynchronize());
    if (ok) {
        for (unsigned i = 0; i < outer_bits.size(); ++i) {
            for (unsigned scale = 0; scale < scale_count; ++scale) {
                ok &= report(results[i * scale_count + scale], outer_bits[i], scale);
            }
        }
    }
    const bool free_results = hip_ok(hipFree(results));
    const bool free_outer = hip_ok(hipFree(outer));
    return ok && free_results && free_outer ? 0 : 1;
}
