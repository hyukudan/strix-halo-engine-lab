// Shared HIP cache-row writer (attn_core/cache_rows.hpp) == host reference (strix/attn/cache_rows.hpp), bitwise, on
// the MODEL_B geometries: MLA rows 576 (512 latent + 64 rope) in a full cache (DSA layers, explicit positions) and in the
// SWA ring of 512 (decode one row per launch past a wrap, and a prefill chunk of 64 rows in a ring of 1024), and the
// indexer token keys (128) in the token ring of 7; a non-finite input raises the flag.
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "attn_core/cache_rows.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace cr = strix::attn::cache_rows;
namespace dev = strix::attn_core::cache_rows;

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* dalloc(std::size_t n) {
    void* p = nullptr;
    ok(hipMalloc(&p, n * sizeof(T)), "hipMalloc");
    return static_cast<T*>(p);
}

std::uint64_t lcg = 12345;
float uniform() {
    lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(static_cast<double>(lcg >> 40U) / 8388608.0 - 1.0) * 8.0F;
}

// Writes `launches` launches of `rows` rows each (consecutive positions from p0, or the explicit list) on the device
// and with the host reference; compares the whole cache and the flag.
bool run(const char* name, cr::Rows layout, std::size_t slots, std::size_t stride, std::uint32_t p0, unsigned rows,
         unsigned launches, const std::vector<std::uint32_t>& list, bool poison) {
    const std::size_t total = static_cast<std::size_t>(rows) * launches;
    std::vector<float> src(total * stride);
    for (auto& v : src) v = uniform();
    if (poison) src[stride + 3U] = std::numeric_limits<float>::quiet_NaN();
    std::vector<std::uint16_t> want(slots * layout.width, 0x5555U), got(want.size());
    bool want_ok = true;
    float* d_src = dalloc<float>(src.size());
    std::uint16_t* d_cache = dalloc<std::uint16_t>(want.size());
    std::uint32_t* d_pos = list.empty() ? nullptr : dalloc<std::uint32_t>(list.size());
    unsigned* d_flag = dalloc<unsigned>(1);
    ok(hipMemcpy(d_src, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice), "copy src");
    ok(hipMemcpy(d_cache, want.data(), want.size() * 2U, hipMemcpyHostToDevice), "copy cache");
    if (d_pos != nullptr) ok(hipMemcpy(d_pos, list.data(), list.size() * 4U, hipMemcpyHostToDevice), "copy pos");
    ok(hipMemset(d_flag, 0, sizeof(unsigned)), "flag");
    for (unsigned l = 0; l < launches; ++l) {
        const std::size_t first = static_cast<std::size_t>(l) * rows;
        const std::uint32_t* hp = list.empty() ? nullptr : list.data() + first;
        const auto fp = static_cast<std::uint32_t>(p0 + first);
        want_ok = cr::store_rows_ref(src.data() + first * stride, stride, hp, fp, rows, layout, want.data()) && want_ok;
        dev::StoreArgs a{d_src + first * stride, stride, d_pos == nullptr ? nullptr : d_pos + first, fp, rows, layout, d_cache, d_flag};
        ok(dev::store_rows(a, nullptr), "store_rows");
    }
    unsigned flag = 0;
    ok(hipMemcpy(got.data(), d_cache, got.size() * 2U, hipMemcpyDeviceToHost), "copy back");
    ok(hipMemcpy(&flag, d_flag, sizeof flag, hipMemcpyDeviceToHost), "flag back");
    std::size_t diff = 0;
    for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != want[i] ? 1U : 0U;
    const bool pass = diff == 0U && (flag != 0U) == !want_ok && want_ok == !poison;
    std::printf("%s %s: %zu rows -> %zu slots x %u, diff %zu, flag %u\n", pass ? "PASS" : "FAIL", name, total, slots,
                layout.width, diff, flag);
    ok(hipFree(d_src), "free");
    ok(hipFree(d_cache), "free");
    if (d_pos != nullptr) ok(hipFree(d_pos), "free");
    ok(hipFree(d_flag), "free");
    return pass;
}

}  // namespace

int main() {
    bool pass = true;
    // DSA layer: full cache, explicit positions (a verify window and a jump), source stride 576 (kv_a row)
    std::vector<std::uint32_t> dsa{0U, 1U, 2U, 3U, 700U, 701U, 702U, 1999U};
    pass = run("modelb dsa full 576", cr::Rows{576U, 0U}, 2048U, 576U, 0U, 4U, 2U, dsa, false) && pass;
    // SWA decode: one row per launch, 1300 positions through the ring of 512 (two wraps)
    pass = run("modelb swa ring 512 decode", cr::Rows{576U, cr::window_ring(512U, 1U)}, 512U, 576U, 0U, 1U, 1300U, {}, false) && pass;
    // SWA prefill chunks of 64 rows in the ring the reader needs for that chunk (1024)
    pass = run("modelb swa ring prefill 64", cr::Rows{576U, cr::window_ring(512U, 64U)}, 1024U, 576U, 37U, 64U, 30U, {}, false) && pass;
    // indexer token keys: width 128 from a wider staging row (stride 160), token ring of 7 (ratio 4, 4 verify rows)
    pass = run("modelb token keys ring 7", cr::Rows{128U, 7U}, 7U, 160U, 5U, 4U, 9U, {}, false) && pass;
    // a NaN input flags the launch
    pass = run("non-finite flag", cr::Rows{576U, 0U}, 16U, 576U, 0U, 4U, 1U, {}, true) && pass;
    std::printf("%s cache rows\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
