// Q8/A8 edge fixtures shared with the prefill qualification suite; adapted to HIP.
#include "strix/hip_ds41_layer.hpp"
#include "strix/ds41_prefill_tuning.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
std::mt19937_64 rng(20261020);
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void env(const char* key, const char* val) { if (val) setenv(key, val, 1); else unsetenv(key); }
template<class T> struct Device {
    T* p = nullptr; std::size_t n;
    explicit Device(std::size_t count): n(count) { check(hipMalloc(&p, n * sizeof(T))); }
    explicit Device(const std::vector<T>& v): Device(v.size()) { put(v); }
    ~Device() { (void)hipFree(p); }
    Device(const Device&) = delete;
    void put(const std::vector<T>& v) { check(hipMemcpy(p, v.data(), n*sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() { std::vector<T> v(n); check(hipMemcpy(v.data(),p,n*sizeof(T),hipMemcpyDeviceToHost)); return v; }
};
std::vector<std::uint8_t> random_q8(std::size_t rows, std::size_t width, bool edges) {
    std::vector<std::uint8_t> w(rows * width / 32 * 34);
    std::uniform_int_distribution<int> e(3, 12), m(0, 1023), sgn(0, 1), code(-127, 127), code_e(-128, 127), pick(0, 31), kind(0, 6),
        rare(0, 63);
    for (std::size_t b = 0; b < w.size(); b += 34) {
        auto h = static_cast<std::uint16_t>((sgn(rng) << 15) | (e(rng) << 10) | m(rng));
        bool zero = false;
        if (edges && pick(rng) == 0) {
            switch (kind(rng)) {
            case 0: zero = true; h = 0; break;
            case 1: h = 0x8000; break;
            case 2: h = 0x7BFF; break;
            case 3: h = 0x0001; break;
            case 4: h = rare(rng) ? 0x0200 : 0x7C00; break;  // Inf (rare, else a subnormal)
            case 5: h = rare(rng) ? 0x8001 : 0x7E00; break;  // NaN (rare)
            default: h = 0xFBFF; break;
            }
        }
        std::memcpy(&w[b], &h, 2);
        for (unsigned j = 0; j < 32; ++j)
            w[b + 2 + j] = zero ? 0 : static_cast<std::uint8_t>(static_cast<std::int8_t>(edges ? code_e(rng) : code(rng)));
    }
    return w;
}

// A8 rows (codes in [-127, 127], deltas, sums unused by the product). Deltas: |d| ~ 2^-10 .. 2^-4; with edges 1 group
// in 32 is special (zero codes, delta 0 / -0 / FLT_MAX / subnormal / Inf / NaN / 2^100).
struct HostA8 {
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};
HostA8 random_a8(std::size_t rows, std::size_t stride, bool edges) {
    HostA8 h;
    h.codes.resize(rows * stride);
    h.deltas.resize(rows * stride / 32);
    h.sums.assign(rows * stride / 32, 0);
    std::uniform_int_distribution<int> code(-127, 127), pick(0, 31), kind(0, 7), rare(0, 63);
    std::uniform_real_distribution<float> ex(-10.0F, -4.0F);
    for (auto& x : h.codes) x = static_cast<std::int8_t>(code(rng));
    for (std::size_t g = 0; g < h.deltas.size(); ++g) {
        float d = std::exp2(ex(rng));
        if (edges && pick(rng) == 0) {
            switch (kind(rng)) {
            case 0: std::memset(&h.codes[g * 32], 0, 32); break;
            case 1: d = 0.0F; break;
            case 2: d = -0.0F; break;
            case 3: d = rare(rng) ? 0x1p60F : std::numeric_limits<float>::max(); break;
            case 4: d = std::numeric_limits<float>::denorm_min(); break;
            case 5: d = rare(rng) ? 0x1p-130F : std::numeric_limits<float>::infinity(); break;
            case 6: d = rare(rng) ? -0x1p-8F : std::numeric_limits<float>::quiet_NaN(); break;
            default: d = 0x1p100F; break;
            }
        }
        h.deltas[g] = d;
    }
    return h;
}

void run(unsigned width, unsigned outputs, unsigned rows, bool edges, bool split) {
    using namespace strix::ds41;
    const unsigned rw = split ? 2U * width : width, xs = rw + 32U, os = outputs + 7U;
    const auto weights = random_q8(outputs, rw, edges);
    const auto a = random_a8(rows, xs, edges);
    Device<std::uint8_t> dw(weights); Device<std::int8_t> dc(a.codes); Device<float> dd(a.deltas);
    std::vector<float> seed(std::size_t{rows}*os, -0.125F);
    Device<float> out(seed);
    Q8A8DensePrefillView v;
    v.weight=dw.p; v.codes=dc.p; v.deltas=dd.p; v.width=width; v.count=outputs;
    v.x_stride=xs; v.out_stride=os; v.rows=rows; v.out=out.p; v.row_width=rw;
    auto launch=[&](bool tuned) {
        out.put(seed); env("STRIX_DS41_PREFILL_TILES", tuned?"1":"0");
        env("STRIX_DS41_PREFILL_DENSE_TILE", "16");
        for(unsigned part=0;part<(split?2U:1U);++part) {
            v.k_offset=part*width; v.codes=dc.p+part*width; v.deltas=dd.p+part*width/32U;
            v.accumulate=split && part!=0U;
            enqueue_q8_0_a8_dense_prefill(v,nullptr);
        }
        check(hipDeviceSynchronize()); return out.get();
    };
    const auto ref=launch(false), got=launch(true);
    std::size_t bad=0;for(std::size_t i=0;i<got.size();++i) if(std::memcmp(&ref[i],&got[i],4)!=0) ++bad;
    std::printf("dense width=%u outputs=%u rows=%u edges=%u split=%u words=%zu unequal=%zu\n",width,outputs,rows,edges,split,got.size(),bad);
    if(bad) throw std::runtime_error("dense tile bits differ");
}
}
int main() {
    try {
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, 0));
        std::printf("device=%s arch=%s shared_per_block=%zu max_threads=%d CUs=%d\n",
                    prop.name, prop.gcnArchName, prop.sharedMemPerBlock, prop.maxThreadsPerBlock, prop.multiProcessorCount);
        for(bool edges:{false,true}) {
            // PFQ site geometries; 17 rows crosses a row tile and leaves a tail.
            run(1280,32768,17,edges,false); run(4096,1024,17,edges,false);
            run(4096,5120,17,edges,true); run(5120,1792,17,edges,false);
            run(5120,4608,17,edges,false); run(2304,5120,17,edges,false);
            for(unsigned width:{32U,512U,1152U,1280U,4096U,5120U})
                for(unsigned rows:{1U,13U,33U}) run(width,35,rows,edges,false);
        }
        env("STRIX_DS41_PREFILL_TILES","0");env("STRIX_DS41_PREFILL_DENSE_TILE",nullptr);
        std::puts("PFQ HIP dense tile fixtures PASS");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
