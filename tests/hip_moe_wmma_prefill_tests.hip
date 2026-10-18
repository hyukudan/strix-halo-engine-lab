// Routed MoE prefill: the iu8 WMMA kernels (form 1) against the dot4 kernels (form 0) through
// the complete known_rows::enqueue_prefill entry (prologue, CSR, gate/up + down quantizer, down).
// Every written byte (prologue codes, intermediates, down image, raw slot dots, statuses, the
// summary word) must be identical. Default mode: the bitwise cases. "bench": timing per form.
#include "strix/hip_moe_q4k_a8_known_rows.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace a8 = strix::moe_q4k_a8;
namespace kr = strix::moe_q4k_a8::known_rows;
void check(hipError_t e, const char* why) {
    if (e != hipSuccess) throw std::runtime_error(std::string(why) + ": " + hipGetErrorString(e));
}
std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}
__device__ std::uint64_t device_mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}
// fp16 scale bits: ordinary scales; flavor 1 also zero, subnormal, negative and max ones;
// flavor 2 also infinities and NaNs (rows fail); flavor 3: those in the down weights only.
__device__ unsigned short scale_bits(std::uint64_t r, unsigned flavor) {
    const unsigned pick = static_cast<unsigned>(r & 4095U);
    if (flavor && pick < 48U) {
        const unsigned short special[12]{0x0000U, 0x8000U, 0x0001U, 0x03ffU, 0x8200U, 0x7bffU, 0xfbffU, 0x0400U, 0x7a00U, 0x7c00U, 0xfc00U, 0x7e00U};
        return special[pick % (flavor == 2U ? 12U : 9U)];
    }
    // 2^-12 .. 2^-6 with random mantissa, sign mostly positive.
    const unsigned exponent = 3U + static_cast<unsigned>((r >> 12U) % 7U), mantissa = static_cast<unsigned>((r >> 16U) & 1023U);
    const unsigned sign = ((r >> 30U) & 15U) == 0U ? 0x8000U : 0U;
    return static_cast<unsigned short>(sign | exponent << 10U | mantissa);
}
// Random packed blocks with format-correct fp16 headers. kind: 12 Q4_K, 13 Q5_K, 7 Q5_1, 8 Q8_0.
__global__ void fill_blocks(unsigned char* bytes, std::uint64_t blocks, unsigned kind, std::uint64_t seed, unsigned flavor) {
    const unsigned block_bytes = kind == 12U ? 144U : kind == 13U ? 176U : kind == 7U ? 24U : 34U;
    for (std::uint64_t b = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x; b < blocks;
         b += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        unsigned char* p = bytes + b * block_bytes;
        std::uint64_t r = device_mix(seed ^ (b * 0x2545f4914f6cdd1dULL));
        for (unsigned i = 0; i < block_bytes; ++i) {
            if ((i & 7U) == 0U) r = device_mix(r);
            p[i] = static_cast<unsigned char>(r >> (8U * (i & 7U)));
        }
        const auto put = [&](unsigned at, unsigned short v) { p[at] = static_cast<unsigned char>(v); p[at + 1U] = static_cast<unsigned char>(v >> 8U); };
        put(0U, scale_bits(device_mix(r + 1U), flavor));
        if (kind != 8U) put(2U, scale_bits(device_mix(r + 2U), flavor));
    }
}
struct Buffer {
    void* base = nullptr; std::uint64_t stride = 0, bytes = 0;
    Buffer() = default;
    Buffer(std::uint64_t rows, std::uint64_t row_bytes) : stride((row_bytes + 255U) / 256U * 256U), bytes(rows * stride) {
        check(hipMalloc(&base, bytes), "allocate fixture buffer");
    }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept { *this = std::move(o); }
    Buffer& operator=(Buffer&& o) noexcept { std::swap(base, o.base); std::swap(stride, o.stride); std::swap(bytes, o.bytes); return *this; }
    ~Buffer() { if (base) (void)hipFree(base); }
    template <class T> kr::StridedRows<T> rows() const { return {static_cast<T*>(base), stride}; }
    std::vector<unsigned char> read() const {
        std::vector<unsigned char> out(bytes); check(hipMemcpy(out.data(), base, bytes, hipMemcpyDeviceToHost), "read buffer"); return out;
    }
    void poison(hipStream_t s) const { check(hipMemsetAsync(base, 0xa5, bytes, s), "poison buffer"); }
};
struct Case {
    const char* name; unsigned experts, rows, gate_type, down_type, flavor; bool duplicates, skew, nan_input = false;
};
struct Fixture {
    static constexpr unsigned H = 2560U, I = 640U, K = 10U;
    Case c; hipStream_t stream = nullptr;
    a8::Request request;
    Buffer gate, up, down, input, routes, weights, codes, deltas, sums, status, output, clear, dcodes, ddeltas, dsums, rawdots, csr;
    kr::PrefillRows view;
    explicit Fixture(const Case& cs) : c(cs) {
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
        request.enabled = true; request.input = a8::InputKind::bf16;
        request.shape = {H, I, c.experts, K, c.rows}; request.gate_type = request.up_type = c.gate_type;
        const unsigned gb = c.gate_type == 12U ? 144U : 176U, db = c.down_type == 7U ? 24U : 34U;
        const std::uint64_t gate_blocks = static_cast<std::uint64_t>(c.experts) * I * (H / 256U);
        const std::uint64_t down_blocks = static_cast<std::uint64_t>(c.experts) * H * (I / 32U);
        gate = Buffer(1, gate_blocks * gb); up = Buffer(1, gate_blocks * gb); down = Buffer(1, down_blocks * db);
        const std::uint64_t seed = mix(c.experts * 131U + c.rows * 7U + c.gate_type * 3U + c.down_type + c.flavor * 99U);
        hipLaunchKernelGGL(fill_blocks, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(gate.base), gate_blocks, c.gate_type, seed ^ 1U, c.flavor == 3U ? 0U : c.flavor);
        hipLaunchKernelGGL(fill_blocks, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(up.base), gate_blocks, c.gate_type, seed ^ 2U, c.flavor == 3U ? 0U : c.flavor);
        hipLaunchKernelGGL(fill_blocks, dim3(1024), dim3(256), 0, stream, static_cast<unsigned char*>(down.base), down_blocks, c.down_type, seed ^ 3U, c.flavor == 3U ? 2U : c.flavor);
        check(hipGetLastError(), "fill weights");
        const unsigned T = c.rows;
        input = Buffer(T, 2U * H); routes = Buffer(T, 4U * K); weights = Buffer(T, 4U * K);
        codes = Buffer(T, H); deltas = Buffer(T, 4U * (H / 32U)); sums = Buffer(T, 4U * (H / 32U)); status = Buffer(T, 4U);
        output = Buffer(T, 4U * K * 2U * I); clear = Buffer(T, 4U * H);
        dcodes = Buffer(T, K * I); ddeltas = Buffer(T, 4U * K * (I / 32U)); dsums = Buffer(T, 4U * K * (I / 32U)); rawdots = Buffer(T, 4U * K * H);
        csr = Buffer(1, strix::q51_known_rows::scratch_layout(T, c.experts, K, true).bytes);
        // Host rows: BF16 activations (normal values, outlier channels, edge: tiny/huge/zero blocks).
        std::vector<std::uint16_t> x(static_cast<std::size_t>(input.stride / 2U) * T, 0U);
        std::vector<std::uint32_t> rt(static_cast<std::size_t>(routes.stride / 4U) * T, 0U);
        std::vector<float> rw(static_cast<std::size_t>(weights.stride / 4U) * T, 0.0F);
        std::uint64_t r = seed;
        const auto bf16 = [](float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U); };
        for (unsigned t = 0; t < T; ++t) {
            for (unsigned k = 0; k < H; ++k) {
                r = mix(r);
                float v = static_cast<float>(static_cast<std::int64_t>(r % 2001U) - 1000) / 500.0F;
                if (k % 97U == 5U) v *= 40.0F;
                if (c.flavor == 1U || c.flavor == 2U) {
                    const unsigned block = (k / 32U + t) % 23U;
                    if (block == 1U) v = 0.0F;
                    else if (block == 2U) v *= 1e-30F;
                    else if (block == 3U) v *= c.flavor == 2U ? 1e30F : 1e5F;
                    else if (block == 4U && k % 32U != 7U) v = 0.0F;
                    else if (block == 5U) v *= 1e-36F; // subnormal deltas
                    else if (block == 6U) v = k % 2U ? 0.5F : -0.5F; // x/delta ties
                }
                x[static_cast<std::size_t>(t) * (input.stride / 2U) + k] = bf16(v);
            }
            if (c.nan_input && t == T / 2U) x[static_cast<std::size_t>(t) * (input.stride / 2U) + 77U] = 0x7fc0U;
            for (unsigned s = 0; s < K; ++s) {
                unsigned e;
                for (;;) {
                    r = mix(r);
                    if (c.skew) { const double u = static_cast<double>(r >> 11U) / 9007199254740992.0; e = static_cast<unsigned>(c.experts * u * u * u); }
                    else e = static_cast<unsigned>(r % c.experts);
                    e = std::min(e, c.experts - 1U);
                    bool seen = false;
                    for (unsigned q = 0; q < s; ++q) seen |= rt[static_cast<std::size_t>(t) * (routes.stride / 4U) + q] == e;
                    if (!seen || (c.duplicates && (r >> 40U) % 7U == 0U)) break;
                }
                rt[static_cast<std::size_t>(t) * (routes.stride / 4U) + s] = e;
                rw[static_cast<std::size_t>(t) * (weights.stride / 4U) + s] = 0.05F + static_cast<float>((r >> 20U) % 1000U) / 1000.0F;
            }
        }
        check(hipMemcpy(input.base, x.data(), input.bytes, hipMemcpyHostToDevice), "upload input");
        check(hipMemcpy(routes.base, rt.data(), routes.bytes, hipMemcpyHostToDevice), "upload routes");
        check(hipMemcpy(weights.base, rw.data(), weights.bytes, hipMemcpyHostToDevice), "upload route weights");
        view.input = input.rows<const void>(); view.routes = routes.rows<const unsigned>(); view.route_weights = weights.rows<const float>();
        view.codes = codes.rows<signed char>(); view.deltas = deltas.rows<float>(); view.sums = sums.rows<int>(); view.status = status.rows<unsigned>();
        view.output = output.rows<float>(); view.clear = clear.rows<float>();
        view.down_codes = dcodes.rows<signed char>(); view.down_deltas = ddeltas.rows<float>(); view.down_sums = dsums.rows<int>();
        view.rawdots = rawdots.rows<float>();
        const auto layout = strix::q51_known_rows::scratch_layout(T, c.experts, K, true);
        const auto ptr = [&](std::uint64_t o) { return reinterpret_cast<unsigned*>(static_cast<unsigned char*>(csr.base) + o); };
        view.csr = {ptr(layout.header), ptr(layout.offsets), ptr(layout.touched), ptr(layout.rows), ptr(layout.slots), ptr(layout.batch_experts),
            ptr(layout.batch_begins)};
        check(hipStreamSynchronize(stream), "fixture setup");
    }
    ~Fixture() { if (stream) (void)hipStreamDestroy(stream); }
    std::vector<const Buffer*> written() const { return {&codes, &deltas, &sums, &status, &output, &clear, &dcodes, &ddeltas, &dsums, &rawdots}; }
    void enqueue(int form) {
        kr::enqueue_prefill(request, view, c.rows, H, static_cast<const unsigned char*>(gate.base), static_cast<const unsigned char*>(up.base),
            c.down_type, static_cast<const unsigned char*>(down.base), stream, form);
    }
    std::vector<std::vector<unsigned char>> run(int form) {
        for (const auto* b : written()) b->poison(stream);
        csr.poison(stream);
        enqueue(form);
        check(hipStreamSynchronize(stream), "run prefill");
        std::vector<std::vector<unsigned char>> out;
        for (const auto* b : written()) out.push_back(b->read());
        std::uint32_t summary = 0;
        check(hipMemcpy(&summary, view.csr.summary, 4U, hipMemcpyDeviceToHost), "read summary");
        out.push_back(std::vector<unsigned char>(reinterpret_cast<unsigned char*>(&summary), reinterpret_cast<unsigned char*>(&summary) + 4));
        return out;
    }
};
const char* const names[] = {"codes", "deltas", "sums", "status", "intermediates", "clear", "down_codes", "down_deltas", "down_sums", "rawdots", "summary"};
bool compare(const Case& c) {
    Fixture f(c);
    const auto ref = f.run(0), got = f.run(1);
    bool ok = true;
    std::size_t nan_words = 0, nan_payload = 0, nonfinite = 0;
    for (std::size_t v = 0; v < ref.size(); ++v) {
        // FP32 views: a NaN's sign/payload is not part of the contract (both kernels write the raw
        // non-finite g of a failed (row, slot) only as a diagnostic); NaN positions must match.
        const bool fp32 = v == 1U || v == 4U || v == 5U || v == 7U || v == 9U;
        std::size_t diff = 0, first = ref[v].size();
        for (std::size_t k = 0; k < ref[v].size(); k += fp32 ? 4U : 1U) {
            if (fp32) {
                std::uint32_t a, b; std::memcpy(&a, ref[v].data() + k, 4); std::memcpy(&b, got[v].data() + k, 4);
                const bool an = (a & 0x7fffffffU) > 0x7f800000U, bn = (b & 0x7fffffffU) > 0x7f800000U;
                nan_words += an; nonfinite += (a & 0x7f800000U) == 0x7f800000U && a != 0xa5a5a5a5U;
                if (a == b) continue;
                if (an && bn) { ++nan_payload; continue; }
            } else if (ref[v][k] == got[v][k]) continue;
            ++diff; first = std::min(first, k);
        }
        if (!diff) continue;
        ok = false;
        std::printf("  %s: %zu %s differ, first at byte %zu (ref %02x got %02x)\n", names[v], diff, fp32 ? "words" : "bytes", first,
            ref[v][first], got[v][first]);
    }
    std::uint32_t summary; std::memcpy(&summary, ref.back().data(), 4);
    std::size_t bad_rows = 0;
    for (unsigned t = 0; t < c.rows; ++t) { std::uint32_t s; std::memcpy(&s, ref[3].data() + t * f.status.stride, 4); bad_rows += s != 0U; }
    std::printf("%-34s %s (summary %#x, rows with status %zu/%u, non-finite words %zu, NaN %zu, NaN sign/payload differences %zu)\n",
        c.name, ok ? "PASS" : "FAIL", summary, bad_rows, c.rows, nonfinite, nan_words, nan_payload);
    return ok;
}
void bench(const Case& c, int reps) {
    Fixture f(c);
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    for (int form = 0; form < 2; ++form) {
        f.enqueue(form);
        check(hipStreamSynchronize(f.stream), "warm");
        std::vector<float> ms;
        for (int i = 0; i < reps; ++i) {
            check(hipEventRecord(start, f.stream), "record");
            f.enqueue(form);
            check(hipEventRecord(stop, f.stream), "record");
            check(hipEventSynchronize(stop), "sync");
            float t = 0; check(hipEventElapsedTime(&t, start, stop), "elapsed"); ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        const double macs = static_cast<double>(c.rows) * Fixture::K * 3.0 * Fixture::H * Fixture::I;
        std::printf("%-30s form %d (%s): median %.3f ms (min %.3f) per %u rows, %.2f effective TOPS\n", c.name, form, form ? "wmma" : "dot4",
            ms[ms.size() / 2], ms.front(), c.rows, 2.0 * macs / (ms[ms.size() / 2] * 1e9));
    }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "test";
        if (mode == "bench1") {
            // bench1 <reps> <T> <gate type> <down type> <skew 0/1>
            if (argc < 7) throw std::invalid_argument("bench1 <reps> <T> <gate type> <down type> <skew>");
            const unsigned T = static_cast<unsigned>(std::atoi(argv[3])), g = static_cast<unsigned>(std::atoi(argv[4]));
            const unsigned d = static_cast<unsigned>(std::atoi(argv[5]));
            const std::string name = "E512 T" + std::to_string(T) + (g == 12U ? " q4k" : " q5k") + (d == 7U ? "/q51" : "/q80") +
                (argv[6][0] == '1' ? " skew" : "");
            bench({name.c_str(), 512U, T, g, d, 0U, false, argv[6][0] == '1', false}, std::atoi(argv[2]));
            return 0;
        }
        if (mode == "bench") {
            const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
            for (unsigned T : {256U, 512U, 1024U, 2048U})
                for (const unsigned g : {12U}) for (const unsigned d : {7U}) {
                    const std::string name = "E512 T" + std::to_string(T) + (g == 12U ? " q4k" : " q5k") + (d == 7U ? "/q51" : "/q80");
                    bench({name.c_str(), 512U, T, g, d, 0U, false, false, false}, reps);
                    const std::string skew = name + " skew";
                    bench({skew.c_str(), 512U, T, g, d, 0U, false, true, false}, reps);
                }
            for (const unsigned g : {13U}) for (const unsigned d : {8U}) {
                const std::string name = std::string("E512 T1024 ") + (g == 12U ? "q4k" : "q5k") + (d == 7U ? "/q51" : "/q80");
                bench({name.c_str(), 512U, 1024U, g, d, 0U, false, false, false}, reps);
            }
            return 0;
        }
        std::vector<Case> cases;
        static std::vector<std::string> keep;
        keep.reserve(64);
        for (const unsigned g : {12U, 13U})
            for (const unsigned d : {7U, 8U})
                for (const unsigned flavor : {0U, 1U, 2U, 3U}) {
                    const std::string fmt = std::string(g == 12U ? "q4k" : "q5k") + (d == 7U ? "/q51" : "/q80") +
                        (flavor == 1U ? " edge" : flavor == 2U ? " nonfinite" : flavor == 3U ? " nonfinite-down" : "");
                    keep.push_back("E512 T256 " + fmt); cases.push_back({keep.back().c_str(), 512U, 256U, g, d, flavor, false, false, false});
                    keep.push_back("E64 T128 dup " + fmt); cases.push_back({keep.back().c_str(), 64U, 128U, g, d, flavor, true, false, false});
                    keep.push_back("E40 T48 skew " + fmt); cases.push_back({keep.back().c_str(), 40U, 48U, g, d, flavor, false, true, false});
                }
        keep.push_back("E512 T1024 q4k/q51 skew"); cases.push_back({keep.back().c_str(), 512U, 1024U, 12U, 7U, 0U, false, true, false});
        keep.push_back("E512 T2048 q4k/q51"); cases.push_back({keep.back().c_str(), 512U, 2048U, 12U, 7U, 0U, false, false, false});
        keep.push_back("E512 T2048 q5k/q80 edge skew"); cases.push_back({keep.back().c_str(), 512U, 2048U, 13U, 8U, 1U, false, true, false});
        keep.push_back("E16 T16 q4k/q51 edge"); cases.push_back({keep.back().c_str(), 16U, 16U, 12U, 7U, 1U, false, false, false});
        keep.push_back("E10 T24 dup q5k/q80"); cases.push_back({keep.back().c_str(), 10U, 24U, 13U, 8U, 0U, true, true, false});
        keep.push_back("E512 T64 q4k/q51 nan input"); cases.push_back({keep.back().c_str(), 512U, 64U, 12U, 7U, 0U, false, false, true});
        unsigned failed = 0;
        for (const auto& c : cases) failed += compare(c) ? 0U : 1U;
        std::printf("%zu cases, %u failed\n", cases.size(), failed);
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
