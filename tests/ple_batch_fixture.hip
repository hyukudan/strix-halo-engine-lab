// Standalone exact PLE batch gate. Build command is in tests/ple_batch_build.sh.
#include "../src/hip_q8_rows_gemv.hpp"
#include "strix/model_contract.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <limits>
namespace strix {
#include "../src/hip_ple_rows_site.inc"
}
using namespace strix;
static void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
static void require(bool yes, const char* what) { if (!yes) throw std::runtime_error(what); }
constexpr std::uint32_t poison = 0xa5a5a5a5U;
constexpr std::size_t guard = 256;
struct Buffer {
    std::uint32_t* allocation = nullptr;
    std::size_t words;
    explicit Buffer(std::size_t n) : words(n) {
        check(hipMalloc(reinterpret_cast<void**>(&allocation), (words + 2 * guard) * 4), "allocate");
        reset();
    }
    ~Buffer() { if (allocation) (void)hipFree(allocation); }
    Buffer(const Buffer&) = delete;
    float* floats() const { return reinterpret_cast<float*>(allocation + guard); }
    void reset() const { check(hipMemset(allocation, 0xa5, (words + 2 * guard) * 4), "poison"); }
    std::vector<std::uint32_t> read() const {
        std::vector<std::uint32_t> v(words + 2 * guard);
        check(hipMemcpy(v.data(), allocation, v.size() * 4, hipMemcpyDeviceToHost), "read");
        for (std::size_t i = 0; i < guard; ++i)
            require(v[i] == poison && v[guard + words + i] == poison, "buffer canary changed");
        return v;
    }
};
struct Weights {
    void* device = nullptr;
    Weights(unsigned width, unsigned outputs, unsigned seed) {
        const std::size_t blocks = static_cast<std::size_t>(width / 32) * outputs;
        std::vector<unsigned char> bytes(blocks * 34);
        // Nonzero finite FP16 scales; signed codes include zero and both extremes.
        const std::uint16_t scales[] = {0x0001, 0x0400, 0x2800, 0x3400, 0x3c00, 0xb400};
        for (std::size_t b = 0; b < blocks; ++b) {
            const auto scale = scales[(b + seed) % 6];
            std::memcpy(bytes.data() + b * 34, &scale, 2);
            for (unsigned j = 0; j < 32; ++j) bytes[b * 34 + 2 + j] = static_cast<unsigned char>((b * 37 + j * 13 + seed) & 255U);
        }
        check(hipMalloc(&device, bytes.size()), "weights");
        check(hipMemcpy(device, bytes.data(), bytes.size(), hipMemcpyHostToDevice), "weights upload");
    }
    ~Weights() { if (device) (void)hipFree(device); }
};
static PleRowsSite<64> site(const Weights& w, unsigned width, unsigned outputs,
    const std::uint16_t* in, unsigned in_stride, float* out, unsigned out_stride) {
    PleRowsSite<64> s{};
    s.segments = q8_rows::segments({{w.device, outputs}});
    s.width = width; s.outputs = outputs; s.base = in; s.in_stride = in_stride;
    s.out = out; s.out_stride = out_stride;
    return s;
}
static void baseline(PleRowsSite<64> s, unsigned groups, hipStream_t stream) {
    for (unsigned g = 0; g < groups; ++g) {
        // Today's production lane path, including its cached weight packing and shared scratch.
        q8_rows::launch_lane_waves<PleRowsSite<64>, 16U>(s, stream);
        s.base += 64ULL * s.in_stride; s.out += 64ULL * s.out_stride;
    }
}
static void packed_reference(PleRowsSite<64> s, unsigned groups, float* x, hipStream_t stream) {
    for (unsigned g = 0; g < groups; ++g) {
        hipLaunchKernelGGL(q8_rows::lane_pack_rows_kernel<PleRowsSite<64>>,
            dim3((64U * s.width + 255U) / 256U), dim3(256), 0, stream, s, x + g * 64ULL * s.width);
        s.base += 64ULL * s.in_stride;
    }
}
static void compare(const Buffer& a, const Buffer& b, unsigned rows, unsigned outputs, unsigned stride) {
    const auto av = a.read(), bv = b.read();
    require(av.size() == bv.size(), "size mismatch");
    for (std::size_t i = guard; i < av.size() - guard; ++i) {
        if (av[i] != bv[i]) {
            std::fprintf(stderr, "word %zu reference %08x batch %08x\n", i - guard, av[i], bv[i]);
            throw std::runtime_error("bitwise mismatch");
        }
        const auto at = i - guard;
        if (at / stride >= rows || at % stride >= outputs) require(av[i] == poison, "output padding overwritten");
    }
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static void timing(const PleRowsSite<64>& key, const PleRowsSite<64>& value, unsigned groups,
    float* x, hipStream_t stream) {
    const auto kw = q8_rows::lane_weights(key, stream), vw = q8_rows::lane_weights(value, stream);
    const auto a = [&] {
        auto k = key, v = value;
        for (unsigned g = 0; g < groups; ++g) {
            baseline(k, 1U, stream); baseline(v, 1U, stream);
            k.base += 64ULL * k.in_stride; v.base += 64ULL * v.in_stride;
            k.out += 64ULL * k.out_stride; v.out += 64ULL * v.out_stride;
        }
    };
    const auto b = [&] { ple_batch_launch(key, kw, groups, x, stream); ple_batch_launch(value, vw, groups, x, stream); };
    a(); b(); check(hipStreamSynchronize(stream), "warm");
    hipEvent_t start{}, stop{};
    check(hipEventCreate(&start), "event"); check(hipEventCreate(&stop), "event");
    const auto measure = [&](bool batch) {
        check(hipEventRecord(start, stream), "start");
        if (batch) b(); else a();
        check(hipEventRecord(stop, stream), "stop"); check(hipEventSynchronize(stop), "wait");
        float ms = 0; check(hipEventElapsedTime(&ms, start, stop), "elapsed"); return ms;
    };
    std::vector<float> aa0, aa1, abba_a, abba_b;
    for (unsigned i = 0; i < 7; ++i) { aa0.push_back(measure(false)); aa1.push_back(measure(false)); }
    for (unsigned i = 0; i < 7; ++i) {
        abba_a.push_back(measure(false)); abba_b.push_back(measure(true));
        abba_b.push_back(measure(true)); abba_a.push_back(measure(false));
    }
    std::printf("128-group key+value medians ms: A/A %.6f %.6f ABBA A %.6f B %.6f\n",
        median(aa0), median(aa1), median(abba_a), median(abba_b));
    check(hipEventDestroy(start), "destroy event"); check(hipEventDestroy(stop), "destroy event");
}
static void run(unsigned width, unsigned expanded, unsigned hidden, unsigned groups, unsigned extra_stride, bool bench) {
    const unsigned rows = groups * 64, in_stride = width + extra_stride;
    std::vector<std::uint16_t> input(static_cast<std::size_t>(rows) * in_stride, 0xa5a5);
    const std::uint16_t edge[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f,
        0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x7f81, 0xff91};
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned f = 0; f < width; ++f)
            input[static_cast<std::size_t>(r) * in_stride + f] = static_cast<std::uint16_t>(0x3c00U + ((r * 71U + f * 17U) % 1024U));
        // Keep most rows finite for useful non-NaN coverage; edge placement varies in every group.
        if (r % 4 == 0)
            for (unsigned e = 0; e < sizeof(edge) / sizeof(edge[0]); ++e)
                input[static_cast<std::size_t>(r) * in_stride + (r * 31U + e * 137U) % width] = edge[e];
    }
    Buffer din((input.size() + 1) / 2);
    check(hipMemcpy(din.floats(), input.data(), input.size() * 2, hipMemcpyHostToDevice), "input");
    Weights wk(width, expanded, 1), wv(width, hidden, 7);
    q8_rows::invalidate_lane_weights();
    Buffer scratch(static_cast<std::size_t>(rows) * width), refpack(scratch.words);
    hipStream_t stream{}; check(hipStreamCreate(&stream), "stream");
    // Padding and suffix rows must remain poisoned; odd output strides exercise address mapping.
    for (unsigned outputs : {expanded, hidden}) {
        const unsigned out_stride = outputs + 3;
        Buffer reference(static_cast<std::size_t>(rows + 1) * out_stride), candidate(reference.words);
        auto a = site(outputs == expanded ? wk : wv, width, outputs,
            reinterpret_cast<const std::uint16_t*>(din.floats()), in_stride, reference.floats(), out_stride);
        auto b = a; b.out = candidate.floats();
        if (extra_stride % 8 != 0) require(!ple_batch_admits(a, groups), "odd stride must fall back");
        baseline(a, groups, stream);
        scratch.reset(); refpack.reset();
        const auto weights = q8_rows::lane_weights(b, stream);
        ple_batch_launch(b, weights, groups, scratch.floats(), stream);
        packed_reference(a, groups, refpack.floats(), stream);
        check(hipGetLastError(), "launch"); check(hipStreamSynchronize(stream), "exact sync");
        compare(reference, candidate, rows, outputs, out_stride);
        const auto p = scratch.read(), rp = refpack.read();
        require(p == rp, "packed FP32 mismatch or canary");
        // Independent CPU address/bit oracle: every staged BF16 word is widened by a shift.
        const unsigned chunks = width / 256;
        for (std::size_t t = 0; t < scratch.words; ++t) {
            const auto local = t % (64ULL * width), group = t / (64ULL * width);
            const unsigned mm = local % 8, r = local / 8 % 4, h = local / 32 % 2;
            const unsigned k = local / 64 % chunks, q = local / (64 * chunks) % 16, g = local / (1024 * chunks);
            const unsigned rev = ((q & 1) << 3) | ((q & 2) << 1) | ((q & 4) >> 1) | ((q & 8) >> 3);
            const unsigned f = rev + 16 * (8 * h + mm) + 256 * k;
            const auto expected = static_cast<std::uint32_t>(input[(group * 64 + 4 * g + r) * in_stride + f]) << 16;
            require(p[guard + t] == expected, "CPU packed-bit oracle mismatch");
        }
    }
    const auto after = din.read();
    require(std::memcmp(after.data() + guard, input.data(), input.size() * 2) == 0, "input mutated");
    std::printf("EXACT width %u key %u value %u groups %u stride %u\n", width, expanded, hidden, groups, in_stride);
    if (bench) {
        // Exact checks above have all passed; no timing after a mismatch.
        Buffer ko(static_cast<std::size_t>(rows) * expanded), vo(static_cast<std::size_t>(rows) * hidden);
        auto key = site(wk, width, expanded, reinterpret_cast<const std::uint16_t*>(din.floats()), in_stride, ko.floats(), expanded);
        auto value = site(wv, width, hidden, key.base, in_stride, vo.floats(), hidden);
        timing(key, value, groups, scratch.floats(), stream);
    }
    check(hipStreamDestroy(stream), "destroy stream");
}
static void cpu_addresses() {
    for (unsigned width : {512U, 2560U}) {
        const unsigned chunks = width / 256U;
        std::vector<unsigned char> seen(64ULL * width, 0);
        for (std::uint64_t t = 0; t < 64ULL * width; ++t) {
            const unsigned mm = t % 8U, r = t / 8U % 4U, h = t / 32U % 2U;
            const unsigned k = t / 64U % chunks, q = t / (64U * chunks) % 16U, g = t / (1024U * chunks);
            const unsigned rev = ((q & 1U) << 3U) | ((q & 2U) << 1U) | ((q & 4U) >> 1U) | ((q & 8U) >> 3U);
            const unsigned f = rev + 16U * (8U * h + mm) + 256U * k, row = 4U * g + r;
            require(row < 64U && f < width && !seen[row * width + f]++, "packed mapping is not bijective");
            for (unsigned groups : {1U, 2U, 3U, 64U, 128U})
                for (unsigned group = 0; group < groups; ++group)
                    for (unsigned stride : {width, width + 1U, width + 8U}) {
                        const auto original = group * 64ULL * stride + row * stride + f;
                        const auto grouped = (group * 64ULL + row) * stride + f;
                        require(original == grouped, "row descriptor address");
                        const auto packed = group * 64ULL * width + t;
                        require(packed / (64ULL * width) == group && packed % (64ULL * width) == t, "packed group offset");
                    }
        }
        for (unsigned outputs : {33U, 65U, 2560U, 10240U})
            for (unsigned group = 0; group < 128U; ++group)
                for (unsigned row = 0; row < 64U; ++row)
                    for (unsigned lane = 0; lane < 32U; ++lane) {
                        const unsigned stride = outputs + 3U;
                        require(group * 64ULL * stride + row * stride + lane ==
                            (group * 64ULL + row) * stride + lane, "store descriptor address");
                    }
    }
    std::puts("PASS: CPU descriptor, pack bijection and store addresses");
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--cpu-only") { cpu_addresses(); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--admission-only") {
            // CPU-only precondition gate; never dereference the synthetic aligned addresses.
            PleRowsSite<64> s{};
            s.base = reinterpret_cast<const std::uint16_t*>(0x10000ULL);
            s.segments = q8_rows::segments({{reinterpret_cast<const void*>(0x20000ULL), 10240U}});
            s.width = 2560; s.outputs = 10240; s.in_stride = 2568; s.out_stride = 10240;
            const bool expected = std::string(argv[2]) == "1";
            require(ple_batch_admits(s, 128) == expected, "admission setting mismatch");
            require(!ple_batch_admits(s, 0) && !ple_batch_admits(s, 65536), "group bounds");
            s.in_stride += 1; require(!ple_batch_admits(s, 128), "odd stride admission"); s.in_stride -= 1;
            s.width = 2304; require(!ple_batch_admits(s, 128), "odd chunks admission"); s.width = 2560;
            s.outputs = 64; require(!ple_batch_admits(s, 128), "small outputs admission");
            std::puts("PASS: CPU admission and fallback checks");
            return 0;
        }
        const ModelContract model{};
        // Match engine expanded = hidden_size * hc_count (Qwen3.8-Flash-Next defaults).
        for (unsigned groups : {1U, 2U, 3U, 64U, 128U}) {
            for (unsigned extra : {8U, 1U}) {
                run(512, 65, 33, groups, extra, false);
                run(static_cast<unsigned>(model.ple_embed_dim), static_cast<unsigned>(model.hidden_size * model.hc_count),
                    static_cast<unsigned>(model.hidden_size), groups, extra, false);
            }
        }
        // Time only after the entire exact matrix passes, using aligned production strides.
        run(static_cast<unsigned>(model.ple_embed_dim), static_cast<unsigned>(model.hidden_size * model.hc_count),
            static_cast<unsigned>(model.hidden_size), 128, 8, true);
        std::puts("PASS: all output words, packed words, padding and canaries exact");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
