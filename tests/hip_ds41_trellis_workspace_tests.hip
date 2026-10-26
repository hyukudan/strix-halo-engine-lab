// Run explicitly on a GPU: PACK_SPEC (K4 dense, K5 head). Compare the adapter against legacy S-plan GEMV.
// Interleave generic GEMV with changing split-K layouts, poison Y and mutate X so a stale output cannot pass.
#include "strix/ds41_trellis.hpp"

#include <hip/hip_runtime.h>
#include <strix_trellis.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strix::ds41;
namespace {
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void lib_check(strix_status e) { if (e != STRIX_SUCCESS) throw std::runtime_error(strix_status_string(e)); }
struct Stream {
    hipStream_t p = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&p, hipStreamNonBlocking)); }
    ~Stream() { (void)hipStreamDestroy(p); }
};
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::uint64_t bytes) { check(hipMalloc(&p, bytes)); }
    ~Buffer() { (void)hipFree(p); }
    float* f() const { return static_cast<float*>(p); }
};
void tensor(strix_tensor_desc& t, float* p, std::uint32_t r, std::uint32_t width) {
    t.data = p;
    t.buffer_bytes = std::uint64_t{r} * width * sizeof(float);
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = r; t.shape[1] = width;
    t.strides[0] = width; t.strides[1] = 1;
}
strix_matmul_desc descriptor(const TrellisSite& site, float* x, float* y, std::uint32_t r) {
    strix_matmul_desc d{};
    strix_matmul_desc_init(&d);
    d.weights = static_cast<strix_weights>(site.weights);
    tensor(d.input, x, r, site.columns); tensor(d.output, y, r, site.rows);
    return d;
}
strix_stream stream_desc(hipStream_t s) {
    strix_stream st{}; st.struct_size = sizeof(st); st.abi_version = STRIX_TRELLIS_ABI_VERSION; st.hip_stream = s;
    return st;
}
}

int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument("usage: workspace-tests PACK_SPEC");
    Stream s0, s1;
    Buffer x0(8U * 5120U * 4U), x1(8U * 5120U * 4U);
    Buffer y0(8ULL * 64640U * 4U), y1(8ULL * 64640U * 4U), reference(8ULL * 64640U * 4U);
    TrellisPack pack(argv[1]);
    TrellisContext ctx;
    const std::vector<const TrellisSite*> sites{
        &ctx.prepare(pack, "layers.0.attn.wq_b", {TrellisSlice::Kind::rows, 0, 16384}),
        &ctx.prepare(pack, "layers.0.attn.wo_b", {TrellisSlice::Kind::columns, 0, 4096}),
        &ctx.prepare(pack, "layers.0.ffn.shared_experts.w1", {TrellisSlice::Kind::rows, 0, 1152}),
        &ctx.prepare(pack, "layers.0.ffn.shared_experts.w3", {TrellisSlice::Kind::rows, 0, 1152}),
        &ctx.prepare(pack, "layers.0.ffn.shared_experts.w2", {TrellisSlice::Kind::columns, 0, 1152}),
        &ctx.prepare(pack, "head", {TrellisSlice::Kind::rows, 0, 64640})};
    const auto& qa = ctx.prepare(pack, "layers.0.attn.wq_a");
    const auto& kv = ctx.prepare(pack, "layers.0.attn.wkv");
    const auto& whole_gate = ctx.prepare(pack, "layers.0.ffn.shared_experts.w1");
    const auto& second_gate = ctx.prepare(pack, "layers.0.ffn.shared_experts.w1", {TrellisSlice::Kind::rows, 1152, 1152});
    ctx.reserve(s0.p, 8); ctx.reserve(s1.p, 8);
    const auto reserved = ctx.library_workspace_bytes();
    auto c = static_cast<strix_context>(ctx.library_context());
    std::uint64_t ws_bytes = 256;
    auto query_sites = sites;
    query_sites.insert(query_sites.end(), {&qa, &kv, &whole_gate, &second_gate});
    for (const auto* site : query_sites) for (std::uint32_t r = 1; r <= 8; ++r) {
        auto d = descriptor(*site, x0.f(), reference.f(), r);
        std::uint64_t b = 0;
        lib_check(strix_matmul_workspace_bytes(c, &d, &b)); ws_bytes = std::max(ws_bytes, b);
    }
    Buffer ws(ws_bytes);
    unsigned calls = 0, failures = 0;
    const auto check_output = [&](const TrellisSite& site, std::uint32_t r, Buffer& x, Buffer& y, hipStream_t stream) {
        auto d = descriptor(site, x.f(), reference.f(), r);
        d.workspace = ws.p; d.workspace_bytes = ws_bytes;
        const auto st = stream_desc(stream);
        lib_check(strix_gemv(c, &d, &st));
        std::vector<float> a(std::size_t{r} * site.rows), b(a.size());
        check(hipMemcpyAsync(a.data(), y.p, a.size() * 4U, hipMemcpyDeviceToHost, stream));
        check(hipMemcpyAsync(b.data(), reference.p, b.size() * 4U, hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        ++calls;
        if (std::memcmp(a.data(), b.data(), a.size() * 4U) != 0) {
            ++failures;
            if (failures <= 8) std::printf("DIFF site %s R%u call%u\n", site.key.c_str(), r, calls);
        }
    };
    const auto compare = [&](const TrellisSite& site, std::uint32_t r, Buffer& x, Buffer& y, hipStream_t stream, unsigned seed) {
        std::vector<float> input(std::size_t{r} * site.columns);
        std::uint32_t state = seed;
        for (auto& v : input) {
            state = state * 1664525U + 1013904223U;
            v = static_cast<float>(static_cast<std::int32_t>((state >> 8U) % 20001U) - 10000) / 4096.0F;
        }
        check(hipMemcpyAsync(x.p, input.data(), input.size() * 4U, hipMemcpyHostToDevice, stream));
        check(hipMemsetAsync(y.p, 0xff, std::size_t{r} * site.rows * 4U, stream));
        enqueue_trellis_linear(site, x.f(), y.f(), r, site.columns, site.rows, false, stream);
        check_output(site, r, x, y, stream);
    };
    // Creation of a subsequent plan may clear counters again. The second sweep reuses every cached plan unchanged.
    for (unsigned sweep = 0; sweep < 3; ++sweep) {
        for (unsigned which = 0; which < 2; ++which) {
            for (std::uint32_t r : {1U, 6U, 8U, 2U, 4U, 3U, 7U, 5U, 1U}) {
                for (std::size_t site = 0; site + 1 < sites.size(); ++site) {
                    auto& x = which ? x1 : x0; auto& y = which ? y1 : y0;
                    const auto stream = which ? s1.p : s0.p;
                    const auto seed = 17U + calls * 97U + sweep;
                    compare(*sites[site], r, x, y, stream, seed);
                    compare(*sites.back(), 8U - ((r + sweep) % 8U), x, y, stream, seed + 29U);
                    compare(*sites[site], r, x, y, stream, seed + 43U);
                }
            }
        }
    }
    const auto& qb = *sites[0];
    std::vector<float> a(6U * qb.columns, 0.375F), b(a.size(), -0.625F);
    check(hipMemcpyAsync(x0.p, a.data(), a.size() * 4U, hipMemcpyHostToDevice, s0.p));
    check(hipMemcpyAsync(x1.p, b.data(), b.size() * 4U, hipMemcpyHostToDevice, s1.p));
    for (unsigned replay = 0; replay < 16; ++replay) {
        check(hipMemsetAsync(y0.p, 0xff, 6ULL * qb.rows * 4U, s0.p));
        check(hipMemsetAsync(y1.p, 0xff, 6ULL * qb.rows * 4U, s1.p));
        enqueue_trellis_linear(qb, x0.f(), y0.f(), 6, qb.columns, qb.rows, false, s0.p);
        enqueue_trellis_linear(qb, x1.f(), y1.f(), 6, qb.columns, qb.rows, false, s1.p);
        check_output(qb, 6, x0, y0, s0.p);
        check_output(qb, 6, x1, y1, s1.p);
    }
    if (ctx.library_workspace_bytes() != reserved) {
        std::printf("DIFF workspace allocation grew after reserve\n"); ++failures;
    }
    // New pair storage is independent of generic scratch and single-projection counters.
    for (unsigned sweep = 0; sweep < 3U; ++sweep) {
        for (unsigned which = 0; which < 2U; ++which) {
            auto& x = which ? x1 : x0; auto& y = which ? y1 : y0;
            const auto stream = which ? s1.p : s0.p;
            for (std::uint32_t r : {1U, 6U, 8U, 2U, 4U, 3U, 7U, 5U, 1U}) {
                compare(qa, r, x, y, stream, 391U + calls);
                compare(kv, r, x, y, stream, 791U + calls);
                check(hipMemsetAsync(y.p, 0xff, 8ULL * 64640U * 4U, stream));
                enqueue_trellis_qakv(qa, kv, x.f(), y.f(), y.f() + 8U * 1280U, r, stream);
                check_output(qa, r, x, y, stream);
                // The kv member occupies a separate output span.
                check(hipMemcpyAsync(y.f(), y.f() + 8U * 1280U, r * 512U * 4U, hipMemcpyDeviceToDevice, stream));
                check_output(kv, r, x, y, stream);
                compare(*sites.back(), r, x, y, stream, 113U + calls);
                // Compare K4 row-halves with the whole S16 product, preserving output row stride.
                if (whole_gate.bits == 4U) {
                    enqueue_trellis_linear(*sites[2], x.f(), y.f(), r, 5120U, 2304U, false, stream);
                    enqueue_trellis_linear(second_gate, x.f(), y.f() + 1152U, r, 5120U, 2304U, false, stream);
                    check_output(whole_gate, r, x, y, stream);
                }
            }
        }
    }
    const auto pair_reserved = ctx.library_workspace_bytes();
    for (unsigned which = 0; which < 2U; ++which) {
        auto& x = which ? x1 : x0; auto& y = which ? y1 : y0;
        const auto stream = which ? s1.p : s0.p;
        enqueue_trellis_qakv(qa, kv, x.f(), y.f(), y.f() + 8U * 1280U, 6U, stream);
        check(hipStreamSynchronize(stream));
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        enqueue_trellis_qakv(qa, kv, x.f(), y.f(), y.f() + 8U * 1280U, 6U, stream);
        check(hipStreamEndCapture(stream, &graph));
        check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (unsigned replay = 0; replay < 16U; ++replay) {
            std::vector<float> values(6U * 5120U, (float(replay) - 8.0F) / 32.0F);
            check(hipMemcpyAsync(x.p, values.data(), values.size() * 4U, hipMemcpyHostToDevice, stream));
            check(hipMemsetAsync(y.p, 0xff, 8U * 1792U * 4U, stream));
            check(hipGraphLaunch(exec, stream));
            check_output(qa, 6U, x, y, stream);
            check(hipMemcpyAsync(y.f(), y.f() + 8U * 1280U, 6U * 512U * 4U, hipMemcpyDeviceToDevice, stream));
            check_output(kv, 6U, x, y, stream);
        }
        check(hipGraphExecDestroy(exec)); check(hipGraphDestroy(graph));
    }
    if (ctx.library_workspace_bytes() != pair_reserved) { ++failures; std::puts("DIFF pair workspace grew after warmup"); }
    std::printf("WORKSPACE_TEST %s calls %u mismatches %u library_bytes %llu version %s\n",
        failures ? "FAIL" : "PASS", calls, failures, static_cast<unsigned long long>(reserved), TrellisContext::library_version());
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what()); return 2;
}
