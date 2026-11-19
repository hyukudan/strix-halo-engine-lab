// Device test of the A8 union rows form (strix/moe/expert_union.hpp): for R = 1, 2, 3, 8 rows that share experts, the union
// probe / down (enqueue_probe_union, enqueue_down_*_union) must write byte-for-byte what R C1 calls write (A8 codes, the
// gate / up / activation intermediates, the down codes and the raw dots), and its device plan must equal expert_union::build.
// Synthetic Q4_K gate / up and Q6_K down weights in the Hy4 preset geometry (latent 1536, I 1536, K 16); no model.
#include "strix/hip_moe_a8_down.hpp"
#include "strix/hip_moe_q4k_a8_probe.hpp"
#include "strix/moe/expert_union.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace a8 = strix::moe_q4k_a8;
namespace eu = strix::moe::expert_union;
constexpr std::uint32_t L = 1536, I = 1536, E = 64, K = 16;
int failures = 0;
void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}
struct Buffer {
    void* p = nullptr;
    std::uint64_t bytes = 0;
    explicit Buffer(std::uint64_t n) : bytes(n) {
        check(hipMalloc(&p, n), "hipMalloc");
        check(hipMemset(p, 0, n), "hipMemset");
    }
    ~Buffer() { static_cast<void>(hipFree(p)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    a8::DeviceBytes all() const { return {reinterpret_cast<std::uintptr_t>(p), bytes}; }
    a8::DeviceBytes row(std::uint32_t r, std::uint32_t rows) const {
        const std::uint64_t each = bytes / rows;
        return {reinterpret_cast<std::uintptr_t>(p) + static_cast<std::uintptr_t>(each * r), each};
    }
    std::vector<unsigned char> host() const {
        std::vector<unsigned char> h(bytes);
        check(hipMemcpy(h.data(), p, bytes, hipMemcpyDeviceToHost), "copy back");
        return h;
    }
};
// Random K-quant blocks with a finite, small super-block scale (bytes 0..1 for Q4_K d and 2..3 dmin; 208..209 for Q6_K d).
std::vector<unsigned char> blocks(std::uint64_t count, std::uint32_t block_bytes, std::mt19937& g) {
    std::vector<unsigned char> w(count * block_bytes);
    for (auto& b : w) b = static_cast<unsigned char>(g() & 0xFFU);
    const std::uint16_t small[] = {0x1C00U, 0x2000U, 0x2400U, 0x9C00U};
    for (std::uint64_t i = 0; i < count; ++i) {
        unsigned char* blk = w.data() + i * block_bytes;
        const std::uint16_t d = small[g() % 4U], m = small[g() % 4U];
        if (block_bytes == 144U) {
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &m, 2);
        } else {
            std::memcpy(blk + 208, &d, 2);
        }
    }
    return w;
}

struct Weights {
    Buffer gate, up, down;
    Weights(std::mt19937& g)
        : gate(std::uint64_t{E} * I * (L / 256U) * 144U), up(gate.bytes),
          down(std::uint64_t{E} * L * a8::down_row_bytes(14, I)) {
        const auto gw = blocks(gate.bytes / 144U, 144U, g), uw = blocks(up.bytes / 144U, 144U, g),
                   dw = blocks(down.bytes / 210U, 210U, g);
        check(hipMemcpy(gate.p, gw.data(), gate.bytes, hipMemcpyHostToDevice), "gate");
        check(hipMemcpy(up.p, uw.data(), up.bytes, hipMemcpyHostToDevice), "up");
        check(hipMemcpy(down.p, dw.data(), down.bytes, hipMemcpyHostToDevice), "down");
    }
};
// One path's buffers for R rows (C1 uses row slices, the union the whole extents).
struct Path {
    Buffer input, routes, codes, deltas, sums, status, output, dcodes, ddeltas, dsums, rawdots, dstatus;
    explicit Path(std::uint32_t R)
        : input(R * L * 4ULL), routes(R * K * 4ULL), codes(R * std::uint64_t{L}), deltas(R * (L / 32U) * 4ULL),
          sums(R * (L / 32U) * 4ULL), status(4), output(R * K * 2ULL * I * 4U), dcodes(R * K * std::uint64_t{I}),
          ddeltas(R * K * (I / 32U) * 4ULL), dsums(R * K * (I / 32U) * 4ULL), rawdots(R * K * std::uint64_t{L} * 4U),
          dstatus(4) {}
};
a8::Request probe_request(std::uint32_t rows) {
    a8::Request rq;
    rq.enabled = true;
    rq.approved_numerical_identity = a8::numerical_identity_for_gate_type(12);
    rq.input = a8::InputKind::fp32;
    rq.shape = {L, I, E, K, rows};
    return rq;
}
a8::DownRequest down_request() {
    a8::DownRequest dq;
    dq.enabled = true;
    dq.weight_type = 14;
    dq.approved_numerical_identity = a8::down_numerical_identity(14);
    dq.shape = {L, I, E, K};
    dq.swiglu_limit = 10.0F;
    return dq;
}

void run_case(std::uint32_t R, const Weights& w, std::mt19937& g, hipStream_t s, bool repeat_in_row) {
    std::vector<std::uint32_t> ids(std::size_t{R} * K);
    for (std::uint32_t r = 0; r < R; ++r)
        for (std::uint32_t k = 0; k < K; ++k) ids[std::size_t{r} * K + k] = (k * 5U + r * 3U) % E;  // distinct per row, shared across rows
    if (repeat_in_row) ids[1] = ids[0];
    std::vector<float> x(std::size_t{R} * L);
    std::normal_distribution<float> nd(0.0F, 1.5F);
    for (auto& v : x) v = nd(g);
    Path c1(R), un(R);
    for (Path* p : {&c1, &un}) {
        check(hipMemcpy(p->input.p, x.data(), p->input.bytes, hipMemcpyHostToDevice), "input");
        check(hipMemcpy(p->routes.p, ids.data(), p->routes.bytes, hipMemcpyHostToDevice), "routes");
    }
    Buffer plan(eu::plan_bytes);
    const auto dq = down_request();
    if (!repeat_in_row) {
        for (std::uint32_t r = 0; r < R; ++r) {
            const a8::Bindings b{c1.input.row(r, R), w.gate.all(), w.up.all(), c1.routes.row(r, R), c1.codes.row(r, R),
                                 c1.deltas.row(r, R), c1.sums.row(r, R), c1.status.all(), c1.output.row(r, R)};
            a8::enqueue_probe(probe_request(1), b, s);
            const a8::DownBindings db{c1.output.row(r, R), w.down.all(), c1.routes.row(r, R), c1.dcodes.row(r, R),
                                      c1.ddeltas.row(r, R), c1.dsums.row(r, R), c1.rawdots.row(r, R), c1.dstatus.all(), {0, 0}};
            a8::enqueue_down_activation_quantize(dq, db, s);
            a8::enqueue_down_rawdots(dq, db, s);
        }
    }
    const a8::Bindings ub{un.input.all(), w.gate.all(), w.up.all(), un.routes.all(), un.codes.all(), un.deltas.all(),
                          un.sums.all(), un.status.all(), un.output.all()};
    a8::enqueue_probe_union(probe_request(R), ub, plan.all(), s);
    const a8::DownBindings udb{un.output.all(), w.down.all(), un.routes.all(), un.dcodes.all(), un.ddeltas.all(),
                               un.dsums.all(), un.rawdots.all(), un.dstatus.all(), {0, 0}};
    a8::enqueue_down_activation_quantize_union(dq, udb, R, plan.all(), s);
    a8::enqueue_down_rawdots_union(dq, udb, R, plan.all(), s);
    check(hipStreamSynchronize(s), "sync");
    const std::string tag = "R" + std::to_string(R) + (repeat_in_row ? " repeat" : "");
    std::uint32_t status = 0;
    std::memcpy(&status, un.status.host().data(), 4);
    if (repeat_in_row) {
        expect(status == a8::invalid_route, tag + ": a repeated expert within a row sets invalid_route");
        return;
    }
    expect(status == 0U, tag + ": union probe status 0");
    std::memcpy(&status, un.dstatus.host().data(), 4);
    expect(status == 0U, tag + ": union down status 0");
    std::memcpy(&status, c1.status.host().data(), 4);
    expect(status == 0U, tag + ": C1 status 0");
    const auto ph = plan.host();
    eu::Header h{};
    std::memcpy(&h, ph.data(), sizeof h);
    const auto ref = eu::build(ids, R, K, E);
    bool plan_ok = h.count == ref.size() && h.rows == R && h.top_k == K;
    for (std::size_t u = 0; plan_ok && u < ref.size(); ++u)
        plan_ok = std::memcmp(ph.data() + sizeof(eu::Header) + u * sizeof(eu::Entry), &ref[u], sizeof(eu::Entry)) == 0;
    expect(plan_ok, tag + ": device plan == expert_union::build");
    std::cout << tag << ": " << h.count << " distinct experts of " << R * K << " assignments\n";
    const std::pair<const Buffer*, const Buffer*> same[] = {
        {&c1.codes, &un.codes},     {&c1.deltas, &un.deltas},   {&c1.sums, &un.sums},   {&c1.output, &un.output},
        {&c1.dcodes, &un.dcodes},   {&c1.ddeltas, &un.ddeltas}, {&c1.dsums, &un.dsums}, {&c1.rawdots, &un.rawdots}};
    const char* names[] = {"codes", "deltas", "sums", "intermediates", "down codes", "down deltas", "down sums", "rawdots"};
    for (std::size_t i = 0; i < 8; ++i) expect(same[i].first->host() == same[i].second->host(), tag + ": " + names[i] + " bitwise == R x C1");
}
}  // namespace

int main() {
    try {
        hipStream_t s = nullptr;
        check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "stream");
        std::mt19937 g(20261119U);
        const Weights w(g);
        for (std::uint32_t R : {1U, 2U, 3U, 8U}) run_case(R, w, g, s, false);
        run_case(2U, w, g, s, true);
        check(hipStreamDestroy(s), "stream destroy");
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " union check(s) failed\n";
        return 1;
    }
    std::cout << "A8 union rows device tests: PASS\n";
    return 0;
}
