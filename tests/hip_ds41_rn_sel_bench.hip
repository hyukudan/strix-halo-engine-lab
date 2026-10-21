// RN_WIDE selection sweep (not a ctest): every Q8_0 x A8 decode site of one tensor-parallel rank (q_a, kv, q_b, o_a,
// both o_b K-halves, the head slice) x weight format (Q8_0 base, U6 c-u6) x rows 2..8: the engine kernel as
// enqueue_q8_0_a8 launches it with STRIX_DS41_RN_WIDE off, every registered shape of the multi-row form
// (hip_ds41_rn_wide.hpp), the =1 default shape and the =2 table choice (hip_ds41_rn_table.hpp). Cold: each launch
// reads a different weight copy (copies over >= 640 MB). Two timings per kernel: "op" = median of one event pair per
// launch (the engine's op timing), "stream" = median over passes of the mean per launch of 24 launches back to back
// (one event pair per pass, as the decode stream runs). Every result is compared bitwise with the one-row engine
// kernel run row by row.
// Second part ("qakv"): 40 layers of q_a + kv, cold, A8 + q_a + kv per op (one event pair) and per stream, at rows
// 1..8, engine vs the =1 default vs the table (to compare with the QA_KV pair numbers).
// usage: strix-hip-ds41-rn-sel-bench [iterations] [sites: all | list] [formats: all | q8 | u6] [rows: all | list]
//                                    [parts: sweep,qakv]

#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/hip_ds41_rn_table.hpp"
#include "strix/hip_ds41_rn_wide.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace strix::ds41;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e)); std::exit(2); }
}
template <class T> T* dev(std::size_t n) { T* p = nullptr; check(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "hipMalloc"); return p; }
template <class T> T* dev(const std::vector<T>& h) { T* p = dev<T>(h.size()); check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); return p; }
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    check(hipDeviceSynchronize(), "sync");
    check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "download");
    return h;
}
std::uint16_t to_f16(float x) { return __builtin_bit_cast(std::uint16_t, static_cast<_Float16>(x)); }

struct Times { double op_us, stream_us; };
// op: median of one event pair per launch; stream: median over passes of the per-launch mean of `burst` launches.
Times time_us(int iterations, const std::function<void(int)>& launch) {
    hipEvent_t a, b;
    check(hipEventCreate(&a), "event");
    check(hipEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) launch(i);
    std::vector<float> t;
    for (int i = 0; i < iterations; ++i) {
        check(hipEventRecord(a, nullptr), "record");
        launch(i + 3);
        check(hipEventRecord(b, nullptr), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        t.push_back(ms * 1000.0F);
    }
    std::sort(t.begin(), t.end());
    const double op = t[t.size() / 2];
    constexpr int burst = 24;
    std::vector<float> s;
    for (int p = 0; p < std::max(5, iterations / 8); ++p) {
        check(hipEventRecord(a, nullptr), "record");
        for (int i = 0; i < burst; ++i) launch(p * burst + i);
        check(hipEventRecord(b, nullptr), "record");
        check(hipEventSynchronize(b), "sync");
        float ms = 0;
        check(hipEventElapsedTime(&ms, a, b), "elapsed");
        s.push_back(ms * 1000.0F / burst);
    }
    std::sort(s.begin(), s.end());
    (void)hipEventDestroy(a);
    (void)hipEventDestroy(b);
    return {op, s[s.size() / 2]};
}

// One rank's view of a site: count outputs of `width` inputs (per_group outputs per activation group); o_b reads the
// K-half [k_offset, k_offset + width) of rows of row_width values.
struct Site { const char* name; std::uint32_t width, count, per_group, row_width, k_offset; };
struct Format { const char* name; std::uint32_t packed, block_bytes; };

std::vector<std::uint8_t> random_rows(std::mt19937& rng, const Format& f, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.001F, 0.05F);
    for (auto& b : host) b = static_cast<std::uint8_t>(byte(rng));
    for (std::size_t blk = 0; blk < bytes / f.block_bytes; ++blk) {
        const auto d = to_f16(sc(rng));
        host[blk * f.block_bytes] = static_cast<std::uint8_t>(d & 0xffU);
        host[blk * f.block_bytes + 1] = static_cast<std::uint8_t>(d >> 8U);
        if (f.packed == 0U)  // Q8_0 codes in -127..127
            for (std::size_t k = 2; k < 34; ++k)
                if (host[blk * 34 + k] == 0x80U) host[blk * 34 + k] = 0x81U;
    }
    return host;
}

std::string shape_name(const RnWideShape& s) {
    return std::to_string(s.outputs_per_wave) + "," + std::to_string(s.waves) + "," + (s.lds ? "1" : "0") + (s.flat ? ",flat" : "") +
           (s.form >= 2U ? "," + std::to_string(s.form) : "");
}

const RnWideShape sweep_shapes[] = {
    {16, 1, true, false},  {16, 2, true, false},  {8, 4, true, false},   {8, 1, true, false},   {8, 2, true, false},
    {16, 4, true, false},  {8, 1, false, false},  {8, 2, false, false},  {8, 4, false, false},  {16, 1, false, false},
    {16, 2, false, false}, {16, 4, false, false}, {4, 4, false, true},   {8, 4, false, true},   {8, 2, false, true},
    {4, 8, false, true},   {8, 4, false, false, 3}, {2, 4, false, false, 3}};

void sweep(const Site& s, const Format& f, const std::vector<std::uint32_t>& row_list, int iterations) {
    std::mt19937 rng(20261020);
    const std::uint32_t groups = s.count / s.per_group, full = s.row_width ? s.row_width : s.width;
    const std::size_t alloc = std::size_t{full} / 32U * f.block_bytes * s.count;  // the tensor (o_b: whole rows)
    const std::size_t read = std::size_t{s.width} / 32U * f.block_bytes * s.count;  // bytes the site reads
    const std::size_t copies = std::max<std::size_t>(4, (640ULL << 20) / alloc + 1);
    const auto host = random_rows(rng, f, alloc);
    auto* w = dev<std::uint8_t>(alloc * copies);
    for (std::size_t c = 0; c < copies; ++c) check(hipMemcpy(w + c * alloc, host.data(), alloc, hipMemcpyHostToDevice), "upload");
    const std::uint32_t x_stride = groups * s.width;
    for (const std::uint32_t rows : row_list) {
        std::vector<std::int8_t> codes(std::size_t{rows} * x_stride);
        std::vector<float> deltas(std::size_t{rows} * x_stride / 32U);
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> dl(0.001F, 0.1F);
        for (auto& c : codes) c = static_cast<std::int8_t>(code(rng));
        for (auto& d : deltas) d = dl(rng);
        auto* dc = dev(codes);
        auto* dd = dev(deltas);
        const std::size_t outs = std::size_t{rows} * s.count;
        auto* out = dev<float>(outs);
        auto view = [&](int i) {
            Q8A8View v;
            v.weight = w + (static_cast<std::size_t>(i) % copies) * alloc;
            v.width = s.width; v.count = s.count; v.per_group = s.per_group; v.first = 0;
            v.row_width = s.row_width; v.k_offset = s.k_offset;
            v.codes = dc; v.deltas = dd; v.rows = rows; v.x_stride = x_stride; v.out_stride = s.count; v.out = out;
            v.packed = f.packed;
            return v;
        };
        for (std::uint32_t r = 0; r < rows; ++r) {  // reference: the one-row engine kernel, row by row
            Q8A8View v = view(0);
            v.rows = 1;
            v.codes = dc + std::size_t{r} * x_stride;
            v.deltas = dd + std::size_t{r} * x_stride / 32U;
            v.out = out + std::size_t{r} * s.count;
            enqueue_q8_0_a8(v, nullptr);
        }
        const auto ref = down(out, outs);
        const auto report = [&](const char* kernel, const std::string& shape, const std::function<void(int)>& launch) {
            check(hipMemset(out, 0xff, outs * 4), "clear");
            launch(0);
            const auto got = down(out, outs);
            const bool exact = std::memcmp(got.data(), ref.data(), outs * 4) == 0;
            const Times t = time_us(iterations, launch);
            std::printf("{\"site\":\"%s\",\"format\":\"%s\",\"rows\":%u,\"kernel\":\"%s\",\"shape\":\"%s\",\"bytes\":%zu,\"op_us\":%.2f,"
                        "\"stream_us\":%.2f,\"op_gbs\":%.1f,\"stream_gbs\":%.1f,\"exact\":%s}\n",
                        s.name, f.name, rows, kernel, shape.c_str(), read, t.op_us, t.stream_us, static_cast<double>(read) / (t.op_us * 1e3),
                        static_cast<double>(read) / (t.stream_us * 1e3), exact ? "true" : "false");
            std::fflush(stdout);
        };
        // STRIX_DS41_RN_WIDE must be unset for this run: enqueue_q8_0_a8 is then the engine kernel.
        report("engine", "old", [&](int i) { enqueue_q8_0_a8(view(i), nullptr); });
        for (const auto& sh : sweep_shapes) {
            if (!rn_wide_supported(view(0), sh)) continue;
            report("rn", shape_name(sh), [&](int i) { enqueue_q8_0_a8_rn(view(i), sh, nullptr); });
        }
        if (rows >= 2) {
            const RnWideShape def = rn_wide_default_shape(view(0));
            if (rn_wide_supported(view(0), def))
                report("rn-default", shape_name(def), [&](int i) { enqueue_q8_0_a8_rn(view(i), def, nullptr); });
            RnWideShape pick;
            const bool rn = rn_table_select(view(0), pick) && rn_wide_supported(view(0), pick);
            report("table", rn ? shape_name(pick) : "old", [&](int i) {
                if (rn) enqueue_q8_0_a8_rn(view(i), pick, nullptr);
                else enqueue_q8_0_a8(view(i), nullptr);
            });
        }
        (void)hipFree(dc); (void)hipFree(dd); (void)hipFree(out);
    }
    (void)hipFree(w);
}

// 40 layers of q_a (5120 -> 1280) and kv (5120 -> 512): A8 of x, then q_a and kv (one A8, as the QA_KV pair does).
void qakv(const Format& f, int iterations) {
    constexpr unsigned H = 5120, QA = 1280, KV = 512, layers = 40;
    std::mt19937 rng(1234);
    const std::size_t qa_bytes = std::size_t{H} / 32U * f.block_bytes * QA, kv_bytes = std::size_t{H} / 32U * f.block_bytes * KV;
    std::vector<std::uint8_t*> qa(layers), kv(layers);
    for (unsigned L = 0; L < layers; ++L) {
        qa[L] = dev(random_rows(rng, f, qa_bytes));
        kv[L] = dev(random_rows(rng, f, kv_bytes));
    }
    std::vector<float> xh(8U * H);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    for (auto& v : xh) v = nd(rng);
    auto* x = dev(xh);
    auto* codes = dev<std::int8_t>(8U * H);
    auto* deltas = dev<float>(8U * H / 32U);
    auto* sums = dev<std::int32_t>(8U * H / 32U);
    auto* out_a = dev<float>(8U * QA);
    auto* out_b = dev<float>(8U * KV);
    const auto view = [&](std::uint8_t* wt, unsigned count, unsigned R, float* out) {
        Q8A8View v;
        v.weight = wt; v.width = H; v.count = count; v.per_group = count; v.codes = codes; v.deltas = deltas; v.rows = R;
        v.x_stride = H; v.out_stride = count; v.out = out; v.packed = f.packed;
        return v;
    };
    const auto per_op = [&](const std::function<void(unsigned)>& fn) {
        hipEvent_t a, b;
        check(hipEventCreate(&a), "event");
        check(hipEventCreate(&b), "event");
        for (unsigned L = 0; L < layers; ++L) fn(L);
        std::vector<double> op, st;
        for (int i = 0; i < iterations; ++i)
            for (unsigned L = 0; L < layers; ++L) {
                check(hipEventRecord(a, nullptr), "record");
                fn(L);
                check(hipEventRecord(b, nullptr), "record");
                check(hipEventSynchronize(b), "sync");
                float ms = 0;
                check(hipEventElapsedTime(&ms, a, b), "elapsed");
                op.push_back(ms * 1000.0);
            }
        for (int i = 0; i < iterations; ++i) {
            check(hipEventRecord(a, nullptr), "record");
            for (unsigned L = 0; L < layers; ++L) fn(L);
            check(hipEventRecord(b, nullptr), "record");
            check(hipEventSynchronize(b), "sync");
            float ms = 0;
            check(hipEventElapsedTime(&ms, a, b), "elapsed");
            st.push_back(ms * 1000.0 / layers);
        }
        (void)hipEventDestroy(a);
        (void)hipEventDestroy(b);
        std::sort(op.begin(), op.end());
        std::sort(st.begin(), st.end());
        return Times{op[op.size() / 2], st[st.size() / 2]};
    };
    for (unsigned R = 1; R <= 8; ++R) {
        enqueue_quantize_a8(x, H, R, codes, deltas, sums, nullptr);
        // mode 0: engine kernels; 1: the =1 default shape; 2: the table
        std::vector<float> ref_a, ref_b;
        for (int mode = 0; mode < 3; ++mode) {
            if (R == 1 && mode > 0) break;
            const auto product = [&](const Q8A8View& v) {
                RnWideShape s;
                if (mode == 1) s = rn_wide_default_shape(v);
                const bool rn = mode == 1 ? rn_wide_supported(v, s) : mode == 2 ? rn_table_select(v, s) && rn_wide_supported(v, s) : false;
                if (rn) enqueue_q8_0_a8_rn(v, s, nullptr);
                else enqueue_q8_0_a8(v, nullptr);
                return rn ? shape_name(s) : std::string("old");
            };
            const std::string sa = product(view(qa[0], QA, R, out_a)), sb = product(view(kv[0], KV, R, out_b));
            const auto ga = down(out_a, std::size_t{R} * QA), gb = down(out_b, std::size_t{R} * KV);
            if (mode == 0) { ref_a = ga; ref_b = gb; }
            const bool exact = std::memcmp(ga.data(), ref_a.data(), ga.size() * 4) == 0 && std::memcmp(gb.data(), ref_b.data(), gb.size() * 4) == 0;
            const Times both = per_op([&](unsigned L) {
                enqueue_quantize_a8(x, H, R, codes, deltas, sums, nullptr);
                product(view(qa[L], QA, R, out_a));
                product(view(kv[L], KV, R, out_b));
            });
            const Times qa_only = per_op([&](unsigned L) { product(view(qa[L], QA, R, out_a)); });
            const Times kv_only = per_op([&](unsigned L) { product(view(kv[L], KV, R, out_b)); });
            const char* names[] = {"engine", "rn-default", "table"};
            std::printf("{\"part\":\"qakv\",\"format\":\"%s\",\"rows\":%u,\"kernel\":\"%s\",\"q_a\":\"%s\",\"kv\":\"%s\",\"a8_qa_kv_op_us\":%.2f,"
                        "\"a8_qa_kv_stream_us\":%.2f,\"qa_op_us\":%.2f,\"qa_stream_us\":%.2f,\"kv_op_us\":%.2f,\"kv_stream_us\":%.2f,\"exact\":%s}\n",
                        f.name, R, names[mode], sa.c_str(), sb.c_str(), both.op_us, both.stream_us, qa_only.op_us, qa_only.stream_us,
                        kv_only.op_us, kv_only.stream_us, exact ? "true" : "false");
            std::fflush(stdout);
        }
    }
    for (unsigned L = 0; L < layers; ++L) { (void)hipFree(qa[L]); (void)hipFree(kv[L]); }
    (void)hipFree(x); (void)hipFree(codes); (void)hipFree(deltas); (void)hipFree(sums); (void)hipFree(out_a); (void)hipFree(out_b);
}

bool listed(const std::string& filter, const std::string& name) {
    if (filter == "all") return true;
    std::size_t p = 0;
    while (p <= filter.size()) {
        const std::size_t e = std::min(filter.find(',', p), filter.size());
        if (filter.compare(p, e - p, name) == 0) return true;
        p = e + 1;
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    if (rn_wide_mode() != 0) { std::fprintf(stderr, "run with STRIX_DS41_RN_WIDE unset (the engine column is the flag-off kernel)\n"); return 2; }
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 40;
    const std::string site_filter = argc > 2 ? argv[2] : "all", format_filter = argc > 3 ? argv[3] : "all";
    const std::string row_filter = argc > 4 ? argv[4] : "all", parts = argc > 5 ? argv[5] : "sweep,qakv";
    std::vector<std::uint32_t> rows;
    for (std::uint32_t r = 2; r <= 8; ++r)
        if (listed(row_filter, std::to_string(r))) rows.push_back(r);
    // One rank of the tensor-parallel decode (--tp-attention): q_a / kv whole, q_b heads [32r, 32r + 32), o_a groups
    // [4r, 4r + 4), o_b this rank's K-half of all 5,120 rows of 8,192 (r0: columns 0..4095, r1: 4096..8191), the head
    // this rank's vocabulary half (64,640 rows).
    const Site sites[] = {{"q_a", 5120, 1280, 1280, 0, 0},     {"kv", 5120, 512, 512, 0, 0},       {"q_b", 1280, 16384, 16384, 0, 0},
                          {"o_a", 4096, 4096, 1024, 0, 0},     {"o_b0", 4096, 5120, 5120, 8192, 0}, {"o_b1", 4096, 5120, 5120, 8192, 4096},
                          {"head", 5120, 64640, 64640, 0, 0}};
    const Format formats[] = {{"q8", 0, 34}, {"u6", 6, 26}};
    for (const auto& f : formats) {
        if (!listed(format_filter, f.name)) continue;
        if (parts.find("sweep") != std::string::npos)
            for (const auto& s : sites)
                if (listed(site_filter, s.name)) sweep(s, f, rows, iterations);
        if (parts.find("qakv") != std::string::npos) qakv(f, std::max(4, iterations / 8));
    }
    return 0;
}
