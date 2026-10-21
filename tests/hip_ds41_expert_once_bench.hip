// Routed experts of a DSpark verify window (R rows) at the runner's launch shapes: what is read per window.
// Real weights of one rank, real recorded routes (the rank app's --routes output, "# seq" headers allowed),
// windows of R consecutive tokens of one sequence sampled evenly over the file. Per R it reports:
//   pairs (row, expert) of this rank, unique experts, the most rows of one expert, and the bytes
//   per window as unique-expert bytes, per-pair bytes, and the bytes the kernels issue (an expert with
//   more entries than a form stages at a time re-reads its rows once per part);
//   cold timings of build_routes, gate/up and down, GB/s over the unique-expert bytes;
//   a streaming-read ceiling over exactly the same unique byte ranges (one launch for gate+up, one
//   for down, the same cold start), i.e. the bandwidth floor of the window's bytes;
//   with --check, every form's outputs bitwise against the default form.
// usage: strix-hip-ds41-expert-once-bench MODEL.gguf --routes FILE [--layers 5,20,35] [--rank R]
//        [--rows 1,2,4,6,8] [--cases N] [--reps N] [--forms -1,0,3,4] [--check]

#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_routes_fast.hpp"
#include "strix/ds41_layer.hpp"

#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strix::ds41::WeightFormat;

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}
int failures = 0;

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) { check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc"); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)hipFree(data); }
    void upload(const std::vector<T>& v) { check(hipMemcpy(data, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> out(count);
        check(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }
    void fill(int byte) { check(hipMemset(data, byte, count * sizeof(T)), "memset"); }
};

void read_into(const std::string& path, std::uint64_t offset, std::uint64_t bytes, std::uint8_t* out) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    std::uint64_t done = 0;
    while (done < bytes) {
        const auto n = ::pread(fd, out + done, std::min<std::uint64_t>(bytes - done, 1ULL << 28), static_cast<off_t>(offset + done));
        if (n <= 0) { ::close(fd); throw std::runtime_error("short read in " + path); }
        done += static_cast<std::uint64_t>(n);
    }
    ::close(fd);
}

struct Tensor {
    const strix::gguf::TensorInfo* info = nullptr;
    WeightFormat format = WeightFormat::q8_0;
    std::uint32_t width = 0, outputs = 0, experts = 1;
    std::uint64_t row_bytes = 0, stride = 0;
};
Tensor tensor(const std::map<std::string, const strix::gguf::TensorInfo*>& tensors, const std::string& name) {
    const auto found = tensors.find(name);
    if (found == tensors.end()) throw std::runtime_error("missing tensor " + name);
    Tensor t;
    t.info = found->second;
    const auto type = t.info->ggml_type;
    if (type != 16U && type != 10U) throw std::runtime_error(name + " is not IQ2_XXS / Q2_K");
    t.format = type == 16U ? WeightFormat::iq2_xxs : WeightFormat::q2_k;
    t.width = static_cast<std::uint32_t>(t.info->dimensions.at(0));
    t.outputs = static_cast<std::uint32_t>(t.info->dimensions.at(1));
    t.experts = static_cast<std::uint32_t>(t.info->dimensions.at(2));
    t.row_bytes = t.width / strix::ds41::block_elements(t.format) * strix::ds41::block_bytes(t.format);
    t.stride = t.row_bytes * t.outputs;
    if (t.stride * t.experts != t.info->byte_size) throw std::runtime_error(name + " byte size does not match its geometry");
    return t;
}

__global__ void flush_kernel(const int* buffer, std::size_t count, int value, int* sink) {
    int sum = 0;
    for (std::size_t i = blockIdx.x * 256ULL + threadIdx.x; i < count; i += gridDim.x * 256ULL) sum += buffer[i];
    if (sum == value) sink[0] = sum;
}
struct Flush {
    Device<int> buffer{64U * 1024U * 1024U};  // 256 MB
    Device<int> sink{1};
    Flush() { buffer.fill(0); check(hipDeviceSynchronize(), "flush buffer"); }
    void operator()(hipStream_t s) {
        hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, s, buffer.data, buffer.count, 0x5a5a5a5a, sink.data);
        check(hipGetLastError(), "flush");
    }
};

// Streaming read of `count` expert ranges of `stride` bytes (one block-row of 256 threads per 64 KiB chunk),
// int4 loads, a sum that is never stored unless equal to a sentinel: the window's unique bytes at the
// best rate a plain reader gets from the same addresses.
__global__ __launch_bounds__(256) void stream_kernel(const unsigned char* a, const unsigned char* b, const std::uint32_t* experts,
                                                      unsigned long long stride, unsigned chunks_per_expert, int* sink) {
    const unsigned e = blockIdx.y, chunk = blockIdx.x;
    if (chunk >= chunks_per_expert) return;
    const unsigned long long begin = static_cast<unsigned long long>(experts[e]) * stride + static_cast<unsigned long long>(chunk) * 65536ULL;
    const unsigned long long end = min(begin + 65536ULL, static_cast<unsigned long long>(experts[e]) * stride + stride);
    int sum = 0;
    for (const unsigned char* base : {a, b}) {
        if (!base) continue;
        for (unsigned long long at = begin + threadIdx.x * 16ULL; at + 16ULL <= end; at += 256ULL * 16ULL) {
            const int4 v = *reinterpret_cast<const int4*>(base + at);
            sum += v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (sum == 0x13572468) sink[0] = sum;
}

__global__ void empty_kernel(int* sink) { if (threadIdx.x == 999U) sink[0] = 1; }

struct Timer {
    hipEvent_t e[8]{};
    Timer() { for (auto& x : e) check(hipEventCreate(&x), "event"); }
    ~Timer() { for (auto& x : e) (void)hipEventDestroy(x); }
    void mark(unsigned i, hipStream_t s) { check(hipEventRecord(e[i], s), "event"); }
    double ms(unsigned a, unsigned b) { float t = 0; check(hipEventSynchronize(e[b]), "sync"); check(hipEventElapsedTime(&t, e[a], e[b]), "elapsed"); return t; }
};

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2U]; }

std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream in(s);
    for (std::string item; std::getline(in, item, ',');) out.push_back(std::stoi(item));
    return out;
}

// Recorded routes: sequences of tokens, each token's ids per layer.
struct Token { std::map<unsigned, std::vector<std::uint32_t>> ids; };
std::vector<std::vector<Token>> load_routes(const std::string& path, const std::vector<int>& layers, unsigned top_k) {
    std::vector<std::vector<Token>> seqs;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        if (line[0] == '#') { seqs.emplace_back(); continue; }
        if (seqs.empty()) seqs.emplace_back();
        Token t;
        std::stringstream fields(line);
        for (std::string field; fields >> field;) {
            const auto colon = field.find(':');
            if (colon == std::string::npos) continue;
            const auto layer = static_cast<int>(std::stoul(field.substr(0, colon)));
            if (std::find(layers.begin(), layers.end(), layer) == layers.end()) continue;
            std::vector<std::uint32_t> ids;
            for (const auto id : parse_ints(field.substr(colon + 1))) ids.push_back(static_cast<std::uint32_t>(id));
            if (ids.size() == top_k) t.ids[static_cast<unsigned>(layer)] = ids;
        }
        seqs.back().push_back(std::move(t));
    }
    return seqs;
}

struct Window {
    std::vector<std::uint32_t> ids;
    std::uint32_t pairs = 0, unique = 0, most = 0;
    std::map<std::uint32_t, std::uint32_t> count;  // this rank's expert -> entries
};
// `cases` windows of R consecutive tokens inside one sequence, starts spread evenly over all tokens.
std::vector<Window> windows(const std::vector<std::vector<Token>>& seqs, unsigned layer, unsigned rows, unsigned cases,
                            std::uint32_t first, std::uint32_t end) {
    std::vector<std::pair<std::size_t, std::size_t>> starts;  // (sequence, token) where a full window fits
    for (std::size_t s = 0; s < seqs.size(); ++s)
        for (std::size_t t = 0; t + rows <= seqs[s].size(); ++t) starts.emplace_back(s, t);
    std::vector<Window> out;
    if (starts.empty()) return out;
    for (unsigned c = 0; c < cases; ++c) {
        const auto [s, t0] = starts[(static_cast<std::size_t>(c) * starts.size()) / cases];
        Window w;
        bool ok = true;
        for (unsigned r = 0; r < rows; ++r) {
            const auto it = seqs[s][t0 + r].ids.find(layer);
            if (it == seqs[s][t0 + r].ids.end()) { ok = false; break; }
            w.ids.insert(w.ids.end(), it->second.begin(), it->second.end());
        }
        if (!ok) continue;
        for (const auto id : w.ids)
            if (id >= first && id < end) { ++w.count[id]; ++w.pairs; }
        w.unique = static_cast<std::uint32_t>(w.count.size());
        for (const auto& [id, n] : w.count) w.most = std::max(w.most, n);
        out.push_back(std::move(w));
    }
    return out;
}

struct RankExperts {
    Tensor gate_t, up_t, down_t;
    std::uint32_t first = 0, end = 0, hidden = 0, inter = 0, top_k = 6, split = 192;
    Device<std::uint8_t> gate, up, down;
    RankExperts(const Tensor& g, const Tensor& u, const Tensor& d, std::uint32_t f, std::uint32_t e, const std::string& path)
        : gate_t(g), up_t(u), down_t(d), first(f), end(e), hidden(g.width), inter(g.outputs), gate(g.stride * (e - f)),
          up(u.stride * (e - f)), down(d.stride * (e - f)) {
        const auto load = [&](const Tensor& t, Device<std::uint8_t>& dev) {
            std::vector<std::uint8_t> host(t.stride * (e - f));
            read_into(path, t.info->absolute_offset + t.stride * f, host.size(), host.data());
            dev.upload(host);
        };
        load(g, gate); load(u, up); load(d, down);
    }
    [[nodiscard]] double gate_up_bytes() const { return 2.0 * static_cast<double>(gate_t.stride); }
    [[nodiscard]] double down_bytes() const { return static_cast<double>(down_t.stride); }
};

struct RoutedBuffers {
    Device<std::uint32_t> ids, experts, begins, rows_out, slots, order, low_counts, counts, unique;
    Device<std::int8_t> codes; Device<float> deltas; Device<std::int32_t> sums;
    Device<std::int8_t> act_codes; Device<float> act_deltas; Device<std::int32_t> act_sums;
    Device<float> raw;
    RoutedBuffers(std::uint32_t rows, std::uint32_t top_k, std::uint32_t hidden, std::uint32_t inter)
        : ids(rows * top_k), experts(rows * top_k + 1U), begins(rows * top_k + 1U), rows_out(rows * top_k), slots(rows * top_k),
          order(rows * top_k), low_counts(rows), counts(2), unique(rows * top_k), codes(static_cast<std::size_t>(rows) * hidden),
          deltas(rows * hidden / 32U), sums(rows * hidden / 32U), act_codes(static_cast<std::size_t>(rows) * top_k * inter),
          act_deltas(rows * top_k * inter / 32U), act_sums(rows * top_k * inter / 32U), raw(static_cast<std::size_t>(rows) * top_k * hidden) {}
    void clear_outputs() { act_codes.fill(0x5b); act_deltas.fill(0x5b); act_sums.fill(0x5b); raw.fill(0x5b); }
};

// The runner's three launches (device_forward routed_experts): build_routes, gate/up, down. Marks 0..3.
// --tight: launch grids over the window's real union (host-known here) instead of rows * top_k slots, to
// measure what the padded (early-exit) blocks of the device-built routes cost.
std::uint32_t tight_count = 0;
void enqueue_routed(const RankExperts& e, RoutedBuffers& b, std::uint32_t rows, hipStream_t s, Timer* timer) {
    using namespace strix::ds41;
    if (timer) timer->mark(0, s);
    BuildRoutesView bv;
    bv.ids = b.ids.data; bv.rows = rows; bv.top_k = e.top_k; bv.split = e.split; bv.first = e.first; bv.end = e.end; bv.base = e.first;
    bv.experts = b.experts.data; bv.begins = b.begins.data; bv.rows_out = b.rows_out.data; bv.slots = b.slots.data;
    bv.order = b.order.data; bv.low_counts = b.low_counts.data; bv.counts = b.counts.data;
    enqueue_build_routes(bv, s);
    if (timer) timer->mark(1, s);
    const std::uint32_t slots = tight_count ? tight_count : rows * e.top_k;
    const RoutesView gate_routes{b.experts.data, b.begins.data, b.rows_out.data, b.slots.data, slots, rows, b.counts.data};
    const RoutesView down_routes{b.experts.data, b.begins.data, b.slots.data, b.slots.data, slots, rows, b.counts.data};
    GateUpView gu;
    gu.gate = e.gate.data; gu.up = e.up.data; gu.expert_stride_bytes = e.gate_t.stride; gu.row_bytes = e.gate_t.row_bytes;
    gu.hidden = e.hidden; gu.inter = e.inter; gu.routes = gate_routes; gu.codes = b.codes.data; gu.deltas = b.deltas.data;
    gu.limit = 10.0F; gu.act_codes = b.act_codes.data; gu.act_deltas = b.act_deltas.data; gu.act_sums = b.act_sums.data;
    enqueue_gate_up(e.gate_t.format, gu, s);
    if (timer) timer->mark(2, s);
    RoutesProductView dv;
    dv.weights = e.down.data; dv.expert_stride_bytes = e.down_t.stride; dv.row_bytes = e.down_t.row_bytes; dv.width = e.inter;
    dv.outputs = e.hidden; dv.routes = down_routes; dv.codes = b.act_codes.data; dv.deltas = b.act_deltas.data; dv.out = b.raw.data;
    enqueue_routes_product(e.down_t.format, dv, s);
    if (timer) timer->mark(3, s);
}

void quantize_rows(RoutedBuffers& b, std::uint32_t rows, std::uint32_t hidden, unsigned seed, hipStream_t s) {
    std::vector<float> x(static_cast<std::size_t>(rows) * hidden);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (auto& v : x) { state = state * 1664525U + 1013904223U; v = static_cast<float>(static_cast<std::int32_t>(state >> 8U) - (1 << 23)) / static_cast<float>(1 << 23); }
    Device<float> dx(x.size());
    dx.upload(x);
    strix::ds41::enqueue_quantize_a8(dx.data, hidden, rows, b.codes.data, b.deltas.data, b.sums.data, s);
    check(hipStreamSynchronize(s), "quantize");
}

// Staged-row caps of the forms (hip_ds41_formats.hip form_shape) and the defaults at R rows.
unsigned form_cap(int form) { return form == 3 ? 4U : form == 4 || form == 5 ? 2U : form == 6 ? 1U : 8U; }
int default_gate_up(unsigned rows) { return rows > 4U ? 3 : 0; }
int default_down(unsigned rows) { return rows > 2U ? 4 : 0; }

struct Options {
    std::string model, routes;
    std::vector<int> layers{20}, forms{-1};
    unsigned rank = 0, cases = 64, reps = 2;
    std::vector<int> rows{1, 2, 4, 6, 8};
    bool check = false, tight = false, routes_check = false;
};

// --routes-check: build_routes_fast_kernel against build_routes_kernel on every window start of every
// sequence (both ranks' expert ranges), all output arrays bytewise; and their event times (warm, after
// a flush, like the runner where the router has just written the ids).
void routes_check(const Options& o, const std::vector<std::vector<Token>>& seqs, Flush& flush, hipStream_t s) {
    using namespace strix::ds41;
    const std::uint32_t K = 6;
    for (const int layer : o.layers)
        for (const int rows_i : o.rows) {
            const auto R = static_cast<std::uint32_t>(rows_i);
            RoutedBuffers a(R, K, 32U, 32U), b(R, K, 32U, 32U);
            std::size_t windows_n = 0, bad = 0;
            std::vector<double> ta, tb;
            Timer timer;
            for (const auto& q : seqs)
                for (std::size_t t0 = 0; t0 + R <= q.size(); t0 += 7U) {  // every 7th start
                    std::vector<std::uint32_t> ids;
                    for (std::uint32_t r = 0; r < R; ++r) {
                        const auto it = q[t0 + r].ids.find(static_cast<unsigned>(layer));
                        if (it == q[t0 + r].ids.end()) break;
                        ids.insert(ids.end(), it->second.begin(), it->second.end());
                    }
                    if (ids.size() != R * K) continue;
                    a.ids.upload(ids); b.ids.upload(ids);
                    for (std::uint32_t rank = 0; rank < 2U; ++rank) {
                        for (auto* x : {&a, &b}) {
                            x->experts.fill(0x77); x->begins.fill(0x77); x->rows_out.fill(0x77); x->slots.fill(0x77);
                            x->order.fill(0x77); x->low_counts.fill(0x77); x->counts.fill(0x77);
                        }
                        const auto view = [&](RoutedBuffers& x) {
                            BuildRoutesView v;
                            v.ids = x.ids.data; v.rows = R; v.top_k = K; v.split = 192; v.first = rank * 192U; v.end = rank * 192U + 192U;
                            v.base = v.first; v.experts = x.experts.data; v.begins = x.begins.data; v.rows_out = x.rows_out.data;
                            v.slots = x.slots.data; v.order = x.order.data; v.low_counts = x.low_counts.data; v.counts = x.counts.data;
                            return v;
                        };
                        const bool timed = windows_n % 64U == 0U;
                        if (timed) flush(s);
                        timer.mark(0, s);
                        enqueue_build_routes(view(a), s);
                        timer.mark(1, s);
                        if (timed) flush(s);
                        timer.mark(2, s);
                        enqueue_build_routes_fast(view(b), s);
                        timer.mark(3, s);
                        check(hipStreamSynchronize(s), "routes");
                        if (timed) { ta.push_back(timer.ms(0, 1) * 1e3); tb.push_back(timer.ms(2, 3) * 1e3); }
                        const auto same = [](const auto& x, const auto& y) {
                            const auto u = x.download(), v = y.download();
                            return std::memcmp(u.data(), v.data(), u.size() * sizeof(u[0])) == 0;
                        };
                        bad += same(a.experts, b.experts) && same(a.begins, b.begins) && same(a.rows_out, b.rows_out) &&
                                       same(a.slots, b.slots) && same(a.order, b.order) && same(a.low_counts, b.low_counts) &&
                                       same(a.counts, b.counts) ? 0U : 1U;
                    }
                    ++windows_n;
                }
            if (bad) ++failures;
            std::cout << "ROUTES L" << layer << " R" << R << ": " << 2U * windows_n - bad << "/" << 2U * windows_n
                      << " (window, rank) builds equal; build_routes median " << median(ta) << " us, fast " << median(tb) << " us\n";
        }
}

struct Sum { double build = 0, gu = 0, dn = 0, s_gu = 0, s_dn = 0, n = 0, pairs = 0, unique = 0, most = 0, issued_gu = 0, issued_dn = 0;
             std::vector<double> t_gu, t_dn; };

void bench_layer(const Options& o, unsigned layer, const RankExperts& e, const std::vector<std::vector<Token>>& seqs, Flush& flush,
                 hipStream_t s, std::map<std::pair<int, int>, Sum>& total) {
    auto& tuning = strix::ds41::moe_kernel_tuning();
    const auto saved = tuning;
    Device<int> sink(1);
    for (const int rows_i : o.rows) {
        const auto rows = static_cast<std::uint32_t>(rows_i);
        const auto cases = windows(seqs, layer, rows, o.cases, e.first, e.end);
        RoutedBuffers b(rows, e.top_k, e.hidden, e.inter), ref(rows, e.top_k, e.hidden, e.inter);
        quantize_rows(b, rows, e.hidden, 11U + rows, s);
        quantize_rows(ref, rows, e.hidden, 11U + rows, s);
        Timer timer;
        for (const int form : o.forms) {
            // form >= 1000: the default forms with STRIX_DS41_EXPERT_SLOTS on, slots_y = form - 1000 (0 = the table)
            const bool slotted = form >= 1000;
            tuning = saved; tuning.routed = slotted ? -1 : form;
            if (slotted) { tuning.slots = 1; tuning.slots_y = form - 1000; }
            const int fg = form < 0 || slotted ? default_gate_up(rows) : form, fd = form < 0 || slotted ? default_down(rows) : form;
            const unsigned cg = std::min(form_cap(fg), rows), cd = std::min(form_cap(fd), rows);
            Sum& sum = total[{rows_i, form}];
            std::size_t bad = 0;
            for (const auto& w : cases) {
                b.ids.upload(w.ids);
                tight_count = o.tight ? std::max<std::uint32_t>(w.unique, 1U) : 0U;
                for (unsigned rep = 0; rep < o.reps; ++rep) {
                    flush(s);
                    enqueue_routed(e, b, rows, s, &timer);
                    const double bu = timer.ms(0, 1), gu = timer.ms(1, 2), dn = timer.ms(2, 3);
                    if (rep + 1U < o.reps) continue;
                    sum.build += bu; sum.gu += gu; sum.dn += dn; sum.t_gu.push_back(gu); sum.t_dn.push_back(dn);
                }
                sum.n += 1; sum.pairs += w.pairs; sum.unique += w.unique; sum.most += w.most;
                for (const auto& [id, n] : w.count) {
                    sum.issued_gu += static_cast<double>((n + cg - 1U) / cg) * e.gate_up_bytes();
                    sum.issued_dn += static_cast<double>((n + cd - 1U) / cd) * e.down_bytes();
                }
                tight_count = 0;
                if (o.check) {
                    b.clear_outputs(); ref.clear_outputs();
                    ref.ids.upload(w.ids);
                    const auto mine = tuning;
                    tuning = saved; tuning.routed = -1; tuning.slots = 0;
                    enqueue_routed(e, ref, rows, s, nullptr);
                    tuning = mine;
                    enqueue_routed(e, b, rows, s, nullptr);
                    check(hipStreamSynchronize(s), "check");
                    const auto same = [](const auto& x, const auto& y) {
                        const auto u = x.download(), v = y.download();
                        return std::memcmp(u.data(), v.data(), u.size() * sizeof(u[0])) == 0;
                    };
                    bad += same(b.act_codes, ref.act_codes) && same(b.act_deltas, ref.act_deltas) && same(b.act_sums, ref.act_sums) &&
                                   same(b.raw, ref.raw) ? 0U : 1U;
                }
            }
            if (o.check) {
                if (bad) ++failures;
                std::cout << "CHECK L" << layer << " R" << rows << " form " << form << ": " << cases.size() - bad << "/" << cases.size()
                          << " windows bitwise equal to the default forms\n";
            }
        }
        tuning = saved;
        // streaming ceiling over the same unique ranges (form key 100)
        Sum& st = total[{rows_i, 100}];
        for (const auto& w : cases) {
            std::vector<std::uint32_t> ids;
            for (const auto& [id, n] : w.count) ids.push_back(id - e.first);
            if (ids.empty()) continue;
            b.unique.upload(ids);
            for (unsigned rep = 0; rep < o.reps; ++rep) {
                flush(s);
                timer.mark(1, s);
                const auto chunks_gu = static_cast<unsigned>((e.gate_t.stride + 65535U) / 65536U);
                hipLaunchKernelGGL(stream_kernel, dim3(chunks_gu, static_cast<unsigned>(ids.size())), dim3(256), 0, s, e.gate.data, e.up.data,
                                   b.unique.data, e.gate_t.stride, chunks_gu, sink.data);
                timer.mark(2, s);
                const auto chunks_dn = static_cast<unsigned>((e.down_t.stride + 65535U) / 65536U);
                hipLaunchKernelGGL(stream_kernel, dim3(chunks_dn, static_cast<unsigned>(ids.size())), dim3(256), 0, s, e.down.data,
                                   static_cast<const unsigned char*>(nullptr), b.unique.data, e.down_t.stride, chunks_dn, sink.data);
                timer.mark(3, s);
                check(hipGetLastError(), "stream");
                const double gu = timer.ms(1, 2), dn = timer.ms(2, 3);
                if (rep + 1U < o.reps) continue;
                st.gu += gu; st.dn += dn; st.t_gu.push_back(gu); st.t_dn.push_back(dn);
            }
            st.n += 1; st.unique += w.unique;
        }
    }
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) { std::cerr << "usage: strix-hip-ds41-expert-once-bench MODEL.gguf --routes FILE [options]\n"; return 2; }
    Options o;
    o.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value"); return argv[++i]; };
        if (a == "--layers") o.layers = parse_ints(next());
        else if (a == "--rank") o.rank = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--routes") o.routes = next();
        else if (a == "--rows") o.rows = parse_ints(next());
        else if (a == "--cases") o.cases = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--reps") o.reps = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--forms") o.forms = parse_ints(next());
        else if (a == "--check") o.check = true;
        else if (a == "--tight") o.tight = true;
        else if (a == "--routes-check") o.routes_check = true;
        else throw std::invalid_argument("unknown option " + a);
    }
    if (o.routes.empty()) throw std::invalid_argument("--routes is required");
    const auto file = strix::gguf::read_file(o.model);
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    for (const auto& t : file.tensors) tensors[t.name] = &t;
    const auto seqs = load_routes(o.routes, o.layers, 6U);
    std::size_t tokens = 0;
    for (const auto& q : seqs) tokens += q.size();
    std::cout << "routes: " << seqs.size() << " sequences, " << tokens << " tokens\n";
    hipStream_t s{};
    check(hipStreamCreate(&s), "stream");
    Flush flush;
    {   // event-to-event cost of a one-block empty kernel after a flush (the floor of a tiny launch)
        Device<int> sink(1);
        Timer t;
        std::vector<double> v;
        for (int i = 0; i < 64; ++i) {
            flush(s);
            t.mark(0, s);
            hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(64), 0, s, sink.data);
            t.mark(1, s);
            v.push_back(t.ms(0, 1) * 1e3);
        }
        std::cout << "empty one-block launch: median " << median(v) << " us\n";
    }
    if (o.routes_check) {
        routes_check(o, seqs, flush, s);
        std::cout << (failures ? "FAILED" : "OK") << " (" << failures << " failures)\n";
        return failures ? 1 : 0;
    }
    std::map<std::pair<int, int>, Sum> total;
    double gu_bytes = 0, dn_bytes = 0;
    for (const int layer : o.layers) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto g = tensor(tensors, prefix + "ffn_gate_exps.weight"), u = tensor(tensors, prefix + "ffn_up_exps.weight"),
                   d = tensor(tensors, prefix + "ffn_down_exps.weight");
        const std::uint32_t half = g.experts / 2U;
        const RankExperts e(g, u, d, o.rank * half, o.rank * half + half, o.model);
        gu_bytes = e.gate_up_bytes(); dn_bytes = e.down_bytes();
        std::cout << "layer " << layer << " rank " << o.rank << ": experts [" << e.first << ", " << e.end << "), gate+up "
                  << e.gate_up_bytes() / 1e6 << " MB, down " << e.down_bytes() / 1e6 << " MB per expert" << std::endl;
        bench_layer(o, static_cast<unsigned>(layer), e, seqs, flush, s, total);
    }
    std::cout << std::fixed;
    std::cout << "\nper layer means over " << o.layers.size() << " layer(s) (cold, last of " << o.reps << " reps per window):\n"
              << "  R form | pairs unique most | MB unique / per-pair / issued(gu+dn) | build us | gate_up us GB/s | down us GB/s | total us GB/s | x40 ms\n";
    for (const auto& [key, v] : total) {
        if (v.n == 0) continue;
        const auto [rows, form] = key;
        const double n = v.n, uq = v.unique / n;
        const double mb_u = uq * (gu_bytes + dn_bytes) / 1e6, mb_p = v.pairs / n * (gu_bytes + dn_bytes) / 1e6;
        const double gu = v.gu / n * 1e3, dn = v.dn / n * 1e3, bu = v.build / n * 1e3;
        const double gbs_gu = uq * gu_bytes / (gu * 1e3), gbs_dn = uq * dn_bytes / (dn * 1e3);
        const double tot = gu + dn + bu;
        std::cout << std::setprecision(2) << "  R" << rows << " " << (form == 100 ? std::string("stream") : std::to_string(form)) << " | "
                  << (form == 100 ? 0.0 : v.pairs / n) << " " << uq << " " << v.most / n << " | " << std::setprecision(1) << mb_u << " / "
                  << (form == 100 ? 0.0 : mb_p) << " / " << (form == 100 ? 0.0 : (v.issued_gu + v.issued_dn) / n / 1e6) << " | " << bu << " | " << gu
                  << " (med " << median(v.t_gu) * 1e3 << ") " << gbs_gu << " | " << dn << " (med " << median(v.t_dn) * 1e3 << ") " << gbs_dn
                  << " | " << tot << " " << mb_u * 1e6 / (tot * 1e3) << " | " << std::setprecision(2) << tot * 40.0 / 1e3 << "\n";
    }
    check(hipStreamDestroy(s), "stream");
    std::cout << (failures ? "FAILED" : "OK") << " (" << failures << " failures)\n";
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
