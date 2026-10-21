// Routed-expert, shared-expert and output-head kernels of the deepseek41 decode at the launch shapes of
// the two-rank runner (routes built on the device per rank, the shared expert as one expert of R rows,
// the head over one rank's vocabulary slice), on real weights and recorded routes:
//   timings with cold caches against the bytes each launch must read, and
//   bitwise checks of every kernel form against the reference form (and of the reference against the
//   host product on some windows).
// usage: strix-hip-ds41-moe-bench MODEL.gguf [--layer N] [--rank R] [--routes FILE] [--rows 1,2,4,8]
//        [--cases N] [--reps N] [--part routed,shared,head] [--check] [--host-check] [--same-rows | --spread]
// --routes: one line per token, "layer:id,id,..." per layer (the rank app's --routes output).

#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_layer.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_moe.hpp"

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
#include <numeric>
#include <random>
#include <set>
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
void expect(bool condition, const std::string& message) {
    if (!condition) { ++failures; std::cout << "FAIL: " << message << '\n'; }
}

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) { check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc"); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)hipFree(data); }
    void upload(const T* host, std::size_t n) { check(hipMemcpy(data, host, n * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    void upload(const std::vector<T>& v) { upload(v.data(), v.size()); }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> out(count);
        check(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }
    void fill(int byte) { check(hipMemset(data, byte, count * sizeof(T)), "memset"); }
};

std::vector<std::uint8_t> read_range(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    std::vector<std::uint8_t> out(bytes);
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    std::uint64_t done = 0;
    while (done < bytes) {
        const auto n = ::pread(fd, out.data() + done, std::min<std::uint64_t>(bytes - done, 1ULL << 28), static_cast<off_t>(offset + done));
        if (n <= 0) { ::close(fd); throw std::runtime_error("short read in " + path); }
        done += static_cast<std::uint64_t>(n);
    }
    ::close(fd);
    return out;
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
    t.format = type == 16U ? WeightFormat::iq2_xxs : type == 10U ? WeightFormat::q2_k : type == 8U ? WeightFormat::q8_0 : WeightFormat::mxfp4;
    if (type != 16U && type != 10U && type != 8U && type != 39U) throw std::runtime_error(name + " has an unexpected type");
    t.width = static_cast<std::uint32_t>(t.info->dimensions.at(0));
    t.outputs = static_cast<std::uint32_t>(t.info->dimensions.at(1));
    t.experts = t.info->dimensions.size() > 2U ? static_cast<std::uint32_t>(t.info->dimensions[2]) : 1U;
    t.row_bytes = t.width / strix::ds41::block_elements(t.format) * strix::ds41::block_bytes(t.format);
    t.stride = t.row_bytes * t.outputs;
    if (t.stride * t.experts != t.info->byte_size) throw std::runtime_error(name + " byte size does not match its geometry");
    return t;
}

// Reads (never dirties) a buffer much larger than the caches, so every timed launch starts cold.
__global__ void flush_kernel(const int* buffer, std::size_t count, int value, int* sink) {
    int sum = 0;
    for (std::size_t i = blockIdx.x * 256ULL + threadIdx.x; i < count; i += gridDim.x * 256ULL) sum += buffer[i];
    if (sum == value) sink[0] = sum;
}
struct Flush {
    Device<int> buffer{64U * 1024U * 1024U};  // 256 MB, eight times the last-level cache
    Device<int> sink{1};
    Flush() { buffer.fill(0); check(hipDeviceSynchronize(), "flush buffer"); }
    void operator()(hipStream_t s) {
        hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, s, buffer.data, buffer.count, 0x5a5a5a5a, sink.data);
        check(hipGetLastError(), "flush");
    }
};

struct Timer {
    hipEvent_t e[4]{};
    Timer() { for (auto& x : e) check(hipEventCreate(&x), "event"); }
    ~Timer() { for (auto& x : e) (void)hipEventDestroy(x); }
    void mark(unsigned i, hipStream_t s) { check(hipEventRecord(e[i], s), "event"); }
    float ms(unsigned a, unsigned b) { float t = 0; check(hipEventSynchronize(e[b]), "sync"); check(hipEventElapsedTime(&t, e[a], e[b]), "elapsed"); return t; }
};

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2U]; }

std::vector<unsigned> parse_list(const std::string& s) {
    std::vector<unsigned> out;
    std::stringstream in(s);
    for (std::string item; std::getline(in, item, ',');) out.push_back(static_cast<unsigned>(std::stoul(item)));
    return out;
}

// Recorded routes of one layer: token -> top_k ids.
std::vector<std::vector<std::uint32_t>> load_routes(const std::string& path, unsigned layer, unsigned top_k) {
    std::vector<std::vector<std::uint32_t>> out;
    std::ifstream in(path);
    for (std::string line; std::getline(in, line);) {
        std::stringstream fields(line);
        for (std::string field; fields >> field;) {
            const auto colon = field.find(':');
            if (colon == std::string::npos || std::stoul(field.substr(0, colon)) != layer) continue;
            std::vector<std::uint32_t> ids;
            for (const auto id : parse_list(field.substr(colon + 1))) ids.push_back(id);
            if (ids.size() == top_k) out.push_back(ids);
        }
    }
    return out;
}

std::vector<float> random_values(std::size_t n, unsigned seed, float sigma) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = normal(random);
    return v;
}

struct Options {
    std::string model, routes;
    unsigned layer = 20, rank = 0, cases = 48, reps = 3;
    std::vector<unsigned> rows{1, 2, 4, 8};
    std::set<std::string> parts{"routed", "shared", "head"};
    bool check = false, host_check = false;
};

// ---- Routed experts of one rank, routes built on the device as in the runner ----

struct RankExperts {
    Tensor gate_t, up_t, down_t;
    std::uint32_t first = 0, end = 0, hidden = 0, inter = 0, top_k = 6, split = 192;
    std::vector<std::uint8_t> host_gate, host_up, host_down;  // this rank's experts only
    Device<std::uint8_t> gate, up, down;
    RankExperts(const Tensor& g, const Tensor& u, const Tensor& d, std::uint32_t f, std::uint32_t e, const std::string& path)
        : gate_t(g), up_t(u), down_t(d), first(f), end(e), hidden(g.width), inter(g.outputs),
          host_gate(read_range(path, g.info->absolute_offset + g.stride * f, g.stride * (e - f))),
          host_up(read_range(path, u.info->absolute_offset + u.stride * f, u.stride * (e - f))),
          host_down(read_range(path, d.info->absolute_offset + d.stride * f, d.stride * (e - f))),
          gate(host_gate.size()), up(host_up.size()), down(host_down.size()) {
        gate.upload(host_gate); up.upload(host_up); down.upload(host_down);
    }
    [[nodiscard]] double expert_bytes_gate_up() const { return 2.0 * static_cast<double>(gate_t.stride); }
    [[nodiscard]] double expert_bytes_down() const { return static_cast<double>(down_t.stride); }
};

struct RoutedBuffers {
    Device<std::uint32_t> ids, experts, begins, rows_out, slots, order, low_counts, counts;
    Device<std::int8_t> codes; Device<float> deltas; Device<std::int32_t> sums;
    Device<std::int8_t> act_codes; Device<float> act_deltas; Device<std::int32_t> act_sums;
    Device<float> raw;
    RoutedBuffers(std::uint32_t rows, std::uint32_t top_k, std::uint32_t hidden, std::uint32_t inter)
        : ids(rows * top_k), experts(rows * top_k + 1U), begins(rows * top_k + 1U), rows_out(rows * top_k), slots(rows * top_k),
          order(rows * top_k), low_counts(rows), counts(2), codes(static_cast<std::size_t>(rows) * hidden), deltas(rows * hidden / 32U),
          sums(rows * hidden / 32U), act_codes(static_cast<std::size_t>(rows) * top_k * inter), act_deltas(rows * top_k * inter / 32U),
          act_sums(rows * top_k * inter / 32U), raw(static_cast<std::size_t>(rows) * top_k * hidden) {}
    void clear_outputs() { act_codes.fill(0x5b); act_deltas.fill(0x5b); act_sums.fill(0x5b); raw.fill(0x5b); }
};

void enqueue_routed(const RankExperts& e, RoutedBuffers& b, std::uint32_t rows, hipStream_t s, Timer* timer) {
    using namespace strix::ds41;
    BuildRoutesView bv;
    bv.ids = b.ids.data; bv.rows = rows; bv.top_k = e.top_k; bv.split = e.split; bv.first = e.first; bv.end = e.end; bv.base = e.first;
    bv.experts = b.experts.data; bv.begins = b.begins.data; bv.rows_out = b.rows_out.data; bv.slots = b.slots.data;
    bv.order = b.order.data; bv.low_counts = b.low_counts.data; bv.counts = b.counts.data;
    enqueue_build_routes(bv, s);
    const RoutesView gate_routes{b.experts.data, b.begins.data, b.rows_out.data, b.slots.data, rows * e.top_k, rows, b.counts.data};
    const RoutesView down_routes{b.experts.data, b.begins.data, b.slots.data, b.slots.data, rows * e.top_k, rows, b.counts.data};
    if (timer) timer->mark(0, s);
    GateUpView gu;
    gu.gate = e.gate.data; gu.up = e.up.data; gu.expert_stride_bytes = e.gate_t.stride; gu.row_bytes = e.gate_t.row_bytes;
    gu.hidden = e.hidden; gu.inter = e.inter; gu.routes = gate_routes; gu.codes = b.codes.data; gu.deltas = b.deltas.data;
    gu.limit = 10.0F; gu.act_codes = b.act_codes.data; gu.act_deltas = b.act_deltas.data; gu.act_sums = b.act_sums.data;
    enqueue_gate_up(e.gate_t.format, gu, s);
    if (timer) timer->mark(1, s);
    RoutesProductView dv;
    dv.weights = e.down.data; dv.expert_stride_bytes = e.down_t.stride; dv.row_bytes = e.down_t.row_bytes; dv.width = e.inter;
    dv.outputs = e.hidden; dv.routes = down_routes; dv.codes = b.act_codes.data; dv.deltas = b.act_deltas.data; dv.out = b.raw.data;
    enqueue_routes_product(e.down_t.format, dv, s);
    if (timer) timer->mark(2, s);
}

struct Window { std::vector<std::uint32_t> ids; std::uint32_t union_mine = 0, most = 0; };
// --same-rows: every row of a window takes the same token's routes (the most overlap, as in the rank
// app's window bench); otherwise consecutive recorded tokens.
bool same_rows = false;
// --spread: the rows of a window come from distant recorded tokens (independent requests in a batch step).
bool spread_rows = false;
std::vector<Window> windows(const std::vector<std::vector<std::uint32_t>>& routes, std::uint32_t rows, std::uint32_t cases,
                            std::uint32_t top_k, std::uint32_t first, std::uint32_t end, unsigned seed) {
    std::vector<Window> out;
    std::mt19937 random(seed);
    for (std::uint32_t c = 0; c < cases; ++c) {
        Window w;
        for (std::uint32_t r = 0; r < rows; ++r) {
            if (!routes.empty()) {
                const std::size_t pick = same_rows ? c : spread_rows ? (static_cast<std::size_t>(c) * 7U + static_cast<std::size_t>(r) * 109U)
                                                                     : static_cast<std::size_t>(c) * rows + r;
                const auto& t = routes[pick % routes.size()];
                w.ids.insert(w.ids.end(), t.begin(), t.end());
            } else {
                std::vector<std::uint32_t> all(384);
                std::iota(all.begin(), all.end(), 0U);
                std::shuffle(all.begin(), all.end(), random);
                w.ids.insert(w.ids.end(), all.begin(), all.begin() + top_k);
            }
        }
        std::map<std::uint32_t, std::uint32_t> count;
        for (const auto id : w.ids)
            if (id >= first && id < end) ++count[id];
        w.union_mine = static_cast<std::uint32_t>(count.size());
        for (const auto& [id, n] : count) w.most = std::max(w.most, n);
        out.push_back(std::move(w));
    }
    return out;
}

void quantize_rows(RoutedBuffers& b, std::uint32_t rows, std::uint32_t hidden, unsigned seed, hipStream_t s) {
    Device<float> x(static_cast<std::size_t>(rows) * hidden);
    x.upload(random_values(x.count, seed, 1.0F));
    strix::ds41::enqueue_quantize_a8(x.data, hidden, rows, b.codes.data, b.deltas.data, b.sums.data, s);
    check(hipStreamSynchronize(s), "quantize");
}

template <class T>
bool same_bytes(const Device<T>& a, const Device<T>& b) {
    const auto x = a.download(), y = b.download();
    return std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
}

void bench_routed(const Options& o, const RankExperts& e, const std::vector<std::vector<std::uint32_t>>& routes, Flush& flush,
                  hipStream_t s) {
    for (const auto rows : o.rows) {
        RoutedBuffers b(rows, e.top_k, e.hidden, e.inter);
        quantize_rows(b, rows, e.hidden, 11U + rows, s);
        const auto cases = windows(routes, rows, o.cases, e.top_k, e.first, e.end, 7U + rows);
        Timer timer;
        std::vector<double> t_gu, t_dn;
        double bytes_gu = 0, bytes_dn = 0, sum_gu = 0, sum_dn = 0, experts = 0;
        for (const auto& w : cases) {
            b.ids.upload(w.ids);
            for (unsigned rep = 0; rep < o.reps; ++rep) {
                flush(s);
                enqueue_routed(e, b, rows, s, &timer);
                const double gu = timer.ms(0, 1), dn = timer.ms(1, 2);
                if (rep + 1U < o.reps) continue;  // the last cold pass of each window
                t_gu.push_back(gu); t_dn.push_back(dn);
                sum_gu += gu; sum_dn += dn;
                bytes_gu += w.union_mine * e.expert_bytes_gate_up();
                bytes_dn += w.union_mine * e.expert_bytes_down();
                experts += w.union_mine;
            }
        }
        std::cout << std::fixed << std::setprecision(1) << "ROUTED R" << rows << " rank " << o.rank << ": mean union " << experts / static_cast<double>(cases.size())
                  << " experts; gate_up " << median(t_gu) * 1000.0 << " us median, " << bytes_gu / (sum_gu * 1e6) << " GB/s; down "
                  << median(t_dn) * 1000.0 << " us, " << bytes_dn / (sum_dn * 1e6) << " GB/s; total " << (sum_gu + sum_dn) / static_cast<double>(cases.size()) * 1000.0
                  << " us/layer, " << (bytes_gu + bytes_dn) / ((sum_gu + sum_dn) * 1e6) << " GB/s\n";
    }
}

// Variant vs reference form, bitwise, on recorded windows (every rows value, all cases).
void check_routed_variants(const Options& o, const RankExperts& e, const std::vector<std::vector<std::uint32_t>>& routes,
                           hipStream_t s, const std::vector<int>& forms) {
    auto& tuning = strix::ds41::moe_kernel_tuning();
    const auto saved = tuning;
    for (const auto rows : o.rows) {
        RoutedBuffers ref(rows, e.top_k, e.hidden, e.inter), cand(rows, e.top_k, e.hidden, e.inter);
        quantize_rows(ref, rows, e.hidden, 21U + rows, s);
        check(hipMemcpy(cand.codes.data, ref.codes.data, ref.codes.count, hipMemcpyDeviceToDevice), "copy");
        check(hipMemcpy(cand.deltas.data, ref.deltas.data, ref.deltas.count * 4U, hipMemcpyDeviceToDevice), "copy");
        const auto cases = windows(routes, rows, o.cases, e.top_k, e.first, e.end, 5U + rows);
        for (const int form : forms) {
            std::size_t bad = 0;
            for (const auto& w : cases) {
                ref.ids.upload(w.ids); cand.ids.upload(w.ids);
                ref.clear_outputs(); cand.clear_outputs();
                tuning = saved; tuning.routed = 0;
                enqueue_routed(e, ref, rows, s, nullptr);
                tuning.routed = form;
                enqueue_routed(e, cand, rows, s, nullptr);
                check(hipStreamSynchronize(s), "routed check");
                const bool same = same_bytes(ref.act_codes, cand.act_codes) && same_bytes(ref.act_deltas, cand.act_deltas) &&
                                  same_bytes(ref.act_sums, cand.act_sums) && same_bytes(ref.raw, cand.raw);
                bad += same ? 0U : 1U;
            }
            expect(bad == 0U, "routed form " + std::to_string(form) + " R" + std::to_string(rows) + ": " + std::to_string(bad) + " windows differ");
            std::cout << "CHECK routed form " << form << " R" << rows << ": " << cases.size() - bad << "/" << cases.size()
                      << " windows bitwise equal to form 0 (codes, deltas, sums, raw slots)\n";
        }
    }
    tuning = saved;
}

// The reference form against the host product (reference_window) on a few windows.
void host_check_routed(const Options& o, const RankExperts& e, const std::vector<std::vector<std::uint32_t>>& routes, hipStream_t s) {
    using namespace strix::ds41;
    ExpertTensors t;
    t.gate_up_format = e.gate_t.format; t.down_format = e.down_t.format; t.hidden = e.hidden; t.inter = e.inter;
    t.gate = e.host_gate; t.up = e.host_up; t.down = e.host_down;
    for (const auto rows : o.rows) {
        RoutedBuffers b(rows, e.top_k, e.hidden, e.inter);
        const auto xs = random_values(static_cast<std::size_t>(rows) * e.hidden, 31U + rows, 1.0F);
        Device<float> x(xs.size());
        x.upload(xs);
        enqueue_quantize_a8(x.data, e.hidden, rows, b.codes.data, b.deltas.data, b.sums.data, s);
        std::vector<A8Activation> host_x;
        for (std::uint32_t r = 0; r < rows; ++r)
            host_x.push_back(quantize_a8(std::span<const float>(xs.data() + static_cast<std::size_t>(r) * e.hidden, e.hidden)));
        const auto cases = windows(routes, rows, 2U, e.top_k, e.first, e.end, 3U + rows);
        for (const auto& w : cases) {
            b.ids.upload(w.ids);
            b.clear_outputs();
            enqueue_routed(e, b, rows, s, nullptr);
            check(hipStreamSynchronize(s), "routed");
            auto window = restrict_routes(build_window_routes(w.ids, rows, e.top_k, e.split), e.first, e.end);
            for (auto& id : window.experts) id -= e.first;
            const std::vector<float> weights(static_cast<std::size_t>(rows) * e.top_k, 0.25F), shared(static_cast<std::size_t>(rows) * e.hidden, 0.0F);
            const auto ref = reference_window(t, host_x, window, weights, shared, e.top_k);
            const auto raw = b.raw.download();
            const auto codes = b.act_codes.download();
            std::size_t bad = 0, compared = 0;
            for (std::size_t j = 0; j < window.experts.size(); ++j)
                for (auto i = window.begins[j]; i < window.begins[j + 1]; ++i) {
                    const auto slot = window.slots[i];
                    for (std::uint32_t c = 0; c < e.inter; ++c) bad += codes[static_cast<std::size_t>(slot) * e.inter + c] != ref.activation[slot].codes[c];
                    for (std::uint32_t h = 0; h < e.hidden; ++h) {
                        const float a = raw[static_cast<std::size_t>(slot) * e.hidden + h], r = ref.raw[static_cast<std::size_t>(slot) * e.hidden + h];
                        bad += std::memcmp(&a, &r, 4) != 0;
                    }
                    ++compared;
                }
            expect(bad == 0U, "routed R" + std::to_string(rows) + " host check: " + std::to_string(bad) + " values differ");
            std::cout << "HOST routed R" << rows << ": " << compared << " slots, " << (bad ? "MISMATCH" : "bitwise equal to the host product") << '\n';
        }
    }
}

// ---- Shared expert (Q8_0), one expert of R rows ----

struct SharedExpert {
    Tensor gate_t, up_t, down_t;
    Device<std::uint8_t> gate, up, down;
    std::vector<std::uint8_t> host_gate, host_up, host_down;
    SharedExpert(const Tensor& g, const Tensor& u, const Tensor& d, const std::string& path)
        : gate_t(g), up_t(u), down_t(d), gate(g.info->byte_size), up(u.info->byte_size), down(d.info->byte_size),
          host_gate(read_range(path, g.info->absolute_offset, g.info->byte_size)),
          host_up(read_range(path, u.info->absolute_offset, u.info->byte_size)),
          host_down(read_range(path, d.info->absolute_offset, d.info->byte_size)) {
        gate.upload(host_gate); up.upload(host_up); down.upload(host_down);
    }
};

struct SharedBuffers {
    Device<std::uint32_t> experts, begins, rows_in, slots;
    std::uint32_t expert_count = 1, max_rows = 0;
    Device<std::int8_t> codes; Device<float> deltas; Device<std::int32_t> sums;
    Device<std::int8_t> act_codes; Device<float> act_deltas; Device<std::int32_t> act_sums;
    Device<float> out;
    SharedBuffers(std::uint32_t rows, std::uint32_t hidden, std::uint32_t inter)
        : experts(1), begins(2), rows_in(rows), slots(rows), max_rows(rows), codes(static_cast<std::size_t>(rows) * hidden),
          deltas(rows * hidden / 32U), sums(rows * hidden / 32U), act_codes(static_cast<std::size_t>(rows) * inter),
          act_deltas(rows * inter / 32U), act_sums(rows * inter / 32U), out(static_cast<std::size_t>(rows) * hidden) {
        const auto r = strix::ds41::build_window_routes(std::vector<std::uint32_t>(rows, 0U), rows, 1U, 1U);
        experts.upload(r.experts); begins.upload(r.begins); rows_in.upload(r.rows); slots.upload(r.slots);
    }
};

void enqueue_shared(const SharedExpert& e, SharedBuffers& b, hipStream_t s, Timer* timer) {
    using namespace strix::ds41;
    const auto H = e.gate_t.width, I = e.gate_t.outputs;
    if (timer) timer->mark(0, s);
    GateUpView gu;
    gu.gate = e.gate.data; gu.up = e.up.data; gu.expert_stride_bytes = e.gate_t.stride; gu.row_bytes = e.gate_t.row_bytes;
    gu.hidden = H; gu.inter = I; gu.routes = {b.experts.data, b.begins.data, b.rows_in.data, b.slots.data, 1U, b.max_rows};
    gu.codes = b.codes.data; gu.deltas = b.deltas.data; gu.limit = 10.0F;
    gu.act_codes = b.act_codes.data; gu.act_deltas = b.act_deltas.data; gu.act_sums = b.act_sums.data;
    enqueue_gate_up(WeightFormat::q8_0, gu, s);
    if (timer) timer->mark(1, s);
    RoutesProductView dv;
    dv.weights = e.down.data; dv.expert_stride_bytes = e.down_t.stride; dv.row_bytes = e.down_t.row_bytes; dv.width = I; dv.outputs = H;
    dv.routes = {b.experts.data, b.begins.data, b.slots.data, b.slots.data, 1U, b.max_rows};
    dv.codes = b.act_codes.data; dv.deltas = b.act_deltas.data; dv.out = b.out.data;
    enqueue_routes_product(WeightFormat::q8_0, dv, s);
    if (timer) timer->mark(2, s);
}

void bench_shared(const Options& o, const SharedExpert& e, Flush& flush, hipStream_t s) {
    const auto H = e.gate_t.width, I = e.gate_t.outputs;
    for (const auto rows : o.rows) {
        SharedBuffers b(rows, H, I);
        Device<float> x(static_cast<std::size_t>(rows) * H);
        x.upload(random_values(x.count, 41U + rows, 1.0F));
        strix::ds41::enqueue_quantize_a8(x.data, H, rows, b.codes.data, b.deltas.data, b.sums.data, s);
        Timer timer;
        std::vector<double> gu, dn;
        for (unsigned rep = 0; rep < o.cases; ++rep) {
            flush(s);
            enqueue_shared(e, b, s, &timer);
            gu.push_back(timer.ms(0, 1)); dn.push_back(timer.ms(1, 2));
        }
        const double g = median(gu), d = median(dn);
        const double bg = 2.0 * static_cast<double>(e.gate_t.stride), bd = static_cast<double>(e.down_t.stride);
        std::cout << std::fixed << std::setprecision(1) << "SHARED R" << rows << ": gate_up " << g * 1000.0 << " us, " << bg / (g * 1e6)
                  << " GB/s; down " << d * 1000.0 << " us, " << bd / (d * 1e6) << " GB/s; total " << (g + d) * 1000.0 << " us, "
                  << (bg + bd) / ((g + d) * 1e6) << " GB/s\n";
    }
}

// The shared expert split over two ranks (exact): rank r computes gate/up of intermediate rows
// [I/2 r, I/2 r + I/2) (whole A8 groups) and, from the assembled activation, down rows [H/2 r, H/2 r + H/2).
// Every value is the same product as in the whole expert. Times rank r's two launches cold and compares
// them with the slices of the whole expert.
void bench_shared_split(const Options& o, const SharedExpert& e, Flush& flush, hipStream_t s) {
    using namespace strix::ds41;
    const auto H = e.gate_t.width, I = e.gate_t.outputs, hI = I / 2U, hH = H / 2U;
    for (const auto rows : o.rows) {
        SharedBuffers ref(rows, H, I);
        Device<float> x(static_cast<std::size_t>(rows) * H);
        x.upload(random_values(x.count, 91U + rows, 1.0F));
        enqueue_quantize_a8(x.data, H, rows, ref.codes.data, ref.deltas.data, ref.sums.data, s);
        enqueue_shared(e, ref, s, nullptr);
        check(hipStreamSynchronize(s), "shared");
        const auto ref_codes = ref.act_codes.download();
        const auto ref_deltas = ref.act_deltas.download();
        const auto ref_out = ref.out.download();
        Device<std::int8_t> codes(static_cast<std::size_t>(rows) * hI);
        Device<float> deltas(rows * hI / 32U), out(static_cast<std::size_t>(rows) * hH);
        Device<std::int32_t> sums(rows * hI / 32U);
        Timer timer;
        for (std::uint32_t rank = 0; rank < 2U; ++rank) {
            const auto launch = [&](bool timed) {
                if (timed) timer.mark(0, s);
                GateUpView gu;
                gu.gate = e.gate.data + e.gate_t.row_bytes * hI * rank; gu.up = e.up.data + e.up_t.row_bytes * hI * rank;
                gu.expert_stride_bytes = e.gate_t.stride; gu.row_bytes = e.gate_t.row_bytes; gu.hidden = H; gu.inter = hI;
                gu.routes = {ref.experts.data, ref.begins.data, ref.rows_in.data, ref.slots.data, 1U, ref.max_rows};
                gu.codes = ref.codes.data; gu.deltas = ref.deltas.data; gu.limit = 10.0F;
                gu.act_codes = codes.data; gu.act_deltas = deltas.data; gu.act_sums = sums.data;
                enqueue_gate_up(WeightFormat::q8_0, gu, s);
                if (timed) timer.mark(1, s);
                RoutesProductView dv;
                dv.weights = e.down.data + e.down_t.row_bytes * hH * rank; dv.expert_stride_bytes = e.down_t.stride;
                dv.row_bytes = e.down_t.row_bytes; dv.width = I; dv.outputs = hH;
                dv.routes = {ref.experts.data, ref.begins.data, ref.slots.data, ref.slots.data, 1U, ref.max_rows};
                dv.codes = ref.act_codes.data; dv.deltas = ref.act_deltas.data; dv.out = out.data;
                enqueue_routes_product(WeightFormat::q8_0, dv, s);
                if (timed) timer.mark(2, s);
            };
            codes.fill(0x5b); deltas.fill(0x5b); out.fill(0x5b);
            launch(false);
            check(hipStreamSynchronize(s), "split");
            const auto c = codes.download();
            const auto d = deltas.download();
            const auto y = out.download();
            std::size_t bad = 0;
            for (std::uint32_t r = 0; r < rows; ++r) {
                bad += std::memcmp(c.data() + static_cast<std::size_t>(r) * hI, ref_codes.data() + static_cast<std::size_t>(r) * I + hI * rank, hI) != 0;
                bad += std::memcmp(d.data() + static_cast<std::size_t>(r) * hI / 32U, ref_deltas.data() + static_cast<std::size_t>(r) * I / 32U + hI / 32U * rank, hI / 8U) != 0;
                bad += std::memcmp(y.data() + static_cast<std::size_t>(r) * hH, ref_out.data() + static_cast<std::size_t>(r) * H + hH * rank, hH * 4U) != 0;
            }
            expect(bad == 0U, "shared split R" + std::to_string(rows) + " rank " + std::to_string(rank));
            std::vector<double> gu, dn;
            for (unsigned rep = 0; rep < o.cases; ++rep) {
                flush(s);
                launch(true);
                gu.push_back(timer.ms(0, 1)); dn.push_back(timer.ms(1, 2));
            }
            const double g = median(gu), dd = median(dn);
            const double bg = static_cast<double>(e.gate_t.stride), bd = static_cast<double>(e.down_t.stride) / 2.0;
            std::cout << std::fixed << std::setprecision(1) << "SHARED-SPLIT R" << rows << " rank " << rank << ": gate_up " << g * 1000.0
                      << " us, " << bg / (g * 1e6) << " GB/s; down " << dd * 1000.0 << " us, " << bd / (dd * 1e6) << " GB/s; total "
                      << (g + dd) * 1000.0 << " us" << (bad ? ", SLICES DIFFER" : ", slices bitwise equal to the whole expert") << '\n';
        }
    }
}

void check_shared_variants(const Options& o, const SharedExpert& e, hipStream_t s, const std::vector<int>& forms) {
    auto& tuning = strix::ds41::moe_kernel_tuning();
    const auto saved = tuning;
    const auto H = e.gate_t.width, I = e.gate_t.outputs;
    for (const auto rows : o.rows) {
        SharedBuffers ref(rows, H, I), cand(rows, H, I);
        for (unsigned seed = 0; seed < 4U; ++seed) {
            Device<float> x(static_cast<std::size_t>(rows) * H);
            x.upload(random_values(x.count, 51U + rows * 7U + seed, seed == 3U ? 30.0F : 1.0F));
            strix::ds41::enqueue_quantize_a8(x.data, H, rows, ref.codes.data, ref.deltas.data, ref.sums.data, s);
            strix::ds41::enqueue_quantize_a8(x.data, H, rows, cand.codes.data, cand.deltas.data, cand.sums.data, s);
            for (const int form : forms) {
                ref.out.fill(0x5b); cand.out.fill(0x5b); ref.act_codes.fill(0x5b); cand.act_codes.fill(0x5b);
                tuning = saved; tuning.shared = 0;
                enqueue_shared(e, ref, s, nullptr);
                tuning.shared = form;
                enqueue_shared(e, cand, s, nullptr);
                check(hipStreamSynchronize(s), "shared check");
                const bool same = same_bytes(ref.act_codes, cand.act_codes) && same_bytes(ref.act_deltas, cand.act_deltas) &&
                                  same_bytes(ref.act_sums, cand.act_sums) && same_bytes(ref.out, cand.out);
                expect(same, "shared form " + std::to_string(form) + " R" + std::to_string(rows) + " seed " + std::to_string(seed));
                if (seed == 3U) std::cout << "CHECK shared form " << form << " R" << rows << ": " << (same ? "bitwise equal" : "DIFFERS") << " to form 0 (4 inputs)\n";
            }
        }
    }
    tuning = saved;
}

void host_check_shared(const Options& o, const SharedExpert& e, hipStream_t s) {
    using namespace strix::ds41;
    const auto H = e.gate_t.width, I = e.gate_t.outputs;
    ExpertTensors t;
    t.gate_up_format = WeightFormat::q8_0; t.down_format = WeightFormat::q8_0; t.hidden = H; t.inter = I;
    t.gate = e.host_gate; t.up = e.host_up; t.down = e.host_down;
    for (const auto rows : o.rows) {
        SharedBuffers b(rows, H, I);
        const auto xs = random_values(static_cast<std::size_t>(rows) * H, 61U + rows, 1.0F);
        Device<float> x(xs.size());
        x.upload(xs);
        enqueue_quantize_a8(x.data, H, rows, b.codes.data, b.deltas.data, b.sums.data, s);
        enqueue_shared(e, b, s, nullptr);
        check(hipStreamSynchronize(s), "shared");
        std::vector<A8Activation> host_x;
        for (std::uint32_t r = 0; r < rows; ++r) host_x.push_back(quantize_a8(std::span<const float>(xs.data() + static_cast<std::size_t>(r) * H, H)));
        const auto routes = build_window_routes(std::vector<std::uint32_t>(rows, 0U), rows, 1U, 1U);
        const std::vector<float> weights(rows, 1.0F), shared(static_cast<std::size_t>(rows) * H, 0.0F);
        const auto ref = reference_window(t, host_x, routes, weights, shared, 1U);
        const auto out = b.out.download();
        std::size_t bad = 0;
        for (std::size_t i = 0; i < out.size(); ++i) bad += std::memcmp(&out[i], &ref.raw[i], 4) != 0;
        expect(bad == 0U, "shared R" + std::to_string(rows) + " host check: " + std::to_string(bad) + " differ");
        std::cout << "HOST shared R" << rows << ": " << (bad ? "MISMATCH" : "bitwise equal to the host product") << '\n';
    }
}

// ---- Output head: this rank's vocabulary slice, Q8_0 x A8 ----

struct Head {
    Tensor t;
    strix::ds41::VocabSlice slice{};
    std::vector<std::uint8_t> host;
    Device<std::uint8_t> weight;
    Head(const Tensor& tensor_info, strix::ds41::VocabSlice s, const std::string& path)
        : t(tensor_info), slice(s), host(read_range(path, tensor_info.info->absolute_offset + tensor_info.row_bytes * s.first, tensor_info.row_bytes * s.count)),
          weight(host.size()) {
        weight.upload(host);
    }
};

void enqueue_head(const Head& h, const std::int8_t* codes, const float* deltas, std::uint32_t rows, float* logits, std::uint32_t vocab,
                  hipStream_t s) {
    strix::ds41::Q8A8View v;
    v.weight = h.weight.data; v.width = h.t.width; v.first = 0; v.count = h.slice.count; v.per_group = h.slice.count;
    v.codes = codes; v.deltas = deltas; v.rows = rows; v.x_stride = h.t.width; v.out_stride = vocab; v.out = logits + h.slice.first;
    strix::ds41::enqueue_q8_0_a8(v, s);
}

void bench_head(const Options& o, const Head& h, std::uint32_t vocab, Flush& flush, hipStream_t s, const std::vector<int>& forms) {
    auto& tuning = strix::ds41::moe_kernel_tuning();
    const auto saved = tuning;
    const auto H = h.t.width;
    for (const auto rows : o.rows) {
        Device<std::int8_t> codes(static_cast<std::size_t>(rows) * H);
        Device<float> deltas(rows * H / 32U);
        Device<std::int32_t> sums(rows * H / 32U);
        Device<float> x(static_cast<std::size_t>(rows) * H), ref(static_cast<std::size_t>(rows) * vocab), cand(static_cast<std::size_t>(rows) * vocab);
        x.upload(random_values(x.count, 71U + rows, 1.0F));
        strix::ds41::enqueue_quantize_a8(x.data, H, rows, codes.data, deltas.data, sums.data, s);
        Timer timer;
        const double bytes = static_cast<double>(h.t.row_bytes) * h.slice.count;
        for (const int form : forms) {
            tuning = saved; tuning.head = form;
            std::vector<double> t;
            for (unsigned rep = 0; rep < o.cases; ++rep) {
                flush(s);
                timer.mark(0, s);
                enqueue_head(h, codes.data, deltas.data, rows, cand.data, vocab, s);
                timer.mark(1, s);
                t.push_back(timer.ms(0, 1));
            }
            const double m = median(t);
            std::string verdict;
            if (o.check) {
                tuning.head = 0;
                ref.fill(0x5b);
                enqueue_head(h, codes.data, deltas.data, rows, ref.data, vocab, s);
                tuning.head = form;
                cand.fill(0x5b);
                enqueue_head(h, codes.data, deltas.data, rows, cand.data, vocab, s);
                check(hipStreamSynchronize(s), "head check");
                const bool same = same_bytes(ref, cand);
                expect(same, "head form " + std::to_string(form) + " R" + std::to_string(rows));
                verdict = same ? ", bitwise equal to form 0" : ", DIFFERS from form 0";
            }
            std::cout << std::fixed << std::setprecision(1) << "HEAD R" << rows << " form " << form << ": " << m * 1000.0 << " us, "
                      << bytes / (m * 1e6) << " GB/s (" << h.slice.count << " rows)" << verdict << '\n';
        }
    }
    tuning = saved;
}

void host_check_head(const Head& h, std::uint32_t vocab, hipStream_t s) {
    using namespace strix::ds41;
    const auto H = h.t.width;
    for (const std::uint32_t rows : {1U, 8U}) {
        const auto xs = random_values(static_cast<std::size_t>(rows) * H, 81U + rows, 1.0F);
        Device<float> x(xs.size());
        x.upload(xs);
        Device<std::int8_t> codes(static_cast<std::size_t>(rows) * H);
        Device<float> deltas(rows * H / 32U), logits(static_cast<std::size_t>(rows) * vocab);
        Device<std::int32_t> sums(rows * H / 32U);
        enqueue_quantize_a8(x.data, H, rows, codes.data, deltas.data, sums.data, s);
        enqueue_head(h, codes.data, deltas.data, rows, logits.data, vocab, s);
        check(hipStreamSynchronize(s), "head");
        const auto out = logits.download();
        std::size_t bad = 0, compared = 0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const auto a = quantize_a8(std::span<const float>(xs.data() + static_cast<std::size_t>(r) * H, H));
            for (std::uint32_t o = 0; o < h.slice.count; o += 97U) {
                const float ref = product_q8_0(std::span<const std::uint8_t>(h.host.data() + h.t.row_bytes * o, h.t.row_bytes), a, H);
                const float got = out[static_cast<std::size_t>(r) * vocab + h.slice.first + o];
                bad += std::memcmp(&ref, &got, 4) != 0;
                ++compared;
            }
        }
        expect(bad == 0U, "head host check R" + std::to_string(rows));
        std::cout << "HOST head R" << rows << ": " << compared << " logits, " << (bad ? "MISMATCH" : "bitwise equal to product_q8_0") << '\n';
    }
}

std::vector<int> forms_from_env(const char* name) {
    std::vector<int> forms{0};
    if (const char* v = std::getenv(name); v && *v)
        for (const auto f : parse_list(v)) if (f != 0U) forms.push_back(static_cast<int>(f));
    forms.push_back(-1);  // the default forms
    return forms;
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) { std::cerr << "usage: strix-hip-ds41-moe-bench MODEL.gguf [options]\n"; return 2; }
    Options o;
    o.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value"); return argv[++i]; };
        if (a == "--layer") o.layer = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--rank") o.rank = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--routes") o.routes = next();
        else if (a == "--rows") o.rows = parse_list(next());
        else if (a == "--cases") o.cases = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--reps") o.reps = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--part") { o.parts.clear(); std::stringstream in(next()); for (std::string p; std::getline(in, p, ',');) o.parts.insert(p); }
        else if (a == "--check") o.check = true;
        else if (a == "--same-rows") same_rows = true;
        else if (a == "--spread") spread_rows = true;
        else if (a == "--host-check") o.host_check = true;
        else throw std::invalid_argument("unknown option " + a);
    }
    const auto file = strix::gguf::read_file(o.model);
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    for (const auto& t : file.tensors) tensors[t.name] = &t;
    hipStream_t s{};
    check(hipStreamCreate(&s), "stream");  // blocking: ordered with the hipMemset clears
    Flush flush;
    const std::string prefix = "blk." + std::to_string(o.layer) + ".";
    // Kernel forms to compare (STRIX_DS41_MOE_BENCH_{ROUTED,SHARED,HEAD}_FORMS=1,2,...); form 0 is the reference.
    const auto routed_forms = forms_from_env("STRIX_DS41_MOE_BENCH_ROUTED_FORMS");
    const auto shared_forms = forms_from_env("STRIX_DS41_MOE_BENCH_SHARED_FORMS");
    const auto head_forms = forms_from_env("STRIX_DS41_MOE_BENCH_HEAD_FORMS");
    auto& tuning = strix::ds41::moe_kernel_tuning();
    const auto defaults = tuning;
    if (o.parts.count("routed")) {
        const auto g = tensor(tensors, prefix + "ffn_gate_exps.weight"), u = tensor(tensors, prefix + "ffn_up_exps.weight"),
                   d = tensor(tensors, prefix + "ffn_down_exps.weight");
        const std::uint32_t half = g.experts / 2U;
        const RankExperts e(g, u, d, o.rank * half, o.rank * half + half, o.model);
        const auto routes = o.routes.empty() ? std::vector<std::vector<std::uint32_t>>{} : load_routes(o.routes, o.layer, e.top_k);
        std::cout << "layer " << o.layer << " rank " << o.rank << ": experts [" << e.first << ", " << e.end << "), " << routes.size()
                  << " recorded tokens; gate/up " << e.expert_bytes_gate_up() / 1e6 << " MB and down " << e.expert_bytes_down() / 1e6 << " MB per expert\n";
        if (o.host_check) host_check_routed(o, e, routes, s);
        if (o.check) check_routed_variants(o, e, routes, s, routed_forms);
        for (const int form : routed_forms) {
            tuning = defaults; tuning.routed = form;
            std::cout << "-- routed form " << form << '\n';
            bench_routed(o, e, routes, flush, s);
        }
        tuning = defaults;
    }
    if (o.parts.count("shared") || o.parts.count("shared-split")) {
        const SharedExpert e(tensor(tensors, prefix + "ffn_gate_shexp.weight"), tensor(tensors, prefix + "ffn_up_shexp.weight"),
                             tensor(tensors, prefix + "ffn_down_shexp.weight"), o.model);
        if (o.parts.count("shared-split")) {
            for (const int form : shared_forms) {
                tuning = defaults; tuning.shared = form;
                std::cout << "-- shared split form " << form << '\n';
                bench_shared_split(o, e, flush, s);
            }
            tuning = defaults;
        }
        if (o.host_check) host_check_shared(o, e, s);
        if (o.check) check_shared_variants(o, e, s, shared_forms);
        for (const int form : o.parts.count("shared") ? shared_forms : std::vector<int>{}) {
            tuning = defaults; tuning.shared = form;
            std::cout << "-- shared form " << form << '\n';
            bench_shared(o, e, flush, s);
        }
        tuning = defaults;
    }
    if (o.parts.count("head")) {
        const auto t = tensor(tensors, "output.weight");
        const auto vocab = t.outputs;
        const Head h(t, strix::ds41::vocab_slice(vocab, 2U, o.rank), o.model);
        if (o.host_check) host_check_head(h, vocab, s);
        bench_head(o, h, vocab, flush, s, head_forms);
    }
    check(hipStreamDestroy(s), "stream");
    std::cout << (failures ? "FAILED" : "OK") << " (" << failures << " failures)\n";
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
