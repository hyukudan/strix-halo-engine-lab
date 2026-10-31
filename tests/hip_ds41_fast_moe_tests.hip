// FAST lever ds41-moe-relaxed-v1 vs the exact batch kernels: same routes and inputs, error of the raw slot vectors and
// the A8 activations, and timings of gate/up and down. Synthetic experts (production geometry: hidden 5120, inter 2304,
// IQ2_XXS gate/up, Q2_K down) or real blk.N (STRIX_DS41_Q2_GGUF=model.gguf, STRIX_DS41_FAST_MOE_LAYER=N).
// usage: strix-hip-ds41-fast-moe-tests [--no-timing] [--reps N]
#include "strix/ds41_moe.hpp"
#include "strix/ds41_prefill.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/hip_ds41_formats.hpp"
#include "strix/hip_ds41_prefill.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strix::ds41::WeightFormat;
int failures = 0;
void expect(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}

template <class T>
struct Device {
    T* data = nullptr;
    std::size_t count = 0;
    explicit Device(std::size_t n) : count(n) {
        check(hipMalloc(reinterpret_cast<void**>(&data), std::max<std::size_t>(n, 1U) * sizeof(T)), "hipMalloc");
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { (void)hipFree(data); }
    void upload(const T* host, std::size_t n) { check(hipMemcpy(data, host, n * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }
    std::vector<T> download() const {
        std::vector<T> out(count);
        check(hipMemcpy(out.data(), data, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return out;
    }
    void fill_pattern() { check(hipMemset(data, 0x7b, count * sizeof(T)), "poison"); }
};

template <class T>
std::size_t bit_mismatches(const std::vector<T>& a, const std::vector<T>& b) {
    std::size_t bad = a.size() != b.size();
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) bad += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return bad;
}

std::vector<std::uint8_t> read_range(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    file.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> out(bytes);
    file.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    if (!file) throw std::runtime_error("short read in " + path);
    return out;
}

// ---- Experts ----

struct Experts {
    WeightFormat gate_up = WeightFormat::iq2_xxs, down = WeightFormat::q2_k;
    std::uint32_t experts = 0, hidden = 0, inter = 0;
    std::vector<std::uint8_t> host[3]; // gate, up, down: whole [experts] tensors
    std::uint64_t gate_row() const { return hidden / strix::ds41::block_elements(gate_up) * strix::ds41::block_bytes(gate_up); }
    std::uint64_t down_row() const { return inter / strix::ds41::block_elements(down) * strix::ds41::block_bytes(down); }
    strix::ds41::ExpertTensors tensors() const {
        strix::ds41::ExpertTensors t;
        t.gate_up_format = gate_up; t.down_format = down; t.hidden = hidden; t.inter = inter;
        t.gate = host[0]; t.up = host[1]; t.down = host[2];
        return t;
    }
};
struct DeviceExperts {
    const Experts& e;
    Device<std::uint8_t> gate, up, down;
    explicit DeviceExperts(const Experts& x) : e(x), gate(x.host[0].size()), up(x.host[1].size()), down(x.host[2].size()) {
        gate.upload(x.host[0]); up.upload(x.host[1]); down.upload(x.host[2]);
    }
};

std::vector<std::uint8_t> synthetic_blocks(WeightFormat format, std::size_t blocks, unsigned seed) {
    using namespace strix::ds41;
    std::mt19937 random(seed);
    std::vector<std::uint8_t> bytes(blocks * block_bytes(format));
    for (auto& b : bytes) b = static_cast<std::uint8_t>(random());
    const auto size = block_bytes(format);
    for (std::size_t b = 0; b < blocks; ++b) {
        auto* p = bytes.data() + b * size;
        // Tame scales: FP16 exponents near 2^-12 .. 2^-9, E8M0 near 2^-10.
        if (format == WeightFormat::iq2_xxs || format == WeightFormat::q8_0) p[1] = static_cast<std::uint8_t>(0x10U | (p[1] & 0x83U));
        else if (format == WeightFormat::q2_k) {
            p[81] = static_cast<std::uint8_t>(0x10U | (p[81] & 0x83U));
            p[83] = static_cast<std::uint8_t>(0x10U | (p[83] & 0x83U));
        } else p[0] = static_cast<std::uint8_t>(118U + p[0] % 8U);
    }
    return bytes;
}
Experts synthetic_experts(WeightFormat gate_up, WeightFormat down, std::uint32_t experts, std::uint32_t hidden, std::uint32_t inter,
                          unsigned seed) {
    using namespace strix::ds41;
    Experts e;
    e.gate_up = gate_up; e.down = down; e.experts = experts; e.hidden = hidden; e.inter = inter;
    const auto gate_blocks = static_cast<std::size_t>(experts) * inter * (hidden / block_elements(gate_up));
    const auto down_blocks = static_cast<std::size_t>(experts) * hidden * (inter / block_elements(down));
    e.host[0] = synthetic_blocks(gate_up, gate_blocks, seed);
    e.host[1] = synthetic_blocks(gate_up, gate_blocks, seed + 1U);
    e.host[2] = synthetic_blocks(down, down_blocks, seed + 2U);
    return e;
}
WeightFormat format_of(std::uint32_t ggml_type) {
    if (ggml_type == 16U) return WeightFormat::iq2_xxs;
    if (ggml_type == 10U) return WeightFormat::q2_k;
    if (ggml_type == 8U) return WeightFormat::q8_0;
    if (ggml_type == 39U) return WeightFormat::mxfp4;
    throw std::runtime_error("unexpected expert tensor type " + std::to_string(ggml_type));
}
Experts file_experts(const std::string& path, const std::string& gate, const std::string& up, const std::string& down) {
    const auto file = strix::gguf::read_file(path);
    std::map<std::string, const strix::gguf::TensorInfo*> tensors;
    for (const auto& t : file.tensors) tensors[t.name] = &t;
    const auto find = [&](const std::string& name) {
        const auto found = tensors.find(name);
        if (found == tensors.end()) throw std::runtime_error("missing tensor " + name);
        return found->second;
    };
    const auto *g = find(gate), *u = find(up), *d = find(down);
    Experts e;
    e.gate_up = format_of(g->ggml_type); e.down = format_of(d->ggml_type);
    e.hidden = static_cast<std::uint32_t>(g->dimensions.at(0));
    e.inter = static_cast<std::uint32_t>(g->dimensions.at(1));
    e.experts = g->dimensions.size() > 2U ? static_cast<std::uint32_t>(g->dimensions[2]) : 1U;
    e.host[0] = read_range(path, g->absolute_offset, g->byte_size);
    e.host[1] = read_range(path, u->absolute_offset, u->byte_size);
    e.host[2] = read_range(path, d->absolute_offset, d->byte_size);
    if (e.host[0].size() != e.gate_row() * e.inter * e.experts || e.host[2].size() != e.down_row() * e.hidden * e.experts)
        throw std::runtime_error(gate + ": byte sizes do not match the geometry");
    return e;
}

// ---- Cases ----

std::vector<float> activations(std::uint32_t rows, std::uint32_t width, unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> values(static_cast<std::size_t>(rows) * width);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = normal(random) * std::ldexp(1.0F, static_cast<int>((i / 32U * 7U) % 11U) - 5);
    for (std::uint32_t r = 0; r < rows; r += 7U) { // a zero group and a tiny group in some rows
        auto* row = values.data() + static_cast<std::size_t>(r) * width;
        std::fill(row, row + 32, 0.0F);
        for (unsigned j = 32; j < 64; ++j) row[j] = 1e-37F * static_cast<float>(j);
    }
    return values;
}
// top_k distinct experts per row; skew > 0 concentrates the choice on a few experts (long CSR lists).
std::vector<std::uint32_t> routing(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, double skew, unsigned seed) {
    std::mt19937 random(seed);
    std::vector<double> weight(experts);
    for (std::uint32_t e = 0; e < experts; ++e) weight[e] = std::exp(-skew * static_cast<double>(e % 17U));
    std::vector<std::uint32_t> ids;
    for (std::uint32_t r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> row;
        while (row.size() < top_k) {
            std::discrete_distribution<std::uint32_t> pick(weight.begin(), weight.end());
            const auto e = pick(random);
            if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
        }
        ids.insert(ids.end(), row.begin(), row.end());
    }
    return ids;
}

struct RouteBuffers {
    Device<std::uint32_t> ids, counts, begins, entry_rows, entry_slots, batch_expert, batch_begin, batch_count, batch_total, order, low;
    strix::ds41::PrefillRoutesBuffers b;
    RouteBuffers(std::uint32_t rows, std::uint32_t top_k, std::uint32_t experts, std::uint32_t batch_rows)
        : ids(static_cast<std::size_t>(rows) * top_k), counts(experts), begins(experts + 1U),
          entry_rows(static_cast<std::size_t>(rows) * top_k), entry_slots(static_cast<std::size_t>(rows) * top_k),
          batch_expert(strix::ds41::prefill_max_batches(rows, top_k, experts, batch_rows)),
          batch_begin(batch_expert.count + 1U), batch_count(batch_expert.count), batch_total(1), order(static_cast<std::size_t>(rows) * top_k),
          low(rows) {
        b = {counts.data, begins.data, entry_rows.data, entry_slots.data, batch_expert.data, batch_begin.data, batch_count.data,
             batch_total.data, order.data, low.data};
    }
};

struct Outputs {
    std::vector<float> gate, up, raw, y;
    std::vector<std::int8_t> codes;
    std::vector<float> deltas;
    std::vector<std::int32_t> sums;
};

// Prefill path: device routes, WMMA gate/up, WMMA down, combine.
struct PrefillRun {
    Device<float> gate, up, raw, y;
    Device<std::int8_t> act_codes;
    Device<float> act_deltas;
    Device<std::int32_t> act_sums;
    PrefillRun(std::size_t slots, std::uint32_t rows, std::uint32_t hidden, std::uint32_t inter, bool keep)
        : gate(keep ? slots * inter : 1U), up(keep ? slots * inter : 1U), raw(slots * hidden), y(static_cast<std::size_t>(rows) * hidden),
          act_codes(slots * inter), act_deltas(slots * inter / 32U), act_sums(slots * inter / 32U) {}
};
struct Inputs {
    std::uint32_t rows = 0, top_k = 0;
    Device<std::int8_t> codes;
    Device<float> deltas;
    Device<std::int32_t> sums;
    Device<float> weights, shared;
    Inputs(std::uint32_t r, std::uint32_t k, std::uint32_t hidden, unsigned seed)
        : rows(r), top_k(k), codes(static_cast<std::size_t>(r) * hidden), deltas(static_cast<std::size_t>(r) * hidden / 32U),
          sums(static_cast<std::size_t>(r) * hidden / 32U), weights(static_cast<std::size_t>(r) * k), shared(static_cast<std::size_t>(r) * hidden) {
        const auto x = activations(r, hidden, seed);
        Device<float> values(x.size());
        values.upload(x);
        strix::ds41::enqueue_quantize_a8(values.data, hidden, r, codes.data, deltas.data, sums.data, nullptr);
        std::mt19937 random(seed + 7U);
        std::uniform_real_distribution<float> w(0.05F, 0.6F);
        std::normal_distribution<float> normal(0.0F, 0.5F);
        std::vector<float> wh(weights.count), sh(shared.count);
        for (auto& v : wh) v = w(random);
        for (auto& v : sh) v = normal(random);
        weights.upload(wh); shared.upload(sh);
        check(hipDeviceSynchronize(), "inputs");
    }
};

void enqueue_prefill(const DeviceExperts& x, const Inputs& in, const strix::ds41::PrefillRoutesView& routes, PrefillRun& run,
                     bool keep, hipEvent_t* events = nullptr) {
    using namespace strix::ds41;
    const auto& e = x.e;
    if (events) check(hipEventRecord(events[0], nullptr), "event");
    PrefillGateUpView gu;
    gu.gate = x.gate.data; gu.up = x.up.data; gu.expert_stride_bytes = e.gate_row() * e.inter; gu.row_bytes = e.gate_row();
    gu.hidden = e.hidden; gu.inter = e.inter; gu.routes = routes; gu.codes = in.codes.data; gu.deltas = in.deltas.data;
    gu.gate_out = keep ? run.gate.data : nullptr; gu.up_out = keep ? run.up.data : nullptr;
    gu.act_codes = run.act_codes.data; gu.act_deltas = run.act_deltas.data; gu.act_sums = run.act_sums.data;
    enqueue_prefill_gate_up(e.gate_up, gu, nullptr);
    if (events) check(hipEventRecord(events[1], nullptr), "event");
    PrefillDownView dn;
    dn.weights = x.down.data; dn.expert_stride_bytes = e.down_row() * e.hidden; dn.row_bytes = e.down_row();
    dn.width = e.inter; dn.outputs = e.hidden; dn.routes = routes; dn.codes = run.act_codes.data; dn.deltas = run.act_deltas.data;
    dn.out = run.raw.data;
    enqueue_prefill_down(e.down, dn, nullptr);
    if (events) check(hipEventRecord(events[2], nullptr), "event");
}


__global__ void flush_kernel(int* p, std::size_t n, int* sink) {
    int acc = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) acc += p[i];
    if (acc == 12345) *sink = acc;
}
bool timing_on = true;
unsigned reps = 7;
double rel_l2(const std::vector<float>& ref, const std::vector<float>& got, std::size_t& nonfinite) {
    double num = 0, den = 0;
    nonfinite = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) { ++nonfinite; continue; }
        const double d = double(got[i]) - ref[i];
        num += d * d; den += double(ref[i]) * ref[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}
void run_case(const DeviceExperts& x, std::uint32_t rows, std::uint32_t top_k, double skew, const std::string& label) {
    using namespace strix::ds41;
    const auto& e = x.e;
    const std::uint32_t batch_rows = 16U;
    const auto ids = routing(rows, top_k, e.experts, skew, 900U + rows);
    RouteBuffers rb(rows, top_k, e.experts, batch_rows);
    rb.ids.upload(ids);
    enqueue_build_prefill_routes(rb.ids.data, rows, top_k, e.experts, e.experts, batch_rows, rb.b, nullptr);
    Inputs in(rows, top_k, e.hidden, 700U + rows);
    const std::size_t slots = static_cast<std::size_t>(rows) * top_k;
    const auto view = prefill_routes_view(rb.b, rows, top_k, e.experts, batch_rows);
    Device<int> flush(64U * 1024U * 1024U), sink(1);
    check(hipMemset(flush.data, 0, flush.count * sizeof(int)), "flush");
    hipEvent_t ev[3];
    for (auto& v : ev) check(hipEventCreate(&v), "event");
    struct Result { std::vector<float> raw, deltas; std::vector<std::int8_t> codes; float gu = 0, dn = 0; };
    const auto run_mode = [&](int mode) {
        set_fast_moe(mode);
        PrefillRun run(slots, rows, e.hidden, e.inter, false);
        check(hipMemset(run.raw.data, 0, run.raw.count * sizeof(float)), "clear");
        enqueue_prefill(x, in, view, run, false);
        check(hipDeviceSynchronize(), "run");
        Result r;
        r.raw = run.raw.download(); r.codes = run.act_codes.download(); r.deltas = run.act_deltas.download();
        if (timing_on) {
            std::vector<float> gu, dn;
            for (unsigned rep = 0; rep < reps; ++rep) {
                hipLaunchKernelGGL(flush_kernel, dim3(1024), dim3(256), 0, nullptr, flush.data, flush.count, sink.data);
                enqueue_prefill(x, in, view, run, false, ev);
                check(hipEventSynchronize(ev[2]), "sync");
                float a = 0, b = 0;
                check(hipEventElapsedTime(&a, ev[0], ev[1]), "elapsed");
                check(hipEventElapsedTime(&b, ev[1], ev[2]), "elapsed");
                gu.push_back(a); dn.push_back(b);
            }
            std::sort(gu.begin(), gu.end()); std::sort(dn.begin(), dn.end());
            r.gu = gu[gu.size() / 2]; r.dn = dn[dn.size() / 2];
        }
        return r;
    };
    const Result exact = run_mode(0), fast = run_mode(1);
    const Result again = run_mode(1);
    set_fast_moe(-1);
    std::size_t nonfinite = 0;
    const double err = rel_l2(exact.raw, fast.raw, nonfinite);
    std::size_t code_diff = 0, code_far = 0;
    for (std::size_t i = 0; i < exact.codes.size(); ++i) {
        code_diff += exact.codes[i] != fast.codes[i];
        code_far += std::abs(int(exact.codes[i]) - int(fast.codes[i])) > 1;
    }
    expect(nonfinite == 0U, label + ": finite raw vectors");
    expect(err < 1e-3, label + ": raw relative L2 < 1e-3");
    expect(code_far == 0U, label + ": A8 codes within 1");
    expect(bit_mismatches(fast.raw, again.raw) == 0U && bit_mismatches(fast.codes, again.codes) == 0U, label + ": repeat bits");
    std::cout << "MOE " << label << " T=" << rows << " experts=" << e.experts << " skew=" << skew << " exact_gu_ms=" << exact.gu
              << " fast_gu_ms=" << fast.gu << " exact_dn_ms=" << exact.dn << " fast_dn_ms=" << fast.dn << " speedup="
              << (fast.gu + fast.dn > 0 ? (exact.gu + exact.dn) / (fast.gu + fast.dn) : 0.0) << " raw_rel_l2=" << err
              << " codes_diff=" << static_cast<double>(code_diff) / static_cast<double>(std::max<std::size_t>(exact.codes.size(), 1U)) << std::endl;
    for (auto& v : ev) (void)hipEventDestroy(v);
}
// Non-canonical batches (the views accept them; the builders never make them): every expert's 4th batch split into two
// batches of 8. The relaxed kernels must cover every entry exactly once (vs the exact kernels on the same view).
void custom_batches_case(const DeviceExperts& x, std::uint32_t rows, std::uint32_t top_k) {
    using namespace strix::ds41;
    const auto& e = x.e;
    const auto ids = routing(rows, top_k, e.experts, 0.0, 333U + rows);
    RouteBuffers rb(rows, top_k, e.experts, 16U);
    rb.ids.upload(ids);
    enqueue_build_prefill_routes(rb.ids.data, rows, top_k, e.experts, e.experts, 16U, rb.b, nullptr);
    check(hipDeviceSynchronize(), "routes");
    const auto total = rb.batch_total.download()[0];
    const auto be = rb.batch_expert.download(), bb = rb.batch_begin.download(), bc = rb.batch_count.download();
    std::vector<std::uint32_t> ne, nb, nc;
    unsigned split = 0;
    for (std::uint32_t i = 0, k = 0; i < total; ++i) {
        k = i > 0U && be[i - 1U] == be[i] ? k + 1U : 0U;
        if (k == 3U && bc[i] == 16U) {
            ne.insert(ne.end(), {be[i], be[i]}); nb.insert(nb.end(), {bb[i], bb[i] + 8U}); nc.insert(nc.end(), {8U, 8U});
            ++split;
        } else { ne.push_back(be[i]); nb.push_back(bb[i]); nc.push_back(bc[i]); }
    }
    nb.push_back(bb[total]);
    Device<std::uint32_t> de(ne.size()), db(nb.size()), dc(nc.size()), dt(1);
    de.upload(ne); db.upload(nb); dc.upload(nc);
    const std::vector<std::uint32_t> t1{static_cast<std::uint32_t>(ne.size())};
    dt.upload(t1);
    auto view = prefill_routes_view(rb.b, rows, top_k, e.experts, 16U);
    view.batch_expert = de.data; view.batch_begin = db.data; view.batch_count = dc.data; view.batch_total = dt.data;
    view.max_batches = static_cast<std::uint32_t>(ne.size());
    Inputs in(rows, top_k, e.hidden, 444U + rows);
    const std::size_t slots = static_cast<std::size_t>(rows) * top_k;
    const auto run_mode = [&](int mode) {
        set_fast_moe(mode);
        PrefillRun run(slots, rows, e.hidden, e.inter, false);
        check(hipMemset(run.raw.data, 0xff, run.raw.count * sizeof(float)), "poison");   // NaN: an unwritten entry shows
        enqueue_prefill(x, in, view, run, false);
        check(hipDeviceSynchronize(), "run");
        return run.raw.download();
    };
    const auto exact = run_mode(0), fast = run_mode(1);
    set_fast_moe(-1);
    std::size_t nonfinite = 0;
    const double err = rel_l2(exact, fast, nonfinite);
    expect(split > 0U, "custom batches: some batches split");
    expect(nonfinite == 0U && err < 1e-3, "custom batches: every entry written once, raw rel L2 < 1e-3");
    std::cout << "MOE custom-batches T=" << rows << " split=" << split << " raw_rel_l2=" << err << " nonfinite=" << nonfinite << std::endl;
}
}  // namespace

int main(int argc, char** argv) try {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-timing") timing_on = false;
        else if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = static_cast<unsigned>(std::stoul(argv[++i]));
    }
    {
        // 48 synthetic experts: T 1024 x top-6 = ~128 rows per expert (an 8K pair prefill), T 4096 = ~512 (32K)
        const auto s = synthetic_experts(WeightFormat::iq2_xxs, WeightFormat::q2_k, 48U, 5120U, 2304U, 11U);
        const DeviceExperts x(s);
        run_case(x, 77U, 6U, 0.0, "synthetic-small");
        custom_batches_case(x, 1024U, 6U);
        run_case(x, 1024U, 6U, 0.0, "synthetic");
        run_case(x, 1024U, 6U, 0.6, "synthetic-skew");
        run_case(x, 4096U, 6U, 0.0, "synthetic");
    }
    if (const char* path = std::getenv("STRIX_DS41_Q2_GGUF"); path && *path) {
        const char* l = std::getenv("STRIX_DS41_FAST_MOE_LAYER");
        const std::string p = "blk." + std::string(l && *l ? l : "20") + ".";
        const auto f = file_experts(path, p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight", p + "ffn_down_exps.weight");
        const DeviceExperts x(f);
        run_case(x, 4096U, 6U, 0.0, "real-" + p);      // 384 experts: ~64 rows each
        run_case(x, 16384U, 6U, 0.0, "real-" + p);     // ~256 rows each
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " (" << failures << " failures)" << std::endl;
    return failures == 0 ? 0 : 1;
} catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << std::endl;
    return 2;
}
