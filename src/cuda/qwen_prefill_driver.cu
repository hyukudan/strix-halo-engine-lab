// CUDA Qwen prefill driver (see include/strix/cuda_qwen_prefill.hpp): model upload, per-block step loop in the HIP
// step order, node state export. The arithmetic lives in the family libraries (qwen_hc, qwen_dense, qwen_gdn, qwen_qsa,
// qwen_moe, qwen_ple, qwen_mtp), each bit-identical to its HIP kernels; this file only wires buffers and order.
//
// HIP step order per layer (src/execution_plan.cpp:100-210; fixture notes/d64-dump-index.txt):
//   [ple (layer 1)] attn_hyper_mix -> {gated_delta_net | indexed_attention (L % 4 == 3)} -> attn_inject -> mlp_hyper_mix
//   -> moe_route -> shared_expert -> moe_experts -> moe_combine -> mlp_inject.
// Block prologue: token_embedding, hyper_expand. The MTP tap is the wide row after layer 47's mlp_inject. The final hyper
// mix and the LM head are not run: the node computes row p itself.
//
// Row-independent steps (HC mixes, dense projections, router, shared and routed experts, inject/combine) may run on
// padded row counts (the MoE needs >= 16 rows, a multiple of 8); steps that carry state across rows (GDN, QSA, PLE)
// run on the exact rows of the block.
#include "strix/cuda_qwen_prefill.hpp"

#include "strix/cuda_qwen_dense.hpp"
#include "strix/cuda_qwen_gdn.hpp"
#include "strix/cuda_qwen_hc.hpp"
#include "strix/cuda_qwen_moe.hpp"
#include "strix/cuda_qwen_plemtp.hpp"
#include "strix/cuda_qwen_qsa.hpp"
#include "strix/gguf_reader.hpp"

#include "qwen_amd_math_tables.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strix::cuda::qwen_prefill {
namespace {

namespace am = strix::cuda::amdmath;
namespace hc = strix::cuda::qwen_hc;
namespace dn = strix::cuda::qwen_dense;
namespace gdn = strix::cuda::qwen_gdn;
namespace qsa = strix::cuda::qwen_qsa;
namespace moe = strix::cuda::qwen_moe;
namespace pm = strix::cuda::qwen_plemtp;

using Clock = std::chrono::steady_clock;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr std::uint32_t kLayers = 48, kHidden = 2560, kWide = 10240, kTopK = 10, kExperts = 512;
constexpr bool is_qsa(std::uint32_t layer) { return layer % 4U == 3U; }
constexpr std::uint32_t gdn_ordinal(std::uint32_t layer) { return layer - (layer + 1U) / 4U; }
constexpr std::uint32_t qsa_ordinal(std::uint32_t layer) { return layer / 4U; }

std::uint32_t round_up(std::uint32_t v, std::uint32_t m) { return (v + m - 1U) / m * m; }

// Device arena of the uploaded model: one cudaMalloc per tensor (keeps alignment simple; ~1200 tensors).
struct Weights {
    std::map<std::string, void*> tensors;
    std::uint64_t bytes = 0;
    void* get(const std::string& name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("model tensor not uploaded: " + name);
        return it->second;
    }
};

template <class T> T* dev_alloc(std::uint64_t count, const char* what) {
    void* p = nullptr;
    ck(cudaMalloc(&p, std::max<std::uint64_t>(count, 1) * sizeof(T)), what);
    return static_cast<T*>(p);
}

}  // namespace

struct Prefill::Impl {
    Config cfg;
    cudaStream_t stream = nullptr;
    cudaStream_t copy_stream = nullptr;                    // release_by_layer: per-layer D2H
    std::vector<cudaEvent_t> layer_done, copy_done;        // [kLayers]
    std::function<void(std::uint32_t)> after_layer;        // called (host) after layer l's launches of a block
    am::Tables tables{};
    qsa::RopeTable* rope = nullptr;
    gguf::FileInfo info;
    Weights w;
    std::vector<hc::HcPacked> hc_attn, hc_ffn;  // per layer
    std::vector<void*> hc_storage;
    std::uint32_t cap = 0;                     // rows per block (padded capacity)

    // ---- per-block activation buffers (row-major, capacity cap)
    std::uint32_t* ids = nullptr;
    std::uint16_t *emb = nullptr, *wide_a = nullptr, *wide_b = nullptr, *hidden = nullptr, *step_out = nullptr;
    float *injection = nullptr, *normalized = nullptr;
    std::int8_t* codes = nullptr;
    float* deltas = nullptr;
    float* gdn_rows = nullptr;
    void* gdn_scratch = nullptr;
    std::uint64_t gdn_stride = 0;
    float *q_gate = nullptr, *qk = nullptr, *qv = nullptr, *index_q = nullptr, *index_k = nullptr, *attn_out = nullptr;
    qsa::Workspace qws{};
    float* logits = nullptr;
    std::uint32_t* routes = nullptr;
    float* route_w = nullptr;
    std::uint16_t *shared_out = nullptr, *routed_out = nullptr, *combined = nullptr;
    float* shared_ws = nullptr;
    // routed MoE per-row buffers
    std::int32_t* moe_sums = nullptr;
    std::uint32_t* moe_status = nullptr;
    float *moe_acc = nullptr, *moe_ddeltas = nullptr, *moe_raw = nullptr;
    std::int8_t *moe_codes = nullptr, *moe_dcodes = nullptr;
    float* moe_deltas = nullptr;
    std::int32_t* moe_dsums = nullptr;
    moe::Csr csr{};
    std::uint16_t* tap = nullptr;  // wide rows after layer 47 (MTP)

    // ---- PLE (one-based layer 2): IQ4_NL table in mapped pinned host memory, gathered on the device
    std::uint8_t* ple_table_host = nullptr;
    const std::uint8_t* ple_table_dev = nullptr;
    std::uint64_t ple_table_rows = 0;
    pm::PleHashContract ple_contract{};
    std::vector<std::uint64_t> ple_ids_host;
    std::uint64_t* ple_ids = nullptr;
    std::uint16_t* ple_rows = nullptr;
    float *ple_keys = nullptr, *ple_values = nullptr, *ple_gates = nullptr;
    std::uint32_t* ple_status = nullptr;
    std::uint32_t ngram[2] = {0, 0};

    // ---- MTP predictor cache rows (sidecar)
    bool mtp = false;
    Weights mw;                    // sidecar tensors on the device
    void* mtp_eh_q4 = nullptr;
    void* mtp_ws = nullptr;
    std::uint32_t* mtp_next = nullptr;
    std::uint32_t* mtp_status = nullptr;
    std::uint16_t *mtp_kv = nullptr, *mtp_index = nullptr;
    std::vector<std::uint32_t> next_host;

    // ---- session state
    std::vector<std::uint16_t*> kv, raw_index;  // per QSA ordinal: [max_positions][1024] / [max_positions][128]
    float* gdn_state = nullptr;                 // [36][48*128*128]
    std::uint16_t* gdn_conv = nullptr;          // [36][10240*4]
    std::uint16_t* ple_conv = nullptr;          // [10240*9]

    // ---- host staging of the exported state (pinned)
    std::uint8_t* host = nullptr;
    std::uint64_t host_bytes = 0;

    std::string describe;

    void upload_model();
    void upload_ple();
    void upload_mtp();
    void allocate();
    void run_block(const std::vector<std::uint32_t>& tokens, std::uint64_t p0, std::uint32_t rows);
};

void Prefill::Impl::upload_model() {
    info = gguf::read_file(cfg.model_gguf);
    std::FILE* f = std::fopen(cfg.model_gguf.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + cfg.model_gguf);
    constexpr std::uint64_t kChunk = std::uint64_t{256} << 20;
    std::uint8_t* staging[2] = {nullptr, nullptr};
    ck(cudaMallocHost(&staging[0], kChunk), "pinned staging");
    ck(cudaMallocHost(&staging[1], kChunk), "pinned staging");
    cudaEvent_t done[2];
    ck(cudaEventCreate(&done[0]), "event");
    ck(cudaEventCreate(&done[1]), "event");
    ck(cudaEventRecord(done[0], stream), "event");
    ck(cudaEventRecord(done[1], stream), "event");
    int slot = 0;
    for (const auto& t : info.tensors) {
        if (t.name == "output.weight" || t.name.rfind("per_layer_token_embd", 0) == 0) continue;  // head (node) / PLE (host)
        void* d = dev_alloc<std::uint8_t>(t.byte_size, t.name.c_str());
        _fseeki64(f, static_cast<long long>(t.absolute_offset), SEEK_SET);
        for (std::uint64_t off = 0; off < t.byte_size; off += kChunk) {
            const auto n = std::min(kChunk, t.byte_size - off);
            ck(cudaEventSynchronize(done[slot]), "staging reuse");
            if (std::fread(staging[slot], 1, static_cast<std::size_t>(n), f) != n) throw std::runtime_error("short read " + t.name);
            ck(cudaMemcpyAsync(static_cast<std::uint8_t*>(d) + off, staging[slot], n, cudaMemcpyHostToDevice, stream), "upload");
            ck(cudaEventRecord(done[slot], stream), "event");
            slot ^= 1;
        }
        w.tensors[t.name] = d;
        w.bytes += t.byte_size;
    }
    ck(cudaStreamSynchronize(stream), "upload sync");
    std::fclose(f);
    cudaFreeHost(staging[0]);
    cudaFreeHost(staging[1]);
    cudaEventDestroy(done[0]);
    cudaEventDestroy(done[1]);
    // HC: pack every mixer's down/up once; the raw Q8_0 copies are dropped.
    hc_attn.resize(kLayers);
    hc_ffn.resize(kLayers);
    auto pack = [&](const std::string& prefix) {
        void* base = dev_alloc<std::uint8_t>(hc::hc_packed_bytes(), "hc packed");
        hc_storage.push_back(base);
        const auto packed = hc::hc_packed_view(base);
        hc::hc_pack(w.get(prefix + "_down.weight"), w.get(prefix + "_up.weight"), packed, stream);
        return packed;
    };
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        hc_attn[l] = pack("blk." + std::to_string(l) + ".hc_attn");
        hc_ffn[l] = pack("blk." + std::to_string(l) + ".hc_ffn");
    }
    ck(cudaStreamSynchronize(stream), "hc pack");
}

void Prefill::Impl::upload_ple() {
    const gguf::TensorInfo* t = nullptr;
    for (const auto& x : info.tensors) if (x.name == "per_layer_token_embd.weight") t = &x;
    if (!t) throw std::runtime_error("per_layer_token_embd.weight not in the model");
    ple_table_rows = t->byte_size / pm::kIq4RowBytes;
    ck(cudaHostAlloc(&ple_table_host, t->byte_size, cudaHostAllocMapped | cudaHostAllocPortable), "pinned PLE table");
    std::FILE* f = std::fopen(cfg.model_gguf.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + cfg.model_gguf);
    _fseeki64(f, static_cast<long long>(t->absolute_offset), SEEK_SET);
    constexpr std::uint64_t kChunk = std::uint64_t{1} << 30;
    for (std::uint64_t off = 0; off < t->byte_size; off += kChunk) {
        const auto n = std::min(kChunk, t->byte_size - off);
        if (std::fread(ple_table_host + off, 1, static_cast<std::size_t>(n), f) != n) throw std::runtime_error("short read PLE table");
    }
    std::fclose(f);
    void* d = nullptr;
    ck(cudaHostGetDevicePointer(&d, ple_table_host, 0), "PLE device view");
    ple_table_dev = static_cast<const std::uint8_t*>(d);
    ple_contract = pm::ple_hash_contract();
}

void Prefill::Impl::upload_mtp() {
    if (cfg.mtp_sidecar_gguf.empty()) return;
    const auto sinfo = gguf::read_file(cfg.mtp_sidecar_gguf);
    std::FILE* f = std::fopen(cfg.mtp_sidecar_gguf.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + cfg.mtp_sidecar_gguf);
    const char* wanted[] = {"blk.48.nextn.enorm.weight", "blk.48.nextn.hnorm.weight", "blk.48.nextn.eh_proj.weight",
                            "blk.48.hc_attn_norm.weight", "blk.48.hc_attn_down.weight", "blk.48.hc_attn_up.weight",
                            "blk.48.indexer.k_proj.weight", "blk.48.attn_k.weight", "blk.48.attn_v.weight",
                            "blk.48.attn_k_norm.weight"};
    for (const char* name : wanted) {
        const gguf::TensorInfo* t = nullptr;
        for (const auto& x : sinfo.tensors) if (x.name == name) t = &x;
        if (!t) throw std::runtime_error(std::string("MTP sidecar lacks ") + name);
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(t->byte_size));
        _fseeki64(f, static_cast<long long>(t->absolute_offset), SEEK_SET);
        if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) throw std::runtime_error(std::string("short read ") + name);
        void* d = dev_alloc<std::uint8_t>(bytes.size(), name);
        ck(cudaMemcpy(d, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "mtp upload");
        mw.tensors[name] = d;
        mw.bytes += bytes.size();
    }
    std::fclose(f);
    const std::uint64_t eh_blocks = std::uint64_t{kHidden} * 2 * kHidden / 32;  // [2560][5120]
    mtp_eh_q4 = dev_alloc<std::uint8_t>(eh_blocks * 18, "mtp eh q4");
    pm::launch_mtp_q8_to_q4(mw.get("blk.48.nextn.eh_proj.weight"), eh_blocks, mtp_eh_q4, stream);
    ck(cudaStreamSynchronize(stream), "mtp q4");
    mtp = true;
}

void Prefill::Impl::allocate() {
    cap = round_up(*std::max_element(cfg.block_schedule.begin(), cfg.block_schedule.end()), 16U);
    const std::uint64_t R = cap;
    ids = dev_alloc<std::uint32_t>(R, "ids");
    emb = dev_alloc<std::uint16_t>(R * kHidden, "emb");
    wide_a = dev_alloc<std::uint16_t>(R * kWide, "wide a");
    wide_b = dev_alloc<std::uint16_t>(R * kWide, "wide b");
    tap = dev_alloc<std::uint16_t>(R * kWide, "tap");
    hidden = dev_alloc<std::uint16_t>(R * kHidden, "hidden");
    step_out = dev_alloc<std::uint16_t>(R * kHidden, "step out");
    injection = dev_alloc<float>(R * 4, "injection");
    normalized = dev_alloc<float>(R * hc::kNormalizedPitch, "hc normalized");
    codes = dev_alloc<std::int8_t>(R * 6144, "a8 codes");
    deltas = dev_alloc<float>(R * (6144 / 32), "a8 deltas");
    gdn_stride = 10240 + 6144 + 48 + 48 + 6144;
    gdn_rows = dev_alloc<float>(R * gdn_stride, "gdn rows");
    ck(cudaMalloc(&gdn_scratch, gdn::core_scratch_bytes(cap)), "gdn scratch");
    q_gate = dev_alloc<float>(R * qsa::kQGateWidth, "q gate");
    qk = dev_alloc<float>(R * qsa::kKvWidth, "k");
    qv = dev_alloc<float>(R * qsa::kKvWidth, "v");
    index_q = dev_alloc<float>(R * qsa::kIndexQWidth, "index q");
    index_k = dev_alloc<float>(R * qsa::kIndexDim, "index k");
    attn_out = dev_alloc<float>(R * qsa::kAttnWidth, "attention out");
    qws = qsa::allocate_workspace(std::min<std::uint32_t>(cap, 1024U), cfg.max_positions);
    logits = dev_alloc<float>(R * kExperts, "router logits");
    routes = dev_alloc<std::uint32_t>(R * kTopK, "routes");
    route_w = dev_alloc<float>(R * kTopK, "route weights");
    shared_out = dev_alloc<std::uint16_t>(R * kHidden, "shared out");
    routed_out = dev_alloc<std::uint16_t>(R * kHidden, "routed out");
    combined = dev_alloc<std::uint16_t>(R * kHidden, "combined");
    shared_ws = dev_alloc<float>(R * 1280, "shared workspace");
    moe_codes = dev_alloc<std::int8_t>(R * kHidden, "moe codes");
    moe_deltas = dev_alloc<float>(R * (kHidden / 32), "moe deltas");
    moe_sums = dev_alloc<std::int32_t>(R * (kHidden / 32), "moe sums");
    moe_status = dev_alloc<std::uint32_t>(R * 4, "moe status");
    moe_acc = dev_alloc<float>(R * kHidden, "moe accumulation");
    moe_dcodes = dev_alloc<std::int8_t>(R * kTopK * 640, "moe down codes");
    moe_ddeltas = dev_alloc<float>(R * kTopK * 640 / 32, "moe down deltas");
    moe_dsums = dev_alloc<std::int32_t>(R * kTopK * 640 / 32, "moe down sums");
    moe_raw = dev_alloc<float>(R * kTopK * kHidden, "moe raw dots");
    const auto sz = moe::csr_sizes(cap);
    csr.summary = dev_alloc<std::uint32_t>(32, "csr");
    csr.offsets = dev_alloc<std::uint32_t>(kExperts + 1, "csr");
    csr.touched = dev_alloc<std::uint32_t>(kExperts, "csr");
    csr.rows = dev_alloc<std::uint32_t>(sz.assignments, "csr");
    csr.slots = dev_alloc<std::uint32_t>(sz.assignments, "csr");
    csr.batch_experts = dev_alloc<std::uint32_t>(sz.max_batches, "csr");
    csr.batch_begins = dev_alloc<std::uint32_t>(sz.max_batches + 1, "csr");
    csr.tile_count = dev_alloc<std::uint32_t>(1, "csr");
    csr.tile_experts = dev_alloc<std::uint32_t>(sz.max_tiles, "csr");
    csr.tile_begins = dev_alloc<std::uint32_t>(sz.max_tiles + 1, "csr");
    // session state
    const std::uint64_t P = cfg.max_positions;
    for (std::uint32_t o = 0; o < 12; ++o) {
        kv.push_back(dev_alloc<std::uint16_t>(P * 1024, "kv"));
        raw_index.push_back(dev_alloc<std::uint16_t>(P * 128, "raw index"));
    }
    gdn_state = dev_alloc<float>(36 * gdn::kMatrixFloats, "gdn state");
    gdn_conv = dev_alloc<std::uint16_t>(36 * gdn::kConvStateHalfs, "gdn conv");
    ple_conv = dev_alloc<std::uint16_t>(10240 * 9, "ple conv");
    ple_ids_host.resize(R * pm::kPleHeads);
    ple_ids = dev_alloc<std::uint64_t>(R * pm::kPleHeads, "ple ids");
    ple_rows = dev_alloc<std::uint16_t>(R * pm::kPleWidth, "ple rows");
    ple_keys = dev_alloc<float>(R * pm::kExpanded, "ple keys");
    ple_values = dev_alloc<float>(R * pm::kHidden, "ple values");
    ple_gates = dev_alloc<float>(R * 4, "ple gates");
    ple_status = dev_alloc<std::uint32_t>(1, "ple status");
    if (mtp) {
        mtp_kv = dev_alloc<std::uint16_t>(P * 1024, "mtp kv");
        mtp_index = dev_alloc<std::uint16_t>(P * 128, "mtp index");
        ck(cudaMalloc(&mtp_ws, pm::mtp_workspace_bytes(cap)), "mtp workspace");
        mtp_next = dev_alloc<std::uint32_t>(R, "mtp next");
        mtp_status = dev_alloc<std::uint32_t>(1, "mtp status");
        next_host.resize(R);
    }
    // host staging: every row component's rows plus the fixed blob
    host_bytes = P * (12ULL * 2048 + 12ULL * 256 + 2048 + 256) + 36 * gdn::kMatrixFloats * 4 + 36 * gdn::kConvStateHalfs * 2 +
                 10240ULL * 9 * 2 + 16 + kWide * 2;
    ck(cudaMallocHost(&host, host_bytes), "pinned export staging");
}

Prefill::Prefill(const Config& config) : impl_(std::make_unique<Impl>()) {
    auto& m = *impl_;
    m.cfg = config;
    ck(cudaSetDevice(config.device), "set device");
    ck(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking), "stream");
    if (config.release_by_layer) {
        ck(cudaStreamCreateWithFlags(&m.copy_stream, cudaStreamNonBlocking), "copy stream");
        m.layer_done.resize(kLayers);
        m.copy_done.resize(kLayers);
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            ck(cudaEventCreateWithFlags(&m.layer_done[l], cudaEventDisableTiming), "layer event");
            ck(cudaEventCreateWithFlags(&m.copy_done[l], cudaEventDisableTiming), "copy event");
        }
    }
    m.tables = am::load_tables(config.tables_dir);
    {
        std::FILE* f = std::fopen(config.rope_table.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open rope table " + config.rope_table);
        std::vector<qsa::RopeTable> cs(static_cast<std::size_t>(config.max_positions) * qsa::kRotaryPairs);
        if (std::fread(cs.data(), sizeof(qsa::RopeTable), cs.size(), f) != cs.size())
            throw std::runtime_error("rope table shorter than max_positions");
        std::fclose(f);
        m.rope = dev_alloc<qsa::RopeTable>(cs.size(), "rope");
        ck(cudaMemcpy(m.rope, cs.data(), cs.size() * sizeof(qsa::RopeTable), cudaMemcpyHostToDevice), "rope upload");
    }
    const auto t0 = Clock::now();
    m.upload_model();
    m.upload_ple();
    m.upload_mtp();
    m.allocate();
    char buf[256];
    std::snprintf(buf, sizeof(buf), "{\"weights_gb\":%.2f,\"upload_s\":%.1f,\"block_rows\":%u,\"max_positions\":%u}",
                  static_cast<double>(m.w.bytes) / 1e9, std::chrono::duration<double>(Clock::now() - t0).count(), m.cap,
                  config.max_positions);
    m.describe = buf;
}

Prefill::~Prefill() = default;

std::string Prefill::describe() const { return impl_->describe; }

void Prefill::Impl::run_block(const std::vector<std::uint32_t>& tokens, std::uint64_t p0, std::uint32_t rows) {
    const std::uint32_t pad = std::max<std::uint32_t>(16U, round_up(rows, 8U));  // row-independent steps
    ck(cudaMemsetAsync(ids, 0, std::uint64_t{pad} * 4, stream), "ids clear");
    ck(cudaMemcpyAsync(ids, tokens.data() + p0, std::uint64_t{rows} * 4, cudaMemcpyHostToDevice, stream), "ids");
    hc::token_embedding(w.get("token_embd.weight"), ids, pad, emb, stream);
    hc::hyper_expand(emb, pad, wide_a, stream);
    std::uint16_t* wide = wide_a;
    std::uint16_t* wide_next = wide_b;
    const auto mix = [&](const std::string& prefix, const hc::HcPacked& packed) {
        hc::HcMixArgs a;
        a.input = wide;
        a.norm = static_cast<const float*>(w.get(prefix + "_norm.weight"));
        a.inject = static_cast<const float*>(w.get(prefix + "_inject.weight"));
        a.packed = packed;
        a.rows = pad;
        a.normalized = normalized;
        a.injection = injection;
        a.output = hidden;
        hc::hc_mix(a, tables, stream);
    };
    const auto inject = [&](const std::uint16_t* value) {
        hc::inject(wide, value, injection, pad, wide_next, stream);
        std::swap(wide, wide_next);
    };
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        const std::string blk = "blk." + std::to_string(l);
        if (l == 1) {
            // PLE (one-based layer 2): host hashing, device gather from the mapped table, key/value projections, tail
            pm::ple_hash_rows(ple_contract, tokens.data() + p0, rows, p0, ngram, ple_ids_host.data());
            ck(cudaMemcpyAsync(ple_ids, ple_ids_host.data(), std::uint64_t{rows} * pm::kPleHeads * 8, cudaMemcpyHostToDevice, stream),
               "ple ids");
            pm::launch_ple_gather_iq4nl(ple_table_dev, ple_table_rows, ple_ids, rows, ple_rows, pm::kPleWidth, ple_status, stream);
            pm::launch_ple_project(ple_rows, pm::kPleWidth, rows, w.get("blk.1.ple_key.weight"), pm::kPleWidth, pm::kExpanded,
                                   ple_keys, pm::kExpanded, stream);
            pm::launch_ple_project(ple_rows, pm::kPleWidth, rows, w.get("blk.1.ple_value.weight"), pm::kPleWidth, pm::kHidden,
                                   ple_values, pm::kHidden, stream);
            pm::PleTailArgs ta{};
            ta.keys = ple_keys; ta.key_stride = pm::kExpanded;
            ta.values = ple_values; ta.value_stride = pm::kHidden;
            ta.input = wide; ta.input_stride = kWide;
            ta.output = wide; ta.output_stride = kWide;
            ta.conv_state = ple_conv;
            ta.norm_key = static_cast<const float*>(w.get("blk.1.ple_norm_key.weight"));
            ta.norm_query = static_cast<const float*>(w.get("blk.1.ple_norm_query.weight"));
            ta.norm_conv = static_cast<const float*>(w.get("blk.1.ple_norm_conv.weight"));
            ta.conv = static_cast<const float*>(w.get("blk.1.ple_conv1d.weight"));
            ta.gates = ple_gates;
            ta.rows = rows;
            ta.epsilon = 1e-6F;
            pm::launch_ple_tail(ta, tables, stream);
        }
        mix(blk + ".hc_attn", hc_attn[l]);
        if (!is_qsa(l)) {
            // gated delta net: qkv||z (A8), core, ssm_out (A8)
            const gdn::Rows gr{gdn_rows, gdn_rows + 10240, gdn_rows + 16384, gdn_rows + 16432, gdn_rows + 16480, gdn_stride};
            dn::Site in;
            in.store = dn::Store::f32_round_bf16;
            in.width = kHidden;
            in.segments = 2;
            in.segment[0] = {w.get(blk + ".attn_qkv.weight"), 10240, gr.qkv, gdn_stride};
            in.segment[1] = {w.get(blk + ".attn_gate.weight"), 6144, gr.z, gdn_stride};
            ck(dn::quantize_project(stream, in, pad, dn::Input::bf16, hidden, kHidden, codes, deltas, cap), "gdn in");
            gdn::Weights gw{static_cast<const float*>(w.get(blk + ".ssm_beta.weight")), static_cast<const float*>(w.get(blk + ".ssm_alpha.weight")),
                            static_cast<const float*>(w.get(blk + ".ssm_conv1d.weight")), static_cast<const float*>(w.get(blk + ".ssm_dt.bias")),
                            static_cast<const float*>(w.get(blk + ".ssm_a")), static_cast<const float*>(w.get(blk + ".ssm_norm.weight"))};
            const auto o = gdn_ordinal(l);
            gdn::launch_core(hidden, kHidden, gr, rows, gdn_conv + o * gdn::kConvStateHalfs, gdn_state + o * gdn::kMatrixFloats,
                             gdn_scratch, gw, tables, stream);
            dn::Site out;
            out.store = dn::Store::bf16;
            out.width = 6144;
            out.segments = 1;
            out.segment[0] = {w.get(blk + ".ssm_out.weight"), kHidden, step_out, kHidden};
            ck(dn::quantize_project(stream, out, pad, dn::Input::f32, gr.core, gdn_stride, codes, deltas, cap), "gdn out");
        } else {
            dn::Site in;
            in.store = dn::Store::f32_round_bf16;
            in.width = kHidden;
            in.segments = 3;
            in.segment[0] = {w.get(blk + ".attn_q.weight"), qsa::kQGateWidth, q_gate, qsa::kQGateWidth};
            in.segment[1] = {w.get(blk + ".attn_k.weight"), qsa::kKvWidth, qk, qsa::kKvWidth};
            in.segment[2] = {w.get(blk + ".attn_v.weight"), qsa::kKvWidth, qv, qsa::kKvWidth};
            ck(dn::quantize_project(stream, in, pad, dn::Input::bf16, hidden, kHidden, codes, deltas, cap), "qsa in");
            const qsa::RowBuffers rb{hidden, q_gate, qk, qv, index_q, index_k, attn_out};
            const qsa::LayerWeights lw{static_cast<const std::uint16_t*>(w.get(blk + ".indexer.q_proj.weight")),
                                       static_cast<const std::uint16_t*>(w.get(blk + ".indexer.k_proj.weight")),
                                       static_cast<const float*>(w.get(blk + ".attn_q_norm.weight")),
                                       static_cast<const float*>(w.get(blk + ".attn_k_norm.weight")),
                                       static_cast<const float*>(w.get(blk + ".indexer.q_norm.weight")),
                                       static_cast<const float*>(w.get(blk + ".indexer.k_norm.weight"))};
            const auto o = qsa_ordinal(l);
            const qsa::LayerState st{kv[o], raw_index[o]};
            qsa::layer_core(rb, rows, p0, lw, st, rope, tables, qws, stream);
            dn::Site out;
            out.store = dn::Store::bf16;
            out.width = qsa::kAttnWidth;
            out.segments = 1;
            out.segment[0] = {w.get(blk + ".attn_output.weight"), kHidden, step_out, kHidden};
            ck(dn::quantize_project(stream, out, pad, dn::Input::f32, attn_out, qsa::kAttnWidth, codes, deltas, cap), "qsa out");
        }
        inject(step_out);
        mix(blk + ".hc_ffn", hc_ffn[l]);
        hc::route_logits(hidden, static_cast<const float*>(w.get(blk + ".ffn_gate_inp.weight")), pad, logits, stream);
        hc::route_select(logits, pad, kTopK, true, tables, routes, route_w, stream);
        dn::SharedExpert sw;
        sw.gate_q8 = w.get(blk + ".ffn_gate_shexp.weight");
        sw.up_q8 = w.get(blk + ".ffn_up_shexp.weight");
        sw.down_q8 = w.get(blk + ".ffn_down_shexp.weight");
        sw.scalar = static_cast<const float*>(w.get(blk + ".ffn_gate_inp_shexp.weight"));
        ck(dn::shared_expert(stream, pad, sw, hidden, kHidden, shared_ws, 1280, shared_out, kHidden, codes, deltas, cap, tables),
           "shared expert");
        moe::Rows mr;
        mr.input = {hidden, kHidden * 2};
        mr.routes = {routes, kTopK * 4};
        mr.route_weights = {route_w, kTopK * 4};
        mr.codes = {moe_codes, kHidden};
        mr.deltas = {moe_deltas, kHidden / 32 * 4};
        mr.sums = {moe_sums, kHidden / 32 * 4};
        mr.status = {moe_status, 16};
        mr.intermediates = {nullptr, 0};
        mr.accumulation = {moe_acc, kHidden * 4};
        mr.down_codes = {moe_dcodes, kTopK * 640};
        mr.down_deltas = {moe_ddeltas, kTopK * 640 / 32 * 4};
        mr.down_sums = {moe_dsums, kTopK * 640 / 32 * 4};
        mr.rawdots = {moe_raw, kTopK * kHidden * 4};
        mr.output = {routed_out, kHidden * 2};
        mr.csr = csr;
        const auto& gate = info.tensors;  // per-layer expert types
        moe::Weights ew;
        for (const auto& t : gate) {
            if (t.name == blk + ".ffn_gate_exps.weight") ew.gate_type = t.ggml_type;
            if (t.name == blk + ".ffn_down_exps.weight") ew.down_type = t.ggml_type;
        }
        ew.gate = static_cast<const std::uint8_t*>(w.get(blk + ".ffn_gate_exps.weight"));
        ew.up = static_cast<const std::uint8_t*>(w.get(blk + ".ffn_up_exps.weight"));
        ew.down = static_cast<const std::uint8_t*>(w.get(blk + ".ffn_down_exps.weight"));
        moe::enqueue(mr, pad, kHidden, ew, tables, nullptr, stream);
        hc::combine(shared_out, routed_out, pad, combined, stream);
        inject(combined);
        if (l == kLayers - 1U) ck(cudaMemcpyAsync(tap, wide, std::uint64_t{pad} * kWide * 2, cudaMemcpyDeviceToDevice, stream), "tap");
        if (after_layer) after_layer(l);
    }
    ck(cudaGetLastError(), "block launches");
}

Timing Prefill::run(const std::vector<std::uint32_t>& tokens, std::uint64_t position, const Sink& sink) {
    auto& m = *impl_;
    if (position > tokens.size() || position == 0) throw std::invalid_argument("prefill position out of range");
    if (position > m.cfg.max_positions) throw std::invalid_argument("prompt longer than max_positions");
    const auto t0 = Clock::now();
    Timing timing;
    // fresh state
    for (std::uint32_t o = 0; o < 12; ++o) {
        ck(cudaMemsetAsync(m.kv[o], 0, position * 2048, m.stream), "kv clear");
        ck(cudaMemsetAsync(m.raw_index[o], 0, position * 256, m.stream), "index clear");
    }
    ck(cudaMemsetAsync(m.gdn_state, 0, 36 * gdn::kMatrixFloats * 4, m.stream), "gdn clear");
    ck(cudaMemsetAsync(m.gdn_conv, 0, 36 * gdn::kConvStateHalfs * 2, m.stream), "conv clear");
    ck(cudaMemsetAsync(m.ple_conv, 0, 10240 * 9 * 2, m.stream), "ple conv clear");
    std::uint64_t p0 = 0;
    std::size_t k = 0;
    std::uint8_t* out = m.host;
    const std::uint8_t* held = nullptr;
    m.ngram[0] = m.ngram[1] = 0;
    while (p0 < position) {
        const auto want = m.cfg.block_schedule[std::min(k, m.cfg.block_schedule.size() - 1)];
        const auto rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(want, position - p0));
        const auto tb = Clock::now();
        struct Pending { std::string name; std::uint64_t first, count; std::uint8_t* data; std::uint64_t bytes; bool fixed = false; };
        std::vector<Pending> pend;
        const bool last_block = p0 + rows == position;
        if (m.cfg.release_by_layer) {
            // Per layer l of every block: the block's QSA rows of l; in the last block also GDN l's recurrent matrix +
            // conv slices (final after it) and the PLE conv after layer 1. D2H on copy_stream after layer_done[l]; an
            // emitter thread hands them over while the next layers compute.
            std::vector<std::vector<Pending>> per(kLayers);
            std::atomic<int> launched{-1};
            m.after_layer = [&](std::uint32_t l) {
                ck(cudaEventRecord(m.layer_done[l], m.stream), "layer done");
                ck(cudaStreamWaitEvent(m.copy_stream, m.layer_done[l], 0), "copy wait");
                const auto copy = [&](const std::string& name, bool fixed_piece, std::uint64_t first, std::uint64_t count,
                                      const void* src, std::uint64_t bytes) {
                    ck(cudaMemcpyAsync(out, src, bytes, cudaMemcpyDeviceToHost, m.copy_stream), "layer d2h");
                    per[l].push_back({name, first, count, out, bytes, fixed_piece});
                    out += bytes;
                };
                if (is_qsa(l)) {
                    const auto o = qsa_ordinal(l);
                    const auto layer = std::to_string(l);
                    copy("kv.L" + layer, false, p0, rows, m.kv[o] + p0 * 1024, std::uint64_t{rows} * 2048);
                    copy("index.L" + layer, false, p0, rows, m.raw_index[o] + p0 * 128, std::uint64_t{rows} * 256);
                } else if (last_block) {
                    const auto g = gdn_ordinal(l);
                    const std::uint64_t mb = std::uint64_t{gdn::kMatrixFloats} * 4, cb = std::uint64_t{gdn::kConvStateHalfs} * 2;
                    copy("recurrent_matrix", true, g * mb, 0, m.gdn_state + std::uint64_t{g} * gdn::kMatrixFloats, mb);
                    copy("gdn_convolution", true, g * cb, 0, m.gdn_conv + std::uint64_t{g} * gdn::kConvStateHalfs, cb);
                }
                if (l == 1 && last_block) copy("ple_convolution", true, 0, 0, m.ple_conv, 10240ULL * 9 * 2);
                ck(cudaEventRecord(m.copy_done[l], m.copy_stream), "copy done");
                launched.store(static_cast<int>(l), std::memory_order_release);
            };
            std::exception_ptr emit_error;
            std::thread emitter([&] {
                try {
                    for (std::uint32_t l = 0; l < kLayers; ++l) {
                        while (launched.load(std::memory_order_acquire) < static_cast<int>(l)) std::this_thread::yield();
                        if (launched.load(std::memory_order_acquire) >= static_cast<int>(kLayers)) return;  // aborted
                        ck(cudaEventSynchronize(m.copy_done[l]), "layer copy");
                        for (const auto& p : per[l]) sink(Piece{p.fixed, p.name, p.first, p.count, p.data, p.bytes});
                    }
                } catch (...) { emit_error = std::current_exception(); }
            });
            try {
                m.run_block(tokens, p0, rows);
            } catch (...) {
                launched.store(static_cast<int>(kLayers), std::memory_order_release);
                emitter.join();
                m.after_layer = nullptr;
                throw;
            }
            emitter.join();
            m.after_layer = nullptr;
            if (emit_error) std::rethrow_exception(emit_error);
        } else {
        m.run_block(tokens, p0, rows);
        // the block's rows of every row component, D2H into the staging, then to the sink
        for (std::uint32_t o = 0; o < 12; ++o) {
            const auto layer = std::to_string(4 * o + 3);
            const auto kb = std::uint64_t{rows} * 2048;
            ck(cudaMemcpyAsync(out, m.kv[o] + p0 * 1024, kb, cudaMemcpyDeviceToHost, m.stream), "kv d2h");
            pend.push_back({"kv.L" + layer, p0, rows, out, kb});
            out += kb;
        }
        for (std::uint32_t o = 0; o < 12; ++o) {
            const auto layer = std::to_string(4 * o + 3);
            const auto ib = std::uint64_t{rows} * 256;
            ck(cudaMemcpyAsync(out, m.raw_index[o] + p0 * 128, ib, cudaMemcpyDeviceToHost, m.stream), "index d2h");
            pend.push_back({"index.L" + layer, p0, rows, out, ib});
            out += ib;
        }
        }
        if (m.mtp) {
            const std::uint64_t last = std::min<std::uint64_t>(p0 + rows, position - 1);  // pairs i in [p0, last)
            if (last > p0) {
                const auto n = static_cast<std::uint32_t>(last - p0);
                for (std::uint32_t r = 0; r < n; ++r) m.next_host[r] = tokens[p0 + r + 1];
                ck(cudaMemcpyAsync(m.mtp_next, m.next_host.data(), std::uint64_t{n} * 4, cudaMemcpyHostToDevice, m.stream), "mtp next");
                pm::MtpWeights mwts{};
                mwts.token_embd = m.w.get("token_embd.weight");
                mwts.vocab = 248320;
                mwts.enorm = static_cast<const float*>(m.mw.get("blk.48.nextn.enorm.weight"));
                mwts.hnorm = static_cast<const float*>(m.mw.get("blk.48.nextn.hnorm.weight"));
                mwts.eh_q4 = m.mtp_eh_q4;
                mwts.hc_norm = static_cast<const float*>(m.mw.get("blk.48.hc_attn_norm.weight"));
                mwts.hc_down = m.mw.get("blk.48.hc_attn_down.weight");
                mwts.hc_up = m.mw.get("blk.48.hc_attn_up.weight");
                mwts.index_k = static_cast<const std::uint16_t*>(m.mw.get("blk.48.indexer.k_proj.weight"));
                mwts.k = m.mw.get("blk.48.attn_k.weight");
                mwts.v = m.mw.get("blk.48.attn_v.weight");
                mwts.k_norm = static_cast<const float*>(m.mw.get("blk.48.attn_k_norm.weight"));
                mwts.rope_cs = reinterpret_cast<const float*>(m.rope);
                mwts.rope_positions = m.cfg.max_positions;
                pm::launch_mtp_cache_rows(mwts, pm::MtpRounding{}, m.tap, m.mtp_next, n, p0, m.mtp_kv, m.mtp_index, m.mtp_ws,
                                          m.mtp_status, m.tables, m.stream);
                const auto kb = std::uint64_t{n} * 2048, ib = std::uint64_t{n} * 256;
                ck(cudaMemcpyAsync(out, m.mtp_kv + p0 * 1024, kb, cudaMemcpyDeviceToHost, m.stream), "mtp kv d2h");
                pend.push_back({"mtp.kv", p0, n, out, kb});
                out += kb;
                ck(cudaMemcpyAsync(out, m.mtp_index + p0 * 128, ib, cudaMemcpyDeviceToHost, m.stream), "mtp index d2h");
                pend.push_back({"mtp.index", p0, n, out, ib});
                out += ib;
            }
            if (p0 + rows == position) {  // held row h_{p-1}: the tap of the last row
                ck(cudaMemcpyAsync(out, m.tap + std::uint64_t{rows - 1} * kWide, kWide * 2, cudaMemcpyDeviceToHost, m.stream), "held");
                held = out;
                out += kWide * 2;
            }
        }
        ck(cudaStreamSynchronize(m.stream), "block");
        const double bs = std::chrono::duration<double>(Clock::now() - tb).count();
        timing.block_s.push_back(bs);
        if (k == 0) timing.first_block_s = bs;
        for (const auto& p : pend) sink(Piece{false, p.name, p.first, p.count, p.data, p.bytes});
        p0 += rows;
        ++k;
    }
    // fixed blob pieces (offsets inside each component)
    const auto fixed = [&](const char* name, const void* dev, std::uint64_t bytes, bool device = true) {
        if (device) ck(cudaMemcpyAsync(out, dev, bytes, cudaMemcpyDeviceToHost, m.stream), "fixed d2h");
        else std::memcpy(out, dev, bytes);
        ck(cudaStreamSynchronize(m.stream), "fixed");
        sink(Piece{true, name, 0, 0, out, bytes});
        out += bytes;
    };
    if (!m.cfg.release_by_layer) {  // else sent per layer with the last block (release_by_layer)
        fixed("recurrent_matrix", m.gdn_state, 36 * gdn::kMatrixFloats * 4);
        fixed("gdn_convolution", m.gdn_conv, 36 * gdn::kConvStateHalfs * 2);
        fixed("ple_convolution", m.ple_conv, 10240 * 9 * 2);
    }
    fixed("ngram_history", m.ngram, 8, false);
    // the session's position word holds the last processed position (node exports: 63 after 64 rows, 1030 after 1031)
    const std::uint64_t last_position = position - 1;
    fixed("position_history", &last_position, 8, false);
    if (held) sink(Piece{true, "mtp.held", 0, 0, held, kWide * 2});
    timing.rows = position;
    timing.total_s = std::chrono::duration<double>(Clock::now() - t0).count();
    return timing;
}

}  // namespace strix::cuda::qwen_prefill
