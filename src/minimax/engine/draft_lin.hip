// MiniMax DSpark drafter EXL3 linears (see draft_lin.hpp).
#include "draft_lin.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace strix::minimax::engine {

namespace {

namespace mmoe = strix::minimax::moe::opt;

void dl_hip(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("draft linears ") + what + ": " + hipGetErrorString(e));
}
void dl_lib(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("draft linears trellis " + what + ": " + strix_status_string(s));
}
strix_stream lib_stream(hipStream_t s) {
    strix_stream st{};
    st.struct_size = sizeof(st);
    st.abi_version = STRIX_TRELLIS_ABI_VERSION;
    st.hip_stream = s;
    return st;
}
void set_tensor(strix_tensor_desc& t, float* p, std::uint64_t rows, std::uint64_t width) {
    t.data = p;
    t.buffer_bytes = rows * width * sizeof(float);
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = width;
    t.strides[0] = width;
    t.strides[1] = 1;
}
const char* const kNames[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj"};
constexpr unsigned kSlots = DraftLinears::kBlock + 1;
// Checkpoint prefix of storage slot s (layer * 7 + kind, or kBlock = fc).
std::string prefix(unsigned s) {
    if (s == DraftLinears::kBlock) return "language_model.model.dspark.fc";
    return "language_model.model.dspark.layers." + std::to_string(s / DraftLinears::kKinds) + ".decoder_layer." +
           kNames[s % DraftLinears::kKinds];
}
constexpr std::uint32_t kK = 1, kV = 2;  // kinds the context appends also run at 32 rows (strix_gemm)

}  // namespace

DraftLinears::Mode DraftLinears::mode_from_env() {
    const char* e = std::getenv("STRIX_MM_DRAFT_LIN");
    if (e == nullptr || std::string(e) == "opt") return Mode::opt;
    if (std::string(e) == "lib") return Mode::lib;
    if (std::string(e) == "load") return Mode::load;
    throw std::invalid_argument("STRIX_MM_DRAFT_LIN must be opt, lib or load");
}

unsigned DraftLinears::slot(unsigned layer, unsigned kind) const {
    if (kind == kKindFc) {
        if (!has_fc_) throw std::invalid_argument("draft linears: the pack has no fc");
        return kFcSlot;
    }
    if (kind >= kKinds || layer >= kLayers) throw std::invalid_argument("draft linears: layer / kind out of range");
    return layer * kKinds + kind;
}

DraftLinears::DraftLinears(const std::string& dir, Mode mode) : pack_(dir), mode_(mode), opt_on_(mode != Mode::lib) {
    std::uint8_t* stage = nullptr;
    try {
        strix_context_desc cd{};
        cd.struct_size = sizeof(cd);
        cd.abi_version = STRIX_TRELLIS_ABI_VERSION;
        cd.device_ordinal = 0;
        dl_lib(strix_context_create(&cd, &ctx_), "context");
        try {
            (void)pack_.info(prefix(kFcSlot) + ".trellis");
            has_fc_ = true;
        } catch (const std::exception&) {
        }
        lib_.assign(kSlots, nullptr);
        dims_.assign(kSlots, Dim{});
        std::uint64_t stage_bytes = 0;
        for (unsigned s = 0; s < kSlots; ++s) {
            if (s == kFcSlot && !has_fc_) continue;
            const std::string p = prefix(s);
            stage_bytes = std::max(stage_bytes, pack_.info(p + ".trellis").bytes + pack_.info(p + ".suh").bytes + pack_.info(p + ".svh").bytes);
        }
        dl_hip(hipHostMalloc(reinterpret_cast<void**>(&stage), stage_bytes, hipHostMallocDefault), "staging");
        for (unsigned s = 0; s < kSlots; ++s) {
            if (s == kFcSlot && !has_fc_) continue;
            const std::string p = prefix(s);
            const detail::TensorInfo &tr = pack_.info(p + ".trellis"), &suh = pack_.info(p + ".suh"), &svh = pack_.info(p + ".svh");
            Dim& d = dims_[s];
            d.rows = static_cast<unsigned>(svh.shape[0]);
            d.cols = static_cast<unsigned>(suh.shape[0]);
            d.bits = static_cast<unsigned>(tr.shape[2] / 16);
            pack_.read(tr, 0, tr.bytes, stage);
            pack_.read(suh, 0, suh.bytes, stage + tr.bytes);
            pack_.read(svh, 0, svh.bytes, stage + tr.bytes + suh.bytes);
            const std::uint8_t *h_tr = stage, *h_suh = stage + tr.bytes, *h_svh = stage + tr.bytes + suh.bytes;
            const bool fc = s == kFcSlot;
            const unsigned kind = fc ? kKindFc : s % kKinds;
            // The library copy: fc always; k / v always (32-row appends); the rest only when lib is selectable.
            if (fc || kind == kK || kind == kV || mode_ != Mode::opt) {
                strix_exl3_parts_desc pd{};
                pd.struct_size = sizeof(pd);
                pd.abi_version = STRIX_TRELLIS_ABI_VERSION;
                pd.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
                pd.bits_x2 = 2 * d.bits;
                pd.rows = d.rows;
                pd.columns = d.cols;
                pd.codebook = STRIX_CODEBOOK_MUL1;
                pd.marker = STRIX_EXL3_MARKER_MUL1;
                pd.trellis = {h_tr, tr.bytes};
                pd.suh = {h_suh, suh.bytes};
                pd.svh = {h_svh, svh.bytes};
                const strix_stream st = lib_stream(nullptr);
                dl_lib(strix_weights_prepare_parts(ctx_, &pd, &st, &lib_[s]), "prepare " + p);
                device_bytes_ += tr.bytes + suh.bytes + svh.bytes;
            }
            if (!fc && mode_ != Mode::lib) {
                const std::uint64_t a = (tr.bytes + 255) & ~std::uint64_t{255}, b = (suh.bytes + 255) & ~std::uint64_t{255};
                void* dev = nullptr;
                dl_hip(hipMalloc(&dev, a + b + svh.bytes), "raw");
                allocations_.push_back(dev);
                auto* bytes = static_cast<std::uint8_t*>(dev);
                dl_hip(hipMemcpy(bytes, h_tr, tr.bytes, hipMemcpyHostToDevice), "raw trellis");
                dl_hip(hipMemcpy(bytes + a, h_suh, suh.bytes, hipMemcpyHostToDevice), "raw suh");
                dl_hip(hipMemcpy(bytes + a + b, h_svh, svh.bytes, hipMemcpyHostToDevice), "raw svh");
                mmoe::Exl3 e;
                e.trellis = reinterpret_cast<const std::uint32_t*>(bytes);
                e.suh = reinterpret_cast<const std::uint16_t*>(bytes + a);
                e.svh = reinterpret_cast<const std::uint16_t*>(bytes + a + b);
                e.input = d.cols;
                e.output = d.rows;
                e.bits = d.bits;
                raw_.push_back(e);  // slot order: raw_[s] for s < kBlock
                device_bytes_ += a + b + svh.bytes;
            }
        }
        dl_hip(hipDeviceSynchronize(), "load");
        dl_hip(hipHostFree(stage), "staging");
        stage = nullptr;
        std::uint64_t max_in = 0, max_out = 0;
        for (const Dim& d : dims_) {
            max_in = std::max<std::uint64_t>(max_in, d.cols);
            max_out = std::max<std::uint64_t>(max_out, d.rows);
        }
        dl_hip(hipMalloc(&sx_, 32 * max_in * sizeof(float)), "scratch");
        allocations_.push_back(sx_);
        dl_hip(hipMalloc(&sy_, 32 * max_out * sizeof(float)), "scratch");
        allocations_.push_back(sy_);
        dl_hip(hipMemset(sx_, 0, 32 * max_in * sizeof(float)), "scratch");
        // Workspaces sized at load for every call the drafter makes (no growth while drafting).
        for (unsigned s = 0; s < kSlots; ++s) {
            if (!lib_has(s)) continue;
            for (const std::uint64_t t : {std::uint64_t{1}, std::uint64_t{7}, std::uint64_t{8}, std::uint64_t{32}}) {
                strix_matmul_desc md{};
                strix_matmul_desc_init(&md);
                md.weights = lib_[s];
                set_tensor(md.input, sx_, t, dims_[s].cols);
                set_tensor(md.output, sy_, t, dims_[s].rows);
                std::uint64_t bytes = 0;
                dl_lib(strix_matmul_workspace_bytes(ctx_, &md, &bytes), "workspace");
                lib_ws_bytes_ = std::max(lib_ws_bytes_, bytes);
            }
        }
        if (lib_ws_bytes_ != 0) {
            dl_hip(hipMalloc(&lib_ws_, lib_ws_bytes_), "lib workspace");
            allocations_.push_back(lib_ws_);
        }
        if (!raw_.empty()) {
            std::size_t ws = 0;
            for (unsigned L = 0; L < kLayers; ++L) {
                const auto& r = raw_;
                const unsigned b = L * kKinds;
                const mmoe::LinearJob qkv[3] = {{r[b], nullptr, 0, nullptr, 0}, {r[b + 1], nullptr, 0, nullptr, 0}, {r[b + 2], nullptr, 0, nullptr, 0}};
                const mmoe::LinearJob gu[2] = {{r[b + 4], nullptr, 0, nullptr, 0}, {r[b + 5], nullptr, 0, nullptr, 0}};
                ws = std::max({ws, mmoe::linear_workspace_bytes(qkv, 3), mmoe::linear_workspace_bytes(qkv + 1, 2),
                               mmoe::linear_workspace_bytes(gu, 2)});
                for (unsigned k = 0; k < kKinds; ++k) {
                    const mmoe::LinearJob one{r[b + k], nullptr, 0, nullptr, 0};
                    ws = std::max(ws, mmoe::linear_workspace_bytes(&one, 1));
                }
            }
            dl_hip(hipMalloc(&opt_ws_, ws), "opt workspace");
            allocations_.push_back(opt_ws_);
            dl_hip(hipMemset(opt_ws_, 0, ws), "opt workspace");  // counters start (and stay) at 0
        }
    } catch (...) {
        if (stage != nullptr) (void)hipHostFree(stage);
        release();
        throw;
    }
}

void DraftLinears::release() noexcept {
    (void)hipDeviceSynchronize();
    for (auto& w : lib_)
        if (w != nullptr) {
            (void)strix_weights_destroy(w);
            w = nullptr;
        }
    for (void* p : allocations_) (void)hipFree(p);
    allocations_.clear();
    if (ctx_ != nullptr) (void)strix_context_destroy(ctx_);
    ctx_ = nullptr;
}

DraftLinears::~DraftLinears() { release(); }

void DraftLinears::set_opt(bool on) {
    if (on != opt_on_ && mode_ != Mode::load) throw std::logic_error("DraftLinears::set_opt: load with STRIX_MM_DRAFT_LIN=load");
    opt_on_ = on;
}

void DraftLinears::lib_mm(unsigned s, const float* x, float* out, std::size_t rows, hipStream_t stream) {
    if (!lib_has(s)) throw std::logic_error("draft linears: " + prefix(s) + " has no library copy");
    strix_matmul_desc md{};
    strix_matmul_desc_init(&md);
    md.weights = lib_[s];
    set_tensor(md.input, const_cast<float*>(x), rows, dims_[s].cols);
    set_tensor(md.output, out, rows, dims_[s].rows);
    std::uint64_t bytes = 0;
    dl_lib(strix_matmul_workspace_bytes(ctx_, &md, &bytes), "workspace");
    if (bytes > lib_ws_bytes_) throw std::logic_error("draft linears: workspace not sized at load");
    md.workspace = bytes ? lib_ws_ : nullptr;
    md.workspace_bytes = bytes ? lib_ws_bytes_ : 0;
    const strix_stream st = lib_stream(stream);
    const bool gemv = rows <= mmoe::kMaxRows;  // the engine's rule (rows <= kMaxStepRows -> strix_gemv)
    dl_lib(gemv ? strix_gemv(ctx_, &md, &st) : strix_gemm(ctx_, &md, &st), gemv ? "gemv" : "gemm");
}

void DraftLinears::run(unsigned layer, unsigned kind, const float* x, std::size_t rows, float* out, hipStream_t stream) {
    if (rows == 0) return;
    if (kind == kKindFc) {
        lib_mm(slot(0, kind), x, out, rows, stream);
        return;
    }
    const std::uint32_t k = kind;
    float* const o[1] = {out};
    run_group(layer, &k, 1, x, rows, o, stream);
}

void DraftLinears::run_group(unsigned layer, const std::uint32_t* kinds, unsigned count, const float* x, std::size_t rows,
                             float* const* outs, hipStream_t stream) {
    if (rows == 0) return;
    if (count == 0 || count > mmoe::kMaxJobs) throw std::invalid_argument("draft linears: 1..4 linears per group");
    for (unsigned i = 0; i < count; ++i)
        if (kinds[i] >= kKinds) throw std::invalid_argument("draft linears: groups take block kinds 0..6");
    if (!opt_on_ || rows > mmoe::kMaxRows) {
        for (unsigned i = 0; i < count; ++i) lib_mm(slot(layer, kinds[i]), x, outs[i], rows, stream);
        return;
    }
    mmoe::LinearJob j[mmoe::kMaxJobs];
    for (unsigned i = 0; i < count; ++i) {
        const unsigned s = slot(layer, kinds[i]);
        j[i] = {raw_[s], x, dims_[s].cols, outs[i], dims_[s].rows};
        if (i > 0 && dims_[s].cols != j[0].x_stride) throw std::invalid_argument("draft linears: a group shares one input width");
    }
    mmoe::enqueue_linears_scheduled(j, count, static_cast<unsigned>(rows), opt_ws_, stream);
}

void DraftLinears::warm_up() {
    for (unsigned s = 0; s < kBlock; ++s) {
        if (lib_has(s)) lib_mm(s, sx_, sy_, 7, nullptr);
        if (s % kKinds == kK || s % kKinds == kV) lib_mm(s, sx_, sy_, 32, nullptr);
    }
    if (has_fc_) {
        lib_mm(kFcSlot, sx_, sy_, 32, nullptr);
        lib_mm(kFcSlot, sx_, sy_, 8, nullptr);  // decode appends (accepted + 1 rows)
    }
    if (!raw_.empty())
        for (unsigned s = 0; s < kBlock; ++s) {
            const mmoe::LinearJob one{raw_[s], sx_, dims_[s].cols, sy_, dims_[s].rows};
            mmoe::enqueue_linears_scheduled(&one, 1, 7, opt_ws_, nullptr);
        }
    dl_hip(hipDeviceSynchronize(), "warm-up");
}

}  // namespace strix::minimax::engine
