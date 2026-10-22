// Device side of ds41_trellis.hpp: the trellis library context, prepared sites, per-stream workspaces and the linear.
#include "strix/ds41_trellis.hpp"

#include "strix/ds41_moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <map>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(STRIX_DS41_TRELLIS_LIB)
#include <strix_trellis.h>
#if defined(STRIX_DS41_TRELLIS_PREPARED) && (!defined(STRIX_TRELLIS_VERSION_MINOR) || STRIX_TRELLIS_VERSION_MINOR < 3)
#error "Prepared TP adapter requires the 0.3 trellis header and runtime"
#endif
#endif

namespace strix::ds41 {
namespace {

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("ds41 trellis: ") + what + ": " + hipGetErrorString(e));
}
hipStream_t as_stream(void* s) { return static_cast<hipStream_t>(s); }
unsigned blocks_for(std::size_t n, unsigned block) { return static_cast<unsigned>((n + block - 1U) / block); }

__global__ void swiglu_f32_kernel(const float* __restrict__ gate, const float* __restrict__ up, std::size_t n, float limit,
                                  float* __restrict__ act) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < n) act[i] = swiglu_clamped(gate[i], up[i], limit);
}
#if defined(STRIX_DS41_TRELLIS_LIB)
// y[r][i] = y[r][i] + t[r][i] (t contiguous rows of n values)
__global__ void accumulate_rows_kernel(float* __restrict__ y, std::size_t y_stride, const float* __restrict__ t, std::uint32_t n,
                                       std::uint32_t rows) {
    const std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= std::size_t{n} * rows) return;
    const std::size_t r = i / n, c = i % n;
    y[r * y_stride + c] = y[r * y_stride + c] + t[i];
}
#endif

}  // namespace

void enqueue_swiglu_f32(const float* gate, const float* up, std::uint32_t rows, std::uint32_t width, float limit, float* act,
                        void* stream) {
    const std::size_t n = std::size_t{rows} * width;
    if (n == 0U) return;
    hipLaunchKernelGGL(swiglu_f32_kernel, dim3(blocks_for(n, 256)), dim3(256), 0, as_stream(stream), gate, up, n, limit, act);
    check(hipGetLastError(), "swiglu f32");
}

// Identity table (v12 library), keyed on (I, N, K): measured R classes on real ours-K3 / K4 / K5 tensors. Only K5 has
// R-dependent shapes; every K3 and K4 shape of every class (q_a, kv, q_b rank rows, o_b K halves, o_a groups, shared
// w1/w3/w2 and their 1152 TP slices, head rank slices) is one class R1..R8 (R-class run). Shapes not listed take
// every count 1..8.
std::uint32_t trellis_submit_sizes(std::uint32_t columns, std::uint32_t rows, std::uint32_t bits) noexcept {
    const auto set = [](std::initializer_list<std::uint32_t> counts) {
        std::uint32_t m = 0;
        for (const auto c : counts) m |= 1U << (c - 1U);
        return m;
    };
    if (bits == 5U && columns == 4096U && rows == 1024U) return set({1, 4, 8});  // wo_a group: split 16 at R 1, 4, 8 (32 else)
    if (bits == 5U && columns == 5120U && rows == 2304U) return set({2, 3, 4, 5, 6, 7});  // shared w1 / w3: R1 | R2..R7 | R8
    if (bits == 5U && columns == 2304U && rows == 5120U) return set({2, 3, 4, 5, 6, 7, 8});  // shared w2: R1 | R2..R8
    return 0xffU;
}
bool trellis_headout_enabled() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("STRIX_DS41_TRELLIS_HEADOUT");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}
std::uint32_t trellis_submit_sizes(const TrellisSite& site) noexcept { return trellis_submit_sizes(site.columns, site.rows, site.bits); }
std::uint32_t trellis_chunk_rows(const TrellisSite& site) noexcept {
    const auto m = trellis_submit_sizes(site);
    std::uint32_t c = 1;
    for (std::uint32_t k = 1; k <= 8U; ++k)
        if ((m >> (k - 1U)) & 1U) c = k;
    return c;
}
std::uint32_t trellis_padded_rows(const TrellisSite& site, std::uint32_t n) noexcept {
    const auto m = trellis_submit_sizes(site);
    for (std::uint32_t k = n; k <= 8U; ++k)
        if ((m >> (k - 1U)) & 1U) return k;
    return trellis_chunk_rows(site);
}

#if defined(STRIX_DS41_TRELLIS_LIB)

namespace {
void lib_check(strix_status s, const std::string& what) {
    if (s != STRIX_SUCCESS) throw std::runtime_error("ds41 trellis: " + what + ": " + strix_status_string(s));
}
strix_stream lib_stream(hipStream_t s) {
    strix_stream st{};
    st.struct_size = sizeof(st);
    st.abi_version = STRIX_TRELLIS_ABI_VERSION;
    st.hip_stream = s;
    return st;
}
void set_tensor(strix_tensor_desc& t, void* data, std::uint32_t rows, std::uint32_t width) {
    t.data = data;
    t.buffer_bytes = std::uint64_t{rows} * width * sizeof(float);
    t.dtype = STRIX_DTYPE_F32;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = width;
    t.strides[0] = width;
    t.strides[1] = 1;
}
}  // namespace

struct TrellisContext::Impl {
    strix_context context = nullptr;
    std::deque<TrellisSite> sites;
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    using PlanKey = std::tuple<std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uintptr_t, std::uint32_t>;
    struct Plan { strix_linear_plan linear = nullptr; strix_tp_dense_plan tp = nullptr; };
    std::map<PlanKey, Plan> plans;
#endif
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    using SplitKey = std::tuple<std::uintptr_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;
    struct HipFree {
        void operator()(void* p) const noexcept { if (p != nullptr) (void)hipFree(p); }
    };
    struct SplitWorkspace {
        std::unique_ptr<void, HipFree> data;
        std::uint64_t bytes = 0;
    };
    std::map<SplitKey, SplitWorkspace> split_workspaces;
    const SplitWorkspace& split_workspace(const TrellisSite& site, std::uint32_t rows, hipStream_t stream, std::uint64_t bytes) {
        const SplitKey key{reinterpret_cast<std::uintptr_t>(stream), site.columns, site.rows, site.bits, rows};
        if (const auto it = split_workspaces.find(key); it != split_workspaces.end()) {
            if (it->second.bytes != bytes) throw std::logic_error("ds41 trellis: inconsistent split-K workspace layout");
            return it->second;
        }
        void* p = nullptr;
        check(hipMalloc(&p, bytes), "split-K persistent workspace");
        SplitWorkspace owned{std::unique_ptr<void, HipFree>(p), bytes};
        return split_workspaces.emplace(key, std::move(owned)).first->second;
    }
#endif
    struct Workspace {
        hipStream_t stream = nullptr;
        void* lib = nullptr;
        std::uint64_t lib_bytes = 0;
        float* xs = nullptr;  // 8 rows x max I (strided or padded input rows)
        float* ys = nullptr;  // 8 rows x max N (strided or padded output rows, accumulate)
        std::uint64_t staging = 0;  // bytes of xs + ys
    };
    std::vector<Workspace> workspaces;
    const Workspace& workspace(hipStream_t s) const {
        for (const auto& w : workspaces)
            if (w.stream == s) return w;
        throw std::logic_error("ds41 trellis: no workspace reserved for this stream");
    }
};

TrellisContext::TrellisContext() : impl_(std::make_unique<Impl>()) {
    int device = 0;
    check(hipGetDevice(&device), "device");
    strix_context_desc d{};
    d.struct_size = sizeof(d);
    d.abi_version = STRIX_TRELLIS_ABI_VERSION;
    d.device_ordinal = device;
    lib_check(strix_context_create(&d, &impl_->context), "context create");
}

TrellisContext::~TrellisContext() {
    if (!impl_) return;
    (void)hipDeviceSynchronize();
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    for (const auto& [key, p] : impl_->plans) {
        (void)key;
        if (p.tp) (void)strix_tp_dense_plan_destroy(p.tp);
        if (p.linear) (void)strix_linear_plan_destroy(p.linear);
    }
#endif
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    impl_->split_workspaces.clear();
#endif
    for (auto& w : impl_->workspaces) {
        (void)hipFree(w.lib);
        (void)hipFree(w.xs);
        (void)hipFree(w.ys);
    }
    for (auto& s : impl_->sites) (void)strix_weights_destroy(static_cast<strix_weights>(s.weights));
    if (impl_->context) (void)strix_context_destroy(impl_->context);
}

bool TrellisContext::available() noexcept { return true; }
const char* TrellisContext::library_version() noexcept {
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    return "strix_trellis 0.3 + split-K 7b5b9b3 (C ABI 1)";
#elif defined(STRIX_DS41_TRELLIS_PREPARED)
    return "strix_trellis 0.3 ef48b33 prepared TP (C ABI 1)";
#else
    return "strix_trellis fused v12 (C ABI 1)";
#endif
}
void* TrellisContext::library_context() const noexcept { return impl_->context; }

namespace {
[[maybe_unused]] bool tp_shape(const TrellisSite& s) {
    if (s.bits != 4U && s.bits != 5U) return false;
    return (s.columns == 1280U && s.rows == 16384U) || (s.columns == 4096U && s.rows == 5120U) ||
           (s.columns == 5120U && (s.rows == 1152U || s.rows == 1280U || s.rows == 512U)) ||
           (s.columns == 1152U && s.rows == 5120U);
}
}

void TrellisContext::submit(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, void* workspace,
                            std::uint64_t bytes, void* stream) {
    strix_matmul_desc d{};
    strix_matmul_desc_init(&d);
    d.weights = static_cast<strix_weights>(site.weights);
    set_tensor(d.input, const_cast<float*>(x), rows, site.columns);
    set_tensor(d.output, y, rows, site.rows);
    d.workspace = workspace;
    d.workspace_bytes = bytes;
    const auto st = lib_stream(as_stream(stream));
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    const Impl::PlanKey key{reinterpret_cast<std::uintptr_t>(site.weights), reinterpret_cast<std::uintptr_t>(x),
        reinterpret_cast<std::uintptr_t>(y), reinterpret_cast<std::uintptr_t>(workspace),
        reinterpret_cast<std::uintptr_t>(stream), rows};
    auto it = impl_->plans.find(key);
    if (it == impl_->plans.end()) {
        hipStreamCaptureStatus capture{};
        check(hipStreamIsCapturing(as_stream(stream), &capture), "plan capture check");
        if (capture != hipStreamCaptureStatusNone)
            throw std::logic_error("ds41 trellis: warm this site/buffer/row plan before capture: " + site.key);
        Impl::Plan p;
        bool tp = tp_shape(site);
#if defined(STRIX_DS41_TRELLIS_SPLITK)
        // Frozen per-shape selection: q_b K5 R8 retains the linear plan.
        // shared_down retains the qualified 0.3 kernel (zero workspace query).
        tp = tp && !(site.columns == 1280U && site.rows == 16384U && site.bits == 5U && rows == 8U);
        if (tp) {
            std::uint64_t required = 0;
            lib_check(strix_tp_dense_workspace_bytes(impl_->context, &d, 0, &required), "split-K plan workspace " + site.key);
            if (required == 0U) {
                d.workspace = nullptr;
                d.workspace_bytes = 0;
            } else {
                // Counter storage persists between submits. Only identical layouts on one stream may share it;
                // generic GEMV and other row counts must never overwrite this allocation.
                const auto& pool = impl_->split_workspace(site, rows, as_stream(stream), required);
                d.workspace = pool.data.get();
                d.workspace_bytes = pool.bytes;
            }
            lib_check(strix_tp_dense_plan_create_splitk(impl_->context, &d, &st, 0, STRIX_TP_SPLITK_SINGLE, &p.tp),
                      "split-K plan " + site.key);
        }
#else
        if (tp) {
            d.workspace = nullptr;
            d.workspace_bytes = 0;
            lib_check(strix_tp_dense_plan_create(impl_->context, &d, &st, &p.tp), "TP plan " + site.key);
        }
#endif
        if (!tp) lib_check(strix_linear_plan_create(impl_->context, &d, &st, STRIX_OPERATION_GEMV, &p.linear),
                            "linear plan " + site.key);
        it = impl_->plans.emplace(key, p).first;
    }
    if (it->second.tp) lib_check(strix_tp_dense_plan_submit(it->second.tp), "TP submit " + site.key);
    else lib_check(strix_linear_plan_submit(it->second.linear), "linear submit " + site.key);
#else
    lib_check(strix_gemv(impl_->context, &d, &st), "gemv " + site.key);
#endif
}

void TrellisContext::release_plans(void* workspace) noexcept {
#if defined(STRIX_DS41_TRELLIS_PREPARED)
    for (auto it = impl_->plans.begin(); it != impl_->plans.end();) {
        if (std::get<3>(it->first) == reinterpret_cast<std::uintptr_t>(workspace)) {
            if (it->second.tp) (void)strix_tp_dense_plan_destroy(it->second.tp);
            if (it->second.linear) (void)strix_linear_plan_destroy(it->second.linear);
            it = impl_->plans.erase(it);
        } else ++it;
    }
#else
    (void)workspace;
#endif
}

const TrellisSite& TrellisContext::prepare(TrellisPack& pack, const std::string& pack_name, const TrellisSlice& slice) {
    if (const auto* s = find(pack_name, slice)) return *s;
    const auto host = pack.load(pack_name, slice);
    strix_exl3_metadata_v1 m{};
    m.struct_size = sizeof(m);
    m.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    m.rows = host.rows;
    m.columns = host.columns;
    m.bits_x2 = 2U * host.bits;
    m.codebook = STRIX_CODEBOOK_MUL1;
    m.marker = host.marker;
    m.trellis_offset = host.trellis_offset;
    m.trellis_bytes = host.trellis_bytes;
    m.suh_offset = host.suh_offset;
    m.suh_bytes = host.suh_bytes;
    m.svh_offset = host.svh_offset;
    m.svh_bytes = host.svh_bytes;
    strix_exl3_transport_desc t{};
    t.struct_size = sizeof(t);
    t.abi_version = STRIX_TRELLIS_ABI_VERSION;
    t.format = STRIX_TRANSPORT_EXL3_CANONICAL;
    t.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
    t.data = host.blob.data();
    t.data_bytes = host.blob.size();
    t.metadata = &m;
    t.metadata_bytes = sizeof(m);
    t.rows = host.rows;
    t.columns = host.columns;
    const auto st = lib_stream(nullptr);
    strix_weights w = nullptr;
    lib_check(strix_weights_prepare(impl_->context, &t, &st, &w), "prepare " + pack_name);
    check(hipDeviceSynchronize(), "prepare sync");
    TrellisSite s;
    s.weights = w;
    s.rows = host.rows;
    s.columns = host.columns;
    s.bits = host.bits;
    s.device_bytes = host.blob.size();
    s.key = pack_name + "@" + slice.describe();
    s.context = this;
    impl_->sites.push_back(std::move(s));
    return impl_->sites.back();
}

const TrellisSite* TrellisContext::find(const std::string& pack_name, const TrellisSlice& slice) const {
    const auto key = pack_name + "@" + slice.describe();
    for (const auto& s : impl_->sites)
        if (s.key == key) return &s;
    return nullptr;
}

void TrellisContext::reserve(void* stream, std::uint32_t max_rows) {
    const auto hs = as_stream(stream);
    for (const auto& w : impl_->workspaces)
        if (w.stream == hs) throw std::logic_error("ds41 trellis: stream reserved twice");
    (void)max_rows;  // every chunk has at most 8 rows (padding may exceed a smaller max_rows)
    Impl::Workspace w;
    w.stream = hs;
    std::uint64_t max_i = 128, max_n = 128;
    for (const auto& s : impl_->sites) {
        max_i = std::max<std::uint64_t>(max_i, s.columns);
        max_n = std::max<std::uint64_t>(max_n, s.rows);
        for (std::uint32_t r = 1; r <= 8U; ++r) {
            strix_matmul_desc d{};
            strix_matmul_desc_init(&d);
            d.weights = static_cast<strix_weights>(s.weights);
            // query only: host descriptors are validated, device memory is not touched (addresses must not overlap)
            set_tensor(d.input, reinterpret_cast<void*>(std::uintptr_t{1} << 40U), r, s.columns);
            set_tensor(d.output, reinterpret_cast<void*>(std::uintptr_t{1} << 41U), r, s.rows);
            std::uint64_t bytes = 0;
            lib_check(strix_matmul_workspace_bytes(impl_->context, &d, &bytes), "workspace query " + s.key);
            w.lib_bytes = std::max(w.lib_bytes, bytes);
#if defined(STRIX_DS41_TRELLIS_SPLITK)
            if (tp_shape(s) && !(s.columns == 1280U && s.rows == 16384U && s.bits == 5U && r == 8U)) {
                lib_check(strix_tp_dense_workspace_bytes(impl_->context, &d, 0, &bytes), "split-K workspace " + s.key);
                if (bytes != 0U) (void)impl_->split_workspace(s, r, hs, bytes);
            }
#endif
        }
    }
    constexpr std::uint32_t chunk_max = 8;
    check(hipMalloc(&w.lib, std::max<std::uint64_t>(w.lib_bytes, 256U)), "workspace");
    check(hipMalloc(reinterpret_cast<void**>(&w.xs), std::size_t{chunk_max} * max_i * sizeof(float)), "x staging");
    check(hipMalloc(reinterpret_cast<void**>(&w.ys), std::size_t{chunk_max} * max_n * sizeof(float)), "y staging");
    w.staging = staging_bytes_for(static_cast<std::uint32_t>(max_i), static_cast<std::uint32_t>(max_n));
    impl_->workspaces.push_back(w);
}

std::uint64_t TrellisContext::staging_bytes() const noexcept {
    std::uint64_t b = 0;
    for (const auto& w : impl_->workspaces) b += w.staging;
    return b;
}
std::uint64_t TrellisContext::library_workspace_bytes() const noexcept {
    std::uint64_t b = 0;
    for (const auto& w : impl_->workspaces) b += std::max<std::uint64_t>(w.lib_bytes, 256U);
#if defined(STRIX_DS41_TRELLIS_SPLITK)
    for (const auto& [key, w] : impl_->split_workspaces) {
        (void)key;
        b += w.bytes;
    }
#endif
    return b;
}

std::uint64_t TrellisContext::device_bytes() const noexcept {
    std::uint64_t b = 0;
    for (const auto& s : impl_->sites) b += s.device_bytes;
    return b;
}
std::size_t TrellisContext::sites() const noexcept { return impl_->sites.size(); }
std::string TrellisContext::announced_identity(const std::vector<std::string>& site_classes, bool shexp_tp) const {
    std::map<std::string, std::uint32_t> bits;  // mix class -> K (one pack per class)
    for (const auto& site : impl_->sites) {
        const auto cls = trellis_mix_class(site.key.substr(0, site.key.find('@')));
        if (const auto it = bits.find(cls); it != bits.end() && it->second != site.bits)
            throw std::logic_error("ds41 trellis: mix class " + cls + " prepared with K" + std::to_string(it->second) + " and K" +
                                   std::to_string(site.bits));
        bits[cls] = site.bits;
    }
    if (bits.empty()) return {};
    // only the classes that were prepared (a selected class without any site in this process announces nothing)
    std::vector<std::string> prepared;
    for (const auto& c : site_classes) {
        const bool any = c == "shexp" ? bits.contains("shexp_w13") || bits.contains("shexp_w2") : bits.contains(c);
        if (any) prepared.push_back(c);
    }
    return trellis_config_identity(bits, prepared, shexp_tp);
}

void enqueue_trellis_linear(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, std::size_t x_stride,
                            std::size_t y_stride, bool accumulate, void* stream) {
    if (rows == 0U) return;
    if (site.context == nullptr || x_stride < site.columns || y_stride < site.rows)
        throw std::invalid_argument("ds41 trellis: bad linear call for " + site.key);
    auto& impl = site.context->impl();
    const auto hs = as_stream(stream);
    const auto& w = impl.workspace(hs);
    const std::uint32_t chunk = trellis_chunk_rows(site);
    const std::size_t xb = std::size_t{site.columns} * 4U;
    const bool headout = trellis_headout_enabled();
    for (std::uint32_t begin = 0; begin < rows; begin += chunk) {
        const std::uint32_t n = std::min(chunk, rows - begin);
        const std::uint32_t m = trellis_padded_rows(site, n);  // library rows of this chunk (n real + m - n copies of the last)
        const float* xin = x + std::size_t{begin} * x_stride;
        if ((x_stride != site.columns && !(headout && n == 1U)) || m != n) {
            check(hipMemcpy2DAsync(w.xs, xb, xin, x_stride * 4U, xb, n, hipMemcpyDeviceToDevice, hs), "x staging copy");
            for (std::uint32_t k = n; k < m; ++k)
                check(hipMemcpyAsync(w.xs + std::size_t{k} * site.columns, xin + std::size_t{n - 1U} * x_stride, xb, hipMemcpyDeviceToDevice, hs),
                      "x padding copy");
            xin = w.xs;
        }
        float* yrow = y + std::size_t{begin} * y_stride;
        const bool direct = !accumulate && (y_stride == site.rows || (headout && n == 1U)) && m == n;
        float* yout = direct ? yrow : w.ys;
        site.context->submit(site, xin, yout, m, w.lib, std::max<std::uint64_t>(w.lib_bytes, 256U), hs);
        if (direct) continue;
        if (accumulate) {
            hipLaunchKernelGGL(accumulate_rows_kernel, dim3(blocks_for(std::size_t{n} * site.rows, 256)), dim3(256), 0, hs, yrow,
                               y_stride, w.ys, site.rows, n);
            check(hipGetLastError(), "accumulate");
        } else {
            check(hipMemcpy2DAsync(yrow, y_stride * 4U, w.ys, std::size_t{site.rows} * 4U, std::size_t{site.rows} * 4U, n,
                                   hipMemcpyDeviceToDevice, hs),
                  "y staging copy");
        }
    }
}

void enqueue_trellis_linear_rows(const TrellisSite& site, const float* x, float* y, std::uint32_t rows, std::size_t x_stride,
                                 std::size_t y_stride, std::uint32_t capacity, std::uint32_t pad_from, void* stream) {
    if (rows == 0U) return;
    if (site.context == nullptr || x_stride < site.columns || y_stride < site.rows || capacity < rows)
        throw std::invalid_argument("ds41 trellis: bad rows call for " + site.key);
    auto& impl = site.context->impl();
    const auto hs = as_stream(stream);
    const auto& w = impl.workspace(hs);
    const std::uint32_t chunk = trellis_chunk_rows(site);
    const std::size_t xb = std::size_t{site.columns} * 4U;
    for (std::uint32_t begin = 0; begin < rows; begin += chunk) {
        const std::uint32_t n = std::min(chunk, rows - begin);
        std::uint32_t m = trellis_padded_rows(site, n);
        if (pad_from != 0U && n >= pad_from && begin + chunk <= capacity) m = std::max(m, chunk);
        const bool fits = begin + m <= capacity;  // rows [n, m) of x / y are the caller's scratch rows
        const float* xin = x + std::size_t{begin} * x_stride;
        if (!(fits && (m == 1U || x_stride == site.columns))) {  // staged as enqueue_trellis_linear
            check(hipMemcpy2DAsync(w.xs, xb, xin, x_stride * 4U, xb, n, hipMemcpyDeviceToDevice, hs), "x staging copy");
            for (std::uint32_t k = n; k < m; ++k)
                check(hipMemcpyAsync(w.xs + std::size_t{k} * site.columns, xin + std::size_t{n - 1U} * x_stride, xb, hipMemcpyDeviceToDevice, hs),
                      "x padding copy");
            xin = w.xs;
        }
        float* yrow = y + std::size_t{begin} * y_stride;
        const bool direct = fits && (m == 1U || y_stride == site.rows);
        site.context->submit(site, xin, direct ? yrow : w.ys, m, w.lib, std::max<std::uint64_t>(w.lib_bytes, 256U), hs);
        if (!direct)
            check(hipMemcpy2DAsync(yrow, y_stride * 4U, w.ys, std::size_t{site.rows} * 4U, std::size_t{site.rows} * 4U, n,
                                   hipMemcpyDeviceToDevice, hs),
                  "y staging copy");
    }
}

#else  // built without the trellis library: the host side works, the device side refuses

struct TrellisContext::Impl {};
TrellisContext::TrellisContext() {
    throw std::runtime_error("ds41 trellis: this build has no trellis library (configure with STRIX_DS41_TRELLIS=ON)");
}
TrellisContext::~TrellisContext() = default;
bool TrellisContext::available() noexcept { return false; }
const char* TrellisContext::library_version() noexcept { return "none"; }
void* TrellisContext::library_context() const noexcept { return nullptr; }
void TrellisContext::submit(const TrellisSite&, const float*, float*, std::uint32_t, void*, std::uint64_t, void*) {
    throw std::logic_error("ds41 trellis: no library");
}
void TrellisContext::release_plans(void*) noexcept {}
const TrellisSite& TrellisContext::prepare(TrellisPack&, const std::string&, const TrellisSlice&) {
    throw std::logic_error("ds41 trellis: no library");
}
const TrellisSite* TrellisContext::find(const std::string&, const TrellisSlice&) const { return nullptr; }
void TrellisContext::reserve(void*, std::uint32_t) { throw std::logic_error("ds41 trellis: no library"); }
std::uint64_t TrellisContext::device_bytes() const noexcept { return 0; }
std::uint64_t TrellisContext::staging_bytes() const noexcept { return 0; }
std::uint64_t TrellisContext::library_workspace_bytes() const noexcept { return 0; }
std::size_t TrellisContext::sites() const noexcept { return 0; }
std::string TrellisContext::announced_identity(const std::vector<std::string>&, bool) const { return {}; }
void enqueue_trellis_linear(const TrellisSite&, const float*, float*, std::uint32_t, std::size_t, std::size_t, bool, void*) {
    throw std::logic_error("ds41 trellis: no library");
}
void enqueue_trellis_linear_rows(const TrellisSite&, const float*, float*, std::uint32_t, std::size_t, std::size_t, std::uint32_t,
                                 std::uint32_t, void*) {
    throw std::logic_error("ds41 trellis: no library");
}

#endif

std::uint64_t TrellisContext::staging_bytes_for(std::uint32_t max_columns, std::uint32_t max_rows) noexcept {
    // reserve(): x / y staging of 8 rows (every chunk has at most 8), widths at least 128
    return 8U * (std::uint64_t{std::max(max_columns, 128U)} + std::max(max_rows, 128U)) * sizeof(float);
}

}  // namespace strix::ds41
