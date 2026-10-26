// The GLM exchange module behind the engine's Exchange seam (see glm_xfer_exchange.hpp).
#include "glm_xfer_exchange.hpp"
#include "glm_post_exchange.hpp"
#include "glm_xfer_device.hpp"

#include "glm/common/glm_math.hpp"

#include <cstdio>
#include <cstdlib>

#include <stdexcept>
#include <string>

namespace strix::glm::xfer {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glm xfer exchange: ") + what + ": " + hipGetErrorString(e));
}

// R17 in place: p = p0 + p1 with p0 = rank 0's partial (the engine's sum2_kernel order).
__global__ void reduce_kernel(float* p, const float* peer, unsigned rank, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    const float a = p[i], b = peer[i];
    p[i] = rank == 0 ? a + b : b + a;
}

// (value bits | id << 32) per row into the TX slot; the single block publishes it.
__global__ void pack_kernel(const float* value, const std::uint32_t* id, std::uint64_t* tx, unsigned rows, ProducerTarget t) {
    for (unsigned r = threadIdx.x; r < rows; r += blockDim.x)
        tx[r] = (static_cast<std::uint64_t>(id[r]) << 32U) | __builtin_bit_cast(std::uint32_t, value[r]);
    glmx_publish(t, gridDim.x);
}

// R18 merge on both ranks, the engine's rule verbatim: take the peer's when pv > v || (pv == v && pi < i).
__global__ void merge_kernel(float* value, std::uint32_t* id, const std::uint64_t* peer, unsigned rows) {
    for (unsigned r = threadIdx.x; r < rows; r += blockDim.x) {
        const float pv = __builtin_bit_cast(float, static_cast<std::uint32_t>(peer[r]));
        const auto pi = static_cast<std::uint32_t>(peer[r] >> 32U);
        if (pv > value[r] || (pv == value[r] && pi < id[r])) {
            value[r] = pv;
            id[r] = pi;
        }
    }
}

}  // namespace

namespace {
void boundary_form(bool& kwait, bool& bracket) {
    const char* v = std::getenv("STRIX_GLMX_BOUNDARY");
    // Default kw: measured token level (N = 1 / 8) kw 16.35 / 22.36 ms vs kw-br 16.50 / 22.41 ms; the separate bracket kernel
    // costs +0.20 / +0.49 ms/token locally (slower rank; 90 extra launches) and its overlap inside the wire window does
    // not pay it back (design notes).
    const std::string f = v == nullptr || *v == '\0' ? "kw" : v;
    if (f != "kw-br" && f != "kw" && f != "br" && f != "plain") throw std::invalid_argument("STRIX_GLMX_BOUNDARY must be kw-br, kw, br or plain");
    kwait = f == "kw-br" || f == "kw";
    bracket = f == "kw-br" || f == "br";
}
}  // namespace

XferExchange::XferExchange(dist::Transport& transport, std::uint32_t rank, unsigned max_rows)
    : transport_(&transport), rank_(rank), max_rows_(max_rows) {
    boundary_form(kwait_, use_bracket_);
}
XferExchange::XferExchange(std::unique_ptr<dist::Transport> transport, std::uint32_t rank, unsigned max_rows)
    : owned_(std::move(transport)), transport_(owned_.get()), rank_(rank), max_rows_(max_rows) {
    boundary_form(kwait_, use_bracket_);
}
XferExchange::~XferExchange() {
    // Not a completion point (destructors cannot report): callers trust results only after complete() / argmax().
    // A failure that was never checked is still printed so it cannot vanish silently.
    if (xf_ != nullptr) {
        try {
            xf_->drain();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "glm xfer exchange: unchecked exchange failure at destruction: %s\n", e.what());
        }
    }
    if (bracket_ != nullptr) (void)hipFree(bracket_);
}

void XferExchange::complete() {
    if (xf_ != nullptr) xf_->drain();
}

Exchanger& XferExchange::ensure(hipStream_t s) {
    if (xf_ == nullptr) {
        Options o;
        o.rank = rank_;
        o.capacity = GlmSchedule::row_bytes(max_rows_);
        o.stream = s;
        o.from_env();
        xf_ = std::make_unique<Exchanger>(*transport_, o);
    } else if (xf_->options().stream != s) {
        throw std::logic_error("glm xfer exchange: every call must use the stream of the first one");
    }
    return *xf_;
}

void XferExchange::reduce(float* p, unsigned rows, hipStream_t s) {
    if (rows == 0 || rows > max_rows_) throw std::invalid_argument("glm xfer exchange: rows");
    Exchanger& x = ensure(s);
    const std::size_t bytes = GlmSchedule::row_bytes(rows);
    x.send_copy(p, bytes, Tag{layer_ % 2 == 0 ? Kind::attn_sum : Kind::ffn_sum, static_cast<std::uint32_t>(layer_ / 2), step_});
    ++layer_;
    const auto* peer = static_cast<const float*>(x.recv());
    const std::size_t n = bytes / sizeof(float);
    reduce_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(p, peer, rank_, n);
    ok(hipGetLastError(), "reduce");
}

void XferExchange::boundary(const float* part, const float* X, const float* mix, float* Xout, unsigned* flag, unsigned rows,
                            hipStream_t s) {
    if (rows == 0 || rows > max_rows_) throw std::invalid_argument("glm xfer exchange: rows");
    Exchanger& x = ensure(s);
    const std::size_t bytes = GlmSchedule::row_bytes(rows);
    x.send_copy(part, bytes, Tag{layer_ % 2 == 0 ? Kind::attn_sum : Kind::ffn_sum, static_cast<std::uint32_t>(layer_ / 2), step_});
    ++layer_;
    if (use_bracket_) {
        if (bracket_ == nullptr) ok(hipMalloc(reinterpret_cast<void**>(&bracket_), GlmSchedule::row_bytes(max_rows_) * kHc), "bracket");
        hc_bracket(X, mix, bracket_, rows, s);  // inside the wire window
    }
    const RxView none{};
    if (kwait_) {
        const RxView v = x.recv_view();
        if (use_bracket_) post_expand_bracket(bracket_, mix, part, nullptr, v, rank_, Xout, flag, rows, s);
        else post_expand(X, mix, part, nullptr, v, rank_, Xout, flag, rows, s);
    } else {
        const auto* peer = static_cast<const float*>(x.recv());
        if (use_bracket_) post_expand_bracket(bracket_, mix, part, peer, none, rank_, Xout, flag, rows, s);
        else post_expand(X, mix, part, peer, none, rank_, Xout, flag, rows, s);
    }
}

void XferExchange::argmax(float* value, std::uint32_t* id, unsigned rows, hipStream_t s) {
    if (rows == 0 || rows > max_rows_) throw std::invalid_argument("glm xfer exchange: rows");
    Exchanger& x = ensure(s);
    auto* tx = static_cast<std::uint64_t*>(x.tx());
    pack_kernel<<<1, 64, 0, s>>>(value, id, tx, rows, x.producer_target());
    ok(hipGetLastError(), "argmax pack");
    x.send(std::size_t{rows} * 8, Tag{Kind::argmax, GlmSchedule::kLayers, step_}, Publish::producer);
    const auto* peer = static_cast<const std::uint64_t*>(x.recv());
    merge_kernel<<<1, 64, 0, s>>>(value, id, peer, rows);
    ok(hipGetLastError(), "argmax merge");
    x.drain();  // the step's CPU sync point: every exchange of the step checked (identity, size, CRC) before use
    ++step_;
    layer_ = 0;
}

}  // namespace strix::glm::xfer
