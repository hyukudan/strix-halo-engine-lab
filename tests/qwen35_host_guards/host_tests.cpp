#include "strix/qwen35/cuda_engine.hpp"
#include "strix/cuda_core/nvfp4.hpp"
#include "strix/profiles/qwen_dense_cuda.hpp"
#include "../../src/core/dense_linear.hpp"
#include "../../src/cuda_core/dense/dense_mma.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cmath>
namespace cc = strix::cuda_core;
namespace core = strix::core;
namespace strix::cuda_core {
DenseArith policy = DenseArith::mma3;
DenseArith dense_arith() { return policy; }
void set_dense_arith(DenseArith p) { policy = p; }
constexpr int kMaxMulti = 4;
struct MultiArgs { int n; const void* w[4]; float* y[4]; std::uint32_t out[4], row_end[4]; };
std::vector<std::pair<std::uint32_t, bool>> routes;
bool valid(const DenseWeight& w, std::uint32_t rows) { return w.data && w.in && w.out && rows && rows <= 32; }
#include "mma_support.inc"
cudaError_t dense_mma_launch(const MmaCall& c, cudaStream_t) {
    for (int i = 0; i < c.n; ++i) routes.emplace_back(c.out[i], true);
    return cudaSuccess;
}
template <bool> cudaError_t dispatch(const MultiArgs& a, const DenseWeight&, const void*, const float*, const float*, std::uint32_t, cudaStream_t) {
    for (int i = 0; i < a.n; ++i) routes.emplace_back(a.out[i], false);
    return cudaSuccess;
}
cudaError_t nvfp4_dense_rows(const DenseWeight&, const float*, float*, std::uint32_t, const float*, cudaStream_t) { return cudaSuccess; }
cudaError_t nvfp4_dense_rows_multi(const Nvfp4Job*, std::uint32_t, const float*, std::uint32_t, cudaStream_t) { return cudaSuccess; }
cudaError_t nvfp4_gate_up_silu_rows(const DenseWeight&, const DenseWeight&, const float*, float*, std::uint32_t, cudaStream_t) { return cudaSuccess; }
#include "dispatch.inc"
#include "workspace.inc"
}
#include "sampler.inc"
#include "maxrel.inc"
int failures = 0;
void check(bool ok, const char* message) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } }
template<class F> bool rejects(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    for (const auto* text : {"-1", "4294967296", "1x", "+1", ""})
        check(rejects([&] { (void)core::parse_u32(text); }), "strict unsigned CLI parsing");
    check(core::parse_u32("4294967295") == 0xFFFFFFFFU, "uint32 boundary");
    const std::uint32_t ids[] = {0, 7, 0xFFFFFFFFU};
    core::validate_token_ids(ids, 2, 8);
    check(rejects([&] { core::validate_token_ids(ids, 3, 8); }), "unsigned token rejection");
    const int negative[] = {-1};
    check(rejects([&] { core::validate_token_ids(negative, 1, 8); }), "negative token rejection");
    check(rejects([&] { core::require_capacity(0, 2, 1); }), "prompt capacity");
    check(rejects([&] { core::require_capacity(1, 1, 1); }), "generation capacity");
    check(rejects([&] { core::require_capacity(SIZE_MAX, 1, SIZE_MAX); }), "capacity wrap");
    check(rejects([&] { (void)core::checked_product({SIZE_MAX, 2}); }), "allocation overflow");
    core::require_capacity(1, 1, 2);
    int stream_a = 0, stream_b = 0;
    float* old = nullptr;
    check(cc::ensure_ws(32, &stream_a, &old) == cudaSuccess, "workspace allocation");
    float* grown = nullptr;
    check(cc::ensure_ws(64, &stream_a, &grown) == cudaSuccess, "workspace growth");
    check(old != grown && live_allocations.count(old), "captured old address survives growth");
    float* other = nullptr;
    (void)cc::ensure_ws(64, &stream_b, &other);
    check(other != grown && other != old, "separate stream workspace");
    current_device = 1;
    float* device_other = nullptr;
    (void)cc::ensure_ws(64, &stream_a, &device_other);
    check(device_other != grown, "separate device workspace");
    cc::nvfp4_release_workspace(&stream_a);
    current_device = 0;
    fail_allocation = true;
    check(cc::ensure_ws(128, &stream_a) != cudaSuccess && live_allocations.count(grown), "failed growth preserves pointers");
    fail_allocation = false;
    cc::nvfp4_release_workspace(&stream_a);
    cc::nvfp4_release_workspace(&stream_b);
    check(live_allocations.empty(), "workspace cleanup after graph lifetime");
    try {
        cc::UniqueHandle<void*, cudaFree> allocation;
        (void)cudaMalloc(allocation.put(), 32);
        std::vector<cc::UniqueHandle<void*, cudaFree>> owners;
        owners.push_back(std::move(allocation));
        throw std::runtime_error("partial load");
    } catch (const std::runtime_error&) {}
    check(live_allocations.empty(), "exception-safe allocation cleanup");
    float x = 0, y0 = 0, y1 = 0;
    cc::DenseWeight ws[] = {{&x, cc::WeightFormat::bf16, 16, 32}, {&x, cc::WeightFormat::bf16, 1, 32}};
    float* ys[] = {&y0, &y1};
    for (auto arithmetic : {cc::DenseArith::mma3, cc::DenseArith::mma2, cc::DenseArith::mma1, cc::DenseArith::fp32}) {
        cc::routes.clear();
        for (int i = 0; i < 2; ++i) (void)cc::dense_rows(ws[i], &x, ys[i], 1, nullptr, nullptr, arithmetic);
        auto separate = cc::routes;
        cc::routes.clear();
        (void)cc::dense_rows_multi(ws, ys, 2, &x, 1, nullptr, nullptr, arithmetic);
        std::sort(separate.begin(), separate.end());
        std::sort(cc::routes.begin(), cc::routes.end());
        check(separate == cc::routes, "mixed jobs retain individual routes");
    }
    strix::qwen35::EngineOptions a;
    a.turbo_weights = strix::qwen35::TurboWeights::onfly;
    a.turbo_lm_head = false;
    a.dense_arithmetic = cc::DenseArith::mma1;
    auto b = a; b.dense_arithmetic = cc::DenseArith::fp32;
    check(engine_identity(a) != engine_identity(b), "retained BF16 arithmetic identity");
    auto identity = engine_identity(a);
    cc::set_dense_arith(cc::DenseArith::fp32);
    check(engine_identity(a) == identity, "identity immutable under global selector");
    {  // --prefill-gdn: strict spelling; chunk16 is a FAST identity on non-exact prefill policies only
        namespace q = strix::qwen35;
        check(q::parse_prefill_gdn("serial") == q::PrefillGdn::serial, "prefill-gdn serial");
        check(q::parse_prefill_gdn("chunk16") == q::PrefillGdn::chunk16, "prefill-gdn chunk16");
        for (const char* bad : {"", "chunk", "Chunk16", "chunk16 ", "auto"})
            check(rejects([&] { (void)q::parse_prefill_gdn(bad); }), "prefill-gdn rejects other spellings");
        q::EngineOptions f = a;
        f.prefill_arith = q::PrefillArith::fast4;
        const std::string serial_id = engine_identity(f);
        f.prefill_gdn = q::PrefillGdn::chunk16;
        const std::string chunk_id = engine_identity(f);
        check(serial_id.ends_with("-pf-fast4") && chunk_id == serial_id + "-gdnc16", "chunk16 FAST identity suffix");
        f.prefill_arith = q::PrefillArith::exact;
        check(engine_identity(f).find("gdnc16") == std::string::npos, "chunk16 ignored under the exact prefill policy");
    }
    {  // --prefill-norm-fold: strict on|off; bitwise scheduling of bytes -> never part of the engine identity
        namespace q = strix::qwen35;
        check(q::parse_prefill_norm_fold("on") && !q::parse_prefill_norm_fold("off"), "prefill-norm-fold on|off");
        for (const char* bad : {"", "1", "ON", "auto", "on "})
            check(rejects([&] { (void)q::parse_prefill_norm_fold(bad); }), "prefill-norm-fold rejects other spellings");
        q::EngineOptions f = a;
        f.prefill_arith = q::PrefillArith::fast4;
        check(f.prefill_norm_fold, "prefill-norm-fold default on (bitwise, measured)");
        const std::string off_id = engine_identity(f);
        f.prefill_norm_fold = false;
        check(engine_identity(f) == off_id, "prefill-norm-fold keeps the FAST4 identity (bitwise)");
    }
    {  // --prefill-tile-feed: strict bulk|lsu; default bulk (measured); scheduling only -> never part of the engine identity
        namespace q = strix::qwen35;
        namespace ft = strix::cuda::core::nvfp4::tile;
        check(q::parse_prefill_tile_feed("bulk") == ft::Feed::bulk && q::parse_prefill_tile_feed("lsu") == ft::Feed::lsu,
              "prefill-tile-feed bulk|lsu");
        for (const char* bad : {"", "BULK", "lsu ", "tma", "on", "off", "1"})
            check(rejects([&] { (void)q::parse_prefill_tile_feed(bad); }), "prefill-tile-feed rejects other spellings");
        q::EngineOptions f = a;
        f.prefill_arith = q::PrefillArith::fast4;
        check(q::EngineOptions{}.prefill_tile_feed == ft::Feed::bulk && ft::kDefaultFeed == ft::Feed::bulk,
              "prefill-tile-feed default bulk (bitwise, measured)");
        f.prefill_tile_feed = ft::Feed::bulk;
        const std::string bulk_id = engine_identity(f);
        f.prefill_tile_feed = ft::Feed::lsu;
        check(engine_identity(f) == bulk_id, "prefill-tile-feed keeps the FAST4 identity (bitwise)");
    }
    {  // NVFP4 split-K policy -> identity on the real EngineOptions (review condition 2): EXACT has no "-skf1"; FAST ends in
       // "-skf1" for every NVFP4 source; without NVFP4 weights FAST and EXACT are one identity; qp::apply sets the option.
        namespace q = strix::qwen35;
        namespace qp = strix::qwen35::profile;
        for (const q::TurboWeights w : {q::TurboWeights::native, q::TurboWeights::onfly, q::TurboWeights::converted}) {
            q::EngineOptions f;
            f.turbo_weights = w;
            const std::string exact_id = engine_identity(f);
            check(exact_id.find("skf1") == std::string::npos, "default identity carries no -skf1");
            f.nvfp4_split = strix::split_k::Profile::fast;
            check(engine_identity(f) == exact_id + "-skf1", "FAST split identity = EXACT identity + -skf1");
        }
        q::EngineOptions off;
        off.turbo_weights = q::TurboWeights::off;
        const std::string off_id = engine_identity(off);
        off.nvfp4_split = strix::split_k::Profile::fast;
        check(engine_identity(off) == off_id, "turbo off: split policy is not part of the identity (no NVFP4 tensor)");
        for (const char* word : {"fast", "default"}) {
            q::EngineOptions o;
            o.turbo_weights = q::TurboWeights::converted;
            const auto r = qp::resolve(strix::profile::parse(word), qp::regime_of(o, qp::Route::windows, 300), {});
            qp::apply(o, *r);
            const bool fast = std::string_view(word) == "fast";
            check((o.nvfp4_split == strix::split_k::Profile::fast) == fast, "qp::apply on EngineOptions sets nvfp4_split");
            check(engine_identity(o).ends_with("-skf1") == fast, "resolved tier -> identity suffix");
        }
    }
    {  // FAST decode attention (decode_attn8): own identity suffix "-d8" on every weight source; the registry row is pending
       // (no qualified regime yet), so `fast` alone leaves it off and only the explicit --decode-attn8 on sets it.
        namespace q = strix::qwen35;
        namespace qp = strix::qwen35::profile;
        for (const q::TurboWeights w : {q::TurboWeights::off, q::TurboWeights::native, q::TurboWeights::converted}) {
            q::EngineOptions o;
            o.turbo_weights = w;
            const std::string id = engine_identity(o);
            check(id.find("-d8") == std::string::npos, "default identity carries no -d8");
            o.decode_attn8 = true;
            check(engine_identity(o).find("-d8") != std::string::npos && engine_identity(o) != id, "decode_attn8 -> -d8 identity");
        }
        for (const auto route : {qp::Route::large_prefill, qp::Route::windows}) {
            q::EngineOptions o;
            o.turbo_weights = q::TurboWeights::converted;
            const auto r = qp::resolve(strix::profile::parse("fast"), qp::regime_of(o, route, 65536), {});
            qp::apply(o, *r);
            check(!o.decode_attn8, "fast without a qualified decode_attn8 row leaves it off");
            q::EngineOptions e;
            e.turbo_weights = q::TurboWeights::converted;
            const auto re = qp::resolve(std::nullopt, qp::regime_of(e, route, 65536), {{"decode_attn8", "on"}});
            qp::apply(e, *re);
            check(e.decode_attn8 && engine_identity(e).find("-d8") != std::string::npos, "--decode-attn8 on sets the option");
        }
    }
    check(portable_sampler() == 0, "POSIX clock sampler compiles and closes");
    check(std::isinf(maxrel({NAN}, {0.0})) && std::isinf(maxrel({0.0F}, {INFINITY})), "nonfinite qualification rejection");
    return failures ? 1 : 0;
}
