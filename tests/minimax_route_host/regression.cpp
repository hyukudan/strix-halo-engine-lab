// Real engine route predicates + real host workspace planner, no HIP calls.
#include "core/linear_plan.hpp"
#include "minimax/engine/route_identity.hpp"
#include "moe_core/grouped_policy.hpp"
#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace route = strix::minimax::route;
namespace mmoe {
inline bool invariant = true;
bool prefill_order_invariant() { return invariant; }
void set_prefill_moe_order(int value) { invariant = value != 0; }
const auto& prefill_moe_order() {
    return strix::moe_core::resolve_grouped_order(invariant ? "invariant" : "adaptive");
}
using route::gemm_splits;
using route::gemm_tile_rows;
struct Exl3 {
    unsigned input = 6144, output = 256, bits = 4;
    const void* trellis = this;
    const void* suh = this;
    const void* svh = this;
};
struct PrefillDenseJob { Exl3 w; float* out; };
struct ExpertSlices { Exl3 gate, up, down; };
constexpr unsigned kHidden = 6144, kInter = 1536, kExperts = 128, kTopE = 4, kI4Experts = kExperts + 1;
struct PrefillMoeView {
    unsigned rows;
    const float* y;
    const std::uint32_t* ids;
    const float* weights;
    const ExpertSlices* experts;
    ExpertSlices shared;
    const float* out;
    const void* i4 = nullptr;
    bool i4_all = true;
};
// Device metadata size is a stand-in: tests check admission and bounds, not ABI.
namespace pfk { struct Item { std::byte metadata[256]; }; }
struct WItem { std::byte metadata[256]; };
#include "planner.inc"
}

// Defaults table stand-in: whether STRIX_MMX_PF_ORDER came from the table (not the user) and the declines seen.
namespace defaults {
inline bool order_defaulted = false;
inline int declines = 0;
bool from_defaults(const char*) { return order_defaulted; }
void decline_default(const char*, const char*, const std::string&) { ++declines; }
}

struct Engine {
    bool gemv = false, linears_loaded = true, moe_opt = true, fp32_prefill = false;
    bool pf_dense_qkv = false, pf_dense_o = false, pf_dense_mlp = false, pf_dense_tail_fixed = true;
    void* pf_ws = this;
    std::size_t pf_ws_bytes = std::size_t{1} << 30;
    struct Options { unsigned max_rows = 4096; } opt;
    float buffer = 0;
    float *q = &buffer, *k = &buffer, *v = &buffer, *part = &buffer;
    float *big1 = &buffer, *big2 = &buffer;
    struct Layer { mmoe::Exl3 rq, rk, rv, ro; mmoe::ExpertSlices rdense; bool moe = false; };
    std::vector<Layer> layers{3};
    Engine() {
        for (auto& y : layers) {
            y.rq.output = 4096;
            y.ro.input = 4096;
            y.ro.output = 6144;
            for (auto* w : {&y.rdense.gate, &y.rdense.up, &y.rdense.down}) w->output = 6144;
        }
    }
#include "engine.inc"
    bool use_dense_mlp(std::size_t n, const Layer& y) const {
        float* out = part;
#include "dense_mlp.inc"
    }
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
void startup(const Engine& e) {
    (void)e;
#include "startup.inc"
}
void startup_tests() {
    Engine e;
    startup(e);
    e.linears_loaded = false;
    rejects([&] { startup(e); }, "LINEARS=off accepted invariant identity");
    e.linears_loaded = true;
    e.moe_opt = false;
    rejects([&] { startup(e); }, "MOE=off accepted invariant identity");
    e.moe_opt = true;
    for (auto member : {&Engine::Layer::rq, &Engine::Layer::rk, &Engine::Layer::rv, &Engine::Layer::ro}) {
        auto& w = e.layers.back().*member;  // inspect all layers, not only layer zero
        w.bits = 2;
        rejects([&] { startup(e); }, "unsupported format accepted invariant identity");
        w.bits = 4;
        w.trellis = nullptr;
        rejects([&] { startup(e); }, "missing raw weights accepted invariant identity");
        w.trellis = &e;
    }
    for (auto* w : {&e.layers.back().rdense.gate, &e.layers.back().rdense.up, &e.layers.back().rdense.down}) {
        w->bits = 7;
        e.pf_dense_tail_fixed = false;
        startup(e);  // default: the dense MLP keeps the library tails, so its weights are not part of the contract
        e.pf_dense_tail_fixed = true;
        rejects([&] { startup(e); }, "unsupported dense MLP weights accepted invariant identity");
        w->bits = 4;
        w->suh = nullptr;
        rejects([&] { startup(e); }, "missing dense MLP scales accepted invariant identity");
        w->suh = &e;
    }
    const mmoe::PrefillDenseJob gu[] = {{e.layers.back().rdense.gate, e.big1}, {e.layers.back().rdense.up, e.big2}};
    e.pf_ws_bytes = mmoe::prefill_dense_workspace_bytes(gu, 2, e.opt.max_rows) - 1;
    rejects([&] { startup(e); }, "workspace that fits projections but not dense gate/up accepted invariant identity");
    e.pf_ws_bytes = 1;
    rejects([&] { startup(e); }, "insufficient workspace accepted invariant identity");
    e.pf_ws = nullptr;
    rejects([&] { startup(e); }, "missing workspace accepted invariant identity");
    mmoe::invariant = false;
    startup(e);  // adaptive keeps its existing fallback contract
    // A defaulted (table) invariant order whose route is unmet declines to adaptive, as on the landing base; the dense
    // tail lever needs the invariant order, so it keeps the hard error.
    Engine d;
    d.pf_dense_tail_fixed = false;
    d.moe_opt = false;  // STRIX_MM_MOE=off with STRIX_MMX_PF_ORDER unset
    defaults::order_defaulted = true;
    mmoe::invariant = true;
    startup(d);
    require(!mmoe::invariant && defaults::declines == 1, "defaulted invariant order did not decline to adaptive");
    mmoe::invariant = true;
    d.pf_dense_tail_fixed = true;
    rejects([&] { startup(d); }, "dense tail lever accepted a declined invariant order");
    require(mmoe::invariant && defaults::declines == 1, "dense tail lever declined the order");
    defaults::order_defaulted = false;
    d.pf_dense_tail_fixed = false;
    rejects([&] { startup(d); }, "explicit invariant order declined to adaptive");
}
void resources_tests() {
    Engine e;
    mmoe::PrefillDenseJob qkv[3] = {{e.layers[0].rq, e.q}, {e.layers[0].rk, e.k}, {e.layers[0].rv, e.v}};
    mmoe::PrefillDenseJob o1[1] = {{e.layers[0].ro, e.part}};
    for (unsigned rows : {1U, 8U, 256U}) {
        qkv[1].w.trellis = nullptr;
        rejects([&] { (void)e.dense_qkv(rows, qkv); }, "qkv silently fell back without raw weights");
        qkv[1].w.trellis = &e;
        e.pf_ws = nullptr;
        rejects([&] { (void)e.dense_qkv(rows, qkv); }, "qkv silently fell back without workspace");
        e.pf_ws = &e;
        qkv[2].w.bits = 2;
        rejects([&] { (void)e.dense_qkv(rows, qkv); }, "qkv silently fell back with unsupported bits");
        qkv[2].w.bits = 4;
    }
    o1[0].w.trellis = nullptr;
    rejects([&] { (void)e.dense_o(1, o1); }, "o silently fell back without raw weights");
    o1[0].w.trellis = &e;
    e.pf_ws_bytes = 1;
    rejects([&] { (void)e.dense_o(1, o1); }, "o silently fell back with small workspace");
    rejects([&] { (void)e.dense_qkv(256, qkv); }, "qkv silently fell back with small workspace");
}
void decode_tests() {
    Engine e;
    e.gemv = true;  // LINEARS=load: raw weights exist, optimized decode branch is bypassed
    e.pf_dense_qkv = e.pf_dense_o = true;
    e.pf_dense_mlp = true;
    const mmoe::PrefillDenseJob qkv[3] = {{e.layers[0].rq, e.q}, {e.layers[0].rk, e.k}, {e.layers[0].rv, e.v}};
    const mmoe::PrefillDenseJob o1[1] = {{e.layers[0].ro, e.part}};
    for (bool inv : {false, true}) {
        mmoe::invariant = inv;
        for (unsigned rows = 1; rows <= 8; ++rows) {
            require(!e.dense_qkv(rows, qkv), "decode/verification qkv entered prefill arithmetic");
            require(!e.dense_o(rows, o1), "decode/verification o entered prefill arithmetic");
            require(!e.use_dense_mlp(rows, e.layers[0]), "decode/verification MLP entered prefill arithmetic");
        }
    }
}
void mlp_tests() {
    Engine e;
    const auto& y = e.layers[0];
    const mmoe::PrefillDenseJob gu[] = {{y.rdense.gate, e.big1}, {y.rdense.up, e.big2}};
    const mmoe::PrefillDenseJob dn[] = {{y.rdense.down, e.part}};
    e.pf_dense_tail_fixed = false;  // default: only the 4096-row preference leaves the library (landing-base routes)
    for (unsigned rows : {1U, 9U, 4096U})
        for (bool preferred : {false, true}) {
            e.pf_dense_mlp = preferred;
            require(e.use_dense_mlp(rows, y) == (rows == 4096 && preferred), "default dense MLP tail left the library chain");
        }
    e.pf_ws = nullptr;
    require(!e.use_dense_mlp(9, y), "default dense MLP tail refused without the opt-in");
    e.pf_ws = &e;
    e.pf_dense_tail_fixed = true;
    for (unsigned rows : {1U, 9U, 4096U}) {
        for (bool preferred : {false, true}) {
            e.pf_dense_mlp = preferred;
            const bool v4 = e.use_dense_mlp(rows, y);
            require(v4 == (rows < 4096 || preferred), "dense MLP did not select the fixed chain for a short cut");
            const auto gate_up = mmoe::make_dense_plan(gu, 2, rows);
            const auto down = mmoe::make_dense_plan(dn, 1, rows);
            const unsigned lib_splits = route::gemm_splits(6144, 6144, 4, rows);
            require((v4 ? gate_up.splits[0] : lib_splits) == 1 && (v4 ? gate_up.splits[1] : lib_splits) == 1 &&
                    (v4 ? down.splits[0] : lib_splits) == 1, "dense MLP launch plan has a variable split");
        }
    }
    e.pf_ws = nullptr;
    rejects([&] { (void)e.use_dense_mlp(9, y); }, "invariant MLP silently fell back without workspace");
    e.pf_ws = &e;
    mmoe::invariant = false;
    require(!e.use_dense_mlp(1, y) && !e.use_dense_mlp(9, y) && e.use_dense_mlp(4096, y), "adaptive MLP route changed");
    e.pf_dense_mlp = false;
    require(!e.use_dense_mlp(4096, y), "adaptive MLP OFF route changed");
}
void grouped_plan_tests() {
    std::array<mmoe::ExpertSlices, mmoe::kExperts> experts{};
    float buffer = 0;
    for (unsigned rows : {1U, 9U, 4096U}) {
        std::vector<std::uint32_t> ids(std::size_t{rows} * mmoe::kTopE);
        std::vector<float> weights(ids.size(), 0.25F);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned k = 0; k < mmoe::kTopE; ++k)
                ids[std::size_t{r} * mmoe::kTopE + k] = (r + k) % mmoe::kExperts;
        const auto p = mmoe::make_plan({rows, &buffer, ids.data(), weights.data(), experts.data(), experts[0], &buffer});
        require(p.groups.size() > 1 && p.groups.back().expert == mmoe::kExperts && p.groups.back().n == rows,
                "missing shared or routed job in grouped plan");
        for (const auto& g : p.groups)
            require(g.s_gu == 1 && g.s_dn == 1, "routed/shared launch plan has a variable split");
    }
}
void route_tests() {
    Engine e;
    for (unsigned bits : {3U, 4U, 5U, 6U, 8U}) {
        mmoe::PrefillDenseJob qkv[3] = {{e.layers[0].rq, e.q}, {e.layers[0].rk, e.k}, {e.layers[0].rv, e.v}};
        mmoe::PrefillDenseJob o1[1] = {{e.layers[0].ro, e.part}};
        for (auto& j : qkv) j.w.bits = bits;
        o1[0].w.bits = bits;
        for (unsigned rows : {1U, 8U, 9U, 17U, 256U, 1024U, 4096U}) {
            for (bool inv : {false, true}) {
                mmoe::invariant = inv;
                e.pf_dense_qkv = e.pf_dense_o = false;
                const bool qroute = e.dense_qkv(rows, qkv), oroute = e.dense_o(rows, o1);
                if (inv) {
                    for (const auto& j : qkv)
                        require(qroute || (rows > 8 && route::gemm_splits(j.w.input, j.w.output, bits, rows) == 1),
                                "engine retained off-contract qkv library route");
                    require(oroute || (rows > 8 && route::gemm_splits(o1[0].w.input, o1[0].w.output, bits, rows) == 1),
                            "engine retained off-contract o library route");
                } else {
                    require(!qroute && !oroute, "adaptive OFF routes changed");
                    e.pf_dense_qkv = e.pf_dense_o = true;
                    require(e.dense_qkv(rows, qkv) == (rows >= 1024), "adaptive qkv threshold changed");
                    require(e.dense_o(rows, o1) == (rows == 4096), "adaptive o threshold changed");
                }
            }
        }
    }
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected test case");
        const std::string_view name(argv[1]);
        if (name == "startup") startup_tests();
        else if (name == "resources") resources_tests();
        else if (name == "decode") decode_tests();
        else if (name == "routes") route_tests();
        else if (name == "mlp") mlp_tests();
        else if (name == "grouped-plan") grouped_plan_tests();
        else throw std::runtime_error("unknown test case");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
