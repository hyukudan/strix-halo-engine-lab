// Host tests of the expert union of a verify window (strix/moe/expert_union.hpp) and of the union-rows admission of the A8
// probe / down contracts (validate_union, validate_down_union). No GPU.
#include "strix/moe/expert_union.hpp"
#include "strix/moe_a8_down_contract.hpp"
#include "strix/moe_q4k_a8_contract.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
namespace eu = strix::moe::expert_union;
namespace a8 = strix::moe_q4k_a8;
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}
template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void plan_reference() {
    // R = 1: the K slots in slot order, every entry held by row 0 only.
    const std::vector<std::uint32_t> one{7, 3, 9};
    const auto p1 = eu::build(one, 1, 3, 16);
    expect(p1.size() == 3 && p1[0].expert == 7 && p1[1].expert == 3 && p1[2].expert == 9, "R1 order = slot order");
    expect(p1[1].slot[0] == 1 && p1[1].slot[1] == eu::no_slot, "R1 slot map");
    // R = 3, K = 3: experts shared across rows appear once, first-occurrence order of (row, slot).
    const std::vector<std::uint32_t> ids{5, 1, 2,   // row 0
                                         2, 6, 5,   // row 1
                                         7, 1, 6};  // row 2
    const auto p = eu::build(ids, 3, 3, 8);
    const std::uint32_t order[] = {5, 1, 2, 6, 7};
    expect(p.size() == 5, "R3 distinct count");
    for (std::size_t u = 0; u < p.size() && u < 5; ++u) expect(p[u].expert == order[u], "R3 first-occurrence order");
    // expert 5: row 0 slot 0, row 1 slot 2; expert 1: row 0 slot 1, row 2 slot 1; expert 6: row 1 slot 1, row 2 slot 2.
    expect(p[0].slot[0] == 0 && p[0].slot[1] == 2 && p[0].slot[2] == eu::no_slot, "R3 expert 5 slots");
    expect(p[1].slot[0] == 1 && p[1].slot[1] == eu::no_slot && p[1].slot[2] == 1, "R3 expert 1 slots");
    expect(p[3].slot[0] == eu::no_slot && p[3].slot[1] == 1 && p[3].slot[2] == 2, "R3 expert 6 slots");
    // Every (row, slot) appears exactly once in the plan.
    unsigned held = 0;
    for (const auto& e : p)
        for (unsigned r = 0; r < eu::max_rows; ++r) held += e.slot[r] != eu::no_slot;
    expect(held == 9, "every assignment held once");
    // Refusals: shape, range, repeat within a row.
    expect(throws([] { (void)eu::build(std::vector<std::uint32_t>(9 * 2, 0), 9, 2, 8); }), "R9 refused");
    expect(throws([] { (void)eu::build(std::vector<std::uint32_t>(17, 0), 1, 17, 32); }), "K17 refused");
    expect(throws([] { (void)eu::build(std::vector<std::uint32_t>{1, 8}, 1, 2, 8); }), "id >= experts refused");
    expect(throws([] { (void)eu::build(std::vector<std::uint32_t>{3, 3}, 1, 2, 8); }), "repeat in a row refused");
    expect(!throws([] { (void)eu::build(std::vector<std::uint32_t>{3, 4, 3, 4}, 2, 2, 8); }), "same experts in two rows admitted");
    expect(eu::plan_bytes == 16U + 128U * 12U, "plan extent");
}

// Hy4 preset geometry: probe input = latent 1536, I = 1536, 512 experts, K = 16; down K-quant Q6_K.
constexpr std::uint32_t L = 1536, I = 1536, E = 512, K = 16;
std::uintptr_t cursor = 0x100000;
a8::DeviceBytes view(std::uint64_t bytes) {
    const a8::DeviceBytes v{cursor, bytes};
    cursor += static_cast<std::uintptr_t>((bytes + 255U) / 256U * 256U);
    return v;
}
a8::Request probe_request(std::uint32_t rows) {
    a8::Request rq;
    rq.enabled = true;
    rq.approved_numerical_identity = a8::numerical_identity_for_gate_type(12);
    rq.input = a8::InputKind::fp32;
    rq.shape = {L, I, E, K, rows};
    return rq;
}
a8::Bindings probe_bindings(std::uint32_t R) {
    const std::uint64_t w = std::uint64_t{E} * I * (L / 256U) * 144U;
    a8::Bindings b;
    b.input = view(R * L * 4ULL); b.gate = view(w); b.up = view(w); b.routes = view(R * K * 4ULL);
    b.codes = view(R * L); b.deltas = view(R * (L / 32U) * 4ULL); b.sums = view(R * (L / 32U) * 4ULL);
    b.status = view(4); b.output = view(R * K * 2ULL * I * 4U);
    return b;
}

void probe_admission() {
    for (std::uint32_t R : {1U, 2U, 4U, 8U}) {
        const auto b = probe_bindings(R);
        const auto plan = view(eu::plan_bytes);
        expect(!throws([&] { a8::validate_union(probe_request(R), b, plan); }), "probe union admits R 1/2/4/8");
    }
    const auto b2 = probe_bindings(2);
    const auto plan = view(eu::plan_bytes);
    expect(throws([&] { a8::validate_union(probe_request(9), probe_bindings(9), plan); }), "probe union refuses R9");
    expect(throws([&] { a8::validate_union(probe_request(2), probe_bindings(3), plan); }), "per-row extents must be R x C1");
    expect(throws([&] { a8::validate_union(probe_request(2), b2, a8::DeviceBytes{plan.address, plan.bytes - 4}); }),
           "plan extent exact");
    expect(throws([&] { a8::validate_union(probe_request(2), b2, a8::DeviceBytes{b2.output.address, eu::plan_bytes}); }),
           "plan disjoint from output");
    auto q5 = probe_request(2);
    q5.gate_type = q5.up_type = 13;
    q5.approved_numerical_identity = a8::numerical_identity_for_gate_type(13);
    expect(throws([&] { a8::validate_union(q5, b2, plan); }), "union is Q4_K gate/up only");
    // C1 itself still refuses logical_rows > 1 (unchanged incumbent).
    expect(throws([&] { a8::validate(probe_request(2), b2); }), "C1 validate unchanged: logical_rows 1 only");
}

void down_admission() {
    for (std::uint32_t R : {1U, 4U, 8U}) {
        a8::DownRequest dq;
        dq.enabled = true;
        dq.weight_type = 14;
        dq.approved_numerical_identity = a8::down_numerical_identity(14);
        dq.shape = {L, I, E, K};
        dq.swiglu_limit = 10.0F;
        a8::DownBindings db;
        db.intermediates = view(R * K * 2ULL * I * 4U);
        db.weights = view(std::uint64_t{E} * L * a8::down_row_bytes(14, I));
        db.routes = view(R * K * 4ULL); db.codes = view(R * K * std::uint64_t{I});
        db.deltas = view(R * K * (I / 32U) * 4ULL); db.sums = view(R * K * (I / 32U) * 4ULL);
        db.rawdots = view(R * K * std::uint64_t{L} * 4U); db.status = view(4); db.aggregate_status = {0, 0};
        const auto plan = view(eu::plan_bytes);
        expect(!throws([&] { a8::validate_down_union(dq, db, R, plan); }), "down union admits Q6_K R 1/4/8");
        expect(throws([&] { a8::validate_down_union(dq, db, R + 1U, plan); }), "down union rows must match the extents");
        auto q51 = dq;
        q51.weight_type = 7;
        q51.approved_numerical_identity = a8::down_numerical_identity(7);
        expect(throws([&] { a8::validate_down_union(q51, db, R, plan); }), "down union is K-quant only");
    }
}
}  // namespace

int main() {
    plan_reference();
    probe_admission();
    down_admission();
    if (failures != 0) {
        std::cerr << failures << " expert union test(s) failed\n";
        return 1;
    }
    std::cout << "expert union host tests: PASS\n";
    return 0;
}
