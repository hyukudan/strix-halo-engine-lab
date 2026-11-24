// No-model fixture of the schedule (placement) trait of the HIP grouped microscaled BF16 tile at the Inkling 2 Mini expert
// shapes (gate_up 2304 x 2304 interleaved halves, down 2304 x 1152; 128 experts, top 8), NVFP4 and MXFP8, at decode row
// counts. Every schedule runs one WMMA chain per output over the K16 units in ascending order, so every arm must store
// the same bits. Checks per route draw (rows 1, 2, 8, 16, 48): the plain gate-up, the activated gate-up (paired
// epilogue) and the down projection of every arm byte-equal to the tall arm, every output finite (outputs are seeded with
// a NaN sentinel first); rows 48 runs the 32-row narrow CTA over two row slices. With --bench: ms per visit (activated
// gate-up + down, the served form) per arm and rows, skewed routes, arms interleaved twice (A B C .. A B C ..), and GB/s
// over the distinct expert bytes. Exit 1 on a failed check.
#include <hip/hip_runtime.h>
#include "strix/hip/legacy_intrinsics.hpp"
#include "strix/hip/legacy_launch.hpp"
#include "moe_core/grouped_microscaled.hpp"
#include "moe_core/device_grouped_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace packed = strix::weight::microscaled;
namespace sf = strix::weight::scalar_float;
namespace grouped = strix::moe_core::grouped_microscaled;
namespace gemm = strix::core::f16_tile_gemm;
using strix::moe_core::device_grouped::ExpertWeights;
using Format = packed::MatrixFormat;

constexpr unsigned kExperts = 128U, kTopK = 8U, kHidden = 2304U, kInter = 1152U, kGate = 2U * kInter;
constexpr unsigned kVariants = 6U;  // route draws cycled per timed rep (distinct experts per rep, cold weights)
constexpr unsigned kMaxRows = 48U, kMaxAssign = kMaxRows * kTopK;
constexpr float kGateCap = 4.0F, kUpCap = 25.0F;

void require_hip(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        std::cerr << "HIP failure: " << what << ": " << hipGetErrorString(status) << '\n';
        std::exit(2);
    }
}

std::uint16_t bf16(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}

template <class T>
T* device_copy(const std::vector<T>& host) {
    T* p = nullptr;
    require_hip(hipMalloc(&p, host.size() * sizeof(T)), "malloc");
    require_hip(hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "copy");
    return p;
}

struct Matrix {
    std::vector<std::uint8_t> codes, scales;
};

// Random valid codes; scales in a range that keeps every product finite.
Matrix make_matrix(Format f, unsigned rows, unsigned cols, std::mt19937& rng) {
    Matrix m;
    const bool nv = f == Format::nvfp4;
    m.codes.resize(std::size_t{rows} * (nv ? cols / 2U : cols));
    m.scales.resize(std::size_t{rows} * cols / (nv ? 16U : 32U));
    for (std::size_t i = 0; i < m.codes.size(); i += 4U) {
        const auto word = static_cast<std::uint32_t>(rng());
        for (std::size_t j = 0; j < 4U && i + j < m.codes.size(); ++j) {
            auto b = static_cast<std::uint8_t>(word >> (8U * j));
            if (!nv && (b & 0x7FU) == 0x7FU) b = static_cast<std::uint8_t>(b ^ 1U);  // no E4M3 NaN
            m.codes[i + j] = b;
        }
    }
    std::uniform_int_distribution<int> s4(0x20, 0x40), s8(118, 124);
    for (auto& s : m.scales) s = static_cast<std::uint8_t>(nv ? s4(rng) : s8(rng));
    return m;
}

struct Routes {
    std::vector<unsigned> ids;  // [rows][8] token-major
    unsigned distinct;
};

// Skewed (Zipf 0.9 over a shuffled expert order) top-8 draws without replacement per row.
Routes draw(unsigned rows, std::mt19937& rng) {
    std::vector<double> p(kExperts, 1.0);
    std::vector<unsigned> perm(kExperts);
    std::iota(perm.begin(), perm.end(), 0U);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (unsigned e = 0; e < kExperts; ++e) p[perm[e]] = 1.0 / std::pow(double(e + 1U), 0.9);
    Routes r{{}, 0U};
    std::vector<int> seen(kExperts, 0);
    for (unsigned row = 0; row < rows; ++row) {
        auto w = p;
        for (unsigned k = 0; k < kTopK; ++k) {
            std::discrete_distribution<unsigned> d(w.begin(), w.end());
            const unsigned e = d(rng);
            w[e] = 0.0;
            r.ids.push_back(e);
            seen[e] = 1;
        }
    }
    r.distinct = static_cast<unsigned>(std::count(seen.begin(), seen.end(), 1));
    return r;
}

struct Experts {
    std::vector<ExpertWeights> host;  // device addresses
    double expert_bytes{};
};

Experts make_experts(Format f, std::mt19937& rng) {
    Experts F;
    const bool nv = f == Format::nvfp4;
    for (unsigned e = 0; e < kExperts; ++e) {
        const float outer = nv ? 0.75F + 0.001F * float(e) : 1.0F;
        const auto gu = make_matrix(f, kGate, kHidden, rng);
        const auto dn = make_matrix(f, kHidden, kInter, rng);
        const auto upload = [&](const Matrix& m, unsigned rows, unsigned cols, packed::RowOrder order) {
            return packed::PackedMatrix{device_copy(m.codes), device_copy(m.scales), rows, cols, outer, order};
        };
        F.host.push_back({upload(gu, kGate, kHidden, packed::RowOrder::interleaved_halves),
                          upload(dn, kHidden, kInter, packed::RowOrder::contiguous)});
        if (e == 0U) F.expert_bytes = double(gu.codes.size() + gu.scales.size() + dn.codes.size() + dn.scales.size());
    }
    return F;
}

struct Device {
    std::uint16_t *gu_in{}, *dn_in{}, *act{};
    float *gu_out{}, *dn_out{};
    grouped::TileRows *gu_tiles{}, *dn_tiles{};
};

Device make_device() {
    Device d;
    require_hip(hipMalloc(&d.gu_in, std::size_t{kMaxAssign} * kHidden * 2U), "m");
    require_hip(hipMalloc(&d.dn_in, std::size_t{kMaxAssign} * kInter * 2U), "m");
    require_hip(hipMalloc(&d.act, std::size_t{kMaxAssign} * kInter * 2U), "m");
    require_hip(hipMalloc(&d.gu_out, std::size_t{kMaxAssign} * kGate * 4U), "m");
    require_hip(hipMalloc(&d.dn_out, std::size_t{kMaxAssign} * kHidden * 4U), "m");
    require_hip(hipMalloc(&d.gu_tiles, kExperts * sizeof(grouped::TileRows)), "m");
    require_hip(hipMalloc(&d.dn_tiles, kExperts * sizeof(grouped::TileRows)), "m");
    return d;
}

// Expert-major tiles of one route draw over gathered rows.
struct Plan {
    std::vector<grouped::TileRows> gu, dn;
    unsigned max_rows{}, distinct{};
};

Plan plan_of(const Experts& F, const Routes& r, unsigned rows, const Device& d) {
    Plan p;
    unsigned first = 0U;
    for (unsigned e = 0; e < kExperts; ++e) {
        unsigned n = 0U;
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned k = 0; k < kTopK; ++k) n += r.ids[row * kTopK + k] == e ? 1U : 0U;
        if (n == 0U) continue;
        p.max_rows = std::max(p.max_rows, n);
        p.gu.push_back({F.host[e].gate_up, d.gu_in + std::size_t{first} * kHidden, d.gu_out + std::size_t{first} * kGate,
                        kHidden, kGate, n});
        p.dn.push_back({F.host[e].down, d.dn_in + std::size_t{first} * kInter, d.dn_out + std::size_t{first} * kHidden,
                        kInter, kHidden, n});
        first += n;
    }
    p.distinct = r.distinct;
    return p;
}

void upload_plan(const Plan& p, const Device& d) {
    require_hip(hipMemcpy(d.gu_tiles, p.gu.data(), p.gu.size() * sizeof(grouped::TileRows), hipMemcpyHostToDevice), "t");
    require_hip(hipMemcpy(d.dn_tiles, p.dn.data(), p.dn.size() * sizeof(grouped::TileRows), hipMemcpyHostToDevice), "t");
}

grouped::Args plain_args(const grouped::TileRows* tiles, unsigned count, unsigned inputs, unsigned outputs, unsigned rows) {
    grouped::Args a{};
    a.tiles = tiles;
    a.count = count;
    a.inputs = inputs;
    a.outputs = outputs;
    a.max_tile_rows = rows;
    return a;
}

grouped::Args activated_args(const Plan& p, const Device& d) {
    auto a = plain_args(d.gu_tiles, unsigned(p.gu.size()), kHidden, kGate, p.max_rows);
    a.pair_rows = d.dn_tiles;
    a.rebase_from = d.dn_in;
    a.rebase_to = d.act;
    a.gate_cap = kGateCap;
    a.up_cap = kUpCap;
    return a;
}

// One schedule arm: the tile launch of grouped_microscaled with an explicit schedule (grid y = row slices of its CTA).
template <class Binding, class Sched>
void run(const grouped::Args& a) {
    const unsigned slices = grouped::tile_row_slices(a.max_tile_rows, grouped::kRowsPerTile, Sched::kRows);
    require_hip(gemm::gemm_packed_bf16<Binding, Sched>(a, a.inputs, a.outputs, a.count, nullptr, slices), "tile");
}

// Arms: the three schedule bodies through the tile launcher, and the public launchers with Schedule::automatic (the
// resolve_schedule path a binder takes).
constexpr unsigned kArms = 4U;
const char* const kArmNames[kArms] = {"tall", "staged", "narrow", "automatic"};

template <Format F, class Fn, class Public>
void with_arm(unsigned arm, Fn&& fn, Public&& pub) {
    switch (arm) {
    case 0: fn.template operator()<grouped::TallSchedule>(); break;
    case 1: fn.template operator()<gemm::IncumbentSchedule>(); break;
    case 2: fn.template operator()<grouped::NarrowSchedule>(); break;
    default: pub(); break;
    }
}

grouped::Args down_args(const Plan& p, const Device& d) {
    auto down = plain_args(d.dn_tiles, unsigned(p.dn.size()), kInter, kHidden, p.max_rows);
    down.rebase_from = d.dn_in;
    down.rebase_to = d.act;
    return down;
}

// The served form: activated gate-up (paired epilogue, BF16 activations into d.act) + down reading them.
template <Format F>
void visit(unsigned arm, const Plan& p, const Device& d) {
    with_arm<F>(
        arm,
        [&]<class S>() {
            run<grouped::ActivatedTileBinding<F>, S>(activated_args(p, d));
            run<grouped::TileBinding<F>, S>(down_args(p, d));
        },
        [&]() {
            require_hip(grouped::launch_activated<F>(activated_args(p, d), nullptr), "activated");
            require_hip(grouped::launch<F>(down_args(p, d), nullptr), "down");
        });
}

template <Format F>
void plain(unsigned arm, const Plan& p, const Device& d) {
    const auto gu = plain_args(d.gu_tiles, unsigned(p.gu.size()), kHidden, kGate, p.max_rows);
    const auto dn = plain_args(d.dn_tiles, unsigned(p.dn.size()), kInter, kHidden, p.max_rows);
    with_arm<F>(
        arm,
        [&]<class S>() {
            run<grouped::TileBinding<F>, S>(gu);
            run<grouped::TileBinding<F>, S>(dn);
        },
        [&]() {
            require_hip(grouped::launch<F>(gu, nullptr), "gate_up");
            require_hip(grouped::launch<F>(dn, nullptr), "down");
        });
}

template <class T>
std::vector<T> host_copy(const T* p, std::size_t n) {
    std::vector<T> h(n);
    require_hip(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}

bool finite_f32(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}
bool finite_bf16(const std::vector<std::uint16_t>& v) {
    return std::all_of(v.begin(), v.end(), [](std::uint16_t h) { return (h & 0x7F80U) != 0x7F80U; });
}

template <Format F>
bool check(const Experts& Fm, const Device& d, std::mt19937& rng) {
    const char* name = F == Format::nvfp4 ? "nvfp4" : "mxfp8";
    bool ok = true;
    for (unsigned rows : {1U, 2U, 8U, 16U, kMaxRows}) {
        const auto r = draw(rows, rng);
        const auto p = plan_of(Fm, r, rows, d);
        upload_plan(p, d);
        const std::size_t A = std::size_t{rows} * kTopK;
        std::vector<float> ref_gu, ref_dn, ref_dn2;
        std::vector<std::uint16_t> ref_act;
        for (unsigned arm = 0; arm < kArms; ++arm) {
            require_hip(hipMemset(d.gu_out, 0xFF, A * kGate * 4U), "s");  // NaN sentinel
            require_hip(hipMemset(d.dn_out, 0xFF, A * kHidden * 4U), "s");
            plain<F>(arm, p, d);
            require_hip(hipDeviceSynchronize(), "plain");
            const auto gu = host_copy(d.gu_out, A * kGate);
            const auto dn = host_copy(d.dn_out, A * kHidden);
            require_hip(hipMemset(d.act, 0xFF, A * kInter * 2U), "s");
            require_hip(hipMemset(d.dn_out, 0xFF, A * kHidden * 4U), "s");
            visit<F>(arm, p, d);
            require_hip(hipDeviceSynchronize(), "visit");
            const auto act = host_copy(d.act, A * kInter);
            const auto dn2 = host_copy(d.dn_out, A * kHidden);
            const bool finite = finite_f32(gu) && finite_f32(dn) && finite_f32(dn2) && finite_bf16(act);
            if (arm == 0U) {
                ref_gu = gu, ref_dn = dn, ref_act = act, ref_dn2 = dn2;
            }
            const bool same = std::memcmp(gu.data(), ref_gu.data(), gu.size() * 4U) == 0 &&
                              std::memcmp(dn.data(), ref_dn.data(), dn.size() * 4U) == 0 &&
                              std::memcmp(act.data(), ref_act.data(), act.size() * 2U) == 0 &&
                              std::memcmp(dn2.data(), ref_dn2.data(), dn2.size() * 4U) == 0;
            const bool pass = finite && same;
            ok = ok && pass;
            std::cout << "check " << name << " rows " << rows << " max_tile_rows " << p.max_rows << " arm "
                      << kArmNames[arm] << (finite ? " finite" : " NONFINITE") << (same ? " same-bits" : " DIFFER")
                      << (pass ? " PASS" : " FAIL") << '\n';
        }
    }
    return ok;
}

template <Format F>
void bench(const Experts& Fm, const Device& d, std::mt19937& rng) {
    const char* name = F == Format::nvfp4 ? "nvfp4" : "mxfp8";
    hipEvent_t e0, e1;
    require_hip(hipEventCreate(&e0), "e");
    require_hip(hipEventCreate(&e1), "e");
    for (unsigned rows : {1U, 2U, 4U, 8U, 16U}) {
        std::vector<Plan> ps;
        double distinct = 0.0;
        for (unsigned v = 0; v < kVariants; ++v) {
            ps.push_back(plan_of(Fm, draw(rows, rng), rows, d));
            distinct += ps.back().distinct;
        }
        distinct /= kVariants;
        const auto time = [&](unsigned arm) {
            std::vector<float> ms;
            for (unsigned rep = 0; rep < 4U * kVariants; ++rep) {
                const auto& p = ps[rep % kVariants];
                upload_plan(p, d);
                require_hip(hipDeviceSynchronize(), "sync");
                require_hip(hipEventRecord(e0, nullptr), "rec");
                visit<F>(arm, p, d);
                require_hip(hipEventRecord(e1, nullptr), "rec");
                require_hip(hipEventSynchronize(e1), "sync");
                float t = 0.0F;
                require_hip(hipEventElapsedTime(&t, e0, e1), "el");
                if (rep >= kVariants) ms.push_back(t);  // the first round warms up
            }
            std::sort(ms.begin(), ms.end());
            return double(ms[ms.size() / 2U]);
        };
        double t[2][kArms];
        for (unsigned pass = 0; pass < 2U; ++pass)
            for (unsigned arm = 0; arm < kArms; ++arm) t[pass][arm] = time(arm);
        const double bytes = distinct * Fm.expert_bytes;
        for (unsigned arm = 0; arm < kArms; ++arm) {
            const double m = 0.5 * (t[0][arm] + t[1][arm]);
            std::cout << "bench " << name << " rows " << rows << " U " << distinct << " arm " << kArmNames[arm]
                      << " ms " << m << " (" << t[0][arm] << " / " << t[1][arm] << ") GB/s " << bytes / m * 1e-6
                      << " vs_tall " << m / (0.5 * (t[0][0] + t[1][0])) << '\n';
        }
    }
}

template <Format F>
bool run_format(const Device& d, std::mt19937& rng, bool timed) {
    const auto experts = make_experts(F, rng);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::vector<std::uint16_t> xg(std::size_t{kMaxAssign} * kHidden), xd(std::size_t{kMaxAssign} * kInter);
    for (auto& v : xg) v = bf16(nd(rng));
    for (auto& v : xd) v = bf16(nd(rng));
    require_hip(hipMemcpy(d.gu_in, xg.data(), xg.size() * 2U, hipMemcpyHostToDevice), "x");
    require_hip(hipMemcpy(d.dn_in, xd.data(), xd.size() * 2U, hipMemcpyHostToDevice), "x");
    const bool ok = check<F>(experts, d, rng);
    if (timed) bench<F>(experts, d, rng);
    for (const auto& e : experts.host) {
        for (const auto* m : {&e.gate_up, &e.down}) {
            require_hip(hipFree(const_cast<std::uint8_t*>(m->codes)), "free");
            require_hip(hipFree(const_cast<std::uint8_t*>(m->block_scales)), "free");
        }
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const bool timed = argc > 1 && std::string(argv[1]) == "--bench";
    std::mt19937 rng(20261124U);
    const auto dev = make_device();
    bool ok = run_format<Format::nvfp4>(dev, rng, timed);
    ok = run_format<Format::mxfp8>(dev, rng, timed) && ok;
    std::cout << (ok ? "ALL PASS" : "FAILED") << '\n';
    return ok ? 0 : 1;
}
