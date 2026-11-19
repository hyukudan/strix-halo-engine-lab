// Host oracle for the resident expanded-operand cache (strix/weight/operand_cache_plan.hpp) and its shared CLI resolver.
// Invariants: (1) admitted entries never overlap and stay inside the arena, so every cached operand keeps exactly the bytes of its
// own expansion (the GEMM reads identical operands with and without the cache); (2) each entry is filled exactly once and every
// later use, on any stream, is ordered after the fill; (3) admission is deterministic first fit under the budget; (4) the default
// and the CLI resolve to the same values.
#include "strix/weight/operand_cache_plan.hpp"
#include "strix/qwen35/prefill_controls.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace oc = strix::weight::operand_cache;
// Same entity: two non-type template arguments naming the same function are the same specialization.
template <auto A, auto B> struct SameEntity { static constexpr bool value = false; };
template <auto A> struct SameEntity<A, A> { static constexpr bool value = true; };
static_assert(SameEntity<&oc::plan, &strix::cuda_core::operand_cache::plan>::value &&
              SameEntity<&oc::budget, &strix::cuda_core::operand_cache::budget>::value,
              "the forwarding header aliases the shared planner");

static int g_fail = 0;
#define REQUIRE(c)                                                                  \
    do {                                                                            \
        if (!(c)) {                                                                 \
            std::fprintf(stderr, "%s:%d: REQUIRE(%s) failed\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                               \
        }                                                                           \
    } while (0)

struct Options {
    oc::Mode prefill_operand_cache = oc::Mode::automatic;
    std::uint32_t prefill_operand_cache_cap_mib = 0;
};

// stand-in for an expansion kernel: a pure function of (weight id, byte index)
static unsigned char expand_byte(std::size_t weight, std::uint64_t i) {
    return static_cast<unsigned char>((weight * 131u + i * 7u + (i >> 9)) & 0xFF);
}

static void sizes() {
    REQUIRE(oc::entry_bytes(oc::Kind::e4m3, 17408, 5120) == 17408ULL * 5120);
    REQUIRE(oc::entry_bytes(oc::Kind::f16, 17408, 5120) == 2ULL * 17408 * 5120);
    // q8: bf16 values, then FP32 scales per 32-block at a 256-aligned offset
    REQUIRE(oc::scales_offset(oc::Kind::q8_bf16, 48, 5120) == oc::align_up(2ULL * 48 * 5120));
    REQUIRE(oc::entry_bytes(oc::Kind::q8_bf16, 48, 5120) == oc::align_up(2ULL * 48 * 5120) + 48ULL * 5120 / 32 * 4);
    REQUIRE(oc::scales_offset(oc::Kind::e4m3, 16, 128) == oc::kNone);
}

static void budget_and_modes() {
    const std::uint64_t G = 1ULL << 30;
    REQUIRE(oc::budget(oc::Mode::off, 80 * G, 96 * G, 0) == 0);
    REQUIRE(oc::budget(oc::Mode::automatic, 80 * G, 96 * G, 0) == 80 * G - 96 * G / 10);  // reserve 10 % of 96 GiB > 4 GiB
    REQUIRE(oc::budget(oc::Mode::on, 20 * G, 24 * G, 0) == 16 * G);                       // reserve floor 4 GiB
    REQUIRE(oc::budget(oc::Mode::automatic, 3 * G, 96 * G, 0) == 0);                      // below the reserve: nothing
    REQUIRE(oc::budget(oc::Mode::automatic, 80 * G, 96 * G, 1 * G) == 1 * G);             // cap
    REQUIRE(oc::admits(oc::Mode::automatic, oc::Kind::e4m3));
    REQUIRE(!oc::admits(oc::Mode::automatic, oc::Kind::f16));
    REQUIRE(!oc::admits(oc::Mode::automatic, oc::Kind::q8_bf16));
    REQUIRE(oc::admits(oc::Mode::on, oc::Kind::f16) && oc::admits(oc::Mode::on, oc::Kind::q8_bf16));
    REQUIRE(!oc::admits(oc::Mode::off, oc::Kind::e4m3));
    oc::Mode m = oc::Mode::off;
    REQUIRE(oc::parse_mode("auto", &m) && m == oc::Mode::automatic);
    REQUIRE(oc::parse_mode("on", &m) && m == oc::Mode::on);
    REQUIRE(oc::parse_mode("off", &m) && m == oc::Mode::off);
    REQUIRE(!oc::parse_mode("yes", &m) && m == oc::Mode::off);
}

static void first_fit_dedupe_overlap() {
    int k[6];
    const std::vector<oc::Candidate> c = {{&k[0], 1000}, {&k[1], 5000}, {&k[2], 300}, {&k[0], 1000}, {&k[3], 0}, {&k[4], 256}};
    // budget 2048: k0 (1024) fits, k1 (5120) does not, k2 (512) fits, dup k0 reuses, k3 zero skipped, k4 (256) fits
    const oc::Plan p = oc::plan(c, 2048);
    REQUIRE(p.admitted == 3 && p.rejected == 1 && p.duplicates == 1);
    REQUIRE(p.offset[0] == 0 && p.offset[1] == oc::kNone && p.offset[2] == 1024 && p.offset[3] == 0 && p.offset[4] == oc::kNone);
    REQUIRE(p.offset[5] == 1536 && p.arena_bytes == 1792);
    REQUIRE(p.wanted_bytes == 1024 + 5120 + 512 + 256);
    const oc::Plan q = oc::plan(c, 2048);  // deterministic
    REQUIRE(q.offset == p.offset && q.arena_bytes == p.arena_bytes);
    const oc::Plan none = oc::plan(c, 0);
    REQUIRE(none.admitted == 0 && none.arena_bytes == 0);
}

// The Qwen dense 27B projection set (48 GDN + 16 attention layers, hidden 5120, inter 17408) under FAST8: every operand fits a
// 96 GiB device with the 17.2 GB pack loaded, entries tile the arena without overlap, and each region holds exactly its own
// expansion after all fills (the bytes the GEMM reads equal a fresh per-call expansion).
static void dense27_arena_bytes() {
    struct Shape { std::uint32_t out, in; };
    std::vector<Shape> shapes;
    for (int l = 0; l < 64; ++l) {
        if (l % 4 != 3) shapes.insert(shapes.end(), {{10240, 5120}, {6144, 5120}, {48, 5120}, {48, 5120}, {5120, 6144}});
        else shapes.insert(shapes.end(), {{12288, 5120}, {1024, 5120}, {1024, 5120}, {5120, 6144}});
        shapes.insert(shapes.end(), {{17408, 5120}, {17408, 5120}, {5120, 17408}});
    }
    REQUIRE(shapes.size() == 496);  // = big_kernel_f8 launches per 4096-row chunk in the receipt profile
    std::vector<int> keys(shapes.size());
    std::vector<oc::Candidate> c;
    std::uint64_t raw = 0;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        c.push_back({&keys[i], oc::entry_bytes(oc::Kind::e4m3, shapes[i].out, shapes[i].in)});
        raw += c.back().bytes;
    }
    REQUIRE(raw == 24'350'556'160ULL);  // 24.35e9 e4m3 bytes = linear elements of the 27B trunk (STATE section 3)
    const std::uint64_t G = 1ULL << 30;
    const oc::Plan p = oc::plan(c, oc::budget(oc::Mode::automatic, 96 * G - 20 * G, 96 * G, 0));
    REQUIRE(p.admitted == 496 && p.rejected == 0 && p.arena_bytes >= raw && p.arena_bytes - raw < 496 * oc::kAlign);
    // no overlap: sort-free check via end <= next offset in layer order (first fit admits in order when everything fits)
    for (std::size_t i = 0; i + 1 < c.size(); ++i) {
        REQUIRE(p.offset[i] % oc::kAlign == 0);
        REQUIRE(p.offset[i] + c[i].bytes <= p.offset[i + 1]);
    }
    REQUIRE(p.offset.back() + c.back().bytes <= p.arena_bytes);
    // bytes oracle on a scaled-down copy (same plan rule, small arena in host memory)
    std::vector<oc::Candidate> s;
    for (std::size_t i = 0; i < 40; ++i) s.push_back({&keys[i], c[i].bytes / 65536 + 1});
    const oc::Plan ps = oc::plan(s, 1 << 20);
    std::vector<unsigned char> arena(ps.arena_bytes, 0xEE);
    for (std::size_t i = 0; i < s.size(); ++i)
        if (ps.offset[i] != oc::kNone)
            for (std::uint64_t b = 0; b < s[i].bytes; ++b) arena[ps.offset[i] + b] = expand_byte(i, b);
    for (std::size_t i = 0; i < s.size(); ++i)
        if (ps.offset[i] != oc::kNone)
            for (std::uint64_t b = 0; b < s[i].bytes; ++b) REQUIRE(arena[ps.offset[i] + b] == expand_byte(i, b));
}

// fill once, then every use (any stream, any chunk, any request) waits on the fill event; a failed fill is retried
static void fill_once_ordering() {
    oc::EntryState st;
    REQUIRE(oc::on_use(st) == oc::Use::fill);
    for (int chunk = 0; chunk < 8; ++chunk) REQUIRE(oc::on_use(st) == oc::Use::wait);
    REQUIRE(st.fills == 1 && st.waits == 8);
    oc::EntryState bad;
    REQUIRE(oc::on_use(bad) == oc::Use::fill);
    oc::on_fill_failed(bad);
    REQUIRE(oc::on_use(bad) == oc::Use::fill && bad.fills == 2);
    REQUIRE(oc::on_use(bad) == oc::Use::wait);
}

static void resolver() {
    Options o;
    strix::qwen35::resolve_operand_cache(o, "", "");
    REQUIRE(o.prefill_operand_cache == oc::Mode::automatic && o.prefill_operand_cache_cap_mib == 0);
    strix::qwen35::resolve_operand_cache(o, "on", "2048");
    REQUIRE(o.prefill_operand_cache == oc::Mode::on && o.prefill_operand_cache_cap_mib == 2048);
    strix::qwen35::resolve_operand_cache(o, "off", "");
    REQUIRE(o.prefill_operand_cache == oc::Mode::off && o.prefill_operand_cache_cap_mib == 2048);
    for (const auto& bad : {std::pair<const char*, const char*>{"maybe", ""}, {"auto", "-1"}, {"auto", "99999999999"}, {"auto", "12x"}}) {
        bool threw = false;
        Options before = o;
        try {
            strix::qwen35::resolve_operand_cache(o, bad.first, bad.second);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        REQUIRE(threw && o.prefill_operand_cache == before.prefill_operand_cache &&
                o.prefill_operand_cache_cap_mib == before.prefill_operand_cache_cap_mib);
    }
}

int main() {
    sizes();
    budget_and_modes();
    first_fit_dedupe_overlap();
    dense27_arena_bytes();
    fill_once_ordering();
    resolver();
    if (g_fail) {
        std::fprintf(stderr, "operand cache host oracle: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("operand cache host oracle: PASS\n");
    return 0;
}
