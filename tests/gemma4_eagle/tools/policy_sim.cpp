// AutoPolicy old-vs-new simulator used for the cost-fit policy revision ("never makes things worse").
// Not built by CMake. Compile once per policy variant, each against its own auto_policy.hpp/.cpp, e.g. (MSVC):
//   cl /O2 /EHsc /std:c++20 /MD /I<variant>/src /I<repo>/src /I<repo>/include sim.cpp <variant>/src/gemma4/engine/auto_policy.cpp
//      strix_gemma4_dflash.lib strix_gemma4_serve.lib strix_gemma4_serve_support.lib
// Prints, per synthetic profile (verify ms per k from the g3 node sweep), auto throughput relative to the best fixed k.
#include "gemma4/engine/auto_policy.hpp"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
namespace e = strix::gemma4::engine;
namespace df = strix::gemma4::dflash;
struct Rng { std::uint64_t s; double u() { s = s*6364136223846793005ULL+1442695040888963407ULL; return double(s >> 11)*0x1p-53; } };
struct Profile { const char* name; double verify[8]; double draft_fixed, draft_per; double q1, q2; unsigned switch_at; };
// returns tokens per ms over a run
double run(const Profile& p, unsigned seed, int forced, unsigned cycles) {
    e::AutoPolicy pol; Rng r{seed*7919ULL+1};
    double ms = 0, tok = 0; std::uint64_t pos = 0;
    for (unsigned c = 0; c < cycles; ++c) {
        if (c % 128 == 0) pos = 0; // a new 256-token request every ~128 cycles
        unsigned k; bool probe = false;
        if (forced >= 0) k = unsigned(forced);
        else { auto d = pol.choose(7,pos); k = d.k; probe = d.probe; }
        const double q = c < p.switch_at ? p.q1 : p.q2;
        df::Acceptance a; a.cycles = 1; a.attempted = k; unsigned acc = 0;
        for (unsigned d = 0; d < k; ++d) { ++a.depth_eligible[d]; ++a.eligible; if (r.u() >= q) break; ++a.depth_accepted[d]; ++a.accepted; ++acc; }
        a.emitted = acc+1;
        const double noise = 1+0.05*(r.u()-0.5);
        const double draft = k ? p.draft_fixed+p.draft_per*k : 0;
        const double verify = p.verify[k]*noise, ctx = 0.2;
        const double total = draft+verify+ctx;
        if (forced < 0) pol.observe(pos,k,total,draft,verify,ctx,a,probe);
        ms += total; tok += a.emitted; pos += a.emitted;
    }
    return tok/ms;
}
int main() {
    const Profile ps[] = {
        {"mtp-g3-en",{20,21.6,23.1,25.4,29.6,31.5,33.4,35.3},0,0.56,0.69,0.69,1u<<30},
        {"mtp-hi-acc",{20,21.6,23.1,25.4,29.6,31.5,33.4,35.3},0,0.56,0.9,0.9,1u<<30},
        {"mtp-low-acc",{20,21.6,23.1,25.4,29.6,31.5,33.4,35.3},0,0.56,0.3,0.3,1u<<30},
        {"mtp-hi-then-low",{20,21.6,23.1,25.4,29.6,31.5,33.4,35.3},0,0.56,0.9,0.25,1500},
        {"mtp-low-then-hi",{20,21.6,23.1,25.4,29.6,31.5,33.4,35.3},0,0.56,0.25,0.9,1500},
        {"dflash-g1",{20,21.6,23.1,25.4,29.6,31.5,33.4,35.3},10.4,0,0.6,0.6,1u<<30},
        {"dflash-code",{20,21.6,23.1,25.4,29.6,31.5,33.4,35.3},10.4,0,0.8,0.8,1u<<30},
        {"flat-verify",{20,20.5,21,21.5,22,22.5,23,23.5},0,0.56,0.8,0.8,1u<<30},
    };
    for (const auto& p : ps) {
        double a = 0; const unsigned N = 24, C = 3000;
        for (unsigned s = 0; s < N; ++s) a += run(p,s,-1,C);
        double best = 0; int bk = 0;
        for (int k = 0; k < 8; ++k) { double f = 0; for (unsigned s = 0; s < N; ++s) f += run(p,s,k,C); if (f > best) { best = f; bk = k; } }
        std::printf("%-16s auto %.5f  best-fixed k=%d %.5f  auto/best %.4f\n", p.name, a/N, bk, best/N, a/best);
    }
}
