#include "gemma4/prefill/attention.hpp"
#include "../gemma4_gpu/test.hpp"
#include <limits>

using namespace strix::gemma4;
namespace pf = strix::gemma4::prefill;
int main() try {
    CHECK(pf::resolve_attention("") == pf::AttentionMode::canonical);
    CHECK(pf::resolve_attention("auto") == pf::AttentionMode::canonical);
    CHECK(pf::resolve_attention("canonical") == pf::AttentionMode::canonical);
    CHECK(pf::resolve_attention("wmma") == pf::AttentionMode::wmma);
    CHECK(pf::resolve_attention(pf::attention_identity) == pf::AttentionMode::wmma);
    refuses([] { (void)pf::resolve_attention("unknown"); });
    for (unsigned dim : {8U, 16U, 32U, 256U, 512U}) {
        std::vector<float> q(dim), k(64ULL * dim), v(k.size());
        for (unsigned d = 0; d < dim; ++d) q[d] = float(int(d % 7) - 3) * 0.015625F;
        for (unsigned i = 0; i < k.size(); ++i) { k[i] = float(int(i % 31) - 15) * 0.03125F; v[i] = float(int(i % 29) - 14) * 0.0625F; }
        for (std::uint64_t position : {0ULL, 15ULL, 16ULL, 31ULL, 511ULL, 512ULL, 65535ULL}) {
            const auto first = position >= 32 ? position - 31 : 0;
            unsigned calls = 0;
            auto read = [&](std::uint64_t p, bool value) {
                CHECK(p >= first && p <= position); ++calls;
                return (value ? v : k).data() + (p % 64) * dim;
            };
            const auto actual = pf::emulate_attention(q, position, 32, read);
            CHECK(calls > 0 && actual.size() == dim);
            for (float a : actual) CHECK(std::isfinite(a));
            // Constant V and dyadic probabilities: denominator/numerator are
            // exact at this small key count, including boundary masks.
            std::fill(q.begin(), q.end(), 0.0F); std::fill(v.begin(), v.end(), 0.5F);
            const auto constant = pf::emulate_attention(q, position, 32, read);
            for (float a : constant) CHECK(a == 0.5F);
            std::fill(v.begin(), v.end(), -0.5F);
            const auto negative = pf::emulate_attention(q, position, 32, read);
            for (float a : negative) CHECK(a == -0.5F);
        }
        q[0] = 65520.0F;
        refuses([&] { (void)pf::emulate_attention(q, 0, 32, [&](std::uint64_t, bool) { return k.data(); }); });
        q[0] = 0.0F; v[0] = std::numeric_limits<float>::infinity();
        refuses([&] { (void)pf::emulate_attention(q, 0, 32, [&](std::uint64_t, bool value) { return value ? v.data() : k.data(); }); });
    }
    std::puts("WMMA attention identity, masking, signs and overflow PASS");
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
