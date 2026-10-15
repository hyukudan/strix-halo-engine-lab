#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
bool better(float a, std::uint32_t ai, float b, std::uint32_t bi) { return a > b || (a == b && ai < bi); }
struct Result { std::vector<std::uint32_t> indices; std::vector<float> weights; };
// Literal legacy insertion; partial_sort is deliberately excluded for NaNs.
Result insertion(const std::vector<float>& p, std::uint32_t k, bool normalize) {
    const auto count = static_cast<std::uint32_t>(p.size());
    Result r{std::vector<std::uint32_t>(k, count), std::vector<float>(k, -INFINITY)};
    for (std::uint32_t e = 0; e < count; ++e) for (std::uint32_t slot = 0; slot < k; ++slot) {
        if (!better(p[e], e, r.weights[slot], r.indices[slot])) continue;
        for (std::uint32_t move = k - 1U; move > slot; --move) {
            r.weights[move] = r.weights[move - 1U]; r.indices[move] = r.indices[move - 1U];
        }
        r.weights[slot] = p[e]; r.indices[slot] = e; break;
    }
    float sum = 0.0F; for (std::uint32_t slot = 0; slot < k; ++slot) sum += r.weights[slot];
    if (!normalize) sum = 1.0F;
    for (std::uint32_t slot = 0; slot < k; ++slot) r.weights[slot] /= sum;
    return r;
}
Result ranks(const std::vector<float>& p, std::uint32_t k, bool normalize) {
    const auto count = static_cast<std::uint32_t>(p.size());
    Result r{std::vector<std::uint32_t>(k, count), std::vector<float>(k, -INFINITY)};
    std::vector<unsigned> writes(k, 0);
    for (std::uint32_t lane = 0; lane < 256U; ++lane) for (std::uint32_t e = lane; e < count; e += 256U) {
        if (p[e] != p[e]) continue;
        std::uint32_t rank = 0;
        for (std::uint32_t t = 0; t < count; ++t) { if (better(p[t], t, p[e], e)) ++rank; if (rank >= k) break; }
        if (rank < k) { ++writes[rank]; r.weights[rank] = p[e]; r.indices[rank] = e; }
    }
    const auto valid = static_cast<std::uint32_t>(std::count_if(p.begin(), p.end(), [](float x) { return !std::isnan(x); }));
    for (std::uint32_t slot = 0; slot < k; ++slot)
        if (writes[slot] != (slot < std::min(k, valid) ? 1U : 0U)) throw std::runtime_error("rank collision or gap");
    float sum = 0.0F; for (std::uint32_t slot = 0; slot < k; ++slot) sum += r.weights[slot];
    if (!normalize) sum = 1.0F;
    for (std::uint32_t slot = 0; slot < k; ++slot) r.weights[slot] /= sum;
    return r;
}
}
int main() try {
    std::uint32_t state = 0x1743a251U; std::size_t cases = 0;
    for (const auto count : {1U, 8U, 16U, 31U, 255U, 256U, 257U, 511U, 512U}) for (unsigned pattern = 0; pattern < 10; ++pattern) {
        std::vector<float> p(count);
        for (std::uint32_t e = 0; e < count; ++e) {
            state ^= state << 13U; state ^= state >> 17U; state ^= state << 5U;
            switch (pattern) {
            case 0: p[e] = 0.25F; break;
            case 1: p[e] = e % 2U ? -0.0F : 0.0F; break;
            case 2: p[e] = std::numeric_limits<float>::quiet_NaN(); break;
            case 3: p[e] = e == 0 ? 1.0F : std::numeric_limits<float>::quiet_NaN(); break;
            case 4: p[e] = e % 3U == 0 ? std::numeric_limits<float>::quiet_NaN() : -INFINITY; break;
            case 5: p[e] = e % 2U ? INFINITY : -INFINITY; break;
            case 6: p[e] = static_cast<float>(e); break;
            case 7: p[e] = static_cast<float>(count - e); break;
            case 8: p[e] = std::bit_cast<float>((e % 2U ? 0x80000000U : 0U) | (e + 1U)); break;
            default: p[e] = static_cast<float>(static_cast<int>(state % 13U) - 6); break;
            }
        }
        for (std::uint32_t k = 1; k <= std::min(count, 16U); ++k) for (const bool normalize : {false, true}) {
            const auto old = insertion(p, k, normalize), candidate = ranks(p, k, normalize);
            if (old.indices != candidate.indices) throw std::runtime_error("rank differs from insertion indices");
            for (std::uint32_t slot = 0; slot < k; ++slot) {
                if (std::isnan(old.weights[slot]) && std::isnan(candidate.weights[slot])) continue;
                if (std::bit_cast<std::uint32_t>(old.weights[slot]) != std::bit_cast<std::uint32_t>(candidate.weights[slot]))
                    throw std::runtime_error("rank differs from insertion weight bits");
            }
            ++cases;
        }
    }
    std::cout << cases << " independent insertion/rank cases passed\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
