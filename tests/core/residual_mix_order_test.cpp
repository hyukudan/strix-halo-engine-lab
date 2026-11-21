#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kLanes = 32;
constexpr unsigned kPartials = kThreads / kLanes;

template <class Fma>
float block_order(unsigned width, Fma accumulate) {
    float partial[kThreads]{};
    for (unsigned thread = 0; thread < kThreads; ++thread) {
        for (unsigned i = thread; i < width; i += kThreads) {
            partial[thread] = accumulate(i, partial[thread]);
        }
    }
    for (unsigned h = kThreads / 2; h > 0; h /= 2) {
        for (unsigned thread = 0; thread < h; ++thread) {
            partial[thread] = partial[thread] + partial[thread + h];
        }
    }
    return partial[0];
}

template <class Fma>
float warp_order(unsigned width, Fma accumulate) {
    float lane_sum[kLanes]{};
    for (unsigned lane = 0; lane < kLanes; ++lane) {
        float v[kPartials]{};
        for (unsigned base = lane; base < width; base += kThreads) {
            for (unsigned k = 0; k < kPartials; ++k) {
                const unsigned i = base + kLanes * k;
                if (i < width) v[k] = accumulate(i, v[k]);
            }
        }
        for (unsigned k = 0; k < 4; ++k) v[k] = v[k] + v[k + 4];
        for (unsigned k = 0; k < 2; ++k) v[k] = v[k] + v[k + 2];
        lane_sum[lane] = v[0] + v[1];
    }
    for (unsigned h = 16; h > 0; h /= 2) {
        for (unsigned lane = 0; lane < h; ++lane) {
            lane_sum[lane] = lane_sum[lane] + lane_sum[lane + h];
        }
    }
    return lane_sum[0];
}

bool check_bits(float block, float warp, const char* operation, unsigned width, unsigned row) {
    if (std::memcmp(&block, &warp, sizeof(float)) == 0) return true;
    std::cerr << operation << " differs at width " << width << ", row " << row << '\n';
    return false;
}

}  // namespace

int main() {
    std::mt19937 rng(0x452b68a1U);
    std::uniform_real_distribution<float> sample(-4.0F, 4.0F);
    for (unsigned width : {2304U, 2048U, 4096U, 96U, 300U}) {
        std::vector<float> source(width);
        std::vector<float> learned(width);
        for (unsigned row = 0; row < 1000; ++row) {
            for (unsigned i = 0; i < width; ++i) {
                source[i] = sample(rng);
                learned[i] = sample(rng);
            }

            const auto squares = [&](unsigned i, float sum) { return std::fma(source[i], source[i], sum); };
            const float block_squares = block_order(width, squares);
            const float warp_squares = warp_order(width, squares);
            if (!check_bits(block_squares, warp_squares, "sum of squares", width, row)) return 1;

            const auto dot = [&](unsigned i, float sum) { return std::fma(learned[i], source[i], sum); };
            if (!check_bits(block_order(width, dot), warp_order(width, dot), "normalized dot", width, row)) {
                return 1;
            }

            const float scale = 1.0F / std::sqrt(block_squares / static_cast<float>(width) + 1e-5F);
            const auto scaled_dot = [&](unsigned i, float sum) {
                return std::fma(learned[i], source[i] * scale, sum);
            };
            if (!check_bits(block_order(width, scaled_dot), warp_order(width, scaled_dot), "raw dot", width, row)) {
                return 1;
            }
        }
    }
    return 0;
}
