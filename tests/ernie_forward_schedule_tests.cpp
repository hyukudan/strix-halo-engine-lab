#include "core/streamed_linear_reference.hpp"
#include "strix/ernie/forward.hpp"

#include <bit>
#include <iostream>

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() try {
    using namespace strix;
    const auto soft = ernie::candidate_activation(ernie::CandidateActivation::soft_clip);
    const auto hard = ernie::candidate_activation(ernie::CandidateActivation::hard_clip);
    check(soft(0, 3, 4, 25) == 0, "zero gate");
    check(soft(std::log(3.0F), -1, 4, 25) < 0, "signed up");
    const float saturation = 4.0F / (1.0F + std::exp(-4.0F));
    check(std::abs(hard(7, -30, 4, 25) - (-25.0F * saturation)) < 1e-5F, "clip both branches");

    core::HostLaneConfig config;
    config.critical_threads = 0;
    config.latency_threads = 0;
    config.background_threads = 4;
    core::HostLane lane(config);
    constexpr std::size_t width = 17, rows = 65;
    std::vector<float> matrix(rows * width), x(width), expected(rows), parallel(rows);
    for (std::size_t i = 0; i < matrix.size(); ++i)
        matrix[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 31.0F;
    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 7.0F;

    auto matches = [&] {
        for (std::size_t row = 0; row < rows; ++row)
            check(std::bit_cast<std::uint32_t>(expected[row]) == std::bit_cast<std::uint32_t>(parallel[row]),
                  "parallel or cached matrix changed bits");
    };
    core::linear_reference(matrix, x, expected);
    core::parallel_linear_reference(lane, 4, matrix, x, parallel);
    matches();
    for (std::size_t first = 0; first < rows; first += 7) {
        const auto count = std::min<std::size_t>(7, rows - first);
        core::linear_reference(std::span(matrix).subspan(first * width, count * width), x,
                               std::span(parallel).subspan(first, count));
    }
    matches();

    core::ReferenceMatrixCache cache(matrix.size() * sizeof(float));
    unsigned loads = 0;
    const auto read = [&](auto, auto values, auto offset) {
        check(offset == 0 && values.size() == matrix.size(), "read shape");
        std::ranges::copy(matrix, values.begin());
        ++loads;
    };
    for (unsigned call = 0; call < 3; ++call) {
        core::parallel_linear_reference(lane, 4, cache.get("matrix", matrix.size(), read), x, parallel);
        matches();
    }
    check(loads == 2, "matrix should be retained on the second load");
    for (unsigned call = 0; call < 3; ++call)
        (void)cache.get("overflow", matrix.size(), read);
    check(loads == 4, "cache failed to retain the newer matrix");
    (void)cache.get("matrix", matrix.size(), read);
    check(loads == 5, "cache retained two matrices in a one-matrix budget");

    std::vector<float> grouped_weights(matrix.begin(), matrix.begin() + 64 * 8);
    std::vector<float> grouped_expected(64), grouped_parallel(64);
    core::grouped_linear_reference(grouped_weights, std::span(x).first(16), grouped_expected, 2);
    core::parallel_linear_reference(lane, 4, grouped_weights, std::span(x).first(16), grouped_parallel, 2);
    check(std::ranges::equal(grouped_expected, grouped_parallel, [](float a, float b) {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    }), "parallel grouped projection changed bits");

    const ernie::ReferenceShape shape{
        .hidden = 4, .vocab = 20, .dense_width = 3, .latent_width = 2, .expert_width = 3,
        .heads = 1, .head_dim = 4, .rope_dim = 2, .q_rank = 2, .o_groups = 1, .o_rank = 2, .vha_rank = 1,
        .route = {.score = moe::RouteScore::sigmoid, .selection_bias = true,
                  .experts = 2, .top_k = 2, .renormalize = true}};
    const ernie::ReferenceRead fixture = [](auto name, auto out, auto offset) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = (name.ends_with("layernorm.weight") || name.ends_with("routed_scaling_factor_param")) ?
                1.0F : static_cast<float>(static_cast<int>((i + offset) % 11) - 5) * 0.02F;
        }
    };
    ernie::CpuReference whole(shape, fixture, soft), separate(shape, fixture, soft);
    constexpr std::uint32_t prompt[] = {1, 2};
    const auto together = whole.append(prompt);
    (void)separate.append(std::span(prompt).first(1));
    const auto serial = separate.append(std::span(prompt).last(1));
    check(std::ranges::equal(together, serial, [](float a, float b) {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    }), "layer-major and token-major sequences differ");
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
