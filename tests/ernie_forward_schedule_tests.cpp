#include "core/streamed_linear_reference.hpp"
#include "cpu/core/dense.hpp"
#include "strix/ernie/forward.hpp"
#include "strix/weight/formats/microscaled.hpp"

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
    struct ActivationCase { float gate, up, expected; };
    constexpr ActivationCase cases[] = {
        {3.75F, 24.50F, 54.0102246F},
        {4.25F, 25.50F, 59.7074687F},
        {-4.25F, 25.50F, -0.8516813F},
        {4.25F, -25.50F, -59.7074687F},
        {8.00F, 50.00F, 92.9037518F},
    };
    for (const auto& c : cases)
        check(std::abs(soft(c.gate, c.up, 4, 25) - c.expected) < 2e-5F, "smooth SiTU at both caps");
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
    const auto ordered = cpu::ordered_fma_kernel(true);
    ordered.matvec(matrix.data(), x.data(), width, parallel.data(), rows);
    matches();
    float interleaved[width * 8], dot8[8];
    for (std::size_t col = 0; col < width; ++col)
        for (std::size_t row = 0; row < 8; ++row)
            interleaved[col * 8 + row] = x[col] * static_cast<float>(row + 1);
    ordered.dot8(matrix.data(), interleaved, width, dot8);
    for (std::size_t row = 0; row < 8; ++row) {
        std::vector<float> input(width);
        for (std::size_t col = 0; col < width; ++col) input[col] = interleaved[col * 8 + row];
        check(std::bit_cast<std::uint32_t>(dot8[row]) ==
              std::bit_cast<std::uint32_t>(core::dot_f32(std::span(matrix).first(width), input)),
              "SIMD teacher-forced dot changed bits");
    }
    const std::uint8_t codes[] = {0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7,
                                  0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87};
    for (unsigned e = 0; e < 253; ++e) {
        const std::uint8_t scales[] = {static_cast<std::uint8_t>(e)};
        float regular[32], table[32], swapped[32];
        weight::microscaled::decode_mxfp4_row(codes, scales, regular);
        weight::microscaled::decode_mxfp4_row_table(codes, scales, table);
        weight::microscaled::decode_mxfp4_row_table(codes, scales, swapped, true);
        for (std::size_t i = 0; i < std::size(regular); ++i) {
            check(std::bit_cast<std::uint32_t>(regular[i]) == std::bit_cast<std::uint32_t>(table[i]),
                  "table decoder changed MXFP4 bits");
            check(std::bit_cast<std::uint32_t>(regular[i ^ 1U]) == std::bit_cast<std::uint32_t>(swapped[i]),
                  "high-nibble probe did not swap adjacent coefficients");
        }
    }
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
        .heads = 2, .head_dim = 4, .rope_dim = 2, .q_rank = 2, .o_groups = 1, .o_rank = 2, .vha_rank = 1,
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
    auto states = whole.embed_rows(prompt);
    ernie::ReferenceRows saved;
    const auto end = whole.prefill(std::move(states), 0, ernie::kLayerCount,
        [&](auto boundary, const auto& rows) { if (boundary == 7) saved = rows; });
    const auto resumed = whole.prefill(std::move(saved), 7, ernie::kLayerCount);
    check(resumed == end, "cached boundary changed the result of causal prefill");
    ernie::CpuReference threaded(shape, fixture, soft, {}, {}, [&](auto count, const auto& work) {
        core::reference_row_ranges(lane, 4, count, work);
    });
    check(threaded.prefill(threaded.embed_rows(prompt), 0, ernie::kLayerCount) == end,
          "attention head scheduler changed the prefill");
    std::vector<float> head(shape.hidden * shape.vocab), scores(shape.vocab);
    fixture("lm_head.weight", std::span(head), 0);
    core::linear_reference(head, whole.head_input(end.back()), scores);
    check(scores == together, "teacher-forced head changed the serial logits");
    const float fixture_logits[] = {1000, 1000, 999};
    const auto score = ernie::score_next_token(fixture_logits, 2);
    check(score.best_id == 0 && std::abs(score.nll - (1 + std::log(2 + std::exp(-1.0)))) < 1e-12,
          "stable loss or deterministic tie break");
    bool rejected = false;
    try {
        const float bad[] = {0, std::numeric_limits<float>::infinity()};
        (void)ernie::score_next_token(bad, 0);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "nonfinite logits accepted");
    (void)separate.append(std::span(prompt).first(1));
    const auto serial = separate.append(std::span(prompt).last(1));
    check(std::ranges::equal(together, serial, [](float a, float b) {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    }), "layer-major and token-major sequences differ");
    const auto bounded_read = [&](auto name, auto out, auto offset) {
        check(name.find(".compressor.") == name.npos && name.find(".indexer.") == name.npos,
              "bounded attention read an unreachable branch");
        fixture(name, out, offset);
    };
    ernie::CpuReference bounded(shape, bounded_read, soft, {}, {}, {}, std::size(prompt));
    (void)bounded.append(std::span(prompt).first(1));
    check(bounded.append(std::span(prompt).last(1)) == serial, "bounded causal append changed logits");
    rejected = false;
    try {
        (void)bounded.append(std::span(prompt).last(1));
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "bounded attention accepted an unreachable position");
    auto checkpoint = shape;
    checkpoint.contracts.router = ernie::RouterContract::summed_scores;
    checkpoint.contracts.vha = ernie::VhaContract::swapped;
    checkpoint.contracts.norm_routed_output = true;
    checkpoint.contracts.checkpoint_attention = true;
    checkpoint.contracts.checkpoint_hc = true;
    checkpoint.contracts.checkpoint_stores = true;
    ernie::CpuReference checkpoint_batch(checkpoint, fixture, soft);
    const auto batch_logits = checkpoint_batch.append(prompt);
    ernie::CpuReference checkpoint_step(checkpoint, bounded_read, soft, {}, {}, {}, std::size(prompt));
    (void)checkpoint_step.append(std::span(prompt).first(1));
    const auto step_logits = checkpoint_step.append(std::span(prompt).last(1));
    check(std::ranges::equal(batch_logits, step_logits, [](float a, float b) {
        return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
    }), "checkpoint prefill and incremental decode changed logits");
    const auto checkpoint_rows = checkpoint_batch.prefill(checkpoint_batch.embed_rows(prompt), 0, ernie::kLayerCount);
    const ernie::CpuReference reverse_heads(checkpoint, fixture, soft, {}, {}, [](auto count, const auto& work) {
        for (auto end = count; end != 0; --end) work(end - 1, end);
    });
    check(reverse_heads.prefill(reverse_heads.embed_rows(prompt), 0, ernie::kLayerCount) == checkpoint_rows,
          "independent head scheduling changed checkpoint attention bits");
    core::linear_reference(head, checkpoint_batch.head_input(checkpoint_rows.back()), scores);
    check(scores == batch_logits, "checkpoint teacher-forced and greedy head differ");

    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
