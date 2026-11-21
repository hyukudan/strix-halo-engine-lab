#include "hy4/ref/hy4_host_ref.hpp"
#include "intel/core/reference.hpp"
#include "strix/core/chunked_dot.hpp"

#include <array>
#include <bit>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace {
using namespace strix;

void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}

using RowAccess = std::tuple<std::string, std::uint64_t>;

class RecordedWeights final : public hy4::Weights {
public:
    explicit RecordedWeights(hy4::Weights& weights) : weights_(weights) {}
    const std::vector<float>& tensor(const std::string& name) override {
        require(name.find("_exps.weight") == std::string::npos, "expert tensor must be streamed by row");
        return weights_.tensor(name);
    }
    std::vector<float> row(const std::string& name, std::uint64_t index) override {
        if (name.find("_exps.weight") != std::string::npos) accesses.emplace_back(name, index);
        return weights_.row(name, index);
    }
    std::vector<RowAccess> accesses;

private:
    hy4::Weights& weights_;
};

class HostOperators final : public core::RoutedReference {
public:
    std::vector<std::vector<std::uint32_t>> routes;
    std::size_t projections = 0;

    void linear(core::F32Rows matrix, std::span<const float> input, std::span<float> output) override {
        require(intel::linear_batch(matrix, input.size(), output.size()) == 1U, "model projection must be one row");
        ++projections;
        for (std::size_t r = 0; r < matrix.rows; ++r)
            output[r] = core::chunked_dot_of([&](std::size_t i) { return matrix.values[r * matrix.width + i]; },
                                            [&](std::size_t i) { return input[i]; }, matrix.width);
    }

    void route(const moe::RoutePolicy& policy, std::span<const float> scores, std::span<const float> bias,
               std::span<std::uint32_t> ids, std::span<float> weights) override {
        require(policy.score == moe::RouteScore::sigmoid && policy.selection_bias &&
                policy.order == moe::RouteOrder::ascending_id && policy.key == moe::RouteKey::ordered_bits,
                "model route policy");
        moe::route_select_ref(policy, scores, bias, ids, weights);
        routes.emplace_back(ids.begin(), ids.end());
    }
};

void route_layout() {
    moe::RoutePolicy policy;
    policy.selection_bias = true;
    policy.key = moe::RouteKey::ordered_bits;
    policy.order = moe::RouteOrder::ascending_id;
    policy.experts = 4;
    policy.top_k = 2;
    policy.scale = 2.0F;
    const std::array scores{1.0F, 1.0F, 0.5F, 0.5F};
    const std::array bias{0.0F, 0.0F, 2.0F, 0.0F};
    std::array<std::uint32_t, 2> ids{};
    std::array<float, 2> weights{};
    std::array<std::uint8_t, 4> scratch{1, 1, 1, 1};
    moe::route_select_row(policy, scores.data(), bias.data(), ids.data(), weights.data(), scratch.data());
    require(ids == std::array<std::uint32_t, 2>{0, 2}, "tie selects lower id and output is ascending");
    require(weights == std::array{(1.0F / 1.5F) * 2.0F, (0.5F / 1.5F) * 2.0F},
            "weights normalize unbiased scores");
}

void model_binding() {
    const auto config = hy4::Config::tiny();
    auto weights = hy4::synthetic_weights(config, 7);
    RecordedWeights model_weights(*weights);
    HostOperators operators;
    hy4::HostRef baseline(config, *weights);
    hy4::HostRef bound(config, model_weights, &operators);
    for (const std::uint32_t token : {1U, 7U, 9U}) {
        operators.routes.clear();
        model_weights.accesses.clear();
        const auto& expected = baseline.step(token);
        const auto& actual = bound.step(token);
        require(expected.size() == actual.size() && baseline.next_token() == bound.next_token(), "logit shape or token");
        for (std::size_t i = 0; i < actual.size(); ++i)
            require(std::bit_cast<std::uint32_t>(expected[i]) == std::bit_cast<std::uint32_t>(actual[i]), "logit bits");
        require(operators.routes.size() == 2U, "only the two MoE blocks invoke the router");
        std::vector<RowAccess> expected_rows;
        for (std::size_t layer = 1; layer < config.layers; ++layer) {
            for (const auto expert : operators.routes[layer - 1U]) {
                for (const std::string_view projection : {"gate", "up", "down"}) {
                    const auto rows = projection == "down" ? config.latent : config.expert_ff;
                    const auto name = "blk." + std::to_string(layer) + ".ffn_" + std::string(projection) + "_exps.weight";
                    for (std::uint32_t row = 0; row < rows; ++row)
                        expected_rows.emplace_back(name, std::uint64_t{expert} * rows + row);
                }
            }
        }
        require(model_weights.accesses == expected_rows, "gate/up/down must read only selected expert slices in route order");
    }
    require(operators.projections > 3U * config.top_k * 2U * 3U, "shared dense executor covers regular projections too");
}
} // namespace

int main() {
    try {
        route_layout();
        model_binding();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Intel Hy4 host binding: %s\n", error.what());
        return 1;
    }
}
