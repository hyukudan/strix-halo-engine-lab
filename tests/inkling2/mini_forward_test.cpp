#include "inkling2/inkling2_mini_forward.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>

namespace {

using namespace strix::inkling2;

struct Fixture {
    std::array<unsigned, kLogicalVisits> stages{};
    std::vector<float> pass_entry_signs;
    unsigned controller_reads = 0;

    std::vector<float> parameter(std::string_view name, unsigned elements) {
        if (name.ends_with("_weight")) ++controller_reads;
        return std::vector<float>(elements, name.ends_with("norm.weight") ? 1.0F : 0.0F);
    }

    std::vector<float> attention(LayerVisit visit, unsigned position, std::span<const float> q,
                                 std::span<const float> k, std::span<const float> v) {
        if (position != 7 || q.size() != kHiddenWidth || k.size() != q.size() || v.size() != q.size()) {
            throw std::runtime_error("invalid visit input");
        }
        if (stages[visit.logical_layer]++ != 0) throw std::runtime_error("attention out of order");
        if (visit.physical_layer == kCoreFirst) pass_entry_signs.push_back(q.front());
        // The prelude leaves a positive residual. Each repeat independently
        // resets to that prelude even though the core leaves a negative one.
        return std::vector<float>(kHiddenWidth, is_core(visit.physical_layer) ? -3.0F : 1.0F);
    }

    std::vector<float> moe(LayerVisit visit, std::span<const float> input) {
        if (input.size() != kHiddenWidth || stages[visit.logical_layer]++ != 2) {
            throw std::runtime_error("MoE out of order");
        }
        return std::vector<float>(kHiddenWidth);
    }

    std::vector<float> convolve(LayerVisit visit, bool after_moe, std::span<const float> input) {
        if (stages[visit.logical_layer]++ != (after_moe ? 3U : 1U)) {
            throw std::runtime_error("convolution out of order");
        }
        return {input.begin(), input.end()};
    }
};

}  // namespace

int main() {
    try {
        Fixture fixture;
        const std::vector<float> embedding(kHiddenWidth, 1.0F);
        const auto result = mini_forward_row(fixture, 7, embedding);
        for (const auto stages : fixture.stages) if (stages != 4) return 1;
        if (fixture.pass_entry_signs != std::vector<float>(kCorePasses, 1.0F)) return 2;
        if (fixture.controller_reads != 2U * 12U * 4U * 3U) return 3;
        const float expected = 1.0F;
        for (const auto value : result) if (value != expected) return 4;
        return 0;
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 5;
    }
}
