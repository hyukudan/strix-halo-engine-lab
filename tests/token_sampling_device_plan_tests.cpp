// plan_device_sample_enqueues (strix/token_sampling_device.hpp): the groups of one device decide call that share a
// transform run in one kernel sequence, a group is never split, results slots follow the enqueue order, and the
// geometry is refused when a group or the call is too large.
#include "strix/token_sampling_device.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

strix::TokenSamplingPolicy policy(float temperature, float top_p, std::uint64_t seed) {
    strix::TokenSamplingPolicy p;
    p.temperature = temperature; p.top_p = top_p; p.seed = seed;
    return p;
}

template<class F>
bool throws(F f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_same_transform() {
    expect(strix::same_token_transform(policy(0.7F, 1.0F, 1), policy(0.7F, 1.0F, 99)), "the seed is not part of the transform");
    expect(!strix::same_token_transform(policy(0.7F, 1.0F, 1), policy(0.8F, 1.0F, 1)), "temperature is");
    expect(!strix::same_token_transform(policy(0.7F, 1.0F, 1), policy(0.7F, 0.9F, 1)), "top_p is");
    auto k = policy(0.7F, 1.0F, 1); k.top_k = 40;
    auto m = policy(0.7F, 1.0F, 1); m.min_p = 0.05F;
    expect(!strix::same_token_transform(policy(0.7F, 1.0F, 1), k) && !strix::same_token_transform(policy(0.7F, 1.0F, 1), m),
        "top_k and min_p are");
}

void test_host_boundary_oracle() {
    constexpr std::array<float, 2> signed_zeros{-0.0F, 0.0F};
    constexpr strix::TokenSamplingPolicy top_one{1.0F, 1U, 1.0F, 0.0F, 0U};
    const auto tie = strix::make_token_distribution(signed_zeros, {}, top_one);
    expect(tie.sample(0.5) == 0U, "the host transform breaks the signed-zero tie by token ID");

    for (const float top_p : {1.0F, 0.75F}) {
        const strix::TokenSamplingPolicy row_policy{1.0F, 0U, top_p, 0.0F, 0U};
        const std::vector<float> logits(top_p < 1.0F ? 8192U : 6144U, 0.0F);
        const auto host = strix::make_token_distribution(logits, {}, row_policy);
        expect(host.tokens().size() == 6144U, "both host boundary policies keep 6144 tokens");
        for (const double u : {std::nextafter(0.5, 0.0), 0.5, std::nextafter(0.5, 1.0)}) {
            expect(host.sample(u) == 3071U, "host CDF sums individually normalized weights at the boundary");
        }
    }
}

void test_plan() {
    using strix::DeviceSampleGroup;
    using strix::DeviceSampleRow;
    strix::TokenSampler a(policy(1.0F, 1.0F, 7)), b(policy(1.0F, 1.0F, 8)), c(policy(0.7F, 0.95F, 7)), d(policy(1.0F, 1.0F, 9));
    const std::vector<DeviceSampleRow> r1(1), r2(2), r3(3), r5(5), r8(8), r9(9);
    // Four requests with one transform and one row each (four concurrent plain decode rows): one sequence.
    {
        const std::array<DeviceSampleGroup, 4> g{{{&a, r1}, {&b, r1}, {&d, r1}, {&a, r1}}};
        const auto p = strix::plan_device_sample_enqueues(g, 8U);
        expect(p.enqueues.size() == 1U && p.rows == 4U && p.enqueues[0].groups == 4U && p.enqueues[0].offset == 0U &&
                   p.order == std::vector<std::uint32_t>({0U, 1U, 2U, 3U}), "one transform, one sequence");
    }
    // Mixed transforms: grouped by first appearance, input order inside a transform; slots follow the enqueue order.
    {
        const std::array<DeviceSampleGroup, 4> g{{{&a, r2}, {&c, r1}, {&b, r3}, {&c, r2}}};
        const auto p = strix::plan_device_sample_enqueues(g, 8U);
        expect(p.order == std::vector<std::uint32_t>({0U, 2U, 1U, 3U}), "order by transform");
        expect(p.enqueues.size() == 2U && p.enqueues[0].first == 0U && p.enqueues[0].groups == 2U && p.enqueues[0].rows == 5U &&
                   p.enqueues[0].offset == 0U && p.enqueues[1].first == 2U && p.enqueues[1].groups == 2U &&
                   p.enqueues[1].rows == 3U && p.enqueues[1].offset == 5U && p.rows == 8U, "two sequences, contiguous slots");
    }
    // A sequence holds at most max_rows rows and never splits a group.
    {
        const std::array<DeviceSampleGroup, 3> g{{{&a, r5}, {&b, r5}, {&d, r2}}};
        const auto p = strix::plan_device_sample_enqueues(g, 8U);
        expect(p.enqueues.size() == 2U && p.enqueues[0].rows == 5U && p.enqueues[1].rows == 7U && p.enqueues[1].offset == 5U,
            "row capacity starts a new sequence");
    }
    // One group: the single-request call keeps its one sequence.
    {
        const std::array<DeviceSampleGroup, 1> g{{{&c, r8}}};
        const auto p = strix::plan_device_sample_enqueues(g, 8U);
        expect(p.enqueues.size() == 1U && p.rows == 8U, "single group");
    }
    // Refusals.
    {
        const std::array<DeviceSampleGroup, 1> over{{{&a, r9}}};
        const std::array<DeviceSampleGroup, 1> empty{{{&a, {}}}};
        const std::array<DeviceSampleGroup, 1> none{{{nullptr, r1}}};
        std::vector<DeviceSampleGroup> many(9, DeviceSampleGroup{&a, r8});
        expect(throws([&] { (void)strix::plan_device_sample_enqueues(over, 8U); }), "group over max_rows");
        expect(throws([&] { (void)strix::plan_device_sample_enqueues(empty, 8U); }), "empty group");
        expect(throws([&] { (void)strix::plan_device_sample_enqueues(none, 8U); }), "group without sampler");
        expect(throws([&] { (void)strix::plan_device_sample_enqueues(many, 8U); }), "call over device_sample_max_rows");
        expect(throws([&] { (void)strix::plan_device_sample_enqueues(none, 0U); }), "zero rows per sequence");
    }
}

}  // namespace

int main() {
    test_same_transform();
    test_host_boundary_oracle();
    test_plan();
    if (failures != 0) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
    std::cout << "device sampling plan tests passed\n";
    return EXIT_SUCCESS;
}
