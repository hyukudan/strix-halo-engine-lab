#include "core/rows_plan.hpp"
#include "strix/weight/registry.hpp"

#include <bit>
#include <cstdio>

namespace rows = strix::core::rows;
namespace weight = strix::weight;

namespace {

void check(bool condition, const char* description) {
    if (!condition) throw std::runtime_error(description);
}

template<class F>
void rejects(F action) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected invalid_argument");
}

constexpr std::array formats{
    rows::RoleFormat{weight::Role::attention_q, rows::kMxfp8Rows},
    rows::RoleFormat{weight::Role::attention_k, rows::kMxfp8Rows},
    rows::RoleFormat{weight::Role::head, weight::kBf16WeightStorage},
    rows::RoleFormat{weight::Role::dense_up, rows::kNvfp4Records},
};

rows::RoleJob make_job(weight::Role role, weight::FormatId format, unsigned out, unsigned in) {
    static std::array<std::array<float, 4>, static_cast<std::size_t>(weight::Role::auxiliary) + 1> storage{};
    const auto& bound = storage[static_cast<std::size_t>(role)];
    return {role, {{&bound[0], format, &bound[1]}, out, in, {&bound[2]}, {&bound[3]}, 0, {}}};
}

void occupancy() {
    strix::device::Profile device{};
    device.units = 3;
    auto q = make_job(weight::Role::attention_q, rows::kMxfp8Rows, 128, 7 * 32);
    auto k = make_job(weight::Role::attention_k, rows::kMxfp8Rows, 16, 5 * 32);
    q.policy = k.policy = rows::SplitPolicy::occupancy; // the opt-in identity; the default is the incumbent tree
    const auto head = make_job(weight::Role::head, weight::kBf16WeightStorage, 32, 128);
    const auto nvfp4 = make_job(weight::Role::dense_up, rows::kNvfp4Records, 128, 4 * 128);
    const std::array jobs{q, k, head, nvfp4};
    const auto plan = rows::plan_rows(jobs, formats, 1, device, {32});
    check(plan.target_blocks() == 12 && plan.rows() == 1 && plan.launches().size() == 3, "four waves and format groups");
    const auto& first = plan.launches()[0];
    check(first.launch.tiles.count == 2 && first.launch.tiles.total() == 9 && first.launch.useful_blocks == 17 &&
          first.reaches_wave_target(plan.target_blocks()) && first.roles[1] == weight::Role::attention_k,
          "heterogeneous split counts count only useful blocks in a homogeneous launch");
    check(first.trees[0].splits == 6 && first.trees[1].splits == 5, "rounded-up target and K limit");
    check(plan.launches()[1].launch.splits[0] == 1 && !plan.launches()[1].reaches_wave_target(plan.target_blocks()),
          "unsplit incumbent shortfall");
    const std::array incumbent_jobs{nvfp4.job};
    const auto incumbent = rows::plan({incumbent_jobs, 1}, device);
    check(plan.launches()[2].trees[0].splits == incumbent.launches()[0].splits[0] &&
          plan.launches()[2].trees[0].arithmetic == strix::split_k::kExactName, "incumbent NVFP4 tree");

    const auto solo = rows::plan_rows({&q, 1}, formats, 32, device, {32});
    check(solo.launches()[0].trees[0] == first.trees[0], "group and row invariance");
    const auto capped = rows::plan_rows({&q, 1}, formats, 1, device, {2});
    check(capped.launches()[0].trees[0].splits == 2 && capped.launches()[0].launch.useful_blocks == 4 &&
          !capped.launches()[0].reaches_wave_target(capped.target_blocks()), "backend grid limit and shortfall");
    device.units = 0xffffffffU;
    const auto wide_target = rows::plan_rows({&q, 1}, formats, 1, device, {32});
    check(wide_target.target_blocks() == 4ULL * device.units && wide_target.launches()[0].trees[0].splits == 7,
          "nonwrapping wave count");
    check(wide_target.launches()[0].trees[0].arithmetic == rows::kMxfp8OccupancySplitIdentity &&
          rows::kMxfp8OccupancySplitIdentity != rows::kMxfp8SplitIdentity, "occupancy tree has its own identity");
    q.splits = 8;
    rejects([&] { (void)rows::plan_rows({&q, 1}, formats, 1, device, {32}); });
    q.splits = 2;
    rejects([&] { (void)rows::plan_rows({&q, 1}, formats, 1, device, {1}); });
    std::puts("PASS B2 rows occupancy");
}

void incumbent_default() {
    // A binding that does not ask for occupancy gets the incumbent rows.hpp count for every shape and device size.
    for (const unsigned units : {1U, 3U, 16U, 40U, 0xffffffffU}) {
        strix::device::Profile device{};
        device.units = units;
        for (const unsigned out : {16U, 48U, 64U, 128U, 256U, 4096U}) {
            for (const unsigned chunks : {1U, 2U, 5U, 7U, 8U, 33U, 128U}) {
                const auto q = make_job(weight::Role::attention_q, rows::kMxfp8Rows, out, chunks * 32);
                const std::array one{q.job};
                const auto incumbent = rows::plan({one, 1}, device).launches()[0].splits[0];
                const auto plan = rows::plan_rows({&q, 1}, formats, 1, device, {65535});
                const auto& tree = plan.launches()[0].trees[0];
                check(tree.splits == incumbent && plan.launches()[0].launch.splits[0] == incumbent &&
                      tree.arithmetic == rows::kMxfp8SplitIdentity, "default plan is the incumbent choice");
            }
        }
    }
    // 128 outputs x 224 inputs on one unit: incumbent seven splits; the occupancy tree (two splits) is another
    // identity with other bits on cancelling terms.
    strix::device::Profile device{};
    device.units = 1;
    auto q = make_job(weight::Role::attention_q, rows::kMxfp8Rows, 128, 7 * 32);
    std::array<float, 7 * 32> x{};
    std::array<std::uint8_t, 7 * 32> codes{};
    std::array<std::uint8_t, 7> scales{};
    constexpr std::array values{0x1p26f, 0.0f, 0.0f, -0x1p26f, 1.0f, 0.0f, 0.0f};
    codes.fill(0x40);
    scales.fill(126);
    for (std::size_t c = 0; c < values.size(); ++c) x[c * 32] = values[c];
    const auto bits = [](float v) { return std::bit_cast<std::uint32_t>(v); };
    const auto def = rows::plan_rows({&q, 1}, formats, 1, device, {32}).launches()[0].trees[0];
    check(def.splits == 7, "incumbent keeps seven splits");
    check(bits(rows::mxfp8_rows_reference(def, codes, scales, x)) == bits(1.0f), "incumbent cancellation bits");
    q.policy = rows::SplitPolicy::occupancy;
    const auto occ = rows::plan_rows({&q, 1}, formats, 1, device, {32}).launches()[0].trees[0];
    check(occ.splits == 2 && occ.arithmetic == rows::kMxfp8OccupancySplitIdentity, "occupancy opt-in selects two");
    check(occ.range(0).end == 3 && occ.range(1).end == 7, "occupancy chunk ranges");
    check(bits(rows::mxfp8_rows_reference(occ, codes, scales, x)) == bits(0.0f), "occupancy cancellation bits");
    std::puts("PASS B2 rows incumbent default");
}

void split_reference() {
    auto q = make_job(weight::Role::attention_q, rows::kMxfp8Rows, 128, 7 * 32);
    strix::device::Profile device{};
    device.units = 1;
    const std::array incumbent_jobs{q.job};
    const auto incumbent = rows::plan({incumbent_jobs, 1}, device);
    const auto incumbent_splits = incumbent.launches()[0].splits[0];
    std::array<float, 7 * 32> x{};
    std::array<std::uint8_t, 7 * 32> codes{};
    std::array<std::uint8_t, 7> scales{};
    constexpr std::array values{0x1p26f, 1.0f, -0x1p26f, 1.0f, 1.0f, 1.0f, 1.0f};
    codes.fill(0x40);
    scales.fill(126);
    for (std::size_t c = 0; c < values.size(); ++c) x[c * 32] = values[c];
    for (const auto splits : {1U, 3U, incumbent_splits}) {
        q.splits = splits;
        const auto plan = rows::plan_rows({&q, 1}, formats, 1, device, {32});
        const auto& tree = plan.launches()[0].trees[0];
        check(tree.splits == splits && tree.arithmetic == rows::kMxfp8SplitIdentity, "pinned split identity");
        if (splits == 3) {
            const auto a = tree.range(0), b = tree.range(1), c = tree.range(2);
            check(a.begin == 0 && a.end == 2 && b.begin == 2 && b.end == 4 && c.begin == 4 && c.end == 7,
                  "uneven split boundaries");
        }
        const auto expected = splits == 3 ? 3.0f : 4.0f;
        const auto actual = rows::mxfp8_rows_reference(tree, codes, scales, x);
        check(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(expected), "split reference bits");
        rejects([&] { (void)rows::mxfp8_rows_reference(tree, codes, std::span{scales}.first(6), x); });
    }
    std::puts("PASS B2 rows split reference");
}

void admission() {
    auto q = make_job(weight::Role::attention_q, rows::kMxfp8Rows, 16, 32);
    strix::device::Profile device{};
    device.units = 1;
    rejects([&] { (void)rows::plan_rows({&q, 1}, {}, 1, device, {32}); });
    const std::array duplicates{formats[0], formats[0]};
    rejects([&] { (void)rows::plan_rows({&q, 1}, duplicates, 1, device, {32}); });
    q.job.w.format.revision = 2;
    rejects([&] { (void)rows::plan_rows({&q, 1}, formats, 1, device, {32}); });

    alignas(8) std::array<std::int8_t, 128> codes{};
    const std::array<float, 4> deltas{};
    const std::array<std::int32_t, 4> sums{};
    rows::A8ReadView view{codes.data(), deltas.data(), sums.data(), 2, 32, 64};
    const auto admit = [&] { rows::validate_a8_consumer(view, 2, 32, weight::kQ8WeightStorage, weight::Exactness::fast); };
    admit();
    view.sums = nullptr;
    admit();
    rejects([&] { rows::validate_a8_consumer(view, 2, 32, weight::kQ8WeightStorage, weight::Exactness::legacy_bitwise); });
    rejects([&] { rows::validate_a8_consumer(view, 2, 32, weight::kBf16WeightStorage, weight::Exactness::fast); });
    rejects([&] { rows::validate_a8_consumer(view, 1, 32, weight::kQ8WeightStorage, weight::Exactness::fast); });
    view.stride = 40;
    rejects(admit);
    view.stride = 64;
    ++view.codes;
    rejects(admit);
    view.codes = codes.data();
    view.deltas = nullptr;
    rejects(admit);
    std::puts("PASS B2 rows role and A8 admission");
}

}  // namespace

int main() {
    try {
        occupancy();
        incumbent_default();
        split_reference();
        admission();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL B2 rows: %s\n", error.what());
        return 1;
    }
    return 0;
}
