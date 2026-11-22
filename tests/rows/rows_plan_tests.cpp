#include "strix/core/rows.hpp"
#include "strix/weight/registry.hpp"

#include <array>
#include <cstdio>
#include <limits>

namespace rows = strix::core::rows;
namespace weight = strix::weight;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
}

void expect_invalid(const rows::Group& group, const strix::device::Profile& device) {
    try {
        (void)rows::plan(group, device);
        check(false, "expected invalid_argument");
    } catch (const std::invalid_argument&) {
    }
}

}  // namespace

int main() {
    std::array<rows::Job, 10> jobs{};
    std::array<float, jobs.size()> weights{}, inputs{}, outputs{}, residuals{}, extra{};
    constexpr std::array<std::uint32_t, jobs.size()> units{1, 3, 5, 2, 4, 1, 2, 3, 4, 1};
    for (unsigned j = 0; j < jobs.size(); ++j) {
        jobs[j] = {{&weights[j], weight::kBf16WeightStorage}, units[j] * 16, 32 * (j + 1),
                   {&inputs[j]}, {&outputs[j]}, static_cast<rows::ChainId>(j), {{&residuals[j]}, {&extra[j]}}};
    }

    strix::device::Profile profile{};
    profile.units = 3;
    const auto wave_plan = rows::plan({jobs, 32, 7}, profile);
    check(wave_plan.rows() == 32 && wave_plan.prologue() == 7 && wave_plan.target_tiles() == 6, "plan metadata");
    constexpr std::array<std::uint32_t, 4> totals{9, 6, 6, 5};
    const auto launches = wave_plan.launches();
    check(launches.size() == totals.size(), "wave target splits the ready jobs");
    unsigned next = 0;
    for (unsigned i = 0; i < launches.size(); ++i) {
        const auto& launch = launches[i];
        const auto& tiles = launch.tiles;
        check(i < totals.size() && tiles.total() == totals[i], "launch tile count");
        std::uint32_t start = 0;
        for (unsigned j = 0; j < tiles.count; ++j, ++next) {
            const auto& item = launch.jobs[j];
            const auto& source = jobs[next];
            check(tiles.first[j] == start && tiles.first[j + 1] == start + units[next], "tile prefix");
            for (std::uint32_t k = 0; k < units[next]; ++k) {
                unsigned member = 0;
                std::uint32_t local = 0;
                strix::decode::locate(tiles.first, tiles.count, start + k, member, local);
                check(member == j && local == k, "lookup at every tile");
                check(launch.jobs[member].x.data == source.x.data && launch.jobs[member].in == source.in,
                      "lookup selects the job's input and stride");
            }
            check(item.w.data == source.w.data && item.y.data == source.y.data && item.out == source.out &&
                  item.epilogue == source.epilogue && item.aux[0].data == source.aux[0].data &&
                  item.aux[1].data == source.aux[1].data, "binding is copied into the plan");
            start += units[next];
        }
    }
    check(next == jobs.size(), "each job appears once, in order");

    profile.units = std::numeric_limits<std::uint32_t>::max();
    const auto capped = rows::plan({jobs, 1}, profile);
    check(capped.target_tiles() == 2ULL * profile.units, "wave target does not wrap");
    check(capped.launches().size() == 2 && capped.launches()[0].tiles.count == 8 &&
          capped.launches()[1].tiles.count == 2, "eight-job cap and remainder");

    jobs[2].w.format = weight::kQ8WeightStorage;
    jobs[3].w.format = weight::kQ8WeightStorage;
    jobs[4].w.format = {weight::kQ8WeightStorage.name, 2};
    const auto formats = rows::plan({jobs, 4}, profile);
    check(formats.launches().size() == 4 && formats.launches()[0].tiles.count == 2 &&
          formats.launches()[1].tiles.count == 2 && formats.launches()[2].tiles.count == 1 &&
          formats.launches()[3].tiles.count == 5, "decoder names and revisions bound launch runs");
    check(capped.launches()[0].jobs[2].w.has_format(weight::kBf16WeightStorage), "plan owns its job table");

    const std::span<const rows::Job> one(jobs.data(), 1);
    const auto isolated = rows::plan({one, 8}, profile);
    check(isolated.launches().size() == 1 && isolated.launches()[0].tiles.total() == 1, "isolated tile stays unsplit");
    expect_invalid({{}, 1}, profile);
    expect_invalid({one, 0}, profile);
    expect_invalid({one, 33}, profile);
    expect_invalid({one, 1}, {});
    const auto original = jobs[0];
    jobs[0].in = 0;
    expect_invalid({one, 1}, profile);
    jobs[0].in = 64 - 1;
    expect_invalid({one, 1}, profile);
    jobs[0] = original;
    jobs[0].out = 0;
    expect_invalid({one, 1}, profile);
    jobs[0].out = 15;
    expect_invalid({one, 1}, profile);
    jobs[0] = original;
    jobs[0].x = {};
    expect_invalid({one, 1}, profile);
    jobs[0] = original;
    jobs[0].y = {};
    expect_invalid({one, 1}, profile);
    jobs[0] = original;
    jobs[0].w = {};
    expect_invalid({one, 1}, profile);

    if (failures != 0) return 1;
    std::puts("PASS rows plan");
    return 0;
}
