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

void split_formats() {
    std::array<float, 7> weights{}, inputs{}, outputs{}, scales{};
    std::array<rows::Job, 7> jobs{};
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        jobs[i].w.data = &weights[i];
        jobs[i].x.data = &inputs[i];
        jobs[i].y.data = &outputs[i];
    }
    jobs[0].w.format = weight::kBf16WeightStorage;
    jobs[1].w.format = weight::kQ8WeightStorage;
    jobs[2].w = {&weights[2], rows::kNvfp4Records, nullptr, 0.25f};
    jobs[3].w.format = rows::kNvfp4Records;
    jobs[4].w.format = rows::kNvfp4Records;
    jobs[5].w = {&weights[5], rows::kMxfp8Rows, &scales[5]};
    jobs[6].w = {&weights[6], rows::kMxfp8Rows, &scales[6]};
    constexpr std::array<std::array<std::uint32_t, 2>, 7> shapes{{
        {16, 64}, {32, 96}, {16, 128}, {272, 8192}, {32, 512}, {80, 32}, {96, 2048}
    }};
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        jobs[i].out = shapes[i][0];
        jobs[i].in = shapes[i][1];
    }

    strix::device::Profile profile{};
    profile.units = 40;
    const auto plan = rows::plan({jobs, 8}, profile);
    const auto launches = plan.launches();
    check(launches.size() == 5, "format boundaries and useful CTAs close launches");
    check(plan.reaches_wave_target() && plan.target_tiles() == 80 && plan.useful_blocks() == 266,
          "mixed-format useful blocks exclude inactive split-grid slots");
    if (launches.size() != 5) return;

    const auto& fp4 = launches[2];
    check(fp4.form == rows::Form::nvfp4_split_k && fp4.chunk_in == 128 &&
          fp4.tiles_per_block == 16 && fp4.splits[0] == 1 && fp4.splits[1] == 64 &&
          fp4.useful_blocks == 129 && fp4.tiles.count == 2 && fp4.jobs[0].w.outer_scale == 0.25f &&
          fp4.split_identity() == strix::split_k::kExactName,
          "NVFP4 split factor and records follow the incumbent");
    const auto& fp8 = launches[4];
    check(fp8.form == rows::Form::mxfp8_split_k && fp8.chunk_in == 32 &&
          fp8.tiles_per_block == 4 && fp8.splits[0] == 1 && fp8.splits[1] == 64 &&
          fp8.useful_blocks == 130 && fp8.jobs[1].w.scales == &scales[6] &&
          fp8.split_identity() == rows::kMxfp8SplitIdentity,
          "MXFP8 split factor and two-plane binding");
    const auto selected = plan.select_launch(1);
    check(selected.rows() == plan.rows() && selected.launches().size() == 1 &&
          selected.launches()[0].jobs[0].w.has_format(weight::kQ8WeightStorage),
          "row-tile binding receives the selected launch");

    for (unsigned i = 0; i < 2; ++i) {
        auto wide_tile = jobs[i];
        wide_tile.out = 80 * rows::kTileOut;
        check(rows::plan({std::span(&wide_tile, 1), 8}, profile).reaches_wave_target(),
              "BF16 and Q8_0 can fill the wave target using output tiles alone");
    }
    const std::span<const rows::Job> wide(jobs.data() + 3, 1);
    const auto single = rows::plan({wide, 32}, profile);
    check(single.launches()[0].tiles_per_block == 8 && single.launches()[0].splits[0] == fp4.splits[1],
          "NVFP4 verify geometry keeps the split order");
    const std::span<const rows::Job> tiny(jobs.data() + 2, 1);
    check(!rows::plan({tiny, 1}, profile).reaches_wave_target(), "one indivisible CTA falls short of target");
    jobs[2].in = 96;
    expect_invalid({tiny, 1}, profile);
    jobs[5].w.scales = nullptr;
    expect_invalid({std::span<const rows::Job>(jobs.data() + 5, 1), 1}, profile);
}

}  // namespace

int main() {
    split_formats();
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
