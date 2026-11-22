#include "k1bind_cases.hpp"
#include "strix/core/rows.hpp"
#include "strix/weight/registry.hpp"

#include <cstdio>
#include <stdexcept>

namespace rows = strix::core::rows;

namespace {

void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}

void inkling(const k1bind_test::Case& test, const strix::device::Profile& profile, unsigned row_count) {
    std::array<float, 3> inputs{};
    std::array<float, rows::kMaxJobs> weights{}, outputs{};
    std::array<rows::Job, rows::kMaxJobs> jobs{};
    unsigned count = 0;
    for (const auto& group : test.groups) {
        for (const auto& projection : group.projections) {
            if (projection.out == 24) continue;  // The sink remains on its FP32 route.
            jobs[count] = {{&weights[count], strix::weight::kBf16WeightStorage}, projection.out, group.width,
                           {&inputs[group.input]}, {&outputs[count]}, rows::ChainId(projection.round_bf16 ? 2U : 0U), {}};
            ++count;
        }
    }
    const auto members = std::span(jobs).first(count);
    const auto plan = rows::plan({members, row_count, 0, rows::Packing::fewest_launches}, profile);
    require(plan.launches().size() == 1 && plan.rows() == row_count && plan.target_tiles() == 376,
            "Inkling launch count, rows or target");
    const auto& launch = plan.launches().front();
    require(launch.form == rows::Form::row_tile && launch.tiles_per_block == 1 && launch.split_identity().empty() &&
                launch.useful_blocks == test.tile_ends.back() && launch.tiles.count == test.tile_ends.size() &&
                plan.reaches_wave_target() == (test.tile_ends.back() >= 376U),
            "Inkling tile geometry or split identity");
    require(launch.tiles.first[0] == 0 && launch.tiles.count == count, "Inkling job count");
    for (unsigned j = 0; j < count; ++j) {
        const auto& job = launch.jobs[j];
        require(launch.tiles.first[j + 1] == test.tile_ends[j] && launch.splits[j] == 1 &&
                    job.w.data == jobs[j].w.data && job.x.data == jobs[j].x.data && job.y.data == jobs[j].y.data &&
                    job.out == jobs[j].out && job.in == jobs[j].in && job.epilogue == jobs[j].epilogue,
                "Inkling tile prefix or job order");
    }
    if (count > 4) {
        const auto incumbent_packing = rows::plan({members, row_count}, profile);
        require(incumbent_packing.launches().size() == 2 && incumbent_packing.launches()[0].tiles.total() == 408,
                "Default rows wave batching remains distinct from the new packing");
    }
}

void qwen_like(const strix::device::Profile& profile) {
    std::array<float, 3> weights{}, outputs{};
    float input{};
    constexpr std::array outs{6144U, 10240U, 12288U};
    const std::array expected_blocks{24U, 40U, 48U};
    const std::array expected_splits{40U, 25U, 21U};
    std::array<rows::Job, outs.size()> jobs{};
    for (unsigned j = 0; j < jobs.size(); ++j)
        jobs[j] = {{&weights[j], rows::kNvfp4Records}, outs[j], 5120, {&input}, {&outputs[j]}, 0, {}};
    const auto plan = rows::plan({jobs, 8}, profile);
    require(plan.launches().size() == jobs.size(), "Qwen-like launch count");
    for (unsigned j = 0; j < jobs.size(); ++j) {
        const auto& launch = plan.launches()[j];
        require(launch.form == rows::Form::nvfp4_split_k && launch.split_identity() == strix::split_k::kExactName &&
                    launch.chunk_in == 128 && launch.tiles_per_block == 16 && launch.tiles.count == 1 &&
                    launch.tiles.total() == outs[j] / rows::kTileOut && launch.splits[0] == expected_splits[j] &&
                    launch.useful_blocks == expected_blocks[j] * expected_splits[j] &&
                    launch.jobs[0].w.data == &weights[j] && launch.jobs[0].y.data == &outputs[j] &&
                    launch.jobs[0].x.data == &input,
                "Qwen-like NVFP4 launch identity or order");
    }
}

}  // namespace

void check_projection_boundary();

int main() {
    strix::device::Profile profile{};
    profile.units = 188;
    try {
        check_projection_boundary();
        for (const auto& test : k1bind_test::cases) {
            for (unsigned row_count = 1; row_count <= 8; ++row_count) inkling(test, profile, row_count);
            std::printf("k1bind host %s PASS tiles=%u\n", test.name, test.tile_ends.back());
        }
        qwen_like(profile);
        std::puts("k1bind host qwen-like PASS");
    } catch (const std::exception& err) {
        std::fprintf(stderr, "k1bind host FAIL: %s\n", err.what());
        return 1;
    }
}
