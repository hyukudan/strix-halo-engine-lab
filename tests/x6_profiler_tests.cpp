#include "strix/exec/profiler.hpp"
#include "strix/serve/json.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace strix::exec;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void save(const std::filesystem::path& path, const strix::serve::Json& json) {
    std::ofstream file(path, std::ios::binary);
    file << json.dump() << '\n';
    file.close();
    require(file.good(), "could not write receipt fixture");
}

}  // namespace

int main(int argc, char** argv) try {
    require(argc == 2, "expected receipt directory");
    const auto directory = std::filesystem::path(argv[1]);
    std::filesystem::create_directories(directory);
    Program program{};
    program.identity = 0xbeef;
    program.predicted = {1000, 4000, 3000, 50, 3};
    program.stages = {{"norm", 2, strix::graph::Format::f32}, {"copy", -1, strix::graph::Format::f32}};
    program.launches.resize(3);
    for (std::size_t i = 0; i < 2; ++i) {
        program.launches[i].entry = "norm.rms";
        program.launches[i].identity = "norm-v1";
        program.launches[i].stages.emplace_back();
    }
    program.launches.back().entry = "exchange";
    program.launches.back().stages.emplace_back().instance = 1;
    KernelEntry entry{};
    entry.key = {strix::graph::Op::norm, "norm.rms", {"plain.f32", 1}, OperandClass::f32_fma, RowsClass::r1};
    entry.identity = "norm-v1";
    const std::array work{LaunchWork{1000, 1000, 10000000, &entry, 1}, LaunchWork{3000, 1000, 0, &entry},
                          LaunchWork{0, 1000, 0, nullptr, 0, true}};
    const ProfileOptions options{work, 1, 4};
    const auto stub = make_step_profile(program, options);
    require(stub.repeats == 0 && !stub.wall_ms && !stub.launches[0].ms, "CPU stub reported measurement");
    require(stub.bytes_per_step == 7000 && stub.exchanges == 1, "stub plan totals");

    strix::receipt::Receipt base;
    base.identity.engine = "x6-cpu-fixture";
    const auto stub_json = strix::serve::parse_json_unique(write_step_profile(stub, base));
    save(directory / "cpu-stub.json", *stub_json.find("receipt"));

    const std::array intervals{1.0, 2.0, 3.0};
    const auto profiled = make_step_profile(program, options, {intervals, 4.0, 0.5, 7, 8.0});
    require(profiled.busy_ms == 6 && profiled.exchange_ms == 3 && profiled.predicted_ms == 1 &&
            profiled.cold_wall_ms == 8 && profiled.repeats == 4, "step timing totals");
    require(profiled.launches[0].byte_roof_pct == 0.2 && profiled.launches[0].compute_roof_pct == 1 &&
            !profiled.launches[1].compute_roof_pct, "roof units or unobserved peak");
    require(write_step_table(profiled).starts_with("launch gpu_ms family entry stages\n2 3 exchange exchange copy\n"),
            "profile table must put the largest interval first");
    const auto projected = profile_receipt(profiled, base);
    const auto& rows = *projected.step;
    require(rows.size() == 3 && rows[0].program == "000000000000beef" && rows[0].launches == 3 &&
            rows[0].wall_ms == 4 && rows[0].device_status == 7, "R1 program row");
    require(rows[2].program.starts_with("000000000000beef/entry/norm-v1/{") && rows[2].wall_ms == 3 &&
            rows[2].launches == 2 && rows[2].bytes_read == 6000, "R1 D4 group");
    auto other_key = entry;
    other_key.key.weight = {"plain.bf16", 1};
    auto split_work = work;
    split_work[1].entry = &other_key;
    const auto split = profile_receipt(make_step_profile(program, {split_work}), base);
    require(split.step->size() == 4 && split.step->at(2).program != split.step->at(3).program &&
            split.step->at(2).launches == 1 && split.step->at(3).launches == 1,
            "one catalogue identity may have multiple D4 keys");
    const auto timed_json = strix::serve::parse_json_unique(write_step_profile(profiled, base));
    require(timed_json.find("gap_ms")->is_null() && timed_json.find("eager_minus_production_ms")->as_double() == 2,
            "eager and graph intervals must remain distinguishable");
    save(directory / "timed.json", *timed_json.find("receipt"));

    auto overflow = work;
    overflow[0].weight_bytes = std::numeric_limits<std::uint64_t>::max();
    try {
        (void)make_step_profile(program, {overflow});
        throw std::runtime_error("accepted overflowing plan bytes");
    } catch (const std::overflow_error&) {
    }
    try {
        (void)make_step_profile(program, options, {std::span(intervals).first(1), 4, 0, 0, {}});
        throw std::runtime_error("accepted incomplete timing trace");
    } catch (const std::invalid_argument&) {
    }
    std::cout << "x6_profiler_host: PASS accounting, timing and R1 projection\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
