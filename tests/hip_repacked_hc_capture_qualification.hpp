#pragma once

// Test-only real-input qualification. Included after the native diagnostic's
// repacked_test fixture definitions. This does not select application kernels.
#include "strix/gguf_reader.hpp"
#include "strix/model_content_hash.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace repacked_test {

std::vector<std::byte> m97_file_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "open qualification input file");
    const std::vector<char> raw((std::istreambuf_iterator<char>(input)), {});
    require(!input.bad(), "qualification input read failed");
    std::vector<std::byte> result(raw.size());
    if (!raw.empty()) std::memcpy(result.data(), raw.data(), raw.size());
    return result;
}

struct M97TensorBytes { std::vector<std::byte> data; std::uint64_t offset = 0; };

M97TensorBytes m97_tensor(const std::filesystem::path& model, const strix::gguf::FileInfo& info,
                         const std::string& name, unsigned type,
                         const std::vector<std::uint64_t>& dimensions) {
    const strix::gguf::TensorInfo* found = nullptr;
    for (const auto& candidate : info.tensors) {
        if (candidate.name == name) {
            require(found == nullptr, "duplicate GGUF tensor name"); found = &candidate;
        }
    }
    require(found != nullptr && found->ggml_type == type && found->dimensions == dimensions,
            "GGUF tensor type/shape mismatch");
    require(found->absolute_offset <= info.file_size && found->byte_size <= info.file_size - found->absolute_offset &&
                found->byte_size <= 3481600ULL &&
                found->absolute_offset <= static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),
            "GGUF tensor window outside bounded source extent");
    M97TensorBytes result;
    result.offset = found->absolute_offset;
    result.data.resize(static_cast<std::size_t>(found->byte_size));
    std::ifstream input(model, std::ios::binary);
    require(input.good(), "open GGUF tensor window");
    input.seekg(static_cast<std::streamoff>(result.offset));
    input.read(reinterpret_cast<char*>(result.data.data()), static_cast<std::streamsize>(result.data.size()));
    require(input.good(), "truncated GGUF tensor window");
    return result;
}

struct M97Library {
    std::string path, sha;
    std::uint64_t size = 0;
    bool operator==(const M97Library&) const = default;
};

std::vector<M97Library> m97_libraries() {
    std::ifstream mappings("/proc/self/maps");
    require(mappings.good(), "read live runtime mappings");
    std::string line;
    std::set<std::string> paths;
    while (std::getline(mappings, line)) {
        const auto start = line.find('/');
        if (start == std::string::npos) continue;
        const auto path = line.substr(start);
        if (path.find("libamdhip64.so") == std::string::npos &&
            path.find("libhsa-runtime64.so") == std::string::npos) continue;
        require(path.find(" (deleted)") == std::string::npos, "deleted live runtime mapping");
        const auto canonical = std::filesystem::canonical(path).string();
        require(canonical.starts_with("/opt/rocm/core-10.1/"), "unexpected live HIP/HSA runtime path");
        paths.insert(canonical);
    }
    require(!mappings.bad() && paths.size() == 2U, "expected exactly two live Core10 HIP/HSA libraries");
    std::vector<M97Library> result;
    for (const auto& path : paths) {
        const auto identity = strix::sha256_file_content(path);
        result.push_back({path, strix::model_content_identity_hex(identity), std::filesystem::file_size(path)});
    }
    return result;
}

struct M97Options {
    std::filesystem::path model, index, output;
    std::string model_sha, index_sha;
    std::vector<Schedule> admitted;
};

bool m97_sha(const std::string& value) {
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

unsigned m97_uint(const std::string& text) {
    require(!text.empty() && text.size() <= 5U &&
                std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "invalid capture integer");
    unsigned result = 0;
    for (char c : text) result = result * 10U + static_cast<unsigned>(c - '0');
    require(result <= 1000U && std::to_string(result) == text, "noncanonical capture integer");
    return result;
}

std::string m97_file_sha(const std::filesystem::path& path) {
    require(std::filesystem::is_regular_file(path) && !std::filesystem::is_symlink(path),
            "expected regular nonsymlink file");
    return strix::model_content_identity_hex(strix::sha256_file_content(path));
}

M97Options m97_options(int argc, char** argv) {
    M97Options options;
    std::set<std::string> seen;
    for (int at = 2; at < argc; at += 2) {
        require(at + 1 < argc, "missing capture option value");
        const std::string key = argv[at], value = argv[at + 1];
        require(seen.insert(key).second, "duplicate capture option");
        if (key == "--model") options.model = value;
        else if (key == "--capture-index") options.index = value;
        else if (key == "--output") options.output = value;
        else if (key == "--model-sha256") options.model_sha = value;
        else if (key == "--capture-index-sha256") options.index_sha = value;
        else if (key == "--schedules") {
            options.admitted = parse_schedules(value);
        } else require(false, "unknown capture option");
    }
    require(!options.model.empty() && !options.index.empty() && !options.output.empty() &&
                m97_sha(options.model_sha) && m97_sha(options.index_sha) && !options.admitted.empty(),
            "need model, index, their SHA256 pins, explicit admitted schedules, and fresh output");
    require(!std::filesystem::exists(options.output) && !std::filesystem::is_symlink(options.output),
            "capture output already occupied");
    return options;
}

std::string m97_key(unsigned position, const std::string& op, unsigned layer) {
    return std::to_string(position) + ':' + op + ':' + std::to_string(layer);
}

struct M97Record {
    unsigned position = 0, layer = 0;
    std::string op, down, inject;
    std::array<std::string, 3> payload_names, payload_shas;
    std::vector<std::byte> normalized, captured;
    std::unique_ptr<Fixture> fixture;
};

std::vector<std::byte> m97_payload(const std::filesystem::path& parent,
                                 const std::string& name, const std::string& sha,
                                 std::size_t size) {
    require(!name.empty() && name != "." && name != ".." &&
                std::all_of(name.begin(), name.end(), [](char c) {
                    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
                }), "unsafe capture basename");
    require(m97_sha(sha), "invalid capture payload SHA256");
    const auto path = parent / name;
    require(std::filesystem::is_regular_file(path) && !std::filesystem::is_symlink(path),
            "capture payload is not regular nonsymlink");
    auto data = m97_file_bytes(path);
    require(data.size() == size && digest(data) == sha, "capture payload extent/hash mismatch");
    return data;
}

// Parse and pin the complete corpus BEFORE allocating a GPU owner. Selection is
// fixed independently of candidate results. Position61 supplies every mixer.
std::vector<M97Record> m97_records(const M97Options& options) {
    require(m97_file_sha(options.index) == options.index_sha, "capture index pin mismatch");
    std::ifstream input(options.index);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)) && line == "strix-hc-capture165-v1",
            "capture index schema mismatch");
    std::set<std::string> expected, seen;
    constexpr std::array<unsigned, 8> selected{0, 1, 2, 3, 23, 24, 46, 47};
    for (unsigned position : {0U, 42U, 43U, 52U, 61U}) {
        for (unsigned layer : selected)
            for (const std::string op : {"attn", "ffn"}) expected.insert(m97_key(position, op, layer));
        expected.insert(m97_key(position, "output", 48));
    }
    for (unsigned layer = 0; layer < 48U; ++layer)
        for (const std::string op : {"attn", "ffn"}) expected.insert(m97_key(61, op, layer));
    require(expected.size() == 165U, "internal fixed selection mismatch");
    std::vector<M97Record> records;
    while (std::getline(input, line)) {
        require(!line.empty() && line.back() != '\r', "noncanonical capture index line");
        std::vector<std::string> columns;
        for (std::size_t start = 0;;) {
            const auto end = line.find('\t', start);
            columns.push_back(line.substr(start, end == std::string::npos ? end : end - start));
            if (end == std::string::npos) break;
            start = end + 1U;
        }
        require(columns.size() == 11U, "capture index column count mismatch");
        M97Record record;
        record.position = m97_uint(columns[0]); record.op = columns[1]; record.layer = m97_uint(columns[2]);
        const auto key = m97_key(record.position, record.op, record.layer);
        require(expected.contains(key) && seen.insert(key).second, "unexpected or duplicate capture record");
        const bool final = record.op == "output";
        const auto prefix = final ? std::string("output_hc") :
                            "blk." + std::to_string(record.layer) + ".hc_" + record.op;
        record.down = columns[9]; record.inject = columns[10];
        require(record.down == prefix + "_down.weight" &&
                    record.inject == (final ? "-" : prefix + "_inject.weight"),
                "capture tensor name/binding mismatch");
        for (std::size_t i = 0; i < 3U; ++i) {
            record.payload_names[i] = columns[3U + i];
            record.payload_shas[i] = columns[6U + i];
        }
        record.normalized = m97_payload(options.index.parent_path(), columns[3], columns[6], 40960U);
        record.captured = m97_payload(options.index.parent_path(), columns[4], columns[7], 1280U);
        if (final) require(columns[5] == "-" && columns[8] == "-", "final mixer has injection payload");
        else {
            auto injection = m97_payload(options.index.parent_path(), columns[5], columns[8], 16U);
            record.captured.insert(record.captured.end(), injection.begin(), injection.end());
        }
        std::vector<float> normalized(10240U);
        std::memcpy(normalized.data(), record.normalized.data(), record.normalized.size());
        require(std::all_of(normalized.begin(), normalized.end(), [](float x) { return std::isfinite(x); }),
                "real capture normalized input is nonfinite");
        records.push_back(std::move(record));
    }
    require(!input.bad() && seen == expected && records.size() == 165U, "incomplete fixed corpus");
    return records;
}

double m97_percentile(std::vector<double> values, double fraction) {
    require(!values.empty(), "empty percentile input");
    std::sort(values.begin(), values.end());
    const double at = fraction * static_cast<double>(values.size() - 1U);
    const auto first = static_cast<std::size_t>(at);
    return values[first] + (values[std::min(first + 1U, values.size() - 1U)] - values[first]) *
                              (at - static_cast<double>(first));
}

struct M97Timing { std::vector<double> baseline, candidate, ratio; };

template<class Sample>
M97Timing m97_pair_protocol(Sample&& sample) {
    // Exactly five discarded pairs, alternating AB/BA. Timed data are 100 ABBA
    // blocks = 200 paired samples, with equal calls and physical owners per arm.
    for (unsigned warm = 0; warm < 5U; ++warm) {
        static_cast<void>(sample((warm & 1U) != 0U));
        static_cast<void>(sample((warm & 1U) == 0U));
    }
    M97Timing result;
    for (unsigned block = 0; block < 100U; ++block) {
        const double a0 = sample(false), b0 = sample(true), b1 = sample(true), a1 = sample(false);
        result.baseline.insert(result.baseline.end(), {a0, a1});
        result.candidate.insert(result.candidate.end(), {b0, b1});
        result.ratio.insert(result.ratio.end(), {a0 / b0, a1 / b1});
    }
    return result;
}

// Every sample verifies final low+injection bits and guards AFTER the stop event.
// Graphs contain only projection launches: no poison, copies or verification.
M97Timing m97_cell(const std::vector<Fixture*>& pool,
                   const std::vector<std::vector<std::byte>>& expected,
                   Schedule schedule, bool graph_mode, unsigned calls) {
    require(!pool.empty() && pool.size() == expected.size(), "invalid timing pool");
    Stream stream;
    Events events;
    Graph baseline_graph, candidate_graph;
    if (graph_mode) {
        for (bool candidate : {false, true}) {
            auto& graph = candidate ? candidate_graph : baseline_graph;
            check(hipStreamBeginCapture(stream.value, hipStreamCaptureModeThreadLocal), "begin real M97 graph");
            try {
                for (unsigned i = 0; i < calls; ++i) {
                    auto* fixture = pool[i % pool.size()];
                    if (candidate) fixture->launch(schedule, stream.value);
                    else fixture->reference(stream.value);
                }
            } catch (...) {
                static_cast<void>(hipStreamEndCapture(stream.value, &graph.graph));
                throw;
            }
            check(hipStreamEndCapture(stream.value, &graph.graph), "end real M97 graph");
            check(hipGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0), "instantiate real M97 graph");
            std::size_t count = 0;
            check(hipGraphGetNodes(graph.graph, nullptr, &count), "count real M97 graph nodes");
            require(count == calls, "M97 graph must contain one kernel per mixer");
        }
    }
    auto sample = [&](bool candidate) {
        for (auto* fixture : pool) fixture->poison(candidate, stream.value);
        check(hipEventRecord(events.start, stream.value), "start real M97 sample");
        if (graph_mode)
            check(hipGraphLaunch(candidate ? candidate_graph.executable : baseline_graph.executable,
                                 stream.value), "launch real M97 graph");
        else {
            for (unsigned i = 0; i < calls; ++i) {
                auto* fixture = pool[i % pool.size()];
                if (candidate) fixture->launch(schedule, stream.value);
                else fixture->reference(stream.value);
            }
        }
        check(hipEventRecord(events.stop, stream.value), "stop real M97 sample");
        check(hipEventSynchronize(events.stop), "drain real M97 sample");
        float milliseconds = 0;
        check(hipEventElapsedTime(&milliseconds, events.start, events.stop), "measure real M97 sample");
        require(std::isfinite(milliseconds) && milliseconds > 0.0F, "invalid real M97 duration");
        for (std::size_t i = 0; i < pool.size(); ++i) {
            require(pool[i]->semantic(candidate) == expected[i], "timed final output differs from captured reference bits");
            pool[i]->guards(candidate);
        }
        return static_cast<double>(milliseconds) * 1000.0 / static_cast<double>(calls);
    };
    auto result = m97_pair_protocol(sample);
    for (auto* fixture : pool) fixture->readonly();
    return result;
}

void m97_timing_json(std::ostream& output, const M97Timing& timing, Schedule schedule,
                     const std::string& mode, unsigned calls, bool attribution) {
    output << "{\"schedule\":" << static_cast<unsigned>(schedule)
           << ",\"mode\":" << quoted(mode) << ",\"calls_per_sample\":" << calls
           << ",\"pairs\":200,\"discarded_warmup_pairs\":5,\"abba_blocks\":100"
              ",\"descriptive_injection_null_attribution\":" << (attribution ? "true" : "false")
           << ",\"baseline_us\":";
    array(output, timing.baseline); output << ",\"candidate_us\":"; array(output, timing.candidate);
    output << ",\"paired_speedup\":"; array(output, timing.ratio);
    output << ",\"median_paired_speedup\":" << median(timing.ratio);
    for (bool candidate : {false, true}) {
        const auto& values = candidate ? timing.candidate : timing.baseline;
        const auto name = candidate ? "candidate" : "baseline";
        output << ",\"" << name << "_median_us\":" << median(values)
               << ",\"" << name << "_p10_us\":" << m97_percentile(values, 0.1)
               << ",\"" << name << "_p90_us\":" << m97_percentile(values, 0.9);
    }
    output << ",\"full_final_outputs_exact\":true,\"guards_intact\":true}";
}

struct M97WeightPin {
    std::string down, inject, down_sha, inject_sha;
    std::uint64_t down_offset = 0, inject_offset = 0;
    std::shared_ptr<WeightOwner> owner;
};

void m97_qualify(const M97Options& options) {
    // External source/compiler admission fixes the schedule subset before this
    // process. No resource-failed schedule is implicitly retried here.
    require(m97_file_sha(options.model) == options.model_sha, "full model SHA256 pin mismatch");
    auto records = m97_records(options);
    const auto model_info = strix::gguf::read_file(options.model);
    const auto exe = std::filesystem::canonical("/proc/self/exe");
    const auto exe_sha = m97_file_sha(exe);
    const auto boot_before = m97_file_bytes("/proc/sys/kernel/random/boot_id");
    int device = -1;
    hipDeviceProp_t properties{};
    check(hipGetDevice(&device), "read M97 device");
    check(hipGetDeviceProperties(&properties, device), "read M97 hardware");
    require(properties.warpSize == 32 && std::string(properties.gcnArchName).starts_with("gfx1151"),
            "M97 requires actual gfx1151 wave32");
    const auto runtime = m97_libraries();

    const Shape shape{10240, 4, 320, 1};
    std::map<std::string, M97WeightPin> weights;
    std::uint64_t forecast = 0;
    for (const auto& record : records) {
        const bool inject = record.inject != "-";
        forecast += fixture_bytes(shape, inject);
        if (!weights.contains(record.down)) {
            M97WeightPin pin;
            pin.down = record.down; pin.inject = record.inject;
            weights.emplace(record.down, std::move(pin));
            forecast += weight_bytes(shape, inject);
        } else require(weights.at(record.down).inject == record.inject, "inconsistent repeated tensor binding");
    }
    require(weights.size() == 97U, "corpus does not cover all97 distinct down owners");
    constexpr std::uint64_t byte_cap = 1ULL << 30U;
    std::size_t free_before = 0, total_before = 0;
    check(hipMemGetInfo(&free_before, &total_before), "read preallocation capacity");
    require(forecast <= byte_cap && forecast <= free_before, "M97 device allocation forecast exceeds cap/free capacity");
    std::uint64_t actual = 0, logical_original = 0;
    for (auto& [name, pin] : weights) {
        auto down = m97_tensor(options.model, model_info, name, 8, {10240, 320});
        require(down.data.size() == 3481600U, "actual down window extent mismatch");
        pin.down_sha = digest(down.data); pin.down_offset = down.offset;
        logical_original += static_cast<std::uint64_t>(down.data.size());
        std::vector<float> injection;
        if (pin.inject != "-") {
            auto raw = m97_tensor(options.model, model_info, pin.inject, 0, {10240, 4});
            require(raw.data.size() == 163840U, "actual inject window extent mismatch");
            pin.inject_sha = digest(raw.data); pin.inject_offset = raw.offset;
            logical_original += static_cast<std::uint64_t>(raw.data.size());
            injection.resize(40960U);
            std::memcpy(injection.data(), raw.data.data(), raw.data.size());
        }
        pin.owner = std::make_shared<WeightOwner>(shape, std::move(down.data), std::move(injection));
        actual += pin.owner->resident_bytes();
    }
    require(logical_original == 353443840ULL, "97 source tensor byte ledger differs from fixed corpus");
    for (auto& record : records) {
        std::vector<float> normalized(10240U);
        std::memcpy(normalized.data(), record.normalized.data(), record.normalized.size());
        record.fixture = std::make_unique<Fixture>(shape, std::move(normalized),
            std::vector<std::byte>{}, std::vector<float>{}, weights.at(record.down).owner);
        actual += record.fixture->resident_bytes();
    }
    require(actual == forecast && actual <= byte_cap, "actual allocation bytes differ from admitted forecast/cap");
    std::size_t free_after = 0, total_after = 0;
    check(hipMemGetInfo(&free_after, &total_after), "read resident capacity");
    require(total_after == total_before, "device capacity changed during setup");

    Stream qualification_stream;
    for (auto& record : records) {
        auto& fixture = *record.fixture;
        fixture.poison(false, qualification_stream.value);
        fixture.reference(qualification_stream.value);
        check(hipStreamSynchronize(qualification_stream.value), "drain corpus reference reproduction");
        require(fixture.semantic(false) == record.captured, "rows8 output differs from frozen captured A bits");
        fixture.guards(false);
        for (auto schedule : options.admitted) {
            fixture.poison(true, qualification_stream.value);
            fixture.launch(schedule, qualification_stream.value);
            check(hipStreamSynchronize(qualification_stream.value), "drain real candidate eager");
            require(fixture.semantic(true) == record.captured, "real candidate eager differs from captured A bits");
            fixture.guards(true);
            Graph graph;
            check(hipStreamBeginCapture(qualification_stream.value, hipStreamCaptureModeThreadLocal),
                  "begin single real qualification graph");
            try { fixture.launch(schedule, qualification_stream.value); }
            catch (...) {
                static_cast<void>(hipStreamEndCapture(qualification_stream.value, &graph.graph));
                throw;
            }
            check(hipStreamEndCapture(qualification_stream.value, &graph.graph), "end single real qualification graph");
            check(hipGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0), "instantiate real qualification graph");
            std::size_t nodes = 0;
            check(hipGraphGetNodes(graph.graph, nullptr, &nodes), "count single qualification graph nodes");
            require(nodes == 1U, "real qualification graph is not one projection kernel");
            for (unsigned replay = 0; replay < 3U; ++replay) {
                fixture.poison(true, qualification_stream.value);
                check(hipGraphLaunch(graph.executable, qualification_stream.value), "replay real qualification graph");
                check(hipStreamSynchronize(qualification_stream.value), "drain real qualification graph");
                require(fixture.semantic(true) == record.captured, "real candidate graph differs from captured A bits");
                fixture.guards(true);
            }
        }
        fixture.readonly();
    }

    std::vector<Fixture*> anchor;
    std::vector<std::vector<std::byte>> anchor_expected;
    auto append_anchor = [&](const std::string& op, unsigned layer) {
        const auto found = std::find_if(records.begin(), records.end(), [&](const auto& record) {
            return record.position == 61U && record.op == op && record.layer == layer;
        });
        require(found != records.end(), "missing real97 anchor mixer");
        anchor.push_back(found->fixture.get()); anchor_expected.push_back(found->captured);
    };
    for (unsigned layer = 0; layer < 48U; ++layer) {
        append_anchor("attn", layer); append_anchor("ffn", layer);
    }
    append_anchor("output", 48);
    require(anchor.size() == 97U, "real97 anchor size mismatch");

    require(!std::filesystem::exists(options.output) && !std::filesystem::is_symlink(options.output),
            "M97 output became occupied");
    std::ofstream output(options.output);
    require(output.good(), "open fresh M97 result");
    output << std::setprecision(17)
           << "{\"schema\":\"hc-exact-repacked-capture165-m97-v3\","
              "\"schedule_family\":\"word-ring-wave-fence-w1-w2-w3-v3\","
              "\"layout\":\"hc-down-q8-j16s2-rpad32-v1\",\"base_old_kernels\":245,"
              "\"capture_cases\":165,\"anchor_position\":61,\"distinct_weight_owners\":97,"
              "\"original97_logical_weight_bytes\":353443840,\"device_byte_cap\":" << byte_cap
           << ",\"forecast_allocation_bytes\":" << forecast << ",\"actual_allocation_bytes\":" << actual
           << ",\"free_before_bytes\":" << free_before << ",\"free_resident_bytes\":" << free_after
           << ",\"device_total_bytes\":" << total_before << ",\"device_index\":" << device
           << ",\"gcn_arch\":" << quoted(properties.gcnArchName)
           << ",\"boot_id_file_sha256\":" << quoted(digest(boot_before))
           << ",\"model_sha256\":" << quoted(options.model_sha)
           << ",\"capture_index_sha256\":" << quoted(options.index_sha)
           << ",\"executable_sha256\":" << quoted(exe_sha)
           << ",\"qualification\":{\"all165_rows8_equal_captured_A_bits\":true,"
              "\"all_admitted_candidates_eager_and_three_graph_replays_equal_A_bits\":true},"
              "\"timing_protocol\":\"five warmup pairs;100ABBA blocks=200pairs;hot16calls;rot97one97mixer sweep\","
              "\"residency\":\"hipMalloc original and packed; common normalized and injection allocations\","
              "\"samples\":[";
    bool first_cell = true;
    double best_speedup = -1.0;
    Schedule best_schedule = options.admitted.front();
    for (auto schedule : options.admitted) {
        for (unsigned mode = 0; mode < 4U; ++mode) {
            const bool rotating = mode >= 2U, graph_mode = (mode & 1U) != 0U;
            const std::vector<Fixture*> pool = rotating ? anchor : std::vector<Fixture*>{anchor.front()};
            const std::vector<std::vector<std::byte>> expected = rotating ? anchor_expected :
                std::vector<std::vector<std::byte>>{anchor_expected.front()};
            const unsigned calls = rotating ? 97U : 16U;
            const auto timing = m97_cell(pool, expected, schedule, graph_mode, calls);
            if (rotating && graph_mode) {
                const double speedup = median(timing.ratio);
                if (speedup > best_speedup) { best_speedup = speedup; best_schedule = schedule; }
            }
            if (!first_cell) output << ',';
            m97_timing_json(output, timing, schedule,
                           mode == 0U ? "eager-hot" : mode == 1U ? "graph-hot" :
                           mode == 2U ? "eager-rot97" : "graph-rot97", calls, false);
            first_cell = false;
            output.flush(); require(output.good(), "write real M97 cell");
        }
    }

    // One descriptive cell only. It does not select a different schedule or
    // relax the primary graph-rot97 screen. The original injection owners stay
    // resident/read-only, and BOTH arms omit injection symmetrically.
    for (auto* fixture : anchor) fixture->set_injection_enabled(false);
    auto down_expected = anchor_expected;
    for (auto& expected : down_expected) expected.resize(1280U);
    const auto attribution = m97_cell(anchor, down_expected, best_schedule, true, 97U);
    output << "],\"attribution\":";
    m97_timing_json(output, attribution, best_schedule, "graph-rot97-injection-null", 97U, true);
    for (auto* fixture : anchor) fixture->set_injection_enabled(fixture->weights->inject.n != 0U);

    // Re-read source windows and every corpus payload, in addition to live
    // device-owner hash/guard checks. These checks are outside all event spans.
    for (auto& record : records) {
        record.fixture->readonly();
        require(m97_payload(options.index.parent_path(), record.payload_names[0], record.payload_shas[0], 40960U) ==
                    record.normalized, "normalized payload changed after timing");
        static_cast<void>(m97_payload(options.index.parent_path(), record.payload_names[1], record.payload_shas[1], 1280U));
        if (record.inject != "-")
            static_cast<void>(m97_payload(options.index.parent_path(), record.payload_names[2], record.payload_shas[2], 16U));
    }
    for (const auto& [name, pin] : weights) {
        require(digest(m97_tensor(options.model, model_info, name, 8, {10240, 320}).data) == pin.down_sha,
                "original down window changed after timing");
        if (pin.inject != "-")
            require(digest(m97_tensor(options.model, model_info, pin.inject, 0, {10240, 4}).data) == pin.inject_sha,
                    "original injection window changed after timing");
    }
    require(m97_file_sha(options.model) == options.model_sha && m97_file_sha(options.index) == options.index_sha &&
                m97_file_sha(exe) == exe_sha && m97_libraries() == runtime &&
                m97_file_bytes("/proc/sys/kernel/random/boot_id") == boot_before,
            "model/index/executable/runtime/boot pins changed during qualification");
    output << ",\"weights\":[";
    bool first_weight = true;
    for (const auto& [name, pin] : weights) {
        output << (first_weight ? "" : ",") << "{\"down_tensor\":" << quoted(name)
               << ",\"down_offset\":" << pin.down_offset << ",\"down_sha256\":" << quoted(pin.down_sha)
               << ",\"inject_tensor\":" << quoted(pin.inject) << ",\"inject_offset\":" << pin.inject_offset
               << ",\"inject_sha256\":" << quoted(pin.inject_sha)
               << ",\"packed_device_sha256\":" << quoted(pin.owner->packed_sha) << '}';
        first_weight = false;
    }
    output << "],\"cases\":[";
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        output << (i ? "," : "") << "{\"position\":" << record.position << ",\"layer\":" << record.layer
               << ",\"op\":" << quoted(record.op) << ",\"down_tensor\":" << quoted(record.down)
               << ",\"normalized_sha256\":" << quoted(record.payload_shas[0])
               << ",\"low_sha256\":" << quoted(record.payload_shas[1])
               << ",\"injection_sha256\":" << quoted(record.payload_shas[2]) << '}';
    }
    output << "],\"loaded_core10_libraries\":[";
    for (std::size_t i = 0; i < runtime.size(); ++i) {
        const auto& library = runtime[i];
        output << (i ? "," : "") << "{\"path\":" << quoted(library.path)
               << ",\"sha256\":" << quoted(library.sha) << ",\"bytes\":" << library.size << '}';
    }
    const bool passed = best_speedup >= 1.5;
    output << "],\"admitted_schedules\":[";
    for (std::size_t i = 0; i < options.admitted.size(); ++i)
        output << (i ? "," : "") << static_cast<unsigned>(options.admitted[i]);
    output << "],\"best_schedule\":" << static_cast<unsigned>(best_schedule)
           << ",\"best_graph_rot97_paired_median\":" << best_speedup
           << ",\"pre_registered_screen_minimum\":1.5,\"M97_screen_passed\":" << (passed ? "true" : "false")
           << ",\"phaseB_authorized\":false,\"model_quality_qualified\":false,"
              "\"default_adopted\":false,\"readonly_inputs_exact\":true,\"finished\":true}\n";
    output.flush(); require(output.good(), "finish real M97 receipt");
}

bool run_capture_cli(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--capture-qualify") return false;
    m97_qualify(m97_options(argc, argv));
    return true;
}

} // namespace repacked_test
