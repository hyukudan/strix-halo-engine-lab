#include "strix/deepseek41_pair_profile_table.hpp"
#include "strix/glm_pair_profile_table.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/profiles/qwen.hpp"
#include "strix/profiles/qwen_dense_hip.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <string_view>

namespace {
namespace p = strix::profile;

struct Binding {
    std::string_view model, engine, source;
    std::span<const p::LadderRow> rows;
    std::span<const p::Lever> levers;
    std::string_view execution;
};

// Read the same tables the engines resolve. Incumbent grants do not describe a timed workload.
constexpr Binding bindings[] = {
    {"deepseek-v4.1-flash", "deepseek41-pair", "include/strix/deepseek41_pair_profile_table.hpp",
     strix::deepseek41::pair_defaults::ladder_rows, strix::deepseek41::pair_defaults::profile_table,
     "HIP pair, known priced prefill; D1 over EXACT, R512 over D1 without replay"},
    {"glm-5.3-flash", "glm-pair", "include/strix/glm_pair_profile_table.hpp",
     strix::glm::rank_defaults::product_ladder_rows, strix::glm::rank_defaults::product_profile_table,
     "HIP USB4 pair ready, admitted 8192-row chunk; full or routed scratch"},
    {"inkling2-mini", "inkling2-mini-hip", "include/strix/profiles/inkling2_mini.hpp",
     strix::inkling2::kMiniLadder, strix::inkling2::kMiniLevers,
     "HIP one node, known mini_ladder_execution; rotated IU4 over the FAST BF16 tile"},
    {"qwen3.8-flash-next", "qwen-hip", "include/strix/qwen_profile_table.hpp",
     strix::qwen::profile::ladder_rows, strix::qwen::profile::table,
     "HIP one lane, known priced prompt execution; FAST plus cap9 over FAST"},
    {"qwen3.8-27b", "qwen35-hip-decode", "include/strix/profiles/qwen_dense_hip.hpp",
     strix::qwen35::hip_profile::kBandRows, strix::qwen35::hip_profile::table,
     "HIP whole-model prefill (one rank or each tensor-parallel rank; decode and spec apps), IU4 g64 pack, one lane, known priced execution; timing with MTP off"},
};

std::string lengths(const auto& points, bool json) {
    std::ostringstream out;
    if (json) out << '[';
    bool first = true;
    for (const auto point : points) {
        if (point == 0) continue;
        if (!first) out << ", ";
        out << point;
        first = false;
    }
    if (json) out << ']';
    else if (first) out << "Unspecified";
    return out.str();
}

bool render(std::ostream& out, bool json) {
    if (json) out << "{\n  \"schema\": \"strix-band-edges/v1\",\n  \"rows\": [";
    else out << "# Prompt band timing inventory\n\n"
             << "Generated from the ladder tables by `strix-band-inventory`. Token ranges are inclusive. "
             << "A band whose lower edge has no timing is listed as EDGE-TIMING PENDING: it keeps its nominal band "
             << "until device edge timings report and its row sets `narrow_to_floor` (then it starts at its first "
             << "winning timing point, shown as the measured floor). "
             << "The lower-edge tolerance is " << p::kBandEdgeTolerance << " tokens for prompt counts.\n\n"
             << "Historical per-request keys are marked RECAPTURE PENDING. "
             << "A recaptured band binds the recorded execution, reference lever chain and candidate identity. "
             << "Recorded keys appear in full in the [JSON inventory](band-edges.json). "
             << "An unspecified gate length means the source does not give a prompt length.\n\n"
             << "| Model / engine | Tier | Nominal band | Effective band | Edge state | Key / candidate | Timing tokens | Gate tokens | Receipts |\n"
             << "|---|---|---:|---:|---|---|---|---|---|\n";
    bool first = true;
    for (const auto& binding : bindings) for (const auto& row : binding.rows) {
        if (row.incumbent) continue;
        const auto floor = p::ladder_min_context(row);
        std::uint64_t measured_floor = 0;
        for (const auto tokens : row.band_evidence.timed)
            if (tokens != 0 && tokens >= row.min_context && tokens <= row.max_context &&
                (measured_floor == 0 || tokens < measured_floor))
                measured_floor = tokens;
        const std::string_view edge_state = !row.band_evidence.lower_edge_unproven ? "proven"
                                            : row.band_evidence.narrow_to_floor     ? "narrowed"
                                                                                    : "EDGE-TIMING PENDING";
        if (row.receipt.empty() || floor == 0 ||
            (row.band_bound && row.band_evidence.lower_edge_unproven == p::band_edge_timed(row)) ||
            (!row.band_bound && !p::band_edge_timed(row))) {
            std::cerr << binding.engine << '/' << row.lever << " has no timing at its effective lower edge\n";
            return false;
        }
        const auto lever = std::ranges::find(binding.levers, row.lever, &p::Lever::name);
        if (lever == binding.levers.end()) return false;
        const auto identity = p::declared_identity(lever->identity, row.value);
        const auto tier = p::name(row.tier);
        const auto key = row.band_bound && row.over.empty()
            ? std::string_view{"bind_band_receipts/LadderArms::over"} : row.over;
        const auto capture = !row.promotion.empty() ? "manifest" :
            row.band_bound && row.over.empty() ? "pending" : "fixed-legacy";
        if (json) {
            if (!first) out << ',';
            const auto field = [&](std::string_view name, std::string_view value) {
                out << "      " << std::quoted(name) << ": " << std::quoted(value) << ",\n";
            };
            out << "\n    {\n";
            field("model", binding.model);
            field("engine", binding.engine);
            field("source", binding.source);
            field("tier", tier);
            field("reference_tier", row.tier == p::Name::fast ? "exact" : "fast");
            field("phase", row.phase == p::Phase::prefill ? "prefill" : "decode");
            field("device", row.device);
            field("lever", row.lever);
            field("value", row.value);
            field("implementation", identity);
            field("execution_key", key);
            field("capture", capture);
            if (!row.promotion.empty()) field("promotion", row.promotion);
            field("execution_scope", binding.execution);
            field("timing_receipt", row.receipt);
            field("gate_receipt", row.band_evidence.gate_scope);
            out << "      \"band\": [" << row.min_context << ", " << row.max_context << "],\n"
                << "      \"lanes\": [" << row.min_lanes << ", " << row.max_lanes << "],\n"
                << "      \"effective_lower_edge\": " << floor << ",\n"
                << "      \"band_bound\": " << std::boolalpha << row.band_bound << ",\n"
                << "      \"lower_edge_unproven\": " << row.band_evidence.lower_edge_unproven << ",\n"
                << "      \"edge_state\": " << std::quoted(edge_state) << ",\n"
                << "      \"measured_floor\": " << measured_floor << ",\n"
                << "      \"speedup\": " << row.speedup << ",\n"
                << "      \"noise\": " << row.noise << ",\n"
                << "      \"timed_tokens\": " << lengths(row.band_evidence.timed, true) << ",\n"
                << "      \"gated_tokens\": " << lengths(row.band_evidence.gated, true) << "\n    }";
        } else {
            out << "| " << binding.model << " / [" << binding.engine << "](../" << binding.source << ") | "
                << tier << " | " << row.min_context << "–" << row.max_context << " | " << floor << "–"
                << row.max_context << " | " << edge_state;
            if (row.band_evidence.lower_edge_unproven) out << " (measured floor " << measured_floor << ')';
            const auto status = !row.promotion.empty() ? "Manifest" :
                row.band_bound && row.over.empty() ? "RECAPTURE PENDING" : "Fixed";
            out << " | " << status << ", `" << row.lever
                << '=' << row.value << '{' << identity << "}` | " << lengths(row.band_evidence.timed, false)
                << " | " << lengths(row.band_evidence.gated, false) << " | " << row.receipt << ' '
                << row.band_evidence.gate_scope << " |\n";
        }
        first = false;
    }
    if (json) out << "\n  ]\n}\n";
    else out << "\nThis inventory covers the tables above. Other model bindings, incumbent grants and standalone "
             << "schedules are tracked separately.\n";
    return !first;
}

bool current(const char* path, bool json) {
    std::ifstream file(path);
    std::ostringstream checked_in, generated;
    checked_in << file.rdbuf();
    if (file && render(generated, json) && checked_in.str() == generated.str()) return true;
    std::cerr << path << ": inventory missing, stale or invalid\n";
    return false;
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "json") return render(std::cout, true) ? 0 : 1;
    if (argc == 2 && std::string_view(argv[1]) == "markdown") return render(std::cout, false) ? 0 : 1;
    if (argc == 4 && std::string_view(argv[1]) == "check")
        return current(argv[2], true) && current(argv[3], false) ? 0 : 1;
    std::cerr << "usage: strix-band-inventory json | markdown | check JSON MARKDOWN\n";
    return 2;
}
