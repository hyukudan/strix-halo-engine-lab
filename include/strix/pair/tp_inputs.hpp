#pragma once
// Inputs of the tensor-parallel placement (split_plan.hpp) that come from a declaration or a start-up measurement,
// never from a number compiled into a binary:
//   rank_rates - per-rank read rate (GB/s): an explicit override (--tp-rates) wins, else the topology declaration
//                (strix/device/topology.hpp memory_gbytes_ps of every rank); neither -> refused (shares and the
//                placement need a rate per rank).
//   trip_cost  - one gather's cost = the link measured at start-up over the gather form the decode runs
//                (dist/link_cost.hpp measure + agree: both ranks hold the same bits), optionally with an explicit
//                latency override (--tp-trip-us) that replaces only the measured latency.
// Home: shared (host shared, pair facility). Bound by: qwen35 dense HIP exact TP decode (apps/hip_qwen35_decode_main.hip).
// Host test: tests/pair_tp_inputs_tests.cpp.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/dist/link_cost.hpp"
#include "strix/pair/column_split.hpp"

namespace strix::pair::tp {

struct RankRates {
    std::vector<double> gbytes_ps;  // one per rank, > 0
    std::string source;             // "flag" | "topology"
};

// `override_gbps`: the flag's list (empty = not given). `declared_gbps`: the declaration's rate per rank (0 = unknown;
// empty = no declaration loaded).
[[nodiscard]] inline RankRates rank_rates(const std::vector<double>& override_gbps, const std::vector<double>& declared_gbps,
                                          std::uint32_t ranks) {
    const auto valid = [&](const std::vector<double>& v) {
        if (v.size() != ranks) return false;
        for (const double g : v) if (!(g > 0.0)) return false;
        return true;
    };
    if (!override_gbps.empty()) {
        if (!valid(override_gbps)) throw std::invalid_argument("tp rates: --tp-rates needs one positive GB/s per rank");
        return {override_gbps, "flag"};
    }
    if (valid(declared_gbps)) return {declared_gbps, "topology"};
    throw std::invalid_argument("tp rates: no read rate per rank: declare memory_gbytes_ps for every node in the topology "
                                "(--topology FILE --node ID or STRIX_TOPOLOGY) or pass --tp-rates G0,G1");
}

// Trip cost of one gather from the agreed link measurement. override_us > 0 replaces the measured latency only.
[[nodiscard]] inline TripCost trip_cost(const dist::LinkCost& measured, double override_us = 0.0) {
    if (!measured.valid()) throw std::invalid_argument("tp trip cost: the link measurement is not valid");
    if (override_us < 0.0) throw std::invalid_argument("tp trip cost: negative --tp-trip-us");
    return TripCost{(override_us > 0.0 ? override_us : measured.latency_us) / 1000.0, measured.bytes_per_ms()};
}

}  // namespace strix::pair::tp
