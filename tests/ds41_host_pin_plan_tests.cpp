#include "strix/ds41_host_pin_plan.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

using namespace strix::ds41;

namespace {
void check(bool ok) {
    if (!ok) throw std::runtime_error("host pin plan check failed");
}
void expect(const std::vector<std::uint64_t>& bytes, std::uint32_t slots, double budget,
            const std::vector<std::uint8_t>& wanted) {
    const auto host = deadline_host_layers(bytes, slots, budget);
    check(host == wanted);
    double used = 0;
    for (std::size_t l = 0; l < host.size(); ++l)
        if (host[l]) used += static_cast<double>(bytes[l]);
    check(used <= budget);
}
}  // namespace

int main() {
    check(parse_host_pin_order("layer") == HostPinOrder::layer);
    check(parse_host_pin_order("deadline") == HostPinOrder::deadline);
    check(std::string_view(host_pin_order_name(HostPinOrder::layer)) == "layer");
    check(std::string_view(host_pin_order_name(HostPinOrder::deadline)) == "deadline");
    check(parse_host_pin_order("early") == HostPinOrder::early);
    check(std::string_view(host_pin_order_name(HostPinOrder::early)) == "early");
    // early: skip the S preloaded streamed layers, then ascending, every layer that still fits
    check(early_host_layers(std::vector<std::uint64_t>{8, 8, 8, 8, 8}, 2, 16) == std::vector<std::uint8_t>{0, 0, 1, 1, 0});
    check(early_host_layers(std::vector<std::uint64_t>{0, 8, 8, 8, 2, 0, 1}, 2, 11) == std::vector<std::uint8_t>{0, 0, 0, 1, 1, 0, 1});
    check(early_host_layers(std::vector<std::uint64_t>{8, 8, 8, 8, 2}, 2, 2) == std::vector<std::uint8_t>{0, 0, 0, 0, 1});
    bool rejected = false;
    try { (void)parse_host_pin_order("latest"); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected);
    expect({}, 2, 100, {});
    expect({0, 0, 0}, 2, 100, {0, 0, 0});
    expect({8, 8, 8, 8, 8}, 2, 0, {0, 0, 0, 0, 0});
    expect({8, 8, 8, 8, 8}, 2, 16, {0, 0, 0, 1, 1});
    expect({8, 8, 8, 8, 8}, 2, 15.999, {0, 0, 0, 0, 1});
    // Device-resident gaps / a nonzero first layer don't consume preload slots.
    expect({0, 8, 0, 8, 8, 0, 8}, 2, 100, {0, 0, 0, 0, 1, 0, 1});
    // The loader clamps to at least one slot, including its one-slot fallback.
    expect({8, 8, 8}, 1, 100, {0, 1, 1});
    expect({8, 8, 8}, 0, 100, {0, 1, 1});
    expect({8, 8}, 8, 100, {0, 0});
    constexpr std::uint64_t gib = 1ULL << 30;
    expect({4 * gib, 4 * gib, 4 * gib, 4 * gib, gib}, 2, 5.0 * double(gib), {0, 0, 0, 1, 1});
    expect({4 * gib, 4 * gib, 4 * gib, 4 * gib, gib}, 2, 5.0 * double(gib) - 1, {0, 0, 0, 0, 1});
    // Small streamed decoder parts belong to the suffix before full encoders.
    expect({8, 8, 8, 8, 8, 3, 2, 1}, 2, 22, {0, 0, 0, 1, 1, 1, 1, 1});
    expect({8, 8, 8, 8, 8, 3, 2, 1}, 2, 14, {0, 0, 0, 0, 1, 1, 1, 1});
    // Don't fill spare budget with an earlier host burst before a later file.
    expect({8, 8, 1, 8, 2}, 2, 3, {0, 0, 0, 0, 1});
    expect({8, 8, 1, 8, 2}, 2, 1, {0, 0, 0, 0, 0});
    // Exhaust all budgets for a heterogeneous synthetic schedule: budget bounds,
    // preload exclusion, contiguous suffix and monotonic placement must all hold.
    const std::vector<std::uint64_t> bytes{0, 16, 16, 16, 0, 16, 7, 4, 1};
    std::vector<std::uint8_t> previous(bytes.size(), 0);
    for (unsigned budget = 0; budget <= 100; ++budget) {
        const auto host = deadline_host_layers(bytes, 2, budget);
        unsigned used = 0;
        bool suffix = false;
        for (std::size_t l = 0; l < bytes.size(); ++l) {
            check(!previous[l] || host[l]);
            if (!bytes[l]) { check(!host[l]); continue; }
            if (host[l]) { suffix = true; used += static_cast<unsigned>(bytes[l]); }
            else check(!suffix);
        }
        check(!host[1] && !host[2] && used <= budget);
        previous = host;
    }
    // interleave: the name round-trips; budget, residents and the idle-preload layers are respected for any budget/ring
    check(parse_host_pin_order("interleave") == HostPinOrder::interleave);
    check(std::string(host_pin_order_name(HostPinOrder::interleave)) == "interleave");
    for (unsigned budget = 0; budget <= 100; budget += 5)
        for (double ring : {1.0, 10.0, 40.0}) {
            const auto host = interleave_host_layers(bytes, 2, budget, ring, 1.0, 0.25);
            unsigned used = 0;
            for (std::size_t l = 0; l < bytes.size(); ++l) {
                if (!bytes[l]) check(!host[l]);
                if (host[l]) used += static_cast<unsigned>(bytes[l]);
            }
            check(used <= budget);
            check(host[1] == (budget >= 16U));  // the first preload slot is pinned first when it fits
            check(ordered_host_layers(HostPinOrder::interleave, bytes, 2, budget, ring) ==
                  interleave_host_layers(bytes, 2, budget, ring));
        }
    {
        // the measured RTX shape: 20 full layers (3.82 GB), 20 mostly resident ones, 2 preload slots, 61.5 GB of host
        // budget and a 4 GiB ring. The layer order leaves every file byte after the host block; interleave must put
        // file layers among the full host layers and still spend the budget (no extra disk bytes).
        std::vector<std::uint64_t> rtx(40, 0);
        for (std::size_t l = 0; l < 20; ++l) rtx[l] = 3'822'000'000ULL;
        for (std::size_t l = 20; l < 40; ++l) rtx[l] = (39 - l) * 30'000'000ULL;
        const double budget = 61.5e9, ring = 4.0 * double(1ULL << 30);
        const auto host = interleave_host_layers(rtx, 2, budget, ring);
        double used = 0.0, streamed = 0.0;
        bool file_before_host = false, seen_file = false;
        check(host[0] && host[1]);  // the preload slots come from pinned memory
        for (std::size_t l = 2; l < 40; ++l) {
            streamed += double(rtx[l]);
            if (host[l]) { used += double(rtx[l]); file_before_host = file_before_host || (seen_file && rtx[l] == rtx[2]); }
            else if (rtx[l]) seen_file = true;
        }
        used += double(rtx[0] + rtx[1]);
        check(used <= budget && used >= budget - double(rtx[2]));
        check(file_before_host);
    }
    std::puts("host pin plan checks pass");
}
