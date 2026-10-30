#pragma once
#include <exception>
#include <array>
#include <algorithm>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace strix::minimax::session {
// Fixed-size startup votes use the existing peer channel before serving begins.
struct StartupVote {
    std::uint32_t enabled = 0, explicit_on = 0, baseline = 0;
    std::array<char, 512> reason{};
    StartupVote() = default;
    StartupVote(bool on, bool requested, const std::string& why)
        : enabled(on), explicit_on(requested), baseline(why == "baseline") {
        std::copy_n(why.data(), std::min(why.size(), reason.size() - 1), reason.data());
    }
    std::string message() const { return std::string(reason.begin(), std::find(reason.begin(), reason.end(), '\0')); }
};
template<class Engine>
StartupVote exchange_startup(Engine& eng, const StartupVote& local) {
    StartupVote peer;
    eng.exchange_bytes(&local, &peer, sizeof(local));
    return peer;
}
template<class Exchange>
StartupVote agree_startup(StartupVote local, unsigned rank, Exchange exchange) {
    const auto peer = exchange(local);
    const bool on = local.enabled && peer.enabled;
    const bool explicit_on = local.explicit_on || peer.explicit_on;
    // A baseline on either rank overrides explicit serving opt-in on both.
    if (local.baseline || peer.baseline) return {false, false, "baseline"};
    const auto& first = rank == 0 ? local : peer;
    const auto& second = rank == 0 ? peer : local;
    return {on, explicit_on, on ? "" : !first.enabled ? first.message() : second.message()};
}
// Factories perform local checks/allocation only, with no peer collectives.
// Publish the runtime only after both ranks have completed construction.
template<class Factory, class Exchange, class Log>
auto admit_runtime_pair(StartupVote local, unsigned rank, Exchange exchange,
                        Factory factory, Log log) -> decltype(factory()) {
    auto vote = agree_startup(local, rank, exchange);
    if (!vote.enabled) {
        log("session cache: off (" + vote.message() + ")");
        if (vote.explicit_on) throw std::runtime_error("session cache: explicit ON refused: " + vote.message());
        return {};
    }
    decltype(factory()) runtime;
    std::string failure;
    try { runtime = factory(); if (!runtime) failure = "runtime unavailable"; }
    catch (const std::exception& e) { failure = e.what(); }
    catch (...) { failure = "runtime initialization failed"; }
    vote = agree_startup({failure.empty(), vote.explicit_on != 0, failure}, rank, exchange);
    if (!vote.enabled) {
        runtime.reset();
        log("session cache: off (" + vote.message() + ")");
        if (vote.explicit_on) throw std::runtime_error("session cache: explicit ON refused: " + vote.message());
        return {};
    }
    log("session cache: on (runtime live)");
    return runtime;
}
// Shared host-only startup policy; explicit admission must never degrade silently.
template<class Factory, class Log>
auto start_runtime(Factory factory, bool explicit_on, Log log) -> decltype(factory()) {
    try {
        auto runtime = factory();
        log("session cache: on (runtime live)");
        return runtime;
    } catch (const std::exception& e) {
        log(std::string(explicit_on ? "session cache: initialization failed (" : "session cache: off (") + e.what() + ")");
        if (explicit_on) throw;
        return {};
    }
}
// Baseline is a serving override even when ON was requested explicitly.
// Keep admission and diagnostics shared with the host-only startup tests.
template<class Factory, class Log>
auto admit_runtime(bool enabled, bool explicit_on, const std::string& reason,
                   Factory factory, Log log) -> decltype(factory()) {
    if (!enabled) {
        log("session cache: off (" + (reason.empty() ? "peer precondition" : reason) + ")");
        if (explicit_on && reason != "baseline")
            throw std::runtime_error("session cache: explicit ON refused: " + reason);
        return {};
    }
    return start_runtime(factory, explicit_on, log);
}
}  // namespace strix::minimax::session
