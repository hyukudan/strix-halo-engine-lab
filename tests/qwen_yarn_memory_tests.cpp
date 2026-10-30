#include "../apps/qwen_yarn_memory.hpp"

#include <iostream>
#include <vector>

using namespace strix::decode_cli;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void await_trip(YarnMemoryWatchdog& watchdog) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!watchdog.tripped() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    require(watchdog.tripped(), "injected pressure must latch without a GPU");
}
void tests() {
    for (const auto* bad : {"MemTotal: 20 kB\n", "MemAvailable: bad kB\n", "MemAvailable: 3 MB\n",
            "MemAvailable: 18446744073709551615 kB\n"}) {
        std::istringstream input(bad);
        require(!yarn_parse_mem_available(input), "missing/malformed/overflow meminfo fails closed");
    }
    std::istringstream valid("Other: 4 kB\nMemAvailable: 2097152 kB\n");
    require(yarn_parse_mem_available(valid) == yarn_memory_stop_bytes, "meminfo KiB are exact bytes");
    std::atomic<std::uint64_t> available{yarn_memory_stop_bytes};
    std::atomic<unsigned> calls{0};
    YarnMemoryWatchdog watchdog([&]() -> std::optional<std::uint64_t> {
        ++calls;
        return available.load();
    }, std::chrono::milliseconds(1));
    YarnMemoryWatchdogScope scope(&watchdog);
    check_yarn_memory(); // threshold equality is safe
    std::vector<unsigned> committed;
    publish_yarn_output([&] { committed.push_back(7); });
    available = yarn_memory_stop_bytes - 1;
    await_trip(watchdog);
    bool stopped = false;
    try { publish_yarn_output([&] { committed.push_back(8); }); }
    catch (const YarnMemoryPressure& error) {
        stopped = true;
        require(error.receipt().find("\"tripped\":true") != std::string::npos, "trip receipt survives clean unwinding");
        require(error.receipt().find("\"trip_available_bytes\":2147483647") != std::string::npos, "receipt pins pressure sample");
    }
    require(stopped && committed == std::vector<unsigned>{7}, "trip prevents further committed outputs");
    available = yarn_memory_stop_bytes * 4;
    stopped = false;
    try { check_yarn_memory(); } catch (const YarnMemoryPressure&) { stopped = true; }
    require(stopped, "pressure recovery cannot reset a trip");
    watchdog.stop();
    const auto stopped_calls = calls.load();
    watchdog.stop();
    require(calls.load() == stopped_calls, "shutdown joins sampler and is idempotent");
    {
        YarnMemoryWatchdogScope native(nullptr);
        check_yarn_memory();
        require(yarn_memory_receipt().empty(), "native route has no watchdog or receipt change");
        publish_yarn_output([&] { committed.push_back(9); });
    }
    for (bool throws : {false, true}) {
        stopped = false;
        try {
            YarnMemoryWatchdog unreadable([&]() -> std::optional<std::uint64_t> {
                if (throws) throw std::runtime_error("injected meminfo read failure");
                return {};
            });
        } catch (const YarnMemoryPressure& error) {
            stopped = error.receipt().find("\"observation_unavailable\":true") != std::string::npos;
        }
        require(stopped, "missing meminfo or failed read cleanly refuses before work");
    }
    {
        unsigned reads = 0;
        YarnMemoryWatchdog short_run([&]() -> std::optional<std::uint64_t> {
            return ++reads == 1 ? yarn_memory_stop_bytes : yarn_memory_stop_bytes - 1;
        }, std::chrono::hours(1));
        stopped = false;
        try { short_run.complete(); } catch (const YarnMemoryPressure&) { stopped = true; }
        require(stopped && reads == 2, "final boundary samples short runs without waiting for the timer");
    }
    std::cout << watchdog.receipt() << '\n';
}
} // namespace
int main() {
    try { tests(); std::cout << "YaRN memory watchdog tests passed\n"; return 0; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
