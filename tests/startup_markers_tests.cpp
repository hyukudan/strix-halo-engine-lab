#include "strix/startup_markers.hpp"
#include <cmath>
#include <array>
#include <functional>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
std::string capture(const std::function<void()>& work) {
    auto* file = std::tmpfile();
    require(file != nullptr, "capture file");
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    require(saved >= 0 && dup2(fileno(file), fileno(stderr)) >= 0, "capture stderr");
    work();
    std::fflush(stderr);
    require(dup2(saved, fileno(stderr)) >= 0, "restore stderr"); close(saved);
    std::rewind(file);
    std::string text; char buffer[1024];
    while (const auto count = std::fread(buffer, 1, sizeof(buffer), file)) text.append(buffer, count);
    std::fclose(file); return text;
}
void exercise() {
    strix::startup::Phase phase("fixture", 4096);
    strix::startup::first_prefill_tile(); strix::startup::first_prefill_tile();
    strix::startup::first_token(); strix::startup::first_token();
}
}
int main() try {
    const char* prior = std::getenv("STRIX_STARTUP_MARKERS");
    const bool had = prior != nullptr; const std::string saved = prior ? prior : "";
    for (const char* value : std::array<const char*,4>{nullptr, "0", "off", "10"}) {
        if (value) setenv("STRIX_STARTUP_MARKERS", value, 1); else unsetenv("STRIX_STARTUP_MARKERS");
        const auto bytes = capture([] { std::fputs("before\n", stderr); exercise(); std::fputs("after\n", stderr); });
        require(bytes == "before\nafter\n", "OFF preserves all surrounding stderr bytes exactly");
    }
    setenv("STRIX_STARTUP_MARKERS", "1", 1);
    const auto text = capture(exercise);
    const std::regex line("strix-startup-marker phase=([a-z_]+) event=([a-z]+) monotonic_ms=([0-9]+\\.[0-9]{3}) chunk=([0-9]+)\n");
    std::size_t offset = 0; double previous = -1; unsigned index = 0;
    const std::string phases[]{"fixture", "first_prefill_tile", "first_token", "fixture"};
    const std::string events[]{"begin", "submit", "ready", "end"};
    while (offset < text.size()) {
        std::smatch match; const auto rest = text.substr(offset);
        require(std::regex_search(rest, match, line) && match.position() == 0, "ON format is one complete stderr line");
        require(index < 4 && match[1] == phases[index] && match[2] == events[index], "phase/event order and once-only milestones");
        const double ms = std::stod(match[3]);
        require(std::isfinite(ms) && ms >= previous && ms >= 0, "timestamps are monotonic milliseconds since process start");
        require(match[4] == (index == 0 || index == 3 ? "4096" : "0"), "chunk identity");
        previous = ms; offset += static_cast<std::size_t>(match.length()); ++index;
    }
    require(index == 4, "exactly one begin/end and one of each milestone");
    if (had) setenv("STRIX_STARTUP_MARKERS", saved.c_str(), 1); else unsetenv("STRIX_STARTUP_MARKERS");
    std::cout << "startup marker byte identity and format pass\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
