#include "observe_driver.hpp"
#include <iostream>
#include <map>

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: strix-direct-gpu-observe-driver --configured-gpu-direct --observe-driver /absolute/char/device --kernel-release RELEASE --driver-build-id HEX\n"
            "Actual owned-FD QUERY/kernel/module observation only; no GPU/import/ARM/payload/socket or qualification.\n"
            "Require privileged serialized ConfigFS gpu_direct ownership through close; unknown stock/copied-mode open may activate paths before ENOTTY.\n"
            "Configured extension mode creates disabled rings and a fresh per-open instance UUID; do not reuse it as the later commissioning FD.\n"
            "Keep hardware observations outside Git.\n"; return 0;
    }
    try {
        std::map<std::string, std::string> options; bool configured = false;
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--configured-gpu-direct") { if (configured) throw std::invalid_argument("duplicate configured-gpu-direct acknowledgment"); configured = true; }
            else if (index + 1 >= argc || !options.emplace(option, argv[++index]).second) throw std::invalid_argument("missing/duplicate observation option");
        }
        if (options.size() != 3 || !options.contains("--observe-driver") || !options.contains("--kernel-release") || !options.contains("--driver-build-id")) throw std::invalid_argument("exact route/kernel/build observation options required");
        const auto route = strix::inspect_direct_stream_device(options.at("--observe-driver"));
        const auto observation = strix::transport::observe_direct_gpu_stream_driver(route, {configured, options.at("--kernel-release"), options.at("--driver-build-id")});
        std::cout << strix::transport::direct_gpu_stream_driver_observation_json(observation);
        return observation.query_observed && observation.error.empty() && !observation.fd_retained ? 0 : 1;
    } catch (const std::exception& error) { std::cerr << "driver observation unavailable: " << error.what() << '\n'; return 1; }
}
