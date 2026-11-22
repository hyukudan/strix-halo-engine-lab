#include "strix/exec/profiler.hpp"

#include <algorithm>
#include <charconv>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {

using strix::exec::bench::Kind;
using strix::exec::bench::Options;

constexpr std::string_view usage =
    "usage: strix-bench launch [--nodes N] [--repeats N]\n"
    "       strix-bench byte-rate [--bytes N[,N...]] [--repeats N]\n"
    "       strix-bench entry --entry ID [--rows N] [--width N] [--repeats N]\n"
    "                        [--byte-gbps N] [--peak-tops N]\n"
    "       strix-bench exchange\n"
    "device selection: [--devices FILE] [--device-id ID] [--device-profile ID]\n"
    "measurement metadata: [--build TEXT] [--power-state TEXT]\n"
    "entry IDs: cuda-sm120-norm-rms-v1, cuda-sm120-norm-rms-bf16-gain-v1\n";

template<class T>
T number(std::string_view text, T lower, T upper, std::string_view flag) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !(value >= lower && value <= upper))
        throw std::invalid_argument("invalid value for " + std::string(flag));
    return value;
}

std::vector<std::uint64_t> buffer_sizes(std::string_view text) {
    std::vector<std::uint64_t> sizes;
    while (true) {
        const auto comma = text.find(',');
        const auto bytes = number<std::uint64_t>(text.substr(0, comma), 16, 1ULL << 30, "--bytes");
        if (bytes % 16 != 0 || sizes.size() == 16 || std::ranges::find(sizes, bytes) != sizes.end())
            throw std::invalid_argument("--bytes requires up to 16 distinct sizes aligned to 16 bytes");
        sizes.push_back(bytes);
        if (comma == text.npos) return sizes;
        text.remove_prefix(comma + 1);
    }
}

Options parse(std::span<const std::string_view> args) {
    Options options;
    const auto mode = args.front();
    if (mode == "launch") options.kind = Kind::launch;
    else if (mode == "byte-rate") options.kind = Kind::byte_rate;
    else if (mode == "entry") options.kind = Kind::entry;
    else if (mode == "exchange") options.kind = Kind::exchange;
    else throw std::invalid_argument("unknown bench: " + std::string(mode));

    std::set<std::string_view> seen;
    for (std::size_t i = 1; i < args.size(); i += 2) {
        const auto flag = args[i];
        if (options.kind == Kind::exchange) throw std::invalid_argument("exchange takes no options");
        if (!seen.insert(flag).second) throw std::invalid_argument("duplicate option: " + std::string(flag));
        if (i + 1 >= args.size() || args[i + 1].empty() || args[i + 1].starts_with("--"))
            throw std::invalid_argument("missing value for " + std::string(flag));
        const auto value = args[i + 1];
        if (flag == "--repeats") options.repeats = number<std::uint32_t>(value, 1, 256, flag);
        else if (flag == "--devices") options.devices = value;
        else if (flag == "--device-id") options.device_id = value;
        else if (flag == "--device-profile") options.device_profile = value;
        else if (flag == "--build") options.build = value;
        else if (flag == "--power-state") options.power_state = value;
        else if (options.kind == Kind::launch && flag == "--nodes")
            options.nodes = number<std::uint32_t>(value, 2, 256, flag);
        else if (options.kind == Kind::byte_rate && flag == "--bytes") options.bytes = buffer_sizes(value);
        else if (options.kind == Kind::entry && flag == "--entry") options.entry = value;
        else if (options.kind == Kind::entry && flag == "--rows") options.rows = number<std::uint32_t>(value, 1, 32, flag);
        else if (options.kind == Kind::entry && flag == "--width") options.width = number<std::uint32_t>(value, 1, 8192, flag);
        else if (options.kind == Kind::entry && flag == "--byte-gbps") options.byte_gbps = number(value, 1e-6, 1e9, flag);
        else if (options.kind == Kind::entry && flag == "--peak-tops") options.peak_tops = number(value, 1e-6, 1e9, flag);
        else throw std::invalid_argument("unknown option for " + std::string(mode) + ": " + std::string(flag));
    }
    if (options.kind == Kind::entry && options.entry.empty()) throw std::invalid_argument("entry requires --entry ID");
    options.command = "strix-bench";
    for (const auto arg : args) {
        options.command.push_back(' ');
        options.command.append(arg);
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        std::cout << usage;
        return argc == 1 ? 2 : 0;
    }
    std::vector<std::string_view> args(argv + 1, argv + argc);
    Options options;
    try {
        options = parse(args);
    } catch (const std::invalid_argument& error) {
        std::cerr << "strix-bench: " << error.what() << '\n';
        return 2;
    }
    if (options.kind == Kind::exchange) {
        std::cout << "{\"schema\":\"strix-bench/v1\",\"bench\":\"exchange\",\"status\":\"unsupported\","
                     "\"reason\":\"single-device runner has no exchange channel\"}\n";
        return 0;
    }
#ifdef STRIX_X6_CUDA
    try {
        const auto output = strix::exec::bench::run_cuda(options);
        if (!output.table.empty()) std::cerr << output.table;
        std::cout << output.json << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "strix-bench: " << error.what() << '\n';
        return 1;
    }
#else
    std::cerr << "strix-bench: CUDA bench support was not built\n";
    return 1;
#endif
}
