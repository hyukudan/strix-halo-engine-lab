#include "strix/serve/json.hpp"
#include "strix/weight/pack_plan.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
constexpr double gib = 1073741824.0;

std::uint64_t gib_bytes(const std::string& text) {
    std::size_t end = 0;
    const auto bytes = std::stod(text, &end) * gib;
    if (end != text.size() || !std::isfinite(bytes) || bytes < 0 ||
        bytes >= static_cast<double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::invalid_argument("invalid GiB amount: " + text);
    return static_cast<std::uint64_t>(bytes);
}
}  // namespace

constexpr std::string_view kUsage =
    "usage: strix-lowbit-pack-plan --model-description FILE --cold sp2|sp3|sp4 --budget-gib N\n"
    "                              [--reserve-gib N] [--hot-experts N]\n"
    "Prints the Strix pack plan (tensor, format, bytes[, coder]) that fits the device budget.\n";

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (const std::string_view arg = argv[i]; arg == "--help" || arg == "-h") {
            std::cout << kUsage;
            return 0;
        }
    try {
        std::string description, cold;
        std::uint64_t budget = 0, reserve = 0, hot = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (++i == argc) throw std::invalid_argument("missing option value");
            const std::string value = argv[i];
            if (option == "--model-description") description = value;
            else if (option == "--cold") cold = value;
            else if (option == "--budget-gib") budget = gib_bytes(value);
            else if (option == "--reserve-gib") reserve = gib_bytes(value);
            else if (option == "--hot-experts") {
                const auto end = value.data() + value.size();
                const auto result = std::from_chars(value.data(), end, hot);
                if (result.ec != std::errc{} || result.ptr != end) throw std::invalid_argument("invalid expert count");
            } else throw std::invalid_argument("unknown option " + std::string(option));
        }
        if (description.empty() || cold.empty() || !budget)
            throw std::invalid_argument("need --model-description FILE --cold sp2|sp3|sp4 --budget-gib N");
        std::ifstream file(description, std::ios::binary);
        if (!file) throw std::runtime_error("cannot open " + description);
        std::ostringstream contents;
        contents << file.rdbuf();
        namespace wei = strix::weight;
        const auto result = wei::plan_strix_pack(strix::serve::parse_json_unique(contents.str()),
                                                 wei::strixpack_format(cold), hot, budget, reserve);
        const bool coders = std::ranges::any_of(result.tensors, [](const auto& tensor) { return !tensor.coder.empty(); });
        std::cout << "tensor\tformat\tbytes" << (coders ? "\tcoder\n" : "\n");
        for (const auto& tensor : result.tensors) {
            std::cout << tensor.tensor << '\t' << (tensor.storage.name.empty() ? "keep" : wei::strixpack_label(tensor.storage))
                      << '\t' << tensor.bytes;
            if (coders) std::cout << '\t' << tensor.coder;
            std::cout << '\n';
        }
        std::cerr << (result.fits() ? "FIT" : "OVER BUDGET") << " bytes=" << result.bytes << " GiB=" << std::fixed
                  << std::setprecision(1) << static_cast<double>(result.bytes) / gib << " budget_bytes=" << result.budget
                  << " reserve_bytes=" << result.reserve << " hot_experts=" << result.hot_experts << '\n';
        return result.fits() ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "strix-lowbit-pack-plan: " << error.what() << '\n';
        return 1;
    }
}
