// strix-hy4-ref: the Hy4-flash CPU oracle (src/hy4/ref) on the real GGUF pack, for the device identity receipts.
//   strix-hy4-ref PART1.gguf PART2.gguf --ids 1,2,3 [--gen N] [--threads T] [--top K]
//   --ids-file F reads the comma/space-separated prompt ids from F (strix-tokenize ... encode output).
// Prints one line per position: pos, input id, argmax, top-K (id:logit), FNV-1a 64 of the logits' bits, step ms;
// then "ids" with the N greedy ids generated after the prompt. Everything is the oracle's declared arithmetic.
#include "hy4/ref/hy4_host_ref.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<std::uint32_t> parse_ids(const std::string& text) {
    std::vector<std::uint32_t> ids;
    std::string token;
    for (const char ch : text + ",") {
        if (ch >= '0' && ch <= '9') {
            token += ch;
        } else if (!token.empty()) {
            ids.push_back(static_cast<std::uint32_t>(std::stoul(token)));
            token.clear();
        }
    }
    return ids;
}

std::uint64_t fnv(const std::vector<float>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const float f : v) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof bits);
        for (int b = 0; b < 4; ++b) {
            h ^= (bits >> (8 * b)) & 0xFFU;
            h *= 1099511628211ULL;
        }
    }
    return h;
}

int run(int argc, char** argv) {
    std::vector<std::filesystem::path> parts;
    std::vector<std::uint32_t> prompt;
    std::uint32_t gen = 0, top = 5, layers = 0;  // layers 0: full depth
    unsigned threads = std::max(1U, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--ids" && more) {
            prompt = parse_ids(argv[++i]);
        } else if (a == "--ids-file" && more) {
            std::ifstream in(argv[++i]);
            std::stringstream s;
            s << in.rdbuf();
            prompt = parse_ids(s.str());
        } else if (a == "--gen" && more) {
            gen = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--threads" && more) {
            threads = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (a == "--top" && more) {
            top = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (a == "--layers" && more) {
            layers = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (!a.empty() && a[0] != '-') {
            parts.emplace_back(a);
        } else {
            throw std::invalid_argument("unknown argument " + a);
        }
    }
    if (parts.empty() || prompt.empty())
        throw std::invalid_argument("usage: strix-hy4-ref PART1.gguf [PART2.gguf] --ids 1,2,3 | --ids-file F [--gen N] "
                                    "[--threads T] [--top K] [--layers N]");
    using clock = std::chrono::steady_clock;
    const auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    strix::hy4::set_ref_threads(threads);
    const auto t0 = clock::now();
    const auto full = strix::hy4::Config::from_gguf(parts.front());
    const auto config = layers == 0U ? full : full.truncated(layers);
    auto weights = strix::hy4::pack_weights(parts);
    strix::hy4::HostRef ref(config, *weights);
    std::cout << "config layers=" << config.layers << " hidden=" << config.hidden << " vocab=" << config.vocab
              << " threads=" << threads << " open_ms=" << std::fixed << std::setprecision(1) << ms(t0, clock::now()) << "\n";
    std::vector<std::uint32_t> generated;
    std::uint32_t token = prompt.front();
    const std::size_t steps = prompt.size() + gen - (gen > 0 ? 1U : 0U);
    double prompt_ms = 0, gen_ms = 0;
    for (std::size_t p = 0; p < steps; ++p) {
        if (p < prompt.size()) token = prompt[p];
        const auto s0 = clock::now();
        const auto& logits = ref.step(token);
        const double step_ms = ms(s0, clock::now());
        (p < prompt.size() ? prompt_ms : gen_ms) += step_ms;
        std::vector<std::uint32_t> order(logits.size());
        std::iota(order.begin(), order.end(), 0U);
        const auto k = std::min<std::size_t>(top, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                          [&logits](std::uint32_t a, std::uint32_t b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
        const auto next = ref.next_token();
        std::cout << "pos " << p << " in " << token << " argmax " << next << " top";
        for (std::size_t j = 0; j < k; ++j) std::cout << ' ' << order[j] << ':' << std::setprecision(4) << logits[order[j]];
        std::cout << " fnv " << std::hex << fnv(logits) << std::dec << " ms " << std::setprecision(0) << step_ms << "\n"
                  << std::flush;
        if (p + 1U >= prompt.size() && generated.size() < gen) {
            generated.push_back(next);
            token = next;
        }
    }
    std::cout << "ids";
    for (const auto id : generated) std::cout << ' ' << id;
    std::cout << "\n" << std::setprecision(1) << "time prompt_ms=" << prompt_ms << " gen_ms=" << gen_ms
              << " total_ms=" << ms(t0, clock::now()) << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "strix-hy4-ref: " << error.what() << "\n";
        return 1;
    }
}
