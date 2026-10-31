#include "prefill.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strix::gemma4::prefill {
Mode resolve(std::string_view value, bool wmma_admitted) {
    if (value.empty() || value == "auto") return wmma_admitted ? Mode::wmma : Mode::exact;
    if (value == "exact") return Mode::exact;
    if (value == "wmma" || value == identity) return Mode::wmma;
    throw std::invalid_argument("STRIX_G4_PREFILL expects auto, exact or wmma");
}
std::vector<Chunk> plan(std::uint64_t frontier, std::size_t rows, std::uint32_t admitted) {
    if (!admitted || admitted > max_positions || frontier > admitted || rows > admitted - frontier)
        throw std::invalid_argument("prefill exceeds admitted context");
    std::vector<Chunk> out;
    while (rows) {
        const auto n = unsigned(std::min<std::size_t>(rows, chunk_rows - frontier % chunk_rows));
        out.push_back({frontier, n, ((n + tile - 1U) / tile) * tile});
        rows -= n;
        frontier += n;
    }
    return out;
}
void append_tail(DraftContext& context, const Result& result, unsigned width) {
    const auto rows = result.tail_positions.size();
    if (!width || width > hidden || rows > 1024 || result.frontier > max_positions ||
        result.tail_taps.size() != rows * 5 * width || rows > result.frontier)
        throw std::invalid_argument("invalid committed prefill tail geometry");
    for (std::size_t r = 0; r < rows; ++r)
        if (result.tail_positions[r] != result.frontier - rows + r)
            throw std::invalid_argument("noncontiguous committed prefill tail");
    for (const float value : result.tail_taps)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite committed prefill tap");
    for (std::size_t first = 0; first < rows; first += 8) {
        const auto count = std::min<std::size_t>(8, rows - first);
        context.append(std::span(result.tail_taps).subspan(first * 5 * width, count * 5 * width),
                       std::span(result.tail_positions).subspan(first, count));
    }
}
void Fidelity::observe(std::span<const float> a, std::span<const float> b) {
    if (a.empty() || a.size() != b.size()) throw std::invalid_argument("fidelity vocabulary mismatch");
    double am = -std::numeric_limits<double>::infinity(), bm = am;
    std::size_t ai = 0, bi = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::invalid_argument("nonfinite fidelity logits");
        if (a[i] > a[ai]) ai = i;
        if (b[i] > b[bi]) bi = i;
        am = std::max(am, double(a[i])); bm = std::max(bm, double(b[i]));
    }
    double az = 0.0, bz = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        az += std::exp(double(a[i]) - am); bz += std::exp(double(b[i]) - bm);
    }
    const double al = am + std::log(az), bl = bm + std::log(bz);
    double divergence = 0.0, squares = 0.0, maximum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        divergence += std::exp(double(a[i]) - al) * ((double(a[i]) - al) - (double(b[i]) - bl));
        const double e = double(a[i]) - double(b[i]);
        squares += e * e; maximum = std::max(maximum, std::abs(e));
    }
    divergence = std::max(0.0, divergence);
    kl.push_back(divergence);
    ++positions; top1_matches += unsigned(ai == bi); values += a.size();
    sum_kl += divergence; squared_error += squares; maximum_error = std::max(maximum_error, maximum);
}
double Fidelity::mean_kl() const { return positions ? sum_kl / double(positions) : std::numeric_limits<double>::infinity(); }
double Fidelity::top1() const { return positions ? double(top1_matches) / double(positions) : 0.0; }
double Fidelity::rmse() const { return values ? std::sqrt(squared_error / double(values)) : std::numeric_limits<double>::infinity(); }
double Fidelity::p99_kl() const {
    if (kl.empty()) return std::numeric_limits<double>::infinity();
    auto sorted = kl;
    std::sort(sorted.begin(), sorted.end());
    return sorted[std::size_t(std::ceil(0.99 * double(sorted.size()))) - 1];
}
bool Fidelity::same_weight_bands_pass() const { return positions && top1() >= 0.995 && mean_kl() <= 1e-4; }
bool Fidelity::bf16_bands_pass() const {
    return same_weight_bands_pass() && p99_kl() <= 1e-3 && rmse() <= 0.02 && maximum_error <= 0.25;
}
} // namespace strix::gemma4::prefill
