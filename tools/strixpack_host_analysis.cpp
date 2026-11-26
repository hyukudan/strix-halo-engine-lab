#include "strix/weight/host_lowbit_gptq.hpp"
#include "strix/weight/strixpack_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>

namespace {
namespace sp = strix::weight::sp;
namespace gptq = strix::weight::lowbit::gptq;

const char* name(sp::CpuIsa isa) {
    if (isa == sp::CpuIsa::avx512) return "avx512";
    if (isa == sp::CpuIsa::avx2) return "avx2";
    return "scalar";
}

template <class S>
void bench() {
    constexpr std::size_t n = 4096, k = 4096;
    constexpr unsigned reps = 7;
    std::vector<std::uint8_t> payload(n * S::row_bytes(k));
    std::vector<float> out(n * k), x(k);
    std::mt19937 rng(0x2468U + S::bits);
    for (auto& b : payload) b = static_cast<std::uint8_t>(rng());
    for (auto& v : x) v = static_cast<float>(rng() % 101U) / 128.0F - 0.5F;
    const auto h = sp::header_for<S>(1.0F, sp::default_table<S::bits>());
    std::vector<sp::CpuIsa> supported;
    for (auto isa : {sp::CpuIsa::scalar, sp::CpuIsa::avx2, sp::CpuIsa::avx512})
        if (sp::cpu_supports(isa)) supported.push_back(isa);

    using clock = std::chrono::steady_clock;
    for (bool project : {false, true}) {
        std::vector<std::vector<double>> times(supported.size());
        double checksum = 0.0;
        const auto measure = [&](std::size_t run, std::size_t arm) {
            const auto isa = supported[arm];
            const auto start = clock::now();
            if (project) sp::cpu_project(S::bits, isa)(h, payload.data(), n, k, x.data(), out.data());
            else sp::cpu_decode(S::bits, isa)(h, payload.data(), n, k, out.data());
            const auto elapsed = std::chrono::duration<double>(clock::now() - start).count();
            if (run) times[arm].push_back(elapsed);
            checksum += static_cast<double>(out[(run + arm) % n]);
        };
        for (unsigned rep = 0; rep <= reps; ++rep)
            for (std::size_t i = 0; i < supported.size(); ++i)
                measure(rep, rep % 2U ? supported.size() - i - 1U : i);
        for (std::size_t arm = 0; arm < supported.size(); ++arm) {
            auto& t = times[arm];
            std::ranges::sort(t);
            std::cout << S::bits << '\t' << (project ? "project" : "decode") << '\t' << name(supported[arm])
                      << '\t' << payload.size() << '\t' << t[t.size() / 2U] * 1000.0 << '\t'
                      << static_cast<double>(payload.size()) / t[t.size() / 2U] << '\t' << checksum << '\n';
        }
    }
}

constexpr std::size_t n = 192, k = 256, samples = 512;

std::vector<double> moments(unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> x(samples * k);
    for (std::size_t r = 0; r < samples; ++r) {
        float z = normal(rng);
        for (std::size_t j = 0; j < k; ++j) {
            x[r * k + j] = z * (0.5F + static_cast<float>(j % 16U) / 8.0F);
            z = 0.85F * z + std::sqrt(1.0F - 0.85F * 0.85F) * normal(rng);
        }
    }
    std::vector<double> h(k * k);
    gptq::accumulate_second_moments(x.data(), samples, k, h);
    return h;
}

struct Study {
    const std::vector<float>& weights;
    const std::vector<double>& eval;
    const std::vector<float>& importance;
    const std::vector<float>& factor;
    const std::vector<std::uint8_t>& dead;
};

// Experimental scale geometry, confined to the study. The v1 wire storage
// keeps four group-32 bytes per superblock.
template <unsigned Bits, unsigned Group>
struct StudyStorage : sp::Storage<Bits> {
    static constexpr unsigned group = Group, groups = 128U / Group, bytes = 16U * Bits + groups;
    static constexpr std::size_t row_bytes(std::size_t width) { return width / 128U * bytes; }
};

template <class S>
void study(const Study& s, const char* distribution) {
    const auto h = sp::make_header<S>(s.weights.data(), n, k, sp::default_table<S::bits>());
    std::vector<std::uint8_t> payload(n * S::row_bytes(k));
    std::vector<float> decoded(n * k);
    std::cout << distribution << '\t' << S::bits << '\t' << S::group << '\t'
              << static_cast<double>(S::bytes) / 128.0;
    for (bool calibrated : {false, true}) {
        gptq::Options opts;
        opts.dequant = decoded.data();
        opts.threads = 1;
        if (calibrated) {
            opts.importance = s.importance.data();
            opts.factor = s.factor.data();
            opts.dead = s.dead.data();
        }
        if (!gptq::encode_rows_sp<S>(h, s.weights.data(), n, k, payload.data(), opts))
            throw std::runtime_error("synthetic encoder failed");
        double error = 0, signal = 0;
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            const double v = s.weights[i];
            const double d = decoded[i] - v;
            error += d * d;
            signal += v * v;
        }
        const auto out = gptq::output_error(s.weights.data(), decoded.data(), n, k, s.eval);
        if (!out.available()) throw std::runtime_error("synthetic output error unavailable");
        std::cout << '\t' << std::sqrt(error / signal) << '\t' << std::sqrt(out.error / out.signal);
    }
    std::cout << '\n';
}

template <unsigned Bits>
void groups(const Study& s, const char* distribution) {
    study<sp::Storage<Bits>>(s, distribution);
    study<StudyStorage<Bits, 64>>(s, distribution);
    study<StudyStorage<Bits, 128>>(s, distribution);
}

void study_all() {
    const auto fit = moments(8273);
    const auto eval = moments(8931);
    std::vector<float> diag(k);
    for (std::size_t j = 0; j < k; ++j) diag[j] = static_cast<float>(fit[j * k + j]);
    std::vector<std::uint8_t> dead;
    const auto factor = gptq::inverse_factor(fit, k, 0.1, dead);
    std::cout << "weights\tbits\tgroup\tB/weight\tRTN_w_RMS\tRTN_y_RMS\tcal_w_RMS\tcal_y_RMS\n";
    for (bool heavy : {false, true}) {
        std::mt19937 rng(heavy ? 2719U : 2693U);
        std::normal_distribution<float> normal;
        std::vector<float> weights(n * k);
        for (float& w : weights) {
            w = normal(rng);
            if (heavy) {
                const float a = normal(rng), b = normal(rng), c = normal(rng);
                w /= std::sqrt(a * a + b * b + c * c);  // Student t(3) / sqrt(3)
            }
        }
        const Study s{weights, eval, diag, factor, dead};
        const auto* distribution = heavy ? "student_t3" : "gaussian";
        groups<4>(s, distribution);
        groups<3>(s, distribution);
        groups<2>(s, distribution);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc == 2 ? argv[1] : "";
        std::cout << std::fixed << std::setprecision(6);
        if (mode == "--bench") {
            std::cout << "selected=" << name(sp::selected_cpu_isa()) << " rows=4096 columns=4096 threads=1\n"
                      << "bits\top\tisa\tpacked_bytes\tmedian_ms\tpacked_bytes/s\tchecksum\n";
            bench<sp::SP4Storage>();
            bench<sp::SP3Storage>();
            bench<sp::SP2Storage>();
        } else if (mode == "--study") {
            study_all();
        } else {
            throw std::invalid_argument("use --bench or --study");
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
