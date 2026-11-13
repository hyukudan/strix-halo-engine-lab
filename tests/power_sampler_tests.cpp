// Host tests of the shared power sampler: fake sysfs trees, no device, no thread unless stated.
#include "strix/power/sampler.hpp"
#include "strix/serve/json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
using namespace strix::power;
namespace fs = std::filesystem;

void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }
constexpr std::int64_t kSecond = 1'000'000'000LL;

struct Tree {
    fs::path root;
    explicit Tree(const char* tag) {
        root = fs::temp_directory_path() / ("strix-power-" + std::string(tag) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
    }
    ~Tree() {
        std::error_code error;
#if defined(__unix__) || defined(__APPLE__)
        for (fs::recursive_directory_iterator it(root, error), end; !error && it != end; it.increment(error))
            if (it->is_regular_file(error)) ::chmod(it->path().c_str(), 0644);
#endif
        fs::remove_all(root, error);
    }
    void put(const std::string& relative, const std::string& text) const {
        const auto path = root / relative;
        fs::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary | std::ios::trunc) << text << "\n";
    }
    void gpu(double watts, double mhz, double celsius, int busy = 99) const {
        put("sys/class/hwmon/hwmon5/name", "amdgpu");
        put("sys/class/hwmon/hwmon5/power1_label", "PPT");
        put("sys/class/hwmon/hwmon5/power1_average", std::to_string(static_cast<long long>(watts * 1e6)));
        put("sys/class/hwmon/hwmon5/freq1_input", std::to_string(static_cast<long long>(mhz * 1e6)));
        put("sys/class/hwmon/hwmon5/temp1_input", std::to_string(static_cast<long long>(celsius * 1e3)));
        put("sys/class/hwmon/hwmon5/device/gpu_busy_percent", std::to_string(busy));
        put("sys/class/hwmon/hwmon2/name", "k10temp");  // another driver: ignored
        put("sys/class/hwmon/hwmon2/temp1_input", "99000");
    }
    // Zone names hold ':' (not a legal file name everywhere): false where the tree cannot be made.
    bool rapl(long long uj, long long range = 1'000'000'000LL) const {
#ifdef _WIN32
        (void)uj; (void)range;
        return false;
#else
        put("sys/class/powercap/intel-rapl:0/name", "package-0");
        put("sys/class/powercap/intel-rapl:0/max_energy_range_uj", std::to_string(range));
        put("sys/class/powercap/intel-rapl:0/energy_uj", std::to_string(uj));
        put("sys/class/powercap/intel-rapl:0:0/name", "core");  // sub-zone: must not be counted twice
        put("sys/class/powercap/intel-rapl:0:0/energy_uj", "7");
        return true;
#endif
    }
    void cpu(long long khz) const {
        put("sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq", std::to_string(khz));
        put("sys/devices/system/cpu/cpufreq/policy1/scaling_cur_freq", std::to_string(khz + 2000));
    }
    [[nodiscard]] std::string path() const { return root.string(); }
};

SourceStatus status_of(const Sampler& sampler, const std::string& name) {
    for (const auto& s : sampler.sources()) if (s.name == name) return s;
    throw std::runtime_error("source not listed: " + name);
}

// Power only (no energy counter): trapezoid integral, interpolated window edges, clock median, temperature maximum.
void power_integral() {
    Tree tree("integral");
    tree.gpu(100.0, 2000.0, 60.0);
    tree.cpu(3'000'000);
    Sampler sampler(make_sources(tree.path()));
    require(sampler.available(), "amdgpu tree not available");
    require(status_of(sampler, "amdgpu-hwmon").available, "amdgpu source");
    require(status_of(sampler, "amdgpu-hwmon").detail.find("power1_average(PPT)") != std::string::npos, "amdgpu detail");
    require(!status_of(sampler, "rapl").available, "rapl must be unavailable without the tree");
    require(status_of(sampler, "rapl").detail.find("no package zone") != std::string::npos, "rapl reason");
    require(status_of(sampler, "cpufreq").available, "cpufreq source");
    sampler.sample_at(10 * kSecond);
    tree.gpu(200.0, 2700.0, 90.0);
    sampler.sample_at(11 * kSecond);
    tree.gpu(200.0, 2500.0, 80.0);
    sampler.sample_at(12 * kSecond);
    require(sampler.sample_count() == 3, "sample count");

    auto r = sampler.summarize(10 * kSecond, 11 * kSecond, 300);
    require(r.available && r.energy_source == "power-integral", "integral receipt");
    require(near(r.energy_j, 150.0) && near(r.avg_w, 150.0) && near(r.peak_w, 200.0), "100..200 W over 1 s");
    require(near(r.tok_per_j, 2.0) && r.tokens == 300 && r.samples == 2, "tok/J");
    require(near(r.gpu_clock_mhz_median, 2350.0) && near(r.temp_c_max, 90.0), "clock median / temperature max");
    require(near(r.cpu_mhz_median, 3001.0), "cpu clock mean of the policies");
    require(near(r.gpu_busy_pct_mean, 99.0), "busy");

    // Edges between samples: 125 W at 10.25 s and 175 W at 10.75 s.
    r = sampler.summarize(10 * kSecond + kSecond / 4, 10 * kSecond + 3 * kSecond / 4, 0);
    require(r.available && near(r.energy_j, 75.0) && near(r.peak_w, 175.0) && r.samples == 0, "interpolated window");
    require(r.tok_per_j == 0.0, "no tokens, no tok/J");
    // A window past the last sample holds the last reading.
    r = sampler.summarize(12 * kSecond, 13 * kSecond, 0);
    require(r.available && near(r.energy_j, 200.0), "held edge");
    require(!sampler.summarize(11 * kSecond, 11 * kSecond, 1).available, "empty window");
    // An older timestamp does not enter the ring.
    sampler.sample_at(5 * kSecond);
    require(sampler.sample_count() == 3, "ring stays ordered");
    double joules = 0.0, seconds = 0.0;
    sampler.lifetime(joules, seconds);
    require(near(joules, 350.0) && near(seconds, 2.0), "lifetime totals");
}

// Energy counter: preferred over the integral, wrap at max_energy_range_uj, sub-zones not double counted.
void counter_wrap() {
    Tree tree("wrap");
    tree.gpu(50.0, 1000.0, 40.0);
    if (!tree.rapl(999'000'000LL)) return;
    Sampler sampler(make_sources(tree.path()));
    require(status_of(sampler, "rapl").available, "rapl source");
    sampler.sample_at(1 * kSecond);
    tree.rapl(5'000'000LL);  // wrapped: 1 J to the range end + 5 J
    sampler.sample_at(2 * kSecond);
    tree.rapl(65'000'000LL);
    sampler.sample_at(3 * kSecond);
    auto r = sampler.summarize(1 * kSecond, 2 * kSecond, 12);
    require(r.available && r.energy_source == "counter", "counter receipt");
    require(near(r.energy_j, 6.0) && near(r.avg_w, 6.0) && near(r.tok_per_j, 2.0), "wrap delta");
    require(near(r.peak_w, 50.0), "peak from the power channel");
    r = sampler.summarize(1 * kSecond, 3 * kSecond, 0);
    require(near(r.energy_j, 66.0) && near(r.avg_w, 33.0), "two segments");
    r = sampler.summarize(2 * kSecond + kSecond / 2, 3 * kSecond, 0);
    require(near(r.energy_j, 30.0), "counter interpolated at the edge");
    double joules = 0.0, seconds = 0.0;
    sampler.lifetime(joules, seconds);
    require(near(joules, 66.0), "lifetime uses the counter");
}

// Counter only (no power channel): peak is the steepest segment.
void counter_only() {
    Tree tree("counter");
    if (!tree.rapl(0)) return;
    Sampler sampler(make_sources(tree.path()));
    require(sampler.available() && !status_of(sampler, "amdgpu-hwmon").available, "rapl only");
    sampler.sample_at(1 * kSecond);
    tree.rapl(10'000'000LL);
    sampler.sample_at(2 * kSecond);
    tree.rapl(40'000'000LL);
    sampler.sample_at(3 * kSecond);
    const auto r = sampler.summarize(1 * kSecond, 3 * kSecond, 0);
    require(r.available && near(r.energy_j, 40.0) && near(r.avg_w, 20.0) && near(r.peak_w, 30.0), "counter only");
    require(r.gpu_clock_mhz_median == 0.0 && r.temp_c_max == 0.0, "absent channels are zero");
}

// Nothing readable: unavailable with a reason, never a throw; the block still renders.
void missing_everything() {
    Tree tree("empty");
    Sampler sampler(make_sources(tree.path()));
    require(!sampler.available(), "empty tree must be unavailable");
    for (const auto& s : sampler.sources()) require(!s.available && !s.detail.empty(), "reason for " + s.name);
    sampler.start();
    sampler.sample_now();
    sampler.stop();
    const auto r = sampler.summarize(0, kSecond, 5);
    require(!r.available && r.reason == "no readable source", "unavailable receipt");
    Block block;
    block.enabled = true;
    block.sources = sampler.sources();
    block.request = block.prefill = block.decode = r;
    const auto json = strix::serve::parse_json(block_json(block));
    require(json.find("schema")->as_string() == "strix-power-receipt-v1", "schema");
    require(!json.find("available")->as_bool() && json.find("sources")->as_array().empty(), "block unavailable");
    require(json.find("source_detail")->as_array().size() == 3, "three built-in sources listed");
    require(json.find("prefill")->find("reason")->as_string() == "no readable source", "phase reason");
}

// A file that exists but cannot be read (the usual state of the package counter): that source is unavailable and says
// why; the others keep working. A partial hwmon directory (power file missing) still gives clock and temperature.
void permission_denied_and_partial() {
    Tree tree("denied");
    tree.gpu(80.0, 2400.0, 70.0);
    tree.rapl(1000);
    fs::remove(tree.root / "sys/class/hwmon/hwmon5/power1_average");
#if defined(__unix__) || defined(__APPLE__)
    const bool root = ::geteuid() == 0;
    ::chmod((tree.root / "sys/class/powercap/intel-rapl:0/energy_uj").c_str(), 0);
#else
    const bool root = true;
#endif
    Sampler sampler(make_sources(tree.path()));
    if (!root) {
        const auto rapl = status_of(sampler, "rapl");
        require(!rapl.available && rapl.detail.find("permission denied") != std::string::npos, "rapl permission reason: " + rapl.detail);
    }
    require(status_of(sampler, "amdgpu-hwmon").available, "partial amdgpu still available");
    sampler.sample_at(kSecond);
    sampler.sample_at(2 * kSecond);
    const auto r = sampler.summarize(kSecond, 2 * kSecond, 0);
    if (!root) {
        require(!r.available && r.reason == "no power or energy channel", "no energy without a power file");
    }
    require(near(sampler.latest().gpu_clock_mhz, 2400.0) && near(sampler.latest().temp_c, 70.0), "clock and temperature");
    // Garbage in a value file: not a number, source unavailable.
    Tree bad("garbage");
    bad.put("sys/class/hwmon/hwmon0/name", "amdgpu");
    bad.put("sys/class/hwmon/hwmon0/power1_average", "n/a");
    Sampler none(make_sources(bad.path()));
    require(!status_of(none, "amdgpu-hwmon").available, "garbage value accepted");
}

// A new device is one registered source: power adds up, temperature takes the maximum, no caller changes.
struct FakeCard final : Source {
    SourceStatus probe() override { return {"fake-card", true, "power temp"}; }
    void read(Sample& s) noexcept override {
        s.power_w = std::isfinite(s.power_w) ? s.power_w + 300.0 : 300.0;
        s.temp_c = std::isfinite(s.temp_c) ? std::max(s.temp_c, 75.0) : 75.0;
    }
};
void registered_source() {
    register_source("fake-card", [](const std::string&) { return std::make_unique<FakeCard>(); });
    Tree tree("extra");
    tree.gpu(100.0, 2000.0, 60.0);
    Sampler sampler(make_sources(tree.path()));
    require(status_of(sampler, "fake-card").available, "registered source listed");
    sampler.sample_at(kSecond);
    sampler.sample_at(2 * kSecond);
    const auto r = sampler.summarize(kSecond, 2 * kSecond, 800);
    require(near(r.energy_j, 400.0) && near(r.temp_c_max, 75.0) && near(r.tok_per_j, 2.0), "sources add up");
    register_source("fake-card", nullptr);
    Sampler again(make_sources(tree.path()));
    require(again.sources().size() == 3, "factory removed");
}

// The thread: samples arrive, stop joins, the ring wraps without losing order.
void thread_and_ring() {
    Tree tree("thread");
    tree.gpu(120.0, 2600.0, 85.0);
    Options options;
    options.period_ms = 2;
    options.capacity = 8;
    Sampler sampler(make_sources(tree.path()), options);
    const auto t0 = Sampler::now_ns();
    sampler.start();
    sampler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    sampler.stop();
    sampler.stop();
    const auto t1 = Sampler::now_ns();
    require(sampler.sample_count() == 8, "ring full after wrap");
    const auto r = sampler.summarize(t1 - 10'000'000, t1, 0);
    require(r.available && near(r.avg_w, 120.0, 1e-3) && near(r.gpu_clock_mhz_median, 2600.0), "threaded receipt");
    double joules = 0.0, seconds = 0.0;
    sampler.lifetime(joules, seconds);
    require(seconds > 0.1 && seconds < static_cast<double>(t1 - t0) * 1e-9 + 1e-3 && near(joules / seconds, 120.0, 1e-3), "lifetime beyond the ring");
}

// The real machine: the default root is the filesystem root, never a path relative to the working directory. Whatever
// this host exposes, a source is either available or names an absolute path in its reason; a short live run then gives
// a receipt when a power source is readable. Prints what it found (useful on a device node).
void real_machine_root() {
    Options options;
    options.period_ms = 20;
    Sampler sampler(make_sources(), options);
    for (const auto& s : sampler.sources()) {
        std::cout << "real machine: " << s.name << (s.available ? " available: " : " unavailable: ") << s.detail << "\n";
#if defined(__linux__)
        require(s.available || s.detail.find("/sys/") != std::string::npos, "relative system path in: " + s.detail);
#endif
    }
    if (!sampler.available()) return;
    const auto t0 = Sampler::now_ns();
    sampler.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sampler.stop();
    sampler.sample_now();
    const auto r = sampler.summarize(t0, Sampler::now_ns(), 0);
    if (r.available)
        std::cout << "real machine: " << r.energy_j << " J, " << r.avg_w << " W avg, " << r.peak_w << " W peak, clock "
                  << r.gpu_clock_mhz_median << " MHz, " << r.temp_c_max << " C, samples " << r.samples << ", " << r.energy_source << "\n";
    else std::cout << "real machine: no receipt: " << r.reason << "\n";
}

// The block JSON: both phases, numbers round-trip through the server's JSON reader.
void block_rendering() {
    Tree tree("json");
    tree.gpu(140.0, 2700.0, 91.0);
    Sampler sampler(make_sources(tree.path()));
    for (int i = 0; i <= 10; ++i) sampler.sample_at(i * kSecond);
    Block block;
    block.enabled = true;
    block.sources = sampler.sources();
    block.request = sampler.summarize(0, 10 * kSecond, 8192 + 127);
    block.prefill = sampler.summarize(0, 5 * kSecond, 8192);
    block.decode = sampler.summarize(5 * kSecond, 10 * kSecond, 127);
    const auto json = strix::serve::parse_json(block_json(block));
    require(json.find("available")->as_bool() && json.find("enabled")->as_bool(), "available");
    require(json.find("sources")->as_array().size() == 1 && json.find("sources")->as_array()[0].as_string() == "amdgpu-hwmon", "sources used");
    const auto* prefill = json.find("prefill");
    require(near(prefill->find("energy_j")->as_double(), 700.0) && near(prefill->find("avg_w")->as_double(), 140.0), "prefill energy");
    require(near(prefill->find("tok_per_j")->as_double(), 8192.0 / 700.0, 1e-5), "prefill tok/J");
    require(near(prefill->find("gpu_clock_mhz_median")->as_double(), 2700.0) && near(prefill->find("temp_c_max")->as_double(), 91.0), "clock / temperature");
    require(prefill->find("samples")->as_double() == 6.0 && prefill->find("energy_source")->as_string() == "power-integral", "samples / source");
    require(near(json.find("decode")->find("tok_per_j")->as_double(), 127.0 / 700.0, 1e-5), "decode tok/J");
    require(near(json.find("request")->find("energy_j")->as_double(), 1400.0), "request energy");
}

}  // namespace

int main() {
    try {
        power_integral();
        counter_wrap();
        counter_only();
        missing_everything();
        permission_denied_and_partial();
        registered_source();
        thread_and_ring();
        block_rendering();
        real_machine_root();
        std::cout << "power sampler tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "power sampler test failed: " << e.what() << "\n";
        return 1;
    }
}
