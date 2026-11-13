#include "strix/power/sampler.hpp"

#include "strix/env_flags.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#define STRIX_POWER_POSIX 1
#else
#define STRIX_POWER_POSIX 0
#endif

namespace strix::power {
namespace {

namespace fs = std::filesystem;

bool finite(double v) noexcept { return std::isfinite(v); }

std::string errno_reason(int error, const std::string& path) {
    if (error == ENOENT || error == ENOTDIR) return "not found: " + path;
    if (error == EACCES || error == EPERM) return "permission denied: " + path;
    return std::string(std::strerror(error)) + ": " + path;
}

// A small text file holding one number, read again at every sample. POSIX keeps the descriptor and re-reads at offset 0.
class ValueFile {
public:
    ValueFile() = default;
    ValueFile(const ValueFile&) = delete;
    ValueFile& operator=(const ValueFile&) = delete;
    ValueFile(ValueFile&& other) noexcept : path_(std::move(other.path_)), fd_(other.fd_) { other.fd_ = -1; }
    ~ValueFile() { close_fd(); }

    // False with the reason when the file cannot be opened or does not hold a number.
    bool open(const std::string& path, std::string& why) {
        close_fd();
        path_ = path;
#if STRIX_POWER_POSIX
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { why = errno_reason(errno, path); path_.clear(); return false; }
#else
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (!file) { why = errno_reason(errno, path); path_.clear(); return false; }
        std::fclose(file);
#endif
        double value = 0.0;
        if (!read(value)) { why = (errno ? errno_reason(errno, path) : "not a number: " + path); close_fd(); path_.clear(); return false; }
        return true;
    }
    [[nodiscard]] bool is_open() const noexcept { return !path_.empty() && (STRIX_POWER_POSIX ? fd_ >= 0 : true); }
    bool read(double& value) const noexcept {
        char text[64];
        std::size_t got = 0;
        errno = 0;
#if STRIX_POWER_POSIX
        if (fd_ < 0) return false;
        const auto n = ::pread(fd_, text, sizeof(text) - 1, 0);
        if (n <= 0) return false;
        got = static_cast<std::size_t>(n);
#else
        std::FILE* file = std::fopen(path_.c_str(), "rb");
        if (!file) return false;
        got = std::fread(text, 1, sizeof(text) - 1, file);
        std::fclose(file);
        if (!got) return false;
#endif
        text[got] = '\0';
        char* end = nullptr;
        const double parsed = std::strtod(text, &end);
        if (end == text || !std::isfinite(parsed)) { errno = 0; return false; }
        value = parsed;
        return true;
    }

private:
    void close_fd() noexcept {
#if STRIX_POWER_POSIX
        if (fd_ >= 0) ::close(fd_);
#endif
        fd_ = -1;
    }
    std::string path_;
    int fd_ = -1;
};

std::string read_word(const fs::path& path) {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (!file) return {};
    char text[96];
    const auto got = std::fread(text, 1, sizeof(text) - 1, file);
    std::fclose(file);
    std::string word(text, got);
    while (!word.empty() && (word.back() == '\n' || word.back() == '\r' || word.back() == ' ')) word.pop_back();
    return word;
}

std::vector<fs::path> children(const fs::path& directory) {
    std::vector<fs::path> out;
    std::error_code error;
    for (fs::directory_iterator it(directory, error), end; !error && it != end; it.increment(error)) out.push_back(it->path());
    std::sort(out.begin(), out.end());
    return out;
}

void add(double& slot, double value) noexcept { slot = finite(slot) ? slot + value : value; }
void set_max(double& slot, double value) noexcept { slot = finite(slot) ? std::max(slot, value) : value; }
void set_if_absent(double& slot, double value) noexcept { if (!finite(slot)) slot = value; }

// amdgpu hwmon: power (the SoC package power tracking value on an integrated device, the card on a discrete one), the
// shader clock, the edge temperature and the busy percentage. World readable on a stock kernel.
class AmdgpuHwmon final : public Source {
public:
    explicit AmdgpuHwmon(std::string root) : root_(std::move(root)) {}
    SourceStatus probe() override {
        SourceStatus status{"amdgpu-hwmon", false, {}};
        const fs::path base = fs::path(root_) / "sys" / "class" / "hwmon";
        std::string first_problem;
        for (const auto& dir : children(base)) {
            if (read_word(dir / "name") != "amdgpu") continue;
            Card card;
            std::string why, what = dir.filename().string() + ":";
            if (card.power.open((dir / "power1_average").string(), why)) what += " power1_average";
            else if (card.power.open((dir / "power1_input").string(), why)) what += " power1_input";
            else if (first_problem.empty()) first_problem = why;
            if (const auto label = read_word(dir / "power1_label"); card.power.is_open() && !label.empty()) what += "(" + label + ")";
            if (card.clock.open((dir / "freq1_input").string(), why)) what += " freq1_input";
            if (card.temp.open((dir / "temp1_input").string(), why)) what += " temp1_input";
            if (card.busy.open((dir / "device" / "gpu_busy_percent").string(), why)) what += " gpu_busy_percent";
            if (!card.power.is_open() && !card.clock.is_open() && !card.temp.is_open()) continue;
            status.detail += (status.detail.empty() ? "" : "; ") + what;
            cards_.push_back(std::move(card));
        }
        status.available = !cards_.empty();
        if (!status.available)
            status.detail = first_problem.empty() ? "no amdgpu hwmon device under " + base.generic_string() : first_problem;
        return status;
    }
    void read(Sample& sample) noexcept override {
        for (const auto& card : cards_) {
            double v = 0.0;
            if (card.power.read(v)) add(sample.power_w, v * 1e-6);
            if (card.clock.read(v)) set_if_absent(sample.gpu_clock_mhz, v * 1e-6);
            if (card.temp.read(v)) set_max(sample.temp_c, v * 1e-3);
            if (card.busy.read(v)) set_if_absent(sample.gpu_busy_pct, v);
        }
    }

private:
    struct Card { ValueFile power, clock, temp, busy; };
    std::string root_;
    std::vector<Card> cards_;
};

// powercap RAPL package energy counters. Usually readable by root only; then the source is unavailable and says so.
class Rapl final : public Source {
public:
    explicit Rapl(std::string root) : root_(std::move(root)) {}
    SourceStatus probe() override {
        SourceStatus status{"rapl", false, {}};
        const fs::path base = fs::path(root_) / "sys" / "class" / "powercap";
        std::string problem;
        for (const auto& dir : children(base)) {
            const auto zone = dir.filename().string();
            // Top-level zones only ("intel-rapl:0"); "intel-rapl:0:0" is a sub-zone already inside its package.
            if (zone.rfind("intel-rapl:", 0) != 0 || std::count(zone.begin(), zone.end(), ':') != 1) continue;
            if (read_word(dir / "name").rfind("package", 0) != 0) continue;
            Zone z;
            std::string why;
            if (!z.energy.open((dir / "energy_uj").string(), why)) { if (problem.empty()) problem = why; continue; }
            ValueFile range;
            if (range.open((dir / "max_energy_range_uj").string(), why)) range.read(z.range_uj);
            z.energy.read(z.last_uj);
            status.detail += (status.detail.empty() ? "" : "; ") + zone + ": energy_uj";
            zones_.push_back(std::move(z));
        }
        status.available = !zones_.empty();
        if (!status.available) status.detail = problem.empty() ? "no package zone under " + base.generic_string() : problem;
        return status;
    }
    void read(Sample& sample) noexcept override {
        bool any = false;
        for (auto& z : zones_) {
            double now = 0.0;
            if (!z.energy.read(now)) continue;
            // The counter wraps at max_energy_range_uj. A drop with no known range is a reset: nothing is added.
            const double delta = now >= z.last_uj ? now - z.last_uj : (z.range_uj > 0.0 ? now + z.range_uj - z.last_uj : 0.0);
            z.last_uj = now;
            total_j_ += delta * 1e-6;
            any = true;
        }
        if (any) add(sample.energy_j, total_j_);
    }

private:
    struct Zone { ValueFile energy; double range_uj = 0.0, last_uj = 0.0; };
    std::string root_;
    std::vector<Zone> zones_;
    double total_j_ = 0.0;
};

// Mean CPU frequency over the cpufreq policies, refreshed at most once per second (a policy read is not free).
class Cpufreq final : public Source {
public:
    explicit Cpufreq(std::string root) : root_(std::move(root)) {}
    SourceStatus probe() override {
        SourceStatus status{"cpufreq", false, {}};
        const fs::path base = fs::path(root_) / "sys" / "devices" / "system" / "cpu" / "cpufreq";
        std::string problem;
        for (const auto& dir : children(base)) {
            if (dir.filename().string().rfind("policy", 0) != 0) continue;
            ValueFile file;
            std::string why;
            if (file.open((dir / "scaling_cur_freq").string(), why)) files_.push_back(std::move(file));
            else if (problem.empty()) problem = why;
        }
        status.available = !files_.empty();
        status.detail = status.available ? std::to_string(files_.size()) + " policies: scaling_cur_freq"
            : problem.empty() ? "no cpufreq policy under " + base.generic_string() : problem;
        return status;
    }
    void read(Sample& sample) noexcept override {
        if (!finite(cached_) || sample.t_ns - last_ns_ >= 1'000'000'000LL || sample.t_ns < last_ns_) {
            double sum = 0.0;
            std::size_t n = 0;
            for (const auto& file : files_) { double v = 0.0; if (file.read(v)) { sum += v; ++n; } }
            cached_ = n ? sum / static_cast<double>(n) * 1e-3 : kAbsent;
            last_ns_ = sample.t_ns;
        }
        if (finite(cached_)) set_if_absent(sample.cpu_mhz, cached_);
    }

private:
    std::string root_;
    std::vector<ValueFile> files_;
    double cached_ = kAbsent;
    std::int64_t last_ns_ = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<std::pair<std::string, SourceFactory>> kinds;
    Registry() {
        kinds.emplace_back("rapl", [](const std::string& root) { return std::make_unique<Rapl>(root); });
        kinds.emplace_back("amdgpu-hwmon", [](const std::string& root) { return std::make_unique<AmdgpuHwmon>(root); });
        kinds.emplace_back("cpufreq", [](const std::string& root) { return std::make_unique<Cpufreq>(root); });
    }
};
Registry& registry() { static Registry r; return r; }

double lerp(double a, double b, double f) noexcept {
    if (finite(a) && finite(b)) return a + (b - a) * f;
    return finite(a) ? a : b;
}
Sample between(const Sample& a, const Sample& b, std::int64_t t) noexcept {
    const double f = b.t_ns > a.t_ns ? static_cast<double>(t - a.t_ns) / static_cast<double>(b.t_ns - a.t_ns) : 0.0;
    Sample out;
    out.t_ns = t;
    out.energy_j = lerp(a.energy_j, b.energy_j, f);
    out.power_w = lerp(a.power_w, b.power_w, f);
    out.gpu_clock_mhz = lerp(a.gpu_clock_mhz, b.gpu_clock_mhz, f);
    out.temp_c = lerp(a.temp_c, b.temp_c, f);
    out.cpu_mhz = lerp(a.cpu_mhz, b.cpu_mhz, f);
    out.gpu_busy_pct = lerp(a.gpu_busy_pct, b.gpu_busy_pct, f);
    return out;
}
double median(std::vector<double>& values) noexcept {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void json_text(std::string& out, const std::string& text) {
    out += '"';
    for (const char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (static_cast<unsigned char>(c) < 0x20) { char u[8]; std::snprintf(u, sizeof(u), "\\u%04x", static_cast<unsigned>(c)); out += u; }
        else out += c;
    }
    out += '"';
}
void json_number(std::string& out, double value) {
    char text[40];
    std::snprintf(text, sizeof(text), "%.6g", finite(value) ? value : 0.0);
    out += text;
}
void json_member(std::string& out, const char* key, double value) { out += ",\""; out += key; out += "\":"; json_number(out, value); }
void receipt_json(std::string& out, const Receipt& r) {
    if (!r.available) { out += "{\"available\":false,\"reason\":"; json_text(out, r.reason); out += '}'; return; }
    out += "{\"available\":true";
    json_member(out, "duration_ms", r.duration_ms);
    json_member(out, "energy_j", r.energy_j);
    json_member(out, "avg_w", r.avg_w);
    json_member(out, "peak_w", r.peak_w);
    out += ",\"tokens\":" + std::to_string(r.tokens);
    json_member(out, "tok_per_j", r.tok_per_j);
    json_member(out, "gpu_clock_mhz_median", r.gpu_clock_mhz_median);
    json_member(out, "temp_c_max", r.temp_c_max);
    json_member(out, "cpu_mhz_median", r.cpu_mhz_median);
    json_member(out, "gpu_busy_pct_mean", r.gpu_busy_pct_mean);
    out += ",\"samples\":" + std::to_string(r.samples) + ",\"energy_source\":";
    json_text(out, r.energy_source);
    out += '}';
}
void sources_json(std::string& out, const std::vector<SourceStatus>& sources) {
    out += "\"sources\":[";
    bool first = true;
    for (const auto& s : sources) if (s.available) { if (!first) out += ','; first = false; json_text(out, s.name); }
    out += "],\"source_detail\":[";
    first = true;
    for (const auto& s : sources) {
        if (!first) out += ',';
        first = false;
        out += "{\"name\":"; json_text(out, s.name);
        out += s.available ? ",\"available\":true,\"detail\":" : ",\"available\":false,\"reason\":";
        json_text(out, s.detail);
        out += '}';
    }
    out += ']';
}

Receipt unavailable(std::string reason) { Receipt r; r.reason = std::move(reason); return r; }

struct Process {
    std::once_flag once;
    bool enabled = true;
    int period_ms = 100;
    std::vector<SourceStatus> sources;
    std::unique_ptr<Sampler> sampler;   // set once, read through `live`
    std::atomic<Sampler*> live{nullptr};
    std::int64_t start_ns = 0;
    std::atomic<std::int64_t> first_token{0};
};
Process& process() { static Process* p = new Process; return *p; }  // never destroyed: used from atexit

void exit_line() noexcept {
    try {
        auto& p = process();
        if (auto* s = p.live.load()) {
            s->sample_now();
            Block block;
            block.enabled = true;
            block.sources = p.sources;
            block.request = s->summarize(p.start_ns, Sampler::now_ns(), 0);
            block.prefill = block.decode = unavailable("process total");
            std::fprintf(stderr, "strix power: %s\n", block_json(block).c_str());
        }
        stop_process_sampler();
    } catch (...) {
    }
}

}  // namespace

void register_source(std::string kind, SourceFactory factory) {
    auto& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    for (auto& entry : r.kinds) if (entry.first == kind) { entry.second = std::move(factory); return; }
    r.kinds.emplace_back(std::move(kind), std::move(factory));
}

std::vector<std::unique_ptr<Source>> make_sources(const std::string& root) {
    auto& r = registry();
    const std::lock_guard<std::mutex> lock(r.mutex);
    std::vector<std::unique_ptr<Source>> out;
    // The real machine is the filesystem root; an empty prefix would make every path relative to the working directory.
    const std::string base = root.empty() ? std::string("/") : root;
    for (const auto& entry : r.kinds) if (entry.second) if (auto source = entry.second(base)) out.push_back(std::move(source));
    return out;
}

Sampler::Sampler(std::vector<std::unique_ptr<Source>> sources, Options options) : sources_(std::move(sources)), options_(options) {
    options_.period_ms = std::clamp(options_.period_ms, 1, 10000);
    options_.capacity = std::max<std::size_t>(options_.capacity, 4);
    for (auto& source : sources_) {
        SourceStatus status;
        try { status = source->probe(); } catch (const std::exception& e) { status.available = false; status.detail = e.what(); }
        if (status.available) live_.push_back(source.get());
        status_.push_back(std::move(status));
    }
    usable_ = !live_.empty();
    if (usable_) ring_.resize(options_.capacity);
}

Sampler::~Sampler() { stop(); }

void Sampler::start() {
    if (!usable_) return;
    const std::lock_guard<std::mutex> lock(wake_mutex_);
    if (started_) return;
    started_ = true;
    stopping_ = false;
    thread_ = std::thread([this] { run(); });
}

void Sampler::stop() {
    {
        const std::lock_guard<std::mutex> lock(wake_mutex_);
        if (!started_) return;
        stopping_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
    const std::lock_guard<std::mutex> lock(wake_mutex_);
    started_ = false;
}

void Sampler::run() {
    std::unique_lock<std::mutex> lock(wake_mutex_);
    while (!stopping_) {
        lock.unlock();
        sample_now();
        lock.lock();
        wake_.wait_for(lock, std::chrono::milliseconds(options_.period_ms), [this] { return stopping_; });
    }
}

void Sampler::sample_now() {
    if (!usable_) return;
    const std::lock_guard<std::mutex> lock(read_mutex_);
    sample_at(now_ns());
}

void Sampler::sample_at(std::int64_t t_ns) {
    if (!usable_) return;
    Sample sample;
    sample.t_ns = t_ns;
    for (auto* source : live_) source->read(sample);
    const std::lock_guard<std::mutex> lock(mutex_);
    if (size_) {
        const Sample& last = ring_[(head_ + ring_.size() - 1) % ring_.size()];
        if (last.t_ns >= t_ns) return;  // keep the ring ordered
        const double dt = static_cast<double>(t_ns - last.t_ns) * 1e-9;
        if (finite(sample.energy_j) && finite(last.energy_j)) lifetime_j_ += sample.energy_j - last.energy_j;
        else if (finite(sample.power_w) && finite(last.power_w)) lifetime_j_ += 0.5 * (sample.power_w + last.power_w) * dt;
        lifetime_s_ += dt;
    }
    ring_[head_] = sample;
    head_ = (head_ + 1) % ring_.size();
    size_ = std::min(size_ + 1, ring_.size());
}

Sample Sampler::latest() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return size_ ? ring_[(head_ + ring_.size() - 1) % ring_.size()] : Sample{};
}

void Sampler::lifetime(double& joules, double& seconds) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    joules = lifetime_j_;
    seconds = lifetime_s_;
}

std::size_t Sampler::sample_count() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return size_;
}

// The samples of [t0, t1] in time order, with an interpolated (or held) point at each edge; `inside` is the number of
// real samples. Empty when the ring has nothing to say about the window.
std::vector<Sample> Sampler::window_locked(std::int64_t t0_ns, std::int64_t t1_ns, std::size_t& inside, bool& lead) const {
    inside = 0;
    lead = false;
    std::vector<Sample> newest_first;
    const Sample* after = nullptr;
    const Sample* before = nullptr;
    for (std::size_t k = 0; k < size_; ++k) {
        const Sample& s = ring_[(head_ + ring_.size() - 1 - k) % ring_.size()];
        if (s.t_ns > t1_ns) { after = &s; continue; }
        if (s.t_ns < t0_ns) { before = &s; break; }
        newest_first.push_back(s);
    }
    inside = newest_first.size();
    std::vector<Sample> points;
    if (!inside && !before && !after) return points;
    points.reserve(inside + 2);
    const auto edge = [&](std::int64_t t, const Sample* left, const Sample* right) {
        Sample out = left && right ? between(*left, *right, t) : left ? *left : *right;
        out.t_ns = t;
        return out;
    };
    const Sample* first_in = inside ? &newest_first.back() : nullptr;
    const Sample* last_in = inside ? &newest_first.front() : nullptr;
    if (!first_in || first_in->t_ns > t0_ns) { points.push_back(edge(t0_ns, before, first_in ? first_in : after)); lead = true; }
    for (auto it = newest_first.rbegin(); it != newest_first.rend(); ++it) points.push_back(*it);
    if (!last_in || last_in->t_ns < t1_ns) points.push_back(edge(t1_ns, last_in ? last_in : before, after));
    return points;
}

Receipt Sampler::summarize(std::int64_t t0_ns, std::int64_t t1_ns, std::uint64_t tokens) const {
    if (!usable_) return unavailable("no readable source");
    if (t1_ns <= t0_ns) return unavailable("empty window");
    std::size_t inside = 0;
    bool lead = false;
    std::vector<Sample> points;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        points = window_locked(t0_ns, t1_ns, inside, lead);
    }
    if (points.size() < 2) return unavailable("no samples");
    Receipt r;
    r.samples = inside;
    r.tokens = tokens;
    const double seconds = static_cast<double>(t1_ns - t0_ns) * 1e-9;
    r.duration_ms = seconds * 1e3;
    const bool counter = finite(points.front().energy_j) && finite(points.back().energy_j);
    double integral = 0.0, peak = 0.0;
    bool any_power = false;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        if (finite(p.power_w)) { any_power = true; peak = std::max(peak, p.power_w); }
        if (!i) continue;
        const auto& q = points[i - 1];
        const double dt = static_cast<double>(p.t_ns - q.t_ns) * 1e-9;
        if (finite(p.power_w) && finite(q.power_w)) integral += 0.5 * (p.power_w + q.power_w) * dt;
        else if (finite(p.power_w) || finite(q.power_w)) integral += (finite(p.power_w) ? p.power_w : q.power_w) * dt;
        // No power channel: the steepest counter segment is the peak.
        if (counter && !any_power && dt > 0.0 && finite(p.energy_j) && finite(q.energy_j)) peak = std::max(peak, (p.energy_j - q.energy_j) / dt);
    }
    if (counter) { r.energy_j = points.back().energy_j - points.front().energy_j; r.energy_source = "counter"; }
    else if (any_power) { r.energy_j = integral; r.energy_source = "power-integral"; }
    else return unavailable("no power or energy channel");
    r.available = true;
    r.avg_w = seconds > 0.0 ? r.energy_j / seconds : 0.0;
    r.peak_w = std::max(peak, any_power ? 0.0 : r.avg_w);
    r.tok_per_j = tokens && r.energy_j > 0.0 ? static_cast<double>(tokens) / r.energy_j : 0.0;
    // Clock, temperature and load over the real samples; the edge points stand in when the window has none.
    const std::size_t lo = inside && lead ? 1 : 0;
    const std::size_t hi = inside ? lo + inside : points.size();
    std::vector<double> clock, cpu;
    double busy = 0.0;
    std::size_t busy_n = 0;
    for (std::size_t i = lo; i < hi; ++i) {
        const auto& p = points[i];
        if (finite(p.gpu_clock_mhz)) clock.push_back(p.gpu_clock_mhz);
        if (finite(p.cpu_mhz)) cpu.push_back(p.cpu_mhz);
        if (finite(p.temp_c)) r.temp_c_max = std::max(r.temp_c_max, p.temp_c);
        if (finite(p.gpu_busy_pct)) { busy += p.gpu_busy_pct; ++busy_n; }
    }
    r.gpu_clock_mhz_median = median(clock);
    r.cpu_mhz_median = median(cpu);
    r.gpu_busy_pct_mean = busy_n ? busy / static_cast<double>(busy_n) : 0.0;
    return r;
}

// ---- process sampler ----------------------------------------------------------------------------------------------

void start_process_sampler() noexcept {
    try {
        auto& p = process();
        std::call_once(p.once, [&p] {
            p.enabled = envcfg::flag("STRIX_POWER_SAMPLER", true, EnvFlagMatch::exact);
            if (!p.enabled) return;
            p.period_ms = std::clamp(envcfg::integer("STRIX_POWER_SAMPLE_MS", 100), 10, 10000);
            Options options;
            options.period_ms = p.period_ms;
            auto sampler = std::make_unique<Sampler>(make_sources(), options);
            p.sources = sampler->sources();
            if (!sampler->available()) return;
            p.start_ns = Sampler::now_ns();
            sampler->start();
            p.sampler = std::move(sampler);
            p.live.store(p.sampler.get());
            std::atexit(exit_line);
        });
    } catch (...) {
    }
}

void stop_process_sampler() noexcept {
    try {
        if (auto* s = process().live.exchange(nullptr)) s->stop();
    } catch (...) {
    }
}

Sampler* process_sampler() noexcept { return process().live.load(); }
bool sampler_enabled() noexcept { start_process_sampler(); return process().enabled; }
std::vector<SourceStatus> process_sources() { start_process_sampler(); return process().sources; }

void mark_first_token() noexcept {
    auto& slot = process().first_token;
    if (slot.load(std::memory_order_relaxed)) return;
    std::int64_t expected = 0;
    slot.compare_exchange_strong(expected, Sampler::now_ns());
}
void reset_first_token() noexcept { process().first_token.store(0); }
std::int64_t first_token_ns() noexcept { return process().first_token.load(); }

namespace {
Block spans_block(std::int64_t p0, std::int64_t d0, std::int64_t end, std::uint64_t prefill_tokens, std::uint64_t decode_tokens) {
    Block block;
    block.enabled = sampler_enabled();
    block.sources = process().sources;
    auto* s = process_sampler();
    if (!s) {
        const auto why = unavailable(block.enabled ? "no readable source" : "STRIX_POWER_SAMPLER=0");
        block.request = block.prefill = block.decode = why;
        return block;
    }
    // A window that ends now needs a closing sample; one that ended long ago already has its neighbours.
    if (Sampler::now_ns() - end < 2'000'000LL * process().period_ms) s->sample_now();
    block.request = s->summarize(p0, end, prefill_tokens + decode_tokens);
    block.prefill = d0 > p0 ? s->summarize(p0, d0, prefill_tokens) : unavailable("no prefill phase");
    block.decode = end > d0 ? s->summarize(d0, end, decode_tokens) : unavailable("no decode phase");
    return block;
}
std::int64_t ms_ns(double ms) noexcept { return finite(ms) && ms > 0.0 ? static_cast<std::int64_t>(ms * 1e6) : 0; }
}  // namespace

Block request_block(std::int64_t end_ns, Phase prefill, Phase decode) {
    const auto d0 = end_ns - ms_ns(decode.ms);
    return spans_block(d0 - ms_ns(prefill.ms), d0, end_ns, prefill.tokens, decode.tokens);
}

Block request_block(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop,
                    std::uint64_t prompt_tokens, std::uint64_t generated_tokens) {
    const auto p0 = Sampler::to_ns(start), end = Sampler::to_ns(stop);
    const auto first = first_token_ns();
    const bool split = first > p0 && first < end && generated_tokens > 1;
    auto block = spans_block(p0, split ? first : end, end, prompt_tokens, split ? generated_tokens - 1 : 0);
    // Several tokens but no first-token mark on this route: only the request total is honest.
    if (!split && generated_tokens > 1 && block.request.available) {
        block.request.tokens = prompt_tokens + generated_tokens;
        block.request.tok_per_j = block.request.energy_j > 0.0 ? static_cast<double>(block.request.tokens) / block.request.energy_j : 0.0;
        block.prefill = block.decode = unavailable("first token not marked on this route");
    }
    return block;
}

std::string block_json(const Block& block) {
    std::string out = "{\"schema\":\"strix-power-receipt-v1\",\"enabled\":";
    out += block.enabled ? "true" : "false";
    out += ",\"available\":";
    out += block.request.available ? "true," : "false,";
    sources_json(out, block.sources);
    out += ",\"request\":"; receipt_json(out, block.request);
    out += ",\"prefill\":"; receipt_json(out, block.prefill);
    out += ",\"decode\":"; receipt_json(out, block.decode);
    out += '}';
    return out;
}

std::string status_json() {
    const bool enabled = sampler_enabled();
    auto& p = process();
    auto* s = process_sampler();
    std::string out = "{\"schema\":\"strix-power-status-v1\",\"enabled\":";
    out += enabled ? "true" : "false";
    out += ",\"available\":";
    out += s ? "true," : "false,";
    sources_json(out, p.sources);
    if (s) {
        s->sample_now();
        const auto last = s->latest();
        out += ",\"period_ms\":" + std::to_string(p.period_ms) + ",\"samples\":" + std::to_string(s->sample_count()) + ",\"latest\":{\"power_w\":";
        json_number(out, last.power_w);
        json_member(out, "gpu_clock_mhz", last.gpu_clock_mhz);
        json_member(out, "temp_c", last.temp_c);
        json_member(out, "cpu_mhz", last.cpu_mhz);
        json_member(out, "gpu_busy_pct", last.gpu_busy_pct);
        double joules = 0.0, seconds = 0.0;
        s->lifetime(joules, seconds);
        out += "},\"since_start\":{\"energy_j\":";
        json_number(out, joules);
        json_member(out, "duration_s", seconds);
        json_member(out, "avg_w", seconds > 0.0 ? joules / seconds : 0.0);
        out += '}';
    }
    out += '}';
    return out;
}

}  // namespace strix::power
