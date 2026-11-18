#include "strix/dist/link_cost.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("link cost contract: ") + what);
}
bool near(double a, double b, double tolerance) { return std::abs(a - b) <= tolerance; }
bool same_bits(const strix::dist::LinkCost& a, const strix::dist::LinkCost& b) {
    return std::memcmp(&a.latency_us, &b.latency_us, sizeof(double)) == 0 &&
           std::memcmp(&a.us_per_byte, &b.us_per_byte, sizeof(double)) == 0 &&
           std::memcmp(&a.tail_us, &b.tail_us, sizeof(double)) == 0;
}

// One rank's end of a pair whose peer message is fixed: exchange() delivers `peer` and records what was sent.
class Loopback final : public strix::dist::Transport {
public:
    explicit Loopback(std::vector<std::byte> peer) : peer_(std::move(peer)) {}
    void send(std::span<const std::byte> message) override { sent.assign(message.begin(), message.end()); }
    void receive(std::span<std::byte> message) override {
        if (message.size() > peer_.size()) throw std::runtime_error("loopback: short peer message");
        std::memcpy(message.data(), peer_.data(), message.size());
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        ++calls;
        send(out);
        receive(in);
    }
    [[nodiscard]] std::string name() const override { return "loopback"; }
    std::vector<std::byte> sent;
    unsigned calls = 0;

private:
    std::vector<std::byte> peer_;
};
std::vector<std::byte> wire(const strix::dist::LinkCost& c) {
    const double v[3] = {c.latency_us, c.us_per_byte, c.tail_us};
    std::vector<std::byte> b(sizeof(v));
    std::memcpy(b.data(), v, sizeof(v));
    return b;
}
} // namespace

int main() try {
    using namespace strix::dist;
    // Exact linear data: 30 us + 0.25 us per KiB.
    {
        std::vector<LinkSample> s;
        for (const std::uint64_t b : {4096ULL, 65536ULL, 1048576ULL})
            s.push_back({b, 30.0 + 0.25 * static_cast<double>(b) / 1024.0, 40.0 + 0.25 * static_cast<double>(b) / 1024.0});
        const auto c = fit(s);
        require(near(c.latency_us, 30.0, 1e-9) && near(c.us_per_byte * 1024.0, 0.25, 1e-12), "linear data recovered");
        require(near(c.tail_us, 10.0, 1e-9) && c.valid(), "tail = p90 - p50");
        require(near(c.exchange_us(12288), 33.0, 1e-9) && near(c.latency_ms(), 0.03, 1e-12), "prediction");
        require(near(c.bytes_per_ms(), 1000.0 * 1024.0 / 0.25, 1e-3), "rate in bytes/ms for chain::Link");
    }
    // Measured anchor (two nodes, USB4 host path, device -> device, pair p50, back to back): the model is within 10 %.
    {
        const std::vector<LinkSample> s{{12288, 36.8, 41.0}, {98304, 57.1, 71.0}, {196608, 98.4, 120.0}, {557056, 193.2, 230.0}};
        const auto c = fit(s);
        require(c.latency_us > 25.0 && c.latency_us < 40.0, "anchor latency");
        const double gbytes_ps = 1e-3 / c.us_per_byte;
        require(gbytes_ps > 3.0 && gbytes_ps < 4.5, "anchor rate");
        for (const auto& x : s) require(std::abs(c.exchange_us(x.bytes) / x.p50_us - 1.0) <= 0.10, "anchor within 10 %");
    }
    // One size: latency only, no rate term (planner treats bytes_per_ms() == 0 as latency-only).
    {
        const std::vector<LinkSample> s{{12288, 36.8, 40.0}, {12288, 37.2, 41.0}};
        const auto c = fit(s);
        require(near(c.latency_us, 37.0, 1e-9) && c.us_per_byte == 0.0 && c.bytes_per_ms() == 0.0 && c.valid(), "one size");
    }
    // Noise that makes the slope negative: flat at the mean, still valid.
    {
        const std::vector<LinkSample> s{{4096, 50.0, 60.0}, {65536, 40.0, 50.0}};
        const auto c = fit(s);
        require(c.us_per_byte == 0.0 && near(c.latency_us, 45.0, 1e-9) && c.valid(), "negative slope -> flat");
    }
    // A fit whose intercept would be <= 0 is anchored at the smallest size: positive latency, non-negative slope.
    {
        const std::vector<LinkSample> s{{1000, 1.0, 1.0}, {2000, 100.0, 100.0}, {3000, 199.0, 199.0}};
        const auto c = fit(s);
        require(c.valid() && near(c.latency_us, 1.0, 1e-12) && c.us_per_byte > 0.0, "anchored intercept");
    }
    // Inputs that cannot be a time are refused.
    {
        bool caught = false;
        try { (void)fit(std::vector<LinkSample>{{4096, 0.0, 1.0}}); } catch (const std::invalid_argument&) { caught = true; }
        require(caught, "zero time refused");
        caught = false;
        try { (void)fit(std::vector<LinkSample>{}); } catch (const std::invalid_argument&) { caught = true; }
        require(caught, "no samples refused");
    }
    // Nearest-rank percentile.
    require(percentile({5.0, 1.0, 3.0, 2.0, 4.0}, 0.5) == 3.0 && percentile({1.0, 2.0}, 0.9) == 2.0 &&
                percentile({7.0}, 0.0) == 7.0, "percentile");
    // measure(): warmup + rounds calls per size, sizes in order, positive samples.
    {
        std::vector<std::uint64_t> seen;
        const std::uint64_t sizes[] = {4096, 65536};
        const auto s = measure([&](std::uint64_t b) { seen.push_back(b); }, sizes, 2U, 5U);
        require(s.size() == 2U && s[0].bytes == 4096U && s[1].bytes == 65536U, "one sample per size, in order");
        require(seen.size() == 14U && seen.front() == 4096U && seen.back() == 65536U, "warmup + rounds calls");
        require(s[0].p50_us > 0.0 && s[0].p90_us >= s[0].p50_us, "positive samples");
        (void)fit(s);  // a sub-resolution fake exchange still fits
    }
    // measure_host(): symmetric exchanges of exactly `bytes` on the transport.
    {
        Loopback t(std::vector<std::byte>(4096));
        const std::uint64_t sizes[] = {512, 4096};
        const auto s = measure_host(t, sizes, 1U, 3U);
        require(s.size() == 2U && t.calls == 8U && t.sent.size() == 4096U, "host form exchanges the requested bytes");
    }
    // agree(): both ranks hold identical bits, = field-wise max, whichever rank combines.
    {
        const LinkCost r0{36.8, 2.9e-4, 10.0}, r1{38.1, 2.7e-4, 25.0};
        Loopback at0(wire(r1)), at1(wire(r0));
        const auto a0 = agree(at0, r0), a1 = agree(at1, r1);
        require(same_bits(a0, a1), "both ranks agree bit for bit");
        require(a0.latency_us == 38.1 && a0.us_per_byte == 2.9e-4 && a0.tail_us == 25.0, "agreement = field-wise max");
        require(at0.sent == wire(r0) && at0.calls == 1U, "one exchange of the local fit");
        Loopback bad(wire(LinkCost{}));
        bool caught = false;
        try { (void)agree(bad, r0); } catch (const std::runtime_error&) { caught = true; }
        require(caught, "an invalid peer fit is refused");
    }
    std::cout << "PASS: dist link cost (fit, anchor within 10 %, measure, rank agreement)\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
