#include "strix/dist/communicator.hpp"

#include "strix/serve/json.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace strix::dist {
namespace {

void check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string("dist communicator: ") + what + ": " + hipGetErrorString(status));
}
[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("dist transport: " + what); }

const std::array<std::uint32_t, 256>& crc_table() {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256U; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1U ? 0x82f63b78U ^ (c >> 1U) : c >> 1U;
            t[i] = c;
        }
        return t;
    }();
    return table;
}
#if defined(__x86_64__)
__attribute__((target("sse4.2"))) std::uint32_t crc32c_sse42(std::span<const std::byte> bytes) noexcept {
    std::uint64_t c = 0xffffffffU;
    std::size_t i = 0;
    for (; i + 8U <= bytes.size(); i += 8U) {
        std::uint64_t word;
        std::memcpy(&word, bytes.data() + i, 8U);
        c = __builtin_ia32_crc32di(c, word);
    }
    auto c32 = static_cast<std::uint32_t>(c);
    for (; i < bytes.size(); ++i) c32 = __builtin_ia32_crc32qi(c32, static_cast<unsigned char>(bytes[i]));
    return c32 ^ 0xffffffffU;
}
#endif
inline void cpu_relax() noexcept {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}

// In-process pair: each direction is a queue of whole messages.
struct Channel {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::vector<std::byte>> messages;
};
class MemoryTransport final : public Transport {
public:
    MemoryTransport(std::shared_ptr<Channel> out, std::shared_ptr<Channel> in) : out_(std::move(out)), in_(std::move(in)) {}
    void send(std::span<const std::byte> message) override {
        {
            std::lock_guard lock(out_->mutex);
            out_->messages.emplace_back(message.begin(), message.end());
        }
        out_->ready.notify_one();
    }
    void receive(std::span<std::byte> message) override {
        std::unique_lock lock(in_->mutex);
        if (!in_->ready.wait_for(lock, std::chrono::seconds(30), [&] { return !in_->messages.empty(); }))
            fail("in-process receive timed out");
        auto front = std::move(in_->messages.front());
        in_->messages.pop_front();
        if (front.size() != message.size()) fail("in-process message size differs from the expected size");
        std::memcpy(message.data(), front.data(), front.size());
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        send(out);
        receive(in);
    }
    [[nodiscard]] std::string name() const override { return "memory"; }

private:
    std::shared_ptr<Channel> out_, in_;
};

// Runs one job at a time on a persistent thread (the progress thread of a transport or stream). After a
// job it spins briefly for the next one (the exchanges of consecutive layers come close together), then
// sleeps; the caller spins for completion.
class Worker {
public:
    Worker() : thread_([this] { run(); }) {}
    ~Worker() {
        {
            std::lock_guard lock(mutex_);
            stop_.store(true, std::memory_order_release);
        }
        wake_.notify_all();
        thread_.join();
    }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    void post(std::function<void()> job) {
        job_ = std::move(job);
        error_ = nullptr;
        {
            std::lock_guard lock(mutex_);
            posted_.fetch_add(1U, std::memory_order_release);
        }
        wake_.notify_one();
    }
    void wait() {
        const auto target = posted_.load(std::memory_order_relaxed);
        while (finished_.load(std::memory_order_acquire) != target) cpu_relax();
        if (error_) std::rethrow_exception(error_);
    }

private:
    void run() {
        std::uint64_t seen = 0;
        while (true) {
            const auto spin_end = std::chrono::steady_clock::now() + std::chrono::microseconds(300);
            while (posted_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < spin_end)
                cpu_relax();
            if (posted_.load(std::memory_order_acquire) == seen) {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stop_.load(std::memory_order_acquire) || posted_.load(std::memory_order_acquire) != seen; });
            }
            if (stop_.load(std::memory_order_acquire)) return;
            ++seen;
            try { job_(); } catch (...) { error_ = std::current_exception(); }
            finished_.store(seen, std::memory_order_release);
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_;
    std::function<void()> job_;
    std::exception_ptr error_;
    std::atomic<std::uint64_t> posted_{0}, finished_{0};
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

void write_all(int fd, std::span<const std::byte> bytes, const char* what) {
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) { cpu_relax(); continue; }
        if (n <= 0) fail(std::string(what) + " write failed: " + std::strerror(errno));
        done += static_cast<std::size_t>(n);
    }
}
void read_all(int fd, std::span<std::byte> bytes, const char* what) {
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto n = ::read(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) { cpu_relax(); continue; }
        if (n == 0) fail(std::string(what) + " closed");
        if (n < 0) fail(std::string(what) + " read failed: " + std::strerror(errno));
        done += static_cast<std::size_t>(n);
    }
}

std::string errno_text(int error) {
    switch (error) {
    case ENODEV: return "ENODEV: the stream device is gone (link unplugged or controller reset)";
    case ETIMEDOUT: return "ETIMEDOUT: the driver timed out (peer or domain gone)";
    case ENXIO: return "ENXIO: no stream attached (peer service gone)";
    case ENOMEM: return "ENOMEM: the driver could not allocate the stream rings (memory pressure)";
    case EPIPE: return "EPIPE: the peer closed the stream (its rank failed or exited)";
    case ECONNRESET: return "ECONNRESET: the peer closed the stream (its rank failed or exited)";
    default: return std::strerror(error);
    }
}

// Host path of USB4 streams. Each stream has a persistent writer and a persistent reader, so a stream's
// part of a message is written and read at the same time and messages of any size go through (a writer
// that ran ahead of its reader blocked once a part exceeded what the two rings buffer). Every transfer
// waits in poll() slices with a deadline and an abort flag: when one stream fails (ENODEV, ETIMEDOUT,
// the peer's CLOSE, the deadline), the other transfers stop, every stream is closed (the peer sees CLOSE
// on its open streams and fails at once instead of waiting) and the transport refuses further use.
// Messages are split in proportion to integer weights at 64-byte boundaries; messages shorter than
// `min_part` bytes per stream use fewer streams (the first k of the order, which alternates cables).
class Usb4HostTransport final : public Transport {
public:
    Usb4HostTransport(std::vector<int> fds, std::vector<std::string> names, std::vector<std::uint32_t> weights,
                      std::size_t min_part = 0, std::chrono::milliseconds timeout = std::chrono::milliseconds(120000))
        : names_(std::move(names)), fds_(std::move(fds)), weights_(std::move(weights)), min_part_(min_part), timeout_(timeout) {
        if (fds_.empty() || fds_.size() > 8U || names_.size() != fds_.size() || weights_.size() != fds_.size()) {
            for (const int fd : fds_) ::close(fd);
            fail("usb4 host transport takes one to eight streams, each with a name and a weight");
        }
        for (const auto w : weights_)
            if (w == 0U) {
                for (const int fd : fds_) ::close(fd);
                fail("usb4 host transport weights must be positive");
            }
        if (const char* t = std::getenv("STRIX_DIST_USB4_TIMEOUT_MS"); t && *t) timeout_ = std::chrono::milliseconds(std::stoll(t));
        for (std::size_t i = 0; i < fds_.size(); ++i) {
            writers_.push_back(std::make_unique<Worker>());
            readers_.push_back(std::make_unique<Worker>());
        }
    }
    ~Usb4HostTransport() override {
        aborted_.store(true, std::memory_order_release);
        writers_.clear();
        readers_.clear();
        close_all();
    }
    void send(std::span<const std::byte> message) override { run(&message, nullptr); }
    void receive(std::span<std::byte> message) override { run(nullptr, &message); }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        if (out.size() != in.size()) fail("usb4 exchange: both directions carry the same size");
        run(&out, &in);
    }
    [[nodiscard]] std::string name() const override {
        std::string out = "usb4-host:";
        for (std::size_t k = 0; k < names_.size(); ++k)
            out += (k ? "," : "") + names_[k] + (fds_.size() > 1U ? "*" + std::to_string(weights_[k]) : "");
        if (min_part_) out += " (min part " + std::to_string(min_part_) + " B)";
        return out;
    }

private:
    // Streams used for a message of `size` bytes, and stream k's part of it.
    [[nodiscard]] std::size_t streams_for(std::size_t size) const {
        if (!min_part_) return fds_.size();
        return std::clamp<std::size_t>(size / min_part_, 1U, fds_.size());
    }
    template <class Span> Span part(Span message, std::size_t k, std::size_t used) const {
        std::uint64_t total = 0, before = 0, through = 0;
        for (std::size_t j = 0; j < used; ++j) {
            total += weights_[j];
            if (j < k) before += weights_[j];
            if (j <= k) through += weights_[j];
        }
        const std::uint64_t size = message.size();
        const auto begin = k == 0U ? 0U : (size * before / total) & ~std::uint64_t{63};
        const auto end = k + 1U == used ? size : (size * through / total) & ~std::uint64_t{63};
        return message.subspan(begin, end - begin);
    }
    void close_all() {
        for (auto& fd : fds_)
            if (fd >= 0) { ::close(fd); fd = -1; }
    }
    // Waits until `fd` is ready for `events` (blocking descriptors) in short slices, so an abort or the
    // deadline stops the wait.
    void wait_ready(std::size_t k, short events, std::chrono::steady_clock::time_point deadline, const char* what) {
        if (spin_) return;
        while (true) {
            if (aborted_.load(std::memory_order_acquire)) throw Aborted{};
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) fail_stream(k, std::string(what) + ": no progress within " + std::to_string(timeout_.count()) + " ms");
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            pollfd p{fds_[k], events, 0};
            const int ready = ::poll(&p, 1, static_cast<int>(std::min<long long>(left, 50)));
            if (ready < 0) {
                if (errno == EINTR) continue;
                fail_stream(k, std::string(what) + ": poll: " + errno_text(errno));
            }
            if (ready == 0) continue;
            if (p.revents & POLLNVAL) fail_stream(k, std::string(what) + ": descriptor closed");
            return; // ready, or an error/hang-up that the read or write reports precisely
        }
    }
    void write_part(std::size_t k, std::span<const std::byte> bytes) {
        std::size_t done = 0;
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        while (done < bytes.size()) {
            wait_ready(k, POLLOUT, deadline, "write");
            const auto n = ::write(fds_[k], bytes.data() + done, bytes.size() - done);
            if (n > 0) { done += static_cast<std::size_t>(n); deadline = std::chrono::steady_clock::now() + timeout_; continue; }
            if (n < 0 && (errno == EINTR || errno == EAGAIN)) {
                if (aborted_.load(std::memory_order_acquire)) throw Aborted{};
                if (spin_) cpu_relax();
                if (std::chrono::steady_clock::now() >= deadline) fail_stream(k, "write: no progress within " + std::to_string(timeout_.count()) + " ms");
                continue;
            }
            fail_stream(k, n == 0 ? std::string("write returned 0") : "write: " + errno_text(errno));
        }
    }
    void read_part(std::size_t k, std::span<std::byte> bytes) {
        std::size_t done = 0;
        auto deadline = std::chrono::steady_clock::now() + timeout_;
        while (done < bytes.size()) {
            wait_ready(k, POLLIN, deadline, "read");
            const auto n = ::read(fds_[k], bytes.data() + done, bytes.size() - done);
            if (n > 0) { done += static_cast<std::size_t>(n); deadline = std::chrono::steady_clock::now() + timeout_; continue; }
            if (n == 0) fail_stream(k, "the peer closed the stream (its rank failed or exited)");
            if (errno == EINTR || errno == EAGAIN) {
                if (aborted_.load(std::memory_order_acquire)) throw Aborted{};
                if (spin_) cpu_relax();
                if (std::chrono::steady_clock::now() >= deadline) fail_stream(k, "read: no progress within " + std::to_string(timeout_.count()) + " ms");
                continue;
            }
            fail_stream(k, "read: " + errno_text(errno));
        }
    }
    struct Aborted {};
    [[noreturn]] void fail_stream(std::size_t k, const std::string& what) {
        const std::string text = "usb4 stream " + names_[k] + ": " + what;
        {
            std::lock_guard lock(root_mutex_);
            if (root_.empty()) root_ = text; // the first failure is the cause; the others follow from it
        }
        aborted_.store(true, std::memory_order_release);
        fail(text);
    }
    void run(const std::span<const std::byte>* out, const std::span<std::byte>* in) {
        if (poisoned_) fail("usb4 host transport: unusable after an earlier stream failure (" + poison_ + ")");
        const std::size_t size = out ? out->size() : in->size();
        const std::size_t used = streams_for(size);
        for (std::size_t k = 0; k < used; ++k) {
            if (out) {
                const auto piece = part(*out, k, used);
                writers_[k]->post([this, k, piece] { guarded([&] { write_part(k, piece); }); });
            }
            if (in) {
                const auto piece = part(*in, k, used);
                readers_[k]->post([this, k, piece] { guarded([&] { read_part(k, piece); }); });
            }
        }
        std::exception_ptr first;
        for (std::size_t k = 0; k < used; ++k) {
            if (out) try { writers_[k]->wait(); } catch (...) { if (!first) first = std::current_exception(); }
            if (in) try { readers_[k]->wait(); } catch (...) { if (!first) first = std::current_exception(); }
        }
        if (aborted_.load(std::memory_order_acquire)) {
            // One stream failed: close every stream now (the peer reads CLOSE on the live ones and fails at
            // once), and keep the reason for any later call.
            close_all();
            poisoned_ = true;
            {
                std::lock_guard lock(root_mutex_);
                poison_ = root_.empty() ? std::string("aborted") : root_;
            }
            fail(poison_ + " (all streams closed; the peer rank sees them close)");
        }
        if (first) std::rethrow_exception(first);
    }
    template <class F> void guarded(F&& f) {
        try { f(); } catch (const Aborted&) {
            throw std::runtime_error("dist transport: usb4 transfer stopped because another stream failed");
        }
    }
    std::vector<std::string> names_;
    std::vector<int> fds_;
    std::vector<std::uint32_t> weights_;
    std::size_t min_part_ = 0;
    std::chrono::milliseconds timeout_;
    const bool spin_ = [] { const char* v = std::getenv("STRIX_DIST_USB4_SPIN"); return v && v[0] == '1'; }();
    std::atomic<bool> aborted_{false};
    bool poisoned_ = false;
    std::mutex root_mutex_;
    std::string poison_, root_;
    std::vector<std::unique_ptr<Worker>> writers_, readers_;
};

bool spin_streams() {
    // STRIX_DIST_USB4_SPIN=1: non-blocking descriptors polled by the stream threads.
    static const bool spin = [] { const char* v = std::getenv("STRIX_DIST_USB4_SPIN"); return v && v[0] == '1'; }();
    return spin;
}
void set_blocking(int fd, bool blocking) {
    const int flags = ::fcntl(fd, F_GETFL);
    if (flags < 0 || ::fcntl(fd, F_SETFL, blocking ? flags & ~O_NONBLOCK : flags | O_NONBLOCK) < 0)
        fail(std::string("fcntl failed: ") + std::strerror(errno));
}

// ---- Automatic pairing -------------------------------------------------------------------------------
constexpr std::uint64_t hello_magic = 0x4f4c4548584e5453ULL; // "STNXHELO"
constexpr std::uint64_t plan_magic = 0x4e414c50584e5453ULL;  // "STNXPLAN"
constexpr std::uint16_t pairing_version = 2;
constexpr std::size_t max_streams = 8;

struct Hello {
    std::uint64_t magic = hello_magic;
    std::uint64_t nonce = 0;
    std::uint64_t peer_nonce = 0;     // acknowledgement: the peer's nonce
    std::uint16_t version = pairing_version;
    std::uint16_t kind = 0;           // 1 hello, 2 acknowledgement
    std::uint32_t rank = 0;
    std::uint32_t stream = 0;         // the sender's candidate index
    std::uint32_t peer_stream = 0;    // acknowledgement: the peer's candidate index
    std::int32_t device = -1;         // the sender's tbstream number
    std::uint32_t generation = 0;     // the sender's open count of this stream
    std::uint32_t peer_generation = 0; // acknowledgement: the peer's generation
    char host[8] = {};
    std::uint32_t crc = 0;
};
static_assert(sizeof(Hello) == 64);
struct Plan {
    std::uint64_t magic = plan_magic;
    std::uint32_t version = pairing_version;
    std::uint32_t count = 0;
    std::uint32_t index[max_streams] = {};  // rank 0's candidate indices, in striping order
    std::uint32_t weight[max_streams] = {};
    std::uint32_t min_part = 0;  // bytes per stream below which a message uses fewer streams (rank 0's value)
    std::uint32_t crc = 0;
    std::uint32_t pad[10] = {};
};
static_assert(sizeof(Plan) == 128);

template <class T> std::uint32_t body_crc(const T& value) {
    return crc32c(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&value), offsetof(T, crc)));
}
std::string read_text(const std::string& path) {
    std::ifstream in(path);
    std::string text;
    std::getline(in, text);
    while (!text.empty() && (text.back() == '\n' || text.back() == ' ' || text.back() == '\r')) text.pop_back();
    return text;
}
bool exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}
double leading_number(const std::string& text) {
    try { return text.empty() ? 0.0 : std::stod(text); } catch (...) { return 0.0; }
}
std::int32_t trailing_number(const std::string& text) {
    std::size_t i = text.size();
    while (i > 0U && text[i - 1U] >= '0' && text[i - 1U] <= '9') --i;
    return i == text.size() ? -1 : static_cast<std::int32_t>(std::stol(text.substr(i)));
}
std::string host_name() {
    char name[256] = {};
    if (::gethostname(name, sizeof(name) - 1U) != 0) return "?";
    return name;
}

struct Candidate {
    Usb4StreamPair info;
    std::string service;
    int fd = -1;
    bool dead = false;
    // handshake state of the current open (generation)
    std::uint32_t generation = 0;
    std::chrono::steady_clock::time_point opened_at{}, last_hello{}, reopen_at{};
    std::array<std::byte, 64> in{};
    std::size_t have = 0;
    std::array<std::byte, 128> out{};
    std::size_t out_size = 0, out_done = 0;
    bool hello_ok = false;     // the peer's hello for this generation was received
    bool ack_ok = false;       // the peer acknowledged our hello of this generation
    bool reply_queued = false; // rank 0: hello + acknowledgement queued; rank 1: acknowledgement queued
    bool reply_written = false;
    Hello peer;
    std::size_t garbage = 0, reopens = 0;
    std::string last_error;
    std::vector<std::string> trace; // STRIX_DIST_USB4_DEBUG=1
};

std::vector<Candidate> layout_candidates(const Usb4AutoOptions& options, std::vector<std::string>& rejected) {
    std::string path = options.layout;
    if (path.empty()) {
        const char* env = std::getenv("STRIX_DIST_USB4_LAYOUT");
        path = env && *env ? env : "/run/usb4stream-layout.json";
    }
    std::ifstream file(path);
    if (!file) fail("cannot read the USB4 stream layout " + path);
    std::stringstream text;
    text << file.rdbuf();
    const auto layout = strix::serve::parse_json(text.str());
    if (!layout.is_object()) fail("the USB4 stream layout " + path + " is not an object");
    std::vector<std::string> exclude = options.exclude;
    if (const char* env = std::getenv("STRIX_DIST_USB4_EXCLUDE"); env && *env) {
        std::stringstream list(env);
        for (std::string item; std::getline(list, item, ',');) if (!item.empty()) exclude.push_back(item);
    }
    std::vector<Candidate> out;
    for (const auto& [key, entry] : layout.as_object()) {
        const auto slash = key.find('/');
        if (slash == std::string::npos || !entry.is_object()) continue; // "version", "topology"
        const auto* device = entry.find("device");
        if (!device || !device->is_string()) { rejected.push_back(key + ": no device in the layout"); continue; }
        Candidate c;
        c.info.key = key;
        c.info.device = device->as_string();
        c.service = key.substr(0, slash);
        if (std::find(exclude.begin(), exclude.end(), key) != exclude.end() ||
            std::find(exclude.begin(), exclude.end(), c.info.device) != exclude.end()) {
            rejected.push_back(key + " " + c.info.device + ": excluded");
            continue;
        }
        const auto reason = [&](const char* field) {
            const auto* r = entry.find(field);
            return r && r->is_string() ? " (" + r->as_string() + ")" : std::string();
        };
        // sysfs decides: the layout is written at one moment (boot, a probe) and its attached/openable
        // flags can be stale in either direction. Never open a stream whose service has no peer now
        // (its controller or link may be down).
        if (!exists(options.sysfs + "/" + c.service)) {
            const auto* attached = entry.find("attached");
            const bool layout_says = attached && attached->is_bool() && !attached->as_bool();
            rejected.push_back(key + " " + c.info.device + ": service " + c.service + " is not in " + options.sysfs +
                               " (no peer on that port)" + (layout_says ? "; the layout agrees" + reason("openable_reason") : std::string()));
            continue;
        }
        if (!exists(c.info.device)) { rejected.push_back(key + ": " + c.info.device + " does not exist"); continue; }
        const auto dot = c.service.rfind('.');
        c.info.xdomain = dot == std::string::npos ? c.service : c.service.substr(0, dot);
        const auto dash = c.info.xdomain.find('-');
        c.info.domain = dash == std::string::npos ? -1 : static_cast<std::int32_t>(leading_number(c.info.xdomain.substr(0, dash)));
        if (const auto* cable = entry.find("cable"); cable && cable->is_string()) c.info.cable = cable->as_string();
        if (const auto* local = entry.find("local"); local && local->is_object())
            if (const auto* port = local->find("usb4_port"); port && port->is_number()) c.info.usb4_port = static_cast<std::int32_t>(port->as_int());
        const auto x = options.sysfs + "/" + c.info.xdomain + "/";
        const double rx = leading_number(read_text(x + "rx_speed")) * std::max(1.0, leading_number(read_text(x + "rx_lanes")));
        const double tx = leading_number(read_text(x + "tx_speed")) * std::max(1.0, leading_number(read_text(x + "tx_lanes")));
        c.info.link_gbps = std::min(rx, tx);
        out.push_back(std::move(c));
    }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.info.key < b.info.key; });
    if (out.size() > max_streams) fail("more than " + std::to_string(max_streams) + " USB4 streams in the layout");
    return out;
}

bool pairing_debug() {
    static const bool debug = [] { const char* v = std::getenv("STRIX_DIST_USB4_DEBUG"); return v && v[0] == '1'; }();
    return debug;
}
void note(Candidate& c, const std::string& what) {
    if (!pairing_debug()) return;
    static const auto origin = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - origin).count();
    c.trace.push_back(std::to_string(us) + "us " + what);
}
std::string describe(const Hello& h) {
    std::ostringstream out;
    out << (h.kind == 1U ? "hello" : h.kind == 2U ? "ack" : "kind?") << " rank " << h.rank << " stream " << h.stream << " gen "
        << h.generation << " nonce " << std::hex << (h.nonce & 0xffffU) << " peer_nonce " << (h.peer_nonce & 0xffffU) << std::dec
        << " peer_stream " << h.peer_stream << " peer_gen " << h.peer_generation;
    return out.str();
}

// The handshake. With the stream module in use, frames written before the peer has opened its end can
// hold back the later frames of that direction, so rank 0 never writes first: rank 1 sends hellos
// (every 250 ms) and, without an answer within 1.5 s of opening, closes the stream, waits 0.5 s (so
// that the CLOSE rank 0 sends while reopening finds no open end here) and opens it again with a new
// generation. Rank 0 answers each new (nonce, generation) hello with its hello and an acknowledgement
// in one write, and reopens at once when it reads the peer's CLOSE. Rank 1 acknowledges rank 0's hello.
// Every frame either side writes is consumed by the other inside the handshake, so the stream is clean
// afterwards.
struct Handshake {
    std::uint32_t rank;
    std::uint64_t nonce;
    std::string host;
    std::chrono::milliseconds answer{1500}, closed_wait{500}, resend{250};
};
Hello make(const Handshake& hs, const Candidate& c, std::uint32_t index, std::uint16_t kind) {
    Hello h;
    h.nonce = hs.nonce; h.rank = hs.rank; h.stream = index; h.kind = kind; h.device = trailing_number(c.info.device);
    h.generation = c.generation;
    std::memcpy(h.host, hs.host.data(), std::min(hs.host.size(), sizeof(h.host)));
    if (kind == 2U) { h.peer_nonce = c.peer.nonce; h.peer_stream = c.peer.stream; h.peer_generation = c.peer.generation; }
    h.crc = body_crc(h);
    return h;
}
void queue(Candidate& c, const Hello& first, const Hello* second = nullptr) {
    std::memcpy(c.out.data(), &first, sizeof(first));
    c.out_size = sizeof(first);
    if (second) { std::memcpy(c.out.data() + sizeof(first), second, sizeof(*second)); c.out_size += sizeof(*second); }
    c.out_done = 0;
}
void open_stream(Candidate& c) {
    c.fd = ::open(c.info.device.c_str(), O_RDWR | O_CLOEXEC | O_NONBLOCK);
    if (c.fd < 0) {
        c.dead = true;
        c.last_error = std::string("open failed: ") + errno_text(errno);
        return;
    }
    ++c.generation;
    c.opened_at = std::chrono::steady_clock::now();
    c.last_hello = {};
    c.have = 0; c.out_size = c.out_done = 0;
    c.hello_ok = c.ack_ok = c.reply_queued = c.reply_written = false;
    note(c, "opened generation " + std::to_string(c.generation));
}
void close_stream(Candidate& c, const std::string& why) {
    note(c, "closing: " + why);
    if (c.fd >= 0) ::close(c.fd);
    c.fd = -1;
    ++c.reopens;
}
bool handshake_done(const Candidate& c, std::uint32_t rank) {
    return rank == 0U ? c.reply_written && c.ack_ok : c.hello_ok && c.ack_ok && c.reply_written;
}
// One non-blocking step on one stream; returns true if anything moved.
bool handshake_step(Candidate& c, std::uint32_t index, const Handshake& hs) {
    const auto now = std::chrono::steady_clock::now();
    if (c.fd < 0) {
        if (hs.rank == 0U || now >= c.reopen_at) open_stream(c);
        return c.fd >= 0;
    }
    const auto lost = [&](const std::string& why) {
        close_stream(c, why);
        if (hs.rank == 0U) open_stream(c);
        else c.reopen_at = std::chrono::steady_clock::now() + hs.closed_wait;
        return true;
    };
    bool moved = false;
    // Rank 1: hellos until rank 0 answers, then a fresh open if it does not.
    if (hs.rank == 1U && !(c.hello_ok && c.ack_ok)) {
        if (now - c.opened_at > hs.answer) return lost("no answer within " + std::to_string(hs.answer.count()) + " ms");
        if (c.out_done == c.out_size && now - c.last_hello >= hs.resend) {
            queue(c, make(hs, c, index, 1U));
            c.last_hello = now;
        }
    }
    if (c.out_done == c.out_size && c.hello_ok && !c.reply_queued && (hs.rank == 1U ? c.ack_ok : true)) {
        if (hs.rank == 0U) {
            const auto hello = make(hs, c, index, 1U), ack = make(hs, c, index, 2U);
            queue(c, hello, &ack);
        } else {
            queue(c, make(hs, c, index, 2U));
        }
        c.reply_queued = true;
    }
    if (c.out_done < c.out_size) {
        const auto n = ::write(c.fd, c.out.data() + c.out_done, c.out_size - c.out_done);
        if (n > 0) {
            c.out_done += static_cast<std::size_t>(n);
            moved = true;
            if (c.out_done == c.out_size) {
                if (c.reply_queued && !c.reply_written) { c.reply_written = true; note(c, hs.rank == 0U ? "wrote hello + ack" : "wrote ack"); }
                else note(c, "wrote hello");
            }
        } else if (n < 0 && errno != EAGAIN && errno != EINTR) {
            c.last_error = std::string("write: ") + std::strerror(errno);
            return lost(c.last_error);
        }
    }
    const auto n = ::read(c.fd, c.in.data() + c.have, c.in.size() - c.have);
    if (n > 0) { c.have += static_cast<std::size_t>(n); moved = true; }
    else if (n == 0) { c.last_error = "peer closed the stream"; return lost("read end of stream (peer CLOSE)"); }
    else if (errno != EAGAIN && errno != EINTR) { c.last_error = std::string("read: ") + std::strerror(errno); return lost(c.last_error); }
    if (c.have == c.in.size()) {
        c.have = 0;
        Hello h;
        std::memcpy(&h, c.in.data(), sizeof(h));
        if (h.magic != hello_magic || h.crc != body_crc(h) || h.version != pairing_version) {
            ++c.garbage; // stale bytes of an earlier session (whole 64-byte units)
            note(c, "read garbage unit");
        } else if (h.kind == 1U) {
            const bool fresh = !c.hello_ok || h.nonce != c.peer.nonce || h.generation != c.peer.generation;
            note(c, "read " + describe(h) + (fresh ? "" : " (repeat)"));
            if (fresh) {
                c.peer = h;
                c.hello_ok = true;
                if (hs.rank == 0U) { c.reply_queued = c.reply_written = c.ack_ok = false; }
            }
        } else if (h.kind == 2U && c.hello_ok && h.nonce == c.peer.nonce && h.generation == c.peer.generation &&
                   h.peer_nonce == hs.nonce && h.peer_stream == index && h.peer_generation == c.generation) {
            note(c, "read " + describe(h) + " (accepted)");
            c.ack_ok = true;
        } else {
            note(c, "read " + describe(h) + " (not for this open)");
            ++c.garbage;
        }
    }
    return moved;
}

void write_plan(int fd, const Plan& plan) {
    write_all(fd, std::span<const std::byte>(reinterpret_cast<const std::byte*>(&plan), sizeof(plan)), "usb4 plan");
}
Plan check_plan(const Plan& plan) {
    if (plan.magic != plan_magic || plan.crc != body_crc(plan) || plan.version != pairing_version || plan.count > max_streams ||
        plan.count == 0U)
        fail("usb4 pairing: malformed plan from the peer");
    return plan;
}
// Pairing reads have a deadline (poll), so a peer that died or a link that went down mid-pairing fails
// the pairing instead of blocking it.
constexpr std::chrono::milliseconds plan_wait{20000};
Plan read_plan(int fd) {
    Plan plan;
    auto* bytes = reinterpret_cast<std::byte*>(&plan);
    std::size_t done = 0;
    const auto deadline = std::chrono::steady_clock::now() + plan_wait;
    while (done < sizeof(plan)) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (left <= 0) fail("usb4 pairing: no plan from the peer within " + std::to_string(plan_wait.count()) + " ms");
        pollfd p{fd, POLLIN, 0};
        const int ready = ::poll(&p, 1, static_cast<int>(left));
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0 || (p.revents & (POLLERR | POLLHUP))) fail("usb4 pairing: the stream failed while waiting for the plan");
        if (ready == 0) continue;
        const auto n = ::read(fd, bytes + done, sizeof(plan) - done);
        if (n == 0) fail("usb4 pairing: the peer closed the stream before the plan");
        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            fail(std::string("usb4 pairing: plan read failed: ") + std::strerror(errno));
        }
        done += static_cast<std::size_t>(n);
    }
    return check_plan(plan);
}
bool same_plan(const Plan& a, const Plan& b) {
    if (a.count != b.count || a.min_part != b.min_part) return false;
    for (std::uint32_t i = 0; i < a.count; ++i)
        if (a.index[i] != b.index[i] || a.weight[i] != b.weight[i]) return false;
    return true;
}
// Rank 0 sends the plan on its first stream; rank 1 echoes it after checking it.
void agree(int fd, std::uint32_t rank, Plan& plan) {
    plan.crc = body_crc(plan);
    if (rank == 0U) {
        write_plan(fd, plan);
        if (!same_plan(read_plan(fd), plan)) fail("usb4 pairing: the peer did not echo the plan");
    } else {
        const auto theirs = read_plan(fd);
        bool ok = theirs.count == plan.count;
        for (std::uint32_t i = 0; ok && i < plan.count; ++i) ok = theirs.index[i] == plan.index[i];
        write_plan(fd, theirs);
        if (!ok) fail("usb4 pairing: the two ranks disagree on the streams");
        plan = theirs;
    }
}

double probe_exchange(const std::vector<int>& fds, std::size_t bytes, std::uint32_t rank) {
    const std::size_t count = fds.size();
    std::vector<std::vector<std::byte>> out(count), in(count);
    for (std::size_t k = 0; k < count; ++k) {
        const std::size_t size = (bytes / count) & ~std::size_t{63};
        out[k].resize(size);
        in[k].resize(size);
        for (std::size_t i = 0; i < size; ++i) out[k][i] = static_cast<std::byte>((i * 131U + k * 7U + rank) & 0xffU);
    }
    const auto start = std::chrono::steady_clock::now();
    // A writer and a reader per stream: the probe is larger than what a stream may buffer.
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(2U * count);
    for (std::size_t k = 0; k < count; ++k) {
        threads.emplace_back([&, k] {
            try { write_all(fds[k], out[k], "usb4 probe"); } catch (...) { errors[2U * k] = std::current_exception(); }
        });
        threads.emplace_back([&, k] {
            try { read_all(fds[k], in[k], "usb4 probe"); } catch (...) { errors[2U * k + 1U] = std::current_exception(); }
        });
    }
    for (auto& t : threads) t.join();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (const auto& e : errors)
        if (e) std::rethrow_exception(e);
    for (std::size_t k = 0; k < count; ++k)
        for (std::size_t i = 0; i < in[k].size(); ++i)
            if (in[k][i] != static_cast<std::byte>((i * 131U + k * 7U + (1U - rank)) & 0xffU))
                fail("usb4 probe: corrupted bytes on stream " + std::to_string(k));
    return seconds;
}

class TcpTransport final : public Transport {
public:
    TcpTransport(int fd, std::chrono::milliseconds timeout) : fd_(fd), timeout_(timeout) {
        const int one = 1;
        ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    ~TcpTransport() override {
        sender_.reset();
        ::close(fd_);
    }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        sender_->post([this, out] { send(out); });
        std::exception_ptr error;
        try { receive(in); } catch (...) { error = std::current_exception(); }
        sender_->wait();
        if (error) std::rethrow_exception(error);
    }
    void send(std::span<const std::byte> message) override {
        std::size_t done = 0;
        while (done < message.size()) {
            wait_for(POLLOUT, "send");
            const auto n = ::send(fd_, message.data() + done, message.size() - done, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) fail(std::string("send failed: ") + std::strerror(errno));
            done += static_cast<std::size_t>(n);
        }
    }
    void receive(std::span<std::byte> message) override {
        std::size_t done = 0;
        while (done < message.size()) {
            wait_for(POLLIN, "receive");
            const auto n = ::recv(fd_, message.data() + done, message.size() - done, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n == 0) fail("peer closed the connection");
            if (n < 0) fail(std::string("receive failed: ") + std::strerror(errno));
            done += static_cast<std::size_t>(n);
        }
    }
    [[nodiscard]] std::string name() const override { return "tcp"; }

private:
    void wait_for(short events, const char* what) const {
        pollfd p{fd_, events, 0};
        const int ready = ::poll(&p, 1, static_cast<int>(timeout_.count()));
        if (ready == 0) fail(std::string(what) + " timed out");
        if (ready < 0 && errno != EINTR) fail(std::string(what) + " poll failed: " + std::strerror(errno));
    }
    int fd_;
    std::chrono::milliseconds timeout_;
    std::unique_ptr<Worker> sender_ = std::make_unique<Worker>();
};

} // namespace

std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept {
#if defined(__x86_64__)
    static const bool hardware = __builtin_cpu_supports("sse4.2");
    if (hardware) return crc32c_sse42(bytes);
#endif
    const auto& table = crc_table();
    std::uint32_t c = 0xffffffffU;
    for (const auto b : bytes) c = table[(c ^ static_cast<std::uint8_t>(b)) & 0xffU] ^ (c >> 8U);
    return c ^ 0xffffffffU;
}

std::pair<std::unique_ptr<Transport>, std::unique_ptr<Transport>> memory_pair() {
    auto a = std::make_shared<Channel>(), b = std::make_shared<Channel>();
    return {std::make_unique<MemoryTransport>(a, b), std::make_unique<MemoryTransport>(b, a)};
}

std::unique_ptr<Transport> usb4_host(const std::vector<std::string>& devices, const std::vector<std::uint32_t>& weights) {
    if (devices.empty() || devices.size() > max_streams) fail("usb4 host transport takes one to eight stream devices");
    if (!weights.empty() && weights.size() != devices.size()) fail("usb4 host transport: one weight per device");
    std::vector<int> fds;
    for (const auto& path : devices) {
        // Non-blocking open: a stream whose link is down fails at once (ENXIO) instead of waiting for it.
        const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) {
            for (const int open_fd : fds) ::close(open_fd);
            fail("cannot open " + path + ": " + std::strerror(errno));
        }
        fds.push_back(fd);
        if (!spin_streams()) set_blocking(fd, true);
    }
    return std::make_unique<Usb4HostTransport>(std::move(fds), devices,
                                               weights.empty() ? std::vector<std::uint32_t>(devices.size(), 1U) : weights);
}

std::unique_ptr<Transport> usb4_host_descriptors(std::vector<int> fds, std::vector<std::string> names, std::vector<std::uint32_t> weights,
                                                 std::size_t min_part, std::chrono::milliseconds timeout) {
    return std::make_unique<Usb4HostTransport>(std::move(fds), std::move(names), std::move(weights), min_part, timeout);
}

std::string Usb4Topology::receipt() const {
    std::ostringstream out;
    std::map<std::string, int> cables;
    for (const auto& s : streams) cables[s.xdomain] += 1;
    out << "usb4 auto-pairing: " << local_host << " <-> " << remote_host;
    if (!remote_unique_id.empty()) out << " (" << remote_unique_id << ")";
    out << ", " << streams.size() << " stream(s) on " << cables.size() << " cable(s)";
    if (min_part_bytes) out << ", messages below " << min_part_bytes << " B per stream use fewer streams";
    for (const auto& s : streams) {
        out << "; " << s.key << ' ' << s.device << " [domain " << s.domain;
        if (s.usb4_port >= 0) out << " port " << s.usb4_port;
        if (!s.cable.empty()) out << " cable " << s.cable;
        out << ", link " << s.link_gbps << " Gb/s] <-> peer tbstream" << s.remote_device << " (index " << s.remote_index
            << "), cable " << s.xdomain << " measured " << s.measured_gbps << " Gb/s, weight " << s.weight;
    }
    for (const auto& r : rejected) out << "; not used: " << r;
    return out.str();
}

std::unique_ptr<Transport> usb4_host_auto(const Usb4AutoOptions& options, Usb4Topology* topology) {
    if (options.rank > 1U) fail("usb4 pairing: rank must be 0 or 1");
    Usb4Topology topo;
    topo.local_host = host_name();
    auto candidates = layout_candidates(options, topo.rejected);
    std::random_device random;
    Handshake hs;
    hs.rank = options.rank;
    hs.host = topo.local_host;
    hs.nonce = (static_cast<std::uint64_t>(random()) << 32U) ^ random() ^
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^ static_cast<std::uint64_t>(::getpid());
    for (auto& c : candidates) open_stream(c);
    const auto close_all = [&] { for (auto& c : candidates) if (c.fd >= 0) { ::close(c.fd); c.fd = -1; } };
    const auto start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point first_done{};
    while (true) {
        bool moved = false, all = true;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            auto& c = candidates[i];
            if (c.dead) continue;
            if (!handshake_done(c, hs.rank)) moved = handshake_step(c, static_cast<std::uint32_t>(i), hs) || moved;
            if (c.dead) continue;
            if (handshake_done(c, hs.rank)) { if (first_done == std::chrono::steady_clock::time_point{}) first_done = std::chrono::steady_clock::now(); }
            else all = false;
        }
        const auto now = std::chrono::steady_clock::now();
        if (all) break;
        if (first_done != std::chrono::steady_clock::time_point{} ? now - first_done > options.grace : now - start > options.timeout) break;
        if (!moved) std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (pairing_debug())
        for (const auto& c : candidates) {
            std::fprintf(stderr, "usb4 pairing trace %s %s (rank %u, nonce %llx):\n", c.info.key.c_str(), c.info.device.c_str(),
                         options.rank, static_cast<unsigned long long>(hs.nonce & 0xffffU));
            for (const auto& line : c.trace) std::fprintf(stderr, "  %s\n", line.c_str());
        }
    std::vector<std::size_t> accepted;
    const auto reject = [&](Candidate& c, const std::string& why) {
        std::string text = c.info.key + " " + c.info.device + ": " + why;
        if (c.reopens) text += ", " + std::to_string(c.reopens) + " reopen(s)";
        if (c.garbage) text += ", " + std::to_string(c.garbage) + " stale 64-byte units discarded";
        if (!c.last_error.empty()) text += ", last error " + c.last_error;
        topo.rejected.push_back(text);
        if (c.fd >= 0) ::close(c.fd);
        c.fd = -1;
    };
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        auto& c = candidates[i];
        if (c.dead && c.fd < 0) { reject(c, "could not be opened"); continue; }
        if (handshake_done(c, hs.rank)) { accepted.push_back(i); continue; }
        reject(c, !c.hello_ok ? "no answer from the peer"
                  : "the peer's hello arrived but the handshake did not complete (one direction only?)");
    }
    const auto fail_all = [&](const std::string& what) {
        close_all();
        std::string all = "usb4 pairing: " + what;
        for (const auto& r : topo.rejected) all += "; " + r;
        fail(all);
    };
    if (accepted.empty()) fail_all("no stream paired with the peer");
    const auto& first_peer = candidates[accepted[0]].peer;
    for (const auto i : accepted) {
        const auto& p = candidates[i].peer;
        if (p.nonce != first_peer.nonce || p.rank != 1U - options.rank || p.nonce == hs.nonce)
            fail_all("stream " + candidates[i].info.key + " leads to " +
                     (p.rank == options.rank ? std::string("a process with the same rank") : std::string("a different peer process")) +
                     "; check the ranks and that only one job uses the streams");
    }
    topo.remote_host = std::string(first_peer.host, strnlen(first_peer.host, sizeof(first_peer.host)));
    topo.remote_unique_id = read_text(options.sysfs + "/" + candidates[accepted[0]].info.xdomain + "/unique_id");
    for (const auto i : accepted) {
        auto& c = candidates[i];
        c.info.rank0_index = options.rank == 0U ? static_cast<std::uint32_t>(i) : c.peer.stream;
        c.info.remote_index = c.peer.stream;
        c.info.remote_device = c.peer.device;
    }
    std::sort(accepted.begin(), accepted.end(), [&](std::size_t a, std::size_t b) {
        return candidates[a].info.rank0_index < candidates[b].info.rank0_index;
    });
    try {
        // Rank 0's accepted streams are a subset of rank 1's (rank 0 completes only on rank 1's
        // acknowledgement, which rank 1 writes when it completes). Rank 0 sends its set on its first
        // stream; rank 1 looks for it on all of its streams, keeps that set and echoes it.
        Plan plan;
        std::size_t plan_stream = accepted[0];
        if (options.rank == 0U) {
            plan.count = static_cast<std::uint32_t>(accepted.size());
            for (std::size_t k = 0; k < accepted.size(); ++k) plan.index[k] = candidates[accepted[k]].info.rank0_index;
            plan.crc = body_crc(plan);
            set_blocking(candidates[plan_stream].fd, true);
            write_plan(candidates[plan_stream].fd, plan);
            if (!same_plan(read_plan(candidates[plan_stream].fd), plan)) fail("usb4 pairing: the peer did not echo the plan");
        } else {
            std::map<std::size_t, std::pair<std::array<std::byte, sizeof(Plan)>, std::size_t>> partial;
            const auto deadline = std::chrono::steady_clock::now() + options.timeout;
            bool found = false;
            while (!found) {
                bool moved = false;
                for (const auto i : accepted) {
                    auto& [buffer, have] = partial[i];
                    const auto n = ::read(candidates[i].fd, buffer.data() + have, buffer.size() - have);
                    if (n > 0) { have += static_cast<std::size_t>(n); moved = true; }
                    else if (n == 0) fail("usb4 pairing: the peer closed " + candidates[i].info.key + " before sending the plan");
                    if (have == buffer.size()) {
                        std::memcpy(&plan, buffer.data(), sizeof(plan));
                        (void)check_plan(plan);
                        plan_stream = i;
                        found = true;
                        break;
                    }
                }
                if (!found && std::chrono::steady_clock::now() > deadline) fail("usb4 pairing: no plan from rank 0");
                if (!moved) std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            std::vector<std::size_t> kept;
            for (std::uint32_t k = 0; k < plan.count; ++k) {
                const auto it = std::find_if(accepted.begin(), accepted.end(), [&](std::size_t i) { return candidates[i].info.rank0_index == plan.index[k]; });
                if (it == accepted.end()) fail("usb4 pairing: rank 0's plan names a stream this rank did not pair");
                kept.push_back(*it);
            }
            if (kept.empty() || kept[0] != plan_stream) fail("usb4 pairing: the plan arrived on an unexpected stream");
            for (const auto i : accepted)
                if (std::find(kept.begin(), kept.end(), i) == kept.end())
                    reject(candidates[i], "paired here, but rank 0 did not complete its handshake (its acknowledgement was late)");
            accepted = kept;
            set_blocking(candidates[plan_stream].fd, true);
            write_plan(candidates[plan_stream].fd, plan);
        }
        // Striping order: the first stream of every cable, then the second of every cable, ... (cables in
        // the order of their first stream), so a message on k streams spreads over the cables. Both ranks
        // derive it from rank 0's indices and the cable partition, which they share.
        {
            std::vector<std::string> cable_order;
            for (const auto i : accepted)
                if (std::find(cable_order.begin(), cable_order.end(), candidates[i].info.xdomain) == cable_order.end())
                    cable_order.push_back(candidates[i].info.xdomain);
            std::map<std::size_t, std::pair<std::size_t, std::size_t>> key;  // candidate -> (position in cable, cable)
            std::map<std::string, std::size_t> seen;
            for (const auto i : accepted) {
                const auto& x = candidates[i].info.xdomain;
                key[i] = {seen[x]++, static_cast<std::size_t>(std::find(cable_order.begin(), cable_order.end(), x) - cable_order.begin())};
            }
            std::stable_sort(accepted.begin(), accepted.end(), [&](std::size_t a, std::size_t b) { return key[a] < key[b]; });
        }
        for (const auto i : accepted)
            if (!spin_streams()) set_blocking(candidates[i].fd, true);
            else set_blocking(candidates[i].fd, false);
        std::vector<int> fds;
        for (const auto i : accepted) fds.push_back(candidates[i].fd);
        // Cables (XDomains) in striping order; each one measured with all its streams at once.
        std::vector<std::string> order;
        for (const auto i : accepted)
            if (std::find(order.begin(), order.end(), candidates[i].info.xdomain) == order.end()) order.push_back(candidates[i].info.xdomain);
        for (const auto& x : order) {
            std::vector<int> cable_fds;
            std::vector<std::size_t> members;
            for (const auto i : accepted)
                if (candidates[i].info.xdomain == x) { cable_fds.push_back(candidates[i].fd); members.push_back(i); }
            double gbps = 0.0;
            if (options.probe_bytes) {
                // A 64-byte barrier on the cable before each repetition: without it the rank that starts
                // later finds the peer's data already buffered and overstates the rate.
                const auto barrier = [&] {
                    std::array<std::byte, 64> token{}, back{};
                    write_all(cable_fds[0], token, "usb4 probe barrier");
                    read_all(cable_fds[0], back, "usb4 probe barrier");
                };
                barrier();
                (void)probe_exchange(cable_fds, std::min<std::size_t>(options.probe_bytes, 256U * 1024U), options.rank); // warm-up
                std::vector<double> times;
                for (int rep = 0; rep < 5; ++rep) {
                    barrier();
                    times.push_back(probe_exchange(cable_fds, options.probe_bytes, options.rank));
                }
                std::sort(times.begin(), times.end());
                gbps = static_cast<double>((options.probe_bytes / cable_fds.size()) & ~std::size_t{63}) * static_cast<double>(cable_fds.size()) * 8.0 / times[2] / 1e9;
            }
            for (const auto i : members) candidates[i].info.measured_gbps = gbps;
        }
        // Weights (rank 0's measurement decides): the cable's rate in Mb/s shared among its streams. The plan
        // lists the streams in striping order now.
        Plan weights = plan;
        for (std::size_t k = 0; k < accepted.size(); ++k) weights.index[k] = candidates[accepted[k]].info.rank0_index;
        weights.min_part = static_cast<std::uint32_t>(options.min_part_bytes);
        if (const char* mp = std::getenv("STRIX_DIST_USB4_MIN_PART"); mp && *mp) weights.min_part = static_cast<std::uint32_t>(std::stoul(mp));
        for (std::size_t k = 0; k < accepted.size(); ++k) {
            const auto& info = candidates[accepted[k]].info;
            const auto members = static_cast<double>(std::count_if(accepted.begin(), accepted.end(), [&](std::size_t i) { return candidates[i].info.xdomain == info.xdomain; }));
            const double rate = options.probe_bytes ? info.measured_gbps : info.link_gbps;
            weights.weight[k] = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(rate * 1000.0 / members));
        }
        agree(fds[0], options.rank, weights);
        std::vector<std::string> names;
        std::vector<std::uint32_t> stream_weights;
        for (std::size_t k = 0; k < accepted.size(); ++k) {
            auto& c = candidates[accepted[k]];
            c.info.weight = weights.weight[k];
            names.push_back(c.info.device + ">tbstream" + std::to_string(c.info.remote_device));
            stream_weights.push_back(weights.weight[k]);
            topo.streams.push_back(c.info);
        }
        for (const auto i : accepted) candidates[i].fd = -1; // owned by the transport from here on
        auto transport = std::make_unique<Usb4HostTransport>(std::move(fds), std::move(names), std::move(stream_weights), weights.min_part,
                                                             options.io_timeout);
        topo.min_part_bytes = weights.min_part;
        if (topology) *topology = std::move(topo);
        return transport;
    } catch (...) {
        close_all();
        throw;
    }
}

std::unique_ptr<Transport> tcp_listen(std::uint16_t port, std::chrono::milliseconds timeout) {
    const int server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) fail("socket failed");
    const int one = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || ::listen(server, 1) != 0) {
        ::close(server);
        fail(std::string("bind/listen failed: ") + std::strerror(errno));
    }
    pollfd p{server, POLLIN, 0};
    if (::poll(&p, 1, static_cast<int>(timeout.count())) <= 0) {
        ::close(server);
        fail("no peer connected before the timeout");
    }
    const int fd = ::accept(server, nullptr, nullptr);
    ::close(server);
    if (fd < 0) fail("accept failed");
    return std::make_unique<TcpTransport>(fd, timeout);
}

std::unique_ptr<Transport> tcp_connect(const std::string& host, std::uint16_t port, std::chrono::milliseconds timeout) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found) != 0 || !found) fail("cannot resolve " + host);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd >= 0 && ::connect(fd, found->ai_addr, found->ai_addrlen) == 0) {
            ::freeaddrinfo(found);
            return std::make_unique<TcpTransport>(fd, timeout);
        }
        if (fd >= 0) ::close(fd);
        if (std::chrono::steady_clock::now() > deadline) {
            ::freeaddrinfo(found);
            fail("cannot connect to " + host);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

class Communicator::Progress {
public:
    Worker worker;
};

Communicator::Communicator(Group group, std::unique_ptr<Transport> transport, std::size_t max_bytes)
    : group_(group), transport_(std::move(transport)), max_bytes_(max_bytes), progress_(std::make_unique<Progress>()) {
    if (group_.world != 2U || group_.rank > 1U || !transport_ || max_bytes_ == 0U)
        throw std::invalid_argument("dist communicator needs two ranks, a transport and a size bound");
    check(hipHostMalloc(reinterpret_cast<void**>(&send_host_), max_bytes_ + sizeof(Trailer), hipHostMallocDefault), "pinned send buffer");
    check(hipHostMalloc(reinterpret_cast<void**>(&receive_host_), max_bytes_ + sizeof(Trailer), hipHostMallocDefault), "pinned receive buffer");
}
Communicator::~Communicator() {
    if (started_) {
        try { progress_->worker.wait(); } catch (...) {}
    }
    progress_.reset();
    if (send_ready_) {
        (void)hipEventSynchronize(send_ready_);
        (void)hipEventDestroy(send_ready_);
    }
    if (receive_done_) {
        (void)hipEventSynchronize(receive_done_);
        (void)hipEventDestroy(receive_done_);
    }
    (void)hipHostFree(send_host_);
    (void)hipHostFree(receive_host_);
}

void Communicator::exchange(hipStream_t stream, const void* local, void* remote, std::size_t bytes, const Tag& tag) {
    start(stream, local, bytes, tag);
    finish(stream, remote);
}

void Communicator::start(hipStream_t stream, const void* local, std::size_t bytes, const Tag& tag) {
    if (started_) throw std::logic_error("dist exchange: start() while an exchange is in flight");
    if (bytes == 0U || bytes > max_bytes_ || !local) throw std::invalid_argument("dist exchange size or buffers");
    const auto entry = std::chrono::steady_clock::now();
    stats_.last_entry = entry;
    // The receive buffer is reused only after the CPU has seen the previous copy out of it complete.
    if (receive_pending_)
        for (hipError_t ready; (ready = hipEventQuery(receive_done_)) != hipSuccess;) {
            if (ready != hipErrorNotReady) check(ready, "previous receive copy");
            cpu_relax();
        }
    receive_pending_ = false;
    // The copy is ordered after the producer of `local` on the stream; the progress thread waits for it on
    // the CPU before computing the CRC and sending.
    check(hipMemcpyAsync(send_host_, local, bytes, hipMemcpyDeviceToHost, stream), "device to host");
    if (!send_ready_) check(hipEventCreateWithFlags(&send_ready_, hipEventDisableTiming), "send event");
    check(hipEventRecord(send_ready_, stream), "send event");
    bytes_ = bytes;
    tag_ = tag;
    started_ = true;
    started_at_ = entry;
    progress_->worker.post([this] {
        // Spin on the copy's event: a blocking synchronize may sleep and add its wake-up to every exchange.
        for (hipError_t ready; (ready = hipEventQuery(send_ready_)) != hipSuccess;) {
            if (ready != hipErrorNotReady) check(ready, "device to host");
            cpu_relax();
        }
        Trailer trailer;
        trailer.epoch = group_.epoch; trailer.plan = group_.plan; trailer.sequence = sequence_; trailer.request = tag_.request;
        trailer.bytes = bytes_; trailer.rank = group_.rank; trailer.tag = tag_.kind; trailer.layer = tag_.layer;
        trailer.crc32c = crc32c(std::span<const std::byte>(send_host_, bytes_));
        std::memcpy(send_host_ + bytes_, &trailer, sizeof(trailer));
        stats_.last_staged = std::chrono::steady_clock::now();
        transport_->exchange(std::span<const std::byte>(send_host_, bytes_ + sizeof(Trailer)),
                             std::span<std::byte>(receive_host_, bytes_ + sizeof(Trailer)));
        stats_.last_received = std::chrono::steady_clock::now();
    });
    returned_at_ = std::chrono::steady_clock::now();
}

void Communicator::finish(hipStream_t stream, void* remote) {
    if (!started_) throw std::logic_error("dist exchange: finish() without start()");
    if (!remote) throw std::invalid_argument("dist exchange receive buffer");
    const auto entry = std::chrono::steady_clock::now();
    started_ = false;
    progress_->worker.wait();
    const auto received = std::chrono::steady_clock::now();
    const auto bytes = bytes_;
    Trailer peer;
    std::memcpy(&peer, receive_host_ + bytes, sizeof(peer));
    if (peer.magic != trailer_magic || peer.epoch != group_.epoch || peer.plan != group_.plan || peer.sequence != sequence_ ||
        peer.request != tag_.request || peer.bytes != bytes || peer.rank != 1U - group_.rank || peer.tag != tag_.kind ||
        peer.layer != tag_.layer)
        throw std::runtime_error("dist exchange: the peer's trailer does not match this collective");
    if (peer.crc32c != crc32c(std::span<const std::byte>(receive_host_, bytes)))
        throw std::runtime_error("dist exchange: payload CRC mismatch");
    check(hipMemcpyAsync(remote, receive_host_, bytes, hipMemcpyHostToDevice, stream), "host to device");
    if (!receive_done_) check(hipEventCreateWithFlags(&receive_done_, hipEventDisableTiming), "receive event");
    check(hipEventRecord(receive_done_, stream), "receive event");
    receive_pending_ = true;
    ++sequence_;
    ++stats_.exchanges;
    stats_.bytes_sent += bytes;
    const auto end = std::chrono::steady_clock::now();
    stats_.stage += std::chrono::duration<double>(stats_.last_staged - started_at_).count();
    stats_.wire += std::chrono::duration<double>(stats_.last_received - stats_.last_staged).count();
    stats_.check += std::chrono::duration<double>(end - received).count();
    stats_.overlap += std::chrono::duration<double>(entry - returned_at_).count();
    stats_.finish_wait += std::chrono::duration<double>(received - entry).count();
    stats_.seconds += std::chrono::duration<double>(end - started_at_).count();
}

} // namespace strix::dist
