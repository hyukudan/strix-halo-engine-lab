#include "strix/model_input.hpp"
#include "strix/ds41_prepared_load.hpp"
#include "strix/ds41_expert_tp.hpp"
#include "strix/ds41_weight_overlay.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#if defined(STRIX_PREPARED_OPENSSL)
#include <openssl/evp.h>
#endif
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strix::ds41::prepared {
namespace {
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error("ds41 prepared: " + why); }
std::string env(const char* key) { const char* v = std::getenv(key); return v ? v : ""; }
void valid_name(const std::string& name) {
    require(!name.empty() && name.size() < 256 && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") == std::string::npos
            && name != "." && name != "..", "invalid tensor name");
}
std::string read_text(const std::filesystem::path& file) {
    require(std::filesystem::file_size(file) <= (4U << 20U), "manifest too large");
    strix::ModelInput f(file);
    require(bool(f), "cannot open manifest");
    return {std::istreambuf_iterator<char>(f), {}};
}
constexpr std::uint64_t alignment = 4096;
std::uint64_t padded(std::uint64_t n) { return (n + alignment - 1) & ~(alignment - 1); }
struct Input {
#if !defined(_WIN32)
    int fd = -1;
    bool direct = true;
    explicit Input(const std::filesystem::path& path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) { direct = false; fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC); }
        require(fd >= 0, "cannot open artifact");
        strix::LoaderCapture::record(path, fd);
    }
    ~Input() { if (fd >= 0) ::close(fd); }
    void read(std::uint64_t offset, std::span<std::uint8_t> out) {
        std::size_t done = 0;
        while (done < out.size()) {
            const auto n = ::pread(fd, out.data() + done, out.size() - done, static_cast<off_t>(offset + done));
            require(n > 0, "short artifact read"); done += static_cast<std::size_t>(n);
        }
        if (!direct) (void)::posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(out.size()), POSIX_FADV_DONTNEED);
    }
#else
    std::ifstream in;
    explicit Input(const std::filesystem::path& path) : in(path, std::ios::binary) { require(bool(in), "cannot open artifact"); }
    void read(std::uint64_t offset, std::span<std::uint8_t> out) {
        in.seekg(static_cast<std::streamoff>(offset));
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size())); require(bool(in), "short artifact read");
    }
#endif
};
struct AlignedChunk {
    std::uint8_t* p = static_cast<std::uint8_t*>(::operator new(chunk_bytes, std::align_val_t(alignment)));
    ~AlignedChunk() { ::operator delete(p, std::align_val_t(alignment)); }
};
void read_at(std::ifstream& in, std::uint64_t offset, std::span<std::uint8_t> out) {
    in.clear(); in.seekg(static_cast<std::streamoff>(offset));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    require(bool(in), "short read");
}
}
const char* digest_backend() noexcept {
#if defined(STRIX_PREPARED_OPENSSL)
    return "openssl-sha256";
#else
    return "portable-sha256";
#endif
}
std::string digest(Bytes bytes) {
#if defined(STRIX_PREPARED_OPENSSL)
    ModelContentIdentity id;
    unsigned size = 0;
    require(EVP_Digest(bytes.data(), bytes.size(), id.bytes.data(), &size, EVP_sha256(), nullptr) == 1 && size == 32, "SHA256 failed");
    return model_content_identity_hex(id);
#else
    return model_content_identity_hex(sha256_bytes(std::as_bytes(bytes)));
#endif
}
std::string chunked_digest(std::uint64_t bytes,
    const std::function<void(std::span<std::uint8_t>, std::uint64_t)>& read, Staging staging) {
    if (bytes == 0) return digest({});
    std::unique_ptr<std::array<AlignedChunk, staging_slots>> owned;
    if (staging[0].empty()) {
        owned = std::make_unique<std::array<AlignedChunk, staging_slots>>();
        for (std::size_t i = 0; i < staging_slots; ++i) staging[i] = {(*owned)[i].p, chunk_bytes};
    }
    for (const auto slot : staging) require(slot.size() >= chunk_bytes, "invalid readback staging size");
    // Futures are destroyed before owned/staging, including if the producer or a digest throws.
    std::array<std::future<std::string>, staging_slots> jobs;
    std::string hashes;
    const auto chunks = (bytes - 1) / chunk_bytes + 1;
    for (std::uint64_t i = 0; i < chunks; ++i) {
        const auto slot = static_cast<std::size_t>(i % staging_slots);
        if (i >= staging_slots) hashes += jobs[slot].get();
        const auto offset = i * chunk_bytes;
        const auto take = static_cast<std::size_t>(std::min<std::uint64_t>(chunk_bytes, bytes - offset));
        const auto part = staging[slot].first(take);
        read(part, offset);
        jobs[slot] = std::async(std::launch::async, [part] { return digest(part); });
    }
    for (auto i = chunks > staging_slots ? chunks - staging_slots : 0; i < chunks; ++i)
        hashes += jobs[static_cast<std::size_t>(i % staging_slots)].get();
    return digest(Bytes(reinterpret_cast<const std::uint8_t*>(hashes.data()), hashes.size()));
}
std::string source_stamp(const std::filesystem::path& path) {
#if !defined(_WIN32)
    struct stat s {};
    require(::stat(path.c_str(), &s) == 0 && S_ISREG(s.st_mode), "cannot stat " + path.string());
    std::ostringstream out;
    out << std::filesystem::canonical(path).string() << ':' << s.st_dev << ':' << s.st_ino << ':' << s.st_size
        << ':' << s.st_mtim.tv_sec << ':' << s.st_mtim.tv_nsec << ':' << s.st_ctim.tv_sec << ':' << s.st_ctim.tv_nsec;
    return out.str();
#else
    return std::filesystem::canonical(path).string() + ':' + std::to_string(std::filesystem::file_size(path)) + ':' +
           std::to_string(std::filesystem::last_write_time(path).time_since_epoch().count());
#endif
}
DenseBinding dense_from_env() {
    return {env("STRIX_DS41_TRELLIS"), env("STRIX_DS41_TRELLIS_MIX"), env("STRIX_DS41_TRELLIS_SITES"), env("STRIX_DS41_TRELLIS_HEADOUT")};
}
std::string name_space(const std::filesystem::path& model, const std::filesystem::path& overlay,
                       const deepseek41::TargetModel& target, unsigned rank, unsigned world, bool tp, bool packed, DenseBinding dense) {
    require(world == 2 && rank < world, "prepared experts require world=2 and rank 0/1");
    std::ostringstream ns;
    const auto file = [&](const std::filesystem::path& p) {
        const auto before = source_stamp(p);
        const auto hash = model_content_identity_hex(sha256_file_content(p));
        require(before == source_stamp(p), "source changed while hashing");
        ns << std::quoted(before) << ' ' << hash << '\n';
    };
    ns << "ds41-prepared-experts-v1\nmodel "; file(model);
    ns << "layout " << target.manifest.layout_identity << "\noverlay ";
    if (!overlay.empty()) file(overlay); else ns << "none\n";
    ns << "rank " << rank << " world " << world << " identity " << (tp ? expert_tp::identity : "ds41-expert-ep-v1")
       << " width " << (tp ? expert_tp::half : expert_tp::inter) << " down " << (packed ? "packed" : "dup") << '\n';
    // The expert payload does not contain dense trellis weights. Bind the requested dense configuration
    // and source files anyway so an artifact cannot silently cross a model/overlay/mix boundary.
    ns << "STRIX_DS41_TRELLIS " << std::quoted(dense.trellis) << '\n'
       << "STRIX_DS41_TRELLIS_MIX " << std::quoted(dense.mix) << '\n'
       << "STRIX_DS41_TRELLIS_SITES " << std::quoted(dense.sites) << '\n'
       << "STRIX_DS41_TRELLIS_HEADOUT " << std::quoted(dense.headout) << '\n';
    std::vector<std::filesystem::path> dirs;
    if (!dense.trellis.empty()) dirs.emplace_back(dense.trellis);
    auto mix = dense.mix;
    if (!mix.empty() && std::filesystem::is_regular_file(mix)) { file(mix); mix = read_text(mix); }
    std::istringstream lines(mix);
    std::string normalized;
    for (std::string line; std::getline(lines, line);) normalized += line.substr(0, line.find('#')) + ',';
    std::istringstream mapping(normalized);
    for (std::string item; std::getline(mapping, item, ',');) {
        const auto eq = item.find('=');
        if (eq != std::string::npos) {
            const auto value = item.substr(eq + 1);
            const auto first = value.find_first_not_of(" \t\r");
            const auto last = value.find_last_not_of(" \t\r");
            require(first != std::string::npos, "empty mix directory");
            dirs.emplace_back(value.substr(first, last - first + 1));
        }
    }
    std::sort(dirs.begin(), dirs.end()); dirs.erase(std::unique(dirs.begin(), dirs.end()), dirs.end());
    for (const auto& dir : dirs) {
        std::vector<std::filesystem::path> files;
        for (const auto& f : std::filesystem::directory_iterator(dir))
            if (f.is_regular_file() && (f.path().extension() == ".safetensors" || f.path().extension() == ".json")) files.push_back(f.path());
        std::sort(files.begin(), files.end());
        for (const auto& f : files) file(f);
    }
    return ns.str();
}
Writer::Writer(const std::filesystem::path& destination, std::string ns) : destination_(destination), ns_(std::move(ns)) {
    require(!std::filesystem::exists(destination_), "destination exists (use a new version directory)");
    temporary_ = destination_; temporary_ += ".incomplete";
    // An atomic directory creation prevents concurrent writers from sharing temporary files.
    if (!destination_.parent_path().empty()) std::filesystem::create_directories(destination_.parent_path());
    require(std::filesystem::create_directory(temporary_), "preparation already in progress");
}
Writer::~Writer() { if (!committed_) { std::error_code e; std::filesystem::remove_all(temporary_, e); } }
void Writer::tensor(const std::string& name, std::uint64_t bytes, const std::function<void(const Sink&)>& produce) {
    valid_name(name); require(bytes > 0, "empty tensor");
    const auto path = temporary_ / (name + ".bin");
    require(!std::filesystem::exists(path), "duplicate tensor");
    std::ofstream out(path, std::ios::binary);
    require(bool(out), "cannot write tensor");
    std::vector<std::uint8_t> pending; pending.reserve(chunk_bytes);
    std::vector<std::string> hashes;
    std::uint64_t written = 0;
    const auto flush = [&] {
        hashes.push_back(digest(pending));
        out.write(reinterpret_cast<const char*>(pending.data()), static_cast<std::streamsize>(pending.size()));
        require(bool(out), "tensor write failed"); pending.clear();
    };
    produce([&](Bytes data, std::uint64_t offset) {
        require(offset == written && data.size() <= bytes - written, "noncontiguous producer");
        written += data.size();
        while (!data.empty()) {
            const auto take = std::min(data.size(), chunk_bytes - pending.size());
            pending.insert(pending.end(), data.begin(), data.begin() + static_cast<std::ptrdiff_t>(take));
            data = data.subspan(take);
            if (pending.size() == chunk_bytes) flush();
        }
    });
    require(written == bytes, "producer length mismatch");
    if (!pending.empty()) flush();
    const std::array<char, alignment> zeros{};
    out.write(zeros.data(), static_cast<std::streamsize>(padded(bytes) - bytes));
    out.close(); require(bool(out), "tensor close failed");
#if !defined(_WIN32)
    const int durable = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(durable >= 0, "cannot reopen artifact for sync");
    const int synced = ::fsync(durable);
    (void)::posix_fadvise(durable, 0, 0, POSIX_FADV_DONTNEED);
    ::close(durable); require(synced == 0, "artifact sync failed");
#endif
    std::ostringstream line;
    line << name << ' ' << bytes << ' ' << hashes.size();
    for (const auto& h : hashes) line << ' ' << h;
    entries_ += line.str() + '\n';
}
void Writer::commit() {
    const auto body = "STRIX-PREPARED-1\n" + ns_ + "END-NAMESPACE\n" + entries_;
    std::ofstream out(temporary_ / "manifest", std::ios::binary);
    out << digest(Bytes(reinterpret_cast<const std::uint8_t*>(body.data()), body.size())) << '\n' << body;
    out.close(); require(bool(out), "manifest write failed");
#if !defined(_WIN32)
    const int fd = ::open((temporary_ / "manifest").c_str(), O_RDONLY | O_CLOEXEC);
    require(fd >= 0, "manifest reopen failed");
    const int synced = ::fsync(fd); ::close(fd); require(synced == 0, "manifest sync failed");
#endif
    std::filesystem::rename(temporary_, destination_); committed_ = true;
}
Reader::Reader(const std::filesystem::path& directory, const std::string& expected) : directory_(directory) {
    const auto text = read_text(directory / "manifest");
    require(text.size() > 65 && text[64] == '\n', "invalid manifest");
    const auto body = text.substr(65);
    require(text.substr(0, 64) == digest(Bytes(reinterpret_cast<const std::uint8_t*>(body.data()), body.size())), "manifest checksum mismatch");
    constexpr std::string_view version = "STRIX-PREPARED-1\n";
    require(body.starts_with(version), "version mismatch: expected STRIX-PREPARED-1, artifact " + body.substr(0, body.find('\n')));
    // Keep the original v1 acceptance rule even when quoted source names contain
    // the delimiter text. Only parse an incompatible namespace to explain it.
    const auto prefix = std::string(version) + expected + "END-NAMESPACE\n";
    if (!body.starts_with(prefix)) {
        const auto end = body.rfind("END-NAMESPACE\n");
        require(end != std::string::npos, "missing END-NAMESPACE");
        const auto actual = body.substr(version.size(), end - version.size());
        // Report the first differing field (including rank/world, EP/TP and down layout),
        // rather than hiding a legitimate incompatibility behind a generic rejection.
        std::istringstream want(expected), have(actual);
        for (unsigned line = 1;; ++line) {
            std::string w, h;
            const bool more_w = bool(std::getline(want, w)), more_h = bool(std::getline(have, h));
            if (!more_w && !more_h) break;
            if (w == h && more_w == more_h) continue;
            std::ostringstream why;
            why << "namespace mismatch at line " << line << ": expected " << std::quoted(w.substr(0, 512))
                << "; artifact " << std::quoted(h.substr(0, 512));
            require(false, why.str());
        }
        require(false, "namespace mismatch: trailing newline differs");
    }
    std::istringstream in(body.substr(prefix.size()));
    for (std::string name; in >> name;) {
        valid_name(name); Entry e; std::uint64_t count = 0;
        require(bool(in >> e.bytes >> count) && e.bytes > 0 && e.bytes < (1ULL << 42U) && count == (e.bytes - 1) / chunk_bytes + 1, "invalid tensor entry");
        for (std::uint64_t i = 0; i < count; ++i) {
            std::string hash; require(bool(in >> hash) && hash.size() == 64 && hash.find_first_not_of("0123456789abcdef") == std::string::npos, "invalid chunk checksum");
            e.hashes.push_back(std::move(hash));
        }
        require(std::filesystem::file_size(directory / (name + ".bin")) == padded(e.bytes), "tensor size mismatch");
        require(entries_.emplace(name, std::move(e)).second, "duplicate tensor entry");
    }
    require(!entries_.empty(), "empty artifact");
}
void Reader::load(const std::string& name, std::uint64_t bytes, const Sink& consume, LoadStats* stats, Staging staging) const {
    const auto it = entries_.find(name);
    require(it != entries_.end() && it->second.bytes == bytes, "missing tensor or geometry mismatch: " + name);
    const auto& entry = it->second;
    const auto path = directory_ / (name + ".bin");
    const auto stamp = source_stamp(path);
    Input in(path);
#if !defined(_WIN32)
    if (stats) stats->direct = stats->direct && in.direct;
#else
    if (stats) stats->direct = false;
#endif
    // A fixed window of staging_slots - 1 read/verify jobs plus the consumed slot. Reads are
    // serialized; SHA work runs outside the I/O lock, so multiple checksums can
    // overlap the next read and the previous upload without a single-core bottleneck.
    std::unique_ptr<std::array<AlignedChunk, staging_slots>> owned;
    if (staging[0].empty()) {
        owned = std::make_unique<std::array<AlignedChunk, staging_slots>>();
        for (std::size_t i = 0; i < staging_slots; ++i) staging[i] = {(*owned)[i].p, chunk_bytes};
    }
    for (const auto slot : staging)
        require(slot.size() >= chunk_bytes && reinterpret_cast<std::uintptr_t>(slot.data()) % alignment == 0, "invalid staging alignment/size");
    std::mutex io_mutex, stats_mutex;
    const auto read_verify = [&](std::size_t index) {
        const auto offset = std::uint64_t{index} * chunk_bytes;
        const auto take = static_cast<std::size_t>(std::min<std::uint64_t>(chunk_bytes, bytes - offset));
        auto* buffer = staging[index % staging_slots].data();
        {
            const std::lock_guard io_lock(io_mutex);
            const auto start = std::chrono::steady_clock::now();
            in.read(offset, std::span(buffer, static_cast<std::size_t>(padded(take))));
            if (stats) {
                const std::lock_guard stats_lock(stats_mutex);
                stats->read_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            }
        }
        const auto start = std::chrono::steady_clock::now();
        require(digest(Bytes(buffer, take)) == entry.hashes[index], "payload checksum mismatch: " + name);
        if (stats) {
            const std::lock_guard stats_lock(stats_mutex);
            stats->verify_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        return take;
    };
    std::array<std::future<std::size_t>, staging_slots> jobs;
    constexpr auto ahead = staging_slots - 1;
    for (std::size_t i = 0; i < std::min(ahead, entry.hashes.size()); ++i)
        jobs[i] = std::async(std::launch::async, read_verify, i);
    for (std::size_t i = 0; i < entry.hashes.size(); ++i) {
        const auto take = jobs[i % staging_slots].get();
        if (i + ahead < entry.hashes.size())
            jobs[(i + ahead) % staging_slots] = std::async(std::launch::async, read_verify, i + ahead);
        // A slot remains owned by this callback until it returns (HIP drains its upload).
        consume(Bytes(staging[i % staging_slots].data(), take), std::uint64_t{i} * chunk_bytes);
    }
    require(source_stamp(path) == stamp, "artifact changed during load");
}
void prepare(const std::filesystem::path& model, const std::filesystem::path& overlay_path,
             const std::filesystem::path& destination, unsigned rank, unsigned world, bool tp, bool packed, unsigned layers) {
    const auto target = deepseek41::validate_target(deepseek41::read_file(model));
    const auto& c = target.config;
    require(world == 2 && rank < world && layers > 0 && layers <= c.num_hidden_layers && (!tp || c.moe_intermediate_size == expert_tp::inter), "unsupported preparation geometry");
    const auto ns = name_space(model, overlay_path, target, rank, world, tp, packed);
    std::unique_ptr<WeightOverlay> overlay;
    if (!overlay_path.empty()) overlay = std::make_unique<WeightOverlay>(overlay_path, target.manifest);
    Writer writer(destination, ns);
    for (const auto& t : target.manifest.tensors) {
        if (!t.name.ends_with("_exps.weight") || !t.layer || *t.layer >= layers) continue;
        const auto* over = overlay ? overlay->find(t.name) : nullptr;
        const auto path = over ? overlay_path : model;
        const auto stamp = source_stamp(path);
        std::ifstream in(path, std::ios::binary);
        const auto offset = over ? over->absolute_offset : t.absolute_offset;
        const auto per = t.byte_size / c.n_routed_experts;
        const bool down = t.name.ends_with("ffn_down_exps.weight");
        const auto row = per / c.moe_intermediate_size;
        const auto out_per = tp ? (down ? c.hidden_size * (packed ? expert_tp::down_row_bytes_packed : expert_tp::down_row_bytes) : per / 2) : per;
        const auto experts = tp ? c.n_routed_experts : c.n_routed_experts / world;
        writer.tensor(t.name, out_per * experts, [&](const Sink& sink) {
            // One producer reads/converts the next expert while the consumer hashes/writes the previous.
            // At this geometry each source/output expert is <16 MiB; never retain a layer/model.
            require(per <= chunk_bytes && out_per <= chunk_bytes, "expert exceeds staging bound");
            std::array<std::vector<std::uint8_t>, 2> output;
            for (auto& b : output) b.resize(static_cast<std::size_t>(out_per));
            std::vector<std::uint8_t> source(static_cast<std::size_t>(per));
            const auto convert = [&](unsigned e) {
                auto& dst = output[e % 2];
                const auto global = tp ? e : rank * experts + e;
                read_at(in, offset + global * per, source);
                if (!tp) std::memcpy(dst.data(), source.data(), dst.size());
                else if (!down) expert_tp::pack_gate_rows(source, row, rank, dst);
                else for (unsigned r = 0; r < c.hidden_size; ++r) {
                    auto d = std::span(dst).subspan(r * (out_per / c.hidden_size), out_per / c.hidden_size);
                    const auto s = Bytes(source).subspan(r * 756U, 756U);
                    if (packed) expert_tp::pack_down_row_packed(s, rank, d); else expert_tp::pack_down_row(s, rank, d);
                }
            };
            auto next = std::async(std::launch::async, convert, 0);
            for (unsigned e = 0; e < experts; ++e) {
                next.get();
                if (e + 1 < experts) next = std::async(std::launch::async, convert, e + 1);
                sink(output[e % 2], std::uint64_t{e} * out_per);
            }
        });
        std::fprintf(stderr, "prepared %s bytes %llu\n", t.name.c_str(), static_cast<unsigned long long>(out_per * experts));
        require(stamp == source_stamp(path), "source changed during preparation");
#if !defined(_WIN32)
        const int advise = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (advise >= 0) {
            (void)::posix_fadvise(advise, static_cast<off_t>(offset), static_cast<off_t>(t.byte_size), POSIX_FADV_DONTNEED);
            ::close(advise);
        }
#endif
    }
    require(ns == name_space(model, overlay_path, target, rank, world, tp, packed), "namespace changed during preparation");
    writer.commit();
}
} // namespace strix::ds41::prepared
