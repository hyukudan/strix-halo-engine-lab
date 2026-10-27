// GLM engine: safetensors pack index and O_DIRECT reads (copied from the MiniMax engine mm_pack.cpp, model-agnostic).
#include "glm_pack.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace strix::glm::engine::detail {

namespace {

constexpr std::uint64_t kAlign = 4096;
constexpr std::uint64_t kBounce = std::uint64_t{64} << 20;

std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void pread_all(int fd, void* dst, std::uint64_t n, std::uint64_t off, const std::string& what) {
    auto* d = static_cast<std::uint8_t*>(dst);
    while (n > 0) {
        const ssize_t r = ::pread(fd, d, static_cast<std::size_t>(n), static_cast<off_t>(off));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw std::runtime_error("pread " + what + ": " + std::strerror(errno));
        d += r;
        off += static_cast<std::uint64_t>(r);
        n -= static_cast<std::uint64_t>(r);
    }
}

// Drop the page cache of [off, off + n) (whole pages around it): a buffered read leaves no resident file pages behind.
// posix_fadvise is advisory: a failure is reported once (the pages may stay cached; the memory monitor sees them).
void advise(int fd, std::uint64_t a, std::uint64_t n, int what, const char* name) {
    static std::atomic<bool> warned{false};
    const int r = ::posix_fadvise(fd, static_cast<off_t>(a), static_cast<off_t>(n), what);
    if (r != 0 && !warned.exchange(true)) std::fprintf(stderr, "pack: WARNING posix_fadvise(%s) failed: %s\n", name, std::strerror(r));
}
void drop_cache(int fd, std::uint64_t off, std::uint64_t n) {
    const std::uint64_t a = off & ~(kAlign - 1), e = (off + n + kAlign - 1) & ~(kAlign - 1);
    advise(fd, a, e - a, POSIX_FADV_DONTNEED, "DONTNEED");
}

float bf16f(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16U;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

PackReader::PackReader(std::string dir) : dir_(std::move(dir)) {
    const std::string idx = read_text(dir_ + "/model.safetensors.index.json");
    const Json j = parse_json(idx.data(), idx.data() + idx.size());
    std::map<std::string, std::uint32_t> file_ids;
    std::vector<std::string> files;
    for (const auto& [k, v] : j.obj().at("weight_map").obj())
        if (file_ids.emplace(v.str(), static_cast<std::uint32_t>(files.size())).second) files.push_back(v.str());
    // Shards the index does not list (the GLM EXL3 pack ships layer 45 = MTP in mtp.safetensors, outside the index).
    std::vector<std::string> unlisted;
    for (const auto& e : std::filesystem::directory_iterator(dir_)) {
        const std::string n = e.path().filename().string();
        if (e.is_regular_file() && n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0 && !file_ids.count(n))
            unlisted.push_back(n);
    }
    std::sort(unlisted.begin(), unlisted.end());
    for (const auto& n : unlisted) {
        file_ids.emplace(n, static_cast<std::uint32_t>(files.size()));
        files.push_back(n);
    }
    void* p = nullptr;
    if (posix_memalign(&p, kAlign, kBounce) != 0) throw std::runtime_error("bounce buffer");
    bounce_ = static_cast<std::uint8_t*>(p);
    try {
        open_shards(files);
    } catch (...) {  // the destructor does not run for a throwing constructor
        for (int fd : fds_) ::close(fd);
        std::free(bounce_);
        throw;
    }
}

// O_DIRECT by default; an O_DIRECT open failure FAILS (strict: no page cache). STRIX_GLM_LOAD_DIRECT=0 = buffered reads
// without readahead + DONTNEED after every read (advisory; the small index JSON is always read buffered).
void PackReader::open_shards(const std::vector<std::string>& files) {
    fds_.reserve(files.size());  // the push_backs below cannot throw: no opened fd is ever unowned
    direct_.reserve(files.size());
    const bool want_direct = std::getenv("STRIX_GLM_LOAD_DIRECT") == nullptr || std::atoi(std::getenv("STRIX_GLM_LOAD_DIRECT")) != 0;
    for (std::uint32_t f = 0; f < files.size(); ++f) {
        const std::string path = dir_ + "/" + files[f];
        int fd = want_direct ? ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT) : -1;
        if (fd < 0 && want_direct)
            throw std::runtime_error("pack: O_DIRECT open failed for " + path + " (" + std::strerror(errno) +
                                     "); refusing buffered reads (STRIX_GLM_LOAD_DIRECT=0 allows them with DONTNEED per read)");
        direct_.push_back(fd >= 0 ? 1 : 0);
        if (fd < 0) {
            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            // no readahead: pages read ahead asynchronously would land after the per-read DONTNEED and stay cached
            if (fd >= 0) advise(fd, 0, 0, POSIX_FADV_RANDOM, "RANDOM");
        }
        if (fd < 0) throw std::runtime_error("open " + path);
        fds_.push_back(fd);
        // Header: 8-byte length + JSON, through the same path as the tensors (O_DIRECT: no page cache).
        std::uint64_t hb = 0;
        read_at(f, 0, 8, &hb);
        if (hb == 0 || hb > (std::uint64_t{1} << 30)) throw std::runtime_error("bad safetensors header length in " + path);
        std::string head(hb, '\0');
        read_at(f, 8, hb, head.data());
        const Json h = parse_json(head.data(), head.data() + head.size());
        for (const auto& [name, v] : h.obj()) {
            if (name == "__metadata__") continue;
            TensorInfo t;
            t.dtype = v.obj().at("dtype").str();
            for (const auto& s : v.obj().at("shape").arr()) t.shape.push_back(static_cast<std::uint64_t>(s.num()));
            const auto& off = v.obj().at("data_offsets").arr();
            const auto b = static_cast<std::uint64_t>(off[0].num()), e = static_cast<std::uint64_t>(off[1].num());
            t.file = f;
            t.offset = 8 + hb + b;
            t.bytes = e - b;
            if (!tensors_.emplace(name, std::move(t)).second) throw std::runtime_error("tensor defined twice: " + name);
        }
    }
}

PackReader::~PackReader() {
    for (int fd : fds_) ::close(fd);
    std::free(bounce_);
}

std::size_t PackReader::direct_files() const { return static_cast<std::size_t>(std::count(direct_.begin(), direct_.end(), 1)); }

std::uint64_t PackReader::bounce_bytes() { return kBounce; }

const TensorInfo& PackReader::info(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not in pack: " + name);
    return it->second;
}

void PackReader::read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst) {
    if (offset + n > t.bytes) throw std::runtime_error("read past tensor end");
    bytes_read_ += n;
    read_at(t.file, t.offset + offset, n, dst);
}

void PackReader::read_at(std::uint32_t file, std::uint64_t pos, std::uint64_t n, void* dst) {
    const int fd = fds_[file];
    auto* d = static_cast<std::uint8_t*>(dst);
    if (!direct_[file]) {
        pread_all(fd, d, n, pos, dir_);
        drop_cache(fd, pos, n);
        return;
    }
    // O_DIRECT: aligned windows through the bounce buffer.
    while (n > 0) {
        const std::uint64_t a0 = pos & ~(kAlign - 1), skip = pos - a0;
        const std::uint64_t want = std::min<std::uint64_t>(n, kBounce - skip);
        const std::uint64_t len = (skip + want + kAlign - 1) & ~(kAlign - 1);
        std::uint64_t got = 0;
        while (got < len) {  // a short read only at the end of the file
            const ssize_t r = ::pread(fd, bounce_ + got, static_cast<std::size_t>(len - got), static_cast<off_t>(a0 + got));
            if (r < 0 && errno == EINTR) continue;
            if (r < 0) throw std::runtime_error(std::string("pread (O_DIRECT): ") + std::strerror(errno));
            if (r == 0) break;
            got += static_cast<std::uint64_t>(r);
        }
        if (got < skip + want) throw std::runtime_error("short O_DIRECT read");
        std::memcpy(d, bounce_ + skip, want);
        d += want;
        pos += want;
        n -= want;
    }
}

std::vector<float> PackReader::floats(const std::string& name) {
    const TensorInfo& t = info(name);
    std::vector<std::uint8_t> raw(t.bytes);
    read(t, 0, t.bytes, raw.data());
    std::vector<float> out;
    if (t.dtype == "BF16") {
        out.resize(t.bytes / 2);
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::uint16_t b;
            std::memcpy(&b, raw.data() + 2 * i, 2);
            out[i] = bf16f(b);
        }
    } else if (t.dtype == "F32") {
        out.resize(t.bytes / 4);
        std::memcpy(out.data(), raw.data(), t.bytes);
    } else if (t.dtype == "F16") {
        out.resize(t.bytes / 2);
        for (std::size_t i = 0; i < out.size(); ++i) {
            _Float16 h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            out[i] = static_cast<float>(h);
        }
    } else {
        throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
    }
    return out;
}

}  // namespace strix::glm::engine::detail
