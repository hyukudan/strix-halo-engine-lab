// GLM engine: safetensors pack index and O_DIRECT reads (copied from the MiniMax engine mm_pack.cpp, model-agnostic).
#include "glm_pack.hpp"

#include "strix/serve/json.hpp"
#include "strix/weight/admission.hpp"
#include "strix/weight/source.hpp"

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

// The index JSON, bounded like every other safetensors manifest reader (no unbounded slurp of a corrupt file).
std::string read_index(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const std::streamoff end = f.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) > weight::kSafetensorsManifestMaxBytes)
        throw std::runtime_error("safetensors index is not a readable file of at most 64 MiB: " + path);
    std::string text(static_cast<std::size_t>(end), '\0');
    f.seekg(0);
    if (!f.read(text.data(), static_cast<std::streamsize>(text.size()))) throw std::runtime_error("cannot read " + path);
    return text;
}

std::size_t dtype_width(const std::string& dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32") return 4;
    return 0;
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
    const std::string index_path = dir_ + "/model.safetensors.index.json";
    const serve::Json j = serve::parse_json_unique(read_index(index_path));
    const serve::Json* weight_map = j.find("weight_map");
    if (weight_map == nullptr || !weight_map->is_object()) throw std::runtime_error("no weight_map object in " + index_path);
    // Shard names come from the file: only plain basenames inside dir_ are opened (shared rule with the weight readers).
    std::map<std::string, std::uint32_t> file_ids;
    for (const auto& [k, v] : weight_map->as_object()) {
        if (!v.is_string() || !weight::shard_basename_ok(v.as_string()))
            throw std::runtime_error("invalid safetensors shard name for " + k + " in " + index_path);
        file_ids.emplace(v.as_string(), 0U);
    }
    std::vector<std::string> files;  // listed shards in name order (file numbers are internal to the reader)
    for (auto& [name, id] : file_ids) {
        id = static_cast<std::uint32_t>(files.size());
        files.push_back(name);
    }
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
        // Header through the shared safetensors indexer: header length bounded by the file size, every tensor's
        // data_offsets checked (two non-negative integers, ordered, inside the file, = shape x dtype width, no overlap).
        std::shared_ptr<const weight::Source> source;
        weight::Status status = weight::open_source(path, source, files[f]);
        weight::TensorIndex index;
        if (status) status = weight::index_safetensors(source, index);
        if (!status) throw std::runtime_error("GLM pack " + path + ": " + status.detail);
        // The validated header must describe the file the tensor reads use (when the source is a file it can name).
        struct stat st{};
        const auto key = weight::source_file_key(*source);
        if (::fstat(fd, &st) != 0 || (key && (key->device != static_cast<std::uint64_t>(st.st_dev) ||
                                              key->file != static_cast<std::uint64_t>(st.st_ino))))
            throw std::runtime_error("GLM pack " + path + ": shard changed while it was opened");
        for (const auto& raw : index.tensors) {
            TensorInfo t;
            t.dtype = raw.wire_dtype;
            t.shape = raw.extents;
            t.file = f;
            t.offset = raw.payload.range.offset;
            t.bytes = raw.payload.range.bytes;
            if (!tensors_.emplace(raw.name, std::move(t)).second) throw std::runtime_error("tensor defined twice: " + raw.name);
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
    if (offset > t.bytes || n > t.bytes - offset) throw std::runtime_error("read past tensor end");
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
    const std::size_t width = dtype_width(t.dtype);
    if (width == 0) throw std::runtime_error("unsupported dtype " + t.dtype + " for " + name);
    if (t.bytes % width != 0) throw std::runtime_error("tensor " + name + " size is not a whole number of " + t.dtype + " elements");
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
