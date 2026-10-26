// MiniMax engine: safetensors pack index and O_DIRECT reads (see mm_common.hpp).
#include "mm_common.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace strix::minimax::engine::detail {

namespace {

constexpr std::uint64_t kAlign = 4096;
constexpr std::uint64_t kBounce = std::uint64_t{64} << 20;

struct JsonParser {
    const char* p;
    const char* end;
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    Json parse() {
        ws();
        if (p >= end) throw std::runtime_error("json: eof");
        if (*p == '{') {
            ++p;
            JsonObj o;
            ws();
            if (*p == '}') { ++p; return {o}; }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                if (*p++ != ':') throw std::runtime_error("json: ':'");
                o.emplace(std::move(k), parse());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != '}') throw std::runtime_error("json: '}'");
                return {o};
            }
        }
        if (*p == '[') {
            ++p;
            JsonArr a;
            ws();
            if (*p == ']') { ++p; return {a}; }
            for (;;) {
                a.push_back(parse());
                ws();
                if (*p == ',') { ++p; continue; }
                if (*p++ != ']') throw std::runtime_error("json: ']'");
                return {a};
            }
        }
        if (*p == '"') return {string()};
        if (std::strncmp(p, "null", 4) == 0) { p += 4; return {nullptr}; }
        if (std::strncmp(p, "true", 4) == 0) { p += 4; return {1.0}; }
        if (std::strncmp(p, "false", 5) == 0) { p += 5; return {0.0}; }
        char* e = nullptr;
        const double d = std::strtod(p, &e);
        if (e == p) throw std::runtime_error("json: value");
        p = e;
        return {d};
    }
    std::string string() {
        if (*p++ != '"') throw std::runtime_error("json: string");
        std::string s;
        while (*p != '"') {
            if (*p == '\\') { ++p; s += *p == 'n' ? '\n' : *p; ++p; continue; }
            s += *p++;
        }
        ++p;
        return s;
    }
};

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

float bf16f(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16U;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

Json parse_json(const char* begin, const char* end) {
    JsonParser jp{begin, end};
    return jp.parse();
}

PackReader::PackReader(std::string dir) : dir_(std::move(dir)) {
    const std::string idx = read_text(dir_ + "/model.safetensors.index.json");
    const Json j = parse_json(idx.data(), idx.data() + idx.size());
    std::map<std::string, std::uint32_t> file_ids;
    std::vector<std::string> files;
    for (const auto& [k, v] : j.obj().at("weight_map").obj())
        if (file_ids.emplace(v.str(), static_cast<std::uint32_t>(files.size())).second) files.push_back(v.str());
    const bool want_direct = std::getenv("STRIX_MM_LOAD_DIRECT") == nullptr || std::atoi(std::getenv("STRIX_MM_LOAD_DIRECT")) != 0;
    for (std::uint32_t f = 0; f < files.size(); ++f) {
        const std::string path = dir_ + "/" + files[f];
        int fd = want_direct ? ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT) : -1;
        direct_.push_back(fd >= 0 ? 1 : 0);
        if (fd < 0) fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("open " + path);
        fds_.push_back(fd);
        // Header: 8-byte length + JSON, read through the aligned bounce once it exists; a plain read is simpler here.
        const int hfd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (hfd < 0) throw std::runtime_error("open " + path);
        std::uint64_t hb = 0;
        pread_all(hfd, &hb, 8, 0, path);
        std::string head(hb, '\0');
        pread_all(hfd, head.data(), hb, 8, path);
        ::close(hfd);
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
            tensors_.emplace(name, std::move(t));
        }
    }
    void* p = nullptr;
    if (posix_memalign(&p, kAlign, kBounce) != 0) throw std::runtime_error("bounce buffer");
    bounce_ = static_cast<std::uint8_t*>(p);
}

PackReader::~PackReader() {
    for (int fd : fds_) ::close(fd);
    std::free(bounce_);
}

const TensorInfo& PackReader::info(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("tensor not in pack: " + name);
    return it->second;
}

void PackReader::read(const TensorInfo& t, std::uint64_t offset, std::uint64_t n, void* dst) {
    if (offset + n > t.bytes) throw std::runtime_error("read past tensor end");
    const int fd = fds_[t.file];
    std::uint64_t pos = t.offset + offset;
    auto* d = static_cast<std::uint8_t*>(dst);
    bytes_read_ += n;
    if (!direct_[t.file]) {
        pread_all(fd, d, n, pos, dir_);
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

}  // namespace strix::minimax::engine::detail
