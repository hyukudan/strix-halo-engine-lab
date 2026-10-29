// CPU-only model reads through the same range and queued-record APIs as the prefill/replica.
#include "strix/ds41_file_reader.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
using strix::ds41::FileReader;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::string json_string(const std::string& text) {
    std::string out = "\"";
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') out += '\\';
        if (c >= 32) out += static_cast<char>(c);
    }
    return out + '"';
}
std::string kernel_version() {
#ifdef _WIN32
    OSVERSIONINFOW v{};
    v.dwOSVersionInfoSize = sizeof(v);
    const auto module = GetModuleHandleW(L"ntdll.dll");
    const auto version = reinterpret_cast<LONG (WINAPI*)(OSVERSIONINFOW*)>(GetProcAddress(module, "RtlGetVersion"));
    if (version && version(&v) == 0)
        return std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion) + "." + std::to_string(v.dwBuildNumber);
#else
    struct utsname u{};
    if (uname(&u) == 0) return u.release;
#endif
    return "unknown";
}
std::string iommu_perfopt() {
#ifndef _WIN32
    std::ifstream in("/sys/module/amdgpu/parameters/iommu_perfopt");
    std::string value;
    if (in >> value) return value;
#endif
    return "unavailable";
}
std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b) throw std::runtime_error("GGUF size overflow");
    return a * b;
}
class Header {
public:
    explicit Header(const std::string& path) : in(path, std::ios::binary), size(std::filesystem::file_size(path)) {
        if (!in) throw std::runtime_error("cannot open GGUF");
    }
    std::uint64_t number(unsigned bytes) {
        std::uint64_t v = 0;
        for (unsigned i = 0; i < bytes; ++i) {
            const int c = in.get();
            if (c == EOF) throw std::runtime_error("truncated GGUF header");
            v |= std::uint64_t(static_cast<unsigned char>(c)) << (i * 8);
        }
        return v;
    }
    std::string string() {
        const auto n = number(8);
        if (n > (1U << 24)) throw std::runtime_error("GGUF string too large");
        std::string s(static_cast<std::size_t>(n), '\0');
        in.read(s.data(), static_cast<std::streamsize>(n));
        if (!in) throw std::runtime_error("truncated GGUF string");
        return s;
    }
    std::uint64_t value(unsigned type, unsigned depth = 0) {
        if (depth > 8) throw std::runtime_error("GGUF metadata nesting");
        switch (type) {
        case 0: case 1: case 7: return number(1);
        case 2: case 3: return number(2);
        case 4: case 5: case 6: return number(4);
        case 10: case 11: case 12: return number(8);
        case 8: string(); return 0;
        case 9: {
            const auto t = static_cast<unsigned>(number(4));
            const auto n = number(8);
            if (n > size) throw std::runtime_error("GGUF metadata count");
            for (std::uint64_t i = 0; i < n; ++i) value(t, depth + 1);
            return 0;
        }
        default: throw std::runtime_error("unknown GGUF metadata type");
        }
    }
    struct Tensor { std::uint64_t offset, bytes; };
    std::map<std::string, Tensor> tensors() {
        if (number(4) != 0x46554747) throw std::runtime_error("not GGUF");
        const auto version = number(4);
        if (version != 2 && version != 3) throw std::runtime_error("unsupported GGUF version");
        const auto nt = number(8), nk = number(8);
        if (nt > (1U << 20) || nk > (1U << 20)) throw std::runtime_error("GGUF header count");
        std::uint64_t alignment = 32;
        for (std::uint64_t k = 0; k < nk; ++k) {
            const auto key = string();
            const auto t = static_cast<unsigned>(number(4));
            const auto v = value(t);
            if (key == "general.alignment") alignment = v;
        }
        if (!alignment || alignment > (1U << 20) || (alignment & (alignment - 1))) throw std::runtime_error("GGUF alignment");
        const std::map<unsigned, std::pair<unsigned, unsigned>> types = {
            {0,{1,4}}, {1,{1,2}}, {2,{32,18}}, {3,{32,20}}, {8,{32,34}}, {10,{256,84}},
            {11,{256,110}}, {12,{256,144}}, {13,{256,176}}, {14,{256,210}}, {16,{256,66}},
            {18,{256,98}}, {24,{1,1}}, {26,{1,4}}, {28,{1,8}}, {30,{1,2}}, {39,{32,17}}};
        std::map<std::string, Tensor> result;
        for (std::uint64_t k = 0; k < nt; ++k) {
            const auto name = string();
            const auto nd = number(4);
            if (!nd || nd > 4) throw std::runtime_error("GGUF dimensions");
            std::uint64_t elems = 1, first = 0;
            for (unsigned d = 0; d < nd; ++d) {
                const auto n = number(8);
                if (!n) throw std::runtime_error("empty GGUF tensor");
                if (!d) first = n;
                elems = multiply(elems, n);
            }
            const auto t = static_cast<unsigned>(number(4));
            const auto off = number(8);
            const auto it = types.find(t);
            if (it == types.end()) throw std::runtime_error("unknown GGUF tensor type: " + name);
            const auto [block, bytes] = it->second;
            if (first % block) throw std::runtime_error("GGUF block alignment");
            if (!result.emplace(name, Tensor{off, multiply(elems / block, bytes)}).second)
                throw std::runtime_error("duplicate tensor");
        }
        const auto pos = static_cast<std::uint64_t>(in.tellg());
        const auto data = (pos + alignment - 1) / alignment * alignment;
        for (auto& [name, tensor] : result) {
            if (data > size || tensor.offset > size - data) throw std::runtime_error("GGUF tensor offset");
            tensor.offset += data;
            if (tensor.bytes > size - tensor.offset) throw std::runtime_error("GGUF tensor bounds");
        }
        return result;
    }
private:
    std::ifstream in;
    std::uint64_t size;
};

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>((v.size() - 1) * q)];
}
struct Buffer {
    std::uint8_t* data;
    explicit Buffer(std::size_t n) {
#ifdef _WIN32
        data = static_cast<std::uint8_t*>(_aligned_malloc(n, FileReader::alignment));
#else
        data = static_cast<std::uint8_t*>(std::aligned_alloc(FileReader::alignment, n));
#endif
        if (!data) throw std::bad_alloc();
    }
    ~Buffer() {
#ifdef _WIN32
        _aligned_free(data);
#else
        std::free(data);
#endif
    }
};
}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) throw std::invalid_argument("MODEL.gguf --mode prefill|units|sparse|engram [--tokens N] [--jobs N] [--weight-dirs LIST] [--drive-readers N] [--weight-verify cached|always] [--buffered] [--json OUT]");
    const std::string model = argv[1];
    strix::ds41::WeightReadOptions opt;
    std::string mode = "prefill", output, preload = "uncontrolled";
    unsigned tokens = 1, jobs = 100;
    bool direct = true, dirs_given = false, verify_given = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto next = [&]() -> std::string {
            if (++i == argc) throw std::invalid_argument("missing " + arg);
            return argv[i];
        };
        if (arg == "--mode") mode = next();
        else if (arg == "--tokens") tokens = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--jobs") jobs = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--weight-dirs") { opt.weight_dirs = next(); dirs_given = true; }
        else if (arg == "--drive-readers") opt.set_drive_readers(next());
        else if (arg == "--weight-verify") { opt.weight_verify = next(); verify_given = true; }
        else if (arg == "--buffered") direct = false;
        else if (arg == "--preload-state") preload = next();
        else if (arg == "--json") output = next();
        else throw std::invalid_argument("unknown " + arg);
    }
    opt.environment(dirs_given, verify_given, model);
    if (!tokens || !jobs) throw std::invalid_argument("tokens and jobs must be positive");
    if (preload != "cold" && preload != "warm" && preload != "uncontrolled") throw std::invalid_argument("--preload-state cold|warm|uncontrolled (recorded, not changed)");
    const auto tensors = Header(model).tensors();
    const auto tensor = [&](unsigned layer, const std::string& suffix) -> Header::Tensor {
        const auto key = "blk." + std::to_string(layer) + "." + suffix + ".weight";
        const auto it = tensors.find(key);
        if (it == tensors.end()) throw std::runtime_error("missing " + key);
        return it->second;
    };
    const auto opened = Clock::now();
    FileReader reader(model, direct, 8, opt);
    const double open_seconds = seconds(opened);
    const auto before = reader.multi_counters();
    Buffer buffer((64U << 20) + 8192);
    std::mt19937_64 rng(42);
    std::vector<double> reads, layers, batches;
    std::uint64_t bytes = 0, covering_bytes = 0;
    const auto range = [&](std::uint64_t off, std::uint64_t n) {
        const auto start = Clock::now();
        reader.read_range(off, static_cast<std::size_t>(n), buffer.data, 8U << 20);
        reads.push_back(seconds(start));
        bytes += n;
        covering_bytes += direct || reader.multi() ? std::min<std::uint64_t>((off + n + 4095) / 4096 * 4096, reader.size()) - off / 4096 * 4096 : n;
    };
    const auto start = Clock::now();
    if (mode == "prefill") {
        for (unsigned l = 0; l < 40; ++l) {
            const auto layer = Clock::now();
            for (const auto* part : {"gate", "up", "down"}) {
                const auto t = tensor(l, std::string("ffn_") + part + "_exps");
                // Match copy_file_to_device's aligned covering read of each 64 MiB payload.
                for (std::uint64_t done = 0; done < t.bytes;) {
                    const auto at = t.offset + done;
                    const auto n = std::min<std::uint64_t>(64U << 20, t.bytes - done);
                    const auto lo = at / 4096 * 4096;
                    const auto hi = std::min<std::uint64_t>((at + n + 4095) / 4096 * 4096, reader.size());
                    range(lo, hi - lo);
                    done += n;
                }
            }
            layers.push_back(seconds(layer));
        }
    } else if (mode == "units") {
        for (unsigned token = 0; token < tokens; ++token) {
            for (unsigned l = 0; l < 40; ++l) {
                const auto layer = Clock::now();
                std::set<unsigned> experts;
                while (experts.size() < 6) experts.insert(static_cast<unsigned>(rng() % 384));
                for (const auto e : experts) {
                    for (const auto* part : {"gate", "up", "down"}) {
                        const auto t = tensor(l, std::string("ffn_") + part + "_exps");
                        if (t.bytes % 384 || t.bytes / 384 > (64U << 20)) throw std::runtime_error("expert slice size");
                        range(t.offset + e * (t.bytes / 384), t.bytes / 384);
                    }
                }
                layers.push_back(seconds(layer));
            }
        }
    } else if (mode == "sparse") {
        // the file-tier decode step (CudaPrefill::step, non-resident layers): the routed experts' parts read one after
        // another as aligned covering ranges into aligned staging
        for (unsigned token = 0; token < tokens; ++token) {
            for (unsigned l = 0; l < 40; ++l) {
                const auto layer = Clock::now();
                std::set<unsigned> experts;
                while (experts.size() < 6) experts.insert(static_cast<unsigned>(rng() % 384));
                for (const auto e : experts) {
                    for (const auto* part : {"gate", "up", "down"}) {
                        const auto t = tensor(l, std::string("ffn_") + part + "_exps");
                        if (t.bytes % 384 || t.bytes / 384 > (64U << 20)) throw std::runtime_error("expert slice size");
                        const std::uint64_t off = t.offset + e * (t.bytes / 384), a0 = off / 4096 * 4096;
                        const std::uint64_t a1 = std::min<std::uint64_t>((off + t.bytes / 384 + 4095) / 4096 * 4096, reader.size());
                        range(a0, a1 - a0);
                    }
                }
                layers.push_back(seconds(layer));
            }
        }
    } else if (mode == "engram") {
        std::array<std::uint64_t, 24> offsets;
        std::array<std::uint8_t, 24 * 264> rows;
        for (unsigned job = 0; job < jobs; ++job) {
            // Sequential layer jobs, as in the engine; one batch of 24 rows at a time.
            for (const auto& [name, t] : tensors) {
                if (!name.starts_with("blk.") || !name.ends_with(".engram_embd.weight")) continue;
                if (t.bytes < 264) throw std::runtime_error("short Engram table");
                for (auto& off : offsets) {
                    off = t.offset + (rng() % (t.bytes / 264)) * 264;
                    covering_bytes += direct || reader.multi() ? std::min<std::uint64_t>((off + 264 + 4095) / 4096 * 4096, reader.size()) - off / 4096 * 4096 : 264;
                }
                const auto batch = Clock::now();
                reader.read_records_queued(offsets, 264, rows.data());
                batches.push_back(seconds(batch));
                bytes += rows.size();
            }
        }
    } else throw std::invalid_argument("--mode prefill|units|sparse|engram");
    const double elapsed = seconds(start);
    const auto stats = reader.stats();
    const auto after = reader.multi_counters();
    std::ostringstream j;
    j.precision(12);
    j << "{\"mode\":\"" << mode << "\",\"multi\":" << (reader.multi() ? "true" : "false")
      << ",\"bytes\":" << bytes << ",\"seconds\":" << elapsed << ",\"gb_s\":" << double(bytes) / 1e9 / elapsed
      << ",\"open_seconds\":" << open_seconds << ",\"direct_mask\":" << (reader.multi() ? after.direct_mask : reader.direct() ? 1 : 0)
      << ",\"verify_mode\":\"" << (reader.multi() ? (after.verify_mode == SS_MFILE_VERIFY_ALWAYS ? "always" : after.verify_mode == SS_MFILE_VERIFY_OFF ? "off" : "cached") : opt.weight_verify)
      << "\",\"drive_readers\":" << opt.drive_readers << ",\"tokens\":" << tokens << ",\"jobs\":" << jobs
      << ",\"host_gib\":0,\"layers_per_tier\":{\"device\":0,\"host\":0,\"file\":40},\"preload_state\":\"" << preload
      << "\",\"page_cache_bytes\":0,\"page_hits\":" << stats.page_hits << ",\"covering_read_bytes\":" << covering_bytes;
    for (const auto& [name, values] : std::array<std::pair<const char*, const std::vector<double>*>,3>{{{"layer",&layers},{"read",&reads},{"job",&batches}}})
        j << ",\"" << name << "\":{\"count\":" << values->size() << ",\"p50_s\":" << percentile(*values, .50) << ",\"p99_s\":" << percentile(*values, .99) << "}";
    j << ",\"location_bytes\":[";
    const auto locations = reader.multi() ? after.locations : 1U;
    for (unsigned l = 0; l < locations; ++l)
        j << (l ? "," : "") << (reader.multi() ? after.bytes[l] - before.bytes[l] : covering_bytes);
    j << "],\"kernel\":" << json_string(kernel_version()) << ",\"amdgpu_iommu_perfopt\":" << json_string(iommu_perfopt())
      << ",\"session\":" << reader.multi_stats_json() << "}";
    std::cout << j.str() << '\n';
    if (!output.empty()) {
        std::ofstream out(output);
        out << j.str() << '\n';
        if (!out) throw std::runtime_error("cannot write --json");
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "ds41 read bench: " << e.what() << '\n';
    return 2;
}
