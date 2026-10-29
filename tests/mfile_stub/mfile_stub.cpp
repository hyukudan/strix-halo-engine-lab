// Buffered, synchronous API contract double. It does not qualify production session verification or scheduling.
#include "strix_stream/ss.h"
#include "json.hpp"
#include "sha256.hpp"
#include "xxh64.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ss_mfile {
    std::filesystem::path primary, data;
    std::uint64_t size = 0, chunk = 0;
    std::map<std::uint64_t, std::uint64_t> slots;
    ss_mfile_counters counters{};
    std::mutex mutex;
};
namespace {
thread_local std::string error;
std::atomic<std::uint64_t> opens{0}, preads{0}, preadvs{0};
std::string contents(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("stub cannot open fixture");
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
const ts::Json& field(const ts::Json& j, const char* name) {
    const auto* v = j.find(name);
    if (!v) throw std::runtime_error("stub fixture missing field");
    return *v;
}
std::uint64_t number(const ts::Json& j, const char* name) { return ts::json_u64(field(j, name), name); }
}
extern "C" {
std::uint64_t strix_mfile_stub_opens() { return opens.load(); }
std::uint64_t strix_mfile_stub_preads() { return preads.load(); }
std::uint64_t strix_mfile_stub_preadvs() { return preadvs.load(); }
const char* ss_last_error() { return error.c_str(); }
void ss_mfile_options_defaults_sized(ss_mfile_options* o, std::uint32_t n) {
    *o = {};
    o->struct_size = n; o->direct_io = 1; o->drive_readers = 4;
}
ss_status ss_mfile_open(const char* path, const char* dirs, const ss_mfile_options* opt, ss_mfile** out) {
    ++opens;
    *out = nullptr;
    try {
        auto f = std::make_unique<ss_mfile>();
        f->primary = std::filesystem::u8path(path);
        f->size = std::filesystem::file_size(f->primary);
        f->counters.struct_size = sizeof(ss_mfile_counters);
        f->counters.locations = 1;
        f->counters.verify_mode = opt->verify_mode;
        if (dirs && *dirs) {
            const auto dir = std::filesystem::u8path(std::string(dirs).substr(0, std::string(dirs).find(';')));
            const auto manifest = dir / (f->primary.filename().string() + ".stripe.json");
            if (std::filesystem::exists(manifest)) {
                auto j = ts::json_parse(contents(manifest));
                const auto expected = field(j, "digest").text;
                j.obj.erase("digest");
                const auto canonical = ts::json_write(j);
                if (std::stoull(expected, nullptr, 16) != ts::xxh64(canonical.data(), canonical.size()))
                    throw std::runtime_error("stub fixture digest mismatch");
                const auto primary = contents(f->primary);
                const auto sha = ts::sha256_hex(primary.data(), primary.size());
                if (sha != field(field(j, "primary"), "sha256").text ||
                    (opt->primary_sha256 && sha != opt->primary_sha256)) throw std::runtime_error("stub primary SHA mismatch");
                f->chunk = number(j, "chunk_bytes");
                f->data = dir / field(field(j, "data"), "name").text;
                const auto data = contents(f->data);
                const auto& hashes = field(j, "hashes").arr;
                for (const auto& r : field(j, "runs").arr) {
                    const auto first = ts::json_u64(r.arr.at(0), "first");
                    const auto count = ts::json_u64(r.arr.at(1), "count");
                    const auto slot = ts::json_u64(r.arr.at(2), "slot");
                    for (std::uint64_t i = 0; i < count; ++i) {
                        const auto n = std::min(f->chunk, f->size - (first + i) * f->chunk);
                        const auto off = (slot + i) * f->chunk;
                        if (off + f->chunk > data.size() || ts::xxh64(data.data() + off, static_cast<std::size_t>(n)) !=
                            std::stoull(hashes.at(static_cast<std::size_t>(slot + i)).text, nullptr, 16))
                            throw std::runtime_error("stub fixture chunk mismatch");
                        f->slots[first + i] = slot + i;
                        f->counters.verified_bytes += n;
                    }
                }
                f->counters.locations = 2;
            }
        }
        *out = f.release();
        return SS_OK;
    } catch (const std::exception& e) { error = e.what(); return SS_E_IO; }
}
void ss_mfile_close(ss_mfile* f) { delete f; }
std::uint64_t ss_mfile_size(const ss_mfile* f) { return f->size; }
std::uint32_t ss_mfile_alignment(const ss_mfile*) { return 4096; }
int ss_mfile_multi(const ss_mfile* f) { return !f->slots.empty(); }
ss_status ss_mfile_pread(ss_mfile* f, void* dst, std::uint64_t bytes, std::uint64_t offset, std::uint64_t* got) {
    ++preads;
    *got = 0;
    if (offset % 4096 || bytes % 4096 || reinterpret_cast<std::uintptr_t>(dst) % 4096 || offset > f->size ||
        bytes > (f->size + 4095) / 4096 * 4096 - offset) {
        error = "stub rejected unaligned/out-of-bounds IO";
        return SS_E_INVALID;
    }
    std::lock_guard<std::mutex> lock(f->mutex);
    const auto end = offset + std::min(bytes, f->size - offset);
    auto* p = static_cast<char*>(dst);
    while (offset < end) {
        const auto c = f->chunk ? offset / f->chunk : 0;
        const auto it = f->slots.find(c);
        const bool secondary = it != f->slots.end();
        const auto n = f->chunk ? std::min(end - offset, f->chunk - offset % f->chunk) : end - offset;
        const auto phys = secondary ? it->second * f->chunk + offset % f->chunk : offset;
        std::ifstream in(secondary ? f->data : f->primary, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(phys));
        in.read(p, static_cast<std::streamsize>(n));
        if (std::uint64_t(in.gcount()) != n) { error = "stub short read"; return SS_E_IO; }
        const unsigned loc = secondary ? 1 : 0;
        f->counters.bytes[loc] += n;
        f->counters.reads[loc]++;
        offset += n; p += n; *got += n;
    }
    return SS_OK;
}
ss_status ss_mfile_preadv(ss_mfile* f, const ss_mfile_io* ios, std::uint32_t n) {
    ++preadvs;
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint64_t got = 0;
        const auto rc = ss_mfile_pread(f, ios[i].dst, ios[i].bytes, ios[i].offset, &got);
        if (rc != SS_OK) return rc;
    }
    return SS_OK;
}
ss_status ss_mfile_get_counters(const ss_mfile* f, ss_mfile_counters* out) {
    auto* mutable_f = const_cast<ss_mfile*>(f);
    std::lock_guard<std::mutex> lock(mutable_f->mutex);
    *out = f->counters;
    return SS_OK;
}
const char* ss_mfile_location_path(const ss_mfile*, int) { return nullptr; }
const char* ss_mfile_open_report(const ss_mfile*) { return ""; }
}
