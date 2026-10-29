#include "strix/ds41_file_reader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#ifdef STRIX_MFILE_CONTRACT_STUB
extern "C" std::uint64_t strix_mfile_stub_opens();
extern "C" std::uint64_t strix_mfile_stub_preads();
extern "C" std::uint64_t strix_mfile_stub_preadvs();
#endif
#ifdef STRIX_HAVE_TIERSTREAM_MFILE
#include "json.hpp"
#include "sha256.hpp"
#include "xxh64.hpp"
#ifndef STRIX_MFILE_CONTRACT_STUB
#include "mfile.hpp"
#endif
#endif

namespace fs = std::filesystem;
using strix::ds41::FileReader;
using strix::ds41::WeightReadOptions;

namespace {
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Temp {
    fs::path path = fs::current_path() / ("reader-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temp() { fs::create_directories(path / "stripe"); }
    ~Temp() { std::error_code ec; fs::remove_all(path, ec); }
};
struct CacheEnvironment {
#ifdef _WIN32
    const char* name = "LOCALAPPDATA";
#else
    const char* name = "XDG_CACHE_HOME";
#endif
    bool present = std::getenv(name) != nullptr;
    std::string saved = present ? std::getenv(name) : "";
    void set(const char* value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value) setenv(name, value, 1); else unsetenv(name);
#endif
    }
    explicit CacheEnvironment(const fs::path& path) {
        fs::create_directories(path);
        set(path.string().c_str());
    }
    ~CacheEnvironment() { set(present ? saved.c_str() : nullptr); }
};
void write(const fs::path& p, const void* bytes, std::size_t n) {
    std::ofstream out(p, std::ios::binary);
    out.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(n));
    check(bool(out), "write fixture");
}
#ifdef STRIX_HAVE_TIERSTREAM_MFILE
std::string hex16(std::uint64_t x) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(x));
    return buf;
}
std::string stripe(const Temp& temp, const std::vector<std::uint8_t>& data) {
    constexpr std::size_t chunk = 1U << 20;
    const auto digest = ts::sha256_hex(data.data(), data.size());
    ts::Json m = ts::Json::object(), primary = ts::Json::object(), placed = ts::Json::object();
    primary["basename"] = ts::Json::string("model.bin");
    primary["size"] = ts::Json::number(std::uint64_t(data.size()));
    primary["sha256"] = ts::Json::string(digest);
    m["schema"] = ts::Json::string("tierstream/stripe-v1");
    m["chunk_bytes"] = ts::Json::number(std::uint64_t(chunk));
    m["primary"] = primary;
    m["runs"] = ts::Json::array();
    m["hashes"] = ts::Json::array();
    std::ofstream out(temp.path / "stripe" / "weights.data", std::ios::binary);
    std::uint64_t slot = 0;
    std::vector<std::uint8_t> page(chunk);
    for (std::size_t c = 1; c * chunk < data.size(); c += 2) {
        const auto n = std::min(chunk, data.size() - c * chunk);
        std::fill(page.begin(), page.end(), 0);
        std::memcpy(page.data(), data.data() + c * chunk, n);
        out.write(reinterpret_cast<const char*>(page.data()), chunk);
        ts::Json run = ts::Json::array();
        run.push(ts::Json::number(std::uint64_t(c)));
        run.push(ts::Json::number(std::uint64_t(1)));
        run.push(ts::Json::number(slot++));
        m["runs"].push(run);
        m["hashes"].push(ts::Json::string(hex16(ts::xxh64(page.data(), n))));
    }
    check(bool(out), "write stripe");
    out.close();
    placed["name"] = ts::Json::string("weights.data");
    placed["size"] = ts::Json::number(slot * chunk);
    m["data"] = placed;
    const auto text = ts::json_write(m);
    m["digest"] = ts::Json::string(hex16(ts::xxh64(text.data(), text.size())));
    const auto manifest = ts::json_write(m);
    write(temp.path / "stripe" / "model.bin.stripe.json", manifest.data(), manifest.size());
    return digest;
}
#endif

void exercise(FileReader& single, FileReader& reader, const std::vector<std::uint8_t>& data) {
    std::mt19937_64 rng(42);
    std::vector<std::uint8_t> a((1U << 20) + 8192), b(a.size());
    for (unsigned i = 0; i < 32; ++i) {
        const auto bytes = std::size_t(rng() % (1U << 20) + 1);
        const auto off = rng() % (data.size() - bytes + 1);
        const auto pad = std::size_t(rng() % 4096);
        single.read_range(off, bytes, a.data() + pad);
        reader.read_range(off, bytes, b.data() + pad);
        check(std::memcmp(a.data() + pad, b.data() + pad, bytes) == 0, "range equality");
        check(std::memcmp(data.data() + off, b.data() + pad, bytes) == 0, "range oracle");
    }
    // An aligned, complete staging range must reach the session as one call.
#ifdef _WIN32
    auto* aligned = static_cast<std::uint8_t*>(_aligned_malloc(64U << 20, FileReader::alignment));
#else
    auto* aligned = static_cast<std::uint8_t*>(std::aligned_alloc(FileReader::alignment, 64U << 20));
#endif
    check(aligned != nullptr, "aligned allocation");
#ifdef STRIX_MFILE_CONTRACT_STUB
    const auto range_calls = strix_mfile_stub_preads();
#endif
    reader.read_range(0, 64U << 20, aligned);
#ifdef STRIX_MFILE_CONTRACT_STUB
    if (reader.multi()) check(strix_mfile_stub_preads() == range_calls + 1, "one staging range call");
#endif
    const bool equal = std::memcmp(aligned, data.data(), 64U << 20) == 0;
    reader.read_range(123, (1U << 20) + 50, aligned + 123);
    check(std::memcmp(aligned + 123, data.data() + 123, (1U << 20) + 50) == 0, "head/middle/tail equality");
    reader.read_range(data.size() - 123, 123, aligned + 1);
    check(std::memcmp(aligned + 1, data.data() + data.size() - 123, 123) == 0, "EOF bounce equality");
#ifdef _WIN32
    _aligned_free(aligned);
#else
    std::free(aligned);
#endif
    check(equal, "staging range equality");
    const std::vector<std::uint64_t> offsets = {0, 4000, 4090, (1U << 20) - 100, (1U << 20) + 4000,
                                               (3U << 20) + 4090, 4090, data.size() - 264};
    a.resize(offsets.size() * 264); b.resize(a.size());
    for (std::size_t cache : {0U, 8U * 4096}) {
        single.set_page_cache(cache); reader.set_page_cache(cache);
        for (unsigned pass = 0; pass < 2; ++pass) {
            single.read_records(offsets, 264, a.data(), 1);
            reader.read_records(offsets, 264, b.data(), 1);
            check(a == b, "records equality");
        }
        if (cache) check(reader.stats().page_hits > 0, "page cache hit");
    }
    single.read_records_queued(offsets, 264, a.data());
#ifdef STRIX_MFILE_CONTRACT_STUB
    const auto queued_calls = strix_mfile_stub_preadvs();
#endif
    reader.read_records_queued(offsets, 264, b.data());
#ifdef STRIX_MFILE_CONTRACT_STUB
    if (reader.multi()) check(strix_mfile_stub_preadvs() == queued_calls + 1, "one queued batch call");
#endif
    check(a == b, "queued equality");
    for (unsigned pass = 0; pass < 3; ++pass) {
        const std::size_t count = pass == 1 ? offsets.size() / 2 : offsets.size();
        reader.read_records_queued(std::span(offsets).first(count), 264, b.data());
        check(std::memcmp(a.data(), b.data(), count * 264) == 0, "queued scratch reuse equality");
    }
    for (std::size_t i = 0; i < offsets.size(); ++i)
        check(std::memcmp(b.data() + i * 264, data.data() + offsets[i], 264) == 0, "record oracle");
}
}  // namespace

int main() try {
    Temp temp;
    CacheEnvironment cache(temp.path / "cache");
    std::vector<std::uint8_t> data((64U << 20) + 123);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>((i * 73 + i / 4096) ^ (i >> 17));
    const auto path = temp.path / "model.bin";
    write(path, data.data(), data.size());
    WeightReadOptions options;
    {
        const char* old_dirs = std::getenv("STRIX_WEIGHT_DIRS");
        const char* old_verify = std::getenv("STRIX_WEIGHT_VERIFY");
        const std::string saved_dirs = old_dirs ? old_dirs : "", saved_verify = old_verify ? old_verify : "";
        const auto env = [](const char* name, const char* value) {
#ifdef _WIN32
            _putenv_s(name, value ? value : "");
#else
            if (value) setenv(name, value, 1); else unsetenv(name);
#endif
        };
        env("STRIX_WEIGHT_DIRS", (temp.path / "stripe").string().c_str()); env("STRIX_WEIGHT_VERIFY", "always");
        options.environment(false, false, path);
        check(options.weight_dirs == (temp.path / "stripe").string() && options.weight_verify == "always", "environment defaults");
        options.weight_dirs.clear(); options.weight_verify = "cached";
        options.environment(true, true, path);
        check(options.weight_dirs.empty() && options.weight_verify == "cached", "flags override environment");
        env("STRIX_WEIGHT_DIRS", nullptr); env("STRIX_WEIGHT_VERIFY", "invalid");
        options.environment(false, false, path);
        check(options.weight_dirs.empty() && options.weight_verify == "cached", "stray verify ignored without dirs");
        options.weight_dirs = temp.path.string();
        options.environment(true, false, path);
        check(options.weight_verify == "cached", "stray verify ignored with primary-only dirs");
        options.weight_dirs = (path / "inaccessible").string();
        check(!options.has_secondary(path), "inaccessible secondary ignored");
        options.environment(true, false, path);
        options.weight_dirs = (temp.path / "stripe").string();
        bool refused = false;
        try { options.environment(true, false, path); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "bad verify refused with secondary");
        options.weight_verify = "cached";
        env("STRIX_WEIGHT_DIRS", old_dirs ? saved_dirs.c_str() : nullptr);
        env("STRIX_WEIGHT_VERIFY", old_verify ? saved_verify.c_str() : nullptr);
    }
    options.weight_dirs = "";
    options.environment(true, true, path);
    check(options.weight_dirs.empty() && options.weight_verify == "cached", "explicit empty options");
    for (const auto* invalid : {"0", "-1", "33", "100000", "4294967296", "999999999999999999999999", "1x"}) {
        bool refused = false;
        try { options.set_drive_readers(invalid); } catch (const std::exception&) { refused = true; }
        check(refused, "invalid drive reader count");
    }
    options.set_drive_readers("32");
    check(options.drive_readers == 32, "maximum drive readers");
    options.set_drive_readers("0004");
    options.weight_dirs.clear();
#ifdef STRIX_HAVE_TIERSTREAM_MFILE
    (void)stripe(temp, data);  // the primary is hashed by tierstream when this manifest is found
#endif
    for (bool direct : {false, true}) {
#ifdef STRIX_MFILE_CONTRACT_STUB
        const auto opens = strix_mfile_stub_opens();
#elif defined(STRIX_HAVE_TIERSTREAM_MFILE)
        const auto sessions = ts::MFile::test_sessions_created();
#endif
        FileReader single(path, direct, 2);
        check(!single.multi(), "default creates no session");
        check(single.multi_stats_json() == "null", "no multi counters on default path");
        options.weight_dirs = temp.path.string() + ";" + (temp.path / ".").string();
        FileReader primary_only(path, direct, 2, options);
        check(!primary_only.multi(), "primary-only list creates no session");
        WeightReadOptions inaccessible;
        inaccessible.weight_dirs = (path / "inaccessible").string();
        FileReader ignored(path, direct, 2, inaccessible);
        check(!ignored.multi(), "inaccessible directories create no session");
#ifdef STRIX_MFILE_CONTRACT_STUB
        check(strix_mfile_stub_opens() == opens, "no session open without secondary");
#elif defined(STRIX_HAVE_TIERSTREAM_MFILE)
        check(ts::MFile::test_sessions_created() == sessions, "no session constructed without secondary");
#endif
        exercise(single, primary_only, data);
        options.weight_dirs = (temp.path / "stripe").string();
        options.weight_verify = "always";
#ifdef STRIX_HAVE_TIERSTREAM_MFILE
        FileReader multi(path, direct, 2, options);
        check(multi.multi(), "stripe admitted");
        exercise(single, multi, data);
        const auto c = multi.multi_counters();
        check(c.locations == 2 && c.bytes[0] && c.bytes[1], "both locations served");
        check(c.verify_mode == SS_MFILE_VERIFY_ALWAYS && c.fallback_bytes == 0 && c.verify_fail == 0, "verification counters");
#else
        bool refused = false;
        try { FileReader multi(path, direct, 2, options); }
        catch (const std::invalid_argument&) { refused = true; }
        check(refused, "unsupported secondary refused");
#endif
    }
    std::cout << "FileReader CPU checks passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
