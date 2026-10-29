#include "../apps/bitcalc_measurement.hpp"

#include <array>
#include <iostream>
#ifdef __linux__
#include <sys/stat.h>
#endif

namespace {
int failures = 0;
void expect(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template<class F> void rejects(F action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    expect(rejected, message);
}
void environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) ::setenv(name, value, 1); else ::unsetenv(name);
#endif
}
struct EnvironmentGuard {
    const char* name;
    std::optional<std::string> saved;
    explicit EnvironmentGuard(const char* key) : name(key) {
        if (const char* value = std::getenv(key)) saved = value;
        environment(name, nullptr);
    }
    ~EnvironmentGuard() { environment(name, saved ? saved->c_str() : nullptr); }
};
}

int main() {
    using strix::decode_cli::BitcalcMeasurement;
    EnvironmentGuard out("STRIX_BITCALC_LOGITS_OUT"), force("STRIX_BITCALC_FORCE_TOKENS");
    const std::filesystem::path root = "bitcalc-measurement-test-files";
    if (!std::filesystem::create_directory(root)) {
        std::cerr << "test fixture directory already exists\n";
        return 1;
    }
    const auto capture = (root / "capture.bin").string();
    const auto partial = capture + ".partial";
    const auto tokens = (root / "tokens.txt").string();
    const std::array<float, 4> values{0.0F, 3.0F, 2.0F, 1.0F};
    environment("STRIX_BITCALC_LOGITS_OUT", capture.c_str());
    const auto prepare = [] { BitcalcMeasurement measured; measured.prepare(4, 1, 1); };
    { std::ofstream file(capture); file << "preserved"; }
    rejects(prepare, "existing destination refused");
    expect(std::filesystem::file_size(capture) == 9U, "existing file preserved");
    std::filesystem::remove(capture);
    { std::ofstream file(partial); file << "preserved"; }
    rejects(prepare, "existing partial refused");
    expect(std::filesystem::file_size(partial) == 9U, "existing partial preserved");
    std::filesystem::remove(partial);
    std::error_code symlink_error;
    std::filesystem::create_symlink("missing-target", capture, symlink_error);
#ifdef __linux__
    expect(!symlink_error, "dangling symlink fixture created");
#endif
    if (!symlink_error) {
        rejects(prepare, "dangling final symlink refused");
        std::filesystem::remove(capture);
        std::filesystem::create_symlink("missing-target", partial);
        rejects(prepare, "dangling partial symlink refused");
        std::filesystem::remove(partial);
    }
    std::filesystem::create_directory(capture);
    rejects(prepare, "directory destination refused");
    std::filesystem::remove(capture);
    {
        BitcalcMeasurement measured;
        measured.prepare(4, 1, 1);
        expect(!std::filesystem::exists(capture) && std::filesystem::is_regular_file(partial),
            "only a regular partial exists during execution");
        expect(measured.hook()(0, 0, 1, values) == 1U, "capture preserves greedy token");
        measured.finish();
        expect(std::filesystem::is_regular_file(capture) && !std::filesystem::exists(partial) &&
            std::filesystem::file_size(capture) == 40U, "finish publishes one unchanged record");
        rejects([&] { measured.finish(); }, "second finish refused");
    }
    std::filesystem::remove(capture);
    {
        BitcalcMeasurement measured;
        measured.prepare(4, 1, 1);
        const auto hook = measured.hook();
        (void)hook(0, 0, 1, values);
        rejects([&] { (void)hook(1, 1, 1, values); }, "capture byte ceiling enforced");
        rejects([&] { measured.finish(); }, "failed write cannot publish at finish");
        expect(!std::filesystem::exists(capture), "failed finish leaves no final file");
    }
    std::filesystem::remove(partial);
#ifdef __linux__
    {
        BitcalcMeasurement measured;
        measured.prepare(4, 1, 1);
        (void)measured.hook()(0, 0, 1, values);
        std::filesystem::remove(partial);
        rejects([&] { measured.finish(); }, "rename failure propagates");
        expect(!std::filesystem::exists(capture), "rename failure leaves no final file");
    }
#endif
    {
        BitcalcMeasurement measured;
        measured.prepare(4, 1, 1);
        (void)measured.hook()(0, 0, 1, values);
        { std::ofstream file(capture); file << "preserved"; }
        rejects([&] { measured.finish(); }, "finish rechecks destination");
        expect(std::filesystem::file_size(capture) == 9U, "finish preserves late destination");
    }
    std::filesystem::remove(capture);
    std::filesystem::remove(partial);
    {
        BitcalcMeasurement measured;
        rejects([&] { measured.prepare(UINT32_MAX, 1, UINT32_MAX); }, "capture byte ceiling overflow refused");
    }
    {
        BitcalcMeasurement measured;
        measured.prepare(4, 1, 1);
        (void)measured.hook()(0, 0, 1, values);
    }
    expect(!std::filesystem::exists(capture), "destructor never publishes an unfinished capture");
    std::filesystem::remove(partial);
    environment("STRIX_BITCALC_LOGITS_OUT", nullptr);
    environment("STRIX_BITCALC_FORCE_TOKENS", tokens.c_str());
    { BitcalcMeasurement deferred; expect(deferred.enabled(), "constructor only records forcing path"); }
    const auto forcing = [] { BitcalcMeasurement measured; measured.prepare(4, 1, 1); };
    rejects(forcing, "missing forcing file refused at prepare");
    std::filesystem::create_directory(tokens);
    rejects(forcing, "forcing directory refused");
    std::filesystem::remove(tokens);
#ifdef __linux__
    expect(::mkfifo(tokens.c_str(), 0600) == 0, "FIFO fixture created");
    rejects(forcing, "forcing FIFO refused without opening");
    std::filesystem::remove(tokens);
#endif
    { std::ofstream file(tokens); file << std::string(4129U, ' '); }
    rejects(forcing, "oversized forcing file refused");
    { std::ofstream file(tokens); file << "0, 1"; }
    rejects(forcing, "too many forcing tokens refused");
    for (const char* invalid : {"4", "4294967296", "-1", "1x", "+1"}) {
        { std::ofstream file(tokens); file << invalid; }
        rejects(forcing, "invalid forcing token refused");
    }
    { std::ofstream file(tokens); file << "0, \t\n"; }
    {
        BitcalcMeasurement measured;
        measured.prepare(4, 1, 2);
        const auto hook = measured.hook();
        expect(hook(0, 0, 1, values) == 0U && hook(1, 1, 1, values) == 1U && measured.forced_steps() == 1U,
            "forcing token zero resumes greedy when exhausted");
        expect(measured.force_tokens_sha256() == strix::model_content_identity_hex(strix::sha256_bytes(
            std::as_bytes(std::span<const char>("0, \t\n", 5U)))), "forcing SHA-256 covers exact bytes");
        measured.finish();
    }
    std::filesystem::remove(tokens);
    std::filesystem::remove(root);
    if (!failures) std::cout << "bitcalc measurement tests passed\n";
    return failures ? 1 : 0;
}
