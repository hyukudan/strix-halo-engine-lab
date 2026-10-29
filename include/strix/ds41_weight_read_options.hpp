#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace strix::ds41 {

struct WeightReadOptions {
    std::string weight_dirs;
    unsigned drive_readers = 4;
    std::string weight_verify = "cached";
    std::string primary_sha256;

    void set_drive_readers(const std::string& value) {
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("--drive-readers must be an integer in 1..32");
        const auto digits = value.find_first_not_of('0');
        if (digits == std::string::npos || value.size() - digits > 2)
            throw std::invalid_argument("--drive-readers must be an integer in 1..32");
        const auto n = std::stoul(value.substr(digits));
        if (!n || n > 32)
            throw std::invalid_argument("--drive-readers must be an integer in 1..32");
        drive_readers = static_cast<unsigned>(n);
    }

    void environment(bool dirs_given, bool verify_given, const std::filesystem::path& primary = {}) {
        if (!dirs_given)
            if (const char* v = std::getenv("STRIX_WEIGHT_DIRS")) weight_dirs = v;
        if (!verify_given) {
            weight_verify = "cached";
            if (has_secondary(primary))
                if (const char* v = std::getenv("STRIX_WEIGHT_VERIFY")) weight_verify = v;
        }
        if (weight_verify != "cached" && weight_verify != "always")
            throw std::invalid_argument("--weight-verify must be cached or always");
        if (drive_readers == 0 || drive_readers > 32)
            throw std::invalid_argument("--drive-readers must be an integer in 1..32");
    }

    // Resolve primary-only lists before opening a session, including spelling/case aliases on Windows.
    [[nodiscard]] bool has_secondary(const std::filesystem::path& primary) const {
        if (weight_dirs.empty()) return false;
        const auto warn = [] {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
                std::fprintf(stderr, "weight streaming: an inaccessible secondary directory was ignored\n");
        };
        const auto canonical = [&](const std::filesystem::path& p) {
            std::error_code ec;
            const auto absolute = std::filesystem::absolute(p, ec);
            if (ec) { warn(); return std::string(); }
            const auto resolved = std::filesystem::weakly_canonical(absolute, ec);
            if (ec) { warn(); return std::string(); }
            auto s = resolved.generic_string();
#ifdef _WIN32
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
            while (s.size() > 1 && s.back() == '/') s.pop_back();
            return s;
        };
        const auto parent = canonical(primary.parent_path().empty() ? std::filesystem::path(".") : primary.parent_path());
        if (parent.empty()) return false;
        for (std::size_t begin = 0; begin < weight_dirs.size();) {
            const auto end = weight_dirs.find(';', begin);
            const auto part = weight_dirs.substr(begin, end == std::string::npos ? end : end - begin);
            if (!part.empty()) {
                const auto dir = std::filesystem::u8path(part);
                std::error_code ec;
                const bool accessible = std::filesystem::is_directory(dir, ec);
                if (ec) warn();
                const auto primary_dir = primary.parent_path().empty() ? std::filesystem::path(".") : primary.parent_path();
                if (accessible) {
                    const auto resolved = canonical(dir);
                    if (!resolved.empty() && !std::filesystem::equivalent(dir, primary_dir, ec) && resolved != parent) return true;
                }
            }
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        return false;
    }

    void require_supported(const std::filesystem::path& primary) const {
#ifndef STRIX_HAVE_TIERSTREAM_MFILE
        if (has_secondary(primary))
            throw std::invalid_argument("--weight-dirs secondary directories require a tierstream build with SS_MFILE");
#else
        (void)primary;
#endif
    }
};

}  // namespace strix::ds41
