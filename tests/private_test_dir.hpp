#pragma once
// A private directory per test run under the test's working directory (ctest runs each test in the build tree, not
// in the system temporary directory): created exclusively (an existing path is never reused) with owner-only access
// set AT CREATION (POSIX mode 0700, then checked: owned by this user, no group/other bits; Windows: a protected DACL
// that grants the owner only), and the constructor throws when that cannot be established, so a test never runs in a
// directory other users can enter. Inside it, file names are handed out only while absent: no other user can plant a
// file or link between that check and the open. The directory and what the test wrote into it are removed at the end.
#include <chrono>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#ifdef _MSC_VER
#pragma comment(lib, "advapi32")
#endif
#else
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strix::test {

namespace detail {
// Creates `dir` owner-only. Returns false when it already exists; throws when it cannot be made private.
inline bool create_private_dir(const std::filesystem::path& dir) {
#ifdef _WIN32
    // Protected DACL (no inherited entries), full access for the owner (OWNER RIGHTS) and nobody else.
    PSECURITY_DESCRIPTOR sd = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;FA;;;OW)", SDDL_REVISION_1, &sd, nullptr))
        throw std::runtime_error("private test directory: no security descriptor");
    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), sd, FALSE};
    const BOOL made = CreateDirectoryW(dir.c_str(), &sa);
    const DWORD error = made ? ERROR_SUCCESS : GetLastError();
    LocalFree(sd);
    if (made) return true;
    if (error == ERROR_ALREADY_EXISTS) return false;
    throw std::runtime_error("private test directory: cannot create " + dir.string());
#else
    if (::mkdir(dir.c_str(), S_IRWXU) != 0) {
        if (errno == EEXIST) return false;
        throw std::runtime_error("private test directory: cannot create " + dir.string());
    }
    struct stat st{};
    if (::lstat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() ||
        (st.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        std::error_code ignored;
        std::filesystem::remove(dir, ignored);
        throw std::runtime_error("private test directory: owner-only access not established for " + dir.string());
    }
    return true;
#endif
}
}  // namespace detail

class PrivateTestDir {
public:
    explicit PrivateTestDir(const std::string& prefix) {
        namespace fs = std::filesystem;
        std::random_device entropy;
        for (int attempt = 0; attempt < 16; ++attempt) {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            const fs::path candidate = fs::current_path() / (prefix + "-" + std::to_string(stamp) + "-" + std::to_string(entropy()));
            if (detail::create_private_dir(candidate)) {  // false when the path exists
                path_ = candidate;
                return;
            }
        }
        throw std::runtime_error("could not create a private test directory under " + fs::current_path().string());
    }
    ~PrivateTestDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    PrivateTestDir(const PrivateTestDir&) = delete;
    PrivateTestDir& operator=(const PrivateTestDir&) = delete;

    // A file name in the directory that does not exist yet (not even as a link).
    [[nodiscard]] std::filesystem::path fresh(const std::string& name) const {
        const auto file = path_ / name;
        std::error_code error;
        if (std::filesystem::exists(std::filesystem::symlink_status(file, error)))
            throw std::runtime_error("test file already exists: " + file.string());
        return file;
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace strix::test
