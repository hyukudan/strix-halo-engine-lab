#include "strix/serve/serial_generation_cli.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/stat.h>
#endif

namespace strix::serve {

#ifdef _WIN32
namespace {
std::optional<FileKey> key_of(HANDLE handle) {
    BY_HANDLE_FILE_INFORMATION info{};
    if (!GetFileInformationByHandle(handle, &info)) return std::nullopt;
    return FileKey{info.dwVolumeSerialNumber,
                   (std::uint64_t(info.nFileIndexHigh) << 32U) | std::uint64_t(info.nFileIndexLow)};
}
}  // namespace
#endif

std::optional<FileKey> file_key(const std::filesystem::path& path) {
#ifdef _WIN32
    const HANDLE handle = CreateFileW(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return std::nullopt;
    const auto key = key_of(handle);
    CloseHandle(handle);
    return key;
#else
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return std::nullopt;
    return FileKey{std::uint64_t(st.st_dev), std::uint64_t(st.st_ino)};
#endif
}

std::optional<FileKey> descriptor_file_key(int descriptor) {
#ifdef _WIN32
    const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(descriptor));
    if (handle == INVALID_HANDLE_VALUE) return std::nullopt;
    return key_of(handle);
#else
    struct stat st{};
    if (::fstat(descriptor, &st) != 0) return std::nullopt;
    return FileKey{std::uint64_t(st.st_dev), std::uint64_t(st.st_ino)};
#endif
}

}  // namespace strix::serve
