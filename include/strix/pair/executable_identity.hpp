#pragma once
// Digest of the running executable, for the pair hello (host only: no device runtime).
#include "strix/model_content_hash.hpp"
#include "strix/pair/handshake.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strix::pair {

[[nodiscard]] inline const ModelContentIdentity& executable_identity() {
    static const auto sha = [] {
#ifdef _WIN32
        std::wstring path(32768U, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (n == 0 || n >= path.size()) throw ProtocolError("pair hello: cannot locate the running executable");
        path.resize(n);
        return sha256_file_content(path, FileHashPolicy::full);
#else
        return sha256_file_content("/proc/self/exe", FileHashPolicy::full);
#endif
    }();
    return sha;
}

}  // namespace strix::pair
