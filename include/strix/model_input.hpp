#pragma once
#include "strix/file_snapshot.hpp"
#include <fstream>
#ifdef __linux__
#include <ext/stdio_filebuf.h>
#include <fcntl.h>
#endif
namespace strix {
#ifdef __linux__
class ModelInput : public std::istream {
    __gnu_cxx::stdio_filebuf<char> buffer_;
public:
    explicit ModelInput(const std::filesystem::path& path, LoaderSnapshot* snapshot = nullptr)
        : std::istream(nullptr), buffer_(::open(path.c_str(), O_RDONLY | O_CLOEXEC), std::ios::in) {
        rdbuf(&buffer_);
        if (!buffer_.is_open()) { setstate(std::ios::failbit); return; }
        if (snapshot) snapshot->push_back(loaded_file(path, buffer_.fd()));
        LoaderCapture::record(path, buffer_.fd());
    }
};
#else
class ModelInput : public std::ifstream {
public:
    explicit ModelInput(const std::filesystem::path& path, LoaderSnapshot* = nullptr)
        : std::ifstream(path, std::ios::binary) {}
};
#endif
}
