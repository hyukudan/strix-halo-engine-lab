#pragma once

#include <filesystem>
#include <random>
#include <stdexcept>

// Create the fixture area exclusively under CTest's working directory.
class VisionTestDir {
public:
    VisionTestDir() {
        std::random_device random;
        for (unsigned attempt = 0; attempt != 32; ++attempt) {
            auto candidate = std::filesystem::current_path() / ("inkling2-vision-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) {
                path = std::move(candidate);
                return;
            }
        }
        throw std::runtime_error("cannot create vision test directory");
    }
    ~VisionTestDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    VisionTestDir(const VisionTestDir&) = delete;
    VisionTestDir& operator=(const VisionTestDir&) = delete;
    std::filesystem::path path;
};
