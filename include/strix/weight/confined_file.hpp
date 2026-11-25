#pragma once

#include "strix/weight/source.hpp"

#include <filesystem>
#include <optional>

namespace strix::weight {

// One operator-selected directory. All subsequent opens use its held directory
// handle, consume one basename, and reject symlinks, reparse points and nonfiles.
class ConfinedDirectory {
public:
    explicit ConfinedDirectory(const std::filesystem::path& directory);
    ~ConfinedDirectory();
    ConfinedDirectory(const ConfinedDirectory&) = delete;
    ConfinedDirectory& operator=(const ConfinedDirectory&) = delete;

    [[nodiscard]] std::shared_ptr<const Source> open(const std::filesystem::path& name) const;
    [[nodiscard]] std::optional<std::string> read(const std::filesystem::path& name, std::size_t max_bytes,
                                                  bool missing_ok = false,
                                                  std::optional<std::size_t> exact_bytes = {}) const;
private:
    struct Impl;
    std::unique_ptr<Impl> directory_;
};

}  // namespace strix::weight
