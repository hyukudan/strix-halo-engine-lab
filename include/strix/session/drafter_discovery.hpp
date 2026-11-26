#pragma once
// Shared discovery of a drafter asset beside a model (MTP heads, draft sidecars). An engine hands over its
// registry of qualified drafters as data (name, shipped relative path, logical file name, exact byte size);
// this file finds the best one that exists next to the model and words the single line that says which file
// was taken and how. The CLI and the HTTP server both go through it, so they cannot disagree.
//
// Order of the three ways a drafter is found (the caller applies the first two, this file does the third):
//   1. an explicit `--mtp-bundle FILE`;
//   2. `MODEL.gguf.mtp-bundle.json` beside the model;
//   3. a registry drafter whose sidecar exists under the model directory at its source path
//      (for example `MTP/NAME.gguf`), by its logical name in the engine's drafter directory (`MTP/LOGICAL`), or by
//      its logical name beside the model, with the registry's exact size.
// A candidate named `preferred` (the recommended one) is tried first; the rest follow in registry order.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace strix::session {

struct DrafterCandidate {
    std::string_view name;          // registry name of the drafter
    std::string_view source_path;   // path of the file inside the model directory as shipped ("MTP/x.gguf")
    std::string_view logical_name;  // file name as it may sit beside the model
    std::uint64_t bytes = 0;        // exact size of the file; 0 accepts any non-empty file
    std::string_view drafter_dir{}; // directory beside the model where the engine's drafters ship ("MTP"); empty: none
};

struct DrafterDiscovery {
    std::filesystem::path path;     // the sidecar that was taken
    std::string name;               // registry name
    std::string how;                // "registry drafter 'base' at MTP/x.gguf" / "... by logical name beside the model"
    std::uint64_t bytes = 0;
    std::size_t index = 0;          // position in the candidate range
};

namespace detail {
inline std::optional<std::uint64_t> regular_file_size(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) return std::nullopt;
    const auto size = std::filesystem::file_size(path, error);
    if (error) return std::nullopt;
    return static_cast<std::uint64_t>(size);
}
} // namespace detail

// Finds the preferred drafter that exists with the registry size. `rejected` (optional) collects files that
// exist under a registry name but have another size, so a fallback notice can say why a file was not taken.
template <class Range>
std::optional<DrafterDiscovery> discover_drafter_sidecar(const std::filesystem::path& model, const Range& candidates,
    std::string_view preferred, std::vector<std::string>* rejected = nullptr) {
    const auto directory = model.has_parent_path() ? model.parent_path() : std::filesystem::path(".");
    std::vector<std::size_t> order;
    std::size_t index = 0;
    for (const auto& candidate : candidates) {
        if (candidate.name == preferred) order.insert(order.begin(), index);
        else order.push_back(index);
        ++index;
    }
    for (const std::size_t wanted : order) {
        std::size_t at = 0;
        for (const auto& candidate : candidates) {
            if (at++ != wanted) continue;
            const std::string in_dir = candidate.drafter_dir.empty() || candidate.logical_name.empty() ? std::string()
                : (std::filesystem::path(std::string(candidate.drafter_dir)) / std::string(candidate.logical_name)).generic_string();
            const std::string_view places[] = {candidate.source_path, in_dir, candidate.logical_name};
            for (std::size_t place = 0; place < 3U; ++place) {
                if (places[place].empty() || (place == 1U && places[1] == places[0]) ||
                    (place == 2U && (places[2] == places[0] || places[2] == places[1])))
                    continue;
                auto path = directory / std::filesystem::path(std::string(places[place]));
                path.make_preferred();
                const auto size = detail::regular_file_size(path);
                if (!size) continue;
                if (candidate.bytes ? *size != candidate.bytes : *size == 0U) {
                    if (rejected)
                        rejected->push_back(path.string() + " is " + std::to_string(*size) + " bytes, registry '" +
                            std::string(candidate.name) + "' expects " + std::to_string(candidate.bytes));
                    continue;
                }
                DrafterDiscovery found;
                found.path = path;
                found.name = std::string(candidate.name);
                found.how = "registry drafter '" + std::string(candidate.name) + "' " +
                    (place == 0U ? "at " + std::string(places[0])
                     : place == 1U ? "by logical name in " + std::string(candidate.drafter_dir) + "/"
                                   : "by logical name beside the model");
                found.bytes = *size;
                found.index = wanted;
                return found;
            }
        }
    }
    return std::nullopt;
}

// The one line an engine prints naming the drafter it took and how it was found.
inline std::string drafter_discovery_line(const std::filesystem::path& file, std::string_view how) {
    std::string line = "mtp: drafter ";
    line += file.string();
    line += " (";
    line += how;
    line += ")";
    return line;
}

// A process prints each distinct drafter line once (a server prepares and then launches from the same arguments).
inline bool announce_drafter_once(const std::string& line) {
    static std::mutex lock;
    static std::set<std::string> printed;
    const std::lock_guard<std::mutex> guard(lock);
    // One line per drafter file whatever the route words (a retry or a second consumer resolves the same file). The
    // route words are the last parenthesis: a path may hold " (" itself (Program Files (x86)).
    const auto open = line.rfind(" (");
    return printed.insert(line.substr(0, open)).second;
}

} // namespace strix::session
