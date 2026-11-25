#pragma once

// Where a model's sidecar drafter (an MTP block, a draft head) comes from, for every entry point (command line and
// server) and every backend. An explicit file wins, an explicit "off" refuses any drafter, and otherwise a drafter is
// looked for beside the checkpoint only when speculation was asked for (auto): the file name is the model's data (its
// TOML / registry entry), never a backend's guess. The receipt line names the source, so a run that drafted and one
// that did not can never be confused.

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace strix::spec {

enum class DrafterSource { off, explicit_file, beside_model, absent, not_requested };

struct DrafterRequest {
    std::string_view file{};   // an explicit drafter file (empty: none given)
    bool off = false;          // the drafter was switched off explicitly
    bool speculating = false;  // speculation is on (auto): a drafter beside the model is used when present
};

struct ResolvedDrafter {
    std::string path;  // empty: no drafter
    DrafterSource source = DrafterSource::not_requested;
    [[nodiscard]] bool enabled() const noexcept { return !path.empty(); }
};

[[nodiscard]] inline std::string_view drafter_source_name(DrafterSource source) noexcept {
    switch (source) {
    case DrafterSource::off: return "off";
    case DrafterSource::explicit_file: return "explicit";
    case DrafterSource::beside_model: return "beside-model";
    case DrafterSource::absent: return "absent";
    case DrafterSource::not_requested: return "not-requested";
    }
    return "unknown";
}

// checkpoint: the model's checkpoint file (its directory holds the sidecar); file_name: the model's sidecar name.
// An explicit file is returned as given (its loader reports a missing file); "off" with a file is refused.
[[nodiscard]] inline ResolvedDrafter resolve_drafter_file(const DrafterRequest& request,
                                                          const std::filesystem::path& checkpoint,
                                                          std::string_view file_name) {
    if (request.off && !request.file.empty())
        throw std::invalid_argument("a drafter file and drafter off are exclusive");
    if (request.off) return {{}, DrafterSource::off};
    if (!request.file.empty()) return {std::string(request.file), DrafterSource::explicit_file};
    if (!request.speculating || file_name.empty()) return {{}, DrafterSource::not_requested};
    std::error_code error;
    const auto beside = checkpoint.parent_path() / std::filesystem::path(file_name);
    if (std::filesystem::is_regular_file(beside, error)) return {beside.string(), DrafterSource::beside_model};
    return {{}, DrafterSource::absent};
}

// "strix-speculation-drafter/v1 route=<route> source=<source> file=<name>" (file=- without a drafter).
[[nodiscard]] inline std::string drafter_receipt(std::string_view route, const ResolvedDrafter& drafter) {
    std::string line = "strix-speculation-drafter/v1 route=";
    line += route;
    line += " source=";
    line += drafter_source_name(drafter.source);
    line += " file=";
    line += drafter.enabled() ? std::filesystem::path(drafter.path).filename().string() : std::string("-");
    return line;
}

}  // namespace strix::spec
