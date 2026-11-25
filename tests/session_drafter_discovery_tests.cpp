// Order and size rules of the shared drafter discovery (include/strix/session/drafter_discovery.hpp).
#include "strix/session/drafter_discovery.hpp"
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace strix::session;
namespace {
void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
void make(const std::filesystem::path& path, std::uint64_t bytes) {
    std::filesystem::create_directories(path.parent_path());
    { std::ofstream file(path, std::ios::binary); file << 'x'; }
    std::filesystem::resize_file(path, bytes);
}
} // namespace

int main() try {
    const auto dir = std::filesystem::temp_directory_path() /
        ("strix-drafter-discovery-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    struct Cleanup { std::filesystem::path p; ~Cleanup() { std::error_code ec; std::filesystem::remove_all(p, ec); } } cleanup{dir};
    const std::array<DrafterCandidate, 3> registry{{
        {"base", "MTP/base.gguf", "base.gguf", 1000},
        {"best", "export/best.gguf", "best.gguf", 2000},
        {"other", "MTP/other.gguf", "other.gguf", 3000}}};
    const auto model = dir / "model.gguf";

    std::vector<std::string> rejected;
    require(!discover_drafter_sidecar(model, registry, "best", &rejected).has_value() && rejected.empty(), "nothing beside the model");

    make(dir / "MTP/base.gguf", 999);
    require(!discover_drafter_sidecar(model, registry, "best", &rejected).has_value() && rejected.size() == 1U &&
        rejected[0].find("is 999 bytes") != std::string::npos && rejected[0].find("'base' expects 1000") != std::string::npos,
        "wrong size is refused and named");

    make(dir / "MTP/base.gguf", 1000);
    auto found = discover_drafter_sidecar(model, registry, "best");
    require(found && found->name == "base" && found->path == dir / "MTP/base.gguf" && found->bytes == 1000U &&
        found->index == 0U && found->how == "registry drafter 'base' at MTP/base.gguf", "source path found");

    make(dir / "best.gguf", 2000);  // logical name beside the model, preferred over the earlier registry entry
    found = discover_drafter_sidecar(model, registry, "best");
    require(found && found->name == "best" && found->path == dir / "best.gguf" &&
        found->how == "registry drafter 'best' by logical name beside the model", "preferred candidate wins, logical name");
    found = discover_drafter_sidecar(model, registry, "");
    require(found && found->name == "base", "without a preferred name the registry order decides");

    make(dir / "export/best.gguf", 2000);  // the source path is tried before the logical name
    found = discover_drafter_sidecar(model, registry, "best");
    require(found && found->path == dir / "export/best.gguf", "source path before logical name");

    // The engine's drafter directory: the logical name inside MTP/ is found, after the source path and before the
    // logical name beside the model; the recommended candidate still comes first.
    {
        const auto mtp_dir = dir / "mtpdir";
        std::filesystem::create_directories(mtp_dir);
        const auto mtp_model = mtp_dir / "model.gguf";
        const std::array<DrafterCandidate, 2> shipped{{{"base", "MTP/base.gguf", "base.gguf", 1000, "MTP"},
                                                      {"best", "export/best.gguf", "best.gguf", 2000, "MTP"}}};
        make(mtp_dir / "MTP/base.gguf", 1000);
        make(mtp_dir / "MTP/best.gguf", 2000);
        auto in_mtp = discover_drafter_sidecar(mtp_model, shipped, "best");
        require(in_mtp && in_mtp->name == "best" && in_mtp->path == mtp_dir / "MTP/best.gguf" &&
            in_mtp->how == "registry drafter 'best' by logical name in MTP/", "recommended drafter found in MTP/ first");
        make(mtp_dir / "best.gguf", 2000);
        in_mtp = discover_drafter_sidecar(mtp_model, shipped, "best");
        require(in_mtp && in_mtp->path == mtp_dir / "MTP/best.gguf", "drafter directory before the logical name beside the model");
        make(mtp_dir / "export/best.gguf", 2000);
        in_mtp = discover_drafter_sidecar(mtp_model, shipped, "best");
        require(in_mtp && in_mtp->path == mtp_dir / "export/best.gguf", "source path before the drafter directory");
        in_mtp = discover_drafter_sidecar(mtp_model, shipped, "");
        require(in_mtp && in_mtp->name == "base" && in_mtp->how == "registry drafter 'base' at MTP/base.gguf",
            "without a preferred name the registry order decides (base at its source path)");
    }

    // A size-0 registry entry accepts any non-empty file only.
    const std::array<DrafterCandidate, 1> loose{{{"loose", "MTP/loose.gguf", "", 0}}};
    make(dir / "MTP/loose.gguf", 1);
    require(discover_drafter_sidecar(model, loose, "loose").has_value(), "size 0 accepts a non-empty file");
    require(!discover_drafter_sidecar(dir / "sub" / "model.gguf", loose, "loose").has_value(), "another model directory sees nothing");

    const auto line = drafter_discovery_line(found->path, found->how);
    require(line.rfind("mtp: drafter ", 0) == 0 && line.find(found->how) != std::string::npos && line.find('\n') == std::string::npos,
        "one line");
    require(announce_drafter_once(line) && !announce_drafter_once(line), "a line prints once");
    std::cout << "drafter discovery tests passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "drafter discovery tests failed: " << error.what() << '\n';
    return 1;
}
