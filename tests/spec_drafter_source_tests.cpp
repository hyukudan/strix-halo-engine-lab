// Sidecar drafter resolution (strix/spec/drafter_source.hpp): explicit file > off > beside the checkpoint under
// speculation; nothing is looked up without speculation; the receipt names the source and the file.
#include "strix/spec/drafter_source.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool value, const char* label) {
    if (!value) {
        std::fprintf(stderr, "FAIL %s\n", label);
        ++failures;
    }
}

}  // namespace

int main() {
    using namespace strix::spec;
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "strix-spec-drafter-source-test";
    std::error_code error;
    fs::create_directories(dir, error);
    const auto checkpoint = dir / "model.safetensors";
    const auto beside = dir / "mtp.safetensors";
    fs::remove(beside, error);

    // Not speculating: nothing is looked up, even when the file exists later.
    auto none = resolve_drafter_file({}, checkpoint, "mtp.safetensors");
    check(!none.enabled() && none.source == DrafterSource::not_requested, "no speculation: no drafter");
    // Speculating without the file: absent.
    auto absent = resolve_drafter_file({.speculating = true}, checkpoint, "mtp.safetensors");
    check(!absent.enabled() && absent.source == DrafterSource::absent, "speculation, no sidecar: absent");
    check(drafter_receipt("resident", absent) == "strix-speculation-drafter/v1 route=resident source=absent file=-",
          "absent receipt");
    { std::ofstream(beside) << "x"; }
    auto found = resolve_drafter_file({.speculating = true}, checkpoint, "mtp.safetensors");
    check(found.enabled() && found.source == DrafterSource::beside_model && fs::path(found.path) == beside,
          "speculation: the sidecar beside the checkpoint");
    check(drafter_receipt("resident", found) ==
              "strix-speculation-drafter/v1 route=resident source=beside-model file=mtp.safetensors",
          "beside-model receipt");
    check(!resolve_drafter_file({.speculating = true}, checkpoint, "").enabled(), "a model without a sidecar name");
    auto off = resolve_drafter_file({.off = true, .speculating = true}, checkpoint, "mtp.safetensors");
    check(!off.enabled() && off.source == DrafterSource::off, "off wins over the sidecar");
    auto given = resolve_drafter_file({.file = "/x/draft.safetensors"}, checkpoint, "mtp.safetensors");
    check(given.path == "/x/draft.safetensors" && given.source == DrafterSource::explicit_file,
          "an explicit file wins and needs no speculation flag");
    bool refused = false;
    try {
        (void)resolve_drafter_file({.file = "/x/d", .off = true}, checkpoint, "mtp.safetensors");
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "file and off together refused");
    // A sidecar beside the model that its loader rejects: context drafts only, the reason in the receipt.
    const auto rejected = decline_drafter(found, "schema mismatch");
    check(!rejected.enabled() && rejected.source == DrafterSource::rejected, "a rejected sidecar: no drafter");
    check(drafter_receipt("resident", rejected) ==
              "strix-speculation-drafter/v1 route=resident source=rejected file=- reason=schema mismatch",
          "receipt: rejected with its reason");
    bool explicit_kept = false;
    try {
        (void)decline_drafter(given, "x");
    } catch (const std::logic_error&) {
        explicit_kept = true;
    }
    check(explicit_kept, "an explicit file is never declined");
    check(drafter_receipt("served", {{}, DrafterSource::context, {}}) ==
              "strix-speculation-drafter/v1 route=served source=context file=-",
          "receipt: a context-only route");
    fs::remove(beside, error);
    fs::remove(dir, error);
    if (failures != 0) {
        std::fprintf(stderr, "spec-drafter-source: %d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("spec-drafter-source: PASS\n");
    return EXIT_SUCCESS;
}
