// Host audit of the Gemma EXL3 core identity: one version per pack arm, the same for the
// command line (streams 1) and the server (streams = slots), at every concurrency.
// No model or accelerator is used.
#include "gemma4/exl3/format.hpp"

#include <iostream>
#include <string_view>
#include <type_traits>

namespace ex = strix::gemma4::exl3;

// Structural witness: the selector has no concurrency input, so slots, streams and
// cohort width cannot change the arithmetic of a row.
static_assert(std::is_invocable_v<decltype(ex::auto_version), std::string_view>);
static_assert(!std::is_invocable_v<decltype(ex::auto_version), std::string_view, unsigned>);
static_assert(!std::is_invocable_v<decltype(ex::configure_identity), std::string_view, unsigned>);

int main() {
    unsigned failures = 0;
    struct Row { const char* arm; unsigned version; };
    // M5 is v1: v3 was measured faster only at 8 streams and slower at 4.
    for (const Row row : {Row{"M4", 3U}, Row{"M5", 1U}, Row{"M6", 3U}, Row{"U5", 1U}, Row{"", 1U}}) {
        for (const unsigned streams : {1U, 2U, 8U, 16U}) {
            // CLI resolves with streams=1, the server with streams=serve_slots(); both call
            // configure_identity(pack_arm) which applies auto_version(arm).
            if (ex::auto_version(row.arm) != row.version) {
                ++failures;
                std::cerr << "DIVERGENT gemma " << row.arm << ": streams=" << streams << " selects v"
                          << ex::auto_version(row.arm) << " instead of v" << row.version << '\n';
            }
        }
    }
    if (ex::forced_version() == 0) {
        // Two entry points in one process resolve to the same fixed identity.
        const unsigned cli = ex::configure_identity("M5"), server = ex::configure_identity("M5");
        if (cli != 1U || server != cli || ex::core_version() != 1U) {
            ++failures;
            std::cerr << "M5 entry points resolve v" << cli << " / v" << server << '\n';
        }
    }
    std::cout << "serve route invariance (gemma): " << failures << " unresolved selectors\n";
    return failures ? 1 : 0;
}
