#pragma once
#include <cstdio>
#include <string>

namespace strix::xport {

// Caller holds the channel's failure mutex. Keep the original reason and emit it
// on the failing rank even if the owning thread is blocked in a device wait.
inline void report_first_failure(std::string& failure, const std::string& why,
                                 unsigned rank, std::FILE* output = stderr) {
    if (!failure.empty()) return;
    failure = why.empty() ? "unknown exchange failure" : why;
    std::fprintf(output, "xport: rank=%u exchange failed: %s\n", rank, failure.c_str());
    std::fflush(output);
}

}  // namespace strix::xport
