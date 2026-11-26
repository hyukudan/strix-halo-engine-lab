#pragma once
#include <cstdio>
#include <string>

namespace strix::xport {

// Caller holds the failure mutex. The first reason is immutable thereafter.
inline bool remember_first_failure(std::string& failure, const std::string& why) {
    if (!failure.empty()) return false;
    failure = why.empty() ? "unknown exchange failure" : why;
    return true;
}

// Call only for the first failure, after publishing the failure and releasing
// waiters, outside the failure mutex. Diagnostic I/O can block.
inline void log_failure(const std::string& failure, unsigned rank,
                        std::FILE* output = stderr) {
    std::fprintf(output, "xport: rank=%u exchange failed: %s\n", rank, failure.c_str());
    std::fflush(output);
}

}  // namespace strix::xport
