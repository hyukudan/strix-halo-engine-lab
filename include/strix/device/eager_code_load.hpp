#pragma once

// Eager GPU code-object loading, ONE home for every HIP binary (host only, no HIP headers).
//
// With the HIP default (deferred loading) a code object is loaded at the first launch of one of its kernels. On a
// unified-memory APU that load allocates the code segment as a KFD userptr BO on host pages (seen with the kfdevict
// probe: 22 MB / 7 MB / 6.6 MB BOs made by the first MoE / dense / trellis launches inside the first timed prefill),
// so the first (cold) request pays the loads and the new BOs join the host memory that page migration can evict.
// HIP_ENABLE_DEFERRED_LOADING=0 loads every code object while the runtime registers them, i.e. during static
// initialisation, before main(): a value set in main() is too late for the running image, so the process restarts its
// own image once with the value in its environment (same pid, same arguments); the restarted image finds the value
// set and does not restart again. Kernel arithmetic does not change (bitwise): only when the code is loaded.
//
// Policy (STRIX_EAGER_CODE_LOAD): unset = the binary's default; "0" = deferred (HIP default); anything else = eager.
// An explicitly supplied HIP_ENABLE_DEFERRED_LOADING always wins, even empty, and is never overwritten.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include "strix/env_flags.hpp"
#include <vector>

#if defined(__unix__)
#include <unistd.h>
#endif

namespace strix::device {

// Returns true when it set HIP_ENABLE_DEFERRED_LOADING=0 (the caller restarts the image for it to apply).
template<class Get, class Put>
inline bool apply_eager_code_loading(bool enabled, Get get, Put put) {
    const char* policy = get("STRIX_EAGER_CODE_LOAD");
    if (!enabled || (policy && policy[0] == '0') || get("HIP_ENABLE_DEFERRED_LOADING")) return false;
    if (!put("HIP_ENABLE_DEFERRED_LOADING", "0"))
        throw std::runtime_error("cannot configure HIP eager code loading");
    return true;
}

// The restart must repeat the process's own arguments: a runner that wraps an engine entry forwards a reduced argv,
// and restarting with that would drop the runner's options.
// Splits /proc/self/cmdline (NUL-terminated arguments); empty when the buffer is not complete.
inline std::vector<std::string> split_process_arguments(std::string_view cmdline) {
    std::vector<std::string> arguments;
    if (cmdline.empty() || cmdline.back() != '\0') return arguments;
    for (std::size_t begin = 0; begin < cmdline.size();) {
        const auto end = cmdline.find('\0', begin);
        arguments.emplace_back(cmdline.substr(begin, end - begin));
        begin = end + 1U;
    }
    return arguments;
}

// Restarts the running image with its original arguments. Returns only when exec failed (message on stderr): the
// process then continues with deferred loading, which is correct, only slower on the first request.
inline void restart_own_image(char** argv) {
#if defined(__unix__)
    std::fflush(nullptr);
    std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
    const auto original = split_process_arguments(
        std::string((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>()));
    std::vector<char*> restart;
    for (const auto& argument : original) restart.push_back(const_cast<char*>(argument.c_str()));
    restart.push_back(nullptr);
    // Exec the resolved binary path, not "/proc/self/exe": the kernel names the new image after the exec'd path, so the
    // latter turns the process name (comm) into "exe" and name-based tools (pgrep, samplers, reapers) lose the engine.
    char self[4096];
    const auto n = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) self[n] = '\0';
    ::execv(n > 0 ? self : "/proc/self/exe", original.empty() ? argv : restart.data());
    std::fprintf(stderr, "eager code loading: restart failed (%s); continuing with deferred loading\n", std::strerror(errno));
#else
    (void)argv;
#endif
}

// Who decides for this binary. shared = the start gate applies the default below; engine = the binary resolves its own
// policy later (it has engine options that can decline eager loading) and calls apply_eager_code_loading itself.
enum class CodeLoad { shared, engine };

// The start gate's call: eager unless STRIX_EAGER_CODE_LOAD=0 or HIP_ENABLE_DEFERRED_LOADING is already set.
inline void eager_code_load_or_restart(char** argv, CodeLoad who = CodeLoad::shared) {
#if defined(__unix__)
    if (who != CodeLoad::shared) return;
    if (apply_eager_code_loading(true, [](const char* name) { return std::getenv(name); },
            [](const char* name, const char* value) { return ::setenv(name, value, 0) == 0; }))
        restart_own_image(argv);
#else
    (void)argv; (void)who;  // no HIP runtime image restart off Linux
#endif
}

// Receipt line of the resolved mode (the restarted image prints it): "code load: eager|deferred (<why>)".
inline std::string code_load_receipt(CodeLoad who = CodeLoad::shared) {
    const char* hip = std::getenv("HIP_ENABLE_DEFERRED_LOADING");
    const char* policy = strix::envcfg::text("STRIX_EAGER_CODE_LOAD");
    const bool eager = hip && std::string_view(hip) == "0";
    std::string why = policy && policy[0] == '0' ? "STRIX_EAGER_CODE_LOAD=0"
        : who == CodeLoad::engine ? "engine policy"
        : hip ? (eager ? "HIP_ENABLE_DEFERRED_LOADING=0" : "HIP_ENABLE_DEFERRED_LOADING set") : "restart unavailable";
    return std::string("code load: ") + (eager ? "eager" : "deferred") + " (" + why + ")";
}

} // namespace strix::device
