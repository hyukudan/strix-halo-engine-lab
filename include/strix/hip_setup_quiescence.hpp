#pragma once

#include "strix/native_setup_quiescence.hpp"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace strix {

// Setup uses borrowed source/destination owners. On unknown queued-work outcome,
// returning through their destructors is unsafe: recover the original device or
// terminate the diagnostic process without normal owner cleanup.
class HipSetupQuiescenceGuard {
public:
    HipSetupQuiescenceGuard() : original_(observe()) {}

    [[nodiscard]] int device_index() const noexcept { return original_.device; }
    [[nodiscard]] const std::string& architecture() const noexcept { return original_.architecture; }

    // Fingerprints the observed tuple for metadata binding. This is neither a
    // context-reset generation nor an independent hardware qualification proof.
    [[nodiscard]] std::uint64_t observed_binding_identity() const noexcept {
        std::uint64_t hash = 14695981039346656037ULL;
        const auto byte = [&](unsigned char value) { hash = (hash ^ value) * 1099511628211ULL; };
        for (const char value : std::string_view("strix-hip-observed-setup-binding-v1")) byte(static_cast<unsigned char>(value));
        for (const int value : {original_.device, original_.runtime, original_.driver}) {
            const auto bits = static_cast<std::uint32_t>(value);
            for (unsigned shift = 0; shift < 32; shift += 8) byte(static_cast<unsigned char>(bits >> shift));
        }
        for (const auto value : original_.uuid) byte(value);
        for (const char value : original_.architecture) byte(static_cast<unsigned char>(value));
        return hash ? hash : 1;
    }

    template <class Work>
    void run(Work&& work) const {
        require_original(); // No work has been submitted by this call yet.
        try {
            std::forward<Work>(work)();
            check(hipDeviceSynchronize());
            require_original();
        } catch (...) {
            recover_or_fail_stop();
            throw;
        }
    }

private:
    struct Identity {
        int device = -1;
        int runtime = 0;
        int driver = 0;
        std::array<unsigned char, 16> uuid{};
        std::string architecture;
        bool operator==(const Identity&) const = default;
    };

    static void check(hipError_t result) {
        if (result != hipSuccess)
            throw std::runtime_error("HIP setup binding or completion check failed");
    }

    static Identity observe() {
        if (const char* override_arch = std::getenv("HSA_OVERRIDE_GFX_VERSION"); override_arch && *override_arch)
            throw std::runtime_error("HIP native setup rejects architecture override");
        Identity result;
        check(hipGetDevice(&result.device));
        check(hipRuntimeGetVersion(&result.runtime));
        check(hipDriverGetVersion(&result.driver));
        hipUUID uuid{};
        check(hipDeviceGetUuid(&uuid, result.device));
        static_assert(sizeof(uuid.bytes) == 16);
        std::memcpy(result.uuid.data(), uuid.bytes, result.uuid.size());
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, result.device));
        if (!std::memchr(properties.gcnArchName, '\0', sizeof(properties.gcnArchName)))
            throw std::runtime_error("HIP setup architecture is not bounded");
        result.architecture = properties.gcnArchName;
        if (result.device < 0 || result.runtime <= 0 || result.driver <= 0 ||
            result.architecture.empty() ||
            std::all_of(result.uuid.begin(), result.uuid.end(), [](auto b) { return b == 0; }))
            throw std::runtime_error("HIP setup requires an observed original device identity");
        return result;
    }

public:
    void require_original() const {
        if (observe() != original_)
            throw std::runtime_error("HIP setup original device binding changed");
    }

    // Error-path entry point for callers that own the enclosing execution
    // sequence. It never returns without the same original-device drain proof.
    void recover_or_fail_stop() const noexcept {
        const bool quiescent = prove_native_setup_quiescence(
            [this] { check(hipSetDevice(original_.device)); require_original(); },
            [] { check(hipDeviceSynchronize()); });
        if (!quiescent) {
            std::fputs("HIP setup completion is unproved; retaining owners until process termination\n", stderr);
            std::fflush(nullptr);
            std::_Exit(3);
        }
    }

private:
    Identity original_;
};

}  // namespace strix
