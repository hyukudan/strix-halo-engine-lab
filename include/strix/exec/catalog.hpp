#pragma once

#include "strix/device/profiles.hpp"
#include "strix/graph/stage_graph.hpp"
#include "strix/lever.hpp"
#include "strix/weight/types.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string_view>

namespace strix::device {

struct Capabilities;

enum class ArchMask : std::uint8_t {
    none = 0,
    sm120 = 1U << 0U,
    gfx1151 = 1U << 1U,
};

constexpr ArchMask operator|(ArchMask lhs, ArchMask rhs) noexcept {
    return static_cast<ArchMask>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

constexpr ArchMask operator&(ArchMask lhs, ArchMask rhs) noexcept {
    return static_cast<ArchMask>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

[[nodiscard]] ArchMask arch_mask(std::string_view arch) noexcept;

} // namespace strix::device

namespace strix::exec {

using FormId = std::string_view;
using ChainId = std::uint32_t;

enum class Backend : std::uint8_t { cpu, hip, cuda, sycl };
enum class OperandClass : std::uint8_t { f32_fma, f16_mma, bf16_mma, a8_imma, a4_imma, fp8_mma, fp4_mma, f16_hilo };
enum class RowsClass : std::uint8_t { r1, r2_8, r9_32, tall };
enum class Math : std::uint8_t { native, ieee_cr, gfx1151_v1 };
enum class Body : std::uint8_t { generated, hand_asm, external };
enum class Pack : std::uint8_t { none, wave, sum, pipe };
enum class ReferenceClass : std::uint8_t { base, host_bitwise };

struct KernelKey {
    graph::Op op;
    FormId form;
    weight::FormatId weight; // Empty name and revision zero for weightless forms.
    OperandClass act;
    RowsClass rows;
    std::uint16_t dim_class = 0; // Runtime dimensions.
    ChainId prologue = 0;
    ChainId epilogue = 0;
    Math math = Math::native;

    [[nodiscard]] bool operator==(const KernelKey& other) const noexcept;
};

struct EntryCost {
    double us;
    std::uint64_t weight_bytes, state_bytes, macs;
    std::uint32_t launches;
};

struct WorkspaceNeed {
    std::uint64_t bytes = 0;
    std::uint32_t alignment = 1;
};

// Counts describe the concrete site and chosen legacy schedule, including any
// split-K scratch and extra launches. backend_args are interpreted by the resolver.
struct LaunchSite {
    std::uint64_t weight_bytes = 0, state_bytes = 0, macs = 0;
    std::uint32_t launches = 1;
    WorkspaceNeed scratch;
    std::uint32_t rows = 1;
    const void* backend_args = nullptr;
};

struct LauncherRef {
    std::string_view symbol, source, mode;
    std::uint32_t max_rows; // Can be smaller than the top of a rows class.
};

struct EntryLaunch {
    using Enqueue = int (*)(void* binding, void* stream);
    Enqueue enqueue = nullptr;
    void* binding = nullptr;

    explicit operator bool() const noexcept { return enqueue != nullptr; }
};

// The backend resolves the typed call and arena addresses at bind time. The
// returned binding borrows arena storage through its last enqueue. An empty
// result means the backend cannot bind this site.
struct EntryBinder {
    using Resolve = EntryLaunch (*)(void* context, const LauncherRef&, const LaunchSite&);
    Resolve resolve = nullptr;
    void* context = nullptr;
};

struct ExternalRef {
    std::string_view library, version;
    std::uint64_t binary_hash = 0;
};

struct KernelEntry {
    KernelKey key;
    device::ArchMask arch;
    std::string_view identity;
    lever::Exactness exact;
    EntryCost (*cost)(const LaunchSite&, const device::Profile&);
    void (*workspace)(const LaunchSite&, WorkspaceNeed&);
    EntryLaunch (*bind)(const LaunchSite&, EntryBinder&);
    const LauncherRef* launcher;
    ReferenceClass reference = ReferenceClass::base;
    Body body = Body::generated;
    Pack pack = Pack::none;
    bool capturable = true;
    bool grid_by_context = false;
    const KernelEntry* null_variant = nullptr;
    const KernelEntry* guard_exact = nullptr;
    ExternalRef external{};
};

class Catalog {
public:
    // Borrows static rows and key strings. Each backend key is unique.
    // Duplicate keys or incomplete entries reject the entire addition.
    void add(std::span<const KernelEntry> entries);
    [[nodiscard]] std::span<const KernelEntry* const> find(const KernelKey&, device::ArchMask) const;
    // The present runtime capability type describes CUDA compute capability.
    // HIP callers pass arch_mask(runtime.arch) through the mask overload.
    [[nodiscard]] std::span<const KernelEntry* const> find(const KernelKey&, const device::Capabilities&) const;
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] static std::span<const KernelEntry> builtin_entries(Backend) noexcept;
    [[nodiscard]] static const Catalog& builtin(Backend);

private:
    struct Less {
        bool operator()(const KernelKey&, const KernelKey&) const noexcept;
    };
    std::map<KernelKey, const KernelEntry*, Less> entries_;
};

} // namespace strix::exec
