#include "cuda/core/mix_catalog.hpp"
#include "strix/exec/catalog.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace strix::exec;
using strix::device::ArchMask;
using strix::graph::Op;

struct Declaration {
    std::string op, form, weight, act, identity, symbol, source, mode, cost, workspace;
    std::uint32_t revision = 0, max_rows = 0;
    bool capturable = false;
};

std::istream& operator>>(std::istream& input, Declaration& expected) {
    return input >> std::quoted(expected.op) >> std::quoted(expected.form) >> std::quoted(expected.weight)
                 >> expected.revision >> std::quoted(expected.act) >> expected.max_rows
                 >> std::quoted(expected.identity) >> std::quoted(expected.symbol) >> std::quoted(expected.source)
                 >> std::quoted(expected.mode) >> expected.capturable >> std::quoted(expected.cost)
                 >> std::quoted(expected.workspace);
}

void require(bool condition, const Declaration& expected, const char* message) {
    if (!condition) throw std::runtime_error(expected.identity + ": " + message);
}

template <class Value, std::size_t Count>
Value parse_enum(std::string_view name, const std::array<std::pair<const char*, Value>, Count>& values) {
    for (const auto& [spelling, value] : values) {
        if (name == spelling) return value;
    }
    throw std::runtime_error("unknown fixture value: " + std::string(name));
}

constexpr std::array ops{
    std::pair{"mix", Op::mix}, std::pair{"select", Op::select}, std::pair{"gather", Op::gather},
    std::pair{"store", Op::store}, std::pair{"attend", Op::attend}, std::pair{"norm", Op::norm},
    std::pair{"elementwise", Op::elementwise}, std::pair{"combine", Op::combine},
    std::pair{"linear", Op::linear}, std::pair{"argmax", Op::argmax},
};

constexpr std::array operands{
    std::pair{"f32_fma", OperandClass::f32_fma}, std::pair{"bf16_mma", OperandClass::bf16_mma},
    std::pair{"fp8_mma", OperandClass::fp8_mma}, std::pair{"i8_dp4a", OperandClass::i8_dp4a},
};

struct BindingProbe {
    const LauncherRef* launcher = nullptr;
    const LaunchSite* site = nullptr;
    unsigned calls = 0;
};

int enqueue_stub(void*, void*) { return 0; }

EntryLaunch resolve(void* context, const LauncherRef& launcher, const LaunchSite& site) {
    auto& probe = *static_cast<BindingProbe*>(context);
    probe.launcher = &launcher;
    probe.site = &site;
    ++probe.calls;
    return {enqueue_stub, context};
}

void check_site(const KernelEntry& entry, const Declaration& expected, std::uint32_t first, std::uint32_t last) {
    BindingProbe probe;
    EntryBinder binder{resolve, &probe};
    LaunchSite site{.weight_bytes = 16U, .state_bytes = 32U, .macs = 1000000U, .launches = 2U,
                    .scratch = {128U, 16U}, .backend_args = &probe};
    for (auto rows : {first, last}) {
        site.rows = rows;
        const auto bound = entry.bind(site, binder);
        require(bound.enqueue == enqueue_stub && bound.binding == &probe && probe.launcher == entry.launcher &&
                probe.site == &site && probe.site->backend_args == &probe, expected, "binder changed metadata or site");
    }
    site.rows = first - 1U;
    require(!entry.bind(site, binder), expected, "row below class reached resolver");
    if (last < std::numeric_limits<std::uint32_t>::max()) {
        site.rows = last + 1U;
        require(!entry.bind(site, binder), expected, "row above class or limit reached resolver");
    }
    site.rows = first;
    binder.resolve = nullptr;
    require(!entry.bind(site, binder) && probe.calls == 2U, expected, "resolver guard changed");

    WorkspaceNeed need{};
    entry.workspace(site, need);
    const auto scratch = expected.workspace == "none" ? WorkspaceNeed{} : site.scratch;
    require(need.bytes == scratch.bytes && need.alignment == scratch.alignment, expected, "workspace changed");

    strix::device::Profile profile{};
    profile.memory_gbytes_ps = 1;
    profile.peaks = {.fp32 = 1, .bf16 = 4, .fp8 = 8, .int8 = 16};
    const auto cost = entry.cost(site, profile);
    const bool unpriced = expected.cost == "unpriced_single_launch";
    require(cost.weight_bytes == site.weight_bytes && cost.state_bytes == site.state_bytes &&
            cost.macs == site.macs && cost.launches == (unpriced ? 1U : site.launches), expected, "cost fields changed");
    if (unpriced || entry.key.act == OperandClass::i8_dp4a) {
        require(std::isinf(cost.us), expected, "unpriced compute received a finite cost");
    } else {
        const double expected_us = entry.key.act == OperandClass::fp8_mma ? 0.25 :
                                   entry.key.act == OperandClass::bf16_mma ? 0.5 : 2.0;
        require(cost.us == expected_us, expected, "roofline used the wrong operand peak");
    }
}

std::size_t check_declaration(const Declaration& expected) {
    const auto op = parse_enum(expected.op, ops);
    const auto act = parse_enum(expected.act, operands);
    const auto& catalog = Catalog::builtin(Backend::cuda);
    const auto mix_rows = strix::cuda::mix::catalog_entries();
    const std::array first{1U, 2U, 9U, 33U};
    const std::array last{1U, 8U, 32U, std::numeric_limits<std::uint32_t>::max()};
    std::size_t count = 0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        const KernelKey key{op, expected.form, {expected.weight, expected.revision}, act, static_cast<RowsClass>(i)};
        const auto found = catalog.find(key, ArchMask::sm120);
        if (first[i] > expected.max_rows) {
            require(found.empty(), expected, "unexpected rows class");
            continue;
        }
        require(found.size() == 1U && catalog.find(key, ArchMask::gfx1151).empty(), expected, "wrong key or arch");
        const auto& entry = *found.front();
        require(entry.identity == expected.identity && entry.capturable == expected.capturable &&
                entry.exact == strix::lever::Exactness::bitwise && entry.reference == ReferenceClass::base,
                expected, "entry identity or flags changed");
        const auto& launcher = *entry.launcher;
        require(launcher.symbol == expected.symbol && launcher.source == expected.source &&
                launcher.mode == expected.mode && launcher.max_rows == expected.max_rows,
                expected, "launcher declaration changed");
        if (op == Op::mix && expected.form == "residual.bank") {
            require(mix_rows.size() == first.size() && &mix_rows[i] == &entry, expected, "K8c view copied rows");
        }
        check_site(entry, expected, first[i], std::min(last[i], expected.max_rows));
        ++count;
    }
    return count;
}

} // namespace

int main() {
    try {
        std::size_t keys = 0;
        std::size_t declarations = 0;
        while (std::cin >> std::ws && std::cin.peek() != std::char_traits<char>::eof()) {
            Declaration expected;
            if (!(std::cin >> expected)) throw std::runtime_error("invalid fixture declaration");
            keys += check_declaration(expected);
            ++declarations;
        }
        if (declarations != 58U || keys != 97U) throw std::runtime_error("fixture did not cover all fragment keys");
        std::cout << "PASS D4 fragments: " << keys << " CUDA keys match launcher metadata and host resolver contracts\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL D4 fragments: " << error.what() << '\n';
        return 1;
    }
}
