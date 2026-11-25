#include "strix/profiles/gemma4.hpp"

#include <iostream>
#include <map>
#include <stdexcept>
#include <type_traits>

namespace pr = strix::gemma4::profile;
namespace common = strix::profile;
namespace {
unsigned checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}
pr::Lookup lookup(std::map<std::string, std::string> values = {}) {
    return [values = std::move(values)](const char* key) -> const char* {
        const auto it = values.find(key);
        return it == values.end() ? nullptr : it->second.c_str();
    };
}
void rows() {
    for (const auto& row : pr::levers()) {
        const auto table = std::span(&row, 1);
        for (const auto p : {pr::Name::standard, pr::Name::fast, pr::Name::quality, pr::Name::turbo}) {
            for (const auto app : {pr::App::engine, pr::App::decode_bench, pr::App::prefill_bench}) {
                const auto label = std::string(row.name) + "/" + pr::name(p) + "/" + pr::app_name(app);
                const bool applies = (row.apps & unsigned(app)) != 0;
                pr::Inputs inputs;
                inputs.env = lookup();
                inputs.cli = {{std::string(row.name), std::string(row.builtin)}};
                auto r = pr::resolve(p, app, inputs, table);
                check(bool(r.find(row.name)) == applies, label + " app mask");
                if (!applies) continue;
                const bool builtin_off = !row.off.empty() && (row.builtin == row.off || row.builtin == "off");
                check(r.entries[0].status == (builtin_off ? pr::Status::off : pr::Status::explicit_cli), label + " CLI builtin");
                if (!row.env.empty()) {
                    // CLI wins even over an environment OFF and discovered asset.
                    inputs.env = lookup({{std::string(row.env), std::string(row.off)}});
                    inputs.assets = {{std::string(row.name), "discovered"}};
                    inputs.cli = {{std::string(row.name), "first"}, {std::string(row.name), "cli"}};
                    r = pr::resolve(p, app, inputs, table);
                    check(r.entries[0].status == pr::Status::explicit_cli && r.entries[0].value == "cli",
                          label + " last CLI > env > discovery > profile");
                    // Empty env is explicit here, including the asset request policy.
                    inputs.cli.clear();
                    inputs.env = lookup({{std::string(row.env), ""}});
                    bool empty_declared = false, empty_runnable = false, empty_refused = false;
                    for (unsigned column = 0; column < 3; ++column)
                        if (row.value[column].empty()) { empty_declared = true; empty_runnable |= row.runnable[column]; }
                    try {
                        r = pr::resolve(p, app, inputs, table);
                        check(r.entries[0].status == pr::Status::explicit_env &&
                              r.entries[0].value == (row.asset ? "discovered" : ""), label + " empty environment");
                    } catch (const std::invalid_argument&) { empty_refused = true; }
                    check(empty_refused == (!row.asset && !row.builtin.empty() && empty_declared && !empty_runnable),
                          label + " empty declared unavailable column refusal");
                    inputs.env = lookup({{std::string(row.env), std::string(row.builtin)}});
                    r = pr::resolve(p, app, inputs, table);
                    check(r.entries[0].status == (builtin_off ? pr::Status::off : pr::Status::explicit_env),
                          label + " env > profile");
                }
                if (row.off.empty()) continue;
                for (const auto off : {row.off, std::string_view("off")}) {
                    inputs.cli.clear();
                    inputs.assets = {{std::string(row.name), "discovered"}};
                    inputs.env = lookup({{std::string(row.env), std::string(off)}});
                    r = pr::resolve(p, app, inputs, table);
                    check(r.entries[0].status == pr::Status::off && r.entries[0].value == "off", label + " env OFF");
                    inputs.cli = {{std::string(row.name), std::string(off)}};
                    inputs.env = lookup({{std::string(row.env), "on"}});
                    r = pr::resolve(p, app, inputs, table);
                    check(r.entries[0].status == pr::Status::off && !r.active(row.name), label + " CLI OFF");
                }
            }
        }
    }
}
void admission() {
    for (const bool ready : {false, true})
        for (const bool runnable : {false, true})
            for (const bool allow : {false, true}) {
                pr::Lever row{.name = "route", .env = "TEST_ROUTE", .value = {"x", "x", "x"}, .builtin = "b",
                              .required = {true, true, true}, .ready = {ready, ready, ready},
                              .runnable = {runnable, runnable, runnable}, .apps = 1};
                pr::Inputs inputs; inputs.env = lookup(); inputs.allow_pending = allow;
                const auto admitted = common::admit_column(ready, runnable, allow);
                for (const auto p : {pr::Name::standard, pr::Name::fast, pr::Name::quality, pr::Name::turbo}) {
                    auto r = pr::resolve(p, pr::App::engine, inputs, std::span(&row, 1));
                    const auto expected = ready ? pr::Status::set : allow && runnable ? pr::Status::unqualified : pr::Status::pending;
                    check(r.entries[0].status == expected, "adapter admission");
                    check(r.entries[0].value == (expected == pr::Status::pending ? "b" : "x"), "pending fallback");
                    check((admitted == common::Status::pending) == (expected == pr::Status::pending), "shared decision");
                }
                // The common API supports provisional columns; this engine has none.
                check(common::admit_column(ready, runnable, allow, true) ==
                      (!ready && !runnable ? common::Status::pending : common::Status::provisional), "provisional admission");
                row.builtin_unqualified = true;
                bool refused = false;
                try { (void)pr::resolve(pr::Name::standard, pr::App::engine, inputs, std::span(&row, 1)); }
                catch (const std::invalid_argument&) { refused = true; }
                check(refused == (!ready && !(allow && runnable)), "unqualified builtin refusal");
            }
}
void receipt() {
    static_assert(std::is_same_v<pr::Name, common::Name>);
    static_assert(std::is_same_v<pr::Kind, common::Kind>);
    const auto r = pr::host_receipt();
    check(r.engine == "gemma4" && r.implementation == "gemma4-profile", "partial profile adapter provenance");
    check(r.identity.arithmetic_contract.empty(), "unbound arithmetic contract stays unknown");
    check(r.identity.model_sha256.empty() && r.identity.pack_hash.empty() && r.identity.table_hash.empty() &&
          r.identity.state_format.empty() && r.identity.prefill_lineage.empty() && r.identity.reduction_plan.empty(),
          "unbound numerical fields stay unknown");
    check(r.build.empty() && r.kernel.empty() && r.device.empty() && r.topology.empty() && r.command.empty() &&
          r.digests.empty(), "unknown execution fields");
    check(r.witness() == "strix-engine-witness " + r.json() + "\n", "canonical witness framing");
    common::NumericalIdentity known;
    known.model_sha256 = "known-model"; known.pack_hash = "known-pack"; known.table_hash = "known-table";
    known.state_format = "bound-state-format"; known.prefill_lineage = "bound-prefill-lineage";
    known.reduction_plan = "bound-reduction-plan";
    // Hold every other field fixed: the adapter must neither supply nor replace a contract.
    const std::array<std::string_view, 4> contracts = {"", "g4-v1", "g4-a8-v1", "g4-exl3-v3"};
    std::array<std::uint64_t, contracts.size()> ids{};
    for (std::size_t i = 0; i < contracts.size(); ++i) {
        known.arithmetic_contract = contracts[i];
        const auto bound = pr::host_receipt(known);
        check(bound.identity.json() == known.json(), "all supplied numerical fields survive verbatim");
        check(bound.identity.arithmetic_contract == contracts[i], "explicit or unknown contract preserved exactly");
        check(bound.implementation == "gemma4-profile" && bound.build.empty() && bound.kernel.empty() &&
              bound.device.empty() && bound.topology.empty() && bound.command.empty() && bound.digests.empty(),
              "bound identity does not manufacture execution provenance");
        ids[i] = bound.identity.id();
        check(ids[i] != r.identity.id(), "bound identity changes composed ID");
        for (std::size_t j = 0; j < i; ++j)
            check(ids[i] != ids[j], "contract alone changes composed ID");
        auto changed_lineage = known;
        changed_lineage.prefill_lineage += "changed";
        const auto changed = pr::host_receipt(changed_lineage);
        check(changed.identity.arithmetic_contract == known.arithmetic_contract &&
              changed.identity.id() != ids[i], "prefill lineage alone changes composed ID for each contract");
    }
}
} // namespace

int main() {
    try { rows(); admission(); receipt(); }
    catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
    std::cout << "gemma4 migration PASS: " << checks << " checks\n";
}
