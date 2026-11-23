#pragma once

#include "strix/profile/gate_records.hpp"
#include "strix/exec/catalog.hpp"

namespace strix::profile {

// The planner declares the keys a model can use at each site class. The
// reference identity is explicit because D4's reference field says BASE or
// host bitwise, not which identity the model uses as its reference.
struct CatalogSite {
    std::string_view site;
    exec::KernelKey key;
    std::string_view reference_identity;
};

struct CatalogLever {
    std::string name; // Site class and versioned entry identity
    std::string identity;
    lever::Exactness exactness;
    exec::ReferenceClass reference;
    Name tier;
};

struct CatalogLevers {
    std::string device;
    std::vector<CatalogLever> rows;
};

// D4 currently supplies exactness but no FAST/TURBO designation. The optional
// list is catalogue metadata keyed by entry identity, shared across sites and
// models. A bit-changing entry without a TURBO designation is a FAST candidate.
[[nodiscard]] CatalogLevers levers_of(const exec::Catalog&, const device::Profile&,
                                      std::span<const CatalogSite> sites,
                                      std::span<const std::string_view> turbo_entries = {});

struct CatalogInputs {
    Inputs options;
    lever::Inputs regime;
    std::string_view model, gate_set, content_class, band;
    std::span<const std::string_view> force_off{};
    std::span<const std::string_view> provisional{};
};

enum class CatalogReason { reference, tier, gate, regime, explicit_switch, model_off };

struct CatalogDecision {
    CatalogLever lever;
    bool on;
    CatalogReason reason;
    std::string receipt;
    bool provisional = false;
    bool gate_not_run = false;
};

struct CatalogResolution {
    Name tier;
    std::vector<CatalogDecision> entries;

    [[nodiscard]] std::string describe() const;
};

[[nodiscard]] CatalogResolution resolve(const CatalogLevers&, const CatalogInputs&,
                                        std::span<const GateRecord> gates,
                                        std::span<const RegimeRecord> regimes);

enum class OptionSource { request, lever, device };

struct CatalogOption {
    std::string option;
    OptionSource source;
    std::string lever; // Populated for a catalogue lever. The option takes 0 or 1.
};

[[nodiscard]] std::vector<CatalogOption> options_table(const CatalogLevers&,
    std::span<const std::string_view> request_options = {},
    std::span<const std::string_view> device_options = {});

// Uses the common tier parser. Unconsumed request, device and lever tokens
// remain available to the native parsers, as with take_options.
void take_catalog_options(std::span<const CatalogOption>, std::vector<std::string>&, Inputs&);

struct CatalogBinding {
    std::string_view lever, legacy_row, on, off;
};

// Explicit bindings translate admitted catalogue choices into legacy cells.
// The returned cells borrow the binding strings. Unbound rows are left to the
// compatibility adapter, including all of today's native override semantics.
[[nodiscard]] std::vector<EffectiveCell> effective_cells(const CatalogResolution&, std::span<const CatalogBinding>);

} // namespace strix::profile
