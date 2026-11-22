#pragma once

#include "strix/engine_profile.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strix::receipt {

inline constexpr char schema_id[] = "strix-receipt/v1";

struct Identity {
    std::string engine;
    std::optional<profile::NumericalIdentity> numerical;
    std::string declaration_hash, catalogue_build_id, chain_table_id;
    std::vector<std::string> programs, numeric_contracts;
    std::string route_fingerprint;
    std::string implementation, build, kernel, device, topology, command;
    std::vector<std::pair<std::string, std::string>> digests;
};

enum class Tier { exact, base, fast, turbo };
enum class Reason { explicit_value, gate, regime, default_value };
struct Lever {
    std::string name, value;
    Reason reason = Reason::default_value;
    std::string off_switch;
};
struct Resolution {
    Tier tier = Tier::exact;
    std::vector<Lever> levers;
    std::string description;
};

struct Tensor {
    std::string name, source, format;
    std::uint64_t bytes = 0;
    std::uint32_t crc = 0;
};
struct Image {
    std::string name, source, lifetime;
    std::uint32_t crc = 0;
};
struct Load {
    std::vector<Tensor> tensors;
    std::vector<Image> images;
    std::vector<std::string> folds, residency_plan;
};

struct ProgramPlan {
    std::string program;
    std::uint64_t launches = 0, packs = 0, sidecars = 0;
};
struct DeclinedFusion {
    std::string program, fusion, missing_catalogue_key;
};
struct Plan {
    std::vector<ProgramPlan> programs;
    std::vector<DeclinedFusion> declined_fusions;
    std::vector<std::string> split_plan;
};

struct ProgramStep {
    std::string program;
    std::optional<double> predicted_ms, wall_ms;
    std::uint64_t launches = 0, bytes_read = 0, exchanges = 0, guards = 0, guard_failures = 0;
    std::uint32_t device_status = 0;
};
struct Speculation {
    std::uint64_t cycles = 0, proposed = 0, accepted = 0;
    std::vector<std::uint64_t> rows_per_verify;
    double drafter_ms = 0;
    std::string controller_state;
};
struct Lease {
    std::string lease;
    std::uint64_t tokens = 0, prefix_hit_length = 0, rounds = 0;
    double queue_ms = 0;
    std::string finish_reason;
};

struct LinkCost {
    std::string peer;
    double latency_ms = 0, bytes_per_second = 0;
};
struct Telemetry {
    double elapsed_ms = 0;
    std::optional<double> clock_mhz, memory_clock_mhz, power_w;
};
struct Device {
    std::string profile_id, power_state;
    std::optional<double> bytes_per_second, clock_mhz, memory_clock_mhz;
    std::vector<LinkCost> links;
    std::vector<Telemetry> telemetry;
};
struct Gate {
    std::string lever, record_id;
};

// An absent section is unobserved, while an empty list records zero entries.
struct Receipt {
    Identity identity;
    std::optional<Resolution> resolution;
    std::optional<Load> load;
    std::optional<Plan> plan;
    std::optional<std::vector<ProgramStep>> step;
    std::optional<Speculation> speculation;
    std::optional<std::vector<Lease>> serving;
    std::optional<std::vector<Device>> devices;
    std::optional<std::vector<Gate>> quality;
};

}  // namespace strix::receipt
