#include "strix/receipt/legacy.hpp"
#include "strix/receipt/writer.hpp"
#include "strix/route_receipt.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
namespace sr = strix::receipt;
using strix::receipt::detail::member;
using strix::serve::parse_json_unique;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

void save(const std::filesystem::path& path, const std::string& json) {
    std::ofstream output(path, std::ios::binary);
    output << json << '\n';
    output.close();
    if (!output) throw std::runtime_error("could not write receipt fixture");
}

sr::Receipt legacy(const std::filesystem::path& directory) {
    const strix::profile::ExecutionReceipt engine{"host-fixture",
        {std::string("fp32\0contract", sizeof("fp32\0contract") - 1) + "\xc3\xa9",
            "weights", "pack", "table", "state", "lineage", "reductions"},
        "host-test", "build", "kernel", "device", "topology", "test \"argument\"", {{"tokens", "abc"}}};
    const auto original = parse_json_unique(engine.json());
    auto mapped = sr::map_legacy_identity(engine.json());
    const auto encoded = sr::write_json(mapped);
    const auto document = parse_json_unique(encoded);
    const auto& identity = member(document, "identity");
    require(member(identity, "numerical").dump() == member(original, "identity").dump(), "eight identity fields changed");
    for (const auto* name : {"engine", "build", "device", "topology", "kernel", "implementation", "command", "digests"})
        require(member(identity, name).dump() == member(original, name).dump(), "engine provenance changed");
    require(encoded == sr::write_json(sr::map_legacy_identity(engine.witness())), "engine witness framing changes identity");
    save(directory / "engine.json", encoded);

    strix::route::set_engine(engine.engine);
    strix::route::lever("fixture", "auto", "host");
    const auto route = strix::route::json(strix::route::Totals{}, "request", false);
    const auto route_original = parse_json_unique(route);
    const auto route_encoded = sr::write_json(sr::map_legacy_identity(route));
    require(route_encoded == sr::write_json(sr::map_legacy_identity("strix-route-receipt " + route + '\n')),
            "route stderr framing changes identity");
    const auto route_document = parse_json_unique(route_encoded);
    const auto& route_identity = member(route_document, "identity");
    require(member(route_identity, "engine").dump() == member(identity, "engine").dump(), "shared engine tag changed");
    require(member(route_identity, "route_fingerprint").dump() == member(route_original, "fingerprint").dump(),
            "route fingerprint changed");
    require(member(route_identity, "numerical").is_null(), "route fingerprint created numerical identity");
    save(directory / "route.json", route_encoded);
    std::cout << "receipt_identity: PASS engine and route\n";
    return mapped;
}

void full_receipt(const std::filesystem::path& directory, sr::Receipt receipt) {
    receipt.identity.declaration_hash = "decl";
    receipt.identity.catalogue_build_id = "catalogue";
    receipt.identity.chain_table_id = "chain";
    receipt.identity.programs = {"step"};
    receipt.identity.numeric_contracts = {"arithmetic"};
    receipt.resolution = sr::Resolution{sr::Tier::fast, {{"host", "on", sr::Reason::gate, "--host off"}}, "resolved"};
    receipt.load = sr::Load{{{"weight", "tensor", "fmt", std::numeric_limits<std::uint64_t>::max(), 42}},
        {{"image", "tensor", "load", 55}}, {"fold"}, {"resident"}};
    receipt.plan = sr::Plan{{{"step", 2, 1, 1}}, {{"step", "fusion", "missing-key"}}, {"split"}};
    receipt.step = {{sr::ProgramStep{"step", 1.5, std::nullopt, 2, 32, 1, 4, 1, 0}}};
    receipt.speculation = sr::Speculation{1, 2, 1, {2}, 0.125, "steady"};
    receipt.serving = {{sr::Lease{"lease", 1, 0, 2, 0.25, "stop"}}};
    receipt.devices = {{sr::Device{"synthetic", "captured", 10, 2, std::nullopt,
        {{"peer", 1, 10}}, {{1, 2, std::nullopt, 3}}}}};
    receipt.quality = {{sr::Gate{"host", "record"}}};
    save(directory / "complete.json", sr::write_json(receipt));
    receipt.speculation->accepted = 3;
    try {
        (void)sr::write_json(receipt);
        throw std::runtime_error("invalid speculation was serialized");
    } catch (const std::invalid_argument& error) {
        require(std::string_view(error.what()).starts_with("speculation.accepted"), "wrong validation error");
    }
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument("usage: receipt-tests OUTPUT_DIR");
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    full_receipt(directory, legacy(directory));
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
