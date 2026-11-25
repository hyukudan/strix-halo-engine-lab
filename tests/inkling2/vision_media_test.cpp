#include "inkling2_mini_media.hpp"
#include "vision_test_dir.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace strix;
namespace ink = strix::inkling2;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void write_tower(const std::filesystem::path& path, const ink::VisionSpec& spec) {
    auto header = serve::Json::object();
    std::string data;
    std::uint32_t seed = 7;
    for (const auto& tensor : ink::vision_tensor_specs(spec)) {
        auto dims = serve::Json::array();
        std::size_t count = 1;
        for (const auto dim : tensor.extents) {
            dims.push(serve::Json(dim));
            count *= static_cast<std::size_t>(dim);
        }
        auto offsets = serve::Json::array();
        offsets.push(serve::Json(std::uint64_t{data.size()}));
        for (std::size_t i = 0; i < count; ++i) {
            seed = 1664525U * seed + 1013904223U;
            const float value = (static_cast<float>(seed >> 24U) - 128.0F) / 512.0F;
            const auto bits = vision::to_bf16(value + (tensor.name.find("ln_") != std::string::npos &&
                                                       tensor.name.ends_with(".weight") ? 1.0F : 0.0F));
            data.push_back(static_cast<char>(bits & 0xffU));
            data.push_back(static_cast<char>(bits >> 8U));
        }
        offsets.push(serve::Json(std::uint64_t{data.size()}));
        auto entry = serve::Json::object();
        entry.set("dtype", serve::Json(std::string(tensor.wire_dtype)));
        entry.set("shape", std::move(dims));
        entry.set("data_offsets", std::move(offsets));
        header.set(tensor.name, std::move(entry));
    }
    auto json = header.dump();
    while (json.size() % 8U) json.push_back(' ');
    std::ofstream output(path, std::ios::binary);
    const std::uint64_t size = json.size();
    for (unsigned shift = 0; shift < 64U; shift += 8U)
        output.put(static_cast<char>((size >> shift) & 0xffU));
    output << json << data;
    require(static_cast<bool>(output), "write tiny safetensors tower");
}

// The run's resolution of the model table (the CLI and server resolver), hermetic: only the image recipe switch.
profile::Resolution resolve_tier(std::optional<profile::Name> tier, const char* recipe) {
    profile::Inputs in;
    in.profile = tier;
    in.env = [recipe](const char* key) -> const char* {
        return std::string_view(key) == "STRIX_INKLING2_VISION_RECIPE" ? recipe : nullptr;
    };
    return profile::resolve(ink::kMiniProfile, in);
}

ink::VisionIdentity resolved_identity(std::optional<profile::Name> tier, const char* recipe) {
    return ink::mini_vision_axial(resolve_tier(tier, recipe)) ? ink::VisionIdentity::axial_v1 : ink::VisionIdentity::legacy;
}

// r65 #1162.1: the image recipe is a row of the model table. Unset or legacy keeps the incumbent under every profile;
// axial@1 is refused under exact (explicit or default) and quality, admitted (unqualified) under fast / turbo.
void test_recipe_admission() {
    using profile::Name;
    for (const auto tier : {std::optional<Name>{}, std::optional{Name::standard}, std::optional{Name::fast},
                            std::optional{Name::quality}, std::optional{Name::turbo}}) {
        require(resolved_identity(tier, nullptr) == ink::VisionIdentity::legacy, "unset recipe is the incumbent");
        require(resolved_identity(tier, "legacy") == ink::VisionIdentity::legacy, "legacy is the incumbent");
        bool refused = false;
        try {
            require(resolved_identity(tier, "axial@1") == ink::VisionIdentity::axial_v1, "admitted axial runs axial");
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        const bool admits = tier == Name::fast || tier == Name::turbo;
        require(refused != admits, "axial@1 is admitted only under fast or turbo");
        bool malformed = false;
        try {
            static_cast<void>(resolved_identity(tier, "axial"));
        } catch (const std::invalid_argument&) {
            malformed = true;
        }
        require(malformed, "an undeclared recipe value is refused");
    }
    // The effective profile receipt names the explicit unqualified row; the incumbent receipt and fingerprint do not.
    const auto fast_axial = resolve_tier(profile::Name::fast, "axial@1");
    const auto fast_unset = resolve_tier(profile::Name::fast, nullptr);
    require(fast_axial.summary().find("inkling2-vision-recipe=axial@1") != std::string::npos, "receipt names axial");
    require(fast_axial.qualification().find("unqualified: inkling2-vision-recipe") != std::string::npos,
            "receipt stamps axial unqualified");
    require(fast_axial.fingerprint() != fast_unset.fingerprint(), "axial enters the fingerprint");
    require(fast_unset.summary().find("inkling2-vision-recipe") == std::string::npos, "builtin stays out of the receipt");
    require(resolve_tier(profile::Name::fast, "legacy").fingerprint() == fast_unset.fingerprint(),
            "explicit legacy is the incumbent identity");
}

void test_serial_rgba() {
    const auto incumbent = resolved_identity(std::nullopt, nullptr);
    const auto selected = resolved_identity(profile::Name::fast, "axial@1");
    require(incumbent == ink::VisionIdentity::legacy, "incumbent image recipe");
    require(selected == ink::VisionIdentity::axial_v1, "explicit axial image recipe");

    // Real CLI media owner, real schema and PNG, deterministic host tower.
    ink::VisionSpec spec;
    spec.layers = 1;
    spec.width = 16;
    spec.heads = 2;
    spec.mlp_width = 24;
    spec.patch = 4;
    spec.pos_grid = 4;
    spec.projection = 20;
    spec.text_width = 12;
    VisionTestDir fixture;
    const auto shard = fixture.path / "tower.safetensors";
    write_tower(shard, spec);
    auto serial = ink::cli::make_mini_media(shard.string(), spec, selected);
    auto legacy_serial = ink::cli::make_mini_media(shard.string(), spec, incumbent);
    auto& bridge = serial->adapter();
    require(bridge.identity().find("inkling2-vision-axial@1|") == 0, "explicit media identity");
    require(legacy_serial->adapter().identity().find("inkling2-vision-v1|") == 0, "legacy media identity");
    // The CLI hands the opener its text resolution, where the image row is deferred (mini_defer_vision_recipe, as in
    // resolve_mini_profile); an image request re-admits it, runs it and prints the image identity and qualification.
    const auto text_tier = [](profile::Name tier, const char* recipe) {
        auto resolved = resolve_tier(tier, recipe);
        ink::mini_defer_vision_recipe(resolved);
        return resolved;
    };
    std::ostringstream receipt;
    const auto opener = ink::cli::mini_media_opener(text_tier(profile::Name::fast, "axial@1"), receipt, spec);
    require(opener(shard.string())->adapter().identity() == bridge.identity(), "opener runs the resolved recipe");
    require(receipt.str().rfind("image identity " + bridge.identity() + "\n", 0) == 0 &&
                receipt.str().find("image profile inkling2-vision-recipe=axial@1 qualification ") != std::string::npos &&
                receipt.str().find("unqualified: inkling2-vision-recipe") != std::string::npos,
            "image run prints its image identity and the row's qualification");
    std::ostringstream incumbent_receipt;
    static_cast<void>(ink::cli::mini_media_opener(text_tier(profile::Name::fast, nullptr), incumbent_receipt, spec)(
        shard.string()));
    require(incumbent_receipt.str() == "image identity " + legacy_serial->adapter().identity() + "\n",
            "the incumbent image run prints only its image identity");
    // EXACT: building the opener (startup, text-only runs) is not refused; an image request is.
    const auto exact_opener = ink::cli::mini_media_opener(text_tier(profile::Name::standard, "axial@1"), receipt, spec);
    bool exact_refused = false;
    try {
        static_cast<void>(exact_opener(shard.string()));
    } catch (const std::invalid_argument& error) {
        exact_refused = std::string_view(error.what()).find("not admitted") != std::string_view::npos;
    }
    require(exact_refused, "explicit EXACT refuses the axial image request");
    std::ifstream png(std::filesystem::path(STRIX_VISION_FIXTURE_DIR) / "rgba_noise.png", std::ios::binary);
    serve::media::MediaItem item;
    item.bytes.assign(std::istreambuf_iterator<char>(png), std::istreambuf_iterator<char>());
    require(!item.bytes.empty(), "read RGBA fixture");
    const auto span = bridge.encode(item, 4096);
    auto legacy_span = legacy_serial->adapter().encode(item, 4096);

    weight::TensorLookup lookup;
    const auto specs = ink::vision_tensor_specs(spec);
    require(static_cast<bool>(lookup.open_safetensors(shard.string(),
            {weight::Container::safetensors, "safetensors@1", specs, ink::kVisionIgnoredPrefix})), "open fixture tower");
    vision::LookupWeights source(lookup);
    const std::span png_bytes(reinterpret_cast<const std::uint8_t*>(item.bytes.data()), item.bytes.size());
    vision::ColorPolicy checker;
    checker.alpha = vision::AlphaPolicy::checkerboard;
    const auto rgb = vision::decode(png_bytes);
    const auto composited = vision::decode(png_bytes, {}, checker);
    const auto encode = [&](ink::VisionIdentity identity, const vision::Image& image) {
        ink::VisionAdapter adapter(spec, "fixture", identity);
        vision::EncoderWeights weights;
        adapter.bind(source, weights);
        vision::ReferenceEncoder encoder(adapter.config(), weights);
        const auto grid = adapter.patches(image);
        auto hidden = encoder.encode(grid, adapter.positions(adapter.config(), weights, grid));
        vision::ProjectedRows projected;
        adapter.project(hidden, grid, encoder, projected);
        return projected.bf16;
    };
    require(span.lease.rows().data == encode(selected, composited), "CLI RGBA rows must use the checkerboard");
    require(span.lease.rows().data != encode(selected, rgb), "RGBA fixture must distinguish alpha policies");
    require(legacy_span.lease.rows().data == encode(incumbent, rgb), "legacy media must keep alpha drop");
}

}  // namespace

int main() try {
    test_recipe_admission();
    test_serial_rgba();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
