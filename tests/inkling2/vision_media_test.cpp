#include "inkling2_mini_media.hpp"
#include "vision_test_dir.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {

using namespace strix;
namespace ink = strix::inkling2;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template <class Call>
bool rejects(const Call& call, std::string_view expected) {
    try {
        call();
    } catch (const std::exception& error) {
        return std::string_view(error.what()).find(expected) != std::string_view::npos;
    }
    return false;
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
    const weight::ConfinedDirectory directory(fixture.path);
    const auto assets = ink::cli::read_mini_vision_checkpoint(directory, shard.filename());
    auto serial = ink::cli::make_mini_media(assets.shard, spec, selected);
    auto legacy_serial = ink::cli::make_mini_media(assets.shard, spec, incumbent);
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
    // The opened input keys the handles it read (the CLI refuses those files as outputs) and the language model
    // loads from that same opened shard: while it loads, open_source of the checkpoint name returns the held file even
    // after the name was replaced, and a load that does not open the name is refused.
    {
        const auto opened = opener(shard.string());
        const auto keys = opened->input_keys();
        const auto shard_key = serve::file_key(shard);
        require(keys.size() == 1U && shard_key && keys[0] == *shard_key, "the image input keys its opened shard");
        const auto original = fixture.path / "original.safetensors";
        std::filesystem::rename(shard, original);
        std::filesystem::copy_file(original, shard);  // the name now reaches another file
        std::shared_ptr<const weight::Source> loaded;
        opened->load_checkpoint(shard.string(), [&] {
            require(static_cast<bool>(weight::open_source(shard.string(), loaded)), "language model opens the name");
        });
        const auto loaded_key = weight::source_file_key(*loaded);
        require(loaded_key && loaded_key->file == shard_key->file && loaded_key->device == shard_key->device,
                "the language model reads the shard the tower read, not the replaced name");
        require(loaded->shard_name() == "tower.safetensors", "the held shard keeps the name's shard namespace");
        require(rejects([&] { opened->load_checkpoint(shard.string(), [] {}); }, "did not load the shard"),
                "a load that does not open the pinned shard is refused");
        loaded.reset();
        std::filesystem::remove(shard);
        std::filesystem::rename(original, shard);
        std::error_code link_error;
        const auto link = fixture.path / "tower-link.safetensors";
        std::filesystem::create_symlink(shard, link, link_error);
        if (!link_error) {  // a linked checkpoint opens its target, as the language model loader does
            const auto linked = opener(link.string());
            require(linked->input_keys()[0] == *serve::file_key(shard), "a linked checkpoint keys its target");
            linked->load_checkpoint(link.string(), [&] {
                require(static_cast<bool>(weight::open_source(link.string(), loaded)), "load through the link");
            });
            require(loaded->shard_name() == "tower-link.safetensors", "the link name keeps its shard namespace");
            loaded.reset();
        }
    }
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

void test_checkpoint_admission() {
    namespace fs = std::filesystem;
    const auto write = [](const fs::path& name, std::string_view bytes) {
        std::ofstream stream(name, std::ios::binary);
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        require(static_cast<bool>(stream), "write checkpoint fixture");
    };
    constexpr std::string_view valid = R"({"vision_config":{"model_type":"inkling2_vision"}})";
    VisionTestDir fixture;
    const auto model_dir = fixture.path / "model";
    fs::create_directory(model_dir);
    const auto config = model_dir / "config.json";
    write(model_dir / "tower.safetensors", "original");
    const weight::ConfinedDirectory directory(model_dir);
    const auto open = [&] { return ink::cli::read_mini_vision_checkpoint(directory, "tower.safetensors"); };
    require(open().spec.width == 1024U, "a shard alone uses the fixed spec");
    require(rejects([&] { (void)ink::cli::open_mini_media((model_dir / "tower.safetensors").string(), ink::VisionIdentity::legacy); },
                    "tower tensors"), "the production opener reaches shard validation without a sidecar");
    std::string capped(valid);
    capped.resize(ink::cli::kMiniVisionConfigBytes, ' ');
    write(config, capped);
    require(open().spec.width == 1024U, "accept config at the one-MiB boundary");
    {
        const auto assets = open();
        const auto config_key = serve::file_key(config);
        require(assets.keys.size() == 2U && config_key && assets.keys[1] == *config_key,
                "the sidecar is keyed by the handle it was read from");
    }
    capped.push_back(' ');
    write(config, capped);
    require(rejects([&] { (void)open(); }, "length"), "refuse config over one MiB");
    write(config, valid.substr(0, valid.size() - 1U));
    require(rejects([&] { (void)open(); }, ""), "refuse a truncated config document");
    write(config, R"({"vision_config":{"model_type":"inkling2_vision","hidden_size":2}})");
    require(rejects([&] { (void)open(); }, "hidden_size"), "refuse unsupported geometry");

    fs::remove(config);
    fs::create_directory(config);
    require(rejects([&] { (void)open(); }, "config.json"), "refuse a directory in the config slot");
    fs::remove(config);
#ifndef _WIN32
    require(::mkfifo(config.c_str(), 0600) == 0, "make fixture FIFO");
    require(rejects([&] { (void)open(); }, "config.json"), "refuse a FIFO in the config slot");
    fs::remove(config);
#endif
    write(fixture.path / "outside.json", valid);
    std::error_code link_error;
    fs::create_symlink(fixture.path / "outside.json", config, link_error);
#ifndef _WIN32
    require(!link_error, "create sidecar symlink fixture");
#endif
    if (!link_error) {
        require(rejects([&] { (void)open(); }, "config.json"), "refuse a config symlink");
        fs::create_symlink(fixture.path / "outside.json", model_dir / "shard-link");
        require(rejects([&] { (void)ink::cli::read_mini_vision_checkpoint(directory, "shard-link"); },
                        "shard-link"), "refuse a shard symlink");
        fs::remove(config);
        fs::create_symlink(fixture.path / "absent.json", config);
        require(rejects([&] { (void)open(); }, "config.json"), "a dangling config symlink is not an absent sidecar");
        fs::remove(config);
    } else {
        std::fprintf(stderr, "symbolic link fixture unavailable: %s\n", link_error.message().c_str());
    }

    write(config, valid);
    fs::rename(model_dir, fixture.path / "held");
    fs::create_directory(model_dir);
    write(config, "{}");
    write(model_dir / "tower.safetensors", "replacement");
    const auto assets = open();
    std::string contents(8, '\0');
    require(assets.spec.width == 1024U &&
            static_cast<bool>(assets.shard->read({0, contents.size()},
                                                {reinterpret_cast<weight::Byte*>(contents.data()), contents.size()})) &&
            contents == "original", "config and shard must open within the same held directory");
}

}  // namespace

int main() try {
    test_recipe_admission();
    test_serial_rgba();
    test_checkpoint_admission();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
