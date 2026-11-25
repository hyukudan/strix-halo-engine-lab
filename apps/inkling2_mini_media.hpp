#pragma once
// Inkling 2 Mini image input of the serial CLI: the family's VisionAdapter, its tower inventory and the encoder
// backend of this build (HIP: vision::Encoder, CUDA: vision::CudaEncoder, CPU: the FP32 host reference), handed to the
// shared vision::SerialMedia. Model-specific only in what it names.
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_vision.hpp"
#include "strix/profiles/inkling2_mini.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/vision/serial_media.hpp"
#include "strix/vision/encoder.hpp"
#include "strix/weight/confined_file.hpp"
#if defined(__CUDACC__)
#include "strix/vision/encoder_cuda.hpp"
#endif

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace strix::inkling2::cli {

inline constexpr std::size_t kMiniVisionConfigBytes = 1U << 20U;

struct MiniVisionCheckpoint {
    VisionSpec spec;
    std::shared_ptr<const weight::Source> shard;
    std::vector<serve::FileKey> keys;  // of the handles read: the shard, then the sidecar when there is one
};

inline MiniVisionCheckpoint read_mini_vision_checkpoint(const weight::ConfinedDirectory& directory,
                                                         const std::filesystem::path& basename) {
    MiniVisionCheckpoint out;
    out.shard = directory.open(basename);
    const auto shard_key = weight::source_file_key(*out.shard);
    if (!shard_key) throw std::runtime_error("could not identify the checkpoint shard");
    out.keys.push_back({shard_key->device, shard_key->file});
    std::optional<weight::FileKey> sidecar_key;
    const auto sidecar = directory.read("config.json", kMiniVisionConfigBytes, true, {}, &sidecar_key);
    if (sidecar_key) out.keys.push_back({sidecar_key->device, sidecar_key->file});
    out.spec = sidecar ? parse_vision_spec(*sidecar) : VisionSpec{};
    return out;
}

inline std::unique_ptr<serve::SerialMediaInput> make_mini_media(std::shared_ptr<const weight::Source> checkpoint,
                                                               const VisionSpec& config,
                                                               VisionIdentity identity) {
    vision::SerialMediaSpec spec;
    const auto shard_bytes = checkpoint->size();
    spec.shard_source = std::move(checkpoint);
    spec.tensors = vision_tensor_specs(config);
    spec.ignored_prefix = std::string(kVisionIgnoredPrefix);
    spec.placeholder = std::string(serve::kInkling2ImagePlaceholder);
    spec.lm_width = config.text_width;
    // Weight identity of the embed rows: the shard's size (no cross-run cache, so no content hash is needed).
    auto adapter = std::make_unique<VisionAdapter>(
        config, "shard-bytes-" + std::to_string(shard_bytes), identity);
    return std::make_unique<vision::SerialMedia>(std::move(adapter), std::move(spec),
        [](const vision::EncoderConfig& c, const vision::EncoderWeights& w) -> std::unique_ptr<vision::EncoderBackend> {
#if defined(__HIPCC__)
            auto encoder = std::make_unique<vision::Encoder>(c);
            encoder->load(w);
            return encoder;
#elif defined(__CUDACC__)
            auto encoder = std::make_unique<vision::CudaEncoder>(c);
            encoder->load(w);
            return encoder;
#else
            return std::make_unique<vision::ReferenceEncoder>(c, w);  // the CPU reference CLI
#endif
        });
}

// One image input of a run: the shared SerialMedia plus the handles it read. The CLI refuses an output that is one of
// those files (before any device work), and the language model loads from the same opened shard as the tower: while
// it loads, open_source of the checkpoint name returns the held shard (weight::SourcePin), and a load that did not
// open it is refused.
class MiniMediaInput final : public serve::SerialMediaInput {
public:
    MiniMediaInput(std::unique_ptr<serve::SerialMediaInput> media, std::vector<serve::FileKey> keys,
                   std::shared_ptr<const weight::Source> shard)
        : media_(std::move(media)), keys_(std::move(keys)), shard_(std::move(shard)) {}
    [[nodiscard]] serve::ApiMediaAdapter& adapter() override { return media_->adapter(); }
    [[nodiscard]] std::vector<serve::FileKey> input_keys() const override { return keys_; }
    void load_checkpoint(const std::string& checkpoint, const std::function<void()>& load) const override {
        const weight::SourcePin pin(checkpoint, shard_);
        load();
        if (pin.uses() == 0U)
            throw std::runtime_error("the language model did not load the shard the image tower was read from");
    }

private:
    std::unique_ptr<serve::SerialMediaInput> media_;
    std::vector<serve::FileKey> keys_;
    std::shared_ptr<const weight::Source> shard_;
};

// Opens the checkpoint's image assets: a linked checkpoint names its target, and the shard and its config.json
// sidecar are read through the target's held directory. The language model later loads from this same opened shard
// (MiniMediaInput::load_checkpoint). `config` overrides the sidecar (host fixtures).
inline std::unique_ptr<serve::SerialMediaInput> open_mini_media(const std::string& checkpoint, VisionIdentity identity,
                                                                const std::optional<VisionSpec>& config = {}) {
    const auto model = std::filesystem::canonical(checkpoint);
    const weight::ConfinedDirectory directory(model.parent_path());
    auto assets = read_mini_vision_checkpoint(directory, model.filename());
    auto shard = assets.shard;
    auto media = make_mini_media(std::move(assets.shard), config ? *config : assets.spec, identity);
    return std::make_unique<MiniMediaInput>(std::move(media), std::move(assets.keys), std::move(shard));
}

// The image opener of one run (SerialGenerationFamily::media): the recipe is the run's own resolution of the row
// inkling2-vision-recipe (kMiniProfile, the resolver the CLI and the server share), never a second lookup. It is
// called only for a request with image parts, and only then admits the row (mini_vision_media_tier: a text-only run
// never validates or applies it); each call writes the effective image identity to `receipt`, and for a recipe other
// than the incumbent the row's value and the image run's qualification.
using MiniMediaOpener = std::function<std::unique_ptr<serve::SerialMediaInput>(const std::string& checkpoint)>;
inline MiniMediaOpener mini_media_opener(const profile::Resolution& tier, std::ostream& receipt,
                                         std::optional<VisionSpec> config = {}) {
    return [tier, &receipt, config](const std::string& checkpoint) {
        const auto media_tier = mini_vision_media_tier(tier);
        const bool axial = mini_vision_axial(media_tier);
        auto media = open_mini_media(checkpoint, axial ? VisionIdentity::axial_v1 : VisionIdentity::legacy, config);
        receipt << "image identity " << media->adapter().identity() << '\n';
        if (axial)
            receipt << "image profile " << kMiniVisionRecipe << "=axial@1 qualification " << media_tier.qualification()
                    << '\n';
        return media;
    };
}

}  // namespace strix::inkling2::cli
