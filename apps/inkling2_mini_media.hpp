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
#if defined(__CUDACC__)
#include "strix/vision/encoder_cuda.hpp"
#endif

#include <filesystem>
#include <functional>
#include <memory>
#include <ostream>
#include <string>

namespace strix::inkling2::cli {

inline std::unique_ptr<serve::SerialMediaInput> make_mini_media(const std::string& checkpoint, const VisionSpec& config,
                                                               VisionIdentity identity) {
    vision::SerialMediaSpec spec;
    spec.shard = checkpoint;
    spec.tensors = vision_tensor_specs(config);
    spec.ignored_prefix = std::string(kVisionIgnoredPrefix);
    spec.placeholder = std::string(serve::kInkling2ImagePlaceholder);
    spec.lm_width = config.text_width;
    // Weight identity of the embed rows: the shard's size (no cross-run cache, so no content hash is needed).
    auto adapter = std::make_unique<VisionAdapter>(
        config, "shard-bytes-" + std::to_string(std::filesystem::file_size(checkpoint)), identity);
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

// The image opener of one run (SerialGenerationFamily::media): the recipe is the run's own resolution of the row
// inkling2-vision-recipe (kMiniProfile, the resolver the CLI and the server share), never a second lookup. It is
// called only for a request with image parts, and only then admits the row (mini_vision_media_tier: a text-only run
// never validates or applies it); each call writes the effective image identity to `receipt`, and for a recipe other
// than the incumbent the row's value and the image run's qualification.
using MiniMediaOpener = std::function<std::unique_ptr<serve::SerialMediaInput>(const std::string& checkpoint)>;
inline MiniMediaOpener mini_media_opener(const profile::Resolution& tier, std::ostream& receipt,
                                         const VisionSpec& config = {}) {
    return [tier, &receipt, config](const std::string& checkpoint) {
        const auto media_tier = mini_vision_media_tier(tier);
        const bool axial = mini_vision_axial(media_tier);
        auto media = make_mini_media(checkpoint, config, axial ? VisionIdentity::axial_v1 : VisionIdentity::legacy);
        receipt << "image identity " << media->adapter().identity() << '\n';
        if (axial)
            receipt << "image profile " << kMiniVisionRecipe << "=axial@1 qualification " << media_tier.qualification()
                    << '\n';
        return media;
    };
}

}  // namespace strix::inkling2::cli
