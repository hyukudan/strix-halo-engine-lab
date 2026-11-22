#pragma once
// Inkling 2 Mini image input of the serial CLI: the family's VisionAdapter, its tower inventory and the encoder
// backend of this build, handed to the shared vision::SerialMedia. Model-specific only in what it names.
#include "inkling2/inkling2_mini_config.hpp"
#include "inkling2/inkling2_vision_adapter.hpp"
#include "strix/serve/inkling2_chat.hpp"
#include "strix/vision/serial_media.hpp"
#if defined(__HIPCC__)
#include "strix/vision/encoder.hpp"
#endif

#include <filesystem>
#include <memory>
#include <string>

namespace strix::inkling2::cli {

inline std::unique_ptr<serve::SerialMediaInput> open_mini_media(const std::string& checkpoint) {
    const VisionConfig config{};
    vision::SerialMediaSpec spec;
    spec.shard = checkpoint;
    spec.tensors = vision_tensor_schema(config);
    spec.ignored_prefix = std::string(kVisionIgnoredPrefix);
    spec.placeholder = std::string(serve::kInkling2ImagePlaceholder);
    spec.lm_width = config.text_hidden_size;
    // Weight identity of the embed rows: the shard's size (no cross-run cache, so no content hash is needed).
    auto adapter = std::make_unique<VisionAdapter>(
        config, "shard-bytes-" + std::to_string(std::filesystem::file_size(checkpoint)));
    return std::make_unique<vision::SerialMedia>(std::move(adapter), std::move(spec),
        [](const vision::EncoderConfig& c, const vision::EncoderWeights& w) -> std::unique_ptr<vision::EncoderBackend> {
#if defined(__HIPCC__)
            auto encoder = std::make_unique<vision::Encoder>(c);
            encoder->load(w);
            return encoder;
#else
            // No CUDA device tower yet: the shared FP32 host reference.
            return std::make_unique<vision::ReferenceEncoder>(c, w);
#endif
        });
}

}  // namespace strix::inkling2::cli
