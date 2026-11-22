#pragma once

#include "strix/weight/decoder.hpp"

namespace strix::weight {

// Resolved index/launch operands. Shape is contiguous-axis first. The slice
// owns its source; format and arithmetic IDs follow the borrowed Id contract.
struct SiteNeed {
    TensorId tensor;
    std::string name;
    FormatId decoder;
    Shape shape;
    Slice payload;
    DecoderProtocol protocol = DecoderProtocol::a;
};

enum class LoadConversion : std::uint8_t {
    stored, q8_transpose, bf16_to_q8, iu4_rotated, runtime,
};

inline constexpr FormatId kIu4RotatedImageStorage{"image.iu4_rotated_tiled", 1};

struct LoadImage {
    std::string name;
    ImagePlan image;
    std::vector<std::size_t> sites;
    LoadConversion conversion = LoadConversion::stored;
};

enum class LoadFoldKind : std::uint8_t { concatenate, one_plus, scale };

struct LoadFold {
    std::size_t image;   // Index into LoadPlan::images.
    LoadFoldKind kind;
    float factor = 1.0F; // Only used by scale.
};

struct LoadPlan {
    std::vector<SiteNeed> sites;
    std::vector<LoadImage> images;
    // Within an image, folds execute in declaration order before conversion.
    // Concatenation stacks along the slowest axis and comes first.
    std::vector<LoadFold> folds;
};

struct LoadImageReceipt {
    TensorId tensor;  // Primary input; image identifies the particular view.
    std::string image;
    std::uint64_t bytes;
    std::uint32_t crc32c;
};

struct LoadFoldReceipt {
    std::size_t fold;
    std::uint32_t crc32c;
};

struct LoadReceipt {
    std::uint64_t plan_identity{};
    std::vector<LoadImageReceipt> images;
    std::vector<LoadFoldReceipt> folds;
};

class ImageSink {
public:
    virtual ~ImageSink() = default;
    // Stage one complete image. The byte span is valid only for this call.
    // A sink that queues uploads retains its own staging storage.
    virtual Status stage(const LoadImage&, const Shape&, std::span<const Byte>) = 0;
    // Stage and a failed commit leave the previously published images intact.
    // On success publish the complete load and fill the CRC32C of each
    // resident payload in stage order, excluding allocation padding.
    virtual Status commit(std::span<std::uint32_t> resident_crc32c) = 0;
    // Drops incomplete staging and retires any queued work that refers to it.
    virtual void abort() noexcept = 0;
};

// Versioned identity of the ordered recipe, including source IDs/ranges,
// excluding addresses and payload reads. Source byte changes are measured by
// the image CRCs. The caller supplies already selected image rows.
[[nodiscard]] std::uint64_t load_plan_identity(const LoadPlan&) noexcept;

// Execute load-lifetime rows only. A failed load leaves the receipt and the
// published images intact. Session/request rows remain with their runtime owner.
[[nodiscard]] Status load(const LoadPlan&, ImageSink&, LoadReceipt&);

} // namespace strix::weight
