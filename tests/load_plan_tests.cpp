#include "strix/weight/load.hpp"

#include "core/iu4a4_rot.hpp"
#include "core/q8_0_transpose.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/weight/registry.hpp"
#include "strix/weight/source.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace {
namespace w = strix::weight;
namespace rot = strix::core::iu4a4_rot;
namespace onfly = strix::core::onfly;

constexpr auto f32 = w::decoder_format(w::DecoderFormat::f32).id;
constexpr auto bf16 = w::decoder_format(w::DecoderFormat::bf16).id;
constexpr auto q8 = w::decoder_format(w::DecoderFormat::q8_0).id;
constexpr auto iu4 = w::decoder_format(w::DecoderFormat::iu4_g64).id;

void require(bool condition, std::string_view detail) {
    if (!condition) throw std::runtime_error(std::string(detail));
}

std::uint32_t crc32c(std::span<const w::Byte> bytes) {
    std::uint32_t crc = 0xffffffffU;
    for (const auto b : bytes) {
        crc ^= std::to_integer<std::uint32_t>(b);
        for (unsigned bit = 0; bit < 8U; ++bit)
            crc = (crc >> 1U) ^ ((crc & 1U) ? 0x82f63b78U : 0U);
    }
    return ~crc;
}

struct Resident {
    std::string name;
    w::Shape shape;
    std::vector<w::Byte> bytes;
};

class CpuSink final : public w::ImageSink {
    std::vector<Resident> pending_;

public:
    std::vector<Resident> resident;
    std::size_t stage_calls{};
    std::size_t abort_calls{};
    bool reject_stage = false;
    bool reject_commit = false;

    w::Status stage(const w::LoadImage& row, const w::Shape& shape, std::span<const w::Byte> bytes) override {
        ++stage_calls;
        pending_.push_back({row.name, shape, {bytes.begin(), bytes.end()}});
        if (reject_stage) return {w::Error::device_failure, "fixture stage failure"};
        return {w::Error::ok, {}};
    }

    w::Status commit(std::span<std::uint32_t> resident_crcs) override {
        if (reject_commit) return {w::Error::device_failure, "fixture commit failure"};
        require(resident_crcs.size() == pending_.size(), "sink commit CRC count");
        resident = std::move(pending_);
        pending_.clear();
        for (std::size_t i = 0; i < resident.size(); ++i) resident_crcs[i] = crc32c(resident[i].bytes);
        return {w::Error::ok, {}};
    }

    void abort() noexcept override {
        ++abort_calls;
        pending_.clear();
    }
};

class UnreadableSource final : public w::Source {
    std::uint64_t size_;

public:
    mutable unsigned reads{};
    explicit UnreadableSource(std::uint64_t size) : size_(size) {}
    w::Status read(w::ByteRange, std::span<w::Byte>) const override {
        ++reads;
        return {w::Error::stale_source, "fixture read failure"};
    }
    std::uint64_t size() const noexcept override { return size_; }
    w::Digest256 identity() const override { return {}; }
};

template<class T, std::size_t extent>
std::span<const w::Byte> wire_bytes(std::span<T, extent> data) {
    static_assert(std::endian::native == std::endian::little);
    return std::as_bytes(data);
}

std::size_t add_site(w::LoadPlan& plan, std::string name, w::FormatId decoder, w::Shape shape,
                     std::span<const w::Byte> bytes, w::DecoderProtocol protocol = w::DecoderProtocol::a) {
    constexpr std::size_t prefix = 3;
    std::vector<w::Byte> shard(prefix, w::Byte{0xa5});
    shard.insert(shard.end(), bytes.begin(), bytes.end());
    const auto index = plan.sites.size();
    plan.sites.push_back({100U + index, std::move(name), decoder, shape,
                          {w::memory_source(shard), {prefix, bytes.size()}}, protocol});
    return index;
}

void add_stored(w::LoadPlan& plan, std::size_t site) {
    const auto& input = plan.sites[site];
    plan.images.push_back({input.name, {input.decoder, input.decoder, input.payload.range.bytes, "stored.v1"}, {site}});
}

void match(const w::LoadPlan& plan, const CpuSink& sink, const w::LoadReceipt& receipt,
           std::size_t image, w::Shape shape, std::span<const w::Byte> expected) {
    const auto& row = plan.images.at(image);
    const auto& resident = sink.resident.at(image);
    const auto& record = receipt.images.at(image);
    require(resident.name == row.name && record.image == row.name, "resident/receipt image name");
    require(resident.shape.dims == shape.dims && resident.shape.rank == shape.rank, "resident image shape");
    require(record.tensor == plan.sites.at(row.sites.front()).tensor && record.bytes == expected.size(),
            "resident/receipt image attribution");
    require(record.crc32c == crc32c(expected), "resident image CRC32C");
    require(resident.bytes.size() == expected.size() &&
            std::memcmp(resident.bytes.data(), expected.data(), expected.size()) == 0, "resident image memcmp");
    require(receipt.plan_identity == w::load_plan_identity(plan), "receipt plan identity");
}

std::vector<std::uint16_t> bf16_pattern(std::size_t count) {
    std::vector<std::uint16_t> data(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto signed_pattern = static_cast<int>((i * 47U + i / 37U) % 181U) - 90;
        const int denom = 8 << (i / 24U % 4U);
        const float value = static_cast<float>(signed_pattern) / static_cast<float>(denom);
        data[i] = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
    }
    std::fill_n(data.begin(), 32, std::uint16_t{0}); // Include a zero block.
    return data;
}

void qwen_layer_set() {
    w::LoadPlan plan;
    const float gains[] = {0.91F, 1.002F, -0.0F, 1.125F};
    const auto norm = add_site(plan, "blk.0.attn_norm.weight", f32, {{4}, 1}, wire_bytes(std::span(gains)));
    constexpr w::Shape dense_shape{{128, 16}, 2};
    const auto dense = bf16_pattern(128U * 16U);
    const auto projection = add_site(plan, "blk.0.ffn_gate.weight", bf16, dense_shape,
                                     wire_bytes(std::span(dense)), w::DecoderProtocol::b);
    add_stored(plan, norm);
    add_stored(plan, projection);

    // HIP load_a4 delegates to a device-only converter. Its existing host twin
    // computes row-major steps. Rearrange them into the HIP resident order.
    constexpr unsigned width = 256, rows = 32, blocks = width / rot::kBlock;
    constexpr w::Shape iu4_shape{{width, rows}, 2};
    std::vector<std::uint8_t> codes(width * rows / 2U);
    for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = static_cast<std::uint8_t>((i * 71U + i / 16U + i / 67U) & 255U);
    std::vector<std::uint16_t> scales(rows * (width / 64U));
    for (std::size_t i = 0; i < scales.size(); ++i)
        scales[i] = static_cast<std::uint16_t>(0x2c00U + i * 19U);
    std::vector<w::Byte> packed;
    for (const auto bytes : {wire_bytes(std::span<const std::uint8_t>(codes)), wire_bytes(std::span<const std::uint16_t>(scales))})
        packed.insert(packed.end(), bytes.begin(), bytes.end());
    const auto qkv = add_site(plan, "blk.1.attn_qkv.weight", iu4, iu4_shape, packed, w::DecoderProtocol::c);
    add_stored(plan, qkv);
    plan.images.push_back({"blk.1.attn_qkv.rotated",
        {iu4, w::kIu4RotatedImageStorage, packed.size(), rot::kIdentity}, {qkv}, w::LoadConversion::iu4_rotated});

    // CUDA's preconverted route reads tile bytes and its F32 global scale
    // directly from the pack. Generate one 16x128 record and a scale entry.
    const auto tile_format = w::gguf_format(strix::gguf::kNvfp4TilePack);
    std::array<std::uint8_t, 1152> tile{};
    for (std::size_t i = 0; i < 1024; ++i) tile[i] = static_cast<std::uint8_t>((37U * i) & 255U);
    std::fill(tile.begin() + 1024, tile.end(), std::uint8_t{0x38});
    const float global_scale[] = {0.125F};
    add_stored(plan, add_site(plan, "blk.0.ffn_up.weight", tile_format, dense_shape, wire_bytes(std::span<const std::uint8_t>(tile))));
    add_stored(plan, add_site(plan, "strix.nvfp4.gscale", f32, {{1}, 1}, wire_bytes(std::span(global_scale))));

    std::vector<std::uint8_t> rotated(codes.size());
    std::vector<float> host_steps(rows * blocks), device_order(host_steps.size());
    for (unsigned col = 0; col < rows; ++col) {
        for (unsigned block = 0; block < blocks; ++block) {
            rot::requantise_block(codes.data(), scales.data(), width, rows, col, block, rotated.data(), host_steps.data());
            const auto offset = ((block * (rows / 16U) + col / 16U) * 2U + col % 2U) * 8U + col % 16U / 2U;
            device_order[offset] = host_steps[col * blocks + block];
        }
    }
    std::vector<w::Byte> reference;
    for (const auto bytes : {wire_bytes(std::span<const std::uint8_t>(rotated)), wire_bytes(std::span<const float>(device_order))})
        reference.insert(reference.end(), bytes.begin(), bytes.end());

    CpuSink sink;
    w::LoadReceipt receipt;
    const auto status = w::load(plan, sink, receipt);
    require(static_cast<bool>(status), status.detail);
    require(receipt.images.size() == 6 && receipt.folds.empty(), "Qwen image/fold count");
    match(plan, sink, receipt, 0, {{4}, 1}, wire_bytes(std::span(gains)));
    match(plan, sink, receipt, 1, dense_shape, wire_bytes(std::span(dense)));
    match(plan, sink, receipt, 2, iu4_shape, packed);
    match(plan, sink, receipt, 3, iu4_shape, reference);
    match(plan, sink, receipt, 4, dense_shape, wire_bytes(std::span<const std::uint8_t>(tile)));
    match(plan, sink, receipt, 5, {{1}, 1}, wire_bytes(std::span(global_scale)));
    std::puts("PASS Qwen 27B stored, converted-pack and rotated IU4 images (CRC32C + memcmp)");
}

void hy4_layer_set() {
    w::LoadPlan plan;
    constexpr unsigned batches = 2, rows = 64, cols = 96;
    constexpr w::Shape key_shape{{cols, rows, batches}, 3};
    const auto key_bf16 = bf16_pattern(batches * rows * cols);
    const auto key = onfly::bf16_to_q8_0_host(key_bf16.data(), batches * rows, cols);
    const auto key_site = add_site(plan, "blk.0.attn_k_b.weight", q8, key_shape, wire_bytes(std::span(key)));
    const auto transposed = strix::core::q8_0::transpose(key.data(), batches, rows, cols);

    constexpr unsigned head_width = 64, head_rows = 144;
    constexpr w::Shape head_shape{{head_width, head_rows}, 2};
    auto head = bf16_pattern(head_width * head_rows);
    // F32 scale is subnormal with a finite reciprocal. The F16 scale is zero
    // while the incumbent still emits nonzero Q8_0 codes.
    std::fill_n(head.begin() + 32, 32, std::uint16_t{0x03c0});
    const auto head_site = add_site(plan, "output.weight", bf16, head_shape, wire_bytes(std::span(head)));
    // Host twin of the device-only core::convert_bf16_q8_0.
    const auto q8_head = onfly::bf16_to_q8_0_host(head.data(), head_rows, head_width);
    require(q8_head[34] == 0 && q8_head[35] == 0 && q8_head[36] != 0, "reference Q8_0 zero-scale block");

    add_stored(plan, key_site);
    plan.images.push_back({"blk.0.attn_k_b.transposed", {q8, q8, transposed.size(), "q8.transpose.v1"},
                           {key_site}, w::LoadConversion::q8_transpose});
    add_stored(plan, head_site);
    plan.images.push_back({"output.q8", {bf16, q8, q8_head.size(), "hy4-head-bf16-q8_0-v1"},
                           {head_site}, w::LoadConversion::bf16_to_q8});

    CpuSink sink;
    w::LoadReceipt receipt;
    const auto status = w::load(plan, sink, receipt);
    require(static_cast<bool>(status), status.detail);
    require(receipt.images.size() == 4 && receipt.folds.empty(), "Hy4 image/fold count");
    match(plan, sink, receipt, 0, key_shape, wire_bytes(std::span(key)));
    match(plan, sink, receipt, 1, {{rows, cols, batches}, 3}, wire_bytes(std::span(transposed)));
    match(plan, sink, receipt, 2, head_shape, wire_bytes(std::span(head)));
    match(plan, sink, receipt, 3, head_shape, wire_bytes(std::span(q8_head)));
    std::puts("PASS Hy4 stored, Q8 transpose and Q8 head images (CRC32C + memcmp)");
}

void folds_and_lifetimes() {
    w::LoadPlan plan;
    const float a[] = {-1.0F, -0.25F}, b[] = {0.125F, 1.0F};
    const auto first = add_site(plan, "a", f32, {{2, 1}, 2}, wire_bytes(std::span(a)));
    const auto second = add_site(plan, "b", f32, {{2, 1}, 2}, wire_bytes(std::span(b)));
    plan.images.push_back({"fused", {f32, f32, 16, "ordered.f32.v1"}, {first, second}});
    plan.folds = {{0, w::LoadFoldKind::concatenate}, {0, w::LoadFoldKind::one_plus}, {0, w::LoadFoldKind::scale, 2.0F}};
    const auto deferred = std::make_shared<UnreadableSource>(sizeof(a));
    const auto deferred_site = plan.sites.size();
    plan.sites.push_back({999, "runtime-source", f32, {{2}, 1}, {deferred, {0, sizeof(a)}}});
    plan.images.push_back({"session", {f32, q8, 34, "runtime.v1", w::ImageLifetime::session, "profile.fast"},
                           {deferred_site}, w::LoadConversion::runtime});
    plan.images.push_back({"request", {f32, f32, 8, "runtime.v1", w::ImageLifetime::request, "draft.subset"},
                           {deferred_site}, w::LoadConversion::runtime});

    CpuSink sink;
    w::LoadReceipt receipt;
    const auto status = w::load(plan, sink, receipt);
    require(static_cast<bool>(status), status.detail);
    require(receipt.images.size() == 1 && sink.stage_calls == 1 && deferred->reads == 0, "load lifetime only");
    const float joined[] = {a[0], a[1], b[0], b[1]};
    const float plus_one[] = {0.0F, 0.75F, 1.125F, 2.0F};
    const float scaled[] = {0.0F, 1.5F, 2.25F, 4.0F};
    match(plan, sink, receipt, 0, {{2, 2}, 2}, wire_bytes(std::span(scaled)));
    const auto fold_frames = {wire_bytes(std::span(joined)), wire_bytes(std::span(plus_one)), wire_bytes(std::span(scaled))};
    require(receipt.folds.size() == fold_frames.size(), "fold receipt count");
    std::size_t index = 0;
    for (const auto frame : fold_frames) {
        require(receipt.folds[index].fold == index && receipt.folds[index].crc32c == crc32c(frame), "ordered fold CRC32C");
        ++index;
    }
    const auto identity = receipt.plan_identity;
    auto changed = plan;
    std::swap(changed.folds[1], changed.folds[2]);
    require(w::load_plan_identity(changed) != identity, "fold ordering in identity");
    changed = plan;
    changed.folds[2].factor = 3.0F;
    require(w::load_plan_identity(changed) != identity, "fold scalar in identity");
    changed = plan;
    changed.images[2].image.lifetime = w::ImageLifetime::session;
    require(w::load_plan_identity(changed) != identity, "deferred lifetime in identity");
    changed = plan;
    changed.sites[0].payload.source.reset();
    require(w::load_plan_identity(changed) == identity, "source address excluded from identity");
    std::puts("PASS ordered folds, load lifetimes and plan identity");

    const auto check_failed = [&](const w::LoadPlan& attempt, w::Error expected) {
        require(w::load(attempt, sink, receipt).code == expected, "unexpected failure status");
        require(receipt.plan_identity == identity && receipt.images.size() == 1, "receipt changed on failure");
        match(plan, sink, receipt, 0, {{2, 2}, 2}, wire_bytes(std::span(scaled)));
    };
    changed = plan;
    changed.images.back().sites.front() = changed.sites.size();
    check_failed(changed, w::Error::malformed);
    require(sink.stage_calls == 1 && sink.abort_calls == 0 && deferred->reads == 0, "validation before I/O");
    changed = plan;
    add_stored(changed, deferred_site);
    check_failed(changed, w::Error::stale_source);
    require(sink.stage_calls == 2 && sink.abort_calls == 1 && deferred->reads == 1, "abort on read failure");
    sink.reject_stage = true;
    check_failed(plan, w::Error::device_failure);
    sink.reject_stage = false;
    sink.reject_commit = true;
    check_failed(plan, w::Error::device_failure);
    require(sink.abort_calls == 3, "abort on sink failures");
    sink.reject_commit = false;
    require(static_cast<bool>(w::load(plan, sink, receipt)) && sink.resident.size() == 1, "successful retry");
    std::puts("PASS rejected loads preserve published images and receipt");
}

} // namespace

int main() {
    try {
        require(crc32c(std::as_bytes(std::span("123456789", 9))) == 0xe3069283U, "CRC32C reference vector");
        qwen_layer_set();
        hy4_layer_set();
        folds_and_lifetimes();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL load_plan_identity: %s\n", e.what());
        return 1;
    }
}
