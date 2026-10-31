#pragma once

// g4-dflash-v1.1 on strict g4-v1 device primitives. Weights and FP32 context
// remain resident. Full eight-row blocks are temporary; only committed target
// taps enter the independent 1024-position rings. No target KV is owned here.
#include "dflash.hpp"
#include "gemma4/kernels/ops.hpp"

namespace strix::gemma4::dflash {

struct DeviceTimings {
    double block_ms = 0, head_ms = 0, append_ms = 0, verify_ms = 0, commit_ms = 0;
};

// Immutable resident coefficients/tables, constructed once by the loaded-model
// owner. Target device views must outlive it. Stream creation never reads or
// validates checkpoint bytes. No context, transactions or scratch live here.
class DraftModel {
public:
    DraftModel(const Weights& weights, RopeView rope, LinearView embedding, LinearView head);
    ~DraftModel();
    DraftModel(const DraftModel&) = delete;
    DraftModel& operator=(const DraftModel&) = delete;
    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] std::size_t resident_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class DeviceDrafter;
};

class DeviceDrafter final : public gemma4::DraftContext {
public:
    // embedding/head are caller-owned DEVICE views, shared with the target when
    // available. Their owner must outlive this object. No target cap is inherited.
    DeviceDrafter(const Weights& weights, RopeView rope, LinearView embedding, LinearView head,
                  bool shared_weights = true);
    explicit DeviceDrafter(std::shared_ptr<const DraftModel> model, bool shared_weights = true);
    ~DeviceDrafter() override;
    DeviceDrafter(const DeviceDrafter&) = delete;
    DeviceDrafter& operator=(const DeviceDrafter&) = delete;
    [[nodiscard]] const Shape& shape() const;
    [[nodiscard]] std::uint64_t frontier() const;
    [[nodiscard]] const DeviceTimings& timings() const;
    [[nodiscard]] const DraftModel& model() const;
    [[nodiscard]] bool reusable() const noexcept;
    // Scheduling boundary only; discards pending work, clears context/tags and
    // timings, retains every allocation. A poisoned session cannot be reused.
    void reset();
    // Append 1..8 contiguous committed rows. Initial tail may start after zero.
    void append(std::span<const float> taps, std::span<const std::uint64_t> positions) override;
    // Stage before target commit, publish after it. Failure/discard leaves ring
    // bytes/tags/frontier unchanged. Publication failure poisons this instance.
    void stage_append(std::span<const float> taps, std::span<const std::uint64_t> positions);
    void publish_append();
    void discard_append() noexcept;
    std::array<std::uint32_t, max_drafts> draft(std::uint32_t anchor, std::uint64_t position);
    // Explicit diagnostic entrypoints: host input/output, same device block/head.
    std::vector<float> forward(std::span<const float> embedded, std::uint64_t position);
    std::array<std::uint32_t, max_drafts> select(std::span<const float> normalized);
    std::vector<float> selected_logits() const; // diagnostic, seven rows after draft policy
    std::vector<float> context_key(std::uint32_t layer, std::uint64_t position) const;
    std::vector<float> context_value(std::uint32_t layer, std::uint64_t position) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Uses the same exact verdict as run_cycle. Target candidates remain in its
// transaction overlay. All retained draft context is staged before target
// publication; rejected rows and temporary block KV are discarded. The caller
// discards an uncommitted transaction on an exception (normal KvTxn ownership).
CycleResult run_device_cycle(Target& target, KvTxn& transaction, DeviceDrafter& drafter,
    std::uint32_t anchor, std::uint32_t k, std::size_t budget,
    std::span<const std::uint32_t> eos = {}, Acceptance* stats = nullptr,
    DeviceTimings* timings = nullptr, std::uint64_t stream = 0, std::uint64_t generation = 0);

} // namespace strix::gemma4::dflash
