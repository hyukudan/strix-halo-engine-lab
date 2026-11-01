#pragma once

// MTP assistant drafter on g4-v1 device primitives. Per draft depth: 44 launches
//   embedding gather, pre_projection, input norm;
//   per layer x4: q, head norm+RoPE (target sliding/global tables), split-K attention partial + merge over the
//   target KV (GQA group 2), o, attention epilogue (+pre-FFN norm), gate_up (GeGLU), down, FFN epilogue
//   (* layer_scalar, + next norm);
//   post_projection (writes the next depth's backbone half in place), centroids, top-32, 4096 candidate dots, select.
// Every per-call value (p, anchor, hidden row, KV descriptors) lives in device memory, so the k-depth chain is captured
// once per depth segment [from, to) as a HIP graph (STRIX_G4_MTP_GRAPHS=0 disables) and replayed. One H2D batch,
// one sync, one download per draft(); extend() continues the chain with one more segment, sync and download.
#include "mtp.hpp"
#include "gemma4/kernels/ops.hpp"

namespace strix::gemma4::eagle::mtp {

class DeviceModel {
public:
    // target_embedding: DEVICE view of the target token_embd (e.g. DeviceTarget::resident_embedding).
    DeviceModel(const Weights& weights, Tables tables, LinearView target_embedding);
    ~DeviceModel();
    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;
    [[nodiscard]] const Config& config() const;
    [[nodiscard]] std::size_t resident_bytes() const;
    [[nodiscard]] std::size_t step_weight_bytes() const; // packed bytes touched per depth (incl. 4096 head rows)
    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
    friend class DeviceDrafter;
};

// Inputs::sliding/full must hold DEVICE pointers (target KV views of layers 22/23).
class DeviceDrafter final : public Drafter {
public:
    explicit DeviceDrafter(std::shared_ptr<const DeviceModel> model);
    ~DeviceDrafter() override;
    DeviceDrafter(const DeviceDrafter&) = delete;
    DeviceDrafter& operator=(const DeviceDrafter&) = delete;
    [[nodiscard]] const Config& config() const override;
    [[nodiscard]] const Timings& timings() const override;
    Draft draft(const Inputs& inputs, std::uint32_t k) override;
    // Real continuation: the chain state (last token, next depth's backbone half, p, anchor, KV descriptors) stays in
    // device buffers, so depths [depth, k) run as one more (cached graph) segment; one sync + download.
    Draft extend(std::uint32_t k) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strix::gemma4::eagle::mtp
