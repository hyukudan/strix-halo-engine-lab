#pragma once
// CUDA device tower of the shared vision runtime: strix::vision::EncoderBackend over the FP32 kernels of
// src/cuda/core/vision. The step order is the shared plan (src/vision/encoder_plan.hpp); the weights are converted to
// FP32 on the device at load. Arithmetic "vision-cuda-f32-v1" is part of the adapter identity (distinct from the HIP
// "vision-f16-v1" FP16-panel tower and from the host "vision-ref-f32" reference).
#include "strix/vision/encoder.hpp"

#include <cstddef>
#include <memory>

namespace strix::vision {

struct CudaEncoderOptions {
    std::size_t mlp_workspace_bytes = 160U << 20U;  // cap of the chunked MLP buffers
};

// Shapes the CUDA kernels take (head dimension <= 128, even; rotary pairs need an even head dimension).
bool cuda_supported(const EncoderConfig&, std::string* why = nullptr);

class CudaEncoder final : public EncoderBackend {
public:
    static constexpr const char* kArithmetic = "vision-cuda-f32-v1";
    explicit CudaEncoder(EncoderConfig, CudaEncoderOptions = {});
    ~CudaEncoder() override;
    CudaEncoder(const CudaEncoder&) = delete;
    CudaEncoder& operator=(const CudaEncoder&) = delete;

    void load(const EncoderWeights&);  // FP32 copies to the device; the weights need not outlive the call
    void unload();
    bool loaded() const;
    std::size_t device_bytes() const;

    EncoderOutput encode(const PatchGrid&, const PositionInputs&) override;
    // Projector / adapter linears (DenseOps): weights are uploaded once per (weight, bias) identity; merge_project is the shared
    // DenseOps graph over this linear.
    void linear(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) override;
    void prepare_linear(const Tensor& w, const Tensor& bias) override;
    // Drops the cached panel of (w, bias); call before freeing or rewriting the storage it was built from. Panels are keyed by
    // the full weight and bias identity (address, type, shape) and are also cleared by load() and unload().
    bool release_linear(const Tensor& w, const Tensor& bias);
    const char* arithmetic() const override { return kArithmetic; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::vision
