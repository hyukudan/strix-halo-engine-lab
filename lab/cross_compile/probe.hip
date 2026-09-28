#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy_probe(const float* x, const float* y,
                                        float* out, float alpha,
                                        unsigned int count) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = alpha * x[i] + y[i];
  }
}

