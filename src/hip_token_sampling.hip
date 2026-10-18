#include "strix/hip_token_sampling.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>
namespace strix {
void read_hip_logits_row(const ExecutionBufferAddress& row,std::uint32_t vocabulary,void* stream,std::vector<float>& out) {
    const auto bytes=static_cast<std::uint64_t>(vocabulary)*sizeof(float);
    if(!row.address || row.bytes<bytes) throw std::invalid_argument("sampled logits row is smaller than the vocabulary");
    out.resize(vocabulary);
    auto check=[](hipError_t e,const char* what){if(e!=hipSuccess) throw std::runtime_error(std::string(what)+": "+hipGetErrorString(e));};
    const auto s=static_cast<hipStream_t>(stream);
    check(hipMemcpyAsync(out.data(),reinterpret_cast<const void*>(row.address),bytes,hipMemcpyDeviceToHost,s),"sampled logits readback");
    check(hipStreamSynchronize(s),"sampled logits readback completion");
}
} // namespace strix
