#include "strix/hip_mtp_draft_head.hpp"
#include "strix/hip_lm_head_launcher.hpp"
#include <hip/hip_runtime.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix {
namespace {
constexpr unsigned threads=256;
constexpr std::uint32_t max_slots=16;
struct Partial {float value;std::uint32_t token,bad,valid;};
struct Reply {std::uint32_t tokens[max_slots];std::uint32_t status;};
void require(bool ok,const char* message) {if(!ok) throw std::invalid_argument(message);}
void check(hipError_t e,const char* message) {if(e!=hipSuccess) throw std::runtime_error(std::string(message)+": "+hipGetErrorString(e));}
// Same monotone key and lowest-token tie rule as the full greedy selector.
__device__ std::uint32_t numeric_key(float value) {
    auto bits=__float_as_uint(value);
    if((bits&0x7fffffffU)==0) bits=0;
    return (bits&0x80000000U)?~bits:(bits^0x80000000U);
}
__device__ Partial choose(Partial a,const Partial b) {
    const auto bad=a.bad|b.bad;
    if(b.valid && (!a.valid || numeric_key(b.value)>numeric_key(a.value) ||
        (numeric_key(b.value)==numeric_key(a.value) && b.token<a.token))) a=b;
    a.bad=bad;return a;
}
__global__ __launch_bounds__(threads) void draft_gather_rows(const std::uint8_t* head,const std::uint32_t* ids,
    std::uint32_t first,std::uint32_t count,std::uint32_t row_bytes,std::uint8_t* compact) {
    const auto row=first+blockIdx.x;if(blockIdx.x>=count) return;
    const auto* source=head+static_cast<std::uint64_t>(ids[row])*row_bytes;
    auto* target=compact+static_cast<std::uint64_t>(row)*row_bytes;
    for(std::uint32_t i=threadIdx.x;i<row_bytes;i+=threads) target[i]=source[i];
}
__global__ __launch_bounds__(threads) void draft_argmax_partials(const float* logits,const std::uint32_t* ids,
    std::uint32_t rows,Partial* partials) {
    __shared__ Partial values[threads];
    const auto lane=static_cast<unsigned>(threadIdx.x);
    const auto index=static_cast<std::uint64_t>(blockIdx.x)*threads+lane;
    Partial value{0.0F,0,0,0};
    if(index<rows) {
        value.value=logits[index];value.token=ids?ids[index]:static_cast<std::uint32_t>(index);
        value.bad=(__float_as_uint(value.value)&0x7f800000U)==0x7f800000U?1U:0U;value.valid=value.bad==0?1U:0U;
    }
    values[lane]=value;__syncthreads();
    for(unsigned stride=threads/2;stride;stride>>=1U) {
        if(lane<stride) values[lane]=choose(values[lane],values[lane+stride]);
        __syncthreads();
    }
    if(!lane) partials[blockIdx.x]=values[0];
}
__global__ __launch_bounds__(threads) void draft_argmax_finish(const Partial* partials,std::uint32_t count,
    std::uint32_t step,std::uint32_t* slots,Reply* reply) {
    __shared__ Partial values[threads];
    const auto lane=static_cast<unsigned>(threadIdx.x);
    Partial value{0.0F,0,0,0};
    for(std::uint32_t i=lane;i<count;i+=threads) value=choose(value,partials[i]);
    values[lane]=value;__syncthreads();
    for(unsigned stride=threads/2;stride;stride>>=1U) {
        if(lane<stride) values[lane]=choose(values[lane],values[lane+stride]);
        __syncthreads();
    }
    if(!lane) {
        const auto best=values[0];const auto failed=best.bad|(best.valid?0U:2U);
        // A failed row still feeds a valid id forward; the chain reader rejects it.
        const auto token=failed?0U:best.token;
        slots[step]=token;reply->tokens[step]=token;
        if(failed) reply->status|=failed<<(2U*step);
    }
}
} // namespace

struct HipMtpDraftHead::Impl {
    HipMtpDraftHeadConfig config;
    const std::uint8_t* head=nullptr;
    std::uint32_t row_bytes=0,capacity=0,rows=0,base_rows=0,context_rows=0,context_dropped=0;
    std::vector<std::uint8_t> seen;
    std::uint8_t* compact=nullptr;
    std::uint32_t* ids=nullptr;
    std::uint32_t* host_ids=nullptr;
    float* logits=nullptr;
    Partial* partials=nullptr;
    std::uint32_t* slots=nullptr;
    Reply* reply=nullptr;
    Reply* device_reply=nullptr;
    hipStream_t stream=nullptr;
    ~Impl() {
        if(stream) (void)hipStreamSynchronize(stream);
        for(void* p:{static_cast<void*>(compact),static_cast<void*>(ids),static_cast<void*>(logits),
                     static_cast<void*>(partials),static_cast<void*>(slots)}) if(p) (void)hipFree(p);
        if(host_ids) (void)hipHostFree(host_ids);
        if(reply) (void)hipHostFree(reply);
    }
    template<class T> T* device(std::uint64_t count) {
        void* p=nullptr;check(hipMalloc(&p,static_cast<std::size_t>(count*sizeof(T))),"MTP draft head allocation");return static_cast<T*>(p);
    }
    // Appends unseen ids to the append-only pinned list, then gathers the new rows.
    std::uint32_t admit(std::span<const std::uint32_t> tokens,std::uint32_t limit) {
        const auto first=rows;std::uint32_t added=0;
        for(const auto token:tokens) {
            if(token>=config.vocab_size || seen[token]) continue;
            if(added>=limit || rows>=capacity) {++context_dropped;continue;}
            seen[token]=1U;host_ids[rows++]=token;++added;
        }
        if(rows==first) return 0;
        check(hipMemcpyAsync(ids+first,host_ids+first,(rows-first)*sizeof(std::uint32_t),hipMemcpyHostToDevice,stream),"MTP draft ids upload");
        hipLaunchKernelGGL(draft_gather_rows,dim3(rows-first),dim3(threads),0,stream,head,ids,first,rows-first,row_bytes,compact);
        check(hipGetLastError(),"MTP draft row gather");
        return added;
    }
};
HipMtpDraftHead::HipMtpDraftHead(const void* q8_head,HipMtpDraftHeadConfig config,void* stream):impl_(std::make_unique<Impl>()) {
    auto& p=*impl_;
    require(q8_head && stream && config.hidden_size && config.hidden_size%32U==0U && config.hidden_size<=10240U &&
        config.vocab_size && config.max_depth && config.max_depth<=max_slots,"MTP draft head geometry is invalid");
    p.head=static_cast<const std::uint8_t*>(q8_head);p.stream=static_cast<hipStream_t>(stream);
    p.row_bytes=config.hidden_size/32U*34U;
    if(config.full) {
        require(config.base_tokens.empty() && !config.context_capacity,"MTP full draft head takes no vocabulary subset");
        p.capacity=p.rows=config.vocab_size;
    } else {
        const auto capacity=static_cast<std::uint64_t>(config.base_tokens.size())+config.context_capacity;
        require(capacity && capacity<=config.vocab_size,"MTP draft vocabulary subset is empty or exceeds the vocabulary");
        p.capacity=static_cast<std::uint32_t>(capacity);
    }
    p.config=std::move(config);
    try {
        p.logits=p.device<float>(p.capacity);
        p.partials=p.device<Partial>((p.capacity-1U)/threads+1U);
        p.slots=p.device<std::uint32_t>(max_slots);
        check(hipMemsetAsync(p.slots,0,max_slots*sizeof(std::uint32_t),p.stream),"MTP draft slot init");
        check(hipHostMalloc(reinterpret_cast<void**>(&p.reply),sizeof(Reply),hipHostMallocMapped|hipHostMallocCoherent),"MTP draft mapped reply");
        check(hipHostGetDevicePointer(reinterpret_cast<void**>(&p.device_reply),p.reply,0),"MTP draft mapped reply view");
        *p.reply=Reply{};
        if(!p.config.full) {
            p.seen.assign(p.config.vocab_size,0U);
            p.compact=p.device<std::uint8_t>(static_cast<std::uint64_t>(p.capacity)*p.row_bytes);
            p.ids=p.device<std::uint32_t>(p.capacity);
            check(hipHostMalloc(reinterpret_cast<void**>(&p.host_ids),p.capacity*sizeof(std::uint32_t),hipHostMallocDefault),"MTP draft pinned ids");
            p.base_rows=p.admit(p.config.base_tokens,static_cast<std::uint32_t>(p.config.base_tokens.size()));
            p.context_dropped=0;
            require(p.base_rows || p.config.context_capacity,"MTP draft base vocabulary has no valid id");
        }
        check(hipStreamSynchronize(p.stream),"MTP draft head setup");
    } catch(...) {impl_.reset();throw;}
}
HipMtpDraftHead::~HipMtpDraftHead()=default;
void HipMtpDraftHead::observe(std::span<const std::uint32_t> tokens) {
    auto& p=*impl_;if(p.config.full) return;
    p.context_rows+=p.admit(tokens,p.config.context_capacity-p.context_rows);
}
void HipMtpDraftHead::begin_chain() {*impl_->reply=Reply{};}
void HipMtpDraftHead::launch(const std::uint16_t* hidden,std::uint32_t step,void* stream) {
    auto& p=*impl_;
    require(step<p.config.max_depth && stream==p.stream && p.rows,"MTP draft head step/stream is invalid");
    launch_hip_q8_head_rows(hidden,p.config.full?static_cast<const void*>(p.head):p.compact,p.config.hidden_size,p.rows,p.logits,stream);
    const auto blocks=(p.rows-1U)/threads+1U;
    hipLaunchKernelGGL(draft_argmax_partials,dim3(blocks),dim3(threads),0,p.stream,p.logits,p.ids,p.rows,p.partials);
    check(hipGetLastError(),"MTP draft argmax partials");
    hipLaunchKernelGGL(draft_argmax_finish,dim3(1),dim3(threads),0,p.stream,p.partials,blocks,step,p.slots,p.device_reply);
    check(hipGetLastError(),"MTP draft argmax finish");
}
const std::uint32_t* HipMtpDraftHead::device_token(std::uint32_t step) const {
    require(step<impl_->config.max_depth,"MTP draft token slot is invalid");return impl_->slots+step;
}
std::vector<std::uint32_t> HipMtpDraftHead::read(std::uint32_t count) const {
    const auto& p=*impl_;require(count<=p.config.max_depth,"MTP draft read exceeds depth");
    const auto status=__atomic_load_n(&p.reply->status,__ATOMIC_ACQUIRE);
    require(!(status&((count>=16U?0U:(1U<<(2U*count)))-1U)),"MTP draft head selected a non-finite row");
    std::vector<std::uint32_t> tokens(p.reply->tokens,p.reply->tokens+count);
    for(const auto token:tokens) require(token<p.config.vocab_size,"MTP draft head produced out-of-vocabulary token");
    return tokens;
}
bool HipMtpDraftHead::full() const noexcept {return impl_->config.full;}
HipMtpDraftHeadStats HipMtpDraftHead::stats() const noexcept {
    const auto& p=*impl_;return {p.rows,p.base_rows,p.context_rows,p.context_dropped};
}
const float* HipMtpDraftHead::logits() const noexcept {return impl_->logits;}
const std::uint32_t* HipMtpDraftHead::ids() const noexcept {return impl_->ids;}
std::vector<std::uint32_t> HipMtpDraftHead::read_token_file(const std::string& path,std::uint32_t vocabulary) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(in.good(),"MTP draft vocabulary file cannot be opened");
    const auto bytes=static_cast<std::uint64_t>(in.tellg());
    require(bytes && bytes%4U==0U && bytes/4U<=vocabulary,"MTP draft vocabulary file must hold 1..vocab uint32 ids");
    std::vector<std::uint32_t> ids(bytes/4U);in.seekg(0);
    std::vector<unsigned char> raw(bytes);in.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(bytes));
    require(in.good(),"MTP draft vocabulary file read failed");
    for(std::size_t i=0;i<ids.size();++i) {
        ids[i]=0;for(unsigned b=0;b<4U;++b) ids[i]|=static_cast<std::uint32_t>(raw[4*i+b])<<(8U*b);
        require(ids[i]<vocabulary,"MTP draft vocabulary id exceeds the model vocabulary");
    }
    return ids;
}
} // namespace strix
