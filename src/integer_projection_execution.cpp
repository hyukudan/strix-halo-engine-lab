#include "strix/integer_projection_execution.hpp"
#include <limits>
#include <stdexcept>

namespace strix::integer_projection {
namespace {
std::uint64_t mul(std::uint64_t a,std::uint64_t b) {
    if(a&&b>std::numeric_limits<std::uint64_t>::max()/a)throw std::overflow_error("projection accounting multiplication overflow");
    return a*b;
}
std::uint64_t add(std::uint64_t a,std::uint64_t b) {
    if(b>std::numeric_limits<std::uint64_t>::max()-a)throw std::overflow_error("projection accounting addition overflow");
    return a+b;
}
std::uint64_t ceil(std::uint64_t a,std::uint64_t b){return a/b+static_cast<std::uint64_t>(a%b!=0);}
std::string phase_name(Phase p){switch(p){case Phase::prefill:return "prefill";case Phase::decode:return "decode";case Phase::verify:return "verify";case Phase::unspecified:break;}throw std::invalid_argument("unknown or unspecified projection phase");}
}
ExecutionPlan ExecutionPlan::build(const ExecutionRequest& r) {
    if(!r.experimental_opt_in||!r.operation_generation)throw std::invalid_argument("explicit experimental opt-in and operation generation required");
    const auto phase=phase_name(r.phase);
    if(r.geometry!=Geometry{}||r.device.index<0||!r.device.instance_generation||r.device.architecture!="gfx1151")
        throw std::invalid_argument("projection requires explicit gfx1151 instance and fixed Wave32 geometry");
    if(!r.maximum_device_bytes||r.maximum_device_bytes>max_working_bytes)
        throw std::invalid_argument("projection requires bounded explicit device admission ceiling");
    validate_shape(r.shape,r.weights);
    const auto numeric=numerical_identity(r.weights,r.activations,r.instruction);
    if(r.approved_numerical_identity!=numeric)throw std::invalid_argument("projection numerical profile is not exactly approved");
    ExecutionPlan plan;plan.request_=r;
    auto& a=plan.accounting_;
    const auto m=static_cast<std::uint64_t>(r.shape.rows),n=static_cast<std::uint64_t>(r.shape.columns),k=static_cast<std::uint64_t>(r.shape.inner),g=k/32;
    const auto req=[&](BufferRole role,std::uint64_t bytes,std::uint64_t alignment){a.buffers[index(role)]={bytes,alignment};a.used_device_bytes=add(a.used_device_bytes,bytes);};
    req(BufferRole::input_f64,mul(mul(m,k),8),8);
    req(BufferRole::packed_weights,mul(n,k/2),1);
    req(BufferRole::weight_scales_f64,mul(mul(n,g),8),8);
    req(BufferRole::activation_codes,mul(m,k),1);
    req(BufferRole::activation_scales_f64,mul(mul(m,g),8),8);
    req(BufferRole::activation_sums_i32,mul(mul(m,g),4),4);
    req(BufferRole::staged_output_f64,mul(mul(m,n),8),8);
    req(BufferRole::destination_f64,mul(mul(m,n),8),8);
    req(BufferRole::integer_groups_i32,mul(mul(mul(m,n),g),4),4);
    req(BufferRole::status_u32,8,4); // flags and observed kernel Wave32.
    a.scratch_bytes=0;
    for(auto role:{BufferRole::activation_codes,BufferRole::activation_scales_f64,BufferRole::activation_sums_i32,
        BufferRole::staged_output_f64,BufferRole::integer_groups_i32,BufferRole::status_u32})
        a.scratch_bytes=add(a.scratch_bytes,a.buffers[index(role)].bytes);
    if(a.used_device_bytes>r.maximum_device_bytes)throw std::length_error("projection exceeds admitted device working set");
    a.source_checkpoint_bytes=mul(mul(n,k/(r.weights==WeightEncoding::q4_0?32U:256U)),r.weights==WeightEncoding::q4_0?18U:136U);
    a.preparation_elements=mul(m,k);a.preparation_groups=mul(m,g);
    a.useful_products=mul(mul(m,n),k);
    const auto gx=ceil(n,16),gy=ceil(m,16);
    if(gx>2147483647U||gy>65535U)throw std::length_error("projection exceeds fixed HIP grid dimensions");
    a.grid_columns=static_cast<std::uint32_t>(gx);a.grid_rows=static_cast<std::uint32_t>(gy);
    a.preparation_blocks=static_cast<std::uint32_t>(ceil(a.preparation_groups,128));
    a.weight_scale_blocks=static_cast<std::uint32_t>(ceil(mul(n,g),128));
    a.issued_products=mul(mul(mul(gx,gy),256),k);
    a.matrix_instructions=mul(mul(gx,gy),k/16);
    plan.identity_="strix-integer-projection-execution-v1:"+phase+":operation="+std::to_string(r.operation_generation)+":"+std::to_string(m)+"x"+std::to_string(n)+"x"+std::to_string(k)+
        ":device="+std::to_string(r.device.index)+":"+std::to_string(r.device.instance_generation)+":"+numeric+
        ":packed-group32:f64-input-output:staged-status-gated-copy";
    return plan;
}
std::uintptr_t buffer_address(const BufferBinding& b) {
    if(b.offset>std::numeric_limits<std::uintptr_t>::max()-b.allocation.base)
        throw std::overflow_error("projection buffer address overflow");
    return b.allocation.base+static_cast<std::uintptr_t>(b.offset);
}
std::uint64_t validate_execution_bindings(const ExecutionPlan& p,const ExecutionBindings& buffers) {
    std::uint64_t actual=0;
    const auto& requirements=p.accounting().buffers;
    for(std::size_t i=0;i<buffer_count;++i) {
        const auto& b=buffers[i];const auto& alloc=b.allocation;const auto req=requirements[i];
        if(!alloc.base||!alloc.bytes||!alloc.identity||!alloc.generation||!alloc.access_generation||alloc.device!=p.request().device||
            alloc.bytes>std::numeric_limits<std::uintptr_t>::max()-alloc.base||b.offset>alloc.bytes||b.bytes>alloc.bytes-b.offset||b.bytes<req.bytes)
            throw std::invalid_argument("projection buffer allocation, instance, generation or extent mismatch");
        const auto address=buffer_address(b);
        const auto role=static_cast<BufferRole>(i);
        if(role!=BufferRole::input_f64&&role!=BufferRole::packed_weights&&role!=BufferRole::weight_scales_f64&&
            alloc.access_generation!=p.request().operation_generation)
            throw std::invalid_argument("projection writable allocation access epoch differs from operation");
        if(address%req.alignment)throw std::invalid_argument("projection buffer alignment mismatch");
        bool seen=false;
        for(std::size_t j=0;j<i;++j) {
            const auto& prior=buffers[j].allocation;
            if(alloc.identity==prior.identity) {
                if(alloc!=prior)throw std::invalid_argument("projection allocation identity reused with different generation/extent/device");
                seen=true;
            } else if(alloc.base<prior.base+prior.bytes&&prior.base<alloc.base+alloc.bytes)
                throw std::invalid_argument("overlapping projection allocations carry different identities");
            const auto other=buffer_address(buffers[j]);
            if(address<other+requirements[j].bytes&&other<address+req.bytes)
                throw std::invalid_argument("projection used buffer ranges alias");
        }
        if(!seen)actual=add(actual,alloc.bytes);
    }
    if(actual>p.request().maximum_device_bytes)throw std::length_error("projection retained allocations exceed admitted full working set");
    return actual;
}
void validate_execution_pins(const ExecutionPlan& p,const ExecutionBindings& b,const AllocationPins& pins) {
    (void)validate_execution_bindings(p,b);
    for(std::size_t i=0;i<buffer_count;++i)
        if(!pins[i]||pins[i]->current_binding()!=b[i].allocation)
            throw std::invalid_argument("projection allocation owner/generation changed or absent");
}
AllocationSnapshots snapshot_allocation_pins(const AllocationPins& pins) {
    AllocationSnapshots result;
    for(std::size_t i=0;i<buffer_count;++i){if(!pins[i])throw std::invalid_argument("missing projection epoch owner");
        result[i]=pins[i]->current_binding();}
    validate_allocation_snapshots(pins,result);return result;
}
void validate_allocation_snapshots(const AllocationPins& pins,const AllocationSnapshots& snapshots) {
    for(std::size_t i=0;i<buffer_count;++i)
        if(!pins[i]||pins[i]->current_binding()!=snapshots[i])
            throw std::invalid_argument("projection allocation/access epoch changed after snapshot");
}
void validate_completed_output(const ExecutionPlan& p,const ExecutionBindings& b,const CompletionEvidence& e) {
    (void)validate_execution_bindings(p,b);
    if(!e.destination_owner||e.destination_owner->current_binding()!=e.destination.allocation||
        e.execution_identity!=p.identity()||e.destination!=b[index(BufferRole::destination_f64)]||
        e.device_status||!e.stream_quiescent||!e.destination_copy_complete||e.destination_poisoned)
        throw std::invalid_argument("projection output lacks matching successful quiescent commit evidence");
}
} // namespace strix::integer_projection
