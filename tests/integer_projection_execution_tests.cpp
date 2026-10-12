#include "strix/integer_projection_execution.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ip=strix::integer_projection;
namespace lp=strix::low_precision;
namespace {
int failures=0;
void expect(bool ok,const char* msg){if(!ok){++failures;std::cerr<<msg<<'\n';}}
template<class E,class F>void rejects(F f,const char* msg){try{f();expect(false,msg);}catch(const E&){}catch(...){expect(false,"wrong exception");}}
ip::ExecutionRequest request(ip::WeightEncoding weights=ip::WeightEncoding::q4_0,std::size_t k=640) {
    ip::ExecutionRequest r;r.experimental_opt_in=true;r.phase=ip::Phase::verify;r.operation_generation=1;r.shape={3,19,k};
    r.weights=weights;r.activations=lp::ActivationEncoding::symmetric_a4;
    r.instruction=weights==ip::WeightEncoding::q4_0?ip::Instruction::iu4:ip::Instruction::iu8;
    r.device={0,17,"gfx1151"};r.maximum_device_bytes=ip::max_working_bytes;
    r.approved_numerical_identity=ip::numerical_identity(r.weights,r.activations,r.instruction);return r;
}
ip::ExecutionBindings buffers(const ip::ExecutionPlan& p) {
    ip::ExecutionBindings b;
    for(std::size_t j=0;j<ip::buffer_count;++j) {
        const auto size=p.accounting().buffers[j].bytes;
        b[j]={{0x100000+static_cast<std::uintptr_t>(j)*0x1000000,size+32,j+1,100+j,p.request().device,p.request().operation_generation},16,size};
    }
    return b;
}
struct Pin final:ip::AllocationPin {
    explicit Pin(ip::AllocationBinding a):binding(std::move(a)){}
    ip::AllocationBinding current_binding()const override{return binding;}
    ip::AllocationBinding binding;
};
struct UnstablePin final:ip::AllocationPin {
    explicit UnstablePin(ip::AllocationBinding a,bool throws):binding(std::move(a)),throwing(throws){}
    ip::AllocationBinding current_binding()const override{
        if(throwing)throw std::runtime_error("owner unavailable");
        auto result=binding;++binding.access_generation;return result;
    }
    mutable ip::AllocationBinding binding;bool throwing;
};
ip::AllocationPins pins(const ip::ExecutionBindings& b){ip::AllocationPins p;for(std::size_t j=0;j<ip::buffer_count;++j)p[j]=std::make_shared<Pin>(b[j].allocation);return p;}
void accounting() {
    for(auto weights:{ip::WeightEncoding::q4_0,ip::WeightEncoding::iq4_xs})
    for(std::size_t k:{std::size_t{640},std::size_t{2560}}) {
        if(weights==ip::WeightEncoding::iq4_xs&&k==640){rejects<std::invalid_argument>([&]{(void)ip::ExecutionPlan::build(request(weights,k));},"K640 cannot create IQ4_XS storage blocks");continue;}
        const auto r=request(weights,k);const auto p=ip::ExecutionPlan::build(r);const auto a=p.accounting();
        const std::uint64_t m=3,n=19,g=k/32;
        expect(a.buffers[ip::index(ip::BufferRole::input_f64)].bytes==m*k*8&&
            a.buffers[ip::index(ip::BufferRole::packed_weights)].bytes==n*k/2&&
            a.buffers[ip::index(ip::BufferRole::weight_scales_f64)].bytes==n*g*8,"independent input/packed storage/group scale accounting");
        expect(a.buffers[ip::index(ip::BufferRole::integer_groups_i32)].bytes==m*n*g*4&&
            a.buffers[ip::index(ip::BufferRole::staged_output_f64)].bytes==m*n*8&&
            a.buffers[ip::index(ip::BufferRole::destination_f64)].bytes==m*n*8,"temporary and destination are counted separately");
        const auto scratch=m*k+m*g*8+m*g*4+m*n*8+m*n*g*4+8;
        expect(a.scratch_bytes==scratch&&a.used_device_bytes==scratch+m*k*8+n*k/2+n*g*8+m*n*8,"complete explicit buffer working set from independent formula");
        expect(a.preparation_elements==m*k&&a.preparation_groups==m*g&&a.matrix_instructions==2*(k/16)&&
            a.useful_products==m*n*k&&a.issued_products==16*32*k,"actual M rows independent of padded matrix issue work");
        expect(a.source_checkpoint_bytes==(weights==ip::WeightEncoding::q4_0?n*(k/32)*18:n*(k/256)*136),"source storage blocks do not become WMMA tiles");
        auto b=buffers(p);expect(ip::validate_execution_bindings(p,b)==a.used_device_bytes+32*ip::buffer_count,"retained allocation capacity includes guards and unused bytes");
        ip::validate_execution_pins(p,b,pins(b));
        // One actual arena allocation may provide disjoint buffers. Count its
        // complete retained extent once, including alignment gaps/unused tail.
        std::uint64_t offset=0;for(std::size_t j=0;j<ip::buffer_count;++j){offset=(offset+7)/8*8;b[j].offset=offset;offset+=a.buffers[j].bytes;}
        const ip::AllocationBinding arena{0x1000000,offset+4096,88,77,r.device,r.operation_generation};
        for(auto& view:b)view.allocation=arena;
        expect(ip::validate_execution_bindings(p,b)==arena.bytes,"shared arena extent deduplicates by exact allocation identity");
    }
}
void contract_rejections() {
    const auto r=request();const auto p=ip::ExecutionPlan::build(r);const auto b=buffers(p);
    for(int change=0;change<12;++change) {
        auto bad=r;
        if(change==0)bad.experimental_opt_in=false;
        if(change==1)bad.approved_numerical_identity="float32-control";
        if(change==2)bad.geometry.wave_size=64;
        if(change==3)bad.geometry.tile_inner=32;
        if(change==4)bad.device.instance_generation=0;
        if(change==5)bad.device.architecture="gfx1100";
        if(change==6)bad.phase=static_cast<ip::Phase>(99);
        if(change==7)bad.activations=lp::ActivationEncoding::float32_control;
        if(change==8)bad.instruction=static_cast<ip::Instruction>(99);
        if(change==9)bad.operation_generation=0;
        if(change==10)bad.phase=ip::Phase::unspecified;
        if(change==11)bad.weights=static_cast<ip::WeightEncoding>(99);
        rejects<std::invalid_argument>([&]{(void)ip::ExecutionPlan::build(bad);},"unknown/incompatible phase/profile/geometry/device is rejected");
    }
    auto small=r;small.maximum_device_bytes=p.accounting().used_device_bytes-1;
    rejects<std::length_error>([&]{(void)ip::ExecutionPlan::build(small);},"preparation/full working set must fit admission ceiling");
    auto huge=r;huge.shape={UINT32_MAX,UINT32_MAX,2560};
    rejects<std::overflow_error>([&]{(void)ip::ExecutionPlan::build(huge);},"dimension/product overflow rejects before allocation");
    for(int change=0;change<11;++change) {
        auto bad=b;auto& dest=bad[ip::index(ip::BufferRole::destination_f64)];
        if(change==0)--dest.bytes;
        if(change==1)++dest.offset;
        if(change==2)dest.allocation.generation=0;
        if(change==3)++dest.allocation.device.instance_generation;
        if(change==4)dest.allocation.base=std::numeric_limits<std::uintptr_t>::max()-8;
        if(change==5)dest.offset=dest.allocation.bytes+1;
        if(change==6)dest=bad[ip::index(ip::BufferRole::staged_output_f64)];
        if(change==7)dest.allocation.identity=bad[0].allocation.identity;
        if(change==8)dest.allocation.base=bad[0].allocation.base;
        if(change==9)dest.allocation.bytes=0;
        if(change==10)++dest.allocation.access_generation;
        rejects<std::invalid_argument>([&]{(void)ip::validate_execution_bindings(p,bad);},"extent/alignment/generation/device/alias checks are enforced");
    }
    auto capacity=b;capacity.back().allocation.bytes=r.maximum_device_bytes;
    rejects<std::length_error>([&]{(void)ip::validate_execution_bindings(p,capacity);},"unused retained allocation capacity counts toward ceiling");
    auto owners=pins(b);owners[0].reset();
    rejects<std::invalid_argument>([&]{ip::validate_execution_pins(p,b,owners);},"absent allocation lifetime owner rejected");
    owners=pins(b);auto changed=std::make_shared<Pin>(b[2].allocation);++changed->binding.generation;owners[2]=changed;
    rejects<std::invalid_argument>([&]{ip::validate_execution_pins(p,b,owners);},"live generation drift rejects pinned metadata");
    owners=pins(b);const auto ready_snapshot=ip::snapshot_allocation_pins(owners);
    changed=std::make_shared<Pin>(b[0].allocation);owners[0]=changed;++changed->binding.access_generation;
    rejects<std::invalid_argument>([&]{ip::validate_allocation_snapshots(owners,ready_snapshot);},"old ready snapshot cannot authorize new input access epochs");
    owners=pins(b);owners[0]=std::make_shared<UnstablePin>(b[0].allocation,false);
    rejects<std::invalid_argument>([&]{(void)ip::snapshot_allocation_pins(owners);},"mixed epoch snapshot is rejected by consistent reread");
    owners=pins(b);owners[0]=std::make_shared<UnstablePin>(b[0].allocation,true);
    rejects<std::runtime_error>([&]{ip::validate_allocation_snapshots(owners,ready_snapshot);},"throwing owner cannot produce a valid ready snapshot");
    std::vector<double> input(r.shape.rows*r.shape.inner,1.0);input[0]=std::numeric_limits<double>::quiet_NaN();
    rejects<std::invalid_argument>([&]{(void)ip::prepare(r.shape,r.activations,input);},"host activation oracle rejects nonfinite input before any native request");
}
void completion_gate() {
    const auto r=request();const auto p=ip::ExecutionPlan::build(r);const auto b=buffers(p);
    const ip::CompletionEvidence complete{p.identity(),b[ip::index(ip::BufferRole::destination_f64)],0,true,true,false,
        std::make_shared<Pin>(b[ip::index(ip::BufferRole::destination_f64)].allocation)};
    ip::validate_completed_output(p,b,complete);
    auto stale=complete;auto owner=std::make_shared<Pin>(stale.destination.allocation);stale.destination_owner=owner;
    ++owner->binding.access_generation;
    rejects<std::invalid_argument>([&]{ip::validate_completed_output(p,b,stale);},"old completion cannot consume rewritten destination access epoch");
    for(int change=0;change<8;++change) {
        auto bad=complete;
        if(change==0)bad.device_status=1;
        if(change==1)bad.stream_quiescent=false;
        if(change==2)bad.destination_copy_complete=false;
        if(change==3)bad.destination_poisoned=true;
        if(change==4)++bad.destination.allocation.generation;
        if(change==5)++bad.destination.offset;
        if(change==6)bad.execution_identity+="tampered";
        if(change==7)bad.destination_owner.reset();
        rejects<std::invalid_argument>([&]{ip::validate_completed_output(p,b,bad);},"failure/partial commit/identity drift never authorizes output");
    }
    auto other=r;other.phase=ip::Phase::decode;const auto phase=ip::ExecutionPlan::build(other);
    rejects<std::invalid_argument>([&]{ip::validate_completed_output(phase,b,complete);},"verify completion cannot substitute decode phase");
    other=r;other.activations=lp::ActivationEncoding::symmetric_a8;other.instruction=ip::Instruction::iu8;
    other.approved_numerical_identity=ip::numerical_identity(other.weights,other.activations,other.instruction);
    const auto numeric=ip::ExecutionPlan::build(other);
    rejects<std::invalid_argument>([&]{ip::validate_completed_output(numeric,b,complete);},"A4 result cannot authorize A8 numerical profile");
    std::weak_ptr<Pin> lifetime;
    ip::CompletionEvidence retained=complete;retained.destination_owner.reset();
    {auto final_owner=std::make_shared<Pin>(complete.destination.allocation);lifetime=final_owner;
        retained.destination_owner=final_owner;}
    expect(!lifetime.expired(),"completed output retains destination after input/scratch owner scope ends");
    ip::validate_completed_output(p,b,retained);retained.destination_owner.reset();
    expect(lifetime.expired(),"destination retires only after completed-output owner is released");
}
}
int main(){accounting();contract_rejections();completion_gate();if(failures)return 1;std::cout<<"integer projection execution contract tests passed\n";}
