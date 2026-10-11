#include "strix/distributed_remote_control.hpp"
#include <iostream>
#include <vector>
namespace {
int failures=0;
void expect(bool c,const char* m){if(!c){std::cerr<<"FAIL: "<<m<<'\n';++failures;}}
class Wire final: public strix::DistributedControlWireBackend{
public:
    strix::DistributedRemoteStepAck execute_remote_step(
        const strix::DistributedRemoteStepCommand& c) override{
        commands.push_back(c);
        if(auto_ack){
            return {c.protocol_version,c.placement_plan_identity,c.request_id,c.generation,
                    c.sequence,c.step_id,c.target_node,true};
        }
        return next_ack;
    }
    void publish_boundary_transfer(
        const strix::DistributedExecutionBoundary&,
        const strix::DistributedBoundaryTransfer& t) override{published.push_back(t);}
    strix::DistributedBoundaryTransfer await_boundary_transfer(
        const strix::DistributedExecutionBoundary&) override{return inbound;}
    bool auto_ack=true;
    strix::DistributedRemoteStepAck next_ack;
    strix::DistributedBoundaryTransfer inbound;
    std::vector<strix::DistributedRemoteStepCommand> commands;
    std::vector<strix::DistributedBoundaryTransfer> published;
};
strix::DistributedPlacementPlan placement(){
    strix::DistributedPlacementPlan p; p.local_node=0; p.plan_identity=0xabc; p.executable=true;
    p.steps={{0,strix::SerialOp::token_embedding,0,false,false},
             {1,strix::SerialOp::hyper_expand,1,true,false},
             {2,strix::SerialOp::lm_head,0,false,false}};
    strix::DistributedExecutionBoundary a; a.boundary_id=0x1001; a.source_step=0; a.destination_step=1; a.source_node=0; a.destination_node=1; a.payload_bytes=128;
    strix::DistributedExecutionBoundary b; b.boundary_id=0x1002; b.source_step=1; b.destination_step=2; b.source_node=1; b.destination_node=0; b.payload_bytes=128;
    p.boundaries={a,b}; return p;
}
strix::DistributedBoundaryTransfer transfer(const strix::DistributedPlacementPlan& p,
    const strix::DistributedExecutionBoundary& b, strix::transport::LinkDirection d){
    strix::DistributedBoundaryTransfer t; t.flow.flow_id=1; t.flow.direction=d;
    t.descriptor.connection_epoch=9; t.descriptor.model_plan_id=p.plan_identity;
    t.descriptor.request_id=11; t.descriptor.generation=2; t.descriptor.boundary_id=b.boundary_id;
    t.descriptor.sequence=1; t.descriptor.shape_fingerprint=3; t.descriptor.dtype_id=7; t.descriptor.byte_length=b.payload_bytes;
    t.link_plan.kind=strix::transport::PlanKind::single_link; t.link_plan.slices={{0,0,b.payload_bytes}}; return t;
}
}
int main(){
    auto p=placement(); strix::DistributedRuntimeRequest req{9,11,2,7}; Wire w;
    strix::SequencedDistributedNativeControlPlane c(p,req,w);
    c.execute_remote_step(p.steps[1]);
    expect(w.commands.size()==1&&w.commands[0].sequence==1&&c.next_sequence()==2&&!c.failed(),"sequenced control accepts matching ACK");
    const auto out=transfer(p,p.boundaries[0],strix::transport::LinkDirection::outbound);
    c.publish_boundary_transfer(p.boundaries[0],out); expect(w.published.size()==1,"outbound metadata forwarded");
    w.inbound=transfer(p,p.boundaries[1],strix::transport::LinkDirection::inbound);
    const auto in=c.await_boundary_transfer(p.boundaries[1]); expect(in.descriptor.boundary_id==p.boundaries[1].boundary_id,"inbound metadata accepted");
    bool duplicate_rejected=false;
    try {static_cast<void>(c.await_boundary_transfer(p.boundaries[1]));}
    catch(const std::runtime_error&){duplicate_rejected=true;}
    expect(duplicate_rejected&&c.failed(),"duplicate logical message fails before native payload exposure");
    for(int mode=0;mode<3;++mode){
        Wire foreign_wire;
        strix::SequencedDistributedNativeControlPlane foreign(p,req,foreign_wire);
        foreign_wire.inbound=transfer(p,p.boundaries[1],strix::transport::LinkDirection::inbound);
        if(mode==0)++foreign_wire.inbound.descriptor.dtype_id;
        if(mode==1)++foreign_wire.inbound.descriptor.protocol_version;
        if(mode==2)++foreign_wire.inbound.descriptor.sequence;
        bool foreign_rejected=false;
        try{static_cast<void>(foreign.await_boundary_transfer(p.boundaries[1]));}
        catch(const std::runtime_error&){foreign_rejected=true;}
        expect(foreign_rejected&&foreign.failed(),"foreign dtype/protocol/logical sequence fails at control preflight");
    }
    Wire badw; badw.auto_ack=false; badw.next_ack.success=true;
    strix::SequencedDistributedNativeControlPlane bad(p,req,badw);
    bool rejected=false; try{bad.execute_remote_step(p.steps[1]);}catch(const std::runtime_error&){rejected=true;}
    expect(rejected&&bad.failed(),"bad ACK poisons control");
    bool poisoned=false; try{bad.execute_remote_step(p.steps[1]);}catch(const std::logic_error&){poisoned=true;}
    expect(poisoned,"poisoned control rejects later work");
    Wire staleW; strix::SequencedDistributedNativeControlPlane stale(p,req,staleW);
    staleW.inbound=transfer(p,p.boundaries[1],strix::transport::LinkDirection::inbound); ++staleW.inbound.descriptor.generation;
    bool staleRejected=false; try{static_cast<void>(stale.await_boundary_transfer(p.boundaries[1]));}catch(const std::runtime_error&){staleRejected=true;}
    expect(staleRejected&&stale.failed(),"stale inbound metadata poisons control");
    if(failures==0) std::cout<<"distributed remote control tests passed\n";
    return failures==0?0:1;
}
