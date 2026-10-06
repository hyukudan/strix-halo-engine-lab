#include "strix/distributed_runtime.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace {
int failures=0;
void expect(bool c,const char* m){if(!c){std::cerr<<"FAIL: "<<m<<'\n';++failures;}}
strix::transport::DualLinkRuntimeScheduler scheduler(){
    std::vector<strix::transport::DirectionalLinkProfile> links(1);
    links[0].outbound.id="usb4-a"; links[0].inbound.id="usb4-a";
    links[0].outbound.peer_node=1; links[0].inbound.peer_node=1;
    links[0].outbound.health=strix::transport::LinkHealth::qualified;
    links[0].inbound.health=strix::transport::LinkHealth::qualified;
    links[0].outbound.payload_bytes_per_second=1e9;
    links[0].inbound.payload_bytes_per_second=1e9;
    links[0].outbound.one_way_latency_seconds=10e-6;
    links[0].inbound.one_way_latency_seconds=10e-6;
    return strix::transport::DualLinkRuntimeScheduler(
        std::move(links));
}
strix::DistributedPlacementPlan plan(){
    strix::DistributedPlacementPlan p; p.local_node=0; p.plan_identity=0xabc; p.executable=true; p.has_remote_work=true; p.remote_nodes={1};
    p.steps={{0,strix::SerialOp::token_embedding,0,false,false},{1,strix::SerialOp::hyper_expand,1,true,false},{2,strix::SerialOp::final_hyper_mix,0,false,false}};
    strix::DistributedExecutionBoundary a; a.boundary_id=0x1001; a.source_step=0; a.destination_step=1; a.source_node=0; a.destination_node=1; a.payload_bytes=4096;
    strix::DistributedExecutionBoundary b; b.boundary_id=0x1002; b.source_step=1; b.destination_step=2; b.source_node=1; b.destination_node=0; b.payload_bytes=4096;
    p.boundaries={a,b}; return p;
}
class Backend final: public strix::DistributedExecutionBackend{
public:
    explicit Backend(strix::DistributedBoundaryTransfer in): inbound(std::move(in)){}
    void execute_local_step(const strix::DistributedStepPlacement& s) override {events.push_back("l"+std::to_string(s.step_id));}
    void execute_remote_step(const strix::DistributedStepPlacement& s) override {events.push_back("r"+std::to_string(s.step_id));}
    void send_boundary(const strix::DistributedExecutionBoundary&,const strix::DistributedBoundaryTransfer& t) override {sent.push_back(t);events.push_back("s");}
    strix::DistributedBoundaryTransfer receive_boundary(const strix::DistributedExecutionBoundary&) override {events.push_back("v"); if(fail) throw std::runtime_error("x"); return inbound;}
    bool fail=false; std::vector<std::string> events; std::vector<strix::DistributedBoundaryTransfer> sent; strix::DistributedBoundaryTransfer inbound;
};
}
int main(){
    auto p=plan(); auto sched=scheduler();
    strix::DistributedRuntimeRequest req{9,11,2,7};
    auto flow=strix::classify_distributed_boundary_flow(p,p.boundaries[1]);
    strix::DistributedBoundaryTransfer inbound; inbound.flow=flow;
    inbound.descriptor=strix::make_distributed_boundary_descriptor(p,p.boundaries[1],9,11,2,1,7);
    inbound.link_plan=sched.plan(flow.flow_id,flow.direction,p.boundaries[1].payload_bytes,flow.priority);
    Backend backend(inbound);
    strix::DistributedExecutionDecision decision; decision.mode=strix::DistributedExecutionMode::distributed;
    auto state=strix::make_distributed_runtime_state(p,req);
    auto result=strix::execute_distributed_runtime(p,decision,req,state,sched,backend);
    expect(backend.events==std::vector<std::string>({"l0","s","r1","v","l2"}),"runtime preserves boundary order");
    expect(result.local_steps==2&&result.remote_steps==1&&result.sent_boundaries==1&&result.received_boundaries==1&&!state.failed,"runtime accounts work");
    expect(backend.sent.size()==1&&backend.sent[0].descriptor.sequence==1,"sender sequence deterministic");
    auto bad_state=strix::make_distributed_runtime_state(p,req); Backend bad(inbound); bad.fail=true; bool threw=false;
    try{static_cast<void>(strix::execute_distributed_runtime(p,decision,req,bad_state,sched,bad));}catch(const std::runtime_error&){threw=true;}
    expect(threw&&bad_state.failed&&bad_state.receive.failed,"backend failure poisons request");
    if(failures==0) std::cout<<"distributed runtime tests passed\n";
    return failures==0?0:1;
}
