#include "native_visibility_probe.hpp"
#include "observed_gpu.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace strix::transport;
using namespace strix::transport::visibility;
template<class Function> bool throws(Function function) { try { function(); } catch (...) { return true; } return false; }

void result_contract() {
    Expected e{7, 9, 11, 4097, Stage::raw_rx};
    Result good{7, 7 ^ completion_mask, 9, 11, 3, 4097, 0, ~0ULL};
    assert(passed(good, e));
    for (unsigned field=0; field<8; ++field) {
        auto bad=good;
        switch(field) {
        case 0: ++bad.started;break;case 1: ++bad.finished;break;
        case 2: ++bad.epoch;break;case 3: ++bad.sequence;break;
        case 4: ++bad.stage;break;case 5: --bad.checked;break;
        case 6: ++bad.mismatches;break;case 7: bad.first_bad=0;break;
        }
        assert(!passed(bad,e));
    }
    assert(!passed(Result{},e));
    auto bad=good;bad.mismatches=1;bad.first_bad=4096;
    assert(complete(bad,e)&&!passed(bad,e));
    bad.first_bad=4097;assert(!complete(bad,e));
    bad=good;bad.mismatches=4098;bad.first_bad=0;assert(!complete(bad,e));
}

void native_probe() {
    const std::uint64_t seed=0xdeadbeef, epoch=37, sequence=8;
    NativeProbe probe([] {}, 1);
    assert(fake_hip::event_flags==(hipEventDisableTiming|hipEventReleaseToSystem));
    for(std::size_t bytes: {1U,255U,256U,257U,4095U,4096U,4097U,65535U,262145U,1048577U}) {
        std::vector<std::uint8_t> payload(bytes);
        for(std::size_t i=0;i<bytes;++i)payload[i]=direct_gpu_commissioning_pattern_byte(i,seed,sequence);
        TransportBufferView view{reinterpret_cast<std::uintptr_t>(payload.data()),bytes,2,TransportMemoryDomain::gpu_device,TransportBufferAccess::read_write};
        const auto verify=[&](Stage stage) { return probe.verify(reinterpret_cast<void*>(1),view,seed,epoch,sequence,stage,std::chrono::milliseconds{1000},std::chrono::milliseconds{1},10); };
        const auto readbacks=fake_hip::readback_bytes;
        auto result=verify(Stage::raw_rx);assert(result.passed());
        assert(result.result.checked==bytes&&fake_hip::readback_bytes-readbacks==64);
        payload[bytes/2]^=1;
        result=verify(Stage::negative_control);
        assert(complete(result.result,result.expected)&&!result.passed());
        assert(result.result.mismatches==1&&result.result.first_bad==bytes/2);
        payload[bytes/2]^=1;
        assert(verify(Stage::consumer).passed());
        fake_hip::omit_verify=true;
        assert(!verify(Stage::raw_rx).passed()); // Init/finalize alone cannot pass.
        fake_hip::omit_verify=false;
        auto old=probe.verify(reinterpret_cast<void*>(1),view,seed,epoch,sequence+1,Stage::raw_rx,std::chrono::milliseconds{1000},std::chrono::milliseconds{1},10);
        assert(!old.passed()); // Reused buffer containing an older generation.
    }
    assert(probe.release());assert(fake_hip::allocations.empty()&&!fake_hip::events);
}

void failure_retention() {
    for(int allocation_failure=1;allocation_failure<=3;++allocation_failure) {
        fake_hip::fail_next=allocation_failure;
        assert(throws([] { NativeProbe probe([] {},1); }));
        assert(fake_hip::allocations.empty()&&!fake_hip::events);
    }
    NativeProbe probe([] {},1);
    std::uint8_t value=direct_gpu_commissioning_pattern_byte(0,0,1);
    TransportBufferView view{reinterpret_cast<std::uintptr_t>(&value),1,2,TransportMemoryDomain::gpu_device,TransportBufferAccess::read_only};
    auto verify=[&] { return probe.verify(reinterpret_cast<void*>(1),view,0,1,1,Stage::raw_rx,std::chrono::milliseconds{100},std::chrono::milliseconds{1},2); };
    fake_hip::query_status=hipErrorNotReady;
    assert(throws(verify));assert(!probe.release());assert(!fake_hip::allocations.empty());
    assert(throws(verify)); // An expired operation did not relinquish resources.
    fake_hip::query_status=hipSuccess;
    probe.mark_stream_quiescent(); // Explicit fake whole-stream quiescence.
    assert(verify().passed());
    fake_hip::fail_next=5; // Event-record failure after the launches and D2H.
    assert(throws(verify));assert(!probe.release());
    probe.mark_stream_quiescent();assert(probe.release());
    assert(fake_hip::allocations.empty()&&!fake_hip::events);
}

struct Delegate : DirectGpuStreamGpu {
    DirectGpuExportBinding binding;
    int current_calls=0,reads=0,stops=0;bool fail_current=false;
    std::vector<int> order;
    const DirectGpuExportBinding& exported_binding() const noexcept override{return binding;}
    void require_current() override{++current_calls;order.push_back(1);if(fail_current)throw std::runtime_error("stale");}
    std::shared_ptr<DirectGpuExportImportLease> import_lease() const noexcept override{return {};}
    void bind_import(const DirectDriverImportBinding&) override{}
    void stage_outbound_segments(const TransferDescriptor&,std::span<const TransportBufferView>,const TransportBufferView&) override{}
    void prepare(const DirectDriverOperation&) override{}
    DirectDriverVisibilityProof publish_terminal(const DirectDriverOperation&,const TransportBufferView&,const GpuCommitRecord&) override{return {};}
    DirectDriverCommitRead read_terminal(const DirectDriverOperation&,const TransportBufferView&) override{++reads;order.push_back(3);return {};}
    TransportBufferView acquired_receive_view(const DirectDriverOperation&) override{return {};}
    void consume_received_segments(const DirectDriverOperation&,std::span<const TransportBufferView>) override{}
    DirectDriverVisibilityProof consumer_completion(const DirectDriverOperation&) override{return {};}
    void retire(const DirectDriverOperation&) override{}
    void request_stop() noexcept override{++stops;}
    DirectDriverQuiescenceStatus poll_quiescence() noexcept override{return DirectDriverQuiescenceStatus::pending;}
    DirectPoolReleaseStatus close() noexcept override{return DirectPoolReleaseStatus::failed_retained;}
};
void observer_order() {
    auto delegate=std::make_shared<Delegate>();
    bool fail=false;
    ObservedCommissioningGpu gpu(delegate,[&](const DirectDriverOperation&) {
        delegate->order.push_back(2);if(fail)throw std::runtime_error("mismatch");
    });
    DirectDriverOperation rx;rx.receive_path=true;
    gpu.read_terminal(rx,{});
    assert((delegate->order==std::vector<int>{1,2,1,3}));
    assert(gpu.close()==DirectPoolReleaseStatus::failed_retained);
    fail=true;assert(throws([&]{gpu.read_terminal(rx,{});}));
    assert(delegate->reads==1&&delegate->stops==1); // No acquire/ACK on mismatch.
    delegate->fail_current=true;delegate->order.clear();
    assert(throws([&]{gpu.read_terminal(rx,{});}));
    assert((delegate->order==std::vector<int>{1}));
    DirectDriverOperation tx;tx.receive_path=false;
    assert(throws([&]{gpu.read_terminal(tx,{});}));
    assert(delegate->reads==1&&delegate->stops==3);
}

int main(int argc,char** argv) {
    assert(argc==2);
    const std::string name=argv[1];
    if(name=="contract")result_contract();
    else if(name=="probe")native_probe();
    else if(name=="retention")failure_retention();
    else if(name=="observer")observer_order();
    else return 2;
    std::cout<<"PASS "<<name<<'\n';
}
