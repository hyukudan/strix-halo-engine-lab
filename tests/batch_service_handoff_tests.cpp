#include "strix/batch_service_handoff.hpp"

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
void alternation() {
    strix::BatchServiceHandoff handoff;
    std::atomic<int> owner{0};std::vector<int> order;int value=0;
    thread_local int engine_tls=41;std::thread::id worker_id;
    for(int generation=0;generation<16;++generation) {
        handoff.notify(); // stale completion/park notification from the preceding generation
        handoff.run([&] {
            if(worker_id==std::thread::id{}) worker_id=std::this_thread::get_id();
            require(worker_id==std::this_thread::get_id(),"prefill worker was replaced");
            require(handoff.state()==strix::BatchServiceHandoff::State::running,"worker lost running lease");
            thread_local int worker_tls=7;
            require(worker_tls==7,"worker TLS was lost");
            for(int boundary=0;boundary<48;++boundary) {
                require(owner.exchange(1)==0,"concurrent worker ownership");
                require(value==boundary*2,"resume did not acquire engine writes");
                order.push_back(1);++value;owner=0;
                handoff.park();
                require(worker_tls==7,"park changed worker TLS");
            }
        },[&] {
            require(owner.exchange(2)==0,"concurrent engine ownership");
            require(engine_tls==41,"worker changed engine TLS");
            require(value%2==1,"service did not acquire worker writes");
            order.push_back(2);++value;owner=0;
            handoff.notify();handoff.notify(); // spurious wakeups cannot transfer ownership
            std::this_thread::yield();
            require(handoff.state()==strix::BatchServiceHandoff::State::servicing,"spurious wake resumed worker");
        });
        require(value==96,"missing handoff");value=0;
        require(handoff.state()==strix::BatchServiceHandoff::State::idle,"completed job retained lease");
    }
    for(std::size_t i=0;i<order.size();++i) require(order[i]==(i%2U?2:1),"handoff did not alternate");
}
void lookahead() {
    for(unsigned limit=0;limit<5;++limit) {
        strix::BatchServiceLookahead queue(limit);std::uint64_t gpu_completed=0;
        for(unsigned boundary=0;boundary<100;++boundary) {
            const auto wait=queue.advance();if(wait) gpu_completed=wait;
            require(queue.submitted()-gpu_completed<=limit,"idle lookahead exceeded bound");
        }
        queue.reset();require(queue.submitted()==0,"stale event generation survived reanchor");
        require(queue.advance()==(limit?0U:1U),"new generation used stale event");
    }
}
void cancellation() {
    strix::BatchServiceHandoff handoff;
    for(unsigned cancelled=0;cancelled<48;++cancelled) {
        unsigned partial=0,published=0;
        try {
            handoff.run([&] {
                for(unsigned boundary=0;boundary<48;++boundary) {
                    ++partial;
                    if(boundary==cancelled) throw strix::KnownPrefillCancelled{};
                    handoff.park();
                }
                ++published;
            },[]{});
            throw std::runtime_error("cancellation was swallowed");
        } catch(const strix::KnownPrefillCancelled&) {}
        require(published==0 && partial==cancelled+1U,"cancel published a partial block");
        partial=0; // fake physical reset, followed by reuse of the same owner
        handoff.run([&]{require(partial==0,"reset lost");++published;},[]{});
        require(published==1,"cancel disabled reusable owner");
    }
}
void shutdown() {
    strix::BatchServiceHandoff handoff;
    bool resumed=false;
    try {
        handoff.run([&]{handoff.park();resumed=true;},[&]{handoff.shutdown();});
        throw std::runtime_error("stopped worker returned success");
    } catch(const strix::BatchServiceHandoff::Stopped&) {}
    require(!resumed,"shutdown resumed parked work");
}
void nested_leaf() {
    strix::BatchServiceHandoff handoff;
    const auto engine=std::this_thread::get_id();
    unsigned completed=0,nested=0,finished=0;
    bool queued=true,free=true;
    handoff.run([&] {
        for(unsigned bundle=1;bundle<=3;++bundle) {
            completed=bundle;
            handoff.park();
            require(finished==1,"outer resumed before nested completion");
        }
    },[&] {
        require(std::this_thread::get_id()==engine,"leaf left engine thread");
        if(!queued || !free) return;
        queued=false;free=false; // prepare only: no work until the leaf below
        require(nested==0 && completed==1,"admission executed or skipped the first outer boundary");
        bool recursive=false;
        try {handoff.run([]{},[]{});} catch(const std::logic_error&) {recursive=true;}
        require(recursive,"occupied worker accepted recursive work");
        for(unsigned unit=0;unit<4;++unit) {
            require(completed==1,"outer progressed during leaf execution");
            ++nested;
            if(unit==1) break; // cancellation after a complete unit
        }
        ++finished;free=true;
    });
    require(nested==2 && finished==1,"completed-unit leaf work finished twice");
    strix::BatchServiceStats total,a,b;
    a.nested_admits=1;a.nested_rows=1192;a.nested_ms=4;a.nest_rows_limit=2048;
    b.nest_skips=2;b.nest_skip_reasons["alias"]=2;
    total.add(a);total.add(b);
    require(total.nested_admits==1 && total.nested_rows==1192 && total.nested_ms==4 && total.nest_rows_limit==2048 &&
        total.nest_skips==2 && total.nest_skip_reasons.at("alias")==2,"nested receipts did not aggregate");
}
void failures() {
    strix::BatchServiceHandoff handoff;
    bool failed=false;
    try {handoff.run([]{throw std::runtime_error("fake backend failure");},[]{});}
    catch(const std::runtime_error&) {failed=true;}
    require(failed,"backend error was swallowed");
    bool resumed=false;failed=false;
    try {handoff.run([&]{handoff.park();resumed=true;},[]{throw std::runtime_error("service callback failure");});}
    catch(const std::runtime_error&) {failed=true;}
    require(failed && resumed,"service failure lost resume or exception");
    bool reused=false;handoff.run([&]{reused=true;},[]{});require(reused,"worker failure lost mailbox");
}
}
int main() {
    try {alternation();lookahead();cancellation();shutdown();failures();nested_leaf();}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
    std::cout<<"batch service handoff tests passed\n";
}
