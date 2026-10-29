// CPU-only: scheduling coverage/capacity and the shared cross-thread clock recorder.
#include "strix/qwen_prefill_schedule.hpp"
#include "strix/qwen_prefill_trace.hpp"
#include <numeric>
#include <iostream>
#include <thread>
using namespace strix::cuda::qwen_prefill;
void require(bool value) { if (!value) throw std::runtime_error("schedule/trace check failed"); }
int main() {
    const std::vector<std::uint32_t> base{1024,2048,4096,8192};
    require(plan_blocks(8191, base) == std::vector<std::uint32_t>({1024,2048,4096,1023}));
    require(plan_blocks(8191, base, true) == std::vector<std::uint32_t>({1023,1024,2048,4096}));
    require(plan_blocks(32767, base, true) == std::vector<std::uint32_t>({1023,1024,2048,4096,8192,8192,8192}));
    require(plan_blocks(8191, {1024,8192}, false, 7167) == std::vector<std::uint32_t>({1024,7167}));
    for (std::uint64_t n=1; n<=65537; ++n) {
        const auto normal = plan_blocks(n, base);
        for (bool tail : {false,true}) for (auto reserve : {0U,1U,1024U,7167U,8192U}) {
            auto p=plan_blocks(n, base, tail, reserve);
            require(p.size()==normal.size());
            require(std::accumulate(p.begin(),p.end(),std::uint64_t{0})==n);
            for (auto r:p) require(r>=1 && r<=8192);
            if (reserve && p.size()>1) {
                const auto other_capacity=std::uint64_t{8192}*(p.size()-1);
                const auto lo=n>other_capacity ? n-other_capacity : 1;
                const auto hi=std::min<std::uint64_t>(8192,n-p.size()+1);
                require(p.back()==std::clamp<std::uint64_t>(reserve,lo,hi));
            }
            if (tail && !reserve) require(p.back()==std::max(normal.back(), normal[normal.size()>1 ? normal.size()-2 : 0]));
        }
    }
    require(plan_blocks(65520, {65520}) == std::vector<std::uint32_t>({65520}));
    for (auto bad : {std::vector<std::uint32_t>{}, {0}, {65521}, {65536}, {65537}}) {
        bool threw=false; try { (void)plan_blocks(1024,bad); } catch(const std::invalid_argument&) { threw=true; }
        require(threw);
    }
    bool threw=false; try { (void)plan_blocks(1,base,false,8193); } catch(const std::invalid_argument&) { threw=true; }
    require(threw);
    strix::qwen_prefill_trace::Trace t;
    std::thread a([&] { for(unsigned i=0;i<1000;++i) t.mark("a",i); });
    std::thread b([&] { for(unsigned i=0;i<1000;++i) t.mark("b",i); });
    a.join(); b.join(); auto events=t.snapshot(); require(events.size()==2000);
    for(const auto& e:events) require(e.ms>=0);
    std::cout << "schedule/trace CPU checks PASS\n";
}
