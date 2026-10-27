// CPU semantic fixture; no device code, timing claims, or fast-math assumptions.
#include "qsa_select_stream_ref.hpp"
using namespace qsa_select_stream_ref;

std::uint64_t comparisons=0, case_count=0, partial_bytes=0, full_bytes=0;
void run_case(const std::vector<U>& input, U tail, const std::string& label, bool exhaustive,
              std::mt19937& rng) {
    ++case_count;
    Image ref=reference(input,tail);
    same(ref,reference(input,tail,false),label+" n="+std::to_string(input.size())
         +" tail="+std::to_string(tail)+" full-radix"); ++comparisons;
    auto test = [&](U slab, U partition) {
        std::string context=label+" n="+std::to_string(input.size())+" tail="+std::to_string(tail)
            +" slab="+std::to_string(slab)+" partition="+std::to_string(partition);
        auto winners=input.size()>K ? stream(input,0,U(input.size()),slab) : std::vector<Record>{};
        same(ref,emit(input,tail,std::move(winners)),context+" stream"); ++comparisons;
        auto two=two_launch(input,tail,slab,partition);
        same(ref,two.image,context+" two-launch"); ++comparisons;
        // Count actual packed partial records, including fewer-eligible cases.
        std::uint64_t expected=0;
        if (input.size()>K) for (U base=0; base<input.size(); base+=partition) {
            U count=0;
            for (U b=base; b<std::min(U(input.size()),base+partition); ++b) count+=eligible(input[b]);
            expected+=std::min(K,count)*8ULL;
        }
        if (two.bytes!=expected) throw std::runtime_error(context+" byte count mismatch");
        partial_bytes+=two.bytes; full_bytes+=input.size()*4ULL;
    };
    if (exhaustive) for (U slab:slabs) for (U partition:partitions) test(slab,partition);
    else test(slabs[rng()%slabs.size()],partitions[rng()%partitions.size()]);
}


int main() {
    try {
        std::mt19937 rng(0x11c0512U);
        for (U kind=0;kind<labels.size();++kind) {
            auto start=case_count, before=comparisons;
            for (U n:sizes) for (U tail=0;tail<4;++tail)
                run_case(make_case(n,kind,rng),tail,labels[kind],true,rng);
            std::cout<<"PASS "<<labels[kind]<<" cases="<<case_count-start<<" comparisons="<<comparisons-before<<'\n';
        }
        auto start=case_count, before=comparisons;
        for (U i=0;i<2400;++i) {
            U n=i%32==0 ? sizes[rng()%sizes.size()] : U(rng()%4609);
            U kind=U(rng()%labels.size());
            run_case(make_case(n,kind,rng),U(rng()%4),"seeded-random-"+std::to_string(i),false,rng);
        }
        std::cout<<"PASS seeded-random cases="<<case_count-start<<" comparisons="<<comparisons-before<<'\n';
        if (!paths.dense || !paths.few || !paths.capacity || !paths.groups)
            throw std::runtime_error("missing reference branch coverage");
        std::cout<<"PASS reference-paths dense="<<paths.dense<<" few-maxima="<<paths.few
                 <<" capacity-fallback="<<paths.capacity<<" candidate-groups="<<paths.groups<<'\n';
        std::cout<<"two-launch byte balance (all eligible; packed records; one row)\n"
                 <<"blocks partition records partial-write partial-read partial-roundtrip full-score-write delta-write\n";
        for (U n:{8192U,32768U,65536U}) for (U p:partitions) {
            std::vector<U> v(n,0x3f800000U);
            auto result=two_launch(v,0,256,p);
            same(reference(v,0),result.image,"byte-table");
            std::uint64_t bytes=result.bytes, full=n*4ULL;
            std::cout<<n<<' '<<p<<' '<<bytes/8<<' '<<bytes<<' '<<bytes<<' '<<2*bytes<<' '
                     <<full<<' '<<std::int64_t(bytes)-std::int64_t(full)<<'\n';
        }
        std::cout<<"PASS total cases="<<case_count<<" comparisons="<<comparisons
                 <<" checked-partial-write-bytes="<<partial_bytes<<" corresponding-full-score-bytes="<<full_bytes
                 <<" obstacles=none\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"FAIL "<<e.what()<<'\n'; return 1;
    }
}
