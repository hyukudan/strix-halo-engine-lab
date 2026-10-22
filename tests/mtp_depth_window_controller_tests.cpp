#include "strix/mtp_depth_window_controller.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using C=strix::MtpDepthWindowController;
int failures=0;
void expect(bool ok,const char* msg) {if(!ok){std::cerr<<msg<<'\n';++failures;}}
void teach(C& c,std::size_t owner,std::size_t n,double d,double residual,int repeats=3) {
    const std::array<unsigned,3> rows{2,4,8};
    const std::vector<float> p(n,.99F);
    std::array<std::uint8_t,3> fits{};fits[owner]=1;
    for(int i=0;i<repeats;++i) {
        c.note_draft(static_cast<std::uint32_t>(n),d);
        const auto chosen=c.choose(p,n,fits);
        expect(chosen==owner,"training respects the only fitting owner");
        const auto accepted=std::min<std::size_t>(n,rows[owner]-1U);
        c.calibrate(p,accepted,rows[owner]-1U);
        c.observe(chosen,accepted+1,d+residual);
    }
}
struct Env {
    std::string key,old;bool had;
    static void set(const char* k,const char* v) {
#ifdef _WIN32
        _putenv_s(k,v?v:"");
#else
        if(v)setenv(k,v,1);else unsetenv(k);
#endif
    }
    Env(const char* k,const char* v):key(k),had(std::getenv(k)!=nullptr) {
        if(had)old=std::getenv(k);
        set(k,v);
    }
    ~Env(){set(key.c_str(),had?old.c_str():nullptr);}
};
}
int main() {
    const std::array<std::uint8_t,3> all{1,1,1};
    {
        C c({2,4,8},.7F);
        teach(c,0,1,1,10);teach(c,1,3,3,12);
        teach(c,2,4,4,1); // padding made this R8 sample unusually cheap
        teach(c,2,7,7,100);
        const std::array<float,7> p{.99F,.99F,.99F,.99F,.99F,.99F,.99F};
        c.note_draft(7,7);
        expect(c.choose(p,7,all)==1,"R8 padded cost must not price a full R8 chain");
        bool saw4=false,saw7=false;
        for(const auto& s:c.cost_stats())if(s.rows==8) {
            if(s.filled==4)saw4=s.ms==1;
            if(s.filled==7)saw7=s.ms==100;
        }
        expect(saw4 && saw7,"owner retains independent residual cost cells");
    }
    {
        C c({2,4,8},.7F,{0,0,0,0,0,0,0});
        teach(c,0,1,1,10);teach(c,1,3,3,12);teach(c,2,7,7,15,200);
        const std::array<float,3> p{.99F,.99F,.99F};
        expect(!c.worth_drafting(p),"selected easy R8 outcomes cannot replace the complete-position prior");
    }
    {
        C c({2,4},.8F);
        const std::array<float,3> p{.9F,.55F,.55F};
        const std::array<double,2> ext{.55,.55};
        const std::array<std::uint8_t,2> fits{1,1};
        c.set_external_rows(1,ext);
        expect(c.choose(p,3,fits)==1,"external source floor is not gated again by MTP min-p");
        c.clear_external_rows();
        expect(c.choose(p,3,fits)==0,"external rows cannot leak into the next request");
    }
    {
        C c({2,4,8},.7F);
        const std::array<float,7> p{.99F,.99F,.99F,.99F,.99F,.99F,.99F};
        const std::array<double,6> ext{.9,.9,.9,.9,.9,.9};
        const std::array<std::uint8_t,3> fits{0,0,1};
        for(int i=0;i<3;++i) {
            c.set_external_rows(1,ext);c.note_draft(1,1.25);
            const auto o=c.choose(p,7,fits);c.observe(o,3,20);
        }
        const auto s=c.draft_stats();
        expect(s.size()==1 && s[0].filled==1 && s[0].samples==3 && s[0].ms==1.25,
            "six external rows must not be charged as six MTP draft steps");
    }
    {
        C c({2,4,8},.7F,{.99,.99,.99,.99,.99,.99,.99});
        const std::array<double,3> residual{10,12,15};
        unsigned r8=0,late_r8=0;
        for(unsigned cycle=0;cycle<450;++cycle) {
            std::vector<float> p;
            for(unsigned k=0;k<7;++k){p.push_back(.99F);if(!c.worth_drafting(p))break;}
            const double draft_ms=static_cast<double>(p.size());
            c.note_draft(static_cast<std::uint32_t>(p.size()),draft_ms);
            const auto o=c.choose(p,p.size(),all);
            const auto rows=o==C::serial?1U:(o==0?2U:o==1?4U:8U);
            const auto accepted=o==C::serial?p.size():std::min<std::size_t>(p.size(),rows-1U);
            double wall=o==C::serial?10*static_cast<double>(accepted+1):residual[o];
            if(o==2 && ++r8==2)wall=95; // first sampled R8 is a host stall
            c.calibrate(p,accepted,o==C::serial?accepted+1:rows-1U);
            c.observe(o,accepted+1,wall+draft_ms);
            if(cycle>=350 && o==2)++late_r8;
        }
        std::uint64_t probes=0;for(const auto& s:c.cost_stats())probes+=s.probes;
        expect(late_r8>=95,"a slow first cost sample cannot permanently starve a deeper owner");
        expect(probes>=1 && probes<=8,"re-probes are bounded");
    }
    {
        Env a("STRIX_MTP_CONTROLLER","depth"),b("STRIX_MTP_DOMAIN","prose"),
            d("STRIX_MTP_MINP_PROSE",".83"),e("STRIX_MTP_DEPTH_PRIOR_PROSE",".8,.7,.6,.5,.4,.3,.2");
        expect(std::abs(strix::mtp_depth_domain_min_probability(.7F)-.83F)<1e-6F,"explicit domain threshold");
        expect(strix::mtp_depth_domain_prior()[6]==.2,"domain lookahead prior is separate at every depth");
        {
            Env off("STRIX_MTP_CONTROLLER",nullptr);
            expect(strix::mtp_depth_domain_min_probability(.7F)==.7F,"domain override is opt-in");
        }
        {
            Env bad("STRIX_MTP_DEPTH_PRIOR_PROSE",".8,.7");bool rejected=false;
            try{static_cast<void>(strix::mtp_depth_domain_prior());}catch(const std::invalid_argument&){rejected=true;}
            expect(rejected,"incomplete depth prior is rejected");
        }
    }
    if(failures)return 1;
    std::cout<<"depth controller prototype PASS\n";
}
