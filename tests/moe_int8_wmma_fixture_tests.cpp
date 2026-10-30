#include "moe_int8_wmma_fixture.hpp"
#include "strix/ggml_dequant.hpp"
#include <cfenv>
#include <span>

int main() {
    using namespace i8fixture;
    try {
        require(std::fegetround()==FE_TONEAREST,"fixture needs RNE");
        for(unsigned type:{12U,7U,8U}) {
            const unsigned blocks=type==7?65536:256;
            const auto p=packed(type,blocks);std::vector<float> ref(elements(type)*blocks);
            strix::ggml_dequant::dequantize_row(type,std::as_bytes(std::span(p)),ref);
            for(unsigned k=0;k<ref.size();++k)require(float(weight(p.data(),k,type))==ref[k],"independent packed decode mismatch");
            std::array<float,32> x{};
            for(unsigned b=0;b<256;++b) {
                for(unsigned k=0;k<32;++k)x[k]=float(int((b*17+k*29)%255)-127)/32;
                const auto g=quantize(x.data(),false);int dot32=0,sum=0;double direct=0;
                const auto* w=p.data()+std::size_t(b)*bytes(type);
                for(unsigned k=0;k<32;++k) { dot32+=g.q[k]*coefficient(w,k,type).q;sum+=g.q[k];direct+=g.q[k]*weight(w,k,type); }
                const auto c=coefficient(w,0,type);
                require(sum==g.sum&&std::abs(direct-(c.d*dot32+c.m*sum))<1e-10,"K32 integer/correction mismatch");
            }
        }
        std::array<float,32> x{};require(quantize(x.data(),false).delta==1,"zero group");
        x[0]=0x1p-125F;require(quantize(x.data(),true).sum==0,"tiny down rule");
        x[0]=std::numeric_limits<float>::denorm_min();require(quantize(x.data(),false).status==4,"tiny input delta rejection");
        x[0]=std::numeric_limits<float>::infinity();require(quantize(x.data(),false).status!=0,"nonfinite rejection");
        x[0]=std::numeric_limits<float>::quiet_NaN();require(quantize(x.data(),true).status!=0,"NaN rejection");
        x.fill(1);x[31]=127;const auto outlier=quantize(x.data(),false);require(outlier.q[31]==127&&outlier.q[0]==1,"outlier group");
        x.fill(0);x[0]=127;x[1]=.5F;x[2]=1.5F;x[3]=-.5F;x[4]=-1.5F;
        const auto ties=quantize(x.data(),false);
        require(ties.q[1]==0&&ties.q[2]==2&&ties.q[3]==0&&ties.q[4]==-2,"RNE ties");
        for(unsigned k=0;k<32;++k)x[k]=k%2?127.0F:-127.0F;
        const auto cancel=quantize(x.data(),false);require(cancel.sum==0,"cancelling code sum");
        auto w=packed(12,1);std::fill(w.begin()+16,w.end(),0x77);
        require(qdot(std::vector<Group>{cancel},w.data(),12)==0,"affine cancellation");
        std::puts("INT8 WMMA CPU fixture PASS");return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"fixture: %s\n",e.what());return 1;}
}
