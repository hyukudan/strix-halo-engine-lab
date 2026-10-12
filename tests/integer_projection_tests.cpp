#include "strix/integer_projection.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ip=strix::integer_projection;
namespace lp=strix::low_precision;
namespace {
int failures=0;
void expect(bool ok,const char* msg) { if(!ok){++failures;std::cerr<<msg<<'\n';} }
template<class E,class F> void rejects(F f,const char* msg) {
    try {f();expect(false,msg);} catch(const E&){} catch(...){expect(false,"wrong exception");}
}
constexpr std::array<int,16> book{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
std::vector<std::byte> fixture(ip::WeightEncoding e,std::size_t n,std::size_t k) {
    const auto block=e==ip::WeightEncoding::q4_0?32U:256U;
    const auto bytes=e==ip::WeightEncoding::q4_0?18U:136U;
    std::vector<std::byte> out(n*(k/block)*bytes);
    for(std::size_t b=0;b<n*(k/block);++b) {
        auto* p=out.data()+b*bytes;
        // Exactly representable +/- 0.5, varies across blocks/columns.
        p[1]=std::byte{static_cast<unsigned char>(b%2?0xb8:0x38)};
        const auto base=e==ip::WeightEncoding::q4_0?2U:8U;
        if(e==ip::WeightEncoding::iq4_xs) {
            unsigned high=0;
            for(unsigned g=0;g<8;++g) {
                const unsigned scale=(7*g+static_cast<unsigned>(b))%64;
                high|=(scale>>4U)<<(2U*g);
                p[4+g/2]|=std::byte{static_cast<unsigned char>((scale&15U)<<(4U*(g%2)))};
            }
            p[2]=std::byte{static_cast<unsigned char>(high&255U)};
            p[3]=std::byte{static_cast<unsigned char>(high>>8U)};
        }
        for(unsigned j=0;j<block/2;++j)
            p[base+j]=std::byte{static_cast<unsigned char>(((j+b)%16)|(((15-j+b)%16)<<4U))};
    }
    return out;
}
void shapes_and_math() {
    for(auto e:{ip::WeightEncoding::q4_0,ip::WeightEncoding::iq4_xs})
    for(auto ae:{lp::ActivationEncoding::symmetric_a8,lp::ActivationEncoding::symmetric_a4})
    for(std::size_t k:{std::size_t{256},std::size_t{2560},std::size_t{640}}) {
        if(e==ip::WeightEncoding::iq4_xs&&k==640)continue;
        const ip::Shape s{3,19,k};
        auto bytes=fixture(e,s.columns,k);
        auto w=ip::stage_weights(e,s.columns,k,bytes);
        std::vector<double> x(s.rows*k);
        for(std::size_t r=0;r<s.rows;++r)for(std::size_t j=0;j<k;++j)
            x[r*k+j]=r==0?0.0:static_cast<double>(static_cast<int>((j+3*r)%23)-11)/4.0;
        const auto a=ip::prepare(s,ae,x);
        const auto instruction=ae==lp::ActivationEncoding::symmetric_a4&&e==ip::WeightEncoding::q4_0?
            ip::Instruction::iu4:ip::Instruction::iu8;
        const auto result=ip::reference(w,a,instruction);
        for(std::size_t r=0;r<s.rows;++r) {
            const auto solo=ip::prepare({1,s.columns,k},ae,std::span<const double>(x).subspan(r*k,k));
            const auto alone=ip::reference(w,solo,instruction);
            expect(std::equal(alone.output.begin(),alone.output.end(),result.output.begin()+static_cast<std::ptrdiff_t>(r*s.columns)),"row alone equals batched output exactly");
            for(std::size_t n=0;n<s.columns;++n) {
                double direct=0;
                for(std::size_t g=0;g<k/32;++g) {
                    std::int64_t dot=0;
                    for(std::size_t j=0;j<32;++j) {
                        const auto block=e==ip::WeightEncoding::q4_0?32U:256U;
                        const auto bs=e==ip::WeightEncoding::q4_0?18U:136U;
                        const auto off=(n*(k/block)+(g*32)/block)*bs;
                        const auto sub=g%(block/32);
                        const auto qoff=off+(e==ip::WeightEncoding::q4_0?2:8+sub*16)+j%16;
                        const auto packed=std::to_integer<unsigned>(bytes[qoff]);
                        const auto code=j<16?packed&15U:packed>>4U;
                        const int wc=e==ip::WeightEncoding::q4_0?static_cast<int>(code)-8:book[code];
                        const unsigned ac=a.codes[r*k+g*32+j];
                        const int q=ae==lp::ActivationEncoding::symmetric_a8?
                            (ac<128?static_cast<int>(ac):static_cast<int>(ac)-256):
                            (ac<8?static_cast<int>(ac):static_cast<int>(ac)-16);
                        dot+=wc*q;
                        if(e==ip::WeightEncoding::iq4_xs) {
                            const auto b=std::span<const std::byte,136>(bytes.data()+off,136);
                            expect(w.scales[n*(k/32)+g]*book[code]==strix::iq4_xs::dequantized_value(b,sub*32+j),"IQ4_XS all groups including negative scales preserve source values");
                        }
                    }
                    expect(result.groups[(r*s.columns+n)*(k/32)+g]==dot,"group dot equals independent code mathematics");
                    direct+=(w.scales[n*(k/32)+g]*a.scales[r*(k/32)+g])*static_cast<double>(dot);
                }
                expect(result.output[r*s.columns+n]==direct,"ordered scales equal direct reference");
            }
        }
        // Exercise both mirrored lane halves, signs, 16-wide column tails and
        // absent activation rows. Check RAW products, before affine correction.
        for(std::size_t nb=0;nb<s.columns;nb+=16)for(std::size_t kb=0;kb<k;kb+=16) {
            const auto tile=ip::fragment_panel(w,a,instruction,0,nb,kb);
            for(std::size_t r=0;r<16;++r)for(std::size_t n=0;n<16;++n) {
                int direct=0;
                if(r<s.rows&&nb+n<s.columns)for(std::size_t j=0;j<16;++j)
                    direct+=ip::weight_code(w,nb+n,kb+j)*ip::activation_code(a,r,kb+j);
                expect(tile[r*16+n]==direct,"packed fragment ABI signs/mirrors and exact zero padding");
            }
        }
    }
}
void negative_cases() {
    rejects<std::invalid_argument>([]{ip::validate_shape({1,2,640},ip::WeightEncoding::iq4_xs);},"K640 IQ4_XS rejected");
    rejects<std::invalid_argument>([]{ip::validate_shape({0,2,32},ip::WeightEncoding::q4_0);},"zero rows rejected");
    rejects<std::overflow_error>([]{ip::validate_shape({UINT32_MAX,UINT32_MAX,UINT32_MAX-31U},ip::WeightEncoding::q4_0);},"shape overflow rejected");
    rejects<std::length_error>([]{ip::validate_shape({256,65536,2560},ip::WeightEncoding::q4_0);},"bounded working set enforced before allocation");
    auto bytes=fixture(ip::WeightEncoding::q4_0,1,32);
    rejects<std::invalid_argument>([&]{(void)ip::stage_weights(ip::WeightEncoding::q4_0,1,64,bytes);},"short storage rejected");
    auto w=ip::stage_weights(ip::WeightEncoding::q4_0,1,32,bytes);
    std::vector<double> x(32,1.0);
    auto a=ip::prepare({1,1,32},lp::ActivationEncoding::symmetric_a4,x);
    const int prior=std::fegetround();
    if(std::fesetround(FE_DOWNWARD)==0) {
        rejects<std::invalid_argument>([&]{(void)ip::stage_weights(ip::WeightEncoding::q4_0,1,32,bytes);},"staging rejects altered rounding");
        rejects<std::invalid_argument>([&]{(void)ip::reference(w,a,ip::Instruction::iu4);},"pre-staged reference rejects altered rounding");
        (void)std::fesetround(prior);
    }
    auto invalid_w=w;invalid_w.encoding=static_cast<ip::WeightEncoding>(9);
    rejects<std::invalid_argument>([&]{(void)ip::weight_code(invalid_w,0,0);},"accessor unknown encoding rejected");
    auto invalid_a=a;invalid_a.encoding=static_cast<lp::ActivationEncoding>(9);
    rejects<std::invalid_argument>([&]{(void)ip::activation_code(invalid_a,0,0);},"activation accessor unknown encoding rejected");
    rejects<std::invalid_argument>([&]{(void)ip::reference(w,a,static_cast<ip::Instruction>(9));},"invalid instruction rejected");
    auto bad=a;bad.sums[0]++;
    rejects<std::invalid_argument>([&]{(void)ip::reference(w,bad,ip::Instruction::iu4);},"tampered sum rejected");
    bad=a;bad.codes[0]=128;
    rejects<std::invalid_argument>([&]{(void)ip::reference(w,bad,ip::Instruction::iu4);},"A4 high bits rejected");
    bad=a;bad.scales[0]=0;
    rejects<std::invalid_argument>([&]{(void)ip::reference(w,bad,ip::Instruction::iu4);},"zero scale rejected");
    bad=a;bad.scales[0]=std::numeric_limits<double>::max();
    auto huge=w;huge.scales[0]=std::numeric_limits<double>::max();
    rejects<std::overflow_error>([&]{(void)ip::reference(huge,bad,ip::Instruction::iu4);},"floating scale product overflow rejected");
    x[0]=std::numeric_limits<double>::infinity();
    rejects<std::invalid_argument>([&]{(void)ip::prepare({1,1,32},lp::ActivationEncoding::symmetric_a8,x);},"nonfinite activation rejected");
    x[0]=std::numeric_limits<double>::quiet_NaN();
    rejects<std::invalid_argument>([&]{(void)ip::prepare({1,1,32},lp::ActivationEncoding::symmetric_a8,x);},"NaN activation rejected");
    bytes[0]=std::byte{0};bytes[1]=std::byte{0x7c};
    rejects<std::invalid_argument>([&]{(void)ip::stage_weights(ip::WeightEncoding::q4_0,1,32,bytes);},"nonfinite weight scale rejected");
    const auto iq=ip::stage_weights(ip::WeightEncoding::iq4_xs,1,256,fixture(ip::WeightEncoding::iq4_xs,1,256));
    const auto ia=ip::prepare({1,1,256},lp::ActivationEncoding::symmetric_a4,std::vector<double>(256,1.0));
    rejects<std::invalid_argument>([&]{(void)ip::reference(iq,ia,ip::Instruction::iu4);},"nonlinear raw IU4 rejected");
    auto maximum=iq;std::fill(maximum.packed.begin(),maximum.packed.end(),0);
    std::fill(maximum.scales.begin(),maximum.scales.end(),1.0);
    const auto ma=ip::prepare({1,1,256},lp::ActivationEncoding::symmetric_a8,std::vector<double>(256,-127.0));
    const auto mr=ip::reference(maximum,ma,ip::Instruction::iu8);
    expect(std::all_of(mr.groups.begin(),mr.groups.end(),[](auto dot){return dot==516128;}),"worst representable signed group remains bounded INT32");
    const auto id=ip::numerical_identity(w.encoding,a.encoding,ip::Instruction::iu4);
    expect(id!=ip::numerical_identity(w.encoding,a.encoding,ip::Instruction::iu8),"instruction layout has explicit numerical identity");
    expect(id!=ip::numerical_identity(w.encoding,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8),"activation semantics identity distinct");
}
}
int main(){shapes_and_math();negative_cases();if(failures)return 1;std::cout<<"integer projection arithmetic tests passed\n";}
