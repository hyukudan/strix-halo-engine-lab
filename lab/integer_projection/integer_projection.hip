#include "strix/hip_integer_projection.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ip=strix::integer_projection;
namespace lp=strix::low_precision;
namespace {
void check(hipError_t e,const char* op){if(e!=hipSuccess)throw std::runtime_error(std::string(op)+": "+hipGetErrorString(e));}
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
ip::DeviceIdentity device(){return {0,1,"gfx1151"};}
std::uint64_t generation(){static std::atomic<std::uint64_t> counter{1};auto value=counter.load();
    for(;;){require(value&&value!=UINT64_MAX,"fixture generations exhausted");if(counter.compare_exchange_weak(value,value+1))return value;}}
bool same_bits(const std::vector<double>& expected,const std::vector<double>& actual,std::size_t start){
    if(actual.size()<start+expected.size())return false;
    for(std::size_t i=0;i<expected.size();++i)if(std::bit_cast<std::uint64_t>(expected[i])!=std::bit_cast<std::uint64_t>(actual[start+i]))return false;
    return true;
}
template<class T>class Buffer {
    struct Storage final:ip::AllocationPin {
        explicit Storage(std::size_t size):count(size){
            const auto gen=generation();binding={0,count*sizeof(T),gen,gen,device(),gen};
            void* p=nullptr;check(hipMalloc(&p,count*sizeof(T)),"fixture allocate");data=static_cast<T*>(p);
            binding.base=reinterpret_cast<std::uintptr_t>(data);
        }
        ~Storage(){if(data&&hipStreamQuery(hipStreamLegacy)==hipSuccess)(void)hipFree(data);}
        ip::AllocationBinding current_binding()const override{return binding;}
        T* data=nullptr;std::size_t count;ip::AllocationBinding binding;
    };
    std::shared_ptr<Storage> owner;
public:
    explicit Buffer(std::size_t n):owner(std::make_shared<Storage>(n)),data(owner->data),count(n){}
    void put(const std::vector<T>& x){require(x.size()==count,"fixture upload length");owner->binding.access_generation=generation();check(hipMemcpy(data,x.data(),count*sizeof(T),hipMemcpyHostToDevice),"fixture upload");}
    std::vector<T>get()const{std::vector<T>x(count);check(hipMemcpy(x.data(),data,count*sizeof(T),hipMemcpyDeviceToHost),"fixture readback");return x;}
    void fill(T value){put(std::vector<T>(count,value));}
    ip::BufferBinding view(std::size_t offset,std::uint64_t bytes)const{return {owner->binding,offset*sizeof(T),bytes};}
    std::shared_ptr<const ip::AllocationPin>pin()const{return owner;}
    void begin_write(std::uint64_t epoch)const{require(epoch>owner->binding.access_generation,"fixture write epoch must advance");owner->binding.access_generation=epoch;}
    T* data;std::size_t count;
};
ip::ExecutionPlan plan(ip::Shape s,ip::WeightEncoding w,lp::ActivationEncoding a,ip::Instruction i,ip::Phase phase){
    ip::ExecutionRequest r;r.experimental_opt_in=true;r.phase=phase;r.operation_generation=generation();r.shape=s;r.weights=w;r.activations=a;r.instruction=i;r.device=device();
    r.approved_numerical_identity=ip::numerical_identity(w,a,i);r.maximum_device_bytes=ip::max_working_bytes;return ip::ExecutionPlan::build(r);
}
std::vector<std::byte> fixture(ip::WeightEncoding e,std::size_t n,std::size_t k) {
    const std::size_t block=e==ip::WeightEncoding::q4_0?32:256,bytes=e==ip::WeightEncoding::q4_0?18:136;
    std::vector<std::byte> out(n*(k/block)*bytes);
    for(std::size_t b=0;b<n*(k/block);++b) {
        auto* p=out.data()+b*bytes;p[1]=std::byte{static_cast<unsigned char>(b%2?0xb8:0x38)};
        if(e==ip::WeightEncoding::iq4_xs){unsigned high=0;
            for(unsigned g=0;g<8;++g){const unsigned scale=(g*9U+static_cast<unsigned>(b))%64;
                high|=(scale>>4U)<<(2*g);p[4+g/2]|=std::byte{static_cast<unsigned char>((scale&15U)<<(4*(g%2)))};}
            p[2]=std::byte{static_cast<unsigned char>(high&255U)};p[3]=std::byte{static_cast<unsigned char>(high>>8U)};}
        for(std::size_t j=0;j<block/2;++j)p[(e==ip::WeightEncoding::q4_0?2:8)+j]=
            std::byte{static_cast<unsigned char>(((j+b)%16)|(((3*j+7*b)%16)<<4U))};
    }
    return out;
}
enum class FixtureMode { ordinary, zero_weights, nonfinite_weights, overflowing_output };
void run_case(ip::Shape s,ip::WeightEncoding e,lp::ActivationEncoding ae,ip::Instruction instruction,
    ip::Phase phase,std::vector<double> x={},FixtureMode mode=FixtureMode::ordinary,unsigned expected_flags=0) {
    auto execution=plan(s,e,ae,instruction,phase);const auto& accounting=execution.accounting();
    if(x.empty()){x.resize(s.rows*s.inner);for(std::size_t r=0;r<s.rows;++r)for(std::size_t j=0;j<s.inner;++j)
        x[r*s.inner+j]=(r==0&&s.rows>1)?0.0:static_cast<double>(static_cast<int>((j+5*r)%29)-14)/8.0;}
    auto w=ip::stage_weights(e,s.columns,s.inner,fixture(e,s.columns,s.inner));
    if(mode==FixtureMode::zero_weights)std::fill(w.scales.begin(),w.scales.end(),0.0);
    if(mode==FixtureMode::nonfinite_weights)w.scales[0]=std::numeric_limits<double>::quiet_NaN();
    if(mode==FixtureMode::overflowing_output){std::fill(w.scales.begin(),w.scales.end(),0.5);std::fill(w.packed.begin(),w.packed.end(),0xff);}
    ip::Activations a;ip::Result ref;
    if(!expected_flags){a=ip::prepare(s,ae,x);ref=ip::reference(w,a,instruction);}
    const auto groups=s.inner/32;
    Buffer<double> input(x.size()),ws(w.scales.size()),as(s.rows*groups+2),stage(s.rows*s.columns+2),out(s.rows*s.columns+2);
    Buffer<std::uint8_t> ac(s.rows*s.inner+2),wc(w.packed.size());
    Buffer<std::int32_t> sums(s.rows*groups+2),ig(s.rows*s.columns*groups+2);Buffer<unsigned>status(2);
    input.put(x);ws.put(w.scales);wc.put(w.packed);status.fill(0);
    const auto reset=[&]{as.fill(12345.0);stage.fill(12345.0);out.fill(12345.0);ac.fill(0xa5);sums.fill(1234567);ig.fill(1234567);};reset();
    execution=plan(s,e,ae,instruction,phase);
    const auto bind=[&](const ip::ExecutionPlan& p){
        ip::HipProjectionInputs b;
        const auto put=[&](ip::BufferRole role,const auto& buffer,std::size_t offset){const auto j=ip::index(role);
            if(role!=ip::BufferRole::input_f64&&role!=ip::BufferRole::packed_weights&&role!=ip::BufferRole::weight_scales_f64)
                buffer.begin_write(p.request().operation_generation);
            b.buffers[j]=buffer.view(offset,p.accounting().buffers[j].bytes);b.pins[j]=buffer.pin();};
        put(ip::BufferRole::input_f64,input,0);put(ip::BufferRole::packed_weights,wc,0);put(ip::BufferRole::weight_scales_f64,ws,0);
        put(ip::BufferRole::activation_codes,ac,1);put(ip::BufferRole::activation_scales_f64,as,1);put(ip::BufferRole::activation_sums_i32,sums,1);
        put(ip::BufferRole::staged_output_f64,stage,1);put(ip::BufferRole::destination_f64,out,1);put(ip::BufferRole::integer_groups_i32,ig,1);put(ip::BufferRole::status_u32,status,0);
        b.producer_ready=ip::HipProjectionReadyEvent::record(device(),hipStreamLegacy,b.pins);return b;
    };
    ip::HipIntegerProjectionLauncher launcher(device(),hipStreamLegacy);
    auto bound=bind(execution);const auto result=launcher.execute(execution,bound);
    if(expected_flags){
        require(result.status==ip::HipProjectionStatus::failed&&result.evidence.stream_quiescent&&
            !result.evidence.destination_copy_complete&&!result.evidence.destination_poisoned&&!launcher.retains_resources(),"negative control lacks failed/quiescent/uncommitted proof");
        require((result.evidence.device_status&expected_flags)==expected_flags,"negative control did not report expected device flags");
        const auto untouched=out.get();require(std::all_of(untouched.begin(),untouched.end(),[](double v){return v==12345;}),"failed projection modified destination");
        if(expected_flags&3U){const auto rejected_codes=ac.get();require(std::all_of(rejected_codes.begin()+1,rejected_codes.begin()+33,
            [](auto code){return code==0xa5;}),"nonfinite/underflow preparation group wrote codes");}
        require(launcher.execute(execution,bound).status==ip::HipProjectionStatus::rejected,"failed launcher admitted another operation");
        std::cout<<"PASS negative flags="<<result.evidence.device_status<<'\n';return;
    }
    require(result.status==ip::HipProjectionStatus::complete,"shared launcher failed: see status");
    ip::validate_completed_output(execution,bound.buffers,result.evidence);
    require(launcher.execute(execution,bound).status==ip::HipProjectionStatus::rejected,"completed operation identity was replayed");
    require(result.actual_device_bytes>=accounting.used_device_bytes&&!launcher.retains_resources(),"complete working set/completion lifetime proof missing");
    const auto dc=ac.get();const auto ds=as.get();const auto dsum=sums.get();const auto got=out.get();const auto integers=ig.get();const auto staged=stage.get();
    require(dc.front()==0xa5&&dc.back()==0xa5&&ds.front()==12345&&ds.back()==12345&&dsum.front()==1234567&&dsum.back()==1234567&&
        got.front()==12345&&got.back()==12345&&integers.front()==1234567&&integers.back()==1234567&&staged.front()==12345&&staged.back()==12345,"shared launcher physical guard overwritten");
    require(std::equal(a.codes.begin(),a.codes.end(),dc.begin()+1)&&same_bits(a.scales,ds,1)&&std::equal(a.sums.begin(),a.sums.end(),dsum.begin()+1),"shared launcher preparation differs from independent CPU oracle");
    require(std::equal(ref.groups.begin(),ref.groups.end(),integers.begin()+1)&&same_bits(ref.output,got,1)&&same_bits(ref.output,staged,1),"shared WMMA integer groups/scale/output bits differ from direct CPU oracle");
    for(std::size_t r=0;r<s.rows;++r){
        std::vector<double> single(x.size(),0);std::copy_n(x.begin()+static_cast<std::ptrdiff_t>(r*s.inner),s.inner,single.begin());input.put(single);reset();
        const auto solo=plan({1,s.columns,s.inner},e,ae,instruction,phase);auto solo_bound=bind(solo);
        auto stale_ready=solo_bound;stale_ready.producer_ready=bound.producer_ready;
        require(launcher.execute(solo,stale_ready).status==ip::HipProjectionStatus::rejected,"old ready event authorized new allocation access epochs");
        const auto solo_result=launcher.execute(solo,solo_bound);require(solo_result.status==ip::HipProjectionStatus::complete,"native solo launcher failed");
        ip::validate_completed_output(solo,solo_bound.buffers,solo_result.evidence);
        const auto qa=ip::prepare({1,s.columns,s.inner},ae,std::span<const double>(x).subspan(r*s.inner,s.inner));const auto qr=ip::reference(w,qa,instruction);
        const auto solo_codes=ac.get();const auto solo_scales=as.get();const auto solo_sums=sums.get();const auto solo_out=out.get();const auto solo_groups=ig.get();
        require(std::equal(qa.codes.begin(),qa.codes.end(),solo_codes.begin()+1)&&same_bits(qa.scales,solo_scales,1)&&std::equal(qa.sums.begin(),qa.sums.end(),solo_sums.begin()+1),"native solo preparation differs");
        require(same_bits(qr.output,solo_out,1)&&same_bits(qr.output,got,1+r*s.columns)&&std::equal(qr.groups.begin(),qr.groups.end(),solo_groups.begin()+1),"native solo/batched output or integer groups differ");
        require(solo_codes[1+s.inner]==0xa5&&solo_scales[1+groups]==12345&&solo_sums[1+groups]==1234567&&
            solo_out[1+s.columns]==12345&&solo_groups[1+qr.groups.size()]==1234567,"solo logical extent guard overwritten");
        bool stale_rejected=false;try{ip::validate_completed_output(execution,bound.buffers,result.evidence);}catch(const std::invalid_argument&){stale_rejected=true;}
        require(stale_rejected,"rewritten destination still authorized old completed output");
    }
    std::cout<<"PASS "<<execution.identity()<<'\n';
}
void controls(){
    for(double bad:{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::denorm_min()}){
        std::vector<double>x(32,0);x[0]=bad;
        run_case({1,1,32},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8,ip::Phase::decode,x,
            FixtureMode::ordinary,std::isfinite(bad)?2U:1U);
    }
    for(auto encoding:{lp::ActivationEncoding::symmetric_a8,lp::ActivationEncoding::symmetric_a4}){
        const int qmax=encoding==lp::ActivationEncoding::symmetric_a4?7:127;
        for(unsigned kind=0;kind<4;++kind){std::vector<double>x(32,0);
            if(kind==0){x[0]=qmax;x[1]=0.5;x[2]=1.5;x[3]=-0.5;x[4]=-1.5;x[5]=-0.0;}
            if(kind==1){x[0]=std::numeric_limits<double>::max();x[1]=-x[0];x[2]=1;}
            if(kind==2){x[0]=std::numeric_limits<double>::denorm_min()*qmax;x[1]=-x[0];x[2]=std::numeric_limits<double>::denorm_min();}
            run_case({1,1,32},ip::WeightEncoding::q4_0,encoding,encoding==lp::ActivationEncoding::symmetric_a4?ip::Instruction::iu4:ip::Instruction::iu8,
                ip::Phase::decode,x,FixtureMode::zero_weights);
        }
    }
    run_case({1,1,32},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8,ip::Phase::decode,
        std::vector<double>(32,1),FixtureMode::nonfinite_weights,16U);
    run_case({1,1,32},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8,ip::Phase::decode,
        std::vector<double>(32,std::numeric_limits<double>::max()),FixtureMode::overflowing_output,8U);
}
}
int main(int argc,char** argv){
    if(argc!=2||std::string(argv[1])!="--run"){std::cout<<"Unqualified shared gfx1151 IU8/IU4 launcher fixture. Explicit --run required.\n";return 2;}
    try{
        const char* override_arch=std::getenv("HSA_OVERRIDE_GFX_VERSION");require(!override_arch||!*override_arch,"architecture override prevents gfx1151 target attribution");
        check(hipSetDevice(0),"select device");controls();
        run_case({3,19,640},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8,ip::Phase::prefill);
        run_case({17,19,640},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu4,ip::Phase::verify);
        run_case({1,640,2560},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu4,ip::Phase::decode);
        run_case({1,19,32},ip::WeightEncoding::q4_0,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu8,ip::Phase::decode);
        run_case({3,19,2560},ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a8,ip::Instruction::iu8,ip::Phase::prefill);
        run_case({17,19,256},ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu8,ip::Phase::verify);
        run_case({1,2560,2560},ip::WeightEncoding::iq4_xs,lp::ActivationEncoding::symmetric_a4,ip::Instruction::iu8,ip::Phase::decode);
        std::cout<<"Synthetic shared-launcher arithmetic checks passed; no model/performance qualification.\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
