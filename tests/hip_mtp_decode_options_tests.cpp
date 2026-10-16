#include "../apps/hip_decode_options.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
strix::decode_cli::Options parse(std::vector<std::string> args){
    std::vector<char*> argv;for(auto& value:args)argv.push_back(value.data());
    return strix::decode_cli::parse_options(static_cast<int>(argv.size()),argv.data());
}
std::vector<std::string> request(){
    const std::string sha(64,'1');
    return {"decoder","target.gguf","1","2","3","--generate","512","--mtp",
        "--mtp-sidecar","sidecar.gguf","--mtp-sidecar-logical-name","mtp-sidecar",
        "--mtp-sidecar-sha256",sha,"--mtp-target-sha256",sha,"--mtp-target-content",sha,
        "--mtp-golden","independent.bin","--mtp-golden-sha256",sha,
        "--mtp-source-repository","publisher/model","--mtp-source-revision","pinned-revision",
        "--mtp-source-path","sidecar.gguf","--mtp-converter-revision","converter-revision",
        "--mtp-converter-sha256",sha,"--mtp-norm-converter-sha256",sha,
        "--mtp-conversion","effective-norm-embedding-first-v1","--mtp-normalized-rounding","fp32",
        "--mtp-projection-rounding","bf16-rne","--mtp-sidecar-residency","register",
        "--mtp-sidecar-bytes","1024","--mtp-header-bytes","512",
        "--mtp-sidecar-retained-bytes","2048","--mtp-sidecar-setup-bytes","2048",
        "--mtp-golden-max-bytes","4096","--mtp-replay-host-bytes","8192",
        "--mtp-runtime-bytes","16384","--mtp-target-capture-bytes","8192",
        "--mtp-policy-id","7","--mtp-candidate-depth","2"};
}
void set(std::vector<std::string>& values,const std::string& key,const std::string& value){
    for(std::size_t i=0;i<values.size();++i){
        if(values[i]==key && i+1U<values.size()){values[i+1U]=value;return;}
        if(values[i].starts_with(key+"=")){values[i]=key+"="+value;return;}
    }
    throw std::logic_error("fixture key absent");
}
void reject(const std::vector<std::string>& args,const char* message){
    bool failed=false;try{(void)parse(args);}catch(const std::exception&){failed=true;}require(failed,message);
}
}
int main(){try{
    const auto legacy=parse({"decoder","target.gguf","1"});
    require(!legacy.mtp.enabled&&!legacy.mtp.supplied,"legacy CLI does not select MTP");
    auto good=parse(request());
    require(good.mtp.enabled&&good.mtp.candidate_depth==2U&&good.mtp.normalized_rounding=="fp32"&&
        good.mtp.projection_rounding=="bf16-rne"&&!good.any_low_level_flag,
        "MTP flags preserve automatic profile selection and explicit cast choices");
    // Parsing precedes automatic context resolution. A normal512 output request
    // must remain admissible until the actual resolved context is known.
    good.context_tokens=8192;strix::decode_cli::validate_resolved_mtp_cli(good);
    auto metadata=request();
    for(const auto* key:{"--mtp-golden","--mtp-golden-sha256"}){
        for(auto it=metadata.begin();it!=metadata.end();++it)if(*it==key){metadata.erase(it,it+2);break;}
    }
    reject(metadata,"ordinary MTP still requires independent golden pins");
    metadata.insert(metadata.end(),{"--mtp-metadata-only","1"});
    const auto probe=parse(metadata);
    require(probe.mtp.metadata_only && probe.mtp.golden.empty() && probe.mtp.golden_sha256.empty(),
        "explicit metadata diagnostic admits real assets without an expected-output file");
    set(metadata,"--mtp-metadata-only","0");reject(metadata,"metadata diagnostic requires its explicit value");
    for(const auto* flag:{"--prepared-eager","--moe-shared-overlap","--gdn-islands","--gpu-greedy-token","--prompt-lookup",
        "--prefix-cache-roundtrip","--device-queue-trace"}){
        auto bad=request();bad.push_back(flag);reject(bad,"incompatible MTP route rejects at parsing before native setup");
    }
    for(unsigned fault=0;fault<13U;++fault){auto bad=request();
        if(fault==0U)set(bad,"--mtp-sidecar-sha256",std::string(64,'0'));
        if(fault==1U)set(bad,"--mtp-golden-sha256","abc");
        if(fault==2U)set(bad,"--mtp-target-content",std::string(64,'g'));
        if(fault==3U)set(bad,"--mtp-runtime-bytes","0");
        if(fault==4U)set(bad,"--mtp-header-bytes","33554433");
        if(fault==5U)set(bad,"--mtp-candidate-depth","9");
        if(fault==6U)set(bad,"--mtp-normalized-rounding","implicit");
        if(fault==7U)set(bad,"--mtp-conversion","raw-norm");
        if(fault==8U){bad.push_back("--mtp-policy-id=8");}
        if(fault==9U){bad.insert(bad.end(),{"--context","3"});}
        if(fault==10U){bad.push_back("--mtp-target-window-rows=2");}
        if(fault==11U)set(bad,"--mtp-policy-id","0");
        if(fault==12U){bad.insert(bad.end(),{"--mtp-max-logit-error","-1"});}
        reject(bad,"MTP source pins, positive budgets, casts, bounds and duplicate options fail closed");
    }
    reject({"decoder","target.gguf","1","--mtp-runtime-bytes","1"},"MTP parameters do not silently select execution");
    auto wide=request();wide.insert(wide.end(),{"--mtp-target-window-rows=2","--mtp-window-arena-bytes=1024",
        "--mtp-window-capture-bytes=1024","--mtp-window-checkpoint-bytes=1024","--mtp-window-total-bytes=8192",
        "--mtp-window-history-intervals=128"});
    // B2: configured windows must be reachable with the admitted draft depth.
    // Check exact admission cause, not any unrelated missing-budget rejection.
    for(const auto rows:{2U,4U,8U}) {
        auto minimum=wide;
        set(minimum,"--mtp-target-window-rows",std::to_string(rows));
        set(minimum,"--mtp-candidate-depth",std::to_string(rows-1U));
        const auto admitted=parse(minimum);
        require(admitted.mtp.window_rows==rows && admitted.mtp.candidate_depth==rows-1U,
            "each closed MTP target window admits its exact minimum draft depth");
        if(rows>2U) {
            auto insufficient=minimum;
            set(insufficient,"--mtp-candidate-depth",std::to_string(rows-2U));
            bool depth_rejected=false;
            try{(void)parse(insufficient);}catch(const std::invalid_argument& e){
                depth_rejected=std::string(e.what())=="MTP target window requires candidate depth at least rows minus one";
            }
            require(depth_rejected,"insufficient draft depth rejects specifically before reserving any target window");
        }
    }
    auto unreachable=wide;set(unreachable,"--mtp-target-window-rows","8");
    bool unreachable_rejected=false;
    try{(void)parse(unreachable);}catch(const std::invalid_argument& e){
        unreachable_rejected=std::string(e.what())=="MTP target window requires candidate depth at least rows minus one";
    }
    require(unreachable_rejected,"depth2 cannot configure an unreachable eight-row target window");
    require(strix::decode_cli::mtp_target_verification_name(0U)=="serial-target-fallback" &&
        strix::decode_cli::mtp_target_verification_name(1U)=="closed-known-window-attempted-with-serial-fallback" &&
        strix::decode_cli::mtp_target_verification_name(7U)==strix::decode_cli::mtp_target_verification_name(1U),
        "receipt verification mode follows actual attempt evidence, never owner configuration alone");
    auto selected=parse(wide);selected.context_tokens=8192;
    bool failed=false;try{strix::decode_cli::validate_resolved_mtp_cli(selected);}catch(const std::invalid_argument&){failed=true;}
    require(failed,"MTP target windows cannot silently enable D1 or paired GDN optimizations");
    selected.deferred_moe_status=selected.virtual_q8_gdn_input=selected.prefetch_q8_gdn_input=selected.virtual_q8_gdn_output=
        selected.wave8_q8_gdn_input=selected.wave8_q8_gdn_output=true;
    strix::decode_cli::validate_resolved_mtp_cli(selected);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
