#include "strix/offline_tuning_preparation.hpp"
#include "strix/model_content_hash.hpp"
#include <chrono>
#include <array>
#include <utility>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace strix;
int failures=0;
void expect(bool value,const char* message) {if(!value){++failures;std::cerr<<message<<'\n';}}
template<class F> void rejects(F action,const char* message) {
    bool bad=false;try{action();}catch(const std::invalid_argument&){bad=true;}expect(bad,message);
}
struct Fixture {
    ModelContract model=ModelContract::qwen38_flash_next();
    ExecutionBindingRequest request;
    CompiledSerialExecution baseline;
    ModelManifest manifest;
    ExecutionTensorAccessPlan access;
    kernels::KernelRegistry registry;
    OfflineTuningLiveContext live;
    Fixture(std::uint64_t numerical=7) {
        ExecutionStep step;step.id=0;step.op=SerialOp::lm_head;baseline.execution.steps.push_back(step);
        ExecutionKernelBinding binding;binding.step_id=0;binding.op=step.op;binding.family=kernels::OperatorFamily::lm_head;
        binding.kernel_id="fixture.lm";baseline.bindings.bindings.push_back(binding);
        kernels::KernelVariant variant;variant.id=*binding.kernel_id;variant.family=*binding.family;
        variant.shape=execution_shape_for(model,step.op,request);variant.resources.workgroup_threads=256;
        variant.numerics={"bf16","fp32","fp32",true,true};variant.qualification=kernels::QualificationStatus::target_qualified;
        variant.compiler_identity="fixture-compiler";variant.input_layout="row-major";variant.output_layout="row-major";
        variant.numerical_contract_identity=numerical;variant.measured_seconds=1.0;registry.add(variant);
        manifest.layout_identity=42;ManifestTensor tensor;tensor.name="output.weight";tensor.ggml_type=8;
        tensor.dimensions={model.hidden_size,model.vocab_size};tensor.byte_size=(model.hidden_size/32U)*34U*model.vocab_size;
        manifest.tensors.push_back(tensor);access.manifest_layout_identity=42;access.plan_identity=43;
        access.accesses.push_back({0,step.op,0,{0}});
        live.gfx_target="gfx1151";live.device_sku="runtime-reported-fixture-MP-domain";live.runtime_reported_processor_count=20;
        live.hip_identity="runtime-fixture";live.binary_sha256=std::string(64,'a');live.model_content_sha256=std::string(64,'b');
        live.compiler_flags_sha256=std::string(64,'c');live.state_layout_identity=44;live.context_tokens=8192;
        live.resolved_configuration="resolved-fixture";live.memory_regime="configured-register";
    }
    PreparedOfflineTuning prepare(LoadedOfflineTuningCache cache={}) {
        return PreparedOfflineTuning(std::move(cache),live,model,baseline,manifest,access,registry,request);
    }
    kernels::KernelQuery query() const {
        kernels::KernelQuery q;q.family=kernels::OperatorFamily::lm_head;
        q.shape=execution_shape_for(model,SerialOp::lm_head,request);return q;
    }
};
struct Temporary {
    std::filesystem::path directory;
    Temporary() {
        const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
        for(unsigned i=0;i<256U;++i) {
            auto candidate=std::filesystem::temp_directory_path()/("strix-tuning-source-fixture-"+std::to_string(stamp)+"-"+std::to_string(i));
            if(std::filesystem::create_directory(candidate)){directory=std::move(candidate);return;}
        }
        throw std::runtime_error("temporary fixture directory unavailable");
    }
    ~Temporary(){std::error_code ignored;std::filesystem::remove_all(directory,ignored);}
};
void save(const std::filesystem::path& path,std::span<const std::byte> bytes) {
    std::ofstream file(path,std::ios::binary|std::ios::trunc);file.exceptions(std::ios::failbit|std::ios::badbit);
    if (!bytes.empty()) file.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
}
}
int main() {
    Fixture f;auto prepared=f.prepare();const auto key=prepared.plan().step_keys[0];
    expect(key.binary_sha256==f.live.binary_sha256&&key.model_content_sha256==f.live.model_content_sha256&&
        key.compute_units==20&&key.numerical_contract_identity==7,"keys use live facts and registered numerical class");
    auto changed=f; // Registry and metadata snapshots are independent values.
    changed.manifest.tensors[0].ggml_type=0;
    expect(changed.prepare().plan().step_keys[0].weight_format!=key.weight_format,"actual selected format affects key");
    kernels::OfflineTuningRecord record;record.key=key;record.kernel_id="fixture.lm";
    record.qualification_receipt_sha256=std::string(64,'d');record.samples=3;record.complete_workload=true;
    record.owners_and_resources_qualified=true;record.median_complete_seconds=1;
    const std::array records{record};
    expect(f.registry.select_tuned(f.query(),key,records).reason==kernels::TuningSelectionReason::qualified_record,
        "only existing registry policy can admit a fully qualified synthetic record");
    f.live.compiler_flags_sha256.clear();auto unknown=f.prepare();
    expect(f.registry.select_tuned(f.query(),unknown.plan().step_keys[0],records).reason==kernels::TuningSelectionReason::invalid_key,
        "unknown live flags never borrow the cache flags");
    Fixture numerical_unknown(0);auto zero=numerical_unknown.prepare();
    expect(numerical_unknown.registry.select_tuned(numerical_unknown.query(),zero.plan().step_keys[0],records).reason==kernels::TuningSelectionReason::invalid_key,
        "unknown numerical class remains unknown and cannot be promoted");
    f.access.manifest_layout_identity=999;
    rejects([&]{(void)f.prepare();},"mismatched source access identity rejects");
    f.access.manifest_layout_identity=f.manifest.layout_identity;f.access.accesses[0].tensor_indices[0]=99;
    rejects([&]{(void)f.prepare();},"out of range actual tensor selection rejects");
    Temporary temporary;const auto path=temporary.directory/"cache.bin";
    auto bytes=kernels::encode_offline_tuning_cache(records);save(path,bytes);
    const auto loaded=load_offline_tuning_cache_file(path);
    expect(loaded.records.size()==1&&loaded.payload_bytes==bytes.size()&&loaded.payload_sha256==model_content_identity_hex(sha256_bytes(bytes)),
        "bounded loader hashes exactly the bytes supplied to the codec");
    auto limits=kernels::OfflineTuningCacheLimits{};limits.max_bytes=bytes.size()-1U;
    rejects([&]{(void)load_offline_tuning_cache_file(path,limits);},"oversized file rejected before allocation");
    bytes.pop_back();save(path,bytes);
    rejects([&]{(void)load_offline_tuning_cache_file(path);},"truncated persisted cache rejected");
    save(path,{});rejects([&]{(void)load_offline_tuning_cache_file(path);},"empty persisted cache rejected");
    return failures?1:0;
}
