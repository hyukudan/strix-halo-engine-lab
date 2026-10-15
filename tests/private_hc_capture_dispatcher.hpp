#pragma once
// Test-only host wrapper. Included only by an independently generated app copy.
#include "strix/dispatch_resources.hpp"
#include "strix/hip_launch_context.hpp"
#include "strix/hip_setup_quiescence.hpp"
#include "strix/model_content_hash.hpp"
#include "strix/model_contract.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::private_hc_capture {
struct Config {
    std::filesystem::path output, model;
    std::array<std::uint32_t,62> inputs{};
    std::array<std::uint32_t,20> golden{};
    std::string source_commit, archive_sha256, model_sha256_expected;
};
inline void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
inline std::string quote(std::string_view value) {
    std::string out="\"";
    constexpr char hex[]="0123456789abcdef";
    for(unsigned char c:value) {
        if(c=='"'||c=='\\'){out+='\\';out+=static_cast<char>(c);}
        else if(c<32U){out+="\\u00";out+=hex[c>>4U];out+=hex[c&15U];}
        else out+=static_cast<char>(c);
    }
    return out+'"';
}
inline std::string digest(std::span<const std::byte> bytes) {
    return model_content_identity_hex(sha256_bytes(bytes));
}
class Dispatcher final:public EagerExecutionDispatcher {
public:
    Dispatcher(EagerExecutionDispatcher& delegate,Config config,
        std::shared_ptr<const HipSetupQuiescenceGuard> guard,
        std::shared_ptr<const void> owner,const ModelContract& model)
        :delegate_(delegate),config_(std::move(config)),guard_(std::move(guard)),owner_(std::move(owner)) {
        require(guard_&&owner_,"capture requires retained original owners");guard_->require_original();
        require(model.hidden_size==2560&&model.hc_count==4&&model.hc_lowrank==320&&model.hidden_layers==48&&model.vocab_size==248320,"capture canonical shape differs");
        require(guard_->architecture().starts_with("gfx1151"),"capture requires actual gfx1151");
        require(config_.output.is_absolute()&&config_.model.is_absolute()&&std::filesystem::is_regular_file(config_.model),"capture paths invalid");
        require(std::filesystem::create_directory(config_.output),"capture output must be fresh");
    }
    void dispatch_kernel(const ExecutionDispatchContext& c)override { dispatch(c,true); }
    void dispatch_runtime_step(const ExecutionDispatchContext& c)override { dispatch(c,false); }
    void finish(std::span<const std::uint32_t> generated,std::uint64_t steps,std::uint64_t final_position) {
        guard_->require_original();
        require(!finished_&&generated.size()==20&&steps==62&&final_position==62,"capture request incomplete");
        require(std::equal(generated.begin(),generated.end(),config_.golden.begin()),"capture generated20 differ from immutable golden");
        require(seen_.size()==165&&positions_.size()==62,"capture case/position coverage differs");
        for(std::uint64_t p=0;p<62;++p)require(positions_.contains(p),"capture position missing");
        for(const auto p:{0ULL,42ULL,43ULL,52ULL,61ULL})for(unsigned op=0;op<97;++op)
            if(p==61||selected(op))require(seen_.contains({p,op}),"capture expected case missing");
        copy_maps("loaded-maps-end.txt");
        // Neither admission artifact exists during partial/failed capture.
        std::ofstream index(config_.output/".capture-index.pending");
        index<<"strix-hc-capture165-v1\n";for(const auto& row:tsv_)index<<row<<'\n';index.close();require(static_cast<bool>(index),"capture index write failed");
        std::ofstream receipt(config_.output/".capture.pending");
        receipt<<"{\"schema\":\"strix-hc-capture165-v1\",\"passed\":true,\"cases\":165,\"core_cases\":85,\"graph_anchor_position\":61,\"graph_cases\":97,\"steps\":62,\"generated20_equal_golden\":true,\"source_commit\":"<<quote(config_.source_commit)
            <<",\"archive_sha256\":"<<quote(config_.archive_sha256)<<",\"capture_index_sha256\":"<<quote(model_content_identity_hex(sha256_file_content(config_.output/".capture-index.pending")))<<",\"model_sha256_expected_parent_pin\":"<<quote(config_.model_sha256_expected)<<",\"model_path\":"<<quote(config_.model.string())<<",\"device_binding_identity\":"<<guard_->observed_binding_identity()<<",\"records\":[";
        for(std::size_t i=0;i<records_.size();++i){if(i)receipt<<',';receipt<<records_[i];}
        receipt<<"]}\n";receipt.close();require(static_cast<bool>(receipt),"capture receipt write failed");
        std::filesystem::rename(config_.output/".capture.pending",config_.output/"capture.json");
        std::filesystem::rename(config_.output/".capture-index.pending",config_.output/"capture-index.tsv");finished_=true;
    }
private:
    struct Weight {std::uintptr_t address;std::uint64_t generation;std::string json;};
    static bool selected(unsigned op) {
        if(op==96)return true;
        const auto layer=op/2U;
        return layer==0||layer==1||layer==2||layer==3||layer==23||layer==24||layer==46||layer==47;
    }
    static void check(hipError_t result) { if(result!=hipSuccess)throw std::runtime_error(hipGetErrorString(result)); }
    void copy_maps(const char* name) {
        std::ifstream source("/proc/self/maps");std::ofstream dest(config_.output/name);dest<<source.rdbuf();require(source&&dest,"capture maps failed");
    }
    std::vector<std::byte> read_device(std::uintptr_t address,std::size_t bytes,hipStream_t stream) {
        // Persistent staging survives every exception until original-device recovery.
        transfer_.resize(bytes);check(hipMemcpyAsync(transfer_.data(),reinterpret_cast<const void*>(address),bytes,hipMemcpyDeviceToHost,stream));
        check(hipStreamSynchronize(stream));guard_->require_original();return transfer_;
    }
    std::string save(const std::string& name,std::uintptr_t address,std::size_t bytes,hipStream_t stream) {
        require(!std::filesystem::exists(config_.output/name),"duplicate capture artifact");
        const auto values=read_device(address,bytes,stream);std::ofstream file(config_.output/name,std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()),static_cast<std::streamsize>(values.size()));file.close();require(static_cast<bool>(file),"capture blob write failed");return digest(values);
    }
    std::string weight(const ExecutionDispatchContext& c,unsigned slot,const std::string& expected,hipStream_t stream) {
        const auto& access=resolve_dispatch_tensor_access(c,slot);require(c.resources&&c.resources->model_manifest&&c.resources->model_weights,"capture missing tensor owners");
        require(access.tensor_indices.size()==1,"capture requires unique tensor access");
        const auto tensor_index=access.tensor_indices.front();
        const auto& manifest=c.resources->model_manifest->tensors.at(tensor_index);
        const auto& runtime=resolve_dispatch_tensor(c,tensor_index);
        require(manifest.name==expected&&runtime.allocation_generation&&runtime.bytes>=manifest.byte_size&&manifest.byte_size<=4000000U,"capture tensor binding differs");
        if(auto it=weights_.find(expected);it!=weights_.end()) {require(it->second.address==runtime.address&&it->second.generation==runtime.allocation_generation,"capture weight owner changed");return it->second.json;}
        std::ifstream source(config_.model,std::ios::binary);source.seekg(static_cast<std::streamoff>(manifest.absolute_offset));
        std::vector<std::byte> file_bytes(static_cast<std::size_t>(manifest.byte_size));source.read(reinterpret_cast<char*>(file_bytes.data()),static_cast<std::streamsize>(file_bytes.size()));require(static_cast<bool>(source),"capture model tensor read failed");
        const auto actual=read_device(runtime.address,file_bytes.size(),stream);require(actual==file_bytes,"capture actual weight bytes differ from model window");
        std::ostringstream out;out<<"{\"name\":"<<quote(manifest.name)<<",\"tensor_index\":"<<tensor_index<<",\"ggml_type\":"<<manifest.ggml_type<<",\"offset\":"<<manifest.absolute_offset<<",\"bytes\":"<<manifest.byte_size<<",\"sha256\":"<<quote(digest(file_bytes))<<",\"address\":"<<runtime.address<<",\"allocation_generation\":"<<runtime.allocation_generation<<",\"owner_node\":"<<runtime.owner_node<<",\"storage\":"<<static_cast<unsigned>(runtime.storage)<<",\"dimensions\":[";
        for(std::size_t i=0;i<manifest.dimensions.size();++i){if(i)out<<',';out<<manifest.dimensions[i];}out<<"]}";
        auto value=out.str();weights_.emplace(expected,Weight{runtime.address,runtime.allocation_generation,value});return value;
    }
    void dispatch(const ExecutionDispatchContext& c,bool kernel) {
        guard_->require_original();
        try {
            const bool attn=c.op==SerialOp::attn_hyper_mix,ffn=c.op==SerialOp::mlp_hyper_mix,final=c.op==SerialOp::final_hyper_mix;
            if(!(attn||ffn||final)){if(kernel)delegate_.dispatch_kernel(c);else delegate_.dispatch_runtime_step(c);return;}
            require(c.logical_rows==1&&c.request_concurrency==1&&!hip_graph_status()&&c.position<62,"capture requires eager serial C1 positions0..61");
            require(c.input_token_id==config_.inputs[c.position],"capture actual input token differs");
            require(final?!c.layer:(c.layer&&*c.layer<48),"capture layer binding differs");
            const unsigned op=final?96U:*c.layer*2U+(ffn?1U:0U);positions_.insert(c.position);
            if(kernel)delegate_.dispatch_kernel(c);else delegate_.dispatch_runtime_step(c);
            if(c.position!=0&&c.position!=42&&c.position!=43&&c.position!=52&&c.position!=61)return;
            if(c.position!=61&&!selected(op))return;
            require(seen_.insert({c.position,op}).second,"duplicate capture case");
            require(c.temporaries.size()==1&&c.writes.size()==(final?1U:2U),"capture workspace/IO topology differs");
            const auto temp=resolve_dispatch_temporary(c,0);const auto input=resolve_dispatch_read(c,0);
            require(temp.use.buffer.buffer_class==ExecutionBufferClass::hyper_workspace&&temp.bytes>=42240U&&temp.address%16U==0&&temp.allocation_generation,"capture workspace extent/alignment/owner differs");
            require(input.bytes>=20480U&&input.allocation_generation,"capture BF16 input extent/owner differs");
            require(c.resources&&c.resources->model_weights&&model_content_identity_hex(c.resources->model_weights->model_content_identity)=="3726d6f0d9ba30b9d655ba303595118941baeca5e6945fc8785f938e0f5455e4","capture actual model content differs");
            const auto stream=hip_execution_stream();check(hipStreamSynchronize(stream));guard_->require_original();
            if(records_.empty())copy_maps("loaded-maps.txt");
            const std::string kind=final?"output":(attn?"attn":"ffn");const auto layer=final?48U:*c.layer;
            const std::string stem="p"+std::to_string(c.position)+"-"+kind+"-l"+std::to_string(layer);
            const auto normfile=stem+"-normalized.f32",lowfile=stem+"-low.f32",injfile=final?std::string("-"):stem+"-injection.f32";
            const auto normsha=save(normfile,temp.address,40960U,stream),lowsha=save(lowfile,temp.address+40960U,1280U,stream);
            std::string injsha="-";ExecutionBufferAddress injection{};
            if(!final){injection=resolve_dispatch_write(c,1);require(injection.bytes>=16U&&injection.allocation_generation,"capture injection extent/owner differs");injsha=save(injfile,injection.address,16U,stream);}
            const auto prefix=final?std::string("output_hc"):"blk."+std::to_string(layer)+".hc_"+(attn?"attn":"ffn");
            const auto down=prefix+"_down.weight",inject=final?std::string("-"):prefix+"_inject.weight";
            std::array<std::string,4> weight_json;for(unsigned i=0;i<(final?3U:4U);++i)weight_json[i]=weight(c,i,prefix+(i==0?"_norm.weight":i==1?"_down.weight":i==2?"_up.weight":"_inject.weight"),stream);
            std::ostringstream row;row<<c.position<<'\t'<<kind<<'\t'<<layer<<'\t'<<normfile<<'\t'<<lowfile<<'\t'<<injfile<<'\t'<<normsha<<'\t'<<lowsha<<'\t'<<injsha<<'\t'<<down<<'\t'<<inject;tsv_.push_back(row.str());
            std::ostringstream record;record<<"{\"position\":"<<c.position<<",\"op\":"<<quote(kind)<<",\"layer\":"<<layer<<",\"input_token_id\":"<<c.input_token_id<<",\"step_id\":"<<c.step_id<<",\"execution_identity\":"<<c.execution_identity<<",\"kernel_id\":"<<quote(c.kernel_id?*c.kernel_id:std::string_view{})<<",\"stream\":"<<reinterpret_cast<std::uintptr_t>(stream)<<",\"workspace_address\":"<<temp.address<<",\"workspace_allocation_generation\":"<<temp.allocation_generation<<",\"input_address\":"<<input.address<<",\"input_allocation_generation\":"<<input.allocation_generation<<",\"input_sha256\":"<<quote(digest(read_device(input.address,20480U,stream)))<<",\"manifest_layout_identity\":"<<c.resources->model_weights->manifest_layout_identity<<",\"model_content_identity\":"<<quote(model_content_identity_hex(c.resources->model_weights->model_content_identity))<<",\"weights\":[";
            for(unsigned i=0;i<(final?3U:4U);++i){if(i)record<<',';record<<weight_json[i];}record<<"]}";records_.push_back(record.str());
        }catch(...){guard_->recover_or_fail_stop();throw;}
    }
    EagerExecutionDispatcher& delegate_;Config config_;std::shared_ptr<const HipSetupQuiescenceGuard> guard_;std::shared_ptr<const void> owner_;
    std::vector<std::byte> transfer_;std::set<std::pair<std::uint64_t,unsigned>> seen_;std::set<std::uint64_t> positions_;
    std::vector<std::string> tsv_,records_;std::map<std::string,Weight> weights_;bool finished_=false;
};
} // namespace strix::private_hc_capture
