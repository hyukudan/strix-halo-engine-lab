#include "hip_mtp_model_qualification.hpp"
#include "hip_mtp_qualification_reference.hpp"
#include "hip_mtp_target_checkpoint_components.hpp"
#include <charconv>
#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include "strix/model_content_hash.hpp"
#include <set>
#include <string>
#include <vector>
int strix_hip_decode_token_entry(int,char**);
namespace strix::tests {
namespace {
struct Request {std::string path,sha,export_path;std::uint64_t file_bytes=0,host_bytes=0,coverage=31,export_bytes=0,export_capture_bytes=0;};
Request request;
std::optional<MtpQualificationReference> reference;
void require(bool v,const char* message){if(!v)throw std::invalid_argument(message);}
std::uint64_t number(std::string_view text){std::uint64_t n=0;const auto r=std::from_chars(text.data(),text.data()+text.size(),n);
    require(r.ec==std::errc{} && r.ptr==text.data()+text.size(),"invalid qualification numeric option");return n;}
std::vector<char*> configure(int argc,char** argv){
    std::set<std::string> seen;std::vector<char*> forward{argv[0]};
    for(int i=1;i<argc;++i){const std::string_view arg=argv[i];if(!arg.starts_with("--qualification-")){forward.push_back(argv[i]);continue;}
        const auto eq=arg.find('=');const std::string key(arg.substr(0,eq));std::string_view value;
        require(seen.insert(key).second,"duplicate qualification option");
        if(eq!=std::string_view::npos)value=arg.substr(eq+1);else {require(i+1<argc,"qualification option requires a value");value=argv[++i];}
        if(key=="--qualification-reference")request.path=value;
        else if(key=="--qualification-sha256")request.sha=value;
        else if(key=="--qualification-file-bytes")request.file_bytes=number(value);
        else if(key=="--qualification-host-bytes")request.host_bytes=number(value);
        else if(key=="--qualification-coverage-mask")request.coverage=number(value);
        else if(key=="--qualification-export-target-input")request.export_path=value;
        else if(key=="--qualification-export-bytes")request.export_bytes=number(value);
        else if(key=="--qualification-export-capture-bytes")request.export_capture_bytes=number(value);
        else throw std::invalid_argument("unknown qualification option");
    }
    if(!request.export_path.empty()){
        require(request.path.empty() && request.sha.empty() && !request.file_bytes && !request.host_bytes &&
            request.export_bytes>=64U && request.export_bytes<=(1ULL<<30U) && request.export_capture_bytes &&
            !std::filesystem::exists(request.export_path),"target input export needs a new path and explicit output/capture budgets, without golden options");
        forward.push_back(nullptr);return forward;
    }
    require(!request.export_bytes && !request.export_capture_bytes && !request.path.empty() && decode_cli::mtp_sha256_text(request.sha) && request.file_bytes>=160U &&
        request.file_bytes<=(1ULL<<30U) && request.host_bytes>=262144U &&
        request.file_bytes<=(request.host_bytes-262144U)/3U && request.coverage<=31U,
        "qualification requires a pinned reference and explicit file/host budgets (host >= 3*file+256KiB)");
    require(std::find_if(forward.begin()+1,forward.end(),[](const char* p){return std::string_view(p)=="--mtp";})!=forward.end(),
        "qualification runner requires the explicit MTP route and all ordinary asset/replay pins");
    reference=read_mtp_qualification_reference(request.path,decode_cli::mtp_cli_digest(request.sha),request.file_bytes);
    forward.push_back(nullptr);return forward;
}
}
bool mtp_target_input_export_requested() noexcept{return !request.export_path.empty();}
void validate_mtp_target_input_export(const decode_cli::Options& o){
    require(!o.mtp.enabled && !o.mtp.supplied && o.execution.mode==DecodeExecutionMode::eager && o.execution.integer_head==DecodeIntegerHead::none &&
        !o.prepared_eager && !o.moe_shared_overlap && !o.moe_token_cost_policy && !o.gdn_islands && !o.gpu_greedy_token &&
        !o.prefix_cache_roundtrip && !o.prompt_lookup && !o.known_prefix_window && o.serving_prefix_files.empty() && !o.device_queue_trace &&
        o.reference_path.empty() && o.raw_recording_path.empty() && o.validation_final_state_path.empty() && o.logprob_token_ids.empty() &&
        o.tokens.size()>=3U && o.tokens.size()<=4097U,
        "target input export requires 3..4097 prompt tokens and an ordinary eager target without MTP or other request controllers");
}
void export_mtp_target_inputs(const decode_cli::Options& o,const ModelContract& model,std::shared_ptr<::HipDecodeExperimentResources> actual,
    const CompiledSerialExecution& compiled,const ExecutionRuntimeResources& resources,SerialSessionContext& session,
    DirectLaunchDispatcher& dispatcher,AllocationGenerationSource& generations,bool d1){
    validate_mtp_target_input_export(o);require(actual && !session.next_position && o.tokens.size()<=o.context_tokens,
        "target input export requires actual parents and a fresh bounded target");
    const auto width=decode_cli::mtp_cli_mul(model.hidden_size,model.hc_count);
    require(width<=UINT32_MAX && o.tokens.size()<=UINT32_MAX,"target input packet dimensions exceed schema");
    const auto bytes=decode_cli::mtp_cli_add(64U,decode_cli::mtp_cli_mul(o.tokens.size(),decode_cli::mtp_cli_add(16U,decode_cli::mtp_cli_mul(width,2U))));
    require(bytes<=request.export_bytes && bytes<=SIZE_MAX && !std::filesystem::exists(request.export_path),"target input packet exceeds budget or already exists");
    HipMtpTargetDeferredConfig deferred;if(d1)deferred={resources.session_state_layout,&generations};
    auto target=HipMtpTargetHidden::create(model,compiled,actual->memory,actual->model,actual->allocator,2U,request.export_capture_bytes,deferred);
    std::vector<std::byte> packet;packet.reserve(static_cast<std::size_t>(bytes));
    const auto word=[&](std::uint64_t value,unsigned n){for(unsigned i=0;i<n;++i)packet.push_back(static_cast<std::byte>((value>>(8U*i))&255U));};
    for(char c:std::string_view("MTPINP01"))word(static_cast<unsigned char>(c),1);
    for(auto b:actual->model->binding().model_content_identity.bytes)word(b,1);
    word(compiled.execution_identity,8);word(compiled.kernel_plan_identity,8);word(width,4);word(o.tokens.size(),4);
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(width));
    const auto stream=static_cast<hipStream_t>(target->stream());
    for(const auto token:o.tokens){
        const auto ticket=target->execute_serial(session,resources,dispatcher,token);const auto view=ticket.view();
        if(hipMemcpyAsync(hidden.data(),reinterpret_cast<const void*>(view.address),static_cast<std::size_t>(view.bytes),hipMemcpyDeviceToHost,stream)!=hipSuccess ||
           hipStreamSynchronize(stream)!=hipSuccess)throw std::runtime_error("target input snapshot readback failed");
        target->guard()->require_original();word(ticket.source_position(),8);word(ticket.input_token(),4);word(ticket.predicted_token(),4);
        for(auto value:hidden)word(value,2);
    }
    require(packet.size()==bytes,"target input packet framing differs");
    std::ofstream file(request.export_path,std::ios::binary|std::ios::out);require(bool(file),"cannot create target input packet");
    file.write(reinterpret_cast<const char*>(packet.data()),static_cast<std::streamsize>(packet.size()));file.close();require(bool(file),"target input packet write failed");
    std::cout<<"{\"mode\":\"target-input-only\",\"mtp_executed\":false,\"mtp_activated\":false,\"rows\":"<<o.tokens.size()
        <<",\"bytes\":"<<packet.size()<<",\"packet_sha256\":\""<<model_content_identity_hex(sha256_bytes(packet))
        <<"\",\"target_file_sha256\":\""<<model_content_identity_hex(actual->mapped->file_sha256())<<"\"}\n";
}
void run_mtp_model_qualification(decode_cli::MtpDecodeQualificationContext& c){
    require(reference.has_value(),"qualification reference was not admitted before setup");
    const auto& g=*reference;
    const auto& owner=*c.predictor;
    require(g.target_content==owner.shared_target().target_model_content_identity && g.sidecar_content==owner.shared_target().sidecar_model_content_identity &&
        g.proposal_identity==owner.plan().proposal_identity && g.plan_identity==owner.plan().plan_identity &&
        g.semantic_identity==owner.plan().forward_semantics.contract_identity && g.native_profile_identity==c.runtime.numerical_profile_identity(),
        "qualification reference does not bind actual assets/plan/semantics/native recipe");
    HipMtpGreedyCycleComponentRequest cycles;
    if(g.kind==MtpQualificationKind::cycles || g.kind==MtpQualificationKind::history_fallback){
        cycles.expected_payload=hip_mtp_greedy_cycle_golden_identity(g.cycles);cycles.maximum_cycles=256U;cycles.host_payload_budget_bytes=request.host_bytes;
        cycles.required_coverage={(request.coverage&1U)!=0U,(request.coverage&2U)!=0U,(request.coverage&4U)!=0U,(request.coverage&8U)!=0U,(request.coverage&16U)!=0U};
    }
    std::uint64_t completed=1;
    switch(g.kind){
    case MtpQualificationKind::history_fallback:{
        require(c.window && !g.cycles.cycles.empty(),"history fixture requires actual window and cycle trace");
        const auto& first=g.cycles.cycles.front();const auto rows=c.window->logical_rows();
        const auto capacity=std::min(c.parents->resources.session_state_layout->context_tokens,owner.model().max_position_embeddings);
        require(c.options.mtp.window_history_intervals && c.session.arithmetic_lineage.intervals.size()>=c.options.mtp.window_history_intervals &&
            c.session.next_position<=capacity && rows<=capacity-c.session.next_position && first.maximum_candidates>=rows-1U &&
            first.proposed_candidates>=rows-1U && first.token_budget>=rows && c.target->retained_rows()>=rows+2U,
            "history fixture must exhaust only the real history bound, with otherwise eligible window");
        for(const auto& row:g.cycles.cycles)require(!row.attempted_window && !row.accepted_window && !row.restored_checkpoint &&
            !row.target_r2_dispatches && !row.target_projection_launches_saved,"exhausted-history reference must require serial fallback");
        [[fallthrough]];
    }
    case MtpQualificationKind::cycles:{
        if(c.window)c.greedy.use_window_verifier(c.window);
        const auto receipt=qualify_hip_mtp_greedy_cycles(c.greedy,c.runtime,std::move(c.initial),g.cycles,cycles);completed=receipt.cycles.size();break;
    }
    case MtpQualificationKind::projection:testing::qualify_hip_mtp_projection_components(c.predictor,g.projection);break;
    case MtpQualificationKind::shared:testing::qualify_hip_mtp_shared_expert_component(c.predictor,g.shared);break;
    case MtpQualificationKind::window:{
        require(bool(c.window),"window fixture needs actual configured native window");
        auto window=c.window->window_owner();c.window.reset();
        testing::qualify_hip_mtp_target_window_capture(*window,c.session,g.window,
            {c.options.mtp.window_capture_bytes,c.options.mtp.window_history_intervals});completed=g.window.chunks.size();break;
    }
    case MtpQualificationKind::lm_head:{
        const auto receipt=qualify_hip_mtp_lm_head_component(c.predictor,g.head,{hip_mtp_lm_head_golden_identity(g.head),request.host_bytes});
        completed=receipt.outputs_compared;break;
    }
    case MtpQualificationKind::checkpoint:{
        require(bool(c.window) && !g.window.chunks.empty(),"checkpoint fixture needs an actual window and explicit mutation inputs");
        require(g.window.initial_position==c.session.next_position,"checkpoint reference initial frontier differs");
        auto window=c.window->window_owner();c.window.reset();
        const auto receipt=qualify_hip_mtp_target_window_checkpoint(c.target,window,c.session,c.parents->resources,c.parents,
            g.window.chunks.front().inputs,{c.options.mtp.window_capture_bytes,c.options.mtp.window_history_intervals},
            c.options.mtp.window_checkpoint_bytes,request.host_bytes,c.generations);
        require(receipt.full_device_bytes_restored && receipt.full_host_bytes_restored && receipt.deferred_status_restored && receipt.exact_session_metadata_restored,
            "checkpoint fixture did not confirm every restoration component");break;
    }
    }
    std::cout<<"{\"mode\":\"native-mtp-model-fixture\",\"kind\":"<<static_cast<std::uint32_t>(g.kind)
        <<",\"completed_records\":"<<completed<<",\"native_fixture_passed\":true,\"performance_qualified\":false,"
        <<"\"reference_file_sha256\":\""<<request.sha<<"\",\"reference_independence\":\"externally-supplied-not-established-by-digest\"}\n";
}
} // namespace strix::tests
int main(int argc,char** argv){try{
    auto forwarded=strix::tests::configure(argc,argv);
    if(forwarded.size()-1U>static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::invalid_argument("qualification argument count overflow");
    return strix_hip_decode_token_entry(static_cast<int>(forwarded.size()-1U),forwarded.data());
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
