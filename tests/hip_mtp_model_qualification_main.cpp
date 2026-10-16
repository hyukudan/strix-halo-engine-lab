#include "hip_mtp_model_qualification.hpp"
#include "hip_mtp_qualification_reference.hpp"
#include "hip_mtp_target_checkpoint_components.hpp"
#include <charconv>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>
int strix_hip_decode_token_entry(int,char**);
namespace strix::tests {
namespace {
struct Request {std::string path,sha;std::uint64_t file_bytes=0,host_bytes=0,coverage=31;};
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
        else throw std::invalid_argument("unknown qualification option");
    }
    require(!request.path.empty() && decode_cli::mtp_sha256_text(request.sha) && request.file_bytes>=160U &&
        request.file_bytes<=(1ULL<<30U) && request.host_bytes>=262144U &&
        request.file_bytes<=(request.host_bytes-262144U)/3U && request.coverage<=31U,
        "qualification requires a pinned reference and explicit file/host budgets (host >= 3*file+256KiB)");
    require(std::find_if(forward.begin()+1,forward.end(),[](const char* p){return std::string_view(p)=="--mtp";})!=forward.end(),
        "qualification runner requires the explicit MTP route and all ordinary asset/replay pins");
    reference=read_mtp_qualification_reference(request.path,decode_cli::mtp_cli_digest(request.sha),request.file_bytes);
    forward.push_back(nullptr);return forward;
}
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
    case MtpQualificationKind::window:
        require(bool(c.window),"window fixture needs actual configured native window");
        testing::qualify_hip_mtp_target_window_capture(*c.window->window_owner(),c.session,g.window,
            {c.options.mtp.window_capture_bytes,c.options.mtp.window_history_intervals});completed=g.window.chunks.size();break;
    case MtpQualificationKind::lm_head:{
        const auto receipt=qualify_hip_mtp_lm_head_component(c.predictor,g.head,{hip_mtp_lm_head_golden_identity(g.head),request.host_bytes});
        completed=receipt.outputs_compared;break;
    }
    case MtpQualificationKind::checkpoint:{
        require(bool(c.window) && !g.window.chunks.empty(),"checkpoint fixture needs an actual window and explicit mutation inputs");
        require(g.window.initial_position==c.session.next_position,"checkpoint reference initial frontier differs");
        const auto receipt=qualify_hip_mtp_target_window_checkpoint(c.target,c.window->window_owner(),c.session,c.parents->resources,c.parents,
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
