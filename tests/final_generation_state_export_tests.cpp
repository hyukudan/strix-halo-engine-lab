#include "strix/final_generation_state_export.hpp"
#include "strix/model_content_hash.hpp"
#include "../apps/hip_decode_options.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
    try { function(); } catch(const std::exception&) { return; }
    throw std::runtime_error("invalid final state export was admitted");
}
std::string text(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    require(file.good(),"missing final state artifact");
    return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
struct Owner { std::vector<std::byte> device,host; };
struct Fixture final : strix::SerialGreedyStepBackend {
    std::shared_ptr<Owner> owner=std::make_shared<Owner>();
    strix::SerialSessionStateLayout layout;
    strix::SessionStateMemoryBinding binding;
    strix::SerialSessionContext session;
    std::array<std::uint32_t,2> prefix{1,2};
    strix::SerialGreedyRequest request{prefix,16,16,3,15};
    strix::SerialGreedyRequestResult result;
    strix::FinalGenerationStateMetadata metadata;
    unsigned drains=0,guards=0,copies=0,publications=0;
    bool fail_read=false,fail_guard=false,fail_publication=false,race_destination=false;
    std::filesystem::path root;
    explicit Fixture(const std::filesystem::path& base,unsigned index,
                     std::uint64_t maximum=3,std::uint32_t eos=15,std::uint64_t context=16)
        : root(base/std::to_string(index)) {
        require(std::filesystem::create_directory(root),"fixture directory collision");
        request.max_new_tokens=maximum; request.eos_token_id=eos; request.context_capacity=context;
        layout.context_tokens=context; layout.device_alignment=1; layout.host_alignment=1;
        constexpr std::array flags{strix::PersistentState::attention_kv,strix::PersistentState::recurrent_matrix,
            strix::PersistentState::convolution_history,strix::PersistentState::convolution_history,
            strix::PersistentState::sparse_index,strix::PersistentState::ngram_history,strix::PersistentState::position_history};
        for(unsigned i=0;i<7U;++i) {
            const bool host=i>=5U;
            auto& total=host?layout.host_total_bytes:layout.device_total_bytes;
            const std::uint64_t count=i==0U?131073U:129U+i;
            layout.allocations.push_back({static_cast<strix::SessionStateComponent>(i),flags[i],
                host?strix::SessionStateStorage::host:strix::SessionStateStorage::device,total,count,false});
            total+=count;
        }
        layout.layout_identity=strix::serial_session_state_layout_identity(strix::ModelContract::qwen38_flash_next(),layout);
        owner->device.resize(static_cast<std::size_t>(layout.device_total_bytes));
        owner->host.resize(static_cast<std::size_t>(layout.host_total_bytes));
        binding={layout.layout_identity,reinterpret_cast<std::uintptr_t>(owner->device.data()),layout.device_total_bytes,19,
                    reinterpret_cast<std::uintptr_t>(owner->host.data()),layout.host_total_bytes,23};
        session.execution_identity=29; session.runtime_compatibility_identity=31; session.state_layout_identity=layout.layout_identity;
        session.state.valid=static_cast<strix::PersistentState>(63U);
        metadata.model_sha256=std::string(64,'1'); metadata.model_content_identity=std::string(64,'2');
        metadata.executable_sha256=std::string(64,'3'); metadata.numerical_identity="synthetic-portable-no-native-claim";
        metadata.output_mode="fixture"; metadata.gpu_identity=std::string(32,'4'); metadata.gpu_architecture="synthetic";
        metadata.kernel_plan_identity=37; metadata.launch_plan_identity=41; metadata.runtime_binding_identity=43;
        metadata.hip_runtime_version=1; metadata.hip_driver_version=1;
        if (maximum == 0U) {
            for (auto input : prefix) execute_state(input,session.next_position);
            result.steps_executed=prefix.size(); result.final_position=session.next_position;
        } else result=strix::execute_serial_greedy_token_request(request,*this);
        if(maximum==3U && eos==15U && context==16U)
            require(result.generated_tokens==std::vector<std::uint32_t>({3,4,5}) && result.steps_executed==4U,
                    "fixture does not use the actual controller bonus convention");
    }
    void execute_state(std::uint32_t input,std::uint64_t position) override {
        require(position==session.next_position,"fixture input frontier mismatch");
        for(auto* buffer:{&owner->device,&owner->host}) for(auto& byte:*buffer)
            byte^=static_cast<std::byte>(input+static_cast<std::uint32_t>(position));
        session.arithmetic_lineage=strix::prepare_serial_row_lineage(session.arithmetic_lineage,position,29,37);
        ++session.next_position; ++session.committed_tokens;
        for(auto& generation:session.state.generations) ++generation;
    }
    std::vector<float> execute(std::uint32_t input,std::uint64_t position) override {
        execute_state(input,position);
        std::vector<float> logits(16,-1.0F); logits[static_cast<std::size_t>(position+2U)]=1.0F; return logits;
    }
    void run(const std::filesystem::path& destination,strix::FinalGenerationStatePolicy policy={}) {
        const auto retained=owner;
        strix::export_final_generation_state(destination,policy,strix::ModelContract::qwen38_flash_next(),
            layout,binding,session,request,result,metadata,retained,
            [&,retained](const auto& allocation,std::uint64_t offset,std::span<std::byte> output) {
                ++copies;
                require(retained.use_count()>=2,"real source owner not retained through read");
                require(output.size()<=65536U && offset<=allocation.bytes && output.size()<=allocation.bytes-offset,
                        "state copy chunk exceeded admitted allocation");
                if(fail_read && copies==2U) throw std::runtime_error("injected partial read");
                const auto& buffer=allocation.storage==strix::SessionStateStorage::host?retained->host:retained->device;
                std::memcpy(output.data(),buffer.data()+allocation.offset+offset,output.size());
            },[&] { ++guards; if(fail_guard) throw std::runtime_error("injected stale binding"); },
              [&] { ++drains; },
              [&] {
                  ++publications;
                  if(fail_publication) throw std::runtime_error("injected changed executable/model");
                  if(race_destination) {
                      require(std::filesystem::create_directory(destination),"cannot create competing destination");
                      std::ofstream(destination/"sentinel")<<"unrelated";
                  }
              });
    }
};
void success(const std::filesystem::path& base,unsigned index) {
    Fixture f(base,index); const auto output=f.root/"finished";
    f.run(output);
    require(std::filesystem::is_directory(output) && !std::filesystem::exists(f.root/"finished.partial"),
            "final state publication was not a whole directory move");
    const auto manifest=text(output/"manifest.json");
    require(manifest.find("\"recurrent_storage\":\"head-key-value-v1\"")!=std::string::npos,
            "export omitted the physical recurrent matrix storage layout");
    require(manifest.find("\"consumed_input_tokens\":[1,2,3,4]")!=std::string::npos &&
            manifest.find("\"generated_tokens\":[3,4,5]")!=std::string::npos &&
            manifest.find("\"unconsumed_emitted_token\":5")!=std::string::npos &&
            manifest.find("\"next_position\":4")!=std::string::npos &&
            manifest.find("\"logits_exported\":false")!=std::string::npos,
            "manifest confused consumed inputs and emitted bonus");
    constexpr std::array names{"attention-kv","recurrent-matrix","gdn-convolution","ple-convolution",
                               "sparse-index","ngram-history","position-history"};
    for(std::size_t i=0;i<7U;++i) {
        const auto& allocation=f.layout.allocations[i];
        const auto& bytes=allocation.storage==strix::SessionStateStorage::host?f.owner->host:f.owner->device;
        const auto actual=text(output/(std::string(names[i])+".raw"));
        require(actual.size()==allocation.bytes &&
                std::memcmp(actual.data(),bytes.data()+allocation.offset,actual.size())==0,"component bytes are incomplete");
        const auto sha=strix::model_content_identity_hex(strix::sha256_file_content(output/(std::string(names[i])+".raw")));
        require(manifest.find(sha)!=std::string::npos,"component hash not bound in manifest");
    }
    require(f.copies==9U && f.drains==1U && f.publications==1U,"chunking/guard/publication contract changed");
    const auto before=text(output/"manifest.json"); rejects([&] { f.run(output); });
    require(text(output/"manifest.json")==before,"existing complete export was overwritten");
}
void failures(const std::filesystem::path& base,unsigned start) {
    for(unsigned mode=0;mode<10U;++mode) {
        Fixture f(base,start+mode); const auto output=f.root/"finished";
        if(mode==0U) f.fail_read=true;
        if(mode==1U) f.fail_guard=true;
        if(mode==2U) f.fail_publication=true;
        if(mode==3U) ++f.result.final_position;
        if(mode==4U) f.session.health=strix::SerialSessionHealth::poisoned;
        if(mode==5U) f.layout.allocations.pop_back();
        if(mode==6U) f.layout.allocations[1].component=f.layout.allocations[0].component;
        if(mode==7U) f.layout.allocations[1].offset=f.layout.allocations[0].offset;
        if(mode==8U) f.metadata.executable_sha256="missing";
        if(mode==9U) {
            require(std::filesystem::create_directory(f.root/"finished.partial"),"cannot prepare stale partial");
            std::ofstream(f.root/"finished.partial/sentinel")<<"keep";
        }
        rejects([&] { f.run(output); });
        require(!std::filesystem::exists(output),"failed export published a partial destination");
        require(f.drains>=1U,"failed export released owners before drain");
        if(mode==0U) require(f.copies==2U && std::filesystem::exists(f.root/"finished.partial"),
                            "partial-read failure was not exercised");
        if(mode==2U) require(f.copies==9U && f.publications==1U &&
                            std::filesystem::exists(f.root/"finished.partial/manifest.json"),
                            "post-copy binding failure was not exercised");
        if(mode>=3U) require(!f.copies,"invalid contract reached a state read");
        if(mode==9U) require(text(f.root/"finished.partial/sentinel")=="keep","stale partial was overwritten");
    }
    Fixture budget(base,start+10U); const auto output=budget.root/"finished";
    rejects([&] { budget.run(output,{1}); });
    require(!budget.copies && !std::filesystem::exists(output),"failed budget read or published state");
    Fixture race(base,start+11U);
    race.race_destination=true;
    rejects([&] { race.run(race.root/"finished"); });
    require(text(race.root/"finished/sentinel")=="unrelated" &&
            !std::filesystem::exists(race.root/"finished/manifest.json") &&
            std::filesystem::exists(race.root/"finished.partial/manifest.json"),
            "atomic publication overwrote a competing destination");
}
void boundaries(const std::filesystem::path& base) {
    unsigned index=40;
    constexpr std::array cases{std::array<std::uint64_t,3>{1,15,16},std::array<std::uint64_t,3>{3,3,16},
                               std::array<std::uint64_t,3>{5,15,2},std::array<std::uint64_t,3>{5,15,3}};
    for(const auto settings:cases) {
        Fixture f(base,index++,settings[0],static_cast<std::uint32_t>(settings[1]),settings[2]);
        f.run(f.root/"finished");
        const auto manifest=text(f.root/"finished/manifest.json");
        require(f.result.steps_executed==f.prefix.size()+f.result.generated_tokens.size()-1U &&
                manifest.find("\"unconsumed_emitted_token\":"+std::to_string(f.result.generated_tokens.back()))!=std::string::npos,
                "EOS/capacity/single-output frontier differs from production controller");
    }
}
void zero_emission(const std::filesystem::path& base) {
    for (unsigned mode=0;mode<8U;++mode) {
        Fixture f(base,60U+mode,0,15,mode==1U ? 2U : 16U);
        const auto output=f.root/"finished";
        if (mode==2U) f.result.generated_tokens.push_back(3U);
        if (mode==3U) f.result.stop_reason=strix::SerialRequestStopReason::context_capacity;
        if (mode==4U) --f.result.steps_executed;
        if (mode==5U) ++f.session.next_position;
        if (mode==6U) f.result.restored_prefix_tokens=1U;
        if (mode==7U) f.session.state.valid=strix::PersistentState::attention_kv;
        if (mode>=2U) {
            rejects([&] { f.run(output); });
            require(!f.copies && !std::filesystem::exists(output),
                    "invalid zero-emission frontier reached state reads/publication");
            continue;
        }
        f.run(output);
        const auto manifest=text(output/"manifest.json");
        require(manifest.find("\"consumed_input_tokens\":[1,2]")!=std::string::npos &&
                manifest.find("\"generated_tokens\":[]")!=std::string::npos &&
                manifest.find("\"unconsumed_emitted_token\":null")!=std::string::npos &&
                manifest.find("\"bonus_pending\":false")!=std::string::npos &&
                manifest.find("\"next_position\":2")!=std::string::npos && f.copies==9U,
                "zero-emission export forged a bonus or omitted full component bytes");
        constexpr std::array names{"attention-kv","recurrent-matrix","gdn-convolution","ple-convolution",
                                   "sparse-index","ngram-history","position-history"};
        for (std::size_t i=0;i<names.size();++i) {
            const auto& allocation=f.layout.allocations[i];
            const auto& bytes=allocation.storage==strix::SessionStateStorage::host?f.owner->host:f.owner->device;
            const auto actual=text(output/(std::string(names[i])+".raw"));
            require(actual.size()==allocation.bytes &&
                    std::memcmp(actual.data(),bytes.data()+allocation.offset,actual.size())==0,
                    "zero-emission export did not retain every component byte");
        }
    }
}
void cli(const std::filesystem::path& base) {
    const auto parse=[](std::vector<std::string> args) {
        std::vector<char*> argv;
        for(auto& arg:args) argv.push_back(arg.data());
        return strix::decode_cli::parse_options(static_cast<int>(argv.size()),argv.data());
    };
    const auto fresh=(base/"cli-output").string();
    auto options=parse({"decoder","unopened.gguf","1","--generate","3","--validation-final-state",fresh});
    require(options.validation_final_state_path==fresh && !std::filesystem::exists(fresh) &&
            !std::filesystem::exists(fresh+".partial"),"CLI admission created export storage");
    options=parse({"decoder","unopened.gguf","1","--generate","3","--gpu-greedy-token","--validation-final-state",fresh});
    require(options.gpu_greedy_token && options.raw_recording_path.empty(),"final-state validation enabled GPU RAW");
    for(const auto& extra:std::vector<std::vector<std::string>>{
        {"--record-raw","raw-output"},{"--reference","unused.json"},{"--prefix-cache-roundtrip"},
        {"--execution-mode","graph"},{"--integer-lm-head","a8-iu8"}}) {
        bool rejected=false;
        try {
            std::vector<std::string> args{"decoder","unopened.gguf","1","--generate","3","--validation-final-state",fresh};
            args.insert(args.end(),extra.begin(),extra.end()); (void)parse(std::move(args));
        } catch(const std::invalid_argument& error) {
            const std::string_view expected=extra.front()=="--reference"
                ? "--generate and --reference are separate execution modes"
                : "final state validation requires eager generation without RAW, reference, roundtrip or integer head";
            rejected=std::string_view(error.what())==expected;
        }
        require(rejected,"CLI exclusion failed for a different/vacuous reason");
    }
    rejects([&] { (void)parse({"decoder","unopened.gguf","1","--validation-final-state",fresh}); });
    rejects([&] { (void)parse({"decoder","unopened.gguf","1","--generate","3","--validation-final-state-max-bytes","1"}); });
    rejects([&] { (void)parse({"decoder","unopened.gguf","1","--generate","3","--validation-final-state",fresh,
                               "--validation-final-state",fresh}); });
}
} // namespace
int main() try {
    const auto unique=std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root=std::filesystem::temp_directory_path()/("strix-final-state-export-"+unique);
    require(std::filesystem::create_directory(root),"cannot reserve final state fixture root");
    // Preserve failures for diagnosis; remove only this explicitly reserved root
    // after every source fixture has succeeded.
    success(root,0); failures(root,10); boundaries(root); zero_emission(root); cli(root);
    require(std::filesystem::canonical(root.parent_path())==
            std::filesystem::canonical(std::filesystem::temp_directory_path()),"fixture cleanup escaped reserved parent");
    std::filesystem::remove_all(root);
    std::cout<<"final state export portable source fixtures passed; no native evidence\n";
    return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
