#include "../apps/qwen_effective_config.hpp"
#include <iostream>

using namespace strix;
using namespace strix::decode_cli;
namespace {
void require(bool v,const char* why) {if(!v)throw std::runtime_error(why);}
void env(const char* key,const std::string& value) {
#ifdef _WIN32
    _putenv_s(key,value.c_str());
#else
    if(value.empty())unsetenv(key);else setenv(key,value.c_str(),1);
#endif
}
Options parse(std::vector<std::string> args) {
    std::vector<char*> pointers;
    for(auto& arg:args)pointers.push_back(arg.data());
    return parse_options(static_cast<int>(pointers.size()),pointers.data());
}
template<class F> void refuses(F f) {
    bool threw=false;try{f();}catch(const std::exception&){threw=true;}
    require(threw,"expected refusal");
}
// A header-only GGUF v3 (no tensors) declaring the drafter-target identity keys discovery reads.
std::string target_gguf(std::uint32_t layers) {
    std::string out("GGUF");
    const auto u32=[&](std::uint32_t v) { for(int i=0;i<4;++i)out.push_back(static_cast<char>((v>>(8*i))&255U)); };
    const auto u64=[&](std::uint64_t v) { for(int i=0;i<8;++i)out.push_back(static_cast<char>((v>>(8*i))&255U)); };
    const auto str=[&](const std::string& v) { u64(v.size());out+=v; };
    u32(3);u64(0);u64(3);
    str("general.architecture");u32(8);str("qwen4exp");
    str("qwen4exp.block_count");u32(4);u32(layers);
    str("qwen4exp.embedding_length");u32(4);u32(2560);
    out.resize((out.size()+31U)/32U*32U,char{0});
    return out;
}
void write(const std::filesystem::path& p,const std::string& text) {
    std::ofstream f(p,std::ios::binary);f<<text;
    if(!f)throw std::runtime_error("fixture write failed");
}
}
int main(int argc,char** argv) try {
    require(argc==2,"recipe argument");
    const auto dir=std::filesystem::temp_directory_path()/("strix-mtp-bundle-test-"+
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::error_code ec;std::filesystem::remove_all(p,ec);}} cleanup{dir};
    env("STRIX_HASH_CACHE",(dir/"hash-cache").string());
    env("STRIX_QWEN_YARN_MTP","");env("STRIX_QWEN_ROPE_YARN","");
    std::ifstream recipe(argv[1]);
    auto json=serve::parse_json_unique(std::string((std::istreambuf_iterator<char>(recipe)),{}));
    auto opts=*json.find("options");
    // Same shipped options, with a tiny byte-pinned sidecar for host-only testing.
    const auto sidecar=dir/"sidecar.gguf";write(sidecar,"abc");
    const std::string sha="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    opts.set("--mtp-sidecar","sidecar.gguf");opts.set("--mtp-sidecar-sha256",sha);
    opts.set("--mtp-sidecar-bytes","3");
    opts.set("--mtp-policy-id",1);
    json.set("options",opts);
    const auto bundle=dir/"target.gguf.mtp-bundle.json";write(bundle,json.dump());
    std::vector<std::string> base{"decode",(dir/"target.gguf").string(),"1","2","3","4","5","6","7","8","9",
        "10","11","12","13","14","15","16","17","18","--rope-yarn","4","--context","1048576","--generate","256"};
    const auto bundled=parse(base);
    require(bundled.mtp.enabled && bundled.mtp_bundle_path==bundle.string(),"automatic discovery");
    require(bundled.mtp.runtime_automatic,"discovered bundle runtime budget grows with context");
    require(bundled.yarn_decode_source==bundle.string(),"bundle source");
    auto import_args=base;
    import_args.insert(import_args.end(),{"--rtx-state-import-file","prefix.json"});
    const auto imported=parse(import_args);
    require(imported.mtp.enabled && imported.mtp_bundle_path==bundle.string() && imported.rtx_state_import_file=="prefix.json",
        "CLI file import preserves bundle admission");

    EffectiveConfig inspected;inspected.options=bundled;inspected.admitted=true;
    auto receipt=serve::parse_json_unique(effective_config_json(inspected));
    require(receipt.find("yarn_decode_route")->find("source")->as_string()==bundle.string(),"effective bundle source");
    require(receipt.find("yarn_mtp_assets")->find("golden")->as_string()=="pending-self-generation","cold bundle receipt");
    auto selected=bundled;
    const auto model=ModelContract::qwen38_flash_next();
    env("XDG_CACHE_HOME",(dir/"admission-cache").string());
    select_mtp_bundle_golden(selected,model);
    require(selected.mtp.golden.empty(),"cold admission leaves golden pending");
    const auto admission_id=mtp_bundle_golden_identity(selected,model);
    const auto admission_file=mtp_golden_cache_dir()/(admission_id+".golden");
    std::filesystem::create_directories(admission_file.parent_path());write(admission_file,"cached");
    select_mtp_bundle_golden(selected,model);
    require(selected.mtp.golden==admission_file.string(),"admission selects cached exact identity");
    require(selected.mtp.golden_sha256==model_content_identity_hex(sha256_file_content(admission_file,FileHashPolicy::cached)) &&
        mtp_sha256_text(selected.mtp.golden_sha256),"admission pins the selected golden digest for the serving check");
    ++selected.context_tokens;select_mtp_bundle_golden(selected,model);
    require(selected.mtp.golden.empty() && selected.mtp.golden_sha256.empty(),"changed geometry misses cache and clears the pin");
    env("XDG_CACHE_HOME","");
    auto explicit_bundle_args=base;
    explicit_bundle_args.push_back("--mtp");
    for(const auto& [key,value]:opts.as_object()) {
        explicit_bundle_args.push_back(key);
        if(value.is_bool())continue;
        auto v=value.is_string()?value.as_string():value.number_source();
        if(key=="--mtp-sidecar")v=(dir/v).string();
        if(key=="--mtp-draft-vocab")v=v.starts_with("auto:")?"auto:"+(dir/v.substr(5)).lexically_normal().string():(dir/v).string();
        explicit_bundle_args.push_back(v);
    }
    explicit_bundle_args.insert(explicit_bundle_args.end(),{"--mtp-golden","reference.golden","--mtp-golden-sha256",sha});
    const auto explicit_options=parse(explicit_bundle_args);
    require(!explicit_options.mtp.runtime_automatic,"explicit CLI runtime budget is a ceiling");
    inspected.options=explicit_options;
    receipt=serve::parse_json_unique(effective_config_json(inspected));
    require(receipt.find("yarn_decode_route")->find("source")->as_string()=="explicit","effective explicit source");
#define SAME(field) require(bundled.mtp.field==explicit_options.mtp.field,"bundle/explicit " #field)
    SAME(sidecar);SAME(sidecar_sha256);SAME(sidecar_logical_name);SAME(sidecar_bytes);
    SAME(target_sha256);SAME(target_content);SAME(repository);SAME(revision);SAME(source_path);
    SAME(converter_revision);SAME(converter_sha256);SAME(norm_converter_sha256);SAME(conversion);
    SAME(normalized_rounding);SAME(projection_rounding);SAME(sidecar_residency);
    SAME(header_bytes);SAME(sidecar_retained_bytes);SAME(sidecar_setup_bytes);SAME(golden_max_bytes);
    SAME(replay_host_bytes);SAME(runtime_bytes);SAME(target_capture_bytes);SAME(policy_identity);
    SAME(max_hidden_error);SAME(max_logit_error);SAME(spec_min_p);SAME(calibration_rows);SAME(draft_chain);
    SAME(draft_vocab);SAME(draft_context_vocab);
    SAME(draft_q4);SAME(candidate_depth);SAME(window_rows);SAME(window_rows_set);SAME(window_prefix);
    SAME(window_hyper_mix);SAME(window_qsa_rows);SAME(window_expert_union);SAME(window_expert_union_bytes);
    SAME(window_arena_bytes);SAME(window_capture_bytes);SAME(window_checkpoint_bytes);SAME(window_total_bytes);
    SAME(window_history_intervals);
#undef SAME
    // Exercise the guide's pins and calibration geometry without the HIP-only organization flags.
    std::vector<std::string> guide_args{"decode", (dir/"target.gguf").string()};
    for (int token = 1; token <= 32; ++token) guide_args.push_back(std::to_string(token));
    const auto pins_begin = std::find(explicit_bundle_args.begin(), explicit_bundle_args.end(), "--mtp");
    const auto pins_end = std::find(pins_begin, explicit_bundle_args.end(), "--mtp-golden");
    guide_args.insert(guide_args.end(), pins_begin, pins_end);
    guide_args.insert(guide_args.end(), {"--generate", "64", "--context", "33280", "--mtp-calibration-rows", "8"});
    auto guide_dump_args = guide_args;
    guide_dump_args.insert(guide_dump_args.end(), {"--known-prefix-window", "--known-prefix-full-rows",
        "--known-prefix-prefill-rows", "8192", "--known-prefix-workspace-budget-bytes", "8589934592",
        "--known-prefix-host-metadata-budget-bytes", "2147483648", "--known-prefix-uma-budget-bytes", "128849018880",
        "--known-prefix-host-headroom-bytes", "4294967296", "--mtp-golden-dump", "qwen-t8192.golden"});
    const auto guide_dump = parse(guide_dump_args);
    require(mtp_tiled_prompt(guide_dump) && guide_dump.known_prefix_prefill_rows == 8192 &&
        guide_dump.mtp.golden_dump == "qwen-t8192.golden" && !guide_dump.http_serve, "guide uses tiled golden dump");
    guide_args.insert(guide_args.end(), {"--http-serve", "--http-batch", "0", "--http-bind", "127.0.0.1",
        "--http-port", "8082", "--mtp-golden", "qwen-t8192.golden", "--mtp-golden-sha256", sha});
    const auto guide_served = parse(guide_args);
    require(http_prefill_pending(guide_served) && guide_served.http_batch_explicit && !guide_served.http_batch &&
        guide_served.mtp.golden == guide_dump.mtp.golden_dump && mtp_calibration_rows(guide_served) == mtp_calibration_rows(guide_dump) &&
        mtp_calibration_rows(guide_served) == 8 && guide_served.mtp.sidecar_logical_name == guide_dump.mtp.sidecar_logical_name,
        "guide uses automatic single HTTP owner with matching MTP pins and calibration geometry");

    const auto stats=file_hash_cache_stats();verify_mtp_bundle_sidecar(sidecar.string(),sha,3);
    verify_mtp_bundle_sidecar(sidecar.string(),"BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",3);
#if defined(__linux__)
    require(file_hash_cache_stats().hits>stats.hits,"persistent hash cache used");
#else
    (void)stats;
#endif
    auto explicit_bundle=base;explicit_bundle.insert(explicit_bundle.end(),{"--mtp-bundle",bundle.string()});
    require(parse(explicit_bundle).mtp.enabled && !parse(explicit_bundle).mtp.runtime_automatic,
        "explicit bundle retains its runtime ceiling");
    explicit_bundle.push_back("--mtp");refuses([&]{parse(explicit_bundle);});
    explicit_bundle.pop_back();explicit_bundle.insert(explicit_bundle.end(),{"--mtp-policy-id","1"});
    refuses([&]{parse(explicit_bundle);});
    env("STRIX_QWEN_YARN_MTP","0");
    inspected.options=parse(base);
    require(!inspected.options.mtp.enabled,"env zero serial");
    receipt=serve::parse_json_unique(effective_config_json(inspected));
    require(receipt.find("yarn_decode_route")->find("source")->as_string()=="env","effective env source");
    auto missing_bundle=base;missing_bundle.insert(missing_bundle.end(),{"--mtp-bundle",(dir/"missing.json").string()});
    require(!parse(missing_bundle).mtp.enabled,"env zero bypasses bundle assets");
    env("STRIX_QWEN_YARN_MTP","1");require(parse(base).mtp.enabled,"env one bundle");
    std::filesystem::remove(bundle);refuses([&]{parse(base);});
    env("STRIX_QWEN_YARN_MTP","");
    std::ostringstream notice;auto* old=std::cerr.rdbuf(notice.rdbuf());
    const auto serial=parse(base);std::cerr.rdbuf(old);
    require(!serial.mtp.enabled && notice.str()=="YaRN: no MTP bundle found; using serial decode\n","no bundle serial notice once");
    write(bundle,json.dump());
    {
        YarnBundleSerialRetry retry;
        require(!parse(base).mtp.enabled,"dump failure retry suppresses automatic MTP");
    }
    require(!yarn_bundle_serial_retry,"retry scope restores resolver");
    std::vector<std::string> native{"decode",(dir/"target.gguf").string(),"1","2","3","--generate","1"};
    require(!parse(native).mtp.enabled,"native ignores adjacent bundle");
    native.insert(native.end(),{"--rope-yarn","off"});
    require(!parse(native).mtp.enabled,"explicit rotary off ignores bundle");
    native.pop_back();native.push_back("4");native.push_back("--baseline");
    require(!parse(native).mtp.enabled,"baseline ignores automatic bundle");
    {
        // The plain production command selects the adjacent bundle; one opt-out flag and one environment switch.
        std::vector<std::string> nd_plain{"decode",(dir/"target.gguf").string()};
        for(int token=1;token<=17;++token)nd_plain.push_back(std::to_string(token));
        nd_plain.insert(nd_plain.end(),{"--generate","8"});
        const auto nd_selected=parse(nd_plain);
        require(nd_selected.mtp.enabled && nd_selected.mtp_native_bundle && nd_selected.mtp.calibration_rows==16U &&
            !nd_selected.route_owned_beyond_mtp,"plain command selects the adjacent bundle");
        require(nd_selected.mtp.runtime_automatic,"native default bundle also sizes runtime by context");
        auto nd_opted_out=nd_plain;nd_opted_out.push_back("--no-mtp");
        require(!parse(nd_opted_out).mtp.enabled,"--no-mtp keeps plain decode");
        env("STRIX_MTP","0");require(!parse(nd_plain).mtp.enabled,"STRIX_MTP=0 keeps plain decode");env("STRIX_MTP","");
        auto nd_owned=nd_plain;nd_owned.push_back("--gpu-greedy-token");
        require(!parse(nd_owned).mtp.enabled,"a route option keeps plain decode");
        const auto nd_moved=dir/"nd_moved.json";std::filesystem::rename(bundle,nd_moved);
        std::ostringstream nd_line;auto* nd_previous=std::cerr.rdbuf(nd_line.rdbuf());
        const auto nd_fallback=parse(nd_plain);std::cerr.rdbuf(nd_previous);
        std::filesystem::rename(nd_moved,bundle);
        const auto nd_text=nd_line.str();
        require(!nd_fallback.mtp.enabled && nd_text.starts_with("MTP default (qwen) unavailable; using plain decode: ") &&
            std::count(nd_text.begin(),nd_text.end(),'\n')==1,"missing bundle is one notice and plain decode");
    }
    {
        // Discovery of a registry drafter beside the model (no bundle json): the shipped MTP/ layout, by size.
        const auto model_dir=dir/"profile-model";std::filesystem::create_directories(model_dir/"MTP");
        const auto pmodel=model_dir/"target.gguf";write(pmodel,target_gguf(48));
        const auto base_file=model_dir/"MTP"/"mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
        std::vector<std::string> plain{"decode",pmodel.string()};
        for(int token=1;token<=17;++token)plain.push_back(std::to_string(token));
        plain.insert(plain.end(),{"--generate","8"});
        const auto defer=[&](std::vector<std::string> args) {
            std::vector<char*> pointers;
            for(auto& arg:args)pointers.push_back(arg.data());
            return parse_options(static_cast<int>(pointers.size()),pointers.data(),true,{},DecodeAssets::defer);
        };
        const auto capture=[&](const std::vector<std::string>& args,std::string& text) {
            std::ostringstream out;auto* previous=std::cerr.rdbuf(out.rdbuf());
            try {auto o=defer(args);std::cerr.rdbuf(previous);text=out.str();return o;}
            catch(...) {std::cerr.rdbuf(previous);throw;}
        };
        std::string text;
        require(!capture(plain,text).mtp.enabled && text.find("no registry drafter beside the model")!=std::string::npos,
            "no sidecar: one fallback notice naming the registry layout");
        {
            // An explicit --http-batch without a drafter names the reason, not only the missing windows.
            auto no_drafter=plain;no_drafter.erase(no_drafter.end()-2,no_drafter.end());
            no_drafter.insert(no_drafter.end(),{"--generate","64","--http-batch","8","--http-serve"});
            std::string reason;
            try {(void)capture(no_drafter,text);} catch(const std::exception& error) {reason=error.what();}
            require(reason.find("--http-batch: missing prefix windows")!=std::string::npos &&
                reason.find("the MTP default was not selected: assets-missing")!=std::string::npos,
                "served command without a drafter names why MTP was not selected");
        }
        write(base_file,"short");
        require(!capture(plain,text).mtp.enabled && text.find("expects 2786568256")!=std::string::npos,"wrong size is not taken and is named");
        std::filesystem::resize_file(base_file,2786568256ULL);
        // Placement alone binds nothing: a model of another identity keeps plain decode and names the draft it left out.
        write(pmodel,target_gguf(64));
        require(!capture(plain,text).mtp.enabled && text.find("draft base targets qwen4exp 48 x 2560")!=std::string::npos,
            "a draft for another target is not taken and is named");
        write(pmodel,target_gguf(48));
        const auto picked=capture(plain,text);
        require(picked.mtp.enabled && picked.mtp_native_bundle && picked.mtp.calibration_rows==16U &&
            picked.mtp.sidecar==base_file.string() && picked.mtp.sidecar_logical_name==base_file.filename().string(),"base sidecar under MTP/");
        require(picked.mtp_bundle_path==(model_dir/"mtp-profile-base.bundle").string() && picked.mtp.runtime_automatic,"profile bundle path");
        require(picked.mtp.spec_min_p==0.85 && picked.mtp.sidecar_bytes==2786568256ULL && picked.mtp.sidecar_sha256.starts_with("5ff540") &&
            picked.mtp.window_rows==8U && picked.mtp.draft_vocab.empty() && picked.mtp.draft_context_vocab==4096U,"base profile pins and min-p");
        require(text.starts_with("mtp: drafter "+base_file.string()) && text.find("registry drafter 'base' at MTP/")!=std::string::npos &&
            std::count(text.begin(),text.end(),'\n')==1,"one line names the file and how");
        {
            // A second consumer of the bundle (golden identity, shipped/cached golden selection) needs no bundle file:
            // the synthesized options travel with the parsed options and give one identity.
            const auto contract=ModelContract::qwen38_flash_next();
            require(!std::filesystem::exists(picked.mtp_bundle_path) && !picked.mtp_bundle_synthesized.empty(),
                "registry bundle has no file and carries its options");
            const auto first_identity=mtp_bundle_golden_identity(picked,contract);
            auto copied=picked;
            require(first_identity==mtp_bundle_golden_identity(copied,contract) &&
                first_identity==mtp_bundle_golden_identity(defer(plain),contract),"second consumer gets the same identity");
            auto golden_selected=picked;select_mtp_bundle_golden(golden_selected,contract);
            require(golden_selected.mtp.golden.empty(),"no cached golden for the synthetic identity: self-generation pending");
            require(picked.mtp_bundle_synthesized.find("--mtp-sidecar"+std::string("\n")+base_file.string())!=std::string::npos,
                "synthesized options hold the sidecar");
        }
        // The same arguments again print nothing new (a server prepares and launches from one command line).
        require(capture(plain,text).mtp.enabled && text.empty(),"each drafter line prints once per process");
        {
            // The launchers (strix run / strix-serve) hand the engine MODEL --prompt-ids FILE --generate N plus the tier
            // (--profile) and the KV storage (--kv): none of them owns a route, so the documented MTP default stays.
            std::string ids;
            for(int token=1;token<=40;++token)ids+=std::to_string(token)+(token%8?" ":"\n");
            const auto ids_file=model_dir/"prompt.ids";write(ids_file,ids);
            const auto few_file=model_dir/"few.ids";write(few_file,"1 2 3 4 5\n");
            for(const std::string tier:{"exact","fast","turbo"}) {
                const std::vector<std::string> launched{"decode",pmodel.string(),"--prompt-ids",ids_file.string(),"--generate","128",
                    "--profile",tier,"--kv","bf16"};
                const auto run=capture(launched,text);
                require(run.mtp.enabled && run.mtp.runtime_automatic && run.mtp.window_prefix && run.mtp.draft_chain,
                    "launcher CLI command (prompt file, tier, KV) keeps the MTP default for a tier");
                auto inline_form=launched;inline_form[6]="--profile="+tier;inline_form.erase(inline_form.begin()+7);
                inline_form[7]="--kv=bf16";inline_form.pop_back();
                require(capture(inline_form,text).mtp.enabled,"inline --profile=/--kv= spelling keeps the MTP default for a tier");
            }
            const std::vector<std::string> short_prompt{"decode",pmodel.string(),"--prompt-ids",few_file.string(),"--generate","128","--profile","fast"};
            require(!capture(short_prompt,text).mtp.enabled,"a prompt file under 17 ids keeps plain decode");
            const std::vector<std::string> owned{"decode",pmodel.string(),"--prompt-ids",ids_file.string(),"--generate","128","--profile","fast",
                "--gpu-greedy-token"};
            require(!capture(owned,text).mtp.enabled,"a route option next to the tier still owns the route");
            // README serving quick start: calibration prompt file, --http-batch 8, tier and KV from the launcher.
            const std::vector<std::string> served_launch{"strix-serve",pmodel.string(),"--prompt-ids",ids_file.string(),"--generate","64",
                "--http-batch","8","--profile","fast","--kv","bf16","--http-serve"};
            const auto served_run=capture(served_launch,text);
            require(served_run.http_serve && served_run.mtp.enabled && served_run.mtp.window_prefix && served_run.mtp.draft_chain &&
                served_run.mtp.window_rows_set==std::vector<std::uint32_t>{2U,4U,8U} && served_run.http_batch_explicit,
                "served quick start resolves tiled MTP serving (prefix windows, draft chain, rows 2,4,8)");
        }
        std::filesystem::create_directories(model_dir/"draft-vocab");
        require(defer(plain).mtp.draft_vocab.find((model_dir/"draft-vocab").string())!=std::string::npos,"draft-vocab beside the model is used");
        // The recommended drafter wins when two exist; a logical-name copy beside the model is found too.
        const auto distill=model_dir/"mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf";write(distill,"x");
        std::filesystem::resize_file(distill,2786568256ULL);
        const auto both=capture(plain,text);
        require(both.mtp.sidecar==distill.string() && both.mtp.spec_min_p==0.7 && text.find("by logical name beside the model")!=std::string::npos,
            "recommended drafter preferred, logical name beside the model");
        // Order: an adjacent bundle json beats the registry; explicit --mtp-bundle beats both.
        auto with_json=json;with_json.set("options",opts);
        const auto model_json=model_dir/"target.gguf.mtp-bundle.json";write(model_json,with_json.dump());
        std::filesystem::copy_file(sidecar,model_dir/"sidecar.gguf",std::filesystem::copy_options::overwrite_existing);
        require(defer(plain).mtp_bundle_path==model_json.string(),"bundle json beside the model beats the registry drafter");
        auto profile_flag_args=plain;profile_flag_args.insert(profile_flag_args.end(),{"--mtp-bundle",bundle.string()});
        require(defer(profile_flag_args).mtp_bundle_path==bundle.string(),"--mtp-bundle beats everything");
        std::filesystem::remove(model_json);
        // YaRN stays serial exactly as before: no registry drafter, no bundle json, no notice.
        auto yarn_args=plain;yarn_args.insert(yarn_args.end(),{"--rope-yarn","4","--context","1048576"});
        std::string yarn_text;const auto yarned=capture(yarn_args,yarn_text);
        require(!yarned.mtp.enabled && yarned.mtp_bundle_path.empty() && yarn_text.find("mtp: drafter")==std::string::npos,
            "YaRN does not discover a registry drafter");
        // Off switches keep working with a discovered drafter.
        auto off_args=plain;off_args.push_back("--no-mtp");require(!defer(off_args).mtp.enabled,"--no-mtp with a discovered drafter");
        env("STRIX_MTP","0");require(!defer(plain).mtp.enabled,"STRIX_MTP=0 with a discovered drafter");env("STRIX_MTP","");
        // The HTTP server takes the same drafter from the same command line (calibration ids first).
        auto serve_args=plain;serve_args.erase(serve_args.end()-2,serve_args.end());
        serve_args.insert(serve_args.end(),{"--generate","8","--http-batch","0","--http-port","8083","--http-serve"});
        const auto served=defer(serve_args);
        require(served.http_serve && served.mtp.enabled && served.mtp.sidecar==distill.string(),"server discovers the drafter like the CLI");
        auto serve_short=serve_args;serve_short.erase(serve_short.begin()+2+10);
        bool short_ok=false;
        try {short_ok=!defer(serve_short).mtp.enabled;} catch(const std::exception&) {short_ok=true;}
        require(short_ok,"fewer than 17 calibration ids do not select MTP on the server");
        // The registry bundle equals the shipped recipe for the same draft (budgets are shared data).
        MtpProfileBundleInputs inputs;inputs.sidecar="mtp-Qwen3.8-Flash-Next-shared-distillG-Q8_0.gguf";
        std::ifstream shipped_file(argv[1]);
        const auto shipped=serve::parse_json_unique(std::string((std::istreambuf_iterator<char>(shipped_file)),{}));
        const auto& shipped_options=shipped.find("options")->as_object();
        inputs.target_sha256=shipped.find("options")->find("--mtp-target-sha256")->as_string();
        inputs.target_content=shipped.find("options")->find("--mtp-target-content")->as_string();
        inputs.vocabulary_dir="draft-vocab";
        const auto built=profile_mtp_bundle_arguments(*find_mtp_draft_profile("distill-g"),inputs);
        std::map<std::string,std::string> built_map;
        for(std::size_t k=1;k<built.argv.size();++k) {
            const auto& key=built.argv[k];
            const bool value=k+1<built.argv.size() && !built.argv[k+1].starts_with("--mtp-");
            built_map[key]=value?built.argv[++k]:"true";
        }
        require(built_map.size()==shipped_options.size(),"registry bundle has the recipe's keys");
        for(const auto& [key,value]:shipped_options) {
            const std::string expected=value.is_bool()?"true":value.is_string()?value.as_string():value.number_source();
            require(built_map.count(key) && (key=="--mtp-draft-vocab"?built_map[key]=="auto:draft-vocab":built_map[key]==expected),
                "registry bundle equals the recipe option");
        }
    }
    write(sidecar,"bad");refuses([&]{parse(base);});
    const auto cache=dir/"cache";int dumps=0;
    const auto fake=[&](const std::filesystem::path& temp){
        ++dumps;require(temp.extension().string().find("tmp-")!=std::string::npos,"temporary dump");
        require(!std::filesystem::exists(cache/"id.golden"),"not published during dump");write(temp,"v3");};
    const auto cached=ensure_mtp_bundle_golden(bundle,"id",fake,[&]{return cache;});
    require(dumps==1 && cached==cache/"id.golden","cache miss publishes");
    require(ensure_mtp_bundle_golden(bundle,"id",fake,[&]{return cache;})==cached && dumps==1,"cache hit");
    std::filesystem::create_directories(dir/"goldens");write(dir/"goldens/id.golden","shipped");
    require(ensure_mtp_bundle_golden(bundle,"id",fake,[&]{return cache;})==dir/"goldens/id.golden","shipped priority");
    refuses([&]{ensure_mtp_bundle_golden(bundle,"failed",[&](const auto& temp){write(temp,"partial");throw std::runtime_error("fake dump failure");},[&]{return cache;});});
    require(!std::filesystem::exists(cache/"failed.golden"),"failed dump not published");
    for(const auto& entry:std::filesystem::directory_iterator(cache))require(entry.path()==cached,"temporary cleaned");
    refuses([&]{ensure_mtp_bundle_golden(bundle,"rename-failure",[&](const auto& temp){
        write(temp,"v3");std::filesystem::create_directory(cache/"rename-failure.golden");
    },[&]{return cache;});});
    for(const auto& entry:std::filesystem::directory_iterator(cache))
        require(entry.path().filename().string().find(".tmp-")==std::string::npos,"rename failure cleans temp");
    write(dir/"blocked","file");
    const auto serial_on_failure=[&](const std::function<void()>& attempt) {
        bool fell_back=false;
        try {with_mtp_bundle_golden_errors(attempt);}
        catch(const YarnBundleCalibrationFailure& error) {
            env("STRIX_QWEN_YARN_MTP","1");
            const auto status=retry_mtp_bundle_serial(error,[&] {
                fell_back=true;require(!parse(base).mtp.enabled,"failed calibration uses fresh serial route");return 0;
            });
            require(status==0,"serial fallback succeeds");env("STRIX_QWEN_YARN_MTP","");
        }
        require(fell_back && !yarn_bundle_serial_retry,"failure fallback and scope reset");
    };
    serial_on_failure([&]{ensure_mtp_bundle_golden(bundle,"blocked",fake,[&]{return dir/"blocked/subdir";});});
    serial_on_failure([&]{ensure_mtp_bundle_golden(bundle,"dump-failed",[](const auto&){throw std::runtime_error("fake dump");},[&]{return cache;});});
    // Admission, preflight and bridge setup use the same retry seam. Fresh serial
    // resolution must see the caller's environment, not the failed MTP defaults.
    write(sidecar,"abc");
    {
        MtpBundleSetupRetry state;
        state.selected=true;
        unsigned attempts=0;
        const auto no_safety_retry=[&]<class Error>(const Error& failure,const char* identity) {
            bool caught=false;
            try {
                with_mtp_bundle_golden_errors([&] {
                    ensure_mtp_bundle_golden(bundle,identity,[&](const auto& temp) {
                        write(temp,"partial");
                        throw failure;
                    },[&]{return cache;});
                });
            } catch(const Error& error) {
                caught=true;
                require(std::string(error.what())==failure.what(),"golden safety diagnostic survives classification");
                const auto status=retry_mtp_bundle_setup(state,error,[&] {
                    ++attempts;
                    return 17;
                });
                require(!status && attempts==0,
                    "golden safety refusal cannot enter the outer bundle retry");
            }
            require(caught,"golden safety exception retains its type");
            require(!std::filesystem::exists(cache/(std::string(identity)+".golden")),
                "safety-refused golden is not published");
            for(const auto& entry:std::filesystem::directory_iterator(cache)) {
                require(entry.path().filename().string().find(".tmp-")==std::string::npos,
                    "safety refusal cleans temporary golden");
            }
        };
        no_safety_retry(ple_reclaim::DiscardSafetyError("fixture overlap"),"discard-refused");
        no_safety_retry(RegisteredPageLockError("fixture lock failure"),"lock-refused");
        bool pressure_caught=false;
        try {with_mtp_bundle_golden_errors([] {throw YarnMemoryPressure("fixture pressure receipt");});}
        catch(const YarnMemoryPressure& error) {
            pressure_caught=error.receipt()=="fixture pressure receipt";
        }
        require(pressure_caught,"golden classification preserves memory-pressure receipt");
    }
    const MtpBundleSetupRetry identity_test_state;
    detail::prefill_identity_override()={};
    env("STRIX_PREFILL_EXACT_FP16","");env("STRIX_PREFILL_HC_OUTPUT_WMMA","1");
    const auto resolve_identity=[&](const Options& options) {
        return resolve_effective_config({options,requested_prefill_environment()},model,{}, {},"",{},{});
    };
    const auto fresh_serial=[&] {
        YarnBundleSerialRetry retry;
        return resolve_identity(parse(base));
    }();
    require(fresh_serial.prefill_option_mask==kPrefillOptionsReady,"fresh serial retains READY arithmetic");
    for(const auto* failure:{"admission refusal","preflight failure","bridge setup failure","golden dump failure"}) {
        env("STRIX_PREFILL_PIPE","0");env("STRIX_QSA_SCORE_RING_ROWS","");
        MtpBundleSetupRetry state;state.selected=true;
        auto attempt=parse(base);
        require(attempt.mtp.enabled,"automatic YaRN bundle selected before setup");
        set_prefill_identity_override(attempt.prefill_exact_fp16,attempt.prefill_disabled);
        const auto resolved=resolve_identity(attempt);
        set_prefill_identity_override(resolved.options.prefill_exact_fp16,resolved.options.prefill_disabled);
        require(resolved.prefill_option_mask==0U && prefill_options_active()==0U,
            "automatic YaRN MTP resolution installs runtime arithmetic restrictions");
        env("STRIX_PREFILL_PIPE","1");env("STRIX_QSA_SCORE_RING_ROWS","64");
        const auto previous_defaulted=qpf2k::defaulted_items;
        qpf2k::defaulted_items=~previous_defaulted;
        std::ostringstream notices;auto* saved=std::cerr.rdbuf(notices.rdbuf());
        unsigned attempts=0;
        const std::runtime_error error(failure);
        const auto status=retry_mtp_bundle_setup(state,error,[&] {
            ++attempts;
            require(std::string(std::getenv("STRIX_PREFILL_PIPE"))=="0" && !std::getenv("STRIX_QSA_SCORE_RING_ROWS"),"retry restores requested environment");
            require(qpf2k::defaulted_items==previous_defaulted,"retry restores default provenance");
            require(!parse(base).mtp.enabled,"bundle failure retries serial even with adjacent bundle");
            const auto serial_config=resolve_identity(parse(base));
            set_prefill_identity_override(serial_config.options.prefill_exact_fp16,serial_config.options.prefill_disabled);
            require(detail::prefill_identity_override().disabled==fresh_serial.options.prefill_disabled &&
                detail::prefill_identity_override().exact==fresh_serial.options.prefill_exact_fp16 &&
                prefill_options_active()==fresh_serial.prefill_option_mask,
                "failed setup restores the fresh serial override and runtime mask");
            require(serial_config.prefill_option_mask==prefill_options_active() &&
                serial_config.options.resolved_prefill_option_mask==prefill_options_active() &&
                effective_config_json(serial_config).find(prefill_identity_fields(prefill_options_active()))!=std::string::npos,
                "serial retry receipt and executed arithmetic declare the same identity");
            require(!retry_mtp_bundle_setup(state,error,[&] {++attempts;return 1;}),"serial failure cannot recurse");
            return 17;
        });
        std::cerr.rdbuf(saved);
        require(status && *status==17 && attempts==1,"one serial retry returns its status");
        const auto notice_text=notices.str();
        require(notice_text.starts_with("MTP bundle setup unavailable; using serial decode: ") &&
            std::count(notice_text.begin(),notice_text.end(),'\n')==1,"one fallback notice");
        state.target_mutated=true;
        require(!retry_mtp_bundle_setup(state,error,[]{return 1;}),"post-mutation error keeps refusal");
        state.target_mutated=false;state.selected=false;
        require(!retry_mtp_bundle_setup(state,error,[]{return 1;}),"explicit MTP keeps refusal");
    }
    env("STRIX_PREFILL_PIPE","");
    identity_test_state.restore();
    // Bundle parsing/setup errors retain their retry classification. Explicit
    // argv errors remain ordinary refusals and never acquire bundle selection.
    write(sidecar,"bad");
    bool bundle_failure=false;
    try {(void)parse(base);}catch(const YarnBundleCalibrationFailure&){bundle_failure=true;}
    require(bundle_failure,"bundle asset setup failure enters serial retry");
    bool explicit_failure=false;
    auto invalid_explicit=explicit_bundle_args;
    const auto pin_option=std::find(invalid_explicit.begin(),invalid_explicit.end(),"--mtp-sidecar-sha256");
    require(pin_option!=invalid_explicit.end(),"explicit fixture has sidecar pin");
    *(pin_option+1)="invalid";
    try {(void)parse(invalid_explicit);}catch(const YarnBundleCalibrationFailure&){throw;}
    catch(const std::exception&){explicit_failure=true;}
    require(explicit_failure,"explicit MTP asset refusal stays explicit");
    env("XDG_CACHE_HOME",(dir/"xdg").string());require(mtp_golden_cache_dir()==dir/"xdg/strix-engine/goldens","XDG path");
    env("XDG_CACHE_HOME","");env("HOME",dir.string());require(mtp_golden_cache_dir()==dir/".cache/strix-engine/goldens","HOME path");
    write(bundle,"{\"version\":1,\"version\":1,\"options\":{}}");refuses([&]{read_mtp_bundle(bundle);});
    opts.set("--mtp-sidecar",std::filesystem::absolute(sidecar).string());json.set("options",opts);write(bundle,json.dump());
    refuses([&]{read_mtp_bundle(bundle);});
    write(bundle,"{\"version\":1,\"options\":{}}");refuses([&]{read_mtp_bundle(bundle);});
    std::cout<<"MTP bundle CPU contracts passed\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
