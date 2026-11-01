#include "strix/session_state_cache.hpp"
#include "strix/model_content_hash.hpp"
#include "../apps/hip_decode_options.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <cstdlib>

namespace {
void check(bool b,const char* why) { if(!b) throw std::runtime_error(why); }
template<class F> void refused(F f,const char* why) {
    bool threw=false; try { f(); } catch(const std::exception&) { threw=true; } check(threw,why);
}
strix::PrefixCacheBuffer heap(std::uint64_t n) {
    return {std::shared_ptr<std::byte>(new std::byte[static_cast<std::size_t>(n?n:1)],std::default_delete<std::byte[]>()),n};
}
strix::ModelContentIdentity hash(std::string_view s) { return strix::sha256_bytes(std::as_bytes(std::span(s.data(),s.size()))); }
strix::PrefixCacheBuffer clone(const strix::PrefixCacheBuffer& b) {
    auto c=heap(b.bytes); std::memcpy(c.data.get(),b.data.get(),static_cast<std::size_t>(b.bytes)); return c;
}
void reroot(strix::PrefixCacheBuffer& b) {
    auto h=strix::sha256_bytes_fast(b.span().first(static_cast<std::size_t>(b.bytes-32)));
    std::memcpy(b.data.get()+b.bytes-32,h.bytes.data(),32);
}
}
void session_tail_plan_test() {
    const char* previous=std::getenv("STRIX_QPF2K_SCHED");
    const std::optional<std::string> saved=previous?std::optional<std::string>{previous}:std::nullopt;
    const auto set=[](const char* v) {
#ifdef _WIN32
        check(_putenv_s("STRIX_QPF2K_SCHED",v?v:"")==0,"tail environment update");
#else
        check((v?setenv("STRIX_QPF2K_SCHED",v,1):unsetenv("STRIX_QPF2K_SCHED"))==0,"tail environment update");
#endif
    };
    set("8");
    const auto make=[](std::size_t n) {
        strix::decode_cli::Options o; o.known_prefix_window=true; o.known_prefix_prefill_explicit=true;
        o.known_prefix_full_rows=true; o.known_prefix_prefill_rows=2048; o.tokens.resize(n); return o;
    };
    for(int mode=0;mode<6;++mode) {
        auto a=make(32768), b=make(77);
        for(auto* o:{&a,&b}) {
            if(mode==0) o->session_save="checkpoint";
            if(mode==1) o->session_load="checkpoint";
            if(mode==2) o->session_live_control=true;
            if(mode==3) o->session_then_prompt="suffix";
            if(mode==4) o->session_logits="logits";
            if(mode==5) o->session_options_requested=true;
            strix::decode_cli::apply_default_known_prefill_tails(*o);
        }
        check(a.known_prefix_tail_rows==b.known_prefix_tail_rows && a.known_prefix_tail_rows.size()==4,"session tail plan depends on input length");
    }
    auto a=make(32768), b=make(77);
    strix::decode_cli::apply_default_known_prefill_tails(a);
    strix::decode_cli::apply_default_known_prefill_tails(b);
    check(a.known_prefix_tail_rows!=b.known_prefix_tail_rows,"non-session pruning changed");
    set(saved?saved->c_str():nullptr);
}
int main() {
    const auto dir=std::filesystem::temp_directory_path()/
        ("strix-session-tests-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        session_tail_plan_test();
        const auto aligned=strix::plan_session_suffix(32768,1024,1024,true);
        check(aligned.tiled && aligned.serial_rows==0,"aligned suffix did not tile");
        const auto unaligned=strix::plan_session_suffix(32769,2047,1024,true);
        check(unaligned.tiled && unaligned.serial_rows==1023,"unaligned suffix boundary differs");
        check(!strix::plan_session_suffix(32769,2046,1024,true).tiled,"short unaligned suffix tiled");
        check(!strix::plan_session_suffix(32768,128,1024,true).tiled,"short suffix changed route");
        check(!strix::plan_session_suffix(32768,2048,1024,false).tiled,"default suffix route changed");
        strix::SessionCompatibility compat{hash("full model content"),hash("exe"),"compiled plan;all env",false};
        strix::SessionCheckpoint c;
        c.compatibility=compat; c.frontier.consumed=3; c.frontier.accepted=3;
        c.frontier.emitted=2; c.frontier.pending=99; c.frontier.tokens={7,8,9};
        auto payload=heap(4096);
        for(std::uint64_t i=0;i<payload.bytes;++i) payload.data.get()[i]=static_cast<std::byte>(i&255U);
        c.sections.push_back({"target.kv",payload}); c.sections.push_back({"target.fixed",heap(0)});
        auto encoded=strix::encode_session_checkpoint(c,heap);
        {
            const std::array<strix::RotaryIdentity,3> rotary{strix::RotaryIdentity{},
                strix::RotaryIdentity(strix::RotaryDescriptor::yarn(2),hash("table2")),
                strix::RotaryIdentity(strix::RotaryDescriptor::yarn(4),hash("table4"))};
            for(const auto& producer:rotary) {
                auto saved=c; saved.compatibility.numerical_config=strix::session_numerical_config("same CLI",producer);
                auto envelope=strix::encode_session_checkpoint(saved,heap);
                (void)strix::decode_session_checkpoint(envelope,saved.compatibility);
                for(const auto& reader:rotary) if(reader!=producer) {
                    auto expected=saved.compatibility;
                    expected.numerical_config=strix::session_numerical_config("same CLI",reader);
                    bool named=false;
                    try {(void)strix::decode_session_checkpoint(envelope,expected);}
                    catch(const std::runtime_error& e) {named=std::string_view(e.what())=="session compatibility mismatch";}
                    check(named,"cross-rotary session envelope must refuse by name");
                }
            }
        }
        auto decoded=strix::decode_session_checkpoint(encoded,compat);
        check(decoded.frontier.tokens==c.frontier.tokens && decoded.frontier.pending==99U &&
            decoded.frontier.emitted==2 && decoded.frontier.sampling=="greedy","frontier roundtrip");
        check(decoded.sections.size()==2 && decoded.sections[0].name=="target.kv" &&
            decoded.sections[0].payload.bytes==4096 && !std::memcmp(decoded.sections[0].payload.data.get(),payload.data.get(),4096),"sections roundtrip");
        check(strix::encode_session_checkpoint(decoded,heap).bytes==encoded.bytes,"canonical size");
        auto roundtrip=strix::encode_session_checkpoint(decoded,heap);
        check(!std::memcmp(roundtrip.data.get(),encoded.data.get(),static_cast<std::size_t>(encoded.bytes)),"canonical bytes");
        for(int which=0;which<4;++which) {
            auto wrong=compat;
            if(which==0) wrong.model=hash("other weight");
            if(which==1) wrong.executable=hash("other exe");
            if(which==2) wrong.numerical_config+=";flag=changed";
            if(which==3) wrong.mtp=true;
            refused([&]{(void)strix::decode_session_checkpoint(encoded,wrong);},"compat mismatch accepted");
        }
        auto bad=clone(encoded); bad.data.get()[static_cast<std::size_t>(decoded.sections[0].payload.data.get()-encoded.data.get()) + 3]^=std::byte{1};
        refused([&]{(void)strix::decode_session_checkpoint(bad,compat);},"corrupt payload accepted");
        reroot(bad); // manifest valid, section SHA wrong: must still refuse
        refused([&]{(void)strix::decode_session_checkpoint(bad,compat);},"section hash not checked");
        auto schema_bad=clone(encoded); schema_bad.data.get()[8U + std::string_view("strix-session-v1").size()] ^= std::byte{1};
        reroot(schema_bad);
        refused([&]{(void)strix::decode_session_checkpoint(schema_bad,compat);},"wrong schema accepted");
        auto offset_bad=clone(encoded);
        offset_bad.data.get()[static_cast<std::size_t>(decoded.sections[0].payload.data.get()-encoded.data.get()) - 8U] ^= std::byte{1};
        reroot(offset_bad);
        refused([&]{(void)strix::decode_session_checkpoint(offset_bad,compat);},"noncanonical section offset accepted");
        for(const auto n:{0ULL,31ULL,32ULL,100ULL,encoded.bytes-1ULL}) {
            auto short_blob=encoded; short_blob.bytes=n;
            refused([&]{(void)strix::decode_session_checkpoint(short_blob,compat);},"truncation accepted");
        }
        auto invalid=c; invalid.frontier.tokens.pop_back();
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"history mismatch accepted");
        invalid=c; invalid.frontier.sampling="random";
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"sampling approximation accepted");
        invalid=c; invalid.frontier.predictor=1;
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"target-only predictor frontier accepted");
        invalid=c; invalid.sections.push_back(c.sections.front());
        refused([&]{(void)strix::encode_session_checkpoint(invalid,heap);},"duplicate sections accepted");
        strix::PrefixStateCache lookup({encoded.bytes * 2,{},0},heap);
        check(lookup.park_session("good",clone(encoded),2),"park encoded session");
        check(strix::lookup_session_checkpoint(lookup,"good",compat).has_value(),"lookup good");
        auto incompatible = compat; incompatible.numerical_config += " changed";
        check(!strix::lookup_session_checkpoint(lookup,"good",incompatible),"lookup mismatch must miss");
        check(lookup.park_session("bad",clone(bad),2),"park corrupt envelope");
        check(!strix::lookup_session_checkpoint(lookup,"bad",compat),"lookup corruption must miss");
        strix::SessionIoStats io;
        strix::publish_session_checkpoint(dir,encoded,&io);
        check(io.bytes==encoded.bytes,"publish bytes");
        { std::ofstream interrupted(dir/".session-interrupted.tmp",std::ios::binary); interrupted<<"partial"; }
        strix::SessionCheckpoint disk_decoded;
        auto disk=strix::read_session_checkpoint(dir,compat,heap,encoded.bytes,&io,&disk_decoded);
        check(disk_decoded.frontier.pending==99U,"single verification read/decode");
        strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",encoded.bytes*4);
        refused([&]{strix::preflight_session_checkpoint(dir,hash("other executable"),"compiled plan;",encoded.bytes*4);},"preflight executable mismatch accepted");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"other numerical setting",encoded.bytes*4);},"preflight numerical mismatch accepted");
        bool allocated=false;
        const auto tracked=[&](std::uint64_t n) { allocated=true; return heap(n); };
        const auto load=[&](std::uint64_t budget,std::uint64_t available) {
            return strix::read_session_checkpoint(dir,compat,tracked,
                strix::session_checkpoint_envelope_limit(budget,available));
        };
        refused([&]{(void)load(1,UINT64_MAX);},"one-byte restore host budget ignored");
        check(!allocated,"restore allocated before host budget refusal");
        refused([&]{(void)load(encoded.bytes*4,encoded.bytes*4-1);},"restore available memory ignored");
        check(!allocated,"restore allocated before available-memory refusal");
        refused([&]{(void)load(UINT64_MAX,0);},"restore unknown memory accepted");
        check(!allocated,"unknown-memory refusal allocated");
        (void)load(encoded.bytes*4,encoded.bytes*4);
        check(allocated,"admitted restore did not allocate");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",1);},"one-byte preflight budget ignored");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",encoded.bytes*4-1);},"preflight four-copy budget ignored");
        check(!std::memcmp(disk.data.get(),encoded.data.get(),static_cast<std::size_t>(encoded.bytes)),"atomic publication roundtrip");
        c.frontier.pending=100; auto second=strix::encode_session_checkpoint(c,heap);
#ifndef _WIN32
        strix::publish_session_checkpoint(dir,second); // replace atomically; old inode remains intact
        auto replaced=strix::read_session_checkpoint(dir,compat,heap,second.bytes);
        check(strix::decode_session_checkpoint(replaced,compat).frontier.pending==100U,"atomic replacement");
        std::exception_ptr reader_failure;
        std::jthread reader([&](std::stop_token stop) {
            try {
                while(!stop.stop_requested()) {
                    auto snapshot=strix::read_session_checkpoint(dir,compat,heap,encoded.bytes);
                    const auto pending=strix::decode_session_checkpoint(snapshot,compat).frontier.pending;
                    check(pending==99U || pending==100U,"reader observed an intermediate publication");
                }
            } catch(...) { reader_failure=std::current_exception(); }
        });
        for(unsigned i=0;i<8;++i) strix::publish_session_checkpoint(dir,i%2U?encoded:second);
        reader.request_stop(); reader.join();
        if(reader_failure) std::rethrow_exception(reader_failure);
#endif
        refused([&]{(void)strix::read_session_checkpoint(dir,compat,heap,encoded.bytes-1);},"read admission budget ignored");
        { std::ofstream truncated(dir/"session.bin",std::ios::binary|std::ios::trunc); truncated<<"bad"; }
        refused([&]{(void)strix::read_session_checkpoint(dir,compat,heap,encoded.bytes);},"truncated disk file accepted");
        refused([&]{strix::preflight_session_checkpoint(dir,compat.executable,"compiled plan;",encoded.bytes*4);},"preflight truncation accepted");
        // Same pinned buffer accounting as content prefixes. A borrowed resume handle remains charged after eviction.
        strix::PrefixStateCache lru({32,{},0},heap);
        check(lru.park_session("a",heap(16),2),"park a");
        check(lru.park_session("b",heap(16),2),"park b");
        check(lru.stats().host_bytes==32,"parking byte count");
        { auto held=lru.resume_session("a"); check(held.bytes==16,"resume a"); }
        check(lru.park_session("c",heap(16),2),"park c");
        check(!lru.resume_session("b").data && lru.resume_session("a").data,"count LRU evicted wrong session");
        auto held=lru.resume_session("a");
        check(lru.delete_session("a"),"delete session");
        check(lru.stats().host_bytes==32,"borrowed pinned memory disappeared from budget");
        held={}; check(lru.stats().host_bytes==16,"released handle still charged");
        check(!lru.park_session("too-large",heap(33),2),"oversized session admitted");
        refused([&]{(void)lru.park_session("../escape",heap(1),2);},"path traversal session id accepted");
        check(lru.delete_session("c") && lru.list_sessions().empty() && lru.stats().host_bytes==0,"delete accounting");
        strix::PrefixStateCache bytes_lru({24,{},0},heap);
        check(bytes_lru.park_session("one",heap(16),10) && bytes_lru.park_session("two",heap(16),10),"byte eviction admission");
        check(bytes_lru.list_sessions()==std::vector<std::string>{"two"},"byte budget did not evict oldest");
        const auto model=dir/"model.bin";
        { std::ofstream f(model,std::ios::binary); f<<"first complete file"; }
        auto first=strix::session_cached_file_hash(model);
        check(first==strix::sha256_file_content(model) && first==strix::session_cached_file_hash(model),"full-content cached digest");
        { std::ofstream f(model,std::ios::binary|std::ios::trunc); f<<"second complete file, different length"; }
        check(first!=strix::session_cached_file_hash(model),"file identity cache failed to invalidate");
        const char* old_env = std::getenv("STRIX_SESSION_CPU_TEST");
        const std::optional<std::string> old_value = old_env ? std::optional<std::string>{old_env} : std::nullopt;
        const auto set_test_env = [](const char* value) {
#ifdef _WIN32
            check(_putenv_s("STRIX_SESSION_CPU_TEST",value?value:"")==0,"test environment update failed");
#else
            check((value?setenv("STRIX_SESSION_CPU_TEST",value,1):unsetenv("STRIX_SESSION_CPU_TEST"))==0,"test environment update failed");
#endif
        };
        set_test_env("one"); const auto env_one = strix::session_numerical_config("same effective CLI");
        set_test_env("two"); const auto env_two = strix::session_numerical_config("same effective CLI");
        set_test_env(old_value ? old_value->c_str() : nullptr);
        check(env_one!=env_two,"all-STRIX environment config failed to invalidate");
        check(env_two!=strix::session_numerical_config("different effective CLI"),"effective CLI not bound");
        for (const auto key : {"HIP_SESSION_CPU_TEST", "HSA_SESSION_CPU_TEST", "AMD_SESSION_CPU_TEST", "ROCR_SESSION_CPU_TEST", "GPU_SESSION_CPU_TEST"}) {
            const char* old = std::getenv(key);
            const std::optional<std::string> saved = old ? std::optional<std::string>{old} : std::nullopt;
            const auto set = [&](const char* value) {
#ifdef _WIN32
                check(_putenv_s(key,value?value:"")==0,"runtime environment update failed");
#else
                check((value?setenv(key,value,1):unsetenv(key))==0,"runtime environment update failed");
#endif
            };
            set("one"); const auto one = strix::session_numerical_config("cli");
            set("two"); check(one!=strix::session_numerical_config("cli"),"runtime env not bound");
            set(nullptr); check(!one.starts_with(strix::session_numerical_config("cli")),"removed runtime env accepted as prefix");
            set(saved ? saved->c_str() : nullptr);
        }
        for(const auto* key:{"STRIX_PREFIX_CACHE_INFLIGHT","STRIX_PREFIX_CACHE_DISK","STRIX_PREFIX_CACHE_DURABLE"}) {
            const char* previous=std::getenv(key);
            const std::optional<std::string> saved=previous?std::optional<std::string>{previous}:std::nullopt;
            const auto set=[&](const char* value) {
#ifdef _WIN32
                check(_putenv_s(key,value?value:"")==0,"operational environment update");
#else
                check((value?setenv(key,value,1):unsetenv(key))==0,"operational environment update");
#endif
            };
            set(nullptr);const auto absent=strix::session_numerical_config("same CLI");
            set("0");check(absent==strix::session_numerical_config("same CLI"),"operational off must preserve numerical signature");
            set("1");check(absent==strix::session_numerical_config("same CLI"),"operational on must preserve numerical signature");
            set(saved?saved->c_str():nullptr);
        }
        for (const char* key : {"STRIX_QPF2K_T16K", "STRIX_QPF2K_INDEX_BATCH", "STRIX_EAGER_CODE_LOAD", "HIP_ENABLE_DEFERRED_LOADING", "STRIX_QPF2K_EARLY_SETUP", "STRIX_QPF2K_HC_LOWBAND", "STRIX_QPF2K_COMPACT_IMAGES"}) {
            const char* old = std::getenv(key);
            const std::optional<std::string> saved = old ? std::optional<std::string>{old} : std::nullopt;
            const auto set = [&](const char* value) {
#ifdef _WIN32
                check(_putenv_s(key,value?value:"")==0,"startup environment update failed");
#else
                check((value?setenv(key,value,1):unsetenv(key))==0,"startup environment update failed");
#endif
            };
            set(nullptr); const auto before = strix::session_numerical_config("cli");
            for (const char* value : {"0","1"}) {
                set(value);
                check(before == strix::session_numerical_config("cli"),"exact startup setting changed numerical session identity");
            }
            set(saved ? saved->c_str() : nullptr);
        }
        std::filesystem::remove_all(dir);
        std::cout<<"session_state_cache_tests PASS\n"; return 0;
    } catch(const std::exception& e) {
        std::cerr<<"session_state_cache_tests FAIL: "<<e.what()<<'\n';
        std::error_code ec; std::filesystem::remove_all(dir,ec); return 1;
    }
}
