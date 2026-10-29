#define STRIX_SESSION_CONSTRUCTION_TEST
#include <source_location>
#include "../apps/cuda_ds41_session_cache.hpp"
#include <iostream>
#ifdef _WIN32
#include <winioctl.h>
#endif
using namespace strix::ds41::session;

// The replaced global operator new/delete pair malloc/free; GCC cannot see that through the replacement.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
static bool watch_allocations=false;
static std::size_t startup_bytes=0,startup_large_allocations=0;
void* operator new(std::size_t n) {
    if(watch_allocations){startup_bytes+=n;if(n>4096)++startup_large_allocations;}
    if(auto p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}

static int checks=0;
static void check(bool ok,std::source_location at=std::source_location::current()) {
    ++checks;if(!ok)throw std::runtime_error("admission check failed at line "+std::to_string(at.line()));
}
template<class F> static void refuses(F f) {
    bool rejected=false;try{f();}catch(const Error&){rejected=true;}check(rejected);
}
static bool directory_alias(const fs::path& target,const fs::path& link) {
    std::error_code ec;fs::create_directory_symlink(fs::absolute(target),link,ec);
    if(!ec)return true;
#ifdef _WIN32
    // Junctions do not need the symbolic-link privilege on Windows.
    const auto absolute=fs::absolute(target).wstring(),substitute=L"\\??\\"+absolute;
    struct Junction {
        DWORD tag=IO_REPARSE_TAG_MOUNT_POINT;WORD length=0,reserved=0;
        WORD substitute_offset=0,substitute_length=0,print_offset=0,print_length=0;
        wchar_t paths[8192]{};
    } data;
    const auto chars=substitute.size()+absolute.size()+2;
    require(chars<=std::size(data.paths),"junction target too long");
    data.substitute_length=static_cast<WORD>(substitute.size()*sizeof(wchar_t));
    data.print_offset=data.substitute_length+sizeof(wchar_t);
    data.print_length=static_cast<WORD>(absolute.size()*sizeof(wchar_t));
    data.length=static_cast<WORD>(8+chars*sizeof(wchar_t));
    std::copy(substitute.begin(),substitute.end(),data.paths);
    std::copy(absolute.begin(),absolute.end(),data.paths+substitute.size()+1);
    fs::create_directory(link);
    const auto handle=CreateFileW(link.c_str(),GENERIC_WRITE,0,nullptr,OPEN_EXISTING,
                                 FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_BACKUP_SEMANTICS,nullptr);
    if(handle!=INVALID_HANDLE_VALUE) {
        DWORD returned=0;
        const bool ok=DeviceIoControl(handle,FSCTL_SET_REPARSE_POINT,&data,data.length+8,nullptr,0,&returned,nullptr)!=0;
        CloseHandle(handle);if(ok)return true;
    }
    fs::remove(link);
#endif
    std::cout<<"SKIP directory alias: "<<ec.message()<<"\n";return false;
}

int main(int argc,char** argv) {
    // Start before Config, paths or test fixtures. Trivial getenv/small strings
    // are allowed; no session I/O, resource construction or allocation >4 KiB.
    watch_allocations=true;
    try {
        Config startup;
        std::unique_ptr<Resources> absent;
        LaunchSeed<Bytes> startup_seed(startup.enabled);
        qualify(startup,"rtx-tp1-decode-rows");
        startup.off("baseline");
        watch_allocations=false;
        check(startup_large_allocations==0&&startup_bytes<=4096);
        check(session_filesystem_operations==0&&session_system_queries==0&&resource_constructions==0);
        check(startup.ram==UINT64_MAX&&startup.dir.empty()&&!absent&&!startup_seed.materialized());
        const auto dir=fs::path(argc>1?argv[1]:"session-admission-test-output")/
            ("run-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Config config;config.enabled=true;config.disk=true;config.dir=dir;
        config.disk_explicit=true;config.disk_bytes=1024*1024;
        Disk disk;disk.open(config,1024);check(disk.enabled());
        Snapshot old;old.compat=sha(bytes("config"));old.attention=bytes("old A");
        auto newer=old;newer.attention=bytes("new A");++newer.committed_generation;
        auto b=old;b.attention=bytes("old B");
        disk.put("A",old);disk.put("B",b);
        const auto a_time=fs::last_write_time(pointer_path(disk.path("A")));
        const auto b_time=fs::last_write_time(pointer_path(disk.path("B")));
        const auto disk_bytes=disk.used();
        const auto a_manifest=read_file(dir/read_pointer(disk.path("A")).directory/"manifest.bin",65536);
        const auto b_manifest=read_file(dir/read_pointer(disk.path("B")).directory/"manifest.bin",65536);
        Lru cache(1024,1);cache.put("A",newer);cache.activate("A");
        const auto live=cache.stage("A");
        // --session-max-parked 1: managed B is read before a later turn
        // check refuses it. Staging must leave active A, RAM and disk intact.
        for(const char* reason:{"mode/fixed mismatch","invalid token file","context capacity","invalid history"}) {
            refuses([&] {
                Lru::Pending pending{restore(disk.path("B"),old.compat,cache.budget())};
                check(!cache.keep_live("B",pending));
                require(false,reason);
                cache.admit("B",std::move(pending));
                disk.touch("B");
            });
            check(cache.list()==std::vector<std::string>{"A"});
            check(cache.evicted()==0&&cache.used()==newer.size());
            check(cache.active()=="A"&&cache.keep_live("A",live));
            check(cache.stage("A").snapshot.attention==newer.attention);
            check(disk.used()==disk_bytes&&fs::last_write_time(pointer_path(disk.path("A")))==a_time&&
                  fs::last_write_time(pointer_path(disk.path("B")))==b_time);
            check(read_file(dir/read_pointer(disk.path("A")).directory/"manifest.bin",65536)==a_manifest&&
                  read_file(dir/read_pointer(disk.path("B")).directory/"manifest.bin",65536)==b_manifest);
        }
        // A successful B admission may evict A. A subsequent disk reload with
        // the same name must import old A, regardless of newer device contents.
        cache.admit("B",{restore(disk.path("B"),old.compat,cache.budget())});
        check(cache.evicted()==1&&cache.active().empty());
        auto reload=Lru::Pending{restore(disk.path("A"),old.compat,cache.budget())};
        check(!cache.keep_live("A",reload));
        auto simulated_gpu=newer.attention;
        if(!cache.keep_live("A",reload))simulated_gpu=reload.snapshot.attention;
        cache.admit("A",reload);cache.activate("A");
        check(simulated_gpu==old.attention&&!cache.keep_live("A",live));
        // Delete/recreate and direct replacement cannot reuse the old live
        // generation, even if the spelling and payload are unchanged.
        for(bool remove_first:{false,true}) {
            const auto stale=cache.stage("A");check(cache.keep_live("A",stale));
            if(remove_first)check(cache.erase("A"));
            cache.put("A",old);check(cache.active().empty());
            check(!cache.keep_live("A",stale));
            cache.activate("A");check(!cache.keep_live("A",stale));
            check(cache.keep_live("A",cache.stage("A")));
        }
        // History fallback is local to the pending copy until admission.
        auto reset=cache.stage("A");reset.generation=0;reset.snapshot=newer;
        check(!cache.keep_live("A",reset));
        check(cache.keep_live("A",cache.stage("A")));
        cache.admit("A",reset);check(cache.active().empty());
        // Refused RAM preflight must not refresh recency either.
        Lru order(1024,2);order.put("A",old);order.put("B",b);order.activate("A");
        refuses([&] {auto pending=order.stage("A");pending.generation=0;require(false,"refused history");});
        order.put("C",old);check(!order.contains("A")&&order.contains("B")&&order.active().empty());
        // Inspecting an immutable external copy or a missing nested source
        // must not leave a lock file or newly created directory behind.
        const auto external=dir/"external";fs::create_directory(external);
        fs::copy(dir/read_pointer(disk.path("A")).directory,external/"A",fs::copy_options::recursive);
        refuses([&]{restore(external/"A",sha(bytes("wrong")),1024);});
        check(!fs::exists(external/".session.lock"));
        refuses([&]{restore(dir/"missing"/"A",old.compat,1024);});
        check(!fs::exists(dir/"missing"));
        // RAM eviction never changes a durable checkpoint, even if RAM is newer.
        Names names;Lru ram(1024,1,&names);
        ram.put("A",newer);ram.activate("A");ram.put("B",b);
        check(!ram.contains("A")&&ram.contains("B")&&ram.active().empty());
        check(restore(disk.path("A"),old.compat,1024).attention==old.attention);
        try{names.turn("A");check(false);}catch(const Error& e){check(e.code=="session_not_found"&&e.status==404);}
        names.admitted("A");refuses([&]{names.missing("A");});
        Names bounded(2);bounded.evicted("one");bounded.evicted("two");bounded.evicted("three");
        check(bounded.size()==2);refuses([&]{bounded.missing("one");});
        disk.put("A",newer);
        for(auto generation:{old.committed_generation,newer.committed_generation}) {
            auto stale=old;stale.committed_generation=generation;
            try{disk.put("A",stale);check(false);}catch(const Error& e){check(e.status==409&&e.code=="session_conflict");}
            check(restore(disk.path("A"),old.compat,1024).attention==newer.attention);
        }
        auto oversized_stale=old;oversized_stale.attention.resize(config.disk_bytes);
        try{disk.put("A",oversized_stale);check(false);}catch(const Error& e){check(e.code=="session_conflict"&&e.status==409);}
        Snapshot foreign=old;foreign.lineage=Snapshot::new_lineage();
        try{disk.put("A",foreign);check(false);}catch(const Error& e){check(e.code=="session_conflict"&&e.status==409);}
        // Every operation of a refused owner is nonblocking and leaves bytes intact.
        const auto unchanged=read_file(pointer_path(disk.path("A")),65536);
        Disk other;
        const auto busy=[&](auto op){try{op();check(false);}catch(const Error& e){check(e.code=="session_tier_busy"&&e.status==409);}};
        busy([&]{other.open(config,1024);});busy([&]{other.ensure_open();});
        busy([&]{other.put("A",foreign);});busy([&]{other.erase("A");});
        busy([&]{other.touch("A");});busy([&]{other.contains("A");});
        busy([&]{other.list();});busy([&]{other.used();});busy([&]{other.path("A");});
        check(read_file(pointer_path(disk.path("A")),65536)==unchanged);

        // Explicit aliases must use the same pointer/generation transaction.
        Config routing_config=config;routing_config.dir=dir/"path-routing";
        Disk routing;routing.open(routing_config,1024);check(routing.enabled());routing.put("A",old);
        const auto save=[&](const fs::path& path,const std::string& id,const Snapshot& snapshot) {
            const auto route=route_explicit_path(path,routing_config.dir,id);
            if(route.managed)routing.put(id,snapshot);else publish(route.path,snapshot);
        };
        const auto load=[&](const fs::path& path,const std::string& id) {
            const auto route=route_explicit_path(path,routing_config.dir,id);
            return restore(route.path,old.compat,1024,{},route.managed);
        };
        auto latest=old;
        const auto routed=[&](const fs::path& path) {
            check(route_explicit_path(path,routing_config.dir,"A").managed);
            ++latest.committed_generation;latest.attention=bytes("generation "+std::to_string(latest.committed_generation));
            save(path,"A",latest);
            check(load(path,"A").committed_generation==latest.committed_generation);
            check(restore(routing.path("A"),old.compat,1024).attention==latest.attention);
            check(!fs::exists(routing.path("A")));
        };
        const auto refused=[&](const fs::path& path,const std::string& id) {
            for(bool saving:{false,true}) {
                try{if(saving)save(path,id,newer);else (void)load(path,id);check(false);}
                catch(const Error& e){check(e.code=="session_conflict"&&e.status==409&&
                    std::string(e.what())=="path inside the managed session tier; use durability disk without a path");}
                check(restore(routing.path("A"),old.compat,1024).attention==latest.attention);
            }
        };
        routed(routing_config.dir/"."/"A");
        fs::create_directory(routing_config.dir/"subdir");routed(routing_config.dir/"subdir"/".."/"A");
        refused(routing_config.dir/"."/"A","B"); // another session cannot publish beside A.current
        refused(routing_config.dir/"subdir"/".."/"B","A");
        refused(pointer_path(routing.path("A")),"A");
        refused(routing_config.dir/read_pointer(routing.path("A")).directory,"A");
        refused(routing_config.dir/"A.g999","A");
#ifdef _WIN32
        auto case_alias=fs::absolute(routing_config.dir/"a").wstring();
        std::transform(case_alias.begin(),case_alias.end(),case_alias.begin(),[](wchar_t c){return c>=L'a'&&c<=L'z'?wchar_t(c-(L'a'-L'A')):c;});
        routed(case_alias);refused(case_alias,"B");
#endif
        if(directory_alias(routing_config.dir,dir/"tier-alias")) {
            routed(dir/"tier-alias"/"A");refused(dir/"tier-alias"/"A","B");
            const auto generation=routing_config.dir/read_pointer(routing.path("A")).directory;
            if(directory_alias(generation,dir/"generation-alias"))refused(dir/"generation-alias","A");
        }
        // Both names existing is a conflict, never a preference for one payload.
        publish_unlocked(routing.path("A"),old);
        for(bool saving:{false,true}) {
            try{if(saving)save(routing.path("A"),"A",newer);else (void)load(routing.path("A"),"A");check(false);}
            catch(const Error& e){check(e.code=="session_conflict"&&e.status==409);}
        }
        try{restore(routing.path("A"),old.compat,1024);check(false);}
        catch(const Error& e){check(e.code=="session_conflict"&&e.status==409);}
        fs::remove_all(routing.path("A"));
        check(restore(routing.path("A"),old.compat,1024).attention==latest.attention);
        publish_unlocked(routing.path("C"),old);
        refused(routing.path("C"),"C");
        // Outside the configured tier, an adjacent pointer is still ambiguous.
        const auto outside_path=dir/"outside_path";
        publish(outside_path,old);write_file(pointer_path(outside_path),read_file(pointer_path(routing.path("A")),65536));
        for(bool saving:{false,true}) {
            try{if(saving)publish(outside_path,newer);else (void)restore(outside_path,old.compat,1024,{},false);check(false);}
            catch(const Error& e){check(e.code=="session_conflict"&&e.status==409);}
        }
        fs::remove(pointer_path(outside_path));check(restore(outside_path,old.compat,1024,{},false).attention==old.attention);

        // No-session startup configures policy only: even a missing nested
        // directory and its parent remain absent until ensure_open is called.
        Config lazy_config=config;lazy_config.dir=dir/"no-session"/"tier";
        {Disk lazy;lazy.configure(lazy_config,1024);check(!lazy.opened()&&!lazy.enabled());
         check(lazy.list().empty()&&lazy.used()==0&&!lazy.contains("A"));}
        check(!fs::exists(dir/"no-session"));
        Disk lazy;lazy.configure(lazy_config,1024);lazy.ensure_open();check(lazy.opened()&&lazy.enabled());
        check(fs::exists(lazy_config.dir/".session.lock"));
        // Optional storage executes no resource constructors until session use.
        const auto constructions=resource_constructions;
        std::optional<Resources> resources;
        check(resource_constructions==constructions);
        resources.emplace(lazy_config,1024,1024);
        check(resource_constructions==constructions+4);
        check(!resources->disk.opened());
        check(resources->ram.list().empty()&&resources->names.size()==0);
        // The worker's max-parked=1 holds active B and parked A, then C evicts A.
        Config parking_config=lazy_config;parking_config.max_parked=1;
        Resources parking(parking_config,1024,1024);
        parking.ram.put("A",old);parking.ram.activate("A");
        parking.ram.put("B",b);parking.ram.activate("B");
        check(parking.ram.contains("A")&&parking.ram.active()=="B");
        parking.ram.admit("A",parking.ram.stage("A"));parking.ram.activate("A");
        parking.ram.put("A",newer);parking.ram.activate("A");
        parking.ram.admit("B",parking.ram.stage("B"));parking.ram.activate("B");
        parking.ram.put("C",old);parking.ram.activate("C");
        check(!parking.ram.contains("A")&&parking.ram.contains("B")&&parking.ram.contains("C"));
        try{parking.names.turn("A");check(false);}catch(const Error& e){check(e.code=="session_not_found"&&e.status==404);}
        check(!parking.disk.opened());
        // Follow default unnamed requests past startup. Import loading itself
        // owns model storage; watch only the session path around each reload.
        {
            struct State {
                int* copies;Bytes attention;
                State(int& n,Bytes b):copies(&n),attention(std::move(b)){}
                State(const State& s):copies(s.copies),attention(s.attention){++*copies;}
            };
            const auto launch_file=dir/"launch-import.bin",alternate_file=dir/"other-import.bin";
            write_file(launch_file,Bytes(65536,7));write_file(alternate_file,Bytes(65536,9));
            int copies=0;State current(copies,read_file(launch_file,65536));
            Config defaults;check(defaults.enabled&&!defaults.explicit_opt_in);
            std::unique_ptr<Resources> ordinary_resources;
            const auto constructions_before=resource_constructions,queries_before=session_system_queries;
            const auto large_before=startup_large_allocations;
            std::uint64_t session_io=0; // filesystem operations inside the session calls only
            int loads=0;
            const auto loader=[&](const fs::path& path){++loads;return State(copies,read_file(path,65536));};
            watch_allocations=true;
            LaunchSeed<State> launch(defaults.enabled,launch_file,loader);
            watch_allocations=false;
            const auto ordinary=[&](const fs::path& path) {
                std::vector<std::string> args{"--import-state",path.string()};
                watch_allocations=true;const auto io0=session_filesystem_operations;
                const auto command=parse(args);check(command.id.empty());
                launch.before_replace(current,defaults.enabled,path);
                session_io+=session_filesystem_operations-io0;watch_allocations=false;
                // Model reload allocations are outside session accounting.
                current.attention=read_file(path,65536);
                launch.after_replace(path);
            };
            // Unnamed requests, same or other import argument: no session work at all.
            ordinary(launch_file);ordinary(launch_file);ordinary(alternate_file);ordinary(alternate_file);
            check(!launch.materialized()&&copies==0&&loads==0&&startup_large_allocations==large_before);
            check(session_io==0);
            check(!ordinary_resources&&resource_constructions==constructions_before&&session_system_queries==queries_before);
            // First session use after another import reloads the launch argument.
            const auto& created=launch.materialize(current);
            check(created.attention==Bytes(65536,7)&&loads==1);
            current.attention[0]=99;
            check(launch.materialize(current).attention==created.attention&&loads==1);
            // Back on the launch argument before any session use: the reloaded state is the seed.
            {
                LaunchSeed<State> back(true,launch_file,loader);const auto l0=loads;
                back.before_replace(current,true,alternate_file);current.attention=read_file(alternate_file,65536);
                back.after_replace(alternate_file);
                back.before_replace(current,true,launch_file);current.attention=read_file(launch_file,65536);
                back.after_replace(launch_file);
                check(back.materialize(current).attention==Bytes(65536,7)&&loads==l0);
            }
            // Another spelling of the launch file is another argument: reload the launch argument.
            {
                LaunchSeed<State> alias(true,launch_file,loader);const auto l0=loads;
                alias.before_replace(current,true,dir/"."/"launch-import.bin");alias.after_replace(dir/"."/"launch-import.bin");
                check(alias.materialize(current).attention==Bytes(65536,7)&&loads==l0+1);
            }
            // The same argument follows the file, exactly as ordinary requests read it.
            {
                LaunchSeed<State> same(true,launch_file,loader);const auto l0=loads;
                write_file(launch_file,Bytes(65536,8));
                same.before_replace(current,true,launch_file);current.attention=read_file(launch_file,65536);
                same.after_replace(launch_file);
                check(same.materialize(current).attention==Bytes(65536,8)&&loads==l0);
                write_file(launch_file,Bytes(65536,7));
            }
            // A failed reload of the launch argument leaves the other import and its provenance:
            // the first session use reloads the launch argument instead of copying the other state.
            {
                LaunchSeed<State> failed(true,launch_file,loader);const auto l0=loads;
                failed.before_replace(current,true,alternate_file);current.attention=read_file(alternate_file,65536);
                failed.after_replace(alternate_file);
                failed.before_replace(current,true,launch_file); // load throws: after_replace is never reached
                check(failed.materialize(current).attention==Bytes(65536,7)&&loads==l0+1);
            }
            // A synthetic launch seed (no file) is preserved before replacement.
            for(const fs::path& file:{fs::path{},fs::path("none")}) {
                LaunchSeed<State> synthetic(true,file,loader);const auto before_synthetic=copies;
                synthetic.before_replace(current,true,file);synthetic.after_replace(file);
                check(synthetic.materialized()&&copies==before_synthetic+1);
            }
        }
        // Lazy launch storage copies once before replacement, or at admission.
        struct Seed {int* copies;explicit Seed(int& n):copies(&n){}Seed(const Seed& s):copies(s.copies){++*copies;}};
        int seed_copies=0;Seed imported(seed_copies);
        LaunchSeed<Seed> seed(true);
        check(!seed.materialized()&&seed_copies==0);
        seed.before_replace(imported,true);
        check(seed.materialized()&&seed_copies==1);
        const auto& preserved=seed.materialize(imported);
        seed.before_replace(imported,true);
        check(&preserved==&seed.materialize(imported)&&seed_copies==1);
        LaunchSeed<Seed> first_admission(true);
        check(!first_admission.materialized());
        check(&first_admission.materialize(imported)!=&imported&&seed_copies==2);
        first_admission.before_replace(imported,true);
        check(seed_copies==2);
        LaunchSeed<Seed> disabled(false);
        const auto bytes_before=startup_bytes;
        watch_allocations=true;
        for(unsigned i=0;i<3;++i)disabled.before_replace(imported,false);
        watch_allocations=false;
        // Explicit sessions remain possible with automatic sessions off: a synthetic seed is kept once.
        check(disabled.materialized()&&seed_copies==3&&startup_bytes-bytes_before<=4096);
        // The kept seed serves a later explicit session without another copy.
        const auto& kept=disabled.materialize(imported);
        check(&kept!=&imported&&seed_copies==3);

        // Simulate abrupt exit at each durable publication boundary. Opening
        // again cleans only unreferenced generations; restore sees old or new.
        for(auto stop:{Disk::Point::files_flushed,Disk::Point::generation_published,Disk::Point::pointer_flushed,
                       Disk::Point::pointer_committed,Disk::Point::old_removed}) {
            Config crash_config=config;crash_config.dir=dir/("crash-"+std::to_string(static_cast<int>(stop)));
            auto crash_owner=std::make_unique<Disk>();auto& crash=*crash_owner;crash.open(crash_config,1024);crash.put("A",old);
            const auto old_pointer=read_pointer(crash.path("A"));bool crashed=false;std::uint64_t peak=0;
            try{crash.put("A",newer,[&](Disk::Point point) {
                if(point==Disk::Point::pointer_committed) {
                    check(fs::exists(crash_config.dir/old_pointer.directory));
                    for(const auto& entry:fs::recursive_directory_iterator(crash_config.dir))if(entry.is_regular_file())peak+=entry.file_size();
                }
                if(point==stop)throw std::runtime_error("simulated termination");
            });}catch(const std::exception&){crashed=true;}
            check(crashed);
            const bool committed=stop==Disk::Point::pointer_committed||stop==Disk::Point::old_removed;
            check(restore(crash.path("A"),old.compat,1024).attention==(committed?newer.attention:old.attention));
            if(committed)check(peak>0&&peak<=crash.budget());
            crash_owner.reset();
            Disk reopened;reopened.open(crash_config,1024);check(reopened.enabled()&&reopened.list()==std::vector<std::string>{"A"});
            const auto current=read_pointer(reopened.path("A"));std::size_t generations=0;
            for(const auto& entry:fs::directory_iterator(crash_config.dir))if(entry.is_directory()) {
                ++generations;check(entry.path().filename()==current.directory);
            }
            check(generations==1);
            const auto recovered=restore(reopened.path("A"),old.compat,1024);
            check(recovered.lineage==old.lineage&&recovered.committed_generation==(committed?newer.committed_generation:old.committed_generation));
        }
        // A partial first generation has no pointer and is never listed or restored.
        Config partial_config=config;partial_config.dir=dir/"partial-first";auto partial_owner=std::make_unique<Disk>();auto& partial=*partial_owner;partial.open(partial_config,1024);
        refuses([&]{partial.put("A",old,[](auto){throw Error("session_corrupt",422,"interrupted");});});
        check(!partial.contains("A")&&partial.list().empty());partial_owner.reset();auto recovered_owner=std::make_unique<Disk>();auto& recovered_partial=*recovered_owner;recovered_partial.open(partial_config,1024);
        check(!fs::exists(partial_config.dir/"A.g1.tmp"));
        // Unselected partial payloads are cleaned; a broken selected generation
        // or pointer refuses and cannot be mistaken for an unknown name.
        recovered_partial.put("A",old);
        fs::create_directory(partial_config.dir/"A.g99.tmp");write_file(partial_config.dir/"A.g99.tmp"/"attention.bin",bytes("partial"));
        recovered_owner.reset();Disk cleaned;cleaned.open(partial_config,1024);check(cleaned.contains("A")&&!fs::exists(partial_config.dir/"A.g99.tmp"));
        const auto selected=read_pointer(cleaned.path("A"));fs::remove(partial_config.dir/selected.directory/"attention.bin");
        check(cleaned.contains("A"));
        try{restore(cleaned.path("A"),old.compat,1024);check(false);}catch(const Error& e){check(e.code=="session_corrupt"&&e.status==422);}
        auto pointer_bytes=read_file(pointer_path(cleaned.path("A")),65536);pointer_bytes.back()^=1;
        write_file(pointer_path(cleaned.path("A")),pointer_bytes);
        try{restore(cleaned.path("A"),old.compat,1024);check(false);}catch(const Error& e){check(e.code=="session_corrupt"&&e.status==422);}
        // Replacement needs both generations in quota; refusal retains old A.
        Config peak_config=config;peak_config.dir=dir/"peak";
        peak_config.disk_bytes=64*1024+old.attention.size()+old.frontier.encode().size();
        Disk peak_disk;peak_disk.open(peak_config,1024);peak_disk.put("A",old);
        refuses([&]{peak_disk.put("A",newer);});
        check(restore(peak_disk.path("A"),old.compat,1024).attention==old.attention);

        // D1 is an explicit save point, even after a newer RAM-only D2.
        Config saved_config=config;saved_config.dir=dir/"saved-tier";saved_config.max_parked=1;
        const auto external_d1=dir/"external-D1";
        publish(external_d1,old);
        {Disk first_owner;first_owner.open(saved_config,1024);first_owner.put("D",old);}
        Resources restarted(saved_config,1024,1024);
        auto d1=restore(external_d1,old.compat,1024,{},false);
        restarted.ram.admit("D",Lru::Pending{d1});
        restarted.ram.activate("D");
        Command turn;turn.id="D";turn.op="turn";
        prepare_ram_command(turn,restarted.ram,restarted.names);
        check(turn.op=="turn"&&!restarted.disk.opened());
        auto d2=d1;d2.attention=bytes("D2");++d2.committed_generation;
        restarted.ram.put("D",d2);restarted.ram.activate("D");
        check(!restarted.disk.opened());
        // Other explicit managed operations open the tier; other names evict D2.
        restarted.disk.ensure_open();restarted.disk.require_available();
        restarted.disk.put("B",b);restarted.ram.put("B",b);restarted.ram.activate("B");
        restarted.disk.put("C",b);restarted.ram.put("C",b);restarted.ram.activate("C");
        check(!restarted.ram.contains("D")&&restarted.disk.contains("D"));
        const auto saved_pointer=read_file(pointer_path(restarted.disk.path("D")),65536);
        const auto misses_without_io=[&] {
            const auto io=session_filesystem_operations;
            for(const auto* op:{"turn","resume"}) {
                turn.op=op;
                try{prepare_ram_command(turn,restarted.ram,restarted.names);check(false);}
                catch(const Error& e){check(e.code=="session_not_found"&&e.status==404);}
            }
            check(session_filesystem_operations==io);
        };
        misses_without_io();
        for(unsigned i=0;i<4100;++i)restarted.names.evicted("name"+std::to_string(i));
        check(!restarted.names.known("D"));misses_without_io();
        // A full-prompt miss recomputes even while D1 exists in the open tier.
        turn.op="turn";prepare_ram_command(turn,restarted.ram,restarted.names,true);
        check(turn.op=="create");
        turn.op="turn";turn.history="full.ids";
        prepare_ram_command(turn,restarted.ram,restarted.names);check(turn.op=="create");
        // Explicit managed restore returns exactly D1, never the evicted D2.
        const auto saved=restore(restarted.disk.path("D"),old.compat,1024);
        check(saved.attention==d1.attention&&saved.drafter==d1.drafter&&saved.frontier.encode()==d1.frontier.encode());
        check(saved.lineage==d1.lineage&&saved.committed_generation==d1.committed_generation);
        check(read_file(pointer_path(restarted.disk.path("D")),65536)==saved_pointer);
        restarted.ram.admit("D",Lru::Pending{saved});
        check(restarted.ram.stage("D").snapshot.attention==old.attention);
        // An inaccessible tier is distinct from confirmed absence for explicit
        // disk operations. A suffix-only missing turn still fails with 404.
        for(bool budget_failure:{false,true}) {
            Config failed=config;failed.dir=dir/(budget_failure?"too-small":"not-a-directory");
            if(budget_failure)failed.disk_bytes=1;else write_file(failed.dir,bytes("file"));
            Disk unavailable;unavailable.open(failed,1024);
            check(!unavailable.available()&&!unavailable.contains("A"));
            try{unavailable.require_available();check(false);}catch(const Error& e){
                check(e.code=="session_tier_unavailable"&&e.status==(budget_failure?507:503));
            }
            unavailable.ensure_open();
            try{unavailable.require_available();check(false);}catch(const Error& e){check(e.code=="session_tier_unavailable");}
            Command missing;missing.id="unknown";missing.op="turn";
            try{prepare_ram_command(missing,restarted.ram,restarted.names);check(false);}
            catch(const Error& e){check(e.code=="session_not_found"&&e.status==404);}
        }

        std::cout<<checks<<" admission checks passed\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
