// Host-only tierstream binding (C ABI calls; the device-gated kernels are
// enqueued by the library on the caller's stream). No device work at include.
#include "tierstream_binding.hpp"
#include "tierstream/ts.h"
#include <stdexcept>
namespace strix::glm::rtxpf::producer {
namespace {
void ts_ok(ss_status s,const char* what){if(s!=TS_OK)throw std::runtime_error(std::string("tierstream ")+what+": "+ts_last_error());}
struct Handles {
    ts_desc* desc=nullptr;ts_plan* plan=nullptr;ts_ctx* ctx=nullptr;ts_file_identity source{};std::vector<ts_dg_unit> seed; // read during ts_open
    ~Handles(){if(ctx)ts_close(ctx);if(plan)ts_plan_free(plan);if(desc)ts_desc_free(desc);}
};
void configure(const TierOpenConfig& c,Handles& h,ts_budget& b,ts_options& o){
    const auto& p=c.budget;expert_need(p.refuse.empty(),"tier budget refused");expert_need(c.unit_bytes%256==0,"unit bytes must be a multiple of 256");
    ts_ok(ts_desc_from_json(c.descriptor_json.data(),c.descriptor_json.size(),&h.desc),"descriptor");
    expert_need(std::uint64_t(ts_desc_total_units(h.desc))==c.tiers.size(),"tier array length != descriptor units");
    std::uint64_t r=0,pin=0,disk=0;for(auto t:c.tiers){r+=t==TS_TIER_RESIDENT;pin+=t==TS_TIER_PINNED;disk+=t==TS_TIER_DISK;}
    expert_need(r==p.resident&&pin==p.pinned&&disk==p.disk&&r+pin+disk==c.tiers.size(),"tier array counts != budget");
    const auto lib=library_budget(p,c.unit_bytes,c.tiers.size(),c.max_ids);
    ts_budget_defaults(&b);b.device_bytes=lib.device_bytes;b.host_pinned_bytes=lib.host_bytes;b.margin_gib=0;b.margin_fraction=0;
    b.tier_mode=TS_TIER_MODE_EXCLUSIVE;b.dynamic_slots_per_layer=c.dynamic_per_layer;b.ring_slots=lib.ring_slots;b.host_spare_units=p.host_spares;
    ts_ok(ts_plan_from_tiers(h.desc,&b,c.tiers.data(),c.tiers.size(),&h.plan),"plan");
    ts_options_defaults(&o);o.backend=c.cuda?TS_BACKEND_CUDA:TS_BACKEND_CPU;o.device=0;o.forward_mode=TS_FORWARD_DEVICE_GATED;o.pin_planned=0;o.verify_hash=0;
    expert_need(p.spare_slots>=1,"the global spare count must be positive");o.dg_max_ids=c.max_ids;o.dg_spare_slots=std::uint32_t(p.spare_slots);
    if(p.three_tier){o.dg_disk_tier=1;o.dg_disk_prefetch=c.disk_prefetch?1u:0u;o.dg_disk_staging=std::uint32_t(p.staging);}
    if(c.source_size){h.source.volume=c.source_volume;h.source.index=c.source_index;h.source.size=c.source_size;o.source_identity_n=1;o.source_identities=&h.source;}
    // The device-gated context charges its own arena, tables, pinned blocks and
    // staging to GLM's ledger by name (29250e6: no whole-device sample at open).
    if(c.ledger)o.budget=static_cast<ts_budget_ledger*>(c.ledger->raw());
    expert_need(c.policy>=0&&c.policy<=2,"tier policy");o.policy=c.policy==0?TS_POLICY_LRU:c.policy==1?TS_POLICY_SCORED:TS_POLICY_SCORED_AUTO;
    if(!c.seed.empty()){h.seed.reserve(c.seed.size());for(auto k:c.seed){expert_need(k<c.tiers.size(),"seed key out of range");h.seed.push_back({std::int32_t(k/c.expected_units),std::int32_t(k%c.expected_units)});}
        // The device slots go to the first distinct seed entries: they must be the tier array's residents.
        for(std::size_t i=0;i<c.seed.size()&&i<p.resident;++i)expert_need(c.tiers[c.seed[i]]==TS_TIER_RESIDENT,"seed head != resident tier keys");
        o.dg_seed=h.seed.data();o.dg_seed_n=std::uint32_t(h.seed.size());}
}
class Binding final:public TierCacheApi {
    Handles h_;TierView view_{};CacheCounters before_{};std::uint64_t unit_=0;unsigned pred_=0,pred_through_=0;
    CacheCounters read(){ts_dg_stats s{};ts_ok(ts_stats_dg(h_.ctx,&s),"stats");CacheCounters c;c.lookups=s.lookups;c.misses=s.misses;c.fetched=s.fetched;c.prefetched=s.prefetched;c.resurrected=s.resurrected;c.covered=s.covered;c.bank_serves=s.bank_serves;c.file_reads=s.disk_reads;c.file_bytes=s.disk_read_bytes;c.file_ns=s.disk_read_ns;
        c.evicted=s.evicted;c.drops=s.disk_drops;c.writebacks=s.writebacks_done;c.stalled=s.stalled;c.timeouts=s.device_timeouts;c.free_slots=s.free_slots;return c;}
public:
    explicit Binding(const TierOpenConfig& c){
        ts_budget b;ts_options o;configure(c,h_,b,o);ts_ok(ts_open(h_.desc,&b,h_.plan,&o,&h_.ctx),"open");
        ts_dg_device_view v{};ts_ok(ts_dg_get_device_view(h_.ctx,&v),"device view");
        expert_need(v.struct_size>=sizeof v&&v.absent_err&&v.protocol==SS_DG_PROTOCOL&&v.n_parts==1&&v.n_gated==c.expected_gated,"tierstream view: protocol/parts/gated layers");
        expert_need(v.parts[0].stride==c.unit_bytes&&v.parts[0].bytes==c.unit_bytes,"tierstream view: arena stride != GLM unit stride");
        for(unsigned g=0;g<c.expected_gated;++g)expert_need(ts_dg_gated_index(h_.ctx,int(g))==int(g),"tierstream: gated ordinals out of order");
        view_.slot_of=v.slot_of;view_.unit_base=v.unit_base;view_.abort=v.abort;view_.absent_err=v.absent_err;view_.guard_slot=v.guard_slot;view_.n_gated=v.n_gated;view_.units=c.expected_units;
        view_.base=v.parts[0].base;view_.stride=v.parts[0].stride;unit_=c.unit_bytes;before_=read();
    }
    void begin()override{ts_ok(ts_begin(h_.ctx,nullptr,0),"begin");}
    void enqueue_begin(void* s)override{ts_ok(ts_dg_enqueue_begin(h_.ctx,s),"enqueue begin");}
    void enqueue_export(unsigned g,const std::int32_t* ids,unsigned n,void* s)override{ts_ok(ts_dg_enqueue_export(h_.ctx,int(g),ids,int(n),int(g<pred_through_?pred_:0),s),"enqueue export");}
    void set_predict_ids(unsigned n,unsigned through)override{expert_need(n<=SS_DG_MAX_IDS&&through<view_.n_gated,"predicted ids");pred_=n;pred_through_=n?through:0;}
    void enqueue_predict(unsigned g,const std::int32_t* ids,unsigned n,void* s)override{expert_need(pred_&&n==pred_&&g<pred_through_,"predict without a promise");ts_ok(ts_dg_enqueue_predict(h_.ctx,int(g),ids,int(n),s),"enqueue predict");}
    std::vector<std::uint32_t> residency()override{std::uint32_t total=0;std::vector<ts_dg_unit> u(std::size_t(view_.n_gated)*view_.units);
        ts_ok(ts_dg_residency(h_.ctx,u.data(),std::uint32_t(u.size()),&total),"residency");std::vector<std::uint32_t> keys;keys.reserve(std::min<std::size_t>(total,u.size()));
        for(std::size_t i=0;i<total&&i<u.size();++i)keys.push_back(std::uint32_t(u[i].gated)*view_.units+std::uint32_t(u[i].unit));return keys;}
    void enqueue_wait(unsigned g,void* s)override{ts_ok(ts_dg_enqueue_wait(h_.ctx,int(g),s),"enqueue wait");}
    void enqueue_end(void* s)override{ts_ok(ts_dg_enqueue_end(h_.ctx,s),"enqueue end");}
    void end()override{ts_ok(ts_dg_end_rows(h_.ctx,1),"token not accepted");}
    TierView view()const override{return view_;}
    void reset_open()override{ts_ok(ts_dg_reset_open(h_.ctx),"reset_open");before_=read();}
    // Prefill through the cache (SS_DG_MAKE_RESIDENT, reader/next 563e999): between
    // tokens; SS_E_CAPACITY changed nothing (false). A held set is never exported:
    // a prefill step exports nothing (rows 0 elsewhere) and reads the kept ids through
    // ss_dg_slot; the set is released right after its step, before any decode token.
    bool supports_make_resident()const override{return true;}
    bool make_resident(unsigned g,const std::int32_t* ids,unsigned n)override{
        const auto s=ts_dg_make_resident(h_.ctx,int(g),ids,int(n));if(s==TS_E_CAPACITY)return false;ts_ok(s,"make_resident");return true;}
    void release_resident(unsigned g)override{ts_ok(ts_dg_keep(h_.ctx,int(g),nullptr,0),"release resident set");}
    void begin_layer_step(unsigned g)override{std::vector<std::uint32_t> rows(view_.n_gated,0);expert_need(g<rows.size(),"gated layer");rows[g]=1;
        ts_ok(ts_begin(h_.ctx,rows.data(),int(rows.size())),"begin single-layer step");}
    // Deltas since the previous collect (cache_delta in expert_tier.hpp).
    Json stats()override{auto now=read();auto j=cache_delta(before_,now,unit_);before_=now;
        // Prefill residency (make_resident) totals since open: counted only here, nowhere else.
        ts_dg_stats s{};ts_ok(ts_stats_dg(h_.ctx,&s),"stats");j.set("prefill_resident_units_total",s.resident_units);j.set("prefill_resident_evicted_total",s.resident_evicted);
        j.set("prefill_resident_bytes_total",s.resident_bytes);j.set("prefill_resident_ns_total",s.resident_ns);return j;}
};
}
// reader/next 563e999 has ss_dg_make_resident: Binding overrides make_resident /
// release_resident / begin_layer_step, so prefill reads experts through the cache.
#ifdef SS_DG_MAKE_RESIDENT
bool tierstream_supports_make_resident(){return true;}
#else
bool tierstream_supports_make_resident(){return false;}
#endif
const char* tierstream_commit(){
#ifdef GLM_TIERSTREAM_COMMIT
    return GLM_TIERSTREAM_COMMIT;
#else
    return "unknown";
#endif
}
std::unique_ptr<TierCacheApi> open_tierstream(const TierOpenConfig& c){expert_need(c.cuda,"open_tierstream needs the CUDA backend");return std::make_unique<Binding>(c);}
std::pair<int,std::string> tierstream_admission_probe(const TierOpenConfig& c){
    Handles h;ts_budget b;ts_options o;
    try{configure(c,h,b,o);}catch(const std::exception& e){return {-1,e.what()};}
    o.backend=TS_BACKEND_CPU;auto s=ts_open(h.desc,&b,h.plan,&o,&h.ctx);return {int(s),s==TS_OK?std::string("opened"):std::string(ts_last_error())};
}
}
