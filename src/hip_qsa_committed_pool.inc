// Included at namespace strix scope. Storage and completion belong to the request.
struct HipQsaCommittedPool::Impl {
    struct Entry {
        std::mutex mutex;
        std::uintptr_t raw=0;
        std::uint64_t raw_bytes=0, generation=0;
        RuntimeTensorBinding norm{};
        std::uint32_t norm_type=0;
        float epsilon=0, theta=0;
        std::uint64_t rotary_dim=0;
        std::shared_ptr<const HipQwenRotaryTable> table;
        QsaPrefillPool pool;
        QsaCommittedFrontier frontier;
        hipEvent_t done=nullptr,ready=nullptr;
        bool recorded=false;
    };
    MtpQsaPoolPlan plan;
    std::array<Entry,14> entries; // 12 target, 1 predictor, 1 exclusive selector scratch lease
    std::mutex registration;
    void* allocation=nullptr;
    int device=-1;
    std::atomic<bool> stopped{false};
    std::atomic<std::uint64_t> epoch{1};
    std::array<ModelContentIdentity,2> content{};
    std::array<std::uint64_t,2> profiles{};
    std::atomic<std::uint64_t> checks{0}, compared{0}, differing{0};
    std::array<std::atomic<std::uint64_t>,13> role_checks{},role_scores{},role_differ{};
    explicit Impl(MtpQsaPoolPlan p):plan(p) {}
};
static_assert(sizeof(HipQsaCommittedPool::Impl)<=16384U);
struct HipQsaCommittedAccess::Impl {
    std::shared_ptr<HipQsaCommittedPool> owner;
    HipQsaCommittedPool::Impl* service=nullptr;
    HipQsaCommittedPool::Impl::Entry* entry=nullptr;
    std::unique_lock<std::mutex> lock;
    hipStream_t stream=nullptr;
    unsigned role=0;
    int exceptions=0;
};
HipQsaCommittedPool::HipQsaCommittedPool()=default;
std::shared_ptr<HipQsaCommittedPool> HipQsaCommittedPool::create(const MtpQsaPoolPlan& plan) {
    const auto expected=mtp_qsa_pool_plan(plan.capacity,plan.maximum_writer_rows,plan.scratch_bytes);
    if (expected.reserved_bytes!=plan.reserved_bytes || expected.entry_stride!=plan.entry_stride ||
        expected.blocks!=plan.blocks || expected.entries!=plan.entries || expected.key_bytes!=plan.key_bytes ||
        expected.maxima_stride!=plan.maxima_stride || expected.provisional_blocks!=plan.provisional_blocks || expected.metadata_bytes!=plan.metadata_bytes)
        throw std::invalid_argument("MTP QSA pool requires its checked plan");
    auto owner=std::shared_ptr<HipQsaCommittedPool>(new HipQsaCommittedPool);
    owner->impl_=std::make_unique<Impl>(plan); auto& p=*owner->impl_;
    check_hip(hipGetDevice(&p.device),"MTP QSA pool device");
    // All bytes allocated before prompt mutation. No per-window or FP32 substitute.
    if (hipMalloc(&p.allocation,plan.key_bytes+plan.scratch_bytes)!=hipSuccess) {
        static_cast<void>(hipGetLastError()); return {};
    }
    for (unsigned i=0;i<13;++i) {
        p.entries[i].pool.packed=reinterpret_cast<std::uint16_t*>(static_cast<unsigned char*>(p.allocation)+i*plan.entry_stride);
        p.entries[i].pool.mtp_role=i;
        p.entries[i].pool.capacity=static_cast<std::uint32_t>(plan.blocks);
    }
    for (auto& e:p.entries) if(hipEventCreateWithFlags(&e.done,hipEventDisableTiming)!=hipSuccess ||
        hipEventCreateWithFlags(&e.ready,hipEventDisableTiming)!=hipSuccess) {
        static_cast<void>(hipGetLastError());return {};
    }
    std::fprintf(stderr,"qsa-mtp-pool enabled=1 format=bf16 entries=13 key_bytes=%llu scratch_bytes=%llu reserved_bytes=%llu epoch=1\n",
        static_cast<unsigned long long>(plan.key_bytes),static_cast<unsigned long long>(plan.scratch_bytes),
        static_cast<unsigned long long>(plan.reserved_bytes));
    return owner;
}
HipQsaCommittedPool::~HipQsaCommittedPool() {
    if (!impl_) return;
    auto& p=*impl_;std::uint64_t rebuilt=0,invalidated=0,promoted=0;
    for (auto& e:p.entries) {
        const std::lock_guard lock(e.mutex);
        if (e.recorded && hipEventSynchronize(e.done)!=hipSuccess) std::terminate();
        if (e.done && hipEventDestroy(e.done)!=hipSuccess) std::terminate();
        if (e.ready && hipEventDestroy(e.ready)!=hipSuccess) std::terminate();
        rebuilt+=e.frontier.rebuilds;invalidated+=e.frontier.invalidations;promoted+=e.frontier.promotions;
    }
    if (p.allocation && hipFree(p.allocation)!=hipSuccess) std::terminate();
    std::fprintf(stderr,"qsa-mtp-pool summary epoch=%llu rebuilt_blocks=%llu invalidations=%llu promoted_blocks=%llu checks=%llu scores=%llu differ=%llu\n",
        static_cast<unsigned long long>(p.epoch),static_cast<unsigned long long>(rebuilt),static_cast<unsigned long long>(invalidated),
        static_cast<unsigned long long>(promoted),static_cast<unsigned long long>(p.checks.load()),
        static_cast<unsigned long long>(p.compared.load()),static_cast<unsigned long long>(p.differing.load()));
}
std::unique_ptr<HipQsaCommittedAccess> HipQsaCommittedPool::before_rewrite(bool predictor,
    std::uintptr_t raw,std::uint64_t raw_bytes,std::uint64_t generation,const RuntimeTensorBinding& norm,
    std::uint32_t norm_type,const ModelContract& model,std::uint64_t first,std::uint32_t rows,void* raw_stream,bool certified) {
    auto& p=*impl_;const auto epoch=p.epoch.load();int device=-1;check_hip(hipGetDevice(&device),"MTP QSA lease device");
    if (!raw_stream || !raw || !generation || !norm.address || !norm.allocation_generation || norm.bytes<256 ||
        raw_bytes>UINTPTR_MAX-raw || device!=p.device || p.stopped.load() || model.indexer_head_dim!=128 ||
        model.indexer_n_heads!=4 || model.indexer_compress_ratio!=4 || raw_bytes/256U<p.plan.capacity)
        throw std::invalid_argument("MTP QSA lease owner/geometry is invalid");
    auto a=std::make_unique<HipQsaCommittedAccess::Impl>();a->owner=shared_from_this();a->service=&p;
    a->stream=static_cast<hipStream_t>(raw_stream);a->exceptions=std::uncaught_exceptions();
    {
        const std::lock_guard registry(p.registration);
        unsigned role=predictor?12U:0U;
        if (!predictor) {
            for (;role<12U;++role) if (!p.entries[role].raw || p.entries[role].raw==raw) break;
            if (role==12U) throw std::invalid_argument("MTP QSA has more than twelve target roles");
        }
        a->role=role;a->entry=&p.entries[role];a->lock=std::unique_lock<std::mutex>(a->entry->mutex);
        if(p.stopped.load() || p.epoch.load()!=epoch) throw std::invalid_argument("MTP QSA lease crossed cancel/epoch boundary");
        auto& e=*a->entry;
        if (!e.raw) {
            e.raw=raw;e.raw_bytes=raw_bytes;e.generation=generation;e.norm=norm;e.norm_type=norm_type;
            e.epsilon=model.rms_norm_epsilon;e.theta=model.rope_theta;e.rotary_dim=model.rotary_dim();e.table=model.qwen_rotary_table;
            // First use may follow completed tiled prefill/cache import. Build lazily from real raw bytes.
            e.frontier.raw_end=first;
        } else if (e.raw!=raw || e.raw_bytes!=raw_bytes || e.generation!=generation || e.norm.address!=norm.address ||
            e.norm.bytes!=norm.bytes || e.norm.allocation_generation!=norm.allocation_generation || e.norm_type!=norm_type ||
            e.epsilon!=model.rms_norm_epsilon || e.theta!=model.rope_theta || e.rotary_dim!=model.rotary_dim() || e.table!=model.qwen_rotary_table)
            throw std::invalid_argument("MTP QSA live entry identity changed");
    }
    auto& e=*a->entry;
    if (e.recorded) check_hip(hipStreamWaitEvent(a->stream,e.done,0),"MTP QSA writer waits for prior consumers");
    e.frontier.before_rewrite(first,rows,p.plan,certified);
    e.pool.valid=e.frontier.committed;
    return std::make_unique<HipQsaCommittedAccess>(std::move(a));
}
std::unique_ptr<HipQsaCommittedAccess> HipQsaCommittedPool::selector_lease(void* raw_stream) {
    auto& p=*impl_;const auto epoch=p.epoch.load();int device=-1;
    check_hip(hipGetDevice(&device),"MTP QSA scratch lease device");
    if (!raw_stream || device!=p.device || p.stopped.load())
        throw std::invalid_argument("MTP QSA scratch owner/device is invalid");
    auto a=std::make_unique<HipQsaCommittedAccess::Impl>();a->owner=shared_from_this();a->service=&p;a->role=13;
    a->entry=&p.entries[13];a->lock=std::unique_lock<std::mutex>(a->entry->mutex);a->stream=static_cast<hipStream_t>(raw_stream);
    if(p.stopped.load() || p.epoch.load()!=epoch) throw std::invalid_argument("MTP QSA scratch crossed cancel/epoch boundary");
    a->exceptions=std::uncaught_exceptions();
    if (a->entry->recorded) check_hip(hipStreamWaitEvent(a->stream,a->entry->done,0),"MTP QSA scratch waits for consumer");
    return std::make_unique<HipQsaCommittedAccess>(std::move(a));
}
HipQsaCommittedAccess::HipQsaCommittedAccess(std::unique_ptr<Impl> p):impl_(std::move(p)) {}
HipQsaCommittedAccess::~HipQsaCommittedAccess() {
    auto& a=*impl_;auto& e=*a.entry;
    if (std::uncaught_exceptions()>a.exceptions) {e.pool.valid=0;e.frontier.invalidate(0);a.service->stopped=true;}
    else if (a.role<13) {
        e.frontier.built(e.pool.valid,a.service->plan);
        e.frontier.committed=std::max(e.frontier.committed,std::min(e.pool.valid,static_cast<std::uint32_t>(e.frontier.raw_end/4)));
    }
    // Last use includes all queued reads and raw writes, also on partial enqueue failure.
    if (hipEventRecord(e.done,a.stream)!=hipSuccess) std::terminate();
    e.recorded=true;
}
void* HipQsaCommittedAccess::storage() const noexcept {return &impl_->entry->pool;}
HipQsaKeyDescriptor HipQsaCommittedAccess::key() const {
    const auto& a=*impl_;check_hip(hipEventRecord(a.entry->ready,a.stream),"MTP QSA key producer fence");
    return {a.entry->pool.packed,a.service->epoch.load(),a.role,128,a.entry->pool.capacity,a.entry->ready};
}
float* HipQsaCommittedAccess::scratch() const noexcept {
    return impl_->role==13?reinterpret_cast<float*>(static_cast<unsigned char*>(impl_->service->allocation)+impl_->service->plan.key_bytes):nullptr;
}
void HipQsaCommittedPool::bind_domain(bool predictor,const ModelContentIdentity& content,std::uint64_t profile) {
    auto& p=*impl_;const std::lock_guard registry(p.registration);const unsigned domain=predictor?1U:0U;
    if (!content.valid() || !profile || (p.profiles[domain] && (p.profiles[domain]!=profile || p.content[domain]!=content)))
        throw std::invalid_argument("MTP QSA immutable model/producer identity changed");
    p.content[domain]=content;p.profiles[domain]=profile;
}
void HipQsaCommittedPool::commit(bool predictor,std::uint64_t end) {
    auto& p=*impl_;const std::lock_guard registry(p.registration);
    if(p.stopped.load()) throw std::invalid_argument("MTP QSA stopped owner cannot commit");
    if (end>p.plan.capacity) throw std::invalid_argument("MTP QSA commit exceeds capacity");
    for (unsigned i=predictor?12U:0U;i<(predictor?13U:12U);++i) {
        auto& e=p.entries[i];const std::lock_guard lock(e.mutex);
        e.frontier.commit(end);e.pool.valid=e.frontier.committed;
    }
}
void HipQsaCommittedPool::invalidate(bool predictor,std::uint64_t first) {
    auto& p=*impl_;
    for (unsigned i=predictor?12U:0U;i<(predictor?13U:12U);++i) {
        auto& e=p.entries[i];const std::lock_guard lock(e.mutex);
        // Restoration/import may run outside the scorer stream. Drain at this explicit lifecycle boundary.
        if (e.recorded) check_hip(hipEventSynchronize(e.done),"MTP QSA invalidation consumer completion");
        e.frontier.invalidate(first);e.pool.valid=e.frontier.committed;
    }
}
void HipQsaCommittedPool::begin_epoch() {
    impl_->stopped=true;
    const std::lock_guard registry(impl_->registration);
    invalidate(false,0);invalidate(true,0);
    for(unsigned role=0;role<13;++role) {
        auto& e=impl_->entries[role];const std::lock_guard lock(e.mutex);
        e.raw=0;e.raw_bytes=0;e.generation=0;e.norm={};e.table.reset();e.pool.rotary_table.reset();
    }
    auto& scratch=impl_->entries[13];
    {const std::lock_guard lock(scratch.mutex);
        if(scratch.recorded) check_hip(hipEventSynchronize(scratch.done),"MTP QSA epoch scratch completion");}
    ++impl_->epoch;impl_->stopped=false;
    // Epoch reset retains allocation leases and model bindings; unproven bytes never remain valid.
}
void HipQsaCommittedPool::cancel() noexcept {
    impl_->stopped=true;
    try {invalidate(false,0);invalidate(true,0);} catch (...) {std::terminate();}
}
void HipQsaCommittedPool::checked(std::uint64_t count,std::uint64_t diff,std::uint32_t role) {
    if(role<13U) {++impl_->role_checks[role];impl_->role_scores[role]+=count;impl_->role_differ[role]+=diff;}
    ++impl_->checks;impl_->compared+=count;impl_->differing+=diff;
    if (diff) throw std::runtime_error("MTP QSA pool self-check mismatch");
}
void HipQsaCommittedPool::print_stats() const {
    auto& p=*impl_;
    for(unsigned role=0;role<13U;++role)
        std::fprintf(stderr,"qsa-mtp-pool role=%u checks=%llu scores=%llu differ=%llu\n",role,
            static_cast<unsigned long long>(p.role_checks[role].load()),static_cast<unsigned long long>(p.role_scores[role].load()),
            static_cast<unsigned long long>(p.role_differ[role].load()));
    std::uint64_t rebuilt=0,invalidated=0,promoted=0;
    for(auto& e:p.entries) {const std::lock_guard lock(e.mutex);
        rebuilt+=e.frontier.rebuilds;invalidated+=e.frontier.invalidations;promoted+=e.frontier.promotions;}
    std::fprintf(stderr,"qsa-mtp-pool stats epoch=%llu rebuilt_blocks=%llu invalidations=%llu promoted_blocks=%llu checks=%llu scores=%llu differ=%llu\n",
        static_cast<unsigned long long>(p.epoch.load()),static_cast<unsigned long long>(rebuilt),static_cast<unsigned long long>(invalidated),
        static_cast<unsigned long long>(promoted),static_cast<unsigned long long>(p.checks.load()),static_cast<unsigned long long>(p.compared.load()),
        static_cast<unsigned long long>(p.differing.load()));
}
std::uint64_t HipQsaCommittedPool::reserved_bytes() const noexcept {return impl_->plan.reserved_bytes;}
