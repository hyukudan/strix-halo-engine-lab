// Included after the internal attention kernels, only in test builds.
#include "hip_qwen_rotary_native_snapshot.inc"
namespace qwen_rotary_diagnostic {
namespace {
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { check_hip(hipMalloc(&p, bytes), "rotary fixture allocation"); }
    ~Buffer() { if (p) (void)hipFree(p); }
    template <class T> T* as() const { return static_cast<T*>(p); }
    void put(const void* source, std::size_t bytes, std::size_t offset = 0) {
        check_hip(hipMemcpy(as<std::byte>() + offset, source, bytes, hipMemcpyHostToDevice), "rotary fixture upload");
    }
};
}
std::vector<float> run(Site site, const ModelContract& model, std::uint64_t position, bool frozen) {
    if (model.qwen_rotary_table) model.qwen_rotary_table->check_position(position);
    if (frozen && model.qwen_rotary_table) throw std::invalid_argument("frozen fixture is native only");
    const bool index = site == Site::index_query || site == Site::pooled || site == Site::batch_pooled ||
        site == Site::score || site == Site::known_score || site == Site::mtp_score;
    const unsigned dim = index ? 128U : 256U;
    const unsigned stride = site == Site::query ? 2U * dim : dim;
    const unsigned block = static_cast<unsigned>(position / 4U);
    const bool pool = site == Site::pooled || site == Site::batch_pooled;
    const bool score = site == Site::score || site == Site::known_score || site == Site::mtp_score;
    std::vector<float> input(stride, 7.0F), weights(dim), query(dim);
    std::vector<std::uint16_t> raw(4U * dim);
    for (unsigned i = 0; i < dim; ++i) {
        input[i] = (i % 3U) ? 1.0F : -1.0F;
        weights[i] = static_cast<float>(1U + i % 7U) / 8.0F;
        query[i] = static_cast<float>(1U + i % 5U) / 8.0F;
        for (unsigned r = 0; r < 4U; ++r) {
            // Four distinct BF16 magnitudes average exactly to one. Pooling
            // still has an exact CPU oracle, but cannot ignore three inputs.
            constexpr std::array<std::uint16_t,4> magnitude{0x3e80U,0x3f40U,0x3fa0U,0x3fe0U};
            raw[r * dim + i] = static_cast<std::uint16_t>(magnitude[r] | (input[i] < 0 ? 0x8000U : 0U));
        }
    }
    Buffer values(stride * sizeof(float)), norm(dim * sizeof(float)), q(dim * sizeof(float));
    values.put(input.data(), input.size() * sizeof(float)); norm.put(weights.data(), weights.size() * sizeof(float));
    q.put(query.data(), query.size() * sizeof(float));
    Buffer keys((pool || score) ? (static_cast<std::size_t>(block) + 1U) * 4U * dim * 2U : 2U);
    if (pool || score) {
        check_hip(hipMemset(keys.p, 0, (static_cast<std::size_t>(block) + 1U) * 4U * dim * 2U), "fixture raw clear");
        keys.put(raw.data(), raw.size() * 2U, static_cast<std::size_t>(block) * raw.size() * 2U);
    }
    Buffer output((pool ? (static_cast<std::size_t>(block) + 1U) * dim : (static_cast<std::size_t>(block) + 1U)) * 4U);
    Buffer table(32U * sizeof(float2)), pos(sizeof(position)), bound(sizeof(uint2));
    pos.put(&position, sizeof(position)); const auto range = make_uint2(block, 1U); bound.put(&range, sizeof(range));
    qsa_index::NormGroupViews<1U> n{};
    n.values[0][0] = values.as<float>(); n.weights[0] = norm.p; n.heads[0] = 1U;
    n.dim[0] = dim; n.stride[0] = stride; n.position[0] = position; n.table = table.as<float2>();
    Buffer groups(sizeof(n)); groups.put(&n, sizeof(n));
    const auto execute = [&]<bool Frozen, class... Source>(Source... source) {
        constexpr auto kind = SmallWeightKind::f32;
#define ROTARY_FIXTURE_KERNEL(Name, ...) [&] { \
    if constexpr (Frozen) return frozen_native::Name<__VA_ARGS__>; \
    else return Name<__VA_ARGS__, Source...>; }()
        if (pool) {
            const auto launch = [&]<bool Batch>() {
                auto k = ROTARY_FIXTURE_KERNEL(block_pool_kernel, kind, Batch);
                // The frozen snapshot predates the packed-key output; a null packed_keys writes nothing extra.
                if constexpr (Frozen)
                    hipLaunchKernelGGL(k, dim3(1), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                        keys.as<std::uint16_t>(), block, dim, 4U, norm.p, 0.0F, 64U, model.rope_theta, output.as<float>(),
                        bound.as<uint2>(), 1U, source...);
                else
                    hipLaunchKernelGGL(k, dim3(1), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                        keys.as<std::uint16_t>(), block, dim, 4U, norm.p, 0.0F, 64U, model.rope_theta, output.as<float>(),
                        bound.as<uint2>(), 1U, static_cast<std::uint16_t*>(nullptr), source...);
            };
            if (site == Site::batch_pooled) launch.template operator()<true>(); else launch.template operator()<false>();
        } else if (score) {
            if (site == Site::score) {
                auto k = ROTARY_FIXTURE_KERNEL(block_score_kernel, kind);
                hipLaunchKernelGGL(k, dim3(block + 1U), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                    q.as<float>(), keys.as<std::uint16_t>(), 1U, dim, 4U, norm.p, 0.0F, 64U, model.rope_theta,
                    output.as<float>(), GraphSelectionArguments{}, source...);
            } else if (site == Site::known_score) {
                QsaRowsScoreViews<1U> v{}; v.index_query[0] = q.as<float>(); v.block_scores[0] = output.as<float>(); v.blocks[0] = block + 1U;
                auto k = ROTARY_FIXTURE_KERNEL(block_score_rows_kernel, kind, 1U);
                hipLaunchKernelGGL(k, dim3(block + 1U), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                    v, keys.as<std::uint16_t>(), 1U, dim, 4U, norm.p, 0.0F, 64U, model.rope_theta, source...);
            } else {
                MtpBatchSelectRows v{}; v.index_query[0] = q.as<float>(); v.raw_keys[0] = keys.as<std::uint16_t>();
                v.block_scores[0] = output.as<float>(); v.blocks[0] = block + 1U;
                auto k = ROTARY_FIXTURE_KERNEL(block_score_rows_kernel, kind);
                hipLaunchKernelGGL(k, dim3(block + 1U), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                    v, 1U, dim, 4U, norm.p, 0.0F, 64U, model.rope_theta, GraphSelectionArguments{}, source...);
            }
        } else if (site == Site::known_rows) {
            QsaRowsNormViews<1U> v{}; v.values[0] = values.as<float>(); v.position[0] = position;
            auto k = ROTARY_FIXTURE_KERNEL(norm_rope_rows_kernel, kind, 1U);
            hipLaunchKernelGGL(k, dim3(1), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                v, 1U, dim, stride, norm.p, 0.0F, 64U, model.rope_theta, source...);
        } else if (site == Site::mtp_norm) {
            MtpBatchNormRopeRows v{}; v.values[0] = values.as<float>(); v.position[0] = position;
            auto k = ROTARY_FIXTURE_KERNEL(norm_rope_rows_kernel, kind);
            hipLaunchKernelGGL(k, dim3(1), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                v, 1U, dim, stride, norm.p, 0.0F, 64U, model.rope_theta, source...);
        } else if (site >= Site::grouped && site <= Site::batch_wave) {
            const auto launch = [&]<bool Batch>() {
                auto k = [&] {
                    if constexpr (Frozen) return frozen_native::rope_table_kernel<1U, Batch>;
                    else return qsa_index::rope_table_kernel<1U, Batch, Source...>;
                }();
                hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, hipStreamLegacy, n, 64U, model.rope_theta,
                    groups.as<qsa_index::NormGroupViews<1U>>(), source...);
                if (site == Site::wave || site == Site::batch_wave)
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::norm_rope_wave_kernel<kind, 1U, Batch>), dim3(1), dim3(256), 0,
                        hipStreamLegacy, n, 0.0F, 64U, groups.as<qsa_index::NormGroupViews<1U>>());
                else hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::norm_rope_group_kernel<kind, 1U, Batch>), dim3(1), dim3(256), 0,
                    hipStreamLegacy, n, 0.0F, 64U, groups.as<qsa_index::NormGroupViews<1U>>());
            };
            if (site == Site::batch_grouped || site == Site::batch_wave) launch.template operator()<true>();
            else launch.template operator()<false>();
        } else {
            auto k = ROTARY_FIXTURE_KERNEL(norm_rope_kernel, kind);
            const GraphPositionKernelArguments dynamic{site == Site::dynamic_query ? pos.as<std::uint64_t>() : nullptr, 0U, nullptr};
            hipLaunchKernelGGL(k, dim3(1), dim3(dim), 2U * dim * sizeof(float), hipStreamLegacy,
                values.as<float>(), 1U, dim, stride, norm.p, 0.0F, 64U, model.rope_theta,
                site == Site::dynamic_query ? 0U : position, dynamic, source...);
        }
#undef ROTARY_FIXTURE_KERNEL
    };
    if (frozen) execute.template operator()<true>();
    else with_qwen_rotary(model, hipStreamLegacy, [&](auto... source) { execute.template operator()<false>(source...); });
    check_hip(hipGetLastError(), "rotary site fixture launch");
    std::vector<float> result(score ? 1U : stride);
    const void* from = score ? output.as<float>() + block : (pool ? output.as<float>() + static_cast<std::size_t>(block) * dim : values.as<float>());
    check_hip(hipMemcpy(result.data(), from, result.size() * sizeof(float), hipMemcpyDeviceToHost), "rotary fixture readback");
    return result;
}

// Faithful sticky-status transaction owner: actual production norm and durable
// raw/KV store kernels, followed by the host token/frontier publication boundary.
// Scratch may change before rejection; durable state and published progress may not.
void publication_probes(const ModelContract& model) {
    struct Owner {
        Buffer raw{512U}, kv{1024U};
        std::array<std::uint16_t,256> raw_before{};
        std::array<std::uint16_t,512> kv_before{};
        bool poisoned=false;
        std::uint64_t frontier=0;
        std::uint32_t token=0xabcdefU;
        Owner() {
            raw_before.fill(0x5a5aU);kv_before.fill(0x6b6bU);
            raw.put(raw_before.data(),sizeof(raw_before)); kv.put(kv_before.data(),sizeof(kv_before));
        }
        void begin() const { if(poisoned) throw std::runtime_error("sticky rotary owner cannot be reused"); }
        bool complete(Buffer& status,Buffer& values,hipStream_t stream) {
            check_hip(hipStreamSynchronize(stream),"publication completion");
            unsigned flag=0;check_hip(hipMemcpy(&flag,status.p,4U,hipMemcpyDeviceToHost),"publication status");
            if(flag) {poisoned=true;return false;}
            QsaRowsStoreViews<1U> stores{};
            stores.index_key[0]=stores.k[0]=stores.v[0]=values.as<float>();stores.position[0]=frontier;
            hipLaunchKernelGGL(HIP_KERNEL_NAME(store_rows_kernel<1U>),dim3(1,1),dim3(256),0,stream,
                stores,raw.as<std::uint16_t>(),kv.as<std::uint16_t>(),256U,256U,
                static_cast<const QsaRowsStoreViews<1U>*>(nullptr));
            check_hip(hipStreamSynchronize(stream),"durable publication wait");
            token=42U;++frontier;return true;
        }
        void unchanged() const {
            std::array<std::uint16_t,256> r{};std::array<std::uint16_t,512> k{};
            check_hip(hipMemcpy(r.data(),raw.p,sizeof(r),hipMemcpyDeviceToHost),"raw refusal readback");
            check_hip(hipMemcpy(k.data(),kv.p,sizeof(k),hipMemcpyDeviceToHost),"KV refusal readback");
            if(r!=raw_before || k!=kv_before || frontier!=0U || token!=0xabcdefU || !poisoned)
                throw std::runtime_error("rejected rotary transaction published durable state/token/frontier");
        }
    };
    const auto& table=*model.qwen_rotary_table;
    Buffer values(1024U),weights(1024U),position(8U),status(4U);
    std::vector<float> ones(256U,1.0F);
    weights.put(ones.data(),1024U);
    const auto reset=[&] { values.put(ones.data(),1024U);const unsigned zero=0;status.put(&zero,4U); };
    const auto norm=[&](hipStream_t stream) {
        const QwenRotarySource source{static_cast<const float2*>(table.data()),table.extent(),status.as<unsigned>()};
        hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_kernel<SmallWeightKind::f32,QwenRotarySource>),dim3(1),dim3(256),2048U,
            stream,values.as<float>(),1U,256U,256U,weights.p,0.0F,64U,model.rope_theta,0ULL,
            (GraphPositionKernelArguments{position.as<std::uint64_t>(),0U,nullptr}),source);
    };
    // Positive control proves this owner really reaches production durable stores.
    { Owner owner;reset();const std::uint64_t valid=1;position.put(&valid,8U);owner.begin();norm(hipStreamLegacy);
      if(!owner.complete(status,values,hipStreamLegacy) || owner.frontier!=1U || owner.token!=42U)
          throw std::runtime_error("publication positive control did not commit");
      std::array<std::uint16_t,256> raw{};std::array<std::uint16_t,512> kv{};
      check_hip(hipMemcpy(raw.data(),owner.raw.p,sizeof(raw),hipMemcpyDeviceToHost),"positive raw publication");
      check_hip(hipMemcpy(kv.data(),owner.kv.p,sizeof(kv),hipMemcpyDeviceToHost),"positive KV publication");
      if(raw==owner.raw_before || kv==owner.kv_before) throw std::runtime_error("positive control omitted durable stores"); }
    for(bool replay:{false,true}) for(auto invalid:{static_cast<std::uint64_t>(table.extent()),UINT64_MAX}) {
        Owner owner;reset();owner.begin();
        hipStream_t stream=nullptr;hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
        check_hip(hipStreamCreate(&stream),"publication stream");
        try {
            if(replay) {
                const std::uint64_t valid=1;position.put(&valid,8U);
                check_hip(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"publication capture");norm(stream);
                check_hip(hipStreamEndCapture(stream,&graph),"publication capture end");
                check_hip(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"publication instantiate");
            }
            position.put(&invalid,8U);
            if(replay) check_hip(hipGraphLaunch(exec,stream),"publication invalid replay");else norm(stream);
            if(owner.complete(status,values,stream)) throw std::runtime_error("invalid rotary owner committed");
            owner.unchanged();
            bool refused=false;try {owner.begin();}catch(const std::runtime_error&){refused=true;}
            if(!refused) throw std::runtime_error("poisoned owner admitted subsequent reuse");
            owner.unchanged();
        } catch(...) {
            if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);
            (void)hipStreamDestroy(stream);throw;
        }
        if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);(void)hipStreamDestroy(stream);
    }
    std::cout<<"site=publication-refusal factor="<<table.descriptor().factor()
        <<" positions_checked=5 bytes_compared="<<4U*2U*(512U+1024U)<<" poisoned_reuse_refusals=4\n";
}

// These tests deliberately bypass run()'s host position guard. Sticky status is
// fixture-only; production table instances trap, with no native ABI change.
void rejection_probes(const ModelContract& model) {
    const auto& table = *model.qwen_rotary_table;
    Buffer values(256U * 4U), weights(256U * 4U), position(sizeof(std::uint64_t)), status(4U);
    std::vector<float> sentinel(256U, 7.0F), gain(256U, 1.0F);
    weights.put(gain.data(), 1024U);
    const auto reset = [&] {
        values.put(sentinel.data(), 1024U);
        const unsigned zero = 0; status.put(&zero, 4U);
    };
    const auto failed = [&] {
        unsigned flag = 0; std::vector<float> output(256U);
        check_hip(hipMemcpy(&flag, status.p, 4U, hipMemcpyDeviceToHost), "rejection status");
        check_hip(hipMemcpy(output.data(), values.p, 1024U, hipMemcpyDeviceToHost), "rejection sentinel");
        if (flag != 1U || output != sentinel) throw std::runtime_error("invalid device position published output");
    };
    const QwenRotarySource source{static_cast<const float2*>(table.data()), table.extent(), status.as<unsigned>()};
    const auto dynamic = [&] (hipStream_t stream) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_kernel<SmallWeightKind::f32, QwenRotarySource>), dim3(1), dim3(256),
            2048U, stream, values.as<float>(), 1U, 256U, 256U, weights.p, 0.0F, 64U, model.rope_theta,
            0ULL, (GraphPositionKernelArguments{position.as<std::uint64_t>(), 0U, nullptr}), source);
    };
    for (std::uint64_t p : {static_cast<std::uint64_t>(table.extent()), UINT64_MAX}) {
        reset(); position.put(&p, sizeof(p)); dynamic(hipStreamLegacy); failed();
        // A late row must not write its values even if earlier workgroups ran.
        QsaRowsNormViews<4U> rows{};
        Buffer row_values(4U*1024U);
        std::vector<float> mixed(4U*256U);
        for (unsigned r = 0; r < 4U; ++r) {
            std::fill_n(mixed.data()+r*256U,256U,7.0F+static_cast<float>(r));
            rows.values[r] = row_values.as<float>()+r*256U; rows.position[r] = r==3U?p:r;
        }
        row_values.put(mixed.data(),mixed.size()*sizeof(float));
        reset();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_rows_kernel<SmallWeightKind::f32, 4U, QwenRotarySource>),
            dim3(1,4), dim3(256), 2048U, hipStreamLegacy, rows, 1U, 256U, 256U, weights.p,
            0.0F, 64U, model.rope_theta, source);
        unsigned mixed_flag=0;
        std::vector<float> mixed_got(mixed.size());
        check_hip(hipMemcpy(mixed_got.data(),row_values.p,mixed.size()*sizeof(float),hipMemcpyDeviceToHost),"mixed-row readback");
        check_hip(hipMemcpy(&mixed_flag,status.p,4U,hipMemcpyDeviceToHost),"mixed-row status");
        if(mixed_flag!=1U || std::memcmp(mixed_got.data()+3U*256U,mixed.data()+3U*256U,1024U))
            throw std::runtime_error("mixed late-invalid norm rows lost isolation");
        for(unsigned r=0;r<3U;++r)
            if(!std::memcmp(mixed_got.data()+r*256U,mixed.data()+r*256U,1024U))
                throw std::runtime_error("valid row was skipped with a late invalid neighbor");
        reset();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(block_pool_kernel<SmallWeightKind::f32, false, QwenRotarySource>),
            dim3(1), dim3(128), 1024U, hipStreamLegacy, static_cast<const std::uint16_t*>(nullptr), table.extent()/4U,
            128U, 4U, weights.p, 0.0F, 64U, model.rope_theta, values.as<float>(), static_cast<const uint2*>(nullptr), 0U,
            static_cast<std::uint16_t*>(nullptr), source);
        failed();
    }
    // Capture at a valid position, then mutate the device position for replay.
    hipStream_t stream = nullptr; hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
    check_hip(hipStreamCreate(&stream), "negative graph stream");
    try {
        auto p = static_cast<std::uint64_t>(table.extent() - 1U); position.put(&p, sizeof(p)); reset();
        check_hip(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "negative graph capture");
        dynamic(stream);
        check_hip(hipStreamEndCapture(stream, &graph), "negative graph end");
        check_hip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "negative graph instantiate");
        check_hip(hipGraphLaunch(exec, stream), "valid boundary graph replay");
        check_hip(hipStreamSynchronize(stream), "valid boundary graph wait");
        unsigned flag = 1; check_hip(hipMemcpy(&flag,status.p,4U,hipMemcpyDeviceToHost), "valid boundary status");
        if (flag) throw std::runtime_error("extent-1 rejected on device");
        for (auto invalid : {static_cast<std::uint64_t>(table.extent()), UINT64_MAX}) {
            reset(); position.put(&invalid, sizeof(invalid));
            check_hip(hipGraphLaunch(exec, stream), "invalid graph replay");
            check_hip(hipStreamSynchronize(stream), "invalid graph wait"); failed();
        }
    } catch (...) {
        if (exec) (void)hipGraphExecDestroy(exec);
        if (graph) (void)hipGraphDestroy(graph);
        (void)hipStreamDestroy(stream); throw;
    }
    (void)hipGraphExecDestroy(exec); (void)hipGraphDestroy(graph); (void)hipStreamDestroy(stream);
}

void multirow_probes(const ModelContract& model) {
    constexpr unsigned Rows = 64U, Groups = 2U, Heads = 2U;
    const auto descriptor = model.qwen_rotary_table->descriptor();
    const auto rounded = [](float value) {
        auto bits = std::bit_cast<std::uint32_t>(value);
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        return std::bit_cast<float>(bits & 0xffff0000U);
    };
    std::array<qsa_index::NormGroupViews<Rows>, Groups> views{};
    std::array<std::array<std::unique_ptr<Buffer>,3>,Groups> buffers;
    std::array<std::array<std::vector<float>,3>,Groups> inputs, expected;
    std::array<std::unique_ptr<Buffer>,Groups> tables;
    std::array<std::unique_ptr<Buffer>,3> weights;
    for (unsigned f=0; f<3U; ++f) {
        const unsigned dim = f==2U ? 128U : 256U;
        std::vector<float> gain(dim);
        for (unsigned i=0;i<dim;++i) gain[i]=static_cast<float>(1U+(i+f)%7U)/8.0F;
        weights[f]=std::make_unique<Buffer>(dim*4U); weights[f]->put(gain.data(),dim*4U);
    }
    for (unsigned g=0;g<Groups;++g) {
        tables[g]=std::make_unique<Buffer>(Rows*32U*sizeof(float2)); views[g].table=tables[g]->as<float2>();
        for (unsigned row=0;row<Rows;++row) {
            const std::array<std::uint64_t,8> boundaries{0,1,8191,8192,262143,262144,descriptor.effective_max()-2U,descriptor.effective_max()-1U};
            views[g].position[row] = row<8U ? boundaries[(row+g)%8U] : (row*3571U+g*31U)%descriptor.effective_max();
        }
        for (unsigned f=0;f<3U;++f) {
            const unsigned dim=f==2U?128U:256U, stride=dim+8U;
            auto& input=inputs[g][f]; input.assign(Rows*Heads*stride,7.0F);
            auto& result=expected[g][f]; result=input;
            for (unsigned row=0;row<Rows;++row) {
                const auto cs=qwen_yarn_row(descriptor,static_cast<std::uint32_t>(views[g].position[row]));
                for (unsigned h=0;h<Heads;++h) {
                    std::vector<float> norm(dim);
                    const unsigned base=(row*Heads+h)*stride;
                    for (unsigned i=0;i<dim;++i) {
                        input[base+i]=(((i*17U)^(row>>(i%6U))^(h<<(i%2U))^(g<<(i%3U))^f)&1U)?1.0F:-1.0F;
                        norm[i]=rounded(input[base+i]*static_cast<float>(1U+(i+f)%7U)/8.0F);
                    }
                    for (unsigned i=0;i<dim;++i) {
                        float value=norm[i];
                        if(i<64U) value=std::fma(norm[i<32U?i+32U:i-32U], i<32U?-cs[i%32U].sine:cs[i%32U].sine,
                            norm[i]*cs[i%32U].cosine);
                        result[base+i]=rounded(value);
                    }
                }
            }
            buffers[g][f]=std::make_unique<Buffer>(input.size()*4U);
            views[g].weights[f]=weights[f]->p; views[g].heads[f]=Heads; views[g].dim[f]=dim; views[g].stride[f]=stride;
            for(unsigned row=0;row<Rows;++row) views[g].values[f][row]=buffers[g][f]->as<float>()+row*Heads*stride;
        }
    }
    std::vector<QsaKnownRows<8U>> tiles(Groups*8U);
    constexpr std::array<float* WorkspacePointers::*,3> fields{&WorkspacePointers::q_gate,&WorkspacePointers::current_k,&WorkspacePointers::index_query};
    for(unsigned g=0;g<Groups;++g) for(unsigned row=0;row<Rows;++row) {
        auto& tile=tiles[g*8U+row/8U]; const auto r=row%8U;
        tile.position[r]=views[g].position[row];
        for(unsigned f=0;f<3U;++f) tile.row[r].workspace.*fields[f]=views[g].values[f][row];
    }
    for(unsigned g=0;g<Groups;++g) {
        const auto prepared=prepare_qpf2k_norm_rows(tiles,g);
        for(unsigned row=0;row<Rows;++row) {
            if(prepared.position[row]!=views[g].position[row]) throw std::runtime_error("QPF2K production position mapping");
            // Production family order is index/query/key.
            for(unsigned f=0;f<3U;++f) views[g].values[f][row]=prepared.values[(f+1U)%3U][row];
        }
    }
    Buffer groups(sizeof(views)); groups.put(views.data(),sizeof(views));
    const QwenRotarySource source{static_cast<const float2*>(model.qwen_rotary_table->data()),descriptor.effective_max()};
    for (bool wave : {false,true}) {
        site_watchdog().begin("qpf2k-multirow");
        std::cout<<"begin site=qpf2k-multirow factor="<<descriptor.factor()<<" wave="<<wave<<std::endl;
        for(unsigned g=0;g<Groups;++g) for(unsigned f=0;f<3U;++f)
            buffers[g][f]->put(inputs[g][f].data(),inputs[g][f].size()*4U);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rope_table_kernel<Rows,true,QwenRotarySource>),dim3(Rows,Groups),dim3(64),0,
            hipStreamLegacy,qsa_index::NormGroupViews<Rows>{},64U,model.rope_theta,groups.as<qsa_index::NormGroupViews<Rows>>(),source);
        if(wave) hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_flash::norm_rope_wave_kernel<SmallWeightKind::f32,Rows,true>),
            dim3((Rows*6U+7U)/8U,Groups),dim3(256),0,hipStreamLegacy,qsa_index::NormGroupViews<Rows>{},0.0F,64U,groups.as<qsa_index::NormGroupViews<Rows>>());
        else hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::norm_rope_group_kernel<SmallWeightKind::f32,Rows,true>),
            dim3(6U,Rows,Groups),dim3(256),0,hipStreamLegacy,qsa_index::NormGroupViews<Rows>{},0.0F,64U,groups.as<qsa_index::NormGroupViews<Rows>>());
        for(unsigned g=0;g<Groups;++g) for(unsigned f=0;f<3U;++f) {
            std::vector<float> got(expected[g][f].size());
            check_hip(hipMemcpy(got.data(),buffers[g][f]->p,got.size()*4U,hipMemcpyDeviceToHost),"multirow readback");
            if(std::memcmp(got.data(),expected[g][f].data(),got.size()*4U))
                throw std::runtime_error("multirow/group/family/head/tail/guard mismatch");
        }
    }
    // Batched MTP uses independent source positions and all norm families.
    for(unsigned f=0;f<3U;++f) {
        site_watchdog().begin("mtp-production-norm");
        std::cout<<"begin site=mtp-production-norm factor="<<descriptor.factor()<<" family="<<f<<" stream=hipStreamLegacy"<<std::endl;
        const unsigned dim=f==2U?128U:256U, stride=dim+8U;
        buffers[1][f]->put(inputs[1][f].data(),inputs[1][f].size()*4U);
        std::vector<MtpAttentionPrepared> prepared(4U);
        std::array<MtpNativeContext,4> contexts{};
        auto mtp_model=model; mtp_model.rms_norm_epsilon=0.0F;
        for(unsigned row=0;row<4U;++row) {
            prepared[row].workspace.*fields[f]=buffers[1][f]->as<float>()+row*Heads*stride;
            const auto source_position=views[1].position[row], seed_position=source_position+1U;
            contexts[row].model=&mtp_model; contexts[row].position=source_position;
            const auto source_cs=qwen_yarn_row(descriptor,static_cast<std::uint32_t>(source_position));
            const auto seed_cs=qwen_yarn_row(descriptor,static_cast<std::uint32_t>(seed_position));
            if(!std::memcmp(source_cs.data(),seed_cs.data(),sizeof(source_cs)))
                throw std::runtime_error("MTP source/seed offset probe is not discriminating");
        }
        RuntimeTensorBinding norm{}; norm.address=reinterpret_cast<std::uintptr_t>(weights[f]->p);
        mtp_batch_norm_rope(prepared,{3U,1U,0U,2U},contexts,fields[f],Heads,dim,stride,norm,hipStreamLegacy);
        std::vector<float> got(4U*Heads*stride);
        check_hip(hipMemcpy(got.data(),buffers[1][f]->p,got.size()*4U,hipMemcpyDeviceToHost),"MTP multirow readback");
        if(std::memcmp(got.data(),expected[1][f].data(),got.size()*4U)) throw std::runtime_error("MTP distinct source-row norm mismatch");
    }
    std::cout<<"site=qpf2k-multirow factor="<<descriptor.factor()<<" positions_checked="<<Rows*Groups
        <<" bytes_compared="<<2U*Groups*Rows*Heads*(264U+264U+136U)*sizeof(float)<<'\n';
    std::cout<<"site=mtp-production-norm factor="<<descriptor.factor()<<" positions_checked=12 bytes_compared="
        <<4U*Heads*(264U+264U+136U)*sizeof(float)<<'\n';
    // A late row of the second group must refuse before writing its table row.
    site_watchdog().begin("qpf2k-multirow-rejection");
    std::cout<<"begin site=qpf2k-multirow-rejection factor="<<descriptor.factor()<<std::endl;
    auto& last=views[1]; last.position[Rows-1U]=descriptor.effective_max();
    groups.put(views.data(),sizeof(views));
    const auto offset=(Rows-1U)*32U*sizeof(float2);
    std::array<float2,32> guard{}; std::memset(guard.data(),0x5a,sizeof(guard));
    tables[1]->put(guard.data(),sizeof(guard),offset);
    Buffer status(4U); const unsigned zero=0; status.put(&zero,4U);
    const QwenRotarySource negative{source.cs,source.extent,status.as<unsigned>()};
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qsa_index::rope_table_kernel<Rows,true,QwenRotarySource>),dim3(Rows,Groups),dim3(64),0,
        hipStreamLegacy,qsa_index::NormGroupViews<Rows>{},64U,model.rope_theta,groups.as<qsa_index::NormGroupViews<Rows>>(),negative);
    std::array<float2,32> got{}; unsigned flag=0;
    check_hip(hipMemcpy(got.data(),tables[1]->as<std::byte>()+offset,sizeof(got),hipMemcpyDeviceToHost),"late invalid table row");
    check_hip(hipMemcpy(&flag,status.p,4U,hipMemcpyDeviceToHost),"late invalid status");
    if(flag!=1U || std::memcmp(got.data(),guard.data(),sizeof(got))) throw std::runtime_error("late invalid group published coefficients");
}

namespace {
void assert_retired(std::weak_ptr<const HipQwenRotaryTable> weak,const void* allocation) {
    // HIP may dispatch graph/user-object destruction callbacks asynchronously.
    for(unsigned attempt=0;attempt<5000U && !weak.expired();++attempt) {
        wait_retirements();std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    wait_retirements();
    if(!weak.expired()) throw std::runtime_error("last reader did not release table owner");
    hipPointerAttribute_t attributes{};
    const auto result=hipPointerGetAttributes(&attributes,allocation);
    (void)hipGetLastError();
    if(result==hipSuccess && attributes.type==hipMemoryTypeDevice)
        throw std::runtime_error("last reader did not retire device allocation");
}
// One lane per gate; coherent mapped memory permits system-scope atomics from
// the CPU and GPU. No HIP callback thread participates in releasing a stream.
__global__ void eager_device_gate(unsigned* release, unsigned* exhausted) {
    for (unsigned long long attempt=0; attempt<(1ULL<<32U); ++attempt) {
        if (__hip_atomic_load(release, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)) return;
    }
    __hip_atomic_store(exhausted, 1U, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
struct EagerGates {
    unsigned* host = nullptr;
    unsigned* device = nullptr;
    EagerGates() {
        check_hip(hipHostMalloc(reinterpret_cast<void**>(&host), 4U*sizeof(unsigned),
            hipHostMallocMapped | hipHostMallocCoherent), "eager mapped gates");
        const auto result=hipHostGetDevicePointer(reinterpret_cast<void**>(&device),host,0);
        if(result!=hipSuccess) { (void)hipHostFree(host); check_hip(result,"eager gate device address"); }
        for(unsigned i=0;i<4U;++i) __atomic_store_n(host+i,0U,__ATOMIC_RELEASE);
    }
    void release(unsigned i) { __atomic_store_n(host+i,1U,__ATOMIC_RELEASE); }
    bool exhausted(unsigned i) const { return __atomic_load_n(host+2U+i,__ATOMIC_ACQUIRE)!=0U; }
    ~EagerGates() { (void)hipHostFree(host); }
};
void eager_lifetime(std::shared_ptr<const HipQwenRotaryTable> table,unsigned count,unsigned failure) {
    site_watchdog().begin("eager-lifetime");
    std::cout<<"begin site=eager-lifetime factor="<<table->descriptor().factor()
        <<" streams="<<count<<" failure="<<failure<<std::endl;
    ModelContract model; model.qwen_rotary_table=std::move(table);
    std::weak_ptr<const HipQwenRotaryTable> weak=model.qwen_rotary_table;
    const void* allocation=model.qwen_rotary_table->data();
    Buffer values(2048U), weights(1024U);
    std::vector<float> ones(512U,1.0F); values.put(ones.data(),2048U); weights.put(ones.data(),1024U);
    std::array<hipStream_t,2> streams{};
    EagerGates gates;
    const bool gated=failure==0U || failure==4U;
    try {
        for(unsigned i=0;i<count;++i) check_hip(hipStreamCreateWithFlags(&streams[i],hipStreamNonBlocking),"independent eager stream");
        for(unsigned i=0;i<count;++i) {
            if(gated) {
                hipLaunchKernelGGL(eager_device_gate,dim3(1),dim3(1),0,streams[i],gates.device+i,gates.device+2U+i);
                check_hip(hipGetLastError(),"eager device gate launch");
            }
            bool refused=false;
            fail_next_reader(failure<4U?failure:0U);
            try {
                with_qwen_rotary(model,streams[i],[&](auto... source) {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_kernel<SmallWeightKind::f32,decltype(source)...>),
                        dim3(1),dim3(256),2048U,streams[i],values.as<float>()+i*256U,1U,256U,256U,
                        weights.p,0.0F,64U,model.rope_theta,1ULL,GraphPositionKernelArguments{},source...);
                    if(failure==4U) throw std::runtime_error("induced failure after first reader submission");
                });
            } catch(const std::runtime_error&) { refused=true; }
            if(refused!=(failure!=0U)) throw std::runtime_error("reader failure injection was not reached");
        }
        model.qwen_rotary_table.reset();
        if(gated && weak.expired()) throw std::runtime_error("eager lease lost without a graph owner");
        for(unsigned i=0;i<count;++i) {
            gates.release(i);
            check_hip(hipStreamSynchronize(streams[i]),"independent eager completion");
            if(gated) for(unsigned j=0;j<count;++j)
                if(gates.exhausted(j)) throw std::runtime_error("eager device gate exhausted before release");
            if(gated && i+1U<count) {
                // Synchronizing a stream does not wait for our CPU event waiter.
                // Prove the first lease is gone while the second gate stays shut.
                for(unsigned attempt=0;attempt<5000U && weak.use_count()>1;++attempt)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if(weak.use_count()!=1) throw std::runtime_error("second stream did not retain the sole remaining lease");
                if(gates.exhausted(i+1U)) throw std::runtime_error("second stream gate exhausted before lease assertion");
            }
            check_hip(hipStreamDestroy(streams[i]),"eager stream destruction"); streams[i]=nullptr;
        }
        assert_retired(weak,allocation);
    } catch(...) {
        for(unsigned i=0;i<count;++i) gates.release(i);
        for(auto stream:streams) if(stream) { (void)hipStreamSynchronize(stream); (void)hipStreamDestroy(stream); }
        throw;
    }
}
}
void device_admission_probes(const ModelContract& model) {
    int active=-1,devices=0;
    check_hip(hipGetDevice(&active),"admission active device");
    check_hip(hipGetDeviceCount(&devices),"admission device count");
    if(devices<2) {
        std::cout<<"site=cross-device-admission factor="<<model.rotary_identity().descriptor().factor()
            <<" positions_checked=0 bytes_compared=0 skipped=requires-two-devices\n";
        return;
    }
    const int other=(active+1)%devices;
    hipStream_t remote=nullptr;hipGraph_t graph=nullptr;
    try {
        check_hip(hipSetDevice(other),"admission select remote");
        check_hip(hipStreamCreate(&remote),"admission remote stream");
        for(bool capture:{false,true}) {
            bool refused=false,submitted=false;
            if(capture) check_hip(hipStreamBeginCapture(remote,hipStreamCaptureModeThreadLocal),"mismatched capture begin");
            try {with_qwen_rotary(model,remote,[&](auto...){submitted=true;});}
            catch(const std::invalid_argument&){refused=true;}
            if(capture) {check_hip(hipStreamEndCapture(remote,&graph),"mismatched capture end");
                if(graph) {check_hip(hipGraphDestroy(graph),"empty mismatched graph");graph=nullptr;} }
            if(!refused || submitted) throw std::runtime_error("mismatched device admitted a rotary reader");
        }
        // Separately exercise a foreign stream while the allocation device is current.
        check_hip(hipSetDevice(active),"admission restore local");
        bool refused=false,submitted=false;
        try {with_qwen_rotary(model,remote,[&](auto...){submitted=true;});}
        catch(const std::invalid_argument&){refused=true;}
        if(!refused || submitted) throw std::runtime_error("foreign stream admitted rotary launch");
        check_hip(hipSetDevice(other),"admission remote cleanup");
        check_hip(hipStreamDestroy(remote),"admission remote destroy");remote=nullptr;
        check_hip(hipSetDevice(active),"admission restore");
    } catch(...) {
        (void)hipSetDevice(other);if(graph)(void)hipGraphDestroy(graph);if(remote)(void)hipStreamDestroy(remote);
        (void)hipSetDevice(active);throw;
    }
    std::cout<<"site=cross-device-admission factor="<<model.rotary_identity().descriptor().factor()
        <<" positions_checked=0 bytes_compared=0 refused_readers=3\n";
}
void lifetime_probes(std::shared_ptr<const HipQwenRotaryTable> table) {
    const auto descriptor=table->descriptor();
    eager_lifetime(std::move(table),1U,0U);
    eager_lifetime(std::make_shared<HipQwenRotaryTable>(descriptor),2U,0U);
    for(unsigned failure=1U;failure<=4U;++failure)
        eager_lifetime(std::make_shared<HipQwenRotaryTable>(descriptor),1U,failure);
    for(unsigned failure:{1U,2U}) {
        site_watchdog().begin("quarantined-lifetime");
        std::cout<<"begin site=quarantined-lifetime factor="<<descriptor.factor()<<" failure="<<failure<<std::endl;
        auto quarantined=std::make_shared<HipQwenRotaryTable>(descriptor);
        const void* allocation=quarantined->data();
        fail_next_retirement(failure);quarantined.reset();wait_retirements();
        hipPointerAttribute_t attributes{};
        check_hip(hipPointerGetAttributes(&attributes,allocation),"quarantined allocation ownership");
        if(attributes.type!=hipMemoryTypeDevice) throw std::runtime_error("failed drain freed allocation");
        // Fixture-only cleanup: this allocation had no readers at all. Production
        // cannot infer that fact after a failure and keeps it quarantined.
        check_hip(hipDeviceSynchronize(),"fixture quarantine cleanup drain");
        check_hip(hipFree(const_cast<void*>(allocation)),"fixture quarantine cleanup free");
    }
    // Graph ownership is tested with a new allocation, independently of eager leases.
    site_watchdog().begin("graph-lifetime");
    std::cout<<"begin site=graph-lifetime factor="<<descriptor.factor()<<std::endl;
    ModelContract model; model.qwen_rotary_table=std::make_shared<HipQwenRotaryTable>(descriptor);
    std::weak_ptr<const HipQwenRotaryTable> weak=model.qwen_rotary_table;
    const void* allocation=model.qwen_rotary_table->data();
    Buffer values(1024U), weights(1024U);
    std::vector<float> ones(256U,1.0F); values.put(ones.data(),1024U); weights.put(ones.data(),1024U);
    hipStream_t stream=nullptr; hipGraph_t graph=nullptr; hipGraphExec_t exec=nullptr;
    check_hip(hipStreamCreate(&stream),"lifetime stream");
    try {
        check_hip(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"lifetime begin");
        with_qwen_rotary(model,stream,[&](auto... source) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(norm_rope_kernel<SmallWeightKind::f32,decltype(source)...>),
                dim3(1),dim3(256),2048U,stream,values.as<float>(),1U,256U,256U,weights.p,0.0F,64U,
                model.rope_theta,1ULL,GraphPositionKernelArguments{},source...);
        });
        check_hip(hipStreamEndCapture(stream,&graph),"lifetime end");
        check_hip(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"lifetime instantiate");
        model.qwen_rotary_table.reset();
        if(weak.expired()) throw std::runtime_error("capture lost table owner");
        check_hip(hipGraphDestroy(graph),"source graph destruction"); graph=nullptr;
        check_hip(hipGraphLaunch(exec,stream),"lifetime replay after caller/graph release");
        check_hip(hipStreamSynchronize(stream),"lifetime replay wait");
        if(weak.expired()) throw std::runtime_error("live executable lost table allocation");
        check_hip(hipGraphExecDestroy(exec),"executable destruction"); exec=nullptr;
        check_hip(hipStreamDestroy(stream),"graph stream destruction"); stream=nullptr;
        assert_retired(weak,allocation);
    } catch (...) {
        if(exec) (void)hipGraphExecDestroy(exec);
        if(graph) (void)hipGraphDestroy(graph);
        if(stream) (void)hipStreamDestroy(stream);throw;
    }
    std::cout<<"site=independent-lifetime factor="<<descriptor.factor()
        <<" positions_checked=8 bytes_compared=0 eager_streams=2 induced_failures=6 retired_allocations=7 quarantine_checks=2\n";
}
} // namespace qwen_rotary_diagnostic
