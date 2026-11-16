// Included after Runner/Store. Only the explicitly selected layer-cut route enters here.
namespace lc = strix::pair::layercut;
namespace mlc = strix::minimax::layercut;

struct CutSlot {
    std::uint8_t* act = nullptr;
    std::uint8_t* kv[2]{};
    cudaEvent_t ready{}, copied{}, layer_ready{};
    std::mutex mutex;
    std::condition_variable cv;
    unsigned pending = 0;
    cudaStream_t copy = nullptr;
    void alloc(unsigned cut, unsigned block, cudaStream_t stream) {
        copy = stream;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&act), std::size_t{block} * mlc::seam_bytes, cudaHostAllocDefault), "cut activation ring");
        for (auto& p : kv) ck(cudaHostAlloc(reinterpret_cast<void**>(&p), std::size_t{cut} * block * 1280, cudaHostAllocDefault), "cut KV ring");
        ck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "cut ready");
        ck(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming), "cut copied");
        ck(cudaEventCreateWithFlags(&layer_ready, cudaEventDisableTiming), "cut layer ready");
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(120), [&] { return pending == 0; }))
            throw std::runtime_error("layercut: ring stalled waiting for Sender release");
    }
    void release() { std::lock_guard<std::mutex> lock(mutex); --pending; cv.notify_all(); }
    ~CutSlot() {
        if (copy) cudaStreamSynchronize(copy); // also retire a partially enqueued D2H on exceptions
        cudaFreeHost(act); for (auto* p : kv) cudaFreeHost(p);
        cudaEventDestroy(ready); cudaEventDestroy(copied); cudaEventDestroy(layer_ready);
    }
};

void run_layercut(const lc::Resolved& cfg, Store& store, const std::vector<std::uint16_t>& table,
                  const float* w_in0, const cu::RopeRow* rope, const std::vector<std::uint32_t>& tokens,
                  const std::string& destinations, const std::string& pair_identity, unsigned repeat, bool bp) {
    const auto endpoints = strix::handoff::parse_path_map(destinations);
    if (endpoints.size() != 2 || tokens.empty() || tokens.size() > UINT32_MAX || repeat == 0)
        throw std::invalid_argument("layercut: needs one nonempty prompt, two --send endpoints and --repeat >= 1");
    const unsigned cut = cfg.cut, block = cfg.block;
    const auto schedule = lc::blocks(tokens.size(), block);
    cu::SeqView seq{};
    seq.cap = static_cast<std::uint32_t>(tokens.size());
    seq.k = dalloc<std::uint16_t>(std::size_t{cut} * seq.cap * 512);
    seq.v = dalloc<std::uint16_t>(std::size_t{cut} * seq.cap * 512);
    seq.ik = dalloc<std::uint16_t>(std::size_t{cut} * seq.cap * 128);
    cu::SeqView* seq_d = dalloc<cu::SeqView>(1);
    Runner run;
    // A separate scratch runner for the short tail keeps R-dependent strides/maps correct.
    run.alloc(std::min<std::size_t>(block, tokens.size()), (seq.cap + 127) / 128);
    run.row_seq = dupload(std::vector<std::uint32_t>(run.R, 0));
    struct CopyStream {
        cudaStream_t value = nullptr;
        CopyStream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cut copy stream"); }
        ~CopyStream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
    } copy_owner;
    const auto copy = copy_owner.value;
    CutSlot ring[2];
    for (auto& slot : ring) slot.alloc(cut, block, copy);
    const std::string id = mlc::identity(cut, pair_identity);
    std::printf("identity %s\n", id.c_str());
    struct Times { std::mutex mutex; Clock::time_point start; double act = 0, kv = 0; };
    for (unsigned it = 0; it < repeat; ++it) {
        strix::handoff::SenderOptions options;
        options.codec = bp ? strix::handoff::Codec::bp_v1 : strix::handoff::Codec::identity;
        options.max_pending_plain_bytes = std::size_t{2} * block * (cut * 1280 + mlc::seam_bytes);
        lc::Meta meta{1, cut, block, mlc::hidden, tokens.size(), lc::token_fnv(tokens), id};
        std::unique_ptr<strix::handoff::Sender> sender[2];
        for (unsigned rank = 0; rank < 2; ++rank)
            sender[rank] = std::make_unique<strix::handoff::Sender>(endpoints[rank],
                strix::handoff::Hello{it + 1ULL, mlc::format, rank, 2, lc::pack(meta)}, options);
        const auto start = Clock::now();
        double compute = 0;
        auto times = std::make_shared<Times>(); times->start = start;
        for (std::size_t b = 0; b < schedule.size(); ++b) {
            auto& slot = ring[b % 2];
            slot.wait(); // bounded backpressure; callbacks return every submitted buffer
            const auto desc = schedule[b];
            if (run.R != desc.rows) {
                run.release(); run.tiles_d = nullptr;
                run.alloc(desc.rows, (seq.cap + 127) / 128);
                run.row_seq = dupload(std::vector<std::uint32_t>(run.R, 0));
            }
            seq.rows = desc.rows; seq.pos0 = static_cast<std::uint32_t>(desc.row0);
            ck(cudaMemcpy(seq_d, &seq, sizeof(seq), cudaMemcpyHostToDevice), "cut sequence");
            cudaFree(run.tiles_d);
            std::vector<uint2> tiles;
            for (unsigned r = 0; r < desc.rows; r += 32) tiles.push_back(make_uint2(r, std::min(32U, desc.rows - r)));
            run.tiles_d = dupload(tiles); run.n_tiles = static_cast<unsigned>(tiles.size());
            std::vector<std::uint32_t> embedding(std::size_t{desc.rows} * kHidden);
            for (unsigned r = 0; r < desc.rows; ++r) {
                const auto token = tokens[static_cast<std::size_t>(desc.row0) + r];
                if (token >= table.size() / kHidden) throw std::invalid_argument("layercut: token outside embedding");
                for (unsigned j = 0; j < kHidden; ++j)
                    embedding[std::size_t{r} * kHidden + j] = std::uint32_t{table[std::size_t{token} * kHidden + j]} << 16;
            }
            const auto begin = Clock::now();
            ck(cudaMemcpy(run.h, embedding.data(), embedding.size() * 4, cudaMemcpyHostToDevice), "cut embed");
            cu::rmsnorm(run.h, w_in0, run.hn, desc.rows, nullptr);
            for (unsigned layer = 0; layer < cut; ++layer) {
                run.layer(hx::bind(store.meta[layer], store.dev[layer]), layer, seq_d, rope);
                ck(cudaEventRecord(slot.layer_ready, nullptr), "cut layer event");
                ck(cudaStreamWaitEvent(copy, slot.layer_ready, 0), "cut copy waits layer");
                for (unsigned rank = 0; rank < 2; ++rank)
                    cu::export_layer(seq, layer, rank, desc.row0, desc.rows,
                        slot.kv[rank] + std::size_t{layer} * block * 1280, copy);
            }
            ck(cudaEventRecord(slot.ready, nullptr), "cut residual ready");
            ck(cudaStreamWaitEvent(copy, slot.ready, 0), "cut waits residual");
            ck(cudaMemcpyAsync(slot.act, run.h, std::size_t{desc.rows} * mlc::seam_bytes, cudaMemcpyDeviceToHost, copy), "cut residual copy");
            ck(cudaEventRecord(slot.copied, copy), "cut copied");
            // Host waits for D2H, never a compute-stream wait on the network. Next block overlaps send and pair compute.
            ck(cudaEventSynchronize(slot.copied), "cut copy complete");
            const double elapsed = secs(begin, Clock::now());
            compute += elapsed;
            {
                std::lock_guard<std::mutex> lock(slot.mutex);
                slot.pending = 2 * cut + (cfg.act == lc::Act::direct ? 2 : 1);
            }
            for (unsigned rank = 0; rank < 2; ++rank) {
                if (rank == 0 || cfg.act == lc::Act::direct) {
                    const strix::bp_codec::Region region{slot.act, std::size_t{desc.rows} * mlc::seam_bytes, 4};
                    sender[rank]->submit({lc::kAct, static_cast<std::uint32_t>(b), cfg.act == lc::Act::relay ? strix::handoff::kRelay : 0,
                                          desc.row0, desc.rows}, {&region, 1}, [times, slotp = &slot] {
                        { std::lock_guard<std::mutex> lock(times->mutex); times->act = std::max(times->act, secs(times->start, Clock::now())); }
                        slotp->release();
                    });
                }
                for (unsigned layer = 0; layer < cut; ++layer) {
                    const strix::bp_codec::Region region{slot.kv[rank] + std::size_t{layer} * block * 1280, std::size_t{desc.rows} * 1280, 2};
                    sender[rank]->submit({lc::kKv, layer, 0, desc.row0, desc.rows}, {&region, 1}, [times, slotp = &slot] {
                        { std::lock_guard<std::mutex> lock(times->mutex); times->kv = std::max(times->kv, secs(times->start, Clock::now())); }
                        slotp->release();
                    });
                }
            }
            std::printf("layercut: run %u block %zu row0 %llu rows %u rtx %.6f s submitted %.6f s\n", it, b,
                static_cast<unsigned long long>(desc.row0), desc.rows, elapsed, secs(start, Clock::now()));
        }
        // Both ranks must see END before either can finish the pair prefill/import collective.
        strix::handoff::Reply replies[2];
        std::exception_ptr failures[2];
        double ttft = 0;
        std::thread finishers[2];
        try {
        for (unsigned rank = 0; rank < 2; ++rank) finishers[rank] = std::thread([&, rank] {
            try { replies[rank] = sender[rank]->finish(); if (rank == 0) ttft = secs(start, Clock::now()); }
            catch (...) { failures[rank] = std::current_exception(); sender[1 - rank]->abort("peer finish failed"); }
        });
        } catch (...) {
            for (auto& s : sender) s->abort("layercut cannot start finish thread");
            for (auto& t : finishers) if (t.joinable()) t.join();
            throw;
        }
        for (auto& t : finishers) t.join();
        for (auto& e : failures) if (e) std::rethrow_exception(e);
        if (replies[0].status || replies[1].status || replies[0].payload.size() != 4)
            throw std::runtime_error("layercut: rejected or malformed token reply");
        std::uint32_t token = 0; std::memcpy(&token, replies[0].payload.data(), 4);
        double act_last = times->act, kv_last = times->kv;
        if (bp) { // bp release means encoded, not sent. The aggregate last-byte time bounds ACT and KV.
            act_last = kv_last = std::max(sender[0]->stats().last_byte, sender[1]->stats().last_byte);
        }
        std::printf("layercut: TTFT %.6f s, rtx compute %.6f s, act last byte %.6f s, kv last byte %.6f s, cut %u block %u; run %u %s token %u%s\n",
            ttft, compute, act_last, kv_last, cut, block, it, it == 0 ? "cold" : "warm", token,
            bp ? " (bp last-byte values are aggregate upper bounds from session open)" : "");
        std::fflush(stdout);
    }
    ck(cudaStreamSynchronize(copy), "cut cleanup");
    run.release(); cudaFree(seq_d); cudaFree(seq.k); cudaFree(seq.v); cudaFree(seq.ik);
}
