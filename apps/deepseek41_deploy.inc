// Included in the serve app's anonymous namespace, after its options and host sizing helpers.
// No device allocation or weight loading is permitted in this provider.
class Ds41FootprintProvider final : public strix::deploy::FootprintProvider {
public:
    explicit Ds41FootprintProvider(const Args& a) : a_(a) {}
    std::string model() const override { return a_.model_id; }
    std::vector<std::string> topologies() const override { return {"tp2"}; }
    std::vector<strix::deploy::Format> formats(const std::string&) const override { return {{"native", "native"}}; }
    std::uint32_t max_slots() const override { return BatchForward::configured_row_limit(); }
    BatchForward::Footprint native_footprint(std::uint32_t context, std::uint32_t slots, std::uint32_t rank) const {
        Args selected = a_;
        selected.context = context; selected.slots = slots;
        return BatchForward::footprint(a_.model, forward_options(selected, nullptr, rank));
    }
    strix::deploy::Footprint footprint(const strix::deploy::Spec& spec, const strix::deploy::Candidate& c,
                                     std::uint32_t rank) const override {
        const auto profile = strix::profile::name(strix::deepseek41::pair_defaults::profile_resolution().profile);
        if (spec.profile != "auto" && spec.profile != profile)
            throw std::invalid_argument("profile pin differs from active --profile; select it explicitly at launch");
        // These routes need header-only sizing before they can be admitted by deployment.
        // Existing launches without --deploy retain their established runtime admission.
        if (!a_.dspark.empty()) throw std::invalid_argument("DSpark: header-only peak footprint not implemented");
        if (!a_.hetero_listen.empty()) throw std::invalid_argument("heterogeneous stage: footprint not implemented");
        if (a_.prefix_cache_mb || a_.session_ram_bytes)
            throw std::invalid_argument("prefix/session cache: peak footprint not implemented; use --cache-profile off and no session RAM");
        const auto f = native_footprint(c.context, c.slots, rank);
        if (f.trellis) throw std::invalid_argument("trellis: pre-load library workspace sizing unavailable");
        strix::deploy::Footprint out;
        out.weights = f.weights;
        // BatchForward's state includes KV rings/carries. Charge the complete state to
        // KV first (never a linear extrapolation across compressed-cache boundaries).
        if (f.state_per_slot && c.slots > (std::numeric_limits<std::uint64_t>::max() - f.pool) / f.state_per_slot)
            throw std::overflow_error("DS41 state byte overflow");
        out.kv_total = f.state_per_slot * c.slots + f.pool;
        out.workspace = f.workspace;
        if (a_.device_prefill) {
            namespace sc = strix::deepseek41::sched;
            if (!sc::auto_wide_pieces(c.context, [](const char* key) {
                    const char* value = std::getenv(key); return value ? std::string(value) : std::string();
                })) throw std::invalid_argument("prefill workspace estimate unqualified for this context/options; use --prefill-path rows");
            strix::deepseek41::PrefillOptions po;
            po.layers = a_.layers; po.max_positions = c.context; po.rank = rank; po.world = 2;
            po.engram_direct_io = a_.engram_direct;
            po.expert_tp = strix::ds41::expert_tp::enabled_from_env();
            po.max_rows = a_.prefill_rows_auto ? std::min(sc::kAutoPrefillRows[0], c.context) : std::min(a_.prefill_rows, c.context);
            if (po.max_rows > sc::kAutoPrefillRows[0]) throw std::invalid_argument("prefill rows exceed measured workspace estimate range");
            const auto scratch = sc::prefill_piece_bytes(po.max_rows, c.context);
            if (scratch > std::numeric_limits<std::uint64_t>::max() - out.workspace)
                throw std::overflow_error("DS41 workspace byte overflow");
            out.workspace += scratch;
            // Empty shared_weights deliberately counts every prefill-owned weight. This
            // is conservative: live prefill can share some BatchForward tensors.
            out.extra.emplace_back("prefill_weights_upper_bound", strix::deepseek41::DevicePrefill::owned_weight_bytes(a_.model, po));
        }
        return out;
    }
private:
    const Args& a_;
};

std::string deployment_link(const std::string& transport) {
    // Never put TCP addresses or ports into a public plan.
    if (transport.starts_with("tcp-")) return "tcp";
    return transport;
}
void prepare_deployment(Args& a) {
    if (a.transport == "emulate") throw std::invalid_argument("deployment: emulated ranks do not provide two measured nodes");
    auto spec = strix::deploy::load_spec(a.deploy);
    if (!spec.pins.contains("context") && !spec.pins.contains("memory.context")) spec.context = a.context;
    if (!spec.pins.contains("slots") && !spec.pins.contains("memory.slots")) spec.slots = a.slots;
    if (!spec.nodes) spec.nodes = 2;
    if (spec.link == "auto") spec.link = deployment_link(a.transport);
    else if (spec.link != deployment_link(a.transport))
        throw std::invalid_argument("deployment link pin differs from --transport");
    if (spec.model.empty()) spec.model = a.model_id;
    a.context = spec.context; a.slots = spec.slots;
    a.deploy_spec = std::make_shared<const strix::deploy::Spec>(std::move(spec));
}

void resolve_live_deployment(Args& a, std::uint32_t rank, strix::dist::Transport& transport) {
    // Host transport exchange before Communicator/BatchForward construction. Both ranks
    // resolve the same rank-ordered, freshly measured inventory, including mixed memory sizes.
    struct Wire {
        std::uint64_t magic, rank, version, cu, total, free, unified;
        std::array<char, 32> gfx, source;
    } mine{}, peer{};
    static_assert(sizeof(Wire) == 120);
    const auto local = strix::deploy::probe_device();
    mine.magic = 0x53544445504c3031ULL; mine.rank = rank;
    mine.version = local.gfx_target_version; mine.cu = local.cu_count;
    mine.total = local.mem_total_bytes; mine.free = local.mem_free_bytes; mine.unified = local.unified ? 1 : 0;
    std::copy_n(local.gfx_target.begin(), std::min(local.gfx_target.size(), mine.gfx.size() - 1), mine.gfx.begin());
    std::copy_n(local.source.begin(), std::min(local.source.size(), mine.source.size() - 1), mine.source.begin());
    transport.exchange(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
    if (peer.magic != mine.magic || peer.rank != 1U - rank || peer.unified > 1 || peer.cu > UINT32_MAX ||
        peer.gfx.back() || peer.source.back()) throw std::runtime_error("deployment: incompatible peer device facts");
    strix::deploy::DeviceFacts remote;
    remote.gfx_target_version = peer.version; remote.gfx_target = peer.gfx.data();
    remote.cu_count = static_cast<std::uint32_t>(peer.cu);
    remote.mem_total_bytes = peer.total; remote.mem_free_bytes = peer.free;
    remote.unified = peer.unified != 0; remote.source = peer.source.data();
    std::vector<strix::deploy::DeviceFacts> devices(2);
    devices[rank] = local; devices[1U - rank] = std::move(remote);
    auto plan = strix::deploy::resolve(*a.deploy_spec, devices, Ds41FootprintProvider(a));
    const auto id = plan.hash();
    std::array<char, 16> own_hash{}, peer_hash{};
    std::copy(id.begin(), id.end(), own_hash.begin());
    transport.exchange(std::as_bytes(std::span(own_hash)), std::as_writable_bytes(std::span(peer_hash)));
    if (own_hash != peer_hash) throw std::runtime_error("deployment: peer plan hash mismatch");
    if (plan.refused) throw std::runtime_error("deployment refused: " + plan.refusal +
        (plan.reasons.empty() ? "" : "; " + plan.reasons.back()));
    a.context = plan.chosen.context; a.slots = plan.chosen.slots;
    a.deployment_plan = std::make_shared<const strix::deploy::Plan>(std::move(plan));
    std::fprintf(stderr, "deploy: id=%s topology=tp2 context=%u slots=%u\n", id.c_str(), a.context, a.slots);
}
