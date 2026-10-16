// Included only by the explicit real-model qualification executable, after its
// loader/snapshot helpers. No automatic CTest, timing, or hardware admission.
struct CompositionFailure final : strix::EagerExecutionDispatcher {
    strix::EagerExecutionDispatcher& inner;
    bool fail_routed = false;
    bool observed_gdn = false;
    explicit CompositionFailure(strix::EagerExecutionDispatcher& target) : inner(target) {}
    void dispatch_kernel(const strix::ExecutionDispatchContext& c) override {
        if (fail_routed && c.op == strix::SerialOp::moe_experts) {
            require(observed_gdn, "failure injection must follow real GDN work");
            throw std::runtime_error("composition injected routed failure");
        }
        inner.dispatch_kernel(c);
        if (c.op == strix::SerialOp::gated_delta_net) observed_gdn = true;
    }
    void dispatch_runtime_step(const strix::ExecutionDispatchContext& c) override {
        inner.dispatch_runtime_step(c);
    }
};
int qualify_islands_overlap(const Options& args) {
    require(args.rows == 8U, "islands-overlap composition requires eight successive token positions");
    const auto teacher = teacher_tokens(args, strix::ModelContract::qwen38_flash_next().vocab_size);
    const auto executable = std::filesystem::canonical("/proc/self/exe");
    const auto executable_sha = hash_file(executable);
    require(std::filesystem::create_directory(args.output), "cannot create fresh composition output directory");
    strix::ExecutionBindingRequest binding;
    strix::ExecutionBufferRequest buffers;
    const auto config = configure(args, binding, buffers);
    const auto owner = load(args, config, binding, buffers);
    require(owner->overlap_plan.has_value(), "composition requires the actual expanded arena plan");
    strix::SerialSessionContext initial_session;
    initial_session.execution_identity = owner->serial.execution_identity;
    initial_session.runtime_compatibility_identity = strix::build_runtime_execution_identities(
        owner->serial.execution_identity, owner->serial.arena, owner->resources).compatibility_identity;
    initial_session.state_layout_identity = owner->layout.layout_identity;
    for (const auto& allocation : owner->layout.allocations)
        initial_session.state.valid = initial_session.state.valid | allocation.state_flag;
    const auto initial = dump_state(owner, args.output, "composition-initial");
    std::vector<std::string> reference_logits;
    std::vector<StateFile> reference_state;
    std::string reference_metadata;
    auto receipt = output_file(args.output / "composition.pending.json");
    receipt << "{\"schema\":\"real-model-islands-overlap-composition-v1\",\"cases\":[";
    unsigned cases = 0;
    // The identical expanded compiled plan/arena is used even when overlap is
    // disabled. This isolates launch organization without changing identities.
    for (unsigned mode = 0; mode < 12U; ++mode) {
        const bool negative = mode == 8U || mode == 9U;
        const bool use_islands = mode >= 10U || negative || (mode & 1U) != 0U;
        const bool use_overlap = mode >= 10U || negative || (mode & 2U) != 0U;
        const bool use_prepared = mode >= 10U || negative || (mode & 4U) != 0U;
        auto session = initial_session;
        restore_state(owner, initial, initial_session, session);
        auto d1 = std::make_shared<HipDeferredMoeStatus>(owner->resources, owner->serial.execution_identity,
            owner->generations.next(), owner->generations.next(), owner, owner->guard);
        auto catalog = std::make_shared<strix::HipLauncherCatalog>(owner->model, config);
        std::shared_ptr<strix::HipGdnIslands> islands;
        if (use_islands) islands = strix::HipGdnIslands::create(owner->serial, catalog->gdn_config(),
            d1->resources(), d1->scope(), owner->resident, d1, owner->guard);
        strix::ExecutionLauncherRegistry registry;
        catalog->register_for_compiled_execution(owner->serial, registry, &d1->resources(), islands);
        const auto launches = strix::compile_launch_plan(owner->serial, registry);
        strix::DirectLaunchDispatcher direct(launches);
        std::unique_ptr<strix::HipMoeOverlapDispatcher> overlap;
        if (use_overlap) overlap = std::make_unique<strix::HipMoeOverlapDispatcher>(owner->serial,
            *owner->overlap_plan, launches, d1->resources(), d1->scope().stream, d1, catalog, owner->guard);
        strix::EagerExecutionDispatcher& selected = overlap
            ? static_cast<strix::EagerExecutionDispatcher&>(*overlap)
            : static_cast<strix::EagerExecutionDispatcher&>(direct);
        CompositionFailure dispatcher(selected);
        dispatcher.fail_routed = mode == 8U;
        std::unique_ptr<strix::PreparedEagerExecution> prepared;
        if (use_prepared) {
            strix::EagerExecutionRequest prototype;
            prototype.state = session.state;
            prototype.input_token_id = teacher.front();
            prepared = std::make_unique<strix::PreparedEagerExecution>(owner->serial,
                d1->resources(), prototype, d1);
        }
        std::uint64_t fork_tokens = 0;
        std::vector<std::string> observed_logits;
        bool expected_failure = false;
        const auto count = negative ? 1U : args.start + args.rows;
        for (std::uint64_t position = 0; position < count; ++position) {
            d1->begin();
            try {
                strix::HipLaunchScope scope(d1->scope());
                if (overlap) {
                    const auto schedule = mode == 10U || (mode == 11U && position % 2U == 0U)
                        ? strix::MoeTokenSchedule::serial_expanded : strix::MoeTokenSchedule::fork_join;
                    overlap->begin_token(session,schedule);
                    if (schedule == strix::MoeTokenSchedule::fork_join) ++fork_tokens;
                }
                const auto completion = [&] {
                    if (overlap) overlap->verify_before_commit();
                    if (mode == 9U) throw std::runtime_error("composition injected completion failure");
                    d1->complete_before_commit();
                };
                if (prepared) (void)strix::execute_prepared_serial_token(*prepared, session,
                    d1->resources(), teacher[static_cast<std::size_t>(position)], dispatcher, completion);
                else (void)strix::execute_serial_token(owner->serial, session, d1->resources(),
                    teacher[static_cast<std::size_t>(position)], dispatcher, completion);
                if (overlap) overlap->mark_committed(session);
            } catch (const std::exception& error) {
                if (overlap) overlap->abort();
                d1->abort();
                const std::string_view message(error.what());
                require(negative && message == (mode == 8U ? "composition injected routed failure" :
                    "composition injected completion failure"), "unexpected composition execution failure");
                require(session.health == strix::SerialSessionHealth::poisoned &&
                    session.arithmetic_lineage.tainted && session.next_position == 0 &&
                    session.committed_tokens == 0 && !owner->prefetch.active,
                    "failed real-model composition published metadata or retained PLE work");
                require(overlap->ticket().state == strix::HipMoeWorkState::failed_drained,
                    "failed composition did not drain both streams");
                expected_failure = true;
                break;
            }
            require(session.health == strix::SerialSessionHealth::ready && !owner->prefetch.active,
                "composition token did not complete before publication");
            if (position >= args.start) {
                const auto path = args.output / ("composition-" + std::to_string(mode) + "-" +
                    std::to_string(position) + ".f32");
                (void)logits(owner, strix::resolve_execution_buffer(owner->serial.execution_identity,
                    owner->serial.arena, owner->memory->scratch(), owner->serial.io.steps.back().writes.front()), path);
                observed_logits.push_back(hash_file(path));
            }
        }
        if (!negative) {
            const auto state = dump_state(owner, args.output, "composition-" + std::to_string(mode));
            std::ostringstream metadata;
            write_session(metadata, session);
            if (!mode) { reference_logits = observed_logits; reference_state = state; reference_metadata = metadata.str(); }
            require(observed_logits == reference_logits && metadata.str() == reference_metadata && state.size() == 7U,
                "composition changed full FP32 logits, six generations, position, commits or lineage");
            for (std::size_t i = 0; i < state.size(); ++i)
                require(state[i].component == reference_state[i].component && state[i].bytes == reference_state[i].bytes &&
                    state[i].sha == reference_state[i].sha && equal_files(state[i].path, reference_state[i].path),
                    "composition changed complete persistent state bytes");
            if (islands) require(islands->receipt().replays == count * islands->receipt().islands,
                "composition island receipt does not count all actual GDN replays");
            if (overlap) require(overlap->stats().committed_tokens == count &&
                overlap->stats().join_waits == fork_tokens * owner->overlap_plan->layers().size() &&
                overlap->stats().fork_join_tokens == fork_tokens && overlap->stats().serial_tokens == count - fork_tokens,
                "composition overlap receipt differs from actual committed tokens/layers");
        } else require(expected_failure, "injected composition failure was not reached");
        receipt << (cases ? "," : "") << "{\"islands\":" << (use_islands ? "true" : "false")
            << ",\"schedule_case\":" << mode
            << ",\"overlap\":" << (use_overlap ? "true" : "false") << ",\"prepared\":"
            << (use_prepared ? "true" : "false") << ",\"injected_failure\":" << (negative ? "true" : "false")
            << ",\"session\":";
        write_session(receipt, session);
        receipt << ",\"logit_sha256\":[";
        for (std::size_t i = 0; i < observed_logits.size(); ++i) receipt << (i ? "," : "") << quote(observed_logits[i]);
        receipt << "]}";
        ++cases;
        // All work is quiescent before reverse-order destruction releases
        // prepared snapshots, overlap side stream, catalog graphs and D1.
        owner->guard->recover_or_fail_stop();
    }
    require(cases == 12U && hash_file(args.model) == args.model_sha &&
        hash_file(args.teacher) == args.teacher_sha && hash_file(executable) == executable_sha,
        "composition source inputs changed or case coverage is incomplete");
    receipt << "],\"finished\":true,\"passed\":true,\"full_model\":true,\"model_sha256\":"
        << quote(args.model_sha) << ",\"teacher_sha256\":" << quote(args.teacher_sha)
        << ",\"executable_sha256\":" << quote(executable_sha)
        << ",\"quality_qualification\":false,\"performance_measurement\":false}\n";
    receipt.close();
    std::filesystem::rename(args.output / "composition.pending.json", args.output / "composition.json");
    return 0;
}
