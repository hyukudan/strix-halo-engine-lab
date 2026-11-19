// Stateful decode fixture: projected inputs are identical on both simulated
// ranks; only owner writes. This does not emulate model-wide feedback or HIP.
float original_pool_channel(const std::uint16_t* key, const std::uint16_t* gate,
                            const float* ape, unsigned i) {
    float l4[4], kk[4], e[4], p[4];
    for (unsigned j = 0; j < 4; ++j) {
        l4[j] = G::bf16_to_float(gate[j * G::kIdxDim + i]) + ape[j * G::kIdxDim + i];
        kk[j] = G::bf16_to_float(key[j * G::kIdxDim + i]);
    }
    float m = l4[0];
    for (unsigned j = 1; j < 4; ++j) m = l4[j] > m ? l4[j] : m;
    for (unsigned j = 0; j < 4; ++j) e[j] = G::declared_exp(l4[j] - m);
    const float sum = ((e[0] + e[1]) + e[2]) + e[3];
    for (unsigned j = 0; j < 4; ++j) p[j] = e[j] / sum;
    return ((p[0] * kk[0] + p[1] * kk[1]) + p[2] * kk[2]) + p[3] * kk[3];
}

struct DecodeCache {
    std::vector<std::uint16_t> latent, key, gate;
    std::vector<float> pooled;
    explicit DecodeCache(std::size_t capacity)
        : latent(capacity * G::kKvLora, 0x7fc1U), key(capacity * G::kIdxDim, 0x7fc1U),
          gate(capacity * G::kIdxDim, 0x7fc1U), pooled(capacity / 4 * G::kIdxDim, -999.0F) {}
    void append(std::uint64_t address, std::uint64_t token, const std::array<float, 4 * G::kIdxDim>& ape,
                bool original = false) {
        std::array<float, G::kIdxDim> raw{}, centered{};
        for (unsigned i = 0; i < G::kIdxDim; ++i)
            raw[i] = static_cast<float>(static_cast<int>((token * 17 + i * 13 + token / 4) % 97) - 48) / 16.0F;
        const float mean = G::tree_sum<128>([&](unsigned i) { return raw[i]; }, G::kIdxDim) / static_cast<float>(G::kIdxDim);
        for (unsigned i = 0; i < G::kIdxDim; ++i) centered[i] = raw[i] - mean;
        const float variance = G::tree_sumsq<128>([&](unsigned i) { return centered[i]; }, G::kIdxDim) / static_cast<float>(G::kIdxDim);
        const float inverse = G::inv_sqrt(variance + G::kLnEps);
        for (unsigned i = 0; i < G::kIdxDim; ++i) {
            key.at(static_cast<std::size_t>(address) * G::kIdxDim + i) = G::float_to_bf16((centered[i] * inverse) * 1.0F + 0.0F);
            gate.at(static_cast<std::size_t>(address) * G::kIdxDim + i) = G::float_to_bf16(
                static_cast<float>(static_cast<int>((token * 3 + i * 7) % 29) - 14) / 16.0F);
        }
        for (unsigned i = 0; i < G::kKvLora; ++i)
            latent.at(static_cast<std::size_t>(address) * G::kKvLora + i) = G::float_to_bf16(
                static_cast<float>(static_cast<int>((token * 7 + i * 13) % 61) - 30) / 16.0F);
        if ((address + 1) % 4 == 0) {
            const auto start = static_cast<std::size_t>(address - 3) * G::kIdxDim;
            for (unsigned i = 0; i < G::kIdxDim; ++i)
                pooled.at(static_cast<std::size_t>(address / 4) * G::kIdxDim + i) =
                    original ? original_pool_channel(key.data() + start, gate.data() + start, ape.data(), i) :
                    strix::attn::group_pool::gated_bf16_channel<4>(key.data() + start, gate.data() + start, ape.data(), i,
                                                                    G::kIdxDim, G::declared_exp);
        }
    }
};

void incremental_decode() {
    constexpr std::uint64_t capacity = 8244; // final half-pair exercises rounded allocation
    DecodeCache full(capacity);
    std::array<DecodeCache, 2> ranks{DecodeCache(K::storage_tokens(capacity)), DecodeCache(K::storage_tokens(capacity))};
    std::array<float, 4 * G::kIdxDim> ape{};
    std::vector<float> query(G::kIdxHeads * G::kIdxDim), weights(G::kIdxHeads);
    for (std::size_t i = 0; i < ape.size(); ++i) ape[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 16.0F;
    for (std::size_t i = 0; i < query.size(); ++i) query[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 16.0F;
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(i % 7 + 1) / 16.0F;
    std::array<std::vector<std::uint32_t>, 2> local_keys;
    std::vector<std::uint32_t> full_keys;
    unsigned selected_new_pools = 0, rejected_new_pools = 0;
    float largest_error = 0.0F;
    std::uint64_t first_attention_bit_difference = 0;
    for (std::uint64_t token = 0; token <= 8240; ++token) {
        full.append(token, token, ape, true); // pre-change kernel formula is the independent control
        const unsigned owner = K::owner(token);
        ranks[owner].append(K::local_token(token), token, ape);
        // Before completion no pool key may be overwritten; the non-owner's
        // entire newly addressed pool remains poisoned until its own append.
        for (unsigned r = 0; r < 2; ++r)
            require(K::owned_prefix(token + 1, r) <= K::storage_tokens(capacity), "owned-prefix capacity overflow");
        if ((token + 1) % 4 == 0) {
            const auto pool = static_cast<std::size_t>(token / 4);
            const auto local = pool / 2;
            for (unsigned i = 0; i < G::kIdxDim; ++i)
                require(full.pooled[pool * G::kIdxDim + i] == ranks[owner].pooled[local * G::kIdxDim + i], "first pooled-key divergence");
            full_keys.push_back(G::ordered_key(score(query.data(), full.pooled.data() + pool * G::kIdxDim, weights.data())));
            for (unsigned r = 0; r < 2; ++r)
                local_keys[r].push_back(r == owner ? G::ordered_key(score(query.data(), ranks[r].pooled.data() + local * G::kIdxDim, weights.data())) : 0xffffffffU);
        }
        if (token < 8192) continue;
        const auto expected = select(full_keys, token);
        std::array<std::vector<std::uint64_t>, 2> selected;
        for (unsigned r = 0; r < 2; ++r) {
            const auto merged = K::merge_score_keys(local_keys[r], local_keys[1 - r], r);
            require(merged == full_keys, "first score-key divergence");
            std::array<unsigned, 256> histogram{}, reference_histogram{};
            for (const auto key : merged) ++histogram[key >> 24U];
            for (const auto key : full_keys) ++reference_histogram[key >> 24U];
            require(histogram == reference_histogram, "first histogram divergence");
            selected[r] = select(merged, token);
            require(selected[r] == expected, "first selected-global-ID divergence");
        }
        if ((token + 1) % 4 == 0) {
            if (std::binary_search(expected.begin(), expected.end(), token)) ++selected_new_pools;
            else ++rejected_new_pools;
        }
        const auto tail = (token + 1) / 4 * 4;
        for (std::uint64_t t = tail; t <= token; ++t)
            require(std::binary_search(expected.begin(), expected.end(), t), "non-owner selection lost mandatory tail");
        std::vector<O::Row> reference;
        for (const auto id : expected) {
            O::Row row{};
            row.id = row.token = id;
            row.value.resize(G::kKvLora);
            float dot = 0.0F;
            for (unsigned i = 0; i < G::kKvLora; ++i) {
                row.value[i] = G::bf16_to_float(full.latent[static_cast<std::size_t>(id) * G::kKvLora + i]);
                const auto stored = ranks[K::owner(id)].latent.at(static_cast<std::size_t>(K::local_token(id)) * G::kKvLora + i);
                require(G::float_to_bf16(row.value[i]) == stored, "first appended-latent divergence");
                dot = std::fma(query[i], row.value[i], dot);
            }
            row.score = dot * G::kMlaScale;
            reference.push_back(std::move(row));
        }
        const auto a = policy_attention<true>(reference, 0), b = policy_attention<true>(reference, 1);
        const auto replica = policy_attention<false>(reference, 0);
        float maximum = -std::numeric_limits<float>::infinity();
        for (const auto& row : reference) maximum = std::max(maximum, row.score);
        std::vector<float> probability(reference.size());
        for (std::size_t i = 0; i < reference.size(); ++i) probability[i] = G::declared_exp(reference[i].score - maximum);
        const float denominator = G::tree_sum<256>([&](unsigned i) { return probability[i]; }, static_cast<unsigned>(probability.size()));
        for (auto& value : probability) value = value / denominator;
        for (unsigned dim = 0; dim < G::kKvLora; ++dim) {
            const float baseline = G::tree_fma<256>([&](unsigned i) { return probability[i]; },
                [&](unsigned i) { return reference[i].value[dim]; }, static_cast<unsigned>(reference.size()));
            const float sharded = K::merge_component(a.m, a.l, a.a[dim], b.m, b.l, b.a[dim]);
            require(__builtin_bit_cast(std::uint32_t, replica.a[dim]) == __builtin_bit_cast(std::uint32_t, baseline),
                    "incremental replicated policy changed bits");
            largest_error = std::max(largest_error, std::abs(sharded - baseline));
            if (sharded != baseline && first_attention_bit_difference == 0) first_attention_bit_difference = token;
            require(std::isfinite(sharded) && std::abs(sharded - baseline) < 1e-5F, "first attention divergence beyond reorder allowance");
        }
    }
    require(selected_new_pools != 0 && rejected_new_pools != 0, "new-pool selection coverage missing");
    std::printf("incremental positions 8192..8240: first cache/key/histogram/ID divergence NONE; first attention rounding difference %llu; attention abs %.9g; new pools selected/rejected %u/%u\n",
                static_cast<unsigned long long>(first_attention_bit_difference), static_cast<double>(largest_error), selected_new_pools, rejected_new_pools);
}

void decode_dump_format() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::path("glm-kvshard-dump-test-" + std::to_string(suffix));
    K::write_decode_dump(root.string(), 1, 3, 7, 8199, {8196, 8197, 8198, 8199},
                         {G::ordered_key(-1.5F), G::ordered_key(2.0F)});
    const auto directory = root / "rank-1";
    const auto selected_path = directory / "epoch-3-layer-7-pos-8199-selected.csv";
    const auto scores_path = directory / "epoch-3-layer-7-pos-8199-pools.csv";
    std::ifstream selected(selected_path), scores(scores_path);
    std::string line;
    std::getline(selected, line); std::getline(selected, line); std::getline(selected, line);
    require(line == "0,8196,1,4096", "dump lost global selection / compact address");
    std::getline(scores, line); std::getline(scores, line); std::getline(scores, line);
    require(line.rfind("0,0,0,403fffff,", 0) == 0 &&
            std::strtof(line.c_str() + 15, nullptr) == -1.5F, "dump score key roundtrip changed");
    selected.close(); scores.close();
    std::filesystem::remove(selected_path); std::filesystem::remove(scores_path);
    std::filesystem::remove(directory); std::filesystem::remove(root);
}
