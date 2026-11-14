#pragma once
// Host-only receipt adapter. This is a partial description of the resolved defaults,
// not a complete execution receipt or evidence of numerical compatibility. Callers
// must add verified asset, state, build, device and workload digests when available.
#include "glm_rank_defaults.hpp"

namespace strix::glm::rank_defaults {

inline strix::profile::NumericalIdentity numerical_identity(const Effective& e) {
    strix::profile::NumericalIdentity identity{};
    // Keep the declared arithmetic separate from the selected R1/R1b mode.
    // R1b is not an assertion of R1 bit equality. See glm_math.hpp and the
    // rank app's R1 prompt / R1b generation contract.
    identity.arithmetic_contract = "src/glm/common/glm_math.hpp: IEEE binary32; R1 ascending fmaf chain; "
                                   "R1b 256-term ascending fmaf blocks then pairwise halving tree; "
                                   "R2 lane-strided fmaf partials with pairwise halving tree";
    identity.reduction_plan = e.identity.value;
    identity.prefill_lineage = e.prefill;
    if (const char* f = std::getenv("STRIX_GLM_MODEL_FORMAT"); f != nullptr && std::string(f) == "fast_i4") {
        identity.reduction_plan += ";model-format=fast_i4;stored-int4-g64-v1;all-routed;exl3-routed=absent";
        identity.arithmetic_contract += "; glm-fast-i4: routed experts IU4 g64 (EXL3 mul1 decode -> per-column K64 FP16 absmax/7 scale, RNE "
                                        "nibbles), prefill W4A4 grouped IU4 (activations per row K64), decode W4A16 (F32 dot2 per group, fmaf scale)";
    }
    if (e.kv_shard) {
        identity.state_format = "glm-dsa-owned-blocks-v1;layout=474c4d4b56533033;block=4;pool-parity;rank/capacity/frontier-bound";
        identity.reduction_plan += std::string(";") + kvshard::numerical_identity;
        identity.arithmetic_contract += "; DSA: global score keys and existing selection; FP32 owned-token m/l/a; declared_exp; lane-strided sum/PV with tree256 and FMA; unnormalized per-rank weights; rank0 then rank1 rescale with separate products/adds and final division before value projection";
        identity.prefill_lineage += "; full attention then block-parity compaction; layout 474c4d4b56533033; block 4";
    }
    return identity;
}

inline strix::profile::ExecutionReceipt host_receipt(const Effective& e) {
    strix::profile::ExecutionReceipt receipt;
    receipt.engine = "glm";
    receipt.implementation = "strix-glm5-rank";
    receipt.identity = numerical_identity(e);
    // Inspection has no model-content hashes, canonical state, build provenance,
    // native kernel/device identity, observed topology or completed-workload digests.
    return receipt;
}

}  // namespace strix::glm::rank_defaults
