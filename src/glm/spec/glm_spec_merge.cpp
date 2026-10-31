#include "glm_spec_merge.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strix::glm::spec {

void merge_rank_order(std::span<MergeRow> mine, MergeCtl& ctl, std::span<const MergeRow> peer,
                      const MergeCtl& pc, std::uint32_t rank) {
    if (rank > 1 || mine.size() != peer.size() || mine.size() > 8)
        throw std::invalid_argument("merge: rank / row count");
    if (pc.cycle != ctl.cycle || pc.step != ctl.step || pc.kind != ctl.kind)
        throw std::runtime_error("spec control word mismatch between ranks (cycle / step / kind)");
    const MergeCtl& c0 = rank == 0 ? ctl : pc;
    const MergeCtl& c1 = rank == 0 ? pc : ctl;
    MergeCtl out = c0;  // rank 0's cost table
    out.replay_flag = c0.replay_flag | c1.replay_flag;
    out.mtp_max = std::max(c0.mtp_max, c1.mtp_max);
    out.mtp_sum = c0.mtp_sum * std::exp(c0.mtp_max - out.mtp_max) + c1.mtp_sum * std::exp(c1.mtp_max - out.mtp_max);
    ctl = out;
    for (std::size_t r = 0; r < mine.size(); ++r) {
        const MergeRow& a = rank == 0 ? mine[r] : peer[r];
        const MergeRow& b = rank == 0 ? peer[r] : mine[r];
        MergeRow m = (b.value > a.value || (b.value == a.value && b.id < a.id)) ? b : a;
        m.flag = a.flag | b.flag;
        mine[r] = m;
    }
}

}  // namespace strix::glm::spec
