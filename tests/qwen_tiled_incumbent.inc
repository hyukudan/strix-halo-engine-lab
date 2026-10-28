// Frozen incumbent no-store block algorithm; environment reads supplied by the test.
        while(phase_==Phase::blocks || phase_==Phase::tails) {
            const bool blocks=phase_==Phase::blocks;
            if(!blocks && !tiled_.ladder_tail_rows.empty()) {
                // STRIX_QPF2K_SCHED bit 8 fail-closed: rows the ladder runs in one unit must not fall to a smaller tail
                // owner or to the serial suffix (row classes sit on a 128-row grid per block; a pruned owner would change
                // their arithmetic silently). Owners larger than the unit are skipped below and re-checked next pass.
                const bool start=config.tiled_start;
                // rows the current owner may take (the step below: pos + rows <= n - 1, or == n with heads)
                const bool theads=start && (tail_>=tiled_.tails.size() ||
                    tiled_.tails[tail_].backend->output_policy()!=KnownWindowOutputPolicy::state_only);
                const std::uint64_t m=n_-pos_-(theads?0U:1U);
                std::uint32_t want=0;
                for(const auto r:tiled_.ladder_tail_rows) if(r && m>=r) {want=r;break;}
                if(want && (tail_>=tiled_.tails.size() || tiled_.tails[tail_].rows<want))
                    throw std::logic_error("qpf2k sched: the ladder's "+std::to_string(want)+"-row unit has no tail owner (pruned)");
            }
            if(!blocks && tail_>=tiled_.tails.size()){phase_=Phase::serial;break;}
            auto& backend=blocks?*tiled_.blocks:*tiled_.tails[tail_].backend;
            auto& owner=blocks?*tiled_.block_owner:*tiled_.tails[tail_].owner;
            const auto rows=blocks?tiled_.rows:tiled_.tails[tail_].rows;
            // STRIX_MTP_TILED_START=0: the last prompt row always runs serially (the previous handoff).
            // Otherwise a block that ends exactly at the last prompt row also computes that row's head, and
            // the first draft cycle starts from the block's last row (its tap and FP32 logits).
            const bool tiled_start=config.tiled_start;
            const bool heads=tiled_start && backend.output_policy()!=KnownWindowOutputPolicy::state_only;
            // STRIX_QPF2K_SCHED bit 16: balanced blocks on the T owner (the prompt's rows down to a multiple of 128 as
            // ceil(rest / T) blocks of a multiple of 128 rows <= T, each the first rows of the T arena; the last rest mod 128
            // rows then run as the ladder runs them). Bit 4: the ladder's leading multiple-of-128 tail units merged into one
            // partial block of the T owner. Not with a prefix store (its captures are at T boundaries) nor small units.
            const bool balanced_raw=(config.schedule&16U)!=0;
            const bool merged_raw=(config.schedule&4U)!=0;
            const bool defaults_off=config.suppressed;
            const bool balanced_env=balanced_raw&&!defaults_off,merged_env=merged_raw&&!defaults_off;
            // Per prompt, decided at its first unit: only where the balanced plan is bitwise equal to the ladder.
            if(blocks && (balanced_env || merged_env) && balanced_<0) {
                std::vector<std::uint32_t> ladder=tiled_.ladder_tail_rows;
                if(ladder.empty()) for(const auto& t:tiled_.tails) ladder.push_back(t.rows);
                // the tiled start: the ladder may end exactly at n (heads); without it the last row is a suffix row
                const std::uint64_t m=heads?n_-pos_:n_-pos_-1U;
                // the balanced plan needs the tiled start (the ladder then covers exactly n - pos rows)
                balanced_=balanced_env && heads && known_prefill_balanced_eligible(m,rows,ladder)?1:0;

            }
            if(blocks && balanced_env && balanced_==1 && !adopted_ && !small_ && true &&
               backend.output_policy()!=KnownWindowOutputPolicy::all_rows && heads) {
                const auto size=known_prefill_balanced_rows(n_-pos_,rows);
                if(size) {
                    const auto r=static_cast<std::uint32_t>(size);
                    return Block{&backend,&owner,r,r==rows,pos_+r==n_};
                }
            }
            // Bit 4 (not balanced): the ladder's leading multiple-of-128 tail units merged into one partial block of the T owner.
            if(blocks && merged_env && balanced_==0 && !adopted_ && !small_ && true &&
               backend.output_policy()!=KnownWindowOutputPolicy::all_rows && heads) {
                std::vector<std::uint32_t> ladder=tiled_.ladder_tail_rows;
                if(ladder.empty()) for(const auto& t:tiled_.tails) ladder.push_back(t.rows);
                if(const auto merged=known_prefill_merged_tail_rows(n_-pos_,rows,ladder)) {
                    const auto r=static_cast<std::uint32_t>(merged);
                    return Block{&backend,&owner,r,false,pos_+r==n_};
                }
            }
            if(!adopted_ && (pos_+rows<=n_-1U || (heads && pos_+rows==n_))) {
                // small_blocks: the T block as units of the largest tail size dividing T (the same rows and bits: a tiled row's
                // arithmetic does not depend on its block's size), e.g. while other requests decode
                if(blocks && small_ && !tiled_.tails.empty() && tiled_.tails[0].rows<rows && rows%tiled_.tails[0].rows==0) {
                    auto& t=tiled_.tails[0];
                    const bool theads=tiled_start && t.backend->output_policy()!=KnownWindowOutputPolicy::state_only;
                    if(pos_+t.rows<=n_-1U || (theads && pos_+t.rows==n_)) return Block{&*t.backend,&*t.owner,t.rows,true,pos_+t.rows==n_};
                }
                return Block{&backend,&owner,rows,blocks,pos_+rows==n_};
            }
            if(blocks)phase_=Phase::tails;else ++tail_;
        }
        return std::nullopt;
