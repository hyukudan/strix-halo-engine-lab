# Depth-aware MTP policy (draft)

`STRIX_MTP_CONTROLLER=depth` enables an experimental greedy host policy. The
default controller is unchanged. Target verification continues to own tokens,
prefix acceptance, rollback and materialization; proposal confidence never
authorizes a target token.

The policy records draft wall cost separately for each actual MTP depth and
non-draft cycle cost for each physical owner and filled proposal count. External
rows share the existing seven-row window and retain their source acceptance
estimates. They neither consume MTP draft steps nor train MTP acceptance bins.
Robust cost estimates use bounded re-probes so a host stall cannot permanently
exclude an owner. Unmeasured cells require local exploration rather than an
estimate copied from another owner.

Acceptance for an observed proposal is calibrated by depth and confidence bin,
using only verified prefix outcomes. Future lookahead uses an explicit fixed
population prior, never an extrapolation of outcomes selected by an R8 policy.
`STRIX_MTP_DOMAIN=general|code|prose|copy` selects explicit caller metadata.
`STRIX_MTP_MINP_<DOMAIN>` overrides only an implicit minimum probability; an
explicit `--mtp-spec-min-p` wins. `STRIX_MTP_DEPTH_PRIOR_<DOMAIN>` accepts seven
comma-separated conditional probabilities measured on complete positions. The
fallback prior is 0.5 at every depth. No domain is inferred from prompt text.

`STRIX_MTP_DRAFT_CACHE=mixed|fast|teacher` selects an independent generation-only
proposal-cache profile. `mixed` preserves existing behavior. `fast` reconstructs
generated K/V rows with the same fast producer as the proposal path; `teacher`
reconstructs every generated row with the ordinary Q8 producer. Prompt priming
is unchanged. Uniform profiles currently require greedy drafts. The receipt
identifies the actual profile separately from native full-forward qualification.
`STRIX_MTP_CACHE_AUDIT=1` performs bounded readback and hashes occupied proposer
cache rows for diagnostic schedule comparisons; it must be off during timing.

These controls remain opt-in proposals. CPU policy tests establish host decision
contracts only. Qualification additionally requires native target tokens, full
logits, canonical and physical state, proposer fidelity across cycle boundaries,
and complete-workload A/B evidence. No draft tree is implemented: each future
branch would require its own recurrent and latent state.
