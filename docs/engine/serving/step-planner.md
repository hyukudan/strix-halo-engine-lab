# Step planner

The batched Qwen serving loop decides, once per step, which prompt runs and whether its unit runs before
the decode round. The decision comes from the shared planner in `include/strix/serve/step_planner.hpp`;
`STRIX_SERVE_STEP_PLANNER=0` (or the baseline mode) restores the loop's own inline policy.

## Switch

| Variable | Default | Meaning |
|---|---|---|
| `STRIX_SERVE_STEP_PLANNER` | on (`0` turns it off) | The shared planner picks the prompt unit and whether it runs before the decode round. Off: the inline policy. |
| `STRIX_SERVE_DECODE_GAP_MS` | `0` | Planner on only. The longest a prompt unit may hold up the decoding requests. `0` means no bound. When set, the unit is a tail-size unit (the largest certified tail) if the observed cost of a full block would exceed the bound. |

Both are printed as `serve lever` lines at the first decode round (`step_planner`, `decode_gap_ms`).

## What the planner does

- Prompt choice: prompts with at most `STRIX_BATCH_SHORT_FIRST_ROWS` rows left (default 2048) go first,
  oldest first; a prompt passed over eight times by younger ones goes next regardless of length.
- Time slicing: a prompt unit runs before the decode round only while the decoding requests are not
  owed time. A unit adds `STRIX_BATCH_PROMPT_SHARE` times its own time (less time suspended for other
  service) to the debt; the decode round and constrained serial work pay it back. With no prompt waiting,
  the debt is cleared.
- Decode gap: with `STRIX_SERVE_DECODE_GAP_MS` set, the unit is chosen from the certified sizes (tail
  unit or full block) using the observed per-row prompt cost.

With the default gap (`0`) the sequence of prompt units and decode rounds equals the loop's own policy,
except that the eight-overtake starvation bound is enforced.

## Exactness

The planner changes scheduling only. A prompt's rows and bits do not depend on the size or the order of
its units, and each decoding request's tokens come from its own rows, so every request returns the ids
of a solo run whatever the plan.
