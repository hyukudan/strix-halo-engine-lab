# MTP draft vocabularies

Token-id lists (little-endian `uint32`) that restrict the MTP draft head's output rows. The draft head only proposes
tokens; the target verifies every one, so these lists change acceptance and speed, never the generated text.

| File | Rows | Use |
|---|---|---|
| `core32k.bin` | 32,768 | Language-independent core: control tokens plus the most frequent tokens of model-style answers (code, English, mixed). |
| `pack-<lang>16k.bin` | 16,384 | Extra tokens for one language (`de`, `es`, `fr`, `it`, `pt`, `ru`, `ja`, `zh`), meant to be used together with the core. |

Selection: `--mtp-draft-vocab auto:data/draft-vocab` loads the core plus the language pack that holds the most
prompt tokens, when that is at least 2 tokens and 5 % of the prompt, otherwise the core alone; the choice is printed
on stderr. Explicit lists still work: `--mtp-draft-vocab data/draft-vocab/core32k.bin,data/draft-vocab/pack-es16k.bin`.

Construction: frequencies of the tokens the tokenizer emits in answer corpora formatted with the model's chat template
(train/test halves), ranked per language against the core; see `tools/build_draft_vocab.py` for the file format.
Measured on the target model: the core alone needs 86 verify cycles where a general 106k-row list needs 89 on a code
prompt (same speed within noise); on a Spanish prompt core + `pack-es16k` matches the full head's acceptance
(154 -> 132 cycles against a general list).
