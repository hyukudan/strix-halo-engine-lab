The same-pack timing and teacher-forced quality tools are documented in
[`scripts/bench/stratacmp/README.md`](../scripts/bench/stratacmp/README.md).
Timing: `scripts/bench/stratacmp/run-fair.py` runs the ABAB matrix and `summarize.py` reports clocks per token. Size distinct inputs with `prepare-prompts.py` and a local tokenizer; the server's actual prompt usage is the context gate.
