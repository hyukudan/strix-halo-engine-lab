# Requested generation output

Repeated `--logprob-token ID` with serial full-output `--generate` returns requested
logprobs using one target call per consumed input and the full-vocabulary FP64
log-sum-exp denominator. IDs are validated before GPU allocation; duplicates and
order are preserved. This is presentation, not a restricted sampling support.
The bounded ledger admits at most64 IDs,262144 context rows and1048576 values.
It reserves rows before inference and each presentation vector before its call.
Receipts identify consumed input positions; the final unconsumed emitted token
never implies another model row. Incompatible scalar/window/lookup/serving/cache/RAW/
reference routes reject explicitly. Defaults allocate no ledger. Source-only: no
new qualification or performance measurement is claimed.
