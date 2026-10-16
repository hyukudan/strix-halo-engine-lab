# Independent MTP forward golden file

`read_mtp_forward_golden_file(path, expected_full_file_sha256, maximum_file_bytes)`
reads an explicitly pinned local reference. The complete bounded file is read
and SHA256 checked before parsing. It does not execute a model or establish that
the supplied producer is independent. `encode_mtp_forward_golden_file` is a
producer-side byte encoder; it never generates reference values.

Version 1 uses little-endian unsigned integers and IEEE binary32 float bits.
There is no native struct serialization. Header fields in order:

| Offset | Field |
| --- | --- |
| 0 | Eight ASCII bytes `STRMTPG1` |
| 8 | uint32 version = 1 |
| 12 | uint32 header bytes = 168 |
| 16 | uint64 complete file bytes |
| 24 | Three 32-byte digests: independent source, target content, sidecar content |
| 120 | Four uint64 IDs: independent reference, structural plan, semantic contract, native numerical profile |
| 152 | Three uint32 dimensions: rows, wide elements per row, vocabulary elements per row |
| 164 | uint32 reserved = 0 |

Every row contains uint64 source position; uint32 source input token, next input
token, selected token, and reserved zero; then the complete source-wide BF16
words, complete next-wide BF16 words, and complete vocabulary FP32 logits.
Rows are ordered fresh positions 0 through rows-1. Both wide arrays have the
header width. All token IDs are within the vocabulary, every value is finite,
and selection is the first maximum of the full FP32 row. Signed zero and finite
subnormal bit patterns are retained. No cross-row equality between the explicit
next input and the following source input is inferred.

Limits are 1..4096 rows, 1..1048576 wide elements, and 1..16777216 vocabulary
elements, additionally restricted by the required file byte budget. All three
digests and all four IDs must be nonzero. Dimensions, exact total bytes, reserved
fields, and every row are validated before allocating decoded row payloads.
Truncation and trailing bytes are rejected. During decode the input file buffer
and decoded vectors coexist; the file budget is not a claim about process RSS.

The file SHA binds these container bytes. It is distinct from the runtime replay
payload identity, which the native consumer computes separately after loading.
No qualification or activation is implied by successful decoding.
