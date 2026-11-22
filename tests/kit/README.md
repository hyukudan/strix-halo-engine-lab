# Host conformance kit

`strix-conformance-capture/v1` fixtures are UTF-8 JSON with an `identity` and two maps of named byte buffers, `inputs` and `expected`. The values use lowercase hex with two digits per byte. Each test case defines the scalar encoding. The seeded cases use little-endian IEEE binary32 (`f32le`) or unsigned 32-bit integers (`u32le`) and one-byte indices for schedules.

`compare_exact` checks every byte, including the bits of tokens, logits and state digests supplied by the caller. `compare_bounded` requires the recorded numerical identity and checks an absolute plus relative per-element bound. Both return an empty string on success and the first difference on failure.

`ctest -L 'R1|R2'` runs the host package tests from a configured and built tree. The three seeded mutations run a passing control before testing that the mutation is rejected.
