# Matrix instruction fixture

`instructions.bin` contains 6,894 project-generated, device-recorded cases of
gfx1151 wave32 `v_wmma_f32_16x16x16_f16`. The file is little endian, with no
header. Each 72-byte record is 16 FP16 A words, 16 FP16 B words, FP32 C bits,
and FP32 result bits. Operands are replicated across matrix rows/columns so one
output element characterizes the scalar accumulator recipe.

The source generator is `../characterize.hip`, seed `0x472abcd1`. The fixture
selects indices 0..511 and every 127th index of each of its nine
32,768-case modes, with duplicates removed. These cover isolated products,
two-product steps, mixed and uniform exponents, signs, half subnormals and FP32
accumulator edge values. Full traces and compilation/timing receipts stay
private. The host fixture test requires no Python, checkpoint or device.

Native qualification must replay the entire characterization trace, including
the exact-cancellation mode, and compare actual GEMMs
against both the integer emulator and the untiled instruction implementation.
An instruction fixture pass is not a model-fidelity or production-admission pass.
