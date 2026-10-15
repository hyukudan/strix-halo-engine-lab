# GDN recurrent LDS layout diagnostic

This diagnostic isolates physical LDS indexing in the 128-thread, 32-column
recurrent state update. The original kernel and its decoder call site remain
unchanged. Three additional template instances use stride 32, padded stride 33,
and transposed LDS indexing. All eight reduction accesses use the selected mapping;
the logical columns, reduction pairs, arithmetic and separate delta storage are
preserved. Persistent state layout remains unchanged.

The explicit-stream wrapper accepts bounded FP32 input, state and core views,
plus optional sticky graph status. It validates shapes, exact extents, alignment,
address ranges and writable-view aliases before launching. It allocates nothing,
clears no status and synchronizes nothing. The caller retains stream and views
until quiescent completion, including after an enqueue exception or graph replay.

The wrapper and additional kernels exist only with the private testing macro.
There is no new decoder option, catalog selection or default. The source fixture
compares the incumbent and all three instances through eager state evolution and
captured reset/resume, including nonzero distinct state, all four tiles, nonfinite
propagation, signed zeros, status rejection, poisoned outputs and canaries.

Status: implemented source, not compiled or executed. Tests and measurements are
deferred to the validation stage. Padding is a bank-conflict hypothesis, not a
measured gain. ISA, resources, native output/state equality and complete-operator
timing are required before selecting a layout. Persistent state transposition,
recurrence over several retained input rows and rollback are separate remaining increments.
