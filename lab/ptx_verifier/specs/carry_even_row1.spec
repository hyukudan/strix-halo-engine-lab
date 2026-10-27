// carry_restore_kernel: p=0, row=1, kHeadDim=512, one CTA.
// The source row offset is part of the contract.
dim N;
dim ONE;
dim STASH_ROWS;
array stash[STASH_ROWS, N];
array state[N];
array valid[ONE];

state[i] = stash[1, i];
valid[i] = 1;
