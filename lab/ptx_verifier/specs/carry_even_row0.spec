// carry_restore_kernel: p=0, row=0, kHeadDim=512, one CTA.
// All state elements and the validity flag must be written.
dim N;
dim ONE;
array stash[N];
array state[N];
array valid[ONE];

state[i] = stash[i];
valid[i] = 1;
