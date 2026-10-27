// hc_expand_kernel: rows=1, hidden=sub_stride=H, pre_carry=null.
// This is a real-arithmetic formula, not an IEEE rounding-order contract.
// The manifest restricts H to 32 so all output indices belong to CTA zero.
dim H;
dim STREAMS;
dim MIX_COUNT;
array sub[H];
array residual[STREAMS, H];
array mixes[MIX_COUNT];
array out[STREAMS, H];

out[k, i] = mixes[4 + k] * sub[i]
          + sum(j in 0..STREAMS, mixes[8 + 4 * j + k] * residual[j, i]);
