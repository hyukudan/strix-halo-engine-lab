"""Generate synthetic format fixtures and independent MPFR scalar/rotary oracles.

Only this maintenance script needs Python, numpy and gmpy2. Tests consume committed bytes.
No checkpoint slices or external inspection records are included.
"""
import importlib.util
import pathlib
import random
import struct
import json
import gmpy2 as mp
import numpy as np

root = pathlib.Path(__file__).parent
out = root/'fixtures'
out.mkdir(exist_ok=True)
mp.get_context().precision = 128
rng = random.Random(471)
with open(out/'rsqrt.bin', 'wb') as f:
    values = [1, 2, 0x7fffff, 0x800000, 0x7f7fffff, 0x3f800000, 0x40800000]
    values += [rng.randrange(1, 0x7f800000) for _ in range(10000)]
    for bits in values:
        x = struct.unpack('<f',struct.pack('<I',bits))[0]
        y = 1/mp.sqrt(mp.mpfr(x))
        assert mp.can_round(y,126,mp.RoundToNearest,mp.RoundToNearest,24)
        with mp.context(mp.ieee(32)): rounded = float(+y)
        f.write(struct.pack('<If',bits,rounded))

with open(out/'quant.bin','wb') as f:
    for kind in ('q8','q4','q6'):
        for block in range(2):
            if kind == 'q8':
                q = [rng.randrange(-128,128) for _ in range(32)]
                d = 0.125 if block else -0.25
                packed = struct.pack('<e',d) + struct.pack('<32b',*q)
                values = np.array(q,dtype=np.float32)*np.float32(d)
            elif kind == 'q4':
                scale = [rng.randrange(64) for _ in range(8)]
                minimum = [rng.randrange(64) for _ in range(8)]
                q = [rng.randrange(16) for _ in range(256)]
                sub = [0]*12
                for j in range(4):
                    sub[j] = scale[j] | ((scale[j+4]>>4)<<6)
                    sub[j+4] = minimum[j] | ((minimum[j+4]>>4)<<6)
                    sub[j+8] = (scale[j+4]&15) | ((minimum[j+4]&15)<<4)
                codes = []
                for b in range(4):
                    codes += [q[b*64+i] | (q[b*64+i+32]<<4) for i in range(32)]
                d, dm = 0.03125, 0.0625
                packed = struct.pack('<ee',d,dm)+bytes(sub)+bytes(codes)
                values = np.array([np.float32(d)*np.float32(scale[i//32]*q[i])-np.float32(dm)*np.float32(minimum[i//32]) for i in range(256)],dtype='<f4')
            else:
                q = [rng.randrange(64) for _ in range(256)]
                scales = [-128,-127,-65,-1,0,1,64,127]*2
                low, high = [0]*128, [0]*64
                for h in range(2):
                    for j in range(4):
                        for i in range(32):
                            value = q[h*128+j*32+i]
                            low[h*64+(j%2)*32+i] |= (value&15) << (4*(j//2))
                            high[h*32+i] |= (value>>4) << (j*2)
                d = 0.015625 if block else -0.03125
                packed = bytes(low)+bytes(high)+struct.pack('<16b',*scales)+struct.pack('<e',d)
                values = np.array([np.float32(d)*np.float32(scales[i//16]*(q[i]-32)) for i in range(256)],dtype='<f4')
            f.write(packed); f.write(values.astype('<f4').tobytes())

spec = importlib.util.spec_from_file_location('g4_rope', root.parent.parent/'src/gemma4/ref/make_rope_tables.py')
rope = importlib.util.module_from_spec(spec); spec.loader.exec_module(rope)
with open(out/'rope.bin','wb') as f:
    for p in (0,511,512,65535):
        a,b = rope.shard((p,p+1)); f.write(a); f.write(b)
np.savez_compressed(out/'arrays.npz', ids=np.array([2,17,4,-1],'<i4'), values=np.arange(21,dtype='<f4').reshape(3,7)*0.25)
# Independent scalar oracle: each basic operation is explicitly rounded to binary32.
f=np.float32
def exp(x):
    x = f(x)
    if x > 88:
        return f(np.inf)
    if x < -103:
        return f(0)
    n = f(np.rint(f(x * f(1.44269504))))
    r = f(f(x - f(n * f(0.693359375))) - f(n * f(-2.12194440e-4)))
    p = f(1.98412698e-4)
    for coefficient in [1.38888889e-3, 8.33333333e-3, 4.16666667e-2, 1.66666667e-1, 0.5, 1, 1]:
        p = f(f(p * r) + f(coefficient))
    k = int(n)
    k1 = int(k / 2)
    k2 = k - k1
    return f(f(p * f(2.0**k1)) * f(2.0**k2))


def tanh(x):
    x = f(x)
    e = exp(f(f(-2) * abs(x)))
    value = f(f(f(1) - e) / f(f(1) + e))
    return -value if np.signbit(x) else value


def gelu(x):
    x = f(x)
    cube = f(f(x * x) * x)
    value = tanh(f(f(0.7978845608028654) * f(x + f(f(0.044715) * cube))))
    return f(f(f(f(0.5) * x) * f(f(1) + value)))


with open(out/'nonlinear.bin', 'wb') as fixture:
    for x in [-100, -30, -10, -3, -1, -0.01, -0.0, 0, 0.01, 1, 3, 10, 30, 88]:
        fixture.write(struct.pack('<ffff', x, exp(x), tanh(x), gelu(x)))

# Synthetic complete configuration, expressed from the supported public contract.
config = dict(model_type='gemma4_text', hidden_size=2560, num_hidden_layers=42, intermediate_size=10240,
    vocab_size=262144, num_attention_heads=8, num_key_value_heads=2, head_dim=256, global_head_dim=512,
    hidden_size_per_layer_input=256, vocab_size_per_layer_input=262144, max_position_embeddings=65536,
    sliding_window=512, num_kv_shared_layers=18, rms_norm_eps=1e-6, final_logit_softcapping=30.0,
    hidden_activation='gelu_pytorch_tanh', tie_word_embeddings=True, attention_bias=False,
    attention_k_eq_v=False, enable_moe_block=False, use_double_wide_mlp=False, attention_dropout=0.0,
    use_bidirectional_attention=None, layer_types=['full_attention' if l%6==5 else 'sliding_attention' for l in range(42)],
    rope_parameters=dict(full_attention=dict(rope_type='proportional',rope_theta=1000000.0,partial_rotary_factor=0.25),
    sliding_attention=dict(rope_type='default',rope_theta=10000.0)))
(out/'config.json').write_text(json.dumps(dict(text_config=config),indent=2)+'\n')

print("generated", sum(p.stat().st_size for p in out.iterdir()), "bytes")
