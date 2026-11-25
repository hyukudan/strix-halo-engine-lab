# Image input

Image input lets a chat request carry pictures. DeepSeek V4.1 takes them on the two-node pair server. GLM-5.3 Flash
has an image tower adapter, a command-line path and a server path that are written but not yet run on the device
(see [GLM-5.3 Flash](#glm-53-flash) below).
[One shared vision runtime](shared-cores.md#the-components) serves every family; a model only supplies an adapter
([vision-adapters.md](vision-adapters.md) describes the adapter interface).

| Model | Image input |
|---|---|
| DeepSeek V4.1 | yes, on the pair, behind `--vision` |
| GLM-5.3 Flash | draft: `strix-glm5-rank --embed-rows` and `strix-glm5-serve --vision`, tower on the CPU; the server runs on the pair (two images answered correctly), `--embed-rows` not yet run |
| Qwen, MiniMax, Gemma | not yet |

## Use

Start the server with `--vision <vision GGUF>` on rank 0, `--prefill-path device` and an explicitly sized
`--prefill-rows N`. Vision with `--deploy`, `--deploy-check` or `--prefill-rows auto` is refused until the admission
estimates include retained source bytes, device peaks, cached/frame rows and transient copies. Manual sizing must
leave room for these allocations on both ranks. The flag is off by default; without
it no vision code runs and rank 1 holds nothing of it. Send an OpenAI-style chat completion with an `image_url` part
(`data:` URL):

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

## GLM-5.3 Flash

The image tower is a separate GGUF projector file (`clip.projector_type = glm5v`, FP32 / FP16 / BF16 storage). It
runs on the CPU, on the shared FP32 host reference, ahead of the run: `strix-glm5-vision` writes the projected rows
of an image to an embed rows file, and both ranks read that file, so they splice the same bytes.

```sh
# 1. one image -> its rows (host only, no GPU; prints the number of rows)
./build/strix-glm5-vision encode MMPROJ.gguf IMAGE.png IMAGE.rows
# 2. the tokenized prompt (32-bit ids, as --prompt takes them) with one <|image|> token (154854) where each image
#    goes -> the prompt with every image span in place
./build/strix-glm5-vision expand PROMPT.u32 PROMPT.image.u32 IMAGE.rows
# 3. once per node, R = 0 or 1, the same two files on both
./build/strix-glm5-rank MODEL_DIR --world 2 --rank R --transport usb4-auto \
    --prompt PROMPT.image.u32 --embed-rows IMAGE.rows --decode 64
```

`expand` turns the n-th `<|image|>` into `<|begin_of_image|>`, one `<|image|>` per row of the n-th rows file and
`<|end_of_image|>`; several images take a comma-separated list in prompt order. The run needs a chunked prefill
(the default). With MTP speculation (the default) the prompt takes the chunked MTP bootstrap, which is
`--prefill gemv`; `--mtp 0` runs plain decode with the same tokens.

The server takes image parts of chat requests with `--vision` on rank 0 (rank 1 needs no file):

```sh
./build/strix-glm5-serve MODEL_DIR --world 2 --rank 1 --transport usb4-auto
./build/strix-glm5-serve MODEL_DIR --world 2 --rank 0 --transport usb4-auto --host 127.0.0.1 --port 8080 \
    --model-id glm-5.3-flash --vision MMPROJ.gguf
```

A request is an OpenAI-style chat completion with `image_url` parts (`data:` URLs), as in [Use](#use). Rank 0
encodes each new image on its CPU in the request's own thread (other requests keep decoding), keeps the rows in
the embed cache (`STRIX_VISION_CACHE_MIB`), and sends them to rank 1 once per image in an embed frame after the
prompt tokens of the first prefill call that needs them. Both ranks then splice the same rows. Image prompts use
the same scheduler as text prompts (chunks beside decoding requests, MTP bootstrap, prefill units); they skip the
prefix cache and the RTX prefill route, and refuse `session_id`. `--vision` needs a chunked prefill, and with MTP
`--prefill gemv`; the server refuses to start otherwise.

Status: the adapter, the tool, the server's host half and the pair protocol (on a fake engine) are tested on the
host against synthetic projector files ([gate status](vision-adapters.md#gate-status)). The device half
(`Engine::set_embed_rows` in the plain and the MTP prefill, `--embed-rows`, the server main) compiles for gfx1151 and,
on a pair (2026-11-24, one USB4 cable), the server runs with the real projector file (`mmproj-F16.gguf`): two
synthetic images were answered correctly, greedy and sampled (red | blue halves; a filled circle beside an outlined
square; 5-8 s per request). Not run yet: `strix-glm5-rank --embed-rows`, images that span several prefill chunks or
units, and text-prompt identity against the previous build on the device.

Cost of the CPU tower: the host reference accumulates in FP64 and is slow by design. On a synthetic tower of width
1024 with 4 blocks (MLP 4096) and a projector of width 4096, on a 4-thread x86 host: 20.9 s for 1,024 patches (a
448 x 448 image, 256 rows) and 116.7 s for 4,096 patches (896 x 896, 1,024 rows), host reference FP32, one run
each. Time grows linearly with the block count and, in attention, with the square of the patch count.

## Data path

1. **Media front door** (`ApiMediaBridge`, `include/strix/vision/api_media.hpp`). Decodes PNG and JPEG under
   `DecodeLimits` (`image.hpp`). WebP and GIF are refused with a 400 ("send PNG or JPEG").
2. **Preprocessing** (`preprocess.hpp`, `ResizeRule`). The upstream resize plan: the aspect ratio is kept on a grey
   canvas, sides are multiples of 14, pixels are BF16, patches are merged 3x3 with zero-padded partial cells. The
   span (`embed_span.hpp`, `SpanLayout`) is at most 1024 rows, counting the row ends, the start and the end markers.
3. **Shared encoder runtime** (`Encoder`, `encoder.hpp`; `src/vision/encoder.hip`). One data-driven ViT tower for every
   family. The HIP arithmetic is `vision-f16-v1`: FP16 panels with a power-of-two scale and FP32 accumulation.
   DeepSeek uses split hi+lo activations.
4. **Projector** (the aligner), in the adapter (`VisionAdapter`, `adapter.hpp`; DeepSeek: `deepseek41/vision_adapter.hpp`).
5. **Splice**. The embedding rows enter the token stream as virtual ids. Image rows carry the router's image flag. For
   Engram, image positions take no part in n-grams and the Engram gate is 0 there, as in the upstream model.
6. **Pair frame** (`include/strix/pair/embed_frame.hpp`). Rank 0 encodes; for each new image it sends rank 1 one frame:
   manifest, BF16 rows and a CRC32C.
7. **Embed cache** (`EmbedCache`, `embed_cache.hpp`). Key = adapter identity + arithmetic + image digest. LRU, default
   256 MiB (`STRIX_VISION_CACHE_MIB`; `0` disables storing). It lives for the process only and is never written to
   disk. Image prompts never enter the prefix cache or sessions.

## Measured

On the two-node pair, DeepSeek V4.1 Q2, one 37x23 PNG fixture upscaled to 1550 patches (200 image rows + 21 text
tokens):

| Case | Time to first token | Detail |
|---|---|---|
| First request (image not cached) | 1.46-1.87 s | image encode 0.93-1.07 s; frame to rank 1 2.0 MB in 4-5 ms; LM prefill about 0.46 s |
| Same image again (cache hit) | 0.53-0.61 s | encode skipped; 80-byte frame |

Decode: 39-41 tok/s.

## Switches

| Switch | Default | Meaning |
|---|---|---|
| `--vision FILE` | off | Server flag, rank 0 only. Needs `--prefill-path device`. |
| `STRIX_VISION_RESIDENT` | 1 | `1`: the tower is uploaded at start; `0`: on the first image. |
| `STRIX_VISION_CACHE_MIB` | 256 | Embed cache size; `0` disables storing. |

See [toggles](../toggles.md).

## Limits

- PNG and JPEG only.
- Images over the row budget are refused.
- Image requests with `session_id` are refused.
- Video is refused.
- GLM-5.3 Flash is a draft: the tower runs on the CPU (minutes per image on a full-size tower, so a first request
  with a new image waits that long), image prompts have no prefix cache entry, and without `--vision`
  `strix-glm5-serve` renders the model's reminder for an image part, as before.
- Needs the device prefill path.
- Encoding is serialized with media preprocessing; peer/split scheduling and overlap with text-prefix prefill remain future work.
- Image positions have zero Engram gates. Synthetic gate tests do not qualify complete served-image fidelity;
  qualification evidence must name the tested revision and cover that arithmetic in the complete image workload.
