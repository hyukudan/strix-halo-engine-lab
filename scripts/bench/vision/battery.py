#!/usr/bin/env python3
# Vision battery client: one greedy chat request per image (data URL) to a served engine; writes a JSON list that
# score.py grades against facts computed from the pixels.
# usage: battery.py PORT OUT.json TAG IMG...   env BATTERY_MODEL (model field, default deepseek-v4.1), BATTERY_MAX_TOKENS (80)
import base64, json, os, sys, time, http.client

PROMPT = ("Describe this image in one or two sentences: which colours and shapes or patterns do you see, "
          "and where are they in the image?")
port, out, tag, imgs = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
res = []
for img in imgs:
    name = img.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    raw = open(img, "rb").read()
    mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
    body = {"model": os.environ.get("BATTERY_MODEL", "deepseek-v4.1"), "stream": True,
            "max_tokens": int(os.environ.get("BATTERY_MAX_TOKENS", "80")), "temperature": 0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())}}]}]}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=float(os.environ.get("BATTERY_TIMEOUT_S", "40")))
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    text, ttft, strix = "", None, {}
    if r.status != 200:
        res.append({"name": name, "terms": tag, "status": r.status, "error": r.read()[:300].decode("utf-8", "replace")})
        print(tag, name, "HTTP", r.status, flush=True)
        continue
    for line in r:
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        j = json.loads(line[5:])
        for ch in j.get("choices", []):
            d = ch.get("delta", {}).get("content")
            if d:
                if ttft is None:
                    ttft = time.time() - t0
                text += d
        for k in j:
            if k.startswith("strix"):
                strix[k] = j[k]
    res.append({"name": name, "terms": tag, "ttft_s": ttft, "wall_s": time.time() - t0, "text": text,
                "strix": json.dumps(strix)[:600]})
    print("%s %-15s ttft %.2f %r" % (tag, name, ttft or -1, text[:160]), flush=True)
    json.dump(res, open(out, "w"), indent=1)  # after every image: a stopped run keeps what it has
