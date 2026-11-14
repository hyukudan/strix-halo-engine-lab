"""Asset-free image regime inspection and invalid-policy gates."""
import argparse
import json
import subprocess


def run(binary):
    def invoke(*args, valid=True):
        p = subprocess.run([binary, "--model", "missing-test-assets", "--print-config", *args],
                           capture_output=True, text=True)
        assert (p.returncode == 0) == valid, (args, p.stdout, p.stderr)
        return json.loads(p.stdout) if valid else None

    assert invoke()["parallel"] == "request"
    pair = invoke("--world", "2", "--rank", "1", "--attn-score-2p15")
    assert pair["parallel"] == "sequence" and pair["attn_score_limit"] == "2^15-extended"
    assert pair["transport"] == "usb4gpu" and pair["transport_effective"] == "pending-pairing"
    assert pair["vae_parallel"]
    assert invoke("--world", "2", "--transport", "usb4auto")["transport"] == "usb4auto"
    assert invoke("--world", "2", "--transport", "usb4-host")["transport"] == "usb4-host"
    batch = invoke("--world", "2", "--rank", "1", "--requests", "5")
    assert batch["parallel"] == "request" and batch["local_requests"] == 2 and batch["transport"] == "none"
    assert invoke("--world", "2", "--rank", "1", "--parallel", "request")["local_requests"] == 0
    assert not invoke("--lazy-prepare")["eager_prepare"]
    assert not invoke("--world", "2", "--vae-parallel", "off")["vae_parallel"]
    assert not invoke("--world", "2", "--latent-only")["vae_parallel"]
    for args in [("--parallel", "cfg"), ("--parallel", "sequence"), ("--requests", "0"),
                 ("--requests", "4097"), ("--seed", "4294967295", "--requests", "2"),
                 ("--world", "3"), ("--rank", "1"), ("--world", "2", "--fast-sparse"),
                 ("--world", "2", "--requests", "2", "--dist"), ("--transport", "private"),
                 ("--vae-parallel", "on"), ("--requests", "2", "--decode-latent", "x")]:
        invoke(*args, valid=False)
    print("pair CLI policy gates passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    run(parser.parse_args().binary)
