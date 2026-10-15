# Experimental SSH-provisioned copied channel contract

`STRIX_ENABLE_SSH_BOUND_BOOTSTRAP=ON` builds an optional OpenSSL3 channel contract.
The default is OFF. Missing build support, algorithms, randomness or verification
fails closed; it never substitutes raw TCP or noncryptographic primitives.
The implementation uses OpenSSL [EVP_MAC HMAC](https://docs.openssl.org/3.0/man7/EVP_MAC-HMAC/),
[EVP_KDF HKDF](https://docs.openssl.org/3.0/man7/EVP_KDF-HKDF/), CSPRNG and
[constant-time tag comparison](https://docs.openssl.org/3.0/man3/CRYPTO_memcmp/).

The trusted pinned-SSH bootstrap owner must independently reconcile and provide
the ephemeral 32-byte secret and expected session facts in RAM. These include
session/node instances, both SSH host-key pin hashes, control/payload nonces,
model content and execution/placement identities, physical route/epoch, logical
epoch/request/generation, codec/dtype and frame/payload budgets. Supplying these
bytes is not proof of an SSH process or a loaded model. This module implements
secret possession and channel binding; actual pinned-SSH provisioning remains
an integration gate. Loopback is not authentication, and this is not mTLS.
Exactly two trusted processes may possess the fresh session secret; a holder can
impersonate either role. The caller must wipe its own secret copy after bootstrap.
This layer provides no forward secrecy. Both channel nonces bind the common
descriptor; the channel enum in the header and HKDF context separates channels.

`SshBoundCopiedByteStream::bootstrap` consumes a healthy bounded loopback TCP
stream after validating the expected contract. Node0 initiates, node1 responds.
The node-instance arrays use that fixed two-node role mapping.
Four HMAC-authenticated role-specific messages exchange fresh initiator/responder
nonces and confirm their common descriptor. The control/payload channel kind,
sender/receiver roles and message phase prevent reflection or channel substitution.
HKDF-SHA256 derives independent direction/channel application keys bound to the
descriptor and fresh challenge transcript. The bootstrap secret is wiped after
derivation; keys are wiped on failure, cancellation and destruction.

Every application frame includes the fixed contract version, phase, channel,
direction, strict sequence, exact payload length, descriptor digest and fresh
transcript nonces, followed by the payload and a full HMAC-SHA256 tag. Data is
released only after verification. Bootstrap and application frames have distinct
domains/keys. Sequence counters cannot wrap, restart or accept replay/reordering.
There is no encryption in this layer; confidentiality comes from the separately
owned authenticated SSH tunnel.
The owner must also guarantee exclusive ownership of forwarded loopback ports:
a local port squatter or relay can read or discard this layer's plaintext without
being able to forge its MAC. Loopback placement alone does not ensure privacy.

The authenticated envelope adds 172 bytes to each underlying TCP frame. Its
configured TCP ceiling must equal the application-frame ceiling plus that
overhead. Existing absolute deadlines/call limits include peer idle and consumer
ACK waits. This module does not create a listener, open a connection or expose
the bare owned socket. Its capability is move-only and a failed/canceled channel
cannot resume. A session owner must cancel every associated channel on epoch
failure and retain its provisioning/tunnel lifetime.
All stream operations, including cancellation, require one serialized progress
owner. Cross-thread cancellation is unsupported; blocked calls end at their
deadline. One connection per secret/descriptor/channel is allowed, and reconnect
must advance the logical epoch. EOF does not establish semantic completion.

The copied payload backend can own a verified payload capability, with exact
frame limits and matching physical route, epoch, owner and link index. It also
rejects transfers whose placement plan/request/generation/epoch/dtype or byte bounds
do not match the admitted descriptor. A control-channel capability cannot enter
that overload. Existing host buffers and producer/consumer proofs remain required.

Host tests cover bidirectional frames, move/cancel, wrong secret, foreign session
facts, direction/channel reflection, malicious relay tampering, replay/reordering,
old-transcript replay after a fresh handshake, raw TCP empty/oversized framing, crypto-OFF
failure and a real loopback authenticated copied-envelope round trip. These are
host contract fixtures, not model execution, GPU visibility or performance evidence.

Remaining work includes the trusted SSH session/provisioning owner, coordinated
control/payload admission and deadlines, actual model/resource observation, HIP
staging/completion proofs, peer worker/session state, position/PLE semantics and
joint token commit. The older raw native TCP control API retains its original
deadline/authentication shortcomings; callers must not label it authenticated
because this optional capability exists. No native two-host inference is qualified.
