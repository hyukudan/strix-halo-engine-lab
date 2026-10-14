# Bounded loopback TCP copied payload

`TcpCopiedPayloadByteStream` provides CPU framing on an exclusively owned,
already-connected TCP socket. Both observed endpoints must be loopback. It
offers no listener or connection service API, and does not authenticate a
remote peer. An encrypted authenticated tunnel is a separate session resource:
adopting its loopback socket is not proof of that session's identity.

The copied transport backend takes exclusive ownership of this stream through a
concrete rvalue overload with explicit matching frame ceilings,
alongside its existing character-device stream. Arbitrary unbounded byte-stream
implementations cannot enter this new overload. The copied payload envelope,
physical link/epoch, logical plan/request/generation, sequence, dtype, exact
payload byte bounds, and separate receive/consumer acknowledgements remain
unchanged. CPU source ownership and explicit producer/consumer completion
proofs remain mandatory. No GPU pointer, DMA, striping or numeric conversion
is introduced.
Failure of `CopiedStreamTransportBackend` poisons and closes its owned TCP stream, preventing
reattachment of another state to unread bytes on that connection.
An owning session must also tear down the epoch after outer distributed-executor
failures; it must not attach a fresh executor to an old transport.

Framing uses the existing four-byte little-endian length prefix. Send gathers
prefix and body into `sendmsg`, accounting partial writes without retaining
the caller's buffer after return. Receive checks the announced nonzero length
against its configured ceiling before allocating a bounded frame vector.
Payload/frame limits can admit two-MiB payloads when their envelope overhead
is budgeted separately. This implementation still allocates/copies framed and
host staging buffers; it is not a pinned zero-allocation receive ring.

The adopted descriptor becomes nonblocking and close-on-exec, with NODELAY
and SIGPIPE suppression. Each frame has one absolute monotonic deadline and
a maximum IO-call count, covering prefix and body together. Partial/error/EOF,
expired or malformed framing permanently poisons and closes the stream.
Socket identity is checked during IO. The descriptor is not exposed for
external close/reuse; the adopter consumes it on validation failure as well.
Timeout is bounded at one minute and frame capacity at 64 MiB.
The receive deadline starts before the first byte and therefore includes peer
idle time and any wait for a consumer acknowledgement. Its configured timeout
must cover that admitted operation's peer staging/consumer work; it is not an
ACK-independent network-progress timer. Operation-level deadline orchestration
for a real model peer remains separate integration work.

Host tests use actual connected loopback TCP sockets. They check partial
framing, empty/oversized/truncated lengths, deadlines, call budgets, moved
ownership and non-TCP rejection. End-to-end copied-envelope tests transfer
20,480-byte, 993,280-byte and two-MiB payloads exactly, retain destination poison
on mismatched dtype/request/generation/logical epoch, and retire ownership only after
the matching consumer proof and acknowledgement. These tests qualify host
framing/contract behavior, not authentication, model execution or GPU visibility.
The strict host build and all 145 host tests pass with this addition.

Remaining integration includes authenticated peer/channel admission, separate
control and payload session ownership, HIP host staging and its completion
proof, the actual per-node model executor, remote peer progress, position/PLE
state contracts and distributed commit/token feedback. No runnable native
two-host inference or performance result is established by this addition.
