# Device queue trace

`--device-queue-trace` is explicit profiling instrumentation for the eager
single-request dispatcher and its actual shared-expert side queue. It requires
host dispatch telemetry and the owned deferred-status stream. Graph, serving, prompt lookup and known-prefix routes
are rejected by this CLI slice. Ordinary benchmark runs do not enable it.

The owner allocates a bounded pool of timing events during setup. The dispatcher
records a pair around each launcher on the actual current HIP stream, after
launcher admission. The side branch is observed inside its real side scope.
There is no allocation, event synchronization or stream synchronization per
observed launcher. Exhaustion drops observations without reusing event handles.
An active capture is detected before any event is recorded.

All queues use one same-device timing-event origin. A newly observed queue waits
on this origin once; these additional waits and the timing events are profiling
overhead. HIP event timestamps mark queue execution boundaries, at approximately
microsecond resolution. Export occurs after the caller's existing completion
and uses event queries and elapsed-time reads without adding a fence. Missing,
invalid or incomplete event times remain unknown. Exceptional owner teardown
drains the original device before releasing events when completion is unknown.
Observation APIs can surface asynchronous target failures. Their errors are
retained, never cleared by the trace. The existing transaction completion calls
`require_healthy_before_commit` before publishing state, so a failed observation
cannot be hidden behind a successful model commit.

These intervals enclose launcher queue work, including host-fed gaps and waits.
They are not individual kernel intervals. Kernel busy time and kernel overlap
remain unknown; the host clock is never subtracted from the device event clock.
Actual accepted fork/join event-record and stream-wait operations are exported
separately, with versioned references to preceding records. Their host acceptance
alone does not prove GPU completion. The trace neither grants numerical
qualification nor establishes a performance improvement.

HIP timing semantics follow the [event management API](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___event.html).

`tools/qualification/queue_trace_evidence.py` reads one existing exported trace
object and writes a fresh JSON receipt without launching any process. It merges
recorded intervals per queue before computing their union and overlap between
distinct queues, retaining the declared request/session/execution/position groups.
Unknown/dropped/error observations cannot count as complete coverage. These are
observed launcher-span metrics including gaps and waits, not kernel utilization,
a full-workload wall interval or a measured device critical path. Kernel busy
time, kernel overlap and device critical path remain null. Source fixtures and
this offline consumer are unexecuted until the final validation stage.
