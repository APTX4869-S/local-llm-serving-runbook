# Troubleshooting — Real Incidents

Every incident below actually happened on this setup (12 GB card, 16 GB RAM,
Windows). Symptoms → root cause → fix. If your error message matches one of
these, skip straight to the fix.

---

## Incident 1 — Startup crash: `cudaMallocHost: out of memory`

**Symptom**: engine exits immediately at launch with a host-memory allocation
error. No GPU error, no crash dump.

**Root cause**: the shipped default pins **8 GiB of host KV cache**, and the host
state adds ~1.15 GiB. On a 16 GB machine, that exceeds free system RAM — the
allocation is on **host** memory, not VRAM, so a big GPU is irrelevant.

**Fix**: `--host-kv-mib 2048` (a quarter of the default). Re-measure: total
runtime stays well inside free RAM.

**Verification**: engine reaches "ready" and `curl /v1/models` returns 200.

---

## Incident 2 — VRAM hard check fails at 262144 context

**Symptom**: engine refuses to start with `--max-context 262144`; a hard VRAM
check reports the runtime needs more than the available budget.

**Root cause**: at 262144 tokens the runtime allocation is ~9.17 GiB. The card
is 12 GB, but under WDDM the desktop/compositor reserves part of the VRAM, and
CUDA only sees the remaining **~5.3 GiB**. The 9.17 GiB request can never fit.

**Fix**: drop context to 131072 (128K). Even at 128K, KV dtype matters — see
Incident 3.

---

## Incident 3 — 128K with fp8 KV "loads" but thrashes (decode −25%)

**Symptom**: server starts, but decode speed collapses (~25% slower) and
latency jitters.

**Root cause**: fp8 KV at 131072 tokens leaves only ~97 MiB of VRAM headroom.
WDDM starts evicting/moving buffers → severe thrash. It "fits" in the sense that
no hard check fails — which is worse, because nothing tells you until you
measure decode speed.

**Fix**: switch the KV cache dtype to 4-bit (`rk4v4`-style). This halves the KV
term, restores headroom, and is **faster than 64K-with-fp8 on both prefill and
decode** (measured).

**Lesson**: a config that "passes the checks" can still be silently broken —
always gate on **measured decode throughput**, not just on successful startup.

---

## Incident 4 — Silent service death (HTTP 000), no error in the log

**Symptom**: requests suddenly return connection refused (`000`); the engine
process is gone; no crash dump (no WER report), exit code 1, no FATAL line in
the log — the process died **idle**, seconds after the last successful response.

**Root cause (established by forensics, not logs)**:
- Event logs show no system-level crash for the process.
- A WER crash report would exist for a segfault — it does not.
- Death during idle + no error output ⇒ the process was **terminated
  externally** (e.g., killed via Task Manager / `taskkill`), not a defect in
  the engine.
- The last request before death had completed a streaming **tool-call** round
  trip (text → image → tool call); while this path is suspicious, a
  scripted reproduction (43 chunks) completed fine, so it was not reproduced.

**Fix / recovery**: restart the launcher — service is ready again in ~6 s.
For diagnosis, keep the heartbeat: poll `/v1/models` and record the last
request id + timing before the gap (the picture above was reconstructed
entirely from such polling data).

**Lesson**: no log ≠ no evidence. Correlate the *timestamp of the last
successful request* with process-death time and check for WER reports before
blaming the engine.

---

## Incident 5 — Streaming + tool-call interleave (flaky edge)

**Symptom**: in early builds, a request that streams a **text → tool-call**
sequence could misbehave right after the tool result was fed back (rare;
not reproducible in stress tests).

**Root cause**: the streaming+tool path is the most stateful one in the server
(partial JSON assembly, two rounds in one connection). The demo client
(`client/demo_client.py tools-demo`) exercises exactly this path on purpose so
the edge stays covered.

**Fix**: keep tool calling working through the **streaming** API, and let the
client merge `delta.tool_calls` fragments (see
`_assemble_streamed_tool_calls`). Prefer single-concurrency so a stuck request
cannot queue behind others.

---

## Incident 6 — `cudaMallocHost` OOM, then persistent `cudaErrorAlreadyMapped`

**Symptom**: the first startup fails with `cudaMallocHost: cudaErrorMemoryAllocation: out of memory` while pinning host state; every later attempt then fails **deterministically** (microsecond-scale) at `pinning host KV` with `cudaErrorAlreadyMapped: resource already mapped` — regardless of `--host-kv-mib` size.

**Environment at the time**: free RAM ~7.5 GB, free VRAM ~11 GB, no leftover serve process, no GPU-driver errors in the Windows event log. The *first* failure happened while the system commit charge was low (~5.5 GB free of 42 GB limit).

**Root cause (hypothesis, not proven)**: the first failed `cudaMallocHost` (host commit pressure) left a stale host mapping at the driver level; later launches collide with it (`already_mapped`), independent of requested size. The same launcher config had started cleanly on earlier days, so it is not a config defect.

**Workaround**:

- Restart the machine — the stale mapping is cleared, and the same launcher starts cleanly again.
- Do NOT keep relaunching: it will keep failing with `AlreadyMapped` until the driver state resets.
- If you are still in the *OOM* stage (first failure, before `AlreadyMapped` appears), free memory first (close large apps) before retrying — a single OOM at startup is what *creates* the stuck state.

**Lesson**: a transient `cudaMallocHost` OOM can poison all later launches with an `AlreadyMapped` error that looks like a code bug. Read the *first* failure line in the log, not the last, before deciding what to fix.

---

## Lessons (TL;DR)

1. On small VRAM cards, check the **WDDM budget**, not the card spec.
2. **Host-side** allocations (host KV) matter on small-RAM machines.
3. 4-bit KV is a feature, not a compromise: 128K @ 4-bit beat 64K @ fp8 on
   throughput in this setup.
4. A config that starts without errors is only "working" once decode
   throughput is measured.
5. When a service dies silently, look for external termination before
   blaming the engine; correlate request timestamps and WER reports.
6. After any crash: restart is cheap (~6 s) — measure, fix, move on.