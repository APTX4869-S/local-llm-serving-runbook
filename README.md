# Local LLM Serving on a 12 GB GPU — A Runbook

Serving a **27B-parameter LLM with a 128K-token context** on a single consumer GPU
(RTX 4070 SUPER, 12 GB VRAM, 16 GB host RAM, Windows 11). Includes:

- the configuration and the memory ledger that makes 128K context fit a 12 GB card,
- an OpenAI-compatible demo client (streaming, tool calling, vision, thinking control),
- a troubleshooting log of **real incidents** that happened in this setup.

> Status: personal production setup, running daily since 2026. All numbers below were
> measured on the hardware in §2; nothing is theoretical.

---

## 1. Why this is non-trivial

The naive path fails three times before you reach a working config:

1. **VRAM check.** The engine's default `--max-context 262144` needs a ~9.17 GiB
   runtime allocation. After the Windows WDDM budget (the desktop/compositor reserves
   VRAM that CUDA cannot use), only **~5.3 GiB** is available → hard check fails.
2. **Host memory.** The default host-side KV pinning (8 GiB) plus host state
   (1.15 GiB) exceeds 16 GB of free RAM → `cudaMallocHost: out of memory` at startup.
3. **KV cache size.** A 128K context in **fp8 KV** does not fit either — it "loads"
   with ~97 MiB of headroom, then thrashes and decodes ~25% slower.

The working config solves all three with **4-bit KV caching (`rk4v4`)** — it halves
the KV term, fits in VRAM, and is *faster* than 64K-with-fp8 on both prefill and
decode (measured, not assumed).

---

## 2. Hardware & environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 4070 SUPER — 12 GB, Ada, sm_89 |
| Host RAM | 15.7–16 GB |
| OS | Windows 11 (WDDM; CUDA toolkit NOT required — driver-side runtime only) |
| Inference engine | OpenAI-compatible local server (MTP draft, `sm_89` build) |
| Model | 27B-class, 4-bit / ternary PTQ, `[MODEL_NAME]` |
| API | OpenAI-compatible at `http://127.0.0.1:8085/v1` |

## 3. The working configuration

```
--max-context     131072        # 128K tokens
--kv-capacity     131072        # KV slots = context
--kv-dtype        rk4v4         # 4-bit KV cache — THE key decision (see §1)
--max-concurrency 1             # single concurrent request; long jobs own the engine
--prefill-chunk   1024          # chunked prefill for memory stability
--host-kv-mib     2048          # host-side KV pinning, reduced from the 8 GiB default
--speculative     mtp --draft-tokens 3    # MTP draft model, 3 draft tokens
--cors            open CORS for browser UIs
--tolerant-tool-calls           # lenient tool-call JSON parsing
--vision                         # image input enabled
```

Why 128K works when fp8 didn't: at 131072 tokens the fp8 KV cache leaves only
~97 MiB of headroom (measured), which causes WDDM thrash and a ~25% slower
decode. Switching the KV cache to 4-bit halves the KV term and restores
headroom. Measured effect: `rk4v4` at 128K is **faster than `fp8` at 64K** on
both prefill and decode.

---

## 4. Quick start

```powershell
# 1. edit paths in scripts\start-windows-12g.bat (engine, model, port)
scripts\start-windows-12g.bat

# 2. verify (expect 200)
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8085/v1/models
```

The engine can serve right after ~6 s. If the service dies (see
`docs/troubleshooting.md` for real cases), restarting the script recovers it.
**One exception**: if startup fails with `cudaErrorAlreadyMapped`, relaunching
will keep failing — reboot the machine once, then the same script starts cleanly
(Incident 6 in the troubleshooting doc).

---

## 5. Demo client

`client/demo_client.py` is a dependency-light (only `requests`) OpenAI-compatible
client that exercises the service end-to-end:

```powershell
python client\demo_client.py health                      # /v1/models probe
python client\demo_client.py chat "Explain what a tensor is"   # plain chat
python client\demo_client.py chat --stream "Tell me a joke"    # streaming
python client\demo_client.py tools-demo                       # tool calling loop
python client\demo_client.py vision --image screenshot.png     # image input
python client\demo_client.py chat --thinking off "9.11 vs 9.9: which is larger?"
```

Notes on behaviors seen in this setup:

- Thinking (`reasoning_content`) is returned as incremental chunks in streaming mode.
- `--thinking off` (via `enable_thinking:false` / `reasoning_effort:low`) roughly
  doubles token latency for short answers, but **complex reasoning degrades** — keep
  thinking on for anything non-trivial (details in `docs/performance.md`).
- Tool calling + streaming is exercised by `tools-demo`, which reconstructs
  `delta.tool_calls` fragments — the exact path that has been crash-prone in early
  builds, so the client mirrors it deliberately.

---

## 6. Measured performance

| Metric | Value | Condition |
|--------|-------|-----------|
| Prefill | ~1.7k tok/s | 128K ctx, `rk4v4` |
| Decode | ~60 tok/s | prose, thinking off |
| Startup to ready | ~6 s | cold start |

Latency/quality trade-offs of thinking modes are in `docs/performance.md`.

---

## 7. Repository layout

```
local-llm-serving-runbook/
├── README.md                  # this file
├── scripts/
│   └── start-windows-12g.bat  # machine-adapted launcher (edit paths)
├── client/
│   ├── requirements.txt
│   └── demo_client.py         # streaming / tools / vision / thinking demo
└── docs/
    ├── performance.md         # memory & speed trade-offs, thinking modes
    └── troubleshooting.md    # real incidents, root causes, fixes
```

---

## 8. License & attribution

Code: MIT. The model is used under its own license — check before redistributing
artifacts. This is a personal-study deployment; do not use it as a production
service without quota/authentication.