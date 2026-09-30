# Performance & Memory Trade-offs

Measured on: RTX 4070 SUPER (12 GB, sm_89), 16 GB RAM, Windows 11, 128K ctx,
4-bit KV (`rk4v4`), MTP draft ×3.

## 1. Context size × KV dtype

| Context | KV dtype | Outcome |
|---------|----------|---------|
| 262144  | default  | hard VRAM check fails (~9.17 GiB needed, ~5.3 GiB available after WDDM budget) |
| 131072  | fp8      | "fits" but ~97 MiB headroom → WDDM thrash, decode **−25%** |
| 131072  | 4-bit    | fits with margin; **faster than 64K@fp8** on prefill and decode |

The takeaway: on a 12 GB card, the KV dtype is the decision that unlocks the
context length. 4-bit KV halves the KV term; the freed headroom eliminates
thrash, which is why it wins even against a *smaller* fp8 context.

## 2. Throughput

| Metric | Value | Condition |
|--------|-------|-----------|
| Prefill | ~1.7k tok/s | 128K ctx, 4-bit KV |
| Decode | ~60 tok/s | prose, thinking off |
| Startup → ready | ~6 s | cold start |

Single concurrency: `--max-concurrency 1`. A long request (huge context or long
generation) **owns the engine** until it finishes — acceptable for personal use;
add a queue upstream if you share the endpoint.

## 3. Thinking modes — latency vs correctness

Same prompt ("9.11 vs 9.9: which is larger?"), measured on this service:

| Mode | Wall time | Output | Correct? |
|------|-----------|--------|----------|
| default (xhigh) | 2.8 s / 113 tok | full reasoning then answer | Yes |
| low | 2.1 s / 111 tok | shorter reasoning | Yes |
| off | 1.4 s / 0 tok | answer only, no reasoning | **No** (answers 9.11) |

What this means in practice:

- **Thinking off** roughly halves latency on short prompts, but non-trivial
  reasoning degrades — and in this particular probe it flipped the answer.
- **Tool calling depends on thinking** (the model must emit a well-formed tool
  call; thinking is what drives that). Do not globally disable thinking on a
  service that should act as an agent.
- Suggested policy: chatty/trivial tasks → `off` or `low`; anything that
  matters → default (or `low` if you want a speed/quality middle ground).

Control via the API: `enable_thinking: false` (off) or
`reasoning_effort: low|medium|xhigh` (the demo client exposes `--thinking`).

## 4. General tuning checklist (in priority order)

1. KV dtype (4-bit) — biggest win on small VRAM.
2. Context length — 128K fits; 256K does not; measure before trusting "fits".
3. Host-KV pinning — reduce from default (8 GiB) to ~2 GiB on 16 GB machines.
4. Prefill chunking — 1024 keeps peak memory stable during long prompts.
5. Speculative decoding (MTP, 3 draft tokens) — free-ish decode speedup; keep.
6. Single concurrency — predictable latency for personal use; queue if shared.