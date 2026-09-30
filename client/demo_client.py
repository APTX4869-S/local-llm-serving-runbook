#!/usr/bin/env python3
"""OpenAI-compatible demo client for a local LLM service.

Exercises the deployment end-to-end:
  * streaming responses (with incremental reasoning chunks, when present)
  * tool/function calling, including the streamed `delta.tool_calls` path
  * vision (image input via base64 data-URI)
  * thinking-mode control (enable_thinking / reasoning_effort)

Only dependency: `requests`.
"""

import argparse
import base64
import datetime
import json
import sys
import time

import requests

DEFAULT_BASE = "http://127.0.0.1:8085/v1"


# ---------------------------------------------------------------------------
# tiny helpers
# ---------------------------------------------------------------------------

def resolve_model(base: str) -> str:
    """Return the first model id the server advertises (or a sensible default)."""
    try:
        r = requests.get(f"{base}/models", timeout=5)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "local-model"  # fallback: change to your model id


def print_reasoning(content: str) -> None:
    """Print thinking trace to stderr so the final answer stays clean on stdout."""
    for line in content.splitlines():
        sys.stderr.write(f"  \033[2m[think] {line}\033[0m\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# 1. plain / streaming chat
# ---------------------------------------------------------------------------

def chat(base: str, model: str, prompt: str, stream: bool, thinking: str | None,
         system: str | None = None) -> None:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if thinking == "off":
        payload["enable_thinking"] = False
    elif thinking == "low":
        payload["reasoning_effort"] = "low"

    t0 = time.time()
    if not stream:
        r = requests.post(f"{base}/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        if msg.get("reasoning_content"):
            print_reasoning(msg["reasoning_content"])
        print(msg["content"] or "")
        print(f"\n[done in {time.time() - t0:.1f}s]", file=sys.stderr)
        return

    # --- streaming ---
    with requests.post(f"{base}/chat/completions", json=payload, stream=True,
                       timeout=300) as resp:
        resp.raise_for_status()
        n_tokens = 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            if delta.get("reasoning_content"):
                # print incremental thinking trace to stderr, answer stays on stdout
                sys.stderr.write(delta["reasoning_content"])
                sys.stderr.flush()
            content = delta.get("content")
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                n_tokens += 1
        print()
        print(f"\n[stream done in {time.time() - t0:.1f}s, ~{n_tokens} content tokens]",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# 2. tool calling (streamed) — the crash-prone path, exercised deliberately
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def execute_tool(name: str, _args: dict) -> str:
    """Our only tool. In a real app you would dispatch on `name`."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def _assemble_streamed_tool_calls(resp) -> tuple[dict, str]:
    """Reconstruct assistant message + raw text from a stream, merging
    delta.tool_calls fragments (indexed call objects). Returns
    (assistant_message, assistant_text)."""
    assistant = {"role": "assistant", "content": None}
    tool_calls: dict[int, dict] = {}
    text_parts: list[str] = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            delta = json.loads(chunk)["choices"][0]["delta"]
        except (KeyError, IndexError, json.JSONDecodeError):
            continue
        if delta.get("reasoning_content"):
            print_reasoning(delta["reasoning_content"])
        if delta.get("content"):
            text_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(idx, {"id": None, "type": "function",
                                               "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            if tc.get("function"):
                if tc["function"].get("name"):
                    slot["function"]["name"] += tc["function"]["name"]
                if tc["function"].get("arguments"):
                    slot["function"]["arguments"] += tc["function"]["arguments"]
    if text_parts:
        text = "".join(text_parts)
        assistant["content"] = text
    if tool_calls:
        assistant["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return assistant, text


def tools_demo(base: str, model: str) -> None:
    """Round 1 asks the model to call a tool; we execute it and feed the result
    back in round 2, then print the final answer."""
    messages = [{"role": "user",
                 "content": "What is the current date and time? Use the tool."}]

    print("== round 1: model decides to call a tool (streamed) ==")
    payload = {"model": model, "messages": messages, "tools": TOOLS,
               "tool_choice": "auto", "stream": True}
    with requests.post(f"{base}/chat/completions", json=payload, stream=True,
                       timeout=300) as resp:
        resp.raise_for_status()
        assistant, text = _assemble_streamed_tool_calls(resp)
    if text:
        print("assistant text:", text[:200])
    if not assistant.get("tool_calls"):
        print("!! model did not call the tool — server may not support tools, or")
        print("   thinking is disabled (tool calling depends on thinking).")
        sys.exit(2)

    print(f"tool_calls -> {json.dumps(assistant['tool_calls'], ensure_ascii=False)}")
    messages.append(assistant)
    for tc in assistant["tool_calls"]:
        result = execute_tool(tc["function"]["name"], {})
        print(f"executing {tc['function']['name']}() -> {result}")
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    print("\n== round 2: final answer (stream) ==")
    payload = {"model": model, "messages": messages, "stream": True}
    with requests.post(f"{base}/chat/completions", json=payload, stream=True,
                       timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            if delta.get("reasoning_content"):
                print_reasoning(delta["reasoning_content"])
            if delta.get("content"):
                sys.stdout.write(delta["content"])
                sys.stdout.flush()
    print()


# ---------------------------------------------------------------------------
# 3. vision
# ---------------------------------------------------------------------------

def vision(base: str, model: str, image_path: str, prompt: str) -> None:
    with open(image_path, "rb") as f:
        mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        data_uri = f"data:{mime};base64," + base64.b64encode(f.read()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]}]
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(f"{base}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    if msg.get("reasoning_content"):
        print_reasoning(msg["reasoning_content"])
    print(msg.get("content") or "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=DEFAULT_BASE, help=f"API base (default {DEFAULT_BASE})")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("health", help="probe /v1/models")
    c.set_defaults(func=lambda a: print(resolve_model(a.base)))

    c = sub.add_parser("chat", help="plain or streaming chat")
    c.add_argument("prompt")
    c.add_argument("--stream", action="store_true")
    c.add_argument("--thinking", choices=["default", "low", "off"], default="default")
    c.add_argument("--system")
    c.set_defaults(func=lambda a: chat(a.base, resolve_model(a.base), a.prompt,
                                       a.stream, a.thinking, a.system))

    c = sub.add_parser("tools-demo", help="tool-calling round trip")
    c.set_defaults(func=lambda a: tools_demo(a.base, resolve_model(a.base)))

    c = sub.add_parser("vision", help="ask the model about an image")
    c.add_argument("--image", required=True)
    c.add_argument("--prompt", default="Describe this image in detail.")
    c.set_defaults(func=lambda a: vision(a.base, resolve_model(a.base), a.image, a.prompt))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())