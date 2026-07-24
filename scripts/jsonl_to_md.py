#!/usr/bin/env python3
"""Render a Codex rollout.jsonl file to readable Markdown.

Usage: jsonl_to_md.py IN.jsonl [OUT.md]
"""
import json
import sys


def item_text(payload: dict) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        parts = []
        for c in content:
            if not isinstance(c, dict):
                continue
            if "text" in c:
                parts.append(c["text"])
            elif c.get("type") in ("input_image", "output_image"):
                parts.append(f"[image: {c.get('type')}]")
        return "\n".join(p for p in parts if p)
    if isinstance(content, str):
        return content
    return payload.get("text", "") or ""


def render(in_path: str) -> str:
    lines = [f"# Session transcript: {in_path}\n"]
    with open(in_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            payload = rec.get("payload", rec)
            rtype = rec.get("type", payload.get("type", ""))
            role = payload.get("role")

            if role in ("user", "assistant", "system"):
                text = item_text(payload)
                if text.strip():
                    lines.append(f"## {role}\n\n{text}\n")
            elif rtype == "function_call" or payload.get("type") == "function_call":
                name = payload.get("name", "?")
                args = payload.get("arguments", "")
                lines.append(f"### tool call: {name}\n\n```\n{args}\n```\n")
            elif rtype == "function_call_output" or payload.get("type") == "function_call_output":
                out = payload.get("output", "")
                if isinstance(out, dict):
                    out = out.get("content", out)
                lines.append(f"### tool output\n\n```\n{str(out)[:4000]}\n```\n")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.rsplit("/", 1)[-1].replace(".jsonl", ".md")
    md = render(in_path)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"wrote {out_path} ({len(md)} chars)")


if __name__ == "__main__":
    main()
