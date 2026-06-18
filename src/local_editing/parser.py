"""DeepSeek-based instruction parser for local image editing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from dotenv import load_dotenv


@dataclass(frozen=True)
class EditInstruction:
    operation: str
    target: str
    source_color: str | None = None
    target_color: str | None = None
    replacement: str | None = None
    edit_prompt: str | None = None


class InstructionParseError(ValueError):
    pass


def parse_instruction(text: str) -> EditInstruction:
    """Parse an image-editing instruction with DeepSeek V4 Flash."""

    load_environment()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise InstructionParseError("Set DEEPSEEK_API_KEY before parsing instructions")

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Parse image-editing instructions for a local editing pipeline. "
                    "Return JSON only. Schema: "
                    '{"operation":"color_change|remove|replace",'
                    '"target":"short visual object phrase",'
                    '"source_color":"color or null",'
                    '"target_color":"color or null",'
                    '"replacement":"replacement object phrase or null",'
                    '"edit_prompt":"concise English inpainting prompt"} '
                    "The target must be the object to ground and segment in the original image. "
                    "For removal, edit_prompt should request realistic background completion. "
                    "For replacement, edit_prompt should describe the replacement object while "
                    "preserving the surrounding scene outside the mask."
                ),
            },
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 300,
        "stream": False,
    }

    http_request = request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise InstructionParseError(f"DeepSeek API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise InstructionParseError(f"DeepSeek API request failed: {exc}") from exc

    data = json.loads(body)
    content = data["choices"][0]["message"]["content"]
    return instruction_from_json(json.loads(content), source_text=text)


def instruction_from_json(payload: dict[str, Any], source_text: str = "") -> EditInstruction:
    operation = str(payload.get("operation", "")).strip().lower()
    if operation not in {"color_change", "remove", "replace"}:
        raise InstructionParseError(f"Unsupported operation from parser: {operation!r} for {source_text!r}")

    target = _clean_nullable(payload.get("target"))
    if not target:
        raise InstructionParseError(f"Parser did not return a target object for {source_text!r}")

    return EditInstruction(
        operation=operation,
        target=target,
        source_color=_clean_nullable(payload.get("source_color")),
        target_color=_clean_nullable(payload.get("target_color")),
        replacement=_clean_nullable(payload.get("replacement")),
        edit_prompt=_clean_nullable(payload.get("edit_prompt")),
    )


def build_inpaint_prompt(instruction: EditInstruction, original_text: str | None = None) -> str:
    if instruction.edit_prompt:
        return instruction.edit_prompt
    if instruction.operation == "color_change":
        color = instruction.target_color or "newly colored"
        return f"a realistic {color} {instruction.target}, preserve the surrounding scene"
    if instruction.operation == "remove":
        return f"remove the {instruction.target}, fill the area with realistic background"
    if instruction.operation == "replace":
        replacement = instruction.replacement or "object"
        return f"a realistic {replacement}, preserve the surrounding scene outside the object"
    if original_text:
        return original_text
    raise InstructionParseError(f"Cannot build an inpainting prompt for {instruction!r}")


def _clean_nullable(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned in {"", "none", "null", "n/a"}:
        return None
    return cleaned


def load_environment() -> None:
    """Load local .env files without overriding shell-provided variables."""

    load_dotenv()
    repo_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_env.exists():
        load_dotenv(repo_env, override=False)
