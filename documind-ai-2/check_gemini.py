"""Quick Gemini connectivity check: python check_gemini.py

Tells you whether your key works, which models it can use, and the exact error Google returns.
"""
from __future__ import annotations

import sys

from config import load_settings


def main() -> int:
    s = load_settings()
    if not s.has_api_key:
        print("X No GEMINI_API_KEY found. Put it in a file named .env next to app.py:")
        print("    GEMINI_API_KEY=your_key_here")
        return 1
    print(f"Key found (starts with {s.gemini_api_key[:4]}..., {len(s.gemini_api_key)} chars). Model: {s.gemini_model}")

    from google import genai
    from google.genai import errors

    from src.llm_service import GeminiService

    client = genai.Client(api_key=s.gemini_api_key)
    try:
        names = [m.name.removeprefix("models/") for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
        print(f"OK Key accepted. {len(names)} text models available, e.g.:")
        for n in [n for n in names if "flash" in n][:12]:
            print("   -", n)
    except errors.APIError as exc:
        print(f"X Listing models failed: HTTP {exc.code}: {exc.message}")
        return 2
    except Exception as exc:
        print(f"X Could not reach Google ({type(exc).__name__}): {exc}")
        return 2

    candidates = [s.gemini_model] + [m for m in ("gemini-2.5-flash-lite", "gemini-2.5-flash") if m != s.gemini_model]
    for model in candidates:
        print(f"\nTrying a tiny request on {model} ...")
        svc = GeminiService(s.with_overrides(gemini_model=model))
        try:
            out = svc.generate("Reply with JSON only.", 'Return {"ok": true}', json_output=True, max_output_tokens=200)
            print("OK Works:", out.strip()[:80])
            print(f"\nUse this in .env:  GEMINI_MODEL={model}")
            return 0
        except Exception as exc:
            print("X", exc)
    return 3


if __name__ == "__main__":
    sys.exit(main())
