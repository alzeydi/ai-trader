"""Compare alternative system prompts against a saved set of historical signals.

Drop additional `.txt` files into src/llm/prompts/ and pass them via --variant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import settings
from src.llm.client import ClaudeClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True,
                        help="paths to alternate system prompts")
    parser.add_argument("--fixtures", default="tests/fixtures/signals.json",
                        help="JSON list of {signal, context, equity}")
    args = parser.parse_args()

    fixtures_path = settings.project_root / args.fixtures
    if not fixtures_path.exists():
        raise SystemExit(f"fixtures not found: {fixtures_path}")
    cases = json.loads(fixtures_path.read_text(encoding="utf-8"))
    client = ClaudeClient()

    for variant in args.variants:
        sys_text = Path(variant).read_text(encoding="utf-8")
        allow = reject = 0
        for case in cases:
            user = json.dumps(case, indent=2)
            raw = client.complete(system=sys_text, user=user)
            try:
                payload = json.loads(raw)
                if str(payload.get("decision", "")).upper() == "ALLOW":
                    allow += 1
                else:
                    reject += 1
            except json.JSONDecodeError:
                reject += 1
        total = allow + reject
        print(f"{variant}: ALLOW={allow}/{total}  REJECT={reject}/{total}")


if __name__ == "__main__":
    main()
