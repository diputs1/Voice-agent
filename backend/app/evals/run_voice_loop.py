from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import Settings
from app.evals.elevenlabs_tests import (
    build_test_manifest,
    run_elevenlabs_agent_tests,
    test_ids_from_cases,
)
from app.evals.voice_tool_harness import (
    load_voice_cases,
    run_local_voice_harness,
    write_voice_report,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run Vin Agent voice harnesses.")
    parser.add_argument(
        "--mode", choices=["local", "elevenlabs", "manifest", "all"], default="local"
    )
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument("--branch-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_voice_cases(args.cases)
    reports = {}

    if args.mode in {"local", "all"}:
        local_report = await run_local_voice_harness(cases, site_id=args.site_id)
        local_path = write_voice_report(local_report)
        local_report["report_path"] = str(local_path)
        reports["local"] = local_report

    if args.mode == "manifest":
        manifest = build_test_manifest(cases)
        _write_or_print(manifest, args.output)
        return 0

    if args.mode in {"elevenlabs", "all"}:
        settings = Settings()
        api_key = settings.elevenlabs_api_key
        agent_id = settings.elevenlabs_agent_id
        if not api_key:
            raise SystemExit("ELEVENLABS_API_KEY is required for --mode elevenlabs/all")
        if not agent_id:
            raise SystemExit("ELEVENLABS_AGENT_ID is required for --mode elevenlabs/all")
        test_ids = sorted(set(args.test_id or test_ids_from_cases(cases)))
        if not test_ids:
            raise SystemExit(
                "No ElevenLabs test IDs configured. Add --test-id or fill elevenlabs_test_ids in voice_cases.json."
            )
        elevenlabs_report = await run_elevenlabs_agent_tests(
            api_key=api_key,
            agent_id=agent_id,
            test_ids=test_ids,
            repeat_count=args.repeat_count,
            branch_id=args.branch_id,
        )
        elevenlabs_path = write_voice_report(elevenlabs_report)
        elevenlabs_report["report_path"] = str(elevenlabs_path)
        reports["elevenlabs"] = elevenlabs_report

    passed = all(report["passed"] for report in reports.values()) if reports else True
    combined = {"passed": passed, "reports": reports}
    _write_or_print(combined, args.output)
    return 0 if passed else 1


def _write_or_print(payload: dict, path: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    print(text)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
