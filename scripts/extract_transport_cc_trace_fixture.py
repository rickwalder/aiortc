#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, TextIO


def record_time_us(record: dict[str, Any]) -> int | None:
    if "feedback_time_us" in record:
        return int(record["feedback_time_us"])
    if "send_time_us" in record:
        return int(record["send_time_us"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--start-us", type=int, required=True)
    parser.add_argument("--end-us", type=int, required=True)
    parser.add_argument(
        "--types",
        default="trace-start,sent,feedback,target-update",
        help="comma-separated record types to retain",
    )
    parser.add_argument(
        "--synthetic-trace-start",
        action="store_true",
        help="insert a trace-start marker at the beginning of the fixture",
    )
    args = parser.parse_args()

    allowed_types = {item.strip() for item in args.types.split(",") if item.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    first_time_us: int | None = None
    last_time_us: int | None = None
    started = False

    with args.input.open(encoding="utf-8") as input_file:
        output_file: TextIO
        if args.output.suffix == ".gz":
            output_file = gzip.open(args.output, "wt", encoding="utf-8")
        else:
            output_file = args.output.open("w", encoding="utf-8")

        with output_file:
            metadata = {
                "type": "fixture-metadata",
                "name": args.name,
                "description": args.description,
                "source": str(args.input),
                "start_us": args.start_us,
                "end_us": args.end_us,
                "retained_types": sorted(allowed_types),
            }
            output_file.write(json.dumps(metadata, separators=(",", ":")) + "\n")

            if args.synthetic_trace_start:
                output_file.write(
                    json.dumps(
                        {
                            "type": "trace-start",
                            "synthetic": True,
                            "fixture_start_us": args.start_us,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                counts["trace-start"] = 1

            for line in input_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_type = record.get("type")
                event_time_us = record_time_us(record)

                if record_type == "trace-start":
                    if event_time_us is None and not started:
                        continue
                    if "trace-start" not in allowed_types:
                        continue
                    continue

                if record_type not in allowed_types:
                    continue
                if event_time_us is None:
                    continue
                if event_time_us < args.start_us:
                    continue
                if event_time_us > args.end_us:
                    if started:
                        break
                    continue

                started = True
                first_time_us = (
                    event_time_us
                    if first_time_us is None
                    else min(first_time_us, event_time_us)
                )
                last_time_us = (
                    event_time_us
                    if last_time_us is None
                    else max(last_time_us, event_time_us)
                )
                counts[record_type] = counts.get(record_type, 0) + 1
                output_file.write(json.dumps(record, separators=(",", ":")) + "\n")

            summary = {
                "type": "fixture-summary",
                "name": args.name,
                "first_time_us": first_time_us,
                "last_time_us": last_time_us,
                "counts": counts,
            }
            output_file.write(json.dumps(summary, separators=(",", ":")) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
