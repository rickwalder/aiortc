#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from pycc import (
    GoogCcController,
    PacedPacketInfo,
    RateConstraints,
    SentPacket,
    TransportPacketsFeedback,
)
from pycc.types import PacketResult


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def replay_trace(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controller = GoogCcController(RateConstraints())
    updates = []

    for record in records:
        if record.get("type") == "trace-start":
            controller = GoogCcController(RateConstraints())
        elif record.get("type") == "sent":
            controller.on_packet_sent(
                SentPacket(
                    transport_sequence_number=record["transport_sequence_number"],
                    send_time_us=record["send_time_us"],
                    size_bytes=record["size_bytes"],
                    ssrc=record["ssrc"],
                    rtp_sequence_number=record["rtp_sequence_number"],
                    is_retransmission=record["is_retransmission"],
                    pacing_info=PacedPacketInfo(record.get("probe_cluster_id", -1)),
                )
            )
        elif record.get("type") == "feedback":
            feedback = TransportPacketsFeedback(
                feedback_time_us=record["feedback_time_us"],
                packet_results=[
                    packet_result_from_trace(packet)
                    for packet in record.get("packets", [])
                ],
                data_in_flight_bytes=record["data_in_flight_bytes"],
            )
            update = controller.on_transport_feedback(feedback)
            if update is not None:
                diagnostics = controller.get_diagnostics()
                updates.append(
                    {
                        "feedback_time_us": record["feedback_time_us"],
                        "target_bitrate_bps": update.target_bitrate_bps,
                        "reason": update.reason,
                        "delay_usage": diagnostics.delay_usage,
                        "trend_ms": diagnostics.trend_ms,
                        "threshold_ms": diagnostics.trend_threshold_ms,
                        "accumulated_delay_ms": diagnostics.accumulated_delay_ms,
                        "smoothed_delay_ms": diagnostics.smoothed_delay_ms,
                        "acked_bitrate_bps": diagnostics.acked_bitrate_bps or 0,
                    }
                )

    return updates


def packet_result_from_trace(packet: list[Any]) -> PacketResult:
    (
        transport_sequence_number,
        send_time_us,
        receive_time_us,
        size_bytes,
        ssrc,
        rtp_sequence_number,
        is_retransmission,
        reported_lost,
        reported_recovered,
        prior_unacked_data_bytes,
        probe_cluster_id,
    ) = packet
    return PacketResult(
        sent_packet=SentPacket(
            transport_sequence_number=transport_sequence_number,
            send_time_us=send_time_us,
            size_bytes=size_bytes,
            ssrc=ssrc,
            rtp_sequence_number=rtp_sequence_number,
            is_retransmission=bool(is_retransmission),
            pacing_info=PacedPacketInfo(probe_cluster_id),
            prior_unacked_data_bytes=prior_unacked_data_bytes,
        ),
        receive_time_us=receive_time_us,
        reported_lost_for_first_time=bool(reported_lost),
        reported_recovered_for_first_time=bool(reported_recovered),
    )


def summarize(records: list[dict[str, Any]]) -> None:
    sent = [record for record in records if record.get("type") == "sent"]
    feedback = [record for record in records if record.get("type") == "feedback"]
    live_updates = [
        record for record in records if record.get("type") == "target-update"
    ]
    replay_updates = replay_trace(records)

    sent_bytes = sum(record["size_bytes"] for record in sent)
    payload_bytes = sum(record["payload_size_bytes"] for record in sent)
    start_us = min((record["send_time_us"] for record in sent), default=0)
    end_us = max(
        [record["send_time_us"] for record in sent]
        + [record["feedback_time_us"] for record in feedback],
        default=start_us,
    )
    duration_s = max(0.001, (end_us - start_us) / 1_000_000)

    print(f"records={len(records)} sent={len(sent)} feedback={len(feedback)}")
    print(
        "duration_s=%.3f sent_bps=%d payload_bps=%d"
        % (
            duration_s,
            int(sent_bytes * 8 / duration_s),
            int(payload_bytes * 8 / duration_s),
        )
    )
    if feedback:
        feedback_sizes = [len(record.get("packets", [])) for record in feedback]
        print(
            "feedback_packets min=%d median=%s max=%d"
            % (min(feedback_sizes), median(feedback_sizes), max(feedback_sizes))
        )
    print_update_summary("live", live_updates)
    print_update_summary("replay", replay_updates)


def print_update_summary(name: str, updates: list[dict[str, Any]]) -> None:
    if not updates:
        print(f"{name}_updates=0")
        return

    targets = [record["target_bitrate_bps"] for record in updates]
    reasons = Counter(record["reason"] for record in updates)
    print(
        f"{name}_updates={len(updates)} "
        f"target_min={min(targets)} target_median={int(median(targets))} "
        f"target_max={max(targets)} reasons={dict(reasons)}"
    )
    for record in updates[:8]:
        print(
            "%s update t=%.3fs target=%d reason=%s usage=%s trend=%.3f "
            "threshold=%.3f acc=%.3f smooth=%.3f acked=%d"
            % (
                name,
                record.get("feedback_time_us", 0) / 1_000_000,
                record["target_bitrate_bps"],
                record["reason"],
                record.get("delay_usage", ""),
                record.get("trend_ms", 0.0),
                record.get("trend_threshold_ms", record.get("threshold_ms", 0.0)),
                record.get("accumulated_delay_ms", 0.0),
                record.get("smoothed_delay_ms", 0.0),
                record.get("acked_bitrate_bps", 0),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    summarize(load_records(args.trace))


if __name__ == "__main__":
    main()
