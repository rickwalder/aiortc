from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)

_TRACE_ENV = "AIORTC_TRANSPORT_CC_TRACE_JSONL"


class TransportCcTraceWriter:
    """
    Append-only JSONL trace for send-side transport-cc diagnostics.

    This is intentionally opt-in and compact. The records are meant for local
    replay / analysis, not for user-facing logs.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("a", encoding="utf-8", buffering=1)
        self._closed = False
        self.write_event(
            "trace-start",
            wall_time=time.time(),
            pid=os.getpid(),
            version=1,
        )

    @classmethod
    def from_environment(cls) -> "TransportCcTraceWriter | None":
        path = os.environ.get(_TRACE_ENV)
        if not path:
            return None
        try:
            return cls(path)
        except OSError:
            logger.warning("could not open transport-cc trace file %s", path)
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file.close()

    def write_sent(self, packet: Any) -> None:
        pacing_info = getattr(packet, "pacing_info", None)
        self.write_event(
            "sent",
            transport_sequence_number=packet.transport_sequence_number,
            send_time_us=packet.send_time_us,
            size_bytes=packet.size_bytes,
            payload_size_bytes=getattr(packet, "payload_size_bytes", 0),
            ssrc=packet.ssrc,
            rtp_sequence_number=packet.rtp_sequence_number,
            is_retransmission=packet.is_retransmission,
            probe_cluster_id=getattr(pacing_info, "probe_cluster_id", -1),
        )

    def write_feedback(self, *, feedback: Any, normalized: Any) -> None:
        packets = []
        for result in normalized.packet_results:
            sent = result.sent_packet
            packets.append(
                [
                    sent.transport_sequence_number,
                    sent.send_time_us,
                    result.receive_time_us,
                    sent.size_bytes,
                    sent.ssrc,
                    sent.rtp_sequence_number,
                    int(sent.is_retransmission),
                    int(result.reported_lost_for_first_time),
                    int(result.reported_recovered_for_first_time),
                    sent.prior_unacked_data_bytes,
                    getattr(sent.pacing_info, "probe_cluster_id", -1),
                ]
            )

        self.write_event(
            "feedback",
            feedback_time_us=normalized.feedback_time_us,
            sender_ssrc=feedback.sender_ssrc,
            media_ssrc=feedback.media_ssrc,
            base_sequence_number=feedback.base_sequence_number,
            feedback_packet_count=feedback.feedback_packet_count,
            packet_status_count=len(feedback.packets),
            data_in_flight_bytes=normalized.data_in_flight_bytes,
            packets=packets,
        )

    def write_target_update(
        self,
        *,
        previous_target: int | None,
        update: Any,
        telemetry: Any,
    ) -> None:
        self.write_event(
            "target-update",
            feedback_time_us=telemetry.last_feedback_time_us,
            target_bitrate_bps=update.target_bitrate_bps,
            stable_target_bitrate_bps=update.stable_target_bitrate_bps,
            previous_target_bitrate_bps=previous_target,
            reason=update.reason,
            loss_fraction=update.loss_fraction,
            rtt_us=update.rtt_us,
            delay_usage=telemetry.delay_usage,
            aimd_state=telemetry.aimd_state,
            acked_bitrate_bps=telemetry.acked_bitrate_bps,
            trend_ms=telemetry.trend_ms,
            raw_trend=telemetry.raw_trend,
            trend_threshold_ms=telemetry.trend_threshold_ms,
            accumulated_delay_ms=telemetry.accumulated_delay_ms,
            smoothed_delay_ms=telemetry.smoothed_delay_ms,
            trend_window_ms=telemetry.trend_window_ms,
            send_delta_ms=telemetry.last_send_delta_ms,
            receive_delta_ms=telemetry.last_receive_delta_ms,
            delay_delta_ms=telemetry.last_delay_delta_ms,
            group_bytes=telemetry.last_group_bytes,
            data_in_flight_bytes=telemetry.data_in_flight_bytes,
            congestion_window_bytes=telemetry.congestion_window_bytes,
            congestion_window_fill_ratio=telemetry.congestion_window_fill_ratio,
            in_alr=telemetry.in_alr,
            alr_budget_ratio=telemetry.alr_budget_ratio,
            link_capacity_bps=telemetry.link_capacity_bps,
            probe_cluster_id=telemetry.probe_cluster_id,
            probe_target_bitrate_bps=telemetry.probe_target_bitrate_bps,
            last_probe_bitrate_bps=telemetry.last_probe_bitrate_bps,
        )

    def write_event(self, event_type: str, **payload: Any) -> None:
        if self._closed:
            return
        payload["type"] = event_type
        self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
