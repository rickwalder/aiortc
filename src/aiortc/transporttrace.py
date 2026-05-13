from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)

NET_TRACE_ENV = "AIORTC_NET_TRACE"


class NetTraceWriter:
    """
    Append-only JSONL trace for aiortc network diagnostics.
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
    def from_environment(cls) -> "NetTraceWriter | None":
        path = os.environ.get(NET_TRACE_ENV)
        if not path:
            return None
        try:
            return cls(path)
        except OSError:
            logger.warning("could not open aiortc network trace file %s", path)
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
            trace="transport-cc",
            regime="twcc-gcc",
            trace_role="sender",
            direction="outbound-rtp",
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
            trace="transport-cc",
            regime="twcc-gcc",
            trace_role="sender",
            direction="inbound-rtcp",
            feedback_time_us=normalized.feedback_time_us,
            sender_ssrc=feedback.sender_ssrc,
            media_ssrc=feedback.media_ssrc,
            base_sequence_number=feedback.base_sequence_number,
            feedback_packet_count=feedback.feedback_packet_count,
            packet_status_count=len(feedback.packets),
            data_in_flight_bytes=normalized.data_in_flight_bytes,
            packets=packets,
        )

    def write_receiver_feedback(
        self, *, feedback: Any, feedback_time_us: int
    ) -> None:
        packets = [
            [
                packet.sequence_number,
                packet.received,
                packet.receive_delta_us,
            ]
            for packet in feedback.packets
        ]
        self.write_event(
            "receiver-feedback",
            trace="transport-cc",
            regime="twcc-gcc",
            trace_role="receiver",
            direction="outbound-rtcp",
            feedback_time_us=feedback_time_us,
            sender_ssrc=feedback.sender_ssrc,
            media_ssrc=feedback.media_ssrc,
            base_sequence_number=feedback.base_sequence_number,
            feedback_packet_count=feedback.feedback_packet_count,
            packet_status_count=len(feedback.packets),
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
            trace="transport-cc",
            regime="twcc-gcc",
            trace_role="sender",
            direction="local-control",
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
            pacing_queue_bytes=telemetry.pacing_queue_bytes,
            pacing_queue_oldest_age_ms=telemetry.pacing_queue_oldest_age_ms,
            congestion_window_bytes=telemetry.congestion_window_bytes,
            congestion_window_fill_ratio=telemetry.congestion_window_fill_ratio,
            in_alr=telemetry.in_alr,
            alr_budget_ratio=telemetry.alr_budget_ratio,
            link_capacity_bps=telemetry.link_capacity_bps,
            probe_cluster_id=telemetry.probe_cluster_id,
            probe_target_bitrate_bps=telemetry.probe_target_bitrate_bps,
            last_probe_bitrate_bps=telemetry.last_probe_bitrate_bps,
        )

    def write_inter_arrival_sample(self, sample: Any) -> None:
        self.write_event(
            "inter-arrival",
            trace="transport-cc",
            regime="twcc-gcc",
            trace_role="sender",
            direction="local-estimator",
            feedback_time_us=sample.feedback_time_us,
            previous_first_send_time_us=sample.previous_first_send_time_us,
            previous_last_send_time_us=sample.previous_last_send_time_us,
            previous_first_receive_time_us=sample.previous_first_receive_time_us,
            previous_last_receive_time_us=sample.previous_last_receive_time_us,
            current_first_send_time_us=sample.current_first_send_time_us,
            current_last_send_time_us=sample.current_last_send_time_us,
            current_first_receive_time_us=sample.current_first_receive_time_us,
            current_last_receive_time_us=sample.current_last_receive_time_us,
            group_bytes=sample.group_bytes,
            send_delta_us=sample.send_delta_us,
            receive_delta_us=sample.receive_delta_us,
            delay_delta_ms=sample.delay_delta_ms,
            trend_ms=sample.trend_ms,
            raw_trend=sample.raw_trend,
            threshold_ms=sample.threshold_ms,
            accumulated_delay_ms=sample.accumulated_delay_ms,
            smoothed_delay_ms=sample.smoothed_delay_ms,
            trend_window_ms=sample.trend_window_ms,
            overuse_counter=sample.overuse_counter,
            overuse_time_us=sample.overuse_time_us,
            delay_usage=sample.delay_usage,
            groups_seen=sample.groups_seen,
        )

    def write_remb_observation(self, telemetry: Any) -> None:
        self.write_event(
            "observation",
            trace="remb",
            regime="remb",
            trace_role="receiver",
            direction="inbound-rtp",
            arrival_time_ms=telemetry.arrival_time_ms,
            send_time_ms=telemetry.send_time_ms,
            abs_send_time=telemetry.abs_send_time,
            payload_size=telemetry.payload_size,
            ssrc=telemetry.ssrc,
            active_ssrcs=list(telemetry.active_ssrcs),
            incoming_bitrate_bps=telemetry.incoming_bitrate_bps,
            target_bitrate_bps=telemetry.target_bitrate_bps,
            valid_estimate=telemetry.valid_estimate,
            detector_state=telemetry.detector_state,
            aimd_state=telemetry.aimd_state,
            estimator_offset_ms=telemetry.estimator_offset_ms,
            estimator_slope=telemetry.estimator_slope,
            estimator_num_deltas=telemetry.estimator_num_deltas,
            detector_threshold_ms=telemetry.detector_threshold_ms,
            detector_overuse_counter=telemetry.detector_overuse_counter,
            detector_overuse_time_ms=telemetry.detector_overuse_time_ms,
            timestamp_delta_ms=telemetry.timestamp_delta_ms,
            arrival_time_delta_ms=telemetry.arrival_time_delta_ms,
            size_delta=telemetry.size_delta,
            probe_count=telemetry.probe_count,
            total_probes_received=telemetry.total_probes_received,
            last_probe_bitrate_bps=telemetry.last_probe_bitrate_bps,
            stream_count=telemetry.stream_count,
        )

    def write_remb_feedback(
        self, *, bitrate: int, ssrcs: list[int], telemetry: Any
    ) -> None:
        self.write_event(
            "feedback",
            trace="remb",
            regime="remb",
            trace_role="receiver",
            direction="outbound-rtcp",
            arrival_time_ms=telemetry.arrival_time_ms,
            bitrate_bps=bitrate,
            ssrcs=ssrcs,
            reason=telemetry.update_reason,
            incoming_bitrate_bps=telemetry.incoming_bitrate_bps,
            detector_state=telemetry.detector_state,
            aimd_state=telemetry.aimd_state,
            estimator_offset_ms=telemetry.estimator_offset_ms,
            estimator_num_deltas=telemetry.estimator_num_deltas,
            detector_threshold_ms=telemetry.detector_threshold_ms,
            detector_overuse_counter=telemetry.detector_overuse_counter,
            detector_overuse_time_ms=telemetry.detector_overuse_time_ms,
            timestamp_delta_ms=telemetry.timestamp_delta_ms,
            arrival_time_delta_ms=telemetry.arrival_time_delta_ms,
            size_delta=telemetry.size_delta,
            probe_count=telemetry.probe_count,
            total_probes_received=telemetry.total_probes_received,
            last_probe_bitrate_bps=telemetry.last_probe_bitrate_bps,
            active_ssrcs=list(telemetry.active_ssrcs),
        )

    def write_remb_receiver_estimate(
        self,
        *,
        bitrate: int,
        ssrcs: list[int],
        now_ms: int,
        accepted: bool,
        reason: str,
        sender_count: int,
        allocations: list[dict[str, Any]] | None = None,
    ) -> None:
        self.write_event(
            "receiver-estimate",
            trace="remb",
            regime="remb",
            trace_role="sender",
            direction="inbound-rtcp",
            now_ms=now_ms,
            bitrate_bps=bitrate,
            ssrcs=ssrcs,
            accepted=accepted,
            reason=reason,
            sender_count=sender_count,
            allocations=allocations or [],
        )

    def write_event(self, event_type: str, **payload: Any) -> None:
        if self._closed:
            return
        payload["type"] = event_type
        payload.setdefault("schema_version", 1)
        payload.setdefault("trace", "unknown")
        payload.setdefault("regime", "unknown")
        payload.setdefault("trace_role", "unknown")
        payload.setdefault("direction", "unknown")
        self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
