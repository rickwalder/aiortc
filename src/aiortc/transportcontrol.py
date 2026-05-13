from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from pycc import (
    RTPFB_TRANSPORT_CC_FMT,
    TRANSPORT_CC_URI,
    GoogCcController,
    InterArrivalSample,
    LeakyBucketPacerModel,
    PacedPacketInfo,
    PacerConfig,
    RateConstraints,
    SentPacket,
    TargetRateUpdate,
    TransportLayerCcPacket,
    TwccRecorder,
    uint16_add,
)

from . import clock
from .rtcrtpparameters import RTCRtcpFeedback, RTCRtpHeaderExtensionParameters
from .transporttrace import TransportCcTraceWriter

RTCP_RTPFB = 205
TRANSPORT_CC_HEADER_EXTENSION_ID = 5


@dataclass(frozen=True)
class TransportControlCapabilities:
    rtcp_feedback: list[RTCRtcpFeedback]
    rtp_header_extensions: list[RTCRtpHeaderExtensionParameters]
    rtcp_feedback_formats: list[tuple[int, int]]


@dataclass(frozen=True)
class TransportControlSentPacket:
    transport_sequence_number: int
    send_time_us: int
    size_bytes: int
    ssrc: int
    rtp_sequence_number: int
    payload_size_bytes: int = 0
    is_retransmission: bool = False
    pacing_info: PacedPacketInfo | None = None


@dataclass(frozen=True)
class TransportControlTelemetry:
    feedback_count: int = 0
    packet_count: int = 0
    received_count: int = 0
    lost_count: int = 0
    first_time_lost_count: int = 0
    recovered_count: int = 0
    sent_packet_count: int = 0
    sent_bytes: int = 0
    acknowledged_bytes: int = 0
    lost_bytes: int = 0
    prior_unacked_bytes: int = 0
    data_in_flight_bytes: int = 0
    pacing_queue_bytes: int = 0
    pacing_queue_oldest_age_ms: int = 0
    oldest_in_flight_age_ms: int = 0
    packet_history_size: int = 0
    next_transport_sequence_number: int = 0
    last_feedback_base_sequence_number: int = 0
    last_feedback_packet_count: int = 0
    last_feedback_time_us: int = 0
    last_target_bitrate_bps: int = 0
    last_update_reason: str = ""
    last_loss_fraction: float = 0.0
    last_rtt_us: int = 0
    delay_usage: str = "normal"
    aimd_state: str = "increase"
    acked_bitrate_bps: int = 0
    in_alr: bool = False
    alr_budget_ratio: float = 0.0
    link_capacity_bps: int = 0
    link_capacity_lower_bps: int = 0
    link_capacity_upper_bps: int = 0
    loss_sample: float = 0.0
    loss_average: float = 0.0
    trend_ms: float = 0.0
    raw_trend: float = 0.0
    accumulated_delay_ms: float = 0.0
    smoothed_delay_ms: float = 0.0
    trend_window_ms: float = 0.0
    trend_threshold_ms: float = 0.0
    overuse_counter: int = 0
    overuse_time_ms: float = 0.0
    groups_seen: int = 0
    last_group_bytes: int = 0
    last_send_delta_ms: float = 0.0
    last_receive_delta_ms: float = 0.0
    last_delay_delta_ms: float = 0.0
    pre_pushback_target_bitrate_bps: int = 0
    pushback_target_bitrate_bps: int = 0
    congestion_window_bytes: int = 0
    congestion_window_fill_ratio: float = 0.0
    pushback_encoding_rate_ratio: float = 1.0
    probe_cluster_id: int = -1
    probe_target_bitrate_bps: int = 0
    last_probe_bitrate_bps: int = 0


def get_transport_control_capabilities(kind: str) -> TransportControlCapabilities:
    if kind != "video":
        return TransportControlCapabilities([], [], [])

    return TransportControlCapabilities(
        rtcp_feedback=[RTCRtcpFeedback(type="transport-cc")],
        rtp_header_extensions=[
            RTCRtpHeaderExtensionParameters(
                id=TRANSPORT_CC_HEADER_EXTENSION_ID,
                uri=TRANSPORT_CC_URI,
            )
        ],
        rtcp_feedback_formats=[(RTCP_RTPFB, RTPFB_TRANSPORT_CC_FMT)],
    )


class PyccTransportControlProvider:
    def __init__(
        self,
        constraints: RateConstraints | None = None,
        trace_writer: TransportCcTraceWriter | None = None,
    ) -> None:
        self._transport_sequence_number = 0
        self._trace_writer = trace_writer
        self._gcc = GoogCcController(
            constraints,
            trace_observer=self._write_inter_arrival_sample,
        )
        self._twcc_recorder: Optional[TwccRecorder] = None
        self._twcc_feedback_ssrc: Optional[int] = None
        self._active = False
        self._feedback_count = 0
        self._packet_count = 0
        self._received_count = 0
        self._lost_count = 0
        self._first_time_lost_count = 0
        self._recovered_count = 0
        self._sent_packet_count = 0
        self._sent_bytes = 0
        self._acknowledged_bytes = 0
        self._lost_bytes = 0
        self._prior_unacked_bytes = 0
        self._pacing_queue_bytes = 0
        self._pacing_queue_oldest_age_ms = 0
        self._last_feedback_time_us = 0
        self._last_feedback_base_sequence_number = 0
        self._last_feedback_packet_count = 0
        self._last_update: TargetRateUpdate | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def packet_history(self):
        return self._gcc.packet_history

    def on_round_trip_time(self, rtt_us: int) -> None:
        self._gcc.on_round_trip_time(rtt_us)

    def next_transport_sequence_number(self) -> int:
        sequence_number = self._transport_sequence_number
        self._transport_sequence_number = uint16_add(
            self._transport_sequence_number, 1
        )
        return sequence_number

    def on_packet_sent(self, packet: TransportControlSentPacket) -> None:
        self._sent_packet_count += 1
        self._sent_bytes += packet.size_bytes
        self._gcc.on_packet_sent(
            SentPacket(
                transport_sequence_number=packet.transport_sequence_number,
                send_time_us=packet.send_time_us,
                size_bytes=packet.size_bytes,
                ssrc=packet.ssrc,
                rtp_sequence_number=packet.rtp_sequence_number,
                is_retransmission=packet.is_retransmission,
                pacing_info=(
                    packet.pacing_info
                    if isinstance(packet.pacing_info, PacedPacketInfo)
                    else PacedPacketInfo()
                ),
            )
        )
        if self._trace_writer is not None:
            self._trace_writer.write_sent(packet)

    def observe_incoming_rtp(
        self,
        *,
        media_ssrc: int,
        transport_sequence_number: int,
        arrival_time_us: int,
        feedback_ssrc: int,
    ) -> list[TransportLayerCcPacket]:
        if (
            self._twcc_recorder is None
            or self._twcc_feedback_ssrc != feedback_ssrc
        ):
            self._twcc_recorder = TwccRecorder(sender_ssrc=feedback_ssrc)
            self._twcc_feedback_ssrc = feedback_ssrc

        self._twcc_recorder.record_packet(
            media_ssrc=media_ssrc,
            transport_sequence_number=transport_sequence_number,
            arrival_time_us=arrival_time_us,
        )
        self._active = True
        return self._twcc_recorder.build_feedback(arrival_time_us)

    def handle_transport_feedback(
        self, feedback: TransportLayerCcPacket, feedback_time_us: int
    ) -> TargetRateUpdate | None:
        normalized = self._gcc.packet_history.on_transport_feedback(
            feedback, feedback_time_us
        )
        if self._trace_writer is not None:
            self._trace_writer.write_feedback(feedback=feedback, normalized=normalized)
        received = normalized.received_with_send_info()
        lost = normalized.lost_with_send_info()
        self._feedback_count += 1
        self._packet_count += len(normalized.packet_results)
        self._received_count += len(received)
        self._lost_count += len(lost)
        self._acknowledged_bytes += sum(
            result.sent_packet.size_bytes for result in received
        )
        self._prior_unacked_bytes += sum(
            result.sent_packet.prior_unacked_data_bytes for result in received
        )
        self._lost_bytes += sum(result.sent_packet.size_bytes for result in lost)
        self._first_time_lost_count += sum(
            1 for result in lost if result.reported_lost_for_first_time
        )
        self._recovered_count += sum(
            1
            for result in received
            if result.reported_recovered_for_first_time
        )
        self._last_feedback_time_us = feedback_time_us
        self._last_feedback_base_sequence_number = feedback.base_sequence_number
        self._last_feedback_packet_count = len(feedback.packets)

        update = self._gcc.on_transport_feedback(normalized)
        self._active = True
        if update is not None:
            self._last_update = update
        return update

    def get_pacer_config(self) -> PacerConfig:
        return self._gcc.get_pacer_config()

    def get_target_bitrate(self) -> int:
        return self._gcc.get_target_bitrate()

    def update_pacing_queue(
        self, queue_bytes: int, oldest_queue_age_ms: int = 0
    ) -> None:
        self._pacing_queue_bytes = max(0, queue_bytes)
        self._pacing_queue_oldest_age_ms = max(0, oldest_queue_age_ms)
        self._gcc.update_pacing_queue(self._pacing_queue_bytes)

    def _write_inter_arrival_sample(self, sample: InterArrivalSample) -> None:
        if self._trace_writer is not None:
            self._trace_writer.write_inter_arrival_sample(sample)

    def get_telemetry(self) -> TransportControlTelemetry:
        update = self._last_update
        packet_history = self._gcc.packet_history
        diagnostics = self._gcc.get_diagnostics()
        oldest_in_flight_send_time_us = packet_history.oldest_in_flight_send_time_us
        if oldest_in_flight_send_time_us is None:
            oldest_in_flight_age_ms = 0
        else:
            oldest_in_flight_age_ms = max(
                0,
                int(
                    (
                        self._last_feedback_time_us
                        - oldest_in_flight_send_time_us
                    )
                    / 1000
                ),
            )
        return TransportControlTelemetry(
            feedback_count=self._feedback_count,
            packet_count=self._packet_count,
            received_count=self._received_count,
            lost_count=self._lost_count,
            first_time_lost_count=self._first_time_lost_count,
            recovered_count=self._recovered_count,
            sent_packet_count=self._sent_packet_count,
            sent_bytes=self._sent_bytes,
            acknowledged_bytes=self._acknowledged_bytes,
            lost_bytes=self._lost_bytes,
            prior_unacked_bytes=self._prior_unacked_bytes,
            data_in_flight_bytes=packet_history.data_in_flight_bytes,
            pacing_queue_bytes=self._pacing_queue_bytes,
            pacing_queue_oldest_age_ms=self._pacing_queue_oldest_age_ms,
            oldest_in_flight_age_ms=oldest_in_flight_age_ms,
            packet_history_size=packet_history.history_size,
            next_transport_sequence_number=self._transport_sequence_number,
            last_feedback_base_sequence_number=self._last_feedback_base_sequence_number,
            last_feedback_packet_count=self._last_feedback_packet_count,
            last_feedback_time_us=self._last_feedback_time_us,
            last_target_bitrate_bps=self._gcc.get_target_bitrate(),
            last_update_reason=update.reason if update is not None else "",
            last_loss_fraction=update.loss_fraction if update is not None else 0.0,
            last_rtt_us=update.rtt_us if update is not None else 0,
            delay_usage=diagnostics.delay_usage,
            aimd_state=diagnostics.aimd_state,
            acked_bitrate_bps=diagnostics.acked_bitrate_bps or 0,
            in_alr=diagnostics.in_alr,
            alr_budget_ratio=diagnostics.alr_budget_ratio,
            link_capacity_bps=diagnostics.link_capacity_bps or 0,
            link_capacity_lower_bps=diagnostics.link_capacity_lower_bps or 0,
            link_capacity_upper_bps=diagnostics.link_capacity_upper_bps or 0,
            loss_sample=diagnostics.loss_sample,
            loss_average=diagnostics.loss_average,
            trend_ms=diagnostics.trend_ms,
            raw_trend=diagnostics.raw_trend,
            accumulated_delay_ms=diagnostics.accumulated_delay_ms,
            smoothed_delay_ms=diagnostics.smoothed_delay_ms,
            trend_window_ms=diagnostics.trend_window_ms,
            trend_threshold_ms=diagnostics.trend_threshold_ms,
            overuse_counter=diagnostics.overuse_counter,
            overuse_time_ms=diagnostics.overuse_time_us / 1000,
            groups_seen=diagnostics.groups_seen,
            last_group_bytes=diagnostics.last_group_bytes,
            last_send_delta_ms=diagnostics.last_send_delta_us / 1000,
            last_receive_delta_ms=diagnostics.last_receive_delta_us / 1000,
            last_delay_delta_ms=diagnostics.last_delay_delta_ms,
            pre_pushback_target_bitrate_bps=(
                diagnostics.pre_pushback_target_bitrate_bps
            ),
            pushback_target_bitrate_bps=diagnostics.pushback_target_bitrate_bps,
            congestion_window_bytes=diagnostics.congestion_window_bytes,
            congestion_window_fill_ratio=diagnostics.congestion_window_fill_ratio,
            pushback_encoding_rate_ratio=diagnostics.pushback_encoding_rate_ratio,
            probe_cluster_id=diagnostics.probe_cluster_id,
            probe_target_bitrate_bps=diagnostics.probe_target_bitrate_bps,
            last_probe_bitrate_bps=diagnostics.last_probe_bitrate_bps or 0,
        )


class AsyncRtpPacer:
    def __init__(self) -> None:
        self._model: LeakyBucketPacerModel | None = None
        self._lock = asyncio.Lock()
        self._active_probe_cluster_id: int | None = None
        self._completed_probe_cluster_ids: set[int] = set()
        self._probe_sent_bytes = 0
        self._probe_sent_packets = 0
        self._last_probe_send_time_us: int | None = None

    async def pace(
        self,
        *,
        size_bytes: int,
        config: PacerConfig,
        now_ms: int | None = None,
    ) -> PacedPacketInfo:
        if size_bytes <= 0 or config.send_bitrate_bps <= 0:
            return PacedPacketInfo()

        async with self._lock:
            pacing_info = self._pacing_info_for_packet(config)
            use_current_clock = now_ms is None
            now_us = (
                clock.current_monotonic_us()
                if use_current_clock
                else now_ms * 1000
            )
            if self._model is None:
                self._model = LeakyBucketPacerModel(config, now_us)
            else:
                self._model.set_config(config, now_us)

            wait_us = max(
                self._wait_time_us(size_bytes, now_us),
                self._probe_wait_time_us(pacing_info, now_us),
            )
            if wait_us > 0:
                await asyncio.sleep(wait_us / 1_000_000)
                if use_current_clock:
                    now_us = clock.current_monotonic_us()
                else:
                    now_us += wait_us

            self._model.on_packet_sent(size_bytes, now_us)
            if pacing_info.is_probe:
                self._probe_sent_packets += 1
                self._probe_sent_bytes += size_bytes
                self._last_probe_send_time_us = now_us
                if (
                    self._probe_sent_packets >= pacing_info.probe_cluster_min_probes
                    and self._probe_sent_bytes >= pacing_info.probe_cluster_min_bytes
                ):
                    self._completed_probe_cluster_ids.add(
                        pacing_info.probe_cluster_id
                    )
            return pacing_info

    def is_probe_pending(self, config: PacerConfig) -> bool:
        probe = config.probe_cluster
        return probe is not None and probe.id not in self._completed_probe_cluster_ids

    def _wait_time_us(self, size_bytes: int, now_us: int) -> int:
        assert self._model is not None
        self._model.update(now_us)
        if self._model.can_send(size_bytes, now_us):
            return 0

        deficit_bytes = size_bytes - self._model.budget_bytes
        if self._model.config.effective_send_bitrate_bps <= 0:
            return 0
        return max(
            0,
            int(
                deficit_bytes
                * 8_000_000
                / self._model.config.effective_send_bitrate_bps
            ),
        )

    def _probe_wait_time_us(
        self, pacing_info: PacedPacketInfo, now_us: int
    ) -> int:
        if (
            not pacing_info.is_probe
            or self._last_probe_send_time_us is None
            or pacing_info.probe_cluster_min_delta_us <= 0
        ):
            return 0

        return max(
            0,
            self._last_probe_send_time_us
            + pacing_info.probe_cluster_min_delta_us
            - now_us,
        )

    def _pacing_info_for_packet(self, config: PacerConfig) -> PacedPacketInfo:
        probe = config.probe_cluster
        if probe is None or probe.id in self._completed_probe_cluster_ids:
            return PacedPacketInfo()
        if self._active_probe_cluster_id != probe.id:
            self._active_probe_cluster_id = probe.id
            self._probe_sent_bytes = 0
            self._probe_sent_packets = 0
            self._last_probe_send_time_us = None
        return probe.pacing_info
