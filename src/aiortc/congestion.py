from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

from .rate import RemoteBitrateEstimator
from .rtp import (
    RTCP_PSFB_APP,
    AnyRtcpPacket,
    RtcpPsfbPacket,
    RtcpTransportLayerCcPacket,
    RtpPacket,
    pack_remb_fci,
)
from .transportcontrol import (
    AsyncRtpPacer,
    PacedPacketInfo,
    PyccTransportControlProvider,
    TransportControlSentPacket,
)
from .transporttrace import TransportCcTraceWriter

logger = logging.getLogger(__name__)
_TELEMETRY_INTERVAL_MS = int(
    os.environ.get("AIORTC_TRANSPORT_CC_TELEMETRY_INTERVAL_MS", "10000")
)
_SIGNIFICANT_TARGET_DROP_RATIO = 0.25


class CongestionControlledSender(Protocol):
    _ssrc: int
    kind: str

    def _get_target_bitrate(self) -> Optional[int]: ...

    def _set_target_bitrate(self, bitrate: int) -> None: ...


class CongestionControlledReceiver(Protocol):
    kind: str

    def _get_rtcp_ssrc(self) -> Optional[int]: ...


@dataclass
class SenderState:
    sender: CongestionControlledSender
    weight: float
    allocated_bitrate: int
    applied_bitrate: Optional[int] = None
    encoded_payload_bytes: int = 0


@dataclass
class TelemetrySnapshot:
    now_ms: int
    sent_bytes: int
    acknowledged_bytes: int
    lost_bytes: int
    prior_unacked_bytes: int
    encoded_payload_bytes_by_ssrc: dict[int, int]


class TransportCongestionController:
    """
    Lightweight transport-scoped overlay congestion controller.

    This first pass intentionally stays simple:
    - one controller per DTLS / bundled RTP transport
    - aggregate receiver REMB feedback at transport scope
    - allocate a single session target across active video senders
    - do not pace or drop packets yet
    """

    def __init__(self) -> None:
        self.__senders: dict[int, SenderState] = {}
        self.__estimates: dict[int, tuple[int, int]] = {}
        self.__receivers: set[CongestionControlledReceiver] = set()
        self.__remote_bitrate_estimator = RemoteBitrateEstimator()
        self.__session_target_bitrate: Optional[int] = None
        self.__trace_writer = TransportCcTraceWriter.from_environment()
        self.__transport_control = PyccTransportControlProvider(
            trace_writer=self.__trace_writer
        )
        self.__rtp_pacer = AsyncRtpPacer()
        self.__last_telemetry_ms: Optional[int] = None
        self.__last_telemetry_snapshot: Optional[TelemetrySnapshot] = None
        self.__last_logged_target_bitrate: Optional[int] = None
        self.__last_logged_delay_usage: Optional[str] = None

    def register_receiver(self, receiver: CongestionControlledReceiver) -> None:
        if getattr(receiver, "kind", None) == "video":
            self.__receivers.add(receiver)

    def unregister_receiver(self, receiver: CongestionControlledReceiver) -> None:
        self.__receivers.discard(receiver)

    def register_sender(self, sender: CongestionControlledSender) -> None:
        if sender.kind != "video":
            return

        self.__senders[sender._ssrc] = SenderState(
            sender=sender,
            weight=1.0,
            allocated_bitrate=0,
            applied_bitrate=None,
        )
        self.__recompute_allocation()

    def unregister_sender(self, sender: CongestionControlledSender) -> None:
        self.__senders.pop(sender._ssrc, None)
        self.__estimates.pop(sender._ssrc, None)
        self.__recompute_allocation()

    def update_receiver_estimate(
        self, bitrate: int, ssrcs: list[int], now_ms: int
    ) -> None:
        if self.__transport_control.active:
            return
        if not self.__senders:
            return

        for ssrc in ssrcs:
            if ssrc in self.__senders:
                self.__estimates[ssrc] = (bitrate, now_ms)

        target = self.__derive_session_target(now_ms)
        if target is None:
            return

        self.__session_target_bitrate = target
        self.__recompute_allocation()

    def set_rtt(self, rtt_ms: int) -> None:
        if rtt_ms <= 0:
            return
        self.__remote_bitrate_estimator.rate_control.rtt = rtt_ms
        self.__transport_control.on_round_trip_time(rtt_ms * 1000)

    def next_transport_sequence_number(self) -> int:
        return self.__transport_control.next_transport_sequence_number()

    def on_packet_sent(
        self,
        *,
        transport_sequence_number: int,
        send_time_us: int,
        size_bytes: int,
        payload_size_bytes: int = 0,
        ssrc: int,
        rtp_sequence_number: int,
        is_retransmission: bool = False,
        pacing_info: Optional[PacedPacketInfo] = None,
    ) -> None:
        self.observe_encoded_frame(ssrc=ssrc, payload_bytes=payload_size_bytes)
        self.__transport_control.on_packet_sent(
            TransportControlSentPacket(
                transport_sequence_number=transport_sequence_number,
                send_time_us=send_time_us,
                size_bytes=size_bytes,
                payload_size_bytes=payload_size_bytes,
                ssrc=ssrc,
                rtp_sequence_number=rtp_sequence_number,
                is_retransmission=is_retransmission,
                pacing_info=pacing_info,
            )
        )

    def observe_encoded_frame(self, *, ssrc: int, payload_bytes: int) -> None:
        state = self.__senders.get(ssrc)
        if state is None or payload_bytes <= 0:
            return
        state.encoded_payload_bytes += payload_bytes

    def handle_transport_feedback(
        self, packet: RtcpTransportLayerCcPacket, now_us: int
    ) -> None:
        previous_target = self.__transport_control.get_target_bitrate()
        update = self.__transport_control.handle_transport_feedback(
            packet.feedback, now_us
        )
        if update is not None:
            self.__recompute_allocation()
            self.__log_target_update(previous_target, update)
            self.__trace_target_update(previous_target, update)
        self.__log_delay_usage_transition()
        self.__log_telemetry(now_us // 1000)

    def get_pacer_config(self):
        return self.__transport_control.get_pacer_config()

    async def pace_rtp_packet(
        self, *, size_bytes: int, now_ms: Optional[int] = None
    ) -> PacedPacketInfo:
        return await self.__rtp_pacer.pace(
            size_bytes=size_bytes,
            config=self.__transport_control.get_pacer_config(),
            now_ms=now_ms,
        )

    def observe_incoming_rtp(
        self,
        receiver: CongestionControlledReceiver,
        packet: RtpPacket,
        arrival_time_ms: int,
    ) -> list[AnyRtcpPacket]:
        feedback: list[AnyRtcpPacket] = []
        if packet.extensions.transport_sequence_number is not None:
            feedback.extend(
                self.__observe_incoming_transport_cc(receiver, packet, arrival_time_ms)
            )

        if (
            getattr(receiver, "kind", None) != "video"
            or packet.extensions.abs_send_time is None
        ):
            return feedback
        if self.__transport_control.active:
            return feedback

        remb = self.__remote_bitrate_estimator.add(
            abs_send_time=packet.extensions.abs_send_time,
            arrival_time_ms=arrival_time_ms,
            payload_size=len(packet.payload) + packet.padding_size,
            ssrc=packet.ssrc,
        )
        if remb is None:
            return feedback

        feedback_ssrc = self.__get_feedback_ssrc()
        if feedback_ssrc is None:
            return feedback

        bitrate, ssrcs = remb

        feedback.append(
            RtcpPsfbPacket(
                fmt=RTCP_PSFB_APP,
                ssrc=feedback_ssrc,
                media_ssrc=0,
                fci=pack_remb_fci(bitrate, ssrcs),
            )
        )
        return feedback

    def __observe_incoming_transport_cc(
        self,
        receiver: CongestionControlledReceiver,
        packet: RtpPacket,
        arrival_time_ms: int,
    ) -> list[RtcpTransportLayerCcPacket]:
        feedback_ssrc = self.__get_feedback_ssrc()
        if feedback_ssrc is None or packet.extensions.transport_sequence_number is None:
            return []

        feedback_packets = self.__transport_control.observe_incoming_rtp(
            media_ssrc=packet.ssrc,
            transport_sequence_number=packet.extensions.transport_sequence_number,
            arrival_time_us=arrival_time_ms * 1000,
            feedback_ssrc=feedback_ssrc,
        )
        return [
            RtcpTransportLayerCcPacket(feedback=feedback)
            for feedback in feedback_packets
        ]

    def __derive_session_target(self, now_ms: int) -> Optional[int]:
        active_ssrcs = list(self.__senders.keys())
        if not active_ssrcs:
            return None

        fresh = []
        for ssrc in active_ssrcs:
            sample = self.__estimates.get(ssrc)
            if sample is None:
                continue
            bitrate, sample_ms = sample
            if now_ms - sample_ms <= 1500:
                fresh.append(bitrate)

        if fresh:
            return min(fresh)

        if self.__session_target_bitrate is not None:
            return self.__session_target_bitrate

        return self.__transport_control.get_target_bitrate()

    def __current_session_target(self) -> int:
        if (
            not self.__transport_control.active
            and self.__session_target_bitrate is not None
        ):
            return self.__session_target_bitrate
        return self.__transport_control.get_target_bitrate()

    def __recompute_allocation(self) -> None:
        if not self.__senders:
            return

        states = list(self.__senders.values())
        session = self.__current_session_target()
        base_allocation = session // len(states)
        remainder = session % len(states)
        for index, state in enumerate(states):
            state.allocated_bitrate = base_allocation + (1 if index < remainder else 0)

        for state in states:
            self.__apply_sender_target(state)

    def __apply_sender_target(self, state: SenderState) -> None:
        bitrate = int(state.allocated_bitrate)
        if state.applied_bitrate == bitrate:
            return
        state.sender._set_target_bitrate(bitrate)
        state.applied_bitrate = bitrate

    def __get_feedback_ssrc(self) -> Optional[int]:
        for receiver in self.__receivers:
            ssrc = receiver._get_rtcp_ssrc()
            if ssrc is not None:
                return ssrc
        return None

    def __log_target_update(self, previous_target, update) -> None:
        telemetry = self.__transport_control.get_telemetry()
        previous = (
            previous_target
            if previous_target is not None
            else self.__last_logged_target_bitrate
        )
        if update.reason == "increase":
            logger.debug(
                "transport-cc target update target_bps=%d stable_bps=%d "
                "previous_bps=%s reason=%s delay_usage=%s aimd=%s acked_bps=%d "
                "in_alr=%s alr_budget=%.2f link_capacity_bps=%d "
                "pre_pushback_bps=%d pushback_bps=%d cwnd=%d cwnd_fill=%.2f "
                "pushback_ratio=%.2f probe_id=%d probe_bps=%d "
                "probe_estimate_bps=%d "
                "loss=%.3f loss_sample=%.3f rtt_ms=%.1f "
                "trend_ms=%.3f raw_trend=%.6f threshold_ms=%.3f "
                "acc_delay_ms=%.3f smooth_delay_ms=%.3f trend_window_ms=%.3f "
                "send_delta_ms=%.3f recv_delta_ms=%.3f delay_delta_ms=%.3f "
                "group_bytes=%d",
                update.target_bitrate_bps,
                update.stable_target_bitrate_bps,
                previous,
                update.reason,
                telemetry.delay_usage,
                telemetry.aimd_state,
                telemetry.acked_bitrate_bps,
                telemetry.in_alr,
                telemetry.alr_budget_ratio,
                telemetry.link_capacity_bps,
                telemetry.pre_pushback_target_bitrate_bps,
                telemetry.pushback_target_bitrate_bps,
                telemetry.congestion_window_bytes,
                telemetry.congestion_window_fill_ratio,
                telemetry.pushback_encoding_rate_ratio,
                telemetry.probe_cluster_id,
                telemetry.probe_target_bitrate_bps,
                telemetry.last_probe_bitrate_bps,
                update.loss_fraction,
                telemetry.loss_sample,
                update.rtt_us / 1000,
                telemetry.trend_ms,
                telemetry.raw_trend,
                telemetry.trend_threshold_ms,
                telemetry.accumulated_delay_ms,
                telemetry.smoothed_delay_ms,
                telemetry.trend_window_ms,
                telemetry.last_send_delta_ms,
                telemetry.last_receive_delta_ms,
                telemetry.last_delay_delta_ms,
                telemetry.last_group_bytes,
            )
            self.__last_logged_target_bitrate = update.target_bitrate_bps
            return

        level = logging.INFO
        if (
            previous is not None
            and previous > 0
            and update.target_bitrate_bps
            < previous * (1.0 - _SIGNIFICANT_TARGET_DROP_RATIO)
        ):
            level = logging.WARNING

        logger.log(
            level,
            "transport-cc target update target_bps=%d stable_bps=%d previous_bps=%s "
            "reason=%s delay_usage=%s aimd=%s acked_bps=%d "
            "in_alr=%s alr_budget=%.2f link_capacity_bps=%d "
            "pre_pushback_bps=%d pushback_bps=%d cwnd=%d cwnd_fill=%.2f "
            "pushback_ratio=%.2f probe_id=%d probe_bps=%d "
            "probe_estimate_bps=%d "
            "loss=%.3f loss_sample=%.3f rtt_ms=%.1f "
            "trend_ms=%.3f raw_trend=%.6f threshold_ms=%.3f "
            "acc_delay_ms=%.3f smooth_delay_ms=%.3f trend_window_ms=%.3f "
            "overuse_count=%d overuse_time_ms=%.3f send_delta_ms=%.3f "
            "recv_delta_ms=%.3f delay_delta_ms=%.3f group_bytes=%d",
            update.target_bitrate_bps,
            update.stable_target_bitrate_bps,
            previous,
            update.reason,
            telemetry.delay_usage,
            telemetry.aimd_state,
            telemetry.acked_bitrate_bps,
            telemetry.in_alr,
            telemetry.alr_budget_ratio,
            telemetry.link_capacity_bps,
            telemetry.pre_pushback_target_bitrate_bps,
            telemetry.pushback_target_bitrate_bps,
            telemetry.congestion_window_bytes,
            telemetry.congestion_window_fill_ratio,
            telemetry.pushback_encoding_rate_ratio,
            telemetry.probe_cluster_id,
            telemetry.probe_target_bitrate_bps,
            telemetry.last_probe_bitrate_bps,
            update.loss_fraction,
            telemetry.loss_sample,
            update.rtt_us / 1000,
            telemetry.trend_ms,
            telemetry.raw_trend,
            telemetry.trend_threshold_ms,
            telemetry.accumulated_delay_ms,
            telemetry.smoothed_delay_ms,
            telemetry.trend_window_ms,
            telemetry.overuse_counter,
            telemetry.overuse_time_ms,
            telemetry.last_send_delta_ms,
            telemetry.last_receive_delta_ms,
            telemetry.last_delay_delta_ms,
            telemetry.last_group_bytes,
        )
        self.__last_logged_target_bitrate = update.target_bitrate_bps

    def __trace_target_update(self, previous_target, update) -> None:
        if self.__trace_writer is None:
            return
        self.__trace_writer.write_target_update(
            previous_target=previous_target,
            update=update,
            telemetry=self.__transport_control.get_telemetry(),
        )

    def __log_delay_usage_transition(self) -> None:
        telemetry = self.__transport_control.get_telemetry()
        usage = telemetry.delay_usage
        previous = self.__last_logged_delay_usage
        if usage == previous:
            return
        self.__last_logged_delay_usage = usage
        level = logging.INFO if usage != "normal" else logging.DEBUG
        logger.log(
            level,
            "transport-cc delay state %s -> %s trend_ms=%.3f "
            "raw_trend=%.6f threshold_ms=%.3f "
            "acc_delay_ms=%.3f smooth_delay_ms=%.3f trend_window_ms=%.3f "
            "overuse_count=%d overuse_time_ms=%.3f acked_bps=%d "
            "in_alr=%s alr_budget=%.2f "
            "send_delta_ms=%.3f recv_delta_ms=%.3f delay_delta_ms=%.3f "
            "group_bytes=%d groups=%d",
            previous or "unknown",
            usage,
            telemetry.trend_ms,
            telemetry.raw_trend,
            telemetry.trend_threshold_ms,
            telemetry.accumulated_delay_ms,
            telemetry.smoothed_delay_ms,
            telemetry.trend_window_ms,
            telemetry.overuse_counter,
            telemetry.overuse_time_ms,
            telemetry.acked_bitrate_bps,
            telemetry.in_alr,
            telemetry.alr_budget_ratio,
            telemetry.last_send_delta_ms,
            telemetry.last_receive_delta_ms,
            telemetry.last_delay_delta_ms,
            telemetry.last_group_bytes,
            telemetry.groups_seen,
        )

    def __log_telemetry(self, now_ms: int) -> None:
        if (
            self.__last_telemetry_ms is not None
            and now_ms - self.__last_telemetry_ms < _TELEMETRY_INTERVAL_MS
        ):
            return
        self.__last_telemetry_ms = now_ms

        telemetry = self.__transport_control.get_telemetry()
        if telemetry.feedback_count <= 0:
            return

        pacer = self.__transport_control.get_pacer_config()
        states = list(self.__senders.values())
        allocated_total = sum(state.allocated_bitrate for state in states)
        applied_total = sum(
            state.applied_bitrate or 0 for state in states
        )
        encoder_total = sum(
            state.sender._get_target_bitrate() or 0 for state in states
        )

        if self.__last_telemetry_snapshot is None:
            elapsed_s = 0.0
            sent_rate_bps = 0
            acked_rate_bps = 0
            lost_rate_bps = 0
            prior_unacked_rate_bps = 0
            encoded_rate_bps_by_ssrc = {state.sender._ssrc: 0 for state in states}
        else:
            elapsed_ms = now_ms - self.__last_telemetry_snapshot.now_ms
            elapsed_s = max(0.001, elapsed_ms / 1000)
            sent_rate_bps = int(
                (telemetry.sent_bytes - self.__last_telemetry_snapshot.sent_bytes)
                * 8
                / elapsed_s
            )
            acked_rate_bps = int(
                (
                    telemetry.acknowledged_bytes
                    - self.__last_telemetry_snapshot.acknowledged_bytes
                )
                * 8
                / elapsed_s
            )
            lost_rate_bps = int(
                (telemetry.lost_bytes - self.__last_telemetry_snapshot.lost_bytes)
                * 8
                / elapsed_s
            )
            prior_unacked_rate_bps = int(
                (
                    telemetry.prior_unacked_bytes
                    - self.__last_telemetry_snapshot.prior_unacked_bytes
                )
                * 8
                / elapsed_s
            )
            encoded_rate_bps_by_ssrc = {}
            for state in states:
                ssrc = state.sender._ssrc
                previous_encoded_bytes = (
                    self.__last_telemetry_snapshot.encoded_payload_bytes_by_ssrc.get(
                        ssrc, 0
                    )
                )
                encoded_rate_bps_by_ssrc[ssrc] = int(
                    (state.encoded_payload_bytes - previous_encoded_bytes)
                    * 8
                    / elapsed_s
                )
        target_bitrate = max(1, telemetry.last_target_bitrate_bps)
        sent_fill_ratio = sent_rate_bps / target_bitrate
        acked_fill_ratio = acked_rate_bps / target_bitrate
        self.__last_telemetry_snapshot = TelemetrySnapshot(
            now_ms=now_ms,
            sent_bytes=telemetry.sent_bytes,
            acknowledged_bytes=telemetry.acknowledged_bytes,
            lost_bytes=telemetry.lost_bytes,
            prior_unacked_bytes=telemetry.prior_unacked_bytes,
            encoded_payload_bytes_by_ssrc={
                state.sender._ssrc: state.encoded_payload_bytes for state in states
            },
        )

        encoded_observed_bps = sum(encoded_rate_bps_by_ssrc.values())
        sender_states = []
        for state in states:
            sender_states.append(
                "%d:alloc=%d applied=%s encoder=%s observed=%d"
                % (
                    state.sender._ssrc,
                    state.allocated_bitrate,
                    state.applied_bitrate,
                    state.sender._get_target_bitrate(),
                    encoded_rate_bps_by_ssrc.get(state.sender._ssrc, 0),
                )
            )

        loss_fraction = (
            telemetry.first_time_lost_count / telemetry.packet_count
            if telemetry.packet_count
            else 0.0
        )
        logger.info(
            "transport-cc telemetry feedbacks=%d packets=%d received=%d "
            "lost=%d first_lost=%d recovered=%d loss=%.3f "
            "target_bps=%d pacer_bps=%d "
            "allocated_bps=%d applied_bps=%d encoder_bps=%d "
            "encoded_observed_bps=%d sent_bps=%d acked_bps=%d lost_bps=%d "
            "prior_unacked_bps=%d sent_fill=%.2f acked_fill=%.2f "
            "in_flight=%d oldest_in_flight_ms=%d history=%d "
            "twcc_next=%d fb_base=%d fb_count=%d delay_usage=%s aimd=%s "
            "acked_estimate_bps=%d in_alr=%s alr_budget=%.2f "
            "pre_pushback_bps=%d pushback_bps=%d cwnd=%d cwnd_fill=%.2f "
            "pushback_ratio=%.2f probe_id=%d probe_bps=%d "
            "probe_estimate_bps=%d "
            "link_capacity_bps=%d link_lower_bps=%d link_upper_bps=%d "
            "loss_sample=%.3f loss_avg=%.3f "
            "trend_ms=%.3f raw_trend=%.6f threshold_ms=%.3f "
            "acc_delay_ms=%.3f smooth_delay_ms=%.3f trend_window_ms=%.3f "
            "overuse_count=%d overuse_time_ms=%.3f groups=%d group_bytes=%d "
            "send_delta_ms=%.3f recv_delta_ms=%.3f delay_delta_ms=%.3f "
            "senders=[%s]",
            telemetry.feedback_count,
            telemetry.packet_count,
            telemetry.received_count,
            telemetry.lost_count,
            telemetry.first_time_lost_count,
            telemetry.recovered_count,
            loss_fraction,
            telemetry.last_target_bitrate_bps,
            pacer.send_bitrate_bps,
            allocated_total,
            applied_total,
            encoder_total,
            encoded_observed_bps,
            sent_rate_bps,
            acked_rate_bps,
            lost_rate_bps,
            prior_unacked_rate_bps,
            sent_fill_ratio,
            acked_fill_ratio,
            telemetry.data_in_flight_bytes,
            telemetry.oldest_in_flight_age_ms,
            telemetry.packet_history_size,
            telemetry.next_transport_sequence_number,
            telemetry.last_feedback_base_sequence_number,
            telemetry.last_feedback_packet_count,
            telemetry.delay_usage,
            telemetry.aimd_state,
            telemetry.acked_bitrate_bps,
            telemetry.in_alr,
            telemetry.alr_budget_ratio,
            telemetry.pre_pushback_target_bitrate_bps,
            telemetry.pushback_target_bitrate_bps,
            telemetry.congestion_window_bytes,
            telemetry.congestion_window_fill_ratio,
            telemetry.pushback_encoding_rate_ratio,
            telemetry.probe_cluster_id,
            telemetry.probe_target_bitrate_bps,
            telemetry.last_probe_bitrate_bps,
            telemetry.link_capacity_bps,
            telemetry.link_capacity_lower_bps,
            telemetry.link_capacity_upper_bps,
            telemetry.loss_sample,
            telemetry.loss_average,
            telemetry.trend_ms,
            telemetry.raw_trend,
            telemetry.trend_threshold_ms,
            telemetry.accumulated_delay_ms,
            telemetry.smoothed_delay_ms,
            telemetry.trend_window_ms,
            telemetry.overuse_counter,
            telemetry.overuse_time_ms,
            telemetry.groups_seen,
            telemetry.last_group_bytes,
            telemetry.last_send_delta_ms,
            telemetry.last_receive_delta_ms,
            telemetry.last_delay_delta_ms,
            "; ".join(sender_states),
        )
