from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Protocol

from pycc import RateConstraints

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
    PyccTransportControlProvider,
    TransportControlSentPacket,
)

logger = logging.getLogger(__name__)
_TELEMETRY_INTERVAL_MS = int(
    os.environ.get("AIORTC_TRANSPORT_CC_TELEMETRY_INTERVAL_MS", "10000")
)
_SIGNIFICANT_TARGET_DROP_RATIO = 0.25
_TRANSPORT_MIN_BITRATE = 1_500_000
_TRANSPORT_START_BITRATE = 1_500_000
_TRANSPORT_MAX_BITRATE = 9_000_000


class CongestionControlledSender(Protocol):
    _ssrc: int
    kind: str

    def _get_target_bitrate(self) -> Optional[int]: ...

    def _get_bitrate_bounds(self) -> tuple[int, int]: ...

    def _set_target_bitrate(self, bitrate: int) -> None: ...


class CongestionControlledReceiver(Protocol):
    kind: str

    def _get_rtcp_ssrc(self) -> Optional[int]: ...


@dataclass
class SenderState:
    sender: CongestionControlledSender
    min_bitrate: int
    max_bitrate: int
    weight: float
    allocated_bitrate: int
    applied_bitrate: Optional[int] = None


@dataclass
class TelemetrySnapshot:
    now_ms: int
    sent_bytes: int
    acknowledged_bytes: int
    lost_bytes: int


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
        self.__transport_control = PyccTransportControlProvider(
            RateConstraints(
                min_bitrate_bps=_TRANSPORT_MIN_BITRATE,
                start_bitrate_bps=_TRANSPORT_START_BITRATE,
                max_bitrate_bps=_TRANSPORT_MAX_BITRATE,
            )
        )
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

        initial = sender._get_target_bitrate() or 1_000_000
        min_bitrate, max_bitrate = sender._get_bitrate_bounds()
        initial = max(min_bitrate, min(initial, max_bitrate))
        self.__senders[sender._ssrc] = SenderState(
            sender=sender,
            min_bitrate=min_bitrate,
            max_bitrate=max_bitrate,
            weight=1.0,
            allocated_bitrate=initial,
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
        send_time_ms: int,
        size_bytes: int,
        ssrc: int,
        rtp_sequence_number: int,
        is_retransmission: bool = False,
    ) -> None:
        self.__transport_control.on_packet_sent(
            TransportControlSentPacket(
                transport_sequence_number=transport_sequence_number,
                send_time_ms=send_time_ms,
                size_bytes=size_bytes,
                ssrc=ssrc,
                rtp_sequence_number=rtp_sequence_number,
                is_retransmission=is_retransmission,
            )
        )

    def handle_transport_feedback(
        self, packet: RtcpTransportLayerCcPacket, now_ms: int
    ) -> None:
        update = self.__transport_control.handle_transport_feedback(
            packet.feedback, now_ms * 1000
        )
        if update is not None:
            previous_target = self.__session_target_bitrate
            self.__session_target_bitrate = update.target_bitrate_bps
            self.__recompute_allocation()
            self.__log_target_update(previous_target, update)
        self.__log_delay_usage_transition()
        self.__log_telemetry(now_ms)

    def get_pacer_config(self):
        return self.__transport_control.get_pacer_config()

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

        return sum(state.allocated_bitrate for state in self.__senders.values())

    def __recompute_allocation(self) -> None:
        if not self.__senders:
            return

        if self.__session_target_bitrate is None:
            for state in self.__senders.values():
                self.__apply_sender_target(state)
            return

        states = list(self.__senders.values())
        floor = sum(state.min_bitrate for state in states)
        ceiling = sum(state.max_bitrate for state in states)
        session = max(floor, min(self.__session_target_bitrate, ceiling))

        for state in states:
            state.allocated_bitrate = state.min_bitrate

        remaining = session - floor
        if remaining <= 0:
            for state in states:
                self.__apply_sender_target(state)
            return

        allocatable = {
            state.sender._ssrc: state.max_bitrate - state.min_bitrate
            for state in states
        }
        active_states = [
            state for state in states if allocatable[state.sender._ssrc] > 0
        ]

        while remaining > 0 and active_states:
            weight_total = sum(state.weight for state in active_states)
            if weight_total <= 0:
                break

            round_remaining = remaining
            grants: dict[int, int] = {}
            remainders: list[tuple[float, SenderState]] = []
            distributed = 0

            for state in active_states:
                room = allocatable[state.sender._ssrc]
                ideal = round_remaining * (state.weight / weight_total)
                grant = min(room, math.floor(ideal))
                grants[state.sender._ssrc] = grant
                distributed += grant
                remainders.append((ideal - math.floor(ideal), state))

            leftover = round_remaining - distributed
            if leftover > 0:
                for _, state in sorted(
                    remainders, key=lambda item: item[0], reverse=True
                ):
                    room = allocatable[state.sender._ssrc]
                    grant = grants[state.sender._ssrc]
                    if room <= grant:
                        continue
                    grants[state.sender._ssrc] += 1
                    distributed += 1
                    leftover -= 1
                    if leftover <= 0:
                        break

            if distributed <= 0:
                for state in active_states:
                    room = allocatable[state.sender._ssrc]
                    if room <= 0:
                        continue
                    grants[state.sender._ssrc] = grants.get(state.sender._ssrc, 0) + 1
                    distributed += 1
                    leftover -= 1
                    if distributed >= remaining:
                        break

            if distributed <= 0:
                break

            for state in active_states:
                grant = grants.get(state.sender._ssrc, 0)
                if grant <= 0:
                    continue
                state.allocated_bitrate += grant
                allocatable[state.sender._ssrc] -= grant

            remaining -= distributed
            active_states = [
                state for state in active_states if allocatable[state.sender._ssrc] > 0
            ]

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
                "loss=%.3f loss_sample=%.3f rtt_ms=%.1f "
                "trend_ms=%.3f threshold_ms=%.3f send_delta_ms=%.3f "
                "recv_delta_ms=%.3f delay_delta_ms=%.3f group_bytes=%d",
                update.target_bitrate_bps,
                update.stable_target_bitrate_bps,
                previous,
                update.reason,
                telemetry.delay_usage,
                telemetry.aimd_state,
                telemetry.acked_bitrate_bps,
                update.loss_fraction,
                update.rtt_us / 1000,
                telemetry.trend_ms,
                telemetry.trend_threshold_ms,
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
            "loss=%.3f loss_sample=%.3f rtt_ms=%.1f "
            "trend_ms=%.3f threshold_ms=%.3f overuse_count=%d "
            "overuse_time_ms=%.3f send_delta_ms=%.3f recv_delta_ms=%.3f "
            "delay_delta_ms=%.3f group_bytes=%d",
            update.target_bitrate_bps,
            update.stable_target_bitrate_bps,
            previous,
            update.reason,
            telemetry.delay_usage,
            telemetry.aimd_state,
            telemetry.acked_bitrate_bps,
            update.loss_fraction,
            telemetry.loss_sample,
            update.rtt_us / 1000,
            telemetry.trend_ms,
            telemetry.trend_threshold_ms,
            telemetry.overuse_counter,
            telemetry.overuse_time_ms,
            telemetry.last_send_delta_ms,
            telemetry.last_receive_delta_ms,
            telemetry.last_delay_delta_ms,
            telemetry.last_group_bytes,
        )
        self.__last_logged_target_bitrate = update.target_bitrate_bps

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
            "transport-cc delay state %s -> %s trend_ms=%.3f threshold_ms=%.3f "
            "overuse_count=%d overuse_time_ms=%.3f acked_bps=%d "
            "send_delta_ms=%.3f recv_delta_ms=%.3f delay_delta_ms=%.3f "
            "group_bytes=%d groups=%d",
            previous or "unknown",
            usage,
            telemetry.trend_ms,
            telemetry.trend_threshold_ms,
            telemetry.overuse_counter,
            telemetry.overuse_time_ms,
            telemetry.acked_bitrate_bps,
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
        allocation_floor = sum(state.min_bitrate for state in states)
        allocation_ceiling = sum(state.max_bitrate for state in states)
        allocated_total = sum(state.allocated_bitrate for state in states)
        applied_total = sum(
            state.applied_bitrate or 0 for state in states
        )
        encoder_total = sum(
            state.sender._get_target_bitrate() or 0 for state in states
        )
        floor_limited = (
            self.__session_target_bitrate is not None
            and self.__session_target_bitrate < allocation_floor
        )

        if self.__last_telemetry_snapshot is None:
            elapsed_s = 0.0
            sent_rate_bps = 0
            acked_rate_bps = 0
            lost_rate_bps = 0
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
        self.__last_telemetry_snapshot = TelemetrySnapshot(
            now_ms=now_ms,
            sent_bytes=telemetry.sent_bytes,
            acknowledged_bytes=telemetry.acknowledged_bytes,
            lost_bytes=telemetry.lost_bytes,
        )

        sender_states = []
        for state in states:
            sender_states.append(
                "%d:alloc=%d applied=%s encoder=%s"
                % (
                    state.sender._ssrc,
                    state.allocated_bitrate,
                    state.applied_bitrate,
                    state.sender._get_target_bitrate(),
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
            "target_bps=%d pacer_bps=%d floor_bps=%d ceiling_bps=%d "
            "allocated_bps=%d applied_bps=%d encoder_bps=%d "
            "floor_limited=%s sent_bps=%d acked_bps=%d lost_bps=%d "
            "in_flight=%d oldest_in_flight_ms=%d history=%d "
            "twcc_next=%d fb_base=%d fb_count=%d delay_usage=%s aimd=%s "
            "acked_estimate_bps=%d loss_sample=%.3f loss_avg=%.3f "
            "trend_ms=%.3f threshold_ms=%.3f overuse_count=%d "
            "overuse_time_ms=%.3f groups=%d group_bytes=%d "
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
            allocation_floor,
            allocation_ceiling,
            allocated_total,
            applied_total,
            encoder_total,
            floor_limited,
            sent_rate_bps,
            acked_rate_bps,
            lost_rate_bps,
            telemetry.data_in_flight_bytes,
            telemetry.oldest_in_flight_age_ms,
            telemetry.packet_history_size,
            telemetry.next_transport_sequence_number,
            telemetry.last_feedback_base_sequence_number,
            telemetry.last_feedback_packet_count,
            telemetry.delay_usage,
            telemetry.aimd_state,
            telemetry.acked_bitrate_bps,
            telemetry.loss_sample,
            telemetry.loss_average,
            telemetry.trend_ms,
            telemetry.trend_threshold_ms,
            telemetry.overuse_counter,
            telemetry.overuse_time_ms,
            telemetry.groups_seen,
            telemetry.last_group_bytes,
            telemetry.last_send_delta_ms,
            telemetry.last_receive_delta_ms,
            telemetry.last_delay_delta_ms,
            "; ".join(sender_states),
        )
