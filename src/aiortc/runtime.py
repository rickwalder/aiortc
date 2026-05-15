from __future__ import annotations

import logging
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol

from rtc_types import (
    BitrateTarget,
    RoundTripTimeObserver,
    RtcpReceiveContext,
    RtcpReceiveObserver,
    RtpReceiveContext,
    RtpReceiveObserver,
    RtpSendContext,
    RtpSendDecision,
    RtpSendInterceptor,
    RtpSentContext,
    RtpSentObserver,
)

from . import clock
from .transporttrace import NetTraceWriter

logger = logging.getLogger(__name__)


def _get_telemetry_interval_ms() -> int:
    interval_ms = os.environ.get("AIORTC_TRANSPORT_CC_TELEMETRY_INTERVAL_MS")
    if interval_ms is not None:
        return int(interval_ms)

    interval_seconds = os.environ.get("AIORTC_TRANSPORT_CC_TELEMETRY_INTERVAL")
    if interval_seconds is not None:
        return int(float(interval_seconds) * 1000)

    return 10000


_TELEMETRY_INTERVAL_MS = _get_telemetry_interval_ms()
_SIGNIFICANT_TARGET_DROP_RATIO = 0.25
_RETRANSMISSION_RATE_LIMIT_WINDOW_MS = 500
_TARGET_APPLY_MIN_DELTA_BPS = 25_000
_TARGET_APPLY_MIN_DELTA_RATIO = 0.01


class BitrateControlledSender(Protocol):
    _ssrc: int
    kind: str

    def _get_target_bitrate(self) -> Optional[int]: ...

    def _set_target_bitrate(self, bitrate: int) -> None: ...


RttHandler = Callable[[int], None]


@dataclass
class SenderState:
    sender: BitrateControlledSender
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
    retransmission_sent_bytes: int
    retransmission_limited_bytes: int
    retransmission_limited_packets: int
    encoded_payload_bytes_by_ssrc: dict[int, int]


class SenderBitrateAllocator:
    def __init__(self, target_bitrate_provider: Callable[[], int]) -> None:
        self.__target_bitrate_provider = target_bitrate_provider
        self.__senders: dict[int, SenderState] = {}
        self.__estimates: dict[int, tuple[int, int]] = {}
        self.__session_target_bitrate: Optional[int] = None

    @property
    def sender_count(self) -> int:
        return len(self.__senders)

    @property
    def states(self) -> list[SenderState]:
        return list(self.__senders.values())

    @property
    def session_target_bitrate(self) -> int:
        return self.__current_session_target()

    def register_sender(self, sender: BitrateControlledSender) -> None:
        if sender.kind != "video":
            return

        self.__senders[sender._ssrc] = SenderState(
            sender=sender,
            weight=1.0,
            allocated_bitrate=0,
            applied_bitrate=None,
        )
        self.recompute_allocation()

    def unregister_sender(self, sender: BitrateControlledSender) -> None:
        self.__senders.pop(sender._ssrc, None)
        self.__estimates.pop(sender._ssrc, None)
        self.recompute_allocation()

    def update_receiver_estimate(
        self, *, bitrate: int, ssrcs: list[int], now_ms: int
    ) -> bool:
        for ssrc in ssrcs:
            if ssrc in self.__senders:
                self.__estimates[ssrc] = (bitrate, now_ms)

        target = self.__derive_session_target(now_ms)
        if target is None:
            return False

        self.__session_target_bitrate = target
        self.recompute_allocation()
        return True

    def set_session_target_bitrate(self, bitrate: int) -> None:
        self.__session_target_bitrate = max(1, int(bitrate))
        self.recompute_allocation()

    def observe_encoded_frame(self, *, ssrc: int, payload_bytes: int) -> None:
        state = self.__senders.get(ssrc)
        if state is None or payload_bytes <= 0:
            return
        state.encoded_payload_bytes += payload_bytes

    def sender_trace_allocations(self) -> list[dict[str, int | None]]:
        return [
            {
                "ssrc": ssrc,
                "allocated_bitrate_bps": state.allocated_bitrate,
                "applied_bitrate_bps": state.applied_bitrate,
                "encoder_bitrate_bps": state.sender._get_target_bitrate(),
            }
            for ssrc, state in self.__senders.items()
        ]

    def recompute_allocation(self) -> None:
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

        return self.__target_bitrate_provider()

    def __current_session_target(self) -> int:
        if self.__session_target_bitrate is not None:
            return self.__session_target_bitrate
        return self.__target_bitrate_provider()

    def __apply_sender_target(self, state: SenderState) -> None:
        allocated_bitrate = int(state.allocated_bitrate)
        bitrate = self.__clamp_sender_target(state.sender, allocated_bitrate)
        is_bound_limited = bitrate != allocated_bitrate
        current = state.sender._get_target_bitrate()
        reference_bitrate = current if current is not None else state.applied_bitrate
        if reference_bitrate == bitrate:
            state.applied_bitrate = bitrate
            return
        if reference_bitrate is not None and not is_bound_limited:
            delta = abs(bitrate - reference_bitrate)
            minimum_delta = max(
                _TARGET_APPLY_MIN_DELTA_BPS,
                int(reference_bitrate * _TARGET_APPLY_MIN_DELTA_RATIO),
            )
            if delta < minimum_delta:
                return
        state.sender._set_target_bitrate(bitrate)
        state.applied_bitrate = state.sender._get_target_bitrate() or bitrate

    def __clamp_sender_target(
        self, sender: BitrateControlledSender, bitrate: int
    ) -> int:
        get_bounds = getattr(sender, "_get_bitrate_bounds", None)
        if get_bounds is None:
            return bitrate
        min_bitrate, max_bitrate = get_bounds()
        return max(min_bitrate, min(max_bitrate, bitrate))


@dataclass(frozen=True)
class _NullTransportTelemetry:
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
    last_target_bitrate_bps: int = 7_500_000
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


class _RateLimiter:
    def __init__(self, window_ms: int) -> None:
        self.__window_ms = window_ms
        self.__max_rate_bps = 0
        self.__events: deque[tuple[int, int]] = deque()
        self.__bytes_in_window = 0

    def set_max_rate(self, rate_bps: int) -> None:
        self.__max_rate_bps = max(0, rate_bps)

    def try_use(self, size_bytes: int, now_ms: int) -> bool:
        if size_bytes <= 0:
            return True

        self.__trim(now_ms)
        if self.__max_rate_bps <= 0:
            return False

        max_window_bytes = self.__max_rate_bps * self.__window_ms // 8000
        if self.__bytes_in_window + size_bytes > max_window_bytes:
            return False

        self.__events.append((now_ms, size_bytes))
        self.__bytes_in_window += size_bytes
        return True

    def __trim(self, now_ms: int) -> None:
        cutoff_ms = now_ms - self.__window_ms
        while self.__events and self.__events[0][0] <= cutoff_ms:
            _, size_bytes = self.__events.popleft()
            self.__bytes_in_window -= size_bytes


class RtcpReceivePipeline:
    def __init__(self) -> None:
        self.__observers: list[RtcpReceiveObserver] = []

    def add(self, observer: RtcpReceiveObserver) -> None:
        self.__observers.append(observer)

    def extend(self, observers: list[RtcpReceiveObserver]) -> None:
        self.__observers.extend(observers)

    def handle(self, packet: object, context: RtcpReceiveContext) -> list[object]:
        feedback_packets = []
        for observer in self.__observers:
            feedback_packets.extend(observer.on_rtcp_received(packet, context))
        return feedback_packets


class RtpReceivePipeline:
    def __init__(self) -> None:
        self.__observers: list[RtpReceiveObserver] = []

    def add(self, observer: RtpReceiveObserver) -> None:
        self.__observers.append(observer)

    def extend(self, observers: list[RtpReceiveObserver]) -> None:
        self.__observers.extend(observers)

    def handle(self, packet: object, context: RtpReceiveContext) -> list[object]:
        feedback_packets = []
        for observer in self.__observers:
            feedback_packets.extend(observer.on_rtp_received(packet, context))
        return feedback_packets


class RtpSendPipeline:
    def __init__(self) -> None:
        self.__interceptors: list[RtpSendInterceptor] = []

    @property
    def interceptors(self) -> list[RtpSendInterceptor]:
        return self.__interceptors

    def add(self, interceptor: RtpSendInterceptor) -> None:
        self.__interceptors.append(interceptor)

    def extend(self, interceptors: list[RtpSendInterceptor]) -> None:
        self.__interceptors.extend(interceptors)

    def prepare(self, packet: object, context: RtpSendContext) -> RtpSendDecision:
        combined_decision = RtpSendDecision()
        for interceptor in self.__interceptors:
            decision = interceptor.prepare_rtp(packet, context)
            if decision.drop_packet:
                return decision
            combined_decision = RtpSendDecision(
                continue_pipeline=decision.continue_pipeline,
                drop_packet=False,
                pace_packet=combined_decision.pace_packet or decision.pace_packet,
            )
            if not decision.continue_pipeline:
                break
        return combined_decision


class RtpSentPipeline:
    def __init__(self) -> None:
        self.__observers: list[RtpSentObserver] = []

    def add(self, observer: RtpSentObserver) -> None:
        self.__observers.append(observer)

    def extend(self, observers: list[RtpSentObserver]) -> None:
        self.__observers.extend(observers)

    def notify(self, packet: object, context: RtpSentContext) -> None:
        for observer in self.__observers:
            observer.on_rtp_sent(packet, context)


class RoundTripTimeDispatcher:
    def __init__(self) -> None:
        self.__rtt_handlers: list[RttHandler] = []

    def add_observer(self, observer: RoundTripTimeObserver) -> None:
        self.__rtt_handlers.append(observer.on_round_trip_time)

    def set_rtt(self, rtt_ms: int) -> None:
        if rtt_ms <= 0:
            return
        for handler in self.__rtt_handlers:
            handler(rtt_ms)


class RetransmissionLimiter:
    def __init__(self, target_bitrate_provider: Callable[[], int]) -> None:
        self.__target_bitrate_provider = target_bitrate_provider
        self.__retransmission_rate_limiter = _RateLimiter(
            _RETRANSMISSION_RATE_LIMIT_WINDOW_MS
        )
        self.__retransmission_sent_bytes = 0
        self.__retransmission_limited_bytes = 0
        self.__retransmission_limited_packets = 0

    @property
    def retransmission_sent_bytes(self) -> int:
        return self.__retransmission_sent_bytes

    @property
    def retransmission_limited_bytes(self) -> int:
        return self.__retransmission_limited_bytes

    @property
    def retransmission_limited_packets(self) -> int:
        return self.__retransmission_limited_packets

    def prepare_rtp(self, packet: object, context: RtpSendContext) -> RtpSendDecision:
        if not context.is_retransmission:
            return RtpSendDecision()

        if self.allow_retransmission(
            size_bytes=context.packet_size_bytes,
            now_ms=context.now_ms,
        ):
            return RtpSendDecision()

        return RtpSendDecision(drop_packet=True)

    def allow_retransmission(
        self, *, size_bytes: int, now_ms: Optional[int] = None
    ) -> bool:
        now_ms = (
            clock.current_monotonic_us() // 1000
            if now_ms is None
            else now_ms
        )
        self.__retransmission_rate_limiter.set_max_rate(
            self.__target_bitrate_provider()
        )
        if self.__retransmission_rate_limiter.try_use(size_bytes, now_ms):
            return True

        self.__retransmission_limited_packets += 1
        self.__retransmission_limited_bytes += max(0, size_bytes)
        return False

    def on_rtp_sent(self, packet: object, context: RtpSentContext) -> None:
        if context.is_retransmission:
            self.__retransmission_sent_bytes += max(0, context.size_bytes)


class EncodedPayloadObserver:
    def __init__(self, bitrate_allocator: SenderBitrateAllocator) -> None:
        self.__bitrate_allocator = bitrate_allocator

    def on_rtp_sent(self, packet: object, context: RtpSentContext) -> None:
        self.__bitrate_allocator.observe_encoded_frame(
            ssrc=packet.ssrc,
            payload_bytes=context.payload_size_bytes,
        )


class BitrateTargetApplier:
    def __init__(
        self,
        bitrate_allocator: SenderBitrateAllocator,
        *,
        trace_writer: NetTraceWriter | None,
        retransmission_limiter: RetransmissionLimiter,
    ) -> None:
        self.__bitrate_allocator = bitrate_allocator
        self.__net_trace_writer = trace_writer
        self.__retransmission_limiter = retransmission_limiter
        self.__transport_feedback_active = False
        self.__last_telemetry_ms: Optional[int] = None
        self.__last_telemetry_snapshot: Optional[TelemetrySnapshot] = None
        self.__last_logged_target_bitrate: Optional[int] = None
        self.__last_logged_delay_usage: Optional[str] = None

    def update_receiver_estimate(
        self, bitrate: int, ssrcs: list[int], now_ms: int
    ) -> None:
        if self.__transport_feedback_active:
            if self.__net_trace_writer is not None:
                self.__net_trace_writer.write_remb_receiver_estimate(
                    bitrate=bitrate,
                    ssrcs=ssrcs,
                    now_ms=now_ms,
                    accepted=False,
                    reason="transport-cc-active",
                    sender_count=self.__bitrate_allocator.sender_count,
                )
            return
        if self.__bitrate_allocator.sender_count == 0:
            if self.__net_trace_writer is not None:
                self.__net_trace_writer.write_remb_receiver_estimate(
                    bitrate=bitrate,
                    ssrcs=ssrcs,
                    now_ms=now_ms,
                    accepted=False,
                    reason="no-senders",
                    sender_count=0,
            )
            return

        accepted = self.__bitrate_allocator.update_receiver_estimate(
            bitrate=bitrate,
            ssrcs=ssrcs,
            now_ms=now_ms,
        )
        if not accepted:
            return
        if self.__net_trace_writer is not None:
            self.__net_trace_writer.write_remb_receiver_estimate(
                bitrate=bitrate,
                ssrcs=ssrcs,
                now_ms=now_ms,
                accepted=True,
                reason="accepted",
                sender_count=self.__bitrate_allocator.sender_count,
                allocations=self.__bitrate_allocator.sender_trace_allocations(),
            )

    def on_bitrate_target(self, target: BitrateTarget) -> None:
        if target.source == "remb":
            self.update_receiver_estimate(
                target.target_bitrate_bps,
                list(target.ssrcs),
                target.now_ms if target.now_ms is not None else clock.current_ms(),
            )
            return

        previous_target = self.__bitrate_allocator.session_target_bitrate
        self.__bitrate_allocator.set_session_target_bitrate(
            target.target_bitrate_bps
        )
        if target.source == "transport-cc":
            self.__transport_feedback_active = True
        if target.update is not None:
            self.__log_target_update(
                previous_target,
                target.update,
                telemetry=target.telemetry,
            )
            self.__trace_target_update(
                previous_target,
                target.update,
                telemetry=target.telemetry,
            )
        self.__log_delay_usage_transition(telemetry=target.telemetry)
        self.__log_telemetry(
            target.now_ms if target.now_ms is not None else clock.current_ms(),
            telemetry=target.telemetry,
            pacer_bitrate_bps=target.pacer_bitrate_bps,
        )

    def __log_target_update(self, previous_target, update, telemetry=None) -> None:
        if telemetry is None:
            telemetry = _NullTransportTelemetry()
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

    def __trace_target_update(self, previous_target, update, telemetry=None) -> None:
        if self.__net_trace_writer is None:
            return
        self.__net_trace_writer.write_target_update(
            previous_target=previous_target,
            update=update,
            telemetry=(
                telemetry
                if telemetry is not None
                else _NullTransportTelemetry()
            ),
        )

    def __log_delay_usage_transition(self, telemetry=None) -> None:
        if telemetry is None:
            telemetry = _NullTransportTelemetry()
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

    def __log_telemetry(
        self,
        now_ms: int,
        telemetry=None,
        pacer_bitrate_bps: int | None = None,
    ) -> None:
        if (
            self.__last_telemetry_ms is not None
            and now_ms - self.__last_telemetry_ms < _TELEMETRY_INTERVAL_MS
        ):
            return
        self.__last_telemetry_ms = now_ms

        if telemetry is None:
            telemetry = _NullTransportTelemetry()
        if telemetry.feedback_count <= 0:
            return

        pacer_bitrate = (
            pacer_bitrate_bps
            if pacer_bitrate_bps is not None
            else 0
        )
        states = self.__bitrate_allocator.states
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
            retransmission_sent_rate_bps = 0
            retransmission_limited_rate_bps = 0
            retransmission_limited_packets = 0
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
            retransmission_sent_rate_bps = int(
                (
                    self.__retransmission_limiter.retransmission_sent_bytes
                    - self.__last_telemetry_snapshot.retransmission_sent_bytes
                )
                * 8
                / elapsed_s
            )
            retransmission_limited_rate_bps = int(
                (
                    self.__retransmission_limiter.retransmission_limited_bytes
                    - self.__last_telemetry_snapshot.retransmission_limited_bytes
                )
                * 8
                / elapsed_s
            )
            retransmission_limited_packets = (
                self.__retransmission_limiter.retransmission_limited_packets
                - self.__last_telemetry_snapshot.retransmission_limited_packets
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
            retransmission_sent_bytes=(
                self.__retransmission_limiter.retransmission_sent_bytes
            ),
            retransmission_limited_bytes=(
                self.__retransmission_limiter.retransmission_limited_bytes
            ),
            retransmission_limited_packets=(
                self.__retransmission_limiter.retransmission_limited_packets
            ),
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
            "prior_unacked_bps=%d rtx_sent_bps=%d rtx_limited_bps=%d "
            "rtx_limited_packets=%d sent_fill=%.2f acked_fill=%.2f "
            "in_flight=%d pacing_queue=%d pacing_queue_ms=%d "
            "oldest_in_flight_ms=%d history=%d "
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
            pacer_bitrate,
            allocated_total,
            applied_total,
            encoder_total,
            encoded_observed_bps,
            sent_rate_bps,
            acked_rate_bps,
            lost_rate_bps,
            prior_unacked_rate_bps,
            retransmission_sent_rate_bps,
            retransmission_limited_rate_bps,
            retransmission_limited_packets,
            sent_fill_ratio,
            acked_fill_ratio,
            telemetry.data_in_flight_bytes,
            telemetry.pacing_queue_bytes,
            telemetry.pacing_queue_oldest_age_ms,
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
