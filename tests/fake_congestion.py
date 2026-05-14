from __future__ import annotations

from dataclasses import dataclass
from struct import pack, unpack
from typing import Any

from aiortc.rtp import (
    RTCP_PSFB_APP,
    RtcpPsfbPacket,
    RtcpTransportLayerCcPacket,
    pack_remb_fci,
)
from pyrtcp import RTCP_RTPFB, RtcpHeader
from rtc_types import (
    BitrateTarget,
    RtcCapabilities,
    RtcpFeedbackCapability,
    RtcpFeedbackPacketCapability,
    RtcpReceiveContext,
    RtcRuntimeContext,
    RtcRuntimeContributions,
    RtpHeaderExtensionCapability,
    RtpReceiveContext,
    RtpSendContext,
    RtpSendDecision,
    RtpSentContext,
)


def install_fake_congestion_components(controller: object, *components: object) -> None:
    context = FakeRuntimeContext(controller)
    for component in components:
        contributions = component.runtime_contributions(context)
        for source in contributions.bitrate_target_sources:
            source.set_bitrate_target_observer(controller)
        for observer in contributions.round_trip_time_observers:
            controller.add_round_trip_time_observer(observer)


class FakeRuntimeContext:
    def __init__(self, controller: object) -> None:
        self.controller = controller
        self.rtcp_codecs: list[object] = []

    @property
    def trace_writer(self) -> object | None:
        return getattr(self.controller, "trace_writer", None)

ABS_SEND_TIME_URI = "http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time"
ABS_SEND_TIME_HEADER_EXTENSION_ID = 3
RTPFB_TRANSPORT_CC_FMT = 15
TRANSPORT_CC_HEADER_EXTENSION_ID = 5
TRANSPORT_CC_URI = (
    "http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01"
)


@dataclass(frozen=True)
class FakePacedPacketInfo:
    probe_cluster_id: int = -1

    @property
    def is_probe(self) -> bool:
        return self.probe_cluster_id >= 0


@dataclass(frozen=True)
class FakePacerConfig:
    send_bitrate_bps: int = 7_500_000
    data_window_bytes: int = 37_500
    probe_cluster: Any | None = None


@dataclass
class FakeTelemetry:
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


@dataclass(frozen=True)
class FakeTargetUpdate:
    target_bitrate_bps: int
    stable_target_bitrate_bps: int
    reason: str = "increase"
    loss_fraction: float = 0.0
    rtt_us: int = 0


@dataclass(frozen=True)
class FakeRembFeedback:
    feedback_ssrc: int
    bitrate: int
    ssrcs: list[int]


@dataclass(frozen=True)
class FakeTransportFeedback:
    sender_ssrc: int
    media_ssrc: int
    base_sequence_number: int
    feedback_packet_count: int = 0

    @property
    def fmt(self) -> int:
        return RTPFB_TRANSPORT_CC_FMT

    @property
    def ssrc(self) -> int:
        return self.sender_ssrc

    def __bytes__(self) -> bytes:
        payload = pack(
            "!LLHH",
            self.sender_ssrc,
            self.media_ssrc,
            self.base_sequence_number,
            self.feedback_packet_count,
        )
        return bytes([0x80 | RTPFB_TRANSPORT_CC_FMT, RTCP_RTPFB, 0, 3]) + payload


class FakeTwccRtcpCodec:
    packet_type = RTCP_RTPFB
    fmt = RTPFB_TRANSPORT_CC_FMT

    def parse_rtcp(
        self, *, header: RtcpHeader, payload: bytes, padding: bytes = b""
    ) -> FakeTransportFeedback:
        sender_ssrc, media_ssrc, base_sequence_number, feedback_packet_count = unpack(
            "!LLHH", payload[:12]
        )
        return FakeTransportFeedback(
            sender_ssrc=sender_ssrc,
            media_ssrc=media_ssrc,
            base_sequence_number=base_sequence_number,
            feedback_packet_count=feedback_packet_count,
        )

    def serialize_rtcp(self, packet: object) -> bytes:
        return bytes(packet)


class FakeRembReceiver:
    def __init__(self) -> None:
        self.rtt_ms: list[int] = []
        self.observed: list[tuple[int, int]] = []
        self.feedback_bitrate = 1_000_000

    def on_round_trip_time(self, rtt_ms: int) -> None:
        self.rtt_ms.append(rtt_ms)

    def observe_incoming_rtp(
        self,
        *,
        receiver: object,
        packet: object,
        arrival_time_ms: int,
        feedback_ssrc: int | None,
    ) -> list[FakeRembFeedback]:
        self.observed.append((packet.ssrc, arrival_time_ms))
        if feedback_ssrc is None or packet.extensions.abs_send_time is None:
            return []
        return [
            FakeRembFeedback(
                feedback_ssrc=feedback_ssrc,
                bitrate=self.feedback_bitrate,
                ssrcs=[packet.ssrc],
            )
        ]


class FakeRembRtpReceiveObserver:
    def __init__(self, receiver: FakeRembReceiver) -> None:
        self.receiver = receiver

    def on_rtp_received(
        self, packet: object, context: RtpReceiveContext
    ) -> list[object]:
        feedback_ssrc = context.receiver._get_rtcp_ssrc()
        return [
            RtcpPsfbPacket(
                fmt=RTCP_PSFB_APP,
                ssrc=feedback.feedback_ssrc,
                media_ssrc=0,
                fci=pack_remb_fci(feedback.bitrate, feedback.ssrcs),
            )
            for feedback in self.receiver.observe_incoming_rtp(
                receiver=context.receiver,
                packet=packet,
                arrival_time_ms=context.arrival_time_ms,
                feedback_ssrc=feedback_ssrc,
            )
        ]


class FakeRembRtcpReceiveObserver:
    def __init__(self) -> None:
        self.observer: object | None = None

    def set_bitrate_target_observer(self, observer: object | None) -> None:
        self.observer = observer

    def on_rtcp_received(
        self, packet: object, context: RtcpReceiveContext
    ) -> list[object]:
        return []


class FakeTransportController:
    def __init__(self) -> None:
        self.active = False
        self.target_bitrate = 7_500_000
        self.next_sequence_number = 0
        self.sent_packets: list[object] = []
        self.pacing_queue_updates: list[tuple[int, int]] = []
        self.rtt_us: list[int] = []
        self.telemetry = FakeTelemetry()

    def on_round_trip_time(self, rtt_us: int) -> None:
        self.rtt_us.append(rtt_us)

    def next_transport_sequence_number(self) -> int:
        value = self.next_sequence_number
        self.next_sequence_number = (self.next_sequence_number + 1) & 0xFFFF
        self.telemetry.next_transport_sequence_number = self.next_sequence_number
        return value

    def on_packet_sent(self, packet: object) -> None:
        self.sent_packets.append(packet)
        self.telemetry.sent_packet_count += 1
        self.telemetry.sent_bytes += packet.size_bytes

    def observe_incoming_rtp(
        self,
        *,
        media_ssrc: int,
        transport_sequence_number: int,
        arrival_time_us: int,
        feedback_ssrc: int,
    ) -> list[FakeTransportFeedback]:
        self.active = True
        self.telemetry.feedback_count += 1
        return [
            FakeTransportFeedback(
                sender_ssrc=feedback_ssrc,
                media_ssrc=media_ssrc,
                base_sequence_number=transport_sequence_number,
                feedback_packet_count=self.telemetry.feedback_count,
            )
        ]

    def handle_transport_feedback(
        self, feedback: FakeTransportFeedback, feedback_time_us: int
    ) -> FakeTargetUpdate:
        self.active = True
        self.telemetry.feedback_count += 1
        self.telemetry.last_feedback_base_sequence_number = (
            feedback.base_sequence_number
        )
        self.telemetry.last_feedback_packet_count = 1
        self.telemetry.last_feedback_time_us = feedback_time_us
        return FakeTargetUpdate(
            target_bitrate_bps=self.target_bitrate,
            stable_target_bitrate_bps=self.target_bitrate,
        )

    def get_pacer_config(self) -> FakePacerConfig:
        return FakePacerConfig(send_bitrate_bps=self.target_bitrate)

    def get_target_bitrate(self) -> int:
        return self.target_bitrate

    def update_pacing_queue(
        self, queue_bytes: int, oldest_queue_age_ms: int = 0
    ) -> None:
        self.pacing_queue_updates.append((queue_bytes, oldest_queue_age_ms))
        self.telemetry.pacing_queue_bytes = queue_bytes
        self.telemetry.pacing_queue_oldest_age_ms = oldest_queue_age_ms

    def get_telemetry(self) -> FakeTelemetry:
        self.telemetry.last_target_bitrate_bps = self.target_bitrate
        return self.telemetry


class FakeTransportRtpSendInterceptor:
    def __init__(self, controller: FakeTransportController) -> None:
        self.controller = controller

    def prepare_rtp(
        self, packet: object, context: RtpSendContext
    ) -> RtpSendDecision:
        if (
            context.is_video
            and context.supports_transport_sequence_number
            and packet.extensions.transport_sequence_number is None
        ):
            packet.extensions.transport_sequence_number = (
                self.controller.next_transport_sequence_number()
            )
            return RtpSendDecision(pace_packet=True)
        return RtpSendDecision()


class FakeTransportRtpSentObserver:
    def __init__(self, controller: FakeTransportController) -> None:
        self.controller = controller

    def on_rtp_sent(self, packet: object, context: RtpSentContext) -> None:
        transport_sequence_number = packet.extensions.transport_sequence_number
        if transport_sequence_number is None:
            return

        @dataclass(frozen=True)
        class Packet:
            transport_sequence_number: int
            send_time_us: int
            size_bytes: int
            payload_size_bytes: int
            ssrc: int
            rtp_sequence_number: int
            is_retransmission: bool
            pacing_info: object | None

        self.controller.on_packet_sent(
            Packet(
                transport_sequence_number=transport_sequence_number,
                send_time_us=context.send_time_us,
                size_bytes=context.size_bytes,
                payload_size_bytes=context.payload_size_bytes,
                ssrc=packet.ssrc,
                rtp_sequence_number=packet.sequence_number,
                is_retransmission=context.is_retransmission,
                pacing_info=context.pacing_info,
            )
        )


class FakeTransportRtpReceiveObserver:
    def __init__(self, controller: FakeTransportController) -> None:
        self.controller = controller

    def on_rtp_received(
        self, packet: object, context: RtpReceiveContext
    ) -> list[object]:
        feedback_ssrc = context.receiver._get_rtcp_ssrc()
        transport_sequence_number = packet.extensions.transport_sequence_number
        if feedback_ssrc is None or transport_sequence_number is None:
            return []
        return [
            RtcpTransportLayerCcPacket(feedback=feedback)
            for feedback in self.controller.observe_incoming_rtp(
                media_ssrc=packet.ssrc,
                transport_sequence_number=transport_sequence_number,
                arrival_time_us=context.arrival_time_us,
                feedback_ssrc=feedback_ssrc,
            )
        ]


class FakeTransportRtcpReceiveObserver:
    def __init__(self, controller: FakeTransportController) -> None:
        self.controller = controller
        self.observer: object | None = None

    def set_bitrate_target_observer(self, observer: object | None) -> None:
        self.observer = observer

    def on_rtcp_received(
        self, packet: object, context: RtcpReceiveContext
    ) -> list[object]:
        feedback = getattr(packet, "feedback", packet)
        if getattr(feedback, "fmt", None) != RTPFB_TRANSPORT_CC_FMT:
            return []
        update = self.controller.handle_transport_feedback(feedback, context.now_us)
        if self.observer is not None:
            self.observer.on_bitrate_target(
                BitrateTarget(
                    target_bitrate_bps=update.target_bitrate_bps,
                    source="transport-cc",
                    reason=update.reason,
                    now_ms=context.now_ms,
                    update=update,
                    telemetry=self.controller.get_telemetry(),
                    pacer_bitrate_bps=self.controller.get_pacer_config().send_bitrate_bps,
                )
            )
        return []


class FakeTransportRoundTripTimeObserver:
    def __init__(self, controller: FakeTransportController) -> None:
        self.controller = controller

    def on_round_trip_time(self, rtt_ms: int) -> None:
        self.controller.on_round_trip_time(rtt_ms * 1000)


class FakeRtpPacer:
    def __init__(self) -> None:
        self.paced_sizes: list[int] = []
        self.pending_probe = False

    async def pace(
        self,
        *,
        size_bytes: int,
        config: object | None = None,
        now_ms: int | None = None,
    ) -> FakePacedPacketInfo:
        self.paced_sizes.append(size_bytes)
        if self.pending_probe:
            self.pending_probe = False
            return FakePacedPacketInfo(probe_cluster_id=7)
        return FakePacedPacketInfo()

    def is_probe_pending(self, config: object | None = None) -> bool:
        return self.pending_probe

    def update_queue(
        self,
        *,
        queue_bytes: int,
        oldest_queue_age_ms: int = 0,
    ) -> None:
        pass


class FakeRemb:
    name = "fake-remb"

    def __init__(self) -> None:
        self.receiver = FakeRembReceiver()

    def capabilities(self, kind: str) -> RtcCapabilities:
        if kind != "video":
            return RtcCapabilities()
        return RtcCapabilities(
            rtcp_feedback=[RtcpFeedbackCapability(type="goog-remb")],
            rtp_header_extensions=[
                RtpHeaderExtensionCapability(
                    uri=ABS_SEND_TIME_URI,
                    preferred_id=ABS_SEND_TIME_HEADER_EXTENSION_ID,
                )
            ],
        )

    def runtime_contributions(
        self, context: RtcRuntimeContext
    ) -> RtcRuntimeContributions:
        rtcp_observer = FakeRembRtcpReceiveObserver()
        return RtcRuntimeContributions(
            rtp_receive_observers=[FakeRembRtpReceiveObserver(self.receiver)],
            rtcp_receive_observers=[rtcp_observer],
            bitrate_target_sources=[rtcp_observer],
            round_trip_time_observers=[self.receiver],
        )


class FakeTransportCc:
    name = "fake-transport-cc"

    def __init__(self) -> None:
        self.controller = FakeTransportController()
        self.pacer = FakeRtpPacer()

    def capabilities(self, kind: str) -> RtcCapabilities:
        if kind != "video":
            return RtcCapabilities()
        return RtcCapabilities(
            rtcp_feedback=[RtcpFeedbackCapability(type="transport-cc")],
            rtp_header_extensions=[
                RtpHeaderExtensionCapability(
                    uri=TRANSPORT_CC_URI,
                    preferred_id=TRANSPORT_CC_HEADER_EXTENSION_ID,
                )
            ],
            rtcp_feedback_packets=[
                RtcpFeedbackPacketCapability(
                    packet_type=RTCP_RTPFB,
                    fmt=RTPFB_TRANSPORT_CC_FMT,
                )
            ],
        )

    def runtime_contributions(
        self, context: RtcRuntimeContext
    ) -> RtcRuntimeContributions:
        rtcp_observer = FakeTransportRtcpReceiveObserver(self.controller)
        return RtcRuntimeContributions(
            rtcp_receive_observers=[rtcp_observer],
            rtp_send_interceptors=[
                FakeTransportRtpSendInterceptor(self.controller)
            ],
            rtp_sent_observers=[FakeTransportRtpSentObserver(self.controller)],
            rtp_receive_observers=[
                FakeTransportRtpReceiveObserver(self.controller)
            ],
            rtp_pacer=self.pacer,
            rtcp_codecs=[FakeTwccRtcpCodec()],
            bitrate_target_sources=[rtcp_observer],
            round_trip_time_observers=[
                FakeTransportRoundTripTimeObserver(self.controller)
            ],
        )
