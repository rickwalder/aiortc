import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

from aiortc.codecs import (
    CODECS,
    HEADER_EXTENSIONS,
    get_codec_parameters,
    get_header_extension_parameters,
    is_rtx,
)
from aiortc.congestion import TransportCongestionController
from aiortc.rtccomponents import get_congestion_control_capabilities
from aiortc.rtcpeerconnection import RTCPeerConnection
from aiortc.rtcrtpparameters import RTCRtcpFeedback
from aiortc.rtp import RtcpPsfbPacket, RtcpTransportLayerCcPacket, RtpPacket
from rtc_types import (
    RtcCapabilities,
    RtcpFeedbackCapability,
    RtcpFeedbackPacketCapability,
)

from .fake_congestion import (
    ABS_SEND_TIME_URI,
    RTCP_RTPFB,
    RTPFB_TRANSPORT_CC_FMT,
    TRANSPORT_CC_HEADER_EXTENSION_ID,
    TRANSPORT_CC_URI,
    FakeRemb,
    FakeTransportCc,
    install_fake_congestion_components,
)
from .utils import asynctest


class DummyReceiver:
    kind = "video"

    def _get_rtcp_ssrc(self) -> int:
        return 4321


class DummySender:
    kind = "video"

    def __init__(self, ssrc: int, target_bitrate: int = 2_500_000) -> None:
        self._ssrc = ssrc
        self.target_bitrate = target_bitrate
        self.applied_bitrates = []

    def _get_target_bitrate(self) -> int:
        return self.target_bitrate

    def _set_target_bitrate(self, bitrate: int) -> None:
        self.target_bitrate = bitrate
        self.applied_bitrates.append(bitrate)


class DummyBoundedSender(DummySender):
    def _get_bitrate_bounds(self) -> tuple[int, int]:
        return (500_000, 3_000_000)


class TransportControlCapabilitiesTest(TestCase):
    def test_congestion_control_capabilities_are_empty_by_default(self) -> None:
        capabilities = get_congestion_control_capabilities("video")

        self.assertEqual(capabilities.rtcp_feedback, [])
        self.assertEqual(capabilities.rtp_header_extensions, [])
        self.assertEqual(capabilities.rtcp_feedback_formats, [])

    def test_configured_components_include_remb_capabilities(self) -> None:
        capabilities = get_congestion_control_capabilities(
            "video",
            components=[FakeRemb()],
        )

        self.assertEqual(
            capabilities.rtcp_feedback,
            [RTCRtcpFeedback(type="goog-remb")],
        )
        self.assertEqual(
            [extension.uri for extension in capabilities.rtp_header_extensions],
            [ABS_SEND_TIME_URI],
        )
        self.assertEqual(capabilities.rtcp_feedback_formats, [])

    def test_configured_components_include_transport_cc_capabilities(self) -> None:
        capabilities = get_congestion_control_capabilities(
            "video",
            components=[FakeTransportCc()],
        )

        self.assertEqual(len(capabilities.rtcp_feedback), 1)
        self.assertEqual(capabilities.rtcp_feedback[0].type, "transport-cc")
        self.assertEqual(len(capabilities.rtp_header_extensions), 1)
        self.assertEqual(
            capabilities.rtp_header_extensions[0].id,
            TRANSPORT_CC_HEADER_EXTENSION_ID,
        )
        self.assertEqual(capabilities.rtp_header_extensions[0].uri, TRANSPORT_CC_URI)
        self.assertEqual(
            capabilities.rtcp_feedback_formats,
            [(RTCP_RTPFB, RTPFB_TRANSPORT_CC_FMT)],
        )

    def test_configured_components_can_be_generic(self) -> None:
        class FakeRegime:
            name = "fake"

            def capabilities(self, kind: str) -> RtcCapabilities:
                if kind != "video":
                    return RtcCapabilities()
                return RtcCapabilities(
                    rtcp_feedback=[RtcpFeedbackCapability(type="fake-cc")],
                    rtcp_feedback_packets=[
                        RtcpFeedbackPacketCapability(packet_type=123, fmt=4)
                    ],
                )

        capabilities = get_congestion_control_capabilities(
            "video",
            components=[FakeRegime()],
        )

        self.assertEqual(capabilities.rtcp_feedback, [RTCRtcpFeedback("fake-cc")])
        self.assertEqual(capabilities.rtp_header_extensions, [])
        self.assertEqual(capabilities.rtcp_feedback_formats, [(123, 4)])

    @asynctest
    async def test_peer_connection_offer_uses_empty_default_components(self) -> None:
        pc = RTCPeerConnection()
        pc.addTransceiver("video")

        offer = await pc.createOffer()

        self.assertNotIn("goog-remb", offer.sdp)
        self.assertNotIn("transport-cc", offer.sdp)
        await pc.close()

    @asynctest
    async def test_peer_connection_offer_uses_configured_components(self) -> None:
        pc = RTCPeerConnection(congestion_control=[FakeRemb()])
        pc.addTransceiver("video")

        offer = await pc.createOffer()

        self.assertIn("goog-remb", offer.sdp)
        self.assertNotIn("transport-cc", offer.sdp)
        await pc.close()

    def test_video_codec_registry_excludes_transport_cc_by_default(self) -> None:
        self.assertNotIn(
            TRANSPORT_CC_URI,
            [extension.uri for extension in HEADER_EXTENSIONS["video"]],
        )
        for codec in CODECS["video"]:
            if not is_rtx(codec):
                self.assertNotIn(
                    RTCRtcpFeedback(type="transport-cc"),
                    codec.rtcpFeedback,
                )

    def test_configured_components_add_transport_cc_to_codecs(self) -> None:
        components = [FakeTransportCc()]
        capabilities = get_congestion_control_capabilities("video", components)
        transport_cc_uri = capabilities.rtp_header_extensions[0].uri
        transport_cc_feedback = capabilities.rtcp_feedback[0]

        self.assertIn(
            transport_cc_uri,
            [
                extension.uri
                for extension in get_header_extension_parameters("video", components)
            ],
        )
        for codec in get_codec_parameters("video", components):
            if not is_rtx(codec):
                self.assertIn(transport_cc_feedback, codec.rtcpFeedback)


class TransportCongestionControllerTest(TestCase):
    def make_controller(self, *components: object) -> TransportCongestionController:
        controller = TransportCongestionController()
        install_fake_congestion_components(controller, *components)
        return controller

    def test_incoming_rtp_delegates_to_component_receiver(self) -> None:
        component = FakeTransportCc()
        controller = self.make_controller(component)
        receiver = DummyReceiver()
        controller.register_receiver(receiver)

        packet = RtpPacket(payload_type=96, sequence_number=1, timestamp=2, ssrc=1234)
        packet.extensions.transport_sequence_number = 77

        feedback = controller.observe_incoming_rtp(receiver, packet, 100)

        self.assertEqual(len(feedback), 1)
        self.assertIsInstance(feedback[0], RtcpTransportLayerCcPacket)
        twcc = feedback[0].feedback
        self.assertEqual(twcc.sender_ssrc, 4321)
        self.assertEqual(twcc.media_ssrc, 1234)
        self.assertEqual(twcc.base_sequence_number, 77)
        self.assertTrue(component.controller.active)

    def test_incoming_remb_suppressed_once_transport_component_active(self) -> None:
        remb = FakeRemb()
        transport_cc = FakeTransportCc()
        controller = self.make_controller(remb, transport_cc)
        receiver = DummyReceiver()
        controller.register_receiver(receiver)

        remb_packet = RtpPacket(
            payload_type=96,
            sequence_number=1,
            timestamp=2,
            ssrc=1234,
        )
        remb_packet.extensions.abs_send_time = 123

        feedback = controller.observe_incoming_rtp(receiver, remb_packet, 100)
        self.assertEqual(len(feedback), 1)
        self.assertIsInstance(feedback[0], RtcpPsfbPacket)

        twcc_packet = RtpPacket(
            payload_type=96,
            sequence_number=2,
            timestamp=3,
            ssrc=1234,
        )
        twcc_packet.extensions.transport_sequence_number = 10
        controller.observe_incoming_rtp(receiver, twcc_packet, 200)

        remb.receiver.observed.clear()
        self.assertEqual(
            controller.observe_incoming_rtp(receiver, remb_packet, 300),
            [],
        )
        self.assertEqual(remb.receiver.observed, [])

    def test_incoming_remb_writes_separate_trace(self) -> None:
        with tempfile.NamedTemporaryFile() as fp:
            with patch.dict(os.environ, {"AIORTC_NET_TRACE": fp.name}):
                controller = self.make_controller(FakeRemb())
            receiver = DummyReceiver()
            controller.register_receiver(receiver)

            packet = RtpPacket(
                payload_type=96,
                sequence_number=1,
                timestamp=2,
                ssrc=1234,
            )
            packet.extensions.abs_send_time = 123

            feedback = controller.observe_incoming_rtp(receiver, packet, 100)
            self.assertEqual(len(feedback), 1)
            controller._TransportCongestionController__net_trace_writer.close()

            fp.file.seek(0)
            records = [
                json.loads(line)
                for line in fp.file.read().decode("utf-8").splitlines()
            ]

        self.assertEqual(records[0]["type"], "trace-start")

    def test_receiver_estimate_updates_sender_allocation(self) -> None:
        controller = TransportCongestionController()
        sender = DummySender(ssrc=1234)
        controller.register_sender(sender)

        controller.update_receiver_estimate(
            bitrate=1_000_000,
            ssrcs=[1234],
            now_ms=100,
        )

        self.assertEqual(sender.target_bitrate, 1_000_000)

    def test_small_receiver_estimate_changes_do_not_reconfigure_sender(self) -> None:
        controller = self.make_controller(FakeTransportCc())
        sender = DummySender(ssrc=1234)
        controller.register_sender(sender)

        controller.update_receiver_estimate(
            bitrate=3_000_000,
            ssrcs=[1234],
            now_ms=100,
        )
        controller.update_receiver_estimate(
            bitrate=3_010_000,
            ssrcs=[1234],
            now_ms=300,
        )
        controller.update_receiver_estimate(
            bitrate=3_040_000,
            ssrcs=[1234],
            now_ms=500,
        )

        self.assertEqual(sender.applied_bitrates[-2:], [3_000_000, 3_040_000])

    def test_sender_target_application_respects_sender_bounds(self) -> None:
        controller = self.make_controller(FakeTransportCc())
        sender = DummyBoundedSender(ssrc=1234)
        controller.register_sender(sender)

        controller.update_receiver_estimate(
            bitrate=14_000_000,
            ssrcs=[1234],
            now_ms=100,
        )
        controller.update_receiver_estimate(
            bitrate=14_100_000,
            ssrcs=[1234],
            now_ms=300,
        )

        self.assertEqual(sender.target_bitrate, 3_000_000)
        self.assertEqual(sender.applied_bitrates, [3_000_000])

    def test_sender_target_application_snaps_to_bound_within_hysteresis(self) -> None:
        controller = self.make_controller(FakeTransportCc())
        sender = DummyBoundedSender(ssrc=1234, target_bitrate=2_980_000)
        controller.register_sender(sender)
        sender.target_bitrate = 2_980_000
        sender.applied_bitrates.clear()

        controller.update_receiver_estimate(
            bitrate=3_010_000,
            ssrcs=[1234],
            now_ms=100,
        )

        self.assertEqual(sender.target_bitrate, 3_000_000)
        self.assertEqual(sender.applied_bitrates, [3_000_000])

    @asynctest
    async def test_pace_rtp_packet_uses_configured_component_pacer(self) -> None:
        component = FakeTransportCc()
        controller = self.make_controller(component)

        await controller.pace_rtp_packet(size_bytes=1200, now_ms=100)
        await controller.pace_rtp_packet(size_bytes=1200, now_ms=100)

        self.assertEqual(component.pacer.paced_sizes, [1200, 1200])

    def test_retransmission_rate_limiter_uses_transport_target_window(self) -> None:
        controller = self.make_controller(FakeTransportCc())

        self.assertTrue(
            controller.allow_retransmission(size_bytes=450_000, now_ms=0)
        )
        self.assertFalse(
            controller.allow_retransmission(size_bytes=20_000, now_ms=0)
        )
        self.assertTrue(
            controller.allow_retransmission(size_bytes=20_000, now_ms=501)
        )

    def test_initial_allocation_splits_transport_target_evenly(self) -> None:
        controller = self.make_controller(FakeTransportCc())
        senders = [DummySender(1000 + i) for i in range(3)]

        for sender in senders:
            controller.register_sender(sender)

        self.assertEqual(
            [sender.target_bitrate for sender in senders],
            [2_500_000] * 3,
        )
