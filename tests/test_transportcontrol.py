from unittest import TestCase
from unittest.mock import patch

from aiortc.codecs import CODECS, HEADER_EXTENSIONS, is_rtx
from aiortc.congestion import TransportCongestionController
from aiortc.rtp import RtcpTransportLayerCcPacket, RtpPacket
from aiortc.transportcontrol import (
    RTCP_RTPFB,
    TRANSPORT_CC_HEADER_EXTENSION_ID,
    AsyncRtpPacer,
    PyccTransportControlProvider,
    TransportControlSentPacket,
    get_transport_control_capabilities,
)
from pycc import RTPFB_TRANSPORT_CC_FMT, TRANSPORT_CC_URI
from pycc.types import PacerConfig

from .utils import asynctest


class DummyReceiver:
    kind = "video"

    def _get_rtcp_ssrc(self) -> int:
        return 4321


class TransportControlCapabilitiesTest(TestCase):
    def test_audio_capabilities_are_empty(self) -> None:
        capabilities = get_transport_control_capabilities("audio")

        self.assertEqual(capabilities.rtcp_feedback, [])
        self.assertEqual(capabilities.rtp_header_extensions, [])
        self.assertEqual(capabilities.rtcp_feedback_formats, [])

    def test_video_capabilities_include_transport_cc(self) -> None:
        capabilities = get_transport_control_capabilities("video")

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

    def test_video_codec_registry_includes_transport_cc_capabilities(self) -> None:
        capabilities = get_transport_control_capabilities("video")
        transport_cc_uri = capabilities.rtp_header_extensions[0].uri
        transport_cc_feedback = capabilities.rtcp_feedback[0]

        self.assertIn(
            transport_cc_uri,
            [extension.uri for extension in HEADER_EXTENSIONS["video"]],
        )
        for codec in CODECS["video"]:
            if not is_rtx(codec):
                self.assertIn(transport_cc_feedback, codec.rtcpFeedback)


class PyccTransportControlProviderTest(TestCase):
    def test_sequence_numbers_wrap(self) -> None:
        provider = PyccTransportControlProvider()
        provider._transport_sequence_number = 0xFFFF

        self.assertEqual(provider.next_transport_sequence_number(), 0xFFFF)
        self.assertEqual(provider.next_transport_sequence_number(), 0)

    def test_observe_incoming_rtp_builds_feedback(self) -> None:
        provider = PyccTransportControlProvider()

        feedback = provider.observe_incoming_rtp(
            media_ssrc=1234,
            transport_sequence_number=55,
            arrival_time_us=100_000,
            feedback_ssrc=4321,
        )

        self.assertTrue(provider.active)
        self.assertEqual(len(feedback), 1)
        self.assertEqual(feedback[0].sender_ssrc, 4321)
        self.assertEqual(feedback[0].media_ssrc, 1234)
        self.assertEqual(feedback[0].base_sequence_number, 55)

    def test_get_pacer_config(self) -> None:
        provider = PyccTransportControlProvider()

        config = provider.get_pacer_config()

        self.assertGreater(config.send_bitrate_bps, 0)
        self.assertGreater(config.data_window_bytes, 0)


class AsyncRtpPacerTest(TestCase):
    @asynctest
    async def test_pace_preserves_order_and_waits_for_budget(self) -> None:
        pacer = AsyncRtpPacer()
        sleeps = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        with patch("aiortc.transportcontrol.asyncio.sleep", new=fake_sleep):
            await pacer.pace(
                size_bytes=1500,
                config=PacerConfig(send_bitrate_bps=300_000, window_us=40_000),
                now_ms=0,
            )
            await pacer.pace(
                size_bytes=1500,
                config=PacerConfig(send_bitrate_bps=300_000, window_us=40_000),
                now_ms=0,
            )

        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0)

    @asynctest
    async def test_pace_serializes_concurrent_senders(self) -> None:
        pacer = AsyncRtpPacer()
        order = []

        async def fake_sleep(delay: float) -> None:
            order.append(("sleep", delay))

        async def pace_packet(name: str) -> None:
            await pacer.pace(
                size_bytes=1500,
                config=PacerConfig(send_bitrate_bps=300_000, window_us=40_000),
                now_ms=0,
            )
            order.append(("sent", name))

        with patch("aiortc.transportcontrol.asyncio.sleep", new=fake_sleep):
            await pace_packet("first")
            await pace_packet("second")

        self.assertEqual(order[-2:], [("sleep", order[-2][1]), ("sent", "second")])

    def test_handle_feedback_activates_provider(self) -> None:
        provider = PyccTransportControlProvider()
        sequence_number = provider.next_transport_sequence_number()
        provider.on_packet_sent(
            TransportControlSentPacket(
                transport_sequence_number=sequence_number,
                send_time_ms=10,
                size_bytes=1000,
                ssrc=1234,
                rtp_sequence_number=99,
            )
        )
        feedback = provider.observe_incoming_rtp(
            media_ssrc=1234,
            transport_sequence_number=sequence_number,
            arrival_time_us=20_000,
            feedback_ssrc=4321,
        )[0]

        provider.handle_transport_feedback(feedback, feedback_time_us=30_000)

        self.assertTrue(provider.active)
        telemetry = provider.get_telemetry()
        self.assertEqual(telemetry.feedback_count, 1)
        self.assertEqual(telemetry.packet_count, 1)
        self.assertEqual(telemetry.received_count, 1)
        self.assertEqual(telemetry.lost_count, 0)
        self.assertGreater(telemetry.last_target_bitrate_bps, 0)


class TransportCongestionControllerTest(TestCase):
    def test_incoming_twcc_rtp_delegates_to_provider(self) -> None:
        controller = TransportCongestionController()
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

    def test_get_pacer_config_delegates_to_provider(self) -> None:
        controller = TransportCongestionController()

        config = controller.get_pacer_config()

        self.assertGreater(config.send_bitrate_bps, 0)
        self.assertGreater(config.data_window_bytes, 0)
