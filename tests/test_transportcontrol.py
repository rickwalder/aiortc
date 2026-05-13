import json
import tempfile
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

from aiortc.codecs import CODECS, HEADER_EXTENSIONS, is_rtx
from aiortc.congestion import TransportCongestionController
from aiortc.rtp import RtcpPsfbPacket, RtcpTransportLayerCcPacket, RtpPacket
from aiortc.transportcontrol import (
    RTCP_RTPFB,
    TRANSPORT_CC_HEADER_EXTENSION_ID,
    AsyncRtpPacer,
    PyccTransportControlProvider,
    TransportControlSentPacket,
    get_transport_control_capabilities,
)
from aiortc.transporttrace import TransportCcTraceWriter
from pycc import RTPFB_TRANSPORT_CC_FMT, TRANSPORT_CC_URI
from pycc.types import PacerConfig, ProbeClusterConfig

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

    def test_get_target_bitrate_uses_pycc_defaults(self) -> None:
        provider = PyccTransportControlProvider()

        self.assertEqual(provider.get_target_bitrate(), 7_500_000)

    def test_update_pacing_queue_is_reflected_in_telemetry(self) -> None:
        provider = PyccTransportControlProvider()

        provider.update_pacing_queue(12345)

        self.assertEqual(provider.get_telemetry().pacing_queue_bytes, 12345)


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

    @asynctest
    async def test_pace_returns_probe_packet_info_until_cluster_minimum(self) -> None:
        pacer = AsyncRtpPacer()
        config = PacerConfig(
            send_bitrate_bps=300_000,
            window_us=40_000,
            probe_cluster=ProbeClusterConfig(
                id=9,
                target_bitrate_bps=6_000_000,
                target_duration_us=2_000,
                target_probe_count=2,
            ),
        )

        first = await pacer.pace(size_bytes=1500, config=config, now_ms=0)
        second = await pacer.pace(size_bytes=1500, config=config, now_ms=0)
        third = await pacer.pace(size_bytes=1500, config=config, now_ms=0)

        self.assertTrue(first.is_probe)
        self.assertEqual(first.probe_cluster_id, 9)
        self.assertTrue(second.is_probe)
        self.assertFalse(third.is_probe)

    @asynctest
    async def test_pace_honors_probe_cluster_min_delta(self) -> None:
        pacer = AsyncRtpPacer()
        sleeps = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        config = PacerConfig(
            send_bitrate_bps=10_000_000,
            window_us=40_000,
            probe_cluster=ProbeClusterConfig(
                id=10,
                target_bitrate_bps=10_000_000,
                target_duration_us=2_000,
                min_probe_delta_us=2_000,
                target_probe_count=2,
            ),
        )

        with patch("aiortc.transportcontrol.asyncio.sleep", new=fake_sleep):
            await pacer.pace(size_bytes=500, config=config, now_ms=0)
            await pacer.pace(size_bytes=500, config=config, now_ms=0)

        self.assertEqual(sleeps, [0.002])

    @asynctest
    async def test_pacer_reports_pending_probe_until_cluster_minimum(self) -> None:
        pacer = AsyncRtpPacer()
        config = PacerConfig(
            send_bitrate_bps=300_000,
            window_us=40_000,
            probe_cluster=ProbeClusterConfig(
                id=11,
                target_bitrate_bps=6_000_000,
                target_duration_us=2_000,
                target_probe_count=2,
            ),
        )

        self.assertTrue(pacer.is_probe_pending(config))

        await pacer.pace(size_bytes=1500, config=config, now_ms=0)
        self.assertTrue(pacer.is_probe_pending(config))
        await pacer.pace(size_bytes=1500, config=config, now_ms=0)

        self.assertFalse(pacer.is_probe_pending(config))

    def test_handle_feedback_activates_provider(self) -> None:
        provider = PyccTransportControlProvider()
        sequence_number = provider.next_transport_sequence_number()
        provider.on_packet_sent(
            TransportControlSentPacket(
                transport_sequence_number=sequence_number,
                send_time_us=10_000,
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
        self.assertEqual(telemetry.sent_packet_count, 1)
        self.assertEqual(telemetry.sent_bytes, 1000)
        self.assertEqual(telemetry.acknowledged_bytes, 1000)
        self.assertEqual(telemetry.lost_bytes, 0)
        self.assertEqual(telemetry.last_feedback_base_sequence_number, sequence_number)
        self.assertEqual(telemetry.last_feedback_packet_count, 1)
        self.assertGreater(telemetry.last_target_bitrate_bps, 0)
        self.assertIn(telemetry.delay_usage, ["normal", "underuse", "overuse"])
        self.assertGreaterEqual(telemetry.trend_threshold_ms, 0)
        self.assertGreaterEqual(telemetry.groups_seen, 0)

    def test_trace_writer_records_sent_and_feedback(self) -> None:
        with tempfile.NamedTemporaryFile() as fp:
            trace_writer = TransportCcTraceWriter(fp.name)
            provider = PyccTransportControlProvider(trace_writer=trace_writer)
            sequence_number = provider.next_transport_sequence_number()
            provider.on_packet_sent(
                TransportControlSentPacket(
                    transport_sequence_number=sequence_number,
                    send_time_us=10_000,
                    size_bytes=1000,
                    payload_size_bytes=900,
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
            trace_writer.close()

            records = [
                json.loads(line)
                for line in fp.file.read().decode("utf-8").splitlines()
            ]

        self.assertEqual(records[0]["type"], "trace-start")
        self.assertEqual(records[1]["type"], "sent")
        self.assertEqual(records[1]["transport_sequence_number"], sequence_number)
        self.assertEqual(records[1]["payload_size_bytes"], 900)
        self.assertEqual(records[2]["type"], "feedback")
        self.assertEqual(records[2]["base_sequence_number"], sequence_number)
        self.assertEqual(records[2]["packets"][0][0], sequence_number)


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

    def test_incoming_twcc_feedback_respects_receiver_cadence(self) -> None:
        controller = TransportCongestionController()
        receiver = DummyReceiver()
        controller.register_receiver(receiver)

        first = RtpPacket(payload_type=96, sequence_number=1, timestamp=2, ssrc=1234)
        first.extensions.transport_sequence_number = 10
        second = RtpPacket(payload_type=96, sequence_number=2, timestamp=3, ssrc=1234)
        second.extensions.transport_sequence_number = 11
        third = RtpPacket(payload_type=96, sequence_number=3, timestamp=4, ssrc=1234)
        third.extensions.transport_sequence_number = 12

        first_feedback = controller.observe_incoming_rtp(receiver, first, 100)
        second_feedback = controller.observe_incoming_rtp(receiver, second, 150)
        third_feedback = controller.observe_incoming_rtp(receiver, third, 200)

        self.assertEqual(len(first_feedback), 1)
        self.assertEqual(second_feedback, [])
        self.assertEqual(len(third_feedback), 1)
        self.assertIsInstance(third_feedback[0], RtcpTransportLayerCcPacket)
        twcc = third_feedback[0].feedback
        self.assertEqual(twcc.base_sequence_number, 11)
        self.assertEqual(
            [packet.sequence_number for packet in twcc.packets],
            [11, 12],
        )

    def test_incoming_twcc_feedback_reports_missing_sequences(self) -> None:
        controller = TransportCongestionController()
        receiver = DummyReceiver()
        controller.register_receiver(receiver)

        first = RtpPacket(payload_type=96, sequence_number=1, timestamp=2, ssrc=1234)
        first.extensions.transport_sequence_number = 10
        third = RtpPacket(payload_type=96, sequence_number=3, timestamp=4, ssrc=1234)
        third.extensions.transport_sequence_number = 12

        controller.observe_incoming_rtp(receiver, first, 100)
        feedback = controller.observe_incoming_rtp(receiver, third, 200)

        self.assertEqual(len(feedback), 1)
        self.assertIsInstance(feedback[0], RtcpTransportLayerCcPacket)
        twcc = feedback[0].feedback
        self.assertEqual(twcc.base_sequence_number, 11)
        self.assertEqual(
            [packet.received for packet in twcc.packets],
            [False, True],
        )

    def test_incoming_twcc_suppresses_remb_fallback_once_active(self) -> None:
        controller = TransportCongestionController()
        receiver = DummyReceiver()
        controller.register_receiver(receiver)
        estimator = controller._TransportCongestionController__remote_bitrate_estimator
        estimator.add = Mock(return_value=(1_000_000, [1234]))

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

        estimator.add.reset_mock()
        self.assertEqual(
            controller.observe_incoming_rtp(receiver, remb_packet, 300),
            [],
        )
        estimator.add.assert_not_called()

    def test_get_pacer_config_delegates_to_provider(self) -> None:
        controller = TransportCongestionController()

        config = controller.get_pacer_config()

        self.assertGreater(config.send_bitrate_bps, 0)
        self.assertGreater(config.data_window_bytes, 0)

    @asynctest
    async def test_pace_rtp_packet_uses_transport_controller_pacer(self) -> None:
        controller = TransportCongestionController()

        with patch(
            "aiortc.congestion.AsyncRtpPacer.pace",
            new_callable=AsyncMock,
        ) as mock_pace:
            await controller.pace_rtp_packet(size_bytes=1200, now_ms=100)
            await controller.pace_rtp_packet(size_bytes=1200, now_ms=100)

        self.assertEqual(mock_pace.await_count, 2)

    def test_retransmission_rate_limiter_uses_transport_target_window(self) -> None:
        controller = TransportCongestionController()

        self.assertTrue(
            controller.allow_retransmission(size_bytes=450_000, now_ms=0)
        )
        self.assertFalse(
            controller.allow_retransmission(size_bytes=20_000, now_ms=0)
        )
        self.assertTrue(
            controller.allow_retransmission(size_bytes=20_000, now_ms=501)
        )

    def test_initial_allocation_splits_pycc_transport_target_evenly(self) -> None:
        controller = TransportCongestionController()
        senders = [DummySender(1000 + i) for i in range(3)]

        for sender in senders:
            controller.register_sender(sender)

        self.assertEqual(
            [sender.target_bitrate for sender in senders],
            [2_500_000] * 3,
        )
