import asyncio
import datetime
from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from aiortc.rtcdtlstransport import (
    SRTP_AEAD_AES_256_GCM,
    SRTP_AES128_CM_SHA1_80,
    RTCCertificate,
    RTCDtlsFingerprint,
    RTCDtlsParameters,
    RTCDtlsTransport,
    RtpRouter,
)
from aiortc.rtcrtpparameters import (
    RTCRtpCodecParameters,
    RTCRtpDecodingParameters,
    RTCRtpHeaderExtensionParameters,
    RTCRtpReceiveParameters,
    RTCRtpSendParameters,
)
from aiortc.rtp import (
    RTCP_PSFB_APP,
    RTCP_PSFB_PLI,
    RTCP_RTPFB_NACK,
    AnyRtcpPacket,
    HeaderExtensionsMap,
    RtcpByePacket,
    RtcpPacket,
    RtcpPsfbPacket,
    RtcpReceiverInfo,
    RtcpRrPacket,
    RtcpRtpfbPacket,
    RtcpSenderInfo,
    RtcpSrPacket,
    RtcpTransportLayerCcPacket,
    RtpPacket,
    pack_remb_fci,
)
from OpenSSL import SSL
from pycc import TRANSPORT_CC_URI, PacedPacketInfo

from .utils import asynctest, dummy_ice_transport_pair, load, set_loss_pattern

RTP = load("rtp.bin")
RTCP = load("rtcp_sr.bin")


class BrokenDataReceiver:
    async def _handle_data(self, data: bytes) -> None:
        raise Exception("some error")


class DummyDataReceiver:
    def __init__(self) -> None:
        self.data: list[bytes] = []

    async def _handle_data(self, data: bytes) -> None:
        self.data.append(data)


class DummyRtpReceiver:
    def __init__(self) -> None:
        self.rtp_packets: list[RtpPacket] = []
        self.rtcp_packets: list[AnyRtcpPacket] = []

    def _handle_disconnect(self) -> None:
        pass

    async def _handle_rtp_packet(self, packet: RtpPacket, arrival_time_ms: int) -> None:
        self.rtp_packets.append(packet)

    async def _handle_rtcp_packet(self, packet: AnyRtcpPacket) -> None:
        self.rtcp_packets.append(packet)


class DummyRtpSender:
    kind = "video"
    _ssrc = 0

    def __init__(self) -> None:
        self._sequence_number = 1000
        self._extensions_map = HeaderExtensionsMap()
        self._extensions_map.configure(
            RTCRtpSendParameters(
                headerExtensions=[
                    RTCRtpHeaderExtensionParameters(id=5, uri=TRANSPORT_CC_URI)
                ]
            )
        )

    def _get_target_bitrate(self) -> int:
        return 1_000_000

    def _get_bitrate_bounds(self) -> tuple[int, int]:
        return (500_000, 3_000_000)

    def _set_target_bitrate(self, bitrate: int) -> None:
        pass

    def _create_rtp_padding_packet(
        self, padding_size: int
    ) -> tuple[RtpPacket, HeaderExtensionsMap]:
        packet = RtpPacket(
            payload_type=100,
            sequence_number=self._sequence_number,
            timestamp=123456,
            ssrc=self._ssrc,
        )
        packet.padding_size = padding_size
        self._sequence_number += 1
        return packet, self._extensions_map

    async def _handle_rtcp_packet(self, packet: AnyRtcpPacket) -> None:
        pass


class TwccDummyRtpReceiver(DummyRtpReceiver):
    kind = "video"

    def _get_rtcp_ssrc(self) -> int:
        return 4321


class RTCCertificateTest(TestCase):
    def test_generate(self) -> None:
        certificate = RTCCertificate.generateCertificate()
        self.assertIsNotNone(certificate)

        expires = certificate.expires
        self.assertIsNotNone(expires)
        self.assertIsInstance(expires, datetime.datetime)

        fingerprints = certificate.getFingerprints()
        self.assertEqual(len(fingerprints), 3)
        self.assertEqual(fingerprints[0].algorithm, "sha-256")
        self.assertEqual(len(fingerprints[0].value), 95)
        self.assertEqual(fingerprints[1].algorithm, "sha-384")
        self.assertEqual(len(fingerprints[1].value), 143)
        self.assertEqual(fingerprints[2].algorithm, "sha-512")
        self.assertEqual(len(fingerprints[2].value), 191)


class RTCDtlsTransportTest(TestCase):
    def assertCounters(
        self,
        transport_a: RTCDtlsTransport,
        transport_b: RTCDtlsTransport,
        packets_sent_a: int,
        packets_sent_b: int,
    ) -> None:
        stats_a = transport_a._get_stats()[transport_a._stats_id]
        stats_b = transport_b._get_stats()[transport_b._stats_id]

        self.assertEqual(stats_a.packetsSent, packets_sent_a)
        self.assertEqual(stats_a.packetsReceived, packets_sent_b)
        self.assertGreater(stats_a.bytesSent, 0)
        self.assertGreater(stats_a.bytesReceived, 0)

        self.assertEqual(stats_b.packetsSent, packets_sent_b)
        self.assertEqual(stats_b.packetsReceived, packets_sent_a)
        self.assertGreater(stats_b.bytesSent, 0)
        self.assertGreater(stats_b.bytesReceived, 0)

        self.assertEqual(stats_a.bytesSent, stats_b.bytesReceived)
        self.assertEqual(stats_b.bytesSent, stats_a.bytesReceived)

    @asynctest
    async def test_data(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyDataReceiver()
        session1._register_data_receiver(receiver1)

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        receiver2 = DummyDataReceiver()
        session2._register_data_receiver(receiver2)

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # send encypted data
        await session1._send_data(b"ping")
        await asyncio.sleep(0.1)
        self.assertEqual(receiver2.data, [b"ping"])

        await session2._send_data(b"pong")
        await asyncio.sleep(0.1)
        self.assertEqual(receiver1.data, [b"pong"])

        # shutdown
        await session1.stop()
        await asyncio.sleep(0.1)
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

        # try closing again
        await session1.stop()
        await session2.stop()

        # try sending after close
        with self.assertRaises(ConnectionError):
            await session1._send_data(b"foo")

    @asynctest
    async def test_data_handler_error(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyDataReceiver()
        session1._register_data_receiver(receiver1)

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        session2._register_data_receiver(BrokenDataReceiver())

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # send encypted data
        await session1._send_data(b"ping")
        await asyncio.sleep(0.1)

        # shutdown
        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_rtp(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyRtpReceiver()
        session1._register_rtp_receiver(
            receiver1,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=1831097322, payloadType=0)],
            ),
        )

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        receiver2 = DummyRtpReceiver()
        session2._register_rtp_receiver(
            receiver2,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=4028317929, payloadType=0)],
            ),
        )

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )
        self.assertCounters(session1, session2, 2, 2)

        # send RTP
        await session1._send_rtp(RTP)
        await asyncio.sleep(0.1)
        self.assertCounters(session1, session2, 3, 2)
        self.assertEqual(len(receiver2.rtcp_packets), 0)
        self.assertEqual(len(receiver2.rtp_packets), 1)

        # send RTCP
        await session2._send_rtp(RTCP)
        await asyncio.sleep(0.1)
        self.assertCounters(session1, session2, 3, 3)
        self.assertEqual(len(receiver1.rtcp_packets), 1)
        self.assertEqual(len(receiver1.rtp_packets), 0)

        # shutdown
        await session1.stop()
        await asyncio.sleep(0.1)
        self.assertCounters(session1, session2, 4, 3)
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

        # try closing again
        await session1.stop()
        await session2.stop()

        # try sending after close
        with self.assertRaises(ConnectionError):
            await session1._send_rtp(RTP)

    @asynctest
    async def test_send_rtp_packet_assigns_twcc_at_transport(self) -> None:
        transport1, _ = dummy_ice_transport_pair()
        session = RTCDtlsTransport(
            transport1,
            [RTCCertificate.generateCertificate()],
        )
        extensions_map = HeaderExtensionsMap()
        extensions_map.configure(
            RTCRtpSendParameters(
                headerExtensions=[
                    RTCRtpHeaderExtensionParameters(
                        id=5,
                        uri=TRANSPORT_CC_URI,
                    )
                ]
            )
        )
        sent_packets: list[RtpPacket] = []

        async def mock_send_rtp(data: bytes) -> None:
            sent_packets.append(RtpPacket.parse(data, extensions_map))

        session._send_rtp = mock_send_rtp  # type: ignore

        packet1 = RtpPacket(payload_type=100, sequence_number=1000, timestamp=1)
        packet1.ssrc = 1234
        packet1.payload = b"abc"
        packet2 = RtpPacket(payload_type=100, sequence_number=1001, timestamp=1)
        packet2.ssrc = 1234
        packet2.payload = b"def"

        self.assertIsNone(packet1.extensions.transport_sequence_number)
        self.assertIsNone(packet2.extensions.transport_sequence_number)

        await session._send_rtp_packet(
            packet1,
            extensions_map,
            is_video=True,
            payload_size_bytes=len(packet1.payload),
        )
        await session._send_rtp_packet(
            packet2,
            extensions_map,
            is_video=True,
            payload_size_bytes=len(packet2.payload),
        )

        self.assertEqual(len(sent_packets), 0)
        self.assertTrue(await session._send_next_rtp_packet_from_queue())
        self.assertTrue(await session._send_next_rtp_packet_from_queue())

        self.assertEqual(
            [packet.extensions.transport_sequence_number for packet in sent_packets],
            [0, 1],
        )

    @asynctest
    async def test_send_video_rtp_packet_without_twcc_bypasses_pacer(self) -> None:
        transport1, _ = dummy_ice_transport_pair()
        session = RTCDtlsTransport(
            transport1,
            [RTCCertificate.generateCertificate()],
        )
        extensions_map = HeaderExtensionsMap()
        sent_packets: list[RtpPacket] = []

        async def mock_send_rtp(data: bytes) -> None:
            sent_packets.append(RtpPacket.parse(data, extensions_map))

        session._send_rtp = mock_send_rtp  # type: ignore
        session._congestion_controller.next_transport_sequence_number = Mock()
        session._congestion_controller.pace_rtp_packet = AsyncMock()
        session._congestion_controller.on_packet_sent = Mock()
        session._congestion_controller.observe_encoded_frame = Mock()

        packet = RtpPacket(payload_type=100, sequence_number=1000, timestamp=1)
        packet.ssrc = 1234
        packet.payload = b"abc"

        await session._send_rtp_packet(
            packet,
            extensions_map,
            is_video=True,
            payload_size_bytes=len(packet.payload),
        )

        self.assertEqual(len(sent_packets), 1)
        self.assertEqual(len(session._rtp_queue), 0)
        session._congestion_controller.next_transport_sequence_number.assert_not_called()
        session._congestion_controller.pace_rtp_packet.assert_not_called()
        session._congestion_controller.on_packet_sent.assert_not_called()
        session._congestion_controller.observe_encoded_frame.assert_called_once_with(
            ssrc=1234,
            payload_bytes=3,
        )

    @asynctest
    async def test_send_rtp_packet_clones_queued_packet_for_twcc(self) -> None:
        transport1, _ = dummy_ice_transport_pair()
        session = RTCDtlsTransport(
            transport1,
            [RTCCertificate.generateCertificate()],
        )
        extensions_map = HeaderExtensionsMap()
        extensions_map.configure(
            RTCRtpSendParameters(
                headerExtensions=[
                    RTCRtpHeaderExtensionParameters(
                        id=5,
                        uri=TRANSPORT_CC_URI,
                    )
                ]
            )
        )
        sent_packets: list[RtpPacket] = []

        async def mock_send_rtp(data: bytes) -> None:
            sent_packets.append(RtpPacket.parse(data, extensions_map))

        session._send_rtp = mock_send_rtp  # type: ignore

        packet = RtpPacket(payload_type=100, sequence_number=1000, timestamp=1)
        packet.ssrc = 1234
        packet.payload = b"abc"

        await session._send_rtp_packet(
            packet,
            extensions_map,
            is_video=True,
            payload_size_bytes=len(packet.payload),
            is_retransmission=True,
        )
        await session._send_rtp_packet(
            packet,
            extensions_map,
            is_video=True,
            payload_size_bytes=len(packet.payload),
            is_retransmission=True,
        )

        self.assertIsNone(packet.extensions.transport_sequence_number)
        self.assertTrue(await session._send_next_rtp_packet_from_queue())
        self.assertTrue(await session._send_next_rtp_packet_from_queue())

        self.assertEqual(
            [packet.extensions.transport_sequence_number for packet in sent_packets],
            [0, 1],
        )

    @asynctest
    async def test_send_rtp_packet_limits_retransmission_before_twcc(self) -> None:
        transport1, _ = dummy_ice_transport_pair()
        session = RTCDtlsTransport(
            transport1,
            [RTCCertificate.generateCertificate()],
        )
        extensions_map = HeaderExtensionsMap()
        extensions_map.configure(
            RTCRtpSendParameters(
                headerExtensions=[
                    RTCRtpHeaderExtensionParameters(
                        id=5,
                        uri=TRANSPORT_CC_URI,
                    )
                ]
            )
        )
        session._congestion_controller.allow_retransmission = Mock(
            return_value=False
        )
        session._congestion_controller.next_transport_sequence_number = Mock()

        packet = RtpPacket(payload_type=100, sequence_number=1000, timestamp=1)
        packet.ssrc = 1234
        packet.payload = b"abc"

        await session._send_rtp_packet(
            packet,
            extensions_map,
            is_video=True,
            payload_size_bytes=len(packet.payload),
            is_retransmission=True,
        )

        self.assertIsNone(packet.extensions.transport_sequence_number)
        self.assertEqual(len(session._rtp_queue), 0)
        session._congestion_controller.next_transport_sequence_number.assert_not_called()

    @asynctest
    async def test_send_probe_padding_packet_uses_twcc_and_probe_info(self) -> None:
        transport1, _ = dummy_ice_transport_pair()
        session = RTCDtlsTransport(
            transport1,
            [RTCCertificate.generateCertificate()],
        )
        sender = DummyRtpSender()
        sender._ssrc = 1234
        session._rtp_router.register_sender(sender, sender._ssrc)
        sent_packets: list[RtpPacket] = []

        async def mock_send_rtp(data: bytes) -> None:
            sent_packets.append(RtpPacket.parse(data, sender._extensions_map))

        session._send_rtp = mock_send_rtp  # type: ignore
        session._congestion_controller.has_probe_pending = Mock(return_value=True)
        session._congestion_controller.pace_rtp_packet = AsyncMock(
            return_value=PacedPacketInfo(probe_cluster_id=7)
        )
        session._congestion_controller.next_transport_sequence_number = Mock(
            return_value=55
        )
        session._congestion_controller.on_packet_sent = Mock()

        sent = await session._send_rtp_probe_padding_packet()

        self.assertTrue(sent)
        self.assertEqual(len(sent_packets), 1)
        self.assertEqual(sent_packets[0].extensions.transport_sequence_number, 55)
        self.assertEqual(sent_packets[0].padding_size, 255)
        session._congestion_controller.on_packet_sent.assert_called_once()

    @asynctest
    async def test_handle_rtp_data_emits_twcc_feedback(self) -> None:
        transport1, _ = dummy_ice_transport_pair()
        session = RTCDtlsTransport(
            transport1,
            [RTCCertificate.generateCertificate()],
        )
        receiver = TwccDummyRtpReceiver()
        parameters = RTCRtpReceiveParameters(
            codecs=[
                RTCRtpCodecParameters(
                    mimeType="video/VP8",
                    clockRate=90000,
                    payloadType=96,
                )
            ],
            encodings=[RTCRtpDecodingParameters(ssrc=1234, payloadType=96)],
            headerExtensions=[
                RTCRtpHeaderExtensionParameters(id=5, uri=TRANSPORT_CC_URI)
            ],
        )
        session._register_rtp_receiver(receiver, parameters)
        extensions_map = HeaderExtensionsMap()
        extensions_map.configure(parameters)
        sent_rtcp: list[AnyRtcpPacket] = []

        async def mock_send_rtp(data: bytes) -> None:
            sent_rtcp.extend(RtcpPacket.parse(data))

        session._send_rtp = mock_send_rtp  # type: ignore

        packet = RtpPacket(
            payload_type=96,
            sequence_number=1000,
            timestamp=123456,
            ssrc=1234,
        )
        packet.extensions.transport_sequence_number = 55
        packet.payload = b"abc"

        await session._handle_rtp_data(packet.serialize(extensions_map), 100)

        self.assertEqual(len(receiver.rtp_packets), 1)
        self.assertEqual(
            receiver.rtp_packets[0].extensions.transport_sequence_number,
            55,
        )
        self.assertEqual(len(sent_rtcp), 1)
        feedback = sent_rtcp[0]
        self.assertIsInstance(feedback, RtcpTransportLayerCcPacket)
        assert isinstance(feedback, RtcpTransportLayerCcPacket)
        self.assertEqual(feedback.feedback.sender_ssrc, 4321)
        self.assertEqual(feedback.feedback.media_ssrc, 1234)
        self.assertEqual(feedback.feedback.base_sequence_number, 55)

    @asynctest
    async def test_handle_rtp_data_twcc_feedback_reports_missing_packets(self) -> None:
        transport1, _ = dummy_ice_transport_pair()
        session = RTCDtlsTransport(
            transport1,
            [RTCCertificate.generateCertificate()],
        )
        receiver = TwccDummyRtpReceiver()
        parameters = RTCRtpReceiveParameters(
            codecs=[
                RTCRtpCodecParameters(
                    mimeType="video/VP8",
                    clockRate=90000,
                    payloadType=96,
                )
            ],
            encodings=[RTCRtpDecodingParameters(ssrc=1234, payloadType=96)],
            headerExtensions=[
                RTCRtpHeaderExtensionParameters(id=5, uri=TRANSPORT_CC_URI)
            ],
        )
        session._register_rtp_receiver(receiver, parameters)
        extensions_map = HeaderExtensionsMap()
        extensions_map.configure(parameters)
        sent_rtcp: list[AnyRtcpPacket] = []

        async def mock_send_rtp(data: bytes) -> None:
            sent_rtcp.extend(RtcpPacket.parse(data))

        session._send_rtp = mock_send_rtp  # type: ignore

        packet1 = RtpPacket(
            payload_type=96,
            sequence_number=1000,
            timestamp=123456,
            ssrc=1234,
        )
        packet1.extensions.transport_sequence_number = 55
        packet1.payload = b"abc"
        packet3 = RtpPacket(
            payload_type=96,
            sequence_number=1002,
            timestamp=123456,
            ssrc=1234,
        )
        packet3.extensions.transport_sequence_number = 57
        packet3.payload = b"ghi"

        await session._handle_rtp_data(packet1.serialize(extensions_map), 100)
        await session._handle_rtp_data(packet3.serialize(extensions_map), 200)

        self.assertEqual(len(sent_rtcp), 2)
        feedback = sent_rtcp[-1]
        self.assertIsInstance(feedback, RtcpTransportLayerCcPacket)
        assert isinstance(feedback, RtcpTransportLayerCcPacket)
        self.assertEqual(feedback.feedback.base_sequence_number, 56)
        self.assertEqual(
            [packet.received for packet in feedback.feedback.packets],
            [False, True],
        )

    @asynctest
    async def test_rtp_malformed(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        # receive truncated RTP
        await session1._handle_rtp_data(RTP[0:8], 0)

        # receive truncated RTCP
        await session1._handle_rtcp_data(RTCP[0:8])

    @asynctest
    async def test_srtp_unprotect_error(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        receiver1 = DummyRtpReceiver()
        session1._register_rtp_receiver(
            receiver1,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=1831097322, payloadType=0)],
            ),
        )

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        receiver2 = DummyRtpReceiver()
        session2._register_rtp_receiver(
            receiver2,
            RTCRtpReceiveParameters(
                codecs=[
                    RTCRtpCodecParameters(
                        mimeType="audio/PCMU", clockRate=8000, payloadType=0
                    )
                ],
                encodings=[RTCRtpDecodingParameters(ssrc=4028317929, payloadType=0)],
            ),
        )

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # send same RTP twice, to trigger error on the receiver side:
        # "replay check failed (bad index)"
        await session1._send_rtp(RTP)
        await session1._send_rtp(RTP)
        await asyncio.sleep(0.1)
        self.assertEqual(len(receiver2.rtcp_packets), 0)
        self.assertEqual(len(receiver2.rtp_packets), 1)

        # shutdown
        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_abrupt_disconnect(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        # break connections -> tasks exits
        await transport1.stop()
        await transport2.stop()
        await asyncio.sleep(0.1)

        # close DTLS
        await session1.stop()
        await session2.stop()

        # check outcome
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

    @asynctest
    async def test_abrupt_disconnect_2(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        def fake_write_ssl() -> None:
            raise ConnectionError

        session1._write_ssl = fake_write_ssl  # type: ignore

        # close DTLS -> ConnectionError
        await session1.stop()
        await session2.stop()
        await asyncio.sleep(0.1)

        # check outcome
        self.assertEqual(session1.state, "closed")
        self.assertEqual(session2.state, "closed")

    @asynctest
    async def test_bad_client_fingerprint(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        bogus_parameters = RTCDtlsParameters(
            fingerprints=[
                RTCDtlsFingerprint(algorithm="sha-256", value="bogus_fingerprint")
            ]
        )
        await asyncio.gather(
            session1.start(bogus_parameters),
            session2.start(session1.getLocalParameters()),
        )
        self.assertEqual(session1.state, "failed")
        self.assertEqual(session2.state, "connected")

        await session1.stop()
        await session2.stop()

    @patch("aiortc.rtcdtlstransport.SSL.Connection.do_handshake")
    @asynctest
    async def test_handshake_error(self, mock_do_handshake: MagicMock) -> None:
        mock_do_handshake.side_effect = SSL.Error(
            [("SSL routines", "", "decryption failed or bad record mac")]
        )

        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )
        self.assertEqual(session1.state, "failed")
        self.assertEqual(session2.state, "failed")

        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_handshake_error_no_common_srtp_profile(self) -> None:
        transport1, transport2 = dummy_ice_transport_pair()

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])
        session1._srtp_profiles = [SRTP_AEAD_AES_256_GCM]

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])
        session2._srtp_profiles = [SRTP_AES128_CM_SHA1_80]

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )
        self.assertEqual(session1.state, "failed")
        self.assertEqual(session2.state, "failed")

        await session1.stop()
        await session2.stop()

    @asynctest
    async def test_lossy_channel(self) -> None:
        """
        Transport with 25% loss eventually connects.
        """
        transport1, transport2 = dummy_ice_transport_pair()
        loss_pattern = [True, False, False, False]
        set_loss_pattern(transport1, loss_pattern)
        set_loss_pattern(transport2, loss_pattern)

        certificate1 = RTCCertificate.generateCertificate()
        session1 = RTCDtlsTransport(transport1, [certificate1])

        certificate2 = RTCCertificate.generateCertificate()
        session2 = RTCDtlsTransport(transport2, [certificate2])

        await asyncio.gather(
            session1.start(session2.getLocalParameters()),
            session2.start(session1.getLocalParameters()),
        )

        await session1.stop()
        await session2.stop()


class RtpRouterTest(TestCase):
    def test_route_rtcp(self) -> None:
        receiver = DummyRtpReceiver()
        sender = DummyRtpSender()

        router = RtpRouter()
        router.register_receiver(receiver, ssrcs=[1234, 2345], payload_types=[96, 97])
        router.register_sender(sender, ssrc=3456)

        # BYE
        packet: AnyRtcpPacket = RtcpByePacket(sources=[1234, 2345])
        self.assertEqual(router.route_rtcp(packet), set([receiver]))

        # RR
        packet = RtcpRrPacket(
            ssrc=1234,
            reports=[
                RtcpReceiverInfo(
                    ssrc=3456,
                    fraction_lost=0,
                    packets_lost=0,
                    highest_sequence=630,
                    jitter=1906,
                    lsr=0,
                    dlsr=0,
                )
            ],
        )
        self.assertEqual(router.route_rtcp(packet), set([sender]))

        # SR
        packet = RtcpSrPacket(
            ssrc=1234,
            sender_info=RtcpSenderInfo(
                ntp_timestamp=0, rtp_timestamp=0, packet_count=0, octet_count=0
            ),
            reports=[
                RtcpReceiverInfo(
                    ssrc=3456,
                    fraction_lost=0,
                    packets_lost=0,
                    highest_sequence=630,
                    jitter=1906,
                    lsr=0,
                    dlsr=0,
                )
            ],
        )
        self.assertEqual(router.route_rtcp(packet), set([receiver, sender]))

        # PSFB - PLI
        packet = RtcpPsfbPacket(fmt=RTCP_PSFB_PLI, ssrc=1234, media_ssrc=3456)
        self.assertEqual(router.route_rtcp(packet), set([sender]))

        # PSFB - REMB
        packet = RtcpPsfbPacket(
            fmt=RTCP_PSFB_APP,
            ssrc=1234,
            media_ssrc=0,
            fci=pack_remb_fci(4160000, [3456]),
        )
        self.assertEqual(router.route_rtcp(packet), set([sender]))

        # PSFB - JUNK
        packet = RtcpPsfbPacket(fmt=RTCP_PSFB_APP, ssrc=1234, media_ssrc=0, fci=b"JUNK")
        self.assertEqual(router.route_rtcp(packet), set())

        # RTPFB
        packet = RtcpRtpfbPacket(fmt=RTCP_RTPFB_NACK, ssrc=1234, media_ssrc=3456)
        self.assertEqual(router.route_rtcp(packet), set([sender]))

    def test_route_rtp(self) -> None:
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()

        router = RtpRouter()
        router.register_receiver(receiver1, ssrcs=[1234, 2345], payload_types=[96, 97])
        router.register_receiver(receiver2, ssrcs=[3456, 4567], payload_types=[98, 99])

        # known SSRC and payload type
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=1234, payload_type=96)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=2345, payload_type=97)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=3456, payload_type=98)), receiver2
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=4567, payload_type=99)), receiver2
        )

        # unknown SSRC, known payload type
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=5678, payload_type=96)), receiver1
        )
        self.assertEqual(router.ssrc_table[5678], receiver1)

        # unknown SSRC and payload type
        self.assertEqual(router.route_rtp(RtpPacket(ssrc=6789, payload_type=100)), None)

    def test_route_rtp_ambiguous_payload_type(self) -> None:
        receiver1 = DummyRtpReceiver()
        receiver2 = DummyRtpReceiver()

        router = RtpRouter()
        router.register_receiver(receiver1, ssrcs=[1234, 2345], payload_types=[96, 97])
        router.register_receiver(receiver2, ssrcs=[3456, 4567], payload_types=[96, 97])

        # known SSRC and payload type
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=1234, payload_type=96)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=2345, payload_type=97)), receiver1
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=3456, payload_type=96)), receiver2
        )
        self.assertEqual(
            router.route_rtp(RtpPacket(ssrc=4567, payload_type=97)), receiver2
        )

        # unknown SSRC, ambiguous payload type
        self.assertEqual(router.route_rtp(RtpPacket(ssrc=5678, payload_type=96)), None)
        self.assertEqual(router.route_rtp(RtpPacket(ssrc=5678, payload_type=97)), None)
