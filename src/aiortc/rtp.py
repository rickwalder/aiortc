import math
import struct
from dataclasses import dataclass, field
from struct import pack, unpack
from typing import Any, Optional, Union

from av import AudioFrame
from pyrtcp import (
    Goodbye,
    ReceiverReport,
    ReceptionReportBlock,
    RtcpHeader,
    RtcpPacketRegistry,
    SdesChunk,
    SdesItem,
    SenderInfo,
    SenderReport,
    SourceDescription,
)
from pyrtcp import (
    pack_remb_fci as pack_wire_remb_fci,
)
from pyrtcp import (
    unpack_remb_fci as unpack_wire_remb_fci,
)
from pyrtp import (
    HeaderExtensionElement,
    RtpError,
    unwrap_rtx_payload,
    wrap_rtx_payload,
)
from pyrtp import (
    RtpPacket as WireRtpPacket,
)
from pyrtp import (
    parse_header_extensions as parse_wire_header_extensions,
)
from pyrtp import (
    serialize_header_extensions as serialize_wire_header_extensions,
)

from .rtcrtpparameters import RTCRtpParameters

# used for NACK and retransmission
RTP_HISTORY_SIZE = 128

# reserved to avoid confusion with RTCP
FORBIDDEN_PAYLOAD_TYPES = range(72, 77)
DYNAMIC_PAYLOAD_TYPES = range(96, 128)

RTP_HEADER_LENGTH = 12
RTCP_HEADER_LENGTH = 4

PACKETS_LOST_MIN = -(1 << 23)
PACKETS_LOST_MAX = (1 << 23) - 1

RTCP_SR = 200
RTCP_RR = 201
RTCP_SDES = 202
RTCP_BYE = 203
RTCP_RTPFB = 205
RTCP_PSFB = 206

RTCP_RTPFB_NACK = 1
RTCP_RTPFB_TRANSPORT_CC = 15

RTCP_PSFB_PLI = 1
RTCP_PSFB_SLI = 2
RTCP_PSFB_RPSI = 3
RTCP_PSFB_FIR = 4
RTCP_PSFB_APP = 15

TRANSPORT_CC_URI = (
    "http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01"
)


@dataclass
class HeaderExtensions:
    abs_send_time: Optional[int] = None
    audio_level: Any = None
    mid: Any = None
    repaired_rtp_stream_id: Any = None
    rtp_stream_id: Any = None
    transmission_offset: Optional[int] = None
    transport_sequence_number: Optional[int] = None


class HeaderExtensionsMap:
    def __init__(self) -> None:
        self.__ids = HeaderExtensions()

    @property
    def has_transport_sequence_number(self) -> bool:
        return self.__ids.transport_sequence_number is not None

    def configure(self, parameters: RTCRtpParameters) -> None:
        for ext in parameters.headerExtensions:
            if ext.uri == "urn:ietf:params:rtp-hdrext:sdes:mid":
                self.__ids.mid = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:sdes:repaired-rtp-stream-id":
                self.__ids.repaired_rtp_stream_id = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id":
                self.__ids.rtp_stream_id = ext.id
            elif (
                ext.uri == "http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time"
            ):
                self.__ids.abs_send_time = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:toffset":
                self.__ids.transmission_offset = ext.id
            elif ext.uri == "urn:ietf:params:rtp-hdrext:ssrc-audio-level":
                self.__ids.audio_level = ext.id
            elif ext.uri == TRANSPORT_CC_URI:
                self.__ids.transport_sequence_number = ext.id

    def get(self, extension_profile: int, extension_value: bytes) -> HeaderExtensions:
        values = HeaderExtensions()
        for x_id, x_value in unpack_header_extensions(
            extension_profile, extension_value
        ):
            if x_id == self.__ids.mid:
                values.mid = x_value.decode("utf8")
            elif x_id == self.__ids.repaired_rtp_stream_id:
                values.repaired_rtp_stream_id = x_value.decode("ascii")
            elif x_id == self.__ids.rtp_stream_id:
                values.rtp_stream_id = x_value.decode("ascii")
            elif x_id == self.__ids.abs_send_time:
                values.abs_send_time = unpack("!L", b"\00" + x_value)[0]
            elif x_id == self.__ids.transmission_offset:
                values.transmission_offset = unpack("!l", x_value + b"\00")[0] >> 8
            elif x_id == self.__ids.audio_level:
                vad_level = unpack("!B", x_value)[0]
                values.audio_level = (vad_level & 0x80 == 0x80, vad_level & 0x7F)
            elif x_id == self.__ids.transport_sequence_number:
                values.transport_sequence_number = unpack("!H", x_value)[0]
        return values

    def set(self, values: HeaderExtensions) -> tuple[int, bytes]:
        extensions = []
        if values.mid is not None and self.__ids.mid:
            extensions.append((self.__ids.mid, values.mid.encode("utf8")))
        if (
            values.repaired_rtp_stream_id is not None
            and self.__ids.repaired_rtp_stream_id
        ):
            extensions.append(
                (
                    self.__ids.repaired_rtp_stream_id,
                    values.repaired_rtp_stream_id.encode("ascii"),
                )
            )
        if values.rtp_stream_id is not None and self.__ids.rtp_stream_id:
            extensions.append(
                (self.__ids.rtp_stream_id, values.rtp_stream_id.encode("ascii"))
            )
        if values.abs_send_time is not None and self.__ids.abs_send_time:
            extensions.append(
                (self.__ids.abs_send_time, pack("!L", values.abs_send_time)[1:])
            )
        if values.transmission_offset is not None and self.__ids.transmission_offset:
            extensions.append(
                (
                    self.__ids.transmission_offset,
                    pack("!l", values.transmission_offset << 8)[0:2],
                )
            )
        if values.audio_level is not None and self.__ids.audio_level:
            extensions.append(
                (
                    self.__ids.audio_level,
                    pack(
                        "!B",
                        (0x80 if values.audio_level[0] else 0)
                        | (values.audio_level[1] & 0x7F),
                    ),
                )
            )
        if (
            values.transport_sequence_number is not None
            and self.__ids.transport_sequence_number
        ):
            extensions.append(
                (
                    self.__ids.transport_sequence_number,
                    pack("!H", values.transport_sequence_number),
                )
            )
        return pack_header_extensions(extensions)


def clamp_packets_lost(count: int) -> int:
    return max(PACKETS_LOST_MIN, min(count, PACKETS_LOST_MAX))


def pack_packets_lost(count: int) -> bytes:
    return pack("!l", count)[1:]


def unpack_packets_lost(d: bytes) -> int:
    if d[0] & 0x80:
        d = b"\xff" + d
    else:
        d = b"\x00" + d
    return unpack("!l", d)[0]


def pack_rtcp_packet(packet_type: int, count: int, payload: bytes) -> bytes:
    assert len(payload) % 4 == 0
    return pack("!BBH", (2 << 6) | count, packet_type, len(payload) // 4) + payload


def pack_remb_fci(bitrate: int, ssrcs: list[int]) -> bytes:
    """
    Pack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    return pack_wire_remb_fci(bitrate, ssrcs)


def unpack_remb_fci(data: bytes) -> tuple[int, list[int]]:
    """
    Unpack the FCI for a Receiver Estimated Maximum Bitrate report.

    https://tools.ietf.org/html/draft-alvestrand-rmcat-remb-03
    """
    return unpack_wire_remb_fci(data)


def is_rtcp(msg: bytes) -> bool:
    return len(msg) >= 2 and msg[1] >= 192 and msg[1] <= 208


def padl(length: int) -> int:
    """
    Return amount of padding needed for a 4-byte multiple.
    """
    return 4 * ((length + 3) // 4) - length


def unpack_header_extensions(
    extension_profile: int, extension_value: bytes | None
) -> list[tuple[int, bytes]]:
    """
    Parse header extensions according to RFC 5285.
    """
    if extension_value is None:
        return []
    try:
        return [
            (element.identifier, element.value)
            for element in parse_wire_header_extensions(
                extension_profile, extension_value
            )
        ]
    except RtpError as exc:
        message = str(exc)
        if message == "RTP two-byte header extension length is truncated":
            message = "RTP two-byte header extension is truncated"
        raise ValueError(message) from exc


def pack_header_extensions(extensions: list[tuple[int, bytes]]) -> tuple[int, bytes]:
    """
    Serialize header extensions according to RFC 5285.
    """
    try:
        return serialize_wire_header_extensions(
            [
                HeaderExtensionElement(identifier=x_id, value=x_value)
                for x_id, x_value in extensions
            ]
        )
    except RtpError as exc:
        raise ValueError(str(exc)) from exc


def compute_audio_level_dbov(frame: AudioFrame) -> int:
    """
    Compute the energy level as spelled out in RFC 6465, Appendix A.
    """
    MAX_SAMPLE_VALUE = 32767
    MAX_AUDIO_LEVEL = 0
    MIN_AUDIO_LEVEL = -127
    rms = 0.0
    buf = bytes(frame.planes[0])
    s = struct.Struct("h")
    for unpacked in s.iter_unpack(buf):
        sample = unpacked[0]
        rms += sample * sample
    rms = math.sqrt(rms / (frame.samples * MAX_SAMPLE_VALUE * MAX_SAMPLE_VALUE))
    if rms > 0:
        db = 20 * math.log10(rms)
        db = max(db, MIN_AUDIO_LEVEL)
        db = min(db, MAX_AUDIO_LEVEL)
    else:
        db = MIN_AUDIO_LEVEL
    return round(db)


@dataclass
class RtcpReceiverInfo:
    ssrc: int
    fraction_lost: int
    packets_lost: int
    highest_sequence: int
    jitter: int
    lsr: int
    dlsr: int

    def _to_pyrtcp(self) -> ReceptionReportBlock:
        return ReceptionReportBlock(
            ssrc=self.ssrc,
            fraction_lost=self.fraction_lost,
            packets_lost=self.packets_lost,
            highest_sequence=self.highest_sequence,
            jitter=self.jitter,
            lsr=self.lsr,
            dlsr=self.dlsr,
        )

    def __bytes__(self) -> bytes:
        return self._to_pyrtcp().serialize()

    @classmethod
    def parse(cls, data: bytes) -> "RtcpReceiverInfo":
        report = ReceptionReportBlock.parse(data)
        return cls(
            ssrc=report.ssrc,
            fraction_lost=report.fraction_lost,
            packets_lost=report.packets_lost,
            highest_sequence=report.highest_sequence,
            jitter=report.jitter,
            lsr=report.lsr,
            dlsr=report.dlsr,
        )


@dataclass
class RtcpSenderInfo:
    ntp_timestamp: int
    rtp_timestamp: int
    packet_count: int
    octet_count: int

    def _to_pyrtcp(self) -> SenderInfo:
        return SenderInfo(
            ntp_timestamp_msw=(self.ntp_timestamp >> 32) & 0xFFFFFFFF,
            ntp_timestamp_lsw=self.ntp_timestamp & 0xFFFFFFFF,
            rtp_timestamp=self.rtp_timestamp,
            packet_count=self.packet_count,
            octet_count=self.octet_count,
        )

    def __bytes__(self) -> bytes:
        return self._to_pyrtcp().serialize()

    @classmethod
    def parse(cls, data: bytes) -> "RtcpSenderInfo":
        sender_info = SenderInfo.parse(data)
        return cls(
            ntp_timestamp=(
                (sender_info.ntp_timestamp_msw << 32)
                | sender_info.ntp_timestamp_lsw
            ),
            rtp_timestamp=sender_info.rtp_timestamp,
            packet_count=sender_info.packet_count,
            octet_count=sender_info.octet_count,
        )


@dataclass
class RtcpSourceInfo:
    ssrc: int
    items: list[tuple[Any, bytes]]

    def _to_pyrtcp(self) -> SdesChunk:
        return SdesChunk(
            ssrc=self.ssrc,
            items=[
                SdesItem(item_type=int(item_type), value=item_value)
                for item_type, item_value in self.items
            ],
        )


@dataclass
class RtcpByePacket:
    sources: list[int]

    def __bytes__(self) -> bytes:
        return Goodbye(sources=self.sources).serialize()

    @classmethod
    def parse(cls, data: bytes, count: int) -> "RtcpByePacket":
        if len(data) < 4 * count:
            raise ValueError("RTCP bye length is invalid")
        bye = Goodbye.parse(
            header=RtcpHeader.for_payload(
                packet_type=RTCP_BYE,
                count=count,
                payload_size_bytes=len(data),
            ),
            payload=data,
        )
        return cls(sources=bye.sources)


@dataclass
class RtcpPsfbPacket:
    """
    Payload-Specific Feedback Message (RFC 4585).
    """

    fmt: int
    ssrc: int
    media_ssrc: int
    fci: bytes = b""

    def __bytes__(self) -> bytes:
        payload = pack("!LL", self.ssrc, self.media_ssrc) + self.fci
        return pack_rtcp_packet(RTCP_PSFB, self.fmt, payload)

    @classmethod
    def parse(cls, data: bytes, fmt: int) -> "RtcpPsfbPacket":
        if len(data) < 8:
            raise ValueError("RTCP payload-specific feedback length is invalid")

        ssrc, media_ssrc = unpack("!LL", data[0:8])
        fci = data[8:]
        return cls(fmt=fmt, ssrc=ssrc, media_ssrc=media_ssrc, fci=fci)


@dataclass
class RtcpRrPacket:
    ssrc: int
    reports: list[RtcpReceiverInfo] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        return ReceiverReport(
            ssrc=self.ssrc,
            reports=[report._to_pyrtcp() for report in self.reports],
        ).serialize()

    @classmethod
    def parse(cls, data: bytes, count: int) -> "RtcpRrPacket":
        if len(data) != 4 + 24 * count:
            raise ValueError("RTCP receiver report length is invalid")

        report = ReceiverReport.parse(
            header=RtcpHeader.for_payload(
                packet_type=RTCP_RR,
                count=count,
                payload_size_bytes=len(data),
            ),
            payload=data,
        )
        return cls(
            ssrc=report.ssrc,
            reports=[
                RtcpReceiverInfo(
                    ssrc=block.ssrc,
                    fraction_lost=block.fraction_lost,
                    packets_lost=block.packets_lost,
                    highest_sequence=block.highest_sequence,
                    jitter=block.jitter,
                    lsr=block.lsr,
                    dlsr=block.dlsr,
                )
                for block in report.reports
            ],
        )


@dataclass
class RtcpRtpfbPacket:
    """
    Generic RTP Feedback Message (RFC 4585).
    """

    fmt: int
    ssrc: int
    media_ssrc: int

    # generick NACK
    lost: list[int] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        payload = pack("!LL", self.ssrc, self.media_ssrc)
        if self.lost:
            pid = self.lost[0]
            blp = 0
            for p in self.lost[1:]:
                d = p - pid - 1
                if d < 16:
                    blp |= 1 << d
                else:
                    payload += pack("!HH", pid, blp)
                    pid = p
                    blp = 0
            payload += pack("!HH", pid, blp)
        return pack_rtcp_packet(RTCP_RTPFB, self.fmt, payload)

    @classmethod
    def parse(cls, data: bytes, fmt: int) -> "RtcpRtpfbPacket":
        if len(data) < 8 or len(data) % 4:
            raise ValueError("RTCP RTP feedback length is invalid")

        ssrc, media_ssrc = unpack("!LL", data[0:8])
        lost = []
        for pos in range(8, len(data), 4):
            pid, blp = unpack("!HH", data[pos : pos + 4])
            lost.append(pid)
            for d in range(0, 16):
                if (blp >> d) & 1:
                    lost.append(pid + d + 1)
        return cls(fmt=fmt, ssrc=ssrc, media_ssrc=media_ssrc, lost=lost)


@dataclass
class RtcpTransportLayerCcPacket:
    """
    Transport-wide congestion-control feedback (draft-holmer transport-cc).
    """

    feedback: Any

    @property
    def fmt(self) -> int:
        return RTCP_RTPFB_TRANSPORT_CC

    @property
    def ssrc(self) -> int:
        return self.feedback.sender_ssrc

    @property
    def media_ssrc(self) -> int:
        return self.feedback.media_ssrc

    def __bytes__(self) -> bytes:
        return bytes(self.feedback)


@dataclass
class RtcpSdesPacket:
    chunks: list[RtcpSourceInfo] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        return SourceDescription(
            chunks=[chunk._to_pyrtcp() for chunk in self.chunks]
        ).serialize()

    @classmethod
    def parse(cls, data: bytes, count: int) -> "RtcpSdesPacket":
        try:
            sdes = SourceDescription.parse(
                header=RtcpHeader.for_payload(
                    packet_type=RTCP_SDES,
                    count=count,
                    payload_size_bytes=len(data),
                ),
                payload=data,
            )
        except ValueError as exc:
            message = str(exc)
            if message == "SDES chunk is truncated":
                raise ValueError("RTCP SDES source is truncated") from exc
            if message.startswith("SDES item"):
                raise ValueError("RTCP SDES item is truncated") from exc
            raise
        return cls(
            chunks=[
                RtcpSourceInfo(
                    ssrc=chunk.ssrc,
                    items=[
                        (item.item_type, item.value)
                        for item in chunk.items
                    ],
                )
                for chunk in sdes.chunks
            ]
        )


@dataclass
class RtcpSrPacket:
    ssrc: int
    sender_info: RtcpSenderInfo
    reports: list[RtcpReceiverInfo] = field(default_factory=list)

    def __bytes__(self) -> bytes:
        return SenderReport(
            ssrc=self.ssrc,
            sender_info=self.sender_info._to_pyrtcp(),
            reports=[report._to_pyrtcp() for report in self.reports],
        ).serialize()

    @classmethod
    def parse(cls, data: bytes, count: int) -> "RtcpSrPacket":
        if len(data) != 24 + 24 * count:
            raise ValueError("RTCP sender report length is invalid")

        report = SenderReport.parse(
            header=RtcpHeader.for_payload(
                packet_type=RTCP_SR,
                count=count,
                payload_size_bytes=len(data),
            ),
            payload=data,
        )
        return RtcpSrPacket(
            ssrc=report.ssrc,
            sender_info=RtcpSenderInfo(
                ntp_timestamp=(
                    (report.sender_info.ntp_timestamp_msw << 32)
                    | report.sender_info.ntp_timestamp_lsw
                ),
                rtp_timestamp=report.sender_info.rtp_timestamp,
                packet_count=report.sender_info.packet_count,
                octet_count=report.sender_info.octet_count,
            ),
            reports=[
                RtcpReceiverInfo(
                    ssrc=block.ssrc,
                    fraction_lost=block.fraction_lost,
                    packets_lost=block.packets_lost,
                    highest_sequence=block.highest_sequence,
                    jitter=block.jitter,
                    lsr=block.lsr,
                    dlsr=block.dlsr,
                )
                for block in report.reports
            ],
        )


AnyRtcpPacket = Union[
    RtcpByePacket,
    RtcpPsfbPacket,
    RtcpRrPacket,
    RtcpRtpfbPacket,
    RtcpTransportLayerCcPacket,
    RtcpSdesPacket,
    RtcpSrPacket,
]


class RtcpPacket:
    @classmethod
    def parse(
        cls,
        data: bytes,
        rtcp_registry: RtcpPacketRegistry | None = None,
    ) -> list[AnyRtcpPacket]:
        pos = 0
        packets: list[AnyRtcpPacket] = []

        while pos < len(data):
            if len(data) < pos + RTCP_HEADER_LENGTH:
                raise ValueError(
                    f"RTCP packet length is less than {RTCP_HEADER_LENGTH} bytes"
                )

            v_p_count, packet_type, length = unpack("!BBH", data[pos : pos + 4])
            version = v_p_count >> 6
            padding = (v_p_count >> 5) & 1
            count = v_p_count & 0x1F
            if version != 2:
                raise ValueError("RTCP packet has invalid version")
            pos += 4

            end = pos + length * 4
            if len(data) < end:
                raise ValueError("RTCP packet is truncated")
            payload = data[pos:end]
            pos = end

            padding_bytes = b""
            if padding:
                if not payload or not payload[-1] or payload[-1] > len(payload):
                    raise ValueError("RTCP packet padding length is invalid")
                padding_bytes = payload[-payload[-1] :]
                payload = payload[0 : -payload[-1]]

            codec = None
            if rtcp_registry is not None:
                codec = rtcp_registry.codec_for(packet_type, count)
            if codec is not None:
                packets.append(
                    codec.parse_rtcp(
                        header=RtcpHeader(
                            count=count,
                            packet_type=packet_type,
                            length=length,
                            padding=bool(padding),
                        ),
                        payload=payload,
                        padding=padding_bytes,
                    )
                )
                continue

            if packet_type == RTCP_BYE:
                packets.append(RtcpByePacket.parse(payload, count))
            elif packet_type == RTCP_SDES:
                packets.append(RtcpSdesPacket.parse(payload, count))
            elif packet_type == RTCP_SR:
                packets.append(RtcpSrPacket.parse(payload, count))
            elif packet_type == RTCP_RR:
                packets.append(RtcpRrPacket.parse(payload, count))
            elif packet_type == RTCP_RTPFB:
                packets.append(RtcpRtpfbPacket.parse(payload, count))
            elif packet_type == RTCP_PSFB:
                packets.append(RtcpPsfbPacket.parse(payload, count))

        return packets


class RtpPacket:
    def __init__(
        self,
        payload_type: int = 0,
        marker: int = 0,
        sequence_number: int = 0,
        timestamp: int = 0,
        ssrc: int = 0,
        payload: bytes = b"",
    ) -> None:
        self.version = 2
        self.marker = marker
        self.payload_type = payload_type
        self.sequence_number = sequence_number
        self.timestamp = timestamp
        self.ssrc = ssrc
        self.csrc: list[int] = []
        self.extensions = HeaderExtensions()
        self.payload = payload
        self.padding_size = 0

    def __repr__(self) -> str:
        return (
            f"RtpPacket(seq={self.sequence_number}, ts={self.timestamp}, "
            f"marker={self.marker}, payload={self.payload_type}, "
            f"{len(self.payload)} bytes)"
        )

    @classmethod
    def parse(
        cls, data: bytes, extensions_map: HeaderExtensionsMap = HeaderExtensionsMap()
    ) -> "RtpPacket":
        if len(data) < RTP_HEADER_LENGTH:
            raise ValueError(
                f"RTP packet length is less than {RTP_HEADER_LENGTH} bytes"
            )

        try:
            wire_packet = WireRtpPacket.parse(data)
        except RtpError as exc:
            raise ValueError(cls.__wire_parse_error_message(data, exc)) from exc

        packet = cls(
            marker=int(wire_packet.marker),
            payload_type=wire_packet.payload_type,
            sequence_number=wire_packet.sequence_number,
            timestamp=wire_packet.timestamp,
            ssrc=wire_packet.ssrc,
            payload=wire_packet.payload,
        )
        packet.csrc = wire_packet.csrc
        packet.padding_size = wire_packet.padding_size
        packet.extensions = extensions_map.get(
            wire_packet.extension_profile, wire_packet.extension_value
        )

        return packet

    def serialize(
        self, extensions_map: HeaderExtensionsMap = HeaderExtensionsMap()
    ) -> bytes:
        extension_profile, extension_value = extensions_map.set(self.extensions)
        try:
            return bytes(
                WireRtpPacket(
                    payload_type=self.payload_type,
                    marker=bool(self.marker),
                    sequence_number=self.sequence_number,
                    timestamp=self.timestamp,
                    ssrc=self.ssrc,
                    csrc=self.csrc,
                    extension_profile=extension_profile,
                    extension_value=extension_value,
                    payload=self.payload,
                    padding_size=self.padding_size,
                )
            )
        except RtpError as exc:
            raise ValueError(str(exc)) from exc

    @staticmethod
    def __wire_parse_error_message(data: bytes, exc: RtpError) -> str:
        message = str(exc)
        if message == "RTP packet has invalid version":
            return message
        if message == "RTP padding length is invalid":
            return "RTP packet padding length is invalid"
        if len(data) >= RTP_HEADER_LENGTH:
            csrc_count = data[0] & 0x0F
            if len(data) < RTP_HEADER_LENGTH + 4 * csrc_count:
                return "RTP packet has truncated CSRC"
        if message == "RTP header extension is truncated":
            return "RTP packet has truncated extension profile / length"
        if message == "RTP header extension value is truncated":
            return "RTP packet has truncated extension value"
        return message


def unwrap_rtx(rtx: RtpPacket, payload_type: int, ssrc: int) -> RtpPacket:
    """
    Recover initial packet from a retransmission packet.
    """
    sequence_number, payload = unwrap_rtx_payload(rtx.payload)
    packet = RtpPacket(
        payload_type=payload_type,
        marker=rtx.marker,
        sequence_number=sequence_number,
        timestamp=rtx.timestamp,
        ssrc=ssrc,
        payload=payload,
    )
    packet.csrc = rtx.csrc
    packet.extensions = rtx.extensions
    return packet


def wrap_rtx(
    packet: RtpPacket, payload_type: int, sequence_number: int, ssrc: int
) -> RtpPacket:
    """
    Create a retransmission packet from a lost packet.
    """
    rtx = RtpPacket(
        payload_type=payload_type,
        marker=packet.marker,
        sequence_number=sequence_number,
        timestamp=packet.timestamp,
        ssrc=ssrc,
        payload=wrap_rtx_payload(packet.sequence_number, packet.payload),
    )
    rtx.csrc = packet.csrc
    rtx.extensions = packet.extensions
    return rtx
