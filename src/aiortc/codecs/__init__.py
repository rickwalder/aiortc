from typing import Optional, Union

from rtc_types import RtcCapabilityProvider

from ..rtccomponents import (
    DEFAULT_CONGESTION_CONTROL_COMPONENTS,
    get_congestion_control_capabilities,
)
from ..rtcrtpparameters import (
    ParametersDict,
    RTCRtcpFeedback,
    RTCRtpCapabilities,
    RTCRtpCodecCapability,
    RTCRtpCodecParameters,
    RTCRtpHeaderExtensionCapability,
    RTCRtpHeaderExtensionParameters,
)
from .base import Decoder, Encoder
from .g711 import PcmaDecoder, PcmaEncoder, PcmuDecoder, PcmuEncoder
from .g722 import G722Decoder, G722Encoder
from .h264 import H264Decoder, H264Encoder, h264_depayload
from .opus import OpusDecoder, OpusEncoder
from .vpx import Vp8Decoder, Vp8Encoder, vp8_depayload

# The clockrate for G.722 is 8kHz even though the sampling rate is 16kHz.
# See https://datatracker.ietf.org/doc/html/rfc3551
G722_CODEC = RTCRtpCodecParameters(
    mimeType="audio/G722", clockRate=8000, channels=1, payloadType=9
)
PCMU_CODEC = RTCRtpCodecParameters(
    mimeType="audio/PCMU", clockRate=8000, channels=1, payloadType=0
)
PCMA_CODEC = RTCRtpCodecParameters(
    mimeType="audio/PCMA", clockRate=8000, channels=1, payloadType=8
)

CODECS: dict[str, list[RTCRtpCodecParameters]] = {
    "audio": [
        RTCRtpCodecParameters(
            mimeType="audio/opus", clockRate=48000, channels=2, payloadType=96
        ),
        G722_CODEC,
        PCMU_CODEC,
        PCMA_CODEC,
    ],
    "video": [],
}
# Note, the id space for these extensions is shared across media types when BUNDLE
# is negotiated. If you add a audio- or video-specific extension, make sure it has
# a unique id.
HEADER_EXTENSIONS: dict[str, list[RTCRtpHeaderExtensionParameters]] = {
    "audio": [
        RTCRtpHeaderExtensionParameters(
            id=1, uri="urn:ietf:params:rtp-hdrext:sdes:mid"
        ),
        RTCRtpHeaderExtensionParameters(
            id=2, uri="urn:ietf:params:rtp-hdrext:ssrc-audio-level"
        ),
    ],
    "video": [],
}


def _build_video_header_extensions(
    congestion_control: tuple[RtcCapabilityProvider, ...]
    | list[RtcCapabilityProvider] = DEFAULT_CONGESTION_CONTROL_COMPONENTS,
) -> list[RTCRtpHeaderExtensionParameters]:
    congestion_capabilities = get_congestion_control_capabilities(
        "video",
        components=congestion_control,
    )
    return [
        RTCRtpHeaderExtensionParameters(
            id=1, uri="urn:ietf:params:rtp-hdrext:sdes:mid"
        ),
        *congestion_capabilities.rtp_header_extensions,
    ]


def _build_video_codecs(
    congestion_control: tuple[RtcCapabilityProvider, ...]
    | list[RtcCapabilityProvider] = DEFAULT_CONGESTION_CONTROL_COMPONENTS,
) -> list[RTCRtpCodecParameters]:
    codecs = []
    dynamic_pt = 97
    congestion_capabilities = get_congestion_control_capabilities(
        "video",
        components=congestion_control,
    )

    def add_video_codec(
        mimeType: str, parameters: Optional[ParametersDict] = None
    ) -> None:
        nonlocal dynamic_pt
        clockRate = 90000
        codecs.extend(
            [
                RTCRtpCodecParameters(
                    mimeType=mimeType,
                    clockRate=clockRate,
                    payloadType=dynamic_pt,
                    rtcpFeedback=[
                        RTCRtcpFeedback(type="nack"),
                        RTCRtcpFeedback(type="nack", parameter="pli"),
                        *congestion_capabilities.rtcp_feedback,
                    ],
                    parameters=parameters or {},
                ),
                RTCRtpCodecParameters(
                    mimeType="video/rtx",
                    clockRate=clockRate,
                    payloadType=dynamic_pt + 1,
                    parameters={"apt": dynamic_pt},
                ),
            ]
        )
        dynamic_pt += 2

    add_video_codec("video/VP8")
    for profile_level_id in ("42001f", "42e01f"):
        add_video_codec(
            "video/H264",
            {
                "level-asymmetry-allowed": "1",
                "packetization-mode": "1",
                "profile-level-id": profile_level_id,
            },
        )
    return codecs


def get_codec_parameters(
    kind: str,
    congestion_control: tuple[RtcCapabilityProvider, ...]
    | list[RtcCapabilityProvider] = DEFAULT_CONGESTION_CONTROL_COMPONENTS,
) -> list[RTCRtpCodecParameters]:
    if kind not in CODECS:
        raise ValueError(f"cannot get capabilities for unknown media {kind}")
    if (
        kind == "video"
        and congestion_control is not DEFAULT_CONGESTION_CONTROL_COMPONENTS
    ):
        return _build_video_codecs(congestion_control)
    return CODECS[kind][:]


def get_header_extension_parameters(
    kind: str,
    congestion_control: tuple[RtcCapabilityProvider, ...]
    | list[RtcCapabilityProvider] = DEFAULT_CONGESTION_CONTROL_COMPONENTS,
) -> list[RTCRtpHeaderExtensionParameters]:
    if kind not in HEADER_EXTENSIONS:
        raise ValueError(f"cannot get capabilities for unknown media {kind}")
    if (
        kind == "video"
        and congestion_control is not DEFAULT_CONGESTION_CONTROL_COMPONENTS
    ):
        return _build_video_header_extensions(congestion_control)
    return HEADER_EXTENSIONS[kind][:]


def init_codecs() -> None:
    CODECS["video"] = _build_video_codecs()
    HEADER_EXTENSIONS["video"] = _build_video_header_extensions()

    # Keep the historic module globals authoritative for the default component set.
    # Non-default component sets build their own video codec/header-extension lists
    # through get_codec_parameters / get_header_extension_parameters.


def depayload(codec: RTCRtpCodecParameters, payload: bytes) -> bytes:
    if codec.name == "VP8":
        return vp8_depayload(payload)
    elif codec.name == "H264":
        return h264_depayload(payload)
    else:
        return payload


def get_capabilities(
    kind: str,
    congestion_control: tuple[RtcCapabilityProvider, ...]
    | list[RtcCapabilityProvider] = DEFAULT_CONGESTION_CONTROL_COMPONENTS,
) -> RTCRtpCapabilities:
    if kind not in CODECS:
        raise ValueError(f"cannot get capabilities for unknown media {kind}")

    codecs = []
    rtx_added = False
    for params in get_codec_parameters(kind, congestion_control):
        if not is_rtx(params):
            codecs.append(
                RTCRtpCodecCapability(
                    mimeType=params.mimeType,
                    clockRate=params.clockRate,
                    channels=params.channels,
                    parameters=params.parameters,
                )
            )
        elif not rtx_added:
            # There will only be a single entry in codecs[] for retransmission
            # via RTX, with sdpFmtpLine not present.
            codecs.append(
                RTCRtpCodecCapability(
                    mimeType=params.mimeType, clockRate=params.clockRate
                )
            )
            rtx_added = True

    headerExtensions = []
    for extension in get_header_extension_parameters(
        kind,
        congestion_control,
    ):
        headerExtensions.append(RTCRtpHeaderExtensionCapability(uri=extension.uri))
    return RTCRtpCapabilities(codecs=codecs, headerExtensions=headerExtensions)


def get_decoder(codec: RTCRtpCodecParameters) -> Decoder:
    mimeType = codec.mimeType.lower()

    if mimeType == "audio/g722":
        return G722Decoder()
    elif mimeType == "audio/opus":
        return OpusDecoder()
    elif mimeType == "audio/pcma":
        return PcmaDecoder()
    elif mimeType == "audio/pcmu":
        return PcmuDecoder()
    elif mimeType == "video/h264":
        return H264Decoder()
    elif mimeType == "video/vp8":
        return Vp8Decoder()
    else:
        raise ValueError(f"No decoder found for MIME type `{mimeType}`")


def get_encoder(codec: RTCRtpCodecParameters) -> Encoder:
    mimeType = codec.mimeType.lower()

    if mimeType == "audio/g722":
        return G722Encoder()
    elif mimeType == "audio/opus":
        return OpusEncoder()
    elif mimeType == "audio/pcma":
        return PcmaEncoder()
    elif mimeType == "audio/pcmu":
        return PcmuEncoder()
    elif mimeType == "video/h264":
        return H264Encoder()
    elif mimeType == "video/vp8":
        return Vp8Encoder()
    else:
        raise ValueError(f"No encoder found for MIME type `{mimeType}`")


def is_rtx(codec: Union[RTCRtpCodecCapability, RTCRtpCodecParameters]) -> bool:
    return codec.name.lower() == "rtx"


init_codecs()
