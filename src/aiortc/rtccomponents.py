from __future__ import annotations

from dataclasses import dataclass

from rtc_types import RtcCapabilities, RtcCapabilityProvider, collect_capabilities

from .rtcrtpparameters import RTCRtcpFeedback, RTCRtpHeaderExtensionParameters


@dataclass(frozen=True)
class CongestionControlCapabilities:
    rtcp_feedback: list[RTCRtcpFeedback]
    rtp_header_extensions: list[RTCRtpHeaderExtensionParameters]
    rtcp_feedback_formats: list[tuple[int, int]]


DEFAULT_CONGESTION_CONTROL_COMPONENTS: tuple[RtcCapabilityProvider, ...] = ()


def normalize_congestion_control_components(
    components: list[RtcCapabilityProvider] | tuple[RtcCapabilityProvider, ...] | None,
) -> tuple[RtcCapabilityProvider, ...]:
    return tuple(components or ())


def get_rtc_capabilities(
    components: list[RtcCapabilityProvider] | tuple[RtcCapabilityProvider, ...],
    kind: str,
) -> RtcCapabilities:
    return collect_capabilities(components, kind)


def adapt_rtc_capabilities(
    capabilities: RtcCapabilities,
) -> CongestionControlCapabilities:
    return CongestionControlCapabilities(
        rtcp_feedback=[
            RTCRtcpFeedback(
                type=feedback.type,
                parameter=feedback.parameter,
            )
            for feedback in capabilities.rtcp_feedback
        ],
        rtp_header_extensions=[
            RTCRtpHeaderExtensionParameters(
                id=extension.preferred_id,
                uri=extension.uri,
            )
            for extension in capabilities.rtp_header_extensions
        ],
        rtcp_feedback_formats=[
            (packet.packet_type, packet.fmt)
            for packet in capabilities.rtcp_feedback_packets
        ],
    )


def get_congestion_control_capabilities(
    kind: str,
    components: list[RtcCapabilityProvider]
    | tuple[RtcCapabilityProvider, ...]
    | None = None,
) -> CongestionControlCapabilities:
    return adapt_rtc_capabilities(
        get_rtc_capabilities(
            normalize_congestion_control_components(components),
            kind,
        )
    )


__all__ = [
    "CongestionControlCapabilities",
    "DEFAULT_CONGESTION_CONTROL_COMPONENTS",
    "RtcCapabilityProvider",
    "adapt_rtc_capabilities",
    "get_congestion_control_capabilities",
    "get_rtc_capabilities",
    "normalize_congestion_control_components",
]
