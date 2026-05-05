from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Protocol

from .rate import RemoteBitrateEstimator
from .rtp import RtcpPsfbPacket, RtpPacket, RTCP_PSFB_APP, pack_remb_fci


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

    def register_receiver(self, receiver: CongestionControlledReceiver) -> None:
        if receiver.kind == "video":
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
        if not self.__senders:
            return

        for ssrc in ssrcs:
            if ssrc in self.__senders:
                self.__estimates[ssrc] = (bitrate, now_ms)

        target = self.__derive_session_target(now_ms)
        if target is None:
            return

        self.__session_target_bitrate = self.__smooth_target(
            self.__session_target_bitrate, target
        )
        self.__recompute_allocation()

    def observe_incoming_rtp(
        self, receiver: CongestionControlledReceiver, packet: RtpPacket, arrival_time_ms: int
    ) -> Optional[RtcpPsfbPacket]:
        if receiver.kind != "video" or packet.extensions.abs_send_time is None:
            return None

        remb = self.__remote_bitrate_estimator.add(
            abs_send_time=packet.extensions.abs_send_time,
            arrival_time_ms=arrival_time_ms,
            payload_size=len(packet.payload) + packet.padding_size,
            ssrc=packet.ssrc,
        )
        if remb is None:
            return None

        feedback_ssrc = self.__get_feedback_ssrc()
        if feedback_ssrc is None:
            return None

        bitrate, ssrcs = remb

        return RtcpPsfbPacket(
            fmt=RTCP_PSFB_APP,
            ssrc=feedback_ssrc,
            media_ssrc=0,
            fci=pack_remb_fci(bitrate, ssrcs),
        )

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
            state.sender._ssrc: state.max_bitrate - state.min_bitrate for state in states
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
                for _, state in sorted(remainders, key=lambda item: item[0], reverse=True):
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

    @staticmethod
    def __smooth_target(
        previous: Optional[int], latest: int
    ) -> int:
        if previous is None:
            return latest
        if latest < previous:
            return max(latest, int(previous * 0.85))

        growth = max(50_000, int(previous * 0.08))
        return min(latest, previous + growth)
