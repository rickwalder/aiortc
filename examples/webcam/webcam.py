import argparse
import asyncio
import json
import logging
import os
import platform
import random
import ssl
from typing import Awaitable, Callable, Optional

from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaPlayer, MediaRelay
from aiortc.rtp import is_rtcp

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("webcam")
pcs = set()
relay = None
webcam = None


def create_local_tracks(
    play_from: str, decode: bool
) -> tuple[Optional[MediaStreamTrack], Optional[MediaStreamTrack]]:
    global relay, webcam

    if play_from:
        # If a file name was given, play from that file.
        player = MediaPlayer(play_from, decode=decode)
        return player.audio, player.video
    else:
        # Otherwise, play from the system's default webcam.
        #
        # In order to serve the same webcam to multiple users we make use of
        # a `MediaRelay`. The webcam will stay open, so it is our responsability
        # to stop the webcam when the application shuts down in `on_shutdown`.
        options = {"framerate": "30", "video_size": "640x480"}
        if relay is None:
            if platform.system() == "Darwin":
                webcam = MediaPlayer(
                    "default:none", format="avfoundation", options=options
                )
            elif platform.system() == "Windows":
                webcam = MediaPlayer(
                    "video=Integrated Camera", format="dshow", options=options
                )
            else:
                webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)
            relay = MediaRelay()
        return None, relay.subscribe(webcam.video)


def force_codec(pc: RTCPeerConnection, sender: RTCRtpSender, forced_codec: str) -> None:
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


def install_outbound_rtp_loss(
    sender: RTCRtpSender,
    *,
    loss_percent: float,
    start_after: float,
    duration: Optional[float],
    seed: Optional[int],
) -> None:
    if loss_percent <= 0:
        return

    original_send_rtp: Callable[[bytes], Awaitable[None]] = sender.transport._send_rtp
    random_source = random.Random(seed)
    drop_probability = loss_percent / 100
    first_rtp_time: Optional[float] = None
    packets = 0
    dropped = 0

    async def send_rtp_with_loss(data: bytes) -> None:
        nonlocal dropped, first_rtp_time, packets

        if is_rtcp(data):
            await original_send_rtp(data)
            return

        loop_time = asyncio.get_running_loop().time()
        if first_rtp_time is None:
            first_rtp_time = loop_time
        elapsed = loop_time - first_rtp_time
        active = elapsed >= start_after and (
            duration is None or elapsed < start_after + duration
        )

        packets += 1
        if active and random_source.random() < drop_probability:
            dropped += 1
            if dropped <= 10 or dropped % 50 == 0:
                logger.info(
                    "dropping outbound RTP packet %d/%d at %.3fs",
                    dropped,
                    packets,
                    elapsed,
                )
            return

        await original_send_rtp(data)

    sender.transport._send_rtp = send_rtp_with_loss  # type: ignore[method-assign]
    logger.info(
        "installed outbound RTP loss: %.2f%% after %.2fs for %s",
        loss_percent,
        start_after,
        "the rest of the call" if duration is None else f"{duration:.2f}s",
    )


async def index(request: web.Request) -> web.Response:
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request: web.Request) -> web.Response:
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request: web.Request) -> web.Response:
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    audio, video = create_local_tracks(
        args.play_from, decode=not args.play_without_decoding
    )

    if audio:
        audio_sender = pc.addTrack(audio)
        if args.audio_codec:
            force_codec(pc, audio_sender, args.audio_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the audio codec using --audio-codec")

    if video:
        video_sender = pc.addTrack(video)
        if args.video_codec:
            force_codec(pc, video_sender, args.video_codec)
        elif args.play_without_decoding:
            raise Exception("You must specify the video codec using --video-codec")
        install_outbound_rtp_loss(
            video_sender,
            loss_percent=args.video_loss_percent,
            start_after=args.video_loss_start,
            duration=args.video_loss_duration,
            seed=args.video_loss_seed,
        )

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app: web.Application) -> None:
    # Close peer connections.
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

    # If a shared webcam was opened, stop it.
    if webcam is not None:
        webcam.video.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument(
        "--audio-codec", help="Force a specific audio codec (e.g. audio/opus)"
    )
    parser.add_argument(
        "--video-codec", help="Force a specific video codec (e.g. video/H264)"
    )
    parser.add_argument(
        "--video-loss-percent",
        type=float,
        default=0.0,
        help="Drop this percentage of outbound video RTP packets",
    )
    parser.add_argument(
        "--video-loss-start",
        type=float,
        default=5.0,
        help="Start dropping outbound video RTP this many seconds after first RTP",
    )
    parser.add_argument(
        "--video-loss-duration",
        type=float,
        help="Stop dropping outbound video RTP after this many seconds",
    )
    parser.add_argument(
        "--video-loss-seed",
        type=int,
        help="Seed for deterministic outbound video RTP loss",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)
