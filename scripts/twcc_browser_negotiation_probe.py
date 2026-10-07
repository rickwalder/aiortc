#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import webbrowser

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from pycc import TRANSPORT_CC_URI

TRANSPORT_CC_FEEDBACK = "transport-cc"


INDEX_HTML = """
<!doctype html>
<meta charset="utf-8">
<title>TWCC negotiation probe</title>
<pre id="log"></pre>
<script>
const log = (...args) => {
  document.getElementById("log").textContent += args.join(" ") + "\\n";
  console.log(...args);
};

async function waitIceGatheringComplete(pc) {
  if (pc.iceGatheringState === "complete") return;
  await new Promise(resolve => {
    pc.addEventListener("icegatheringstatechange", () => {
      if (pc.iceGatheringState === "complete") resolve();
    });
  });
}

async function main() {
  const pc = new RTCPeerConnection();
  pc.addTransceiver("video", {direction: "sendrecv"});
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitIceGatheringComplete(pc);

  const response = await fetch("/offer", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(pc.localDescription),
  });
  const answer = await response.json();
  await pc.setRemoteDescription(answer);

  const offerSdp = pc.localDescription.sdp;
  const answerSdp = pc.remoteDescription.sdp;
  log("browser offer extmap transport-cc:", offerSdp.includes("%(uri)s"));
  log("browser offer rtcp-fb transport-cc:", offerSdp.includes("%(feedback)s"));
  log("aiortc answer extmap transport-cc:", answerSdp.includes("%(uri)s"));
  log("aiortc answer rtcp-fb transport-cc:", answerSdp.includes("%(feedback)s"));
}

main().catch(err => log(err.stack || err));
</script>
""" % {
    "uri": TRANSPORT_CC_URI,
    "feedback": TRANSPORT_CC_FEEDBACK,
}


def has_transport_cc(sdp: str) -> tuple[bool, bool]:
    return TRANSPORT_CC_URI in sdp, TRANSPORT_CC_FEEDBACK in sdp


async def index(request: web.Request) -> web.Response:
    return web.Response(text=INDEX_HTML, content_type="text/html")


async def offer(request: web.Request) -> web.Response:
    params = await request.json()
    pc = RTCPeerConnection()
    request.app["pcs"].add(pc)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    offer_ext, offer_fb = has_transport_cc(params["sdp"])
    answer_ext, answer_fb = has_transport_cc(pc.localDescription.sdp)
    print(f"browser offer extmap transport-cc: {offer_ext}")
    print(f"browser offer rtcp-fb transport-cc: {offer_fb}")
    print(f"aiortc answer extmap transport-cc: {answer_ext}")
    print(f"aiortc answer rtcp-fb transport-cc: {answer_fb}")

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def cleanup(app: web.Application) -> None:
    await asyncio.gather(*(pc.close() for pc in app["pcs"]), return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a browser page that probes TWCC SDP negotiation."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    app = web.Application()
    app["pcs"] = set()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_cleanup.append(cleanup)

    url = f"http://{args.host}:{args.port}/"
    if not args.no_open:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    print(f"Open {url}")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
