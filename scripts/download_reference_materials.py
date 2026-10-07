#!/usr/bin/env python3
"""
Download reference material for WebRTC TWCC / GCC work.

The output directory is intentionally ignored by git. Re-run this script whenever
we want a fresh local snapshot of specs, docs, and implementation references.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import json
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reference-materials-dont-commit"
USER_AGENT = "aiortc-twcc-gcc-reference-downloader/1.0"


@dataclass(frozen=True)
class Material:
    path: str
    url: str
    title: str
    why: str
    gitiles_text: bool = False


@dataclass(frozen=True)
class GitHubDirectory:
    owner: str
    repo: str
    ref: str
    source_path: str
    output_path: str
    title: str
    why: str


def gitiles_url(path: str, ref: str = "main") -> str:
    quoted_path = quote(path)
    return f"https://webrtc.googlesource.com/src/+show/{ref}/{quoted_path}?format=TEXT"


MATERIALS = [
    Material(
        path="specs/draft-holmer-rmcat-transport-wide-cc-extensions-01.html",
        url="https://datatracker.ietf.org/doc/html/draft-holmer-rmcat-transport-wide-cc-extensions-01",
        title="Transport-wide Congestion Control RTP header extension draft",
        why="Defines the RTP transport sequence number extension and RTCP transport feedback format used for browser TWCC interop.",
    ),
    Material(
        path="specs/draft-ietf-rmcat-gcc-02.pdf",
        url="https://www.ietf.org/archive/id/draft-ietf-rmcat-gcc-02.pdf",
        title="Google Congestion Control draft",
        why="Core GCC algorithm reference: delay-based control, loss-based control, and interaction model.",
    ),
    Material(
        path="specs/draft-alvestrand-rmcat-remb-03.html",
        url="https://datatracker.ietf.org/doc/html/draft-alvestrand-rmcat-remb-03",
        title="REMB draft",
        why="Current aiortc congestion-control baseline and fallback mechanism to compare against TWCC/GCC behavior.",
    ),
    Material(
        path="specs/rfc8888-rtcp-ccfb.html",
        url="https://www.rfc-editor.org/rfc/rfc8888.html",
        title="RFC 8888 RTCP feedback for congestion control",
        why="Standardized RTCP congestion-control feedback. libwebrtc supports it alongside TWCC, but browser interop still commonly depends on TWCC.",
    ),
    Material(
        path="specs/rfc8285-rtp-header-extensions.html",
        url="https://www.rfc-editor.org/rfc/rfc8285.html",
        title="RFC 8285 RTP header extensions",
        why="Base RTP extension mechanism used by abs-send-time and transport-wide sequence numbers.",
    ),
    Material(
        path="specs/rfc8834-webrtc-rtp-usage.html",
        url="https://www.rfc-editor.org/rfc/rfc8834.html",
        title="RFC 8834 WebRTC RTP usage",
        why="WebRTC RTP/RTCP profile guidance, including feedback and RTP extension usage expectations.",
    ),
    Material(
        path="specs/rfc4585-rtp-avpf.html",
        url="https://www.rfc-editor.org/rfc/rfc4585.html",
        title="RFC 4585 RTP/AVPF",
        why="Base RTP feedback profile used for NACK, RTCP feedback timing, and feedback packet structure.",
    ),
    Material(
        path="specs/rfc5761-rtcp-mux.html",
        url="https://www.rfc-editor.org/rfc/rfc5761.html",
        title="RFC 5761 RTP and RTCP mux",
        why="Relevant because aiortc routes RTP and RTCP on the same DTLS transport.",
    ),
    Material(
        path="specs/rfc8843-bundle.html",
        url="https://www.rfc-editor.org/rfc/rfc8843.html",
        title="RFC 8843 BUNDLE",
        why="Relevant to transport-scoped congestion control across bundled media sections.",
    ),
    Material(
        path="w3c/webrtc-stats.html",
        url="https://w3c.github.io/webrtc-stats/webrtc-stats",
        title="W3C WebRTC Stats",
        why="Defines browser-observable bitrate, RTP, and transport stats useful for validating TWCC/GCC behavior.",
    ),
    Material(
        path="w3c/webrtc-pc.html",
        url="https://w3c.github.io/webrtc-pc/",
        title="W3C WebRTC peer connection",
        why="API-level browser behavior reference for senders, receivers, transceivers, and RTCPeerConnection lifecycle.",
    ),
    Material(
        path="libwebrtc/docs/transport-wide-cc-02.md",
        url=gitiles_url("docs/native-code/rtp-hdrext/transport-wide-cc-02/README.md"),
        title="libwebrtc transport-wide-cc-02 doc",
        why="Canonical browser implementation note for TWCC-02 negotiation and RFC 8888 mutual exclusion.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/rtcp_packet/transport_feedback.h",
        url=gitiles_url("modules/rtp_rtcp/source/rtcp_packet/transport_feedback.h"),
        title="libwebrtc transport_feedback.h",
        why="RTCP TWCC packet API and data model.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/rtcp_packet/transport_feedback.cc",
        url=gitiles_url("modules/rtp_rtcp/source/rtcp_packet/transport_feedback.cc"),
        title="libwebrtc transport_feedback.cc",
        why="RTCP TWCC packet parse/serialize behavior.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/rtcp_packet/congestion_control_feedback.h",
        url=gitiles_url(
            "modules/rtp_rtcp/source/rtcp_packet/congestion_control_feedback.h"
        ),
        title="libwebrtc congestion_control_feedback.h",
        why="RFC 8888 CCFB packet API for comparison with TWCC.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/rtcp_packet/congestion_control_feedback.cc",
        url=gitiles_url(
            "modules/rtp_rtcp/source/rtcp_packet/congestion_control_feedback.cc"
        ),
        title="libwebrtc congestion_control_feedback.cc",
        why="RFC 8888 CCFB parse/serialize behavior.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/congestion_controller/rtp/transport_feedback_adapter.h",
        url=gitiles_url(
            "modules/congestion_controller/rtp/transport_feedback_adapter.h"
        ),
        title="libwebrtc transport_feedback_adapter.h",
        why="Adapter from RTCP feedback packets to network controller packet feedback.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/congestion_controller/rtp/transport_feedback_adapter.cc",
        url=gitiles_url(
            "modules/congestion_controller/rtp/transport_feedback_adapter.cc"
        ),
        title="libwebrtc transport_feedback_adapter.cc",
        why="Tracks sent packets and converts TWCC feedback into usable send/receive deltas.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/api/transport/network_control.h",
        url=gitiles_url("api/transport/network_control.h"),
        title="libwebrtc network_control.h",
        why="Network controller interface that GoogCC implements.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/api/transport/network_types.h",
        url=gitiles_url("api/transport/network_types.h"),
        title="libwebrtc network_types.h",
        why="Packet feedback and bitrate control structs passed through the network controller.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/call/rtp_transport_controller_send.h",
        url=gitiles_url("call/rtp_transport_controller_send.h"),
        title="libwebrtc rtp_transport_controller_send.h",
        why="High-level sender-side controller wiring around feedback, pacing, and bitrate allocation.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/call/rtp_transport_controller_send.cc",
        url=gitiles_url("call/rtp_transport_controller_send.cc"),
        title="libwebrtc rtp_transport_controller_send.cc",
        why="Shows how browser-grade sender control consumes TWCC and CCFB feedback.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/goog_cc/goog_cc_network_control.h",
        url=gitiles_url(
            "modules/congestion_controller/goog_cc/goog_cc_network_control.h"
        ),
        title="libwebrtc GoogCcNetworkController header",
        why="Current libwebrtc GCC controller interface and state.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/goog_cc/goog_cc_network_control.cc",
        url=gitiles_url(
            "modules/congestion_controller/goog_cc/goog_cc_network_control.cc"
        ),
        title="libwebrtc GoogCcNetworkController implementation",
        why="Current libwebrtc GCC controller implementation.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/goog_cc/delay_based_bwe.h",
        url=gitiles_url("modules/congestion_controller/goog_cc/delay_based_bwe.h"),
        title="libwebrtc delay_based_bwe.h",
        why="Delay-based bandwidth estimator interface.",
        gitiles_text=True,
    ),
    Material(
        path="libwebrtc/goog_cc/delay_based_bwe.cc",
        url=gitiles_url("modules/congestion_controller/goog_cc/delay_based_bwe.cc"),
        title="libwebrtc delay_based_bwe.cc",
        why="Delay-based bandwidth estimator implementation.",
        gitiles_text=True,
    ),
    Material(
        path="pion/docs/twcc-package.html",
        url="https://pion-interceptor-62.mintlify.app/api/packages/twcc",
        title="Pion TWCC package docs",
        why="Clean non-C++ implementation reference for TWCC sender/receiver interceptors.",
    ),
    Material(
        path="pion/docs/gcc-package.html",
        url="https://pion-interceptor-62.mintlify.app/api/packages/gcc",
        title="Pion GCC package docs",
        why="Clean non-C++ implementation reference for GCC estimator pieces.",
    ),
    Material(
        path="gstreamer/docs/rtpgccbwe.html",
        url="https://gstreamer.freedesktop.org/documentation/rsrtp/rtpgccbwe.html",
        title="GStreamer rtpgccbwe docs",
        why="GStreamer Rust RTP element that implements GCC and depends on TWCC.",
    ),
    Material(
        path="gstreamer/docs/webrtcsink.html",
        url="https://gstreamer.freedesktop.org/documentation/rswebrtc/webrtcsink.html",
        title="GStreamer webrtcsink docs",
        why="Server-side WebRTC reference exposing GCC congestion control as a productized behavior.",
    ),
    Material(
        path="notes/webrtc-for-the-curious-media-communication.html",
        url="https://webrtcforthecurious.com/docs/06-media-communication/",
        title="WebRTC for the Curious media communication",
        why="Readable conceptual overview for RTP feedback, TWCC, REMB, and congestion control.",
    ),
]


GITHUB_DIRECTORIES = [
    GitHubDirectory(
        owner="pion",
        repo="interceptor",
        ref="main",
        source_path="pkg/twcc",
        output_path="pion/interceptor/pkg/twcc",
        title="Pion TWCC source directory",
        why="Go TWCC header-extension and feedback sender/receiver implementation.",
    ),
    GitHubDirectory(
        owner="pion",
        repo="interceptor",
        ref="main",
        source_path="pkg/gcc",
        output_path="pion/interceptor/pkg/gcc",
        title="Pion GCC source directory",
        why="Go GCC estimator implementation with small, readable components and tests.",
    ),
    GitHubDirectory(
        owner="pion",
        repo="interceptor",
        ref="main",
        source_path="pkg/cc",
        output_path="pion/interceptor/pkg/cc",
        title="Pion congestion-control interceptor directory",
        why="Glue layer connecting packet feedback to bandwidth estimators.",
    ),
]


def fetch_url(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def decode_gitiles_text(data: bytes) -> bytes:
    # Gitiles TEXT responses are base64 with optional whitespace.
    return base64.b64decode(b"".join(data.split()), validate=False)


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def fetch_github_directory(directory: GitHubDirectory) -> list[Material]:
    api_url = (
        "https://api.github.com/repos/"
        f"{directory.owner}/{directory.repo}/contents/{directory.source_path}"
        f"?ref={directory.ref}"
    )
    entries = json.loads(fetch_url(api_url).decode("utf8"))
    materials = []
    for entry in entries:
        if entry.get("type") != "file":
            continue

        name = entry["name"]
        if not name.endswith((".go", ".md", ".txt")):
            continue

        source_path = entry["path"]
        output_path = f"{directory.output_path}/{name}"
        materials.append(
            Material(
                path=output_path,
                url=f"https://raw.githubusercontent.com/{directory.owner}/{directory.repo}/{directory.ref}/{source_path}",
                title=f"{directory.title}: {name}",
                why=directory.why,
            )
        )
    return materials


def download_material(
    material: Material, output_dir: Path, dry_run: bool, continue_on_error: bool
) -> bool:
    destination = output_dir / material.path
    if dry_run:
        print(f"would download {material.url} -> {destination}")
        return True

    try:
        data = fetch_url(material.url)
        if material.gitiles_text:
            data = decode_gitiles_text(data)
        write_file(destination, data)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        print(f"failed: {material.url} -> {destination}: {exc}", file=sys.stderr)
        if not continue_on_error:
            raise
        return False

    print(f"downloaded {material.path}")
    return True


def render_index(materials: list[Material], fetched_at: datetime) -> str:
    lines = [
        "# TWCC / GCC Reference Materials",
        "",
        "This directory is generated by `scripts/download_reference_materials.py` and is ignored by git.",
        "",
        f"Fetched: {fetched_at.isoformat()}",
        "",
    ]

    for material in sorted(materials, key=lambda item: item.path):
        lines.extend(
            [
                f"## {material.path}",
                "",
                f"- Title: {material.title}",
                f"- Source: {material.url}",
                f"- Why: {material.why}",
                "",
            ]
        )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TWCC / GCC reference materials.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/download_reference_materials.py
              python scripts/download_reference_materials.py --dry-run
            """
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print downloads without writing files",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue downloading after individual fetch failures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    materials = list(MATERIALS)
    for directory in GITHUB_DIRECTORIES:
        try:
            materials.extend(fetch_github_directory(directory))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            ValueError,
        ) as exc:
            print(
                f"failed to enumerate GitHub directory {directory.source_path}: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                raise

    successes = 0
    for material in materials:
        if download_material(
            material, output_dir, args.dry_run, args.continue_on_error
        ):
            successes += 1

    if not args.dry_run:
        index = render_index(materials, datetime.now(timezone.utc))
        write_file(output_dir / "INDEX.md", index.encode("utf8"))
        print(f"wrote {(output_dir / 'INDEX.md').relative_to(ROOT)}")

    failures = len(materials) - successes
    if failures:
        print(f"{failures} download(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
