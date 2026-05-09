#!/usr/bin/env python3
"""Render a synthetic localization debug image directly in a terminal.

This is dependency-free on purpose: it generates a PNG in memory and writes it
using either the kitty graphics protocol or iTerm2's inline image protocol.
"""

from __future__ import annotations

import argparse
import base64
import math
import os
import struct
import sys
import zlib


Color = tuple[int, int, int, int]


def clamp(value: int, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, value))


def blend_pixel(buf: bytearray, width: int, height: int, x: int, y: int, color: Color) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    r, g, b, a = color
    i = (y * width + x) * 4
    if a >= 255:
        buf[i : i + 4] = bytes((r, g, b, 255))
        return
    inv = 255 - a
    buf[i] = (r * a + buf[i] * inv) // 255
    buf[i + 1] = (g * a + buf[i + 1] * inv) // 255
    buf[i + 2] = (b * a + buf[i + 2] * inv) // 255
    buf[i + 3] = 255


def line(buf: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: Color, thickness: int = 1) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    radius = max(0, thickness // 2)
    while True:
        for yy in range(y0 - radius, y0 + radius + 1):
            for xx in range(x0 - radius, x0 + radius + 1):
                blend_pixel(buf, width, height, xx, yy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def circle(buf: bytearray, width: int, height: int, cx: int, cy: int, radius: int, color: Color, fill: bool = True) -> None:
    r2 = radius * radius
    inner = max(0, radius - 2)
    inner2 = inner * inner
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d2 <= r2 and (fill or d2 >= inner2):
                blend_pixel(buf, width, height, x, y, color)


def draw_arrow(buf: bytearray, width: int, height: int, x: int, y: int, theta: float, color: Color) -> None:
    length = max(24, width // 18)
    x1 = int(x + math.cos(theta) * length)
    y1 = int(y - math.sin(theta) * length)
    line(buf, width, height, x, y, x1, y1, color, 5)
    for offset in (2.45, -2.45):
        hx = int(x1 + math.cos(theta + offset) * length * 0.32)
        hy = int(y1 - math.sin(theta + offset) * length * 0.32)
        line(buf, width, height, x1, y1, hx, hy, color, 4)
    circle(buf, width, height, x, y, 8, (255, 255, 255, 230))
    circle(buf, width, height, x, y, 5, color)


def make_debug_rgba(width: int, height: int) -> bytearray:
    buf = bytearray(width * height * 4)

    for y in range(height):
        for x in range(width):
            nx = x / max(1, width - 1)
            ny = y / max(1, height - 1)
            grid = 10 if (x % 40 == 0 or y % 40 == 0) else 0
            wave = int(10 * math.sin(nx * 16.0) + 8 * math.cos(ny * 13.0))
            v = clamp(222 + wave - grid)
            i = (y * width + x) * 4
            buf[i : i + 4] = bytes((v, v, v, 255))

    # Occupied cells and unknown patches, map-debug style.
    for k in range(18):
        cx = int(width * (0.10 + 0.80 * ((k * 37) % 101) / 100))
        cy = int(height * (0.12 + 0.76 * ((k * 61) % 97) / 96))
        rw = 18 + (k * 19) % 70
        rh = 14 + (k * 23) % 54
        shade = 42 + (k * 13) % 25
        for y in range(cy - rh, cy + rh):
            for x in range(cx - rw, cx + rw):
                if 0 <= x < width and 0 <= y < height:
                    wobble = math.sin((x + k) * 0.08) + math.cos((y - k) * 0.11)
                    if wobble > -1.2:
                        blend_pixel(buf, width, height, x, y, (shade, shade + 2, shade + 4, 210))

    # Laser scan fan.
    origin = (int(width * 0.46), int(height * 0.58))
    for a in [math.radians(v) for v in range(-115, 116, 5)]:
        dist = width * (0.16 + 0.08 * (1 + math.sin(3 * a)))
        x1 = int(origin[0] + math.cos(a + 0.35) * dist)
        y1 = int(origin[1] - math.sin(a + 0.35) * dist)
        line(buf, width, height, origin[0], origin[1], x1, y1, (40, 160, 255, 42), 1)
        circle(buf, width, height, x1, y1, 2, (25, 135, 245, 150))

    # Global localization particles.
    for k in range(260):
        angle = k * 2.39996323
        radius = math.sqrt(k / 260.0)
        spread_x = width * 0.27
        spread_y = height * 0.20
        x = int(origin[0] + math.cos(angle) * radius * spread_x + math.sin(k) * 8)
        y = int(origin[1] + math.sin(angle) * radius * spread_y + math.cos(k * 0.7) * 7)
        circle(buf, width, height, x, y, 2, (255, 112, 30, 145))

    # Global path and estimated pose.
    pts: list[tuple[int, int]] = []
    for t in range(90):
        u = t / 89
        x = int(width * (0.12 + 0.78 * u))
        y = int(height * (0.76 - 0.42 * u + 0.07 * math.sin(u * math.tau * 2.1)))
        pts.append((x, y))
    for a, b in zip(pts, pts[1:]):
        line(buf, width, height, a[0], a[1], b[0], b[1], (38, 190, 95, 230), 4)

    draw_arrow(buf, width, height, origin[0], origin[1], -0.35, (230, 30, 75, 255))

    # Border to reveal image boundaries and scaling quality.
    line(buf, width, height, 1, 1, width - 2, 1, (20, 24, 30, 255), 2)
    line(buf, width, height, width - 2, 1, width - 2, height - 2, (20, 24, 30, 255), 2)
    line(buf, width, height, width - 2, height - 2, 1, height - 2, (20, 24, 30, 255), 2)
    line(buf, width, height, 1, height - 2, 1, 1, (20, 24, 30, 255), 2)
    return buf


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    rows = []
    stride = width * 4
    for y in range(height):
        rows.append(b"\x00" + rgba[y * stride : (y + 1) * stride])
    raw = b"".join(rows)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
            png_chunk(b"IDAT", zlib.compress(raw, 6)),
            png_chunk(b"IEND", b""),
        ]
    )


def write_kitty(png: bytes, width: int, height: int) -> None:
    payload = base64.b64encode(png)
    chunk_size = 4096
    out = sys.stdout.buffer
    for start in range(0, len(payload), chunk_size):
        chunk = payload[start : start + chunk_size]
        more = 1 if start + chunk_size < len(payload) else 0
        if start == 0:
            header = f"\033_Ga=T,f=100,t=d,s={width},v={height},m={more};".encode()
        else:
            header = f"\033_Gm={more};".encode()
        out.write(header + chunk + b"\033\\")
    out.write(b"\n")
    out.flush()


def write_iterm2(png: bytes, width: int, height: int) -> None:
    payload = base64.b64encode(png)
    header = f"\033]1337;File=inline=1;width={width}px;height={height}px;preserveAspectRatio=1:".encode()
    sys.stdout.buffer.write(header + payload + b"\a\n")
    sys.stdout.buffer.flush()


def detect_protocol() -> str:
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    term = os.environ.get("TERM", "").lower()
    if "iterm" in term_program:
        return "iterm2"
    if "wezterm" in term_program or "kitty" in term or os.environ.get("KITTY_WINDOW_ID"):
        return "kitty"
    return "kitty"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("auto", "kitty", "iterm2"), default="auto")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--save", metavar="PATH", help="also write the generated PNG to PATH")
    args = parser.parse_args()

    width = max(160, args.width)
    height = max(120, args.height)
    png = encode_png_rgba(width, height, make_debug_rgba(width, height))

    if args.save:
        with open(args.save, "wb") as f:
            f.write(png)

    protocol = detect_protocol() if args.protocol == "auto" else args.protocol
    if protocol == "iterm2":
        write_iterm2(png, width, height)
    else:
        write_kitty(png, width, height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
