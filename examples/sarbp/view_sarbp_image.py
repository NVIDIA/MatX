#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2026, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
View a SAR backprojection image produced by the sarbp executable.

The .raw file is single-precision complex float (interleaved real/imag,
row-major).  Image dimensions are read from the companion .sarbp input file
or specified manually with --size.

Usage:
    python view_sarbp_image.py output.raw
    python view_sarbp_image.py output.raw --sarbp input.sarbp
    python view_sarbp_image.py output.raw --size 2048x2048
    python view_sarbp_image.py output.raw --dynamic-range 50
    python view_sarbp_image.py output.raw --save output.png
"""

import argparse
import os
import struct
import sys

import numpy as np

SARBP_MAGIC = b"SARBP\x02\x00\x00"
SARBP_HEADER_SIZE = 256


def read_sarbp_header(sarbp_path):
    """Read image dimensions and voxel grid info from a .sarbp file header.

    Returns a dict with keys: image_height, image_width, voxel_start_x,
    voxel_start_y, voxel_stride_x, voxel_stride_y, or None on failure.
    """
    with open(sarbp_path, "rb") as f:
        hdr = f.read(SARBP_HEADER_SIZE)

    if len(hdr) < SARBP_HEADER_SIZE or hdr[:8] != SARBP_MAGIC:
        return None

    return {
        "image_width":    struct.unpack_from("<I", hdr, 16)[0],
        "image_height":   struct.unpack_from("<I", hdr, 20)[0],
        "voxel_start_x":  struct.unpack_from("<d", hdr, 56)[0],
        "voxel_start_y":  struct.unpack_from("<d", hdr, 64)[0],
        "voxel_stride_x": struct.unpack_from("<d", hdr, 80)[0],
        "voxel_stride_y": struct.unpack_from("<d", hdr, 88)[0],
    }


def find_sarbp_file(raw_path):
    """Look for a .sarbp file matching the .raw file name."""
    base = os.path.splitext(raw_path)[0]
    candidate = base + ".sarbp"
    if os.path.isfile(candidate):
        return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="View a SAR backprojection image (.raw complex float)"
    )
    parser.add_argument("raw_file", help="Path to .raw output file from sarbp")
    parser.add_argument("--sarbp", default=None,
                        help="Path to .sarbp input file (for reading image dimensions). "
                             "Default: looks for a .sarbp file with the same base name.")
    parser.add_argument("--size", default=None,
                        help="Image dimensions as HEIGHTxWIDTH (e.g. 2048x2048). "
                             "Overrides .sarbp header. If neither --sarbp nor --size is "
                             "given, assumes a square image based on file size.")
    parser.add_argument("--dynamic-range", type=float, default=70,
                        help="Display dynamic range in dB below peak (default: 70)")
    parser.add_argument("--save", default=None,
                        help="Save image to file (e.g. output.png) instead of displaying")
    parser.add_argument("--cmap", default="gray",
                        help="Matplotlib colormap (default: gray)")
    args = parser.parse_args()

    # Determine image dimensions (priority: --size > --sarbp / auto-discover > square guess)
    height, width = None, None
    sarbp_info = None
    if args.size:
        parts = args.size.lower().split("x")
        if len(parts) != 2:
            print(f"ERROR: --size must be HEIGHTxWIDTH, got '{args.size}'",
                  file=sys.stderr)
            sys.exit(1)
        height, width = int(parts[0]), int(parts[1])

    sarbp_path = args.sarbp or find_sarbp_file(args.raw_file)
    if sarbp_path:
        sarbp_info = read_sarbp_header(sarbp_path)
        if sarbp_info and height is None:
            height = sarbp_info["image_height"]
            width = sarbp_info["image_width"]
            print(f"Read image size {height}x{width} from {sarbp_path}")

    # Load raw data
    data = np.fromfile(args.raw_file, dtype=np.complex64)

    if height is None or width is None:
        # Assume square
        n = int(np.sqrt(len(data)))
        if n * n == len(data):
            height, width = n, n
            print(f"Assuming square image: {height}x{width}")
        else:
            print("ERROR: cannot determine image dimensions. "
                  "The pixel count is not a perfect square. "
                  "Use --size HEIGHTxWIDTH or --sarbp to specify dimensions.",
                  file=sys.stderr)
            sys.exit(1)

    expected = height * width
    if len(data) != expected:
        print(f"ERROR: expected {expected} pixels ({height}x{width}), "
              f"got {len(data)}", file=sys.stderr)
        sys.exit(1)

    img = data.reshape(height, width)

    # Convert to dB magnitude, normalized so peak = 0 dB
    mag = np.abs(img)
    mag_db = 20.0 * np.log10(mag + 1e-12)
    mag_db -= mag_db.max()
    vmax = 0.0
    vmin = -args.dynamic_range

    # Import matplotlib only when needed so the script can be used without
    # a display if only --save is used.
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    else:
        # Pick an interactive backend by checking if the underlying toolkit
        # is actually importable. matplotlib.use() must be called before
        # pyplot is imported and cannot be recovered after a failed import.
        candidates = [
            ("TkAgg",   "tkinter"),
            ("QtAgg",   "PyQt5.QtWidgets"),
            ("GTK4Agg", "gi"),
        ]
        for backend, probe in candidates:
            try:
                __import__(probe)
                matplotlib.use(backend)
                break
            except (ImportError, ModuleNotFoundError):
                continue
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # Use physical coordinates from .sarbp header when available
    extent = None
    has_coords = False
    if sarbp_info and sarbp_info["voxel_stride_x"] != 0 and sarbp_info["voxel_stride_y"] != 0:
        x0 = sarbp_info["voxel_start_x"]
        y0 = sarbp_info["voxel_start_y"]
        dx = sarbp_info["voxel_stride_x"]
        dy = sarbp_info["voxel_stride_y"]
        x1 = x0 + dx * (width - 1)
        y1 = y0 + dy * (height - 1)
        # imshow extent: [left, right, bottom, top]
        # origin="upper" means row 0 is at top, so top=y0, bottom=y1
        extent = [x0 - dx / 2, x1 + dx / 2, y1 + dy / 2, y0 - dy / 2]
        has_coords = True

    im = ax.imshow(mag_db, cmap=args.cmap, vmin=vmin, vmax=vmax,
                   origin="upper", aspect="equal", extent=extent)
    fig.colorbar(im, ax=ax, label="dB", shrink=0.8)
    ax.set_title("SAR Backprojection Image")
    if has_coords:
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
    else:
        ax.set_xlabel("Range (pixels)")
        ax.set_ylabel("Azimuth (pixels)")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
