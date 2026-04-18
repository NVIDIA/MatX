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
CPHD to MatX sar_bp Input Converter
====================================

Reads a CPHD (Compensated Phase History Data) file and produces a .sarbp
binary file consumed by the MatX sarbp example. The .sarbp format packages
the phase history, platform positions, and image grid parameters into a
single file with no external dependencies at load time.

Processing pipeline:
  1. Read CPHD signal array (FX domain) and per-vector parameters
  2. Optional: decimate pulses for faster processing
  3. Compute antenna phase centres (APC) in ECEF, convert to local ENU
  4. Define output image grid in local ENU coordinates
  5. Write .sarbp output

Usage:
    python cphd_to_sarbp_input.py <cphd_file> [options]
    python cphd_to_sarbp_input.py path/to/file.cphd --image-size 2048 --pulse-stride 4
    python cphd_to_sarbp_input.py path/to/file.cphd -o output.sarbp
"""

import argparse
import mmap
import os
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# .sarbp binary format constants
# ---------------------------------------------------------------------------
SARBP_MAGIC = b"SARBP\x02\x00\x00"   # 8 bytes: 5-char magic + version(2) + 2 reserved
SARBP_VERSION = 2
SARBP_HEADER_SIZE = 256               # fixed header size in bytes (padded for alignment)


def ecef_to_enu(ecef_points: np.ndarray, ref_ecef: np.ndarray,
                ref_lat_rad: float, ref_lon_rad: float) -> np.ndarray:
    """Convert ECEF coordinates to local East-North-Up (ENU) frame.

    Parameters
    ----------
    ecef_points : (N, 3) array -- ECEF coordinates [m]
    ref_ecef    : (3,) array   -- ECEF origin of ENU frame [m]
    ref_lat_rad : float        -- geodetic latitude of origin [rad]
    ref_lon_rad : float        -- geodetic longitude of origin [rad]

    Returns
    -------
    enu : (N, 3) array -- East, North, Up coordinates [m]
    """
    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    # Rotation matrix from ECEF to ENU
    R = np.array([
        [-sin_lon,             cos_lon,            0       ],
        [-sin_lat * cos_lon,  -sin_lat * sin_lon,  cos_lat ],
        [ cos_lat * cos_lon,   cos_lat * sin_lon,  sin_lat ],
    ])

    diff = ecef_points - ref_ecef[np.newaxis, :]
    return diff @ R.T


def read_cphd(cphd_path: str, pulse_stride: int = 1, max_pulses: int = 0,
              use_streaming: bool = False, int16_mode: bool = False):
    """Read CPHD file using sarpy and extract all needed data.

    Parameters
    ----------
    cphd_path    : path to .cphd file
    pulse_stride : take every Nth pulse (for faster processing)
    max_pulses   : limit total number of pulses (0 = no limit)
    use_streaming: if True, defer signal loading and return the reader object
                   for streamed access; if False, load all signal into memory
    int16_mode   : if True (and not streaming), load raw int16 I/Q samples
                   via read_raw() instead of scaled complex64

    Returns
    -------
    dict with keys: signal (complex64, or None if streaming/int16),
                    signal_int16 (int16 shape (N, samples, 2), or None),
                    reader (or None), pulse_indices,
                    tx_pos, rx_pos, srp_pos, fx1, fx2, sc0, scss,
                    iarp_ecef, iarp_lat_deg, iarp_lon_deg, iarp_hae,
                    center_freq_hz, bandwidth_hz, sgn
    """
    from sarpy.io.phase_history.converter import open_phase_history

    print(f"Opening CPHD: {cphd_path}")
    reader = open_phase_history(cphd_path)
    meta = reader.cphd_meta

    # Sample format (CF8 = complex float32, CI4 = complex int16, CI2 = complex int8)
    signal_format = meta.Data.SignalArrayFormat
    print(f"  Signal format: {signal_format}")
    if int16_mode and signal_format != "CI4":
        print(f"  NOTE: int16_mode requested but source is {signal_format}; "
              f"falling back to complex64 output.")
        int16_mode = False

    # Global parameters
    domain = meta.Global.DomainType
    assert domain == "FX", f"Expected FX domain, got {domain}"
    sgn = meta.Global.SGN  # phase sign convention

    fx_min = meta.Global.FxBand.FxMin
    fx_max = meta.Global.FxBand.FxMax
    center_freq = (fx_min + fx_max) / 2.0
    bandwidth = fx_max - fx_min

    print(f"  Domain: {domain}, SGN: {sgn}")
    print(f"  FxBand: {fx_min/1e9:.4f} - {fx_max/1e9:.4f} GHz")
    print(f"  Center freq: {center_freq/1e9:.4f} GHz, BW: {bandwidth/1e6:.1f} MHz")

    # Scene reference point
    sc = meta.SceneCoordinates
    iarp_ecef = np.array([sc.IARP.ECF.X, sc.IARP.ECF.Y, sc.IARP.ECF.Z])
    iarp_lat = sc.IARP.LLH.Lat
    iarp_lon = sc.IARP.LLH.Lon
    iarp_hae = sc.IARP.LLH.HAE

    print(f"  IARP: lat={iarp_lat:.6f}, lon={iarp_lon:.6f}, hae={iarp_hae:.1f} m")

    # Data dimensions
    num_vectors, num_samples = reader.data_size
    print(f"  Data: {num_vectors} vectors x {num_samples} samples")

    # Per-vector parameters
    print("  Reading PVP ...", end=" ", flush=True)
    pvp = reader.read_pvp_array(0)
    print("done.")

    # Determine pulse indices
    pulse_indices = np.arange(0, num_vectors, pulse_stride)
    if max_pulses > 0 and len(pulse_indices) > max_pulses:
        pulse_indices = pulse_indices[:max_pulses]
    num_pulses = len(pulse_indices)
    print(f"  Using {num_pulses} pulses (stride={pulse_stride})")

    # Read signal data (or defer for streaming)
    signal = None
    signal_int16 = None
    complex_bytes = num_pulses * num_samples * 8  # complex64
    int16_bytes = num_pulses * num_samples * 4    # int16 I/Q
    if not use_streaming:
        t0 = time.time()
        if int16_mode:
            print(f"  Reading raw int16 signal ...", end=" ", flush=True)
            # Allocate native little-endian int16 output
            signal_int16 = np.empty((num_pulses, num_samples, 2), dtype='<i2')
            for i, vi in enumerate(pulse_indices):
                raw = reader.read_raw(slice(int(vi), int(vi) + 1),
                                      slice(0, num_samples), index=0, squeeze=True)
                # raw shape: (num_samples, 2), big-endian int16
                signal_int16[i] = raw.byteswap().view(np.dtype('<i2'))
            dt = time.time() - t0
            print(f"done ({dt:.1f}s, {signal_int16.nbytes/1e9:.2f} GB)")
        else:
            print(f"  Reading signal data ...", end=" ", flush=True)
            signal = np.zeros((num_pulses, num_samples), dtype=np.complex64)
            for i, vi in enumerate(pulse_indices):
                signal[i, :] = reader[int(vi):int(vi)+1, :, 0]
            # Note: sarpy's read() applies AmpSF
            dt = time.time() - t0
            print(f"done ({dt:.1f}s, {signal.nbytes/1e9:.2f} GB)")
    else:
        bytes_est = int16_bytes if int16_mode else complex_bytes
        print(f"  Signal data: {bytes_est/1e9:.1f} GB (will stream from file)")

    # Extract positions for selected pulses (ECEF, metres)
    tx_pos = pvp['TxPos'][pulse_indices]    # (N, 3)
    rx_pos = pvp['RcvPos'][pulse_indices]   # (N, 3)
    srp_pos = pvp['SRPPos'][pulse_indices]  # (N, 3)

    # Frequency parameters per vector
    fx1 = pvp['FX1'][pulse_indices]
    fx2 = pvp['FX2'][pulse_indices]
    sc0 = pvp['SC0'][pulse_indices]
    scss = pvp['SCSS'][pulse_indices]

    # Per-pulse amplitude scale factor (if present)
    if 'AmpSF' in pvp.dtype.names:
        ampsf = pvp['AmpSF'][pulse_indices]
        print(f"  AmpSF: [{ampsf.min():.4e}, {ampsf.max():.4e}]")
    else:
        ampsf = None

    # Velocities for Doppler filtering
    tx_vel = pvp['TxVel'][pulse_indices]
    rx_vel = pvp['RcvVel'][pulse_indices]

    # PRF from pulse timing
    tx_times = pvp['TxTime']
    prf = 1.0 / np.median(np.diff(tx_times))

    # Signal time-of-arrival support (for scene extent computation)
    toa1 = pvp['TOA1'][pulse_indices]  # differential one-way time [s]
    toa2 = pvp['TOA2'][pulse_indices]

    return dict(
        signal=signal,
        signal_int16=signal_int16,
        int16_mode=int16_mode,
        reader=reader if use_streaming else None,
        pulse_indices=pulse_indices,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
        tx_vel=tx_vel,
        rx_vel=rx_vel,
        srp_pos=srp_pos,
        fx1=fx1, fx2=fx2,
        sc0=sc0, scss=scss,
        ampsf=ampsf,
        toa1=toa1, toa2=toa2,
        iarp_ecef=iarp_ecef,
        iarp_lat_deg=iarp_lat,
        iarp_lon_deg=iarp_lon,
        iarp_hae=iarp_hae,
        center_freq_hz=center_freq,
        bandwidth_hz=bandwidth,
        sgn=sgn,
        num_samples=num_samples,
        num_vectors=num_vectors,
        prf=prf,
    )



def doppler_prefilter(range_profiles: np.ndarray, apc_enu: np.ndarray,
                      vel_enu: np.ndarray, image_extent_m: float,
                      center_freq_hz: float, prf: float) -> np.ndarray:
    """Filter range profiles to keep only Doppler content within the image area.

    For satellite SAR, the antenna footprint is much larger than a typical
    image area.  Energy from scatterers outside the image occupies Doppler
    frequencies that can alias into the image band during backprojection.
    This filter removes that out-of-band energy.  The practical impact on
    image quality varies by dataset and may be small in some cases.

    Parameters
    ----------
    range_profiles : (num_pulses, num_range_bins) complex array
    apc_enu        : (num_pulses, 3) platform positions in ENU [m]
    vel_enu        : (num_pulses, 3) platform velocities in ENU [m/s]
    image_extent_m : diagonal extent of the output image [m]
    center_freq_hz : carrier frequency [Hz]
    prf            : pulse repetition frequency [Hz]

    Returns
    -------
    filtered : (num_pulses, num_range_bins) complex array
    """
    num_pulses = range_profiles.shape[0]
    filter_mask, info = _build_doppler_mask(
        num_pulses, apc_enu, vel_enu, image_extent_m, center_freq_hz, prf)
    print(info)

    # Apply filter in Doppler domain and IFFT back to slow time
    rp_doppler = np.fft.fft(range_profiles, axis=0)
    rp_doppler *= filter_mask[:, np.newaxis]
    filtered = np.fft.ifft(rp_doppler, axis=0)

    return filtered.astype(np.complex64)


def _build_doppler_mask(num_pulses: int, apc_enu: np.ndarray,
                        vel_enu: np.ndarray, image_extent_m: float,
                        center_freq_hz: float, prf: float):
    """Compute the Doppler filter mask (shared by bulk and streamed paths).

    Returns (filter_mask, info_str) where filter_mask is (num_pulses,) float32.
    """
    c = 299792458.0
    lam = c / center_freq_hz

    mid = num_pulses // 2
    pos0 = apc_enu[mid]
    vel0 = vel_enu[mid]
    R0 = np.linalg.norm(pos0)
    los = -pos0 / R0

    vel_along_los = np.dot(vel0, los) * los
    vel_cross = vel0 - vel_along_los
    v_cross_mag = np.linalg.norm(vel_cross)

    scene_doppler_bw = 2.0 * v_cross_mag * image_extent_m / (lam * R0)
    filter_half_bw = scene_doppler_bw / 2.0 * 1.2

    doppler_axis = np.fft.fftfreq(num_pulses, d=1.0 / prf)
    filter_mask = np.zeros(num_pulses, dtype=np.float32)
    for i in range(num_pulses):
        fd = abs(doppler_axis[i])
        if fd <= filter_half_bw * 0.8:
            filter_mask[i] = 1.0
        elif fd <= filter_half_bw:
            x = (fd - filter_half_bw * 0.8) / (filter_half_bw * 0.2)
            filter_mask[i] = 0.5 * (1.0 + np.cos(np.pi * x))

    num_kept = int(np.sum(filter_mask > 0))
    info = (f"Doppler prefilter:\n"
            f"  Scene Doppler BW: {scene_doppler_bw:.1f} Hz "
            f"(image extent: {image_extent_m:.0f} m)\n"
            f"  Filter half-BW: {filter_half_bw:.1f} Hz (with 20% margin)\n"
            f"  PRF: {prf:.1f} Hz\n"
            f"  Keeping {num_kept}/{num_pulses} Doppler bins")
    return filter_mask, info


def _build_sarbp_header(num_pulses, num_range_bins, image_size,
                        center_freq, del_r, bandwidth, pixel_spacing_m,
                        is_fx_domain, sgn, num_samples_raw, prf,
                        grazing_angle_deg, int16_mode=False):
    """Build the 256-byte .sarbp file header."""
    half_extent = (image_size / 2.0) * pixel_spacing_m
    voxel_start_x = -half_extent
    voxel_start_y = -half_extent
    voxel_start_z = 0.0
    if image_size > 1:
        voxel_stride_x = (2.0 * half_extent) / (image_size - 1)
        voxel_stride_y = (2.0 * half_extent) / (image_size - 1)
    else:
        voxel_stride_x = pixel_spacing_m
        voxel_stride_y = pixel_spacing_m

    header = bytearray(SARBP_HEADER_SIZE)
    header[0:8] = SARBP_MAGIC
    struct.pack_into('<I', header, 8, num_pulses)
    struct.pack_into('<I', header, 12, num_range_bins)
    struct.pack_into('<I', header, 16, image_size)
    struct.pack_into('<I', header, 20, image_size)
    struct.pack_into('<d', header, 24, center_freq)
    struct.pack_into('<d', header, 32, del_r)
    struct.pack_into('<d', header, 40, bandwidth)
    struct.pack_into('<d', header, 48, pixel_spacing_m)
    struct.pack_into('<d', header, 56, voxel_start_x)
    struct.pack_into('<d', header, 64, voxel_start_y)
    struct.pack_into('<d', header, 72, voxel_start_z)
    struct.pack_into('<d', header, 80, voxel_stride_x)
    struct.pack_into('<d', header, 88, voxel_stride_y)
    flags = (0x1 if is_fx_domain else 0) | (0x2 if int16_mode else 0)
    struct.pack_into('<I', header, 96, flags)
    struct.pack_into('<i', header, 100, sgn)
    struct.pack_into('<I', header, 104,
                     num_samples_raw if num_samples_raw > 0 else num_range_bins)
    pulse_hdr_size = 56 if int16_mode else 48
    struct.pack_into('<I', header, 108, pulse_hdr_size)
    struct.pack_into('<d', header, 112, prf)
    struct.pack_into('<d', header, 120, grazing_angle_deg)
    return header


def write_sarbp_file_streamed(output_path: str, reader, pulse_indices: np.ndarray,
                              apc_enu: np.ndarray, range_to_mcp: np.ndarray,
                              center_freq: float, del_r: float, bandwidth: float,
                              pixel_spacing_m: float, image_size: int,
                              is_fx_domain: bool = False, sgn: int = -1,
                              toa1: np.ndarray = None, toa2: np.ndarray = None,
                              num_samples_raw: int = 0, prf: float = 0.0,
                              grazing_angle_deg: float = 0.0,
                              doppler_mask: np.ndarray = None,
                              chunk_bins: int = 1000,
                              int16_mode: bool = True,
                              ampsf: np.ndarray = None):
    """Write a .sarbp file by streaming from a CPHD reader.

    When doppler_mask is provided, reads column chunks (all pulses x chunk_bins
    range bins) from the CPHD reader, applies the Doppler filter, and writes
    the filtered samples to the output file.  Without doppler_mask, streams
    pulses directly.

    Peak memory with Doppler filter: ~num_pulses * chunk_bins * 16 bytes.
    """
    num_pulses = len(pulse_indices)
    num_range_bins = reader.data_size[1]

    header = _build_sarbp_header(
        num_pulses, num_range_bins, image_size,
        center_freq, del_r, bandwidth, pixel_spacing_m,
        is_fx_domain, sgn, num_samples_raw, prf, grazing_angle_deg,
        int16_mode=int16_mode)

    if int16_mode:
        pulse_header_size = 56
        samples_size = num_range_bins * 4  # int16 I/Q = 4 bytes
    else:
        pulse_header_size = 48
        samples_size = num_range_bins * 8  # complex64 = 8 bytes
    record_size = pulse_header_size + samples_size
    total_size = SARBP_HEADER_SIZE + num_pulses * record_size

    mode_str = "int16" if int16_mode else "complex64"
    print(f"Writing .sarbp file (streamed): {output_path} ({mode_str} samples)")
    print(f"  File header : {SARBP_HEADER_SIZE} bytes")
    print(f"  Pulse record: {record_size} bytes x {num_pulses} pulses")
    print(f"  Total       : {total_size / (1024**3):.2f} GB")

    # Per-pulse scales (for int16 mode): use AmpSF if available, otherwise 1.0
    if int16_mode:
        scales = (np.asarray(ampsf, dtype=np.float64)
                  if ampsf is not None else np.ones(num_pulses, dtype=np.float64))

    # --- Step 1: Write file skeleton (header + pulse headers, preallocate) ---
    t0 = time.time()
    print("  Writing pulse headers ...", end=" ", flush=True)
    with open(output_path, 'wb') as f:
        # Preallocate file (sparse on filesystems that support it)
        f.truncate(total_size)

        # Write file header
        f.write(header)

        # Write pulse headers at their correct offsets
        for i in range(num_pulses):
            t1 = float(toa1[i]) if toa1 is not None else 0.0
            t2 = float(toa2[i]) if toa2 is not None else 0.0
            if int16_mode:
                pulse_hdr = struct.pack('<ddddddd',
                                        apc_enu[i, 0], apc_enu[i, 1],
                                        apc_enu[i, 2], range_to_mcp[i],
                                        t1, t2, float(scales[i]))
            else:
                pulse_hdr = struct.pack('<dddddd',
                                        apc_enu[i, 0], apc_enu[i, 1],
                                        apc_enu[i, 2], range_to_mcp[i],
                                        t1, t2)
            f.seek(SARBP_HEADER_SIZE + i * record_size)
            f.write(pulse_hdr)
    dt = time.time() - t0
    print(f"done ({dt:.1f}s)")

    # --- Step 2: Fill samples via mmap, reading column chunks from CPHD ---
    samples_offset = SARBP_HEADER_SIZE + pulse_header_size

    with open(output_path, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), 0)
        if int16_mode:
            # Each row: num_range_bins * 2 int16 values (I/Q interleaved)
            dst_arr = np.ndarray(
                shape=(num_pulses, num_range_bins * 2),
                dtype=np.int16,
                buffer=mm,
                offset=samples_offset,
                strides=(record_size, 2))
        else:
            dst_arr = np.ndarray(
                shape=(num_pulses, num_range_bins),
                dtype=np.complex64,
                buffer=mm,
                offset=samples_offset,
                strides=(record_size, 8))

        t0 = time.time()
        for j in range(0, num_range_bins, chunk_bins):
            j_end = min(j + chunk_bins, num_range_bins)
            cols = j_end - j

            if int16_mode:
                # Read raw int16 I/Q directly -- no AmpSF scaling, no float round-trip.
                # read_raw returns shape (1, cols, 2) big-endian int16 per pulse.
                iq = np.empty((num_pulses, cols * 2), dtype='<i2')
                for i, vi in enumerate(pulse_indices):
                    raw = reader.read_raw(slice(int(vi), int(vi) + 1),
                                          slice(j, j_end), index=0, squeeze=True)
                    # raw shape: (cols, 2), dtype big-endian int16
                    iq[i, :] = raw.reshape(-1).byteswap().view(np.dtype('<i2'))
                dst_arr[:, j * 2:j_end * 2] = iq
            else:
                # Read formatted complex64 (AmpSF already applied by sarpy)
                chunk = np.zeros((num_pulses, cols), dtype=np.complex64)
                for i, vi in enumerate(pulse_indices):
                    chunk[i, :] = reader[int(vi):int(vi) + 1, j:j_end, 0]

                # Apply Doppler filter if requested
                if doppler_mask is not None:
                    spectrum = np.fft.fft(chunk, axis=0)
                    spectrum *= doppler_mask[:, np.newaxis]
                    chunk = np.fft.ifft(spectrum, axis=0).astype(np.complex64)

                dst_arr[:, j:j_end] = chunk

            elapsed = time.time() - t0
            pct = j_end / num_range_bins * 100
            print(f"\r  Bins {j_end}/{num_range_bins} ({pct:.0f}%)", end="", flush=True)

        mm.flush()
        mm.close()

    dt = time.time() - t0
    print(f"\n  Done ({dt:.1f}s).")



def write_sarbp_file(output_path: str, range_profiles: np.ndarray,
                     apc_enu: np.ndarray, range_to_mcp: np.ndarray,
                     center_freq: float, del_r: float, bandwidth: float,
                     pixel_spacing_m: float, image_size: int,
                     is_fx_domain: bool = False, sgn: int = -1,
                     toa1: np.ndarray = None, toa2: np.ndarray = None,
                     num_samples_raw: int = 0, prf: float = 0.0,
                     grazing_angle_deg: float = 0.0,
                     int16_mode: bool = True,
                     ampsf: np.ndarray = None,
                     int16_samples: np.ndarray = None):
    """Write a single .sarbp binary file.

    File layout
    -----------
    File header (SARBP_HEADER_SIZE = 256 bytes, little-endian):
      Offset  Size  Type      Field
      0       8     char[8]   magic  ("SARBP\\x02\\x00\\x00")
      8       4     uint32    num_pulses
      12      4     uint32    num_range_bins
      16      4     uint32    image_width
      20      4     uint32    image_height
      24      8     float64   center_frequency  [Hz]
      32      8     float64   del_r             [m]
      40      8     float64   bandwidth         [Hz]
      48      8     float64   pixel_spacing     [m]
      56      8     float64   voxel_start_x     [m]  (East, first pixel centre)
      64      8     float64   voxel_start_y     [m]  (North, first pixel centre)
      72      8     float64   voxel_start_z     [m]  (Up, typically 0)
      80      8     float64   voxel_stride_x    [m]  (pixel pitch, East)
      88      8     float64   voxel_stride_y    [m]  (pixel pitch, North)
      96      4     uint32    flags  (bit 0: 1=FX domain, 0=range compressed)
      100     4     int32     sgn    (phase sign convention: -1 or +1)
      104     4     uint32    num_samples_raw   (original FX sample count before processing)
      108     4     uint32    pulse_header_size (bytes per pulse header, currently 48)
      112     8     float64   prf               [Hz]
      120     8     float64   grazing_angle     [deg]
      128     128   -         reserved (zero-filled)

    Per-pulse record (repeated num_pulses times):
      Pulse header (48 bytes):
        0     8     float64   platform_pos_x  [m]  (East)
        8     8     float64   platform_pos_y  [m]  (North)
        16    8     float64   platform_pos_z  [m]  (Up)
        24    8     float64   range_to_mcp    [m]
        32    8     float64   toa1            [s]  (differential one-way TOA, near edge)
        40    8     float64   toa2            [s]  (differential one-way TOA, far edge)
      I/Q samples:
        48    num_range_bins * 8   complex64[]  (interleaved float32 real/imag)
    """
    if int16_mode:
        if int16_samples is None:
            raise ValueError("int16_mode=True requires int16_samples")
        num_pulses, num_range_bins, two = int16_samples.shape
        assert two == 2, f"int16_samples last dim must be 2 (I/Q), got {two}"
    else:
        num_pulses, num_range_bins = range_profiles.shape

    header = _build_sarbp_header(
        num_pulses, num_range_bins, image_size,
        center_freq, del_r, bandwidth, pixel_spacing_m,
        is_fx_domain, sgn, num_samples_raw, prf, grazing_angle_deg,
        int16_mode=int16_mode)

    if int16_mode:
        pulse_header_size = 56
        samples_size = num_range_bins * 4  # int16 I/Q = 4 bytes
    else:
        pulse_header_size = 48
        samples_size = num_range_bins * 8  # complex64 = 8 bytes
    pulse_record_size = pulse_header_size + samples_size
    total_size = SARBP_HEADER_SIZE + num_pulses * pulse_record_size

    mode_str = "int16" if int16_mode else "complex64"
    print(f"Writing .sarbp file: {output_path} ({mode_str} samples)")
    print(f"  File header : {SARBP_HEADER_SIZE} bytes")
    print(f"  Pulse record: {pulse_record_size} bytes x {num_pulses} pulses")
    print(f"  Total       : {total_size / (1024**2):.1f} MB")

    # Ensure int16 samples are in native (little-endian) byte order for the file
    if int16_mode:
        s = np.ascontiguousarray(int16_samples)
        if s.dtype.byteorder == '>':
            s = s.byteswap().view(np.dtype('<i2'))
        elif s.dtype != np.dtype('<i2'):
            s = s.astype(np.dtype('<i2'))

    with open(output_path, 'wb') as f:
        f.write(header)

        if not int16_mode:
            rp = np.ascontiguousarray(range_profiles, dtype=np.complex64)

        for i in range(num_pulses):
            t1 = toa1[i] if toa1 is not None else 0.0
            t2 = toa2[i] if toa2 is not None else 0.0

            if int16_mode:
                # Per-pulse scale factor: AmpSF from PVP, or 1.0 if absent.
                # CUDA multiplies int16 samples by this to recover complex float.
                scale = float(ampsf[i]) if ampsf is not None else 1.0
                pulse_hdr = struct.pack('<ddddddd',
                                        apc_enu[i, 0], apc_enu[i, 1], apc_enu[i, 2],
                                        range_to_mcp[i], t1, t2, scale)
                f.write(pulse_hdr)
                f.write(s[i].tobytes())
            else:
                pulse_hdr = struct.pack('<dddddd',
                                        apc_enu[i, 0], apc_enu[i, 1], apc_enu[i, 2],
                                        range_to_mcp[i], t1, t2)
                f.write(pulse_hdr)
                f.write(rp[i, :].tobytes())

    print(f"  Done.")


def process_cphd(cphd_path: str, output_path: str,
                 image_size: int = 0, pixel_spacing_m: float = 0.0,
                 pulse_stride: int = 1, max_pulses: int = 0,
                 doppler_filter: bool = False,
                 force_streaming: bool = False,
                 aperture_angle_deg: float = 2.0,
                 float_samples: bool = False):
    """Full CPHD to sar_bp pipeline.

    Parameters
    ----------
    cphd_path         : path to .cphd file
    output_path       : output .sarbp file path
    image_size        : output image dimension (square, 0 = auto from scene extent)
    pixel_spacing_m   : pixel spacing (0 = auto, 25% oversampled native resolution)
    pulse_stride      : decimate pulses by this factor
    max_pulses        : max pulses to use (0 = all)
    doppler_filter    : apply Doppler prefilter to remove out-of-scene energy
    aperture_angle_deg: max angle from closest approach to include (default: 2.0)
    """

    if pulse_stride > 1:
        print(f"NOTE: pulse_stride={pulse_stride} reduces effective PRF, "
              f"which may cause Doppler aliasing.")

    # Sample format: int16 by default; float is forced when Doppler filter is on
    # (FFT across pulses requires complex float) or explicitly requested.
    int16_mode = not (doppler_filter or float_samples)

    c = 299792458.0  # speed of light [m/s]

    # --- Step 1: Read CPHD metadata; decide whether to load signal into RAM ---
    # Estimate signal size to decide bulk vs streamed path
    from sarpy.io.phase_history.converter import open_phase_history
    probe = open_phase_history(cphd_path)
    num_vectors_total, num_samples = probe.data_size
    est_pulses = num_vectors_total // pulse_stride
    if max_pulses > 0:
        est_pulses = min(est_pulses, max_pulses)
    # Use complex64 size for the streaming threshold; if the source is CI4 and
    # int16_mode stays active, actual in-memory footprint is half this.
    signal_bytes = est_pulses * num_samples * 8
    del probe

    # Use streaming when signal would exceed ~16 GB
    MAX_SIGNAL_BYTES = 16 * 1024**3
    use_streaming = (signal_bytes > MAX_SIGNAL_BYTES or force_streaming)

    data = read_cphd(cphd_path, pulse_stride=pulse_stride,
                     max_pulses=max_pulses,
                     use_streaming=use_streaming,
                     int16_mode=int16_mode)

    # read_cphd may downgrade int16_mode -> False if source isn't CI4
    int16_mode = data['int16_mode']

    num_pulses = len(data['pulse_indices'])
    num_range_bins = data['num_samples']

    # --- Step 2: Signal data (FX domain, range compression deferred to CUDA) ---
    if use_streaming:
        if force_streaming:
            print(f"Streaming mode (forced, signal: {signal_bytes/1e9:.1f} GB)")
        else:
            print(f"Streaming mode (signal: {signal_bytes/1e9:.1f} GB > "
                  f"{MAX_SIGNAL_BYTES/1e9:.0f} GB threshold)")
    elif int16_mode:
        int16_samples = data['signal_int16']
        num_pulses, num_range_bins, _ = int16_samples.shape
        print(f"FX-domain int16 data: {int16_samples.shape}")
    else:
        range_profiles = data['signal']
        num_pulses, num_range_bins = range_profiles.shape
        print(f"FX-domain data: {range_profiles.shape}")

    # --- Step 3: Compute antenna phase centres in ENU ---
    # Monostatic: APC = (TxPos + RcvPos) / 2
    apc_ecef = (data['tx_pos'] + data['rx_pos']) / 2.0  # (N, 3)

    ref_lat_rad = np.deg2rad(data['iarp_lat_deg'])
    ref_lon_rad = np.deg2rad(data['iarp_lon_deg'])

    apc_enu = ecef_to_enu(apc_ecef, data['iarp_ecef'],
                          ref_lat_rad, ref_lon_rad)  # (N, 3)

    print(f"APC ENU extent: "
          f"E=[{apc_enu[:,0].min():.1f}, {apc_enu[:,0].max():.1f}] m, "
          f"N=[{apc_enu[:,1].min():.1f}, {apc_enu[:,1].max():.1f}] m, "
          f"U=[{apc_enu[:,2].min():.1f}, {apc_enu[:,2].max():.1f}] m")

    # --- Step 3b: Aperture selection for stripmap / sliding spotlight ---
    # For long collects, only a subset of pulses illuminate the scene center.
    # Using all pulses adds incoherent noise from out-of-beam scatterers.
    # Compute look angle from each APC to scene center (ENU origin).
    ranges = np.linalg.norm(apc_enu, axis=1)
    closest_idx = np.argmin(ranges)
    los_vectors = -apc_enu / ranges[:, np.newaxis]  # unit LOS from APC to origin
    los_closest = los_vectors[closest_idx]

    # Angular deviation from closest approach LOS
    cos_angles = np.clip(np.dot(los_vectors, los_closest), -1.0, 1.0)
    angle_from_closest = np.degrees(np.arccos(cos_angles))
    total_angular_extent = angle_from_closest.max()

    print(f"  Aperture: total angular extent = {total_angular_extent:.2f}deg, "
          f"limit = {aperture_angle_deg:.1f}deg")

    if total_angular_extent > aperture_angle_deg:
        mask = angle_from_closest <= aperture_angle_deg
        sel = np.where(mask)[0]
        n_before = num_pulses
        num_pulses = len(sel)

        apc_enu = apc_enu[sel]
        if not use_streaming:
            if int16_mode:
                int16_samples = int16_samples[sel]
            else:
                range_profiles = range_profiles[sel]
        # Trim all per-pulse arrays in data dict
        for key in ['toa1', 'toa2', 'tx_pos', 'rx_pos', 'tx_vel', 'rx_vel',
                     'srp_pos', 'fx1', 'fx2', 'sc0', 'scss', 'ampsf']:
            if key in data and data[key] is not None:
                data[key] = data[key][sel]
        data['pulse_indices'] = data['pulse_indices'][sel]

        print(f"    Trimmed to {num_pulses}/{n_before} pulses "
              f"(angular range: {angle_from_closest[sel].min():.3f}deg "
              f"to {angle_from_closest[sel].max():.3f}deg)")
    else:
        print(f"    No trimming needed")

    # --- Step 4: Compute sar_bp parameters and native resolution ---
    bandwidth = data['bandwidth_hz']
    center_freq = data['center_freq_hz']
    lam = c / center_freq
    del_r = c / (2.0 * bandwidth)  # range bin spacing = slant-range resolution

    # Mid-aperture geometry for resolution/extent computation
    mid = num_pulses // 2
    pos_mid = apc_enu[mid]
    R0 = np.linalg.norm(pos_mid)
    los = -pos_mid / R0
    grazing = np.arcsin(abs(los[2]))
    incidence = np.pi / 2.0 - grazing

    # Ground-range resolution (slant-range projected onto ground plane)
    ground_range_res = del_r / np.sin(incidence)

    # Cross-range resolution: lambda * R / (2 * v_cross * T_aperture)
    vel_ecef = (data['tx_vel'] + data['rx_vel']) / 2.0
    vel_enu = ecef_to_enu(vel_ecef, np.zeros(3), ref_lat_rad, ref_lon_rad)
    vel_mid = vel_enu[mid]
    vel_along_los = np.dot(vel_mid, los) * los
    vel_cross = vel_mid - vel_along_los
    v_cross = np.linalg.norm(vel_cross)
    T_aperture = num_pulses / data['prf']
    cross_range_res = lam * R0 / (2.0 * v_cross * T_aperture)

    print(f"Native resolution: slant-range={del_r:.3f} m, "
          f"ground-range={ground_range_res:.3f} m, "
          f"cross-range={cross_range_res:.3f} m")
    print(f"  Grazing angle: {np.degrees(grazing):.1f} deg, "
          f"slant range: {R0/1e3:.1f} km")

    # Auto pixel spacing: 25% oversampled relative to the coarsest native
    # resolution (isotropic square grid).
    if pixel_spacing_m <= 0:
        native_res = max(ground_range_res, cross_range_res)
        oversampling = 1.25
        pixel_spacing_m = native_res / oversampling
        print(f"Auto pixel spacing: {pixel_spacing_m:.3f} m "
              f"(native res = {native_res:.3f} m, {oversampling:.1f}x oversampled)")

    # Auto image size: cover the full scene with data support
    if image_size <= 0:
        # Range extent: from signal time support (TOA1/TOA2), projected to ground
        # Use the intersection across all pulses for full support
        toa1_max = np.max(data['toa1'])  # most restrictive near-range
        toa2_min = np.min(data['toa2'])  # most restrictive far-range
        slant_range_extent = (toa2_min - toa1_max) * c / 2.0
        ground_range_extent = slant_range_extent / np.sin(incidence)

        # Cross-range extent: use the same extent as ground-range to produce
        # a scene that covers the full signal swath symmetrically.  The antenna
        # footprint is typically larger, but points far off-scene have no useful
        # targets and just add computation and noise.
        cross_range_extent = ground_range_extent

        # The image grid is ENU-aligned; range/cross-range are rotated.
        # Compute the range and cross-range ground-plane directions.
        los_horiz = los.copy()
        los_horiz[2] = 0.0
        los_horiz /= np.linalg.norm(los_horiz)
        cross_dir = vel_cross / v_cross
        cross_horiz = cross_dir.copy()
        cross_horiz[2] = 0.0
        cross_horiz /= np.linalg.norm(cross_horiz)

        # For a square ENU grid, compute the extent needed to inscribe the
        # rotated range x cross-range rectangle
        half_r = ground_range_extent / 2.0
        half_c = cross_range_extent / 2.0
        # Project onto E and N axes
        half_E = (abs(half_r * los_horiz[0]) +
                  abs(half_c * cross_horiz[0]))
        half_N = (abs(half_r * los_horiz[1]) +
                  abs(half_c * cross_horiz[1]))
        scene_extent = 2.0 * max(half_E, half_N)

        image_size = int(np.ceil(scene_extent / pixel_spacing_m))
        # Round up to next multiple of 64 for GPU-friendly alignment
        image_size = ((image_size + 63) // 64) * 64

        print(f"Auto image size: {image_size}x{image_size} "
              f"({image_size * pixel_spacing_m:.0f} m)")
        print(f"  Ground-range extent: {ground_range_extent:.0f} m "
              f"(slant: {slant_range_extent:.0f} m)")
        print(f"  Cross-range extent:  {cross_range_extent:.0f} m "
              f"(matched to range)")

    print(f"Image grid: {image_size}x{image_size}, "
          f"spacing={pixel_spacing_m:.3f} m, "
          f"extent={image_size * pixel_spacing_m:.1f} m")

    # --- Step 5b: Doppler prefilter (bulk path only; streaming applies it after write) ---
    if doppler_filter and not use_streaming:
        image_extent = image_size * pixel_spacing_m * np.sqrt(2.0)  # diagonal
        prf_effective = data['prf'] / pulse_stride
        range_profiles = doppler_prefilter(
            range_profiles, apc_enu, vel_enu,
            image_extent, data['center_freq_hz'], prf_effective)

    # --- Step 6: Compute range_to_mcp ---
    # MCP = scene centre = ENU origin = (0, 0, 0) in local frame
    range_to_mcp = np.linalg.norm(apc_enu, axis=1)  # (N,)
    print(f"Range to MCP: [{range_to_mcp.min():.1f}, {range_to_mcp.max():.1f}] m, "
          f"mean={range_to_mcp.mean():.1f} m")

    # --- Step 7: Save .sarbp output ---
    if use_streaming:
        # Single-pass streamed path: read column chunks from CPHD,
        # optionally Doppler-filter, write to sarbp via mmap
        doppler_mask = None
        if doppler_filter:
            image_extent = image_size * pixel_spacing_m * np.sqrt(2.0)
            prf_effective = data['prf'] / pulse_stride
            doppler_mask, info = _build_doppler_mask(
                num_pulses, apc_enu, vel_enu,
                image_extent, data['center_freq_hz'], prf_effective)
            print(info)

        write_sarbp_file_streamed(
            output_path, data['reader'], data['pulse_indices'],
            apc_enu, range_to_mcp,
            center_freq, del_r, bandwidth, pixel_spacing_m, image_size,
            is_fx_domain=True, sgn=data['sgn'],
            toa1=data['toa1'], toa2=data['toa2'],
            num_samples_raw=data['num_samples'],
            prf=data['prf'],
            grazing_angle_deg=np.degrees(grazing),
            doppler_mask=doppler_mask,
            int16_mode=int16_mode,
            ampsf=data['ampsf'])
    else:
        write_sarbp_file(
            output_path,
            None if int16_mode else range_profiles,
            apc_enu, range_to_mcp,
            center_freq, del_r, bandwidth, pixel_spacing_m, image_size,
            is_fx_domain=True, sgn=data['sgn'],
            toa1=data['toa1'], toa2=data['toa2'],
            num_samples_raw=data['num_samples'],
            prf=data['prf'],
            grazing_angle_deg=np.degrees(grazing),
            int16_mode=int16_mode,
            ampsf=data['ampsf'],
            int16_samples=int16_samples if int16_mode else None)

    print(f"\nTo use with MatX sarbp example:")
    print(f"  sarbp {output_path} [output_image.raw]")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CPHD file to MatX sar_bp inputs"
    )
    parser.add_argument("cphd_file", help="Path to .cphd file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .sarbp file path (default: input with .sarbp extension)")
    parser.add_argument("--image-size", type=int, default=0,
                        help="Output image size in pixels (square, 0 = auto from scene extent)")
    parser.add_argument("--pixel-spacing", type=float, default=0.0,
                        help="Pixel spacing in metres (0 = auto, 25%% oversampled native res)")
    parser.add_argument("--pulse-stride", type=int, default=1,
                        help="Use every Nth pulse (default: 1 = all)")
    parser.add_argument("--max-pulses", type=int, default=0,
                        help="Max number of pulses (0 = all)")
    parser.add_argument("--doppler-filter", action="store_true",
                        help="Enable Doppler prefilter (forces complex64 sample output)")
    parser.add_argument("--float-samples", action="store_true",
                        help="Force complex64 samples instead of int16 (default: int16)")
    parser.add_argument("--stream", action="store_true",
                        help="Force streamed (low-memory) write path regardless of file size")
    parser.add_argument("--aperture-angle", type=float, default=2.0,
                        help="Max angle (degrees) from closest approach to scene center. "
                             "Pulses beyond this angle are discarded. (default: 2.0)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.cphd_file)[0] + ".sarbp"

    process_cphd(
        args.cphd_file,
        args.output,
        image_size=args.image_size,
        pixel_spacing_m=args.pixel_spacing,
        pulse_stride=args.pulse_stride,
        max_pulses=args.max_pulses,
        doppler_filter=args.doppler_filter,
        force_streaming=args.stream,
        aperture_angle_deg=args.aperture_angle,
        float_samples=args.float_samples,
    )


if __name__ == "__main__":
    main()
