"""
Heightmap Tile Stitcher - GUI Application

Visual tool for stitching independently-made heightmap tiles.
Two export modes:
  1. Stitch & Export Tiles - blend edges, save as separate tile files
  2. Merge to Single File - combine all tiles into one heightmap with blended seams

Usage:
    python stitch_tiles_ui.py

Requirements:
    pip install numpy Pillow
"""

import logging
import os
import re
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# File logger - writes everything to stitch_log.txt
# ---------------------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stitch_log.txt")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stitcher")
log.info(f"Python {sys.version}")
log.info(f"Pillow {Image.__version__}")
log.info(f"NumPy {np.__version__}")
log.info(f"Working dir: {os.getcwd()}")
log.info(f"Log file: {LOG_PATH}")


# ---------------------------------------------------------------------------
# Core stitching logic
# ---------------------------------------------------------------------------

def load_tile_png(path):
    log.debug(f"load_tile_png: opening {path}")
    img = Image.open(path)
    log.debug(f"load_tile_png: mode={img.mode}, size={img.size}")
    arr = np.array(img, dtype=np.uint16)
    log.debug(f"load_tile_png: array shape={arr.shape}, min={arr.min()}, max={arr.max()}")
    return arr


def save_tile_png(path, data):
    img = Image.fromarray(data.astype(np.uint16))
    img.save(path)


def load_tile_r16(path, size_hint=None):
    raw = np.fromfile(path, dtype=np.uint16)
    total = raw.size

    # Auto-detect size: try exact square root first
    sqrt = int(np.sqrt(total))
    if sqrt * sqrt == total:
        size = sqrt
        log.info(f"load_tile_r16: auto-detected size {size}x{size} from {total} values")
    elif size_hint and size_hint * size_hint == total:
        size = size_hint
    else:
        # Try common UE5 landscape sizes
        for try_size in [8129, 4033, 2017, 1009, 505, 253]:
            if try_size * try_size == total:
                size = try_size
                log.info(f"load_tile_r16: matched known size {size}x{size}")
                break
        else:
            raise ValueError(
                f"{path} has {total} values which is not a perfect square. "
                f"sqrt={sqrt}, closest known sizes: "
                f"8129({8129*8129}), 4033({4033*4033}), 2017({2017*2017}), 1009({1009*1009})")

    return raw.reshape((size, size))


def save_tile_r16(path, data):
    data.astype(np.uint16).tofile(path)


def find_tiles(tile_dir, fmt=None):
    # Supports both _x0_y0 and _y0_x0 naming conventions (Gaea uses y first)
    # If fmt is None, accept both .png and .r16
    pattern_xy = re.compile(r'_x(\d+)_y(\d+)')
    pattern_yx = re.compile(r'_y(\d+)_x(\d+)')
    valid_exts = [f".{fmt}"] if fmt else [".png", ".r16"]
    tiles = {}
    for f in os.listdir(tile_dir):
        if not any(f.lower().endswith(ext) for ext in valid_exts):
            continue
        match = pattern_xy.search(f)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            tiles[(x, y)] = os.path.join(tile_dir, f)
            continue
        match = pattern_yx.search(f)
        if match:
            y, x = int(match.group(1)), int(match.group(2))
            tiles[(x, y)] = os.path.join(tile_dir, f)
    return tiles


def _smooth_weights(n):
    """Smoothstep curve (ease-in-out) instead of linear."""
    t = np.linspace(0.0, 1.0, n)
    return 1.0 - (3.0 * t * t - 2.0 * t * t * t)


# ---------------------------------------------------------------------------
# Ultra quality: Graph Cut + 2D Poisson + Erosion
# ---------------------------------------------------------------------------

def _find_optimal_seam(tile_a_strip, tile_b_strip, axis):
    """
    Find the minimum-cost seam through the overlap zone using dynamic programming.
    The seam follows the path where the two tiles are most similar.
    Returns a mask: 1.0 = use tile_a, 0.0 = use tile_b.
    axis: 0 = vertical seam (for horizontal neighbors), 1 = horizontal seam (for vertical neighbors)
    """
    # Cost = squared difference between the two strips
    cost = (tile_a_strip - tile_b_strip) ** 2

    h, w = cost.shape

    if axis == 0:
        # Vertical seam: find best path top-to-bottom through columns (vectorized)
        dp = cost.copy()

        for row in range(1, h):
            center = dp[row - 1, :]
            left = np.empty(w)
            left[0] = np.inf
            left[1:] = dp[row - 1, :-1]
            right = np.empty(w)
            right[-1] = np.inf
            right[:-1] = dp[row - 1, 1:]
            dp[row, :] += np.minimum(center, np.minimum(left, right))

        # Trace back
        seam_cols = np.zeros(h, dtype=np.int32)
        seam_cols[-1] = np.argmin(dp[-1])
        for row in range(h - 2, -1, -1):
            sc = seam_cols[row + 1]
            lo = max(0, sc - 1)
            hi = min(w, sc + 2)
            seam_cols[row] = lo + np.argmin(dp[row, lo:hi])

        # Build mask vectorized
        falloff = max(w // 6, 8)
        col_idx = np.arange(w).reshape(1, w)
        seam_2d = seam_cols.reshape(h, 1)
        t = (col_idx - (seam_2d - falloff)) / (2.0 * falloff)
        t = np.clip(t, 0.0, 1.0)
        mask = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    else:
        # Horizontal seam: find best path left-to-right through rows (vectorized)
        dp = cost.copy()

        for col in range(1, w):
            center = dp[:, col - 1]
            up = np.empty(h)
            up[0] = np.inf
            up[1:] = dp[:-1, col - 1]
            down = np.empty(h)
            down[-1] = np.inf
            down[:-1] = dp[1:, col - 1]
            dp[:, col] += np.minimum(center, np.minimum(up, down))

        seam_rows = np.zeros(w, dtype=np.int32)
        seam_rows[-1] = np.argmin(dp[:, -1])
        for col in range(w - 2, -1, -1):
            sr = seam_rows[col + 1]
            lo = max(0, sr - 1)
            hi = min(h, sr + 2)
            seam_rows[col] = lo + np.argmin(dp[lo:hi, col])

        # Build mask vectorized
        falloff = max(h // 6, 8)
        row_idx = np.arange(h).reshape(h, 1)
        seam_2d = seam_rows.reshape(1, w)
        t = (row_idx - (seam_2d - falloff)) / (2.0 * falloff)
        t = np.clip(t, 0.0, 1.0)
        mask = 1.0 - (3.0 * t * t - 2.0 * t * t * t)

        mask = np.zeros((h, w), dtype=np.float64)
        falloff = max(h // 8, 4)
        for col in range(w):
            sr = seam_rows[col]
            for row in range(h):
                if row <= sr - falloff:
                    mask[row, col] = 1.0
                elif row >= sr + falloff:
                    mask[row, col] = 0.0
                else:
                    t = (row - (sr - falloff)) / (2.0 * falloff)
                    mask[row, col] = 1.0 - (3.0 * t * t - 2.0 * t * t * t)

    return mask


def _poisson_solve_2d(composite, mask, tile_a_strip, tile_b_strip, iterations=200):
    """
    Full 2D Poisson blend. Solves for heights that have the same gradients
    as the source tiles but are continuous across the seam.
    Uses Gauss-Seidel relaxation.
    """
    h, w = composite.shape
    result = composite.copy()

    # Compute target gradients from the masked combination of both tiles
    # Use gradients from tile_a where mask > 0.5, tile_b where mask <= 0.5
    grad_x = np.zeros_like(result)
    grad_y = np.zeros_like(result)

    grad_x_a = np.diff(tile_a_strip, axis=1, prepend=tile_a_strip[:, :1])
    grad_y_a = np.diff(tile_a_strip, axis=0, prepend=tile_a_strip[:1, :])
    grad_x_b = np.diff(tile_b_strip, axis=1, prepend=tile_b_strip[:, :1])
    grad_y_b = np.diff(tile_b_strip, axis=0, prepend=tile_b_strip[:1, :])

    # Blend gradients using the seam mask
    grad_x = grad_x_a * mask + grad_x_b * (1.0 - mask)
    grad_y = grad_y_a * mask + grad_y_b * (1.0 - mask)

    # Laplacian of the target (divergence of gradient field)
    laplacian = np.zeros_like(result)
    laplacian[:, 1:] += grad_x[:, 1:] - grad_x[:, :-1]
    laplacian[1:, :] += grad_y[1:, :] - grad_y[:-1, :]

    # Interior mask: don't touch boundary pixels
    interior = np.ones((h, w), dtype=bool)
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False

    # Gauss-Seidel relaxation
    for _ in range(iterations):
        new_val = (
            np.roll(result, 1, axis=0) +
            np.roll(result, -1, axis=0) +
            np.roll(result, 1, axis=1) +
            np.roll(result, -1, axis=1) -
            laplacian
        ) / 4.0
        result[interior] = new_val[interior]

    return result


def _hydraulic_erosion(heightmap, iterations=50, drop_count=None):
    """
    Vectorized hydraulic erosion - processes batches of droplets simultaneously.
    ~50-100x faster than single-droplet loop.
    """
    h, w = heightmap.shape
    result = heightmap.astype(np.float64).copy()

    if drop_count is None:
        drop_count = max(20000, h * w // 20)

    rng = np.random.RandomState(42)
    batch_size = min(4096, drop_count)
    max_steps = 80

    inertia = 0.3
    capacity = 16.0
    deposition_rate = 0.15
    erosion_rate = 0.4
    evaporation = 0.015
    min_slope = 0.5

    for batch_start in range(0, drop_count, batch_size):
        n = min(batch_size, drop_count - batch_start)

        px = rng.uniform(3, w - 4, size=n)
        py = rng.uniform(3, h - 4, size=n)
        ddx = np.zeros(n)
        ddy = np.zeros(n)
        sediment = np.zeros(n)
        water = np.ones(n)
        speed = np.ones(n)
        alive = np.ones(n, dtype=bool)

        for step in range(max_steps):
            if not alive.any():
                break

            ix = px.astype(np.int32)
            iy = py.astype(np.int32)

            oob = (ix < 2) | (ix >= w - 3) | (iy < 2) | (iy >= h - 3)
            alive &= ~oob
            if not alive.any():
                break

            a_ix = np.clip(ix, 0, w - 2)
            a_iy = np.clip(iy, 0, h - 2)
            a_ix1 = np.minimum(a_ix + 1, w - 1)
            a_iy1 = np.minimum(a_iy + 1, h - 1)

            h00 = result[a_iy, a_ix]
            h10 = result[a_iy, a_ix1]
            h01 = result[a_iy1, a_ix]
            h11 = result[a_iy1, a_ix1]

            fx = px - ix
            fy = py - iy

            gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx

            ddx = ddx * inertia - gx * (1 - inertia)
            ddy = ddy * inertia - gy * (1 - inertia)
            length = np.maximum(np.sqrt(ddx * ddx + ddy * ddy), 1e-6)
            ddx /= length
            ddy /= length

            new_px = px + ddx
            new_py = py + ddy
            new_ix = np.clip(new_px.astype(np.int32), 0, w - 1)
            new_iy = np.clip(new_py.astype(np.int32), 0, h - 1)

            oob2 = (new_px < 2) | (new_px >= w - 3) | (new_py < 2) | (new_py >= h - 3)
            alive &= ~oob2

            new_h = result[new_iy, new_ix]
            old_h = result[a_iy, a_ix]
            delta_h = new_h - old_h

            cap = np.maximum(-delta_h, min_slope) * speed * water * capacity

            # Deposit
            depositing = alive & ((sediment > cap) | (delta_h > 0))
            dep_amount = np.where(delta_h <= 0,
                                  (sediment - cap) * deposition_rate,
                                  np.minimum(sediment, delta_h))
            dep_amount = np.maximum(dep_amount, 0) * depositing

            # Erode
            eroding = alive & ~depositing
            ero_amount = np.minimum((cap - sediment) * erosion_rate, np.maximum(-delta_h, 0))
            ero_amount = np.maximum(ero_amount, 0) * eroding

            # Apply with brush (center + 4 neighbors)
            valid_dep = depositing & (dep_amount > 0)
            valid_ero = eroding & (ero_amount > 0)

            if valid_dep.any():
                dy_arr, dx_arr = a_iy[valid_dep], a_ix[valid_dep]
                val = dep_amount[valid_dep]
                np.add.at(result, (dy_arr, dx_arr), val)
                np.add.at(result, (np.clip(dy_arr - 1, 0, h - 1), dx_arr), val * 0.25)
                np.add.at(result, (np.clip(dy_arr + 1, 0, h - 1), dx_arr), val * 0.25)
                np.add.at(result, (dy_arr, np.clip(dx_arr - 1, 0, w - 1)), val * 0.25)
                np.add.at(result, (dy_arr, np.clip(dx_arr + 1, 0, w - 1)), val * 0.25)

            if valid_ero.any():
                dy_arr, dx_arr = a_iy[valid_ero], a_ix[valid_ero]
                val = ero_amount[valid_ero]
                np.add.at(result, (dy_arr, dx_arr), -val)
                np.add.at(result, (np.clip(dy_arr - 1, 0, h - 1), dx_arr), -val * 0.25)
                np.add.at(result, (np.clip(dy_arr + 1, 0, h - 1), dx_arr), -val * 0.25)
                np.add.at(result, (dy_arr, np.clip(dx_arr - 1, 0, w - 1)), -val * 0.25)
                np.add.at(result, (dy_arr, np.clip(dx_arr + 1, 0, w - 1)), -val * 0.25)

            sediment += ero_amount - dep_amount
            px = np.where(alive, new_px, px)
            py = np.where(alive, new_py, py)
            speed = np.sqrt(np.maximum(speed * speed + delta_h, 0.01))
            water *= (1 - evaporation)
            alive &= (water > 0.01)

    return result


def _thermal_erosion(heightmap, iterations=30, talus_angle=4.0):
    """
    Thermal/talus erosion - material slides downhill when slope exceeds
    the talus angle. Creates natural scree slopes and softens sharp edges.
    Fully vectorized with numpy.
    """
    result = heightmap.astype(np.float64).copy()
    h, w = result.shape

    # Talus threshold - max height difference before material slides
    # Higher = more aggressive erosion
    talus = talus_angle

    for _ in range(iterations):
        # Compute height differences to all 4 neighbors
        diff_right = np.zeros_like(result)
        diff_left = np.zeros_like(result)
        diff_down = np.zeros_like(result)
        diff_up = np.zeros_like(result)

        diff_right[:, :-1] = result[:, :-1] - result[:, 1:]
        diff_left[:, 1:] = result[:, 1:] - result[:, :-1]
        diff_down[:-1, :] = result[:-1, :] - result[1:, :]
        diff_up[1:, :] = result[1:, :] - result[:-1, :]

        # Find max difference at each cell
        d_max = np.maximum(np.maximum(diff_right, diff_left),
                           np.maximum(diff_down, diff_up))

        # Only erode where slope exceeds talus angle
        erode_mask = d_max > talus

        # Total excess difference
        d_total = np.zeros_like(result)
        excess_r = np.maximum(diff_right - talus, 0)
        excess_l = np.maximum(diff_left - talus, 0)
        excess_d = np.maximum(diff_down - talus, 0)
        excess_u = np.maximum(diff_up - talus, 0)
        d_total = excess_r + excess_l + excess_d + excess_u

        # Avoid division by zero
        d_total_safe = np.maximum(d_total, 1e-10)

        # Amount to move = half the excess (distribute to neighbors)
        move = d_max * 0.5 * erode_mask

        # Distribute to each neighbor proportionally
        result[:, :-1] -= move[:, :-1] * (excess_r[:, :-1] / d_total_safe[:, :-1]) * 0.25
        result[:, 1:] += move[:, :-1] * (excess_r[:, :-1] / d_total_safe[:, :-1]) * 0.25

        result[:, 1:] -= move[:, 1:] * (excess_l[:, 1:] / d_total_safe[:, 1:]) * 0.25
        result[:, :-1] += move[:, 1:] * (excess_l[:, 1:] / d_total_safe[:, 1:]) * 0.25

        result[:-1, :] -= move[:-1, :] * (excess_d[:-1, :] / d_total_safe[:-1, :]) * 0.25
        result[1:, :] += move[:-1, :] * (excess_d[:-1, :] / d_total_safe[:-1, :]) * 0.25

        result[1:, :] -= move[1:, :] * (excess_u[1:, :] / d_total_safe[1:, :]) * 0.25
        result[:-1, :] += move[1:, :] * (excess_u[1:, :] / d_total_safe[1:, :]) * 0.25

    return result


def _laplacian_pyramid_blend(strip_a, strip_b, seam_mask, num_levels=6):
    """
    Multi-scale Laplacian pyramid blend.
    Blends low frequencies (big terrain shapes) over a wide area and
    high frequencies (fine detail) over a narrow area along the seam.
    This is what Photoshop uses for seamless compositing - adapted for terrain.
    """
    h, w = strip_a.shape

    # Build Gaussian pyramids for both strips and the mask
    def _build_gaussian_pyramid(img, levels):
        pyramid = [img.copy()]
        current = img.copy()
        for _ in range(levels - 1):
            # Downsample by 2x using area averaging
            new_h = max(1, current.shape[0] // 2)
            new_w = max(1, current.shape[1] // 2)
            pil_img = Image.fromarray(current)
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            current = np.array(pil_img, dtype=np.float64)
            pyramid.append(current)
        return pyramid

    def _build_laplacian_pyramid(gauss_pyr):
        lap_pyr = []
        for i in range(len(gauss_pyr) - 1):
            # Upsample the next level
            target_h, target_w = gauss_pyr[i].shape
            upsampled = np.array(
                Image.fromarray(gauss_pyr[i + 1]).resize((target_w, target_h), Image.BILINEAR),
                dtype=np.float64)
            # Laplacian = current level - upsampled next level (captures the detail)
            lap_pyr.append(gauss_pyr[i] - upsampled)
        # Last level is just the lowest frequency
        lap_pyr.append(gauss_pyr[-1])
        return lap_pyr

    # Clamp levels based on image size
    min_dim = min(h, w)
    num_levels = min(num_levels, max(2, int(np.log2(min_dim))))

    ga_pyr = _build_gaussian_pyramid(strip_a, num_levels)
    gb_pyr = _build_gaussian_pyramid(strip_b, num_levels)
    gm_pyr = _build_gaussian_pyramid(seam_mask, num_levels)

    la_pyr = _build_laplacian_pyramid(ga_pyr)
    lb_pyr = _build_laplacian_pyramid(gb_pyr)

    # Blend each level using the mask at that resolution
    blended_pyr = []
    for la, lb, gm in zip(la_pyr, lb_pyr, gm_pyr):
        # Resize mask to match this level if needed
        if gm.shape != la.shape:
            gm = np.array(
                Image.fromarray(gm).resize((la.shape[1], la.shape[0]), Image.BILINEAR),
                dtype=np.float64)
        blended_pyr.append(la * gm + lb * (1.0 - gm))

    # Reconstruct from blended pyramid
    result = blended_pyr[-1]
    for i in range(len(blended_pyr) - 2, -1, -1):
        target_h, target_w = blended_pyr[i].shape
        upsampled = np.array(
            Image.fromarray(result).resize((target_w, target_h), Image.BILINEAR),
            dtype=np.float64)
        result = upsampled + blended_pyr[i]

    return result


def _generate_noise_mask(height, width, seed=42):
    """Generate a fractal noise mask for organic seam boundaries.
    Returns values 0-1 with organic, terrain-like variation."""
    rng = np.random.RandomState(seed)
    mask = np.zeros((height, width), dtype=np.float64)

    # Layer multiple octaves of noise for fractal look
    for octave in range(5):
        freq = 2 ** octave
        amp = 1.0 / (2 ** octave)
        # Generate low-res noise and upscale
        noise_h = max(2, height // (32 // freq))
        noise_w = max(2, width // (32 // freq))
        small = rng.rand(noise_h, noise_w)
        # Bilinear upscale using PIL
        from PIL import Image as _Img
        up = np.array(_Img.fromarray(small).resize((width, height), _Img.BILINEAR))
        mask += up * amp

    # Normalize to 0-1
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)
    return mask


def _poisson_blend_1d(strip_a, strip_b, axis):
    """
    Gradient-domain (Poisson) blend between two strips.
    Blends the gradients from both strips then integrates to get
    heights that preserve terrain detail from both sides.
    axis=1 for horizontal blending, axis=0 for vertical.
    """
    h, w = strip_a.shape

    # Compute gradients along the blend axis
    grad_a = np.diff(strip_a, axis=axis)
    grad_b = np.diff(strip_b, axis=axis)

    # Blend gradients with position-based weight
    if axis == 1:
        n = w
        t = np.linspace(0.0, 1.0, n - 1).reshape(1, n - 1)
    else:
        n = h
        t = np.linspace(0.0, 1.0, n - 1).reshape(n - 1, 1)

    # Smoothstep weight for gradient blending
    t_smooth = 3.0 * t * t - 2.0 * t * t * t
    blended_grad = grad_a * (1.0 - t_smooth) + grad_b * t_smooth

    # Integrate gradients starting from strip_a's first row/col
    if axis == 1:
        result = np.zeros_like(strip_a)
        result[:, 0] = strip_a[:, 0]
        for i in range(1, w):
            result[:, i] = result[:, i - 1] + blended_grad[:, i - 1]
        # Anchor the end to strip_b's last col with smooth correction
        end_diff = strip_b[:, -1] - result[:, -1]
        correction = end_diff.reshape(-1, 1) * np.linspace(0, 1, w).reshape(1, w)
        result += correction
    else:
        result = np.zeros_like(strip_a)
        result[0, :] = strip_a[0, :]
        for i in range(1, h):
            result[i, :] = result[i - 1, :] + blended_grad[i - 1, :]
        end_diff = strip_b[-1, :] - result[-1, :]
        correction = end_diff.reshape(1, -1) * np.linspace(0, 1, h).reshape(h, 1)
        result += correction

    return result


def _gaussian_smooth_1d(arr, sigma):
    """Smooth a 1D array with a Gaussian kernel. Heavy smoothing to match terrain scale."""
    size = int(sigma * 6) | 1  # Ensure odd
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Pad to avoid edge artifacts
    padded = np.pad(arr, size // 2, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def _gaussian_blur_2d(data, sigma):
    """Apply Gaussian blur to a 2D array. Separable for speed."""
    size = int(sigma * 6) | 1
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # Horizontal pass
    padded = np.pad(data, ((0, 0), (size // 2, size // 2)), mode='reflect')
    h_blurred = np.zeros_like(data)
    for i in range(data.shape[0]):
        h_blurred[i] = np.convolve(padded[i], kernel, mode='valid')[:data.shape[1]]

    # Vertical pass
    padded = np.pad(h_blurred, ((size // 2, size // 2), (0, 0)), mode='reflect')
    result = np.zeros_like(data)
    for j in range(data.shape[1]):
        result[:, j] = np.convolve(padded[:, j], kernel, mode='valid')[:data.shape[0]]

    return result


def _deep_height_equalize_h(a_f, b_f, margin):
    """Pass 1: Deep height equalization for horizontal neighbors.
    Extends terrain shape from each tile deep into the other with natural falloff.
    Instead of just shifting values, blends the actual terrain profile across."""
    h = a_f.shape[0]
    w_a, w_b = a_f.shape[1], b_f.shape[1]

    # Step 1: Global level matching with ultra-wide Gaussian
    sample_w = min(margin, 128)
    a_edge_band = a_f[:, -sample_w:].mean(axis=1)
    b_edge_band = b_f[:, :sample_w].mean(axis=1)
    sigma = max(128, h // 8)
    diff = _gaussian_smooth_1d(a_edge_band - b_edge_band, sigma)

    # Sharp falloff - strong near seam, drops dramatically at ~30% depth
    b_x = np.arange(w_b, dtype=np.float64) / w_b  # 0 to 1
    b_weight = np.exp(-4.0 * (b_x / 0.3) ** 3).reshape(1, w_b)  # Cubic exp = steep drop
    b_f += diff.reshape(-1, 1) * b_weight * 0.4

    a_x = np.arange(w_a, dtype=np.float64) / w_a
    a_weight = np.exp(-4.0 * ((1.0 - a_x) / 0.3) ** 3).reshape(1, w_a)
    a_f -= diff.reshape(-1, 1) * a_weight * 0.3

    # Step 2: Terrain shape extension - blend A's terrain features into B
    # Only extend where A actually has terrain features (not flat areas)
    extend_depth = min(int(w_b * 0.5), w_b - 1)
    source_depth = min(margin * 2, w_a // 2)

    # Get A's terrain profile at the edge (smoothed to terrain scale)
    a_profile = a_f[:, -source_depth:].copy()
    a_profile_smooth = _gaussian_blur_2d(a_profile, max(8, source_depth // 8))

    # Measure how much terrain detail A has at each row (feature strength)
    # High values = mountains/ridges, low values = flat plains
    a_edge_detail = np.abs(a_f[:, -source_depth:] - a_profile_smooth).mean(axis=1)
    a_edge_detail = _gaussian_smooth_1d(a_edge_detail, max(32, h // 16))
    # Normalize to 0-1
    detail_max = max(a_edge_detail.max(), 1.0)
    feature_strength = np.clip(a_edge_detail / detail_max, 0.1, 1.0)

    for col in range(extend_depth):
        t = col / extend_depth
        # Steeper falloff further from seam
        influence = (1.0 - t) ** 2.0

        src_col = min(int((1.0 - t) * source_depth), source_depth - 1)

        # Scale influence by feature strength - only push where A has terrain
        row_influence = influence * feature_strength * 0.5
        b_f[:, col] = b_f[:, col] * (1.0 - row_influence) + a_profile_smooth[:, src_col] * row_influence

    # Step 3: Match contrast at the seam
    a_band = a_f[:, -sample_w:]
    b_band = b_f[:, :sample_w]
    a_std = max(a_band.std(), 1.0)
    b_std = max(b_band.std(), 1.0)
    scale = np.clip(a_std / b_std, 0.7, 1.4)
    if abs(scale - 1.0) > 0.02:
        b_mean_local = b_f[:, :extend_depth].mean()
        b_centered = b_f[:, :extend_depth] - b_mean_local
        blend_w = np.exp(-0.5 * (np.arange(extend_depth, dtype=np.float64) / (extend_depth * 0.4)) ** 2).reshape(1, extend_depth)
        b_f[:, :extend_depth] = b_mean_local + b_centered * (1.0 + (scale - 1.0) * blend_w)


def _deep_height_equalize_v(a_f, b_f, margin):
    """Pass 1: Deep height equalization for vertical neighbors.
    Extends terrain shape with natural falloff."""
    w = a_f.shape[1]
    h_a, h_b = a_f.shape[0], b_f.shape[0]

    sample_h = min(margin, 128)
    a_edge_band = a_f[-sample_h:, :].mean(axis=0)
    b_edge_band = b_f[:sample_h, :].mean(axis=0)
    sigma = max(128, w // 8)
    diff = _gaussian_smooth_1d(a_edge_band - b_edge_band, sigma)

    # Ultra-wide Gaussian correction
    # Sharp falloff - drops dramatically at ~30% depth
    b_y = np.arange(h_b, dtype=np.float64) / h_b
    b_weight = np.exp(-4.0 * (b_y / 0.3) ** 3).reshape(h_b, 1)
    b_f += diff.reshape(1, -1) * b_weight * 0.4

    a_y = np.arange(h_a, dtype=np.float64) / h_a
    a_weight = np.exp(-4.0 * ((1.0 - a_y) / 0.3) ** 3).reshape(h_a, 1)
    a_f -= diff.reshape(1, -1) * a_weight * 0.3

    # Terrain shape extension into B - only where features exist
    extend_depth = min(int(h_b * 0.5), h_b - 1)
    source_depth = min(margin * 2, h_a // 2)

    a_profile = a_f[-source_depth:, :].copy()
    a_profile_smooth = _gaussian_blur_2d(a_profile, max(8, source_depth // 8))

    a_edge_detail = np.abs(a_f[-source_depth:, :] - a_profile_smooth).mean(axis=0)
    a_edge_detail = _gaussian_smooth_1d(a_edge_detail, max(32, w // 16))
    detail_max = max(a_edge_detail.max(), 1.0)
    feature_strength = np.clip(a_edge_detail / detail_max, 0.1, 1.0)

    for row in range(extend_depth):
        t = row / extend_depth
        influence = (1.0 - t) ** 2.0
        src_row = min(int((1.0 - t) * source_depth), source_depth - 1)
        col_influence = influence * feature_strength * 0.5
        b_f[row, :] = b_f[row, :] * (1.0 - col_influence) + a_profile_smooth[src_row, :] * col_influence

    # Match contrast
    a_band = a_f[-sample_h:, :]
    b_band = b_f[:sample_h, :]
    a_std = max(a_band.std(), 1.0)
    b_std = max(b_band.std(), 1.0)
    scale = np.clip(a_std / b_std, 0.7, 1.4)
    if abs(scale - 1.0) > 0.02:
        b_mean_local = b_f[:extend_depth, :].mean()
        b_centered = b_f[:extend_depth, :] - b_mean_local
        blend_h = np.exp(-0.5 * (np.arange(extend_depth, dtype=np.float64) / (extend_depth * 0.4)) ** 2).reshape(extend_depth, 1)
        b_f[:extend_depth, :] = b_mean_local + b_centered * (1.0 + (scale - 1.0) * blend_h)


def _feature_spill_h(a_f, b_f, margin):
    """Spill bright features from each tile into the other at the seam.
    Where tile A's edge is brighter than tile B, extend A's terrain into B
    with a soft falloff, and vice versa. This prevents features from being
    cut off at the tile boundary."""
    h = a_f.shape[0]
    spill_depth = margin * 2  # How far features can spill

    # Sample edge strips
    a_edge = a_f[:, -margin:]
    b_edge = b_f[:, :margin]

    # Per-row: which tile is brighter at each row
    a_brightness = a_edge.mean(axis=1)
    b_brightness = b_edge.mean(axis=1)

    # Gaussian smooth to get broad brightness trends
    sigma = max(32, h // 16)
    a_bright_smooth = _gaussian_smooth_1d(a_brightness, sigma)
    b_bright_smooth = _gaussian_smooth_1d(b_brightness, sigma)

    diff = a_bright_smooth - b_bright_smooth  # positive = A is brighter

    # Where A is brighter: spill A's features into B
    # Where B is brighter: spill B's features into A
    spill_w = min(spill_depth, b_f.shape[1], a_f.shape[1])

    # Gaussian falloff for the spill
    spill_x = np.arange(spill_w, dtype=np.float64)
    spill_fade = np.exp(-0.5 * (spill_x / (spill_w * 0.3)) ** 2).reshape(1, spill_w)

    # Sample a wide band at each edge (not single pixel) for smoother values
    edge_band = min(margin // 2, 64)
    a_edge_vals = a_f[:, -edge_band:].mean(axis=1)
    b_edge_vals = b_f[:, :edge_band].mean(axis=1)

    # Smooth the edge values to avoid per-row noise
    edge_sigma = max(16, h // 32)
    a_edge_smooth = _gaussian_smooth_1d(a_edge_vals, edge_sigma)
    b_edge_smooth = _gaussian_smooth_1d(b_edge_vals, edge_sigma)

    for row in range(h):
        d = diff[row]
        if abs(d) < 20:
            continue

        strength = min(abs(d) / 5000.0, 0.6)

        if d > 0:
            a_val = a_edge_smooth[row]
            b_vals = b_f[row, :spill_w]
            blend = a_val * spill_fade[0] * strength + b_vals * (1.0 - spill_fade[0] * strength)
            b_f[row, :spill_w] = blend
        else:
            b_val = b_edge_smooth[row]
            a_vals = a_f[row, -spill_w:]
            spill_fade_rev = spill_fade[0, ::-1]
            blend = b_val * spill_fade_rev * strength + a_vals * (1.0 - spill_fade_rev * strength)
            a_f[row, -spill_w:] = blend



def _feature_spill_v(a_f, b_f, margin):
    """Spill bright features vertically between tiles."""
    w = a_f.shape[1]
    spill_depth = margin * 2

    a_edge = a_f[-margin:, :]
    b_edge = b_f[:margin, :]

    a_brightness = a_edge.mean(axis=0)
    b_brightness = b_edge.mean(axis=0)

    sigma = max(32, w // 16)
    a_bright_smooth = _gaussian_smooth_1d(a_brightness, sigma)
    b_bright_smooth = _gaussian_smooth_1d(b_brightness, sigma)

    diff = a_bright_smooth - b_bright_smooth

    spill_h = min(spill_depth, b_f.shape[0], a_f.shape[0])
    spill_y = np.arange(spill_h, dtype=np.float64)
    spill_fade = np.exp(-0.5 * (spill_y / (spill_h * 0.3)) ** 2).reshape(spill_h, 1)

    edge_band = min(margin // 2, 64)
    a_edge_vals = a_f[-edge_band:, :].mean(axis=0)
    b_edge_vals = b_f[:edge_band, :].mean(axis=0)

    edge_sigma = max(16, w // 32)
    a_edge_smooth = _gaussian_smooth_1d(a_edge_vals, edge_sigma)
    b_edge_smooth = _gaussian_smooth_1d(b_edge_vals, edge_sigma)

    for col in range(w):
        d = diff[col]
        if abs(d) < 20:
            continue

        strength = min(abs(d) / 5000.0, 0.6)

        if d > 0:
            a_val = a_edge_smooth[col]
            b_vals = b_f[:spill_h, col]
            blend = a_val * spill_fade[:, 0] * strength + b_vals * (1.0 - spill_fade[:, 0] * strength)
            b_f[:spill_h, col] = blend
        else:
            b_val = b_edge_smooth[col]
            a_vals = a_f[-spill_h:, col]
            spill_fade_rev = spill_fade[::-1, 0]
            blend = b_val * spill_fade_rev * strength + a_vals * (1.0 - spill_fade_rev * strength)
            a_f[-spill_h:, col] = blend



def blend_horizontal(tile_a, tile_b, margin, quality="high", opts=None):
    if opts is None:
        opts = {"blur": False, "height_preserve": True, "feature_spill": True, "terrain_extend": True}

    a_f = tile_a.astype(np.float64)
    b_f = tile_b.astype(np.float64)
    h = a_f.shape[0]

    if quality == "fast":
        height_diff = a_f[:, -1].mean() - b_f[:, 0].mean()
        if abs(height_diff) > 1.0:
            fade = np.linspace(1.0, 0.0, margin).reshape(1, margin)
            b_f[:, :margin] += height_diff * fade
        w = _smooth_weights(margin).reshape(1, margin)
        blended = a_f[:, -margin:] * w + b_f[:, :margin] * (1.0 - w)
    else:
        # Pass 1: Deep height equalization (terrain extend)
        if opts.get("terrain_extend", True):
            _deep_height_equalize_h(a_f, b_f, margin)

        # Pass 1b: Feature spill
        if opts.get("feature_spill", True):
            _feature_spill_h(a_f, b_f, margin)

        tile_a[:] = np.clip(a_f, 0, 65535).astype(np.uint16)
        tile_b[:] = np.clip(b_f, 0, 65535).astype(np.uint16)

        strip_a = a_f[:, -margin:]
        strip_b = b_f[:, :margin]

        if quality == "high":
            poisson_result = _poisson_blend_1d(strip_a, strip_b, axis=1)
            w = _smooth_weights(margin).reshape(1, margin)
            crossfade = poisson_result * w + strip_b * (1.0 - w)

            if opts.get("height_preserve", True):
                max_blend = np.maximum(strip_a, strip_b)
                diff_ratio = np.abs(strip_a - strip_b) / max(np.abs(strip_a - strip_b).max(), 1)
                blended = crossfade * (1.0 - diff_ratio * 0.5) + max_blend * (diff_ratio * 0.5)
            else:
                blended = crossfade

            if opts.get("blur", False):
                blended = _gaussian_blur_2d(blended, max(8, margin // 10))

        elif quality == "ultra":
            log.info("ULTRA: Feature cloning blend...")

            # Step 1: Simple smoothstep base blend
            w_blend = _smooth_weights(margin).reshape(1, margin)
            base = strip_a * w_blend + strip_b * (1.0 - w_blend)

            # Step 2: Find bright features on both sides and clone them across
            # Sample wider than margin to find features to clone
            clone_depth = min(margin * 3, a_f.shape[1] // 2, b_f.shape[1] // 2)
            a_source = a_f[:, -clone_depth:]
            b_source = b_f[:, :clone_depth]

            # Extract terrain detail by subtracting the local average
            blur_sigma = max(32, margin // 4)
            a_smooth = _gaussian_blur_2d(a_source, blur_sigma)
            b_smooth = _gaussian_blur_2d(b_source, blur_sigma)
            a_detail = a_source - a_smooth  # ridges, valleys, features
            b_detail = b_source - b_smooth

            # Find where features are prominent (high local contrast)
            a_prominence = np.abs(a_detail)
            b_prominence = np.abs(b_detail)

            # Threshold: only clone the top features
            a_thresh = np.percentile(a_prominence, 70)
            b_thresh = np.percentile(b_prominence, 70)

            # Build clone masks - where features are strong
            a_feature_mask = np.clip((a_prominence - a_thresh) / max(a_thresh, 1), 0, 1)
            b_feature_mask = np.clip((b_prominence - b_thresh) / max(b_thresh, 1), 0, 1)

            # Gaussian smooth the masks so cloned features blend softly
            a_feature_mask = _gaussian_blur_2d(a_feature_mask, blur_sigma // 2)
            b_feature_mask = _gaussian_blur_2d(b_feature_mask, blur_sigma // 2)

            # Clone A's features into the blend zone
            # Map source columns to blend zone columns
            for col in range(margin):
                # How far into the blend zone (0=left/A side, 1=right/B side)
                t = col / margin
                # Source column in A (sample from deeper as we go further right)
                src_col_a = int(clone_depth - 1 - col * clone_depth / margin)
                src_col_a = max(0, min(src_col_a, clone_depth - 1))
                # Clone strength fades as we cross into B's territory
                clone_strength_a = (1.0 - t) * 1.0
                # Apply A's detail features onto the base blend
                base[:, col] += a_detail[:, src_col_a] * a_feature_mask[:, src_col_a] * clone_strength_a

            # Clone B's features into the blend zone
            for col in range(margin):
                t = col / margin
                src_col_b = int(col * clone_depth / margin)
                src_col_b = max(0, min(src_col_b, clone_depth - 1))
                clone_strength_b = t * 1.0
                base[:, col] += b_detail[:, src_col_b] * b_feature_mask[:, src_col_b] * clone_strength_b

            blended = base
            log.info("ULTRA: Feature clone blend complete.")

    blended = np.clip(blended, 0, 65535).astype(np.uint16)
    tile_a[:, -margin:] = blended
    tile_b[:, :margin] = blended


def blend_vertical(tile_a, tile_b, margin, quality="high", opts=None):
    if opts is None:
        opts = {"blur": False, "height_preserve": True, "feature_spill": True, "terrain_extend": True}

    a_f = tile_a.astype(np.float64)
    b_f = tile_b.astype(np.float64)
    w = a_f.shape[1]

    if quality == "fast":
        height_diff = a_f[-1, :].mean() - b_f[0, :].mean()
        if abs(height_diff) > 1.0:
            fade = np.linspace(1.0, 0.0, margin).reshape(margin, 1)
            b_f[:margin, :] += height_diff * fade
        wt = _smooth_weights(margin).reshape(margin, 1)
        blended = a_f[-margin:, :] * wt + b_f[:margin, :] * (1.0 - wt)
    else:
        if opts.get("terrain_extend", True):
            _deep_height_equalize_v(a_f, b_f, margin)

        if opts.get("feature_spill", True):
            _feature_spill_v(a_f, b_f, margin)

        tile_a[:] = np.clip(a_f, 0, 65535).astype(np.uint16)
        tile_b[:] = np.clip(b_f, 0, 65535).astype(np.uint16)

        strip_a = a_f[-margin:, :]
        strip_b = b_f[:margin, :]

        if quality == "high":
            poisson_result = _poisson_blend_1d(strip_a, strip_b, axis=0)
            wt = _smooth_weights(margin).reshape(margin, 1)
            crossfade = poisson_result * wt + strip_b * (1.0 - wt)

            if opts.get("height_preserve", True):
                max_blend = np.maximum(strip_a, strip_b)
                diff_ratio = np.abs(strip_a - strip_b) / max(np.abs(strip_a - strip_b).max(), 1)
                blended = crossfade * (1.0 - diff_ratio * 0.5) + max_blend * (diff_ratio * 0.5)
            else:
                blended = crossfade

            if opts.get("blur", False):
                blended = _gaussian_blur_2d(blended, max(8, margin // 10))

        elif quality == "ultra":
            log.info("ULTRA V: Feature cloning blend...")

            # Step 1: Simple smoothstep base blend
            wt_blend = _smooth_weights(margin).reshape(margin, 1)
            base = strip_a * wt_blend + strip_b * (1.0 - wt_blend)

            # Step 2: Clone bright features across the seam
            clone_depth = min(margin * 3, a_f.shape[0] // 2, b_f.shape[0] // 2)
            a_source = a_f[-clone_depth:, :]
            b_source = b_f[:clone_depth, :]

            blur_sigma = max(32, margin // 4)
            a_smooth = _gaussian_blur_2d(a_source, blur_sigma)
            b_smooth = _gaussian_blur_2d(b_source, blur_sigma)
            a_detail = a_source - a_smooth
            b_detail = b_source - b_smooth

            a_prominence = np.abs(a_detail)
            b_prominence = np.abs(b_detail)
            a_thresh = np.percentile(a_prominence, 70)
            b_thresh = np.percentile(b_prominence, 70)

            a_feature_mask = np.clip((a_prominence - a_thresh) / max(a_thresh, 1), 0, 1)
            b_feature_mask = np.clip((b_prominence - b_thresh) / max(b_thresh, 1), 0, 1)
            a_feature_mask = _gaussian_blur_2d(a_feature_mask, blur_sigma // 2)
            b_feature_mask = _gaussian_blur_2d(b_feature_mask, blur_sigma // 2)

            for row in range(margin):
                t = row / margin
                src_row_a = int(clone_depth - 1 - row * clone_depth / margin)
                src_row_a = max(0, min(src_row_a, clone_depth - 1))
                clone_strength_a = (1.0 - t) * 1.0
                base[row, :] += a_detail[src_row_a, :] * a_feature_mask[src_row_a, :] * clone_strength_a

            for row in range(margin):
                t = row / margin
                src_row_b = int(t * clone_depth)
                src_row_b = max(0, min(src_row_b, clone_depth - 1))
                clone_strength_b = t * 1.0
                base[row, :] += b_detail[src_row_b, :] * b_feature_mask[src_row_b, :] * clone_strength_b

            blended = base
            log.info("ULTRA V: Feature clone blend complete.")

    blended = np.clip(blended, 0, 65535).astype(np.uint16)
    tile_a[-margin:, :] = blended
    tile_b[:margin, :] = blended


def merge_tiles_to_single(tile_data, margin, quality="high", log_fn=None, opts=None):
    """
    Merge all tiles into a single heightmap array.
    For ultra quality, uses the full blend pipeline (graph cut + Poisson + erosion)
    by blending tile pairs directly before compositing.
    """
    if not tile_data:
        return None

    # Compact grid coordinates: remove gaps so tiles are always adjacent.
    # e.g. tiles at (0,0) and (2,0) become (0,0) and (1,0)
    x_vals = sorted(set(x for x, y in tile_data))
    y_vals = sorted(set(y for x, y in tile_data))
    x_remap = {old: new for new, old in enumerate(x_vals)}
    y_remap = {old: new for new, old in enumerate(y_vals)}

    compacted = {}
    for (gx, gy), data in tile_data.items():
        compacted[(x_remap[gx], y_remap[gy])] = data

    if log_fn:
        if len(x_remap) != max(x_remap.keys()) - min(x_remap.keys()) + 1 or \
           len(y_remap) != max(y_remap.keys()) - min(y_remap.keys()) + 1:
            log_fn("Compacted grid (removed gaps between tiles)")

    # Now work with compacted coordinates
    # Tiles can be different sizes - handle per-row heights and per-col widths
    grid_cols = len(x_vals)
    grid_rows = len(y_vals)

    # Get max dimensions per column and row for layout
    col_widths = {}
    row_heights = {}
    for (cx, cy), data in compacted.items():
        th, tw = data.shape
        col_widths[cx] = max(col_widths.get(cx, 0), tw)
        row_heights[cy] = max(row_heights.get(cy, 0), th)

    # Calculate pixel offsets for each column/row (with overlap)
    col_offsets = {}
    x_pos = 0
    for c in range(grid_cols):
        col_offsets[c] = x_pos
        x_pos += col_widths[c]
        if c < grid_cols - 1:
            x_pos -= margin  # overlap with next column

    row_offsets = {}
    y_pos = 0
    for r in range(grid_rows):
        row_offsets[r] = y_pos
        y_pos += row_heights[r]
        if r < grid_rows - 1:
            y_pos -= margin  # overlap with next row

    out_w = x_pos
    out_h = y_pos

    if log_fn:
        log_fn(f"Output size: {out_w}x{out_h} pixels")
        log_fn(f"Grid: {grid_cols}x{grid_rows} tiles, overlap {margin}px")

    # --- Deep height equalization for high/ultra ---
    _opts = opts or {"blur": False, "height_preserve": True, "feature_spill": True, "terrain_extend": True}
    if quality in ("high", "ultra") and _opts.get("terrain_extend", True):
        if log_fn:
            log_fn("Applying deep terrain equalization...")
        for (cx, cy) in sorted(compacted.keys()):
            if (cx + 1, cy) in compacted:
                a_f = compacted[(cx, cy)].astype(np.float64)
                b_f = compacted[(cx + 1, cy)].astype(np.float64)
                _deep_height_equalize_h(a_f, b_f, margin)
                compacted[(cx, cy)] = np.clip(a_f, 0, 65535).astype(np.uint16)
                compacted[(cx + 1, cy)] = np.clip(b_f, 0, 65535).astype(np.uint16)
            if (cx, cy + 1) in compacted:
                a_f = compacted[(cx, cy)].astype(np.float64)
                b_f = compacted[(cx, cy + 1)].astype(np.float64)
                _deep_height_equalize_v(a_f, b_f, margin)
                compacted[(cx, cy)] = np.clip(a_f, 0, 65535).astype(np.uint16)
                compacted[(cx, cy + 1)] = np.clip(b_f, 0, 65535).astype(np.uint16)
        if log_fn:
            log_fn("Deep equalization complete.")

    # --- Height matching pass (additional flat offset correction) ---
    height_corrections = {c: 0.0 for c in compacted}

    # Use the first tile (top-left) as reference, correct others relative to it
    processed = set()
    queue = [(0, 0)]
    processed.add((0, 0))

    while queue:
        cx, cy = queue.pop(0)
        if (cx, cy) not in compacted:
            continue
        cur_data = compacted[(cx, cy)].astype(np.float64)

        # Check right neighbor
        if (cx + 1, cy) in compacted and (cx + 1, cy) not in processed:
            nb_data = compacted[(cx + 1, cy)].astype(np.float64)
            cur_edge = cur_data[:, -1].mean() + height_corrections[(cx, cy)]
            nb_edge = nb_data[:, 0].mean() + height_corrections[(cx + 1, cy)]
            height_corrections[(cx + 1, cy)] += cur_edge - nb_edge
            processed.add((cx + 1, cy))
            queue.append((cx + 1, cy))

        # Check bottom neighbor
        if (cx, cy + 1) in compacted and (cx, cy + 1) not in processed:
            nb_data = compacted[(cx, cy + 1)].astype(np.float64)
            cur_edge = cur_data[-1, :].mean() + height_corrections[(cx, cy)]
            nb_edge = nb_data[0, :].mean() + height_corrections[(cx, cy + 1)]
            height_corrections[(cx, cy + 1)] += cur_edge - nb_edge
            processed.add((cx, cy + 1))
            queue.append((cx, cy + 1))

        # Check left neighbor
        if (cx - 1, cy) in compacted and (cx - 1, cy) not in processed:
            nb_data = compacted[(cx - 1, cy)].astype(np.float64)
            cur_edge = cur_data[:, 0].mean() + height_corrections[(cx, cy)]
            nb_edge = nb_data[:, -1].mean() + height_corrections[(cx - 1, cy)]
            height_corrections[(cx - 1, cy)] += cur_edge - nb_edge
            processed.add((cx - 1, cy))
            queue.append((cx - 1, cy))

        # Check top neighbor
        if (cx, cy - 1) in compacted and (cx, cy - 1) not in processed:
            nb_data = compacted[(cx, cy - 1)].astype(np.float64)
            cur_edge = cur_data[0, :].mean() + height_corrections[(cx, cy)]
            nb_edge = nb_data[-1, :].mean() + height_corrections[(cx, cy - 1)]
            height_corrections[(cx, cy - 1)] += cur_edge - nb_edge
            processed.add((cx, cy - 1))
            queue.append((cx, cy - 1))

    if log_fn:
        for c, corr in height_corrections.items():
            if abs(corr) > 1.0:
                log_fn(f"Height correction tile {c}: {corr:+.0f}")

    if log_fn:
        log_fn(f"Blend quality: {quality}")

    # --- Ultra mode: blend tile pairs directly, then paste onto canvas ---
    if quality == "ultra":
        if log_fn:
            log_fn("Ultra: Blending tile pairs with graph cut + Poisson + erosion...")

        # Apply height corrections to tile copies
        work = {}
        for (cx, cy), data in compacted.items():
            work[(cx, cy)] = np.clip(
                data.astype(np.float64) + height_corrections[(cx, cy)], 0, 65535
            ).astype(np.uint16)

        # Blend all horizontal neighbor pairs
        for (cx, cy) in sorted(work.keys()):
            if (cx + 1, cy) in work:
                if log_fn:
                    log_fn(f"  Ultra H-blend: ({cx},{cy}) <-> ({cx+1},{cy})")
                blend_horizontal(work[(cx, cy)], work[(cx + 1, cy)], margin, "ultra", opts)

        # Blend all vertical neighbor pairs
        for (cx, cy) in sorted(work.keys()):
            if (cx, cy + 1) in work:
                if log_fn:
                    log_fn(f"  Ultra V-blend: ({cx},{cy}) <-> ({cx},{cy+1})")
                blend_vertical(work[(cx, cy)], work[(cx, cy + 1)], margin, "ultra", opts)

        # Paste blended tiles onto canvas
        # Since blend functions already handled the seam, use simple averaging
        # in overlap zones (both tiles have matching data there now)
        canvas = np.zeros((out_h, out_w), dtype=np.float64)
        weight = np.zeros((out_h, out_w), dtype=np.float64)

        for (cx, cy), data in work.items():
            th, tw = data.shape
            rx = col_offsets[cx]
            ry = row_offsets[cy]

            # Equal weight everywhere - the blend functions already matched
            # the overlap pixels, so averaging preserves their work
            canvas[ry:ry + th, rx:rx + tw] += data.astype(np.float64)
            weight[ry:ry + th, rx:rx + tw] += 1.0

        weight[weight == 0] = 1.0
        result = canvas / weight

        # Smooth seam intersection corners (only if blur enabled)
        if not _opts.get("blur", False):
            result = np.clip(result, 0, 65535).astype(np.uint16)
            return result

        corner_blur_r = margin
        blur_sigma = max(12, margin // 8)

        if log_fn:
            log_fn("Smoothing seam intersections...")

        for (cx, cy) in work:
            # Check if this tile has both a right and bottom neighbor
            # = there's a corner at the bottom-right of this tile
            if (cx + 1, cy) in work and (cx, cy + 1) in work:
                # Corner position in output space
                corner_x = col_offsets[cx] + col_widths[cx] - margin // 2
                corner_y = row_offsets[cy] + row_heights[cy] - margin // 2

                # Extract region around the corner
                x1 = max(0, corner_x - corner_blur_r)
                x2 = min(out_w, corner_x + corner_blur_r)
                y1 = max(0, corner_y - corner_blur_r)
                y2 = min(out_h, corner_y + corner_blur_r)

                if x2 > x1 and y2 > y1:
                    region = result[y1:y2, x1:x2].astype(np.float64)
                    blurred = _gaussian_blur_2d(region, blur_sigma)

                    # Blend the blurred version with original using a circular mask
                    rh, rw = region.shape
                    cy_local = (y2 - y1) // 2
                    cx_local = (x2 - x1) // 2
                    yy, xx = np.ogrid[:rh, :rw]
                    dist = np.sqrt((xx - cx_local) ** 2 + (yy - cy_local) ** 2)
                    blend_mask = np.clip(1.0 - dist / corner_blur_r, 0, 1)
                    # Smoothstep the mask
                    blend_mask = 3 * blend_mask**2 - 2 * blend_mask**3

                    result[y1:y2, x1:x2] = (
                        blurred * blend_mask + region * (1.0 - blend_mask)
                    ).astype(np.uint16)

        result = np.clip(result, 0, 65535).astype(np.uint16)
        return result

    # --- Compositing pass (fast/high) ---
    canvas = np.zeros((out_h, out_w), dtype=np.float64)
    weight = np.zeros((out_h, out_w), dtype=np.float64)

    for (cx, cy), data in compacted.items():
        th, tw = data.shape
        rx = col_offsets[cx]
        ry = row_offsets[cy]

        # Apply height correction to this tile
        corrected = data.astype(np.float64) + height_corrections[(cx, cy)]

        # Build per-tile weight mask
        w_mask = np.ones((th, tw), dtype=np.float64)

        # Fade left edge
        if (cx - 1, cy) in compacted and margin > 0:
            m = min(margin, tw)
            fade = _smooth_weights(m)[::-1].reshape(1, m)
            w_mask[:, :m] *= fade

        # Fade right edge
        if (cx + 1, cy) in compacted and margin > 0:
            m = min(margin, tw)
            fade = _smooth_weights(m).reshape(1, m)
            w_mask[:, -m:] *= fade

        # Fade top edge
        if (cx, cy - 1) in compacted and margin > 0:
            m = min(margin, th)
            fade = _smooth_weights(m)[::-1].reshape(m, 1)
            w_mask[:m, :] *= fade

        # Fade bottom edge
        if (cx, cy + 1) in compacted and margin > 0:
            m = min(margin, th)
            fade = _smooth_weights(m).reshape(m, 1)
            w_mask[-m:, :] *= fade

        canvas[ry:ry + th, rx:rx + tw] += corrected * w_mask
        weight[ry:ry + th, rx:rx + tw] += w_mask

    # Normalize by weight
    weight[weight == 0] = 1.0
    result = canvas / weight

    # Post-blur the blend zones (only if blur option is on)
    if _opts.get("blur", False) and quality == "high" and len(compacted) > 1:
        blur_sigma = max(4, margin // 32)
        for (cx, cy) in compacted:
            if (cx + 1, cy) in compacted:
                rx = col_offsets[cx]
                tw = col_widths[cx]
                ry_start = row_offsets[cy]
                th = row_heights[cy]
                x_start = rx + tw - margin
                x_end = rx + tw
                if x_end <= out_w and x_start >= 0:
                    region = result[ry_start:ry_start + th, x_start:x_end].astype(np.float64)
                    blurred = _gaussian_blur_2d(region, blur_sigma)
                    result[ry_start:ry_start + th, x_start:x_end] = np.clip(blurred, 0, 65535).astype(np.uint16)
            if (cx, cy + 1) in compacted:
                rx_start = col_offsets[cx]
                tw = col_widths[cx]
                ry = row_offsets[cy]
                th = row_heights[cy]
                y_start = ry + th - margin
                y_end = ry + th
                if y_end <= out_h and y_start >= 0:
                    region = result[y_start:y_end, rx_start:rx_start + tw].astype(np.float64)
                    blurred = _gaussian_blur_2d(region, blur_sigma)
                    result[y_start:y_end, rx_start:rx_start + tw] = np.clip(blurred, 0, 65535).astype(np.uint16)
    result = np.clip(result, 0, 65535).astype(np.uint16)

    return result


def make_thumbnail(data, thumb_size=128):
    """Convert a uint16 heightmap array to an 8-bit PIL Image thumbnail."""
    normalized = (data.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
    img = Image.fromarray(normalized)
    img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# UI Constants - Trilithium / industrial metallic theme
# ---------------------------------------------------------------------------

GRID_MAX = 8
CELL_SIZE = 130
THUMB_SIZE = 120
PAD = 5

# Trilithium palette - brushed metal + dark LCD insets
BG_COLOR = "#8a8a88"         # Main background - brushed metal gray
BG_DARK = "#7a7a78"          # Title bar / darker frames
BG_LIGHT = "#989896"         # Lighter raised areas
PANEL_BG = "#3a3e3a"         # Dark LCD inset panels
PANEL_BORDER = "#606060"     # Inset panel border
CELL_COLOR = "#2e322e"       # Empty grid cell - dark inset
CELL_HOVER = "#3a3e3a"
CELL_FILLED = "#343834"      # Filled cell
CELL_SELECTED = "#4a5a4a"    # Selected - lighter
TEXT_COLOR = "#1a1a1a"        # Dark text on metal
TEXT_DIM = "#484848"          # Dimmed text
TEXT_BRIGHT = "#000000"       # Bold black
ACCENT = "#7a7a78"
SUCCESS = "#8a8a88"
WARNING = "#9a9a60"
BORDER_BLEND = "#6a6a68"
MERGE_COLOR = "#7a7a88"

# Trilithium metallic buttons
BTN_FACE = "#949494"          # Raised button face
BTN_LIGHT = "#b8b8b8"         # Highlight edge
BTN_SHADOW = "#5a5a5a"        # Shadow edge
BTN_TEXT = "#1a1a1a"           # Dark text

# Fonts - smaller, tighter like Winamp
FONT_TITLE = ("Terminal", 11, "bold")
FONT_LABEL = ("Terminal", 9, "bold")
FONT_BODY = ("Terminal", 8)
FONT_MONO = ("Consolas", 8)
FONT_SMALL = ("Terminal", 8)
FONT_BTN = ("Terminal", 9, "bold")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class StitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heightmap Tile Stitcher")
        self.root.configure(bg=BG_COLOR, bd=2, relief=tk.RAISED)
        self.root.minsize(400, 300)
        self.root.overrideredirect(True)  # Remove Windows title bar

        self.root.update_idletasks()
        self._screen_w = self.root.winfo_screenwidth()
        self._screen_h = self.root.winfo_screenheight()
        self._settings_w = 225
        self._chrome_h = 205
        self._chrome_w = 55

        # Track window dragging
        self._drag_x = 0
        self._drag_y = 0


        # State
        self.tile_paths = {}    # (x,y) -> file path
        self.tile_data = {}     # (x,y) -> numpy array
        self.tile_thumbs = {}   # (x,y) -> ImageTk.PhotoImage
        self.tile_format = tk.StringVar(value="png")
        self.r16_size = tk.IntVar(value=4033)
        self.margin = tk.IntVar(value=1024)
        self.blend_quality = tk.StringVar(value="high")
        self.opt_blur = tk.BooleanVar(value=False)
        self.opt_height_preserve = tk.BooleanVar(value=True)
        self.opt_feature_spill = tk.BooleanVar(value=True)
        self.opt_terrain_extend = tk.BooleanVar(value=True)
        self.grid_cols = tk.IntVar(value=4)
        self.grid_rows = tk.IntVar(value=4)
        self.selected_cell = None
        self.cell_widgets = {}

        self._build_ui()

    def _make_btn(self, parent, text, command, style="normal"):
        """Create a beveled metallic button - Trilithium style."""
        colors = {
            "normal":  (BTN_FACE, BTN_TEXT),
            "accent":  (BTN_FACE, BTN_TEXT),
            "action":  (BTN_FACE, BTN_TEXT),
            "danger":  (BTN_FACE, BTN_TEXT),
        }
        bg, fg = colors.get(style, colors["normal"])
        btn = tk.Button(parent, text=text, command=command,
                        bg=bg, fg=fg, font=FONT_BTN,
                        relief=tk.RAISED, bd=2, padx=10, pady=2,
                        activebackground=BTN_LIGHT, activeforeground="#000000",
                        disabledforeground="#6a6a6a",
                        highlightbackground=BTN_SHADOW,
                        cursor="hand2")
        return btn

    def _make_inset(self, parent, **kwargs):
        """Create a dark inset panel (LCD-style display area)."""
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief=tk.SUNKEN, **kwargs)
        return frame

    def _resize_for_grid(self):
        """Resize window to fit the current grid dimensions."""
        cols = self.grid_cols.get()
        rows = self.grid_rows.get()

        # Grid area: cells + labels + padding
        grid_w = cols * (CELL_SIZE + 4) + 22  # cells + gaps + y-label column
        grid_h = rows * (CELL_SIZE + 4) + 18  # cells + gaps + x-label row

        # Total window size
        win_w = grid_w + self._settings_w + self._chrome_w
        win_h = grid_h + self._chrome_h

        # Clamp to screen
        win_w = min(win_w, self._screen_w - 40)
        win_h = min(win_h, self._screen_h - 80)

        # Center on screen
        x = (self._screen_w - win_w) // 2
        y = (self._screen_h - win_h) // 2

        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    def _start_drag(self, event):
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _do_drag(self, event):
        self.root.geometry(f"+{event.x_root - self._drag_x}+{event.y_root - self._drag_y}")

    def _minimize(self):
        self.root.withdraw()

    def _close(self):
        # Kill everything - root, toplevel, and the process
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            self.root.master.destroy()
        except Exception:
            pass
        sys.exit(0)

    def _build_ui(self):
        # --- Custom Winamp-style title bar ---
        title_bar = tk.Frame(self.root, bg=BG_DARK, bd=0, height=22)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)

        # Left grip texture
        grip_left = tk.Canvas(title_bar, bg=BG_DARK, width=60, height=22,
                              highlightthickness=0, bd=0)
        grip_left.pack(side=tk.LEFT, fill=tk.Y)
        for gy in range(4, 20, 3):
            for gx in range(4, 56, 4):
                grip_left.create_rectangle(gx, gy, gx+1, gy+1, fill=BTN_LIGHT, outline="")
                grip_left.create_rectangle(gx+1, gy+1, gx+2, gy+2, fill=BTN_SHADOW, outline="")

        # Title text - centered
        title_label = tk.Label(title_bar, text="HEIGHTMAP TILE STITCHER",
                               font=("Terminal", 9, "bold"), bg=BG_DARK, fg=TEXT_COLOR)
        title_label.pack(side=tk.LEFT, padx=2, expand=True)

        # Window buttons - tiny, tight, Winamp style
        btn_close = tk.Button(title_bar, text="X", command=self._close,
                              bg=BTN_FACE, fg="#000000", font=("Terminal", 9, "bold"),
                              relief=tk.RAISED, bd=1, width=2, padx=2, pady=0,
                              activebackground="#cc4444", cursor="hand2")
        btn_close.pack(side=tk.RIGHT, padx=(0, 3), pady=1)

        btn_min = tk.Button(title_bar, text="_", command=self._minimize,
                            bg=BTN_FACE, fg="#000000", font=("Terminal", 9, "bold"),
                            relief=tk.RAISED, bd=1, width=2, padx=2, pady=0,
                            activebackground=BTN_LIGHT, cursor="hand2")
        btn_min.pack(side=tk.RIGHT, padx=(0, 1), pady=1)

        # Right grip texture
        grip_right = tk.Canvas(title_bar, bg=BG_DARK, width=60, height=22,
                               highlightthickness=0, bd=0)
        grip_right.pack(side=tk.RIGHT, fill=tk.Y)
        for gy in range(4, 20, 3):
            for gx in range(4, 56, 4):
                grip_right.create_rectangle(gx, gy, gx+1, gy+1, fill=BTN_LIGHT, outline="")
                grip_right.create_rectangle(gx+1, gy+1, gx+2, gy+2, fill=BTN_SHADOW, outline="")

        # Make title bar draggable
        for widget in [title_bar, title_label, grip_left, grip_right]:
            widget.bind("<Button-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._do_drag)

        # --- Top toolbar - integrated into bg ---
        toolbar = tk.Frame(self.root, bg=BG_COLOR, bd=0)
        toolbar.pack(fill=tk.X, padx=6, pady=(4, 0))

        toolbar_inner = tk.Frame(toolbar, bg=BG_COLOR, pady=2, padx=2)
        toolbar_inner.pack(fill=tk.X)

        self._make_btn(toolbar_inner, "LOAD FOLDER", self._load_folder,
                       "accent").pack(side=tk.LEFT, padx=2)
        self._make_btn(toolbar_inner, "ADD TILE", self._add_single_tile
                       ).pack(side=tk.LEFT, padx=2)
        self._make_btn(toolbar_inner, "REMOVE", self._remove_selected
                       ).pack(side=tk.LEFT, padx=2)
        self._make_btn(toolbar_inner, "CLEAR", self._clear_grid,
                       "danger").pack(side=tk.LEFT, padx=2)

        # Separator groove - embossed line like Winamp
        sep = tk.Frame(toolbar_inner, width=2, bd=1, relief=tk.GROOVE, bg=BG_COLOR)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        # Export buttons
        self._make_btn(toolbar_inner, "EXPORT", self._paint_export
                       ).pack(side=tk.RIGHT, padx=2)

        self.btn_stitch = self._make_btn(toolbar_inner, "STITCH TILES",
                                         self._stitch, "accent")
        self.btn_stitch.pack(side=tk.RIGHT, padx=2)
        self.btn_stitch.configure(state=tk.DISABLED)

        self.btn_merge = self._make_btn(toolbar_inner, "MERGE TO FILE",
                                        self._merge, "action")
        self.btn_merge.pack(side=tk.RIGHT, padx=2)
        self.btn_merge.configure(state=tk.DISABLED)

        # --- Status bar - pack BEFORE content so it's at bottom ---
        status_bar = tk.Frame(self.root, bg=BG_COLOR, bd=0)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=(4, 6))

        # Progress bar canvas
        self.progress_canvas = tk.Canvas(status_bar, height=8, bg=PANEL_BG,
                                         highlightthickness=0, bd=1, relief=tk.SUNKEN)
        self.progress_canvas.pack(fill=tk.X)

        # Generate 75-stop smooth gradient: red -> orange -> yellow -> green -> cyan -> blue
        self._progress_colors = []
        for i in range(75):
            t = i / 74.0
            if t < 0.2:       # Red -> Orange
                s = t / 0.2
                r, g, b = 220, int(40 + 140 * s), 30
            elif t < 0.35:    # Orange -> Yellow
                s = (t - 0.2) / 0.15
                r, g, b = 220, int(180 + 60 * s), 30
            elif t < 0.5:     # Yellow -> Green
                s = (t - 0.35) / 0.15
                r, g, b = int(220 - 180 * s), int(220 - 20 * s), int(30 + 10 * s)
            elif t < 0.65:    # Green -> Teal
                s = (t - 0.5) / 0.15
                r, g, b = int(40 - 10 * s), int(200 - 10 * s), int(40 + 140 * s)
            elif t < 0.8:     # Teal -> Cyan
                s = (t - 0.65) / 0.15
                r, g, b = int(30 + 10 * s), int(190 + 30 * s), int(180 + 40 * s)
            else:             # Cyan -> Blue-white
                s = (t - 0.8) / 0.2
                r, g, b = int(40 + 80 * s), int(220 - 20 * s), int(220 + 20 * s)
            self._progress_colors.append(f"#{r:02x}{g:02x}{b:02x}")

        self._progress_value = 0.0
        self._progress_pulse_phase = 0.0
        self._progress_animating = False

        # Status text overlaid
        self.status_label = tk.Label(status_bar, text="Ready", font=FONT_BODY,
                                     bg=BG_COLOR, fg="#b0b0a8", anchor=tk.W, padx=2)
        self.status_label.pack(fill=tk.X)

        # --- Main content - PanedWindow for draggable divider ---
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                               bg=BG_COLOR, bd=0, sashwidth=6,
                               sashrelief=tk.RAISED, sashpad=0,
                               opaqueresize=True)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 0))

        # Left: container with tab buttons + switchable views
        left_panel = tk.Frame(paned, bg=BG_COLOR, bd=0)

        # Tab buttons
        tab_bar = tk.Frame(left_panel, bg=BG_COLOR)
        tab_bar.pack(fill=tk.X)

        self._active_tab = tk.StringVar(value="tiles")

        self.tab_tiles_btn = tk.Button(tab_bar, text="TILES", font=FONT_BTN,
                                       bg=PANEL_BG, fg="#b0b0a8", bd=1, relief=tk.SUNKEN,
                                       padx=12, pady=1, cursor="hand2",
                                       command=lambda: self._switch_tab("tiles"))
        self.tab_tiles_btn.pack(side=tk.LEFT, padx=(0, 1))

        self.tab_result_btn = tk.Button(tab_bar, text="RESULT", font=FONT_BTN,
                                        bg=BTN_FACE, fg=BTN_TEXT, bd=1, relief=tk.RAISED,
                                        padx=12, pady=1, cursor="hand2",
                                        command=lambda: self._switch_tab("result"))
        self.tab_result_btn.pack(side=tk.LEFT, padx=(0, 1))

        self.tab_paint_btn = tk.Button(tab_bar, text="PAINT", font=FONT_BTN,
                                       bg=BTN_FACE, fg=BTN_TEXT, bd=1, relief=tk.RAISED,
                                       padx=12, pady=1, cursor="hand2",
                                       command=lambda: self._switch_tab("paint"))
        self.tab_paint_btn.pack(side=tk.LEFT)

        # Tile grid view
        self.grid_outer = tk.Frame(left_panel, bg=PANEL_BG, bd=1, relief=tk.GROOVE)
        self.grid_outer.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.grid_outer, bg=PANEL_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.grid_inner = tk.Frame(self.canvas, bg=PANEL_BG)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.grid_inner, anchor=tk.NW)
        self.grid_inner.bind("<Configure>",
                             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Result preview view (hidden initially)
        self.result_outer = tk.Frame(left_panel, bg=PANEL_BG, bd=1, relief=tk.GROOVE)
        self.result_canvas = tk.Canvas(self.result_outer, bg=PANEL_BG, highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        self._result_photo = None  # Keep reference to prevent GC
        self._result_data = None   # Raw result for re-rendering on resize
        self._result_zoom = 1.0    # Zoom level (1.0 = fit to canvas)
        self._result_pan_x = 0.5   # Pan center X (0-1, 0.5 = centered)
        self._result_pan_y = 0.5   # Pan center Y
        self._result_dragging = False
        self._result_drag_start = (0, 0)

        # Bind zoom and pan
        self.result_canvas.bind("<Configure>", self._on_result_resize)
        self.result_canvas.bind("<MouseWheel>", self._on_result_zoom)
        self.result_canvas.bind("<Button-1>", self._on_result_drag_start)
        self.result_canvas.bind("<B1-Motion>", self._on_result_drag)
        self.result_canvas.bind("<Double-Button-1>", self._on_result_reset_zoom)

        # Paint view (hidden initially)
        self.paint_outer = tk.Frame(left_panel, bg=PANEL_BG, bd=1, relief=tk.GROOVE)
        self.paint_canvas = tk.Canvas(self.paint_outer, bg=PANEL_BG, highlightthickness=0,
                                      cursor="crosshair")
        self.paint_canvas.pack(fill=tk.BOTH, expand=True)
        self._paint_photo = None
        self._paint_data = None      # Working copy of result for painting
        self._paint_undo_stack = []  # List of numpy arrays for undo
        self._paint_zoom = 1.0
        self._paint_pan_x = 0.5
        self._paint_pan_y = 0.5
        self._painting = False

        # Paint settings
        self.paint_brush_size = tk.IntVar(value=200)
        self.paint_opacity = tk.DoubleVar(value=0.5)
        self.paint_color = tk.DoubleVar(value=0.5)  # 0=black, 1=white in 0-1
        self.paint_brush_type = tk.StringVar(value="soft")

        # Paint bindings
        self.paint_canvas.bind("<Button-1>", self._on_paint_start)
        self.paint_canvas.bind("<B1-Motion>", self._on_paint_stroke)
        self.paint_canvas.bind("<ButtonRelease-1>", self._on_paint_end)
        self.paint_canvas.bind("<MouseWheel>", self._on_paint_zoom)
        self.paint_canvas.bind("<Button-3>", self._on_paint_pan_start)
        self.paint_canvas.bind("<B3-Motion>", self._on_paint_pan)
        self.paint_canvas.bind("<Configure>", self._on_paint_resize)

        # Generate brush stamps
        self._brush_stamps = self._generate_brush_stamps()

        paned.add(left_panel, stretch="always", minsize=300)

        # Right: settings panel
        SETTINGS_W = CELL_SIZE * 2 + 12
        settings = tk.Frame(paned, bg=BG_COLOR, bd=0)

        paned.add(settings, stretch="never", minsize=160, width=225)

        # Settings header - Winamp section title style
        hdr = tk.Frame(settings, bg=BG_COLOR, bd=1, relief=tk.GROOVE)
        hdr.pack(fill=tk.X, pady=(0, 6))
        tk.Label(hdr, text="SETTINGS", font=FONT_LABEL,
                 bg=BG_COLOR, fg=TEXT_DIM).pack(pady=1)

        # Export Format
        self._setting_label(settings, "EXPORT FORMAT")
        fmt_inset = self._make_inset(settings)
        fmt_inset.pack(fill=tk.X, pady=(0, 6))
        fmt_inner = tk.Frame(fmt_inset, bg=PANEL_BG)
        fmt_inner.pack(fill=tk.X, padx=1, pady=1)
        tk.Radiobutton(fmt_inner, text="PNG 16-bit", variable=self.tile_format,
                       value="png", bg=BTN_FACE, fg=BTN_TEXT, selectcolor=BG_LIGHT,
                       activebackground=BG_LIGHT, activeforeground=BTN_TEXT,
                       font=FONT_BODY, indicatoron=0, padx=4, pady=1,
                       relief=tk.RAISED, bd=1).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        tk.Radiobutton(fmt_inner, text="R16 raw", variable=self.tile_format,
                       value="r16", bg=BTN_FACE, fg=BTN_TEXT, selectcolor=BG_LIGHT,
                       activebackground=BG_LIGHT, activeforeground=BTN_TEXT,
                       font=FONT_BODY, indicatoron=0, padx=4, pady=1,
                       relief=tk.RAISED, bd=1).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)

        # Blend margin
        self._setting_label(settings, "BLEND MARGIN")

        # Margin presets - radiobuttons so selected stays pressed
        preset_inset = self._make_inset(settings)
        preset_inset.pack(fill=tk.X, pady=(0, 6))
        preset_inner = tk.Frame(preset_inset, bg=PANEL_BG)
        preset_inner.pack(fill=tk.X, padx=1, pady=1)
        for label, val in [("256", 256), ("512", 512), ("1024", 1024), ("2048", 2048), ("4096", 4096)]:
            tk.Radiobutton(preset_inner, text=label, variable=self.margin,
                           value=val, bg=BTN_FACE, fg=BTN_TEXT, selectcolor=BG_LIGHT,
                           activebackground=BG_LIGHT, activeforeground=BTN_TEXT,
                           font=FONT_SMALL, indicatoron=0, padx=2, pady=0,
                           relief=tk.RAISED, bd=1, cursor="hand2"
                           ).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)

        # Blend quality
        self._setting_label(settings, "BLEND QUALITY")
        qual_inset = self._make_inset(settings)
        qual_inset.pack(fill=tk.X, pady=(0, 6))
        qual_inner = tk.Frame(qual_inset, bg=PANEL_BG)
        qual_inner.pack(fill=tk.X, padx=1, pady=1)
        for qlabel, qval in [("Fast", "fast"), ("High", "high"), ("Ultra", "ultra")]:
            tk.Radiobutton(qual_inner, text=qlabel, variable=self.blend_quality,
                           value=qval, bg=BTN_FACE, fg=BTN_TEXT, selectcolor=BG_LIGHT,
                           activebackground=BG_LIGHT, activeforeground=BTN_TEXT,
                           font=FONT_BODY, indicatoron=0, padx=4, pady=1,
                           relief=tk.RAISED, bd=1).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)

        # Processing toggles
        self._setting_label(settings, "OPTIONS")
        opt_inset = self._make_inset(settings)
        opt_inset.pack(fill=tk.X, pady=(0, 6))
        opt_inner = tk.Frame(opt_inset, bg=PANEL_BG, padx=4, pady=2)
        opt_inner.pack(fill=tk.X)
        for label, var in [("Blur", self.opt_blur),
                           ("Height Preserve", self.opt_height_preserve),
                           ("Feature Spill", self.opt_feature_spill),
                           ("Terrain Extend", self.opt_terrain_extend)]:
            tk.Checkbutton(opt_inner, text=label, variable=var,
                           bg=PANEL_BG, fg="#b0b0a8", selectcolor=PANEL_BG,
                           activebackground=PANEL_BG, activeforeground="#b0b0a8",
                           font=FONT_SMALL, anchor=tk.W,
                           ).pack(fill=tk.X)

        # Grid size
        self._setting_label(settings, "GRID SIZE")
        grid_inset = self._make_inset(settings)
        grid_inset.pack(fill=tk.X, pady=(0, 6))
        grid_inner = tk.Frame(grid_inset, bg=PANEL_BG)
        grid_inner.pack(fill=tk.X, padx=4, pady=3)
        tk.Label(grid_inner, text="COLS", bg=PANEL_BG, fg="#b0b0a8",
                 font=FONT_BODY).pack(side=tk.LEFT)
        tk.Spinbox(grid_inner, from_=1, to=GRID_MAX, width=3,
                   textvariable=self.grid_cols, command=self._rebuild_grid,
                   font=FONT_MONO, bg=PANEL_BG, fg="#b0b0a8",
                   buttonbackground=BTN_FACE, insertbackground=TEXT_COLOR,
                   relief=tk.SUNKEN, bd=1).pack(side=tk.LEFT, padx=(4, 12))
        tk.Label(grid_inner, text="ROWS", bg=PANEL_BG, fg="#b0b0a8",
                 font=FONT_BODY).pack(side=tk.LEFT)
        tk.Spinbox(grid_inner, from_=1, to=GRID_MAX, width=3,
                   textvariable=self.grid_rows, command=self._rebuild_grid,
                   font=FONT_MONO, bg=PANEL_BG, fg="#b0b0a8",
                   buttonbackground=BTN_FACE, insertbackground=TEXT_COLOR,
                   relief=tk.SUNKEN, bd=1).pack(side=tk.LEFT, padx=4)

        # Save path
        self._setting_label(settings, "SAVE PATH")
        save_inset = self._make_inset(settings)
        save_inset.pack(fill=tk.X, pady=(0, 6))
        save_inner = tk.Frame(save_inset, bg=PANEL_BG)
        save_inner.pack(fill=tk.X, padx=1, pady=1)

        self.save_path = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Downloads"))
        self.save_path_entry = tk.Entry(save_inner, textvariable=self.save_path,
                                        bg=PANEL_BG, fg="#b0b0a8", font=FONT_SMALL,
                                        insertbackground="#b0b0a8", relief=tk.FLAT, bd=0)
        self.save_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2), pady=2)

        tk.Button(save_inner, text="...", command=self._browse_save_path,
                  bg=BTN_FACE, fg=BTN_TEXT, font=FONT_SMALL,
                  relief=tk.RAISED, bd=1, padx=4, pady=0, cursor="hand2"
                  ).pack(side=tk.RIGHT, padx=(0, 2), pady=2)


        # Log
        # Paint tools
        self._setting_label(settings, "PAINT TOOLS")
        paint_inset = self._make_inset(settings)
        paint_inset.pack(fill=tk.X, pady=(0, 6))
        paint_inner = tk.Frame(paint_inset, bg=PANEL_BG, padx=4, pady=2)
        paint_inner.pack(fill=tk.X)

        # Brush type
        tk.Label(paint_inner, text="BRUSH", bg=PANEL_BG, fg="#b0b0a8",
                 font=FONT_SMALL).pack(anchor=tk.W)
        brush_frame = tk.Frame(paint_inner, bg=PANEL_BG)
        brush_frame.pack(fill=tk.X, pady=(0, 4))
        for label, val in [("Soft", "soft"), ("Round", "round"), ("Noise1", "noise1"),
                           ("Noise2", "noise2"), ("Noise3", "noise3"), ("Tri", "triangle")]:
            tk.Radiobutton(brush_frame, text=label, variable=self.paint_brush_type,
                           value=val, bg=BTN_FACE, fg=BTN_TEXT, selectcolor=BG_LIGHT,
                           activebackground=BG_LIGHT, activeforeground=BTN_TEXT,
                           font=("Terminal", 7), indicatoron=0, padx=2, pady=0,
                           relief=tk.RAISED, bd=1).pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)

        # Brush size
        tk.Label(paint_inner, text="SIZE", bg=PANEL_BG, fg="#b0b0a8",
                 font=FONT_SMALL).pack(anchor=tk.W)
        self.paint_size_slider = tk.Scale(paint_inner, from_=50, to=2500, orient=tk.HORIZONTAL,
                                          variable=self.paint_brush_size, bg=PANEL_BG, fg="#b0b0a8",
                                          troughcolor="#282c28", highlightthickness=0,
                                          font=("Terminal", 7), sliderlength=10, width=10,
                                          showvalue=1)
        self.paint_size_slider.pack(fill=tk.X, pady=(0, 2))

        # Opacity
        tk.Label(paint_inner, text="OPACITY", bg=PANEL_BG, fg="#b0b0a8",
                 font=FONT_SMALL).pack(anchor=tk.W)
        self.paint_opacity_slider = tk.Scale(paint_inner, from_=0.01, to=1.0, resolution=0.01,
                                             orient=tk.HORIZONTAL, variable=self.paint_opacity,
                                             bg=PANEL_BG, fg="#b0b0a8", troughcolor="#282c28",
                                             highlightthickness=0, font=("Terminal", 7),
                                             sliderlength=10, width=10, showvalue=1)
        self.paint_opacity_slider.pack(fill=tk.X, pady=(0, 2))

        # Color (B&W gradient)
        tk.Label(paint_inner, text="HEIGHT VALUE", bg=PANEL_BG, fg="#b0b0a8",
                 font=FONT_SMALL).pack(anchor=tk.W)
        color_frame = tk.Frame(paint_inner, bg=PANEL_BG)
        color_frame.pack(fill=tk.X, pady=(0, 2))
        # Draw a gradient bar for color picking
        self.color_canvas = tk.Canvas(color_frame, height=16, bg=PANEL_BG,
                                       highlightthickness=0, bd=1, relief=tk.SUNKEN)
        self.color_canvas.pack(fill=tk.X)
        self.color_canvas.bind("<Button-1>", self._on_color_pick)
        self.color_canvas.bind("<B1-Motion>", self._on_color_pick)
        self.color_canvas.bind("<Configure>", self._draw_color_gradient)

        # Undo button
        undo_frame = tk.Frame(paint_inner, bg=PANEL_BG)
        undo_frame.pack(fill=tk.X, pady=(4, 0))
        tk.Button(undo_frame, text="UNDO", command=self._paint_undo,
                  bg=BTN_FACE, fg=BTN_TEXT, font=FONT_SMALL,
                  relief=tk.RAISED, bd=1, cursor="hand2").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(undo_frame, text="APPLY", command=self._paint_apply,
                  bg=BTN_FACE, fg=BTN_TEXT, font=FONT_SMALL,
                  relief=tk.RAISED, bd=1, cursor="hand2").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(1, 0))

        export_frame = tk.Frame(paint_inner, bg=PANEL_BG)
        export_frame.pack(fill=tk.X, pady=(2, 0))
        tk.Button(export_frame, text="EXPORT", command=self._paint_export,
                  bg=BTN_FACE, fg=BTN_TEXT, font=FONT_SMALL,
                  relief=tk.RAISED, bd=1, cursor="hand2").pack(fill=tk.X)

        # Bind Ctrl+Z globally
        self.root.bind("<Control-z>", lambda e: self._paint_undo())

        # Build initial grid and size window
        self._rebuild_grid()
        self._resize_for_grid()

        # Log panel - bottom of window, between content and status bar
        log_bar = tk.Frame(self.root, bg=BG_COLOR, bd=0)
        log_bar.pack(fill=tk.X, side=tk.BOTTOM, before=paned, padx=6, pady=(0, 2))

        log_header = tk.Frame(log_bar, bg=BG_COLOR)
        log_header.pack(fill=tk.X)
        tk.Label(log_header, text="LOG", font=FONT_SMALL, bg=BG_COLOR,
                 fg=TEXT_DIM).pack(side=tk.LEFT)

        log_inset = self._make_inset(log_bar)
        log_inset.pack(fill=tk.X)
        self.log_text = tk.Text(log_inset, height=3, bg=PANEL_BG, fg="#b0b0a8",
                                font=FONT_MONO, relief=tk.FLAT, padx=6, pady=2,
                                state=tk.DISABLED, wrap=tk.WORD,
                                insertbackground="#b0b0a8")
        self.log_text.pack(fill=tk.X)

    def _switch_tab(self, tab):
        self._active_tab.set(tab)
        # Hide all views
        self.grid_outer.pack_forget()
        self.result_outer.pack_forget()
        self.paint_outer.pack_forget()

        # Reset all tab buttons
        for btn in [self.tab_tiles_btn, self.tab_result_btn, self.tab_paint_btn]:
            btn.configure(bg=BTN_FACE, fg=BTN_TEXT, relief=tk.RAISED)

        if tab == "tiles":
            self.grid_outer.pack(fill=tk.BOTH, expand=True)
            self.tab_tiles_btn.configure(bg=PANEL_BG, fg="#b0b0a8", relief=tk.SUNKEN)
        elif tab == "result":
            self.result_outer.pack(fill=tk.BOTH, expand=True)
            self.tab_result_btn.configure(bg=PANEL_BG, fg="#b0b0a8", relief=tk.SUNKEN)
            self._render_result_preview()
        elif tab == "paint":
            self.paint_outer.pack(fill=tk.BOTH, expand=True)
            self.tab_paint_btn.configure(bg=PANEL_BG, fg="#b0b0a8", relief=tk.SUNKEN)
            # Copy result data to paint canvas if not already
            if self._paint_data is None and self._result_data is not None:
                self._paint_data = self._result_data.copy()
                self._paint_undo_stack = []
            self._render_paint_canvas()

    def _set_result(self, data):
        """Store the result data and enable the result tab."""
        self._result_data = data
        self._result_zoom = 1.0
        self._result_pan_x = 0.5
        self._result_pan_y = 0.5
        # Auto-switch to result tab
        self._switch_tab("result")

    def _render_result_preview(self):
        """Render the result data with zoom and pan support."""
        if self._result_data is None:
            self.result_canvas.delete("all")
            self.result_canvas.create_text(
                self.result_canvas.winfo_width() // 2,
                self.result_canvas.winfo_height() // 2,
                text="No result yet.\nRun Stitch or Merge first.",
                fill="#b0b0a8", font=FONT_LABEL, justify=tk.CENTER)
            return

        cw = self.result_canvas.winfo_width()
        ch = self.result_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        data = self._result_data
        dh, dw = data.shape

        # Base scale = fit to canvas
        base_scale = min(cw / dw, ch / dh)
        scale = base_scale * self._result_zoom

        # Size of the zoomed image
        img_w = max(1, int(dw * scale))
        img_h = max(1, int(dh * scale))

        # Pan offset - what portion of the image is visible
        # Pan values 0-1 map to the center of the viewport
        view_cx = self._result_pan_x * img_w
        view_cy = self._result_pan_y * img_h

        # Top-left corner of image relative to canvas
        x_off = int(cw / 2 - view_cx)
        y_off = int(ch / 2 - view_cy)

        # Crop the data to only render the visible portion (performance)
        # Convert canvas bounds to image pixel coords
        src_x1 = max(0, int(-x_off / scale))
        src_y1 = max(0, int(-y_off / scale))
        src_x2 = min(dw, int((cw - x_off) / scale) + 1)
        src_y2 = min(dh, int((ch - y_off) / scale) + 1)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return

        # Extract and render only the visible portion
        crop = data[src_y1:src_y2, src_x1:src_x2]
        normalized = (crop.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
        img = Image.fromarray(normalized)

        crop_w = max(1, int((src_x2 - src_x1) * scale))
        crop_h = max(1, int((src_y2 - src_y1) * scale))
        img = img.resize((crop_w, crop_h), Image.LANCZOS if scale < 4 else Image.NEAREST)

        self._result_photo = ImageTk.PhotoImage(img)
        self.result_canvas.delete("all")

        # Place the cropped image at the right offset
        place_x = x_off + int(src_x1 * scale)
        place_y = y_off + int(src_y1 * scale)
        self.result_canvas.create_image(place_x, place_y, anchor=tk.NW, image=self._result_photo)

        # Info overlay
        zoom_pct = int(self._result_zoom * 100)
        info = f"{dw}x{dh}  {zoom_pct}%"
        self.result_canvas.create_text(
            cw - 4, ch - 4, anchor=tk.SE,
            text=info, fill="#b0b0a8", font=FONT_SMALL)

        if self._result_zoom > 1.0:
            self.result_canvas.create_text(
                4, ch - 4, anchor=tk.SW,
                text="Scroll=Zoom  Drag=Pan  DblClick=Reset",
                fill="#606060", font=FONT_SMALL)

    def _on_result_zoom(self, event):
        """Scroll wheel zoom centered on mouse position."""
        if self._result_data is None:
            return

        # Zoom factor
        if event.delta > 0:
            factor = 1.3
        else:
            factor = 1 / 1.3

        old_zoom = self._result_zoom
        self._result_zoom = max(0.5, min(20.0, self._result_zoom * factor))

        # Adjust pan to zoom toward mouse position
        cw = self.result_canvas.winfo_width()
        ch = self.result_canvas.winfo_height()
        if cw > 0 and ch > 0:
            mx = event.x / cw  # Mouse position 0-1
            my = event.y / ch
            # Shift pan toward mouse
            zoom_ratio = self._result_zoom / old_zoom
            self._result_pan_x = mx + (self._result_pan_x - mx) * zoom_ratio
            self._result_pan_y = my + (self._result_pan_y - my) * zoom_ratio

        self._render_result_preview()

    def _on_result_drag_start(self, event):
        """Start panning."""
        self._result_dragging = True
        self._result_drag_start = (event.x, event.y)

    def _on_result_drag(self, event):
        """Pan the zoomed view."""
        if not self._result_dragging or self._result_data is None:
            return

        dx = event.x - self._result_drag_start[0]
        dy = event.y - self._result_drag_start[1]
        self._result_drag_start = (event.x, event.y)

        data = self._result_data
        dh, dw = data.shape
        cw = self.result_canvas.winfo_width()
        ch = self.result_canvas.winfo_height()

        base_scale = min(cw / dw, ch / dh)
        img_w = dw * base_scale * self._result_zoom
        img_h = dh * base_scale * self._result_zoom

        # Convert pixel drag to pan fraction
        if img_w > 0 and img_h > 0:
            self._result_pan_x -= dx / img_w
            self._result_pan_y -= dy / img_h
            self._result_pan_x = max(0.0, min(1.0, self._result_pan_x))
            self._result_pan_y = max(0.0, min(1.0, self._result_pan_y))

        self._render_result_preview()

    def _on_result_reset_zoom(self, event):
        """Double-click to reset zoom and pan."""
        self._result_zoom = 1.0
        self._result_pan_x = 0.5
        self._result_pan_y = 0.5
        self._render_result_preview()

    def _on_result_resize(self, event):
        """Re-render preview when canvas resizes."""
        if self._active_tab.get() == "result" and self._result_data is not None:
            self._render_result_preview()

    # --- Paint system ---

    def _generate_brush_stamps(self):
        """Pre-generate base brush shapes and noise textures separately."""
        stamps = {}
        size = 128  # Base shape size

        # Soft brush - Gaussian falloff
        y, x = np.ogrid[-size:size+1, -size:size+1]
        dist = np.sqrt(x*x + y*y) / size
        stamps["soft"] = np.clip(1.0 - dist, 0, 1) ** 2

        # Round brush - hard circle
        stamps["round"] = (dist <= 1.0).astype(np.float64)

        # Triangle brush
        tri = np.zeros((size*2+1, size*2+1), dtype=np.float64)
        for row in range(size*2+1):
            t = row / (size*2)
            half_w = int((1.0 - t) * size)
            if half_w > 0:
                tri[row, size-half_w:size+half_w+1] = 1.0 - t
        stamps["triangle"] = tri

        # Pre-generate large tileable noise textures with wide spread
        self._noise_textures = {}
        for i, seed in enumerate([101, 202, 303]):
            rng = np.random.RandomState(seed)
            tex_size = 2048  # Large texture for wide spread
            # Sparse noise - mostly zero with scattered particles
            noise = np.zeros((tex_size, tex_size), dtype=np.float64)
            # Different densities for each brush
            densities = [0.003, 0.006, 0.01]  # Very sparse particles
            n_particles = int(tex_size * tex_size * densities[i])
            px = rng.randint(0, tex_size, n_particles)
            py = rng.randint(0, tex_size, n_particles)
            vals = rng.rand(n_particles) * 0.5 + 0.5  # 0.5-1.0
            noise[py, px] = vals
            # Blur particles to create soft splotches
            blur_sizes = [12, 8, 5]  # Different splotch sizes
            noise = _gaussian_blur_2d(noise, blur_sizes[i])
            # Normalize
            nmax = noise.max()
            if nmax > 0:
                noise = noise / nmax
            self._noise_textures[f"noise{i+1}"] = noise

        return stamps

    def _get_brush_stamp(self, size):
        """Get a brush stamp at the requested size.
        For noise brushes: scale the soft mask shape but tile the fine noise texture."""
        brush_type = self.paint_brush_type.get()

        # Get base shape (soft falloff mask)
        if brush_type.startswith("noise"):
            base_shape = self._brush_stamps["soft"]
        else:
            base_shape = self._brush_stamps.get(brush_type, self._brush_stamps["soft"])

        # Resize shape to requested size
        shape_img = Image.fromarray(base_shape)
        shape_resized = np.array(shape_img.resize((size, size), Image.BILINEAR), dtype=np.float64)

        # For noise brushes, tile the fine noise texture over the brush area
        if brush_type.startswith("noise") and brush_type in self._noise_textures:
            noise_tex = self._noise_textures[brush_type]
            nh, nw = noise_tex.shape
            # Tile noise to cover brush size (noise stays at fixed pixel density)
            tiles_x = (size // nw) + 2
            tiles_y = (size // nh) + 2
            tiled = np.tile(noise_tex, (tiles_y, tiles_x))[:size, :size]
            # Multiply shape mask by noise
            shape_resized = shape_resized * tiled

        return shape_resized

    def _canvas_to_pixel(self, cx, cy):
        """Convert paint canvas coords to pixel coords in paint_data."""
        if self._paint_data is None:
            return None, None
        cw = self.paint_canvas.winfo_width()
        ch = self.paint_canvas.winfo_height()
        dh, dw = self._paint_data.shape

        base_scale = min(cw / dw, ch / dh)
        scale = base_scale * self._paint_zoom

        img_w = dw * scale
        img_h = dh * scale
        x_off = cw / 2 - self._paint_pan_x * img_w
        y_off = ch / 2 - self._paint_pan_y * img_h

        px = int((cx - x_off) / scale)
        py = int((cy - y_off) / scale)
        return px, py

    def _on_paint_start(self, event):
        if self._paint_data is None:
            return
        # Save undo state
        self._paint_undo_stack.append(self._paint_data.copy())
        if len(self._paint_undo_stack) > 30:
            self._paint_undo_stack.pop(0)
        self._painting = True
        self._last_paint_pos = (event.x, event.y)
        # Cache the brush stamp for this stroke
        self._cached_stamp = self._get_brush_stamp(self.paint_brush_size.get())
        self._apply_brush_at(event.x, event.y)
        self._paint_dirty = True
        self._schedule_paint_render()

    def _on_paint_stroke(self, event):
        if not self._painting or self._paint_data is None:
            return
        # Interpolate between last position and current for smooth strokes
        lx, ly = self._last_paint_pos
        dx, dy = event.x - lx, event.y - ly
        dist = max(1, int(np.sqrt(dx*dx + dy*dy)))
        # Step size = 1/4 of brush size in canvas pixels for smooth coverage
        cw = self.paint_canvas.winfo_width()
        ch = self.paint_canvas.winfo_height()
        dh, dw = self._paint_data.shape
        base_scale = min(cw / dw, ch / dh) * self._paint_zoom
        step_px = max(2, int(self.paint_brush_size.get() * base_scale * 0.25))
        steps = max(1, dist // step_px)

        for i in range(1, steps + 1):
            t = i / steps
            ix = int(lx + dx * t)
            iy = int(ly + dy * t)
            self._apply_brush_at(ix, iy)

        self._last_paint_pos = (event.x, event.y)
        self._paint_dirty = True
        self._schedule_paint_render()

    def _on_paint_end(self, event):
        self._painting = False
        self._cached_stamp = None
        self._render_paint_canvas()

    def _schedule_paint_render(self):
        """Throttle render to max ~15fps during painting."""
        if not hasattr(self, '_paint_render_pending') or not self._paint_render_pending:
            self._paint_render_pending = True
            self.root.after(66, self._do_paint_render)  # ~15fps

    def _do_paint_render(self):
        self._paint_render_pending = False
        if self._paint_dirty:
            self._paint_dirty = False
            self._render_paint_canvas()

    def _apply_brush_at(self, cx, cy):
        """Apply cached brush stamp at canvas position. Fast path."""
        px, py = self._canvas_to_pixel(cx, cy)
        if px is None:
            return

        dh, dw = self._paint_data.shape
        opacity = self.paint_opacity.get()
        color_val = self.paint_color.get() * 65535.0

        stamp = self._cached_stamp
        if stamp is None:
            return
        sh, sw = stamp.shape
        half_h, half_w = sh // 2, sw // 2

        y1 = max(0, py - half_h)
        y2 = min(dh, py - half_h + sh)
        x1 = max(0, px - half_w)
        x2 = min(dw, px - half_w + sw)

        if y2 <= y1 or x2 <= x1:
            return

        sy1 = y1 - (py - half_h)
        sy2 = sy1 + (y2 - y1)
        sx1 = x1 - (px - half_w)
        sx2 = sx1 + (x2 - x1)

        region = self._paint_data[y1:y2, x1:x2].astype(np.float64)
        brush_alpha = stamp[sy1:sy2, sx1:sx2] * opacity

        self._paint_data[y1:y2, x1:x2] = np.clip(
            region * (1.0 - brush_alpha) + color_val * brush_alpha,
            0, 65535
        ).astype(np.uint16)

    def _on_paint_zoom(self, event):
        if self._paint_data is None:
            return
        factor = 1.3 if event.delta > 0 else 1 / 1.3
        old_zoom = self._paint_zoom
        self._paint_zoom = max(0.5, min(20.0, self._paint_zoom * factor))

        cw = self.paint_canvas.winfo_width()
        ch = self.paint_canvas.winfo_height()
        if cw > 0 and ch > 0:
            mx, my = event.x / cw, event.y / ch
            zoom_ratio = self._paint_zoom / old_zoom
            self._paint_pan_x = mx + (self._paint_pan_x - mx) * zoom_ratio
            self._paint_pan_y = my + (self._paint_pan_y - my) * zoom_ratio
        self._render_paint_canvas()

    def _on_paint_pan_start(self, event):
        self._paint_drag_start = (event.x, event.y)

    def _on_paint_pan(self, event):
        if self._paint_data is None:
            return
        dx = event.x - self._paint_drag_start[0]
        dy = event.y - self._paint_drag_start[1]
        self._paint_drag_start = (event.x, event.y)

        dh, dw = self._paint_data.shape
        cw = self.paint_canvas.winfo_width()
        ch = self.paint_canvas.winfo_height()
        base_scale = min(cw / dw, ch / dh)
        img_w = dw * base_scale * self._paint_zoom
        img_h = dh * base_scale * self._paint_zoom

        if img_w > 0 and img_h > 0:
            self._paint_pan_x -= dx / img_w
            self._paint_pan_y -= dy / img_h
            self._paint_pan_x = max(0.0, min(1.0, self._paint_pan_x))
            self._paint_pan_y = max(0.0, min(1.0, self._paint_pan_y))
        self._render_paint_canvas()

    def _on_paint_resize(self, event):
        if self._active_tab.get() == "paint" and self._paint_data is not None:
            self._render_paint_canvas()

    def _render_paint_canvas(self):
        """Render paint data with zoom/pan (same logic as result preview)."""
        if self._paint_data is None:
            self.paint_canvas.delete("all")
            self.paint_canvas.create_text(
                self.paint_canvas.winfo_width() // 2,
                self.paint_canvas.winfo_height() // 2,
                text="No data. Run Merge first,\nthen switch to Paint tab.",
                fill="#b0b0a8", font=FONT_LABEL, justify=tk.CENTER)
            return

        cw = self.paint_canvas.winfo_width()
        ch = self.paint_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        data = self._paint_data
        dh, dw = data.shape
        base_scale = min(cw / dw, ch / dh)
        scale = base_scale * self._paint_zoom

        img_w = max(1, int(dw * scale))
        img_h = max(1, int(dh * scale))

        view_cx = self._paint_pan_x * img_w
        view_cy = self._paint_pan_y * img_h
        x_off = int(cw / 2 - view_cx)
        y_off = int(ch / 2 - view_cy)

        src_x1 = max(0, int(-x_off / scale))
        src_y1 = max(0, int(-y_off / scale))
        src_x2 = min(dw, int((cw - x_off) / scale) + 1)
        src_y2 = min(dh, int((ch - y_off) / scale) + 1)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return

        crop = data[src_y1:src_y2, src_x1:src_x2]
        normalized = (crop.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
        img = Image.fromarray(normalized)
        crop_w = max(1, int((src_x2 - src_x1) * scale))
        crop_h = max(1, int((src_y2 - src_y1) * scale))
        img = img.resize((crop_w, crop_h), Image.LANCZOS if scale < 4 else Image.NEAREST)

        self._paint_photo = ImageTk.PhotoImage(img)
        self.paint_canvas.delete("all")
        place_x = x_off + int(src_x1 * scale)
        place_y = y_off + int(src_y1 * scale)
        self.paint_canvas.create_image(place_x, place_y, anchor=tk.NW, image=self._paint_photo)

        zoom_pct = int(self._paint_zoom * 100)
        self.paint_canvas.create_text(cw - 4, ch - 4, anchor=tk.SE,
                                       text=f"{dw}x{dh}  {zoom_pct}%  Undos:{len(self._paint_undo_stack)}",
                                       fill="#b0b0a8", font=FONT_SMALL)

    def _on_color_pick(self, event):
        """Pick height value from the gradient bar."""
        w = self.color_canvas.winfo_width()
        if w > 0:
            self.paint_color.set(max(0.0, min(1.0, event.x / w)))
            self._draw_color_gradient(None)

    def _draw_color_gradient(self, event=None):
        """Draw the B&W gradient bar with indicator."""
        self.color_canvas.delete("all")
        w = self.color_canvas.winfo_width()
        h = self.color_canvas.winfo_height()
        if w < 2:
            return
        # Draw gradient
        for x in range(w):
            t = x / w
            v = int(t * 255)
            color = f"#{v:02x}{v:02x}{v:02x}"
            self.color_canvas.create_line(x, 0, x, h, fill=color)
        # Draw indicator
        ix = int(self.paint_color.get() * w)
        self.color_canvas.create_rectangle(ix - 2, 0, ix + 2, h, outline="#ff0000", width=1)

    def _paint_undo(self):
        """Undo last paint stroke."""
        if self._paint_undo_stack:
            self._paint_data = self._paint_undo_stack.pop()
            self._render_paint_canvas()

    def _paint_apply(self):
        """Apply painted edits back to the result data."""
        if self._paint_data is not None:
            self._result_data = self._paint_data.copy()
            self._log("Paint edits applied to result.")

    def _paint_export(self):
        """Export the painted heightmap directly."""
        if self._paint_data is None:
            self._log("Nothing to export - paint first.")
            return

        import random, string
        rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        dh, dw = self._paint_data.shape
        ext = "." + self.tile_format.get()
        save_dir = self.save_path.get()

        if save_dir and os.path.isdir(save_dir):
            out_path = os.path.join(save_dir, f"painted_{rand_id}_{dw}x{dh}{ext}")
        elif save_dir:
            out_path = os.path.join(os.path.dirname(save_dir), f"painted_{rand_id}_{dw}x{dh}{ext}")
        else:
            filetypes = [("PNG 16-bit", "*.png"), ("R16 raw", "*.r16")]
            out_path = filedialog.asksaveasfilename(
                title="Export painted heightmap",
                filetypes=filetypes, defaultextension=".png",
                initialfile=f"painted_{rand_id}_{dw}x{dh}")
            if not out_path:
                return

        if out_path.lower().endswith(".r16"):
            save_tile_r16(out_path, self._paint_data)
        else:
            save_tile_png(out_path, self._paint_data)

        file_mb = os.path.getsize(out_path) / 1024 / 1024
        self._log(f"Exported: {os.path.basename(out_path)} ({file_mb:.1f} MB)")
        # Also apply to result
        self._result_data = self._paint_data.copy()

    def _browse_save_path(self):
        filetypes = [("PNG 16-bit", "*.png"), ("R16 raw", "*.r16")]
        path = filedialog.asksaveasfilename(
            title="Set default save file",
            filetypes=filetypes,
            defaultextension=".png",
            initialfile="merged")
        if path:
            self.save_path.set(path)

    def _setting_label(self, parent, text):
        tk.Label(parent, text=text, font=FONT_LABEL,
                 bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor=tk.W, pady=(6, 2))

    def _set_progress(self, value, status_text=None):
        """Update the multicolor progress bar. value: 0.0 to 1.0"""
        self._progress_value = max(0.0, min(1.0, value))
        self._draw_progress_bar()

        if status_text:
            self.status_label.configure(text=status_text)

        # Start pulsing animation if in progress
        if 0.0 < self._progress_value < 1.0:
            if not self._progress_animating:
                self._progress_animating = True
                self._animate_progress()
        else:
            self._progress_animating = False

    def _draw_progress_bar(self):
        """Render the progress bar with pulse effect."""
        self.progress_canvas.delete("all")
        w = self.progress_canvas.winfo_width()
        h = self.progress_canvas.winfo_height()
        if w < 2:
            w = 400

        fill_w = int(w * self._progress_value)
        if fill_w <= 0:
            return

        colors = self._progress_colors
        seg_count = len(colors)
        seg_w = max(1, fill_w / seg_count)

        # Pulse: a bright highlight that sweeps across the bar
        pulse = self._progress_pulse_phase
        pulse_width = 0.15  # Width of the pulse glow

        for i, base_color in enumerate(colors):
            x1 = int(i * seg_w)
            x2 = int((i + 1) * seg_w)
            if x2 > fill_w:
                x2 = fill_w
            if x1 >= x2:
                continue

            # Parse base color
            r = int(base_color[1:3], 16)
            g = int(base_color[3:5], 16)
            b = int(base_color[5:7], 16)

            # Add pulse brightness
            seg_pos = i / seg_count
            dist = abs(seg_pos - pulse)
            if dist < pulse_width:
                glow = (1.0 - dist / pulse_width) * 0.4
                r = min(255, int(r + (255 - r) * glow))
                g = min(255, int(g + (255 - g) * glow))
                b = min(255, int(b + (255 - b) * glow))

            color = f"#{r:02x}{g:02x}{b:02x}"
            self.progress_canvas.create_rectangle(x1, 0, x2, h, fill=color, outline="")

    def _animate_progress(self):
        """Pulse animation loop."""
        if not self._progress_animating:
            return
        self._progress_pulse_phase += 0.03
        if self._progress_pulse_phase > 1.0:
            self._progress_pulse_phase = -0.15  # Wrap around with gap
        self._draw_progress_bar()
        self.root.after(50, self._animate_progress)

    def _reset_progress(self):
        """Clear the progress bar."""
        self._progress_value = 0.0
        self._progress_animating = False
        self.progress_canvas.delete("all")

    def _log(self, msg, tag=None):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_info(self, text):
        pass  # Tile info panel removed

    # --- Grid management ---

    def _rebuild_grid(self, *_):
        for widget in self.grid_inner.winfo_children():
            widget.destroy()
        self.cell_widgets.clear()

        cols = self.grid_cols.get()
        rows = self.grid_rows.get()

        # Lock column/row sizes so cells don't bounce around
        self.grid_inner.grid_columnconfigure(0, minsize=18, weight=0)
        for x in range(cols):
            self.grid_inner.grid_columnconfigure(x + 1, minsize=CELL_SIZE + 4, weight=0)
        self.grid_inner.grid_rowconfigure(0, minsize=14, weight=0)
        for y in range(rows):
            self.grid_inner.grid_rowconfigure(y + 1, minsize=CELL_SIZE + 4, weight=0)

        # Column headers
        tk.Label(self.grid_inner, text="", bg=PANEL_BG, width=2).grid(row=0, column=0, sticky="nsew")
        for x in range(cols):
            tk.Label(self.grid_inner, text=f"x{x}", bg=PANEL_BG, fg=TEXT_DIM,
                     font=FONT_SMALL).grid(row=0, column=x + 1, pady=(2, 1), sticky="nsew")

        for y in range(rows):
            # Row header
            tk.Label(self.grid_inner, text=f"y{y}", bg=PANEL_BG, fg=TEXT_DIM,
                     font=FONT_SMALL).grid(row=y + 1, column=0, padx=(2, 1), sticky="nsew")

            for x in range(cols):
                # Each cell is a sunken inset
                cell = tk.Frame(self.grid_inner, width=CELL_SIZE, height=CELL_SIZE,
                                bg=CELL_COLOR, bd=2, relief=tk.SUNKEN)
                cell.grid(row=y + 1, column=x + 1, padx=2, pady=2, sticky="nsew")
                cell.grid_propagate(False)

                coord = (x, y)
                is_selected = coord == self.selected_cell
                has_thumb = coord in self.tile_thumbs
                has_path = coord in self.tile_paths

                if has_thumb:
                    lbl = tk.Label(cell, image=self.tile_thumbs[coord], bg=CELL_COLOR, bd=0)
                    if is_selected:
                        cell.configure(relief=tk.GROOVE, bg=CELL_SELECTED)
                    else:
                        cell.configure(bg=CELL_FILLED)
                elif has_path:
                    lbl = tk.Label(cell, text="LOADING...", bg=CELL_FILLED, fg=TEXT_COLOR,
                                   font=FONT_SMALL)
                    cell.configure(bg=CELL_FILLED)
                    if is_selected:
                        cell.configure(relief=tk.GROOVE, bg=CELL_SELECTED)
                else:
                    lbl = tk.Label(cell, text=f"{x},{y}", bg=CELL_COLOR,
                                   fg="#2a2e2a", font=FONT_BODY, justify=tk.CENTER)
                    if is_selected:
                        cell.configure(relief=tk.GROOVE, bg=CELL_SELECTED)

                lbl.pack(expand=True, fill=tk.BOTH)

                # Click handlers
                lbl.bind("<Button-1>", lambda e, c=coord: self._cell_clicked(c))
                cell.bind("<Button-1>", lambda e, c=coord: self._cell_clicked(c))

                # Hover - raise border on hover
                for w in [cell, lbl]:
                    w.bind("<Enter>", lambda e, f=cell: f.configure(relief=tk.RAISED))
                    w.bind("<Leave>", lambda e, f=cell, c=coord: f.configure(
                        relief=tk.GROOVE if c == self.selected_cell else tk.SUNKEN))

                self.cell_widgets[coord] = cell

        # Blend edge indicators - highlight neighboring filled cells
        self._draw_edge_indicators()
        self._update_buttons()
        # Update status bar
        n = len(self.tile_data)
        if hasattr(self, 'status_label'):
            self.status_label.configure(
                text=f"{n} tile{'s' if n != 1 else ''} loaded  |  "
                     f"Grid: {cols}x{rows}  |  "
                     f"Margin: {self.margin.get()}px")

        # Auto-resize window when grid dimensions change
        grid_key = (cols, rows)
        if not hasattr(self, '_last_grid_key') or self._last_grid_key != grid_key:
            self._last_grid_key = grid_key
            self._resize_for_grid()

    def _draw_edge_indicators(self):
        """Highlight cells that have neighbors (will be blended)."""
        for (x, y) in self.tile_paths:
            if (x, y) not in self.cell_widgets:
                continue
            has_neighbor = ((x + 1, y) in self.tile_paths or (x - 1, y) in self.tile_paths or
                            (x, y + 1) in self.tile_paths or (x, y - 1) in self.tile_paths)
            if has_neighbor and (x, y) != self.selected_cell:
                self.cell_widgets[(x, y)].configure(bg=CELL_FILLED)

    def _cell_clicked(self, coord):
        x, y = coord
        self.selected_cell = coord

        if coord in self.tile_paths:
            # Show info about loaded tile
            path = self.tile_paths[coord]
            info = f"Position: x{x}, y{y}\n"
            info += f"File: {os.path.basename(path)}\n"
            if coord in self.tile_data:
                d = self.tile_data[coord]
                info += f"Size: {d.shape[1]}x{d.shape[0]}\n"
                info += f"Min: {d.min()}  Max: {d.max()}\n"
                info += f"Mean: {d.mean():.0f}\n"
                info += f"Memory: {d.nbytes / 1024 / 1024:.1f} MB"
            self._set_info(info)
            self._rebuild_grid()
        else:
            # Empty cell - open file picker and load directly into this cell
            self._rebuild_grid()
            filetypes = [("Heightmap files", "*.png *.r16"), ("PNG files", "*.png"),
                         ("R16 files", "*.r16"), ("All files", "*.*")]
            path = filedialog.askopenfilename(title=f"Select heightmap for ({x},{y})",
                                             filetypes=filetypes)
            if not path:
                self._set_info(f"Position: x{x}, y{y}\n\nEmpty")
                return

            self.tile_paths[coord] = path
            self._log(f"Added {os.path.basename(path)} at ({x},{y})")
            log.info(f"Cell click load: tile_paths[({x},{y})] = {path}")
            self._load_tile_data([coord])

    def _update_buttons(self):
        """Enable/disable export buttons based on loaded tiles."""
        has_neighbors = False
        for (x, y) in self.tile_paths:
            if (x + 1, y) in self.tile_paths or (x, y + 1) in self.tile_paths:
                has_neighbors = True
                break

        has_any = len(self.tile_data) >= 2
        self.btn_stitch.configure(state=tk.NORMAL if has_neighbors else tk.DISABLED)
        self.btn_merge.configure(state=tk.NORMAL if has_any else tk.DISABLED)

    # --- Loading tiles ---

    def _load_folder(self):
        log.info("_load_folder called")
        folder = filedialog.askdirectory(title="Select folder containing heightmap tiles")
        log.info(f"Folder selected: {folder}")
        if not folder:
            return

        # Search for all heightmap files (both png and r16)
        found = find_tiles(folder)  # None = accept both formats
        log.info(f"find_tiles(all formats): {found}")

        if not found:
            all_files = os.listdir(folder)
            heightmaps = [f for f in all_files if f.lower().endswith((".png", ".r16"))]
            if heightmaps:
                messagebox.showinfo("No Grid Names Found",
                                    f"Found {len(heightmaps)} heightmap files but none with "
                                    f"_xN_yN or _yN_xN naming.\n\n"
                                    f"Use 'Add Tile' to load them individually and "
                                    f"place them on the grid manually.")
            else:
                messagebox.showwarning("No Tiles Found",
                                       f"No .png or .r16 files found in:\n{folder}")
            return

        # Determine grid size needed
        max_x = max(x for x, y in found) + 1
        max_y = max(y for x, y in found) + 1
        self.grid_cols.set(min(max_x, GRID_MAX))
        self.grid_rows.set(min(max_y, GRID_MAX))

        self._clear_grid(silent=True)
        self.tile_paths.update(found)

        self._log(f"Found {len(found)} tiles in {os.path.basename(folder)}")

        self._load_tile_data(list(found.keys()))

    def _add_single_tile(self):
        """Add tile to selected cell or first empty cell."""
        log.info("_add_single_tile called")

        # Find target cell - selected empty cell, or first empty cell
        target = None
        if self.selected_cell and self.selected_cell not in self.tile_paths:
            target = self.selected_cell
        else:
            for try_y in range(self.grid_rows.get()):
                for try_x in range(self.grid_cols.get()):
                    if (try_x, try_y) not in self.tile_paths:
                        target = (try_x, try_y)
                        break
                if target:
                    break

        if not target:
            messagebox.showinfo("Grid Full", "All grid cells are filled.\nIncrease grid size or remove a tile.")
            return

        x, y = target
        filetypes = [("Heightmap files", "*.png *.r16"), ("PNG files", "*.png"),
                     ("R16 files", "*.r16"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title=f"Select heightmap for ({x},{y})",
                                          filetypes=filetypes)
        if not path:
            return

        self.tile_paths[(x, y)] = path
        log.info(f"tile_paths[({x},{y})] = {path}")
        self._log(f"Added {os.path.basename(path)} at ({x},{y})")
        self._load_tile_data([(x, y)])

    def _remove_selected(self):
        if self.selected_cell and self.selected_cell in self.tile_paths:
            coord = self.selected_cell
            name = os.path.basename(self.tile_paths[coord])
            del self.tile_paths[coord]
            self.tile_data.pop(coord, None)
            self.tile_thumbs.pop(coord, None)
            self.selected_cell = None
            self._log(f"Removed {name} from ({coord[0]},{coord[1]})")
            self._rebuild_grid()

    def _load_tile_data(self, coords):
        """Load tile data and generate thumbnails."""
        log.info(f"_load_tile_data called with coords={coords}")
        if not coords:
            log.info("No coords to load, returning")
            self._log("No coords to load.")
            return

        coords = list(coords)
        self._log(f"Loading {len(coords)} tiles...")
        self.root.update_idletasks()

        r16_size = self.r16_size.get()
        count = 0

        for coord in coords:
            if coord not in self.tile_paths:
                log.warning(f"SKIP {coord} - not in tile_paths")
                self._log(f"  SKIP {coord} - not in tile_paths")
                continue
            path = self.tile_paths[coord]
            name = os.path.basename(path)
            log.info(f"Loading tile {coord}: {path}")
            try:
                self._log(f"  Loading {name} at {coord}...")
                self.root.update_idletasks()

                if path.lower().endswith(".r16"):
                    log.info(f"  Format: R16, size={r16_size}")
                    data = load_tile_r16(path, r16_size)
                else:
                    log.info(f"  Format: PNG")
                    data = load_tile_png(path)

                self.tile_data[coord] = data
                log.info(f"  Loaded: shape={data.shape}, dtype={data.dtype}, min={data.min()}, max={data.max()}")
                self._log(f"  Data: {data.shape[1]}x{data.shape[0]}, {data.dtype}")

                # Create thumbnail
                log.info(f"  Creating thumbnail...")
                thumb_pil = make_thumbnail(data, THUMB_SIZE)
                log.info(f"  Thumbnail PIL: size={thumb_pil.size}, mode={thumb_pil.mode}")
                photo = ImageTk.PhotoImage(thumb_pil)
                log.info(f"  PhotoImage created: {photo.width()}x{photo.height()}")

                # Keep extra reference to prevent garbage collection
                if not hasattr(self, '_photo_refs'):
                    self._photo_refs = []
                self._photo_refs.append(photo)
                self.tile_thumbs[coord] = photo

                self._log(f"  OK: {name}")
                log.info(f"  OK: {name}")
                count += 1
            except Exception as e:
                log.error(f"  FAILED loading {name}: {e}")
                log.error(traceback.format_exc())
                self._log(f"  ERROR {name}: {e}")

        log.info(f"Load complete: {count}/{len(coords)}")
        log.info(f"tile_data keys: {list(self.tile_data.keys())}")
        log.info(f"tile_thumbs keys: {list(self.tile_thumbs.keys())}")
        self._log(f"Loaded {count}/{len(coords)} tiles.")
        self._rebuild_grid()

    def _clear_grid(self, silent=False):
        self.tile_paths.clear()
        self.tile_data.clear()
        self.tile_thumbs.clear()
        self.selected_cell = None
        if not silent:
            self._log("Grid cleared.")
        self._rebuild_grid()

    # --- Stitching (separate tiles) ---

    def _stitch(self):
        log.info(f"_stitch called. tile_data={list(self.tile_data.keys())}, tile_paths={list(self.tile_paths.keys())}")
        if not self.tile_data:
            log.warning("tile_data is empty!")
            messagebox.showwarning("No Data", "Load tiles first before stitching.")
            return

        not_loaded = [c for c in self.tile_paths if c not in self.tile_data]
        if not_loaded:
            messagebox.showwarning("Tiles Not Loaded",
                                   f"{len(not_loaded)} tiles haven't finished loading yet.")
            return

        # Use save path directory if set, otherwise prompt
        save_dir = self.save_path.get()
        if save_dir:
            out_dir = save_dir if os.path.isdir(save_dir) else os.path.dirname(save_dir)
            if not out_dir:
                out_dir = save_dir
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = filedialog.askdirectory(title="Select output folder for stitched tiles")
            if not out_dir:
                return
            self.save_path.set(out_dir)

        margin = self.margin.get()
        fmt = self.tile_format.get()

        first_tile = next(iter(self.tile_data.values()))
        tile_h, tile_w = first_tile.shape
        if margin >= min(tile_h, tile_w) // 2:
            messagebox.showerror("Margin Too Large",
                                 f"Blend margin ({margin}px) is too large for "
                                 f"tile size ({tile_w}x{tile_h}).\n\n"
                                 f"Maximum margin: {min(tile_h, tile_w) // 2 - 1}px")
            return

        export_fmt = self.tile_format.get()
        quality = self.blend_quality.get()
        opts = {
            "blur": self.opt_blur.get(),
            "height_preserve": self.opt_height_preserve.get(),
            "feature_spill": self.opt_feature_spill.get(),
            "terrain_extend": self.opt_terrain_extend.get(),
        }

        self.btn_stitch.configure(state=tk.DISABLED, text="Stitching...")
        self.btn_merge.configure(state=tk.DISABLED)

        def worker():
            try:
                work_data = {c: d.copy() for c, d in self.tile_data.items()}

                h_pairs = []
                v_pairs = []
                for (x, y) in work_data:
                    if (x + 1, y) in work_data:
                        h_pairs.append(((x, y), (x + 1, y)))
                    if (x, y + 1) in work_data:
                        v_pairs.append(((x, y), (x, y + 1)))

                total_edges = len(h_pairs) + len(v_pairs)
                total_steps = total_edges + len(work_data)  # edges + saves
                step = 0

                self.root.after(0, lambda: self._log(
                    f"\nStitching {total_edges} edges (margin={margin}px)..."))
                self.root.after(0, lambda: self._set_progress(0.0, "Blending edges..."))

                for a, b in h_pairs:
                    self.root.after(0, lambda a=a, b=b: self._log(
                        f"  H-blend: ({a[0]},{a[1]}) <-> ({b[0]},{b[1]})"))
                    blend_horizontal(work_data[a], work_data[b], margin, quality, opts)
                    step += 1
                    p = step / total_steps
                    self.root.after(0, lambda p=p: self._set_progress(p, f"Blending... {int(p*100)}%"))

                for a, b in v_pairs:
                    self.root.after(0, lambda a=a, b=b: self._log(
                        f"  V-blend: ({a[0]},{a[1]}) <-> ({b[0]},{b[1]})"))
                    blend_vertical(work_data[a], work_data[b], margin, quality, opts)
                    step += 1
                    p = step / total_steps
                    self.root.after(0, lambda p=p: self._set_progress(p, f"Blending... {int(p*100)}%"))

                self.root.after(0, lambda: self._log(f"\nSaving as .{export_fmt} to {out_dir}..."))
                self.root.after(0, lambda: self._set_progress(0.8, "Saving files..."))

                for coord, data in work_data.items():
                    orig_name = os.path.basename(self.tile_paths[coord])
                    base = os.path.splitext(orig_name)[0]
                    out_name = f"{base}.{export_fmt}"
                    out_path = os.path.join(out_dir, out_name)
                    if export_fmt == "r16":
                        save_tile_r16(out_path, data)
                    else:
                        save_tile_png(out_path, data)
                    self.root.after(0, lambda n=out_name: self._log(f"  Saved {n}"))
                    step += 1
                    p = step / total_steps
                    self.root.after(0, lambda p=p: self._set_progress(p, f"Saving... {int(p*100)}%"))

                for coord, data in work_data.items():
                    thumb_img = make_thumbnail(data, THUMB_SIZE)
                    self.tile_thumbs[coord] = ImageTk.PhotoImage(thumb_img)
                    self.tile_data[coord] = data

                # Build preview by stitching tiles into one image
                preview = merge_tiles_to_single(work_data, margin, quality="fast")

                self.root.after(0, lambda: self._set_progress(1.0, "Stitching complete!"))
                self.root.after(0, lambda: self._log(
                    f"\nStitching complete! {len(work_data)} tiles saved to {out_dir}"))
                if preview is not None:
                    self.root.after(0, lambda p=preview: self._set_result(p))

            except Exception as e:
                self.root.after(0, lambda: self._log(f"\nERROR: {e}"))
                self.root.after(0, lambda: messagebox.showerror("Stitch Failed", str(e)))

            finally:
                self.root.after(0, lambda: self.btn_stitch.configure(
                    state=tk.NORMAL, text="STITCH TILES"))
                self.root.after(0, lambda: self.btn_merge.configure(state=tk.NORMAL))
                self.root.after(0, self._rebuild_grid)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    # --- Merging (single output file) ---

    def _merge(self):
        if not self.tile_data:
            messagebox.showwarning("No Data", "Load tiles first before merging.")
            return

        not_loaded = [c for c in self.tile_paths if c not in self.tile_data]
        if not_loaded:
            messagebox.showwarning("Tiles Not Loaded",
                                   f"{len(not_loaded)} tiles haven't finished loading yet.")
            return

        margin = self.margin.get()
        quality = self.blend_quality.get()
        opts = {
            "blur": self.opt_blur.get(),
            "height_preserve": self.opt_height_preserve.get(),
            "feature_spill": self.opt_feature_spill.get(),
            "terrain_extend": self.opt_terrain_extend.get(),
        }

        # Calculate output size using compacted grid (same as merge logic)
        x_vals = sorted(set(x for x, y in self.tile_data))
        y_vals = sorted(set(y for x, y in self.tile_data))
        x_remap = {old: new for new, old in enumerate(x_vals)}
        y_remap = {old: new for new, old in enumerate(y_vals)}

        compacted = {}
        for (gx, gy), data in self.tile_data.items():
            compacted[(x_remap[gx], y_remap[gy])] = data

        col_widths = {}
        row_heights = {}
        for (cx, cy), data in compacted.items():
            th, tw = data.shape
            col_widths[cx] = max(col_widths.get(cx, 0), tw)
            row_heights[cy] = max(row_heights.get(cy, 0), th)

        out_w = sum(col_widths.values()) - (len(col_widths) - 1) * margin
        out_h = sum(row_heights.values()) - (len(row_heights) - 1) * margin
        out_mb = out_w * out_h * 2 / 1024 / 1024

        # Warn if output is very large
        if out_mb > 2048:
            if not messagebox.askyesno("Large Output",
                                       f"Output will be {out_w}x{out_h} pixels "
                                       f"({out_mb:.0f} MB).\n\n"
                                       f"This may take a while and use a lot of RAM.\n"
                                       f"Continue?"):
                return

        # Auto-generate filename with random suffix
        import random, string
        rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        ext = "." + self.tile_format.get()
        save_dir = self.save_path.get()

        if save_dir and os.path.isdir(save_dir):
            out_path = os.path.join(save_dir, f"merged_{rand_id}_{out_w}x{out_h}{ext}")
        elif save_dir:
            # save_path is a file path, use its directory
            out_path = os.path.join(os.path.dirname(save_dir), f"merged_{rand_id}_{out_w}x{out_h}{ext}")
        else:
            filetypes = [("PNG 16-bit", "*.png"), ("R16 raw", "*.r16")]
            out_path = filedialog.asksaveasfilename(
                title="Save merged heightmap as",
                filetypes=filetypes,
                defaultextension=".png",
                initialfile=f"merged_{rand_id}_{out_w}x{out_h}")
            if not out_path:
                return
            self.save_path.set(os.path.dirname(out_path))

        self.btn_merge.configure(state=tk.DISABLED, text="Merging...")
        self.btn_stitch.configure(state=tk.DISABLED)

        def worker():
            try:
                self.root.after(0, lambda: self._log(
                    f"\nMerging {len(self.tile_data)} tiles into {out_w}x{out_h}..."))
                self.root.after(0, lambda: self._set_progress(0.05, "Height matching..."))

                def progress_log(msg):
                    self.root.after(0, lambda m=msg: self._log(f"  {m}"))
                    # Update progress based on message content
                    if "Height correction" in msg:
                        self.root.after(0, lambda: self._set_progress(0.15, "Height matching..."))
                    elif "Blend quality" in msg:
                        self.root.after(0, lambda: self._set_progress(0.25, "Compositing tiles..."))
                    elif "Output size" in msg:
                        self.root.after(0, lambda: self._set_progress(0.10, "Calculating layout..."))

                result = merge_tiles_to_single(
                    self.tile_data, margin, quality=quality,
                    log_fn=progress_log, opts=opts
                )

                self.root.after(0, lambda: self._set_progress(0.75, "Saving file..."))
                self.root.after(0, lambda: self._log(f"Saving to {os.path.basename(out_path)}..."))

                if out_path.lower().endswith(".r16"):
                    save_tile_r16(out_path, result)
                else:
                    save_tile_png(out_path, result)

                file_mb = os.path.getsize(out_path) / 1024 / 1024
                self.root.after(0, lambda: self._set_progress(1.0, "Merge complete!"))
                self.root.after(0, lambda: self._log(
                    f"\nMerge complete! {out_w}x{out_h} ({file_mb:.1f} MB)"))
                self.root.after(0, lambda: self._log(f"Saved to: {out_path}"))
                self.root.after(0, lambda r=result: self._set_result(r))

            except Exception as e:
                self.root.after(0, lambda: self._log(f"\nERROR: {e}"))
                self.root.after(0, lambda: self._set_progress(0.0, "Error!"))

            finally:
                self.root.after(0, lambda: self.btn_merge.configure(
                    state=tk.NORMAL, text="MERGE TO FILE"))
                self.root.after(0, lambda: self.btn_stitch.configure(state=tk.NORMAL))
                self.root.after(0, self._rebuild_grid)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


# ---------------------------------------------------------------------------
# Tile placement dialog (when filename has no coordinates)
# ---------------------------------------------------------------------------

class TilePlacementDialog:
    def __init__(self, parent, max_x, max_y, filename="", def_x=0, def_y=0):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Place Tile")
        self.dialog.configure(bg=BG_COLOR)
        self.dialog.geometry("320x200")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Title inset
        title_frame = tk.Frame(self.dialog, bg=PANEL_BG, bd=2, relief=tk.SUNKEN)
        title_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        title_text = f"PLACE: {filename}" if filename else "PLACE TILE"
        tk.Label(title_frame, text=title_text, bg=PANEL_BG, fg=TEXT_COLOR,
                 font=FONT_BODY).pack(pady=4, padx=4)

        tk.Label(self.dialog, text="Grid Position:", bg=BG_COLOR, fg=TEXT_COLOR,
                 font=FONT_LABEL).pack(pady=(8, 4))

        frame = tk.Frame(self.dialog, bg=BG_COLOR)
        frame.pack()

        self.x_var = tk.IntVar(value=def_x)
        self.y_var = tk.IntVar(value=def_y)

        tk.Label(frame, text="X:", bg=BG_COLOR, fg=TEXT_COLOR, font=FONT_BODY).grid(row=0, column=0, padx=4)
        tk.Spinbox(frame, from_=0, to=max(max_x - 1, 7), width=4,
                   textvariable=self.x_var, font=FONT_MONO,
                   bg=PANEL_BG, fg="#b0b0a8", buttonbackground=BTN_FACE,
                   insertbackground=TEXT_COLOR).grid(row=0, column=1, padx=4)
        tk.Label(frame, text="Y:", bg=BG_COLOR, fg=TEXT_COLOR, font=FONT_BODY).grid(row=0, column=2, padx=4)
        tk.Spinbox(frame, from_=0, to=max(max_y - 1, 7), width=4,
                   textvariable=self.y_var, font=FONT_MONO,
                   bg=PANEL_BG, fg="#b0b0a8", buttonbackground=BTN_FACE,
                   insertbackground=TEXT_COLOR).grid(row=0, column=3, padx=4)

        btn_frame = tk.Frame(self.dialog, bg=BG_COLOR)
        btn_frame.pack(pady=12)

        tk.Button(btn_frame, text="OK", command=self._ok,
                  bg=BTN_FACE, fg=BTN_TEXT, font=FONT_BTN,
                  relief=tk.RAISED, bd=2, padx=16, cursor="hand2").pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="CANCEL", command=self._cancel,
                  bg=BTN_FACE, fg=BTN_TEXT, font=FONT_BTN,
                  relief=tk.RAISED, bd=2, padx=16, cursor="hand2").pack(side=tk.LEFT, padx=4)

        self.dialog.wait_window()

    def _ok(self):
        self.result = (self.x_var.get(), self.y_var.get())
        self.dialog.destroy()

    def _cancel(self):
        self.dialog.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Taskbar root - visible in taskbar, manages minimize/restore
    _root = tk.Tk()
    _root.title("Heightmap Tile Stitcher")
    _root.geometry("1x1+0+0")
    _root.attributes("-alpha", 0.0)

    # Actual app window
    _app_win = tk.Toplevel(_root)
    app = StitcherApp(_app_win)

    # --- Minimize: hide app, iconify root (shows in taskbar) ---
    def _minimize():
        _app_win.withdraw()
        _root.iconify()

    # --- Restore: when root is restored from taskbar ---
    def _restore(event=None):
        _root.deiconify()
        _root.attributes("-alpha", 0.0)  # Keep root invisible
        _app_win.deiconify()
        _app_win.overrideredirect(True)
        _app_win.lift()
        _app_win.after(50, _app_win.focus_force)

    # --- Close: destroy everything ---
    def _close():
        _root.destroy()

    # Bind restore event
    _root.bind("<Map>", _restore)

    # Wire up the app's buttons
    app._minimize = _minimize
    app._close = _close

    # Handle if user Alt-F4s the root
    _root.protocol("WM_DELETE_WINDOW", _close)

    _root.mainloop()
