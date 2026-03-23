//! Terrain-aware heightmap tile blending.
//!
//! Three quality levels:
//! - **Fast**: smoothstep crossfade with edge height matching
//! - **High**: Laplacian pyramid multi-resolution blend with smoothstep mask
//! - **Ultra**: gradient-aware graph-cut seam + Laplacian pyramid blend
//!            + wide-band Poisson smooth + thermal erosion

use crate::pyramid;
use crate::tile::Tile;

// ── Named constants for magic numbers ────────────────────────────────────────
const EDGE_REACH_FRACTION: f64 = 0.4;      // Global correction reaches 40% into tile
const PROFILE_REACH_FRACTION: f64 = 0.25;   // Per-row correction reaches 25%
const PROFILE_SAMPLE_PIXELS: usize = 8;     // Edge sample band width
const GRADIENT_LAMBDA: f64 = 0.5;           // Gradient weight in seam cost
const THERMAL_TALUS_HIGH: f64 = 3.0;        // Talus angle for High quality
const THERMAL_TALUS_ULTRA: f64 = 2.0;       // Talus angle for Ultra quality
const THERMAL_ITERS_HIGH: usize = 15;       // Erosion passes for High
const THERMAL_ITERS_ULTRA: usize = 40;      // Erosion passes for Ultra
const EROSION_TRANSFER_RATE: f64 = 0.4;     // Material transfer fraction

#[derive(Clone)]
pub struct BlendOpts {
    pub terrain_extend: bool,
}

impl Default for BlendOpts {
    fn default() -> Self {
        Self { terrain_extend: true }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Quality { Fast, High, Ultra }

// ── Weight curves ────────────────────────────────────────────────────────────

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smooth weights from 1.0 -> 0.0 across n samples.
fn smooth_weights(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1.0 - smoothstep(i as f64 / (n - 1).max(1) as f64)).collect()
}

// ── 1-D Gaussian smoothing ──────────────────────────────────────────────────

fn gaussian_smooth_1d(arr: &[f64], sigma: f64) -> Vec<f64> {
    let size = ((sigma * 6.0) as usize) | 1;
    let half = size / 2;
    let kernel: Vec<f64> = {
        let raw: Vec<f64> = (0..size).map(|i| {
            let x = i as f64 - half as f64;
            (-0.5 * (x / sigma).powi(2)).exp()
        }).collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&v| v / s).collect()
    };
    let n = arr.len();
    (0..n).map(|i| {
        kernel.iter().enumerate().map(|(ki, &kv)| {
            let si = (i as isize + ki as isize - half as isize).max(0).min(n as isize - 1) as usize;
            arr[si] * kv
        }).sum()
    }).collect()
}

// ── Multi-pass edge height matching ──────────────────────────────────────────
// 1. Global mean offset: match average edge elevations
// 2. Per-pixel profile matching: match the elevation profile along the seam
//    so mountains connect to mountains, valleys to valleys
// 3. Cosine falloff deep into each tile (40% of tile width) to avoid bands

fn match_edge_heights_h(a: &mut [f64], aw: usize, ah: usize,
                        b: &mut [f64], bw: usize, _margin: usize) {
    // --- Pass 1: Global mean offset ---
    let a_mean: f64 = (0..ah).map(|r| a[r * aw + aw - 1]).sum::<f64>() / ah as f64;
    let b_mean: f64 = (0..ah).map(|r| b[r * bw]).sum::<f64>() / ah as f64;
    let global_diff = a_mean - b_mean;

    let reach_a = (aw as f64 * EDGE_REACH_FRACTION).max(64.0);
    let reach_b = (bw as f64 * EDGE_REACH_FRACTION).max(64.0);

    for row in 0..ah {
        for col in 0..aw {
            let dist = (aw - 1 - col) as f64;
            if dist < reach_a {
                let t = dist / reach_a;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                a[row * aw + col] -= global_diff * 0.5 * w;
            }
        }
        for col in 0..bw {
            let dist = col as f64;
            if dist < reach_b {
                let t = dist / reach_b;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                b[row * bw + col] += global_diff * 0.5 * w;
            }
        }
    }

    // --- Pass 2: Per-row profile matching ---
    let sample = PROFILE_SAMPLE_PIXELS.min(aw).min(bw);
    let mut a_edge = vec![0.0; ah];
    let mut b_edge = vec![0.0; ah];
    for row in 0..ah {
        for i in 0..sample {
            a_edge[row] += a[row * aw + (aw - sample + i)];
            b_edge[row] += b[row * bw + i];
        }
        a_edge[row] /= sample as f64;
        b_edge[row] /= sample as f64;
    }

    let sigma = (ah as f64 / 8.0).max(32.0);
    let residual: Vec<f64> = a_edge.iter().zip(&b_edge).map(|(a, b)| a - b).collect();
    let residual = gaussian_smooth_1d(&residual, sigma);

    let reach2_a = (aw as f64 * PROFILE_REACH_FRACTION).max(48.0);
    let reach2_b = (bw as f64 * PROFILE_REACH_FRACTION).max(48.0);

    for row in 0..ah {
        let d = residual[row] * 0.5;
        for col in 0..aw {
            let dist = (aw - 1 - col) as f64;
            if dist < reach2_a {
                let t = dist / reach2_a;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                a[row * aw + col] -= d * w;
            }
        }
        for col in 0..bw {
            let dist = col as f64;
            if dist < reach2_b {
                let t = dist / reach2_b;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                b[row * bw + col] += d * w;
            }
        }
    }
}

fn match_edge_heights_v(a: &mut [f64], aw: usize, ah: usize,
                        b: &mut [f64], bw: usize, bh: usize, _margin: usize) {
    let a_mean: f64 = (0..aw).map(|c| a[(ah - 1) * aw + c]).sum::<f64>() / aw as f64;
    let b_mean: f64 = (0..aw).map(|c| b[c]).sum::<f64>() / aw as f64;
    let global_diff = a_mean - b_mean;

    let reach_a = (ah as f64 * EDGE_REACH_FRACTION).max(64.0);
    let reach_b = (bh as f64 * EDGE_REACH_FRACTION).max(64.0);

    for col in 0..aw {
        for row in 0..ah {
            let dist = (ah - 1 - row) as f64;
            if dist < reach_a {
                let t = dist / reach_a;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                a[row * aw + col] -= global_diff * 0.5 * w;
            }
        }
        for row in 0..bh {
            let dist = row as f64;
            if dist < reach_b {
                let t = dist / reach_b;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                b[row * bw + col] += global_diff * 0.5 * w;
            }
        }
    }

    // Per-column profile matching
    let sample = PROFILE_SAMPLE_PIXELS.min(ah).min(bh);
    let mut a_edge = vec![0.0; aw];
    let mut b_edge = vec![0.0; aw];
    for col in 0..aw {
        for i in 0..sample {
            a_edge[col] += a[(ah - sample + i) * aw + col];
            b_edge[col] += b[i * bw + col];
        }
        a_edge[col] /= sample as f64;
        b_edge[col] /= sample as f64;
    }

    let sigma = (aw as f64 / 8.0).max(32.0);
    let residual: Vec<f64> = a_edge.iter().zip(&b_edge).map(|(a, b)| a - b).collect();
    let residual = gaussian_smooth_1d(&residual, sigma);

    let reach2_a = (ah as f64 * PROFILE_REACH_FRACTION).max(48.0);
    let reach2_b = (bh as f64 * PROFILE_REACH_FRACTION).max(48.0);

    for col in 0..aw {
        let d = residual[col] * 0.5;
        for row in 0..ah {
            let dist = (ah - 1 - row) as f64;
            if dist < reach2_a {
                let t = dist / reach2_a;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                a[row * aw + col] -= d * w;
            }
        }
        for row in 0..bh {
            let dist = row as f64;
            if dist < reach2_b {
                let t = dist / reach2_b;
                let w = 0.5 * (1.0 + (t * std::f64::consts::PI).cos());
                b[row * bw + col] += d * w;
            }
        }
    }
}

// ── Gradient computation ────────────────────────────────────────────────────
// Compute partial derivatives (Sobel-style 3x3) for gradient-aware seam finding.

/// Compute gradient magnitude squared at each pixel in the strip.
/// Returns (grad_x, grad_y) arrays - each the central-difference derivative.
fn compute_gradients(data: &[f64], w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let mut gx = vec![0.0; w * h];
    let mut gy = vec![0.0; w * h];
    for row in 1..h.saturating_sub(1) {
        for col in 1..w.saturating_sub(1) {
            let idx = row * w + col;
            gx[idx] = (data[idx + 1] - data[idx - 1]) * 0.5;
            gy[idx] = (data[(row + 1) * w + col] - data[(row - 1) * w + col]) * 0.5;
        }
    }
    (gx, gy)
}

// ── Gradient-aware graph-cut seam finding ────────────────────────────────────
// Cost function includes both height difference AND gradient difference.
// The seam naturally follows ridgelines and avoids cutting across rivers.

/// Gradient-aware cost at a single pixel: squared height diff + lambda * squared gradient diff.
fn gradient_aware_cost(
    strip_a: &[f64], strip_b: &[f64],
    gx_a: &[f64], gy_a: &[f64],
    gx_b: &[f64], gy_b: &[f64],
    idx: usize, lambda: f64,
) -> f64 {
    let dh = strip_a[idx] - strip_b[idx];
    let dgx = gx_a[idx] - gx_b[idx];
    let dgy = gy_a[idx] - gy_b[idx];
    dh * dh + lambda * (dgx * dgx + dgy * dgy)
}

/// Find optimal vertical seam through an overlap strip (for horizontal neighbors).
/// Uses gradient-aware cost: height diff + gradient diff.
fn find_seam_vertical(strip_a: &[f64], strip_b: &[f64], h: usize, w: usize) -> Vec<usize> {
    if w == 0 || h == 0 {
        return vec![0; h];
    }

    let (gx_a, gy_a) = compute_gradients(strip_a, w, h);
    let (gx_b, gy_b) = compute_gradients(strip_b, w, h);

    // Build cost map
    let mut cost = vec![0.0f64; h * w];
    for i in 0..h * w {
        cost[i] = gradient_aware_cost(strip_a, strip_b, &gx_a, &gy_a, &gx_b, &gy_b, i, GRADIENT_LAMBDA);
    }

    // DP: accumulate min cost top-to-bottom
    let mut dp = cost.clone();
    for row in 1..h {
        for col in 0..w {
            let above = dp[(row - 1) * w + col];
            let above_l = if col > 0 { dp[(row - 1) * w + col - 1] } else { f64::INFINITY };
            let above_r = if col < w - 1 { dp[(row - 1) * w + col + 1] } else { f64::INFINITY };
            dp[row * w + col] += above.min(above_l).min(above_r);
        }
    }

    // Trace back from bottom
    let mut seam = vec![0usize; h];
    seam[h - 1] = (0..w).min_by(|&a, &b|
        dp[(h - 1) * w + a].total_cmp(&dp[(h - 1) * w + b])).unwrap();
    for row in (0..h - 1).rev() {
        let sc = seam[row + 1];
        let lo = sc.saturating_sub(1);
        let hi = (sc + 2).min(w);
        let mut best = lo;
        let mut best_val = dp[row * w + lo];
        for c in lo..hi {
            if dp[row * w + c] < best_val {
                best_val = dp[row * w + c];
                best = c;
            }
        }
        seam[row] = best;
    }
    seam
}

/// Find optimal horizontal seam through an overlap strip (for vertical neighbors).
fn find_seam_horizontal(strip_a: &[f64], strip_b: &[f64], h: usize, w: usize) -> Vec<usize> {
    if w == 0 || h == 0 {
        return vec![0; w];
    }

    let (gx_a, gy_a) = compute_gradients(strip_a, w, h);
    let (gx_b, gy_b) = compute_gradients(strip_b, w, h);

    let mut cost = vec![0.0f64; h * w];
    for i in 0..h * w {
        cost[i] = gradient_aware_cost(strip_a, strip_b, &gx_a, &gy_a, &gx_b, &gy_b, i, GRADIENT_LAMBDA);
    }

    let mut dp = cost.clone();
    for col in 1..w {
        for row in 0..h {
            let left = dp[row * w + col - 1];
            let left_u = if row > 0 { dp[(row - 1) * w + col - 1] } else { f64::INFINITY };
            let left_d = if row < h - 1 { dp[(row + 1) * w + col - 1] } else { f64::INFINITY };
            dp[row * w + col] += left.min(left_u).min(left_d);
        }
    }

    let mut seam = vec![0usize; w];
    seam[w - 1] = (0..h).min_by(|&a, &b|
        dp[a * w + w - 1].total_cmp(&dp[b * w + w - 1])).unwrap();
    for col in (0..w - 1).rev() {
        let sr = seam[col + 1];
        let lo = sr.saturating_sub(1);
        let hi = (sr + 2).min(h);
        let mut best = lo;
        let mut best_val = dp[lo * w + col];
        for r in lo..hi {
            if dp[r * w + col] < best_val {
                best_val = dp[r * w + col];
                best = r;
            }
        }
        seam[col] = best;
    }
    seam
}

/// Build a smooth seam mask from seam positions (vertical seam for horizontal neighbors).
/// 1.0 = use tile A, 0.0 = use tile B, smooth transition around the seam.
fn seam_to_mask_vertical(seam: &[usize], h: usize, w: usize) -> Vec<f64> {
    let falloff = (w / 6).max(4);
    let mut mask = vec![0.0; h * w];
    for row in 0..h {
        let sc = seam[row] as f64;
        for col in 0..w {
            let dist = col as f64 - sc;
            let t = ((dist + falloff as f64) / (2.0 * falloff as f64)).clamp(0.0, 1.0);
            mask[row * w + col] = 1.0 - smoothstep(t);
        }
    }
    mask
}

fn seam_to_mask_horizontal(seam: &[usize], h: usize, w: usize) -> Vec<f64> {
    let falloff = (h / 6).max(4);
    let mut mask = vec![0.0; h * w];
    for col in 0..w {
        let sr = seam[col] as f64;
        for row in 0..h {
            let dist = row as f64 - sr;
            let t = ((dist + falloff as f64) / (2.0 * falloff as f64)).clamp(0.0, 1.0);
            mask[row * w + col] = 1.0 - smoothstep(t);
        }
    }
    mask
}

// ── Wide-band Poisson smoothing ─────────────────────────────────────────────
// Solves for heights that have continuous gradients across the seam.
// Uses Gauss-Seidel relaxation with boundary conditions pinned to the
// already-blended data at the edges of the solve band.

fn poisson_smooth(data: &mut [f64], w: usize, h: usize,
                  strip_a: &[f64], strip_b: &[f64], mask: &[f64],
                  iterations: usize) {
    // Compute target Laplacian from blended gradients
    let mut lap = vec![0.0f64; h * w];
    for row in 1..h.saturating_sub(1) {
        for col in 1..w.saturating_sub(1) {
            let idx = row * w + col;
            let m = mask[idx];
            let lap_a = strip_a.get((row.wrapping_sub(1)) * w + col).copied().unwrap_or(0.0)
                      + strip_a.get((row + 1) * w + col).copied().unwrap_or(0.0)
                      + strip_a.get(idx.wrapping_sub(1)).copied().unwrap_or(0.0)
                      + strip_a.get(idx + 1).copied().unwrap_or(0.0)
                      - 4.0 * strip_a[idx];
            let lap_b = strip_b.get((row.wrapping_sub(1)) * w + col).copied().unwrap_or(0.0)
                      + strip_b.get((row + 1) * w + col).copied().unwrap_or(0.0)
                      + strip_b.get(idx.wrapping_sub(1)).copied().unwrap_or(0.0)
                      + strip_b.get(idx + 1).copied().unwrap_or(0.0)
                      - 4.0 * strip_b[idx];
            lap[idx] = lap_a * m + lap_b * (1.0 - m);
        }
    }

    // Gauss-Seidel relaxation with boundary conditions fixed
    for _ in 0..iterations {
        for row in 1..h.saturating_sub(1) {
            for col in 1..w.saturating_sub(1) {
                let idx = row * w + col;
                let neighbors = data[(row - 1) * w + col]
                              + data[(row + 1) * w + col]
                              + data[idx - 1]
                              + data[idx + 1];
                data[idx] = (neighbors - lap[idx]) / 4.0;
            }
        }
    }
}

// ── Thermal erosion along seam zone ─────────────────────────────────────────

fn thermal_erode(data: &mut [f64], w: usize, h: usize, iterations: usize, talus: f64) {
    for _ in 0..iterations {
        for row in 1..h.saturating_sub(1) {
            for col in 1..w.saturating_sub(1) {
                let idx = row * w + col;
                let center = data[idx];

                let diffs = [
                    center - data[idx + 1],
                    center - data[idx - 1],
                    center - data[(row + 1) * w + col],
                    center - data[(row - 1) * w + col],
                ];

                let d_max = diffs.iter().copied().fold(0.0f64, f64::max);
                if d_max <= talus { continue; }

                let mut total_excess = 0.0;
                let excesses: Vec<f64> = diffs.iter().map(|&d| {
                    let e = (d - talus).max(0.0);
                    total_excess += e;
                    e
                }).collect();

                if total_excess <= 0.0 { continue; }

                let move_amount = d_max * EROSION_TRANSFER_RATE;
                let neighbor_offsets: [isize; 4] = [1, -1, w as isize, -(w as isize)];

                for (i, &offset) in neighbor_offsets.iter().enumerate() {
                    if excesses[i] > 0.0 {
                        let fraction = excesses[i] / total_excess;
                        let transfer = move_amount * fraction;
                        data[idx] -= transfer * 0.5;
                        data[(idx as isize + offset) as usize] += transfer * 0.5;
                    }
                }
            }
        }
    }
}

// ── Public blend functions ───────────────────────────────────────────────────

pub fn blend_horizontal(tile_a: &mut Tile, tile_b: &mut Tile,
                        margin: usize, quality: Quality, opts: &BlendOpts) {
    let h = tile_a.height;
    let aw = tile_a.width;
    let bw = tile_b.width;

    let mut a_f: Vec<f64> = tile_a.data.iter().map(|&v| v as f64).collect();
    let mut b_f: Vec<f64> = tile_b.data.iter().map(|&v| v as f64).collect();

    // Step 1: Smooth edge height matching (all quality levels)
    if opts.terrain_extend {
        match_edge_heights_h(&mut a_f, aw, h, &mut b_f, bw, margin);
    }

    // Extract overlap strips
    let mut strip_a = vec![0.0; h * margin];
    let mut strip_b = vec![0.0; h * margin];
    for row in 0..h {
        for i in 0..margin {
            strip_a[row * margin + i] = a_f[row * aw + (aw - margin + i)];
            strip_b[row * margin + i] = b_f[row * bw + i];
        }
    }

    let blended = match quality {
        Quality::Fast => {
            // Simple smoothstep crossfade
            let weights = smooth_weights(margin);
            let mut result = vec![0.0; h * margin];
            for row in 0..h {
                for i in 0..margin {
                    let idx = row * margin + i;
                    result[idx] = strip_a[idx] * weights[i] + strip_b[idx] * (1.0 - weights[i]);
                }
            }
            result
        }
        Quality::High => {
            // Laplacian pyramid multi-resolution blend with smoothstep mask
            let mask: Vec<f64> = (0..h * margin).map(|i| {
                let col = i % margin;
                1.0 - smoothstep(col as f64 / (margin - 1).max(1) as f64)
            }).collect();
            let levels = 5.min(((margin.min(h) as f64).log2() as usize).saturating_sub(1)).max(2);
            let mut result = pyramid::pyramid_blend(&strip_a, &strip_b, &mask, margin, h, levels);
            // Light thermal erosion to naturalize
            thermal_erode(&mut result, margin, h, THERMAL_ITERS_HIGH, THERMAL_TALUS_HIGH);
            result
        }
        Quality::Ultra => {
            // Gradient-aware graph-cut seam → Laplacian pyramid blend
            //   → wide-band Poisson smooth → thermal erosion
            let seam = find_seam_vertical(&strip_a, &strip_b, h, margin);
            let mask = seam_to_mask_vertical(&seam, h, margin);

            // Multi-resolution blend using the seam mask
            let levels = 5.min(((margin.min(h) as f64).log2() as usize).saturating_sub(1)).max(2);
            let mut result = pyramid::pyramid_blend(&strip_a, &strip_b, &mask, margin, h, levels);

            // Wide-band Poisson smooth to eliminate residual gradient discontinuities
            let poisson_iters = (margin / 2).max(150).min(500);
            poisson_smooth(&mut result, margin, h, &strip_a, &strip_b, &mask, poisson_iters);

            // Thermal erosion for natural terrain transitions
            thermal_erode(&mut result, margin, h, THERMAL_ITERS_ULTRA, THERMAL_TALUS_ULTRA);

            result
        }
    };

    // Write terrain corrections to ENTIRE tiles first
    if opts.terrain_extend {
        for i in 0..a_f.len() {
            tile_a.data[i] = a_f[i].clamp(0.0, 65535.0) as u16;
        }
        for i in 0..b_f.len() {
            tile_b.data[i] = b_f[i].clamp(0.0, 65535.0) as u16;
        }
    }

    // Write full blend to both tiles' overlap zones.
    for row in 0..h {
        for i in 0..margin {
            let v = blended[row * margin + i].clamp(0.0, 65535.0) as u16;
            tile_a.set(aw - margin + i, row, v);
            tile_b.set(i, row, v);
        }
    }
}

pub fn blend_vertical(tile_a: &mut Tile, tile_b: &mut Tile,
                      margin: usize, quality: Quality, opts: &BlendOpts) {
    let w = tile_a.width;
    let ah = tile_a.height;
    let bh = tile_b.height;
    let aw = tile_a.width;
    let bw = tile_b.width;

    let mut a_f: Vec<f64> = tile_a.data.iter().map(|&v| v as f64).collect();
    let mut b_f: Vec<f64> = tile_b.data.iter().map(|&v| v as f64).collect();

    if opts.terrain_extend {
        match_edge_heights_v(&mut a_f, aw, ah, &mut b_f, bw, bh, margin);
    }

    let mut strip_a = vec![0.0; margin * w];
    let mut strip_b = vec![0.0; margin * w];
    for i in 0..margin {
        for col in 0..w {
            strip_a[i * w + col] = a_f[(ah - margin + i) * aw + col];
            strip_b[i * w + col] = b_f[i * bw + col];
        }
    }

    let blended = match quality {
        Quality::Fast => {
            let weights = smooth_weights(margin);
            let mut result = vec![0.0; margin * w];
            for i in 0..margin {
                for col in 0..w {
                    let idx = i * w + col;
                    result[idx] = strip_a[idx] * weights[i] + strip_b[idx] * (1.0 - weights[i]);
                }
            }
            result
        }
        Quality::High => {
            let mask: Vec<f64> = (0..margin * w).map(|i| {
                let row = i / w;
                1.0 - smoothstep(row as f64 / (margin - 1).max(1) as f64)
            }).collect();
            let levels = 5.min(((margin.min(w) as f64).log2() as usize).saturating_sub(1)).max(2);
            let mut result = pyramid::pyramid_blend(&strip_a, &strip_b, &mask, w, margin, levels);
            thermal_erode(&mut result, w, margin, THERMAL_ITERS_HIGH, THERMAL_TALUS_HIGH);
            result
        }
        Quality::Ultra => {
            let seam = find_seam_horizontal(&strip_a, &strip_b, margin, w);
            let mask = seam_to_mask_horizontal(&seam, margin, w);

            let levels = 5.min(((margin.min(w) as f64).log2() as usize).saturating_sub(1)).max(2);
            let mut result = pyramid::pyramid_blend(&strip_a, &strip_b, &mask, w, margin, levels);

            let poisson_iters = (margin / 2).max(150).min(500);
            poisson_smooth(&mut result, w, margin, &strip_a, &strip_b, &mask, poisson_iters);

            thermal_erode(&mut result, w, margin, THERMAL_ITERS_ULTRA, THERMAL_TALUS_ULTRA);
            result
        }
    };

    // Write terrain corrections to ENTIRE tiles first
    if opts.terrain_extend {
        for i in 0..a_f.len() {
            tile_a.data[i] = a_f[i].clamp(0.0, 65535.0) as u16;
        }
        for i in 0..b_f.len() {
            tile_b.data[i] = b_f[i].clamp(0.0, 65535.0) as u16;
        }
    }

    // Write full blend to both tiles' overlap zones.
    for i in 0..margin {
        for col in 0..w {
            let v = blended[i * w + col].clamp(0.0, 65535.0) as u16;
            tile_a.set(col, ah - margin + i, v);
            tile_b.set(col, i, v);
        }
    }
}

// ── Merge tiles to single heightmap ──────────────────────────────────────────

pub fn merge_tiles_to_single(
    tiles: &std::collections::HashMap<(i32, i32), Tile>,
    margin: usize,
    quality: Quality,
    opts: &BlendOpts,
    log_fn: &mut dyn FnMut(&str),
) -> Option<Tile> {
    if tiles.is_empty() { return None; }

    // Compact grid
    let mut x_vals: Vec<i32> = tiles.keys().map(|&(x, _)| x).collect();
    let mut y_vals: Vec<i32> = tiles.keys().map(|&(_, y)| y).collect();
    x_vals.sort(); x_vals.dedup();
    y_vals.sort(); y_vals.dedup();

    let x_remap: std::collections::HashMap<i32, i32> = x_vals.iter().enumerate().map(|(i, &v)| (v, i as i32)).collect();
    let y_remap: std::collections::HashMap<i32, i32> = y_vals.iter().enumerate().map(|(i, &v)| (v, i as i32)).collect();

    let mut compacted: std::collections::HashMap<(i32, i32), Tile> = std::collections::HashMap::new();
    for (&(gx, gy), data) in tiles {
        compacted.insert((x_remap[&gx], y_remap[&gy]), data.clone());
    }

    let grid_cols = x_vals.len();
    let grid_rows = y_vals.len();

    let mut col_widths = vec![0usize; grid_cols];
    let mut row_heights = vec![0usize; grid_rows];
    for (&(cx, cy), tile) in &compacted {
        col_widths[cx as usize] = col_widths[cx as usize].max(tile.width);
        row_heights[cy as usize] = row_heights[cy as usize].max(tile.height);
    }

    let mut col_offsets = vec![0usize; grid_cols];
    let mut x_pos = 0usize;
    for c in 0..grid_cols {
        col_offsets[c] = x_pos;
        x_pos += col_widths[c];
        if c < grid_cols - 1 { x_pos -= margin; }
    }
    let mut row_offsets = vec![0usize; grid_rows];
    let mut y_pos = 0usize;
    for r in 0..grid_rows {
        row_offsets[r] = y_pos;
        y_pos += row_heights[r];
        if r < grid_rows - 1 { y_pos -= margin; }
    }
    let out_w = x_pos;
    let out_h = y_pos;
    log_fn(&format!("Output size: {out_w}x{out_h} pixels"));

    // BFS height matching
    let mut corrections: std::collections::HashMap<(i32, i32), f64> = std::collections::HashMap::new();
    for &k in compacted.keys() { corrections.insert(k, 0.0); }

    let mut processed = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((0i32, 0i32));
    processed.insert((0, 0));

    while let Some((cx, cy)) = queue.pop_front() {
        if !compacted.contains_key(&(cx, cy)) { continue; }
        let cur = &compacted[&(cx, cy)];
        let cur_corr = corrections[&(cx, cy)];

        for &(nx, ny, is_right, is_down) in &[
            (cx+1, cy, true, false), (cx, cy+1, false, true),
            (cx-1, cy, true, false), (cx, cy-1, false, true)]
        {
            if !compacted.contains_key(&(nx, ny)) || processed.contains(&(nx, ny)) { continue; }
            let nb = &compacted[&(nx, ny)];
            let nb_corr = corrections[&(nx, ny)];

            let (cur_edge, nb_edge) = if is_right && nx > cx {
                let ce: f64 = (0..cur.height).map(|r| cur.get_f64(cur.width - 1, r)).sum::<f64>() / cur.height as f64;
                let ne: f64 = (0..nb.height).map(|r| nb.get_f64(0, r)).sum::<f64>() / nb.height as f64;
                (ce, ne)
            } else if is_right {
                let ce: f64 = (0..cur.height).map(|r| cur.get_f64(0, r)).sum::<f64>() / cur.height as f64;
                let ne: f64 = (0..nb.height).map(|r| nb.get_f64(nb.width - 1, r)).sum::<f64>() / nb.height as f64;
                (ce, ne)
            } else if is_down && ny > cy {
                let ce: f64 = (0..cur.width).map(|c| cur.get_f64(c, cur.height - 1)).sum::<f64>() / cur.width as f64;
                let ne: f64 = (0..nb.width).map(|c| nb.get_f64(c, 0)).sum::<f64>() / nb.width as f64;
                (ce, ne)
            } else {
                let ce: f64 = (0..cur.width).map(|c| cur.get_f64(c, 0)).sum::<f64>() / cur.width as f64;
                let ne: f64 = (0..nb.width).map(|c| nb.get_f64(c, nb.height - 1)).sum::<f64>() / nb.width as f64;
                (ce, ne)
            };

            *corrections.get_mut(&(nx, ny)).unwrap() += (cur_edge + cur_corr) - (nb_edge + nb_corr);
            processed.insert((nx, ny));
            queue.push_back((nx, ny));
        }
    }

    // Apply corrections and blend
    let mut work: std::collections::HashMap<(i32, i32), Tile> = std::collections::HashMap::new();
    for (&coord, tile) in &compacted {
        let corr = corrections[&coord];
        let mut t = tile.clone();
        if corr.abs() > 0.5 {
            log_fn(&format!("Height correction ({},{}): {:+.0}", coord.0, coord.1, corr));
            for v in &mut t.data { *v = (*v as f64 + corr).clamp(0.0, 65535.0) as u16; }
        }
        work.insert(coord, t);
    }

    // Blend neighbor pairs
    let mut h_pairs: Vec<_> = Vec::new();
    let mut v_pairs: Vec<_> = Vec::new();
    for &(cx, cy) in work.keys() {
        if work.contains_key(&(cx + 1, cy)) { h_pairs.push(((cx, cy), (cx + 1, cy))); }
        if work.contains_key(&(cx, cy + 1)) { v_pairs.push(((cx, cy), (cx, cy + 1))); }
    }
    h_pairs.sort(); v_pairs.sort();

    for (a, b) in &h_pairs {
        log_fn(&format!("  H-blend ({},{}) <-> ({},{})", a.0, a.1, b.0, b.1));
        let mut ta = work.remove(a).unwrap();
        let mut tb = work.remove(b).unwrap();
        blend_horizontal(&mut ta, &mut tb, margin, quality, opts);
        work.insert(*a, ta); work.insert(*b, tb);
    }
    for (a, b) in &v_pairs {
        log_fn(&format!("  V-blend ({},{}) <-> ({},{})", a.0, a.1, b.0, b.1));
        let mut ta = work.remove(a).unwrap();
        let mut tb = work.remove(b).unwrap();
        blend_vertical(&mut ta, &mut tb, margin, quality, opts);
        work.insert(*a, ta); work.insert(*b, tb);
    }

    // Composite onto canvas
    let weights = smooth_weights(margin);
    let mut canvas = vec![0.0f64; out_w * out_h];
    let mut weight_map = vec![0.0f64; out_w * out_h];

    for (&(cx, cy), tile) in &work {
        let rx = col_offsets[cx as usize];
        let ry = row_offsets[cy as usize];
        let tw = tile.width;
        let th = tile.height;

        for ty in 0..th {
            for tx in 0..tw {
                let mut w = 1.0;
                if compacted.contains_key(&(cx - 1, cy)) && tx < margin {
                    w *= weights[margin - 1 - tx];
                }
                if compacted.contains_key(&(cx + 1, cy)) && tx >= tw - margin {
                    w *= weights[tx - (tw - margin)];
                }
                if compacted.contains_key(&(cx, cy - 1)) && ty < margin {
                    w *= weights[margin - 1 - ty];
                }
                if compacted.contains_key(&(cx, cy + 1)) && ty >= th - margin {
                    w *= weights[ty - (th - margin)];
                }

                let ox = rx + tx;
                let oy = ry + ty;
                if ox < out_w && oy < out_h {
                    let idx = oy * out_w + ox;
                    canvas[idx] += tile.get(tx, ty) as f64 * w;
                    weight_map[idx] += w;
                }
            }
        }
    }

    let mut result = Tile::zeros(out_w, out_h);
    for i in 0..out_w * out_h {
        let w = if weight_map[i] > 0.0 { weight_map[i] } else { 1.0 };
        result.data[i] = (canvas[i] / w).clamp(0.0, 65535.0) as u16;
    }

    Some(result)
}
