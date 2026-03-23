//! Multi-resolution Laplacian pyramid blending for heightmaps.
//!
//! Decomposes tiles into frequency bands, blends each band with a
//! different-width mask (wide for low frequencies, narrow for high),
//! then reconstructs.  This avoids ghosting from simple crossfade
//! and preserves detail at all scales.

/// Downsample a 2D grid by 2x using box filter (average of 2x2 blocks).
fn downsample(src: &[f64], w: usize, h: usize) -> (Vec<f64>, usize, usize) {
    let nw = w / 2;
    let nh = h / 2;
    let mut dst = vec![0.0; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let sx = x * 2;
            let sy = y * 2;
            let mut sum = src[sy * w + sx];
            let mut count = 1.0;
            if sx + 1 < w {
                sum += src[sy * w + sx + 1];
                count += 1.0;
            }
            if sy + 1 < h {
                sum += src[(sy + 1) * w + sx];
                count += 1.0;
            }
            if sx + 1 < w && sy + 1 < h {
                sum += src[(sy + 1) * w + sx + 1];
                count += 1.0;
            }
            dst[y * nw + x] = sum / count;
        }
    }
    (dst, nw, nh)
}

/// Upsample a 2D grid by 2x using bilinear interpolation.
fn upsample(src: &[f64], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<f64> {
    let mut dst = vec![0.0; tw * th];
    for y in 0..th {
        for x in 0..tw {
            let fx = (x as f64) / (tw as f64) * (sw as f64) - 0.5;
            let fy = (y as f64) / (th as f64) * (sh as f64) - 0.5;
            let x0 = (fx.floor() as isize).max(0) as usize;
            let y0 = (fy.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let tx = fx - fx.floor();
            let ty = fy - fy.floor();
            dst[y * tw + x] = src[y0 * sw + x0] * (1.0 - tx) * (1.0 - ty)
                + src[y0 * sw + x1] * tx * (1.0 - ty)
                + src[y1 * sw + x0] * (1.0 - tx) * ty
                + src[y1 * sw + x1] * tx * ty;
        }
    }
    dst
}

/// A level in a Gaussian/Laplacian pyramid, carrying its dimensions.
struct PyramidLevel {
    data: Vec<f64>,
    w: usize,
    h: usize,
}

/// Build a Gaussian pyramid (sequence of progressively downsampled images).
fn build_gaussian_pyramid(src: &[f64], w: usize, h: usize, levels: usize) -> Vec<PyramidLevel> {
    let mut pyr = Vec::with_capacity(levels);
    pyr.push(PyramidLevel { data: src.to_vec(), w, h });
    for _ in 1..levels {
        let prev = pyr.last().unwrap();
        if prev.w < 4 || prev.h < 4 {
            break;
        }
        let (down, nw, nh) = downsample(&prev.data, prev.w, prev.h);
        pyr.push(PyramidLevel { data: down, w: nw, h: nh });
    }
    pyr
}

/// Build a Laplacian pyramid from a Gaussian pyramid.
/// Each level = gaussian[i] - upsample(gaussian[i+1]).
/// The coarsest level is kept as-is (low-pass residual).
fn build_laplacian_pyramid(gauss: &[PyramidLevel]) -> Vec<PyramidLevel> {
    let n = gauss.len();
    let mut lap = Vec::with_capacity(n);
    for i in 0..n - 1 {
        let up = upsample(&gauss[i + 1].data, gauss[i + 1].w, gauss[i + 1].h, gauss[i].w, gauss[i].h);
        let diff: Vec<f64> = gauss[i]
            .data
            .iter()
            .zip(&up)
            .map(|(&a, &b)| a - b)
            .collect();
        lap.push(PyramidLevel {
            data: diff,
            w: gauss[i].w,
            h: gauss[i].h,
        });
    }
    // Coarsest level stored directly
    let last = gauss.last().unwrap();
    lap.push(PyramidLevel {
        data: last.data.clone(),
        w: last.w,
        h: last.h,
    });
    lap
}

/// Collapse (reconstruct) from a Laplacian pyramid back to full resolution.
fn collapse_laplacian(lap: &[PyramidLevel]) -> Vec<f64> {
    let n = lap.len();
    let mut current = lap[n - 1].data.clone();
    let mut cw = lap[n - 1].w;
    let mut ch = lap[n - 1].h;
    for i in (0..n - 1).rev() {
        let up = upsample(&current, cw, ch, lap[i].w, lap[i].h);
        current = up
            .iter()
            .zip(&lap[i].data)
            .map(|(&u, &l)| u + l)
            .collect();
        cw = lap[i].w;
        ch = lap[i].h;
    }
    current
}

/// Blend two same-sized strips using Laplacian pyramid multi-resolution blending.
///
/// `mask` is a per-pixel weight (1.0 = tile A, 0.0 = tile B) at full resolution.
/// The mask is also decomposed into a Gaussian pyramid so that each frequency
/// band is blended with the appropriately-smoothed mask.
///
/// `levels` controls pyramid depth (5 is a good default for 1024px margins).
pub fn pyramid_blend(
    strip_a: &[f64],
    strip_b: &[f64],
    mask: &[f64],
    w: usize,
    h: usize,
    levels: usize,
) -> Vec<f64> {
    assert_eq!(strip_a.len(), w * h);
    assert_eq!(strip_b.len(), w * h);
    assert_eq!(mask.len(), w * h);

    let levels = levels.min(((w.min(h) as f64).log2() as usize).saturating_sub(1)).max(2);

    let ga = build_gaussian_pyramid(strip_a, w, h, levels);
    let gb = build_gaussian_pyramid(strip_b, w, h, levels);
    let gm = build_gaussian_pyramid(mask, w, h, levels);

    let la = build_laplacian_pyramid(&ga);
    let lb = build_laplacian_pyramid(&gb);

    // Blend each Laplacian level using the corresponding mask level
    let mut blended_lap: Vec<PyramidLevel> = Vec::with_capacity(la.len());
    for i in 0..la.len() {
        let n = la[i].data.len();
        let mut data = vec![0.0; n];
        for j in 0..n {
            let m = gm[i].data[j.min(gm[i].data.len() - 1)];
            data[j] = la[i].data[j] * m + lb[i].data[j] * (1.0 - m);
        }
        blended_lap.push(PyramidLevel {
            data,
            w: la[i].w,
            h: la[i].h,
        });
    }

    collapse_laplacian(&blended_lap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip_identity() {
        // A constant image should survive pyramid round-trip
        let w = 64;
        let h = 64;
        let val = 1000.0;
        let src = vec![val; w * h];
        let gauss = build_gaussian_pyramid(&src, w, h, 4);
        let lap = build_laplacian_pyramid(&gauss);
        let result = collapse_laplacian(&lap);
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - val).abs() < 1.0,
                "pixel {i} drifted: expected {val}, got {v}"
            );
        }
    }

    #[test]
    fn test_blend_identity_mask() {
        // mask=1.0 everywhere should return strip_a
        let w = 32;
        let h = 32;
        let a: Vec<f64> = (0..w * h).map(|i| (i * 10) as f64).collect();
        let b = vec![0.0; w * h];
        let mask = vec![1.0; w * h];
        let result = pyramid_blend(&a, &b, &mask, w, h, 4);
        for i in 0..w * h {
            assert!(
                (result[i] - a[i]).abs() < 2.0,
                "pixel {i}: expected {}, got {}",
                a[i],
                result[i]
            );
        }
    }
}
