//! Tile I/O: load/save 16-bit PNG and raw R16 heightmaps, discover tiles on disk.

use image::{ImageReader, Luma};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

static RE_XY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_x(\d+)_y(\d+)").unwrap());
static RE_YX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_y(\d+)_x(\d+)").unwrap());

/// Grid coordinate (col, row).
pub type Coord = (i32, i32);

/// A 2-D heightmap stored as row-major u16 values.
#[derive(Clone)]
pub struct Tile {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u16>,
}

impl Tile {
    pub fn new(width: usize, height: usize, data: Vec<u16>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0u16; width * height] }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.width + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u16) {
        self.data[y * self.width + x] = v;
    }

    #[inline]
    pub fn get_f64(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x] as f64
    }

    /// Make an 8-bit grayscale thumbnail for display.
    pub fn thumbnail(&self, thumb_size: usize) -> Vec<u8> {
        let mut out = vec![0u8; thumb_size * thumb_size];
        for ty in 0..thumb_size {
            for tx in 0..thumb_size {
                let sx = tx * self.width / thumb_size;
                let sy = ty * self.height / thumb_size;
                let v = self.get(sx.min(self.width - 1), sy.min(self.height - 1));
                out[ty * thumb_size + tx] = (v as f64 / 65535.0 * 255.0) as u8;
            }
        }
        out
    }

    /// Convert entire tile to 8-bit grayscale for preview.
    pub fn to_gray8(&self) -> Vec<u8> {
        self.data.iter().map(|&v| (v as f64 / 65535.0 * 255.0) as u8).collect()
    }

    pub fn min_val(&self) -> u16 {
        self.data.iter().copied().min().unwrap_or(0)
    }

    pub fn max_val(&self) -> u16 {
        self.data.iter().copied().max().unwrap_or(0)
    }

    pub fn mean_val(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        self.data.iter().map(|&v| v as f64).sum::<f64>() / self.data.len() as f64
    }
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

pub fn load_tile_png(path: &Path) -> Result<Tile, String> {
    let img = ImageReader::open(path)
        .map_err(|e| format!("Cannot open {}: {e}", path.display()))?
        .decode()
        .map_err(|e| format!("Cannot decode {}: {e}", path.display()))?;
    let gray16 = img.into_luma16();
    let (w, h) = gray16.dimensions();
    let data: Vec<u16> = gray16.into_raw();
    Ok(Tile::new(w as usize, h as usize, data))
}

/// Save as 16-bit PNG (full precision heightmap).
pub fn save_tile_png16(path: &Path, tile: &Tile) -> Result<(), String> {
    let mut img = image::ImageBuffer::<Luma<u16>, Vec<u16>>::new(
        tile.width as u32, tile.height as u32,
    );
    for (i, &v) in tile.data.iter().enumerate() {
        let x = (i % tile.width) as u32;
        let y = (i / tile.width) as u32;
        img.put_pixel(x, y, Luma([v]));
    }
    img.save(path).map_err(|e| format!("Cannot save {}: {e}", path.display()))
}

/// Save as 8-bit PNG (normalized to 0-255).
pub fn save_tile_png8(path: &Path, tile: &Tile) -> Result<(), String> {
    let mut img = image::ImageBuffer::<Luma<u8>, Vec<u8>>::new(
        tile.width as u32, tile.height as u32,
    );
    for (i, &v) in tile.data.iter().enumerate() {
        let x = (i % tile.width) as u32;
        let y = (i / tile.width) as u32;
        img.put_pixel(x, y, Luma([(v as f64 / 65535.0 * 255.0) as u8]));
    }
    img.save(path).map_err(|e| format!("Cannot save {}: {e}", path.display()))
}

pub fn load_tile_r16(path: &Path, size_hint: Option<usize>) -> Result<Tile, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    let total = bytes.len() / 2;
    if bytes.len() % 2 != 0 {
        return Err(format!("{} has odd byte count {}", path.display(), bytes.len()));
    }

    // Auto-detect size
    let sqrt = (total as f64).sqrt() as usize;
    let size = if sqrt * sqrt == total {
        sqrt
    } else if let Some(hint) = size_hint {
        if hint * hint == total { hint }
        else {
            // Try common UE5 landscape sizes
            [8129, 4033, 2017, 1009, 505, 253].iter()
                .find(|&&s| s * s == total)
                .copied()
                .ok_or_else(|| format!("{} has {total} values which is not a perfect square", path.display()))?
        }
    } else {
        [8129, 4033, 2017, 1009, 505, 253].iter()
            .find(|&&s| s * s == total)
            .copied()
            .ok_or_else(|| format!("{} has {total} values which is not a perfect square", path.display()))?
    };

    let data: Vec<u16> = bytes.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok(Tile::new(size, size, data))
}

pub fn save_tile_r16(path: &Path, tile: &Tile) -> Result<(), String> {
    let bytes: Vec<u8> = tile.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).map_err(|e| format!("Cannot write {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

/// Split a merged heightmap back into a grid of tiles.
/// `cols` and `rows` define the grid dimensions.
/// Tiles share one edge row/column with their neighbors so UE5 landscape
/// tiles line up exactly (shared edge vertices).
pub fn split_into_tiles(merged: &Tile, cols: usize, rows: usize) -> HashMap<Coord, Tile> {
    if cols == 0 || rows == 0 || merged.width < cols || merged.height < rows {
        return HashMap::new();
    }

    let mut tiles = HashMap::new();

    // Each tile is (merged_w - cols + 1) / cols +1 wide to account for shared edges.
    // Actually simpler: divide evenly, then each tile's edge overlaps by 1 pixel.
    let tile_w = (merged.width - 1) / cols + 1; // e.g. 16129 / 2 + 1 = 8065
    let tile_h = (merged.height - 1) / rows + 1;

    for gy in 0..rows {
        for gx in 0..cols {
            // Start position: each tile starts 1 pixel before the previous tile ends
            let sx = gx * (tile_w - 1);
            let sy = gy * (tile_h - 1);
            let ew = tile_w.min(merged.width - sx);
            let eh = tile_h.min(merged.height - sy);

            let mut data = vec![0u16; ew * eh];
            for ty in 0..eh {
                for tx in 0..ew {
                    let mx = sx + tx;
                    let my = sy + ty;
                    if mx < merged.width && my < merged.height {
                        data[ty * ew + tx] = merged.get(mx, my);
                    }
                }
            }
            tiles.insert((gx as i32, gy as i32), Tile::new(ew, eh, data));
        }
    }
    tiles
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Find tiles in a directory. Tries coordinate patterns first (_xN_yN),
/// then falls back to assigning all PNG files to grid positions left-to-right, top-to-bottom.
pub fn find_tiles(tile_dir: &Path) -> HashMap<Coord, PathBuf> {
    let mut tiles = HashMap::new();
    let mut uncoordinated: Vec<PathBuf> = Vec::new();

    let entries = match fs::read_dir(tile_dir) {
        Ok(e) => e,
        Err(_) => return tiles,
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let lower = name.to_lowercase();
        if !lower.ends_with(".png") {
            continue;
        }
        if let Some(caps) = RE_XY.captures(&name) {
            if let (Ok(x), Ok(y)) = (caps[1].parse::<i32>(), caps[2].parse::<i32>()) {
                tiles.insert((x, y), entry.path());
            } else {
                uncoordinated.push(entry.path());
            }
        } else if let Some(caps) = RE_YX.captures(&name) {
            if let (Ok(y), Ok(x)) = (caps[1].parse::<i32>(), caps[2].parse::<i32>()) {
                tiles.insert((x, y), entry.path());
            } else {
                uncoordinated.push(entry.path());
            }
        } else {
            uncoordinated.push(entry.path());
        }
    }

    // If no coordinated tiles found, auto-assign uncoordinated PNGs to grid
    if tiles.is_empty() && !uncoordinated.is_empty() {
        uncoordinated.sort();
        let count = uncoordinated.len();
        let cols = (count as f64).sqrt().ceil() as i32;
        for (i, path) in uncoordinated.into_iter().enumerate() {
            let x = i as i32 % cols;
            let y = i as i32 / cols;
            tiles.insert((x, y), path);
        }
    }

    tiles
}
