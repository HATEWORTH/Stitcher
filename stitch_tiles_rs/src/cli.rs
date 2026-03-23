//! Stitch independently-made heightmap tiles by blending their shared edges.
//!
//! Supports both 16-bit PNG (.png) and raw 16-bit (.r16) heightmaps.
//! Tiles are arranged on a grid and neighboring edges are linearly blended
//! over a configurable margin so they meet seamlessly.
//!
//! Usage:
//!     stitch_tiles <tile_dir> [--margin 128] [--size 4033] [--format png]

use clap::Parser;
use image::{GrayImage, ImageReader, Luma};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "stitch_tiles", about = "Stitch heightmap tiles by blending edges")]
struct Cli {
    /// Directory containing tile files
    tile_dir: PathBuf,

    /// Blend margin in pixels
    #[arg(long, default_value_t = 128)]
    margin: usize,

    /// Tile dimensions for .r16 files
    #[arg(long, default_value_t = 4033)]
    size: usize,

    /// Tile file format
    #[arg(long, default_value = "png", value_parser = ["png", "r16"])]
    format: String,

    /// Preview mode - show grid and neighbors without modifying files
    #[arg(long)]
    preview: bool,

    /// Output directory (default: <tile_dir>/stitched)
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Tile I/O
// ---------------------------------------------------------------------------

/// A 2-D heightmap stored as row-major u16 values.
struct Tile {
    width: usize,
    height: usize,
    data: Vec<u16>,
}

impl Tile {
    fn new(width: usize, height: usize, data: Vec<u16>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> u16 {
        self.data[y * self.width + x]
    }

    #[inline]
    fn set(&mut self, x: usize, y: usize, v: u16) {
        self.data[y * self.width + x] = v;
    }
}

fn load_tile_png(path: &Path) -> Tile {
    let img = ImageReader::open(path)
        .unwrap_or_else(|e| { eprintln!("ERROR: Cannot open {}: {e}", path.display()); process::exit(1); })
        .decode()
        .unwrap_or_else(|e| { eprintln!("ERROR: Cannot decode {}: {e}", path.display()); process::exit(1); });

    let gray16 = img.into_luma16();
    let (w, h) = gray16.dimensions();
    let data: Vec<u16> = gray16.into_raw();
    Tile::new(w as usize, h as usize, data)
}

fn save_tile_png(path: &Path, tile: &Tile) {
    let img = GrayImage::from_raw(
        tile.width as u32,
        tile.height as u32,
        tile.data.iter().flat_map(|&v| v.to_ne_bytes()).collect(),
    )
    .expect("failed to create image buffer");

    // GrayImage is 8-bit; we need 16-bit output. Build a Luma16 image instead.
    let mut img16 = image::ImageBuffer::<Luma<u16>, Vec<u16>>::new(
        tile.width as u32,
        tile.height as u32,
    );
    for (i, &v) in tile.data.iter().enumerate() {
        let x = (i % tile.width) as u32;
        let y = (i / tile.width) as u32;
        img16.put_pixel(x, y, Luma([v]));
    }
    drop(img); // not needed

    img16.save(path).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot save {}: {e}", path.display());
        process::exit(1);
    });
}

fn load_tile_r16(path: &Path, size: usize) -> Tile {
    let bytes = fs::read(path).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot read {}: {e}", path.display());
        process::exit(1);
    });
    let expected = size * size * 2;
    if bytes.len() != expected {
        eprintln!(
            "ERROR: {} has {} bytes, expected {} ({size}x{size} u16)",
            path.display(),
            bytes.len(),
            expected
        );
        process::exit(1);
    }
    let data: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Tile::new(size, size, data)
}

fn save_tile_r16(path: &Path, tile: &Tile) {
    let bytes: Vec<u8> = tile.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot write {}: {e}", path.display());
        process::exit(1);
    });
}

// ---------------------------------------------------------------------------
// Tile discovery
// ---------------------------------------------------------------------------

type Coord = (i32, i32);

fn find_tiles(tile_dir: &Path, fmt: &str) -> HashMap<Coord, PathBuf> {
    let re_xy = Regex::new(r"_x(\d+)_y(\d+)").unwrap();
    let re_yx = Regex::new(r"_y(\d+)_x(\d+)").unwrap();
    let ext = format!(".{fmt}");
    let mut tiles = HashMap::new();

    let entries = fs::read_dir(tile_dir).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot read directory {}: {e}", tile_dir.display());
        process::exit(1);
    });

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.to_lowercase().ends_with(&ext) {
            continue;
        }
        if let Some(caps) = re_xy.captures(&name) {
            let x: i32 = caps[1].parse().unwrap();
            let y: i32 = caps[2].parse().unwrap();
            tiles.insert((x, y), entry.path());
        } else if let Some(caps) = re_yx.captures(&name) {
            let y: i32 = caps[1].parse().unwrap();
            let x: i32 = caps[2].parse().unwrap();
            tiles.insert((x, y), entry.path());
        }
    }

    tiles
}

// ---------------------------------------------------------------------------
// Blending
// ---------------------------------------------------------------------------

/// Blend the right edge of tile_a with the left edge of tile_b (horizontal neighbors).
fn blend_horizontal(tile_a: &mut Tile, tile_b: &mut Tile, margin: usize) {
    let h = tile_a.height;
    for row in 0..h {
        for i in 0..margin {
            let wa = 1.0 - (i as f64 / (margin - 1) as f64); // 1.0 -> 0.0
            let wb = 1.0 - wa;

            let col_a = tile_a.width - margin + i;
            let va = tile_a.get(col_a, row) as f64;
            let vb = tile_b.get(i, row) as f64;

            let blended = (va * wa + vb * wb).clamp(0.0, 65535.0) as u16;
            tile_a.set(col_a, row, blended);
            tile_b.set(i, row, blended);
        }
    }
}

/// Blend the bottom edge of tile_a with the top edge of tile_b (vertical neighbors).
fn blend_vertical(tile_a: &mut Tile, tile_b: &mut Tile, margin: usize) {
    let w = tile_a.width;
    for i in 0..margin {
        let wa = 1.0 - (i as f64 / (margin - 1) as f64); // 1.0 -> 0.0
        let wb = 1.0 - wa;

        let row_a = tile_a.height - margin + i;
        for col in 0..w {
            let va = tile_a.get(col, row_a) as f64;
            let vb = tile_b.get(col, i) as f64;

            let blended = (va * wa + vb * wb).clamp(0.0, 65535.0) as u16;
            tile_a.set(col, row_a, blended);
            tile_b.set(col, i, blended);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // Find tiles
    let tile_paths = find_tiles(&cli.tile_dir, &cli.format);
    if tile_paths.is_empty() {
        eprintln!(
            "No .{} tiles found in {}",
            cli.format,
            cli.tile_dir.display()
        );
        eprintln!("Expected filenames like: tile_x0_y0.{}", cli.format);
        process::exit(1);
    }

    // Grid bounds
    let max_x = tile_paths.keys().map(|&(x, _)| x).max().unwrap();
    let max_y = tile_paths.keys().map(|&(_, y)| y).max().unwrap();
    println!(
        "Found {} tiles in a {}x{} grid",
        tile_paths.len(),
        max_x + 1,
        max_y + 1
    );
    println!("Blend margin: {}px", cli.margin);

    // Show grid
    println!("\nGrid layout:");
    for y in 0..=max_y {
        let mut row = String::new();
        for x in 0..=max_x {
            if tile_paths.contains_key(&(x, y)) {
                row.push_str(&format!(" [{x},{y}]"));
            } else {
                row.push_str("  ... ");
            }
        }
        println!("{row}");
    }

    // Find neighbor pairs
    let mut h_pairs: Vec<(Coord, Coord)> = Vec::new();
    let mut v_pairs: Vec<(Coord, Coord)> = Vec::new();
    for &(x, y) in tile_paths.keys() {
        if tile_paths.contains_key(&(x + 1, y)) {
            h_pairs.push(((x, y), (x + 1, y)));
        }
        if tile_paths.contains_key(&(x, y + 1)) {
            v_pairs.push(((x, y), (x, y + 1)));
        }
    }
    // Sort for deterministic ordering
    h_pairs.sort();
    v_pairs.sort();

    println!(
        "\nEdges to blend: {} horizontal, {} vertical",
        h_pairs.len(),
        v_pairs.len()
    );

    // Preview mode
    if cli.preview {
        println!("\n[Preview mode - no files modified]");
        for (a, b) in &h_pairs {
            println!("  H: ({},{}) <-> ({},{})", a.0, a.1, b.0, b.1);
        }
        for (a, b) in &v_pairs {
            println!("  V: ({},{}) <-> ({},{})", a.0, a.1, b.0, b.1);
        }
        return;
    }

    // Output directory
    let out_dir = cli.output.unwrap_or_else(|| cli.tile_dir.join("stitched"));
    fs::create_dir_all(&out_dir).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot create output directory: {e}");
        process::exit(1);
    });

    // Load all tiles
    println!("\nLoading tiles...");
    let mut tiles: HashMap<Coord, Tile> = HashMap::new();
    for (&coord, path) in &tile_paths {
        let name = path.file_name().unwrap().to_string_lossy();
        print!("  Loading {name}...");
        let tile = if cli.format == "png" {
            load_tile_png(path)
        } else {
            load_tile_r16(path, cli.size)
        };
        println!(" {}x{}", tile.width, tile.height);
        tiles.insert(coord, tile);
    }

    // Validate margin
    let first_tile = tiles.values().next().unwrap();
    let tile_size = first_tile.height;
    if cli.margin >= tile_size / 2 {
        eprintln!(
            "ERROR: Margin {} is too large for tile size {tile_size}",
            cli.margin
        );
        process::exit(1);
    }

    // Blend horizontal neighbors
    println!("\nBlending horizontal edges...");
    for (a, b) in &h_pairs {
        println!("  ({},{}) <-> ({},{})", a.0, a.1, b.0, b.1);
        // Temporarily remove both tiles to satisfy borrow checker
        let mut ta = tiles.remove(a).unwrap();
        let mut tb = tiles.remove(b).unwrap();
        blend_horizontal(&mut ta, &mut tb, cli.margin);
        tiles.insert(*a, ta);
        tiles.insert(*b, tb);
    }

    // Blend vertical neighbors
    println!("Blending vertical edges...");
    for (a, b) in &v_pairs {
        println!("  ({},{}) <-> ({},{})", a.0, a.1, b.0, b.1);
        let mut ta = tiles.remove(a).unwrap();
        let mut tb = tiles.remove(b).unwrap();
        blend_vertical(&mut ta, &mut tb, cli.margin);
        tiles.insert(*a, ta);
        tiles.insert(*b, tb);
    }

    // Save stitched tiles
    println!("\nSaving to {}...", out_dir.display());
    for (&coord, tile) in &tiles {
        let orig_name = tile_paths[&coord]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let out_path = out_dir.join(&orig_name);
        print!("  Saving {orig_name}...");
        if cli.format == "png" {
            save_tile_png(&out_path, tile);
        } else {
            save_tile_r16(&out_path, tile);
        }
        println!(" OK");
    }

    println!(
        "\nDone! {} stitched tiles written to: {}",
        tiles.len(),
        out_dir.display()
    );
    println!(
        "Blend margin was {}px on each side of every shared edge.",
        cli.margin
    );
}
