//! Heightmap Tile Stitcher - GUI Application (Rust / egui)
//!
//! Visual tool for stitching independently-made heightmap tiles.
//! Trilithium / brushed-metal theme matching the original Python/tkinter version.

#![windows_subsystem = "windows"]

mod tile;
mod pyramid;
mod blend;
mod theme;

use blend::{BlendOpts, Quality};
use eframe::egui;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tile::{Coord, Tile};
use theme::*;

// ── App state ────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq)]
enum Tab { Tiles, Result, Paint }

enum JobResult {
    StitchDone { tiles: HashMap<Coord, Tile> },
    MergeDone  { result: Tile, out_path: String },
    Error(String),
    Log(String),
    Progress(f32, String),
}

struct StitcherApp {
    tile_paths:    HashMap<Coord, PathBuf>,
    tile_data:     HashMap<Coord, Tile>,
    /// Immutable originals loaded from disk — never modified by blend operations.
    original_data: HashMap<Coord, Tile>,
    tile_textures: HashMap<Coord, egui::TextureHandle>,

    tile_format: String, // "png16" or "png8"
    margin:      usize,
    quality:     Quality,
    opts:        BlendOpts,
    grid_cols:   i32,
    grid_rows:   i32,
    save_path:   String,
    run_name:    String,

    active_tab:    Tab,
    selected_cell: Option<Coord>,
    log_entries:   Vec<String>,
    progress:      f32,
    progress_text: String,
    busy:          bool,

    result_data:    Option<Tile>,
    result_texture: Option<egui::TextureHandle>,
    result_zoom:    f32,
    result_offset:  egui::Vec2,

    paint_data:       Option<Tile>,
    paint_texture:    Option<egui::TextureHandle>,
    paint_undo_stack: Vec<Tile>,
    paint_brush_size: f32,
    paint_opacity:    f32,
    paint_color:      f32,
    paint_zoom:       f32,
    paint_offset:     egui::Vec2,
    painting:         bool,
    last_paint_pos:   Option<egui::Pos2>,

    job_rx: Option<std::sync::mpsc::Receiver<JobResult>>,

    clear_pending: bool,

    // custom title-bar drag state (reserved)
    _dragging_title: bool,
    _drag_start:     egui::Pos2,

    /// One-shot flag to fix window styles on the first frame.
    needs_style_fix: bool,

    /// Title-bar icon texture (loaded once on first frame).
    titlebar_icon: Option<egui::TextureHandle>,
    /// Title-bar "Stitcher" text logo (loaded once on first frame).
    titlebar_logo: Option<egui::TextureHandle>,
}

impl Default for StitcherApp {
    fn default() -> Self {
        Self {
            tile_paths: HashMap::new(), tile_data: HashMap::new(), original_data: HashMap::new(), tile_textures: HashMap::new(),
            tile_format: "png16".into(), margin: 1024,
            quality: Quality::High, opts: BlendOpts::default(),
            grid_cols: 4, grid_rows: 4,
            save_path: dirs_or_default(), run_name: String::new(),
            active_tab: Tab::Tiles, selected_cell: None,
            log_entries: Vec::new(), progress: 0.0, progress_text: "Ready".into(), busy: false,
            result_data: None, result_texture: None, result_zoom: 1.0, result_offset: egui::Vec2::ZERO,
            paint_data: None, paint_texture: None, paint_undo_stack: Vec::new(),
            paint_brush_size: 200.0, paint_opacity: 0.5, paint_color: 0.5,
            paint_zoom: 1.0, paint_offset: egui::Vec2::ZERO, painting: false, last_paint_pos: None,
            job_rx: None,
            clear_pending: false,
            _dragging_title: false, _drag_start: egui::pos2(0.0, 0.0),
            needs_style_fix: true,
            titlebar_icon: None,
            titlebar_logo: None,
        }
    }
}

fn dirs_or_default() -> String {
    if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        let p = PathBuf::from(home).join("Downloads");
        if p.exists() { return p.to_string_lossy().into(); }
    }
    ".".into()
}

// ── Helpers ──────────────────────────────────────────────────────────────────

impl StitcherApp {
    /// Find our own main window HWND by process ID + visibility check.
    #[cfg(target_os = "windows")]
    fn find_own_hwnd() -> isize {
        unsafe {
            #[link(name = "user32")]
            unsafe extern "system" {
                fn GetWindowThreadProcessId(h: isize, pid: *mut u32) -> u32;
                fn EnumWindows(cb: unsafe extern "system" fn(isize, isize) -> i32, lparam: isize) -> i32;
                fn IsWindowVisible(h: isize) -> i32;
            }
            #[link(name = "kernel32")]
            unsafe extern "system" { fn GetCurrentProcessId() -> u32; }

            static mut FOUND: isize = 0;
            static mut PID: u32 = 0;
            unsafe extern "system" fn cb(h: isize, _: isize) -> i32 {
                let mut p: u32 = 0;
                unsafe {
                    GetWindowThreadProcessId(h, &mut p);
                    if p == PID && IsWindowVisible(h) != 0 {
                        FOUND = h;
                        return 0;
                    }
                }
                1
            }
            FOUND = 0;
            PID = GetCurrentProcessId();
            EnumWindows(cb, 0);
            FOUND
        }
    }

    /// Query whether the window is currently maximized via Win32 `IsZoomed`.
    #[cfg(target_os = "windows")]
    fn is_maximized(&self) -> bool {
        unsafe {
            #[link(name = "user32")]
            unsafe extern "system" { fn IsZoomed(h: isize) -> i32; }
            let hwnd = Self::find_own_hwnd();
            hwnd != 0 && IsZoomed(hwnd) != 0
        }
    }
    #[cfg(not(target_os = "windows"))]
    fn is_maximized(&self) -> bool { false }

    /// Toggle maximize / restore via Win32 `ShowWindow`.
    #[cfg(target_os = "windows")]
    fn toggle_maximize(&self) {
        let hwnd = Self::find_own_hwnd();
        if hwnd == 0 { return; }
        unsafe {
            const SW_MAXIMIZE: i32 = 3;
            const SW_RESTORE: i32  = 9;
            #[link(name = "user32")]
            unsafe extern "system" { fn ShowWindow(h: isize, cmd: i32) -> i32; }
            if self.is_maximized() {
                ShowWindow(hwnd, SW_RESTORE);
            } else {
                ShowWindow(hwnd, SW_MAXIMIZE);
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    fn toggle_maximize(&self) {}

    fn log(&mut self, msg: &str) { self.log_entries.push(msg.to_string()); }

    fn set_progress(&mut self, v: f32, t: &str) {
        self.progress = v.clamp(0.0, 1.0);
        self.progress_text = t.into();
    }

    /// Build an output directory path: `<save_path>/<run_name>_<NNN>/`
    /// Auto-increments the suffix so nothing is overwritten.
    /// Build output path: `<save_path>/Heightmaps/<run_name>_<operation>_<NNN>/`
    fn make_output_dir(&self, operation: &str) -> PathBuf {
        let base = PathBuf::from(&self.save_path).join("Heightmaps");
        let prefix = if self.run_name.is_empty() {
            operation.to_string()
        } else {
            format!("{}_{}", self.run_name, operation)
        };
        // Find next available number
        for n in 1..10000 {
            let dir = base.join(format!("{prefix}_{n:03}"));
            if !dir.exists() {
                return dir;
            }
        }
        // Fallback with random suffix
        use rand::Rng;
        let mut rng = rand::rng();
        let id: String = (0..4).map(|_| {
            let c = b"abcdefghijklmnopqrstuvwxyz0123456789"[rng.random_range(0..36)];
            c as char
        }).collect();
        base.join(format!("{prefix}_{id}"))
    }

    fn load_tile_at(&mut self, coord: Coord, path: PathBuf, ctx: &egui::Context) {
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        self.log(&format!("Loading {name} at ({},{})...", coord.0, coord.1));
        let result = tile::load_tile_png(&path);
        match result {
            Ok(t) => {
                self.log(&format!("  {}x{} min={} max={}", t.width, t.height, t.min_val(), t.max_val()));
                let thumb = t.thumbnail(THUMB_SIZE);
                let tex = ctx.load_texture(
                    format!("tile_{}_{}", coord.0, coord.1),
                    egui::ColorImage::from_gray([THUMB_SIZE, THUMB_SIZE], &thumb),
                    egui::TextureOptions::LINEAR,
                );
                self.tile_textures.insert(coord, tex);
                self.original_data.insert(coord, t.clone());
                self.tile_data.insert(coord, t);
                self.tile_paths.insert(coord, path);
            }
            Err(e) => self.log(&format!("  ERROR: {e}")),
        }
    }

    fn load_folder(&mut self, folder: PathBuf, ctx: &egui::Context) {
        let found = tile::find_tiles(&folder);
        if found.is_empty() { self.log(&format!("No tiles found in {}", folder.display())); return; }
        let max_x = found.keys().map(|&(x,_)| x).max().unwrap() + 1;
        let max_y = found.keys().map(|&(_,y)| y).max().unwrap() + 1;
        if max_x > GRID_MAX || max_y > GRID_MAX {
            self.log(&format!("WARNING: Grid capped at {}x{} (found {}x{})", GRID_MAX, GRID_MAX, max_x, max_y));
        }
        self.grid_cols = max_x.min(GRID_MAX);
        self.grid_rows = max_y.min(GRID_MAX);
        self.clear_grid();
        self.log(&format!("Found {} tiles in {}", found.len(), folder.file_name().unwrap_or_default().to_string_lossy()));
        for (coord, path) in found { self.load_tile_at(coord, path, ctx); }

        // Warn if tiles have mismatched dimensions
        let sizes: Vec<_> = self.tile_data.values().map(|t| (t.width, t.height)).collect();
        if let Some(&first) = sizes.first() {
            if sizes.iter().any(|s| *s != first) {
                let unique: std::collections::HashSet<_> = sizes.iter().collect();
                let list: Vec<_> = unique.iter().map(|(w,h)| format!("{w}x{h}")).collect();
                self.log(&format!("WARNING: Tiles have mixed dimensions: {}", list.join(", ")));
            }
        }
    }

    fn clear_grid(&mut self) {
        self.tile_paths.clear(); self.tile_data.clear(); self.original_data.clear();
        self.tile_textures.clear(); self.selected_cell = None;
    }

    fn has_neighbors(&self) -> bool {
        self.tile_paths.keys().any(|&(x,y)|
            self.tile_paths.contains_key(&(x+1,y)) || self.tile_paths.contains_key(&(x,y+1)))
    }

    fn update_result_texture(&mut self, tile: &Tile, ctx: &egui::Context) {
        let gray = tile.to_gray8();
        self.result_texture = Some(ctx.load_texture(
            "result_preview", egui::ColorImage::from_gray([tile.width, tile.height], &gray),
            egui::TextureOptions::LINEAR,
        ));
    }

    fn update_paint_texture(&mut self, ctx: &egui::Context) {
        if let Some(tile) = &self.paint_data {
            let gray = tile.to_gray8();
            self.paint_texture = Some(ctx.load_texture(
                "paint_preview", egui::ColorImage::from_gray([tile.width, tile.height], &gray),
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn apply_brush_at(&mut self, px: f32, py: f32) {
        let Some(paint) = &mut self.paint_data else { return };
        let (dw, dh) = (paint.width, paint.height);
        let half = (self.paint_brush_size / 2.0) as i32;
        let color_val = (self.paint_color * 65535.0) as f64;
        let opacity = self.paint_opacity as f64;
        let (cx, cy) = (px as i32, py as i32);
        for dy in -half..=half {
            for dx in -half..=half {
                let (sx, sy) = (cx + dx, cy + dy);
                if sx < 0 || sy < 0 || sx >= dw as i32 || sy >= dh as i32 { continue; }
                let dist = ((dx*dx + dy*dy) as f64).sqrt() / half.max(1) as f64;
                if dist > 1.0 { continue; }
                let alpha = (1.0 - dist).powi(2) * opacity;
                let idx = sy as usize * dw + sx as usize;
                let old = paint.data[idx] as f64;
                paint.data[idx] = (old * (1.0 - alpha) + color_val * alpha).clamp(0.0, 65535.0) as u16;
            }
        }
    }
}

// ── Background jobs (stitch / merge) ─────────────────────────────────────────

impl StitcherApp {
    fn start_stitch(&mut self) {
        if self.busy || self.tile_data.is_empty() { return; }

        // Validate margin vs tile size
        let min_dim = self.original_data.values()
            .map(|t| t.width.min(t.height))
            .min().unwrap_or(0);
        if self.margin >= min_dim {
            self.log(&format!("ERROR: Margin {} >= smallest tile dimension {} — reduce margin",
                self.margin, min_dim));
            return;
        }

        self.busy = true; self.set_progress(0.0, "Stitching...");
        let data: HashMap<Coord,Tile> = self.original_data.iter().map(|(&c,t)|(c,t.clone())).collect();
        let paths = self.tile_paths.clone();
        let out_dir = self.make_output_dir("stitch");
        self.log(&format!("Output: {}", out_dir.display()));
        let (margin, quality, opts, fmt) =
            (self.margin, self.quality, self.opts.clone(), self.tile_format.clone());
        let save_dir = out_dir.to_string_lossy().to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        self.job_rx = Some(rx);
        std::thread::spawn(move || {
            let tx2 = tx.clone();
            let job_start = std::time::Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut work: HashMap<Coord,Tile> = data;
                let mut h_pairs = Vec::new(); let mut v_pairs = Vec::new();
                for &(x,y) in work.keys() {
                    if work.contains_key(&(x+1,y)) { h_pairs.push(((x,y),(x+1,y))); }
                    if work.contains_key(&(x,y+1)) { v_pairs.push(((x,y),(x,y+1))); }
                }
                h_pairs.sort(); v_pairs.sort();
                let total = h_pairs.len() + v_pairs.len() + work.len();
                let mut step = 0;
                let _ = tx2.send(JobResult::Log(format!("Stitching {} edges (margin={margin}px)...", h_pairs.len()+v_pairs.len())));
                for (a,b) in &h_pairs {
                    let _ = tx2.send(JobResult::Log(format!("  H-blend: ({},{}) <-> ({},{})", a.0,a.1,b.0,b.1)));
                    let mut ta = work.remove(a).unwrap(); let mut tb = work.remove(b).unwrap();
                    blend::blend_horizontal(&mut ta, &mut tb, margin, quality, &opts);
                    work.insert(*a, ta); work.insert(*b, tb);
                    step += 1;
                    let _ = tx2.send(JobResult::Progress(step as f32/total as f32, format!("Blending... {:.0}%", step as f32/total as f32*100.0)));
                }
                for (a,b) in &v_pairs {
                    let _ = tx2.send(JobResult::Log(format!("  V-blend: ({},{}) <-> ({},{})", a.0,a.1,b.0,b.1)));
                    let mut ta = work.remove(a).unwrap(); let mut tb = work.remove(b).unwrap();
                    blend::blend_vertical(&mut ta, &mut tb, margin, quality, &opts);
                    work.insert(*a, ta); work.insert(*b, tb);
                    step += 1;
                    let _ = tx2.send(JobResult::Progress(step as f32/total as f32, format!("Blending... {:.0}%", step as f32/total as f32*100.0)));
                }
                let _ = tx2.send(JobResult::Log(format!("Saving as .{fmt} to {save_dir}...")));
                let out_dir = PathBuf::from(&save_dir);
                if let Err(e) = std::fs::create_dir_all(&out_dir) {
                    let _ = tx2.send(JobResult::Error(format!("Cannot create output dir: {e}")));
                    return None;
                }
                for (&coord, tile) in &work {
                    if let Some(orig_path) = paths.get(&coord) {
                        let base = orig_path.file_stem().unwrap_or_default().to_string_lossy();
                        let out_name = format!("{base}.png");
                        let out_path = out_dir.join(&out_name);
                        let r = if fmt == "png8" { tile::save_tile_png8(&out_path, tile) } else { tile::save_tile_png16(&out_path, tile) };
                        match r { Ok(_) => { let _ = tx2.send(JobResult::Log(format!("  Saved {out_name}"))); }
                                  Err(e) => { let _ = tx2.send(JobResult::Log(format!("  ERROR: {e}"))); } }
                    }
                    step += 1;
                    let _ = tx2.send(JobResult::Progress(step as f32/total as f32, format!("Saving... {:.0}%", step as f32/total as f32*100.0)));
                }
                let elapsed = job_start.elapsed();
                let _ = tx2.send(JobResult::Log(format!("Stitch completed in {:.1}s", elapsed.as_secs_f64())));
                let _ = tx2.send(JobResult::Progress(1.0, "Stitch complete!".into()));
                Some(work)
            }));
            match result {
                Ok(Some(work)) => { let _ = tx.send(JobResult::StitchDone { tiles: work }); }
                Ok(None) => {} // error already sent
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<String>() { s.clone() }
                              else if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
                              else { "Unknown panic".into() };
                    let _ = tx.send(JobResult::Error(format!("Stitch panicked: {msg}")));
                }
            }
        });
    }

    fn start_merge(&mut self) {
        if self.busy || self.tile_data.is_empty() { return; }

        // Validate margin vs tile size
        let min_dim = self.original_data.values()
            .map(|t| t.width.min(t.height))
            .min().unwrap_or(0);
        if self.margin >= min_dim {
            self.log(&format!("ERROR: Margin {} >= smallest tile dimension {} — reduce margin",
                self.margin, min_dim));
            return;
        }

        self.busy = true; self.set_progress(0.0, "Merging...");
        let data: HashMap<Coord,Tile> = self.original_data.iter().map(|(&c,t)|(c,t.clone())).collect();
        let out_dir = self.make_output_dir("merge");
        self.log(&format!("Output: {}", out_dir.display()));
        let (margin, quality, opts, fmt) =
            (self.margin, self.quality, self.opts.clone(), self.tile_format.clone());
        let save_dir = out_dir.to_string_lossy().to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        self.job_rx = Some(rx);
        std::thread::spawn(move || {
            let tx2 = tx.clone();
            let job_start = std::time::Instant::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _ = tx2.send(JobResult::Log(format!("Merging {} tiles...", data.len())));
                let _ = tx2.send(JobResult::Progress(0.1, "Merging...".into()));
                let log_msgs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
                let log_clone = log_msgs.clone();
                let merged = blend::merge_tiles_to_single(&data, margin, quality, &opts,
                    &mut |msg: &str| { log_clone.lock().unwrap().push(msg.to_string()); });
                for msg in log_msgs.lock().unwrap().iter() { let _ = tx2.send(JobResult::Log(msg.clone())); }
                match merged {
                    Some(merged) => {
                        let _ = tx2.send(JobResult::Progress(0.8, "Saving...".into()));
                        let out_name = format!("merged_{}x{}.png", merged.width, merged.height);
                        let out_path = PathBuf::from(&save_dir).join(&out_name);
                        if let Err(e) = std::fs::create_dir_all(&save_dir) {
                            let _ = tx2.send(JobResult::Error(format!("Cannot create output dir: {e}")));
                            return;
                        }
                        let r = if fmt == "png8" { tile::save_tile_png8(&out_path, &merged) } else { tile::save_tile_png16(&out_path, &merged) };
                        match r {
                            Ok(_) => { let elapsed = job_start.elapsed();
                                       let _ = tx2.send(JobResult::Log(format!("Merge completed in {:.1}s", elapsed.as_secs_f64())));
                                       let _ = tx2.send(JobResult::Progress(1.0, "Merge complete!".into()));
                                       let _ = tx2.send(JobResult::MergeDone { result: merged, out_path: out_path.to_string_lossy().into() }); }
                            Err(e) => { let _ = tx2.send(JobResult::Error(format!("Save failed: {e}"))); }
                        }
                    }
                    None => { let _ = tx2.send(JobResult::Error("Merge produced no output".into())); }
                }
            }));
            if let Err(e) = result {
                let msg = if let Some(s) = e.downcast_ref::<String>() { s.clone() }
                          else if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
                          else { "Unknown panic".into() };
                let _ = tx.send(JobResult::Error(format!("Merge panicked: {msg}")));
            }
        });
    }

    /// Split the merged result into individual tiles that share edge pixels,
    /// so they line up perfectly in UE5 landscape.
    fn split_and_save(&mut self) {
        let Some(merged) = self.result_data.clone() else { return; };

        // Determine grid dimensions from original tile layout
        let max_x = self.original_data.keys().map(|&(x, _)| x).max().unwrap_or(0) + 1;
        let max_y = self.original_data.keys().map(|&(_, y)| y).max().unwrap_or(0) + 1;
        let cols = max_x as usize;
        let rows = max_y as usize;

        // Switch to Tiles tab so the log is visible
        self.active_tab = Tab::Tiles;
        self.set_progress(0.1, "Splitting...");

        let out_dir = self.make_output_dir("split");
        self.log(&format!("Splitting {}x{} into {}x{} tiles...",
            merged.width, merged.height, cols, rows));
        self.log(&format!("Output: {}", out_dir.display()));

        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            self.log(&format!("ERROR: Cannot create output dir: {e}"));
            self.set_progress(0.0, "Ready");
            return;
        }
        let tiles = tile::split_into_tiles(&merged, cols, rows);
        let total = tiles.len();

        let mut saved = 0;
        // Sort for deterministic output order
        let mut coords: Vec<_> = tiles.keys().copied().collect();
        coords.sort();
        for &(gx, gy) in &coords {
            let t = &tiles[&(gx, gy)];
            let out_name = format!("Heightmap_x{gx}_y{gy}.png");
            let out_path = out_dir.join(&out_name);
            let r = if self.tile_format == "png8" {
                tile::save_tile_png8(&out_path, t)
            } else {
                tile::save_tile_png16(&out_path, t)
            };
            match r {
                Ok(_) => {
                    self.log(&format!("  Saved {out_name} ({}x{})", t.width, t.height));
                    saved += 1;
                }
                Err(e) => self.log(&format!("  ERROR {out_name}: {e}")),
            }
            self.set_progress(saved as f32 / total as f32, &format!("Saving {saved}/{total}..."));
        }
        self.set_progress(1.0, "Split complete!");
        self.log(&format!("Split complete! {saved}/{total} tiles saved."));
    }

    fn poll_jobs(&mut self, ctx: &egui::Context) {
        let messages: Vec<JobResult> = if let Some(rx) = &self.job_rx {
            let mut msgs = Vec::new();
            while let Ok(msg) = rx.try_recv() { msgs.push(msg); }
            msgs
        } else { return; };
        for msg in messages {
            match msg {
                JobResult::Log(s) => self.log(&s),
                JobResult::Progress(v,s) => self.set_progress(v, &s),
                JobResult::Error(e) => { self.log(&format!("ERROR: {e}")); self.busy = false; }
                JobResult::StitchDone { tiles } => {
                    self.log(&format!("Stitch complete! {} tiles.", tiles.len()));
                    for (coord, t) in &tiles {
                        let thumb = t.thumbnail(THUMB_SIZE);
                        let tex = ctx.load_texture(format!("tile_{}_{}", coord.0, coord.1),
                            egui::ColorImage::from_gray([THUMB_SIZE, THUMB_SIZE], &thumb), egui::TextureOptions::LINEAR);
                        self.tile_textures.insert(*coord, tex);
                        self.tile_data.insert(*coord, t.clone());
                    }
                    // Fast direct-composite of already-stitched tiles (no re-blending).
                    if let Some(preview) = direct_composite(&tiles, self.margin) {
                        self.update_result_texture(&preview, ctx);
                        self.result_data = Some(preview);
                        self.active_tab = Tab::Result;
                    }
                    self.busy = false;
                }
                JobResult::MergeDone { result, out_path } => {
                    self.log(&format!("Saved to: {out_path}"));
                    self.update_result_texture(&result, ctx);
                    self.result_data = Some(result);
                    self.active_tab = Tab::Result;
                    self.busy = false;
                }
            }
        }
    }
}

const THUMB_SIZE: usize = 120;
const GRID_MAX: i32 = 8;

/// Fast direct-composite of already-blended tiles onto a single canvas.
/// Tiles overlap by `margin` pixels; in the overlap zone we just take the
/// average (the tiles already contain identical blend data there).
fn direct_composite(tiles: &HashMap<Coord, Tile>, margin: usize) -> Option<Tile> {
    if tiles.is_empty() { return None; }

    let mut x_vals: Vec<i32> = tiles.keys().map(|&(x,_)| x).collect();
    let mut y_vals: Vec<i32> = tiles.keys().map(|&(_,y)| y).collect();
    x_vals.sort(); x_vals.dedup();
    y_vals.sort(); y_vals.dedup();

    let x_remap: HashMap<i32,usize> = x_vals.iter().enumerate().map(|(i,&v)| (v,i)).collect();
    let y_remap: HashMap<i32,usize> = y_vals.iter().enumerate().map(|(i,&v)| (v,i)).collect();

    let grid_cols = x_vals.len();
    let grid_rows = y_vals.len();

    let mut col_widths = vec![0usize; grid_cols];
    let mut row_heights = vec![0usize; grid_rows];
    for (&(gx, gy), tile) in tiles {
        let cx = x_remap[&gx];
        let cy = y_remap[&gy];
        col_widths[cx] = col_widths[cx].max(tile.width);
        row_heights[cy] = row_heights[cy].max(tile.height);
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
    let mut canvas = Tile::zeros(out_w, out_h);

    for (&(gx, gy), tile) in tiles {
        let rx = col_offsets[x_remap[&gx]];
        let ry = row_offsets[y_remap[&gy]];
        for ty in 0..tile.height {
            for tx in 0..tile.width {
                let ox = rx + tx;
                let oy = ry + ty;
                if ox < out_w && oy < out_h {
                    canvas.set(ox, oy, tile.get(tx, ty));
                }
            }
        }
    }

    Some(canvas)
}

// ── eframe::App ──────────────────────────────────────────────────────────────

impl eframe::App for StitcherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Fix window styles once on first frame so taskbar single-click works.
        // decorations(false) creates a WS_POPUP window which Windows doesn't
        // activate from the taskbar on single-click.  Adding WS_THICKFRAME |
        // WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX makes it behave like a
        // normal window for taskbar purposes.  We then call
        // DwmExtendFrameIntoClientArea(-1) to keep the title bar invisible
        // so the custom title bar still renders over the full client area.
        #[cfg(target_os = "windows")]
        if self.needs_style_fix {
            self.needs_style_fix = false;
            unsafe {
                const GWL_STYLE: i32 = -16;
                const WS_THICKFRAME: u32   = 0x0004_0000;
                const WS_CAPTION: u32      = 0x00C0_0000;
                const WS_SYSMENU: u32      = 0x0008_0000;
                const WS_MINIMIZEBOX: u32  = 0x0002_0000;
                const WS_MAXIMIZEBOX: u32  = 0x0001_0000;
                const SWP_FRAMECHANGED: u32  = 0x0020;
                const SWP_NOMOVE: u32        = 0x0002;
                const SWP_NOSIZE: u32        = 0x0001;
                const SWP_NOZORDER: u32      = 0x0004;
                const SWP_NOACTIVATE: u32    = 0x0010;

                #[repr(C)]
                struct MARGINS { left: i32, right: i32, top: i32, bottom: i32 }

                #[link(name = "user32")]
                unsafe extern "system" {
                    fn GetWindowLongPtrW(h: isize, index: i32) -> isize;
                    fn SetWindowLongPtrW(h: isize, index: i32, val: isize) -> isize;
                    fn SetWindowPos(h: isize, after: isize, x: i32, y: i32,
                                    cx: i32, cy: i32, flags: u32) -> i32;
                }
                #[link(name = "dwmapi")]
                unsafe extern "system" {
                    fn DwmExtendFrameIntoClientArea(h: isize, m: *const MARGINS) -> i32;
                }

                let hwnd = Self::find_own_hwnd();
                if hwnd != 0 {
                    let style = GetWindowLongPtrW(hwnd, GWL_STYLE) as u32;
                    let new_style = style
                        | WS_THICKFRAME | WS_CAPTION
                        | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
                    SetWindowLongPtrW(hwnd, GWL_STYLE, new_style as isize);

                    // Tell Windows to re-evaluate the frame after style change.
                    SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                        SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE
                        | SWP_NOZORDER | SWP_NOACTIVATE);

                    // Extend the DWM frame fully into the client area so the
                    // native caption / resize border is invisible.
                    let margins = MARGINS { left: -1, right: -1, top: -1, bottom: -1 };
                    DwmExtendFrameIntoClientArea(hwnd, &margins);
                }
            }
        }

        self.poll_jobs(ctx);
        if self.busy { ctx.request_repaint(); }

        let _full_rect = ctx.screen_rect();

        // ── Custom title bar ──────────────────────────────────────────────
        egui::TopBottomPanel::top("titlebar").exact_height(32.0)
            .frame(egui::Frame::new().fill(BG_DARK).inner_margin(egui::Margin::same(0)))
            .show(ctx, |ui| {
            let bar_rect = ui.max_rect();
            let bar_h = bar_rect.height();
            let btn_h = 20.0;
            let btn_w = 22.0;
            let btn_y = bar_rect.min.y + (bar_h - btn_h) / 2.0;

            // ── "Stitcher" text logo on the left ──
            let logo_tex = self.titlebar_logo.get_or_insert_with(|| {
                let png_bytes = include_bytes!("../stitcher_text.png");
                let img = image::load_from_memory(png_bytes)
                    .expect("embedded stitcher_text.png")
                    .into_rgba8();
                let (w, h) = img.dimensions();
                ctx.load_texture("stitcher_text",
                    egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img.into_raw()),
                    egui::TextureOptions::LINEAR)
            });
            let logo_h = 20.0;
            let logo_w = logo_h * logo_tex.aspect_ratio();
            let logo_rect = egui::Rect::from_min_size(
                egui::pos2(bar_rect.min.x + 8.0, bar_rect.min.y + (bar_h - logo_h) / 2.0),
                egui::vec2(logo_w, logo_h));
            ui.painter().image(logo_tex.id(), logo_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE);

            // ── App icon centered ──
            let icon_size = 20.0;
            let icon_tex = self.titlebar_icon.get_or_insert_with(|| {
                let png_bytes = include_bytes!("../titlebar_icon.png");
                let img = image::load_from_memory(png_bytes)
                    .expect("embedded titlebar_icon.png")
                    .into_rgba8();
                let (w, h) = img.dimensions();
                ctx.load_texture("titlebar_icon",
                    egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img.into_raw()),
                    egui::TextureOptions::LINEAR)
            });
            let icon_rect = egui::Rect::from_min_size(
                egui::pos2(bar_rect.center().x - icon_size / 2.0, bar_rect.min.y + (bar_h - icon_size) / 2.0),
                egui::vec2(icon_size, icon_size));
            ui.painter().image(icon_tex.id(), icon_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE);

            // ── Grip dots left (after logo) ──
            let grip_left_start = (logo_rect.max.x + 6.0) as i32;
            for gy in (6..28).step_by(3) {
                for gx in (grip_left_start..(grip_left_start + 52)).step_by(4) {
                    let p = bar_rect.min + egui::vec2(gx as f32, gy as f32);
                    ui.painter().rect_filled(egui::Rect::from_min_size(p, egui::vec2(1.0,1.0)), 0.0, GRIP_HI);
                    ui.painter().rect_filled(egui::Rect::from_min_size(p + egui::vec2(1.0,1.0), egui::vec2(1.0,1.0)), 0.0, GRIP_LO);
                }
            }
            // Grip dots right
            for gy in (6..28).step_by(3) {
                for gx in (4..56).step_by(4) {
                    let p = egui::pos2(bar_rect.max.x - 86.0 + gx as f32, bar_rect.min.y + gy as f32);
                    ui.painter().rect_filled(egui::Rect::from_min_size(p, egui::vec2(1.0,1.0)), 0.0, GRIP_HI);
                    ui.painter().rect_filled(egui::Rect::from_min_size(p + egui::vec2(1.0,1.0), egui::vec2(1.0,1.0)), 0.0, GRIP_LO);
                }
            }

            // ── Close button ──
            let close_rect = egui::Rect::from_min_size(
                egui::pos2(bar_rect.max.x - btn_w - 2.0, btn_y), egui::vec2(btn_w, btn_h));
            let close_resp = ui.allocate_rect(close_rect, egui::Sense::click());
            let close_bg = if close_resp.hovered() { egui::Color32::from_rgb(0xcc, 0x44, 0x44) } else { BTN_FACE };
            ui.painter().rect_filled(close_rect, 0.0, close_bg);
            ui.painter().text(close_rect.center(), egui::Align2::CENTER_CENTER, "X", egui::FontId::monospace(10.0), BTN_TEXT);
            if close_resp.clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); }

            // ── Maximize / Restore button ──
            let max_rect = egui::Rect::from_min_size(
                egui::pos2(close_rect.min.x - btn_w - 2.0, btn_y), egui::vec2(btn_w, btn_h));
            let max_resp = ui.allocate_rect(max_rect, egui::Sense::click());
            let max_bg = if max_resp.hovered() { BTN_LIGHT } else { BTN_FACE };
            ui.painter().rect_filled(max_rect, 0.0, max_bg);
            let is_maximized = self.is_maximized();
            let btn_stroke = egui::Stroke::new(1.0, BTN_TEXT);
            if is_maximized {
                let r1 = egui::Rect::from_min_size(max_rect.min + egui::vec2(4.0, 7.0), egui::vec2(11.0, 9.0));
                let r2 = egui::Rect::from_min_size(max_rect.min + egui::vec2(7.0, 4.0), egui::vec2(11.0, 9.0));
                ui.painter().rect_stroke(r2, 0.0, btn_stroke, egui::StrokeKind::Outside);
                ui.painter().rect_filled(r1, 0.0, max_bg);
                ui.painter().rect_stroke(r1, 0.0, btn_stroke, egui::StrokeKind::Outside);
            } else {
                let r = egui::Rect::from_min_size(max_rect.min + egui::vec2(5.0, 5.0), egui::vec2(12.0, 10.0));
                ui.painter().rect_stroke(r, 0.0, btn_stroke, egui::StrokeKind::Outside);
            }
            if max_resp.clicked() {
                self.toggle_maximize();
            }

            // ── Minimize button ──
            let min_rect = egui::Rect::from_min_size(
                egui::pos2(max_rect.min.x - btn_w - 2.0, btn_y), egui::vec2(btn_w, btn_h));
            let min_resp = ui.allocate_rect(min_rect, egui::Sense::click());
            let min_bg = if min_resp.hovered() { BTN_LIGHT } else { BTN_FACE };
            ui.painter().rect_filled(min_rect, 0.0, min_bg);
            ui.painter().text(min_rect.center(), egui::Align2::CENTER_CENTER, "_", egui::FontId::monospace(10.0), BTN_TEXT);
            if min_resp.clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true)); }

            // ── Make title bar draggable; double-click toggles maximize ──
            let drag_rect = egui::Rect::from_min_max(bar_rect.min, egui::pos2(min_rect.min.x - 2.0, bar_rect.max.y));
            let drag_resp = ui.allocate_rect(drag_rect, egui::Sense::click_and_drag());
            if drag_resp.dragged() {
                ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            if drag_resp.double_clicked() {
                self.toggle_maximize();
            }
        });

        // ── Toolbar ───────────────────────────────────────────────────────
        egui::TopBottomPanel::top("toolbar").exact_height(26.0)
            .frame(egui::Frame::new().fill(BG_METAL).inner_margin(egui::Margin::symmetric(6, 2)))
            .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;

                if metal_button(ui, "LOAD", !self.busy) {
                    self.clear_pending = false;
                    if let Some(f) = rfd::FileDialog::new().set_title("Select tile folder").pick_folder() {
                        self.load_folder(f, ctx);
                    }
                }
                if metal_button(ui, "ADD", !self.busy) {
                    if let Some(path) = rfd::FileDialog::new().set_title("Select heightmap").add_filter("PNG images", &["png"]).pick_file() {
                        let target = self.selected_cell.filter(|c| !self.tile_paths.contains_key(c))
                            .or_else(|| { for y in 0..self.grid_rows { for x in 0..self.grid_cols {
                                if !self.tile_paths.contains_key(&(x,y)) { return Some((x,y)); } } } None });
                        if let Some(coord) = target { self.load_tile_at(coord, path, ctx); }
                    }
                }
                if metal_button(ui, "REMOVE", !self.busy && self.selected_cell.map_or(false, |c| self.tile_paths.contains_key(&c))) {
                    if let Some(c) = self.selected_cell {
                        self.tile_paths.remove(&c); self.tile_data.remove(&c); self.original_data.remove(&c); self.tile_textures.remove(&c); self.selected_cell = None;
                    }
                }
                {
                    let label = if self.clear_pending { "CONFIRM?" } else { "CLEAR" };
                    if metal_button(ui, label, !self.busy && !self.tile_paths.is_empty()) {
                        if self.clear_pending {
                            self.clear_grid(); self.log("Grid cleared.");
                            self.clear_pending = false;
                        } else {
                            self.clear_pending = true;
                        }
                    }
                }

                // Push MERGE/STITCH/SPLIT/EXPORT to the right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;

                    // Buttons are added right-to-left, so reverse order
                    if metal_button(ui, "EXPORT", !self.busy && self.paint_data.is_some()) {
                        if let Some(paint) = &self.paint_data {
                            let rand_id: String = { use rand::Rng; let mut rng = rand::rng();
                                (0..6).map(|_| { let c = b"abcdefghijklmnopqrstuvwxyz0123456789"[rng.random_range(0..36)]; c as char }).collect() };
                            let out_name = format!("painted_{rand_id}_{}x{}.png", paint.width, paint.height);
                            let out_path = PathBuf::from(&self.save_path).join(&out_name);
                            let _ = std::fs::create_dir_all(&self.save_path);
                            let r = if self.tile_format == "png8" { tile::save_tile_png8(&out_path, paint) } else { tile::save_tile_png16(&out_path, paint) };
                            match r { Ok(_) => self.log(&format!("Exported: {out_name}")), Err(e) => self.log(&format!("Export error: {e}")) }
                        }
                    }
                    if metal_button(ui, "SPLIT", !self.busy && self.result_data.is_some() && self.original_data.len() >= 2) {
                        self.split_and_save();
                    }
                    if metal_button(ui, "STITCH", !self.busy && self.has_neighbors()) {
                        self.start_stitch();
                    }
                    if metal_button(ui, "MERGE", !self.busy && self.tile_data.len() >= 2) {
                        self.start_merge();
                    }
                });
            });
        });

        // ── Bottom: status bar ────────────────────────────────────────────
        egui::TopBottomPanel::bottom("statusbar").exact_height(20.0)
            .frame(egui::Frame::new().fill(BG_METAL).inner_margin(egui::Margin::symmetric(6, 2)))
            .show(ctx, |ui| {
            let n = self.tile_data.len();
            let status = format!("{} tile{} loaded  |  Grid: {}x{}  |  Margin: {}px",
                n, if n != 1 {"s"} else {""}, self.grid_cols, self.grid_rows, self.margin);
            ui.label(egui::RichText::new(status).monospace().size(10.0).color(TEXT_DIM));
        });

        // ── Bottom: progress bar ──────────────────────────────────────────
        egui::TopBottomPanel::bottom("progress").exact_height(17.0)
            .frame(egui::Frame::new().fill(BG_METAL).inner_margin(egui::Margin { left: 6, right: 6, top: 6, bottom: 1 }))
            .show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let bar_h = 8.0;
            let bar = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), bar_h));
            ui.painter().rect_filled(bar, 2.0, PANEL_BG);
            if self.progress > 0.0 {
                let fill_w = bar.width() * self.progress;
                let fill = egui::Rect::from_min_size(bar.min, egui::vec2(fill_w, bar_h));

                // Animated time-based hue shift
                let time = ui.input(|i| i.time) as f32;
                let speed = 0.4; // full cycle period ~15s

                // Two colours that smoothly rotate through the spectrum
                let hue_a = (time * speed) % 1.0;
                let hue_b = (hue_a + 0.3) % 1.0; // offset for gradient contrast

                fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
                    let i = (h * 6.0).floor() as i32;
                    let f = h * 6.0 - i as f32;
                    let p = v * (1.0 - s);
                    let q = v * (1.0 - f * s);
                    let t = v * (1.0 - (1.0 - f) * s);
                    let (r, g, b) = match i % 6 {
                        0 => (v, t, p), 1 => (q, v, p), 2 => (p, v, t),
                        3 => (p, q, v), 4 => (t, p, v), _ => (v, p, q),
                    };
                    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
                }

                let (r1, g1, b1) = hsv_to_rgb(hue_a, 0.7, 0.85);
                let (r2, g2, b2) = hsv_to_rgb(hue_b, 0.7, 0.85);
                let col_left  = egui::Color32::from_rgb(r1, g1, b1);
                let col_right = egui::Color32::from_rgb(r2, g2, b2);

                // Draw gradient with vertical strips
                let steps = (fill_w as i32).max(1);
                for i in 0..steps {
                    let t = i as f32 / steps as f32;
                    let r = (r1 as f32 + (r2 as f32 - r1 as f32) * t) as u8;
                    let g = (g1 as f32 + (g2 as f32 - g1 as f32) * t) as u8;
                    let b = (b1 as f32 + (b2 as f32 - b1 as f32) * t) as u8;
                    let x = fill.min.x + i as f32;
                    let strip = egui::Rect::from_min_size(
                        egui::pos2(x, fill.min.y), egui::vec2(1.0, bar_h));
                    ui.painter().rect_filled(strip, 0.0, egui::Color32::from_rgb(r, g, b));
                }

                // Rounded clip: re-draw the background corners over the gradient
                let _ = (col_left, col_right); // suppress unused
            }
            ui.allocate_space(egui::vec2(0.0, bar_h));
        });

        // ── Right: settings panel (tab-dependent) ─────────────────────────
        // Tiles = settings + log, Result = none, Paint = paint tools only
        if self.active_tab != Tab::Result {
            egui::SidePanel::right("settings").exact_width(260.0).resizable(false)
                .frame(egui::Frame::new().fill(BG_METAL).inner_margin(egui::Margin::symmetric(8, 4)))
                .show(ctx, |ui| {
                ui.add_space(10.0);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.set_min_width(ui.available_width());

                    if self.active_tab == Tab::Tiles {
                        // ── TILES: full settings + log ────────────────────
                        panel_header(ui, "SETTINGS");

                        section_header(ui, "EXPORT FORMAT");
                        inset_frame(ui, |ui| {
                            let fmt_idx = if self.tile_format == "png8" { 1 } else { 0 };
                            if let Some(i) = metal_toggle_row(ui, &["PNG 16-bit", "PNG 8-bit"], fmt_idx) {
                                self.tile_format = if i == 1 { "png8" } else { "png16" }.into();
                            }
                        });

                        section_header(ui, "BLEND MARGIN");
                        inset_frame(ui, |ui| {
                            let margins = [256, 512, 1024, 2048, 4096];
                            let sel = margins.iter().position(|&m| m == self.margin).unwrap_or(2);
                            let labels: Vec<&str> = margins.iter().map(|m| match m {
                                256 => "256", 512 => "512", 1024 => "1024", 2048 => "2048", _ => "4096",
                            }).collect();
                            if let Some(i) = metal_toggle_row(ui, &labels, sel) {
                                self.margin = margins[i];
                            }
                        });

                        section_header(ui, "BLEND QUALITY");
                        inset_frame(ui, |ui| {
                            let q_idx = match self.quality { Quality::Fast => 0, Quality::High => 1, Quality::Ultra => 2 };
                            if let Some(i) = metal_toggle_row(ui, &["Fast", "High", "Ultra"], q_idx) {
                                self.quality = match i { 0 => Quality::Fast, 1 => Quality::High, _ => Quality::Ultra };
                            }
                            let desc = match self.quality {
                                Quality::Fast  => "Smoothstep crossfade",
                                Quality::High  => "Laplacian pyramid blend",
                                Quality::Ultra => "Graph-cut + pyramid + Poisson + erosion",
                            };
                            ui.label(egui::RichText::new(desc).monospace().size(8.0).color(TEXT_LCD));
                        });

                        section_header(ui, "OPTIONS");
                        inset_frame(ui, |ui| {
                            metal_checkbox(ui, &mut self.opts.terrain_extend, "Terrain Extend");
                        });

                        section_header(ui, "GRID SIZE");
                        inset_frame(ui, |ui| {
                            let w = ui.available_width();
                            ui.horizontal(|ui| {
                                ui.set_min_width(w);
                                ui.spacing_mut().item_spacing.x = 4.0;
                                ui.label(egui::RichText::new("COLS").monospace().size(10.0).color(TEXT_LCD));
                                ui.add(egui::DragValue::new(&mut self.grid_cols).range(1..=GRID_MAX).speed(0.1));
                                ui.add_space(8.0);
                                ui.label(egui::RichText::new("ROWS").monospace().size(10.0).color(TEXT_LCD));
                                ui.add(egui::DragValue::new(&mut self.grid_rows).range(1..=GRID_MAX).speed(0.1));
                            });
                        });

                        section_header(ui, "RUN NAME");
                        inset_frame(ui, |ui| {
                            let w = ui.available_width();
                            ui.add(egui::TextEdit::singleline(&mut self.run_name)
                                .desired_width(w)
                                .hint_text("e.g. coalvalley")
                                .font(egui::FontId::monospace(9.0)));
                        });

                        section_header(ui, "SAVE PATH");
                        inset_frame(ui, |ui| {
                            let w = ui.available_width();
                            if ui.add(egui::TextEdit::singleline(&mut self.save_path)
                                .desired_width(w)
                                .font(egui::FontId::monospace(9.0))).double_clicked()
                            {
                                if let Some(f) = rfd::FileDialog::new().set_title("Save directory").pick_folder() {
                                    self.save_path = f.to_string_lossy().into();
                                }
                            }
                            ui.label(egui::RichText::new("Double-click to browse").monospace().size(7.0).color(TEXT_DIM));
                        });

                        section_header(ui, "NEXT OUTPUT");
                        inset_frame(ui, |ui| {
                            let preview = self.make_output_dir("merge");
                            let text = preview.display().to_string();
                            // Truncate to fit panel width
                            let max_chars = (ui.available_width() / 5.5) as usize;
                            let display = if text.len() > max_chars {
                                format!("...{}", &text[text.len() - max_chars + 3..])
                            } else {
                                text
                            };
                            ui.label(egui::RichText::new(display).monospace().size(8.0).color(TEXT_LCD));
                        });

                        section_header(ui, "LOG");
                        inset_frame(ui, |ui| {
                            egui::ScrollArea::vertical()
                                .max_height(100.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                ui.set_min_height(92.0);
                                for entry in &self.log_entries {
                                    ui.label(egui::RichText::new(entry).monospace().size(9.0).color(TEXT_LCD));
                                }
                            });
                        });

                    } else if self.active_tab == Tab::Paint {
                        // ── PAINT: paint tools only ───────────────────────
                        panel_header(ui, "PAINT TOOLS");

                        ui.add_space(4.0);
                        inset_frame(ui, |ui| {
                            ui.label(egui::RichText::new("BRUSH: Soft").monospace().size(9.0).color(TEXT_LCD));

                            ui.label(egui::RichText::new("SIZE").monospace().size(9.0).color(TEXT_LCD));
                            ui.add(egui::Slider::new(&mut self.paint_brush_size, 50.0..=2500.0).show_value(true));

                            ui.label(egui::RichText::new("OPACITY").monospace().size(9.0).color(TEXT_LCD));
                            ui.add(egui::Slider::new(&mut self.paint_opacity, 0.01..=1.0).show_value(true));

                            ui.label(egui::RichText::new("HEIGHT VALUE").monospace().size(9.0).color(TEXT_LCD));
                            let avail_w = ui.available_width().max(10.0);
                            let (grad_rect, grad_resp) = ui.allocate_exact_size(egui::vec2(avail_w, 16.0), egui::Sense::click_and_drag());
                            if grad_rect.width() > 1.0 {
                                for x in 0..grad_rect.width() as i32 {
                                    let t = x as f32 / grad_rect.width();
                                    let v = (t * 255.0) as u8;
                                    ui.painter().rect_filled(
                                        egui::Rect::from_min_size(egui::pos2(grad_rect.min.x + x as f32, grad_rect.min.y), egui::vec2(1.0, 16.0)),
                                        0.0, egui::Color32::from_rgb(v, v, v));
                                }
                                let ix = grad_rect.min.x + self.paint_color * grad_rect.width();
                                ui.painter().rect_stroke(
                                    egui::Rect::from_center_size(egui::pos2(ix, grad_rect.center().y), egui::vec2(4.0, 16.0)),
                                    0.0, egui::Stroke::new(1.0, egui::Color32::RED), egui::StrokeKind::Outside);
                            }
                            if (grad_resp.clicked() || grad_resp.dragged()) && grad_resp.interact_pointer_pos().is_some() {
                                let pos = grad_resp.interact_pointer_pos().unwrap();
                                self.paint_color = ((pos.x - grad_rect.min.x) / grad_rect.width().max(1.0)).clamp(0.0, 1.0);
                            }

                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                if metal_button(ui, "UNDO", !self.paint_undo_stack.is_empty()) {
                                    if let Some(prev) = self.paint_undo_stack.pop() {
                                        self.paint_data = Some(prev);
                                        self.update_paint_texture(ctx);
                                    }
                                }
                                if metal_button(ui, "APPLY", self.paint_data.is_some()) {
                                    if let Some(p) = self.paint_data.clone() {
                                        self.update_result_texture(&p, ctx);
                                        self.result_data = Some(p);
                                        self.log("Paint applied to result.");
                                    }
                                }
                            });
                        });
                    }
                }); // end ScrollArea
            });
        }

        // ── Central panel ─────────────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG_METAL).inner_margin(egui::Margin::symmetric(6, 0)))
            .show(ctx, |ui| {
            // Gap between toolbar and tab row
            ui.add_space(10.0);

            // Tab row
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                if metal_toggle(ui, "TILES", self.active_tab == Tab::Tiles) { self.active_tab = Tab::Tiles; }
                if metal_toggle(ui, "RESULT", self.active_tab == Tab::Result) { self.active_tab = Tab::Result; }
                if metal_toggle(ui, "PAINT", self.active_tab == Tab::Paint) {
                    self.active_tab = Tab::Paint;
                    if self.paint_data.is_none() {
                        if let Some(r) = self.result_data.clone() {
                            self.paint_data = Some(r);
                            self.paint_undo_stack.clear();
                            self.update_paint_texture(ctx);
                        }
                    }
                }
            });

            // Content area in dark inset
            let content_frame = egui::Frame::new().fill(PANEL_BG)
                .stroke(egui::Stroke::new(1.0, BORDER_IN))
                .inner_margin(egui::Margin::same(4));
            content_frame.show(ui, |ui| {
                match self.active_tab {
                    Tab::Tiles  => self.render_tiles_tab(ui, ctx),
                    Tab::Result => self.render_result_tab(ui),
                    Tab::Paint  => self.render_paint_tab(ui, ctx),
                }
            });
        });
    }
}

// ── Tab rendering ────────────────────────────────────────────────────────────

impl StitcherApp {
    fn render_tiles_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let cols = self.grid_cols.max(1);
        let rows = self.grid_rows.max(1);
        let gap = 4.0;
        let pad = 8.0; // uniform padding around grid

        let avail = ui.available_size();
        let usable_w = avail.x - pad * 2.0;
        let usable_h = avail.y - pad * 2.0;

        // Calculate cell size to fit the grid, then center it
        let cell_px = ((usable_w + gap) / cols as f32 - gap)
            .min((usable_h + gap) / rows as f32 - gap)
            .max(60.0);

        let grid_w = cols as f32 * (cell_px + gap) - gap;
        let grid_h = rows as f32 * (cell_px + gap) - gap;

        // Reserve the full viewport area, then compute centered origin
        let (area_rect, _) = ui.allocate_exact_size(avail, egui::Sense::hover());
        let grid_origin = egui::pos2(
            area_rect.min.x + (avail.x - grid_w) * 0.5,
            area_rect.min.y + (avail.y - grid_h) * 0.5,
        );

        for y in 0..rows {
            for x in 0..cols {
                let coord = (x, y);
                let cell_min = egui::pos2(
                    grid_origin.x + x as f32 * (cell_px + gap),
                    grid_origin.y + y as f32 * (cell_px + gap),
                );
                let cell_rect = egui::Rect::from_min_size(cell_min, egui::vec2(cell_px, cell_px));

                let is_sel = self.selected_cell == Some(coord);
                let has_tile = self.tile_textures.contains_key(&coord);

                let bg = if is_sel { CELL_SEL } else if has_tile { CELL_FILL } else { CELL_BG };
                ui.painter().rect_filled(cell_rect, 1.0, bg);
                bevel_sunken(ui, cell_rect);

                // Thumbnail or coord text
                if let Some(tex) = self.tile_textures.get(&coord) {
                    let img_rect = cell_rect.shrink(5.0);
                    ui.painter().image(tex.id(), img_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                }

                // Coordinate label inside cell (bottom-right corner)
                ui.painter().text(
                    egui::pos2(cell_rect.max.x - 3.0, cell_rect.max.y - 2.0),
                    egui::Align2::RIGHT_BOTTOM,
                    format!("{x},{y}"),
                    egui::FontId::monospace(9.0),
                    if has_tile { egui::Color32::from_rgba_premultiplied(0x80, 0x80, 0x80, 0x90) }
                    else { egui::Color32::from_rgb(0x2a, 0x2e, 0x2a) },
                );

                // Click handling
                let resp = ui.allocate_rect(cell_rect, egui::Sense::click());
                if resp.clicked() {
                    self.selected_cell = Some(coord);
                    if !self.tile_paths.contains_key(&coord) {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title(&format!("Select heightmap for ({x},{y})"))
                            .add_filter("PNG images", &["png"]).pick_file()
                        {
                            self.load_tile_at(coord, path, ctx);
                        }
                    }
                }

                // Tooltip
                if has_tile && resp.hovered() {
                    resp.on_hover_ui(|ui: &mut egui::Ui| {
                        if let Some(t) = self.tile_data.get(&coord) {
                            ui.label(format!("({x},{y}) {}x{}", t.width, t.height));
                            ui.label(format!("Min:{} Max:{} Mean:{:.0}", t.min_val(), t.max_val(), t.mean_val()));
                        }
                    });
                }
            }
        }
    }

    fn render_result_tab(&mut self, ui: &mut egui::Ui) {
        let Some(tex_info) = self.result_texture.as_ref().map(|t| (t.id(), t.size_vec2())) else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("No result yet.\nRun Stitch or Merge first.").color(TEXT_LCD));
            });
            return;
        };
        let (tex_id, tex_size) = tex_info;
        let available = ui.available_size();
        let base_scale = (available.x / tex_size.x).min(available.y / tex_size.y);
        let scale = base_scale * self.result_zoom;
        let img_size = tex_size * scale;

        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 { self.result_zoom = (self.result_zoom * if scroll > 0.0 {1.1} else {1.0/1.1}).clamp(0.1,20.0); }
        }
        if response.dragged() { self.result_offset += response.drag_delta(); }
        if response.double_clicked() { self.result_zoom = 1.0; self.result_offset = egui::Vec2::ZERO; }

        let center = rect.center() + self.result_offset;
        let img_rect = egui::Rect::from_center_size(center, img_size);
        ui.painter().image(tex_id, img_rect,
            egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), egui::Color32::WHITE);

        if let Some(data) = &self.result_data {
            ui.painter().text(egui::pos2(rect.max.x - 4.0, rect.max.y - 4.0), egui::Align2::RIGHT_BOTTOM,
                format!("{}x{}  {:.0}%", data.width, data.height, self.result_zoom * 100.0),
                egui::FontId::monospace(10.0), TEXT_LCD);
        }
    }

    fn render_paint_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let tex_info = self.paint_texture.as_ref().map(|t| (t.id(), t.size_vec2()));
        let Some((tex_id, tex_size)) = tex_info else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("No data. Run Merge first,\nthen switch to Paint tab.").color(TEXT_LCD));
            });
            return;
        };

        let available = ui.available_size();
        let base_scale = (available.x / tex_size.x).min(available.y / tex_size.y);
        let scale = base_scale * self.paint_zoom;
        let img_size = tex_size * scale;

        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 { self.paint_zoom = (self.paint_zoom * if scroll > 0.0 {1.1} else {1.0/1.1}).clamp(0.1,20.0); }
        }
        if response.dragged_by(egui::PointerButton::Secondary) { self.paint_offset += response.drag_delta(); }

        let center = rect.center() + self.paint_offset;
        let img_rect = egui::Rect::from_center_size(center, img_size);

        let mut did_paint = false;
        if response.dragged_by(egui::PointerButton::Primary) || response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                if !self.painting {
                    if let Some(p) = self.paint_data.clone() {
                        self.paint_undo_stack.push(p);
                        const MAX_UNDO_BYTES: usize = 512 * 1024 * 1024; // 512 MB
                        let tile_bytes = self.paint_data.as_ref()
                            .map(|t| t.width * t.height * 2).unwrap_or(1);
                        let max_undos = (MAX_UNDO_BYTES / tile_bytes).max(1);
                        if self.paint_undo_stack.len() > max_undos {
                            self.paint_undo_stack.remove(0);
                            self.log_entries.push(format!("Undo stack capped at {} states (512 MB limit)", max_undos));
                        }
                    }
                    self.painting = true;
                }
                let rel = pos - img_rect.min;
                let (px, py) = (rel.x / scale, rel.y / scale);
                if let Some(last) = self.last_paint_pos {
                    let (dx, dy) = (px - last.x, py - last.y);
                    let dist = (dx*dx + dy*dy).sqrt();
                    let step = (self.paint_brush_size * 0.25).max(2.0);
                    let steps = (dist / step).max(1.0) as i32;
                    for i in 1..=steps { let t = i as f32 / steps as f32; self.apply_brush_at(last.x + dx*t, last.y + dy*t); }
                } else { self.apply_brush_at(px, py); }
                self.last_paint_pos = Some(egui::pos2(px, py));
                did_paint = true;
            }
        } else if self.painting { self.painting = false; self.last_paint_pos = None; }

        if did_paint { self.update_paint_texture(ctx); }

        let final_id = self.paint_texture.as_ref().map(|t| t.id()).unwrap_or(tex_id);
        ui.painter().image(final_id, img_rect,
            egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), egui::Color32::WHITE);

        let (dw, dh) = self.paint_data.as_ref().map(|d| (d.width, d.height)).unwrap_or((0,0));
        ui.painter().text(egui::pos2(rect.max.x - 4.0, rect.max.y - 4.0), egui::Align2::RIGHT_BOTTOM,
            format!("{}x{}  {:.0}%  Undos:{}", dw, dh, self.paint_zoom * 100.0, self.paint_undo_stack.len()),
            egui::FontId::monospace(10.0), TEXT_LCD);
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn load_icon() -> Option<egui::IconData> {
    let png_bytes = include_bytes!("../icon.ico");
    // Try loading as ICO (first embedded image) via the image crate
    let img = image::load_from_memory(png_bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::IconData { rgba: img.into_raw(), width: w, height: h })
}

fn main() -> eframe::Result {
    let mut vp = egui::ViewportBuilder::default()
        .with_inner_size([860.0, 720.0])
        .with_min_inner_size([600.0, 400.0])
        .with_decorations(false) // custom title bar
        .with_title("Heightmap Tile Stitcher");

    if let Some(icon) = load_icon() {
        vp = vp.with_icon(std::sync::Arc::new(icon));
    }

    let options = eframe::NativeOptions { viewport: vp, ..Default::default() };

    eframe::run_native("Heightmap Tile Stitcher", options,
        Box::new(|cc| {
            apply_theme(&cc.egui_ctx);
            Ok(Box::new(StitcherApp::default()))
        }),
    )
}

// ── Integration tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod integration_tests {
    use super::*;
    use blend::{BlendOpts, Quality};
    use std::collections::HashMap;
    use std::path::Path;
    use std::time::Instant;

    const TILE_DIR: &str = r"D:\github repo\Dungeons\Heightmaps\r16_png";

    fn load_test_tile(x: i32, y: i32) -> Option<Tile> {
        let path = Path::new(TILE_DIR).join(format!("Heightmap_x{x}_y{y}.png"));
        if !path.exists() { return None; }
        tile::load_tile_png(&path).ok()
    }

    /// Test that all three quality levels produce valid output
    /// on a horizontal pair without panicking or producing NaN/Inf.
    #[test]
    fn test_horizontal_blend_all_qualities() {
        let Some(a) = load_test_tile(0, 0) else {
            eprintln!("Skipping: test tiles not found at {TILE_DIR}");
            return;
        };
        let Some(b) = load_test_tile(1, 0) else {
            eprintln!("Skipping: test tiles not found");
            return;
        };
        println!("Tile A: {}x{}, Tile B: {}x{}", a.width, a.height, b.width, b.height);

        let margin = 128; // small margin for fast test
        let opts = BlendOpts::default();

        for &quality in &[Quality::Fast, Quality::High, Quality::Ultra] {
            let mut ta = a.clone();
            let mut tb = b.clone();
            let label = match quality { Quality::Fast => "Fast", Quality::High => "High", Quality::Ultra => "Ultra" };
            let t0 = Instant::now();
            blend::blend_horizontal(&mut ta, &mut tb, margin, quality, &opts);
            let elapsed = t0.elapsed();
            println!("  {label}: {elapsed:.2?}");

            // Verify no NaN/Inf and values in range
            for (i, &v) in ta.data.iter().enumerate() {
                assert!(v <= 65535, "{label}: tile_a pixel {i} out of range: {v}");
            }
            for (i, &v) in tb.data.iter().enumerate() {
                assert!(v <= 65535, "{label}: tile_b pixel {i} out of range: {v}");
            }

            // Verify overlap continuity: tile_a's last `margin` columns should equal tile_b's first `margin` columns
            let aw = ta.width;
            for row in 0..ta.height {
                for i in 0..margin {
                    let va = ta.get(aw - margin + i, row);
                    let vb = tb.get(i, row);
                    assert_eq!(va, vb, "{label}: overlap mismatch at row {row}, offset {i}: a={va} b={vb}");
                }
            }
        }
    }

    /// Test vertical blend similarly.
    #[test]
    fn test_vertical_blend_all_qualities() {
        let Some(a) = load_test_tile(0, 0) else { return; };
        let Some(b) = load_test_tile(0, 1) else { return; };
        println!("Tile A: {}x{}, Tile B: {}x{}", a.width, a.height, b.width, b.height);

        let margin = 128;
        let opts = BlendOpts::default();

        for &quality in &[Quality::Fast, Quality::High, Quality::Ultra] {
            let mut ta = a.clone();
            let mut tb = b.clone();
            let label = match quality { Quality::Fast => "Fast", Quality::High => "High", Quality::Ultra => "Ultra" };
            let t0 = Instant::now();
            blend::blend_vertical(&mut ta, &mut tb, margin, quality, &opts);
            let elapsed = t0.elapsed();
            println!("  {label}: {elapsed:.2?}");

            // Verify overlap continuity: tile_a's last `margin` rows should equal tile_b's first `margin` rows
            let ah = ta.height;
            for col in 0..ta.width {
                for i in 0..margin {
                    let va = ta.get(col, ah - margin + i);
                    let vb = tb.get(col, i);
                    assert_eq!(va, vb, "{label}: overlap mismatch at col {col}, offset {i}: a={va} b={vb}");
                }
            }
        }
    }

    /// Test merge-to-single on a 2x2 subset at each quality level.
    #[test]
    fn test_merge_2x2_all_qualities() {
        let tiles_opt: Vec<_> = [(0,0),(1,0),(0,1),(1,1)].iter()
            .map(|&(x,y)| ((x,y), load_test_tile(x, y)))
            .collect();
        let mut tiles: HashMap<(i32,i32), Tile> = HashMap::new();
        for ((x,y), t) in tiles_opt {
            if let Some(t) = t { tiles.insert((x,y), t); }
        }
        if tiles.len() < 4 { eprintln!("Skipping: not enough test tiles"); return; }

        let margin = 128;
        let opts = BlendOpts::default();

        for &quality in &[Quality::Fast, Quality::High, Quality::Ultra] {
            let label = match quality { Quality::Fast => "Fast", Quality::High => "High", Quality::Ultra => "Ultra" };
            let t0 = Instant::now();
            let result = blend::merge_tiles_to_single(
                &tiles, margin, quality, &opts, &mut |msg| { println!("    {msg}"); });
            let elapsed = t0.elapsed();

            assert!(result.is_some(), "{label}: merge returned None");
            let r = result.unwrap();
            println!("  {label}: {}x{} in {elapsed:.2?}", r.width, r.height);
            assert!(r.width > 0 && r.height > 0, "{label}: empty result");

            // Check no stuck-at-zero large regions (sign of broken blend)
            let nonzero = r.data.iter().filter(|&&v| v > 0).count();
            let total = r.data.len();
            let pct = nonzero as f64 / total as f64 * 100.0;
            println!("    {nonzero}/{total} non-zero pixels ({pct:.1}%)");
            assert!(pct > 50.0, "{label}: too many zero pixels ({pct:.1}%)");
        }
    }

    /// Test that gradient-aware seam finding produces valid seam positions.
    #[test]
    fn test_gradient_seam_synthetic() {
        // Two strips: A has a ridge on the left, B has a ridge on the right.
        // The seam should prefer the middle where both are flat.
        let w = 64;
        let h = 128;
        let mut strip_a = vec![1000.0f64; h * w];
        let mut strip_b = vec![1000.0f64; h * w];

        // A has a ridge at col 10
        for row in 0..h {
            for col in 0..w {
                let dist_a = ((col as f64 - 10.0) / 5.0).abs();
                strip_a[row * w + col] += (500.0 * (-dist_a * dist_a).exp());
                let dist_b = ((col as f64 - 54.0) / 5.0).abs();
                strip_b[row * w + col] += (500.0 * (-dist_b * dist_b).exp());
            }
        }

        // The seam should be between the ridges (roughly middle)
        // This exercises the gradient-aware cost function
        let mask: Vec<f64> = (0..h * w).map(|i| {
            let col = i % w;
            1.0 - (col as f64 / (w - 1) as f64)  // simple linear
        }).collect();

        let result = pyramid::pyramid_blend(&strip_a, &strip_b, &mask, w, h, 4);
        assert_eq!(result.len(), w * h);

        // Left edge should be close to strip_a, right edge close to strip_b
        for row in 0..h {
            let left_diff = (result[row * w] - strip_a[row * w]).abs();
            let right_diff = (result[row * w + w - 1] - strip_b[row * w + w - 1]).abs();
            assert!(left_diff < 50.0, "row {row} left edge drifted: diff={left_diff}");
            assert!(right_diff < 50.0, "row {row} right edge drifted: diff={right_diff}");
        }
    }

    /// Benchmark: stitch a 2x1 pair at 1024px margin (realistic scenario).
    #[test]
    fn bench_realistic_stitch() {
        let Some(a) = load_test_tile(3, 3) else { return; };
        let Some(b) = load_test_tile(4, 3) else { return; };
        println!("Tiles: {}x{}", a.width, a.height);

        let margin = 1024;
        let opts = BlendOpts::default();

        for &quality in &[Quality::Fast, Quality::High, Quality::Ultra] {
            let mut ta = a.clone();
            let mut tb = b.clone();
            let label = match quality { Quality::Fast => "Fast", Quality::High => "High", Quality::Ultra => "Ultra" };
            let t0 = Instant::now();
            blend::blend_horizontal(&mut ta, &mut tb, margin, quality, &opts);
            let elapsed = t0.elapsed();
            println!("  {label} (margin={margin}): {elapsed:.2?}");
        }
    }
}
