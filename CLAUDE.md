# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heightmap Tile Stitcher — a Python/Tkinter GUI tool for blending and merging independently-created heightmap tiles, targeting terrain workflows (e.g., Unreal Engine landscapes).

## Running the Application

```bash
pip install numpy Pillow
python Stitcher_V1.py
```

The main source file is `Stitcher_V1.py` (single-file application, ~3300 lines). Tkinter is required (bundled with Python). Logs are written to `stitch_log.txt` in the script's directory.

## No Build System or Tests

There is no build step, no package manager config, no CI/CD, and no automated tests. The app is run directly as a Python script and tested manually through the GUI.

## Architecture

### Single-File Structure

The entire application lives in one Python file, organized in these sections (top to bottom):

1. **File I/O** — `load_tile_png`, `save_tile_png`, `load_tile_r16`, `save_tile_r16`, `find_tiles`. Supports 16-bit grayscale PNG and Unreal Engine R16 raw format.
2. **Low-level blending primitives** — seam finding (`_find_optimal_seam`), weight functions, erosion algorithms (hydraulic/thermal), Gaussian smoothing, Poisson solving.
3. **Height equalization** — `_deep_height_equalize_h/v`, `_feature_spill_h/v` for matching heights across tile boundaries.
4. **Advanced blending** — Laplacian pyramid blending, Poisson blending.
5. **High-level blend functions** — `blend_horizontal`, `blend_vertical` with three quality tiers (fast/high/ultra).
6. **Merge pipeline** — `merge_tiles_to_single` orchestrates grid compaction, height equalization, global height matching via BFS, blending, and final composition.
7. **UI constants** — color scheme (retro Winamp/Trilithium aesthetic), fonts, grid dimensions.
8. **GUI classes** — `StitcherApp(tk.Toplevel)` is the main controller; `TilePlacementDialog` is a modal for manual grid positioning.
9. **Entry point** — dual-window Tkinter setup (invisible root + visible app window).

### Key Data Model

- Tiles stored as `dict[(x: int, y: int)] -> numpy.ndarray[uint16]`
- UI state managed via Tkinter `StringVar`/`IntVar`/`BooleanVar`
- Tab-based interface: LOAD, TILES, RESULT, PAINT

### Two Export Modes

1. **Stitch & Export Tiles** — blend seams, save as separate tile files
2. **Merge to Single File** — combine all tiles into one heightmap

### Three Blend Quality Levels

- **Fast**: linear fade + crossfade
- **High** (default): deep height equalization + feature spill + Poisson blending
- **Ultra**: graph-cut seam finding + Laplacian pyramid blending + erosion passes

### Threading Pattern

Long-running operations (stitching/merging) run on daemon worker threads, posting UI updates back via `root.after()`.

### Naming Conventions

- `_` prefix for private/internal functions (e.g., `_deep_height_equalize_h`)
- PascalCase for classes, UPPER_SNAKE_CASE for constants
- R16 tile sizes auto-detected from known UE5 landscape dimensions (8129, 4033, 2017, 1009, 505, 253)
