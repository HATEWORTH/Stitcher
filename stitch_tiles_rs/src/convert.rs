//! Convert all .r16 heightmaps in a directory to 16-bit PNGs.

mod tile;

use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: convert <r16_directory> [output_directory]");
        std::process::exit(1);
    }));
    let out_dir = std::env::args().nth(2).map(PathBuf::from)
        .unwrap_or_else(|| dir.join("png"));

    std::fs::create_dir_all(&out_dir).expect("Cannot create output directory");

    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("Cannot read directory")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().to_lowercase().ends_with(".r16"))
        .collect();
    files.sort_by_key(|e| e.file_name());

    println!("Converting {} .r16 files to 16-bit PNG...", files.len());
    println!("Output: {}", out_dir.display());

    for entry in &files {
        let name = entry.file_name().to_string_lossy().to_string();
        print!("  {name}...");

        match tile::load_tile_r16(&entry.path(), None) {
            Ok(t) => {
                let png_name = name.replace(".r16", ".png").replace(".R16", ".png");
                let out_path = out_dir.join(&png_name);
                match tile::save_tile_png16(&out_path, &t) {
                    Ok(_) => println!(" {}x{} OK", t.width, t.height),
                    Err(e) => println!(" SAVE ERROR: {e}"),
                }
            }
            Err(e) => println!(" LOAD ERROR: {e}"),
        }
    }

    println!("\nDone!");
}
