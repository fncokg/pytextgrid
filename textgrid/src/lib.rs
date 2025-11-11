//! A library for reading and writing Praat TextGrid files.
//!
//! This library provides functionality to parse TextGrid files in both long and short formats,
//! convert them to various data structures, and write them back to files.

mod converter;
mod parser_long;
mod parser_short;
mod textgrid;
mod utils;
mod writer;

pub use textgrid::{Item, TextGrid, Tier};

use parser_long::read_from_file_long;
use parser_short::read_from_file_short;
use std::io::Result;
use utils::fast_map;

/// Reads a TextGrid file from the specified path.
///
/// # Arguments
///
/// * `fname` - The path to the TextGrid file
/// * `strict` - Whether to perform strict validation on the parsed data
/// * `file_type` - The format of the file: "long", "short", or "auto" to detect automatically
///
/// # Returns
///
/// Returns a `Result` containing the parsed `TextGrid` on success, or an error on failure.
///
/// # Panics
///
/// Panics if an unknown file type is provided.
///
/// # Examples
///
/// ```no_run
/// use textgrid::read_from_file;
///
/// // Auto-detect format
/// let tg = read_from_file("example.TextGrid", true, "auto").unwrap();
/// println!("TextGrid has {} tiers", tg.tiers.len());
///
/// // Explicitly specify long format
/// let tg_long = read_from_file("example.TextGrid", false, "long").unwrap();
/// ```
pub fn read_from_file(fname: &str, strict: bool, file_type: &str) -> Result<TextGrid> {
    match file_type {
        "long" => read_from_file_long(fname, strict),
        "short" => read_from_file_short(fname, strict),
        "auto" => {
            let content = std::fs::read_to_string(fname)?;
            if content.contains("item []") {
                read_from_file_long(fname, strict)
            } else {
                read_from_file_short(fname, strict)
            }
        }
        _ => panic!("Unknown file type: {}", file_type),
    }
}

/// Reads multiple TextGrid files and converts them to data format in parallel.
///
/// # Arguments
///
/// * `fnames` - A vector of file paths to TextGrid files
/// * `strict` - Whether to perform strict validation on the parsed data
/// * `file_type` - The format of the files: "long", "short", or "auto"
///
/// # Returns
///
/// Returns a vector of tuples, where each tuple contains:
/// * `tmin` - The minimum time of the TextGrid
/// * `tmax` - The maximum time of the TextGrid
/// * A vector of tier data, where each tier contains:
///   - Tier name (String)
///   - Whether it's an interval tier (bool)
///   - A vector of items (tmin, tmax, label)
///
/// Files that fail to parse return `(0.0, 0.0, Vec::new())`.
///
/// # Examples
///
/// ```no_run
/// use textgrid::files_to_data;
///
/// let files = vec![
///     String::from("file1.TextGrid"),
///     String::from("file2.TextGrid"),
/// ];
/// let data = files_to_data(&files, true, "auto");
/// for (tmin, tmax, tiers) in data {
///     println!("TextGrid: {:.2} - {:.2}, {} tiers", tmin, tmax, tiers.len());
/// }
/// ```
pub fn files_to_data(
    fnames: &Vec<String>,
    strict: bool,
    file_type: &str,
) -> Vec<(f64, f64, Vec<(String, bool, Vec<(f64, f64, String)>)>)> {
    let map_fun = |tgt_fname: &String| {
        let tgt_result = read_from_file(tgt_fname, strict, file_type);
        match tgt_result {
            Ok(tgt) => tgt.to_data(),
            Err(_) => (0.0, 0.0, Vec::new()),
        }
    };
    let datas: Vec<(f64, f64, Vec<(String, bool, Vec<(f64, f64, String)>)>)> =
        fast_map(fnames, map_fun, 20);
    datas
}

/// Reads multiple TextGrid files and converts them to vector format in parallel.
///
/// # Arguments
///
/// * `fnames` - A vector of file paths to TextGrid files
/// * `strict` - Whether to perform strict validation on the parsed data
/// * `file_type` - The format of the files: "long", "short", or "auto"
///
/// # Returns
///
/// Returns a vector of tuples, where each tuple contains:
/// * A vector of tmin values (`Vec<f64>`)
/// * A vector of tmax values (`Vec<f64>`)
/// * A vector of labels (`Vec<String>`)
/// * A vector of tier names (`Vec<String>`)
/// * A vector of interval tier flags (`Vec<bool>`)
///
/// Files that fail to parse return empty vectors.
///
/// # Examples
///
/// ```no_run
/// use textgrid::files_to_vectors;
///
/// let files = vec![String::from("example.TextGrid")];
/// let vectors = files_to_vectors(&files, false, "auto");
/// for (tmins, tmaxs, labels, tier_names, is_intervals) in vectors {
///     println!("Found {} items", tmins.len());
/// }
/// ```
pub fn files_to_vectors(
    fnames: &Vec<String>,
    strict: bool,
    file_type: &str,
) -> Vec<(Vec<f64>, Vec<f64>, Vec<String>, Vec<String>, Vec<bool>)> {
    let map_fun = |tgt_fname: &String| {
        let tgt_result = read_from_file(tgt_fname, strict, file_type);
        match tgt_result {
            Ok(tgt) => tgt.to_vectors(),
            Err(_) => (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        }
    };
    let vectors: Vec<(Vec<f64>, Vec<f64>, Vec<String>, Vec<String>, Vec<bool>)> =
        fast_map(fnames, map_fun, 20);
    vectors
}
