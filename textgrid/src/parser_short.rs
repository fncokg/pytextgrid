//! Parser for short-format TextGrid files.
//!
//! This module provides functionality to parse TextGrid files in the short format,
//! which uses a more compact representation without explicit key-value pairs.

use crate::textgrid::*;
use crate::utils::{parse_float, parse_str, parse_uint};
use std::io::Result;

/// Parses a single tier from the short format lines.
///
/// # Arguments
///
/// * `lines` - A slice of all lines in the file
/// * `start_index` - The index where the tier data starts
///
/// # Returns
///
/// Returns a tuple containing:
/// * The parsed `Tier`
/// * The index of the next line after this tier's data
///
/// # Panics
///
/// Panics if an unknown tier class is encountered.
#[inline]
fn parse_tier(lines: &[&str], start_index: usize) -> (Tier, usize) {
    let mut tier = Tier::new();
    tier.interval_tier = match lines[start_index].trim_matches('"') {
        "IntervalTier" => true,
        "TextTier" => false,
        _ => {
            panic!("Unknown tier class: {}", lines[start_index]);
        }
    };
    tier.name = parse_str(lines[start_index + 1]);
    tier.tmin = parse_float(lines[start_index + 2]);
    tier.tmax = parse_float(lines[start_index + 3]);
    tier.size = parse_uint(lines[start_index + 4]);
    let mut cursor = start_index + 5;
    for _ in 0..tier.size {
        let item: Item;
        if tier.interval_tier {
            item = Item {
                tmin: parse_float(lines[cursor]),
                tmax: parse_float(lines[cursor + 1]),
                label: parse_str(lines[cursor + 2]),
            };
            cursor += 3;
        } else {
            let number = parse_float(lines[cursor]);
            item = Item {
                tmin: number,
                tmax: number,
                label: parse_str(lines[cursor + 1]),
            };
            cursor += 2;
        }
        tier.items.push(item);
    }
    (tier, cursor)
}

/// Reads and parses a TextGrid file in short format.
///
/// # Arguments
///
/// * `fname` - The path to the TextGrid file
/// * `strict` - Whether to perform strict validation on the parsed data
///
/// # Returns
///
/// Returns a `Result` containing the parsed `TextGrid` on success, or an error on failure.
///
/// # Errors
///
/// Returns an error if:
/// * The file cannot be read
/// * The file content is invalid
/// * Validation fails (when `strict` is true)
///
/// # Examples
///
/// ```no_run
/// use textgrid::read_from_file;
///
/// // Read using explicit "short" format specifier
/// let tg = read_from_file("example.TextGrid", false, "short").unwrap();
/// println!("Loaded TextGrid with {} tiers", tg.tiers.len());
/// ```
pub(crate) fn read_from_file_short(fname: &str, strict: bool) -> Result<TextGrid> {
    let content = std::fs::read_to_string(fname).unwrap();
    let mut tg = TextGrid::new();
    tg.name = std::path::Path::new(fname)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let lines: Vec<&str> = content.lines().map(|l| l.trim()).collect();

    tg.tmin = parse_float(lines[3]);
    tg.tmax = parse_float(lines[4]);
    tg.size = parse_uint(lines[6]);

    let mut cursor = 7;
    for _ in 0..tg.size {
        let (tier, next_cursor) = parse_tier(&lines, cursor);
        tg.tiers.push(tier);
        cursor = next_cursor;
    }

    if strict {
        tg.assert_valid()?;
    }
    Ok(tg)
}
