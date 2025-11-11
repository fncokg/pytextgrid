//! Parser for long-format TextGrid files.
//!
//! This module provides functionality to parse TextGrid files in the long format,
//! which uses explicit key-value pairs with equals signs.

use crate::textgrid::*;
use crate::utils::{parse_float, parse_str, parse_uint};
use std::io::Result;

/// Represents the current parsing state when reading a long-format TextGrid file.
enum State {
    /// Parsing the TextGrid header
    Header,
    /// Parsing an individual item (interval or point)
    Item,
    /// Parsing a tier definition
    Tier,
    /// Parsing the tier list declaration
    TierList,
}

/// Parses a key-value pair from a line.
///
/// # Arguments
///
/// * `line` - The line to parse
///
/// # Returns
///
/// Returns `Some((key, value))` if the line contains a key-value pair separated by '=',
/// otherwise returns `None`.
#[inline]
fn parse_kv(line: &str) -> Option<(&str, &str)> {
    let parts: Vec<&str> = line.splitn(2, '=').collect();
    if parts.len() == 2 {
        Some((parts[0].trim(), parts[1].trim()))
    } else {
        None
    }
}

/// Parses a key-value pair and updates an Item accordingly.
///
/// # Arguments
///
/// * `line` - The line to parse
/// * `item` - The item to update with parsed values
#[inline]
fn parse_item_kv(line: &str, item: &mut Item) {
    if let Some((key, value)) = parse_kv(line) {
        match key {
            "xmin" => item.tmin = parse_float(value),
            "xmax" => item.tmax = parse_float(value),
            "text" => item.label = parse_str(value),
            "number" => {
                item.tmin = parse_float(value);
                item.tmax = item.tmin;
            }
            "mark" => item.label = parse_str(value),
            _ => {}
        }
    }
}

/// Parses a key-value pair and updates a Tier accordingly.
///
/// # Arguments
///
/// * `line` - The line to parse
/// * `tier` - The tier to update with parsed values
#[inline]
fn parse_tier_kv(line: &str, tier: &mut Tier) {
    if let Some((key, value)) = parse_kv(line) {
        match key {
            "class" => match value.trim_matches('"') {
                "IntervalTier" => tier.interval_tier = true,
                "TextTier" => tier.interval_tier = false,
                _ => {
                    panic!("Unknown tier class: {}", value);
                }
            },
            "name" => tier.name = parse_str(value),
            "intervals: size" => tier.size = parse_uint(value),
            "points: size" => tier.size = parse_uint(value),
            "xmin" => tier.tmin = parse_float(value),
            "xmax" => tier.tmax = parse_float(value),
            _ => {}
        }
    }
}

/// Parses a key-value pair and updates a TextGrid accordingly.
///
/// # Arguments
///
/// * `line` - The line to parse
/// * `tg` - The TextGrid to update with parsed values
#[inline]
fn parse_tg_kv(line: &str, tg: &mut TextGrid) {
    if let Some((key, value)) = parse_kv(line) {
        match key {
            "xmin" => tg.tmin = parse_float(value),
            "xmax" => tg.tmax = parse_float(value),
            "size" => tg.size = parse_uint(value),
            _ => {}
        }
    }
}

/// Reads and parses a TextGrid file in long format.
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
/// // Read using explicit "long" format specifier
/// let tg = read_from_file("example.TextGrid", true, "long").unwrap();
/// println!("Loaded TextGrid with {} tiers", tg.tiers.len());
/// ```
pub(crate) fn read_from_file_long(fname: &str, strict: bool) -> Result<TextGrid> {
    let content = std::fs::read_to_string(fname).unwrap();
    let mut tg = TextGrid::new();
    tg.name = std::path::Path::new(fname)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let mut state = State::Header;
    for line in content.lines().map(|l| l.trim()) {
        if line.starts_with("item []") {
            state = State::TierList;
        } else if line.starts_with("item [") {
            state = State::Tier;
            tg.add_empty_tier();
        } else if line.starts_with("intervals [") || line.starts_with("points [") {
            state = State::Item;
            tg.tiers.last_mut().unwrap().add_empty_item();
        } else {
            // parse key-value pairs
            match state {
                State::Header => parse_tg_kv(line, &mut tg),
                State::Tier => parse_tier_kv(line, &mut tg.tiers.last_mut().unwrap()),
                State::Item => parse_item_kv(
                    line,
                    &mut tg.tiers.last_mut().unwrap().items.last_mut().unwrap(),
                ),
                // TierList has no key-value pairs
                State::TierList => (),
            }
        }
    }
    if strict {
        tg.assert_valid()?;
    }
    Ok(tg)
}
