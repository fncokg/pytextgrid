//! Utility functions for parsing and parallel processing.
//!
//! This module provides helper functions for:
//! * Parallel mapping operations with automatic parallelization based on data size
//! * Parsing string values to various types

use num_cpus;
use rayon::prelude::*;

// Parallel mapping helper functions

/// Maps a function over a vector, using parallel processing for large collections.
///
/// Automatically switches between sequential and parallel processing based on
/// the number of items and available CPU cores. Used internally throughout the library.
#[inline]
pub(crate) fn fast_map<T, F, R>(items: &Vec<T>, func: F, min_len: usize) -> Vec<R>
where
    F: Fn(&T) -> R + Sync + Send,
    R: Send,
    T: Sync,
{
    match items.len() >= (num_cpus::get() * min_len / 2) {
        false => items.iter().map(func).collect::<Vec<R>>(),
        true => items
            .par_iter()
            .with_min_len(min_len)
            .map(func)
            .collect::<Vec<R>>(),
    }
}

/// Maps a function over a vector with indices, using parallel processing for large collections.
///
/// Similar to `fast_map` but provides the index along with each element.
/// Used for operations that need to know the position of items.
#[inline]
pub(crate) fn fast_enumerate_map<T, F, R>(items: &Vec<T>, func: F, min_len: usize) -> Vec<R>
where
    F: Fn((usize, &T)) -> R + Sync + Send,
    R: Send,
    T: Sync,
{
    match items.len() >= (num_cpus::get() * min_len / 2) {
        false => items.iter().enumerate().map(func).collect::<Vec<R>>(),
        true => items
            .par_iter()
            .enumerate()
            .with_min_len(min_len)
            .map(func)
            .collect::<Vec<R>>(),
    }
}

/// Maps a function over a vector with ownership transfer, using parallel processing for large collections.
///
/// Consumes the input vector and transfers ownership to the mapping function.
/// Used when the original data is no longer needed.
#[inline]
pub(crate) fn fast_move_map<T, F, R>(items: Vec<T>, func: F, min_len: usize) -> Vec<R>
where
    F: Fn(T) -> R + Sync + Send,
    T: Send,
    R: Send,
{
    match items.len() >= (num_cpus::get() * min_len / 2) {
        false => items.into_iter().map(func).collect::<Vec<R>>(),
        true => items
            .into_par_iter()
            .with_min_len(min_len)
            .map(func)
            .collect::<Vec<R>>(),
    }
}

// Parsing helper functions

/// Parses a string to a floating-point number.
///
/// Returns 0.0 if parsing fails, providing a fallback for malformed data.
#[inline]
pub(crate) fn parse_float(s: &str) -> f64 {
    s.parse().unwrap_or(0.0)
}

/// Parses a string by removing surrounding quotes.
///
/// Removes leading and trailing double quote characters from TextGrid string values.
#[inline]
pub(crate) fn parse_str(s: &str) -> String {
    s.trim_matches('"').to_string()
}

/// Parses a string to an unsigned integer.
///
/// Returns 0 if parsing fails, providing a fallback for malformed data.
#[inline]
pub(crate) fn parse_uint(s: &str) -> usize {
    s.parse().unwrap_or(0)
}
