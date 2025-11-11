use crate::textgrid::{Item, TextGrid, Tier};
use crate::utils::{fast_map, fast_move_map};
use std::io::{Error, ErrorKind, Result};

/// Finds the extreme value (minimum or maximum) in a collection of items.
///
/// # Arguments
///
/// * `items` - A vector of items to search
/// * `key` - A function that extracts the value to compare from each item
/// * `find_max` - If `true`, finds the maximum value; if `false`, finds the minimum
///
/// # Returns
///
/// Returns `Some(f64)` with the extreme value if the collection is not empty,
/// otherwise returns `None`.
#[inline]
fn get_extreme<T, K>(items: &Vec<T>, key: K, find_max: bool) -> Option<f64>
where
    K: Fn(&T) -> f64,
{
    if items.is_empty() {
        return None;
    }
    let ext: f64;
    if find_max {
        ext = items
            .iter()
            .map(|item| key(item))
            .fold(f64::MIN, |a, b| a.max(b));
    } else {
        ext = items
            .iter()
            .map(|item| key(item))
            .fold(f64::MAX, |a, b| a.min(b));
    }
    Some(ext)
}

/// Gets an extreme value from a collection with an optional default.
///
/// # Arguments
///
/// * `default` - An optional default value to return if provided
/// * `items` - A vector of items to search if no default is provided
/// * `key` - A function that extracts the value to compare from each item
/// * `find_max` - If `true`, finds the maximum value; if `false`, finds the minimum
///
/// # Returns
///
/// Returns the default value if provided, otherwise computes and returns the extreme value,
/// or 0.0 if the collection is empty.
///
/// Used when constructing tiers or TextGrids where time bounds may be explicitly provided
/// or should be computed from the data.
#[inline]
fn get_optional_extreme<T, F>(default: Option<f64>, items: &Vec<T>, key: F, find_max: bool) -> f64
where
    F: Fn(&T) -> f64,
{
    match default {
        Some(val) => val,
        None => get_extreme(items, key, find_max).unwrap_or(0.0),
    }
}

/// Creates a `Tier` from a collection of items.
///
/// # Arguments
///
/// * `items` - A vector of items for the tier
/// * `tier_name` - The name of the tier
/// * `is_interval` - Whether this is an interval tier (true) or point tier (false)
/// * `tmin` - Optional minimum time; if not provided, will be computed from items
/// * `tmax` - Optional maximum time; if not provided, will be computed from items
///
/// # Returns
///
/// Returns a `Tier` with the provided or computed values.
///
/// Helper function used by conversion methods to construct tiers with consistent logic.
#[inline]
fn make_tier(
    items: Vec<Item>,
    tier_name: String,
    is_interval: bool,
    tmin: Option<f64>,
    tmax: Option<f64>,
) -> Tier {
    Tier {
        name: tier_name,
        interval_tier: is_interval,
        tmin: get_optional_extreme(tmin, &items, |item| item.tmin, false),
        tmax: get_optional_extreme(tmax, &items, |item| item.tmax, true),
        size: items.len(),
        items,
    }
}

/// Creates a `TextGrid` from a collection of tiers.
///
/// # Arguments
///
/// * `tiers` - A vector of tiers for the TextGrid
/// * `name` - Optional name for the TextGrid; defaults to "ConvertedTextGrid" if not provided
/// * `tmin` - Optional minimum time; if not provided, will be computed from tiers
/// * `tmax` - Optional maximum time; if not provided, will be computed from tiers
///
/// # Returns
///
/// Returns a `Result` containing the `TextGrid` if validation passes, otherwise returns an error.
///
/// # Errors
///
/// Returns an error if the TextGrid validation fails.
///
/// Helper function used by conversion methods to construct TextGrids with consistent validation.
#[inline]
fn make_textgrid(
    tiers: Vec<Tier>,
    name: Option<String>,
    tmin: Option<f64>,
    tmax: Option<f64>,
) -> Result<TextGrid> {
    let tgt = TextGrid {
        name: match name {
            Some(n) => n,
            None => String::from("ConvertedTextGrid"),
        },
        tmin: get_optional_extreme(tmin, &tiers, |tier| tier.tmin, false),
        tmax: get_optional_extreme(tmax, &tiers, |tier| tier.tmax, true),
        size: tiers.len(),
        tiers,
    };
    tgt.assert_valid()?;
    Ok(tgt)
}

impl TextGrid {
    /// Converts the TextGrid to a nested data structure.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// * `tmin` - The minimum time of the TextGrid
    /// * `tmax` - The maximum time of the TextGrid
    /// * A vector of tier data, where each tier contains:
    ///   - Tier name (String)
    ///   - Whether it's an interval tier (bool)
    ///   - A vector of items (tmin, tmax, label)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use textgrid::read_from_file;
    ///
    /// let tg = read_from_file("example.TextGrid", false, "auto").unwrap();
    /// let (tmin, tmax, tiers) = tg.to_data();
    /// println!("TextGrid spans {:.2} to {:.2} seconds", tmin, tmax);
    /// for (tier_name, is_interval, items) in tiers {
    ///     println!("Tier '{}' has {} items", tier_name, items.len());
    /// }
    /// ```
    pub fn to_data(&self) -> (f64, f64, Vec<(String, bool, Vec<(f64, f64, String)>)>) {
        let mut data = Vec::new();
        let map_fun = |item: &Item| (item.tmin, item.tmax, item.label.clone());
        for tier in self.tiers.iter() {
            let tier_data = (
                tier.name.clone(),
                tier.interval_tier,
                fast_map(&tier.items, map_fun, 20),
            );
            data.push(tier_data);
        }
        (self.tmin, self.tmax, data)
    }

    /// Creates a TextGrid from a nested data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of tier data, where each tier contains:
    ///   - Tier name (String)
    ///   - Whether it's an interval tier (bool)
    ///   - A vector of items (tmin, tmax, label)
    /// * `name` - Optional name for the TextGrid
    /// * `tmin` - Optional minimum time; if not provided, will be computed from tiers
    /// * `tmax` - Optional maximum time; if not provided, will be computed from tiers
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `TextGrid` if successful, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the TextGrid validation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use textgrid::TextGrid;
    ///
    /// let data = vec![
    ///     (
    ///         String::from("words"),
    ///         true,  // interval tier
    ///         vec![
    ///             (0.0, 0.5, String::from("hello")),
    ///             (0.5, 1.0, String::from("world")),
    ///         ],
    ///     ),
    /// ];
    ///
    /// let tg = TextGrid::from_data(
    ///     data,
    ///     Some(String::from("example")),
    ///     Some(0.0),
    ///     Some(1.0),
    /// ).unwrap();
    ///
    /// assert_eq!(tg.tiers.len(), 1);
    /// assert_eq!(tg.tiers[0].items.len(), 2);
    /// ```
    pub fn from_data(
        data: Vec<(String, bool, Vec<(f64, f64, String)>)>,
        name: Option<String>,
        tmin: Option<f64>,
        tmax: Option<f64>,
    ) -> Result<TextGrid> {
        let mut tiers = Vec::new();
        for (tier_name, is_interval, items_data) in data.into_iter() {
            let map_fun = |item_data: (f64, f64, String)| Item {
                tmin: item_data.0,
                tmax: item_data.1,
                label: item_data.2,
            };
            let items = fast_move_map(items_data, map_fun, 20);
            if items.is_empty() {
                continue;
            }
            let tier = make_tier(items, tier_name, is_interval, tmin, tmax);
            tiers.push(tier);
        }
        let tgt = make_textgrid(tiers, name, tmin, tmax)?;
        Ok(tgt)
    }

    /// Converts the TextGrid to flat vectors of data.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// * A vector of tmin values (`Vec<f64>`)
    /// * A vector of tmax values (`Vec<f64>`)
    /// * A vector of labels (`Vec<String>`)
    /// * A vector of tier names (`Vec<String>`)
    /// * A vector of interval tier flags (`Vec<bool>`)
    ///
    /// All vectors have the same length, with one entry per item across all tiers.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use textgrid::read_from_file;
    ///
    /// let tg = read_from_file("example.TextGrid", false, "auto").unwrap();
    /// let (tmins, tmaxs, labels, tier_names, is_intervals) = tg.to_vectors();
    ///
    /// for i in 0..tmins.len() {
    ///     println!("{:.2}-{:.2}: {} (tier: {})",
    ///              tmins[i], tmaxs[i], labels[i], tier_names[i]);
    /// }
    /// ```
    pub fn to_vectors(&self) -> (Vec<f64>, Vec<f64>, Vec<String>, Vec<String>, Vec<bool>) {
        let mut tmins = Vec::new();
        let mut tmaxs = Vec::new();
        let mut labels = Vec::new();
        let mut tier_names = Vec::new();
        let mut is_intervals: Vec<bool> = Vec::new();
        for tier in self.tiers.iter() {
            for item in tier.items.iter() {
                tmins.push(item.tmin);
                tmaxs.push(item.tmax);
                labels.push(item.label.clone());
                is_intervals.push(tier.interval_tier);
                tier_names.push(tier.name.clone());
            }
        }
        (tmins, tmaxs, labels, tier_names, is_intervals)
    }

    /// Creates a TextGrid from flat vectors of data.
    ///
    /// # Arguments
    ///
    /// * `tmins` - A vector of start times for all items
    /// * `tmaxs` - A vector of end times for all items
    /// * `labels` - A vector of text labels for all items
    /// * `tier_names` - A vector of tier names for all items
    /// * `is_intervals` - A vector of flags indicating whether each item's tier is an interval tier
    /// * `tmin` - Optional minimum time for the TextGrid
    /// * `tmax` - Optional maximum time for the TextGrid
    /// * `name` - Optional name for the TextGrid
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `TextGrid` if successful, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * Input vectors have different lengths
    /// * The TextGrid validation fails
    ///
    /// # Notes
    ///
    /// * Items are grouped by tier name and sorted by tmin within each tier
    /// * The order of tiers in the output matches the first appearance order in the input vectors
    ///
    /// # Examples
    ///
    /// ```
    /// use textgrid::TextGrid;
    ///
    /// let tmins = vec![0.0, 0.5];
    /// let tmaxs = vec![0.5, 1.0];
    /// let labels = vec![String::from("hello"), String::from("world")];
    /// let tier_names = vec![String::from("words"), String::from("words")];
    /// let is_intervals = vec![true, true];
    ///
    /// let tg = TextGrid::from_vectors(
    ///     tmins, tmaxs, labels, tier_names, is_intervals,
    ///     Some(0.0), Some(1.0), Some(String::from("example"))
    /// ).unwrap();
    ///
    /// assert_eq!(tg.tiers.len(), 1);
    /// assert_eq!(tg.tiers[0].items.len(), 2);
    /// ```
    pub fn from_vectors(
        tmins: Vec<f64>,
        tmaxs: Vec<f64>,
        labels: Vec<String>,
        tier_names: Vec<String>,
        is_intervals: Vec<bool>,
        tmin: Option<f64>,
        tmax: Option<f64>,
        name: Option<String>,
    ) -> Result<TextGrid> {
        if tmins.len() != tmaxs.len()
            || tmins.len() != labels.len()
            || tmins.len() != tier_names.len()
            || tmins.len() != is_intervals.len()
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Input vectors must have the same length",
            ));
        }
        let mut tier_map: std::collections::HashMap<String, (Vec<Item>, bool)> =
            std::collections::HashMap::new();

        // Also preserve the order of tier names, thus the order of the final TextGrid tiers is kept the same as the first appearance in vectors
        let mut tier_name_order: Vec<String> = Vec::new();
        for i in 0..tmins.len() {
            let item = Item {
                tmin: tmins[i],
                tmax: tmaxs[i],
                label: labels[i].clone(),
            };
            let tier_name: &String = &tier_names[i];
            if !tier_map.contains_key(tier_name) {
                tier_name_order.push(tier_name.clone());
                tier_map.insert(tier_name.clone(), (Vec::new(), is_intervals[i]));
            }
            tier_map.get_mut(tier_name).unwrap().0.push(item);
        }
        let mut tiers = Vec::new();
        for tier_name in tier_name_order.into_iter() {
            let (mut items, is_interval) = tier_map.remove(&tier_name).unwrap();
            // sort items by tmin
            items.sort_by(|a, b| a.tmin.partial_cmp(&b.tmin).unwrap());
            let tier = make_tier(items, tier_name, is_interval, tmin, tmax);
            tiers.push(tier);
        }
        let tgt = make_textgrid(tiers, name, tmin, tmax)?;
        Ok(tgt)
    }
}
