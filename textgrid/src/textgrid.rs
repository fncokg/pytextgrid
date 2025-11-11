use std::io::{Error, ErrorKind, Result};

/// Epsilon value for floating-point time comparisons.
const TIME_EPSILON: f64 = 1e-6;

/// Represents an item (interval or point) in a TextGrid tier.
///
/// For point tiers, `tmin` equals `tmax`.
pub struct Item {
    /// Start time of the item.
    pub tmin: f64,
    /// End time of the item.
    pub tmax: f64,
    /// Text label associated with the item.
    pub label: String,
}

/// Represents a tier in a TextGrid.
///
/// A tier can be either an interval tier (with time ranges) or a point tier (with time points).
pub struct Tier {
    /// Name of the tier.
    pub name: String,
    /// Number of items in the tier.
    pub size: usize,
    /// Vector of items contained in the tier.
    pub items: Vec<Item>,
    /// Whether this is an interval tier (true) or point tier (false).
    pub interval_tier: bool,
    /// Minimum time of the tier.
    pub tmin: f64,
    /// Maximum time of the tier.
    pub tmax: f64,
}

/// Represents a Praat TextGrid object.
///
/// A TextGrid contains multiple tiers and defines a time range.
pub struct TextGrid {
    /// Minimum time of the TextGrid.
    pub tmin: f64,
    /// Maximum time of the TextGrid.
    pub tmax: f64,
    /// Number of tiers in the TextGrid.
    pub size: usize,
    /// Name of the TextGrid.
    pub name: String,
    /// Vector of tiers contained in the TextGrid.
    pub tiers: Vec<Tier>,
}

impl Item {
    /// Creates a new empty `Item` with default values.
    ///
    /// # Returns
    ///
    /// Returns an `Item` with tmin and tmax set to 0.0 and an empty label.
    ///
    /// # Examples
    ///
    /// ```
    /// use textgrid::Item;
    ///
    /// let item = Item::new();
    /// assert_eq!(item.tmin, 0.0);
    /// assert_eq!(item.tmax, 0.0);
    /// assert_eq!(item.label, "");
    /// ```
    pub fn new() -> Self {
        Item {
            tmin: 0.0,
            tmax: 0.0,
            label: String::new(),
        }
    }
}

impl Tier {
    /// Creates a new empty `Tier` with default values.
    ///
    /// # Returns
    ///
    /// Returns a `Tier` with default values: empty name, zero size, empty items vector,
    /// interval_tier set to true, and tmin/tmax set to 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use textgrid::Tier;
    ///
    /// let tier = Tier::new();
    /// assert_eq!(tier.name, "");
    /// assert_eq!(tier.size, 0);
    /// assert!(tier.interval_tier);
    /// assert_eq!(tier.items.len(), 0);
    /// ```
    pub fn new() -> Self {
        Tier {
            name: String::new(),
            size: 0,
            items: Vec::new(),
            interval_tier: true,
            tmin: 0.0,
            tmax: 0.0,
        }
    }

    /// Adds an empty item to the tier's item list.
    ///
    /// Used during parsing to add placeholder items that will be populated with data.
    pub fn add_empty_item(&mut self) {
        self.items.push(Item::new());
    }

    /// Validates the tier structure and constraints.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the tier is valid, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The `size` field doesn't match the actual number of items
    /// * The time bounds are invalid
    /// * Any item has invalid time bounds (for interval tiers)
    /// * Any point tier item has tmin != tmax
    /// * Any adjacent items overlap
    pub fn assert_valid(&self) -> Result<()> {
        if self.size != self.items.len() {
            return Err(data_error("Tier size does not match number of items"));
        }
        assert_valid_time_bounds(self.tmin, self.tmax, &format!("tier {}", self.name))?;
        for item_idx in 0..self.items.len() {
            let this_item = &self.items[item_idx];

            if self.interval_tier {
                assert_valid_time_bounds(
                    this_item.tmin,
                    this_item.tmax,
                    &format!("item {} in tier {}", item_idx, self.name),
                )?;
            } else {
                if (this_item.tmin - this_item.tmax).abs() > TIME_EPSILON {
                    return Err(data_error(&format!(
                        "Item {} should have tmin == tmax in PointTier {}",
                        item_idx, self.name
                    )));
                }
            }
            if item_idx + 1 < self.items.len() {
                let next_item = &self.items[item_idx + 1];
                if this_item.tmax - next_item.tmin > TIME_EPSILON {
                    return Err(data_error(&format!(
                        "Items {} and {} overlap in tier {}",
                        item_idx,
                        item_idx + 1,
                        self.name
                    )));
                }
            }
        }
        Ok(())
    }
}

impl TextGrid {
    /// Creates a new empty `TextGrid` with default values.
    ///
    /// # Returns
    ///
    /// Returns a `TextGrid` with tmin/tmax set to 0.0, zero size, empty name, and no tiers.
    ///
    /// # Examples
    ///
    /// ```
    /// use textgrid::TextGrid;
    ///
    /// let tg = TextGrid::new();
    /// assert_eq!(tg.tmin, 0.0);
    /// assert_eq!(tg.tmax, 0.0);
    /// assert_eq!(tg.size, 0);
    /// assert_eq!(tg.tiers.len(), 0);
    /// ```
    pub fn new() -> Self {
        TextGrid {
            tmin: 0.0,
            tmax: 0.0,
            size: 0,
            name: String::new(),
            tiers: Vec::new(),
        }
    }

    /// Adds an empty tier to the TextGrid's tier list.
    ///
    /// Used during parsing to add placeholder tiers that will be populated with data.
    pub fn add_empty_tier(&mut self) {
        self.tiers.push(Tier::new());
    }

    /// Validates the TextGrid structure and all its tiers.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the TextGrid and all its tiers are valid, otherwise returns an error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The `size` field doesn't match the actual number of tiers
    /// * The time bounds are invalid
    /// * Any tier is invalid (see [`Tier::assert_valid`])
    pub fn assert_valid(&self) -> Result<()> {
        if self.size != self.tiers.len() {
            return Err(data_error("TextGrid size does not match number of tiers"));
        }

        assert_valid_time_bounds(self.tmin, self.tmax, "TextGrid")?;

        for tier in &self.tiers {
            tier.assert_valid()?;
        }

        Ok(())
    }
}

// Helper functions for error handling and validation

/// Creates an `Error` with `InvalidData` kind.
///
/// Convenience function for creating data validation errors consistently.
#[inline]
fn data_error(msg: &str) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Validates time bounds for any TextGrid element.
///
/// Checks that tmin is non-negative, tmax is positive, and tmax > tmin.
/// Used internally by tier and TextGrid validation.
#[inline]
fn assert_valid_time_bounds(tmin: f64, tmax: f64, where_msg: &str) -> Result<()> {
    if tmin < 0.0 || tmax <= 0.0 {
        return Err(data_error(&format!(
            "Time bounds should be non-negative in {}",
            where_msg
        )));
    }
    if tmax - tmin <= TIME_EPSILON {
        return Err(data_error(&format!(
            "tmin should be less than tmax in {}",
            where_msg
        )));
    }
    Ok(())
}
