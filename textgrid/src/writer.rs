use crate::textgrid::{Item, TextGrid, Tier};
use crate::utils::{fast_enumerate_map, fast_map};

impl Tier {
    /// Converts the tier to a string representation in long TextGrid format.
    ///
    /// # Arguments
    ///
    /// * `index` - The 0-based index of this tier in the TextGrid
    ///
    /// # Returns
    ///
    /// Returns a string containing the tier data in long format with proper indentation and formatting.
    ///
    /// Used internally by `TextGrid::to_long_textgrid_string` to serialize tiers.
    pub fn to_long_textgrid_string(&self, index: usize) -> String {
        let (tier_class, tier_name) = if self.interval_tier {
            ("IntervalTier", "intervals")
        } else {
            ("TextTier", "points")
        };
        let mut output = format!(
            "    item [{}]:\r\n        class = \"{}\" \r\n        name = \"{}\" \r\n        xmin = {} \r\n        xmax = {} \r\n        {}: size = {} \r\n",
            index + 1,
            tier_class,
            self.name.replace('"', "\\\""),
            self.tmin,
            self.tmax,
            tier_name,
            self.items.len()
        );
        let map_fun = |(index, item): (usize, &Item)| -> String {
            match tier_class {
                "IntervalTier" => {
                    format!(
                        "        intervals [{}]:\r\n            xmin = {} \r\n            xmax = {} \r\n            text = \"{}\" \r\n",
                        index + 1,
                        item.tmin,
                        item.tmax,
                        item.label.replace('"', "\\\"")
                    )
                }
                "TextTier" => {
                    format!(
                        "        points [{}]:\r\n            number = {} \r\n            mark = \"{}\" \r\n",
                        index + 1,
                        item.tmin,
                        item.label.replace('"', "\\\"")
                    )
                }
                _ => String::new(),
            }
        };
        let item_strings = fast_enumerate_map(&self.items, map_fun, 20);
        output.push_str(item_strings.join("").as_str());
        output
    }

    /// Converts the tier to a string representation in short TextGrid format.
    ///
    /// # Returns
    ///
    /// Returns a string containing the tier data in short format with minimal formatting.
    ///
    /// Used internally by `TextGrid::to_short_textgrid_string` to serialize tiers.
    pub fn to_short_textgrid_string(&self) -> String {
        let tier_class;
        if self.interval_tier {
            tier_class = "IntervalTier";
        } else {
            tier_class = "TextTier";
        }
        let mut output = format!(
            "\"{}\"\r\n\"{}\"\r\n{}\r\n{}\r\n{}\r\n",
            tier_class,
            self.name.replace('"', "\\\""),
            self.tmin,
            self.tmax,
            self.items.len()
        );
        let map_fun = |item: &Item| -> String {
            match tier_class {
                "IntervalTier" => {
                    format!(
                        "{}\r\n{}\r\n\"{}\"\r\n",
                        item.tmin,
                        item.tmax,
                        item.label.replace('"', "\\\"")
                    )
                }
                "TextTier" => {
                    format!(
                        "{}\r\n\"{}\"\r\n",
                        item.tmin,
                        item.label.replace('"', "\\\"")
                    )
                }
                _ => String::new(),
            }
        };
        let item_strings = fast_map(&self.items, map_fun, 20);
        output.push_str(item_strings.join("").as_str());
        output
    }
}

impl TextGrid {
    /// Converts the TextGrid to a string representation in long format.
    ///
    /// The long format is the traditional Praat TextGrid format with explicit key-value pairs
    /// and detailed structure. Note: In the long format, many lines are ended with a space character
    /// for compatibility with Praat.
    ///
    /// # Returns
    ///
    /// Returns a string containing the complete TextGrid data in long format.
    pub fn to_long_textgrid_string(&self) -> String {
        // Note: In the long format, many lines are ended with a space character.
        // I don't know why and it seems unnecessary, but to be compatible, we add them here.
        let nitems = self.tiers.len();
        let tiers_existence = if nitems > 0 { "<exists>" } else { "<absent>" };
        let mut output = format!(
            "File type = \"ooTextFile\"\r\nObject class = \"TextGrid\"\r\n\r\nxmin = {} \r\nxmax = {} \r\ntiers? {} \r\nsize = {} \r\nitem []: \r\n",
            self.tmin, self.tmax, tiers_existence, nitems,
        );
        for (i, item) in self.tiers.iter().enumerate() {
            output.push_str(&item.to_long_textgrid_string(i));
        }
        output
    }

    /// Converts the TextGrid to a string representation in short format.
    ///
    /// The short format is a more compact representation of the TextGrid data.
    ///
    /// # Returns
    ///
    /// Returns a string containing the complete TextGrid data in short format.
    pub fn to_short_textgrid_string(&self) -> String {
        let nitems = self.tiers.len();
        let tiers_existence = if nitems > 0 { "<exists>" } else { "<absent>" };
        let mut output = format!(
            "File type = \"ooTextFile\"\r\nObject class = \"TextGrid\"\r\n\r\n{}\r\n{}\r\n{}\r\n{}\r\n",
            self.tmin, self.tmax, tiers_existence, nitems,
        );
        for item in self.tiers.iter() {
            output.push_str(&item.to_short_textgrid_string());
        }
        output
    }

    /// Saves the TextGrid to a file.
    ///
    /// # Arguments
    ///
    /// * `filename` - The path where the file will be saved
    /// * `long` - If `true`, saves in long format; if `false`, saves in short format
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use textgrid::{TextGrid, read_from_file};
    ///
    /// let tg = read_from_file("input.TextGrid", false, "auto").unwrap();
    ///
    /// // Save in long format
    /// tg.save_textgrid("output_long.TextGrid", true);
    ///
    /// // Save in short format
    /// tg.save_textgrid("output_short.TextGrid", false);
    /// ```
    pub fn save_textgrid(&self, filename: &str, long: bool) {
        let content = if long {
            self.to_long_textgrid_string()
        } else {
            self.to_short_textgrid_string()
        };
        std::fs::write(filename, content).unwrap();
    }

    /// Saves the TextGrid to a CSV file.
    ///
    /// # Arguments
    ///
    /// * `filename` - The path where the CSV file will be saved
    ///
    /// # Format
    ///
    /// The CSV file will have the following columns:
    /// * `tmin` - Start time of the item
    /// * `tmax` - End time of the item
    /// * `label` - Text label of the item
    /// * `tier` - Name of the tier
    /// * `is_interval` - Whether the tier is an interval tier
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be created or written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use textgrid::read_from_file;
    ///
    /// let tg = read_from_file("input.TextGrid", false, "auto").unwrap();
    /// tg.save_csv("output.csv");
    /// ```
    pub fn save_csv(&self, filename: &str) {
        let mut wtr = csv::WriterBuilder::new()
            .delimiter(b',')
            .quote_style(csv::QuoteStyle::NonNumeric)
            .from_writer(std::fs::File::create(filename).unwrap());
        let (_, _, data) = self.to_data();
        wtr.write_record(&["tmin", "tmax", "label", "tier", "is_interval"])
            .unwrap();
        for (tier_name, is_interval, items) in data {
            for (tmin, tmax, label) in items {
                wtr.write_record(&[
                    tmin.to_string(),
                    tmax.to_string(),
                    label,
                    tier_name.clone(),
                    is_interval.to_string(),
                ])
                .unwrap();
            }
        }
    }
}
