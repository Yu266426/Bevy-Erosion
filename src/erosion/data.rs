use bevy::math::Vec2;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErosionDataType {
    /// Tracks total amount of sediment deposited at each point
    Deposition,
    /// Tracks total amount of material eroded at each point
    Erosion,
    /// Tracks water * speed through each point
    WaterFlux,
    /// Tracks average flow speed at each point
    FlowSpeed,
    /// Tracks total amount of water
    Wetness,
    /// Tracks number of particles that have affected each point
    ParticleDensity,
}

#[derive(Debug, Clone)]
pub struct ErosionData {
    resolution: usize,
    /// Data layers indexed by ErosionDataType
    layers: Vec<Vec<f32>>,
    /// Normalized flow direction vectors
    flow_directions: Option<Vec<Vec2>>,
    /// Which data types are being collected
    enabled_types: Vec<ErosionDataType>,
}

impl ErosionData {
    pub fn new(
        resolution: usize,
        data_types: &[ErosionDataType],
        collect_flow_directions: bool,
    ) -> Self {
        let mut enabled_types = Vec::new();
        let mut layers = Vec::new();

        for &data_type in data_types {
            enabled_types.push(data_type);
            layers.push(vec![0.0; resolution * resolution]);
        }

        let flow_directions = if collect_flow_directions {
            Some(vec![Vec2::ZERO; resolution * resolution])
        } else {
            None
        };

        Self {
            resolution,
            layers,
            flow_directions,
            enabled_types,
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    pub fn basic(resolution: usize) -> Self {
        Self::new(
            resolution,
            &[
                ErosionDataType::Deposition,
                ErosionDataType::Erosion,
                ErosionDataType::WaterFlux,
            ],
            true,
        )
    }

    pub fn comprehensive(resolution: usize) -> Self {
        Self::new(
            resolution,
            &[
                ErosionDataType::Deposition,
                ErosionDataType::Erosion,
                ErosionDataType::WaterFlux,
                ErosionDataType::FlowSpeed,
                ErosionDataType::Wetness,
                ErosionDataType::ParticleDensity,
            ],
            true,
        )
    }

    #[inline]
    pub fn get_index(&self, row: usize, col: usize) -> usize {
        debug_assert!(
            row < self.resolution,
            "Row index out of bounds: {} >= {}",
            row,
            self.resolution
        );
        debug_assert!(
            col < self.resolution,
            "Column index out of bounds: {} >= {}",
            col,
            self.resolution
        );
        row * self.resolution + col
    }

    /// Records data at the specified position
    ///
    /// # Arguments
    /// * `data_type` - The type of data to record
    /// * `pos` - The position in grid space
    /// * `value` - The value to add to the current value
    ///
    /// # Returns
    /// `true` if the data was recorded, `false` if the data type is not enabled
    pub fn record(&mut self, data_type: ErosionDataType, pos: &Vec2, value: f32) -> bool {
        if let Some(layer_index) = self.get_layer_index(data_type) {
            let col = pos.x as usize;
            let row = pos.y as usize;

            if row < self.resolution && col < self.resolution {
                let index = self.get_index(row, col);
                self.layers[layer_index][index] += value;
                return true;
            }
        }
        false
    }

    /// Records data using bilinear interpolation
    ///
    /// # Arguments
    /// * `data_type` - The type of data to record
    /// * `pos` - The position in grid space (can be fractional)
    /// * `value` - The value to add to the current values
    ///
    /// # Returns
    /// `true` if the data was recorded, `false` if the data type is not enabled
    pub fn record_bilinear(&mut self, data_type: ErosionDataType, pos: &Vec2, value: f32) -> bool {
        if let Some(layer_index) = self.get_layer_index(data_type) {
            let col_f = pos.x;
            let row_f = pos.y;

            // Ensure position is within bounds
            if col_f < 0.0
                || row_f < 0.0
                || col_f >= self.resolution as f32 - 1.0
                || row_f >= self.resolution as f32 - 1.0
            {
                return false;
            }

            // Calculate grid cell coordinates and weights
            let col0 = col_f.floor() as usize;
            let row0 = row_f.floor() as usize;
            let col1 = (col0 + 1).min(self.resolution - 1);
            let row1 = (row0 + 1).min(self.resolution - 1);

            let col_weight = col_f - col0 as f32;
            let row_weight = row_f - row0 as f32;

            // Calculate bilinear weights for each corner
            let w00 = (1.0 - col_weight) * (1.0 - row_weight);
            let w01 = (1.0 - col_weight) * row_weight;
            let w10 = col_weight * (1.0 - row_weight);
            let w11 = col_weight * row_weight;

            // Add weighted values to each corner
            let idx00 = self.get_index(row0, col0);
            let idx01 = self.get_index(row1, col0);
            let idx10 = self.get_index(row0, col1);
            let idx11 = self.get_index(row1, col1);

            self.layers[layer_index][idx00] += value * w00;
            self.layers[layer_index][idx01] += value * w01;
            self.layers[layer_index][idx10] += value * w10;
            self.layers[layer_index][idx11] += value * w11;

            return true;
        }
        false
    }

    /// Records flow direction at the specified position
    ///
    /// # Arguments
    /// * `pos` - The position in grid space
    /// * `direction` - The direction vector to add (will be normalized and averaged with existing)
    ///
    /// # Returns
    /// `true` if the direction was recorded, `false` if flow direction tracking is disabled
    pub fn record_flow_direction(&mut self, pos: &Vec2, direction: Vec2) -> bool {
        if let Some(ref mut flow_dirs) = self.flow_directions {
            let col = pos.x as usize;
            let row = pos.y as usize;

            if row < self.resolution && col < self.resolution {
                let index = row * self.resolution + col;

                // Normalize direction and average with existing direction
                let dir_norm = direction.normalize_or_zero();
                if dir_norm != Vec2::ZERO {
                    let current = flow_dirs[index];

                    // Weighted average - give more weight to stronger flows
                    let speed = direction.length();
                    let current_weight = current.length();
                    let new_weight = speed;

                    let total_weight = current_weight + new_weight;
                    if total_weight > 0.0 {
                        flow_dirs[index] =
                            (current * current_weight + dir_norm * new_weight) / total_weight;
                    } else {
                        flow_dirs[index] = dir_norm;
                    }
                }
                return true;
            }
        }
        false
    }

    fn get_layer_index(&self, data_type: ErosionDataType) -> Option<usize> {
        self.enabled_types.iter().position(|&t| t == data_type)
    }

    pub fn get_data(&self, data_type: ErosionDataType) -> Option<&[f32]> {
        self.get_layer_index(data_type)
            .map(|idx| self.layers[idx].as_slice())
    }

    pub fn get_data_mut(&mut self, data_type: ErosionDataType) -> Option<&mut [f32]> {
        self.get_layer_index(data_type)
            .map(move |idx| self.layers[idx].as_mut_slice())
    }

    pub fn get_flow_directions(&self) -> Option<&[Vec2]> {
        self.flow_directions.as_ref().map(|dirs| dirs.as_slice())
    }

    /// Normalizes the data for the specified type to the range [0, 1]
    ///
    /// # Returns
    /// The (min, max) values before normalization
    pub fn normalize(&mut self, data_type: ErosionDataType) -> Option<(f32, f32)> {
        if let Some(layer_index) = self.get_layer_index(data_type) {
            let data = &mut self.layers[layer_index];

            // Find min and max values
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;

            for &val in data.iter() {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            // Normalize if range is non-zero
            let range = max_val - min_val;
            if range > 1e-6 {
                for val in data.iter_mut() {
                    *val = (*val - min_val) / range;
                }
            } else if min_val != 0.0 {
                // If all values are the same but not zero, set them to 1.0
                for val in data.iter_mut() {
                    *val = 1.0;
                }
            }

            Some((min_val, max_val))
        } else {
            None
        }
    }

    /// Normalizes all data layers to the range [0, 1]
    pub fn normalize_all(&mut self) -> Vec<(ErosionDataType, f32, f32)> {
        let mut results = Vec::new();

        // Create a copy of enabled_types to avoid borrowing conflicts
        let data_types = self.enabled_types.clone();

        for data_type in data_types {
            if let Some((min, max)) = self.normalize(data_type) {
                results.push((data_type, min, max));
            }
        }

        results
    }

    pub fn sample_bilinear(
        &self,
        data_type: ErosionDataType,
        norm_coords: (f32, f32),
    ) -> Option<f32> {
        if let Some(layer_index) = self.get_layer_index(data_type) {
            let (norm_col, norm_row) = norm_coords;

            // Clamp normalized inputs to [0.0, 1.0] range
            let norm_row = norm_row.clamp(0.0, 1.0);
            let norm_col = norm_col.clamp(0.0, 1.0);

            // Scale normalized coordinates to floating-point grid coordinates
            let r_f = norm_row * (self.resolution - 1) as f32;
            let c_f = norm_col * (self.resolution - 1) as f32;

            // Get the integer parts (top-left corner of the cell for interpolation)
            let r0 = r_f.floor() as usize;
            let c0 = c_f.floor() as usize;

            // Fractional parts for interpolation weights
            let r_frac = r_f - r0 as f32;
            let c_frac = c_f - c0 as f32;

            // Coordinates of the four surrounding grid points
            let r1 = (r0 + 1).min(self.resolution - 1);
            let c1 = (c0 + 1).min(self.resolution - 1);

            // Sample the values at the four surrounding points
            let val_r0c0 = self.layers[layer_index][self.get_index(r0, c0)]; // Top-left
            let val_r0c1 = self.layers[layer_index][self.get_index(r0, c1)]; // Top-right
            let val_r1c0 = self.layers[layer_index][self.get_index(r1, c0)]; // Bottom-left
            let val_r1c1 = self.layers[layer_index][self.get_index(r1, c1)]; // Bottom-right

            // Interpolate along the columns (c-axis) for each row (r0, r1)
            let interp_at_r0 = val_r0c0 * (1.0 - c_frac) + val_r0c1 * c_frac;
            let interp_at_r1 = val_r1c0 * (1.0 - c_frac) + val_r1c1 * c_frac;

            // Interpolate along the rows (r-axis) using the results from column interpolations
            let final_value = interp_at_r0 * (1.0 - r_frac) + interp_at_r1 * r_frac;

            Some(final_value)
        } else {
            None
        }
    }

    pub fn sample_flow_direction_bilinear(&self, norm_coords: (f32, f32)) -> Option<Vec2> {
        if let Some(ref flow_dirs) = self.flow_directions {
            let (norm_col, norm_row) = norm_coords;

            // Clamp normalized inputs to [0.0, 1.0] range
            let norm_row = norm_row.clamp(0.0, 1.0);
            let norm_col = norm_col.clamp(0.0, 1.0);

            // Scale normalized coordinates to floating-point grid coordinates
            let r_f = norm_row * (self.resolution - 1) as f32;
            let c_f = norm_col * (self.resolution - 1) as f32;

            // Get the integer parts (top-left corner of the cell for interpolation)
            let r0 = r_f.floor() as usize;
            let c0 = c_f.floor() as usize;

            // Fractional parts for interpolation weights
            let r_frac = r_f - r0 as f32;
            let c_frac = c_f - c0 as f32;

            // Coordinates of the four surrounding grid points
            let r1 = (r0 + 1).min(self.resolution - 1);
            let c1 = (c0 + 1).min(self.resolution - 1);

            // Sample the values at the four surrounding points
            let val_r0c0 = flow_dirs[self.get_index(r0, c0)]; // Top-left
            let val_r0c1 = flow_dirs[self.get_index(r0, c1)]; // Top-right
            let val_r1c0 = flow_dirs[self.get_index(r1, c0)]; // Bottom-left
            let val_r1c1 = flow_dirs[self.get_index(r1, c1)]; // Bottom-right

            // Interpolate along the columns (c-axis) for each row (r0, r1)
            let interp_at_r0 = val_r0c0 * (1.0 - c_frac) + val_r0c1 * c_frac;
            let interp_at_r1 = val_r1c0 * (1.0 - c_frac) + val_r1c1 * c_frac;

            // Interpolate along the rows (r-axis) using the results from column interpolations
            let final_value = interp_at_r0 * (1.0 - r_frac) + interp_at_r1 * r_frac;

            Some(final_value)
        } else {
            None
        }
    }
}

impl Index<ErosionDataType> for ErosionData {
    type Output = [f32];

    fn index(&self, index: ErosionDataType) -> &Self::Output {
        match self.get_data(index) {
            Some(data) => data,
            None => &[],
        }
    }
}

impl IndexMut<ErosionDataType> for ErosionData {
    fn index_mut(&mut self, index: ErosionDataType) -> &mut Self::Output {
        match self.get_data_mut(index) {
            Some(data) => data,
            None => &mut [],
        }
    }
}
