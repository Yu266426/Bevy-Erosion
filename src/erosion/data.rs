use crate::common::{Field2D, FloatField2D};
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
    layers: Vec<Field2D<f32>>,
    /// Normalized flow direction vectors
    flow_directions: Option<Field2D<Vec2>>,
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
            layers.push(Field2D::new(resolution));
        }

        let flow_directions = if collect_flow_directions {
            Some(Field2D::new(resolution))
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
                let index = self.layers[layer_index].get_index(row, col);
                self.layers[layer_index].change(index, value);
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
            self.layers[layer_index].record_bilinear(pos, value);
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

            if row >= self.resolution || col >= self.resolution {
                return false;
            }

            let index = flow_dirs.get_index(row, col);

            let current_dir = flow_dirs.data()[index];

            let new_dir = Self::calculate_new_flow_direction(current_dir, direction);

            flow_dirs.data_mut()[index] = new_dir;

            return true;
        }
        false
    }

    /// Helper method to calculate new flow direction based on existing direction and new input
    fn calculate_new_flow_direction(current: Vec2, direction: Vec2) -> Vec2 {
        // Normalize direction
        let dir_norm = direction.normalize_or_zero();
        if dir_norm == Vec2::ZERO {
            return current;
        }

        // Weighted average - give more weight to stronger flows
        let speed = direction.length();
        let current_weight = current.length();
        let new_weight = speed;

        let total_weight = current_weight + new_weight;

        if total_weight > 0.0 {
            (current * current_weight + dir_norm * new_weight) / total_weight
        } else {
            dir_norm
        }
    }

    fn get_layer_index(&self, data_type: ErosionDataType) -> Option<usize> {
        self.enabled_types.iter().position(|&t| t == data_type)
    }

    pub fn get_data(&self, data_type: ErosionDataType) -> Option<&[f32]> {
        self.get_layer_index(data_type)
            .map(|idx| self.layers[idx].data())
    }

    pub fn get_data_mut(&mut self, data_type: ErosionDataType) -> Option<&mut [f32]> {
        self.get_layer_index(data_type)
            .map(move |idx| self.layers[idx].data_mut())
    }

    pub fn get_flow_directions(&self) -> Option<&[Vec2]> {
        self.flow_directions.as_ref().map(|dirs| dirs.data())
    }

    /// Normalizes the data for the specified type to the range [0, 1]
    ///
    /// # Returns
    /// The (min, max) values before normalization
    pub fn normalize(&mut self, data_type: ErosionDataType) -> Option<(f32, f32)> {
        if let Some(layer_index) = self.get_layer_index(data_type) {
            let result = self.layers[layer_index].normalize();
            Some(result)
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
            let value = self.layers[layer_index].sample_bilinear(norm_coords);
            Some(value)
        } else {
            None
        }
    }

    pub fn sample_flow_direction_bilinear(&self, norm_coords: (f32, f32)) -> Option<Vec2> {
        // Early return if we don't have flow directions
        let flow_dirs = self.flow_directions.as_ref()?;

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
        let val_r0c0 = flow_dirs[flow_dirs.get_index(r0, c0)]; // Top-left
        let val_r0c1 = flow_dirs[flow_dirs.get_index(r0, c1)]; // Top-right
        let val_r1c0 = flow_dirs[flow_dirs.get_index(r1, c0)]; // Bottom-left
        let val_r1c1 = flow_dirs[flow_dirs.get_index(r1, c1)]; // Bottom-right

        // Interpolate along the columns (c-axis) for each row (r0, r1)
        let interp_at_r0 = val_r0c0 * (1.0 - c_frac) + val_r0c1 * c_frac;
        let interp_at_r1 = val_r1c0 * (1.0 - c_frac) + val_r1c1 * c_frac;

        // Interpolate along the rows (r-axis) using the results from column interpolations
        let final_value = interp_at_r0 * (1.0 - r_frac) + interp_at_r1 * r_frac;

        Some(final_value)
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
