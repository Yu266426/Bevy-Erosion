//! Generic 2D field abstraction
//!
//! This module provides a reusable abstraction for 2D grid data,
//! which can be used by both the heightmap and erosion data components.

use bevy::math::Vec2;
use std::ops::{Index, IndexMut};

/// A generic 2D field of values with utility functions for sampling and manipulation.
///
/// This structure provides common functionality for working with 2D grid data,
/// including indexing, coordinate conversion, and interpolated sampling.
#[derive(Debug, Clone)]
pub struct Field2D<T> {
    /// The resolution of the field (width and height)
    resolution: usize,
    /// The actual data storage, laid out in row-major order
    data: Vec<T>,
}

impl<T: Clone + Copy + Default> Field2D<T> {
    /// Creates a new field with the specified resolution
    ///
    /// # Arguments
    /// * `resolution` - The width and height of the field in grid cells
    /// * `scale` - Optional scale factor for gradient calculations
    ///
    /// # Panics
    /// Panics if resolution is 0
    pub fn new(resolution: usize) -> Self {
        if resolution == 0 {
            panic!("Field2D resolution cannot be 0.");
        }

        Self {
            resolution,
            data: vec![T::default(); resolution * resolution],
        }
    }

    /// Creates a new field with the specified resolution and initial value
    ///
    /// # Arguments
    /// * `resolution` - The width and height of the field in grid cells
    /// * `initial_value` - The value to fill the field with
    /// * `scale` - Optional scale factor for gradient calculations
    ///
    /// # Panics
    /// Panics if resolution is 0
    pub fn new_with_value(resolution: usize, initial_value: T) -> Self {
        if resolution == 0 {
            panic!("Field2D resolution cannot be 0.");
        }

        Self {
            resolution,
            data: vec![initial_value; resolution * resolution],
        }
    }

    /// Returns the resolution of the field
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Returns a reference to the underlying data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Calculates the index in the data array for the given row and column
    ///
    /// # Arguments
    /// * `row` - The row index
    /// * `col` - The column index
    ///
    /// # Returns
    /// The index in the data array
    ///
    /// # Panics
    /// In debug mode, panics if row or col are out of bounds
    #[inline]
    pub fn get_index(&self, row: usize, col: usize) -> usize {
        if cfg!(debug_assertions) {
            assert!(
                row < self.resolution,
                "Row index out of bounds: {} >= {}",
                row,
                self.resolution
            );
            assert!(
                col < self.resolution,
                "Column index out of bounds: {} >= {}",
                col,
                self.resolution
            );
        }
        row * self.resolution + col
    }

    /// Converts an index to a (row, col) position
    ///
    /// # Arguments
    /// * `index` - The index in the data array
    ///
    /// # Returns
    /// A tuple of (row, col)
    ///
    /// # Panics
    /// In debug mode, panics if index is out of bounds
    #[inline]
    pub fn get_pos(&self, index: usize) -> (usize, usize) {
        debug_assert!(
            index < self.data.len(),
            "Index out of bounds: {} >= {}",
            index,
            self.data.len()
        );
        (index / self.resolution, index % self.resolution) // (row, col)
    }

    /// Changes the value at the specified index by the given amount
    ///
    /// # Arguments
    /// * `index` - The index in the data array
    /// * `value` - The new value to set
    pub fn set(&mut self, index: usize, value: T) {
        self.data[index] = value;
    }
}

impl<T: Clone + Copy + std::ops::Add<Output = T>> Field2D<T> {
    /// Changes the value at the specified index by adding the given amount
    ///
    /// # Arguments
    /// * `index` - The index in the data array
    /// * `amount` - The amount to add to the current value
    pub fn change(&mut self, index: usize, amount: T) {
        self.data[index] = self.data[index] + amount;
    }
}

impl<T: Clone + Copy + Default> Index<usize> for Field2D<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Clone + Copy + Default> IndexMut<usize> for Field2D<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Extension trait for Field2D<f32> with floating-point specific operations
pub trait FloatField2D {
    /// Samples the field at the given normalized coordinates
    fn sample(&self, norm_coords: (f32, f32)) -> f32;

    /// Samples the field at the given normalized coordinates using bilinear interpolation
    fn sample_bilinear(&self, norm_coords: (f32, f32)) -> f32;

    /// Gets the gradient of the field at the given normalized coordinates
    fn get_gradient(&self, norm_coords: (f32, f32)) -> Vec2;

    /// Records a value at the given position with bilinear interpolation
    fn record_bilinear(&mut self, pos: &Vec2, value: f32);

    /// Calculates the min and max values in the field
    fn calculate_range(&self) -> (f32, f32);

    /// Normalizes the field values to the range [0, 1]
    fn normalize(&mut self) -> (f32, f32);
}

impl FloatField2D for Field2D<f32> {
    fn sample(&self, norm_coords: (f32, f32)) -> f32 {
        let (norm_col, norm_row) = norm_coords;

        // Clamp normalized coordinates to [0.0, 1.0] to handle out-of-range inputs gracefully
        let norm_row = norm_row.clamp(0.0, 1.0);
        let norm_col = norm_col.clamp(0.0, 1.0);

        // Scale normalized coordinates to grid space
        let r = (norm_row * (self.resolution - 1) as f32) as usize;
        let c = (norm_col * (self.resolution - 1) as f32) as usize;

        // Clamp grid coordinates to be within [0, resolution - 1]
        let r_clamped = r.min(self.resolution - 1);
        let c_clamped = c.min(self.resolution - 1);

        let index = self.get_index(r_clamped, c_clamped);
        self.data[index]
    }

    fn sample_bilinear(&self, norm_coords: (f32, f32)) -> f32 {
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
        let val_r0c0 = self.data[self.get_index(r0, c0)]; // Top-left
        let val_r0c1 = self.data[self.get_index(r0, c1)]; // Top-right
        let val_r1c0 = self.data[self.get_index(r1, c0)]; // Bottom-left
        let val_r1c1 = self.data[self.get_index(r1, c1)]; // Bottom-right

        // Interpolate along the columns (c-axis) for each row (r0, r1)
        let interp_at_r0 = val_r0c0 * (1.0 - c_frac) + val_r0c1 * c_frac;
        let interp_at_r1 = val_r1c0 * (1.0 - c_frac) + val_r1c1 * c_frac;

        // Interpolate along the rows (r-axis) using the results from column interpolations
        interp_at_r0 * (1.0 - r_frac) + interp_at_r1 * r_frac
    }

    fn get_gradient(&self, norm_coords: (f32, f32)) -> Vec2 {
        if self.resolution <= 1 {
            // For a 0x0 map (which `new` should prevent) or a 1x1 map,
            // the surface is conceptually flat, so the gradient is zero
            return Vec2::ZERO;
        }

        let (norm_x_center, norm_y_center) = norm_coords;

        // Use a delta proportional to the resolution for better accuracy
        let delta_norm = 0.5 / self.resolution as f32;

        // Calculate heights at neighboring points for central difference approximation
        let h_x_plus = self.sample_bilinear((norm_x_center + delta_norm, norm_y_center));
        let h_x_minus = self.sample_bilinear((norm_x_center - delta_norm, norm_y_center));
        let h_y_plus = self.sample_bilinear((norm_x_center, norm_y_center + delta_norm));
        let h_y_minus = self.sample_bilinear((norm_x_center, norm_y_center - delta_norm));

        // Calculate gradients using central difference formula
        let gradient_x = (h_x_plus - h_x_minus) / (2.0 * delta_norm);
        let gradient_y = (h_y_plus - h_y_minus) / (2.0 * delta_norm);

        // Apply scale if available
        Vec2::new(gradient_x, gradient_y)
    }

    fn record_bilinear(&mut self, pos: &Vec2, value: f32) {
        let col_f = pos.x;
        let row_f = pos.y;

        // Ensure position is within bounds
        if col_f < 0.0
            || row_f < 0.0
            || col_f >= self.resolution as f32 - 1.0
            || row_f >= self.resolution as f32 - 1.0
        {
            return;
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

        self.data[idx00] += value * w00;
        self.data[idx01] += value * w01;
        self.data[idx10] += value * w10;
        self.data[idx11] += value * w11;
    }

    fn calculate_range(&self) -> (f32, f32) {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for &val in &self.data {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        (min_val, max_val)
    }

    fn normalize(&mut self) -> (f32, f32) {
        // Find min and max values
        let (min_val, max_val) = self.calculate_range();

        // Normalize if range is non-zero
        let range = max_val - min_val;
        if range > 1e-6 {
            for val in self.data.iter_mut() {
                *val = (*val - min_val) / range;
            }
        } else if min_val != 0.0 {
            // If all values are the same but not zero, set them to 1.0
            for val in self.data.iter_mut() {
                *val = 1.0;
            }
        }

        (min_val, max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::EPSILON;

    #[test]
    fn test_field2d_creation() {
        let field = Field2D::<f32>::new(10);
        assert_eq!(field.resolution(), 10);
        assert_eq!(field.data().len(), 100);

        // Check default values
        for &val in field.data() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_field2d_new_with_value() {
        let field = Field2D::<f32>::new_with_value(5, 1.5);
        assert_eq!(field.resolution(), 5);
        assert_eq!(field.data().len(), 25);

        // Check initial values
        for &val in field.data() {
            assert_eq!(val, 1.5);
        }
    }

    #[test]
    #[should_panic(expected = "Field2D resolution cannot be 0")]
    fn test_field2d_zero_resolution() {
        // This should panic
        let _ = Field2D::<f32>::new(0);
    }

    #[test]
    fn test_index_calculation() {
        let field = Field2D::<f32>::new(10);

        // Check index calculations
        assert_eq!(field.get_index(0, 0), 0);
        assert_eq!(field.get_index(0, 5), 5);
        assert_eq!(field.get_index(1, 0), 10);
        assert_eq!(field.get_index(5, 5), 55);

        // Check position calculations
        assert_eq!(field.get_pos(0), (0, 0));
        assert_eq!(field.get_pos(5), (0, 5));
        assert_eq!(field.get_pos(10), (1, 0));
        assert_eq!(field.get_pos(55), (5, 5));
    }

    #[test]
    fn test_set_and_get() {
        let mut field = Field2D::<f32>::new(5);

        // Set values
        let idx1 = field.get_index(1, 2);
        let idx2 = field.get_index(3, 4);
        field.set(idx1, 3.14);
        field.set(idx2, 2.71);

        // Check values directly
        assert_eq!(field[idx1], 3.14);
        assert_eq!(field[idx2], 2.71);

        // Use IndexMut
        let idx3 = field.get_index(0, 0);
        field[idx3] = 1.23;
        assert_eq!(field[idx3], 1.23);
    }

    #[test]
    fn test_change() {
        let mut field = Field2D::<f32>::new(5);

        // Set initial values
        field.set(field.get_index(1, 2), 3.0);

        // Change values
        field.change(field.get_index(1, 2), 2.0);

        // Check result
        assert_eq!(field[field.get_index(1, 2)], 5.0);
    }

    #[test]
    fn test_sample() {
        let mut field = Field2D::<f32>::new(5);

        // Set up a simple height gradient
        for row in 0..5 {
            for col in 0..5 {
                field.set(field.get_index(row, col), (row + col) as f32);
            }
        }

        // Test exact sampling (should match the value at the nearest grid point)
        assert_eq!(field.sample((0.0, 0.0)), 0.0);
        assert_eq!(field.sample((1.0, 0.0)), 4.0);
        assert_eq!(field.sample((0.0, 1.0)), 4.0);
        assert_eq!(field.sample((1.0, 1.0)), 8.0);

        // Test clamping
        assert_eq!(field.sample((-0.5, -0.5)), 0.0);
        assert_eq!(field.sample((1.5, 1.5)), 8.0);
    }

    #[test]
    fn test_sample_bilinear() {
        let mut field = Field2D::<f32>::new(3);

        // Set up a simple pattern
        // [0 1 2]
        // [1 2 3]
        // [2 3 4]
        field.set(field.get_index(0, 0), 0.0);
        field.set(field.get_index(0, 1), 1.0);
        field.set(field.get_index(0, 2), 2.0);
        field.set(field.get_index(1, 0), 1.0);
        field.set(field.get_index(1, 1), 2.0);
        field.set(field.get_index(1, 2), 3.0);
        field.set(field.get_index(2, 0), 2.0);
        field.set(field.get_index(2, 1), 3.0);
        field.set(field.get_index(2, 2), 4.0);

        // Test exact grid points
        assert_eq!(field.sample_bilinear((0.0, 0.0)), 0.0);
        assert_eq!(field.sample_bilinear((1.0, 1.0)), 4.0);

        // Test mid-points (should be average of surrounding values)
        assert!((field.sample_bilinear((0.5, 0.0)) - 0.5).abs() < EPSILON);
        assert!((field.sample_bilinear((0.0, 0.5)) - 0.5).abs() < EPSILON);
        assert!((field.sample_bilinear((0.5, 0.5)) - 1.25).abs() < EPSILON);
    }

    #[test]
    fn test_gradient() {
        let mut field = Field2D::<f32>::new(5);

        // Set up a simple gradient field
        // In x-direction: increases by 1.0 per cell
        // In y-direction: increases by 0.5 per cell
        for row in 0..5 {
            for col in 0..5 {
                field.set(field.get_index(row, col), col as f32 + (row as f32 * 0.5));
            }
        }

        // Check gradient
        let grad = field.get_gradient((0.5, 0.5));
        assert!((grad.x - 1.0).abs() < 0.1); // Should be approximately 1.0
        assert!((grad.y - 0.5).abs() < 0.1); // Should be approximately 0.5
    }

    #[test]
    fn test_record_bilinear() {
        let mut field = Field2D::<f32>::new(3);

        // Record a value at the center of the field
        field.record_bilinear(&Vec2::new(1.0, 1.0), 1.0);

        // Check that the value was distributed to all corners
        assert!((field[field.get_index(1, 1)] - 1.0).abs() < EPSILON);

        // Record at a position that should distribute to 4 cells
        field = Field2D::<f32>::new(3);
        field.record_bilinear(&Vec2::new(0.5, 0.5), 1.0);

        // Each corner should get part of the value
        assert!((field[field.get_index(0, 0)] - 0.25).abs() < EPSILON);
        assert!((field[field.get_index(0, 1)] - 0.25).abs() < EPSILON);
        assert!((field[field.get_index(1, 0)] - 0.25).abs() < EPSILON);
        assert!((field[field.get_index(1, 1)] - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_normalize() {
        let mut field = Field2D::<f32>::new(3);

        // Set values from 1.0 to 9.0
        for i in 0..9 {
            field.set(i, (i + 1) as f32);
        }

        // Normalize
        let (min, max) = field.normalize();

        // Check min/max
        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);

        // Check normalized values
        assert!((field[0] - 0.0).abs() < EPSILON);
        assert!((field[4] - 0.5).abs() < EPSILON);
        assert!((field[8] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_same_values() {
        let mut field = Field2D::<f32>::new(2);

        // Set all values to 5.0
        for i in 0..4 {
            field.set(i, 5.0);
        }

        // Normalize
        let (min, max) = field.normalize();

        // Check min/max
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);

        // All values should be set to 1.0 since they were non-zero
        for i in 0..4 {
            assert_eq!(field[i], 1.0);
        }
    }

    #[test]
    fn test_calculate_range() {
        let mut field = Field2D::<f32>::new(3);

        // Set values
        field.set(0, -1.0);
        field.set(4, 10.0);
        field.set(8, 5.0);

        // Calculate range
        let (min, max) = field.calculate_range();

        // Check min/max
        assert_eq!(min, -1.0);
        assert_eq!(max, 10.0);
    }
}
