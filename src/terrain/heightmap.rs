use crate::common::{Field2D, FloatField2D};
use bevy::math::Vec2;
use noisy_bevy::simplex_noise_2d;

#[derive(Debug, Clone)]
pub struct Heightmap {
    field: Field2D<f32>,
    offset: (f32, f32),
    scale: f32,
}

impl Heightmap {
    pub fn new(resolution: usize, scale: f32, offset: (f32, f32)) -> Self {
        Self {
            field: Field2D::new(resolution),
            offset,
            scale,
        }
    }

    pub fn resolution(&self) -> usize {
        self.field.resolution()
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn data(&self) -> &[f32] {
        self.field.data()
    }

    #[inline]
    pub fn get_index(&self, row: usize, col: usize) -> usize {
        self.field.get_index(row, col)
    }

    #[inline]
    pub fn get_pos(&self, index: usize) -> (usize, usize) {
        self.field.get_pos(index)
    }

    #[inline]
    pub fn change_field(&mut self, index: usize, amount: f32) {
        self.field.change(index, amount);
    }

    pub fn sample(&self, norm_coords: (f32, f32)) -> f32 {
        self.field.sample(norm_coords)
    }

    pub fn sample_bilinear(&self, norm_coords: (f32, f32)) -> f32 {
        self.field.sample_bilinear(norm_coords)
    }

    pub fn get_gradient(&self, norm_coords: (f32, f32)) -> Vec2 {
        self.field.get_gradient(norm_coords) / self.scale
    }

    /// Generates terrain using multiple octaves of simplex noise
    ///
    /// # Arguments
    /// * `octaves` - Number of noise layers to combine
    /// * `lacunarity` - How much detail increases with each octave (typically >= 2.0)
    /// * `persistence` - How much effect each octave has (typically ~0.5)
    /// * `exp` - Exponent to apply to the final noise value for non-linear height distribution
    ///
    /// # Panics
    /// Panics if octaves is 0
    pub fn generate(&mut self, octaves: u32, lacunarity: f32, persistence: f32, exp: f32) {
        // Lacunarity usually >= 2.0 (~ How much detail each octave increases)
        // persistence usually ~ 0.5 (~ How much effect each octave has)
        if octaves == 0 {
            panic!("Number of octaves must be at least 1.");
        }

        let resolution = self.field.resolution();
        let inv_resolution = 1.0 / resolution as f32;
        let scaled_inv_resolution = inv_resolution * self.scale;

        for x in 0..resolution {
            for y in 0..resolution {
                let index = self.field.get_index(y, x);

                let noise_base_x = x as f32 * scaled_inv_resolution + self.offset.0;
                let noise_base_y = y as f32 * scaled_inv_resolution + self.offset.1;

                let mut total_noise_contribution = 0.0;
                let mut current_frequency_multiplier = 1.0;
                let mut current_amplitude = 1.0;
                let mut sum_of_amplitudes = 0.0;

                for _octave_num in 0..octaves {
                    let noise_sample_x = noise_base_x * current_frequency_multiplier;
                    let noise_sample_y = noise_base_y * current_frequency_multiplier;

                    let raw_noise = sample_noise(noise_sample_x, noise_sample_y);
                    total_noise_contribution += raw_noise * current_amplitude;

                    sum_of_amplitudes += current_amplitude;

                    current_frequency_multiplier *= lacunarity;
                    current_amplitude *= persistence;
                }

                let mut final_value;
                if sum_of_amplitudes > 0.0 {
                    final_value = (total_noise_contribution / sum_of_amplitudes) * 0.5 + 0.5;
                } else {
                    // Edge case, should not usually happen
                    final_value = 0.5;
                }

                final_value = final_value.powf(exp);

                self.field.set(index, final_value);
            }
        }
    }

    pub fn calculate_height_range(&self) -> (f32, f32) {
        self.field.calculate_range()
    }
}

#[inline]
fn sample_noise(x: f32, y: f32) -> f32 {
    return simplex_noise_2d(Vec2::new(x, y));
}
