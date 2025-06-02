use bevy::math::Vec2;
use noisy_bevy::simplex_noise_2d;

pub struct Heightmap {
    resolution: usize,
    scale: f32,
    offset: (f32, f32),
    field: Vec<f32>,
}

impl Heightmap {
    pub fn new(resolution: usize, scale: f32, offset: (f32, f32)) -> Self {
        if resolution == 0 {
            panic!("HeightMap resolution cannot be 0.");
        }

        Self {
            resolution,
            scale,
            offset,
            field: vec![0.0; resolution * resolution],
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn data(&self) -> &[f32] {
        &self.field
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

    #[inline]
    pub fn get_pos(&self, index: usize) -> (usize, usize) {
        debug_assert!(
            index < self.field.len(),
            "Index out of bounds: {} >= {}",
            index,
            self.field.len()
        );
        (index % self.resolution, index / self.resolution) // (col, row)
    }

    #[inline]
    pub fn change_field(&mut self, index: usize, amount: f32) {
        self.field[index] += amount;
    }

    pub fn sample(&self, norm_coords: (f32, f32)) -> f32 {
        let (norm_col, norm_row) = norm_coords;

        // Clamp normalized coordinates to [0.0, 1.0] to handle out-of-range inputs gracefully.
        let norm_row = norm_row.clamp(0.0, 1.0);
        let norm_col = norm_col.clamp(0.0, 1.0);

        // Scale normalized coordinates to grid space.
        // Multiplying by `self.resolution` and casting to `usize` effectively floors the value.
        // E.g., norm_row in [0.0, 1.0) maps to row index in [0, resolution - 1].
        // If norm_row is 1.0, `r` becomes `resolution`, which needs to be clamped.
        let r = (norm_row * (self.resolution - 1) as f32) as usize;
        let c = (norm_col * (self.resolution - 1) as f32) as usize;

        // Clamp grid coordinates to be within `[0, resolution - 1]`.
        let r_clamped = r.min(self.resolution - 1);
        let c_clamped = c.min(self.resolution - 1);

        let index = self.get_index(r_clamped, c_clamped);

        self.field[index]
    }

    pub fn sample_bilinear(&self, norm_coords: (f32, f32)) -> f32 {
        let (norm_col, norm_row) = norm_coords;

        // Clamp normalized inputs to [0.0, 1.0] range.
        let norm_row = norm_row.clamp(0.0, 1.0);
        let norm_col = norm_col.clamp(0.0, 1.0);

        // Scale normalized coordinates to floating-point grid coordinates.
        // This maps the [0.0, 1.0] range to [0.0, resolution - 1.0] range.
        let r_f = norm_row * (self.resolution - 1) as f32;
        let c_f = norm_col * (self.resolution - 1) as f32;

        // Get the integer parts (top-left corner of the cell for interpolation).
        let r0 = r_f.floor() as usize;
        let c0 = c_f.floor() as usize;

        // Fractional parts for interpolation weights.
        let r_frac = r_f - r0 as f32;
        let c_frac = c_f - c0 as f32;

        // Coordinates of the four surrounding grid points.
        // r0, c0 are already <= self.resolution - 1 due to clamping of norm_row/col and scaling.
        // r1 and c1 need to be clamped to self.resolution - 1 to handle edges.
        let r1 = (r0 + 1).min(self.resolution - 1);
        let c1 = (c0 + 1).min(self.resolution - 1);

        // Sample the values at the four surrounding points.
        let val_r0c0 = self.field[self.get_index(r0, c0)]; // Top-left
        let val_r0c1 = self.field[self.get_index(r0, c1)]; // Top-right
        let val_r1c0 = self.field[self.get_index(r1, c0)]; // Bottom-left
        let val_r1c1 = self.field[self.get_index(r1, c1)]; // Bottom-right

        // Interpolate along the columns (c-axis) for each row (r0, r1).
        let interp_at_r0 = val_r0c0 * (1.0 - c_frac) + val_r0c1 * c_frac;
        let interp_at_r1 = val_r1c0 * (1.0 - c_frac) + val_r1c1 * c_frac;

        // Interpolate along the rows (r-axis) using the results from column interpolations.
        let final_value = interp_at_r0 * (1.0 - r_frac) + interp_at_r1 * r_frac;

        final_value
    }

    pub fn get_gradient(&self, norm_coords: (f32, f32)) -> Vec2 {
        if self.resolution <= 1 {
            // For a 0x0 map (which `new` should prevent) or a 1x1 map,
            // the surface is conceptually flat, so the gradient is zero.
            return Vec2::ZERO;
        }

        let (norm_x_center, norm_y_center) = norm_coords;

        let delta_norm = 0.5 / self.resolution as f32;

        let h_x_plus = self.sample_bilinear((norm_x_center + delta_norm, norm_y_center));
        let h_x_minus = self.sample_bilinear((norm_x_center - delta_norm, norm_y_center));

        let gradient_x = (h_x_plus - h_x_minus) / (2.0 * delta_norm);

        let h_y_plus = self.sample_bilinear((norm_x_center, norm_y_center + delta_norm));
        let h_y_minus = self.sample_bilinear((norm_x_center, norm_y_center - delta_norm));

        let gradient_y = (h_y_plus - h_y_minus) / (2.0 * delta_norm);

        Vec2::new(gradient_x, gradient_y)
    }

    pub fn generate(&mut self, octaves: u32, lacunarity: f32, persistence: f32, exp: f32) {
        // Lucanarity usually >= 2.0 (~ How much detail each octave increases)
        // persistence usually ~ 0.5 (~ How much effect each octave has)
        if octaves == 0 {
            panic!("Number of octaves must be at least 1.");
        }

        let inv_resolution = 1.0 / self.resolution as f32;
        let scaled_inv_resolution = inv_resolution * self.scale;

        for x in 0..self.resolution {
            for y in 0..self.resolution {
                let index = self.get_index(x, y);

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

                self.field[index] = final_value;
            }
        }
    }
}

#[inline]
fn sample_noise(x: f32, y: f32) -> f32 {
    return simplex_noise_2d(Vec2::new(x, y));
}
