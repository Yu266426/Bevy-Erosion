use bevy::math::{FloatPow, Vec2};
use nanorand::{Rng, WyRand};
use particle::Particle;

use crate::terrain::heightmap::Heightmap;

mod constants;
pub mod data;
mod particle;

pub use constants::ErosionConstants;
pub use data::{ErosionData, ErosionDataType};

#[derive(Debug, Clone)]
pub struct Erosion {
    pub erosion: f32,
    pub inertia: f32,
    pub capacity: f32,
    pub evaporation: f32,
    pub deposition: f32,
    pub min_slope: f32,
    pub radius: i32,
    pub gravity: f32,
    pub flat_threshold: f32,
    pub pool_friction: f32,
    pub initial_speed: f32,
    pub initial_water: f32,
    pub max_steps_for_particle: usize,
    pub collect_erosion_data: bool,
}

impl Default for Erosion {
    fn default() -> Self {
        Self {
            erosion: 0.7,
            inertia: 0.1,
            capacity: 6.0,
            evaporation: 0.02,
            deposition: 0.1,
            min_slope: 0.0001,
            radius: 2,
            gravity: 1.0,
            flat_threshold: 0.04,
            pool_friction: 0.9,
            initial_speed: 0.9,
            initial_water: 1.0,
            max_steps_for_particle: 64,
            collect_erosion_data: true,
        }
    }
}

impl Erosion {
    pub fn erode(&self, heightmap: &mut Heightmap, num_particles: usize) -> Option<ErosionData> {
        let mut rng = WyRand::new();

        let mut erosion_data = if self.collect_erosion_data {
            Some(ErosionData::comprehensive(heightmap.resolution()))
        } else {
            None
        };

        let mut tuned_self = self.clone();

        let heightmap_scale_factor = heightmap.resolution() as f32 / 512.0;

        tuned_self.initial_water *= heightmap_scale_factor;
        tuned_self.capacity *= heightmap_scale_factor;
        tuned_self.radius = ((self.radius as f32) * heightmap_scale_factor).round() as i32;
        tuned_self.max_steps_for_particle =
            ((self.max_steps_for_particle as f32) * heightmap_scale_factor).round() as usize;

        if heightmap_scale_factor.abs() > 1e-6 {
            tuned_self.min_slope /= heightmap_scale_factor;
            tuned_self.flat_threshold /= heightmap_scale_factor;
        }

        for _ in 0..num_particles {
            tuned_self.erode_particle(&mut rng, heightmap, erosion_data.as_mut());
        }

        erosion_data
    }

    pub fn erode_particle(
        &self,
        rng: &mut WyRand,
        heightmap: &mut Heightmap,
        mut erosion_data: Option<&mut ErosionData>,
    ) {
        let resolution = (heightmap.resolution() - 1) as f32;
        let inv_resolution = 1.0 / resolution;

        let starting_x = rng.generate::<f32>();
        let starting_y = rng.generate::<f32>();

        let mut p = Particle::new(
            Vec2::new(starting_x * resolution, starting_y * resolution),
            Vec2::ZERO,
            self.initial_speed,
            self.initial_water,
        );

        for _ in 0..self.max_steps_for_particle {
            let prev_pos = p.pos;
            let prev_norm_coords = (p.pos.x * inv_resolution, p.pos.y * inv_resolution);

            let prev_height = heightmap.sample_bilinear(prev_norm_coords);
            let gradient = heightmap.get_gradient(prev_norm_coords) / heightmap.scale();

            p.dir = (p.dir * self.inertia - gradient * (1.0 - self.inertia)).normalize_or_zero();

            let movement = (p.dir * p.speed).clamp_length_max(1.0);
            p.pos += movement;

            // Record flow data if enabled
            if let Some(data) = erosion_data.as_mut() {
                data.record_bilinear(ErosionDataType::WaterFlux, &prev_pos, p.water * p.speed);
                data.record_bilinear(ErosionDataType::FlowSpeed, &prev_pos, p.speed);
                data.record_flow_direction(&prev_pos, movement);
                data.record_bilinear(ErosionDataType::ParticleDensity, &prev_pos, 1.0);
                data.record_bilinear(ErosionDataType::Wetness, &prev_pos, p.water);
            }

            if p.dir.length_squared() <= ErosionConstants::MIN_DIR_LENGTH_SQUARED
                || p.speed < ErosionConstants::MIN_SPEED
                || p.pos.x <= 0.0
                || p.pos.x >= resolution
                || p.pos.y <= 0.0
                || p.pos.y >= resolution
            {
                break;
            }

            let norm_coords = (p.pos.x * inv_resolution, p.pos.y * inv_resolution);
            let new_height = heightmap.sample_bilinear(norm_coords);

            let height_diff = new_height - prev_height;

            let carry_capacity =
                self.min_slope.max(-height_diff) * p.speed * p.water * self.capacity;

            // If carrying more than capacity or going uphill, drop some sediment
            if height_diff > 0.0 || p.sediment > carry_capacity {
                let amount_to_deposit = if height_diff > 0.0 {
                    p.sediment.min(height_diff)
                } else {
                    (p.sediment - carry_capacity) * self.deposition
                };

                p.sediment -= amount_to_deposit;
                Self::deposit_heightmap(heightmap, &prev_pos, amount_to_deposit);

                // Record deposition data if enabled
                if let Some(data) = erosion_data.as_mut() {
                    data.record_bilinear(ErosionDataType::Deposition, &prev_pos, amount_to_deposit);
                }
            }
            // Otherwise, take up some sediment
            else {
                let amount_to_erode =
                    ((carry_capacity - p.sediment) * self.erosion).min(-height_diff);

                p.sediment += amount_to_erode;
                Self::erode_heightmap(heightmap, &prev_pos, amount_to_erode, self.radius);

                // Record erosion data if enabled
                if let Some(data) = erosion_data.as_mut() {
                    data.record_bilinear(ErosionDataType::Erosion, &prev_pos, amount_to_erode);
                }
            }

            p.speed = (p.speed.squared() + height_diff * self.gravity)
                .max(0.000001)
                .sqrt();
            if gradient.length() < self.flat_threshold {
                p.speed *= self.pool_friction;
            }

            p.water *= 1.0 - self.evaporation;
        }
    }

    fn deposit_heightmap(heightmap: &mut Heightmap, pos: &Vec2, amount: f32) {
        if pos.x < 0.0 || pos.y < 0.0 {
            return;
        }

        let col = pos.x as usize;
        let row = pos.y as usize;

        let u = pos.x.fract();
        let v = pos.y.fract();

        let map_resolution = heightmap.resolution() - 1;

        if col >= map_resolution || row >= map_resolution {
            return;
        }

        heightmap.change_field(
            heightmap.get_index(row, col),
            amount * (1.0 - u) * (1.0 - v),
        );

        let can_affect_col_plus_1 = col + 1 < map_resolution;
        let can_affect_row_plus_1 = row + 1 < map_resolution;

        if can_affect_row_plus_1 {
            heightmap.change_field(heightmap.get_index(row + 1, col), amount * (1.0 - u) * v);
        }

        if can_affect_col_plus_1 {
            heightmap.change_field(heightmap.get_index(row, col + 1), amount * u * (1.0 - v));
        }

        if can_affect_row_plus_1 && can_affect_col_plus_1 {
            heightmap.change_field(heightmap.get_index(row + 1, col + 1), amount * u * v);
        }
    }

    pub fn erode_heightmap(heightmap: &mut Heightmap, pos: &Vec2, amount: f32, radius: i32) {
        // Simple case
        if radius < 1 {
            Self::deposit_heightmap(heightmap, pos, -amount);
            return;
        }

        let map_resolution = heightmap.resolution() - 1;

        let x_start = pos.x as i32 - radius;
        let y_start = pos.y as i32 - radius;

        let x_end = (map_resolution as i32).min(x_start + 2 * radius + 1);
        let y_end = (map_resolution as i32).min(y_start + 2 * radius + 1);

        let x_start = x_start.max(0);
        let y_start = y_start.max(0);

        // Construct erosion kernel
        let mut kernel = vec![vec![0.0; (2 * radius + 1) as usize]; (2 * radius + 1) as usize];
        let mut kernel_sum = 0.0;
        for y in y_start..y_end {
            for x in x_start..x_end {
                let d_x = x as f32 - pos.x;
                let d_y = y as f32 - pos.y;

                let distance = (d_x.squared() + d_y.squared()).sqrt();

                let w = 0.0_f32.max(radius as f32 - distance);

                kernel_sum += w;
                kernel[(y - y_start) as usize][(x - x_start) as usize] = w;
            }
        }

        // Normalise weights, then erode
        for y in y_start..y_end {
            for x in x_start..x_end {
                let kernel_val =
                    kernel[(y - y_start) as usize][(x - x_start) as usize] / kernel_sum;

                heightmap.change_field(
                    heightmap.get_index(y as usize, x as usize),
                    -amount * kernel_val,
                );
            }
        }
    }
}
