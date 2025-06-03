use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        mesh::VertexAttributeValues,
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
};
use heightmap::Heightmap;
use terrain_types::{get_color, get_terrain_type};

use crate::erosion::Erosion;

pub mod heightmap;
pub mod terrain_types;

/// Represents a 3D terrain with a 2D heightmap
pub struct Terrain {
    /// The heightmap defines the terrain's elevation
    pub heightmap: Heightmap,
}

impl Terrain {
    pub fn new(resolution: usize, scale: f32, offset: (f32, f32)) -> Self {
        Self {
            heightmap: Heightmap::new(resolution, scale, offset),
        }
    }

    pub fn generate(&mut self, octaves: u32, lacunarity: f32, persistence: f32, exp: f32) {
        self.heightmap
            .generate(octaves, lacunarity, persistence, exp);
    }

    pub fn erode(&mut self, erosion_params: &Erosion, num_particles: usize) {
        erosion_params.erode(&mut self.heightmap, num_particles);
    }

    /// Creates a 3D mesh from the heightmap
    ///
    /// # Arguments
    /// * `resolution` - The resolution of the generated mesh
    /// * `size` - The size of the mesh in world units
    ///
    /// # Returns
    /// A Bevy mesh with vertices positioned according to the heightmap
    ///
    /// # Panics
    /// Panics if the mesh attribute format is unexpected
    pub fn create_mesh(&self, resolution: u32, size: f32) -> Mesh {
        let inv_size = 1.0 / size;

        let mut plane = Mesh::from(
            Plane3d::default()
                .mesh()
                .size(size, size)
                .subdivisions(resolution),
        );

        let pos_attribute = plane.attribute_mut(Mesh::ATTRIBUTE_POSITION).unwrap();
        let VertexAttributeValues::Float32x3(pos_attribute) = pos_attribute else {
            panic!("Unexpected vertex format, expected Float32x3");
        };

        // Apply heightmap to mesh vertices
        pos_attribute.iter_mut().for_each(|pos| {
            pos[1] = self
                .heightmap
                .sample_bilinear((pos[0] * inv_size + 0.5, pos[2] * inv_size + 0.5));
        });

        // Calculate vertex normals based on modified positions
        plane.compute_normals();

        plane
    }

    pub fn generate_terrain_texture(&self, resolution: usize) -> Vec<u8> {
        let inv_resolution = 1.0 / resolution as f32;
        let mut splat_map_data = vec![0; resolution * resolution * 4];

        // Get the height range from the heightmap or calculate it if not cached
        let (min_height, max_height) = match self.heightmap.height_range() {
            Some(range) => range,
            None => {
                let mut min_h = f32::MAX;
                let mut max_h = f32::MIN;

                // Single pass through the heightmap to find min/max
                for &h in self.heightmap.data() {
                    if h < min_h {
                        min_h = h;
                    }
                    if h > max_h {
                        max_h = h;
                    }
                }

                (min_h, max_h)
            }
        };

        let height_range = max_height - min_height;

        // Pre-allocate reusable vectors for multi-threading if added later
        let chunks = splat_map_data.chunks_mut(resolution * 4);

        // Generate texture data
        for (row, chunk) in chunks.enumerate().take(resolution) {
            for col in 0..resolution {
                let norm_coords = (col as f32 * inv_resolution, row as f32 * inv_resolution);

                let height = self.heightmap.sample_bilinear(norm_coords);
                let slope = self.heightmap.get_gradient(norm_coords).length();

                let terrain_type = get_terrain_type((height - min_height) / height_range, slope);
                let color = get_color(terrain_type);

                let pixel_index = col * 4;
                chunk[pixel_index] = color.0;
                chunk[pixel_index + 1] = color.1;
                chunk[pixel_index + 2] = color.2;
                chunk[pixel_index + 3] = 255;
            }
        }

        splat_map_data
    }

    pub fn create_terrain_image(&self, resolution: usize) -> Image {
        let texture_resolution = resolution as u32;
        let texture_data = self.generate_terrain_texture(resolution);

        Image::new(
            Extent3d {
                width: texture_resolution,
                height: texture_resolution,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            texture_data,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::RENDER_WORLD,
        )
    }
}
