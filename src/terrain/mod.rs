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

use crate::erosion::{Erosion, ErosionData, ErosionDataType};

pub mod heightmap;
pub mod terrain_types;

/// Represents a 3D terrain with a 2D heightmap and erosion data
pub struct Terrain {
    /// The heightmap defines the terrain's elevation
    pub heightmap: Heightmap,
    /// Optional erosion data collected during erosion simulation
    pub erosion_data: Option<ErosionData>,
}

impl Terrain {
    pub fn new(resolution: usize, scale: f32, offset: (f32, f32)) -> Self {
        Self {
            heightmap: Heightmap::new(resolution, scale, offset),
            erosion_data: None,
        }
    }

    pub fn generate(&mut self, octaves: u32, lacunarity: f32, persistence: f32, exp: f32) {
        self.heightmap
            .generate(octaves, lacunarity, persistence, exp);
    }

    pub fn erode(&mut self, erosion_params: &Erosion, num_particles: usize) {
        self.erosion_data = erosion_params.erode(&mut self.heightmap, num_particles);
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

        let Some(ref erosion_data) = self.erosion_data else {
            // TODO: Make basic version with just height and gradient
            panic!("Must collect erosion data to generate texture");
        };
        let mut data_copy = erosion_data.clone();
        data_copy.normalize_all();

        // Get the height range from the heightmap or calculate it if not cached
        let (min_height, max_height) = self.heightmap.calculate_height_range();
        let height_range = max_height - min_height;

        // Pre-allocate reusable vectors for multi-threading if added later
        let chunks = splat_map_data.chunks_mut(resolution * 4);

        // Generate texture data
        for (row, chunk) in chunks.enumerate().take(resolution) {
            for col in 0..resolution {
                let norm_coords = (col as f32 * inv_resolution, row as f32 * inv_resolution);

                let height = self.heightmap.sample_bilinear(norm_coords);
                let slope = self.heightmap.get_gradient(norm_coords).length();

                // Get erosion attributes if available
                let mut wetness = 0.0;
                let mut water_flux = 0.0;
                let mut deposition = 0.0;
                let mut erosion_amount = 0.0;

                if let Some(val) = data_copy.sample_bilinear(ErosionDataType::Wetness, norm_coords)
                {
                    wetness = val;
                }
                if let Some(val) =
                    data_copy.sample_bilinear(ErosionDataType::WaterFlux, norm_coords)
                {
                    water_flux = val;
                }
                if let Some(val) =
                    data_copy.sample_bilinear(ErosionDataType::Deposition, norm_coords)
                {
                    deposition = val;
                }
                if let Some(val) = data_copy.sample_bilinear(ErosionDataType::Erosion, norm_coords)
                {
                    erosion_amount = val;
                }

                let terrain_type = get_terrain_type(
                    (height - min_height) / height_range,
                    slope / self.heightmap.scale(),
                    wetness,
                    water_flux,
                    deposition,
                    erosion_amount,
                );
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

    /// Creates a grayscale texture from a specific erosion data type
    pub fn create_erosion_data_image(
        &self,
        data_type: ErosionDataType,
        resolution: usize,
    ) -> Option<Image> {
        if let Some(ref erosion_data) = self.erosion_data {
            let mut data_copy = erosion_data.clone();
            data_copy.normalize(data_type);

            if let Some(_) = data_copy.get_data(data_type) {
                let texture_resolution = resolution as u32;
                let mut texture_data = vec![0; resolution * resolution * 4];

                for y in 0..resolution {
                    for x in 0..resolution {
                        let norm_coords =
                            (x as f32 / resolution as f32, y as f32 / resolution as f32);

                        if let Some(value) = data_copy.sample_bilinear(data_type, norm_coords) {
                            let pixel_index = (y * resolution + x) * 4;
                            let intensity = (value * 255.0) as u8;

                            texture_data[pixel_index] = intensity;
                            texture_data[pixel_index + 1] = intensity;
                            texture_data[pixel_index + 2] = intensity;
                            texture_data[pixel_index + 3] = 255;
                        }
                    }
                }

                return Some(Image::new(
                    Extent3d {
                        width: texture_resolution,
                        height: texture_resolution,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    texture_data,
                    TextureFormat::Rgba8UnormSrgb,
                    RenderAssetUsages::RENDER_WORLD,
                ));
            }
        }
        None
    }

    /// Creates a flow direction visualization texture
    pub fn create_flow_visualization(&self, resolution: usize) -> Option<Image> {
        if let Some(ref erosion_data) = self.erosion_data {
            let texture_resolution = resolution as u32;
            let mut texture_data = vec![0; resolution * resolution * 4];

            for y in 0..resolution {
                for x in 0..resolution {
                    let norm_coords = (x as f32 / resolution as f32, y as f32 / resolution as f32);

                    if let Some(direction) =
                        erosion_data.sample_flow_direction_bilinear(norm_coords)
                    {
                        let pixel_index = (y * resolution + x) * 4;

                        // Map x direction to red channel (-1 to 1 -> 0 to 255)
                        let red = ((direction.x + 1.0) * 127.5) as u8;
                        // Map y direction to green channel (-1 to 1 -> 0 to 255)
                        let green = ((direction.y + 1.0) * 127.5) as u8;
                        // Map magnitude to blue channel (0 to 1 -> 0 to 255)
                        let blue = (direction.length() * 255.0) as u8;

                        texture_data[pixel_index] = red;
                        texture_data[pixel_index + 1] = green;
                        texture_data[pixel_index + 2] = blue;
                        texture_data[pixel_index + 3] = 255;
                    } else {
                        let pixel_index = (y * resolution + x) * 4;
                        texture_data[pixel_index + 3] = 255; // Set alpha to opaque
                    }
                }
            }

            return Some(Image::new(
                Extent3d {
                    width: texture_resolution,
                    height: texture_resolution,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                texture_data,
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::RENDER_WORLD,
            ));
        }
        None
    }

    /// Creates a combined multi-channel texture from multiple erosion data types
    pub fn create_erosion_map(&self, resolution: usize) -> Option<Image> {
        if let Some(ref erosion_data) = self.erosion_data {
            let mut data_copy = erosion_data.clone();

            // Normalize all the data we'll use
            data_copy.normalize(ErosionDataType::Deposition);
            data_copy.normalize(ErosionDataType::WaterFlux);
            data_copy.normalize(ErosionDataType::Erosion);
            data_copy.normalize(ErosionDataType::Wetness);

            let texture_resolution = resolution as u32;
            let mut texture_data = vec![0; resolution * resolution * 4];

            for y in 0..resolution {
                for x in 0..resolution {
                    let norm_coords = (x as f32 / resolution as f32, y as f32 / resolution as f32);

                    // Red channel: Deposition
                    let deposition = data_copy
                        .sample_bilinear(ErosionDataType::Deposition, norm_coords)
                        .unwrap_or(0.0);

                    // Green channel: WaterFlux
                    let water_flux = data_copy
                        .sample_bilinear(ErosionDataType::WaterFlux, norm_coords)
                        .unwrap_or(0.0);

                    // Blue channel: Erosion
                    let erosion = data_copy
                        .sample_bilinear(ErosionDataType::Erosion, norm_coords)
                        .unwrap_or(0.0);

                    // Alpha channel: Wetness (or just 255 for fully opaque)
                    let wetness = data_copy
                        .sample_bilinear(ErosionDataType::Wetness, norm_coords)
                        .unwrap_or(0.0);

                    let pixel_index = (y * resolution + x) * 4;
                    texture_data[pixel_index] = (deposition * 255.0) as u8;
                    texture_data[pixel_index + 1] = (water_flux * 255.0) as u8;
                    texture_data[pixel_index + 2] = (erosion * 255.0) as u8;
                    texture_data[pixel_index + 3] = (wetness * 255.0) as u8;
                }
            }

            return Some(Image::new(
                Extent3d {
                    width: texture_resolution,
                    height: texture_resolution,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                texture_data,
                TextureFormat::Rgba8Unorm, // Use Unorm instead of UnormSrgb for data textures
                RenderAssetUsages::RENDER_WORLD,
            ));
        }
        None
    }
}
