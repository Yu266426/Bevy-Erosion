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

pub mod heightmap;
pub mod terrain_types;

pub fn create_mesh(heightmap: &Heightmap, resolution: u32) -> Mesh {
    let size = 10.0;
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

    pos_attribute.iter_mut().for_each(|pos| {
        pos[1] = heightmap.sample_bilinear((pos[0] * inv_size + 0.5, pos[2] * inv_size + 0.5));
    });

    plane.compute_normals();

    plane
}

pub fn generate_terrain_texture(heightmap: &Heightmap, resolution: usize) -> Vec<u8> {
    let inv_resolution = 1.0 / resolution as f32;

    let mut splat_map_data = vec![0; resolution * resolution * 4];

    let mut min_height = f32::MAX;
    let mut max_height = f32::MIN;
    for row in 0..resolution {
        for col in 0..resolution {
            let norm_coords = (col as f32 * inv_resolution, row as f32 * inv_resolution);

            let h = heightmap.sample_bilinear(norm_coords);
            if h < min_height {
                min_height = h;
            }
            if h > max_height {
                max_height = h;
            }
        }
    }
    let height_range = max_height - min_height;

    for row in 0..resolution {
        for col in 0..resolution {
            let norm_coords = (col as f32 * inv_resolution, row as f32 * inv_resolution);

            let height = heightmap.sample_bilinear(norm_coords);
            let slope = heightmap.get_gradient(norm_coords).length();

            let terrain_type = get_terrain_type((height - min_height) / height_range, slope);

            let pixel_index = (row * resolution + col) * 4;
            let color = get_color(terrain_type);

            splat_map_data[pixel_index] = color.0;
            splat_map_data[pixel_index + 1] = color.1;
            splat_map_data[pixel_index + 2] = color.2;
            splat_map_data[pixel_index + 3] = 255;
        }
    }

    return splat_map_data;
}

pub fn create_terrain_image(heightmap: &Heightmap, resolution: usize) -> Image {
    let texture_resolution = resolution as u32;
    let texture_data = generate_terrain_texture(heightmap, texture_resolution as usize);

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
