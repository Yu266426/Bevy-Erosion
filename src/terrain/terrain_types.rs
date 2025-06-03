#[derive(Debug, Clone, Copy)]
pub enum TerrainType {
    Rock,
    Grass,
    Water,
    Snow,
    Sand,
    Mud,
    RiverBed,
    Sediment,
}

pub fn get_terrain_type(
    norm_height: f32,
    norm_slope: f32,
    wetness: f32,
    water_flux: f32,
    deposition: f32,
    erosion_amount: f32,
) -> TerrainType {
    // Water is determined by height and flatness
    if norm_height < 0.2 && norm_slope < 0.15 {
        // High water flux indicates active water flow (river bed)
        if water_flux > 0.3 {
            TerrainType::RiverBed
        } else {
            TerrainType::Water
        }
    } else if water_flux > 0.25 {
        TerrainType::RiverBed
    }
    // Snow on high elevations with gentle slopes
    else if norm_height > 0.6 && norm_slope < 8.0 {
        TerrainType::Snow
    }
    // Steep slopes or erosion indicate rock
    else if norm_slope > 2.4 || erosion_amount > 0.3 {
        TerrainType::Rock
    }
    // Sand in areas with high deposition near water
    else if deposition > 0.5 && norm_height < 0.3 {
        TerrainType::Sand
    }
    // Mud in wet low areas
    else if wetness > 0.15 && norm_height < 0.4 {
        TerrainType::Mud
    }
    // Sediment in areas with moderate deposition
    else if deposition > 0.4 {
        TerrainType::Sediment
    }
    // Default to grass
    else {
        TerrainType::Grass
    }
}

pub fn get_color(terrain_type: TerrainType) -> (u8, u8, u8) {
    match terrain_type {
        TerrainType::Rock => (114, 121, 122),
        TerrainType::Grass => (47, 125, 27),
        TerrainType::Water => (60, 132, 214),
        TerrainType::Snow => (230, 230, 230),
        TerrainType::Sand => (225, 212, 167),
        TerrainType::Mud => (94, 75, 47),
        TerrainType::RiverBed => (85, 108, 137),
        TerrainType::Sediment => (190, 164, 135),
    }
}
