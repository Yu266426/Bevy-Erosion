#[derive(Debug, Clone, Copy)]
pub enum TerrainType {
    Rock,
    Grass,
    Water,
    Snow,
}

pub fn get_terrain_type(norm_height: f32, slope: f32) -> TerrainType {
    if norm_height > 0.6 && slope < 8.0 {
        TerrainType::Snow
    } else if slope > 2.4 {
        TerrainType::Rock
    } else if norm_height < 0.2 && slope < 0.15 {
        TerrainType::Water
    } else {
        TerrainType::Grass
    }
}

pub fn get_color(terrain_type: TerrainType) -> (u8, u8, u8) {
    match terrain_type {
        TerrainType::Rock => (114, 121, 122),
        TerrainType::Grass => (47, 125, 27),
        TerrainType::Water => (60, 132, 214),
        TerrainType::Snow => (230, 230, 230),
    }
}
