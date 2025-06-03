use bevy::math::Vec2;

#[derive(Debug)]
pub struct Particle {
    pub pos: Vec2,
    pub dir: Vec2,
    pub speed: f32,
    pub water: f32,
    pub sediment: f32,
}

impl Particle {
    pub fn new(pos: Vec2, dir: Vec2, initial_speed: f32, initial_water: f32) -> Self {
        Self {
            pos,
            dir: dir.normalize_or_zero(),
            speed: initial_speed,
            water: initial_water,
            sediment: 0.0,
        }
    }
}
