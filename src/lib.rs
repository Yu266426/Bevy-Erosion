use std::f32::consts::TAU;

use bevy::math::Vec2;
use nanorand::Rng;

pub mod common;
pub mod erosion;
pub mod terrain;

#[allow(dead_code)]
pub(crate) fn random_unit_vec2() -> bevy::math::Vec2 {
    let mut rng = nanorand::tls_rng();

    let angle = rng.generate::<f32>() * TAU;
    let x = angle.cos();
    let y = angle.sin();

    Vec2::new(x, y)
}
