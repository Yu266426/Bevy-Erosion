//! Constants used in the erosion simulation

/// Constants used in the erosion algorithm
pub struct ErosionConstants;

impl ErosionConstants {
    /// Minimum speed threshold for particles before they stop moving
    pub const MIN_SPEED: f32 = 1e-4;

    /// Minimum squared length of direction vector before considering it zero
    pub const MIN_DIR_LENGTH_SQUARED: f32 = 1e-6;
}
