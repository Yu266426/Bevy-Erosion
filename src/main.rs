use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    pbr::{Atmosphere, DirectionalLightShadowMap, light_consts::lux},
    picking::backend::ray::RayMap,
    prelude::*,
    render::camera::Exposure,
};
use bevy_erosion::{erosion::Erosion, terrain::Terrain};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

const TERRAIN_SIZE: f32 = 10.0;

fn main() -> AppExit {
    // unsafe {
    //     std::env::set_var("RUST_BACKTRACE", "1");
    // }

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Erosion".into(),
                name: Some("berosion.app".into()),
                resolution: (1400., 1000.).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(PanOrbitCameraPlugin)
        .insert_resource(ClearColor(Color::srgb(0.2, 0.2, 0.2)))
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .insert_resource(AmbientLight {
            // brightness: 15000.0,
            brightness: 50000.0,
            ..default()
        })
        .insert_resource(ErosionSim {
            terrain: Terrain::new(512, 2.0, 5.0, (-0.5, 0.0)),
            max_iterations: 100_000,
            instant: true,
            iterations: 0,
            iteration_step_size: 1000,
        })
        .add_systems(Startup, setup_scene)
        .add_systems(Update, run_sim)
        // .add_systems(Update, check_gradient)
        .run()
}

#[derive(Resource)]
struct ErosionSim {
    /// The terrain being eroded
    pub terrain: Terrain,
    /// Maximum number of iterations to run
    pub max_iterations: usize,
    /// Whether to run all iterations at startup
    pub instant: bool,
    /// Current iteration count
    pub iterations: usize,
    /// Number of iterations to run per update
    pub iteration_step_size: usize,
}

/// Sets up the scene with camera, lighting, and initial terrain
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut erosion_sim: ResMut<ErosionSim>,
) {
    // Setup camera with orbital controls
    setup_camera(&mut commands);

    // Setup lighting
    setup_lighting(&mut commands);

    // Generate and erode terrain
    setup_terrain(
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut images,
        &mut erosion_sim,
    );
}

/// Sets up the camera with orbital controls
fn setup_camera(commands: &mut Commands) {
    commands.spawn((
        PanOrbitCamera {
            focus: Vec3::new(0.0, 100.0, 0.0), // Spawn higher up to avoid sun breaking
            ..default()
        },
        Camera {
            hdr: true,
            ..default()
        },
        Transform::from_xyz(0., 112., 0.),
        Atmosphere::EARTH,
        Exposure::SUNLIGHT,
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
    ));
}

/// Sets up the scene lighting
fn setup_lighting(commands: &mut Commands) {
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(-1.0, 1.0, -1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Sets up the terrain mesh and material
fn setup_terrain(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    erosion_sim: &mut ErosionSim,
) {
    let mut comparison_terrain = Terrain::new(1024, 2.0, 5.0, (-0.5, 0.0));

    // Generate the heightmap using fractal noise
    erosion_sim.terrain.generate(8, 2.0, 0.5, 2.5);
    comparison_terrain.generate(8, 2.0, 0.5, 2.5);

    // Apply erosion if instant mode is enabled
    if erosion_sim.instant {
        let iterations = erosion_sim.max_iterations;
        erosion_sim.terrain.erode(&Erosion::default(), iterations);
    }
    comparison_terrain.erode(&Erosion::default(), erosion_sim.max_iterations);

    // Create terrain texture and material
    // let texture_handle = images.add(erosion_sim.terrain.create_terrain_image(2048));
    // let comparison_texture_handle = images.add(comparison_terrain.create_terrain_image(2048));
    let texture_handle = images.add(erosion_sim.terrain.create_erosion_map(2048).unwrap());
    let comparison_texture_handle =
        images.add(comparison_terrain.create_erosion_map(2048).unwrap());
    // let texture_handle = images.add(
    // erosion_sim
    // .terrain
    // .create_erosion_data_image(bevy_erosion::erosion::ErosionDataType::Wetness, 2048)
    // .unwrap(),
    // );

    let terrain_material = StandardMaterial {
        base_color: Color::WHITE,
        // base_color: Color::srgb(0.6, 0.6, 0.6),
        base_color_texture: Some(texture_handle),
        reflectance: 0.3,
        ..default()
    };
    let comparison_terrain_material = StandardMaterial {
        base_color: Color::WHITE,
        // base_color: Color::srgb(0.6, 0.6, 0.6),
        base_color_texture: Some(comparison_texture_handle),
        reflectance: 0.3,
        ..default()
    };

    // Spawn the terrain entity
    commands.spawn((
        Name::new("Terrain"),
        Mesh3d(meshes.add(erosion_sim.terrain.create_mesh(1024, TERRAIN_SIZE))),
        MeshMaterial3d(materials.add(terrain_material.clone())),
        Transform::from_xyz(0.0, 100.0, 0.0),
    ));
    commands.spawn((
        Name::new("Terrain"),
        Mesh3d(meshes.add(comparison_terrain.create_mesh(2048, TERRAIN_SIZE))),
        MeshMaterial3d(materials.add(comparison_terrain_material.clone())),
        Transform::from_xyz(11.0, 100.0, 0.0),
        // Transform::from_xyz(0.0, 101.0, 0.0),
    ));
}

/// Runs the erosion simulation each frame based on user input
fn run_sim(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    terrain: Single<Entity, With<Mesh3d>>,
    mut erosion_sim: ResMut<ErosionSim>,
    key_input: Res<ButtonInput<KeyCode>>,
) {
    // Skip if in instant mode (all erosion already applied at startup)
    if erosion_sim.instant {
        return;
    }

    // Add more iterations when space is pressed
    if key_input.just_pressed(KeyCode::Space) {
        erosion_sim.max_iterations += 10_000;
    }

    // Stop when we've reached the maximum iterations
    if erosion_sim.iterations >= erosion_sim.max_iterations {
        return;
    }

    // Run erosion simulation for this frame
    run_erosion_step(&mut erosion_sim);

    // Update the terrain mesh and texture
    update_terrain_visuals(
        &mut commands,
        &mut meshes,
        &mut images,
        &mut materials,
        terrain.entity(),
        &erosion_sim,
    );
}

/// Runs a single step of the erosion simulation
fn run_erosion_step(erosion_sim: &mut ErosionSim) {
    let iteration_step_size = erosion_sim.iteration_step_size;
    erosion_sim.iterations += iteration_step_size;
    erosion_sim
        .terrain
        .erode(&Erosion::default(), iteration_step_size);
}

/// Updates the terrain mesh and texture based on the current heightmap
fn update_terrain_visuals(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
    terrain_entity: Entity,
    erosion_sim: &ErosionSim,
) {
    // Create updated mesh from heightmap
    let mesh = erosion_sim.terrain.create_mesh(1024, TERRAIN_SIZE);

    // Create updated texture from heightmap
    let texture_handle = images.add(erosion_sim.terrain.create_terrain_image(2048));

    // Create updated material with the new texture
    let terrain_material = StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(texture_handle),
        reflectance: 0.3,
        ..default()
    };

    // Update the terrain entity with new mesh and material
    commands.entity(terrain_entity).insert((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(terrain_material)),
    ));
}

/// Visualizes terrain gradients at raycast hit points for debugging
/// This function is used for debugging the gradient calculation
#[allow(dead_code)]
fn check_gradient(
    mut ray_cast: MeshRayCast,
    mut gizmos: Gizmos,
    ray_map: Res<RayMap>,
    erosion_sim: Res<ErosionSim>,
) {
    // Constants for visualization
    const SPHERE_RADIUS: f32 = 0.05;
    const SPHERE_COLOR: Color = Color::srgb(0.6, 0.6, 0.8);
    const ARROW_COLOR: Color = Color::srgb(0.7, 0.7, 0.6);
    const GRADIENT_SCALE: f32 = 0.1; // Scale down gradient for visualization
    const STEP_FACTOR: f32 = 20.0;
    const HEIGHT_OFFSET: f32 = 100.0; // Offset to position the terrain in world space

    for (_, ray) in ray_map.iter() {
        // Cast ray and get hit point
        let Some((_, hit)) = ray_cast
            .cast_ray(*ray, &MeshRayCastSettings::default())
            .first()
        else {
            continue;
        };

        // Draw sphere at hit point
        gizmos.sphere(hit.point, SPHERE_RADIUS, SPHERE_COLOR);

        // Calculate terrain coordinates from world position
        let norm_coords = (
            hit.point.x / TERRAIN_SIZE + 0.5,
            hit.point.z / TERRAIN_SIZE + 0.5,
        );

        // Get gradient at hit point
        let gradient = erosion_sim.terrain.heightmap.get_gradient(norm_coords) * GRADIENT_SCALE;

        // Calculate next point along gradient
        let new_norm_coords = (
            (hit.point.x + gradient.x / STEP_FACTOR) / TERRAIN_SIZE + 0.5,
            (hit.point.z + gradient.y / STEP_FACTOR) / TERRAIN_SIZE + 0.5,
        );

        // Sample height at the next point
        let next_height = erosion_sim
            .terrain
            .heightmap
            .sample_bilinear(new_norm_coords)
            * erosion_sim.terrain.height_scale
            + HEIGHT_OFFSET;

        // Calculate direction vector for visualization
        let dir_vector = Vec3::new(
            hit.point.x + gradient.x / STEP_FACTOR,
            next_height,
            hit.point.z + gradient.y / STEP_FACTOR,
        ) - hit.point;

        // Draw arrow showing gradient direction
        gizmos.arrow(hit.point, hit.point + dir_vector * STEP_FACTOR, ARROW_COLOR);
    }
}
