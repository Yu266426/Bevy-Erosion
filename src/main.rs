use std::env;

use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    pbr::{Atmosphere, DirectionalLightShadowMap, light_consts::lux},
    picking::backend::ray::RayMap,
    prelude::*,
    render::{camera::Exposure, mesh::VertexAttributeValues},
};
use bevy_erosion::{erosion::Erosion, heightmap::Heightmap};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

fn main() -> AppExit {
    unsafe {
        env::set_var("RUST_BACKTRACE", "1");
    }

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
            brightness: 15000.0,
            ..default()
        })
        .insert_resource(ErosionSim {
            heightmap: Heightmap::new(512, 2.0, (0.5, 0.5)),
            max_iterations: 100_000,
            // max_iterations: 0,
            iterations: 1,
        })
        .add_systems(Startup, setup_scene)
        .add_systems(Update, run_sim)
        // .add_systems(Update, check_gradient)
        .run()
}

#[derive(Resource)]
struct ErosionSim {
    pub max_iterations: usize,
    pub iterations: usize,
    pub heightmap: Heightmap,
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut erosion_sim: ResMut<ErosionSim>,
) {
    commands.spawn((
        PanOrbitCamera {
            focus: Vec3::new(0.0, 100.0, 0.0), // Spawn higher up to avoid sun breaking.
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

    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        // Transform::from_xyz(1.0, 1.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
        Transform::from_xyz(-1.0, 1.0, -1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let terrain_material = StandardMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.6),
        perceptual_roughness: 0.75,

        ..default()
    };

    erosion_sim.heightmap.generate(8, 2.0, 0.5, 2.5);
    // Erosion {
    //     initial_water: 2.0,
    //     max_steps_for_particle: 128,
    //     ..default()
    // }
    // .erode(&mut erosion_sim.heightmap, 100_000);

    // let mut rng = nanorand::WyRand::new();
    // for i in 0..100 {
    //     let positions = Erosion::default().erode_particle(&mut rng, &mut erosion_sim.heightmap);

    //     if i % 8 != 0 {
    //         continue;
    //     }

    //     commands.spawn((
    //         Mesh3d(meshes.add(Sphere::new(0.01))),
    //         MeshMaterial3d(materials.add(Color::srgb(0.6, 0.8, 0.6))),
    //         Transform::from_xyz(
    //             positions[0].x * 10.0 - 5.0,
    //             positions[0].y + 100.0,
    //             positions[0].z * 10.0 - 5.0,
    //         ),
    //     ));

    //     let mut i = 0;
    //     for position in &positions[1..] {
    //         i += 1;
    //         if i % 3 != 0 {
    //             continue;
    //         }

    //         commands.spawn((
    //             Mesh3d(meshes.add(Sphere::new(0.008))),
    //             MeshMaterial3d(materials.add(Color::srgb(0.6, 0.6, 0.8))),
    //             Transform::from_xyz(
    //                 position.x * 10.0 - 5.0,
    //                 position.y + 100.0,
    //                 position.z * 10.0 - 5.0,
    //             ),
    //         ));
    //     }
    // }

    commands.spawn((
        Name::new("Terrain"),
        Mesh3d(meshes.add(create_mesh(&erosion_sim.heightmap, 1024))),
        MeshMaterial3d(materials.add(terrain_material.clone())),
        Transform::from_xyz(0.0, 100.0, 0.0),
    ));
}

fn create_mesh(heightmap: &Heightmap, resolution: u32) -> Mesh {
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
        pos[1] = heightmap
            .sample_bilinear((pos[0] * inv_size + 0.5, pos[2] * inv_size + 0.5))
            .clamp(-0.5, 1.5);
    });

    plane.compute_normals();

    plane
}

fn run_sim(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    terrain: Single<Entity, With<Mesh3d>>,
    mut erosion_sim: ResMut<ErosionSim>,
    key_input: Res<ButtonInput<KeyCode>>,
) {
    if key_input.just_pressed(KeyCode::Space) {
        erosion_sim.max_iterations += 10_000;
    }

    if erosion_sim.iterations > erosion_sim.max_iterations {
        return;
    }

    let new_iters = 1000;

    erosion_sim.iterations += new_iters;
    Erosion::default().erode(&mut erosion_sim.heightmap, new_iters);

    let mesh = create_mesh(&erosion_sim.heightmap, 1024);
    commands
        .entity(terrain.entity())
        .insert(Mesh3d(meshes.add(mesh)));
}

fn check_gradient(
    mut ray_cast: MeshRayCast,
    mut gizmos: Gizmos,
    ray_map: Res<RayMap>,
    erosion_sim: Res<ErosionSim>,
) {
    for (_, ray) in ray_map.iter() {
        let Some((_, hit)) = ray_cast
            .cast_ray(*ray, &MeshRayCastSettings::default())
            .first()
        else {
            continue;
        };

        gizmos.sphere(hit.point, 0.05, Color::srgb(0.6, 0.6, 0.8));

        let norm_coords = (hit.point.x / 10.0 + 0.5, hit.point.z / 10.0 + 0.5);
        let gradient = erosion_sim.heightmap.get_gradient(norm_coords) / 10.0;
        // println!("Gradient {} at {:?}", gradient, norm_coords);

        let step_factor = 20.0;
        let new_norm_coords = (
            (hit.point.x + gradient.x / step_factor) / 10.0 + 0.5,
            (hit.point.z + gradient.y / step_factor) / 10.0 + 0.5,
        );
        let next_height = erosion_sim.heightmap.sample_bilinear(new_norm_coords) + 100.0;

        let dir_vector = Vec3::new(
            hit.point.x + gradient.x / step_factor,
            next_height,
            hit.point.z + gradient.y / step_factor,
        ) - hit.point;

        gizmos.arrow(
            hit.point,
            hit.point + dir_vector * step_factor,
            Color::srgb(0.7, 0.7, 0.6),
        );
    }
}
