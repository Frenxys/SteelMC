use std::{
    io::Cursor,
    sync::{Arc, Weak},
};

use glam::DVec3;
use simdnbt::{ToNbtTag as _, borrow::read_compound, owned::NbtCompound};
use steel_registry::blocks::properties::{
    AttachFace, BlockStateProperties, DoubleBlockHalf, PistonType,
};
use steel_registry::{
    data_components::vanilla_components::CUSTOM_NAME, test_support::init_test_registry,
    vanilla_blocks, vanilla_damage_types, vanilla_entities, vanilla_game_rules::MOB_GRIEFING,
    vanilla_items,
};
use steel_utils::types::{GameType, UpdateFlags};
use steel_utils::{BlockPos, ChunkPos, Direction, Downcast as _};
use text_components::TextComponent;
use uuid::Uuid;

use super::*;
use crate::behavior::init_behaviors;
use crate::block_entity::{
    SharedBlockEntity, entities::PistonMovingBlockEntity, init_block_entities,
};
use crate::entity::EntityFluidContact;
use crate::entity::entities::{
    ChestMinecartEntity, ItemFrameEntity, LeashFenceKnotEntity, PigEntity,
};
use crate::player::ResetReason;
use crate::test_support::{TestPlayerBuilder, fresh_test_world, insert_ready_full_chunk};
use crate::world::{
    ClipBlockShape, ClipFluid, DefaultExplosionDamageCalculator, ExplosionInteraction,
    ExplosionOptions, ExplosionOutcome,
};

#[test]
fn vanilla_damage_formula_uses_distance_and_exposure() {
    struct TestExplosion;

    impl Explosion for TestExplosion {
        fn world(&self) -> &Arc<World> {
            panic!("test formula does not access the world")
        }

        fn damage_source(&self) -> &DamageSource {
            panic!("test formula does not access the damage source")
        }

        fn block_interaction(&self) -> BlockInteraction {
            BlockInteraction::Keep
        }

        fn indirect_source_entity(&self) -> Option<&dyn Entity> {
            None
        }

        fn direct_source_entity(&self) -> Option<&dyn Entity> {
            None
        }

        fn radius(&self) -> f32 {
            4.0
        }

        fn center(&self) -> DVec3 {
            DVec3::ZERO
        }

        fn can_trigger_blocks(&self) -> bool {
            false
        }

        fn should_affect_blocklike_entities(&self) -> bool {
            false
        }
    }

    init_test_registry();
    let entity = ItemEntity::new(
        &vanilla_entities::ITEM,
        1,
        DVec3::new(4.0, 0.0, 0.0),
        Weak::new(),
    );
    let damage =
        DefaultExplosionDamageCalculator.entity_damage_amount(&TestExplosion, &entity, 1.0);
    assert_eq!(damage.to_bits(), 22.0_f32.to_bits());
}

#[test]
fn default_damage_source_preserves_direct_source_position() {
    init_test_registry();
    let position = DVec3::new(1.25, 64.0, -3.5);
    let entity = ItemEntity::new(&vanilla_entities::ITEM, 17, position, Weak::new());

    let source = default_explosion_damage_source(Some(&entity), None);

    assert_eq!(source.direct_entity_id, Some(entity.id()));
    assert_eq!(source.causing_entity_id, None);
    assert_eq!(source.source_position, Some(position));
    assert_eq!(source.damage_type.key, vanilla_damage_types::EXPLOSION.key);
}

fn assert_pinned_exposure_matches_live(world: &World, center: DVec3, entity: &dyn Entity) -> f32 {
    let exposure = EntityExplosionExposure::capture(entity);
    let Ok(region) = exposure.read_region(center) else {
        panic!("small exposure read region should be representable");
    };
    let pinned = world
        .with_gameplay_block_state_read_batch(region, |reader| exposure.calculate(reader, center));
    let Ok(pinned) = pinned else {
        panic!("static exposure should complete through the pinned reader");
    };
    let live = exposure.calculate(world, center);

    assert_eq!(pinned.to_bits(), live.to_bits());
    assert_eq!(
        seen_percent(world, center, entity).to_bits(),
        live.to_bits()
    );
    pinned
}

#[test]
fn pinned_exposure_matches_live_for_clear_and_obstructed_paths() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("pinned_explosion_exposure");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let center = DVec3::new(0.5, 64.125, 0.5);
    let entity = ItemEntity::new(
        &vanilla_entities::ITEM,
        18,
        DVec3::new(2.5, 64.0, 0.5),
        Arc::downgrade(&world),
    );

    assert_eq!(
        assert_pinned_exposure_matches_live(&world, center, &entity).to_bits(),
        1.0_f32.to_bits()
    );

    assert!(world.set_block(
        BlockPos::new(1, 64, 0),
        vanilla_blocks::STONE.default_state(),
        UpdateFlags::UPDATE_ALL,
    ));
    assert_eq!(
        assert_pinned_exposure_matches_live(&world, center, &entity).to_bits(),
        0.0_f32.to_bits()
    );
}

#[test]
fn exposure_clipping_uses_the_entity_collision_context() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_entity_collision_context");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let powder_snow_pos = BlockPos::new(1, 64, 0);
    assert!(world.set_block(
        powder_snow_pos,
        vanilla_blocks::POWDER_SNOW.default_state(),
        UpdateFlags::UPDATE_ALL,
    ));
    let entity = ItemEntity::new(
        &vanilla_entities::ITEM,
        18,
        DVec3::new(2.5, 64.0, 0.5),
        Arc::downgrade(&world),
    );
    let center = DVec3::new(0.5, 64.125, 0.5);

    assert_eq!(
        assert_pinned_exposure_matches_live(&world, center, &entity).to_bits(),
        1.0_f32.to_bits()
    );

    entity.set_fall_distance(3.0);
    assert_eq!(
        assert_pinned_exposure_matches_live(&world, center, &entity).to_bits(),
        0.0_f32.to_bits()
    );
}

#[test]
fn moving_piston_exposure_retries_with_live_block_entities() {
    init_test_registry();
    init_behaviors();
    init_block_entities();
    let world = fresh_test_world("moving_piston_explosion_exposure");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let piston_pos = BlockPos::new(1, 64, 0);
    let moving_state = vanilla_blocks::MOVING_PISTON
        .default_state()
        .set_value(&BlockStateProperties::FACING, Direction::East)
        .set_value(&BlockStateProperties::PISTON_TYPE, PistonType::Normal);
    let moved_state = vanilla_blocks::PISTON
        .default_state()
        .set_value(&BlockStateProperties::FACING, Direction::East)
        .set_value(&BlockStateProperties::EXTENDED, true);
    assert!(world.set_block(piston_pos, moving_state, UpdateFlags::UPDATE_NONE));
    let block_entity: SharedBlockEntity = Arc::new(PistonMovingBlockEntity::new_moving(
        Arc::downgrade(&world),
        piston_pos,
        moving_state,
        moved_state,
        Direction::East,
        false,
        true,
    ));
    assert!(world.set_block_entity(block_entity));
    let center = DVec3::new(0.5, 64.125, 0.5);
    let entity = ItemEntity::new(
        &vanilla_entities::ITEM,
        20,
        DVec3::new(2.5, 64.0, 0.5),
        Arc::downgrade(&world),
    );
    let exposure = EntityExplosionExposure::capture(&entity);
    let Ok(region) = exposure.read_region(center) else {
        panic!("small exposure read region should be representable");
    };

    let pinned = world
        .with_gameplay_block_state_read_batch(region, |reader| exposure.calculate(reader, center));
    assert_eq!(pinned, Err(GameplayBlockReadWindowError::RetryLive));

    let live = exposure.calculate(world.as_ref(), center);
    for from in exposure.sample_positions() {
        let full_miss = world
            .clip_with_reader_and_collision_context(
                world.as_ref(),
                from,
                center,
                ClipBlockShape::Collider,
                ClipFluid::None,
                exposure.collision_context,
            )
            .is_miss();
        assert_eq!(
            World::is_block_collision_path_clear_with_reader(
                world.as_ref(),
                from,
                center,
                exposure.collision_context,
            ),
            full_miss,
        );
    }
    assert_eq!(
        seen_percent(&world, center, &entity).to_bits(),
        live.to_bits()
    );
}

#[test]
fn explosion_applies_damage_and_impulse_to_nearby_entities() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_entity_effects");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let item = Arc::new(ItemEntity::new(
        &vanilla_entities::ITEM,
        19,
        DVec3::new(1.5, 64.0, 0.5),
        Arc::downgrade(&world),
    ));
    item.set_item(ItemStack::with_count(&vanilla_items::STONE, 1));
    let entity: SharedEntity = item.clone();
    let Ok(()) = world.try_add_entity(entity) else {
        panic!("test item must be added to its loaded chunk");
    };

    let ExplosionOutcome {
        affected_block_count: _,
    } = world.explode(ExplosionOptions::new(
        DVec3::new(0.5, 64.0, 0.5),
        2.0,
        ExplosionInteraction::None,
    ));

    assert!(item.is_removed());
    assert!(item.velocity().x > 0.0);
}

#[test]
fn non_destructive_explosion_ignores_items_when_mob_griefing_is_disabled() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_ignores_blocklike_entities");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    assert!(world.set_game_rule(&MOB_GRIEFING, false));
    let item = Arc::new(ItemEntity::new(
        &vanilla_entities::ITEM,
        20,
        DVec3::new(1.5, 64.0, 0.5),
        Arc::downgrade(&world),
    ));
    item.set_item(ItemStack::with_count(&vanilla_items::STONE, 1));
    let entity: SharedEntity = item.clone();
    let Ok(()) = world.try_add_entity(entity) else {
        panic!("test item must be added to its loaded chunk");
    };

    world.explode(ExplosionOptions::new(
        DVec3::new(0.5, 64.0, 0.5),
        2.0,
        ExplosionInteraction::None,
    ));

    assert!(!item.is_removed());
    assert_eq!(item.get_health(), 5);
    assert_eq!(item.velocity(), DVec3::ZERO);
}

#[test]
fn mob_explosion_does_not_push_vehicles_when_mob_griefing_is_disabled() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_ignores_vehicles_without_mob_griefing");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    assert!(world.set_game_rule(&MOB_GRIEFING, false));
    let minecart = Arc::new(ChestMinecartEntity::new(
        &vanilla_entities::CHEST_MINECART,
        21,
        DVec3::new(1.5, 64.0, 0.5),
        Arc::downgrade(&world),
    ));
    let entity: SharedEntity = minecart.clone();
    let Ok(()) = world.try_add_entity(entity) else {
        panic!("test chest minecart must be added to its loaded chunk");
    };
    let pig = PigEntity::new(
        &vanilla_entities::PIG,
        22,
        DVec3::new(0.5, 64.0, 0.5),
        Arc::downgrade(&world),
    );
    let mut options =
        ExplosionOptions::new(DVec3::new(0.5, 64.0, 0.5), 2.0, ExplosionInteraction::Mob);
    options.source = Some(&pig);

    world.explode(options);

    assert_eq!(minecart.velocity(), DVec3::ZERO);
}

#[test]
fn non_blocklike_explosion_does_not_push_block_attached_entities() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_ignores_block_attached_entities");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    assert!(world.set_game_rule(&MOB_GRIEFING, false));
    let (item_frame, leash_knot) = add_block_attached_targets(&world, 23);

    world.explode(ExplosionOptions::new(
        DVec3::new(0.5, 64.0, 0.5),
        2.0,
        ExplosionInteraction::None,
    ));

    assert_eq!(item_frame.velocity(), DVec3::ZERO);
    assert_eq!(leash_knot.velocity(), DVec3::ZERO);
}

#[test]
fn submerged_source_explosion_does_not_push_block_attached_entities() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("submerged_explosion_ignores_block_attached_entities");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let source = ItemEntity::new(
        &vanilla_entities::ITEM,
        25,
        DVec3::new(0.5, 64.0, 0.5),
        Arc::downgrade(&world),
    );
    source
        .base()
        .set_fluid_contact(EntityFluidContact::from_parts(1.0, 0.0, false, false));
    assert!(source.is_in_water());
    let (item_frame, leash_knot) = add_block_attached_targets(&world, 26);
    let mut options =
        ExplosionOptions::new(DVec3::new(0.5, 64.0, 0.5), 2.0, ExplosionInteraction::Block);
    options.source = Some(&source);

    world.explode(options);

    assert_eq!(item_frame.velocity(), DVec3::ZERO);
    assert_eq!(leash_knot.velocity(), DVec3::ZERO);
}

fn add_block_attached_targets(
    world: &Arc<World>,
    first_id: i32,
) -> (Arc<ItemFrameEntity>, Arc<LeashFenceKnotEntity>) {
    let item_frame = Arc::new(ItemFrameEntity::new_attached(
        &vanilla_entities::ITEM_FRAME,
        first_id,
        BlockPos::new(1, 64, 0),
        Direction::West,
        Arc::downgrade(world),
    ));
    let item_frame_entity: SharedEntity = item_frame.clone();
    let Ok(()) = world.try_add_entity(item_frame_entity) else {
        panic!("test item frame must be added to its loaded chunk");
    };
    let leash_knot = Arc::new(LeashFenceKnotEntity::new_attached(
        &vanilla_entities::LEASH_KNOT,
        first_id + 1,
        BlockPos::new(0, 64, 1),
        Arc::downgrade(world),
    ));
    let leash_knot_entity: SharedEntity = leash_knot.clone();
    let Ok(()) = world.try_add_entity(leash_knot_entity) else {
        panic!("test leash knot must be added to its loaded chunk");
    };
    (item_frame, leash_knot)
}

#[test]
fn trigger_explosion_activates_controls_without_destroying_them() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("trigger_explosion_controls");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let lever_pos = BlockPos::new(2, 64, 2);
    let button_pos = BlockPos::new(8, 64, 2);
    let fence_gate_pos = BlockPos::new(14, 64, 2);
    for pos in [lever_pos, button_pos] {
        assert!(world.set_block(
            pos.below(),
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
    }
    assert!(
        world.set_block(
            lever_pos,
            vanilla_blocks::LEVER
                .default_state()
                .set_value(&BlockStateProperties::ATTACH_FACE, AttachFace::Floor),
            UpdateFlags::UPDATE_NONE,
        )
    );
    assert!(
        world.set_block(
            button_pos,
            vanilla_blocks::STONE_BUTTON
                .default_state()
                .set_value(&BlockStateProperties::ATTACH_FACE, AttachFace::Floor),
            UpdateFlags::UPDATE_NONE,
        )
    );
    assert!(world.set_block(
        fence_gate_pos,
        vanilla_blocks::OAK_FENCE_GATE.default_state(),
        UpdateFlags::UPDATE_NONE,
    ));

    trigger_block_positions(&world, &mut [lever_pos, button_pos, fence_gate_pos]);

    let lever = world.get_block_state(lever_pos);
    assert_eq!(lever.get_block(), &vanilla_blocks::LEVER);
    assert!(lever.get_value(&BlockStateProperties::POWERED));
    let button = world.get_block_state(button_pos);
    assert_eq!(button.get_block(), &vanilla_blocks::STONE_BUTTON);
    assert!(button.get_value(&BlockStateProperties::POWERED));
    let fence_gate = world.get_block_state(fence_gate_pos);
    assert_eq!(fence_gate.get_block(), &vanilla_blocks::OAK_FENCE_GATE);
    assert!(fence_gate.get_value(&BlockStateProperties::OPEN));
    assert!(!fence_gate.get_value(&BlockStateProperties::POWERED));
}

#[test]
fn trigger_explosion_respects_door_and_trapdoor_wind_charge_rules() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("trigger_explosion_doors");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let oak_door_pos = BlockPos::new(1, 64, 1);
    let iron_door_pos = BlockPos::new(4, 64, 1);
    let powered_door_pos = BlockPos::new(7, 64, 1);
    let upper_door_pos = BlockPos::new(10, 64, 1);
    let copper_door_pos = BlockPos::new(13, 64, 1);
    let oak_trapdoor_pos = BlockPos::new(1, 64, 5);
    let iron_trapdoor_pos = BlockPos::new(4, 64, 5);
    let powered_trapdoor_pos = BlockPos::new(7, 64, 5);
    let copper_trapdoor_pos = BlockPos::new(10, 64, 5);
    for (pos, state) in [
        (oak_door_pos, vanilla_blocks::OAK_DOOR.default_state()),
        (iron_door_pos, vanilla_blocks::IRON_DOOR.default_state()),
        (
            powered_door_pos,
            vanilla_blocks::OAK_DOOR
                .default_state()
                .set_value(&BlockStateProperties::POWERED, true),
        ),
        (
            upper_door_pos,
            vanilla_blocks::OAK_DOOR.default_state().set_value(
                &BlockStateProperties::DOUBLE_BLOCK_HALF,
                DoubleBlockHalf::Upper,
            ),
        ),
        (copper_door_pos, vanilla_blocks::COPPER_DOOR.default_state()),
        (
            oak_trapdoor_pos,
            vanilla_blocks::OAK_TRAPDOOR.default_state(),
        ),
        (
            iron_trapdoor_pos,
            vanilla_blocks::IRON_TRAPDOOR.default_state(),
        ),
        (
            powered_trapdoor_pos,
            vanilla_blocks::OAK_TRAPDOOR
                .default_state()
                .set_value(&BlockStateProperties::POWERED, true),
        ),
        (
            copper_trapdoor_pos,
            vanilla_blocks::COPPER_TRAPDOOR.default_state(),
        ),
    ] {
        assert!(world.set_block(pos, state, UpdateFlags::UPDATE_NONE));
    }

    trigger_block_positions(
        &world,
        &mut [
            oak_door_pos,
            iron_door_pos,
            powered_door_pos,
            upper_door_pos,
            copper_door_pos,
            oak_trapdoor_pos,
            iron_trapdoor_pos,
            powered_trapdoor_pos,
            copper_trapdoor_pos,
        ],
    );

    for pos in [oak_door_pos, copper_door_pos] {
        assert!(
            world
                .get_block_state(pos)
                .get_value(&BlockStateProperties::OPEN)
        );
    }
    for pos in [iron_door_pos, powered_door_pos, upper_door_pos] {
        assert!(
            !world
                .get_block_state(pos)
                .get_value(&BlockStateProperties::OPEN)
        );
    }
    for pos in [oak_trapdoor_pos, copper_trapdoor_pos] {
        assert!(
            world
                .get_block_state(pos)
                .get_value(&BlockStateProperties::OPEN)
        );
    }
    for pos in [iron_trapdoor_pos, powered_trapdoor_pos] {
        assert!(
            !world
                .get_block_state(pos)
                .get_value(&BlockStateProperties::OPEN)
        );
    }
}

#[test]
fn trigger_explosion_extinguishes_candles_without_destroying_them() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("trigger_explosion_candles");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let candle_pos = BlockPos::new(2, 64, 2);
    let candle_cake_pos = BlockPos::new(6, 64, 2);
    for pos in [candle_pos, candle_cake_pos] {
        assert!(world.set_block(
            pos.below(),
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
    }
    assert!(
        world.set_block(
            candle_pos,
            vanilla_blocks::CANDLE
                .default_state()
                .set_value(&BlockStateProperties::LIT, true),
            UpdateFlags::UPDATE_NONE,
        )
    );
    assert!(
        world.set_block(
            candle_cake_pos,
            vanilla_blocks::CANDLE_CAKE
                .default_state()
                .set_value(&BlockStateProperties::LIT, true),
            UpdateFlags::UPDATE_NONE,
        )
    );

    trigger_block_positions(&world, &mut [candle_pos, candle_cake_pos]);

    let candle = world.get_block_state(candle_pos);
    assert_eq!(candle.get_block(), &vanilla_blocks::CANDLE);
    assert!(!candle.get_value(&BlockStateProperties::LIT));
    let candle_cake = world.get_block_state(candle_cake_pos);
    assert_eq!(candle_cake.get_block(), &vanilla_blocks::CANDLE_CAKE);
    assert!(!candle_cake.get_value(&BlockStateProperties::LIT));
}

fn trigger_block_positions(world: &Arc<World>, positions: &mut [BlockPos]) {
    let explosion = ServerExplosion::new(
        world,
        None,
        None,
        None,
        None,
        DVec3::ZERO,
        1.0,
        false,
        BlockInteraction::TriggerBlock,
    );
    explosion.interact_with_blocks(positions);
}

#[test]
fn resistant_center_block_stops_explosion_rays() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_ray_resistance");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let center_pos = BlockPos::new(0, 64, 0);
    assert!(world.set_block(
        center_pos,
        vanilla_blocks::OBSIDIAN.default_state(),
        UpdateFlags::UPDATE_ALL,
    ));
    let explosion = ServerExplosion::new(
        &world,
        None,
        None,
        None,
        None,
        DVec3::new(0.5, 64.5, 0.5),
        4.0,
        false,
        BlockInteraction::Destroy,
    );

    let affected = explosion.calculate_exploded_positions(|| 0.5);

    assert!(affected.is_empty());
}

#[test]
fn explosion_rays_use_vanilla_world_bounds_in_both_lanes() {
    init_test_registry();
    let world = fresh_test_world("explosion_world_bounds");
    let bounds = ExplosionWorldBounds::from_world(&world);
    let min_y = world.get_min_y();
    let max_y = world.get_max_y();
    let cases = [
        (BlockPos::new(-30_000_000, min_y, -30_000_000), true),
        (BlockPos::new(29_999_999, max_y, 29_999_999), true),
        (BlockPos::new(-30_000_001, min_y, 0), false),
        (BlockPos::new(30_000_000, min_y, 0), false),
        (BlockPos::new(0, min_y, -30_000_001), false),
        (BlockPos::new(0, min_y, 30_000_000), false),
        (BlockPos::new(0, min_y - 1, 0), false),
        (BlockPos::new(0, max_y + 1, 0), false),
    ];

    for (pos, expected) in cases {
        assert_eq!(world.is_in_world_bounds(pos), expected, "live pos={pos:?}");
        assert_eq!(bounds.contains(pos), expected, "immutable pos={pos:?}");
    }
}

#[test]
fn immutable_block_rays_match_sequential_order_and_repeat() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("immutable_explosion_rays");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let center = DVec3::new(8.5, 64.5, 8.5);
    assert!(world.set_block(
        BlockPos::from(center),
        vanilla_blocks::STONE.default_state(),
        UpdateFlags::UPDATE_ALL,
    ));
    let immutable_calculator = DefaultExplosionDamageCalculator;
    let immutable = ServerExplosion::new(
        &world,
        None,
        None,
        None,
        Some(&immutable_calculator),
        center,
        4.0,
        false,
        BlockInteraction::Destroy,
    );
    let sequential = ServerExplosion::new(
        &world,
        None,
        None,
        None,
        None,
        center,
        4.0,
        false,
        BlockInteraction::Destroy,
    );
    let calculate = |explosion: &ServerExplosion<'_>| {
        let mut index = 0_u32;
        explosion.calculate_exploded_positions(|| {
            let value = ((index * 37 + 11) % 101) as f32 / 100.0;
            index += 1;
            value
        })
    };

    let sequential_positions = calculate(&sequential);
    let first_immutable_positions = calculate(&immutable);
    let second_immutable_positions = calculate(&immutable);

    assert_eq!(first_immutable_positions, sequential_positions);
    assert_eq!(second_immutable_positions, first_immutable_positions);

    let mut ray_index = 0_u32;
    let rays = immutable.draw_immutable_rays(|| {
        let value = ((ray_index * 37 + 11) % 101) as f32 / 100.0;
        ray_index += 1;
        value
    });
    assert_eq!(ray_index as usize, RAY_COUNT);
    let context = ExplosionRayContext {
        center,
        bounds: ExplosionWorldBounds::from_world(&world),
    };
    let reader = LiveExplosionBlockReader { world: &world };
    let pure_sequential =
        calculate_immutable_rays_sequential(&rays, context, &reader, &immutable_calculator);
    for worker_threads in [1, 4, 8, 12, 16] {
        let worker_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_threads)
            .build()
            .expect("explosion test pool should start");
        let target_shards = worker_threads * 2;
        let pure_parallel = worker_pool.install(|| {
            calculate_immutable_rays_sharded(
                &rays,
                context,
                &reader,
                &immutable_calculator,
                target_shards,
            )
        });
        let repeated_parallel = worker_pool.install(|| {
            calculate_immutable_rays_sharded(
                &rays,
                context,
                &reader,
                &immutable_calculator,
                target_shards,
            )
        });

        assert_eq!(pure_parallel, pure_sequential);
        assert_eq!(repeated_parallel, pure_parallel);
    }
}

#[test]
fn explosion_entity_query_excludes_spectators() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_excludes_spectators");
    let player =
        TestPlayerBuilder::new(Arc::clone(&world), Uuid::from_u128(1), "Spectator", 27).build();
    player.restore_game_modes(GameType::Spectator, None);
    player
        .abilities
        .lock()
        .update_for_game_mode(GameType::Spectator);
    let position = player.position();
    insert_ready_full_chunk(&world, ChunkPos::from_block_pos(BlockPos::from(position)));
    assert!(world.add_player(Arc::clone(&player), ResetReason::InitialJoin));

    world.explode(ExplosionOptions::new(
        position - DVec3::X,
        2.0,
        ExplosionInteraction::None,
    ));

    assert_eq!(player.velocity(), DVec3::ZERO);
}

#[test]
fn destructive_explosion_removes_blocks_and_spawns_their_loot() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("explosion_block_destruction");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let center_pos = BlockPos::new(0, 64, 0);
    assert!(world.set_block(
        center_pos,
        vanilla_blocks::STONE.default_state(),
        UpdateFlags::UPDATE_ALL,
    ));
    let mut explosion = ServerExplosion::new(
        &world,
        None,
        None,
        None,
        None,
        DVec3::new(0.5, 64.5, 0.5),
        4.0,
        false,
        BlockInteraction::Destroy,
    );

    explosion.explode();

    assert!(world.get_block_state(center_pos).is_air());
    let drops = world.get_entities_in_aabb_matching(
        &WorldAabb::new(-1.0, 63.0, -1.0, 2.0, 67.0, 2.0),
        |entity| entity.entity_type() == &vanilla_entities::ITEM,
    );
    assert!(drops.iter().any(|entity| {
        entity
            .as_ref()
            .downcast_ref::<ItemEntity>()
            .is_some_and(|item| item.get_item().is(&vanilla_items::COBBLESTONE))
    }));
}

#[test]
fn explosion_loot_preserves_live_block_entity_components() {
    init_test_registry();
    init_behaviors();
    init_block_entities();
    let world = fresh_test_world("explosion_block_entity_components");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let center_pos = BlockPos::new(0, 64, 0);
    let chest_state = vanilla_blocks::CHEST.default_state();
    assert!(world.set_block(center_pos, chest_state, UpdateFlags::UPDATE_ALL));
    let Some(block_entity) = world.get_block_entity(center_pos) else {
        panic!("placed chest must create its block entity");
    };
    let custom_name = TextComponent::from("Explosion chest");
    let mut nbt = NbtCompound::new();
    nbt.insert("CustomName", custom_name.clone().to_nbt_tag());
    let mut encoded = Vec::new();
    nbt.write(&mut encoded);
    let Ok(borrowed) = read_compound(&mut Cursor::new(encoded.as_slice())) else {
        panic!("test block entity NBT must reborrow");
    };
    block_entity.load_additional(&borrowed);
    let mut explosion = ServerExplosion::new(
        &world,
        None,
        None,
        None,
        None,
        DVec3::new(0.5, 64.5, 0.5),
        4.0,
        false,
        BlockInteraction::Destroy,
    );

    explosion.explode();

    let drops = world.get_entities_in_aabb_matching(
        &WorldAabb::new(-1.0, 63.0, -1.0, 2.0, 67.0, 2.0),
        |entity| entity.entity_type() == &vanilla_entities::ITEM,
    );
    assert!(drops.iter().any(|entity| {
        entity
            .as_ref()
            .downcast_ref::<ItemEntity>()
            .is_some_and(|item| item.get_item().get(CUSTOM_NAME) == Some(&custom_name))
    }));
}

#[test]
fn combined_explosion_drops_never_exceed_sixteen() {
    init_test_registry();
    let stack = ItemStack::with_count(&vanilla_items::STONE, 10);
    let mut stacks = Vec::new();

    add_or_append_stack(&mut stacks, stack.clone(), BlockPos::ZERO);
    add_or_append_stack(&mut stacks, stack, BlockPos::ZERO);

    assert_eq!(stacks.len(), 2);
    assert_eq!(stacks[0].stack.count(), 16);
    assert_eq!(stacks[1].stack.count(), 4);
}
