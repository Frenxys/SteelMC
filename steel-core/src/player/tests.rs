use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use glam::DVec3;
use simdnbt::{ToNbtTag as _, borrow::read_compound, owned::NbtCompound};
use steel_protocol::packets::game::EquipmentSlotItem;
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::blocks::properties::{BlockStateProperties, Direction, DoubleBlockHalf};
use steel_registry::data_component_predicate::DataComponentMatchers;
use steel_registry::data_components::vanilla_components::{CAN_BREAK, CUSTOM_NAME, EQUIPPABLE};
use steel_registry::data_components::{AdventureModePredicate, BlockPredicate};
use steel_registry::{
    RegistryHolderSet, item_stack::ItemStack, test_support::init_test_registry, vanilla_attributes,
    vanilla_blocks, vanilla_damage_types, vanilla_entities, vanilla_game_rules, vanilla_items,
    vanilla_menu_types,
};
use steel_utils::locks::IntoShared as _;
use steel_utils::types::{Difficulty, GameType, InteractionHand, UpdateFlags};
use steel_utils::{BlockPos, ChunkPos, Downcast as _, DowncastType, DowncastTypeKey, WorldAabb};
use text_components::TextComponent;
use uuid::Uuid;

use crate::behavior::{InteractionResult, init_behaviors};
use crate::block_entity::init_block_entities;
use crate::entity::{
    Entity, EntitySyncedData, LivingEntity, damage::DamageSource, entities::ItemEntity,
    next_entity_id,
};
use crate::inventory::{
    click::{Click, DragKind, QuickCraft},
    container::{Container as _, SimpleContainer},
    equipment::{EntityEquipment, EquipmentSlot},
    menu::{Menu, MenuBehavior, MenuBuilder, MenuKind, kinds::BasicKind},
};
use crate::permission::{PermissionEntry, PermissionKey, PermissionMetadataSet, PermissionSet};
use crate::test_support::{
    TestPlayerBuilder, fresh_test_world, hard_damage_test_world, insert_ready_full_chunk,
    test_world,
};
use crate::world::World;

use super::{
    DEATH_DURATION, Player, PlayerPermissionState, ResetReason, experience::Experience,
    experience::first_point_level_up_sound, game_mode::block_breaking::BlockBreakAction,
    lifecycle::nullable_game_mode_id, player_data::PersistentPlayerData,
};

fn test_player(world: Arc<World>) -> Arc<Player> {
    let player = TestPlayerBuilder::new(world, Uuid::from_u128(1), "TestPlayer", 1).build();
    player.set_client_loaded(true);
    player
}

macro_rules! impl_test_menu_kind_downcast {
    ($type:ty, $key:literal) => {
        // SAFETY: This test-owned key uniquely identifies the concrete menu
        // kind within the test process.
        unsafe impl DowncastType for $type {
            const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new($key);
        }
    };
}

struct CountRemovals {
    count: Arc<AtomicUsize>,
}

impl_test_menu_kind_downcast!(CountRemovals, "steel:test/menu/player/count_removals");

impl MenuKind for CountRemovals {
    fn removed(&mut self, _behavior: &mut MenuBehavior, _player: &Player) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

struct ReopenOnRemoved {
    replacement_removals: Arc<AtomicUsize>,
}

impl_test_menu_kind_downcast!(ReopenOnRemoved, "steel:test/menu/player/reopen_on_removed");

impl MenuKind for ReopenOnRemoved {
    fn removed(&mut self, _behavior: &mut MenuBehavior, player: &Player) {
        let replacement_removals = Arc::clone(&self.replacement_removals);
        player.open_menu("Replacement", move |container_id, _world| {
            empty_test_menu(
                player,
                container_id,
                CountRemovals {
                    count: replacement_removals,
                },
            )
        });
    }
}

fn empty_test_menu(player: &Player, container_id: u8, kind: impl MenuKind + 'static) -> Menu {
    let mut builder = MenuBuilder::new(&vanilla_menu_types::GENERIC_9X1, container_id);
    builder.section(SimpleContainer::new(9).into_shared(), 9);
    builder.player_inventory(&player.inventory);
    builder.build(kind)
}

fn permission_key(value: &str) -> PermissionKey {
    match PermissionKey::parse(value) {
        Ok(key) => key,
        Err(error) => panic!("test permission key should parse: {error}"),
    }
}

#[test]
fn permission_state_replacement_is_versioned_and_keeps_both_rule_sets() {
    let mut state = PlayerPermissionState::default();
    let overrides =
        PermissionSet::from_entries([PermissionEntry::deny(permission_key("steel.fly"))]);
    let effective =
        PermissionSet::from_entries([PermissionEntry::allow(permission_key("steel.build"))]);

    let first = state.replace(
        vec!["builder".to_owned()],
        overrides.clone(),
        PermissionMetadataSet::new(),
        effective.clone(),
        PermissionMetadataSet::new(),
    );
    let second = state.replace(
        vec!["moderator".to_owned()],
        overrides,
        PermissionMetadataSet::new(),
        effective,
        PermissionMetadataSet::new(),
    );

    assert_eq!(first, 1);
    assert_eq!(second, 2);
    assert_eq!(state.groups, ["moderator"]);
    assert!(!state.overrides.allows_key(&permission_key("steel.fly")));
    assert!(state.effective.allows_key(&permission_key("steel.build")));
}

#[test]
fn respawn_request_is_allowed_after_dead_reconnect() {
    assert!(Player::should_process_respawn(0.0));
}

#[test]
fn ai_step_copies_player_yaw_to_head_yaw() {
    init_test_registry();
    init_behaviors();
    let player = test_player(Arc::clone(test_world()));
    player.set_rotation((90.0, 15.0));
    player.set_y_head_rot(-45.0);

    let _ = player.ai_step();

    assert_eq!(player.y_head_rot().to_bits(), 90.0_f32.to_bits());
}

#[test]
fn respawn_request_is_ignored_while_alive() {
    assert!(!Player::should_process_respawn(20.0));
}

#[test]
fn respawn_request_uses_health_not_death_processed_guard() {
    struct RespawnGateInput {
        health: f32,
        death_processed: bool,
    }

    let input = RespawnGateInput {
        health: 20.0,
        death_processed: true,
    };

    assert!(input.death_processed);
    assert!(!Player::should_process_respawn(input.health));
}

#[test]
fn end_credits_respawn_keeps_vanilla_attribute_data_only() {
    assert_eq!(ResetReason::InitialJoin.respawn_data_kept(), 0x00);
    assert_eq!(ResetReason::Respawn.respawn_data_kept(), 0x00);
    assert_eq!(ResetReason::EndCredits.respawn_data_kept(), 0x01);
    assert_eq!(ResetReason::WorldChange.respawn_data_kept(), 0x03);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the full death-to-removal flow must verify every menu item disposition together"
)]
fn death_keeps_menu_items_until_entity_removal() {
    init_test_registry();
    let world = fresh_test_world("death_menu_cleanup");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    assert!(world.set_game_rule(&vanilla_game_rules::KEEP_INVENTORY, true));
    let player = test_player(Arc::clone(&world));
    let kept_item = ItemStack::new(&vanilla_items::DIAMOND);
    player.inventory.lock().set_item(0, kept_item);
    let transient = SimpleContainer::new(9).into_shared();
    transient
        .lock()
        .set_item(0, ItemStack::with_count(&vanilla_items::STONE, 3));
    let crafting = player.crafting_container();
    crafting
        .lock()
        .set_item(0, ItemStack::with_count(&vanilla_items::DIRT, 2));
    *player.inventory_menu.lock().behavior_mut().carried_mut() =
        ItemStack::new(&vanilla_items::STICK);
    player.inventory_menu.lock().clicked(
        Click::QuickCraft(QuickCraft::Start {
            kind: DragKind::Left,
        }),
        &player,
    );
    assert_eq!(
        player.inventory_menu.lock().behavior().quickcraft(),
        Some(DragKind::Left)
    );

    let menu_container = Arc::clone(&transient);
    let inventory = Arc::clone(&player.inventory);
    player.open_menu("Death cleanup", move |container_id, _world| {
        let mut builder = MenuBuilder::new(&vanilla_menu_types::GENERIC_9X1, container_id);
        let transient_slots = builder.section(menu_container, 9);
        builder.player_inventory(&inventory);
        builder.drain([transient_slots]);
        builder.build(BasicKind {})
    });

    player.set_health(0.0);
    player.die(&DamageSource::environment(&vanilla_damage_types::GENERIC));

    assert_eq!(transient.lock().get_item(0).count(), 3);
    assert_eq!(crafting.lock().get_item(0).count(), 2);
    assert!(
        player
            .inventory_menu
            .lock()
            .behavior()
            .carried()
            .is(&vanilla_items::STICK)
    );
    assert!(
        world
            .get_entities_in_aabb_matching(
                &WorldAabb::new(-2.0, -1.0, -2.0, 2.0, 3.0, 2.0),
                |entity| entity.entity_type() == &vanilla_entities::ITEM,
            )
            .is_empty()
    );

    for _ in 1..DEATH_DURATION {
        player.tick_death();
    }
    assert_eq!(transient.lock().get_item(0).count(), 3);

    player.tick_death();

    assert!(transient.lock().get_item(0).is_empty());
    assert!(crafting.lock().get_item(0).is_empty());
    assert!(player.inventory_menu.lock().behavior().carried().is_empty());
    assert_eq!(
        player.inventory_menu.lock().behavior().quickcraft(),
        Some(DragKind::Left)
    );
    assert!(
        player
            .inventory
            .lock()
            .get_item(0)
            .is(&vanilla_items::DIAMOND)
    );
    let dropped = world.get_entities_in_aabb_matching(
        &WorldAabb::new(-2.0, -1.0, -2.0, 2.0, 3.0, 2.0),
        |entity| entity.entity_type() == &vanilla_entities::ITEM,
    );
    assert_eq!(dropped.len(), 3);
    let mut dropped_stacks = Vec::new();
    for entity in dropped {
        let Some(item) = entity.as_ref().downcast_ref::<ItemEntity>() else {
            panic!("dropped entity should retain its concrete item type");
        };
        dropped_stacks.push(item.get_item());
    }
    assert!(
        dropped_stacks
            .iter()
            .any(|item| item.is(&vanilla_items::STONE) && item.count() == 3)
    );
    assert!(
        dropped_stacks
            .iter()
            .any(|item| item.is(&vanilla_items::DIRT) && item.count() == 2)
    );
    assert!(
        dropped_stacks
            .iter()
            .any(|item| item.is(&vanilla_items::STICK) && item.count() == 1)
    );
}

#[test]
fn death_respawn_drops_menu_items_exactly_once() {
    init_test_registry();
    let world = fresh_test_world("death_respawn_menu_cleanup");
    insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
    let player = test_player(Arc::clone(&world));
    let transient = SimpleContainer::new(9).into_shared();
    transient
        .lock()
        .set_item(0, ItemStack::with_count(&vanilla_items::STONE, 3));
    {
        let mut inventory_menu = player.inventory_menu.lock();
        *inventory_menu.behavior_mut().carried_mut() = ItemStack::new(&vanilla_items::STICK);
        inventory_menu.clicked(
            Click::QuickCraft(QuickCraft::Start {
                kind: DragKind::Left,
            }),
            &player,
        );
        *inventory_menu.behavior_mut().carried_mut() = ItemStack::empty();
    }

    let menu_container = Arc::clone(&transient);
    let inventory = Arc::clone(&player.inventory);
    player.open_menu("Respawn cleanup", move |container_id, _world| {
        let mut builder = MenuBuilder::new(&vanilla_menu_types::GENERIC_9X1, container_id);
        let transient_slots = builder.section(menu_container, 9);
        builder.player_inventory(&inventory);
        builder.drain([transient_slots]);
        builder.build(BasicKind {})
    });

    player.set_health(0.0);
    player.die(&DamageSource::environment(&vanilla_damage_types::GENERIC));
    player.reset_state_for_death_respawn();
    let _ = player.base.clear_removed();
    player.reset(Arc::clone(&world), ResetReason::Respawn);
    {
        let mut inventory_menu = player.inventory_menu.lock();
        assert_eq!(inventory_menu.behavior().quickcraft(), None);
        *inventory_menu.behavior_mut().carried_mut() = ItemStack::new(&vanilla_items::STICK);
        inventory_menu.clicked(
            Click::QuickCraft(QuickCraft::Start {
                kind: DragKind::Left,
            }),
            &player,
        );
        assert_eq!(inventory_menu.behavior().quickcraft(), Some(DragKind::Left));
    }

    assert!(transient.lock().get_item(0).is_empty());
    assert!(
        player
            .inventory
            .lock()
            .items()
            .iter()
            .all(|item| !item.is(&vanilla_items::STONE))
    );
    let dropped = world.get_entities_in_aabb_matching(
        &WorldAabb::new(-2.0, -1.0, -2.0, 2.0, 3.0, 2.0),
        |entity| entity.entity_type() == &vanilla_entities::ITEM,
    );
    assert_eq!(dropped.len(), 1);
    let Some(item) = dropped[0].as_ref().downcast_ref::<ItemEntity>() else {
        panic!("dropped entity should retain its concrete item type");
    };
    assert_eq!(item.get_item().count(), 3);
}

#[test]
fn end_credits_removes_all_menus_before_detaching() {
    init_test_registry();
    let world = fresh_test_world("end_credits_menu_removal");
    let player = test_player(Arc::clone(&world));
    assert!(world.add_player(Arc::clone(&player), ResetReason::InitialJoin));
    let _ = player.mark_joined_world();

    player
        .crafting_container()
        .lock()
        .set_item(0, ItemStack::with_count(&vanilla_items::STONE, 2));
    *player.inventory_menu.lock().behavior_mut().carried_mut() =
        ItemStack::with_count(&vanilla_items::DIRT, 3);
    let replacement_removals = Arc::new(AtomicUsize::new(0));
    player.open_menu("Reopen on removal", |container_id, _world| {
        empty_test_menu(
            &player,
            container_id,
            ReopenOnRemoved {
                replacement_removals: Arc::clone(&replacement_removals),
            },
        )
    });

    player.show_end_credits();

    assert!(player.has_won_game());
    assert!(!player.has_container_open());
    assert!(world.players.get_by_uuid(&player.gameprofile.id).is_none());
    assert_eq!(replacement_removals.load(Ordering::Relaxed), 0);
    let inventory = player.inventory.lock();
    for (item, expected) in [(&vanilla_items::STONE, 2), (&vanilla_items::DIRT, 3)] {
        let count: i32 = inventory
            .items()
            .iter()
            .filter(|stack| stack.is(item))
            .map(ItemStack::count)
            .sum();
        assert_eq!(count, expected);
    }
}

#[test]
fn disabled_damage_game_rule_matches_vanilla_player_damage_gates() {
    init_test_registry();

    let cases = [
        (
            &vanilla_damage_types::DROWN,
            &vanilla_game_rules::DROWNING_DAMAGE,
        ),
        (
            &vanilla_damage_types::FALL,
            &vanilla_game_rules::FALL_DAMAGE,
        ),
        (
            &vanilla_damage_types::LAVA,
            &vanilla_game_rules::FIRE_DAMAGE,
        ),
        (
            &vanilla_damage_types::FREEZE,
            &vanilla_game_rules::FREEZE_DAMAGE,
        ),
    ];

    for (damage_type, rule) in cases {
        let source = DamageSource::environment(damage_type);
        let mapped = Player::disabled_damage_game_rule(&source);
        assert!(mapped.is_some_and(|mapped| mapped.key() == rule.key()));
    }
}

#[test]
fn disabled_damage_game_rule_ignores_unrelated_damage() {
    init_test_registry();
    let source = DamageSource::environment(&vanilla_damage_types::GENERIC);

    assert!(Player::disabled_damage_game_rule(&source).is_none());
}

#[test]
fn hurt_uses_explicit_world_difficulty() {
    let attached_world = Arc::clone(test_world());
    let damage_world = hard_damage_test_world();
    let player = test_player(attached_world);
    let source = DamageSource::environment(&vanilla_damage_types::EXPLOSION);

    assert_eq!(player.get_world().difficulty(), Difficulty::Normal);
    assert_eq!(damage_world.difficulty(), Difficulty::Hard);
    assert_eq!(player.get_health().to_bits(), 20.0_f32.to_bits());

    assert!(player.hurt(damage_world, &source, 4.0));
    assert_eq!(player.get_health().to_bits(), 14.0_f32.to_bits());
}

#[test]
fn conditional_damage_does_not_scale_for_player_or_unresolved_causes() {
    let world = hard_damage_test_world();
    let causing_player = test_player(Arc::clone(world));
    let source = DamageSource::environment(&vanilla_damage_types::FIREWORKS);

    assert!(!source.scales_with_difficulty(Some(causing_player.as_ref())));

    let target = test_player(Arc::clone(world));
    let unresolved_source = source.with_causing_entity(2);
    assert!(target.hurt(world, &unresolved_source, 4.0));
    assert_eq!(target.get_health().to_bits(), 16.0_f32.to_bits());
}

#[test]
fn player_damage_applies_armor_and_absorption() {
    init_test_registry();
    let world = Arc::clone(test_world());
    let player = test_player(Arc::clone(&world));
    {
        let mut attributes = player.attributes().lock();
        attributes.set_base_value(vanilla_attributes::ARMOR, 20.0);
        attributes.set_base_value(vanilla_attributes::MAX_ABSORPTION, 3.0);
    }
    player.set_absorption_amount(3.0);
    let source = DamageSource::environment(&vanilla_damage_types::FIREWORKS);

    assert!(player.hurt(&world, &source, 10.0));

    assert_eq!(player.get_health().to_bits(), 19.0_f32.to_bits());
    assert_eq!(player.get_absorption_amount().to_bits(), 0.0_f32.to_bits());
}

#[test]
fn player_absorption_amount_clamps_to_attribute_range() {
    let world = Arc::clone(test_world());
    let player = test_player(world);
    player
        .attributes()
        .lock()
        .set_base_value(vanilla_attributes::MAX_ABSORPTION, 4.0);

    player.set_absorption_amount(10.0);
    assert_eq!(player.get_absorption_amount().to_bits(), 4.0_f32.to_bits());

    player.set_absorption_amount(-1.0);
    assert_eq!(player.get_absorption_amount().to_bits(), 0.0_f32.to_bits());
}

#[test]
fn player_damage_hurts_armor_equipment() {
    init_test_registry();
    let world = Arc::clone(test_world());
    let player = test_player(Arc::clone(&world));
    player.inventory.lock().set(
        EquipmentSlot::Chest,
        ItemStack::new(&vanilla_items::DIAMOND_CHESTPLATE),
    );
    let source = DamageSource::environment(&vanilla_damage_types::FIREWORKS);

    assert!(player.hurt(&world, &source, 8.0));

    let inventory = player.inventory.lock();
    assert_eq!(
        inventory.get_ref(EquipmentSlot::Chest).get_damage_value(),
        2,
    );
}

#[test]
fn equipping_player_target_uses_inventory_equipment_storage() {
    init_test_registry();
    let world = Arc::clone(test_world());
    let source = test_player(Arc::clone(&world));
    let target =
        TestPlayerBuilder::new(world, Uuid::from_u128(2), "Target", next_entity_id()).build();
    let mut helmet = ItemStack::new(&vanilla_items::DIAMOND_HELMET);
    let Some(mut equippable) = helmet.get_equippable().cloned() else {
        panic!("diamond helmet should have equippable data");
    };
    equippable.equip_on_interact = true;
    helmet.set(EQUIPPABLE, equippable);
    source.inventory.lock().set_selected_item(helmet.clone());

    let result = LivingEntity::interact_living_entity_with_equippable(
        target.as_ref(),
        source.as_ref(),
        InteractionHand::MainHand,
    );

    assert_eq!(result, InteractionResult::Success);
    assert!(source.inventory.lock().get_selected_item().is_empty());
    assert_eq!(
        target.inventory.lock().get_ref(EquipmentSlot::Head),
        &helmet
    );
    assert_eq!(
        target
            .living_base()
            .equipment()
            .lock()
            .get_ref(EquipmentSlot::Head),
        &helmet,
        "LivingEntityBase and Player::inventory must share one equipment backing",
    );
    LivingEntity::detect_equipment_updates(target.as_ref());
    assert_eq!(
        Entity::drain_dirty_equipment(target.as_ref()),
        vec![EquipmentSlotItem {
            slot: EquipmentSlot::Head,
            item_stack: helmet,
        }]
    );
}

#[test]
fn living_tick_detects_raw_inventory_equipment_mutation() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));
    let (base_armor, base_toughness) = {
        let attributes = player.attributes().lock();
        (
            attributes.required_value(vanilla_attributes::ARMOR),
            attributes.required_value(vanilla_attributes::ARMOR_TOUGHNESS),
        )
    };

    {
        let mut inventory = player.inventory.lock();
        inventory.items_mut()[39] = ItemStack::new(&vanilla_items::DIAMOND_HELMET);
    }

    LivingEntity::detect_equipment_updates(player.as_ref());

    {
        let attributes = player.attributes().lock();
        assert_eq!(
            attributes
                .required_value(vanilla_attributes::ARMOR)
                .to_bits(),
            (base_armor + 3.0).to_bits()
        );
        assert_eq!(
            attributes
                .required_value(vanilla_attributes::ARMOR_TOUGHNESS)
                .to_bits(),
            (base_toughness + 2.0).to_bits()
        );
    }
    assert_eq!(
        Entity::drain_dirty_equipment(player.as_ref()),
        vec![EquipmentSlotItem {
            slot: EquipmentSlot::Head,
            item_stack: ItemStack::new(&vanilla_items::DIAMOND_HELMET),
        }]
    );
    LivingEntity::detect_equipment_updates(player.as_ref());
    assert!(Entity::drain_dirty_equipment(player.as_ref()).is_empty());
}

#[test]
fn equipment_detection_tracks_selected_main_hand() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));
    {
        let mut inventory = player.inventory.lock();
        inventory.set_item(0, ItemStack::new(&vanilla_items::STICK));
        inventory.set_item(1, ItemStack::new(&vanilla_items::OAK_LOG));
    }

    LivingEntity::detect_equipment_updates(player.as_ref());
    assert_eq!(
        Entity::drain_dirty_equipment(player.as_ref()),
        vec![EquipmentSlotItem {
            slot: EquipmentSlot::MainHand,
            item_stack: ItemStack::new(&vanilla_items::STICK),
        }]
    );

    player.inventory.lock().set_selected_slot(1);
    LivingEntity::detect_equipment_updates(player.as_ref());
    assert_eq!(
        Entity::drain_dirty_equipment(player.as_ref()),
        vec![EquipmentSlotItem {
            slot: EquipmentSlot::MainHand,
            item_stack: ItemStack::new(&vanilla_items::OAK_LOG),
        }]
    );
}

#[test]
fn equipment_detection_suppresses_exact_hand_swap_packet() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));
    {
        let mut inventory = player.inventory.lock();
        inventory.set_selected_item(ItemStack::new(&vanilla_items::STICK));
        inventory.set_offhand_item(ItemStack::new(&vanilla_items::SHIELD));
    }
    LivingEntity::detect_equipment_updates(player.as_ref());
    let initial = Entity::drain_dirty_equipment(player.as_ref());
    assert_eq!(initial.len(), 2);

    assert!(player.inventory.lock().swap_hands());
    LivingEntity::detect_equipment_updates(player.as_ref());

    assert!(Entity::drain_dirty_equipment(player.as_ref()).is_empty());
}

#[test]
fn equipment_detection_coalesces_before_tracker_drain() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));
    player.inventory.lock().set(
        EquipmentSlot::Head,
        ItemStack::new(&vanilla_items::IRON_HELMET),
    );
    LivingEntity::detect_equipment_updates(player.as_ref());

    player.inventory.lock().set(
        EquipmentSlot::Head,
        ItemStack::new(&vanilla_items::DIAMOND_HELMET),
    );
    LivingEntity::detect_equipment_updates(player.as_ref());

    assert_eq!(
        Entity::drain_dirty_equipment(player.as_ref()),
        vec![EquipmentSlotItem {
            slot: EquipmentSlot::Head,
            item_stack: ItemStack::new(&vanilla_items::DIAMOND_HELMET),
        }]
    );
}

#[test]
fn nullable_game_mode_id_matches_vanilla_encoding() {
    assert_eq!(nullable_game_mode_id(None), -1);
    assert_eq!(nullable_game_mode_id(Some(GameType::Creative)), 1);
}

#[test]
fn clear_matching_items_uses_inventory_crafting_then_carried_order() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));
    player
        .inventory
        .lock()
        .set_item(0, ItemStack::with_count(&vanilla_items::STONE, 3));
    {
        let inventory_menu = player.inventory_menu.lock();
        inventory_menu
            .crafting_container()
            .expect("inventory menu should have a crafting grid")
            .lock()
            .set_item(0, ItemStack::with_count(&vanilla_items::STONE, 2));
    }
    *player.inventory_menu.lock().behavior_mut().carried_mut() =
        ItemStack::with_count(&vanilla_items::STONE, 4);

    let stone = |stack: &ItemStack| stack.is(&vanilla_items::STONE);
    assert_eq!(player.clear_or_count_matching_items(&stone, 5), 5);
    assert!(player.inventory.lock().get_item(0).is_empty());
    assert!(
        player
            .inventory_menu
            .lock()
            .crafting_container()
            .expect("inventory menu should have a crafting grid")
            .lock()
            .get_item(0)
            .is_empty()
    );
    assert_eq!(player.inventory_menu.lock().behavior().carried().count(), 4);

    assert_eq!(player.clear_or_count_matching_items(&stone, 0), 4);
    assert_eq!(player.inventory_menu.lock().behavior().carried().count(), 4);
    assert_eq!(player.clear_or_count_matching_items(&stone, -1), 4);
    assert!(player.inventory_menu.lock().behavior().carried().is_empty());
}

#[test]
fn point_level_up_sound_uses_first_crossed_five_level_boundary() {
    assert_eq!(first_point_level_up_sound(0, 4, 100), None);
    assert_eq!(first_point_level_up_sound(0, 5, 100), Some(5));
    assert_eq!(first_point_level_up_sound(4, 12, 100), Some(5));
    assert_eq!(first_point_level_up_sound(5, 10, 100), Some(10));
    assert_eq!(first_point_level_up_sound(5, 10, -100), None);
}

#[test]
fn point_grants_update_entity_score_with_java_wrapping() {
    let player = test_player(Arc::clone(test_world()));
    player.set_score(i32::MAX - 10);

    player.give_experience_points(100);

    assert_eq!(player.score(), (i32::MAX - 10).wrapping_add(100));
    assert_eq!(player.experience.lock().total_points(), 100);
}

#[test]
fn persistent_player_data_restores_independent_experience_fields_and_score() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));
    *player.experience.lock() = Experience::from_parts(7, 0.5, 32);
    player.set_score(19);
    let persistent = PersistentPlayerData::from_player(&player);

    *player.experience.lock() = Experience::default();
    player.set_score(-1);
    persistent.apply_to_player_without_location(&player);

    let experience = player.experience.lock();
    assert_eq!(experience.level(), 7);
    assert_eq!(experience.progress().to_bits(), 0.5_f32.to_bits());
    assert_eq!(experience.total_points(), 32);
    drop(experience);
    assert_eq!(player.score(), 19);
}

#[test]
fn persistent_player_data_restores_equipment_inventory_slots() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));
    let helmet = ItemStack::new(&vanilla_items::DIAMOND_HELMET);
    let saddle = ItemStack::new(&vanilla_items::SADDLE);
    {
        let mut inventory = player.inventory.lock();
        inventory.set(EquipmentSlot::Head, helmet.clone());
        inventory.set(EquipmentSlot::Saddle, saddle.clone());
    }
    let persistent = PersistentPlayerData::from_player(&player);

    {
        let mut inventory = player.inventory.lock();
        inventory.clear();
    }
    persistent.apply_to_player_without_location(&player);

    let inventory = player.inventory.lock();
    assert_eq!(inventory.get_ref(EquipmentSlot::Head), &helmet);
    assert_eq!(inventory.get_ref(EquipmentSlot::Saddle), &saddle);
}

#[test]
fn effect_visibility_refresh_preserves_spectator_invisibility() {
    init_test_registry();
    let player = test_player(Arc::clone(test_world()));

    player.restore_game_modes(GameType::Spectator, Some(GameType::Survival));
    player.living_base.mark_effects_dirty();
    player.update_dirty_mob_effect_entity_data();
    assert!(player.entity_data.is_base_invisible_flag());

    player.restore_game_modes(GameType::Survival, Some(GameType::Spectator));
    player.living_base.mark_effects_dirty();
    player.update_dirty_mob_effect_entity_data();
    assert!(!player.entity_data.is_base_invisible_flag());
}

#[test]
fn block_action_restriction_precedes_redstone_ore_attack() {
    init_test_registry();
    init_behaviors();
    let world = fresh_test_world("redstone_ore_block_action_restriction");
    let pos = BlockPos::new(1, 64, 0);
    insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));
    assert!(world.set_block(
        pos,
        vanilla_blocks::REDSTONE_ORE.default_state(),
        UpdateFlags::UPDATE_ALL,
    ));

    let player = test_player(Arc::clone(&world));
    player.base.set_position_local(DVec3::new(1.0, 64.0, 0.0));

    for game_mode in [GameType::Spectator, GameType::Adventure] {
        player.restore_game_modes(game_mode, None);
        player.abilities.lock().update_for_game_mode(game_mode);
        player.block_breaking.lock().handle_block_break_action(
            &player,
            &world,
            pos,
            BlockBreakAction::Start,
            Direction::Up,
        );
        assert!(
            !world
                .get_block_state(pos)
                .get_value(&BlockStateProperties::LIT)
        );
    }

    let predicate = BlockPredicate::new(
        Some(RegistryHolderSet::direct(vec![
            &vanilla_blocks::REDSTONE_ORE,
        ])),
        None,
        None,
        DataComponentMatchers::ANY,
    );
    let can_break =
        AdventureModePredicate::new(vec![predicate]).expect("one block predicate is valid");
    let mut tool = ItemStack::new(&vanilla_items::DIAMOND_PICKAXE);
    tool.set(CAN_BREAK, can_break);
    player.inventory.lock().set_selected_item(tool);

    player.block_breaking.lock().handle_block_break_action(
        &player,
        &world,
        pos,
        BlockBreakAction::Start,
        Direction::Up,
    );
    assert!(
        world
            .get_block_state(pos)
            .get_value(&BlockStateProperties::LIT)
    );
}

#[test]
fn player_breaks_double_plant_loot_before_either_half_is_removed() {
    init_test_registry();
    init_behaviors();

    for (world_key, broken_half) in [
        ("double_plant_break_lower", DoubleBlockHalf::Lower),
        ("double_plant_break_upper", DoubleBlockHalf::Upper),
    ] {
        let world = fresh_test_world(world_key);
        let lower_pos = BlockPos::new(8, 64, 8);
        let upper_pos = lower_pos.above();
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(lower_pos));

        let lower = vanilla_blocks::TALL_GRASS.default_state().set_value(
            &BlockStateProperties::DOUBLE_BLOCK_HALF,
            DoubleBlockHalf::Lower,
        );
        let upper = vanilla_blocks::TALL_GRASS.default_state().set_value(
            &BlockStateProperties::DOUBLE_BLOCK_HALF,
            DoubleBlockHalf::Upper,
        );
        let placement_flags = UpdateFlags::UPDATE_NONE | UpdateFlags::UPDATE_KNOWN_SHAPE;
        assert!(world.set_block(lower_pos, lower, placement_flags));
        assert!(world.set_block(upper_pos, upper, placement_flags));

        let player = test_player(Arc::clone(&world));
        player.base.set_position_local(DVec3::new(8.5, 64.0, 8.5));
        player
            .inventory
            .lock()
            .set_selected_item(ItemStack::new(&vanilla_items::SHEARS));
        let break_pos = match broken_half {
            DoubleBlockHalf::Lower => lower_pos,
            DoubleBlockHalf::Upper => upper_pos,
        };
        player.block_breaking.lock().handle_block_break_action(
            &player,
            &world,
            break_pos,
            BlockBreakAction::Start,
            Direction::Up,
        );

        assert!(world.get_block_state(lower_pos).is_air());
        assert!(world.get_block_state(upper_pos).is_air());
        let dropped = world.get_entities_in_aabb_matching(
            &WorldAabb::new(6.0, 62.0, 6.0, 11.0, 68.0, 11.0),
            |entity| entity.entity_type() == &vanilla_entities::ITEM,
        );
        assert_eq!(dropped.len(), 1);
        let Some(item) = dropped[0].as_ref().downcast_ref::<ItemEntity>() else {
            panic!("double-plant loot should spawn an item entity");
        };
        let stack = item.get_item();
        assert!(stack.is(&vanilla_items::SHORT_GRASS));
        assert_eq!(stack.count(), 2);
    }
}

#[test]
fn player_break_loot_preserves_removed_block_entity_components() {
    init_test_registry();
    init_behaviors();
    init_block_entities();

    let world = fresh_test_world("player_break_block_entity_components");
    let pos = BlockPos::new(8, 64, 8);
    insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));
    assert!(world.set_block(
        pos,
        vanilla_blocks::CHEST.default_state(),
        UpdateFlags::UPDATE_ALL,
    ));
    let Some(block_entity) = world.get_block_entity(pos) else {
        panic!("placed chest must create its block entity");
    };
    let custom_name = TextComponent::from("Player-broken chest");
    let mut nbt = NbtCompound::new();
    nbt.insert("CustomName", custom_name.clone().to_nbt_tag());
    let mut encoded = Vec::new();
    nbt.write(&mut encoded);
    let Ok(borrowed) = read_compound(&mut Cursor::new(encoded.as_slice())) else {
        panic!("test block entity NBT must reborrow");
    };
    block_entity.load_additional(&borrowed);

    let player = test_player(Arc::clone(&world));
    player.base.set_position_local(DVec3::new(8.5, 64.0, 8.5));
    player
        .inventory
        .lock()
        .set_selected_item(ItemStack::new(&vanilla_items::NETHERITE_AXE));
    let mut block_breaking = player.block_breaking.lock();
    block_breaking.handle_block_break_action(
        &player,
        &world,
        pos,
        BlockBreakAction::Start,
        Direction::Up,
    );
    block_breaking.handle_block_break_action(
        &player,
        &world,
        pos,
        BlockBreakAction::Stop,
        Direction::Up,
    );
    for _ in 0..64 {
        if world.get_block_state(pos).is_air() {
            break;
        }
        block_breaking.tick(&player, &world);
    }
    drop(block_breaking);

    assert!(world.get_block_state(pos).is_air());
    let drops = world.get_entities_in_aabb_matching(
        &WorldAabb::new(6.0, 62.0, 6.0, 11.0, 68.0, 11.0),
        |entity| entity.entity_type() == &vanilla_entities::ITEM,
    );
    assert!(drops.iter().any(|entity| {
        entity
            .as_ref()
            .downcast_ref::<ItemEntity>()
            .is_some_and(|item| item.get_item().get(CUSTOM_NAME) == Some(&custom_name))
    }));
}
