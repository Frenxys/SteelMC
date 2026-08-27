//! Beacon block entity implementation.
//!
//! Beacons track the pyramid level and configured effects. Every 80 game ticks
//! they re-check the supporting pyramid and, while the beam has an unobstructed
//! path to the sky, apply the configured status effects to nearby players.

use std::{
    str::FromStr as _,
    sync::{Arc, Weak},
};

use glam::DVec3;
use simdnbt::borrow::{BaseNbtCompound as BorrowedNbtCompound, NbtCompound as NbtCompoundView};
use simdnbt::owned::NbtCompound;
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::mob_effect::MobEffectRef;
use steel_registry::{
    REGISTRY, RegistryExt, TaggedRegistryExt, vanilla_block_entity_types, vanilla_block_tags,
    vanilla_blocks, vanilla_entities, vanilla_mob_effects,
};
use steel_utils::locks::SyncMutex;
use steel_utils::{
    BlockPos, BlockStateId, Downcast as _, DowncastType, DowncastTypeKey, Identifier, WorldAabb,
};

use crate::block_entity::{BlockEntity, BlockEntityBase};
use crate::chunk::light::MAX_LIGHT_LEVEL;
use crate::entity::{LivingEntity as _, MobEffectInstance};
use crate::player::Player;
use crate::world::World;

/// Maximum beacon pyramid level.
const MAX_LEVELS: i32 = 4;

const BEACON_TICK_INTERVAL: i64 = 80;

const BASE_EFFECT_RANGE: f64 = 10.0;
const EFFECT_RANGE_PER_LEVEL: f64 = 10.0;

/// The four valid beacon effects, indexed by pyramid level tier.
pub(crate) const BEACON_EFFECTS: [&[MobEffectRef]; 4] = [
    &[vanilla_mob_effects::SPEED, vanilla_mob_effects::HASTE],
    &[
        vanilla_mob_effects::RESISTANCE,
        vanilla_mob_effects::JUMP_BOOST,
    ],
    &[vanilla_mob_effects::STRENGTH],
    &[vanilla_mob_effects::REGENERATION],
];

/// Mutable beacon state shared with the menu's data slots.
pub struct BeaconState {
    pub(crate) levels: i32,
    pub(crate) primary_power: Option<MobEffectRef>,
    pub(crate) secondary_power: Option<MobEffectRef>,
}

impl BeaconState {
    const fn new() -> Self {
        Self {
            levels: 0,
            primary_power: None,
            secondary_power: None,
        }
    }

    pub(crate) fn filter_effect(effect: Option<MobEffectRef>) -> Option<MobEffectRef> {
        effect.filter(|effect| {
            BEACON_EFFECTS
                .iter()
                .copied()
                .flatten()
                .any(|valid| valid.key == effect.key)
        })
    }

    /// Mirrors vanilla `validateEffects`: returns whether the combination is legal for the
    /// given pyramid level.
    pub(crate) fn validate_effects(
        primary: Option<MobEffectRef>,
        secondary: Option<MobEffectRef>,
        levels: i32,
    ) -> bool {
        if secondary.is_some() && levels < MAX_LEVELS {
            return false;
        }
        let primary_level = Self::required_levels_for(primary);
        let secondary_level = Self::required_levels_for(secondary);
        if primary_level > levels || secondary_level > levels {
            return false;
        }
        // Regeneration (tier 4) is secondary-only.
        if primary_level >= MAX_LEVELS {
            return false;
        }
        secondary_level == 0
            || secondary_level >= MAX_LEVELS
            || primary.zip(secondary).is_some_and(|(p, s)| p.key == s.key)
    }

    /// Returns the 1-indexed tier that unlocks `effect`, `0` for `None`, or `i32::MAX` for
    /// effects not in `BEACON_EFFECTS`.
    fn required_levels_for(effect: Option<MobEffectRef>) -> i32 {
        let Some(effect) = effect else {
            return 0;
        };
        for (i, tier) in BEACON_EFFECTS.iter().enumerate() {
            if tier.iter().any(|e| e.key == effect.key) {
                return i as i32 + 1;
            }
        }
        i32::MAX
    }
}

/// Beacon block entity.
pub struct BeaconBlockEntity {
    base: Arc<BlockEntityBase>,
    state: Arc<SyncMutex<BeaconState>>,
}

// SAFETY: This key is owned by Steel and uniquely identifies `BeaconBlockEntity`.
unsafe impl DowncastType for BeaconBlockEntity {
    const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new("steel:block_entity/beacon");
}

impl BeaconBlockEntity {
    /// Creates a new beacon block entity.
    #[must_use]
    pub fn new(level: Weak<World>, pos: BlockPos, state: BlockStateId) -> Self {
        let base = Arc::new(BlockEntityBase::new(
            &vanilla_block_entity_types::BEACON,
            level,
            pos,
            state,
        ));
        Self {
            base,
            state: Arc::new(SyncMutex::new(BeaconState::new())),
        }
    }

    pub(crate) fn state(&self) -> Arc<SyncMutex<BeaconState>> {
        Arc::clone(&self.state)
    }

    /// Returns whether the beam has an unobstructed path to the sky.
    fn is_beam_open(world: &World, pos: BlockPos) -> bool {
        for y in (pos.y() + 1)..=world.get_max_y() {
            let state = world.get_block_state(BlockPos::new(pos.x(), y, pos.z()));
            if state.get_block() != &vanilla_blocks::BEDROCK
                && state.get_light_dampening() >= MAX_LIGHT_LEVEL
            {
                return false;
            }
        }
        true
    }

    /// Recomputes the beacon's pyramid level, mirroring vanilla `updateBase`.
    fn update_base(world: &World, pos: BlockPos) -> i32 {
        let mut levels = 0;
        for step in 1..=MAX_LEVELS {
            let layer_y = pos.y() - step;
            if layer_y < world.get_min_y() {
                break;
            }

            let mut valid = true;
            'outer: for layer_x in (pos.x() - step)..=(pos.x() + step) {
                for layer_z in (pos.z() - step)..=(pos.z() + step) {
                    let state = world.get_block_state(BlockPos::new(layer_x, layer_y, layer_z));
                    if !REGISTRY.blocks.is_in_tag(
                        state.get_block(),
                        &vanilla_block_tags::BlockTag::BEACON_BASE_BLOCKS,
                    ) {
                        valid = false;
                        break 'outer;
                    }
                }
            }

            if !valid {
                break;
            }
            levels = step;
        }
        levels
    }

    fn apply_effects(&self, world: &Arc<World>, pos: BlockPos, levels: i32) {
        let (primary, secondary) = {
            let state = self.state.lock();
            (state.primary_power, state.secondary_power)
        };
        let Some(primary) = primary else {
            return;
        };

        let range = f64::from(levels) * EFFECT_RANGE_PER_LEVEL + BASE_EFFECT_RANGE;
        let base_amplifier =
            i32::from(levels >= 4 && secondary.is_some_and(|s| s.key == primary.key));
        let duration = (9 + levels * 2) * 20;

        // Vanilla: `new AABB(pos).inflate(range).expandTowards(0, level.getHeight(), 0)`.
        let world_height = f64::from(world.get_max_y() - world.get_min_y() + 1);
        let min = DVec3::new(
            f64::from(pos.x()) - range,
            f64::from(pos.y()) - range,
            f64::from(pos.z()) - range,
        );
        let max = DVec3::new(
            f64::from(pos.x()) + 1.0 + range,
            f64::from(pos.y()) + 1.0 + range + world_height,
            f64::from(pos.z()) + 1.0 + range,
        );
        let aabb = WorldAabb::from_min_max(min, max);

        for entity in world.get_entities_in_aabb_matching(&aabb, |entity| {
            entity.entity_type() == &vanilla_entities::PLAYER
        }) {
            let Some(player) = entity.downcast_ref::<Player>() else {
                continue;
            };
            player.add_mob_effect(
                MobEffectInstance::with_duration(primary, duration, base_amplifier)
                    .with_ambient(true)
                    .with_visible(true),
            );

            if let Some(secondary) = secondary.filter(|s| levels >= 4 && s.key != primary.key) {
                player.add_mob_effect(
                    MobEffectInstance::with_duration(secondary, duration, 0)
                        .with_ambient(true)
                        .with_visible(true),
                );
            }
        }
    }

    fn store_effect(nbt: &mut NbtCompound, field: &str, effect: Option<MobEffectRef>) {
        if let Some(effect) = effect {
            nbt.insert(field, effect.key.to_string());
        }
    }

    fn load_effect(nbt: &BorrowedNbtCompound<'_>, field: &str) -> Option<MobEffectRef> {
        let nbt_view: NbtCompoundView<'_, '_> = nbt.into();
        let key = Identifier::from_str(&nbt_view.string(field)?.to_string()).ok()?;
        let effect = REGISTRY.mob_effects.by_key(&key)?;
        BeaconState::filter_effect(Some(effect))
    }
}

impl BlockEntity for BeaconBlockEntity {
    fn base(&self) -> &BlockEntityBase {
        &self.base
    }

    fn load_additional(&self, nbt: &BorrowedNbtCompound<'_>) {
        let mut state = self.state.lock();
        state.primary_power = Self::load_effect(nbt, "primary_effect");
        state.secondary_power = Self::load_effect(nbt, "secondary_effect");
    }

    fn save_additional(&self, nbt: &mut NbtCompound) {
        let state = self.state.lock();
        Self::store_effect(nbt, "primary_effect", state.primary_power);
        Self::store_effect(nbt, "secondary_effect", state.secondary_power);
        nbt.insert("Levels", state.levels);
    }

    fn get_update_tag(&self) -> Option<NbtCompound> {
        let mut nbt = NbtCompound::new();
        {
            let state = self.state.lock();
            Self::store_effect(&mut nbt, "primary_effect", state.primary_power);
            Self::store_effect(&mut nbt, "secondary_effect", state.secondary_power);
            nbt.insert("Levels", state.levels);
        }
        Some(nbt)
    }

    fn tick(&self, world: &Arc<World>) {
        if world.game_time() % BEACON_TICK_INTERVAL != 0 {
            return;
        }

        let pos = self.get_block_pos();
        if Self::is_beam_open(world, pos) {
            let levels = Self::update_base(world, pos);
            self.state.lock().levels = levels;
            if levels > 0 {
                self.apply_effects(world, pos, levels);
            }
        } else {
            self.state.lock().levels = 0;
        }
    }
}
