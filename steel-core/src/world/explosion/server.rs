use std::sync::Arc;

use glam::DVec3;
use rand::seq::SliceRandom;
use rustc_hash::{FxHashMap, FxHashSet};
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::item_stack::ItemStack;
use steel_registry::vanilla_entity_type_tags::EntityTypeTag;
use steel_registry::vanilla_game_rules::MOB_GRIEFING;
use steel_registry::{
    REGISTRY, TaggedRegistryExt as _, vanilla_attributes, vanilla_damage_types, vanilla_entities,
    vanilla_game_events,
};
use steel_utils::types::{GameType, UpdateFlags};
use steel_utils::{BlockPos, WorldAabb};

use crate::behavior::BLOCK_BEHAVIORS;
use crate::behavior::blocks::FireBlock;
use crate::entity::damage::DamageSource;
use crate::entity::entities::ItemEntity;
use crate::entity::{Entity, SharedEntity};
use crate::world::game_event::GameEventContext;
use crate::world::{ClipBlockShape, ClipFluid, World};

use super::{BlockInteraction, Explosion, ExplosionDamageCalculator, SelectedDamageCalculator};

const RAY_GRID_SIZE: i32 = 16;
const RAY_STEP: f64 = 0.3_f32 as f64;
const RAY_POWER_DECAY: f32 = 0.225_000_01;
const MIN_DAMAGE_RADIUS: f32 = 1.0e-5;
const NORMALIZE_EPSILON: f64 = 1.0e-5_f32 as f64;
const MAX_DROPS_PER_COMBINED_STACK: i32 = 16;

pub(super) struct ServerExplosion<'a> {
    world: &'a Arc<World>,
    fire: bool,
    block_interaction: BlockInteraction,
    center: DVec3,
    source: Option<&'a dyn Entity>,
    indirect_source: Option<SharedEntity>,
    radius: f32,
    damage_source: DamageSource,
    damage_calculator: SelectedDamageCalculator<'a>,
    pub(super) hit_players: FxHashMap<i32, DVec3>,
}

impl<'a> ServerExplosion<'a> {
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the Vanilla ServerExplosion construction boundary"
    )]
    pub(super) fn new(
        world: &'a Arc<World>,
        source: Option<&'a dyn Entity>,
        damage_source: Option<DamageSource>,
        damage_calculator: Option<&'a dyn ExplosionDamageCalculator>,
        center: DVec3,
        radius: f32,
        fire: bool,
        block_interaction: BlockInteraction,
    ) -> Self {
        let indirect_source = source
            .filter(|source| source.as_living_entity().is_none())
            .and_then(Entity::explosion_indirect_source);
        let indirect_source_entity = source
            .filter(|source| source.as_living_entity().is_some())
            .or(indirect_source.as_deref());
        let damage_source = damage_source
            .unwrap_or_else(|| default_explosion_damage_source(source, indirect_source_entity));
        let damage_calculator = match damage_calculator {
            Some(calculator) => SelectedDamageCalculator::Custom(calculator),
            None => source.map_or(
                SelectedDamageCalculator::Default,
                SelectedDamageCalculator::Entity,
            ),
        };
        Self {
            world,
            fire,
            block_interaction,
            center,
            source,
            indirect_source,
            radius,
            damage_source,
            damage_calculator,
            hit_players: FxHashMap::default(),
        }
    }

    pub(super) fn explode(&mut self) -> usize {
        self.world.game_event_at(
            &vanilla_game_events::EXPLODE,
            self.center,
            &GameEventContext::new(self.source, None),
        );
        let mut affected = self.calculate_exploded_positions(rand::random::<f32>);
        self.hurt_entities();
        if self.interacts_with_blocks() {
            self.interact_with_blocks(&mut affected);
        }
        if self.fire {
            self.create_fire(&affected);
        }
        affected.len()
    }

    fn calculate_exploded_positions(&self, mut next_float: impl FnMut() -> f32) -> Vec<BlockPos> {
        let mut affected = FxHashSet::default();

        for xx in 0..RAY_GRID_SIZE {
            for yy in 0..RAY_GRID_SIZE {
                for zz in 0..RAY_GRID_SIZE {
                    if xx != 0
                        && xx != RAY_GRID_SIZE - 1
                        && yy != 0
                        && yy != RAY_GRID_SIZE - 1
                        && zz != 0
                        && zz != RAY_GRID_SIZE - 1
                    {
                        continue;
                    }

                    let mut xd = f64::from(xx as f32 / 15.0 * 2.0 - 1.0);
                    let mut yd = f64::from(yy as f32 / 15.0 * 2.0 - 1.0);
                    let mut zd = f64::from(zz as f32 / 15.0 * 2.0 - 1.0);
                    let direction_length = (xd * xd + yd * yd + zd * zd).sqrt();
                    xd /= direction_length;
                    yd /= direction_length;
                    zd /= direction_length;

                    let mut remaining_power = self.radius * (0.7 + next_float() * 0.6);
                    let mut ray_pos = self.center;
                    while remaining_power > 0.0 {
                        let pos = BlockPos::from(ray_pos);
                        let state = self.world.get_block_state(pos);
                        let fluid = state.get_fluid_state();
                        if !self.world.is_in_valid_bounds(pos) {
                            break;
                        }

                        if let Some(resistance) = self
                            .damage_calculator
                            .block_explosion_resistance(self, self.world, pos, state, fluid)
                        {
                            remaining_power -= (resistance + 0.3) * 0.3;
                        }

                        if remaining_power > 0.0
                            && self.damage_calculator.should_block_explode(
                                self,
                                self.world,
                                pos,
                                state,
                                remaining_power,
                            )
                        {
                            affected.insert(pos);
                        }

                        ray_pos += DVec3::new(xd, yd, zd) * RAY_STEP;
                        remaining_power -= RAY_POWER_DECAY;
                    }
                }
            }
        }

        affected.into_iter().collect()
    }

    fn hurt_entities(&mut self) {
        if self.radius < MIN_DAMAGE_RADIUS {
            return;
        }

        let double_radius = self.radius * 2.0;
        let radius = f64::from(double_radius);
        let bounds = WorldAabb::from_min_max(
            DVec3::new(
                (self.center.x - radius - 1.0).floor(),
                (self.center.y - radius - 1.0).floor(),
                (self.center.z - radius - 1.0).floor(),
            ),
            DVec3::new(
                (self.center.x + radius + 1.0).floor(),
                (self.center.y + radius + 1.0).floor(),
                (self.center.z + radius + 1.0).floor(),
            ),
        );
        let source_id = self.source.map(Entity::id);
        let entities = self
            .world
            .get_entities_in_aabb_matching(&bounds, |entity| source_id != Some(entity.id()));
        let redirect_owner = self.damage_source.causing_entity_id.and_then(|owner_id| {
            self.indirect_source
                .as_ref()
                .filter(|owner| owner.id() == owner_id)
                .cloned()
                .or_else(|| self.world.get_entity_by_id(owner_id))
        });

        for entity in entities {
            if entity.ignore_explosion(self) {
                continue;
            }
            let distance = entity.position().distance(self.center) / radius;
            if distance > 1.0 {
                continue;
            }

            let delta = entity.explosion_damage_origin() - self.center;
            let delta_length = delta.length();
            let direction = if delta_length < NORMALIZE_EPSILON {
                DVec3::ZERO
            } else {
                delta / delta_length
            };
            let should_damage = self
                .damage_calculator
                .should_damage_entity(self, entity.as_ref());
            let knockback_multiplier = self.damage_calculator.knockback_multiplier(entity.as_ref());
            let exposure = if !should_damage && knockback_multiplier == 0.0 {
                0.0
            } else {
                seen_percent(self.center, entity.as_ref())
            };

            if should_damage {
                let amount =
                    self.damage_calculator
                        .entity_damage_amount(self, entity.as_ref(), exposure);
                entity.hurt(self.world, &self.damage_source, amount);
            }

            let knockback_resistance = entity.as_living_entity().map_or(0.0, |living| {
                living
                    .attributes()
                    .lock()
                    .required_value(vanilla_attributes::EXPLOSION_KNOCKBACK_RESISTANCE)
            });
            let knockback_power = (1.0 - distance)
                * f64::from(exposure)
                * f64::from(knockback_multiplier)
                * (1.0 - knockback_resistance);
            let knockback = direction * knockback_power;
            entity.push_impulse(knockback);

            if REGISTRY.entity_types.is_in_tag(
                entity.entity_type(),
                &EntityTypeTag::REDIRECTABLE_PROJECTILE,
            ) {
                if let Some(projectile) = entity.as_projectile() {
                    projectile.set_owner_entity(redirect_owner.as_ref());
                }
            } else if let Some(player) = entity.as_player()
                && !player.is_spectator()
                && (player.game_mode() != GameType::Creative || !player.abilities.lock().flying)
            {
                self.hit_players.insert(player.id(), knockback);
            }

            entity.on_explosion_hit(self.source);
        }
    }

    fn interact_with_blocks(&self, affected: &mut [BlockPos]) {
        affected.shuffle(&mut rand::rng());
        let mut stacks = Vec::new();

        for &pos in affected.iter() {
            let state = self.world.get_block_state(pos);
            BLOCK_BEHAVIORS
                .get_behavior(state.get_block())
                .on_explosion_hit(state, self.world, pos, self, &mut |stack, stack_pos| {
                    add_or_append_stack(&mut stacks, stack, stack_pos);
                });
        }

        for stack in stacks {
            self.world.pop_resource(stack.pos, stack.stack);
        }
    }

    fn create_fire(&self, affected: &[BlockPos]) {
        for &pos in affected {
            if rand::random_range(0..3) == 0
                && self.world.get_block_state(pos).is_air()
                && self.world.get_block_state(pos.below()).is_solid_render()
            {
                self.world.set_block(
                    pos,
                    FireBlock::get_state(self.world.as_ref(), pos),
                    UpdateFlags::UPDATE_ALL,
                );
            }
        }
    }

    fn interacts_with_blocks(&self) -> bool {
        self.block_interaction != BlockInteraction::Keep
    }

    pub(super) fn is_small(&self) -> bool {
        self.radius < 2.0 || !self.interacts_with_blocks()
    }
}

impl Explosion for ServerExplosion<'_> {
    fn world(&self) -> &Arc<World> {
        self.world
    }

    fn damage_source(&self) -> &DamageSource {
        &self.damage_source
    }

    fn block_interaction(&self) -> BlockInteraction {
        self.block_interaction
    }

    fn indirect_source_entity(&self) -> Option<&dyn Entity> {
        self.source
            .filter(|source| source.as_living_entity().is_some())
            .or(self.indirect_source.as_deref())
    }

    fn direct_source_entity(&self) -> Option<&dyn Entity> {
        self.source
    }

    fn radius(&self) -> f32 {
        self.radius
    }

    fn center(&self) -> DVec3 {
        self.center
    }

    fn can_trigger_blocks(&self) -> bool {
        if self.block_interaction != BlockInteraction::TriggerBlock {
            return false;
        }
        self.source.is_none_or(|source| {
            source.entity_type() != &vanilla_entities::BREEZE_WIND_CHARGE
                || self.world.get_game_rule(&MOB_GRIEFING)
        })
    }

    fn should_affect_blocklike_entities(&self) -> bool {
        let is_wind_charge = self.source.is_some_and(|source| {
            source.entity_type() == &vanilla_entities::BREEZE_WIND_CHARGE
                || source.entity_type() == &vanilla_entities::WIND_CHARGE
        });
        !is_wind_charge
            && (self.world.get_game_rule(&MOB_GRIEFING)
                || self.block_interaction.should_affect_blocklike_entities())
    }
}

fn default_explosion_damage_source(
    direct: Option<&dyn Entity>,
    indirect: Option<&dyn Entity>,
) -> DamageSource {
    let damage_type = if direct.is_some() && indirect.is_some() {
        &vanilla_damage_types::PLAYER_EXPLOSION
    } else {
        &vanilla_damage_types::EXPLOSION
    };
    let mut source = DamageSource::environment(damage_type);
    if let Some(entity) = direct {
        source = source
            .with_direct_entity(entity.id())
            .with_source_position(entity.position());
    }
    if let Some(entity) = indirect {
        source = source.with_causing_entity(entity.id());
    }
    source
}

fn seen_percent(center: DVec3, entity: &dyn Entity) -> f32 {
    let bounding_box = entity.bounding_box();
    let x_step = 1.0 / (bounding_box.width() * 2.0 + 1.0);
    let y_step = 1.0 / (bounding_box.height() * 2.0 + 1.0);
    let z_step = 1.0 / (bounding_box.depth() * 2.0 + 1.0);
    let x_offset = (1.0 - (1.0 / x_step).floor() * x_step) / 2.0;
    let z_offset = (1.0 - (1.0 / z_step).floor() * z_step) / 2.0;
    if x_step < 0.0 || y_step < 0.0 || z_step < 0.0 {
        return 0.0;
    }

    let Some(world) = entity.level() else {
        return 0.0;
    };
    let mut hits = 0_u32;
    let mut count = 0_u32;
    let mut x = 0.0;
    while x <= 1.0 {
        let mut y = 0.0;
        while y <= 1.0 {
            let mut z = 0.0;
            while z <= 1.0 {
                let from = DVec3::new(
                    bounding_box.min_x()
                        + (bounding_box.max_x() - bounding_box.min_x()) * x
                        + x_offset,
                    bounding_box.min_y() + (bounding_box.max_y() - bounding_box.min_y()) * y,
                    bounding_box.min_z()
                        + (bounding_box.max_z() - bounding_box.min_z()) * z
                        + z_offset,
                );
                if world
                    .clip_for_entity(
                        from,
                        center,
                        ClipBlockShape::Collider,
                        ClipFluid::None,
                        entity,
                    )
                    .is_miss()
                {
                    hits += 1;
                }
                count += 1;
                z += z_step;
            }
            y += y_step;
        }
        x += x_step;
    }
    hits as f32 / count as f32
}

struct StackCollector {
    pos: BlockPos,
    stack: ItemStack,
}

fn add_or_append_stack(stacks: &mut Vec<StackCollector>, mut stack: ItemStack, pos: BlockPos) {
    for collector in stacks.iter_mut() {
        if ItemEntity::are_mergeable(&collector.stack, &stack) {
            let available = collector
                .stack
                .max_stack_size()
                .min(MAX_DROPS_PER_COMBINED_STACK)
                - collector.stack.count();
            let transferred = available.min(stack.count());
            collector.stack = collector
                .stack
                .copy_with_count(collector.stack.count() + transferred);
            stack.shrink(transferred);
            if stack.is_empty() {
                return;
            }
        }
    }
    stacks.push(StackCollector { pos, stack });
}

#[cfg(test)]
mod tests;
