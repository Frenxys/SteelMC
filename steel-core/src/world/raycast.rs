use super::*;

/// Controls how a block position is treated during a raytrace traversal.
///
/// Returned by the predicate closure passed to [`World::raytrace`].
#[derive(Debug)]
pub enum RaytraceAction {
    /// Skip this block and continue traversal (transparent block).
    Pass,
    /// Test the block's voxel shape for a precise ray intersection.
    CheckShape,
    /// Immediately treat this block as a hit without shape testing.
    ImmediateHit,
}

/// Block shape channel used by vanilla-style world clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipBlockShape {
    /// `ClipContext.Block.COLLIDER`
    Collider,
    /// `ClipContext.Block.OUTLINE`
    Outline,
    /// `ClipContext.Block.VISUAL`
    Visual,
    /// `ClipContext.Block.FALLDAMAGE_RESETTING`
    FallDamageResetting {
        /// Whether the clip context entity is a player.
        entity_is_player: bool,
    },
}

/// Fluid shape filter used by vanilla-style world clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipFluid {
    /// `ClipContext.Fluid.NONE`
    None,
    /// `ClipContext.Fluid.SOURCE_ONLY`
    SourceOnly,
    /// `ClipContext.Fluid.ANY`
    Any,
    /// `ClipContext.Fluid.WATER`
    Water,
}

/// Result of a vanilla-style world clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipHitResult {
    /// Exact hit location in world coordinates.
    pub location: DVec3,
    /// Hit face, or the miss direction for misses.
    pub direction: Direction,
    /// Block position containing the hit or miss endpoint.
    pub block_pos: BlockPos,
    /// Whether this result is a miss.
    pub miss: bool,
    /// Whether the ray started inside the hit shape.
    pub inside: bool,
    /// Whether this hit was synthesized by the world border.
    pub world_border_hit: bool,
}

struct BlockShapeClipRay {
    from: DVec3,
    difference: DVec3,
    inside_test_point: DVec3,
}

enum BlockShapeBoxHit {
    Inside,
    Entry { t: f64, direction: Direction },
}

impl BlockShapeClipRay {
    fn new(from: DVec3, to: DVec3) -> Option<Self> {
        let difference = to - from;
        if difference.length_squared() < 1.0e-7 {
            return None;
        }

        Some(Self {
            from,
            difference,
            inside_test_point: from + difference * 0.001,
        })
    }
}

impl ClipHitResult {
    /// Returns whether this clip missed all selected block and fluid shapes.
    #[must_use]
    pub const fn is_miss(self) -> bool {
        self.miss
    }
}

impl World {
    /// Checks if a ray intersects with a block's selection box.
    pub fn ray_outline_check(
        &self,
        block_pos: BlockPos,
        from: DVec3,
        to: DVec3,
    ) -> (bool, Option<Direction>) {
        let state = self.get_block_state(block_pos);
        let shape = state.get_outline_shape_at(block_pos);

        match Self::clip_shape(block_pos, from, to, shape) {
            Some(hit) => (true, Some(hit.direction)),
            None => (false, None),
        }
    }

    /// Performs a vanilla-style block/fluid clip in the world.
    #[must_use]
    pub fn clip(
        &self,
        start_pos: DVec3,
        end_pos: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
    ) -> ClipHitResult {
        self.clip_with_reader(self, start_pos, end_pos, block_shape, fluid)
    }

    /// Performs a block/fluid clip with block reads routed through `reader`.
    #[must_use]
    pub(crate) fn clip_with_reader<R: LevelReader>(
        &self,
        reader: &R,
        start_pos: DVec3,
        end_pos: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
    ) -> ClipHitResult {
        self.clip_with_reader_and_collision_context(
            reader,
            start_pos,
            end_pos,
            block_shape,
            fluid,
            BlockCollisionContext::empty(),
        )
    }

    /// Performs a vanilla-style block/fluid clip using an entity collision context.
    #[must_use]
    pub fn clip_for_entity(
        &self,
        start_pos: DVec3,
        end_pos: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
        entity: &dyn Entity,
    ) -> ClipHitResult {
        self.clip_for_entity_with_reader(self, start_pos, end_pos, block_shape, fluid, entity)
    }

    /// Performs an entity-context clip with block reads routed through `reader`.
    #[must_use]
    pub(crate) fn clip_for_entity_with_reader<R: LevelReader>(
        &self,
        reader: &R,
        start_pos: DVec3,
        end_pos: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
        entity: &dyn Entity,
    ) -> ClipHitResult {
        let context = BlockCollisionContext::entity(entity.position().y, entity.is_descending())
            .with_fall_distance(entity.fall_distance())
            .with_can_walk_on_powder_snow(entity.can_walk_on_powder_snow())
            .with_falling_block(entity.entity_type() == &vanilla_entities::FALLING_BLOCK);
        self.clip_with_reader_and_collision_context(
            reader,
            start_pos,
            end_pos,
            block_shape,
            fluid,
            context,
        )
    }

    pub(crate) fn clip_with_reader_and_collision_context<R: LevelReader>(
        &self,
        reader: &R,
        start_pos: DVec3,
        end_pos: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
        collision_context: BlockCollisionContext,
    ) -> ClipHitResult {
        Self::traverse_blocks(start_pos, end_pos, |block| {
            self.clip_block_and_fluid(
                reader,
                block,
                start_pos,
                end_pos,
                block_shape,
                fluid,
                collision_context,
            )
        })
        .unwrap_or_else(|| Self::clip_miss(start_pos, end_pos))
    }

    /// Returns whether Vanilla's collider-only, fluid-free clip would miss every block shape.
    ///
    /// This retains the authoritative block behavior and entity collision context while omitting
    /// hit location, face, and interaction-shape work that cannot affect hit versus miss.
    pub(crate) fn is_block_collision_path_clear_with_reader<R: LevelReader>(
        reader: &R,
        start_pos: DVec3,
        end_pos: DVec3,
        collision_context: BlockCollisionContext,
    ) -> bool {
        let Some(ray) = BlockShapeClipRay::new(start_pos, end_pos) else {
            return true;
        };
        let block_behaviors = &*BLOCK_BEHAVIORS;

        Self::traverse_blocks(start_pos, end_pos, |block_pos| {
            let state = reader.get_collision_candidate_state(block_pos)?;
            if block_behaviors.is_collision_shape_guaranteed_empty(state) {
                return None;
            }
            let shape = block_behaviors
                .get_behavior(state.get_block())
                .get_resolved_collision_shape(state, reader, block_pos, collision_context);
            if shape.is_empty() {
                return None;
            }
            Self::shape_blocks_ray(block_pos, &ray, shape.iter()).then_some(())
        })
        .is_none()
    }

    /// Performs vanilla `CollisionGetter.clipIncludingBorder`.
    #[must_use]
    pub fn clip_including_border(
        &self,
        start_pos: DVec3,
        end_pos: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
    ) -> ClipHitResult {
        self.clip_including_border_with_reader(self, start_pos, end_pos, block_shape, fluid)
    }

    /// Performs `CollisionGetter.clipIncludingBorder` through an injected reader.
    #[must_use]
    pub(crate) fn clip_including_border_with_reader<R: LevelReader>(
        &self,
        reader: &R,
        start_pos: DVec3,
        end_pos: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
    ) -> ClipHitResult {
        let hit = self.clip_with_reader(reader, start_pos, end_pos, block_shape, fluid);
        let border = self.world_border_snapshot();
        if border.is_within_bounds_with_margin(start_pos.x, start_pos.z, 0.0)
            && !border.is_within_bounds_with_margin(hit.location.x, hit.location.z, 0.0)
        {
            let delta = hit.location - start_pos;
            let location = border.clamp_vec3_to_bound(hit.location);
            return ClipHitResult {
                location,
                direction: Self::approximate_nearest_direction(delta),
                block_pos: BlockPos::from(location),
                miss: false,
                inside: false,
                world_border_hit: true,
            };
        }
        hit
    }

    /// Traverses block positions in Vanilla `BlockGetter.traverseBlocks` order.
    fn traverse_blocks<T>(
        start_pos: DVec3,
        end_pos: DVec3,
        mut visitor: impl FnMut(BlockPos) -> Option<T>,
    ) -> Option<T> {
        if start_pos == end_pos {
            return None;
        }

        let adjust = -1.0e-7f64;
        let to = end_pos.lerp(start_pos, adjust);
        let from = start_pos.lerp(end_pos, adjust);
        let mut block = BlockPos::new(
            from.x.floor() as i32,
            from.y.floor() as i32,
            from.z.floor() as i32,
        );

        if let Some(result) = visitor(block) {
            return Some(result);
        }

        let difference = to - from;
        let step = difference.signum().as_ivec3();
        let delta = DVec3::new(
            if step.x == 0 {
                f64::MAX
            } else {
                f64::from(step.x) / difference.x
            },
            if step.y == 0 {
                f64::MAX
            } else {
                f64::from(step.y) / difference.y
            },
            if step.z == 0 {
                f64::MAX
            } else {
                f64::from(step.z) / difference.z
            },
        );
        let mut next = DVec3::new(
            delta.x
                * if step.x > 0 {
                    1.0 - (from.x - from.x.floor())
                } else {
                    from.x - from.x.floor()
                },
            delta.y
                * if step.y > 0 {
                    1.0 - (from.y - from.y.floor())
                } else {
                    from.y - from.y.floor()
                },
            delta.z
                * if step.z > 0 {
                    1.0 - (from.z - from.z.floor())
                } else {
                    from.z - from.z.floor()
                },
        );

        while next.x <= 1.0 || next.y <= 1.0 || next.z <= 1.0 {
            if next.x < next.y {
                if next.x < next.z {
                    block.0.x += step.x;
                    next.x += delta.x;
                } else {
                    block.0.z += step.z;
                    next.z += delta.z;
                }
            } else if next.y < next.z {
                block.0.y += step.y;
                next.y += delta.y;
            } else {
                block.0.z += step.z;
                next.z += delta.z;
            }

            if let Some(result) = visitor(block) {
                return Some(result);
            }
        }

        None
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the injected reader extends Vanilla's block/fluid clip inputs without changing their shape"
    )]
    pub(super) fn clip_block_and_fluid<R: LevelReader>(
        &self,
        reader: &R,
        pos: BlockPos,
        from: DVec3,
        to: DVec3,
        block_shape: ClipBlockShape,
        fluid: ClipFluid,
        collision_context: BlockCollisionContext,
    ) -> Option<ClipHitResult> {
        let state = reader.get_block_state(pos);
        let shape = self.clip_block_shape(reader, state, pos, block_shape, collision_context);
        let block_result = Self::clip_boxes(pos, from, to, shape.iter())
            .map(|hit| Self::clip_with_interaction_override(pos, from, to, state, hit));
        let fluid_result = if fluid == ClipFluid::None {
            None
        } else {
            Self::clip_fluid_shape(reader, pos, from, to, state, fluid)
        };

        match (block_result, fluid_result) {
            (Some(block_hit), Some(fluid_hit)) => {
                let block_distance = from.distance_squared(block_hit.location);
                let fluid_distance = from.distance_squared(fluid_hit.location);
                if block_distance <= fluid_distance {
                    Some(block_hit)
                } else {
                    Some(fluid_hit)
                }
            }
            (Some(hit), None) | (None, Some(hit)) => Some(hit),
            (None, None) => None,
        }
    }

    pub(super) fn clip_with_interaction_override(
        pos: BlockPos,
        from: DVec3,
        to: DVec3,
        state: BlockStateId,
        block_hit: ClipHitResult,
    ) -> ClipHitResult {
        let Some(override_hit) =
            Self::clip_shape(pos, from, to, state.get_interaction_shape_at(pos))
        else {
            return block_hit;
        };

        if from.distance_squared(override_hit.location) < from.distance_squared(block_hit.location)
        {
            ClipHitResult {
                direction: override_hit.direction,
                ..block_hit
            }
        } else {
            block_hit
        }
    }

    pub(super) fn clip_block_shape<R: LevelReader>(
        &self,
        reader: &R,
        state: BlockStateId,
        pos: BlockPos,
        shape: ClipBlockShape,
        collision_context: BlockCollisionContext,
    ) -> ResolvedBlockCollisionShape {
        match shape {
            ClipBlockShape::Collider => BLOCK_BEHAVIORS
                .get_behavior(state.get_block())
                .get_resolved_collision_shape(state, reader, pos, collision_context),
            ClipBlockShape::Outline => {
                ResolvedBlockCollisionShape::borrowed(state.get_outline_shape_at(pos))
            }
            ClipBlockShape::Visual => {
                ResolvedBlockCollisionShape::borrowed(state.get_visual_shape_at(pos))
            }
            ClipBlockShape::FallDamageResetting { entity_is_player } => {
                ResolvedBlockCollisionShape::borrowed(OffsetVoxelShape::without_offset(
                    self.fall_damage_resetting_shape(state, entity_is_player),
                ))
            }
        }
    }

    pub(super) fn fall_damage_resetting_shape(
        &self,
        state: BlockStateId,
        entity_is_player: bool,
    ) -> VoxelShape {
        let block = state.get_block();
        if block.has_tag(&BlockTag::FALL_DAMAGE_RESETTING) {
            return VoxelShape::FULL_BLOCK;
        }

        if !entity_is_player {
            return VoxelShape::EMPTY;
        }

        if block == &vanilla_blocks::END_GATEWAY || block == &vanilla_blocks::END_PORTAL {
            return VoxelShape::FULL_BLOCK;
        }

        if block == &vanilla_blocks::NETHER_PORTAL
            && self.get_game_rule(&PLAYERS_NETHER_PORTAL_DEFAULT_DELAY) == 0
        {
            return VoxelShape::FULL_BLOCK;
        }

        VoxelShape::EMPTY
    }

    pub(super) fn clip_fluid_shape<R: LevelReader>(
        reader: &R,
        pos: BlockPos,
        from: DVec3,
        to: DVec3,
        state: BlockStateId,
        fluid: ClipFluid,
    ) -> Option<ClipHitResult> {
        let fluid_state = state.get_fluid_state();
        let can_pick = match fluid {
            ClipFluid::None => false,
            ClipFluid::SourceOnly => fluid_state.is_source(),
            ClipFluid::Any => !fluid_state.is_empty(),
            ClipFluid::Water => fluid_state.is_water(),
        };
        if !can_pick {
            return None;
        }

        let height = Self::fluid_clip_height(reader, pos, fluid_state);
        Self::clip_local_aabb(
            pos,
            from,
            to,
            BlockLocalAabb::new(0.0, 0.0, 0.0, 1.0, height, 1.0),
        )
    }

    pub(super) fn fluid_clip_height<R: LevelReader>(
        reader: &R,
        pos: BlockPos,
        fluid_state: FluidState,
    ) -> f64 {
        let above_fluid = reader.get_block_state(pos.above()).get_fluid_state();
        Self::fluid_clip_height_from_above(fluid_state, above_fluid)
    }

    pub(super) fn fluid_clip_height_from_above(
        fluid_state: FluidState,
        above_fluid: FluidState,
    ) -> f64 {
        if FLUID_BEHAVIORS
            .get_behavior(fluid_state.fluid_id)
            .is_same(above_fluid.fluid_id)
        {
            1.0
        } else {
            f64::from(fluid_state.own_height())
        }
    }

    pub(super) fn clip_shape(
        block_pos: BlockPos,
        from: DVec3,
        to: DVec3,
        shape: OffsetVoxelShape,
    ) -> Option<ClipHitResult> {
        Self::clip_boxes(block_pos, from, to, shape.iter())
    }

    fn clip_boxes(
        block_pos: BlockPos,
        from: DVec3,
        to: DVec3,
        boxes: impl IntoIterator<Item = BlockLocalAabb>,
    ) -> Option<ClipHitResult> {
        let ray = BlockShapeClipRay::new(from, to)?;

        let mut closest: Option<(f64, Direction)> = None;

        for shape in boxes {
            match Self::clip_shape_box(block_pos, &ray, shape) {
                Some(BlockShapeBoxHit::Inside) => {
                    return Some(ClipHitResult {
                        location: ray.inside_test_point,
                        direction: Self::approximate_nearest_direction(ray.difference).opposite(),
                        block_pos,
                        miss: false,
                        inside: true,
                        world_border_hit: false,
                    });
                }
                Some(BlockShapeBoxHit::Entry { t, direction })
                    if closest.is_none_or(|(best_t, _)| t < best_t) =>
                {
                    closest = Some((t, direction));
                }
                Some(BlockShapeBoxHit::Entry { .. }) | None => {}
            }
        }

        closest.map(|(t, direction)| ClipHitResult {
            location: ray.from + ray.difference * t,
            direction,
            block_pos,
            miss: false,
            inside: false,
            world_border_hit: false,
        })
    }

    fn shape_blocks_ray(
        block_pos: BlockPos,
        ray: &BlockShapeClipRay,
        boxes: impl IntoIterator<Item = BlockLocalAabb>,
    ) -> bool {
        boxes
            .into_iter()
            .any(|shape| Self::clip_shape_box(block_pos, ray, shape).is_some())
    }

    fn clip_shape_box(
        block_pos: BlockPos,
        ray: &BlockShapeClipRay,
        shape: BlockLocalAabb,
    ) -> Option<BlockShapeBoxHit> {
        if shape.is_empty() {
            return None;
        }

        let block_vec = DVec3::new(
            f64::from(block_pos.x()),
            f64::from(block_pos.y()),
            f64::from(block_pos.z()),
        );
        if Self::local_aabb_contains_world_point(shape, block_vec, ray.inside_test_point) {
            return Some(BlockShapeBoxHit::Inside);
        }

        let world_min = DVec3::new(shape.min_x(), shape.min_y(), shape.min_z()) + block_vec;
        let world_max = DVec3::new(shape.max_x(), shape.max_y(), shape.max_z()) + block_vec;
        Self::intersects_aabb_with_delta(ray.from, ray.difference, world_min, world_max)
            .map(|(t, direction)| BlockShapeBoxHit::Entry { t, direction })
    }

    pub(super) fn clip_local_aabb(
        block_pos: BlockPos,
        from: DVec3,
        to: DVec3,
        aabb: BlockLocalAabb,
    ) -> Option<ClipHitResult> {
        if aabb.is_empty() {
            return None;
        }

        let ray = BlockShapeClipRay::new(from, to)?;
        match Self::clip_shape_box(block_pos, &ray, aabb)? {
            BlockShapeBoxHit::Inside => Some(ClipHitResult {
                location: ray.inside_test_point,
                direction: Self::approximate_nearest_direction(ray.difference).opposite(),
                block_pos,
                miss: false,
                inside: true,
                world_border_hit: false,
            }),
            BlockShapeBoxHit::Entry { t, direction } => Some(ClipHitResult {
                location: ray.from + ray.difference * t,
                direction,
                block_pos,
                miss: false,
                inside: false,
                world_border_hit: false,
            }),
        }
    }

    pub(super) fn local_aabb_contains_world_point(
        aabb: BlockLocalAabb,
        block_vec: DVec3,
        point: DVec3,
    ) -> bool {
        let local = point - block_vec;
        !aabb.is_empty() && aabb.contains(local)
    }

    pub(super) fn clip_miss(from: DVec3, to: DVec3) -> ClipHitResult {
        ClipHitResult {
            location: to,
            direction: Self::approximate_nearest_direction(from - to),
            block_pos: BlockPos::from(to),
            miss: true,
            inside: false,
            world_border_hit: false,
        }
    }

    pub(super) fn approximate_nearest_direction(vector: DVec3) -> Direction {
        let mut result = Direction::North;
        let mut highest_dot = 0.0;
        for direction in [
            Direction::Down,
            Direction::Up,
            Direction::North,
            Direction::South,
            Direction::West,
            Direction::East,
        ] {
            let dot = vector.dot(direction.offset_vec().as_dvec3());
            if dot > highest_dot {
                highest_dot = dot;
                result = direction;
            }
        }
        result
    }

    /// Returns Vanilla's segment entry parameter and face for one AABB.
    #[expect(
        clippy::too_many_lines,
        reason = "the linear axis order mirrors Vanilla's parity-sensitive clip-point sequence"
    )]
    fn intersects_aabb_with_delta(
        start: DVec3,
        difference: DVec3,
        min: DVec3,
        max: DVec3,
    ) -> Option<(f64, Direction)> {
        let mut best_t = 1.0;
        let mut hit_direction = None;

        macro_rules! clip_point {
            (
                $da:expr, $db:expr, $dc:expr, $point:expr,
                $min_b:expr, $max_b:expr, $min_c:expr, $max_c:expr,
                $direction:expr, $from_a:expr, $from_b:expr, $from_c:expr
            ) => {{
                let t = ($point - $from_a) / $da;
                let point_b = $from_b + t * $db;
                let point_c = $from_c + t * $dc;
                if 0.0 < t
                    && t < best_t
                    && $min_b - 1.0e-7 < point_b
                    && point_b < $max_b + 1.0e-7
                    && $min_c - 1.0e-7 < point_c
                    && point_c < $max_c + 1.0e-7
                {
                    best_t = t;
                    hit_direction = Some($direction);
                }
            }};
        }

        if difference.x > 1.0e-7 {
            clip_point!(
                difference.x,
                difference.y,
                difference.z,
                min.x,
                min.y,
                max.y,
                min.z,
                max.z,
                Direction::West,
                start.x,
                start.y,
                start.z
            );
        } else if difference.x < -1.0e-7 {
            clip_point!(
                difference.x,
                difference.y,
                difference.z,
                max.x,
                min.y,
                max.y,
                min.z,
                max.z,
                Direction::East,
                start.x,
                start.y,
                start.z
            );
        }

        if difference.y > 1.0e-7 {
            clip_point!(
                difference.y,
                difference.z,
                difference.x,
                min.y,
                min.z,
                max.z,
                min.x,
                max.x,
                Direction::Down,
                start.y,
                start.z,
                start.x
            );
        } else if difference.y < -1.0e-7 {
            clip_point!(
                difference.y,
                difference.z,
                difference.x,
                max.y,
                min.z,
                max.z,
                min.x,
                max.x,
                Direction::Up,
                start.y,
                start.z,
                start.x
            );
        }

        if difference.z > 1.0e-7 {
            clip_point!(
                difference.z,
                difference.x,
                difference.y,
                min.z,
                min.x,
                max.x,
                min.y,
                max.y,
                Direction::North,
                start.z,
                start.x,
                start.y
            );
        } else if difference.z < -1.0e-7 {
            clip_point!(
                difference.z,
                difference.x,
                difference.y,
                max.z,
                min.x,
                max.x,
                min.y,
                max.y,
                Direction::South,
                start.z,
                start.x,
                start.y
            );
        }

        hit_direction.map(|direction| (best_t, direction))
    }

    /// Performs a raytrace in the world.
    ///
    /// Adapted from Pumpkin project.
    pub fn raytrace<F>(
        &self,
        start_pos: DVec3,
        end_pos: DVec3,
        hit_check: F,
    ) -> (Option<BlockPos>, Option<Direction>)
    where
        F: Fn(BlockPos, &Self) -> RaytraceAction,
    {
        if start_pos == end_pos {
            return (None, None);
        }

        let adjust = -1.0e-7f64;
        let to = end_pos.lerp(start_pos, adjust);
        let from = start_pos.lerp(end_pos, adjust);

        let mut block = BlockPos::new(
            from.x.floor() as i32,
            from.y.floor() as i32,
            from.z.floor() as i32,
        );

        match hit_check(block, self) {
            RaytraceAction::ImmediateHit => return (Some(block), None),
            RaytraceAction::CheckShape => {
                let (hit, face) = self.ray_outline_check(block, start_pos, end_pos);
                if hit {
                    return (Some(block), face);
                }
            }
            RaytraceAction::Pass => {}
        }

        let difference = to - from;

        let step = difference.signum().as_ivec3();

        let delta = DVec3::new(
            if step.x == 0 {
                f64::MAX
            } else {
                (f64::from(step.x)) / difference.x
            },
            if step.y == 0 {
                f64::MAX
            } else {
                (f64::from(step.y)) / difference.y
            },
            if step.z == 0 {
                f64::MAX
            } else {
                (f64::from(step.z)) / difference.z
            },
        );

        let mut next = DVec3::new(
            delta.x
                * (if step.x > 0 {
                    1.0 - (from.x - from.x.floor())
                } else {
                    from.x - from.x.floor()
                }),
            delta.y
                * (if step.y > 0 {
                    1.0 - (from.y - from.y.floor())
                } else {
                    from.y - from.y.floor()
                }),
            delta.z
                * (if step.z > 0 {
                    1.0 - (from.z - from.z.floor())
                } else {
                    from.z - from.z.floor()
                }),
        );

        while next.x <= 1.0 || next.y <= 1.0 || next.z <= 1.0 {
            // Vanilla parity: traverseBlocks tie-breaking — Z wins on any tie.
            // X wins only when strictly less than both Y and Z.
            // Y wins only when strictly less than both X and Z.
            // Everything else (including all ties) goes to Z.
            let block_direction = if next.x < next.y && next.x < next.z {
                block.0.x += step.x;
                next.x += delta.x;
                if step.x > 0 {
                    Direction::West
                } else {
                    Direction::East
                }
            } else if next.y < next.x && next.y < next.z {
                block.0.y += step.y;
                next.y += delta.y;
                if step.y > 0 {
                    Direction::Down
                } else {
                    Direction::Up
                }
            } else {
                block.0.z += step.z;
                next.z += delta.z;
                if step.z > 0 {
                    Direction::North
                } else {
                    Direction::South
                }
            };

            match hit_check(block, self) {
                RaytraceAction::ImmediateHit => {
                    return (Some(block), Some(block_direction));
                }
                RaytraceAction::CheckShape => {
                    let (hit, face) = self.ray_outline_check(block, start_pos, end_pos);
                    if hit {
                        return (Some(block), face);
                    }
                }
                RaytraceAction::Pass => {}
            }
        }

        (None, None)
    }
}
