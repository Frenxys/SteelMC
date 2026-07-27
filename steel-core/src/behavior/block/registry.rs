use super::{
    BlockBehavior, BlockPlaceContext, BlockPos, BlockRef, BlockStateId, CollisionShapeSource,
    Direction, REGISTRY, RegistryEntry, RegistryExt, ScheduledTickAccess,
    schedule_water_tick_if_waterlogged,
};

/// Default placement plus the common water tick used by unported waterlogged blocks.
pub struct DefaultBlockBehavior {
    block: BlockRef,
}

impl DefaultBlockBehavior {
    /// Creates a new default block behavior for the given block.
    #[must_use]
    pub const fn new(block: BlockRef) -> Self {
        Self { block }
    }
}

impl BlockBehavior for DefaultBlockBehavior {
    fn collision_shape_source(&self, _state: BlockStateId) -> CollisionShapeSource {
        if self.block.config.dynamic_shape {
            CollisionShapeSource::Behavior
        } else {
            CollisionShapeSource::RegistryStatic
        }
    }

    fn update_shape(
        &self,
        state: BlockStateId,
        world: &dyn ScheduledTickAccess,
        pos: BlockPos,
        _direction: Direction,
        _neighbor_pos: BlockPos,
        _neighbor_state: BlockStateId,
    ) -> BlockStateId {
        schedule_water_tick_if_waterlogged(state, world, pos);
        state
    }

    // This fallback only preserves generic block placement. Blocks with
    // class-specific placement rules still need their own behavior.
    fn get_state_for_placement(&self, _context: &BlockPlaceContext<'_>) -> Option<BlockStateId> {
        Some(self.block.default_state())
    }
}

/// Registry for block behaviors.
///
/// Created after the main registry is frozen. All blocks are initialized with
/// default behaviors, then custom behaviors are registered for specific blocks.
pub struct BlockBehaviorRegistry {
    behaviors: Vec<Box<dyn BlockBehavior>>,
    guaranteed_empty_collision: Box<[bool]>,
}

impl BlockBehaviorRegistry {
    /// Get all behaviors.
    #[cfg(feature = "flint")]
    #[must_use]
    pub fn get_behaviors(&self) -> &[Box<dyn BlockBehavior>] {
        &self.behaviors
    }

    /// Creates a new behavior registry with default behaviors for all blocks.
    #[must_use]
    pub fn new() -> Self {
        let block_count = REGISTRY.blocks.len();
        let state_count = REGISTRY.blocks.state_to_block_lookup.len();
        let mut behaviors: Vec<Box<dyn BlockBehavior>> = Vec::with_capacity(block_count);

        // Initialize all blocks with default behavior
        for (_, block) in REGISTRY.blocks.iter() {
            behaviors.push(Box::new(DefaultBlockBehavior::new(block)));
        }

        let mut registry = Self {
            behaviors,
            guaranteed_empty_collision: vec![false; state_count].into_boxed_slice(),
        };
        for (_, block) in REGISTRY.blocks.iter() {
            registry.refresh_guaranteed_empty_collision(block);
        }
        registry
    }

    /// Sets a custom behavior for a block.
    pub fn set_behavior(&mut self, block: BlockRef, behavior: Box<dyn BlockBehavior>) {
        let id = block.id();
        self.behaviors[id] = behavior;
        self.refresh_guaranteed_empty_collision(block);
    }

    fn refresh_guaranteed_empty_collision(&mut self, block: BlockRef) {
        let base_state = REGISTRY.blocks.get_base_state_id(block).0;
        let behavior = self.behaviors[block.id()].as_ref();

        for offset in 0..block.state_count() {
            let state = BlockStateId(base_state + offset);
            self.guaranteed_empty_collision[usize::from(state.0)] =
                behavior.collision_shape_source(state) == CollisionShapeSource::RegistryStatic
                    && REGISTRY.blocks.get_static_collision_shape(state).is_empty();
        }
    }

    /// Returns whether the installed behavior guarantees an empty collider for this state.
    #[must_use]
    #[inline]
    pub fn is_collision_shape_guaranteed_empty(&self, state: BlockStateId) -> bool {
        self.guaranteed_empty_collision
            .get(usize::from(state.0))
            .copied()
            .unwrap_or(false)
    }

    /// Gets the behavior for a block.
    #[must_use]
    pub fn get_behavior(&self, block: BlockRef) -> &dyn BlockBehavior {
        let id = block.id();
        self.behaviors[id].as_ref()
    }

    /// Gets the behavior for a block by its ID.
    #[must_use]
    pub fn get_behavior_by_id(&self, id: usize) -> Option<&dyn BlockBehavior> {
        self.behaviors.get(id).map(AsRef::as_ref)
    }

    /// Gets the behavior for a block state.
    #[must_use]
    pub fn get_behavior_for_state(&self, state: BlockStateId) -> Option<&dyn BlockBehavior> {
        let block = REGISTRY.blocks.by_state_id(state)?;
        Some(self.get_behavior(block))
    }
}

impl Default for BlockBehaviorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use steel_registry::{
        blocks::shapes::VoxelShape, test_support::init_test_registry, vanilla_blocks,
    };

    use super::*;
    use crate::behavior::BlockCollisionContext;
    use crate::world::LevelReader;

    struct ContextDependentCollision;

    impl BlockBehavior for ContextDependentCollision {
        fn get_state_for_placement(
            &self,
            _context: &BlockPlaceContext<'_>,
        ) -> Option<BlockStateId> {
            None
        }

        fn get_collision_shape(
            &self,
            _state: BlockStateId,
            _world: &dyn LevelReader,
            _pos: BlockPos,
            context: BlockCollisionContext,
        ) -> VoxelShape {
            if context.is_placement() {
                VoxelShape::EMPTY
            } else {
                VoxelShape::FULL_BLOCK
            }
        }
    }

    #[test]
    fn default_static_empty_collision_is_cached() {
        init_test_registry();
        let registry = BlockBehaviorRegistry::new();

        assert!(registry.is_collision_shape_guaranteed_empty(vanilla_blocks::AIR.default_state()));
        assert!(
            !registry.is_collision_shape_guaranteed_empty(vanilla_blocks::STONE.default_state())
        );
    }

    #[test]
    fn behavior_replacement_invalidates_guaranteed_empty_collision() {
        init_test_registry();
        let mut registry = BlockBehaviorRegistry::new();
        let block = &vanilla_blocks::LIGHT;
        let states = REGISTRY.blocks.matching_states(block, &[]);
        assert!(states.len() > 1);
        assert!(
            states
                .iter()
                .all(|&state| registry.is_collision_shape_guaranteed_empty(state))
        );

        registry.set_behavior(block, Box::new(ContextDependentCollision));

        assert!(
            states
                .iter()
                .all(|&state| !registry.is_collision_shape_guaranteed_empty(state))
        );
    }

    #[test]
    fn default_dynamic_collision_is_not_cached() {
        init_test_registry();
        let registry = BlockBehaviorRegistry::new();

        assert!(vanilla_blocks::MOVING_PISTON.config.dynamic_shape);
        assert!(
            !registry
                .is_collision_shape_guaranteed_empty(vanilla_blocks::MOVING_PISTON.default_state())
        );
    }
}
