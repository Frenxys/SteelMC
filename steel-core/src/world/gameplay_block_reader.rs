//! Closure-scoped gameplay block readers for bounded repeated queries.

use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::blocks::properties::Direction;
use steel_registry::blocks::shapes::SupportType;
use steel_registry::vanilla_blocks;
use steel_utils::{BlockPos, BlockStateId};

use super::{LevelReader, World};
use crate::behavior::BLOCK_BEHAVIORS;
use crate::block_entity::SharedBlockEntity;
use crate::chunk::chunk_map::{
    FullChunkReadWindow, PinnedBlockEntitySnapshot, PinnedSectionReadWindow,
    try_with_full_chunk_read_window,
};

pub use crate::chunk::chunk_map::{GameplayBlockReadRegion, GameplayBlockReadWindowError};

trait BlockStateWindow: Sync {
    fn block_state(&self, pos: BlockPos) -> Option<BlockStateId>;
}

impl BlockStateWindow for FullChunkReadWindow<'_> {
    #[inline]
    fn block_state(&self, pos: BlockPos) -> Option<BlockStateId> {
        Self::block_state(self, pos)
    }
}

impl BlockStateWindow for PinnedSectionReadWindow<'_> {
    #[inline]
    fn block_state(&self, pos: BlockPos) -> Option<BlockStateId> {
        Self::block_state(self, pos)
    }
}

struct GameplayBlockReader<'world, 'window> {
    world: &'world World,
    window: &'window dyn BlockStateWindow,
    mode: ReaderMode<'window>,
}

#[derive(Clone, Copy)]
enum ReaderMode<'window> {
    Live(&'window FullChunkReadWindow<'window>),
    Pinned {
        sections: &'window PinnedSectionReadWindow<'window>,
        block_entities: &'window PinnedBlockEntitySnapshot,
    },
}

impl LevelReader for GameplayBlockReader<'_, '_> {
    #[inline]
    fn get_block_state(&self, pos: BlockPos) -> BlockStateId {
        if !self.world.is_in_valid_bounds(pos) {
            return vanilla_blocks::VOID_AIR.default_state();
        }

        match self.mode {
            ReaderMode::Live(_) => self
                .window
                .block_state(pos)
                .unwrap_or_else(|| self.world.get_block_state(pos)),
            ReaderMode::Pinned { .. } => self
                .window
                .block_state(pos)
                .unwrap_or_else(|| vanilla_blocks::AIR.default_state()),
        }
    }

    fn get_block_entity(&self, pos: BlockPos) -> Option<SharedBlockEntity> {
        if !self.world.is_in_valid_bounds(pos) {
            return None;
        }

        match self.mode {
            ReaderMode::Live(window) => window
                .block_entity(pos)
                .unwrap_or_else(|| self.world.get_block_entity(pos)),
            ReaderMode::Pinned {
                sections,
                block_entities,
            } => block_entities.block_entity(sections, pos),
        }
    }

    fn is_face_sturdy_for(
        &self,
        state: BlockStateId,
        pos: BlockPos,
        direction: Direction,
        support_type: SupportType,
    ) -> bool {
        BLOCK_BEHAVIORS
            .get_behavior(state.get_block())
            .is_face_sturdy(state, self, pos, direction, support_type)
    }

    fn raw_brightness(&self, pos: BlockPos, sky_darkening: u8) -> u8 {
        match self.mode {
            ReaderMode::Live(_) => {
                <World as LevelReader>::raw_brightness(self.world, pos, sky_darkening)
            }
            ReaderMode::Pinned { sections, .. } => {
                sections.request_live_retry();
                0
            }
        }
    }

    fn can_see_sky(&self, pos: BlockPos) -> bool {
        match self.mode {
            ReaderMode::Live(_) => self.world.can_see_sky(pos),
            ReaderMode::Pinned { sections, .. } => {
                sections.request_live_retry();
                false
            }
        }
    }

    fn ambient_light(&self) -> f32 {
        <World as LevelReader>::ambient_light(self.world)
    }

    fn min_y(&self) -> i32 {
        self.world.get_min_y()
    }

    fn height(&self) -> i32 {
        self.world.get_height()
    }
}

/// Strict state-only view for a pinned gameplay block-read batch.
///
/// The view is `Sync` and may be shared with scoped worker threads. It does not
/// allocate or inspect block entities. Broader `LevelReader` queries that cannot
/// be answered from the state snapshot request a whole-operation live retry.
pub struct GameplayBlockStateReadBatch<'world, 'window> {
    world: &'world World,
    window: &'window PinnedSectionReadWindow<'window>,
}

impl GameplayBlockStateReadBatch<'_, '_> {
    /// Gets a state from the pinned section snapshot.
    ///
    /// World-invalid positions return void air. A read beyond the bounded cells
    /// returns provisional air and makes the enclosing batch return
    /// [`GameplayBlockReadWindowError::RetryLive`]. A covered unavailable chunk
    /// returns air directly, matching [`World::get_block_state`].
    #[inline]
    #[must_use]
    pub fn get_block_state(&self, pos: BlockPos) -> BlockStateId {
        if !self.world.is_in_valid_bounds(pos) {
            return vanilla_blocks::VOID_AIR.default_state();
        }
        self.window
            .block_state(pos)
            .unwrap_or_else(|| vanilla_blocks::AIR.default_state())
    }

    /// Returns whether the exact requested cuboid has only behavior-authoritative,
    /// guaranteed-empty collision shapes.
    ///
    /// The caller must still ensure its operation cannot read beyond the region.
    /// Unavailable covered sections read as air, while invalid world positions
    /// read as void air, so both fallback states are part of the proof.
    pub(crate) fn requested_region_has_only_guaranteed_empty_colliders(&self) -> bool {
        let block_behaviors = &*BLOCK_BEHAVIORS;
        if !block_behaviors.is_collision_shape_guaranteed_empty(vanilla_blocks::AIR.default_state())
            || !block_behaviors
                .is_collision_shape_guaranteed_empty(vanilla_blocks::VOID_AIR.default_state())
        {
            return false;
        }

        self.window.requested_region_has_no_collision_candidates()
    }
}

impl LevelReader for GameplayBlockStateReadBatch<'_, '_> {
    #[inline]
    fn get_block_state(&self, pos: BlockPos) -> BlockStateId {
        Self::get_block_state(self, pos)
    }

    #[inline]
    fn get_collision_candidate_state(&self, pos: BlockPos) -> Option<BlockStateId> {
        if !self.world.is_in_valid_bounds(pos) {
            let state = vanilla_blocks::VOID_AIR.default_state();
            return (!BLOCK_BEHAVIORS.is_collision_shape_guaranteed_empty(state)).then_some(state);
        }
        self.window.collision_candidate_state(pos)
    }

    fn get_block_entity(&self, _pos: BlockPos) -> Option<SharedBlockEntity> {
        self.window.request_live_retry();
        None
    }

    fn is_face_sturdy_for(
        &self,
        state: BlockStateId,
        pos: BlockPos,
        direction: Direction,
        support_type: SupportType,
    ) -> bool {
        BLOCK_BEHAVIORS
            .get_behavior(state.get_block())
            .is_face_sturdy(state, self, pos, direction, support_type)
    }

    fn raw_brightness(&self, _pos: BlockPos, _sky_darkening: u8) -> u8 {
        self.window.request_live_retry();
        0
    }

    fn can_see_sky(&self, _pos: BlockPos) -> bool {
        self.window.request_live_retry();
        false
    }

    fn ambient_light(&self) -> f32 {
        <World as LevelReader>::ambient_light(self.world)
    }

    fn min_y(&self) -> i32 {
        self.world.get_min_y()
    }

    fn height(&self) -> i32 {
        self.world.get_height()
    }
}

impl World {
    /// Executes a bounded gameplay query with stable full-chunk guards.
    ///
    /// Individual section locks are released after each lookup, so serialized
    /// gameplay mutations made by the caller remain visible to later reads. A
    /// lookup outside the bounded cells uses the normal live-world path.
    pub fn with_gameplay_block_reader<R>(
        &self,
        region: GameplayBlockReadRegion,
        operation: impl FnOnce(&dyn LevelReader) -> R,
    ) -> Result<R, GameplayBlockReadWindowError> {
        try_with_full_chunk_read_window(&self.chunk_map, region, |window| {
            let reader = GameplayBlockReader {
                world: self,
                window,
                mode: ReaderMode::Live(window),
            };
            operation(&reader)
        })
    }

    /// Executes a strict state-only batch with covered chunk sections pinned.
    ///
    /// This is the lowest-overhead bulk path: it does not snapshot block entities
    /// and its reader is shareable by scoped worker threads. The operation must be
    /// pure because an uncovered read discards its result and returns
    /// [`GameplayBlockReadWindowError::RetryLive`] after all guards are released.
    pub fn with_gameplay_block_state_read_batch<R>(
        &self,
        region: GameplayBlockReadRegion,
        operation: impl FnOnce(&GameplayBlockStateReadBatch<'_, '_>) -> R,
    ) -> Result<R, GameplayBlockReadWindowError> {
        try_with_full_chunk_read_window(&self.chunk_map, region, |chunks| {
            chunks.try_with_pinned_sections(|window| {
                let reader = GameplayBlockStateReadBatch {
                    world: self,
                    window,
                };
                operation(&reader)
            })
        })?
    }

    /// Executes a bounded pure read batch with covered chunk sections pinned.
    ///
    /// The caller must not mutate covered state or cause other externally visible
    /// side effects. Dynamic behavior receives this scoped reader, keeping its
    /// recursive block reads on the same pinned view. If it touches an uncovered
    /// cell, unresolved block entity, or unpinned light data, the provisional
    /// result is discarded and [`GameplayBlockReadWindowError::RetryLive`] is
    /// returned after every guard is released. The caller may then rerun the
    /// entire pure operation with [`Self::with_gameplay_block_reader`].
    pub fn with_gameplay_block_read_batch<R>(
        &self,
        region: GameplayBlockReadRegion,
        operation: impl FnOnce(&(dyn LevelReader + Sync)) -> R,
    ) -> Result<R, GameplayBlockReadWindowError> {
        try_with_full_chunk_read_window(&self.chunk_map, region, |chunks| {
            let block_entities = chunks.try_block_entity_snapshot()?;
            chunks.try_with_pinned_sections(|window| {
                let reader = GameplayBlockReader {
                    world: self,
                    window,
                    mode: ReaderMode::Pinned {
                        sections: window,
                        block_entities: &block_entities,
                    },
                };
                operation(&reader)
            })
        })?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use steel_registry::blocks::properties::{BlockStateProperties, Direction, PistonType};
    use steel_registry::blocks::shapes::SupportType;
    use steel_registry::{test_support::init_test_registry, vanilla_blocks};
    use steel_utils::types::UpdateFlags;
    use steel_utils::{BlockPos, ChunkPos, SectionPos};

    use super::*;
    use crate::behavior::init_behaviors;
    use crate::block_entity::{
        SharedBlockEntity, entities::PistonMovingBlockEntity, init_block_entities,
    };
    use crate::chunk::chunk_access::ChunkStatus;
    use crate::chunk::chunk_holder::ChunkHolder;
    use crate::test_support::{fresh_test_world, insert_ready_full_chunk};

    fn read_region(first: BlockPos, second: BlockPos) -> GameplayBlockReadRegion {
        let Ok(region) = GameplayBlockReadRegion::from_corners(first, second) else {
            panic!("small test region should be representable");
        };
        region
    }

    fn requested_region_has_only_guaranteed_empty_colliders(
        reader: &GameplayBlockStateReadBatch<'_, '_>,
    ) -> bool {
        reader.requested_region_has_only_guaranteed_empty_colliders()
    }

    fn section_write_is_available(holder: &ChunkHolder, pos: BlockPos) -> bool {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Full) else {
            return false;
        };
        let Some(chunk) = chunk.as_full() else {
            return false;
        };
        let chunk_min_section = SectionPos::block_to_section_coord(chunk.min_y());
        let section_y = SectionPos::block_to_section_coord(pos.y());
        let Ok(section_index) = usize::try_from(i64::from(section_y - chunk_min_section)) else {
            return false;
        };
        let Some(section) = chunk.sections.sections.get(section_index) else {
            return false;
        };
        section.try_write().is_some()
    }

    #[test]
    fn live_reader_observes_mutations_made_inside_its_scope() {
        init_test_registry();
        init_behaviors();
        let world = fresh_test_world("live_gameplay_block_reader");
        let pos = BlockPos::new(3, 64, 5);
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));
        assert!(world.set_block(
            pos,
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));

        let result = world.with_gameplay_block_reader(read_region(pos, pos), |reader| {
            assert_eq!(
                reader.get_block_state(pos),
                vanilla_blocks::STONE.default_state()
            );
            assert!(world.set_block(
                pos,
                vanilla_blocks::DIRT.default_state(),
                UpdateFlags::UPDATE_NONE,
            ));
            assert_eq!(
                reader.get_block_state(pos),
                vanilla_blocks::DIRT.default_state()
            );
        });

        assert!(result.is_ok(), "live read window should be allocated");
    }

    #[test]
    fn pinned_reader_retains_then_releases_covered_section_guards() {
        init_test_registry();
        let world = fresh_test_world("pinned_gameplay_block_reader");
        let pos = BlockPos::new(3, 64, 5);
        let holder = insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));
        assert!(section_write_is_available(&holder, pos));

        let result = world.with_gameplay_block_state_read_batch(read_region(pos, pos), |reader| {
            fn assert_sync<T: Sync + ?Sized>(_value: &T) {}
            assert_sync(reader);
            assert_eq!(
                reader.get_block_state(pos),
                vanilla_blocks::AIR.default_state()
            );
            assert!(
                !section_write_is_available(&holder, pos),
                "the whole containing section stays pinned during the batch"
            );
        });

        assert!(result.is_ok(), "pinned read window should be allocated");
        assert!(section_write_is_available(&holder, pos));
    }

    #[test]
    fn pinned_state_reader_preserves_world_bounds_and_unavailable_air() {
        init_test_registry();
        init_behaviors();
        let world = fresh_test_world("pinned_gameplay_block_reader_bounds");
        let loaded = BlockPos::new(0, 64, 0);
        let unloaded = BlockPos::new(16, 64, 0);
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(loaded));
        assert!(world.set_block(
            loaded,
            vanilla_blocks::CAVE_AIR.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));

        let bounded =
            world.with_gameplay_block_state_read_batch(read_region(loaded, loaded), |reader| {
                assert_eq!(
                    reader.get_block_state(loaded),
                    vanilla_blocks::CAVE_AIR.default_state()
                );
                assert_eq!(
                    reader.get_block_state(BlockPos::new(0, world.get_min_y() - 1, 0)),
                    vanilla_blocks::VOID_AIR.default_state()
                );
                assert_eq!(
                    reader.get_block_state(BlockPos::new(
                        BlockPos::MAX_HORIZONTAL_COORDINATE,
                        64,
                        0,
                    )),
                    vanilla_blocks::VOID_AIR.default_state()
                );
            });
        assert!(bounded.is_ok(), "world bounds need no live retry");

        let pinned = world
            .with_gameplay_block_state_read_batch(read_region(loaded, unloaded), |reader| {
                reader.get_block_state(unloaded)
            });
        assert_eq!(pinned, Ok(vanilla_blocks::AIR.default_state()));
    }

    #[test]
    fn pinned_collision_empty_proof_tracks_palette_contents_and_unavailable_air() {
        init_test_registry();
        init_behaviors();
        let world = fresh_test_world("pinned_collision_empty_proof");
        let pos = BlockPos::new(3, 96, 5);
        let outside_requested_region = BlockPos::new(12, 96, 12);
        let unavailable = BlockPos::new(16, 96, 5);
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));

        let proves_empty = |first, second| {
            world.with_gameplay_block_state_read_batch(read_region(first, second), |reader| {
                reader.requested_region_has_only_guaranteed_empty_colliders()
            })
        };

        assert_eq!(proves_empty(pos, unavailable), Ok(true));
        assert!(world.set_block(
            outside_requested_region,
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert_eq!(proves_empty(pos, pos), Ok(true));
        assert_eq!(proves_empty(pos, outside_requested_region), Ok(false));

        assert!(world.set_block(
            pos,
            vanilla_blocks::LIGHT.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert_eq!(proves_empty(pos, pos), Ok(true));
        assert_eq!(
            world.with_gameplay_block_state_read_batch(read_region(pos, pos), |reader| {
                LevelReader::get_collision_candidate_state(reader, pos)
            }),
            Ok(None)
        );

        assert!(world.set_block(
            pos,
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert_eq!(proves_empty(pos, pos), Ok(false));
        assert_eq!(
            world.with_gameplay_block_state_read_batch(read_region(pos, pos), |reader| {
                LevelReader::get_collision_candidate_state(reader, pos)
            }),
            Ok(Some(vanilla_blocks::STONE.default_state()))
        );

        assert!(world.set_block(
            pos,
            vanilla_blocks::MOVING_PISTON.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert_eq!(proves_empty(pos, pos), Ok(false));

        assert!(world.set_block(
            pos,
            vanilla_blocks::AIR.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert_eq!(proves_empty(pos, pos), Ok(true));
    }

    #[test]
    fn pinned_collision_empty_proof_clips_signed_section_boundaries() {
        init_test_registry();
        init_behaviors();
        let world = fresh_test_world("pinned_collision_signed_boundaries");
        for chunk_z in -1..=0 {
            for chunk_x in -1..=0 {
                insert_ready_full_chunk(&world, ChunkPos::new(chunk_x, chunk_z));
            }
        }

        let requested_min = BlockPos::new(-1, 95, -1);
        let requested_max = BlockPos::new(0, 96, 0);
        let outside_requested_region = BlockPos::new(-16, 80, -16);
        assert!(world.set_block(
            outside_requested_region,
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));

        let proves_empty = || {
            world.with_gameplay_block_state_read_batch(
                read_region(requested_min, requested_max),
                requested_region_has_only_guaranteed_empty_colliders,
            )
        };
        assert_eq!(proves_empty(), Ok(true));

        assert!(world.set_block(
            requested_max,
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert_eq!(proves_empty(), Ok(false));
    }

    #[test]
    fn pinned_reader_requests_live_retry_beyond_its_section_cells() {
        init_test_registry();
        init_behaviors();
        let world = fresh_test_world("pinned_gameplay_block_reader_fallback");
        let covered = BlockPos::new(15, 64, 0);
        let outside = BlockPos::new(16, 64, 0);
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(covered));
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(outside));
        assert!(world.set_block(
            outside,
            vanilla_blocks::STONE.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));

        let pinned = world
            .with_gameplay_block_state_read_batch(read_region(covered, covered), |reader| {
                reader.get_block_state(outside)
            });
        assert_eq!(pinned, Err(GameplayBlockReadWindowError::RetryLive));

        let collision = world
            .with_gameplay_block_state_read_batch(read_region(covered, covered), |reader| {
                LevelReader::get_collision_candidate_state(reader, outside)
            });
        assert_eq!(collision, Err(GameplayBlockReadWindowError::RetryLive));

        let live = world.with_gameplay_block_reader(read_region(covered, covered), |reader| {
            reader.get_block_state(outside)
        });
        assert_eq!(live, Ok(vanilla_blocks::STONE.default_state()));
    }

    #[test]
    fn pinned_reader_requests_live_retry_for_light_queries() {
        init_test_registry();
        let world = fresh_test_world("pinned_gameplay_block_reader_light_retry");
        let pos = BlockPos::new(3, 64, 5);
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));

        let result = world.with_gameplay_block_read_batch(read_region(pos, pos), |reader| {
            reader.raw_brightness(pos, 0)
        });
        assert_eq!(result, Err(GameplayBlockReadWindowError::RetryLive));
    }

    #[test]
    fn state_only_reader_requests_live_retry_for_block_entity_queries() {
        init_test_registry();
        let world = fresh_test_world("state_only_gameplay_block_reader_entity_retry");
        let pos = BlockPos::new(3, 64, 5);
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));

        let result = world.with_gameplay_block_state_read_batch(read_region(pos, pos), |reader| {
            LevelReader::get_block_entity(reader, pos).is_some()
        });

        assert_eq!(result, Err(GameplayBlockReadWindowError::RetryLive));
    }

    #[test]
    fn pinned_reader_requests_live_retry_for_missing_block_entity() {
        init_test_registry();
        init_behaviors();
        init_block_entities();
        let world = fresh_test_world("pinned_gameplay_block_reader_missing_entity");
        let pos = BlockPos::new(3, 64, 5);
        let holder = insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));
        assert!(world.set_block(
            pos,
            vanilla_blocks::CHEST.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));
        assert!(world.get_block_entity(pos).is_some());
        {
            let Some(chunk) = holder.try_chunk(ChunkStatus::Full) else {
                panic!("ready holder should expose its full chunk");
            };
            let Some(chunk) = chunk.as_full() else {
                panic!("ready holder should contain a LevelChunk");
            };
            assert!(chunk.remove_block_entity(pos));
        }

        let pinned = world.with_gameplay_block_read_batch(read_region(pos, pos), |reader| {
            reader.get_block_entity(pos).is_some()
        });
        assert_eq!(pinned, Err(GameplayBlockReadWindowError::RetryLive));

        let live = world.with_gameplay_block_reader(read_region(pos, pos), |reader| {
            reader.get_block_entity(pos).is_some()
        });
        assert_eq!(live, Ok(true));
    }

    #[test]
    fn oversized_dense_region_reports_its_distinct_setup_error() {
        assert_eq!(
            GameplayBlockReadRegion::from_corners(
                BlockPos::new(i32::MIN, i32::MIN, i32::MIN),
                BlockPos::new(i32::MAX, i32::MAX, i32::MAX),
            ),
            Err(GameplayBlockReadWindowError::RegionTooLarge)
        );
    }

    #[test]
    fn pinned_dynamic_moving_piston_shape_matches_live_block_entity_view() {
        init_test_registry();
        init_behaviors();
        let world = fresh_test_world("pinned_gameplay_block_reader_piston");
        let pos = BlockPos::new(8, 64, 8);
        insert_ready_full_chunk(&world, ChunkPos::from_block_pos(pos));
        let moving_state = vanilla_blocks::MOVING_PISTON
            .default_state()
            .set_value(&BlockStateProperties::FACING, Direction::East)
            .set_value(&BlockStateProperties::PISTON_TYPE, PistonType::Normal);
        let moved_state = vanilla_blocks::PISTON
            .default_state()
            .set_value(&BlockStateProperties::FACING, Direction::East)
            .set_value(&BlockStateProperties::EXTENDED, true);
        assert!(moving_state.has_block_entity());
        assert!(world.set_block(pos, moving_state, UpdateFlags::UPDATE_NONE));
        let block_entity: SharedBlockEntity = Arc::new(PistonMovingBlockEntity::new_moving(
            Arc::downgrade(&world),
            pos,
            moving_state,
            moved_state,
            Direction::East,
            false,
            true,
        ));
        assert!(world.set_block_entity(block_entity));

        let live_west =
            world.is_face_sturdy_for(moving_state, pos, Direction::West, SupportType::Full);
        assert!(
            live_west,
            "the retracting source keeps its rear support face"
        );

        let result = world.with_gameplay_block_read_batch(read_region(pos, pos), |reader| {
            assert!(reader.get_block_entity(pos).is_some());
            assert_eq!(
                reader.is_face_sturdy_for(moving_state, pos, Direction::West, SupportType::Full,),
                live_west
            );
            assert!(!reader.is_face_sturdy_for(
                moving_state,
                pos,
                Direction::East,
                SupportType::Full,
            ));
        });

        assert!(result.is_ok(), "pinned read window should be allocated");
    }
}
