//! Bounded chunk and section guards for repeated gameplay block reads.

use std::{
    collections::TryReserveError,
    error::Error,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use parking_lot::RwLockReadGuard;
use rustc_hash::{FxHashMap, FxHashSet};
use steel_registry::blocks::block_state_ext::BlockStateExt as _;
use steel_registry::vanilla_blocks;
use steel_utils::{BlockPos, BlockStateId, ChunkPos, SectionPos};

use super::ChunkMap;
use crate::behavior::BLOCK_BEHAVIORS;
use crate::block_entity::{BlockEntityLifecycleExt as _, SharedBlockEntity};
use crate::chunk::{
    chunk_access::{ChunkAccess, ChunkStatus},
    chunk_holder::ChunkHolder,
    level_chunk::LevelChunk,
    section::ChunkSection,
};

/// Failure to represent or allocate a bounded gameplay block-read window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameplayBlockReadWindowError {
    /// The dense region has more cells than the platform can index.
    RegionTooLarge,
    /// Storage for the window could not be reserved.
    AllocationFailed,
    /// The pure pinned operation touched data that requires the live reader.
    RetryLive,
}

impl fmt::Display for GameplayBlockReadWindowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RegionTooLarge => {
                formatter.write_str("gameplay block-read region has too many dense cells")
            }
            Self::AllocationFailed => {
                formatter.write_str("gameplay block-read window allocation failed")
            }
            Self::RetryLive => formatter
                .write_str("pinned gameplay block read touched data that requires a live retry"),
        }
    }
}

impl Error for GameplayBlockReadWindowError {}

impl From<TryReserveError> for GameplayBlockReadWindowError {
    fn from(_error: TryReserveError) -> Self {
        Self::AllocationFailed
    }
}

/// Dense chunk-column and section bounds for repeated gameplay block reads.
///
/// The supplied block corners are expanded to their containing chunk columns and
/// 16-block-tall sections. This keeps recursive neighbor reads within an already
/// acquired cell when they remain in the same chunk section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameplayBlockReadRegion {
    min_block: BlockPos,
    max_block: BlockPos,
    min_chunk_x: i32,
    max_chunk_x: i32,
    min_chunk_z: i32,
    max_chunk_z: i32,
    min_section_y: i32,
    max_section_y: i32,
    chunk_width: usize,
    chunk_depth: usize,
    section_count: usize,
    chunk_slot_count: usize,
    section_slot_count: usize,
}

impl GameplayBlockReadRegion {
    /// Creates dense read bounds containing both block-position corners.
    ///
    /// Corners may be supplied in either order. An error is returned when the
    /// resulting dense chunk/section index cannot be represented by `usize`.
    pub fn from_corners(
        first: BlockPos,
        second: BlockPos,
    ) -> Result<Self, GameplayBlockReadWindowError> {
        let min_block = BlockPos::new(
            first.x().min(second.x()),
            first.y().min(second.y()),
            first.z().min(second.z()),
        );
        let max_block = BlockPos::new(
            first.x().max(second.x()),
            first.y().max(second.y()),
            first.z().max(second.z()),
        );
        let min_chunk_x = SectionPos::block_to_section_coord(min_block.x());
        let max_chunk_x = SectionPos::block_to_section_coord(max_block.x());
        let min_chunk_z = SectionPos::block_to_section_coord(min_block.z());
        let max_chunk_z = SectionPos::block_to_section_coord(max_block.z());
        let min_section_y = SectionPos::block_to_section_coord(min_block.y());
        let max_section_y = SectionPos::block_to_section_coord(max_block.y());

        let chunk_width = inclusive_span(min_chunk_x, max_chunk_x)?;
        let chunk_depth = inclusive_span(min_chunk_z, max_chunk_z)?;
        let section_count = inclusive_span(min_section_y, max_section_y)?;
        let Some(chunk_slot_count) = chunk_width.checked_mul(chunk_depth) else {
            return Err(GameplayBlockReadWindowError::RegionTooLarge);
        };
        let Some(section_slot_count) = chunk_slot_count.checked_mul(section_count) else {
            return Err(GameplayBlockReadWindowError::RegionTooLarge);
        };

        Ok(Self {
            min_block,
            max_block,
            min_chunk_x,
            max_chunk_x,
            min_chunk_z,
            max_chunk_z,
            min_section_y,
            max_section_y,
            chunk_width,
            chunk_depth,
            section_count,
            chunk_slot_count,
            section_slot_count,
        })
    }

    #[inline]
    fn chunk_slot(self, pos: BlockPos) -> Option<usize> {
        let chunk_pos = ChunkPos::from_block_pos(pos);
        let x = chunk_pos.0.x;
        let z = chunk_pos.0.y;
        if x < self.min_chunk_x
            || x > self.max_chunk_x
            || z < self.min_chunk_z
            || z > self.max_chunk_z
        {
            return None;
        }

        let x_offset = usize::try_from(i64::from(x) - i64::from(self.min_chunk_x)).ok()?;
        let z_offset = usize::try_from(i64::from(z) - i64::from(self.min_chunk_z)).ok()?;
        z_offset
            .checked_mul(self.chunk_width)?
            .checked_add(x_offset)
    }

    #[inline]
    fn section_slot(self, pos: BlockPos) -> Option<usize> {
        let chunk_slot = self.chunk_slot(pos)?;
        let section_y = SectionPos::block_to_section_coord(pos.y());
        if section_y < self.min_section_y || section_y > self.max_section_y {
            return None;
        }

        let section_offset =
            usize::try_from(i64::from(section_y) - i64::from(self.min_section_y)).ok()?;
        chunk_slot
            .checked_mul(self.section_count)?
            .checked_add(section_offset)
    }
}

fn inclusive_span(min: i32, max: i32) -> Result<usize, GameplayBlockReadWindowError> {
    let span = i64::from(max) - i64::from(min) + 1;
    usize::try_from(span).map_err(|_error| GameplayBlockReadWindowError::RegionTooLarge)
}

/// Executes a closure with stable identities and read guards for the region's
/// currently published, generation-permitted full chunks.
pub(crate) fn try_with_full_chunk_read_window<R>(
    chunk_map: &ChunkMap,
    region: GameplayBlockReadRegion,
    operation: impl FnOnce(&FullChunkReadWindow<'_>) -> R,
) -> Result<R, GameplayBlockReadWindowError> {
    let mut holders = Vec::new();
    holders.try_reserve_exact(region.chunk_slot_count)?;

    let mut chunk_z = region.min_chunk_z;
    loop {
        let mut chunk_x = region.min_chunk_x;
        loop {
            holders.push(chunk_map.lookup_active_holder(ChunkPos::new(chunk_x, chunk_z)));
            if chunk_x == region.max_chunk_x {
                break;
            }
            chunk_x += 1;
        }
        if chunk_z == region.max_chunk_z {
            break;
        }
        chunk_z += 1;
    }

    let mut chunks = Vec::new();
    chunks.try_reserve_exact(region.chunk_slot_count)?;
    for holder in &holders {
        let guard = holder.as_ref().and_then(|holder| {
            if holder.is_status_disallowed(ChunkStatus::Full) {
                return None;
            }
            holder.try_chunk(ChunkStatus::Full)
        });
        chunks.push(guard);
    }

    let window = FullChunkReadWindow {
        region,
        _holders: &holders,
        chunks,
    };
    Ok(operation(&window))
}

/// Live chunk-column view for a bounded gameplay read operation.
///
/// Section locks are acquired only for individual reads, so serialized gameplay
/// mutations made during the closure remain visible to later reads.
pub(crate) struct FullChunkReadWindow<'holders> {
    region: GameplayBlockReadRegion,
    // Retains the owners that the chunk guards borrow for the full window scope.
    _holders: &'holders [Option<Arc<ChunkHolder>>],
    chunks: Vec<Option<RwLockReadGuard<'holders, ChunkAccess>>>,
}

impl FullChunkReadWindow<'_> {
    #[inline]
    fn full_chunk_at_slot(&self, slot: usize) -> Option<&LevelChunk> {
        self.chunks.get(slot)?.as_deref()?.as_full()
    }

    /// Gets a live block state when the position is covered by a currently
    /// available chunk in this window.
    #[inline]
    pub(crate) fn block_state(&self, pos: BlockPos) -> Option<BlockStateId> {
        let slot = self.region.chunk_slot(pos)?;
        let section_y = SectionPos::block_to_section_coord(pos.y());
        if section_y < self.region.min_section_y || section_y > self.region.max_section_y {
            return None;
        }
        Some(self.full_chunk_at_slot(slot)?.get_block_state(pos))
    }

    /// Gets a live block entity without reacquiring the covered chunk guard.
    ///
    /// The outer `Option` distinguishes an uncovered position from a covered
    /// position whose chunk or block entity is unavailable.
    #[inline]
    #[expect(
        clippy::option_option,
        reason = "the outer option distinguishes uncovered cells from covered cells without an entity"
    )]
    pub(crate) fn block_entity(&self, pos: BlockPos) -> Option<Option<SharedBlockEntity>> {
        let slot = self.region.chunk_slot(pos)?;
        let section_y = SectionPos::block_to_section_coord(pos.y());
        if section_y < self.region.min_section_y || section_y > self.region.max_section_y {
            return None;
        }
        Some(
            self.full_chunk_at_slot(slot)
                .and_then(|chunk| chunk.get_block_entity_immediate(pos)),
        )
    }

    /// Snapshots concrete block entities for the broader pinned `LevelReader` layer.
    ///
    /// This performs no promotion or factory calls. Unresolved entries are recorded
    /// so a query can request a whole-operation live retry after section guards drop.
    pub(crate) fn try_block_entity_snapshot(
        &self,
    ) -> Result<PinnedBlockEntitySnapshot, GameplayBlockReadWindowError> {
        let mut block_entities = FxHashMap::default();
        let mut unresolved = FxHashSet::default();
        for chunk_slot in 0..self.region.chunk_slot_count {
            let Some(chunk) = self.full_chunk_at_slot(chunk_slot) else {
                continue;
            };

            let existing = chunk
                .block_entity_storage()
                .get_all_without_lifecycle_filter();
            let pending = chunk.pending_block_entity_positions();
            block_entities.try_reserve(existing.len())?;
            unresolved.try_reserve(existing.len())?;
            unresolved.try_reserve(pending.len())?;
            for block_entity in existing {
                let pos = block_entity.get_block_pos();
                if self.region.chunk_slot(pos) != Some(chunk_slot)
                    || self.region.section_slot(pos).is_none()
                {
                    continue;
                }
                if block_entity.is_removed() {
                    unresolved.insert(pos);
                } else {
                    block_entities.insert(pos, block_entity);
                }
            }
            for pos in pending {
                if self.region.chunk_slot(pos) == Some(chunk_slot)
                    && self.region.section_slot(pos).is_some()
                {
                    unresolved.insert(pos);
                }
            }
        }

        Ok(PinnedBlockEntitySnapshot {
            block_entities,
            unresolved,
        })
    }

    /// Executes a pure read closure while retaining every covered chunk-section
    /// read guard. Covered sections must not be mutated until the closure returns.
    pub(crate) fn try_with_pinned_sections<R>(
        &self,
        operation: impl FnOnce(&PinnedSectionReadWindow<'_>) -> R,
    ) -> Result<R, GameplayBlockReadWindowError> {
        let mut sections = Vec::new();
        sections.try_reserve_exact(self.region.section_slot_count)?;

        for chunk_slot in 0..self.region.chunk_slot_count {
            let full_chunk = self.full_chunk_at_slot(chunk_slot);
            let mut section_y = self.region.min_section_y;
            loop {
                let guard = full_chunk.and_then(|chunk| {
                    let chunk_min_section = SectionPos::block_to_section_coord(chunk.min_y());
                    let section_index =
                        usize::try_from(i64::from(section_y) - i64::from(chunk_min_section))
                            .ok()?;
                    chunk
                        .sections
                        .sections
                        .get(section_index)
                        .map(|section| section.read())
                });
                sections.push(guard);
                if section_y == self.region.max_section_y {
                    break;
                }
                section_y += 1;
            }
        }

        let window = PinnedSectionReadWindow {
            region: self.region,
            sections,
            retry_live: AtomicBool::new(false),
        };
        let result = operation(&window);
        if window.retry_live.load(Ordering::Relaxed) {
            return Err(GameplayBlockReadWindowError::RetryLive);
        }
        Ok(result)
    }
}

/// Section-pinned view for a bounded, pure gameplay block-read operation.
pub(crate) struct PinnedSectionReadWindow<'chunks> {
    region: GameplayBlockReadRegion,
    sections: Vec<Option<RwLockReadGuard<'chunks, ChunkSection>>>,
    retry_live: AtomicBool,
}

impl PinnedSectionReadWindow<'_> {
    /// Returns whether the exact requested block cuboid contains no collision candidates.
    ///
    /// Unavailable slots are omitted because pinned reads represent them as air.
    /// Callers must separately validate that fallback state.
    pub(crate) fn requested_region_has_no_collision_candidates(&self) -> bool {
        let mut section_slot = 0;
        let mut chunk_z = self.region.min_chunk_z;
        loop {
            let mut chunk_x = self.region.min_chunk_x;
            loop {
                let mut section_y = self.region.min_section_y;
                loop {
                    let Some(section) = self.sections.get(section_slot) else {
                        return false;
                    };
                    if let Some(section) = section.as_deref() {
                        let section_min_x = i64::from(chunk_x) * 16;
                        let section_min_y = i64::from(section_y) * 16;
                        let section_min_z = i64::from(chunk_z) * 16;
                        let local_min_x = (i64::from(self.region.min_block.x()).max(section_min_x)
                            - section_min_x) as usize;
                        let local_max_x = (i64::from(self.region.max_block.x())
                            .min(section_min_x + 15)
                            - section_min_x) as usize;
                        let local_min_y = (i64::from(self.region.min_block.y()).max(section_min_y)
                            - section_min_y) as usize;
                        let local_max_y = (i64::from(self.region.max_block.y())
                            .min(section_min_y + 15)
                            - section_min_y) as usize;
                        let local_min_z = (i64::from(self.region.min_block.z()).max(section_min_z)
                            - section_min_z) as usize;
                        let local_max_z = (i64::from(self.region.max_block.z())
                            .min(section_min_z + 15)
                            - section_min_z) as usize;
                        if section.has_collision_candidate_in_box(
                            local_min_x,
                            local_max_x,
                            local_min_y,
                            local_max_y,
                            local_min_z,
                            local_max_z,
                        ) {
                            return false;
                        }
                    }

                    section_slot += 1;
                    if section_y == self.region.max_section_y {
                        break;
                    }
                    section_y += 1;
                }
                if chunk_x == self.region.max_chunk_x {
                    break;
                }
                chunk_x += 1;
            }
            if chunk_z == self.region.max_chunk_z {
                break;
            }
            chunk_z += 1;
        }
        true
    }

    #[inline]
    fn block_state_lookup(&self, pos: BlockPos) -> PinnedBlockStateLookup {
        let Some(section_slot) = self.region.section_slot(pos) else {
            return PinnedBlockStateLookup::Uncovered;
        };
        let Some(section) = self.sections.get(section_slot) else {
            return PinnedBlockStateLookup::Uncovered;
        };
        let Some(section) = section.as_deref() else {
            return PinnedBlockStateLookup::Unavailable;
        };
        PinnedBlockStateLookup::State(section.states().get(
            (pos.x() & 15) as usize,
            (pos.y() & 15) as usize,
            (pos.z() & 15) as usize,
        ))
    }

    /// Gets a pinned block state when the position is covered by a currently
    /// admitted chunk and section in this window.
    #[inline]
    pub(crate) fn block_state(&self, pos: BlockPos) -> Option<BlockStateId> {
        match self.block_state_lookup(pos) {
            PinnedBlockStateLookup::State(state) => Some(state),
            PinnedBlockStateLookup::Unavailable => Some(vanilla_blocks::AIR.default_state()),
            PinnedBlockStateLookup::Uncovered => {
                self.retry_live.store(true, Ordering::Relaxed);
                None
            }
        }
    }

    /// Returns the exact state only when its collision shape may be non-empty.
    ///
    /// Uncovered reads conservatively return provisional air after requesting a
    /// live retry, so callers cannot mistake an incomplete snapshot for a proof.
    #[inline]
    pub(crate) fn collision_candidate_state(&self, pos: BlockPos) -> Option<BlockStateId> {
        let lookup = match self.region.section_slot(pos) {
            Some(section_slot) => match self.sections.get(section_slot) {
                Some(Some(section)) => section
                    .collision_candidate_state(
                        (pos.x() & 15) as usize,
                        (pos.y() & 15) as usize,
                        (pos.z() & 15) as usize,
                    )
                    .map_or(PinnedCollisionStateLookup::GuaranteedEmpty, |state| {
                        PinnedCollisionStateLookup::State(state)
                    }),
                Some(None) => PinnedCollisionStateLookup::Unavailable,
                None => PinnedCollisionStateLookup::Uncovered,
            },
            None => PinnedCollisionStateLookup::Uncovered,
        };

        match lookup {
            PinnedCollisionStateLookup::GuaranteedEmpty => None,
            PinnedCollisionStateLookup::State(state) => Some(state),
            PinnedCollisionStateLookup::Unavailable => {
                let air = vanilla_blocks::AIR.default_state();
                (!BLOCK_BEHAVIORS.is_collision_shape_guaranteed_empty(air)).then_some(air)
            }
            PinnedCollisionStateLookup::Uncovered => {
                self.request_live_retry();
                Some(vanilla_blocks::AIR.default_state())
            }
        }
    }

    /// Requests a whole-operation live retry for a reader surface that is not
    /// part of this pinned block/section snapshot.
    #[inline]
    pub(crate) fn request_live_retry(&self) {
        self.retry_live.store(true, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy)]
enum PinnedBlockStateLookup {
    State(BlockStateId),
    Unavailable,
    Uncovered,
}

#[derive(Clone, Copy)]
enum PinnedCollisionStateLookup {
    GuaranteedEmpty,
    State(BlockStateId),
    Unavailable,
    Uncovered,
}

/// Concrete block entities captured before a broader `LevelReader` pins sections.
pub(crate) struct PinnedBlockEntitySnapshot {
    block_entities: FxHashMap<BlockPos, SharedBlockEntity>,
    unresolved: FxHashSet<BlockPos>,
}

impl PinnedBlockEntitySnapshot {
    /// Returns a snapshotted block entity or requests a whole-operation live retry.
    #[inline]
    pub(crate) fn block_entity(
        &self,
        sections: &PinnedSectionReadWindow<'_>,
        pos: BlockPos,
    ) -> Option<SharedBlockEntity> {
        let state = match sections.block_state_lookup(pos) {
            PinnedBlockStateLookup::State(state) => state,
            PinnedBlockStateLookup::Unavailable => return None,
            PinnedBlockStateLookup::Uncovered => {
                sections.request_live_retry();
                return None;
            }
        };
        if let Some(block_entity) = self.block_entities.get(&pos) {
            return Some(Arc::clone(block_entity));
        }
        if state.has_block_entity() || self.unresolved.contains(&pos) {
            sections.request_live_retry();
        }
        None
    }
}
