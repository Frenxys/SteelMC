//! World-level entity ownership and lookup.
//!
//! Steel deliberately uses a simpler loaded/simulated split than vanilla's
//! entity section manager. The manager owns runtime entity lookup regardless
//! of chunk load state; chunks are still the persistence boundary, and only
//! full simulated chunks tick entities.

mod spatial_index;

use std::{
    collections::{BTreeMap, VecDeque},
    error::Error,
    fmt, mem, ptr, slice,
    sync::Arc,
};

use glam::DVec3;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use steel_registry::vanilla_entities;
use steel_utils::locks::{SyncMutex, SyncRwLock};
use steel_utils::{ChunkPos, PackedSectionPos, SectionPos, WorldAabb};
use uuid::Uuid;

use super::{
    BoundEntityCallback, Entity, EntityCallbackToken, EntityLevelCallback, EntitySpatialChange,
    EntitySpatialCommitResult, EntitySpatialUpdate, InactiveEntityCallback, NullEntityCallback,
    RemovalReason, SharedEntity, snapshot_old_pos_and_rot_for_tick,
    tick_vehicle_passengers_with_ticked_if,
};
use spatial_index::{EntitySpatialIndex, EntitySpatialMembership};

/// Error returned when adding an entity to the runtime world fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddEntityError {
    /// The entity is in a chunk that is not active in the world entity manager.
    ChunkNotLoaded {
        /// Entity network ID.
        entity_id: i32,
        /// Chunk containing the entity.
        chunk: ChunkPos,
    },
    /// Another live entity with the same persistent UUID is already registered.
    DuplicateUuid {
        /// Entity network ID.
        entity_id: i32,
        /// Duplicate persistent UUID.
        uuid: Uuid,
    },
    /// The entity is already removed and cannot be added to the live world.
    RemovedEntity {
        /// Entity network ID.
        entity_id: i32,
    },
}

impl fmt::Display for AddEntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChunkNotLoaded { entity_id, chunk } => {
                write!(f, "entity {entity_id} is in non-loaded chunk {chunk:?}")
            }
            Self::DuplicateUuid { entity_id, uuid } => {
                write!(f, "entity {entity_id} has duplicate UUID {uuid}")
            }
            Self::RemovedEntity { entity_id } => {
                write!(f, "entity {entity_id} is already removed")
            }
        }
    }
}

impl Error for AddEntityError {}

/// Error returned when a live entity move cannot be committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityMoveError {
    /// The entity is no longer managed as live world state.
    NotLive {
        /// Entity network ID.
        entity_id: i32,
    },
    /// The entity is deliberately frozen outside live world membership.
    Inactive {
        /// Entity network ID.
        entity_id: i32,
    },
    /// The entity tried to move into a chunk outside active world ownership.
    UnloadedDestination {
        /// Entity network ID.
        entity_id: i32,
        /// Destination chunk.
        chunk: ChunkPos,
    },
}

impl fmt::Display for EntityMoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLive { entity_id } => {
                write!(f, "entity {entity_id} is not live in the world")
            }
            Self::Inactive { entity_id } => {
                write!(f, "entity {entity_id} is inactive outside live world state")
            }
            Self::UnloadedDestination { entity_id, chunk } => {
                write!(
                    f,
                    "entity {entity_id} cannot move into non-loaded chunk {chunk:?}"
                )
            }
        }
    }
}

impl Error for EntityMoveError {}

/// Whether the manager owns persistence for an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityOwnership {
    /// Normal non-player entity owned by the world entity manager.
    ManagerOwned,
    /// Entity whose lifetime is owned elsewhere, such as a player.
    External,
}

/// Entity visibility for a chunk column.
///
/// Mirrors vanilla `Visibility`: hidden chunks keep entity data inactive,
/// tracked chunks expose entities to lookup/tracking, and ticking chunks also
/// run manager-owned entity ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityVisibility {
    /// Not accessible to entity lookup/tracking and not ticking.
    Hidden,
    /// Accessible to entity lookup/tracking but not ticking.
    Tracked,
    /// Accessible to entity lookup/tracking and ticking.
    Ticking,
}

impl EntityVisibility {
    /// Returns whether entities in this visibility are accessible to queries and tracking.
    #[must_use]
    pub const fn is_accessible(self) -> bool {
        matches!(self, Self::Tracked | Self::Ticking)
    }

    /// Returns whether entities in this visibility are eligible for ticking.
    #[must_use]
    pub const fn is_ticking(self) -> bool {
        matches!(self, Self::Ticking)
    }
}

/// Entity lifecycle changes caused by manager membership or visibility updates.
#[derive(Default)]
pub struct EntityLifecycleChanges {
    /// Entities that became tracked.
    pub tracking_started: Vec<SharedEntity>,
    /// Entities that stopped being tracked.
    pub tracking_stopped: Vec<SharedEntity>,
    /// Entities that entered the world entity tick list.
    pub ticking_started: Vec<SharedEntity>,
    /// Entities that left the world entity tick list.
    pub ticking_stopped: Vec<SharedEntity>,
}

impl fmt::Debug for EntityLifecycleChanges {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EntityLifecycleChanges")
            .field("tracking_started", &self.tracking_started.len())
            .field("tracking_stopped", &self.tracking_stopped.len())
            .field("ticking_started", &self.ticking_started.len())
            .field("ticking_stopped", &self.ticking_stopped.len())
            .finish()
    }
}

impl EntityLifecycleChanges {
    fn extend(&mut self, other: Self) {
        self.tracking_started.extend(other.tracking_started);
        self.tracking_stopped.extend(other.tracking_stopped);
        self.ticking_started.extend(other.ticking_started);
        self.ticking_stopped.extend(other.ticking_stopped);
    }
}

/// Section/chunk membership update caused by a committed entity move.
#[derive(Clone)]
pub struct EntityMoveUpdate {
    entity: SharedEntity,
    ownership: EntityOwnership,
    /// Entity network ID.
    pub entity_id: i32,
    /// Previous section membership.
    pub old_section: SectionPos,
    /// New section membership.
    pub new_section: SectionPos,
    /// Previous chunk membership.
    pub old_chunk: ChunkPos,
    /// New chunk membership.
    pub new_chunk: ChunkPos,
    /// Whether the entity was visible to normal world/tracker queries before the move.
    pub old_accessible: bool,
    /// Whether the entity is visible to normal world/tracker queries after the move.
    pub new_accessible: bool,
    /// Whether the manager-owned entity was in the tick list before the move.
    pub old_ticking: bool,
    /// Whether the manager-owned entity is in the tick list after the move.
    pub new_ticking: bool,
    spatial_update: EntitySpatialUpdate,
}

pub(crate) enum EntityManagerEffect {
    TrackingStart(SharedEntity),
    TrackingStop(i32),
    SpatialMove(EntityMoveUpdate),
    Removal {
        entity: SharedEntity,
        chunk: ChunkPos,
        ownership: EntityOwnership,
    },
}

#[derive(Default)]
struct EntityManagerEffectQueue {
    pending: VecDeque<EntityManagerEffect>,
    dispatching: bool,
}

struct EntityManagerEffectDrainGuard<'a> {
    queue: &'a SyncMutex<EntityManagerEffectQueue>,
    armed: bool,
}

impl Drop for EntityManagerEffectDrainGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.queue.lock().dispatching = false;
        }
    }
}

impl EntityMoveUpdate {
    pub(crate) const fn spatial_update(&self) -> EntitySpatialUpdate {
        self.spatial_update
    }

    pub(crate) fn entity(&self) -> &SharedEntity {
        &self.entity
    }

    pub(crate) const fn ownership(&self) -> EntityOwnership {
        self.ownership
    }

    /// Returns whether the entity changed sections.
    #[must_use]
    pub fn section_changed(&self) -> bool {
        self.old_section != self.new_section
    }

    /// Returns whether the entity changed chunks.
    #[must_use]
    pub fn chunk_changed(&self) -> bool {
        self.old_chunk != self.new_chunk
    }

    /// Returns whether the entity crossed an accessibility boundary.
    #[must_use]
    pub const fn accessibility_changed(&self) -> bool {
        self.old_accessible != self.new_accessible
    }

    /// Returns whether this move made a previously hidden entity accessible.
    #[must_use]
    pub const fn became_accessible(&self) -> bool {
        !self.old_accessible && self.new_accessible
    }

    /// Returns whether this move made a previously accessible entity hidden.
    #[must_use]
    pub const fn became_inaccessible(&self) -> bool {
        self.old_accessible && !self.new_accessible
    }

    /// Returns whether this move made a previously non-ticking entity tick.
    #[must_use]
    pub const fn became_ticking(&self) -> bool {
        !self.old_ticking && self.new_ticking
    }

    /// Returns whether this move made a previously ticking entity stop ticking.
    #[must_use]
    pub const fn became_non_ticking(&self) -> bool {
        self.old_ticking && !self.new_ticking
    }
}

/// Saveable entity that could not be persisted by a chunk save pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsavedEntityReport {
    /// Entity network ID.
    pub entity_id: i32,
    /// Entity persistent UUID.
    pub uuid: Uuid,
    /// Chunk containing the entity.
    pub chunk: ChunkPos,
}

/// Entity changes produced when a chunk becomes loaded.
#[derive(Default)]
pub struct ChunkEntityLoadResult {
    /// Retained entities restored to live world membership.
    pub restored: Vec<SharedEntity>,
    /// Live entities in this chunk whose tracking became visible again.
    pub tracking_started: Vec<SharedEntity>,
    /// Live entities in this chunk whose ticking became active again.
    pub ticking_started: Vec<SharedEntity>,
    /// Whether recovery created save-pending entity state for this chunk.
    pub needs_save: bool,
}

/// Entity changes produced when a chunk starts unloading.
#[derive(Default)]
pub struct ChunkEntityUnloadStart {
    /// Entities removed from live ownership and retained for chunk recovery.
    pub retained: Vec<SharedEntity>,
    /// Entities whose tracker visibility should stop for this chunk transition.
    pub tracking_stopped: Vec<SharedEntity>,
    /// Entities whose ticking should stop for this chunk transition.
    pub ticking_stopped: Vec<SharedEntity>,
}

#[derive(Clone)]
struct EntityEntry {
    entity: SharedEntity,
    callback_token: EntityCallbackToken,
    uuid: Uuid,
    section: SectionPos,
    chunk: ChunkPos,
    committed_position: DVec3,
    committed_bounding_box: WorldAabb,
    committed_spatial_revision: u64,
    ownership: EntityOwnership,
    section_insertion_order: u64,
    spatial_membership: EntitySpatialMembership,
    retained_callback: Option<Arc<dyn EntityLevelCallback>>,
}

struct EntityQueryCandidate {
    entity: SharedEntity,
    bounding_box: WorldAabb,
    order: (i32, PackedSectionPos, u64),
}

impl EntityEntry {
    fn new(
        entity: SharedEntity,
        ownership: EntityOwnership,
        spatial_update: EntitySpatialUpdate,
        callback_token: EntityCallbackToken,
    ) -> Self {
        let committed_position = spatial_update.position();
        let section = SectionPos::from_entity_pos(committed_position);
        let chunk = ChunkPos::new(section.x(), section.z());
        let committed_bounding_box = spatial_update.bounding_box();
        Self {
            uuid: entity.uuid(),
            entity,
            callback_token,
            section,
            chunk,
            committed_position,
            committed_bounding_box,
            committed_spatial_revision: spatial_update.revision(),
            ownership,
            section_insertion_order: 0,
            spatial_membership: EntitySpatialMembership::default(),
            retained_callback: None,
        }
    }

    fn refresh_spatial_update(&mut self, spatial_update: EntitySpatialUpdate) {
        self.committed_position = spatial_update.position();
        self.committed_bounding_box = spatial_update.bounding_box();
        self.committed_spatial_revision = spatial_update.revision();
        self.section = SectionPos::from_entity_pos(self.committed_position);
        self.chunk = ChunkPos::new(self.section.x(), self.section.z());
    }

    #[must_use]
    fn should_save(&self) -> bool {
        self.ownership == EntityOwnership::ManagerOwned
            && (!self.entity.is_removed()
                || self
                    .entity
                    .removal_reason()
                    .is_some_and(RemovalReason::should_save))
            && !self.entity.is_passenger()
            && !self.entity.has_exactly_one_player_passenger()
            && self.entity.entity_type().can_serialize
    }
}

#[derive(Default)]
struct ManagerState {
    chunk_visibility: FxHashMap<ChunkPos, EntityVisibility>,
    live_by_id: FxHashMap<i32, EntityEntry>,
    live_by_uuid: FxHashMap<Uuid, i32>,
    accessible_order: OrderedEntityIds,
    by_section: BTreeMap<PackedSectionPos, OrderedEntityIds>,
    by_chunk: FxHashMap<ChunkPos, FxHashSet<i32>>,
    spatial_index: EntitySpatialIndex,
    next_section_insertion_order: u64,
    unloading_by_chunk: FxHashMap<ChunkPos, Vec<EntityEntry>>,
    save_pending_by_chunk: FxHashMap<ChunkPos, Vec<EntityEntry>>,
    tick_list: EntityTickList,
}

#[derive(Default)]
struct OrderedEntityIds {
    ids: Vec<i32>,
}

impl OrderedEntityIds {
    fn insert(&mut self, entity_id: i32) -> bool {
        if self.ids.contains(&entity_id) {
            return false;
        }
        self.ids.push(entity_id);
        true
    }

    fn remove(&mut self, entity_id: i32) -> bool {
        let Some(index) = self.ids.iter().position(|id| *id == entity_id) else {
            return false;
        };
        self.ids.remove(index);
        true
    }

    const fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn iter(&self) -> impl Iterator<Item = &i32> {
        self.ids.iter()
    }
}

#[derive(Default)]
struct EntityTickList {
    active: FxHashMap<i32, SharedEntity>,
    order: Vec<i32>,
}

impl EntityTickList {
    fn add(&mut self, entity: &SharedEntity) -> bool {
        let entity_id = entity.id();
        if self.active.insert(entity_id, entity.clone()).is_some() {
            return false;
        }
        self.order.push(entity_id);
        true
    }

    fn remove(&mut self, entity_id: i32) -> Option<SharedEntity> {
        let removed = self.active.remove(&entity_id)?;
        self.order.retain(|id| *id != entity_id);
        Some(removed)
    }

    fn contains(&self, entity_id: i32) -> bool {
        self.active.contains_key(&entity_id)
    }

    fn snapshot(&self) -> Vec<SharedEntity> {
        self.order
            .iter()
            .filter_map(|id| self.active.get(id))
            .cloned()
            .collect()
    }
}

/// Central world entity manager.
pub struct WorldEntityManager {
    state: SyncRwLock<ManagerState>,
    effects: SyncMutex<EntityManagerEffectQueue>,
}

impl fmt::Debug for WorldEntityManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.read();
        let effects = self.effects.lock();
        f.debug_struct("WorldEntityManager")
            .field("chunk_visibility", &state.chunk_visibility.len())
            .field("live_entities", &state.live_by_id.len())
            .field("unloading_chunks", &state.unloading_by_chunk.len())
            .field("pending_effects", &effects.pending.len())
            .field("dispatching_effects", &effects.dispatching)
            .finish()
    }
}

impl WorldEntityManager {
    /// Creates an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: SyncRwLock::new(ManagerState::default()),
            effects: SyncMutex::new(EntityManagerEffectQueue::default()),
        }
    }

    fn enqueue_tracking_effects(
        &self,
        tracking_stopped: &[SharedEntity],
        tracking_started: &[SharedEntity],
    ) {
        if tracking_stopped.is_empty() && tracking_started.is_empty() {
            return;
        }

        let mut effects = self.effects.lock();
        effects.pending.extend(
            tracking_stopped
                .iter()
                .map(|entity| EntityManagerEffect::TrackingStop(entity.id())),
        );
        effects.pending.extend(
            tracking_started
                .iter()
                .cloned()
                .map(EntityManagerEffect::TrackingStart),
        );
    }

    fn enqueue_lifecycle_effects(&self, changes: &EntityLifecycleChanges) {
        self.enqueue_tracking_effects(&changes.tracking_stopped, &changes.tracking_started);
    }

    fn enqueue_effect(&self, effect: EntityManagerEffect) {
        self.effects.lock().pending.push_back(effect);
    }

    /// Applies manager effects in the same order as their state commits.
    ///
    /// The queue lock is never held while `handler` runs. Reentrant drain calls
    /// return immediately; the active drainer will observe any newly queued work.
    pub(crate) fn drain_effects(&self, mut handler: impl FnMut(EntityManagerEffect)) {
        {
            let mut effects = self.effects.lock();
            if effects.dispatching {
                return;
            }
            effects.dispatching = true;
        }

        let mut guard = EntityManagerEffectDrainGuard {
            queue: &self.effects,
            armed: true,
        };
        loop {
            let effect = {
                let mut effects = self.effects.lock();
                let Some(effect) = effects.pending.pop_front() else {
                    effects.dispatching = false;
                    guard.armed = false;
                    return;
                };
                effect
            };
            handler(effect);
        }
    }

    /// Returns whether runtime entity ownership for this chunk is loaded.
    ///
    /// This is Vanilla's separate `areEntitiesLoaded` gate used by block-entity
    /// ticking; it is intentionally not the stricter entity-ticking visibility.
    #[must_use]
    pub(crate) fn is_chunk_loaded(&self, pos: ChunkPos) -> bool {
        self.state.read().chunk_visibility.contains_key(&pos)
    }

    /// Marks a chunk as loaded and reactivates retained unloading entities.
    ///
    /// # Panics
    ///
    /// Panics if retained manager state is missing its live callback binding.
    pub fn on_chunk_loaded(&self, pos: ChunkPos) -> ChunkEntityLoadResult {
        let mut state = self.state.write();
        state
            .chunk_visibility
            .entry(pos)
            .or_insert(EntityVisibility::Hidden);

        let mut result = ChunkEntityLoadResult::default();
        if let Some(entries) = state.unloading_by_chunk.remove(&pos) {
            result.restored.reserve(entries.len());
            for mut entry in entries {
                if entry.entity.is_removed() {
                    entry.retained_callback = None;
                    entry
                        .entity
                        .base()
                        .replace_level_callback(Arc::new(NullEntityCallback));
                    if entry.should_save() {
                        result.needs_save = true;
                        state
                            .save_pending_by_chunk
                            .entry(pos)
                            .or_default()
                            .push(entry);
                    }
                    continue;
                }

                let Some(callback) = entry.retained_callback.take() else {
                    panic!(
                        "retained entity {} had no live callback to restore",
                        entry.entity.id()
                    );
                };
                entry.entity.base().replace_level_callback(callback);
                let spatial_update = entry.entity.base().spatial_update();
                entry.refresh_spatial_update(spatial_update);
                if entry.entity.is_removed() {
                    entry
                        .entity
                        .base()
                        .replace_level_callback(Arc::new(NullEntityCallback));
                    if entry.should_save() {
                        result.needs_save = true;
                        state
                            .save_pending_by_chunk
                            .entry(pos)
                            .or_default()
                            .push(entry);
                    }
                    continue;
                }
                let entity = entry.entity.clone();
                Self::insert_live_entry(&mut state, entry);
                let lifecycle = Self::apply_entity_lifecycle_after_insert(&mut state, entity.id());
                result.tracking_started.extend(lifecycle.tracking_started);
                result.ticking_started.extend(lifecycle.ticking_started);
                result.restored.push(entity);
            }
        }

        self.enqueue_tracking_effects(&[], &result.tracking_started);
        result
    }

    /// Updates the entity visibility for a chunk column.
    pub fn update_chunk_visibility(
        &self,
        pos: ChunkPos,
        visibility: EntityVisibility,
    ) -> EntityLifecycleChanges {
        let mut state = self.state.write();
        let previous = state
            .chunk_visibility
            .insert(pos, visibility)
            .unwrap_or(EntityVisibility::Hidden);

        if previous == visibility {
            return EntityLifecycleChanges::default();
        }

        let changes = Self::apply_chunk_visibility_change(&mut state, pos, previous, visibility);
        self.enqueue_lifecycle_effects(&changes);
        changes
    }

    fn push_unique_entity(
        entity: &SharedEntity,
        seen: &mut FxHashSet<i32>,
        entities: &mut Vec<SharedEntity>,
    ) {
        if seen.insert(entity.id()) {
            entities.push(entity.clone());
        }
    }

    /// Moves manager-owned root entities in `pos` out of live world membership while
    /// retaining them for possible chunk recovery.
    pub fn begin_chunk_unload(&self, pos: ChunkPos) -> ChunkEntityUnloadStart {
        let mut state = self.state.write();
        let previous_visibility = state
            .chunk_visibility
            .remove(&pos)
            .unwrap_or(EntityVisibility::Hidden);

        let ids = Self::entity_ids_in_chunk_order(&state, pos);

        let mut result = ChunkEntityUnloadStart::default();
        let lifecycle = Self::apply_chunk_visibility_change(
            &mut state,
            pos,
            previous_visibility,
            EntityVisibility::Hidden,
        );
        let mut tracking_stopped_ids = lifecycle
            .tracking_stopped
            .iter()
            .map(|entity| entity.id())
            .collect::<FxHashSet<_>>();
        result.tracking_stopped = lifecycle.tracking_stopped;
        result.ticking_stopped = lifecycle.ticking_stopped;

        let mut root_ids = Vec::new();
        for entity_id in ids {
            let Some(entry) = state.live_by_id.get(&entity_id) else {
                continue;
            };
            if entry.ownership != EntityOwnership::ManagerOwned {
                continue;
            }

            Self::push_unique_entity(
                &entry.entity,
                &mut tracking_stopped_ids,
                &mut result.tracking_stopped,
            );
            if !entry.entity.is_passenger() {
                root_ids.push(entity_id);
            }
        }

        let mut retained = Vec::new();
        let mut visited = FxHashSet::default();
        for entity_id in root_ids {
            Self::retain_unloading_entity_tree(
                &mut state,
                entity_id,
                &mut visited,
                &mut retained,
                &mut result.retained,
                &mut tracking_stopped_ids,
                &mut result.tracking_stopped,
            );
        }

        if !retained.is_empty() {
            state
                .unloading_by_chunk
                .entry(pos)
                .or_default()
                .extend(retained);
        }

        self.enqueue_tracking_effects(&result.tracking_stopped, &[]);
        result
    }

    fn retain_unloading_entity_tree(
        state: &mut ManagerState,
        entity_id: i32,
        visited: &mut FxHashSet<i32>,
        retained: &mut Vec<EntityEntry>,
        retained_entities: &mut Vec<SharedEntity>,
        tracking_stopped_ids: &mut FxHashSet<i32>,
        tracking_stopped: &mut Vec<SharedEntity>,
    ) {
        if !visited.insert(entity_id) {
            return;
        }

        let Some(mut entry) = Self::remove_live_entry(state, entity_id) else {
            return;
        };

        if entry.ownership != EntityOwnership::ManagerOwned {
            let restored_id = entry.entity.id();
            Self::insert_live_entry(state, entry);
            let entity_to_tick = state.live_by_id.get(&restored_id).and_then(|entry| {
                let visibility = Self::lifecycle_visibility_for(
                    entry,
                    Self::chunk_visibility(state, entry.chunk),
                );
                visibility.is_ticking().then(|| entry.entity.clone())
            });
            if let Some(entity) = entity_to_tick {
                state.tick_list.add(&entity);
            }
            return;
        }

        let passengers = entry.entity.passengers();
        let callback = entry
            .entity
            .base()
            .replace_level_callback(Arc::new(InactiveEntityCallback::new(entity_id)));
        assert!(
            entry.retained_callback.replace(callback).is_none(),
            "live entity {entity_id} already retained a callback"
        );
        Self::push_unique_entity(&entry.entity, tracking_stopped_ids, tracking_stopped);
        retained_entities.push(Arc::clone(&entry.entity));
        retained.push(entry);
        for passenger in passengers {
            Self::retain_unloading_entity_tree(
                state,
                passenger.id(),
                visited,
                retained,
                retained_entities,
                tracking_stopped_ids,
                tracking_stopped,
            );
        }
    }

    /// Finalizes an unloading chunk. Retained entities are detached and dropped.
    pub fn finalize_chunk_unload(&self, pos: ChunkPos) {
        let entries = self
            .state
            .write()
            .unloading_by_chunk
            .remove(&pos)
            .unwrap_or_default();

        for entry in entries {
            entry
                .entity
                .base()
                .replace_level_callback(Arc::new(NullEntityCallback));
            entry.entity.set_removed(RemovalReason::UnloadedToChunk);
        }
    }

    #[cfg(test)]
    /// Registers an entity with a local test callback.
    pub fn add_live_entity(
        &self,
        entity: SharedEntity,
        ownership: EntityOwnership,
    ) -> Result<EntityLifecycleChanges, AddEntityError> {
        self.add_live_entity_with_callback(
            entity,
            ownership,
            BoundEntityCallback::new(|_| Arc::new(NullEntityCallback)),
        )
    }

    /// Registers a live runtime entity and atomically binds its manager callback.
    ///
    /// # Panics
    ///
    /// Panics if an entity with the same session network ID is already present. Duplicate runtime
    /// IDs indicate corrupted manager ownership and cannot be recovered without losing identity.
    pub(crate) fn add_live_entity_with_callback(
        &self,
        entity: SharedEntity,
        ownership: EntityOwnership,
        callback: BoundEntityCallback,
    ) -> Result<EntityLifecycleChanges, AddEntityError> {
        let (callback, callback_token) = callback.into_parts();
        let mut entry = Self::checked_live_entry(entity, ownership, callback_token)?;
        let entity_id = entry.entity.id();
        let mut state = self.state.write();
        let previous_callback = entry.entity.base().replace_level_callback(callback);
        let spatial_update = entry.entity.base().spatial_update();
        entry.refresh_spatial_update(spatial_update);
        if entry.entity.is_removed() {
            entry
                .entity
                .base()
                .replace_level_callback(previous_callback);
            return Err(AddEntityError::RemovedEntity { entity_id });
        }
        if let Err(error) =
            Self::validate_live_entries(&state, slice::from_ref(&entry), ownership, true)
        {
            entry
                .entity
                .base()
                .replace_level_callback(previous_callback);
            return Err(error);
        }
        Self::insert_live_entry(&mut state, entry);
        let lifecycle = Self::apply_entity_lifecycle_after_insert(&mut state, entity_id);
        self.enqueue_lifecycle_effects(&lifecycle);
        Ok(lifecycle)
    }

    #[cfg(test)]
    /// Registers an entity tree with local test callbacks.
    pub fn add_live_entity_tree(
        &self,
        entities: &[SharedEntity],
        ownership: EntityOwnership,
    ) -> Result<EntityLifecycleChanges, AddEntityError> {
        let callbacks = entities
            .iter()
            .map(|_| BoundEntityCallback::new(|_| Arc::new(NullEntityCallback)))
            .collect::<Vec<_>>();
        self.add_live_entity_tree_with_callbacks(entities, ownership, callbacks)
    }

    /// Adds a related group of live entities and binds all callbacks atomically.
    ///
    /// Use this for persisted vehicle/passenger trees so registration either
    /// publishes the whole tree or leaves world indexes and callback bindings unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the entity tree contains the same session network ID more
    /// than once or `callbacks` does not contain exactly one callback per entity.
    pub(crate) fn add_live_entity_tree_with_callbacks(
        &self,
        entities: &[SharedEntity],
        ownership: EntityOwnership,
        callbacks: Vec<BoundEntityCallback>,
    ) -> Result<EntityLifecycleChanges, AddEntityError> {
        assert_eq!(
            entities.len(),
            callbacks.len(),
            "live entity tree requires exactly one callback per entity"
        );
        let mut entries = Vec::with_capacity(entities.len());
        let mut level_callbacks = Vec::with_capacity(callbacks.len());
        for (entity, callback) in entities.iter().zip(callbacks) {
            let (callback, callback_token) = callback.into_parts();
            entries.push(Self::checked_live_entry(
                Arc::clone(entity),
                ownership,
                callback_token,
            )?);
            level_callbacks.push(callback);
        }

        let mut seen_ids = FxHashSet::default();
        let mut seen_uuids = FxHashSet::default();
        for entry in &entries {
            let entity_id = entry.entity.id();
            assert!(
                seen_ids.insert(entity_id),
                "entity id {entity_id} appears more than once in a live entity tree"
            );
            if !seen_uuids.insert(entry.uuid) {
                return Err(AddEntityError::DuplicateUuid {
                    entity_id,
                    uuid: entry.uuid,
                });
            }
        }

        let mut state = self.state.write();
        let mut previous_callbacks = Vec::with_capacity(entries.len());
        for (entry, callback) in entries.iter_mut().zip(&level_callbacks) {
            previous_callbacks.push(
                entry
                    .entity
                    .base()
                    .replace_level_callback(Arc::clone(callback)),
            );
            let spatial_update = entry.entity.base().spatial_update();
            entry.refresh_spatial_update(spatial_update);
        }
        let validation = entries
            .iter()
            .find(|entry| entry.entity.is_removed())
            .map_or_else(
                || Self::validate_live_entries(&state, &entries, ownership, false),
                |entry| {
                    Err(AddEntityError::RemovedEntity {
                        entity_id: entry.entity.id(),
                    })
                },
            );
        if let Err(error) = validation {
            for (entry, previous_callback) in entries.iter().zip(previous_callbacks) {
                entry
                    .entity
                    .base()
                    .replace_level_callback(previous_callback);
            }
            return Err(error);
        }
        let entity_ids = entries
            .iter()
            .map(|entry| entry.entity.id())
            .collect::<Vec<_>>();
        for entry in entries {
            Self::insert_live_entry(&mut state, entry);
        }
        let mut lifecycle = EntityLifecycleChanges::default();
        for entity_id in entity_ids {
            lifecycle.extend(Self::apply_entity_lifecycle_after_insert(
                &mut state, entity_id,
            ));
        }
        self.enqueue_lifecycle_effects(&lifecycle);
        Ok(lifecycle)
    }

    fn checked_live_entry(
        entity: SharedEntity,
        ownership: EntityOwnership,
        callback_token: EntityCallbackToken,
    ) -> Result<EntityEntry, AddEntityError> {
        if entity.is_removed() {
            return Err(AddEntityError::RemovedEntity {
                entity_id: entity.id(),
            });
        }

        let spatial_update = entity.base().spatial_update();
        Ok(EntityEntry::new(
            entity,
            ownership,
            spatial_update,
            callback_token,
        ))
    }

    fn validate_live_entries(
        state: &ManagerState,
        entries: &[EntityEntry],
        ownership: EntityOwnership,
        require_loaded_chunks: bool,
    ) -> Result<(), AddEntityError> {
        for entry in entries {
            let entity_id = entry.entity.id();
            assert!(
                !Self::contains_id(state, entity_id),
                "entity id {entity_id} is already registered in the world entity manager"
            );
            if Self::contains_uuid(state, entry.uuid) {
                return Err(AddEntityError::DuplicateUuid {
                    entity_id,
                    uuid: entry.uuid,
                });
            }
            if require_loaded_chunks
                && ownership == EntityOwnership::ManagerOwned
                && !state.chunk_visibility.contains_key(&entry.chunk)
            {
                return Err(AddEntityError::ChunkNotLoaded {
                    entity_id,
                    chunk: entry.chunk,
                });
            }
        }
        Ok(())
    }

    /// Removes this exact live entity for an explicit removal reason.
    ///
    /// A different entity that reused the same network ID is left untouched.
    pub fn remove_live_entity(
        &self,
        entity: &dyn Entity,
        reason: RemovalReason,
    ) -> Option<SharedEntity> {
        self.remove_live_entity_matching(entity, None, reason)
    }

    /// Removes an entity only if the callback still owns its current manager binding.
    pub(crate) fn remove_live_entity_if_bound(
        &self,
        entity: &dyn Entity,
        callback_token: &EntityCallbackToken,
        reason: RemovalReason,
    ) -> Option<SharedEntity> {
        self.remove_live_entity_matching(entity, Some(callback_token), reason)
    }

    fn remove_live_entity_matching(
        &self,
        entity: &dyn Entity,
        callback_token: Option<&EntityCallbackToken>,
        reason: RemovalReason,
    ) -> Option<SharedEntity> {
        let entity_id = entity.id();
        let mut state = self.state.write();
        let current = state.live_by_id.get(&entity_id)?;
        if !ptr::eq(current.entity.as_ref(), entity) {
            return None;
        }
        if callback_token.is_some_and(|token| !current.callback_token.matches(token)) {
            return None;
        }
        let entry = Self::remove_live_entry(&mut state, entity_id)?;
        let entity = entry.entity.clone();
        let chunk = entry.chunk;
        let ownership = entry.ownership;
        entity
            .base()
            .replace_level_callback(Arc::new(NullEntityCallback));

        if reason.should_save() && entry.should_save() {
            state
                .save_pending_by_chunk
                .entry(entry.chunk)
                .or_default()
                .push(entry);
        }

        self.enqueue_effect(EntityManagerEffect::Removal {
            entity: Arc::clone(&entity),
            chunk,
            ownership,
        });
        Some(entity)
    }

    /// Acknowledges that selected save-pending entities for `chunk` were persisted.
    pub fn on_chunk_saved(&self, chunk: ChunkPos, saved_entity_ids: &[i32]) {
        if saved_entity_ids.is_empty() {
            return;
        }

        let saved_entity_ids = saved_entity_ids.iter().copied().collect::<FxHashSet<_>>();
        let mut state = self.state.write();
        let Some(entries) = state.save_pending_by_chunk.get_mut(&chunk) else {
            return;
        };

        entries.retain(|entry| !saved_entity_ids.contains(&entry.entity.id()));
        if entries.is_empty() {
            state.save_pending_by_chunk.remove(&chunk);
        }
    }

    /// Returns whether `chunk` has removed runtime entities waiting for a save acknowledgement.
    #[must_use]
    pub fn has_save_pending_for_chunk(&self, chunk: ChunkPos) -> bool {
        self.state
            .read()
            .save_pending_by_chunk
            .get(&chunk)
            .is_some_and(|entries| !entries.is_empty())
    }

    /// Validates that a live entity can move to `new_pos`.
    pub fn validate_move(&self, entity_id: i32, new_pos: DVec3) -> Result<(), EntityMoveError> {
        let state = self.state.read();
        let Some(entry) = state.live_by_id.get(&entity_id) else {
            return Err(EntityMoveError::NotLive { entity_id });
        };

        if entry.ownership == EntityOwnership::ManagerOwned {
            let new_section = SectionPos::from_entity_pos(new_pos);
            let new_chunk = ChunkPos::new(new_section.x(), new_section.z());
            if !Self::can_move_manager_owned_to_chunk(&state, entry, new_chunk) {
                return Err(EntityMoveError::UnloadedDestination {
                    entity_id,
                    chunk: new_chunk,
                });
            }
        }

        Ok(())
    }

    fn missing_move_commit_result(
        entity_id: i32,
        change: &EntitySpatialChange<'_>,
    ) -> Result<Option<EntityMoveUpdate>, EntityMoveError> {
        if change.callback_is_current() {
            Err(EntityMoveError::NotLive { entity_id })
        } else {
            Ok(None)
        }
    }

    /// Commits manager indexes after a live entity position change.
    ///
    /// # Panics
    ///
    /// Panics if `change` is not a position mutation or one coherent spatial
    /// revision identifies different positions in the manager and base.
    pub fn commit_move(
        &self,
        entity_id: i32,
        change: &EntitySpatialChange<'_>,
    ) -> Result<Option<EntityMoveUpdate>, EntityMoveError> {
        let Some((_old_pos, requested_position)) = change.position_change() else {
            panic!("entity move commit received a non-position spatial change");
        };
        let mut state = self.state.write();
        let Some(current) = state.live_by_id.get(&entity_id) else {
            return Self::missing_move_commit_result(entity_id, change);
        };
        if !change.originates_from(current.entity.base()) {
            return Err(EntityMoveError::NotLive { entity_id });
        }
        if current.committed_spatial_revision != change.expected().revision() {
            return Ok(None);
        }
        assert_eq!(
            current.committed_position,
            change.expected().position(),
            "entity manager and base prepared different positions"
        );

        let new_section = SectionPos::from_entity_pos(requested_position);
        let new_chunk = ChunkPos::new(new_section.x(), new_section.z());
        if current.ownership == EntityOwnership::ManagerOwned
            && !Self::can_move_manager_owned_to_chunk(&state, current, new_chunk)
        {
            return Err(EntityMoveError::UnloadedDestination {
                entity_id,
                chunk: new_chunk,
            });
        }

        let old_section = current.section;
        let old_chunk = current.chunk;
        let old_accessible = Self::is_accessible(&state, current);
        let new_accessible = Self::is_accessible_at(&state, current.ownership, new_chunk);
        let old_visibility =
            Self::lifecycle_visibility_for(current, Self::chunk_visibility(&state, old_chunk));
        let new_visibility =
            Self::lifecycle_visibility_for(current, Self::chunk_visibility(&state, new_chunk));
        let old_ticking = old_visibility.is_ticking();
        let new_ticking = new_visibility.is_ticking();
        let entity = current.entity.clone();
        let ownership = current.ownership;
        let EntitySpatialCommitResult::Committed(spatial_update) = change.commit() else {
            return Ok(None);
        };
        assert_eq!(
            spatial_update.position(),
            requested_position,
            "entity move committed a different position than its requested destination"
        );
        Self::refresh_spatial_membership(&mut state, entity_id, spatial_update.bounding_box());
        if let Some(entry) = state.live_by_id.get_mut(&entity_id) {
            entry.committed_position = spatial_update.position();
            entry.committed_spatial_revision = spatial_update.revision();
        }
        if old_section != new_section || old_chunk != new_chunk {
            Self::remove_from_section(&mut state, old_section, entity_id);
            Self::remove_from_chunk(&mut state, old_chunk, entity_id);

            let new_section_insertion_order = Self::take_section_insertion_order(&mut state);
            if let Some(entry) = state.live_by_id.get_mut(&entity_id) {
                entry.section = new_section;
                entry.chunk = new_chunk;
                entry.section_insertion_order = new_section_insertion_order;
            }

            state
                .by_section
                .entry(PackedSectionPos::from(new_section))
                .or_default()
                .insert(entity_id);
            state
                .by_chunk
                .entry(new_chunk)
                .or_default()
                .insert(entity_id);

            if old_accessible && !new_accessible {
                state.accessible_order.remove(entity_id);
            } else if !old_accessible && new_accessible {
                state.accessible_order.insert(entity_id);
            }

            if old_ticking && !new_ticking {
                state.tick_list.remove(entity_id);
            } else if !old_ticking && new_ticking {
                state.tick_list.add(&entity);
            }
        }

        let update = EntityMoveUpdate {
            entity,
            ownership,
            entity_id,
            old_section,
            new_section,
            old_chunk,
            new_chunk,
            old_accessible,
            new_accessible,
            old_ticking,
            new_ticking,
            spatial_update,
        };
        self.enqueue_effect(EntityManagerEffect::SpatialMove(update.clone()));
        Ok(Some(update))
    }

    /// Commits a non-position spatial change with its manager indexes.
    pub(crate) fn commit_spatial_change(
        &self,
        entity_id: i32,
        change: &EntitySpatialChange<'_>,
    ) -> EntitySpatialCommitResult {
        let mut state = self.state.write();
        let Some(current) = state.live_by_id.get(&entity_id) else {
            assert!(
                !change.callback_is_current(),
                "entity {entity_id} current manager callback had no live entry"
            );
            return EntitySpatialCommitResult::Retry;
        };
        if !change.originates_from(current.entity.base()) {
            assert!(
                !change.callback_is_current(),
                "entity {entity_id} current manager callback targeted a different live entity"
            );
            return EntitySpatialCommitResult::Retry;
        }
        if current.committed_spatial_revision != change.expected().revision() {
            return EntitySpatialCommitResult::Retry;
        }
        let committed_position = current.committed_position;
        assert!(
            change.position_change().is_none(),
            "non-position spatial commit received an entity move"
        );
        assert_eq!(
            committed_position,
            change.expected().position(),
            "non-position entity spatial change did not start from the manager position"
        );
        let result = change.commit();
        let EntitySpatialCommitResult::Committed(spatial_update) = result else {
            return result;
        };
        assert_eq!(
            committed_position,
            spatial_update.position(),
            "non-position entity spatial change moved the entity"
        );
        Self::refresh_spatial_membership(&mut state, entity_id, spatial_update.bounding_box());
        if let Some(entry) = state.live_by_id.get_mut(&entity_id) {
            entry.committed_spatial_revision = spatial_update.revision();
        }
        result
    }

    fn can_move_manager_owned_to_chunk(
        state: &ManagerState,
        entry: &EntityEntry,
        new_chunk: ChunkPos,
    ) -> bool {
        state.chunk_visibility.contains_key(&new_chunk)
            || (entry.entity.is_passenger()
                && Self::has_live_loaded_root_vehicle(state, &entry.entity))
    }

    fn has_live_loaded_root_vehicle(state: &ManagerState, entity: &SharedEntity) -> bool {
        let mut visited = FxHashSet::default();
        visited.insert(entity.id());

        let mut passenger = Arc::clone(entity);
        let Some(mut vehicle) = passenger.vehicle() else {
            return false;
        };

        loop {
            assert!(
                visited.insert(vehicle.id()),
                "cyclic passenger relationship involving entity {}",
                entity.id()
            );
            if vehicle.is_removed() || !vehicle.has_passenger(passenger.as_ref()) {
                return false;
            }

            let Some(vehicle_entry) = state.live_by_id.get(&vehicle.id()) else {
                return false;
            };

            let Some(next_vehicle) = vehicle.vehicle() else {
                return match vehicle_entry.ownership {
                    EntityOwnership::External => true,
                    EntityOwnership::ManagerOwned => {
                        state.chunk_visibility.contains_key(&vehicle_entry.chunk)
                    }
                };
            };

            passenger = vehicle;
            vehicle = next_vehicle;
        }
    }

    #[must_use]
    /// Gets a live entity by session network ID.
    pub fn get_by_id(&self, entity_id: i32) -> Option<SharedEntity> {
        self.state
            .read()
            .live_by_id
            .get(&entity_id)
            .map(|entry| entry.entity.clone())
    }

    /// Returns true if this exact entity is live or retained for chunk-unload recovery.
    pub fn contains_live_or_unloading_entity(&self, entity: &SharedEntity) -> bool {
        let state = self.state.read();
        state
            .live_by_id
            .get(&entity.id())
            .is_some_and(|entry| Arc::ptr_eq(&entry.entity, entity))
            || state
                .unloading_by_chunk
                .values()
                .flatten()
                .any(|entry| Arc::ptr_eq(&entry.entity, entity))
    }

    #[must_use]
    /// Gets a live entity by session network ID if it is visible to vanilla gameplay lookups.
    pub fn get_accessible_by_id(&self, entity_id: i32) -> Option<SharedEntity> {
        let state = self.state.read();
        let entry = state.live_by_id.get(&entity_id)?;
        Self::is_accessible(&state, entry).then(|| entry.entity.clone())
    }

    #[must_use]
    /// Gets a live entity by persistent UUID.
    pub fn get_by_uuid(&self, uuid: &Uuid) -> Option<SharedEntity> {
        let state = self.state.read();
        let entity_id = state.live_by_uuid.get(uuid)?;
        state
            .live_by_id
            .get(entity_id)
            .map(|entry| entry.entity.clone())
    }

    #[must_use]
    /// Gets live entities whose bounding boxes intersect `aabb` and match `predicate`.
    pub fn get_entities_in_aabb_matching(
        &self,
        aabb: &WorldAabb,
        mut predicate: impl FnMut(&dyn Entity) -> bool,
    ) -> Vec<SharedEntity> {
        self.entity_query_candidates(aabb)
            .into_iter()
            .filter(|candidate| predicate(candidate.entity.as_ref()))
            .map(|candidate| candidate.entity)
            .collect()
    }

    /// Returns whether any live entity intersects `aabb` and matches `predicate`.
    #[must_use]
    pub fn has_entity_in_aabb_matching(
        &self,
        aabb: &WorldAabb,
        mut predicate: impl FnMut(&dyn Entity) -> bool,
    ) -> bool {
        self.entity_query_candidates(aabb)
            .into_iter()
            .any(|candidate| predicate(candidate.entity.as_ref()))
    }

    /// Gets matching live entity bounding boxes that intersect `aabb`.
    #[must_use]
    pub fn get_entity_bounding_boxes_in_aabb_matching(
        &self,
        aabb: &WorldAabb,
        mut predicate: impl FnMut(&dyn Entity) -> bool,
    ) -> Vec<WorldAabb> {
        self.entity_query_candidates(aabb)
            .into_iter()
            .filter_map(|candidate| {
                predicate(candidate.entity.as_ref()).then_some(candidate.bounding_box)
            })
            .collect()
    }

    #[must_use]
    /// Gets the nearest live entity whose bounding box intersects `aabb` and matches `predicate`.
    pub fn nearest_entity_in_aabb_matching(
        &self,
        aabb: &WorldAabb,
        origin: DVec3,
        mut predicate: impl FnMut(&dyn Entity) -> bool,
    ) -> Option<SharedEntity> {
        self.get_entities_in_aabb(aabb)
            .into_iter()
            .filter(|entity| predicate(entity.as_ref()))
            .min_by(|first, second| {
                first
                    .position()
                    .distance_squared(origin)
                    .total_cmp(&second.position().distance_squared(origin))
            })
    }

    #[must_use]
    /// Gets live entities whose bounding boxes intersect `aabb`.
    pub fn get_entities_in_aabb(&self, aabb: &WorldAabb) -> Vec<SharedEntity> {
        self.entity_query_candidates(aabb)
            .into_iter()
            .map(|candidate| candidate.entity)
            .collect()
    }

    /// Gets all live entities visible to vanilla gameplay lookups.
    #[must_use]
    pub fn get_accessible_entities(&self) -> Vec<SharedEntity> {
        let state = self.state.read();
        state
            .accessible_order
            .iter()
            .filter_map(|entity_id| state.live_by_id.get(entity_id))
            .filter(|entry| Self::is_accessible(&state, entry))
            .map(|entry| Arc::clone(&entry.entity))
            .collect()
    }

    /// Snapshots intersecting entities before extensible predicates run.
    ///
    /// Fine-grained candidates are gathered under the manager read lock, then
    /// sorted outside it into Vanilla section and section-insertion order.
    fn entity_query_candidates(&self, aabb: &WorldAabb) -> SmallVec<[EntityQueryCandidate; 16]> {
        let (mut candidates, needs_sort) = {
            let state = self.state.read();
            let mut candidates = SmallVec::<[EntityQueryCandidate; 16]>::new();
            let (minimum, maximum) = Self::entity_query_section_bounds(aabb);
            let used_spatial_index =
                state
                    .spatial_index
                    .try_visit_candidate_ids(*aabb, |entity_id| {
                        let Some(entry) = state.live_by_id.get(&entity_id) else {
                            return;
                        };
                        if !Self::section_is_within_query_bounds(entry.section, minimum, maximum)
                            || !Self::is_accessible(&state, entry)
                        {
                            return;
                        }

                        let bounding_box = entry.committed_bounding_box;
                        if bounding_box.intersects(*aabb) {
                            candidates.push(Self::entity_query_candidate(entry, bounding_box));
                        }
                    });

            if !used_spatial_index {
                Self::visit_intersecting_entity_query_entries_by_section(
                    &state,
                    aabb,
                    |entry, bounding_box| {
                        candidates.push(Self::entity_query_candidate(entry, bounding_box));
                    },
                );
            }
            (candidates, used_spatial_index)
        };

        if needs_sort {
            candidates.sort_unstable_by_key(|candidate| candidate.order);
        }
        candidates
    }

    fn entity_query_candidate(
        entry: &EntityEntry,
        bounding_box: WorldAabb,
    ) -> EntityQueryCandidate {
        EntityQueryCandidate {
            entity: Arc::clone(&entry.entity),
            bounding_box,
            order: (
                entry.section.x(),
                PackedSectionPos::from(entry.section),
                entry.section_insertion_order,
            ),
        }
    }

    fn visit_intersecting_entity_query_entries_by_section(
        state: &ManagerState,
        aabb: &WorldAabb,
        mut visitor: impl FnMut(&EntityEntry, WorldAabb),
    ) {
        let (minimum, maximum) = Self::entity_query_section_bounds(aabb);
        for x in minimum.x()..=maximum.x() {
            let first = PackedSectionPos::from(SectionPos::new(x, 0, 0));
            let last = PackedSectionPos::from(SectionPos::new(x, -1, -1));
            for (packed, entity_ids) in state.by_section.range(first..=last) {
                let section = packed.to_section_pos();
                if section.y() < minimum.y()
                    || section.y() > maximum.y()
                    || section.z() < minimum.z()
                    || section.z() > maximum.z()
                {
                    continue;
                }

                let manager_owned_accessible =
                    Self::chunk_visibility(state, ChunkPos::new(section.x(), section.z()))
                        .is_accessible();
                for entity_id in entity_ids.iter() {
                    let Some(entry) = state.live_by_id.get(entity_id) else {
                        continue;
                    };
                    if entry.ownership != EntityOwnership::External && !manager_owned_accessible {
                        continue;
                    }

                    let bounding_box = entry.committed_bounding_box;
                    if bounding_box.intersects(*aabb) {
                        visitor(entry, bounding_box);
                    }
                }
            }
        }
    }

    const fn section_is_within_query_bounds(
        section: SectionPos,
        minimum: SectionPos,
        maximum: SectionPos,
    ) -> bool {
        section.x() >= minimum.x()
            && section.x() <= maximum.x()
            && section.y() >= minimum.y()
            && section.y() <= maximum.y()
            && section.z() >= minimum.z()
            && section.z() <= maximum.z()
    }

    fn entity_ids_in_chunk_order(state: &ManagerState, chunk: ChunkPos) -> Vec<i32> {
        let first = PackedSectionPos::from(SectionPos::new(chunk.0.x, 0, chunk.0.y));
        let last = PackedSectionPos::from(SectionPos::new(chunk.0.x, -1, chunk.0.y));
        state
            .by_section
            .range(first..=last)
            .flat_map(|(_, entity_ids)| entity_ids.iter().copied())
            .collect()
    }

    fn entity_query_section_bounds(aabb: &WorldAabb) -> (SectionPos, SectionPos) {
        let min_section = SectionPos::from_entity_pos(DVec3::new(
            aabb.min_x() - 2.0,
            aabb.min_y() - 4.0,
            aabb.min_z() - 2.0,
        ));
        let max_section = SectionPos::from_entity_pos(DVec3::new(
            aabb.max_x() + 2.0,
            aabb.max_y(),
            aabb.max_z() + 2.0,
        ));
        (min_section, max_section)
    }

    /// Reports saveable entities whose chunks were not part of a chunk save pass.
    #[must_use]
    pub fn saveable_entities_outside_chunks(
        &self,
        saved_chunks: &[ChunkPos],
    ) -> Vec<UnsavedEntityReport> {
        let saved_chunks = saved_chunks.iter().copied().collect::<FxHashSet<_>>();
        let state = self.state.read();
        let mut seen = FxHashSet::default();
        let mut reports = Vec::new();

        for entry in state.live_by_id.values() {
            Self::push_unsaved_entity_report(&saved_chunks, &mut seen, &mut reports, entry);
        }

        for entries in state.unloading_by_chunk.values() {
            for entry in entries {
                Self::push_unsaved_entity_report(&saved_chunks, &mut seen, &mut reports, entry);
            }
        }

        for entries in state.save_pending_by_chunk.values() {
            for entry in entries {
                Self::push_unsaved_entity_report(&saved_chunks, &mut seen, &mut reports, entry);
            }
        }

        reports.sort_by_key(|report| (report.chunk.0.x, report.chunk.0.y, report.entity_id));
        reports
    }

    #[must_use]
    /// Gets entities that should be serialized for `chunk`.
    pub fn get_saveable_entities_for_chunk(&self, chunk: ChunkPos) -> Vec<SharedEntity> {
        let state = self.state.read();
        let mut result = Vec::new();
        let mut seen_ids = FxHashSet::default();
        let mut seen_uuids = FxHashSet::default();

        for entity_id in Self::entity_ids_in_chunk_order(&state, chunk) {
            let Some(entry) = state.live_by_id.get(&entity_id) else {
                continue;
            };
            Self::push_saveable_entity(&mut result, &mut seen_ids, &mut seen_uuids, entry);
        }

        if let Some(entries) = state.unloading_by_chunk.get(&chunk) {
            for entry in entries {
                Self::push_saveable_entity(&mut result, &mut seen_ids, &mut seen_uuids, entry);
            }
        }

        if let Some(entries) = state.save_pending_by_chunk.get(&chunk) {
            for entry in entries {
                Self::push_saveable_entity(&mut result, &mut seen_ids, &mut seen_uuids, entry);
            }
        }

        result
    }

    #[must_use]
    /// Gets live entities currently indexed in `chunk`.
    pub fn live_entities_in_chunk(&self, chunk: ChunkPos) -> Vec<SharedEntity> {
        let state = self.state.read();
        Self::entity_ids_in_chunk_order(&state, chunk)
            .into_iter()
            .filter_map(|entity_id| state.live_by_id.get(&entity_id))
            .map(|entry| entry.entity.clone())
            .collect()
    }

    #[must_use]
    /// Returns the number of live indexed entities.
    pub fn count(&self) -> usize {
        self.state.read().live_by_id.len()
    }

    /// Ticks live entities currently in the ticking visibility set.
    pub fn tick_entities(&self, _tick_count: i32, runs_normally: bool) -> FxHashSet<ChunkPos> {
        let mut dirty_chunks = FxHashSet::default();
        let mut ticked_entities = FxHashSet::default();
        let tick_candidates = self.ticking_entities_snapshot();
        for entity in tick_candidates {
            if !self.can_tick_entity_now(entity.id()) {
                continue;
            }

            if entity.is_removed() {
                continue;
            }

            if Self::is_entity_frozen_by_tick_rate(entity.as_ref(), runs_normally) {
                continue;
            }

            let entity_chunk = self.live_manager_owned_entity_chunk(entity.id());
            entity.check_despawn();
            if entity.is_removed() {
                if let Some(chunk) = entity_chunk {
                    dirty_chunks.insert(chunk);
                }
                continue;
            }

            if Self::is_valid_passenger_or_stop_riding(&entity) {
                continue;
            }

            if !ticked_entities.insert(entity.id()) {
                continue;
            }

            self.tick_non_passenger(&entity, &mut ticked_entities, &mut dirty_chunks);
        }
        dirty_chunks
    }

    fn ticking_entities_snapshot(&self) -> Vec<SharedEntity> {
        self.state.read().tick_list.snapshot()
    }

    fn live_manager_owned_entity_chunk(&self, entity_id: i32) -> Option<ChunkPos> {
        self.state
            .read()
            .live_by_id
            .get(&entity_id)
            .filter(|entry| entry.ownership == EntityOwnership::ManagerOwned)
            .map(|entry| entry.chunk)
    }

    fn chunk_visibility(state: &ManagerState, chunk: ChunkPos) -> EntityVisibility {
        state
            .chunk_visibility
            .get(&chunk)
            .copied()
            .unwrap_or(EntityVisibility::Hidden)
    }

    fn effective_visibility(
        entry: &EntityEntry,
        chunk_visibility: EntityVisibility,
    ) -> EntityVisibility {
        if entry.entity.is_always_ticking() {
            return EntityVisibility::Ticking;
        }
        if entry.ownership == EntityOwnership::External {
            return EntityVisibility::Tracked;
        }
        chunk_visibility
    }

    fn lifecycle_visibility_for(
        entry: &EntityEntry,
        chunk_visibility: EntityVisibility,
    ) -> EntityVisibility {
        Self::effective_visibility(entry, chunk_visibility)
    }

    fn apply_entity_lifecycle_after_insert(
        state: &mut ManagerState,
        entity_id: i32,
    ) -> EntityLifecycleChanges {
        let Some(entry) = state.live_by_id.get(&entity_id) else {
            return EntityLifecycleChanges::default();
        };
        let visibility =
            Self::lifecycle_visibility_for(entry, Self::chunk_visibility(state, entry.chunk));
        let entity = entry.entity.clone();
        let should_tick = visibility.is_ticking();

        let mut lifecycle = EntityLifecycleChanges::default();
        if visibility.is_accessible() {
            lifecycle.tracking_started.push(entity.clone());
        }
        if should_tick && state.tick_list.add(&entity) {
            lifecycle.ticking_started.push(entity);
        }
        lifecycle
    }

    fn apply_chunk_visibility_change(
        state: &mut ManagerState,
        chunk: ChunkPos,
        previous: EntityVisibility,
        new: EntityVisibility,
    ) -> EntityLifecycleChanges {
        let entity_ids = Self::entity_ids_in_chunk_order(state, chunk);
        let mut lifecycle = EntityLifecycleChanges::default();

        for entity_id in entity_ids {
            let Some(entry) = state.live_by_id.get(&entity_id) else {
                continue;
            };
            if entry.ownership != EntityOwnership::ManagerOwned {
                continue;
            }

            let old_visibility = Self::lifecycle_visibility_for(entry, previous);
            let new_visibility = Self::lifecycle_visibility_for(entry, new);
            if old_visibility == new_visibility {
                continue;
            }

            let entity = entry.entity.clone();
            if old_visibility.is_ticking()
                && !new_visibility.is_ticking()
                && state.tick_list.remove(entity_id).is_some()
            {
                lifecycle.ticking_stopped.push(entity.clone());
            }

            if old_visibility.is_accessible() && !new_visibility.is_accessible() {
                state.accessible_order.remove(entity_id);
                lifecycle.tracking_stopped.push(entity.clone());
            } else if !old_visibility.is_accessible() && new_visibility.is_accessible() {
                state.accessible_order.insert(entity_id);
                lifecycle.tracking_started.push(entity.clone());
            }

            if !old_visibility.is_ticking()
                && new_visibility.is_ticking()
                && state.tick_list.add(&entity)
            {
                lifecycle.ticking_started.push(entity);
            }
        }

        lifecycle
    }

    fn is_entity_frozen_by_tick_rate(entity: &dyn Entity, runs_normally: bool) -> bool {
        !runs_normally
            && entity.entity_type() != &vanilla_entities::PLAYER
            && entity.count_player_passengers() == 0
    }

    fn has_pending_world_change_in_vehicle_chain(entity: &SharedEntity) -> bool {
        if entity.is_world_change_pending() {
            return true;
        }

        let mut visited = FxHashSet::default();
        visited.insert(entity.id());
        let mut vehicle = entity.vehicle();
        while let Some(current) = vehicle {
            assert!(
                visited.insert(current.id()),
                "cyclic passenger relationship involving entity {}",
                entity.id()
            );
            if current.is_world_change_pending() {
                return true;
            }
            vehicle = current.vehicle();
        }
        false
    }

    fn is_accessible(state: &ManagerState, entry: &EntityEntry) -> bool {
        Self::is_accessible_at(state, entry.ownership, entry.chunk)
    }

    fn is_accessible_at(state: &ManagerState, ownership: EntityOwnership, chunk: ChunkPos) -> bool {
        ownership == EntityOwnership::External
            || Self::chunk_visibility(state, chunk).is_accessible()
    }

    fn is_valid_passenger_or_stop_riding(entity: &SharedEntity) -> bool {
        let Some(vehicle) = entity.vehicle() else {
            return false;
        };

        if !vehicle.is_removed() && vehicle.has_passenger(entity.as_ref()) {
            Self::assert_acyclic_vehicle_chain(entity);
            return true;
        }

        entity.stop_riding();
        false
    }

    fn assert_acyclic_vehicle_chain(entity: &SharedEntity) {
        let mut visited = FxHashSet::default();
        visited.insert(entity.id());

        let mut vehicle = entity.vehicle();
        while let Some(current) = vehicle {
            assert!(
                visited.insert(current.id()),
                "cyclic passenger relationship involving entity {}",
                entity.id()
            );
            vehicle = current.vehicle();
        }
    }

    fn tick_non_passenger(
        &self,
        entity: &SharedEntity,
        ticked_entities: &mut FxHashSet<i32>,
        dirty_chunks: &mut FxHashSet<ChunkPos>,
    ) {
        snapshot_old_pos_and_rot_for_tick(entity.as_ref());
        entity.advance_tick_count();
        entity.tick();
        self.mark_dirty_after_tick(entity, dirty_chunks);
        self.tick_vehicle_passengers_with_ticked(entity.as_ref(), ticked_entities, dirty_chunks);
    }

    fn tick_vehicle_passengers_with_ticked(
        &self,
        vehicle: &dyn Entity,
        ticked_entities: &mut FxHashSet<i32>,
        dirty_chunks: &mut FxHashSet<ChunkPos>,
    ) {
        let mut post_tick = |entity: &SharedEntity| {
            self.mark_dirty_after_tick(entity, dirty_chunks);
        };
        tick_vehicle_passengers_with_ticked_if(
            vehicle,
            ticked_entities,
            &mut post_tick,
            &mut |entity| self.can_tick_entity_now(entity.id()),
        );
    }

    fn mark_dirty_after_tick(&self, entity: &SharedEntity, dirty_chunks: &mut FxHashSet<ChunkPos>) {
        if self.live_manager_owned_entity_chunk(entity.id()).is_some() {
            dirty_chunks.insert(ChunkPos::from_entity_pos(entity.position()));
        }
    }

    fn can_tick_entity_now(&self, entity_id: i32) -> bool {
        let state = self.state.read();
        let Some(entry) = state.live_by_id.get(&entity_id) else {
            return false;
        };
        if Self::has_pending_world_change_in_vehicle_chain(&entry.entity) {
            return false;
        }

        match entry.ownership {
            EntityOwnership::External => {
                entry.entity.entity_type() == &vanilla_entities::PLAYER
                    || state.tick_list.contains(entity_id)
            }
            EntityOwnership::ManagerOwned => state.tick_list.contains(entity_id),
        }
    }

    fn insert_live_entry(state: &mut ManagerState, mut entry: EntityEntry) {
        let entity_id = entry.entity.id();
        assert!(
            entry.retained_callback.is_none(),
            "live entity {entity_id} retained an inactive callback binding"
        );
        let is_accessible = Self::is_accessible_at(state, entry.ownership, entry.chunk);
        assert!(
            !state.live_by_id.contains_key(&entity_id),
            "entity id {entity_id} is already registered in the world entity manager"
        );
        assert!(
            state.live_by_uuid.insert(entry.uuid, entity_id).is_none(),
            "entity uuid {} is already registered in the world entity manager",
            entry.uuid
        );
        entry.section_insertion_order = Self::take_section_insertion_order(state);
        entry.spatial_membership =
            EntitySpatialMembership::for_bounding_box(entry.committed_bounding_box);
        state
            .spatial_index
            .insert(entity_id, &entry.spatial_membership);
        state
            .by_section
            .entry(PackedSectionPos::from(entry.section))
            .or_default()
            .insert(entity_id);
        state
            .by_chunk
            .entry(entry.chunk)
            .or_default()
            .insert(entity_id);
        state.live_by_id.insert(entity_id, entry);
        if is_accessible {
            state.accessible_order.insert(entity_id);
        }
    }

    fn contains_uuid(state: &ManagerState, uuid: Uuid) -> bool {
        state.live_by_uuid.contains_key(&uuid)
            || state
                .unloading_by_chunk
                .values()
                .flatten()
                .any(|entry| entry.uuid == uuid)
            || state
                .save_pending_by_chunk
                .values()
                .flatten()
                .any(|entry| entry.uuid == uuid)
    }

    fn contains_id(state: &ManagerState, entity_id: i32) -> bool {
        state.live_by_id.contains_key(&entity_id)
            || state
                .unloading_by_chunk
                .values()
                .flatten()
                .any(|entry| entry.entity.id() == entity_id)
            || state
                .save_pending_by_chunk
                .values()
                .flatten()
                .any(|entry| entry.entity.id() == entity_id)
    }

    fn push_saveable_entity(
        result: &mut Vec<SharedEntity>,
        seen_ids: &mut FxHashSet<i32>,
        seen_uuids: &mut FxHashSet<Uuid>,
        entry: &EntityEntry,
    ) {
        if !entry.should_save() || !seen_ids.insert(entry.entity.id()) {
            return;
        }
        assert!(
            seen_uuids.insert(entry.uuid),
            "duplicate saveable entity uuid {} in world entity manager",
            entry.uuid
        );
        result.push(entry.entity.clone());
    }

    fn push_unsaved_entity_report(
        saved_chunks: &FxHashSet<ChunkPos>,
        seen: &mut FxHashSet<i32>,
        reports: &mut Vec<UnsavedEntityReport>,
        entry: &EntityEntry,
    ) {
        if saved_chunks.contains(&entry.chunk)
            || !entry.should_save()
            || !seen.insert(entry.entity.id())
        {
            return;
        }

        reports.push(UnsavedEntityReport {
            entity_id: entry.entity.id(),
            uuid: entry.uuid,
            chunk: entry.chunk,
        });
    }

    fn remove_live_entry(state: &mut ManagerState, entity_id: i32) -> Option<EntityEntry> {
        let entry = state.live_by_id.remove(&entity_id)?;
        state
            .spatial_index
            .remove(entity_id, &entry.spatial_membership);
        state.tick_list.remove(entity_id);
        state.live_by_uuid.remove(&entry.uuid);
        state.accessible_order.remove(entity_id);
        Self::remove_from_section(state, entry.section, entity_id);
        Self::remove_from_chunk(state, entry.chunk, entity_id);
        Some(entry)
    }

    fn take_section_insertion_order(state: &mut ManagerState) -> u64 {
        assert!(
            state.next_section_insertion_order != u64::MAX,
            "world entity manager exhausted section insertion order"
        );
        let order = state.next_section_insertion_order;
        state.next_section_insertion_order += 1;
        order
    }

    fn refresh_spatial_membership(
        state: &mut ManagerState,
        entity_id: i32,
        bounding_box: WorldAabb,
    ) {
        let new_membership = EntitySpatialMembership::for_bounding_box(bounding_box);
        let old_membership = {
            let Some(entry) = state.live_by_id.get_mut(&entity_id) else {
                return;
            };
            entry.committed_bounding_box = bounding_box;
            if entry.spatial_membership == new_membership {
                return;
            }
            mem::take(&mut entry.spatial_membership)
        };

        state.spatial_index.remove(entity_id, &old_membership);
        state.spatial_index.insert(entity_id, &new_membership);
        if let Some(entry) = state.live_by_id.get_mut(&entity_id) {
            entry.spatial_membership = new_membership;
        }
    }

    fn remove_from_section(state: &mut ManagerState, section: SectionPos, entity_id: i32) {
        let packed = PackedSectionPos::from(section);
        let remove_section = if let Some(entity_ids) = state.by_section.get_mut(&packed) {
            entity_ids.remove(entity_id);
            entity_ids.is_empty()
        } else {
            false
        };
        if remove_section {
            state.by_section.remove(&packed);
        }
    }

    fn remove_from_chunk(state: &mut ManagerState, chunk: ChunkPos, entity_id: i32) {
        let remove_chunk = if let Some(entity_ids) = state.by_chunk.get_mut(&chunk) {
            entity_ids.remove(&entity_id);
            entity_ids.is_empty()
        } else {
            false
        };
        if remove_chunk {
            state.by_chunk.remove(&chunk);
        }
    }
}

impl Default for WorldEntityManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
