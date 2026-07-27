//! Entity lifecycle callbacks for movement and removal tracking.

use std::sync::{Arc, Weak};

use super::{
    EntityMoveError, EntitySpatialChange, EntitySpatialCommitResult, SharedEntity, WeakEntity,
};
use crate::world::World;

/// Reasons an entity can be removed from the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemovalReason {
    /// Entity was killed/destroyed.
    Killed,
    /// Entity was discarded (e.g., too far from players).
    Discarded,
    /// Entity unloaded with chunk.
    UnloadedToChunk,
    /// Entity moved to another loaded world.
    ChangedWorld,
    /// Entity is persisted inside a player `RootVehicle` payload.
    StoredWithPlayer,
}

impl RemovalReason {
    /// Returns true if entity data should be destroyed (not saved).
    #[must_use]
    pub const fn should_destroy(self) -> bool {
        matches!(self, Self::Killed | Self::Discarded)
    }

    /// Returns true if the entity should be saved when removed.
    ///
    /// In vanilla, only `UnloadedToChunk` saves - the entity persists in chunk storage.
    /// `ChangedWorld` and `StoredWithPlayer` do not save because the entity
    /// is retained by another owner instead of current-world entity storage.
    #[must_use]
    pub const fn should_save(self) -> bool {
        matches!(self, Self::UnloadedToChunk)
    }
}

/// Callback interface for entity lifecycle events.
///
/// Mirrors vanilla's `EntityInLevelCallback`.
pub trait EntityLevelCallback: Send + Sync {
    /// Returns whether direct local position writes may bypass lifecycle callbacks.
    fn allows_local_position_update(&self) -> bool {
        false
    }

    /// Validates and atomically commits an ordinary position change.
    fn commit_move(
        &self,
        change: &EntitySpatialChange<'_>,
    ) -> Result<EntitySpatialCommitResult, EntityMoveError>;

    /// Atomically commits a bounding-box, dimensions, or respawn spatial change.
    fn commit_spatial_change(&self, change: &EntitySpatialChange<'_>) -> EntitySpatialCommitResult;

    /// Called when entity is removed from the world.
    fn on_remove(&self, reason: RemovalReason);
}

struct EntityCallbackIdentity;

/// Unforgeable identity for one manager callback installation.
#[derive(Clone)]
pub(crate) struct EntityCallbackToken(Arc<EntityCallbackIdentity>);

impl EntityCallbackToken {
    fn new() -> Self {
        Self(Arc::new(EntityCallbackIdentity))
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// A level callback paired with the identity authenticated by the entity manager.
pub(crate) struct BoundEntityCallback {
    callback: Arc<dyn EntityLevelCallback>,
    token: EntityCallbackToken,
}

impl BoundEntityCallback {
    pub(crate) fn new(
        create: impl FnOnce(EntityCallbackToken) -> Arc<dyn EntityLevelCallback>,
    ) -> Self {
        let token = EntityCallbackToken::new();
        let callback = create(token.clone());
        Self { callback, token }
    }

    #[cfg(test)]
    pub(crate) fn callback_for_test(&self) -> Arc<dyn EntityLevelCallback> {
        Arc::clone(&self.callback)
    }

    pub(crate) fn into_parts(self) -> (Arc<dyn EntityLevelCallback>, EntityCallbackToken) {
        (self.callback, self.token)
    }
}

/// Null callback for entities not yet in the world.
pub struct NullEntityCallback;

impl EntityLevelCallback for NullEntityCallback {
    fn allows_local_position_update(&self) -> bool {
        true
    }

    fn commit_move(
        &self,
        change: &EntitySpatialChange<'_>,
    ) -> Result<EntitySpatialCommitResult, EntityMoveError> {
        Ok(change.commit())
    }

    fn commit_spatial_change(&self, change: &EntitySpatialChange<'_>) -> EntitySpatialCommitResult {
        change.commit()
    }

    fn on_remove(&self, _reason: RemovalReason) {}
}

/// Callback for entities retained outside live world membership.
pub struct InactiveEntityCallback {
    entity_id: i32,
}

impl InactiveEntityCallback {
    /// Creates an inactive callback for a retained non-live entity.
    #[must_use]
    pub const fn new(entity_id: i32) -> Self {
        Self { entity_id }
    }
}

impl EntityLevelCallback for InactiveEntityCallback {
    fn commit_move(
        &self,
        _change: &EntitySpatialChange<'_>,
    ) -> Result<EntitySpatialCommitResult, EntityMoveError> {
        Err(EntityMoveError::Inactive {
            entity_id: self.entity_id,
        })
    }

    fn commit_spatial_change(&self, change: &EntitySpatialChange<'_>) -> EntitySpatialCommitResult {
        change.commit()
    }

    fn on_remove(&self, _reason: RemovalReason) {}
}

/// Callback for players.
///
/// Players are owned by `World.players`, but the world entity manager still
/// indexes their live position for lookup and tracking updates.
pub struct PlayerEntityCallback {
    entity_id: i32,
    world: Weak<World>,
}

impl PlayerEntityCallback {
    /// Creates a new callback for a player.
    #[must_use]
    pub const fn new(entity_id: i32, world: Weak<World>) -> Self {
        Self { entity_id, world }
    }

    pub(crate) fn bind(entity_id: i32, world: Weak<World>) -> BoundEntityCallback {
        BoundEntityCallback::new(|_| Arc::new(Self::new(entity_id, world)))
    }
}

impl EntityLevelCallback for PlayerEntityCallback {
    fn commit_move(
        &self,
        change: &EntitySpatialChange<'_>,
    ) -> Result<EntitySpatialCommitResult, EntityMoveError> {
        let Some(world) = self.world.upgrade() else {
            return Err(EntityMoveError::NotLive {
                entity_id: self.entity_id,
            });
        };
        let Some((old_pos, new_pos)) = change.position_change() else {
            panic!("player move callback received a non-position spatial change");
        };

        let Some(update) = world
            .entity_manager()
            .commit_move(self.entity_id, change)
            .inspect_err(|error| {
                log::warn!(
                    "Failed to commit player entity move from {old_pos:?} to {new_pos:?}: {error}"
                );
            })?
        else {
            return Ok(EntitySpatialCommitResult::Retry);
        };

        world.drain_entity_manager_effects();

        Ok(EntitySpatialCommitResult::Committed(
            update.spatial_update(),
        ))
    }

    fn commit_spatial_change(&self, change: &EntitySpatialChange<'_>) -> EntitySpatialCommitResult {
        let Some(world) = self.world.upgrade() else {
            return change.commit();
        };
        world
            .entity_manager()
            .commit_spatial_change(self.entity_id, change)
    }

    fn on_remove(&self, _reason: RemovalReason) {
        // Player removal is handled by World::remove_player, not through this callback
    }
}

/// Callback attached to each entity for tracking chunk/section movement.
///
/// Mirrors vanilla's `PersistentEntitySectionManager.Callback`.
pub struct EntityChunkCallback {
    entity_id: i32,
    entity: WeakEntity,
    callback_token: EntityCallbackToken,
    world: Weak<World>,
}

impl EntityChunkCallback {
    pub(crate) fn bind(entity: &SharedEntity, world: Weak<World>) -> BoundEntityCallback {
        BoundEntityCallback::new(|callback_token| {
            Arc::new(Self {
                entity_id: entity.id(),
                entity: Arc::downgrade(entity),
                callback_token,
                world,
            })
        })
    }
}

impl EntityLevelCallback for EntityChunkCallback {
    fn commit_move(
        &self,
        change: &EntitySpatialChange<'_>,
    ) -> Result<EntitySpatialCommitResult, EntityMoveError> {
        let Some(world) = self.world.upgrade() else {
            return Err(EntityMoveError::NotLive {
                entity_id: self.entity_id,
            });
        };
        let Some((old_pos, new_pos)) = change.position_change() else {
            panic!("entity move callback received a non-position spatial change");
        };

        let Some(update) = world
            .entity_manager()
            .commit_move(self.entity_id, change)
            .inspect_err(|error| {
                log::warn!("Failed to commit entity move from {old_pos:?} to {new_pos:?}: {error}");
            })?
        else {
            return Ok(EntitySpatialCommitResult::Retry);
        };

        world.drain_entity_manager_effects();

        Ok(EntitySpatialCommitResult::Committed(
            update.spatial_update(),
        ))
    }

    fn commit_spatial_change(&self, change: &EntitySpatialChange<'_>) -> EntitySpatialCommitResult {
        let Some(world) = self.world.upgrade() else {
            return change.commit();
        };
        world
            .entity_manager()
            .commit_spatial_change(self.entity_id, change)
    }

    fn on_remove(&self, reason: RemovalReason) {
        let Some(world) = self.world.upgrade() else {
            return;
        };
        let Some(entity) = self.entity.upgrade() else {
            return;
        };

        world.entity_manager().remove_live_entity_if_bound(
            entity.as_ref(),
            &self.callback_token,
            reason,
        );
        world.drain_entity_manager_effects();
    }
}
