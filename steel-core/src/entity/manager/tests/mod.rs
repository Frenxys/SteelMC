use std::sync::{Arc, Barrier, Weak};

use steel_registry::entity_type::EntityTypeRef;
use steel_registry::vanilla_entities;
use steel_registry::{entity_data::EntityPose, entity_type::EntityDimensions};
use steel_utils::locks::SyncMutex;
use uuid::Uuid;

use crate::entity::{Entity, EntityBase, EntityLevelCallback, WeakEntity};

use super::*;

struct ManagerSpatialCallback {
    entity_id: i32,
    entity: WeakEntity,
    callback_token: EntityCallbackToken,
    manager: Weak<WorldEntityManager>,
    commit_entered: Option<Arc<Barrier>>,
}

impl EntityLevelCallback for ManagerSpatialCallback {
    fn commit_move(
        &self,
        change: &EntitySpatialChange<'_>,
    ) -> Result<EntitySpatialCommitResult, EntityMoveError> {
        if let Some(commit_entered) = &self.commit_entered {
            commit_entered.wait();
        }
        let Some(manager) = self.manager.upgrade() else {
            return Err(EntityMoveError::NotLive {
                entity_id: self.entity_id,
            });
        };
        let Some(update) = manager.commit_move(self.entity_id, change)? else {
            return Ok(EntitySpatialCommitResult::Retry);
        };
        Ok(EntitySpatialCommitResult::Committed(
            update.spatial_update(),
        ))
    }

    fn commit_spatial_change(&self, change: &EntitySpatialChange<'_>) -> EntitySpatialCommitResult {
        if let Some(commit_entered) = &self.commit_entered {
            commit_entered.wait();
        }
        let Some(manager) = self.manager.upgrade() else {
            return change.commit();
        };
        manager.commit_spatial_change(self.entity_id, change)
    }

    fn on_remove(&self, reason: RemovalReason) {
        let Some(manager) = self.manager.upgrade() else {
            return;
        };
        let Some(entity) = self.entity.upgrade() else {
            return;
        };
        manager.remove_live_entity_if_bound(entity.as_ref(), &self.callback_token, reason);
    }
}

fn manager_spatial_callback(
    manager: &Arc<WorldEntityManager>,
    entity: &SharedEntity,
) -> BoundEntityCallback {
    manager_spatial_callback_with_commit_gate(manager, entity, None)
}

fn manager_spatial_callback_with_commit_gate(
    manager: &Arc<WorldEntityManager>,
    entity: &SharedEntity,
    commit_entered: Option<Arc<Barrier>>,
) -> BoundEntityCallback {
    BoundEntityCallback::new(|callback_token| {
        Arc::new(ManagerSpatialCallback {
            entity_id: entity.id(),
            entity: Arc::downgrade(entity),
            callback_token,
            manager: Arc::downgrade(manager),
            commit_entered,
        })
    })
}

fn add_live_entity_with_spatial_callback(
    manager: &Arc<WorldEntityManager>,
    entity: SharedEntity,
    ownership: EntityOwnership,
) -> Result<EntityLifecycleChanges, AddEntityError> {
    let callback = manager_spatial_callback(manager, &entity);
    manager.add_live_entity_with_callback(entity, ownership, callback)
}

struct ManagerTestEntity {
    base: EntityBase,
    entity_type: EntityTypeRef,
    always_ticking: bool,
    registration_check_gate: Option<Arc<RegistrationCheckGate>>,
}

struct RegistrationCheckGate {
    check_count: SyncMutex<usize>,
    callback_installed: Barrier,
    release_registration: Barrier,
}

impl RegistrationCheckGate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            check_count: SyncMutex::new(0),
            callback_installed: Barrier::new(2),
            release_registration: Barrier::new(2),
        })
    }

    fn on_removed_check(&self) {
        let should_block = {
            let mut check_count = self.check_count.lock();
            *check_count += 1;
            *check_count == 2
        };
        if should_block {
            self.callback_installed.wait();
            self.release_registration.wait();
        }
    }
}

impl ManagerTestEntity {
    fn shared(id: i32, uuid: Uuid, position: DVec3) -> SharedEntity {
        Self::shared_with_type(id, uuid, position, &vanilla_entities::ITEM)
    }

    fn shared_with_type(
        id: i32,
        uuid: Uuid,
        position: DVec3,
        entity_type: EntityTypeRef,
    ) -> SharedEntity {
        Arc::new(Self {
            base: EntityBase::with_uuid(id, uuid, position, entity_type.dimensions, Weak::new()),
            entity_type,
            always_ticking: false,
            registration_check_gate: None,
        })
    }

    fn shared_with_registration_check_gate(
        id: i32,
        uuid: Uuid,
        position: DVec3,
        registration_check_gate: Arc<RegistrationCheckGate>,
    ) -> SharedEntity {
        Arc::new(Self {
            base: EntityBase::with_uuid(
                id,
                uuid,
                position,
                vanilla_entities::ITEM.dimensions,
                Weak::new(),
            ),
            entity_type: &vanilla_entities::ITEM,
            always_ticking: false,
            registration_check_gate: Some(registration_check_gate),
        })
    }

    fn shared_always_ticking(id: i32, uuid: Uuid, position: DVec3) -> SharedEntity {
        Arc::new(Self {
            base: EntityBase::with_uuid(
                id,
                uuid,
                position,
                vanilla_entities::ITEM.dimensions,
                Weak::new(),
            ),
            entity_type: &vanilla_entities::ITEM,
            always_ticking: true,
            registration_check_gate: None,
        })
    }
}

struct MovingTickTestEntity {
    base: EntityBase,
    tick_position: DVec3,
    tick_rotation: (f32, f32),
}

impl MovingTickTestEntity {
    fn shared(
        id: i32,
        uuid: Uuid,
        position: DVec3,
        tick_position: DVec3,
        tick_rotation: (f32, f32),
    ) -> SharedEntity {
        Arc::new(Self {
            base: EntityBase::with_uuid(
                id,
                uuid,
                position,
                vanilla_entities::ITEM.dimensions,
                Weak::new(),
            ),
            tick_position,
            tick_rotation,
        })
    }
}

crate::entity::impl_test_downcast_type!(MovingTickTestEntity);

impl Entity for MovingTickTestEntity {
    fn base(&self) -> &EntityBase {
        &self.base
    }

    fn entity_type(&self) -> EntityTypeRef {
        &vanilla_entities::ITEM
    }

    fn tick(&self) {
        self.default_tick();
        if let Err(error) = self.try_set_position(self.tick_position) {
            panic!("moving tick test entity failed to move during tick: {error}");
        }
        self.set_rotation(self.tick_rotation);
    }
}

struct AddDuringTickTestEntity {
    base: EntityBase,
    manager: Arc<WorldEntityManager>,
    entity_to_add: SyncMutex<Option<SharedEntity>>,
}

impl AddDuringTickTestEntity {
    fn shared(
        id: i32,
        uuid: Uuid,
        position: DVec3,
        manager: Arc<WorldEntityManager>,
        entity_to_add: SharedEntity,
    ) -> SharedEntity {
        Arc::new(Self {
            base: EntityBase::with_uuid(
                id,
                uuid,
                position,
                vanilla_entities::ITEM.dimensions,
                Weak::new(),
            ),
            manager,
            entity_to_add: SyncMutex::new(Some(entity_to_add)),
        })
    }
}

crate::entity::impl_test_downcast_type!(AddDuringTickTestEntity);

impl Entity for AddDuringTickTestEntity {
    fn base(&self) -> &EntityBase {
        &self.base
    }

    fn entity_type(&self) -> EntityTypeRef {
        &vanilla_entities::ITEM
    }

    fn tick(&self) {
        self.default_tick();
        let Some(entity) = self.entity_to_add.lock().take() else {
            return;
        };
        if let Err(error) = self
            .manager
            .add_live_entity(entity, EntityOwnership::ManagerOwned)
        {
            panic!("add-during-tick test entity failed to add live entity: {error}");
        }
    }
}

struct DespawnOnCheckTestEntity {
    base: EntityBase,
}

impl DespawnOnCheckTestEntity {
    fn shared(id: i32, uuid: Uuid, position: DVec3) -> SharedEntity {
        Arc::new(Self {
            base: EntityBase::with_uuid(
                id,
                uuid,
                position,
                vanilla_entities::ITEM.dimensions,
                Weak::new(),
            ),
        })
    }
}

crate::entity::impl_test_downcast_type!(DespawnOnCheckTestEntity);

impl Entity for DespawnOnCheckTestEntity {
    fn base(&self) -> &EntityBase {
        &self.base
    }

    fn entity_type(&self) -> EntityTypeRef {
        &vanilla_entities::ITEM
    }

    fn check_despawn(&self) {
        self.set_removed(RemovalReason::Discarded);
    }
}

crate::entity::impl_test_downcast_type!(ManagerTestEntity);

impl Entity for ManagerTestEntity {
    fn base(&self) -> &EntityBase {
        &self.base
    }

    fn entity_type(&self) -> EntityTypeRef {
        self.entity_type
    }

    fn is_always_ticking(&self) -> bool {
        self.always_ticking
    }

    fn is_removed(&self) -> bool {
        if let Some(registration_check_gate) = &self.registration_check_gate {
            registration_check_gate.on_removed_check();
        }
        self.base.is_removed()
    }
}

fn entity(id: i32, uuid_seed: u128, position: DVec3) -> SharedEntity {
    ManagerTestEntity::shared(id, Uuid::from_u128(uuid_seed), position)
}

fn assert_empty_lifecycle(changes: EntityLifecycleChanges) {
    assert!(changes.tracking_started.is_empty());
    assert!(changes.tracking_stopped.is_empty());
    assert!(changes.ticking_started.is_empty());
    assert!(changes.ticking_stopped.is_empty());
}

fn load_chunk(manager: &WorldEntityManager, chunk: ChunkPos) {
    let result = manager.on_chunk_loaded(chunk);
    assert!(result.restored.is_empty());
    assert!(result.tracking_started.is_empty());
    assert!(result.ticking_started.is_empty());
    assert!(!result.needs_save);
    assert_empty_lifecycle(manager.update_chunk_visibility(chunk, EntityVisibility::Ticking));
}

fn track_chunk(manager: &WorldEntityManager, chunk: ChunkPos) {
    let result = manager.on_chunk_loaded(chunk);
    assert!(result.restored.is_empty());
    assert!(result.tracking_started.is_empty());
    assert!(result.ticking_started.is_empty());
    assert!(!result.needs_save);
    assert_empty_lifecycle(manager.update_chunk_visibility(chunk, EntityVisibility::Tracked));
}

mod lifecycle;
mod movement_and_visibility;
mod persistence;
mod queries_and_order;
mod ticking;
