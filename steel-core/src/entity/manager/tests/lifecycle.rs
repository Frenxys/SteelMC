use super::*;
use std::thread;

#[test]
fn add_live_entity_rejects_manager_owned_unloaded_chunk() {
    let manager = WorldEntityManager::new();
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));

    assert!(matches!(
        manager.add_live_entity(entity.clone(), EntityOwnership::ManagerOwned),
        Err(AddEntityError::ChunkNotLoaded {
            entity_id: 1,
            chunk,
        }) if chunk == ChunkPos::new(0, 0)
    ));
    assert_eq!(manager.count(), 0);
    assert!(manager.get_by_id(entity.id()).is_none());
}

#[test]
fn add_live_entity_accepts_external_unloaded_chunk() {
    let manager = WorldEntityManager::new();
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));

    assert!(
        manager
            .add_live_entity(entity.clone(), EntityOwnership::External)
            .is_ok()
    );
    assert_eq!(manager.count(), 1);

    let Some(live_entity) = manager.get_by_id(entity.id()) else {
        panic!("entity in unloaded chunk should be live");
    };
    assert!(Arc::ptr_eq(&entity, &live_entity));
}

#[test]
fn add_live_entity_rejects_duplicate_uuid_without_registering_second_entity() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));

    let uuid = Uuid::from_u128(5);
    let first = ManagerTestEntity::shared(1, uuid, DVec3::new(1.0, 64.0, 1.0));
    let second = ManagerTestEntity::shared(2, uuid, DVec3::new(2.0, 64.0, 1.0));

    assert!(
        manager
            .add_live_entity(first.clone(), EntityOwnership::ManagerOwned)
            .is_ok()
    );
    assert!(matches!(
        manager.add_live_entity(second, EntityOwnership::ManagerOwned),
        Err(AddEntityError::DuplicateUuid {
            entity_id: 2,
            uuid: duplicate,
        }) if duplicate == uuid
    ));

    let Some(live_first) = manager.get_by_id(1) else {
        panic!("first entity should stay registered");
    };
    assert!(Arc::ptr_eq(&first, &live_first));
    assert!(manager.get_by_id(2).is_none());
    assert_eq!(manager.count(), 1);
}

#[test]
fn manager_binding_invalidates_a_pre_registration_null_attempt() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let destination = DVec3::new(4.0, 64.0, 1.0);
    let stale_change = entity.base().position_change_for_test(destination);

    let result = manager.add_live_entity_with_callback(
        Arc::clone(&entity),
        EntityOwnership::ManagerOwned,
        manager_spatial_callback(&manager, &entity),
    );
    assert!(result.is_ok(), "entity should bind atomically: {result:?}");

    assert!(matches!(
        stale_change.commit(),
        EntitySpatialCommitResult::Retry
    ));
    assert_eq!(entity.position(), DVec3::new(1.0, 64.0, 1.0));

    assert!(entity.try_set_position(destination).is_ok());
    assert_eq!(entity.position(), destination);
    assert_eq!(
        manager.get_entities_in_aabb(&entity.bounding_box()).len(),
        1
    );
}

#[test]
fn failed_single_and_tree_registration_retry_spatial_writers_after_rollback() {
    fn assert_writer_retries_after_rollback(register_tree: bool, change_bounding_box: bool) {
        let manager = Arc::new(WorldEntityManager::new());
        load_chunk(&manager, ChunkPos::new(0, 0));
        let duplicate_uuid = Uuid::from_u128(1);
        let existing = ManagerTestEntity::shared(1, duplicate_uuid, DVec3::new(1.0, 64.0, 1.0));
        assert!(
            manager
                .add_live_entity(existing, EntityOwnership::ManagerOwned)
                .is_ok()
        );

        let registration_gate = RegistrationCheckGate::new();
        let candidate = ManagerTestEntity::shared_with_registration_check_gate(
            2,
            duplicate_uuid,
            DVec3::new(2.0, 64.0, 1.0),
            Arc::clone(&registration_gate),
        );
        let commit_entered = Arc::new(Barrier::new(2));
        let callback = manager_spatial_callback_with_commit_gate(
            &manager,
            &candidate,
            Some(Arc::clone(&commit_entered)),
        );
        let registration_manager = Arc::clone(&manager);
        let registration_entity = Arc::clone(&candidate);
        let registration = thread::spawn(move || {
            if register_tree {
                registration_manager.add_live_entity_tree_with_callbacks(
                    slice::from_ref(&registration_entity),
                    EntityOwnership::ManagerOwned,
                    vec![callback],
                )
            } else {
                registration_manager.add_live_entity_with_callback(
                    registration_entity,
                    EntityOwnership::ManagerOwned,
                    callback,
                )
            }
        });

        registration_gate.callback_installed.wait();
        let destination = DVec3::new(5.0, 64.0, 1.0);
        let destination_box = WorldAabb::new(4.5, 63.5, 0.5, 5.5, 65.5, 1.5);
        let writer_entity = Arc::clone(&candidate);
        let writer = thread::spawn(move || {
            if change_bounding_box {
                writer_entity.base().set_bounding_box(destination_box);
                Ok(())
            } else {
                writer_entity.try_set_position(destination)
            }
        });
        commit_entered.wait();
        registration_gate.release_registration.wait();

        let Ok(registration_result) = registration.join() else {
            panic!("registration thread panicked");
        };
        assert!(matches!(
            registration_result,
            Err(AddEntityError::DuplicateUuid {
                entity_id: 2,
                uuid,
            }) if uuid == duplicate_uuid
        ));
        let Ok(writer_result) = writer.join() else {
            panic!("spatial writer thread panicked");
        };
        assert!(
            writer_result.is_ok(),
            "writer should retry through the restored local callback: {writer_result:?}"
        );
        if change_bounding_box {
            assert_eq!(candidate.bounding_box(), destination_box);
        } else {
            assert_eq!(candidate.position(), destination);
        }
        assert!(manager.get_by_id(candidate.id()).is_none());
    }

    assert_writer_retries_after_rollback(false, false);
    assert_writer_retries_after_rollback(false, true);
    assert_writer_retries_after_rollback(true, false);
    assert_writer_retries_after_rollback(true, true);
}

#[test]
fn effect_queue_preserves_move_remove_readd_order_during_reentrant_drain() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    load_chunk(&manager, ChunkPos::new(1, 0));
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let result = manager.add_live_entity_with_callback(
        Arc::clone(&entity),
        EntityOwnership::ManagerOwned,
        manager_spatial_callback(&manager, &entity),
    );
    assert!(result.is_ok(), "entity should bind atomically: {result:?}");
    manager.drain_effects(|_| {});

    assert!(entity.try_set_position(DVec3::new(17.0, 64.0, 1.0)).is_ok());

    let mut observed = Vec::new();
    manager.drain_effects(|effect| match effect {
        EntityManagerEffect::SpatialMove(update) => {
            observed.push("move");
            let entity = Arc::clone(update.entity());
            assert!(
                manager
                    .remove_live_entity(update.entity().as_ref(), RemovalReason::ChangedWorld)
                    .is_some()
            );
            let result = manager.add_live_entity_with_callback(
                Arc::clone(&entity),
                EntityOwnership::ManagerOwned,
                manager_spatial_callback(&manager, &entity),
            );
            assert!(
                result.is_ok(),
                "entity should rebind atomically: {result:?}"
            );
            manager.drain_effects(|_| panic!("reentrant drain must not overtake active effects"));
        }
        EntityManagerEffect::Removal { entity, .. } => {
            assert_eq!(entity.id(), 1);
            observed.push("remove");
        }
        EntityManagerEffect::TrackingStart(entity) => {
            assert_eq!(entity.id(), 1);
            observed.push("start");
        }
        EntityManagerEffect::TrackingStop(entity_id) => {
            panic!("unexpected tracking stop for entity {entity_id}")
        }
    });

    assert_eq!(observed, vec!["move", "remove", "start"]);
    assert!(Arc::ptr_eq(
        &manager
            .get_by_id(entity.id())
            .expect("entity should be live"),
        &entity
    ));
}

#[test]
fn delayed_callback_cannot_remove_the_same_entity_after_readd() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let original_callback = manager_spatial_callback(&manager, &entity);
    let delayed_callback = original_callback.callback_for_test();
    let result = manager.add_live_entity_with_callback(
        Arc::clone(&entity),
        EntityOwnership::ManagerOwned,
        original_callback,
    );
    assert!(result.is_ok(), "entity should bind atomically: {result:?}");

    assert!(
        manager
            .remove_live_entity(entity.as_ref(), RemovalReason::ChangedWorld)
            .is_some()
    );
    let result = manager.add_live_entity_with_callback(
        Arc::clone(&entity),
        EntityOwnership::ManagerOwned,
        manager_spatial_callback(&manager, &entity),
    );
    assert!(
        result.is_ok(),
        "entity should rebind atomically: {result:?}"
    );

    delayed_callback.on_remove(RemovalReason::Discarded);

    let live = manager
        .get_by_id(entity.id())
        .expect("the current callback binding should remain live");
    assert!(Arc::ptr_eq(&live, &entity));
    let destination = DVec3::new(4.0, 64.0, 1.0);
    assert!(entity.try_set_position(destination).is_ok());
    assert_eq!(entity.position(), destination);
}

#[test]
#[should_panic(expected = "entity 1 current manager callback had no live entry")]
fn current_non_position_callback_without_a_live_entry_panics() {
    let manager = Arc::new(WorldEntityManager::new());
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let callback = manager_spatial_callback(&manager, &entity);
    entity
        .base()
        .replace_level_callback(callback.callback_for_test());

    let change = entity
        .base()
        .bounding_box_change_for_test(WorldAabb::new(0.5, 63.5, 0.5, 1.5, 65.5, 1.5));

    manager.commit_spatial_change(entity.id(), &change);
}

#[test]
#[should_panic(expected = "entity 1 current manager callback targeted a different live entity")]
fn current_non_position_callback_for_a_different_live_entity_panics() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    let live = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let result = add_live_entity_with_spatial_callback(
        &manager,
        Arc::clone(&live),
        EntityOwnership::ManagerOwned,
    );
    assert!(result.is_ok(), "live entity should register: {result:?}");

    let stale = entity(1, 2, DVec3::new(2.0, 64.0, 1.0));
    let callback = manager_spatial_callback(&manager, &stale);
    stale
        .base()
        .replace_level_callback(callback.callback_for_test());

    let change = stale
        .base()
        .bounding_box_change_for_test(WorldAabb::new(1.5, 63.5, 0.5, 2.5, 65.5, 1.5));

    manager.commit_spatial_change(stale.id(), &change);
}

#[test]
fn stale_same_id_entity_cannot_mutate_or_remove_its_replacement() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    let original = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let result = manager.add_live_entity_with_callback(
        Arc::clone(&original),
        EntityOwnership::ManagerOwned,
        manager_spatial_callback(&manager, &original),
    );
    assert!(
        result.is_ok(),
        "original should bind atomically: {result:?}"
    );

    let stale_move = original
        .base()
        .position_change_for_test(DVec3::new(3.0, 64.0, 1.0));
    let stale_box = original
        .base()
        .bounding_box_change_for_test(WorldAabb::new(3.0, 63.0, 0.0, 4.0, 65.0, 2.0));
    assert!(
        manager
            .remove_live_entity(original.as_ref(), RemovalReason::ChangedWorld)
            .is_some()
    );

    let replacement_position = DVec3::new(8.0, 64.0, 1.0);
    let replacement = entity(1, 2, replacement_position);
    let result = manager.add_live_entity_with_callback(
        Arc::clone(&replacement),
        EntityOwnership::ManagerOwned,
        manager_spatial_callback(&manager, &replacement),
    );
    assert!(
        result.is_ok(),
        "replacement should bind atomically: {result:?}"
    );

    assert!(matches!(
        manager.commit_move(original.id(), &stale_move),
        Err(EntityMoveError::NotLive { entity_id: 1 })
    ));
    assert_eq!(
        manager.commit_spatial_change(original.id(), &stale_box),
        EntitySpatialCommitResult::Retry
    );
    assert!(
        manager
            .remove_live_entity(original.as_ref(), RemovalReason::Discarded)
            .is_none()
    );

    let live = manager
        .get_by_id(replacement.id())
        .expect("replacement should remain live");
    assert!(Arc::ptr_eq(&live, &replacement));
    assert_eq!(replacement.position(), replacement_position);
}

#[test]
fn add_live_entity_tree_rejects_duplicate_uuid_and_restores_callback_bindings() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));

    let existing_uuid = Uuid::from_u128(5);
    let existing = ManagerTestEntity::shared(1, existing_uuid, DVec3::new(1.0, 64.0, 1.0));
    let result = manager.add_live_entity(Arc::clone(&existing), EntityOwnership::ManagerOwned);
    assert!(
        result.is_ok(),
        "existing entity should register before duplicate UUID test: {result:?}"
    );

    let vehicle = entity(2, 6, DVec3::new(2.0, 64.0, 2.0));
    let passenger = ManagerTestEntity::shared(3, existing_uuid, DVec3::new(2.0, 64.0, 2.0));
    EntityBase::restore_passenger_relationship(&vehicle, &passenger);
    let destination = DVec3::new(3.0, 64.0, 2.0);
    let stale_change = vehicle.base().position_change_for_test(destination);

    assert!(matches!(
        manager.add_live_entity_tree(
            &[Arc::clone(&vehicle), Arc::clone(&passenger)],
            EntityOwnership::ManagerOwned,
        ),
        Err(AddEntityError::DuplicateUuid {
            entity_id: 3,
            uuid,
        }) if uuid == existing_uuid
    ));
    assert!(manager.get_by_id(2).is_none());
    assert!(manager.get_by_id(3).is_none());
    assert_eq!(manager.count(), 1);
    assert!(matches!(
        stale_change.commit(),
        EntitySpatialCommitResult::Retry
    ));
    assert!(vehicle.try_set_position(destination).is_ok());
    assert_eq!(vehicle.position(), destination);
}

#[test]
#[should_panic(expected = "entity id 1 is already registered in the world entity manager")]
fn duplicate_entity_id_is_a_loud_invariant_failure() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));

    assert!(
        manager
            .add_live_entity(
                entity(1, 1, DVec3::new(1.0, 64.0, 1.0)),
                EntityOwnership::ManagerOwned,
            )
            .is_ok()
    );
    let _ = manager.add_live_entity(
        entity(1, 2, DVec3::new(2.0, 64.0, 1.0)),
        EntityOwnership::ManagerOwned,
    );
}
