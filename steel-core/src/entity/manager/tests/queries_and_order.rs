use super::*;

#[test]
fn aabb_matching_query_filters_accessible_entities() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));

    let first = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let second = entity(2, 2, DVec3::new(3.0, 64.0, 1.0));
    let outside = entity(3, 3, DVec3::new(30.0, 64.0, 1.0));
    assert!(
        manager
            .add_live_entity(first, EntityOwnership::ManagerOwned)
            .is_ok()
    );
    assert!(
        manager
            .add_live_entity(second.clone(), EntityOwnership::ManagerOwned)
            .is_ok()
    );
    assert!(matches!(
        manager.add_live_entity(outside, EntityOwnership::ManagerOwned),
        Err(AddEntityError::ChunkNotLoaded { .. })
    ));

    let aabb = WorldAabb::new(0.0, 63.0, 0.0, 5.0, 66.0, 3.0);
    let result = manager.get_entities_in_aabb_matching(&aabb, |entity| entity.id() == 2);

    assert_eq!(result.len(), 1);
    assert!(Arc::ptr_eq(&result[0], &second));
}

#[test]
fn visibility_transitions_separate_tracking_and_ticking() {
    let manager = WorldEntityManager::new();
    let chunk = ChunkPos::new(0, 0);
    let result = manager.on_chunk_loaded(chunk);
    assert!(result.restored.is_empty());
    assert!(result.tracking_started.is_empty());
    assert!(result.ticking_started.is_empty());

    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let changes = match manager.add_live_entity(entity.clone(), EntityOwnership::ManagerOwned) {
        Ok(changes) => changes,
        Err(error) => panic!("entity should register in active hidden chunk: {error}"),
    };
    assert_empty_lifecycle(changes);
    assert!(
        manager
            .get_entities_in_aabb(&entity.bounding_box())
            .is_empty()
    );

    let changes = manager.update_chunk_visibility(chunk, EntityVisibility::Tracked);
    assert_eq!(changes.tracking_started.len(), 1);
    assert!(Arc::ptr_eq(&changes.tracking_started[0], &entity));
    assert!(changes.ticking_started.is_empty());
    manager.tick_entities(0, true);
    assert_eq!(entity.tick_count(), 0);

    let changes = manager.update_chunk_visibility(chunk, EntityVisibility::Ticking);
    assert!(changes.tracking_started.is_empty());
    assert_eq!(changes.ticking_started.len(), 1);
    assert!(Arc::ptr_eq(&changes.ticking_started[0], &entity));
    manager.tick_entities(1, true);
    assert_eq!(entity.tick_count(), 1);

    let changes = manager.update_chunk_visibility(chunk, EntityVisibility::Tracked);
    assert!(changes.tracking_stopped.is_empty());
    assert_eq!(changes.ticking_stopped.len(), 1);
    assert!(Arc::ptr_eq(&changes.ticking_stopped[0], &entity));
    manager.tick_entities(2, true);
    assert_eq!(entity.tick_count(), 1);

    let changes = manager.update_chunk_visibility(chunk, EntityVisibility::Hidden);
    assert_eq!(changes.tracking_stopped.len(), 1);
    assert!(Arc::ptr_eq(&changes.tracking_stopped[0], &entity));
    assert!(changes.ticking_stopped.is_empty());
    assert!(
        manager
            .get_entities_in_aabb(&entity.bounding_box())
            .is_empty()
    );
}

#[test]
fn fine_grained_query_keeps_external_entities_accessible_in_hidden_chunks() {
    let manager = WorldEntityManager::new();
    let chunk = ChunkPos::new(0, 0);
    let result = manager.on_chunk_loaded(chunk);
    assert!(result.restored.is_empty());
    let managed_entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let external = entity(2, 2, DVec3::new(1.5, 64.0, 1.0));
    assert!(
        manager
            .add_live_entity(managed_entity, EntityOwnership::ManagerOwned)
            .is_ok()
    );
    assert!(
        manager
            .add_live_entity(Arc::clone(&external), EntityOwnership::External)
            .is_ok()
    );

    let result = manager.get_entities_in_aabb(&WorldAabb::new(0.0, 63.0, 0.0, 2.0, 66.0, 2.0));

    assert_eq!(result.len(), 1);
    assert!(Arc::ptr_eq(&result[0], &external));
}

#[test]
fn has_aabb_matching_query_respects_bounds_accessibility_and_predicate() {
    let manager = WorldEntityManager::new();
    let loaded_chunk = ChunkPos::new(0, 0);
    let hidden_chunk = ChunkPos::new(1, 0);
    load_chunk(&manager, loaded_chunk);
    load_chunk(&manager, hidden_chunk);

    let filtered_out = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let matching = entity(2, 2, DVec3::new(3.0, 64.0, 1.0));
    let hidden = entity(3, 3, DVec3::new(17.0, 64.0, 1.0));
    for entity in [filtered_out, matching, hidden] {
        assert!(
            manager
                .add_live_entity(entity, EntityOwnership::ManagerOwned)
                .is_ok()
        );
    }

    let loaded_aabb = WorldAabb::new(0.0, 63.0, 0.0, 5.0, 66.0, 3.0);
    assert!(manager.has_entity_in_aabb_matching(&loaded_aabb, |entity| entity.id() == 2));
    assert!(!manager.has_entity_in_aabb_matching(&loaded_aabb, |entity| entity.id() == 3));

    manager.begin_chunk_unload(hidden_chunk);
    let hidden_aabb = WorldAabb::new(16.0, 63.0, 0.0, 18.0, 66.0, 3.0);
    assert!(!manager.has_entity_in_aabb_matching(&hidden_aabb, |entity| entity.id() == 3));
}

#[test]
fn aabb_predicates_can_mutate_manager_after_candidate_snapshot() {
    let manager = WorldEntityManager::new();
    let chunk = ChunkPos::new(0, 0);
    load_chunk(&manager, chunk);

    let matching = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    assert!(
        manager
            .add_live_entity(matching, EntityOwnership::ManagerOwned)
            .is_ok()
    );

    let aabb = WorldAabb::new(0.0, 63.0, 0.0, 3.0, 66.0, 3.0);
    let matched = manager.has_entity_in_aabb_matching(&aabb, |entity| {
        let changes = manager.update_chunk_visibility(chunk, EntityVisibility::Hidden);
        assert_eq!(changes.tracking_stopped.len(), 1);
        entity.id() == 1
    });

    assert!(matched);
    assert!(manager.get_entities_in_aabb(&aabb).is_empty());
}

#[test]
fn aabb_matching_bounding_box_query_returns_only_matching_intersections() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));

    let filtered_out = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let matching = entity(2, 2, DVec3::new(3.0, 64.0, 1.0));
    let outside = entity(3, 3, DVec3::new(8.0, 64.0, 1.0));
    let expected_box = matching.bounding_box();
    for entity in [filtered_out, Arc::clone(&matching), outside] {
        assert!(
            add_live_entity_with_spatial_callback(&manager, entity, EntityOwnership::ManagerOwned,)
                .is_ok()
        );
    }

    let aabb = WorldAabb::new(2.0, 63.0, 0.0, 4.0, 66.0, 3.0);
    let replacement_box = WorldAabb::new(6.0, 63.0, 0.0, 7.0, 65.0, 1.0);
    let mut saw_outside_entity = false;
    let result = manager.get_entity_bounding_boxes_in_aabb_matching(&aabb, |entity| {
        saw_outside_entity |= entity.id() == 3;
        if entity.id() == 2 {
            entity.base().set_bounding_box(replacement_box);
        }
        entity.id() > 1
    });

    assert_eq!(result, vec![expected_box]);
    assert_eq!(matching.bounding_box(), replacement_box);
    assert!(!saw_outside_entity);
    let replacement_ids = manager
        .get_entities_in_aabb(&WorldAabb::new(5.5, 62.5, -0.5, 7.5, 65.5, 1.5))
        .into_iter()
        .map(|entity| entity.id())
        .collect::<Vec<_>>();
    assert_eq!(replacement_ids, vec![2]);
}

#[test]
fn nearest_aabb_matching_query_returns_closest_match() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));

    let near_filtered_out = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let near_match = entity(2, 2, DVec3::new(3.0, 64.0, 1.0));
    let far_match = entity(3, 3, DVec3::new(8.0, 64.0, 1.0));
    for entity in [near_filtered_out, near_match.clone(), far_match] {
        assert!(
            manager
                .add_live_entity(entity, EntityOwnership::ManagerOwned)
                .is_ok()
        );
    }

    let aabb = WorldAabb::new(0.0, 63.0, 0.0, 10.0, 66.0, 3.0);
    let result =
        manager.nearest_entity_in_aabb_matching(&aabb, DVec3::ZERO, |entity| entity.id() > 1);

    let Some(result) = result else {
        panic!("nearest matching entity should be found");
    };
    assert!(Arc::ptr_eq(&result, &near_match));
}

#[test]
fn accessible_entities_keep_tracking_start_order() {
    let manager = WorldEntityManager::new();
    let first_chunk = ChunkPos::new(0, 0);
    let second_chunk = ChunkPos::new(1, 0);
    load_chunk(&manager, first_chunk);
    load_chunk(&manager, second_chunk);

    let first = entity(30, 30, DVec3::new(1.0, 80.0, 1.0));
    let second = entity(10, 10, DVec3::new(17.0, 64.0, 1.0));
    let third = entity(20, 20, DVec3::new(2.0, 64.0, 1.0));
    for entity in [Arc::clone(&first), Arc::clone(&second), Arc::clone(&third)] {
        assert!(
            manager
                .add_live_entity(entity, EntityOwnership::ManagerOwned)
                .is_ok()
        );
    }

    let entity_ids = manager
        .get_accessible_entities()
        .into_iter()
        .map(|entity| entity.id())
        .collect::<Vec<_>>();
    assert_eq!(entity_ids, vec![30, 10, 20]);

    let changes = manager.update_chunk_visibility(first_chunk, EntityVisibility::Hidden);
    assert_eq!(changes.tracking_stopped.len(), 2);
    let changes = manager.update_chunk_visibility(first_chunk, EntityVisibility::Tracked);
    assert_eq!(changes.tracking_started.len(), 2);

    let entity_ids = manager
        .get_accessible_entities()
        .into_iter()
        .map(|entity| entity.id())
        .collect::<Vec<_>>();
    assert_eq!(entity_ids, vec![10, 20, 30]);
}

#[test]
fn aabb_queries_use_vanilla_section_order_then_section_insertion_order() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));
    load_chunk(&manager, ChunkPos::new(0, 1));

    let later_section = entity(1, 1, DVec3::new(1.0, 64.0, 17.0));
    let first_same_section = entity(2, 2, DVec3::new(10.0, 64.0, 1.0));
    let second_same_section = entity(3, 3, DVec3::new(1.0, 64.0, 1.0));
    for entity in [
        later_section,
        Arc::clone(&first_same_section),
        Arc::clone(&second_same_section),
    ] {
        assert!(
            manager
                .add_live_entity(entity, EntityOwnership::ManagerOwned)
                .is_ok()
        );
    }

    let aabb = WorldAabb::new(0.0, 63.0, 0.0, 18.0, 66.0, 18.0);
    let entity_ids = manager
        .get_entities_in_aabb(&aabb)
        .into_iter()
        .map(|entity| entity.id())
        .collect::<Vec<_>>();
    assert_eq!(entity_ids, vec![2, 3, 1]);
}

#[test]
fn dense_aabb_query_keeps_insertion_order_across_spatial_cells() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));

    for insertion_index in 0..64 {
        let spatial_index = 63 - insertion_index;
        let entity_id = insertion_index + 1;
        assert!(
            manager
                .add_live_entity(
                    entity(
                        entity_id,
                        entity_id as u128,
                        DVec3::new(
                            f64::from(spatial_index % 8) + 0.5,
                            64.0,
                            f64::from(spatial_index / 8) + 0.5,
                        ),
                    ),
                    EntityOwnership::ManagerOwned,
                )
                .is_ok()
        );
    }

    let entity_ids = manager
        .get_entities_in_aabb(&WorldAabb::new(0.0, 63.0, 0.0, 8.0, 66.0, 8.0))
        .into_iter()
        .map(|entity| entity.id())
        .collect::<Vec<_>>();

    assert_eq!(entity_ids, (1..=64).collect::<Vec<_>>());
}

#[test]
fn entity_query_section_bounds_match_vanilla_search_grace() {
    let aabb = WorldAabb::new(0.0, 66.0, 0.0, 15.0, 79.0, 15.0);

    let (minimum, maximum) = WorldEntityManager::entity_query_section_bounds(&aabb);

    assert_eq!(minimum, SectionPos::new(-1, 3, -1));
    assert_eq!(maximum, SectionPos::new(1, 4, 1));
}

#[test]
fn fine_grained_query_deduplicates_an_entity_spanning_multiple_cells() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));
    let entity = entity(1, 1, DVec3::new(3.0, 64.0, 3.0));
    entity
        .base()
        .set_bounding_box(WorldAabb::new(1.0, 63.0, 1.0, 5.0, 67.0, 5.0));
    assert!(
        manager
            .add_live_entity(Arc::clone(&entity), EntityOwnership::ManagerOwned)
            .is_ok()
    );

    let result = manager.get_entities_in_aabb(&WorldAabb::new(1.5, 63.5, 1.5, 4.5, 66.5, 4.5));

    assert_eq!(result.len(), 1);
    assert!(Arc::ptr_eq(&result[0], &entity));
}

#[test]
fn fine_grained_query_prunes_a_dense_vanilla_section() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));
    for z in 0..16 {
        for x in 0..16 {
            let id = z * 16 + x + 1;
            assert!(
                manager
                    .add_live_entity(
                        entity(
                            id,
                            id as u128,
                            DVec3::new(f64::from(x) + 0.5, 64.0, f64::from(z) + 0.5),
                        ),
                        EntityOwnership::ManagerOwned,
                    )
                    .is_ok()
            );
        }
    }

    let query = WorldAabb::new(0.0, 63.0, 0.0, 2.0, 66.0, 2.0);
    let state = manager.state.read();
    let mut candidate_ids = Vec::new();
    assert!(
        state
            .spatial_index
            .try_visit_candidate_ids(query, |entity_id| candidate_ids.push(entity_id)),
        "small query should use the fine-grained spatial index"
    );

    assert_eq!(candidate_ids.len(), 4);
    assert_eq!(state.live_by_id.len(), 256);
}

#[test]
fn committed_same_section_move_refreshes_fine_grained_cells() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    assert!(
        add_live_entity_with_spatial_callback(
            &manager,
            Arc::clone(&entity),
            EntityOwnership::ManagerOwned,
        )
        .is_ok()
    );

    let old_query = WorldAabb::new(0.0, 63.0, 0.0, 2.0, 66.0, 2.0);
    let new_query = WorldAabb::new(8.0, 63.0, 0.0, 10.0, 66.0, 2.0);
    assert_eq!(manager.get_entities_in_aabb(&old_query).len(), 1);
    assert!(entity.try_set_position(DVec3::new(9.0, 64.0, 1.0)).is_ok());

    assert!(manager.get_entities_in_aabb(&old_query).is_empty());
    let result = manager.get_entities_in_aabb(&new_query);
    assert_eq!(result.len(), 1);
    assert!(Arc::ptr_eq(&result[0], &entity));
}

#[test]
fn direct_box_and_pose_changes_refresh_fine_grained_cells() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    assert!(
        add_live_entity_with_spatial_callback(
            &manager,
            Arc::clone(&entity),
            EntityOwnership::ManagerOwned,
        )
        .is_ok()
    );

    let original_query = WorldAabb::new(0.0, 63.0, 0.0, 2.0, 66.0, 2.0);
    let custom_query = WorldAabb::new(6.0, 63.0, 0.0, 8.0, 66.0, 2.0);
    entity
        .base()
        .set_bounding_box(WorldAabb::new(6.5, 63.5, 0.5, 7.5, 65.5, 1.5));
    assert!(manager.get_entities_in_aabb(&original_query).is_empty());
    assert_eq!(manager.get_entities_in_aabb(&custom_query).len(), 1);

    entity
        .base()
        .set_pose_and_dimensions(EntityPose::Sneaking, EntityDimensions::new(0.8, 1.2, 0.9));
    assert!(manager.get_entities_in_aabb(&custom_query).is_empty());
    assert_eq!(manager.get_entities_in_aabb(&original_query).len(), 1);
}

#[test]
fn crossing_sections_reinserts_after_existing_section_entities() {
    let manager = Arc::new(WorldEntityManager::new());
    load_chunk(&manager, ChunkPos::new(0, 0));
    load_chunk(&manager, ChunkPos::new(1, 0));
    let moving = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    let staying = entity(2, 2, DVec3::new(2.0, 64.0, 1.0));
    for entity in [Arc::clone(&moving), Arc::clone(&staying)] {
        assert!(
            add_live_entity_with_spatial_callback(&manager, entity, EntityOwnership::ManagerOwned,)
                .is_ok()
        );
    }

    assert!(moving.try_set_position(DVec3::new(17.0, 64.0, 1.0)).is_ok());
    assert!(moving.try_set_position(DVec3::new(1.0, 64.0, 1.0)).is_ok());

    let ids = manager
        .get_entities_in_aabb(&WorldAabb::new(0.0, 63.0, 0.0, 3.0, 66.0, 2.0))
        .into_iter()
        .map(|entity| entity.id())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![2, 1]);
}

#[test]
fn oversized_entity_uses_fallback_without_duplicate_results() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    entity
        .base()
        .set_bounding_box(WorldAabb::new(-10.0, 50.0, -10.0, 12.0, 75.0, 12.0));
    assert!(
        manager
            .add_live_entity(Arc::clone(&entity), EntityOwnership::ManagerOwned)
            .is_ok()
    );

    let result = manager.get_entities_in_aabb(&WorldAabb::new(0.0, 63.0, 0.0, 2.0, 66.0, 2.0));

    assert_eq!(result.len(), 1);
    assert!(Arc::ptr_eq(&result[0], &entity));
}

#[test]
fn oversized_fallback_does_not_widen_vanilla_section_search() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(3, 0));
    let entity = entity(1, 1, DVec3::new(49.0, 64.0, 1.0));
    entity
        .base()
        .set_bounding_box(WorldAabb::new(-1.0, 63.0, 0.0, 50.0, 65.0, 2.0));
    assert!(
        manager
            .add_live_entity(entity, EntityOwnership::ManagerOwned)
            .is_ok()
    );

    assert!(
        manager
            .get_entities_in_aabb(&WorldAabb::new(0.0, 63.0, 0.0, 2.0, 65.0, 2.0))
            .is_empty()
    );
}

#[test]
fn unload_removes_spatial_membership_and_restore_rebuilds_it_once() {
    let manager = WorldEntityManager::new();
    let chunk = ChunkPos::new(0, 0);
    load_chunk(&manager, chunk);
    let entity = entity(1, 1, DVec3::new(1.0, 64.0, 1.0));
    assert!(
        manager
            .add_live_entity(Arc::clone(&entity), EntityOwnership::ManagerOwned)
            .is_ok()
    );
    let query = WorldAabb::new(0.0, 63.0, 0.0, 2.0, 66.0, 2.0);
    assert_eq!(manager.get_entities_in_aabb(&query).len(), 1);

    let unload = manager.begin_chunk_unload(chunk);
    assert_eq!(unload.retained.len(), 1);
    assert!(manager.get_entities_in_aabb(&query).is_empty());
    let inactive_box = WorldAabb::new(6.5, 63.5, 0.5, 7.5, 65.5, 1.5);
    entity.base().set_bounding_box(inactive_box);
    let inactive_query = WorldAabb::new(6.0, 63.0, 0.0, 8.0, 66.0, 2.0);

    let loaded = manager.on_chunk_loaded(chunk);
    assert_eq!(loaded.restored.len(), 1);
    manager.update_chunk_visibility(chunk, EntityVisibility::Tracked);
    assert!(manager.get_entities_in_aabb(&query).is_empty());
    let result = manager.get_entities_in_aabb(&inactive_query);
    assert_eq!(result.len(), 1);
    assert!(Arc::ptr_eq(&result[0], &entity));
}

#[test]
fn large_query_fallback_keeps_vanilla_section_order() {
    let manager = WorldEntityManager::new();
    load_chunk(&manager, ChunkPos::new(0, 0));
    load_chunk(&manager, ChunkPos::new(1, 0));
    let later_section = entity(1, 1, DVec3::new(17.0, 64.0, 1.0));
    let earlier_section = entity(2, 2, DVec3::new(1.0, 64.0, 1.0));
    for entity in [later_section, earlier_section] {
        assert!(
            manager
                .add_live_entity(entity, EntityOwnership::ManagerOwned)
                .is_ok()
        );
    }

    let ids = manager
        .get_entities_in_aabb(&WorldAabb::new(-1.0, 0.0, -1.0, 66.0, 67.0, 66.0))
        .into_iter()
        .map(|entity| entity.id())
        .collect::<Vec<_>>();

    assert_eq!(ids, vec![2, 1]);
}
