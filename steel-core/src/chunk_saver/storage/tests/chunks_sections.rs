use super::*;

#[test]
fn proto_heightmap_save_preserves_existing_maps_and_load_primes_missing_maps() {
    init_test_registry();

    let pos = ChunkPos::new(3, -4);
    let proto = Chunk::new(single_empty_section(), pos, 0, 16, Weak::new());
    proto.set_status(ChunkStatus::Noise);
    proto.heightmaps.write().prime_from_sections(
        &[HeightmapType::WorldSurfaceWg],
        0,
        16,
        &proto.sections.sections,
    );
    let chunk = ChunkAccess::Proto(proto);

    let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk, ChunkStatus::Noise, &[], false)
    else {
        panic!("dirty proto chunk should prepare for saving");
    };
    assert_eq!(prepared.persistent.heightmaps.len(), 1);
    assert_eq!(
        prepared.persistent.heightmaps[0].heightmap_type,
        HeightmapType::WorldSurfaceWg.persistence_id()
    );

    let loaded = ChunkStorage::persistent_to_chunk(
        &prepared.persistent,
        pos,
        ChunkStatus::Noise,
        0,
        16,
        Weak::new(),
    );
    let ChunkAccess::Proto(loaded) = loaded.chunk else {
        panic!("noise status should load as a proto chunk");
    };
    let heightmaps = loaded.heightmaps.read();
    assert!(heightmaps.get(HeightmapType::WorldSurfaceWg).is_some());
    assert!(heightmaps.get(HeightmapType::OceanFloorWg).is_some());
}

#[test]
fn carvers_heightmap_save_excludes_stale_worldgen_maps() {
    init_test_registry();

    let proto = Chunk::new(
        single_empty_section(),
        ChunkPos::new(3, -4),
        0,
        16,
        Weak::new(),
    );
    proto.set_status(ChunkStatus::Carvers);
    {
        let mut heightmaps = proto.heightmaps.write();
        heightmaps.replace(Heightmap::new(HeightmapType::WorldSurfaceWg, 0, 16));
        heightmaps.replace(Heightmap::new(HeightmapType::WorldSurface, 0, 16));
    }
    let chunk = ChunkAccess::Proto(proto);

    let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk, ChunkStatus::Carvers, &[], false)
    else {
        panic!("dirty proto chunk should prepare for saving");
    };
    assert_eq!(prepared.persistent.heightmaps.len(), 1);
    assert_eq!(
        prepared.persistent.heightmaps[0].heightmap_type,
        HeightmapType::WorldSurface.persistence_id()
    );
}

#[test]
fn proto_carving_mask_presence_roundtrips_when_empty() {
    init_test_registry();

    let pos = ChunkPos::new(3, -4);
    let proto = Chunk::new(single_empty_section(), pos, 0, 16, Weak::new());
    proto.set_status(ChunkStatus::Carvers);
    drop(proto.get_or_create_carving_mask());
    let chunk = ChunkAccess::Proto(proto);

    let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk, ChunkStatus::Carvers, &[], false)
    else {
        panic!("dirty proto chunk should prepare for saving");
    };
    assert_eq!(prepared.persistent.carving_mask, Some(Vec::new()));

    let loaded = ChunkStorage::persistent_to_chunk(
        &prepared.persistent,
        pos,
        ChunkStatus::Carvers,
        0,
        16,
        Weak::new(),
    );
    let ChunkAccess::Proto(loaded_proto) = loaded.chunk else {
        panic!("carvers status should load as proto chunk");
    };

    assert!(loaded_proto.carving_mask.read().is_some());
}

#[tokio::test]
async fn ram_only_storage_restores_the_status_bundled_with_the_prepared_save() {
    init_test_registry();

    let pos = ChunkPos::new(3, -4);
    let proto = Chunk::new(single_empty_section(), pos, 0, 16, Weak::new());
    proto.set_status(ChunkStatus::Carvers);
    let chunk = ChunkAccess::Proto(proto);
    let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk, ChunkStatus::Carvers, &[], false)
    else {
        panic!("dirty proto chunk should prepare for saving");
    };

    let storage = RamOnlyStorage::empty_world();
    let Ok(true) = storage.save_chunk_data(prepared).await else {
        panic!("prepared chunk should save to RAM storage");
    };
    let Ok(Some(loaded)) = storage.load_chunk(pos, 0, 16, Weak::new()).await else {
        panic!("saved chunk should load from RAM storage");
    };

    assert_eq!(loaded.status, ChunkStatus::Carvers);
    assert!(matches!(loaded.chunk, ChunkAccess::Proto(_)));
}

#[test]
fn proto_carving_mask_bits_roundtrip_through_persistent_chunk() {
    init_test_registry();

    let pos = ChunkPos::new(3, -4);
    let proto = Chunk::new(single_empty_section(), pos, 0, 16, Weak::new());
    proto.set_status(ChunkStatus::Carvers);
    {
        let mut mask = proto.get_or_create_carving_mask();
        mask.set(7, 5, 11);
    }
    let chunk = ChunkAccess::Proto(proto);

    let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk, ChunkStatus::Carvers, &[], false)
    else {
        panic!("dirty proto chunk should prepare for saving");
    };
    assert!(
        prepared
            .persistent
            .carving_mask
            .as_ref()
            .is_some_and(|packed| !packed.is_empty())
    );

    let loaded = ChunkStorage::persistent_to_chunk(
        &prepared.persistent,
        pos,
        ChunkStatus::Carvers,
        0,
        16,
        Weak::new(),
    );
    let ChunkAccess::Proto(loaded_proto) = loaded.chunk else {
        panic!("carvers status should load as proto chunk");
    };

    let mask_guard = loaded_proto.carving_mask.read();
    let Some(mask) = mask_guard.as_ref() else {
        panic!("carving mask should restore from persistent chunk");
    };
    assert!(mask.get(7, 5, 11));
    assert!(!mask.get(8, 5, 11));
}

#[test]
fn proto_postprocessing_roundtrips_through_persistent_chunk() {
    init_test_registry();

    let pos = ChunkPos::new(-2, 1);
    let marked = BlockPos::new(-17, -63, 31);
    let proto = Chunk::new(single_empty_section(), pos, -64, 16, Weak::new());
    proto.set_status(ChunkStatus::Noise);
    proto.mark_pos_for_postprocessing(marked);
    let packed = Chunk::pack_postprocessing_offset(marked);
    let chunk = ChunkAccess::Proto(proto);

    let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk, ChunkStatus::Noise, &[], false)
    else {
        panic!("dirty proto chunk should prepare for saving");
    };

    assert_eq!(prepared.persistent.postprocessing, vec![vec![packed]]);

    let loaded = ChunkStorage::persistent_to_chunk(
        &prepared.persistent,
        pos,
        ChunkStatus::Noise,
        -64,
        16,
        Weak::new(),
    );
    let ChunkAccess::Proto(loaded_proto) = loaded.chunk else {
        panic!("noise status should load as proto chunk");
    };

    assert_eq!(loaded_proto.postprocessing.read()[0], vec![packed]);
}

#[test]
fn full_chunk_postprocessing_roundtrips_through_persistent_chunk() {
    init_test_registry();
    init_runtime_registries();

    let pos = ChunkPos::new(-2, 1);
    let marked = BlockPos::new(-17, -63, 31);
    let packed = Chunk::pack_postprocessing_offset(marked);
    let persistent = ChunkStorage::to_persistent(
        &single_empty_section(),
        &[],
        &[],
        &[],
        Vec::new(),
        Vec::new(),
        Vec::new(),
        PersistentLightData::default(),
        None,
        vec![vec![packed]],
        Vec::new(),
        Vec::new(),
        Vec::new(),
        pos,
    );

    let loaded = ChunkStorage::persistent_to_chunk(
        &persistent,
        pos,
        ChunkStatus::Full,
        -64,
        16,
        Weak::new(),
    );
    let ChunkAccess::Full(loaded_full) = loaded.chunk else {
        panic!("full status should load as a full chunk");
    };
    assert_eq!(
        loaded_full.postprocessing_for_serialization(),
        vec![vec![packed]]
    );

    let chunk = ChunkAccess::Full(loaded_full);
    chunk.mark_dirty();
    let Some(prepared) = ChunkStorage::prepare_chunk_save(&chunk, ChunkStatus::Full, &[], false)
    else {
        panic!("dirty full chunk should prepare for saving");
    };

    assert_eq!(prepared.persistent.postprocessing, vec![vec![packed]]);
}
