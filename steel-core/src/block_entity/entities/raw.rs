//! NBT-preserving fallback block entity.

use std::{io::Cursor, sync::Weak};

use simdnbt::borrow::{
    BaseNbtCompound as BorrowedNbtCompound, NbtCompound as NbtCompoundView,
    read_compound as read_borrowed_compound,
};
use simdnbt::owned::NbtCompound;
use steel_registry::block_entity_type::BlockEntityTypeRef;
use steel_utils::{BlockPos, BlockStateId, DowncastType, DowncastTypeKey, locks::SyncMutex};

use crate::block_entity::{BlockEntity, BlockEntityBase};
use crate::world::World;

struct RawBlockEntityState {
    data: NbtCompound,
}

/// Steel-specific fallback for block entity types whose runtime behavior is not implemented yet.
///
/// Vanilla has concrete classes for every block entity type. Steel uses this only to preserve
/// worldgen and disk NBT until the corresponding typed implementation is added.
pub struct RawBlockEntity {
    base: BlockEntityBase,
    state: SyncMutex<RawBlockEntityState>,
}

// SAFETY: This key identifies the Steel fallback implementation, independently
// of the Minecraft block-entity registry entry stored inside it.
unsafe impl DowncastType for RawBlockEntity {
    const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new("steel:block_entity/raw");
}

impl RawBlockEntity {
    /// Creates a raw block entity without additional NBT.
    #[must_use]
    pub fn new(
        block_entity_type: BlockEntityTypeRef,
        level: Weak<World>,
        pos: BlockPos,
        state: BlockStateId,
    ) -> Self {
        Self {
            base: BlockEntityBase::new(block_entity_type, level, pos, state),
            state: SyncMutex::new(RawBlockEntityState {
                data: NbtCompound::new(),
            }),
        }
    }

    /// Creates a raw block entity with already-owned additional NBT.
    #[must_use]
    pub fn with_data(
        block_entity_type: BlockEntityTypeRef,
        level: Weak<World>,
        pos: BlockPos,
        state: BlockStateId,
        data: NbtCompound,
    ) -> Self {
        let mut encoded = Vec::new();
        data.write(&mut encoded);
        let entity = Self::new(block_entity_type, level, pos, state);
        if let Ok(borrowed) = read_borrowed_compound(&mut Cursor::new(encoded.as_slice())) {
            entity.load_with_components(&borrowed);
        } else {
            log::warn!(
                "Failed to reborrow owned data for raw block entity {}",
                block_entity_type.key,
            );
            entity.state.lock().data = data;
        }
        entity
    }
}

impl BlockEntity for RawBlockEntity {
    fn base(&self) -> &BlockEntityBase {
        &self.base
    }

    fn load_additional(&self, nbt: &BorrowedNbtCompound<'_>) {
        let nbt_view: NbtCompoundView<'_, '_> = nbt.into();
        let mut data = nbt_view.to_owned();
        while data.remove("components").is_some() {}
        self.state.lock().data = data;
    }

    fn save_additional(&self, nbt: &mut NbtCompound) {
        *nbt = self.state.lock().data.clone();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use steel_registry::{
        data_components::{DataComponentMap, vanilla_components::CUSTOM_NAME},
        test_support::init_test_registry,
        vanilla_block_entity_types, vanilla_blocks,
    };
    use text_components::TextComponent;

    use super::*;

    #[test]
    fn full_metadata_replaces_stale_raw_metadata() {
        init_test_registry();
        let mut data = NbtCompound::new();
        data.insert("id", "minecraft:chest");
        data.insert("x", 100_i32);
        data.insert("custom", 7_i32);
        let entity = RawBlockEntity::with_data(
            &vanilla_block_entity_types::BARREL,
            Weak::new(),
            BlockPos::new(2, 70, -4),
            vanilla_blocks::BARREL.default_state(),
            data,
        );

        let saved = entity.save_with_full_metadata();
        let custom = entity.save_custom_only();

        assert_eq!(
            saved.string("id").map(ToString::to_string),
            Some("minecraft:barrel".to_owned())
        );
        assert_eq!(saved.int("x"), Some(2));
        assert_eq!(saved.int("y"), Some(70));
        assert_eq!(saved.int("z"), Some(-4));
        assert_eq!(saved.int("custom"), Some(7));
        assert!(!custom.contains("id"));
        assert!(!custom.contains("x"));
        assert_eq!(custom.int("custom"), Some(7));
    }

    #[test]
    fn stored_components_survive_raw_block_entity_load_and_save() {
        init_test_registry();
        let custom_name = TextComponent::from("Stored raw name");
        let mut components = DataComponentMap::new();
        components.set(CUSTOM_NAME, Some(custom_name.clone()));
        let mut data = NbtCompound::new();
        data.insert("components", components.to_nbt_tag_ref());
        let entity = RawBlockEntity::with_data(
            &vanilla_block_entity_types::BARREL,
            Weak::new(),
            BlockPos::new(2, 70, -4),
            vanilla_blocks::BARREL.default_state(),
            data,
        );

        let collected = entity
            .collect_components()
            .expect("stored component snapshot should collect");
        let saved = entity.save_without_metadata();

        assert_eq!(collected.get_ref(CUSTOM_NAME), Some(&custom_name));
        assert!(saved.compound("components").is_some());
        assert!(!entity.save_custom_only().contains("components"));
    }

    #[test]
    #[should_panic(expected = "invalid block entity minecraft:barrel state minecraft:stone")]
    fn constructor_rejects_a_type_state_mismatch() {
        init_test_registry();
        let _ = RawBlockEntity::new(
            &vanilla_block_entity_types::BARREL,
            Weak::new(),
            BlockPos::new(2, 70, -4),
            vanilla_blocks::STONE.default_state(),
        );
    }
}
