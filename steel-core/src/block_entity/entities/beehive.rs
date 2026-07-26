//! Beehive block entity implementation.

use std::{io, sync::Weak};

use simdnbt::borrow::{BaseNbtCompound as BorrowedNbtCompound, NbtCompound as NbtCompoundView};
use simdnbt::owned::{NbtCompound, NbtList, NbtTag};
use steel_registry::{
    data_components::{
        BeehiveOccupant, Bees, DataComponentMap, EntityData, vanilla_components::BEES,
    },
    vanilla_block_entity_types, vanilla_entities,
};
use steel_utils::{BlockPos, BlockStateId, DowncastType, DowncastTypeKey, locks::SyncMutex};

use crate::block_entity::{BlockEntity, BlockEntityBase};
use crate::world::World;

/// Maximum number of occupants in a vanilla beehive.
pub const BEEHIVE_MAX_OCCUPANTS: usize = 3;
/// Minimum occupation time for bees without nectar.
pub const BEEHIVE_MIN_OCCUPATION_TICKS_NECTARLESS: i32 = 600;

struct BeeOccupant {
    entity_data: NbtCompound,
    ticks_in_hive: i32,
    min_ticks_in_hive: i32,
}

impl BeeOccupant {
    fn worldgen(ticks_in_hive: i32) -> Self {
        Self {
            entity_data: default_bee_entity_data(),
            ticks_in_hive,
            min_ticks_in_hive: BEEHIVE_MIN_OCCUPATION_TICKS_NECTARLESS,
        }
    }

    fn load(nbt: NbtCompoundView<'_, '_>) -> Self {
        let entity_data = nbt
            .compound("entity_data")
            .map_or_else(default_bee_entity_data, |entity_data| {
                entity_data.to_owned()
            });
        let ticks_in_hive = nbt.int("ticks_in_hive").unwrap_or(0);
        let min_ticks_in_hive = nbt
            .int("min_ticks_in_hive")
            .unwrap_or(BEEHIVE_MIN_OCCUPATION_TICKS_NECTARLESS);

        Self {
            entity_data,
            ticks_in_hive,
            min_ticks_in_hive,
        }
    }

    fn save(&self) -> NbtCompound {
        let mut nbt = NbtCompound::new();
        nbt.insert("entity_data", self.entity_data.clone());
        nbt.insert("ticks_in_hive", self.ticks_in_hive);
        nbt.insert("min_ticks_in_hive", self.min_ticks_in_hive);
        nbt
    }

    fn to_component(&self) -> io::Result<BeehiveOccupant> {
        let entity_data =
            EntityData::from_owned_nbt(&NbtTag::Compound(self.entity_data.clone()))
                .ok_or_else(|| io::Error::other("invalid beehive occupant entity data"))?;
        Ok(BeehiveOccupant::new(
            entity_data,
            self.ticks_in_hive,
            self.min_ticks_in_hive,
        ))
    }
}

fn default_bee_entity_data() -> NbtCompound {
    let mut entity_data = NbtCompound::new();
    entity_data.insert("id", vanilla_entities::BEE.key.to_string());
    entity_data
}

struct BeehiveState {
    stored: Vec<BeeOccupant>,
}

impl BeehiveState {
    fn push_occupant(&mut self, occupant: BeeOccupant) -> bool {
        if self.stored.len() >= BEEHIVE_MAX_OCCUPANTS {
            return false;
        }

        self.stored.push(occupant);
        true
    }
}

/// Beehive and bee nest block entity.
///
/// Currently stores and persists occupants for worldgen bee nests. Full vanilla
/// occupant ticking/release is blocked on bee entity support.
pub struct BeehiveBlockEntity {
    base: BlockEntityBase,
    state: SyncMutex<BeehiveState>,
}

// SAFETY: This key is owned by Steel and uniquely identifies `BeehiveBlockEntity`.
unsafe impl DowncastType for BeehiveBlockEntity {
    const TYPE_KEY: DowncastTypeKey = DowncastTypeKey::new("steel:block_entity/beehive");
}

impl BeehiveBlockEntity {
    /// Creates a new beehive block entity.
    #[must_use]
    pub fn new(level: Weak<World>, pos: BlockPos, state: BlockStateId) -> Self {
        Self {
            base: BlockEntityBase::new(&vanilla_block_entity_types::BEEHIVE, level, pos, state),
            state: SyncMutex::new(BeehiveState { stored: Vec::new() }),
        }
    }

    /// Stores a vanilla worldgen bee occupant.
    ///
    /// Mirrors `BeehiveBlockEntity.Occupant.create(ticksInHive)`.
    pub fn store_worldgen_bee(&self, ticks_in_hive: i32) {
        let stored = {
            self.state
                .lock()
                .push_occupant(BeeOccupant::worldgen(ticks_in_hive))
        };
        if stored {
            BlockEntity::set_changed(self);
        }
    }

    /// Returns the number of stored occupants.
    #[must_use]
    pub fn occupant_count(&self) -> usize {
        self.state.lock().stored.len()
    }

    /// Returns whether the hive currently stores no occupants.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.lock().stored.is_empty()
    }
}

impl BlockEntity for BeehiveBlockEntity {
    fn base(&self) -> &BlockEntityBase {
        &self.base
    }

    fn load_additional(&self, nbt: &BorrowedNbtCompound<'_>) {
        let nbt: NbtCompoundView<'_, '_> = nbt.into();
        let mut stored = Vec::new();

        if let Some(bees) = nbt.list("bees")
            && let Some(compounds) = bees.compounds()
        {
            for compound in compounds {
                if stored.len() >= BEEHIVE_MAX_OCCUPANTS {
                    break;
                }
                stored.push(BeeOccupant::load(compound));
            }
        }

        self.state.lock().stored = stored;
    }

    fn save_additional(&self, nbt: &mut NbtCompound) {
        let bees = self
            .state
            .lock()
            .stored
            .iter()
            .map(BeeOccupant::save)
            .collect::<Vec<_>>();
        nbt.insert("bees", NbtList::Compound(bees));
    }

    fn collect_implicit_components(&self, components: &mut DataComponentMap) -> io::Result<()> {
        let bees = self
            .state
            .lock()
            .stored
            .iter()
            .map(BeeOccupant::to_component)
            .collect::<io::Result<Vec<_>>>()?;
        components.set(BEES, Some(Bees::new(bees)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Weak;

    use steel_registry::{
        data_components::vanilla_components::BEES, test_support::init_test_registry, vanilla_blocks,
    };
    use steel_utils::BlockPos;

    use super::*;

    #[test]
    fn component_snapshot_contains_live_beehive_occupants() {
        init_test_registry();
        let block_entity = BeehiveBlockEntity::new(
            Weak::new(),
            BlockPos::ZERO,
            vanilla_blocks::BEEHIVE.default_state(),
        );
        block_entity.store_worldgen_bee(37);

        let components = block_entity
            .collect_components()
            .expect("valid bee occupants should collect");
        let bees = components
            .get_ref(BEES)
            .expect("beehive snapshot should contain bees");

        assert_eq!(bees.bees().len(), 1);
        assert_eq!(bees.bees()[0].ticks_in_hive(), 37);
        assert_eq!(
            bees.bees()[0].min_ticks_in_hive(),
            BEEHIVE_MIN_OCCUPATION_TICKS_NECTARLESS,
        );
    }
}
