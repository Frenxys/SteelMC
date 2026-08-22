//! Beacon menu.
//!
//! A single payment slot plus the player inventory. Three data slots mirror the
//! beacon's pyramid level and configured effects, matching vanilla's
//! `DATA_LEVELS`, `DATA_PRIMARY`, and `DATA_SECONDARY`.

use std::sync::Arc;

use steel_registry::{
    REGISTRY, RegistryEntry, TaggedRegistryExt, item_stack::ItemStack, mob_effect::MobEffectRef,
    vanilla_item_tags, vanilla_menu_types,
};
use steel_utils::locks::SyncMutex;

use crate::block_entity::entities::{BEACON_SLOTS, BeaconState};
use crate::inventory::prelude::*;
use crate::player::player_inventory::PlayerInventory;

/// Builds the beacon menu.
#[must_use]
pub fn beacon(
    inventory: Shared<PlayerInventory>,
    container_id: u8,
    container: impl Into<ContainerRef>,
    state: Arc<SyncMutex<BeaconState>>,
) -> Menu {
    let container = container.into();

    let mut builder = MenuBuilder::new(&vanilla_menu_types::BEACON, container_id);
    // Vanilla gates the payment slot with `ItemTags.BEACON_PAYMENT_ITEMS`.
    let payment = builder.section_with(
        &container,
        BEACON_SLOTS,
        SectionKind::restricted(|_, stack| {
            REGISTRY.items.is_in_tag(
                stack.item(),
                &vanilla_item_tags::ItemTag::BEACON_PAYMENT_ITEMS,
            )
        }),
    );
    let player = builder.player_inventory(&inventory);
    let levels = builder.data_slot(0);
    let primary = builder.data_slot(0);
    let secondary = builder.data_slot(0);

    builder.route(payment, player.all(), FillDirection::Forward);
    builder.route(player.all(), payment, FillDirection::Forward);

    builder.build(BeaconKind {
        container,
        state,
        levels,
        primary,
        secondary,
    })
}

/// Encodes an effect as vanilla's `BeaconMenu.encodeEffect`: `0` for none, or
/// the effect's registry id plus one.
fn encode_effect(effect: Option<MobEffectRef>) -> i16 {
    effect.map_or(0, |effect| effect.id() as i16 + 1)
}

/// Per-menu beacon state: the backing container and data-slot handles.
pub struct BeaconKind {
    container: ContainerRef,
    state: Arc<SyncMutex<BeaconState>>,
    levels: DataSlot,
    primary: DataSlot,
    secondary: DataSlot,
}

// SAFETY: This Steel-owned key uniquely identifies the concrete menu kind
// within the process.
unsafe impl steel_utils::DowncastType for BeaconKind {
    const TYPE_KEY: steel_utils::DowncastTypeKey =
        steel_utils::DowncastTypeKey::new("steel:menu/beacon");
}

impl BeaconKind {
    /// Pushes the beacon's current levels and effects into the data slots.
    fn sync_data_slots(&self, behavior: &mut MenuBehavior) {
        let state = self.state.lock();
        self.levels.set(behavior, state.levels as i16);
        self.primary
            .set(behavior, encode_effect(state.primary_power));
        self.secondary
            .set(behavior, encode_effect(state.secondary_power));
    }
}

impl MenuKind for BeaconKind {
    /// Returns true while the beacon block entity is still valid for the player.
    fn still_valid(&self, _behavior: &MenuBehavior, player: &Player) -> bool {
        self.container.still_valid(player)
    }

    fn on_open(
        &mut self,
        behavior: &mut MenuBehavior,
        _guard: &mut ContainerLockGuard,
        _player: &Player,
    ) {
        self.sync_data_slots(behavior);
    }

    fn on_tick(
        &mut self,
        behavior: &mut MenuBehavior,
        _guard: &mut ContainerLockGuard,
        _player: &Player,
    ) {
        self.sync_data_slots(behavior);
    }

    fn on_update_effects(
        &mut self,
        behavior: &mut MenuBehavior,
        guard: &mut ContainerLockGuard,
        primary: Option<MobEffectRef>,
        secondary: Option<MobEffectRef>,
    ) {
        let Some(payment) = guard.get(self.container.container_id()) else {
            return;
        };
        if payment.get_item(0).is_empty() {
            return;
        }

        {
            let mut state = self.state.lock();
            state.primary_power = BeaconState::filter_effect(primary);
            state.secondary_power = BeaconState::filter_effect(secondary);
        }

        // Vanilla removes the payment through `Slot.remove(1)` and marks the
        // block entity changed; route through the guard so the owner is notified.
        guard.set_item(self.container.container_id(), 0, ItemStack::empty());
        self.sync_data_slots(behavior);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use steel_registry::{
        init_vanilla_registry, item_stack::ItemStack, vanilla_blocks, vanilla_items,
        vanilla_mob_effects,
    };
    use steel_utils::types::UpdateFlags;
    use steel_utils::{BlockPos, ChunkPos, Downcast as _};

    use crate::behavior::init_behaviors;
    use crate::block_entity::{entities::BeaconBlockEntity, init_block_entities};
    use crate::inventory::click::{Click, MouseButton};
    use crate::inventory::lock::ContainerRef;
    use crate::test_support::{TestPlayerBuilder, fresh_test_world, insert_ready_full_chunk};

    use super::beacon;

    #[test]
    fn beacon_menu_rejects_other_items_and_consumes_iron_on_effect_selection() {
        init_vanilla_registry();
        init_behaviors();
        init_block_entities();
        let world = fresh_test_world("beacon_menu_iron_payment");
        insert_ready_full_chunk(&world, ChunkPos::new(0, 0));
        let player = TestPlayerBuilder::new(Arc::clone(&world), "BeaconTester", 1).build();

        let pos = BlockPos::new(8, 64, 8);
        assert!(world.set_block(
            pos,
            vanilla_blocks::BEACON.default_state(),
            UpdateFlags::UPDATE_NONE,
        ));

        let block_entity = world.get_block_entity(pos).expect("beacon block entity");
        let beacon_entity = block_entity
            .downcast_ref::<BeaconBlockEntity>()
            .expect("beacon block entity type");
        let container_ref =
            ContainerRef::from_block_entity(block_entity.clone()).expect("beacon container");

        let mut menu = beacon(
            player.inventory.clone(),
            1,
            container_ref.clone(),
            beacon_entity.state(),
        );

        // Non-payment items stay on the cursor.
        *menu.behavior_mut().carried_mut() = ItemStack::new(&vanilla_items::DIRT);
        menu.clicked(
            Click::Pickup {
                slot: 0,
                button: MouseButton::Left,
            },
            &player,
        );
        assert!(menu.behavior().carried().is(&vanilla_items::DIRT));
        assert!({
            let guard = menu.behavior().lock_all_containers();
            guard
                .get(container_ref.container_id())
                .expect("payment container locked")
                .get_item(0)
                .is_empty()
        });

        // Iron is accepted, capped at one item like vanilla's payment slot.
        *menu.behavior_mut().carried_mut() = ItemStack::with_count(&vanilla_items::IRON_INGOT, 2);
        menu.clicked(
            Click::Pickup {
                slot: 0,
                button: MouseButton::Left,
            },
            &player,
        );
        assert_eq!(menu.behavior().carried().count(), 1);
        assert!({
            let guard = menu.behavior().lock_all_containers();
            guard
                .get(container_ref.container_id())
                .expect("payment container locked")
                .get_item(0)
                .is(&vanilla_items::IRON_INGOT)
        });

        // Selecting an effect consumes the payment and stores the selection.
        menu.update_effects(
            Some(vanilla_mob_effects::STRENGTH),
            None,
            &player.connection,
        );
        assert!({
            let guard = menu.behavior().lock_all_containers();
            guard
                .get(container_ref.container_id())
                .expect("payment container locked")
                .get_item(0)
                .is_empty()
        });
        let state = beacon_entity.state();
        let state = state.lock();
        assert_eq!(
            state.primary_power.map(|effect| effect.key.clone()),
            Some(vanilla_mob_effects::STRENGTH.key.clone())
        );
        assert!(state.secondary_power.is_none());
    }
}
