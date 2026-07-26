//! Shared storage and persistence for Vanilla base container block entities.

use std::{io, mem};

use simdnbt::borrow::NbtCompound as NbtCompoundView;
use simdnbt::owned::{NbtCompound, NbtList, NbtTag};
use simdnbt::{FromNbtTag as _, ToNbtTag as _};
use steel_registry::{
    data_components::{
        DataComponentMap, ItemContainerContents,
        vanilla_components::{CONTAINER, CUSTOM_NAME, LOCK},
    },
    item_predicate::LockCode,
    item_stack::ItemStack,
    item_stack_template::ItemStackTemplate,
};
use text_components::TextComponent;

/// Inventory, custom-name, and lock data shared by container block entities.
///
pub(crate) struct BaseContainer {
    items: Vec<ItemStack>,
    custom_name: Option<TextComponent>,
    lock: Option<LockCode>,
}

impl BaseContainer {
    #[must_use]
    pub(crate) fn new(size: usize) -> Self {
        Self {
            items: vec![ItemStack::empty(); size],
            custom_name: None,
            lock: None,
        }
    }

    pub(crate) fn load_metadata(&mut self, nbt: &NbtCompoundView<'_, '_>) {
        self.custom_name = nbt
            .get("CustomName")
            .and_then(|tag| TextComponent::from_nbt(&tag.to_owned()));
        self.lock = nbt.get("lock").and_then(LockCode::from_nbt_tag);
    }

    pub(crate) fn load_items(&mut self, nbt: &NbtCompoundView<'_, '_>) {
        self.items = Self::items_from_nbt(nbt, self.items.len());
    }

    pub(crate) fn items_from_nbt(nbt: &NbtCompoundView<'_, '_>, size: usize) -> Vec<ItemStack> {
        let mut result = vec![ItemStack::empty(); size];
        let Some(items) = nbt.list("Items").and_then(|items| items.compounds()) else {
            return result;
        };
        for compound in items {
            let Some(slot) = compound.byte("Slot").map(|slot| slot as u8 as usize) else {
                continue;
            };
            if slot < result.len()
                && let Some(item) = ItemStack::from_borrowed_compound(&compound)
            {
                result[slot] = item;
            }
        }
        result
    }

    pub(crate) fn save_metadata(&self, nbt: &mut NbtCompound) {
        if let Some(custom_name) = &self.custom_name {
            nbt.insert("CustomName", custom_name.to_nbt_tag());
        }
        if let Some(lock) = &self.lock {
            nbt.insert("lock", lock.to_nbt_tag_ref());
        }
    }

    pub(crate) fn save_items(&self, nbt: &mut NbtCompound) {
        Self::save_item_slice(nbt, &self.items);
    }

    pub(crate) fn collect_implicit_components(
        &self,
        components: &mut DataComponentMap,
    ) -> io::Result<()> {
        components.set(CUSTOM_NAME, self.custom_name.clone());
        components.set(
            LOCK,
            self.lock
                .as_ref()
                .filter(|lock| *lock != &LockCode::NO_LOCK)
                .cloned(),
        );
        let Some(last_non_empty_slot) = self.items.iter().rposition(|item| !item.is_empty()) else {
            components.set(CONTAINER, Some(ItemContainerContents::empty()));
            return Ok(());
        };
        let items = self.items[..=last_non_empty_slot]
            .iter()
            .map(|item| {
                if item.is_empty() {
                    Ok(None)
                } else {
                    ItemStackTemplate::from_stack(item).map(Some)
                }
            })
            .collect::<io::Result<Vec<_>>>()?;
        components.set(CONTAINER, Some(ItemContainerContents::new(items)?));
        Ok(())
    }

    pub(crate) fn save_item_slice(nbt: &mut NbtCompound, item_slice: &[ItemStack]) {
        let mut items = Vec::new();
        for (slot, item) in item_slice.iter().enumerate() {
            if item.is_empty() {
                continue;
            }
            let NbtTag::Compound(mut item_nbt) = item.to_nbt_tag_ref() else {
                continue;
            };
            item_nbt.insert("Slot", slot as i8);
            items.push(item_nbt);
        }
        nbt.insert("Items", NbtList::Compound(items));
    }

    pub(crate) fn replace_items(&mut self, items: Vec<ItemStack>) -> Result<(), Vec<ItemStack>> {
        if items.len() != self.items.len() {
            return Err(items);
        }
        self.items = items;
        Ok(())
    }

    #[must_use]
    pub(crate) fn items(&self) -> &[ItemStack] {
        &self.items
    }

    pub(crate) fn items_mut(&mut self) -> &mut [ItemStack] {
        &mut self.items
    }

    pub(crate) fn set_item(&mut self, slot: usize, mut stack: ItemStack) {
        if slot >= self.items.len() {
            return;
        }
        let max_stack_size = 99.min(stack.max_stack_size());
        if !stack.is_empty() && stack.count() > max_stack_size {
            stack.set_count(max_stack_size);
        }
        self.items[slot] = stack;
    }

    pub(crate) fn clear_items(&mut self) {
        self.items.fill(ItemStack::empty());
    }

    /// Removes every item while retaining the fixed slot count.
    pub(crate) fn take_items(&mut self) -> Vec<ItemStack> {
        let size = self.items.len();
        mem::replace(&mut self.items, vec![ItemStack::empty(); size])
    }

    #[must_use]
    pub(crate) fn display_name(&self, default: TextComponent) -> TextComponent {
        self.custom_name.clone().unwrap_or(default)
    }

    #[must_use]
    pub(crate) const fn has_custom_name(&self) -> bool {
        self.custom_name.is_some()
    }

    #[must_use]
    pub(crate) fn has_lock(&self) -> bool {
        self.lock
            .as_ref()
            .is_some_and(|lock| lock != &LockCode::NO_LOCK)
    }
}

#[cfg(test)]
mod tests {
    use steel_registry::{
        data_components::{DataComponentMap, vanilla_components::CONTAINER},
        item_stack::ItemStack,
        test_support::init_test_registry,
        vanilla_items,
    };

    use super::BaseContainer;

    #[test]
    fn component_snapshot_trims_trailing_empty_container_slots() {
        init_test_registry();
        let mut container = BaseContainer::new(3);
        container.set_item(1, ItemStack::with_count(&vanilla_items::DIAMOND, 2));
        let mut components = DataComponentMap::new();

        container
            .collect_implicit_components(&mut components)
            .expect("valid live container items should collect");

        let contents = components
            .get_ref(CONTAINER)
            .expect("container snapshot should always include contents");
        assert!(contents.items()[0].is_none());
        let item = contents.items()[1]
            .as_ref()
            .expect("occupied slot should contain an item template");
        assert_eq!(item.item(), &*vanilla_items::DIAMOND);
        assert_eq!(item.count(), 2);
        assert_eq!(contents.items().len(), 2);

        let empty_container = BaseContainer::new(3);
        let mut empty_components = DataComponentMap::new();
        empty_container
            .collect_implicit_components(&mut empty_components)
            .expect("empty live container should collect");
        assert!(
            empty_components
                .get_ref(CONTAINER)
                .expect("empty snapshot should contain container contents")
                .items()
                .is_empty()
        );
    }
}
