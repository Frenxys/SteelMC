use glam::IVec3;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use steel_utils::WorldAabb;

// Performance granularity only: exact AABB filtering and the fallbacks below
// keep query behavior independent of this value. Reprofile 1/2/4-block cells
// if candidate collection remains hot.
const CELL_SIZE: f64 = 2.0;
// Vanilla's ordinary four-block size limit occupies at most 27 two-block cells;
// larger custom boxes use the exact oversized-entity lane.
const MAX_ENTITY_CELLS: u64 = 64;
// Bounds empty-cell probes for pathological volumes before exact section traversal.
const MAX_QUERY_CELLS: u64 = 32_768;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct SpatialCell(IVec3);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexedEntity {
    entity_id: i32,
    minimum_axes: MinimumAxes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MinimumAxes(u8);

impl MinimumAxes {
    const X: u8 = 1 << 0;
    const Y: u8 = 1 << 1;
    const Z: u8 = 1 << 2;

    const fn for_cell(cell: SpatialCell, minimum: SpatialCell) -> Self {
        let mut axes = 0;
        if cell.0.x == minimum.0.x {
            axes |= Self::X;
        }
        if cell.0.y == minimum.0.y {
            axes |= Self::Y;
        }
        if cell.0.z == minimum.0.z {
            axes |= Self::Z;
        }
        Self(axes)
    }

    const fn is_first_overlap_cell(self, cell: SpatialCell, query_minimum: SpatialCell) -> bool {
        (cell.0.x == query_minimum.0.x || self.0 & Self::X != 0)
            && (cell.0.y == query_minimum.0.y || self.0 & Self::Y != 0)
            && (cell.0.z == query_minimum.0.z || self.0 & Self::Z != 0)
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub(super) enum EntitySpatialMembership {
    #[default]
    Unindexed,
    Cells(SmallVec<[SpatialCell; 8]>),
    Oversized,
}

impl EntitySpatialMembership {
    pub(super) fn for_bounding_box(bounding_box: WorldAabb) -> Self {
        let Some(bounds) = SpatialCellBounds::for_aabb(bounding_box) else {
            return Self::Oversized;
        };
        let Some(bounds) = bounds else {
            return Self::Unindexed;
        };
        if bounds.cell_count() > MAX_ENTITY_CELLS {
            return Self::Oversized;
        }

        let mut cells = SmallVec::new();
        bounds.visit(|cell| cells.push(cell));
        Self::Cells(cells)
    }
}

#[derive(Default)]
pub(super) struct EntitySpatialIndex {
    cells: FxHashMap<SpatialCell, SmallVec<[IndexedEntity; 8]>>,
    oversized: FxHashSet<i32>,
}

impl EntitySpatialIndex {
    pub(super) fn insert(&mut self, entity_id: i32, membership: &EntitySpatialMembership) {
        match membership {
            EntitySpatialMembership::Unindexed => {}
            EntitySpatialMembership::Cells(cells) => {
                let Some(minimum) = cells.first().copied() else {
                    return;
                };
                for cell in cells {
                    let indexed_entities = self.cells.entry(*cell).or_default();
                    if !indexed_entities
                        .iter()
                        .any(|indexed| indexed.entity_id == entity_id)
                    {
                        indexed_entities.push(IndexedEntity {
                            entity_id,
                            minimum_axes: MinimumAxes::for_cell(*cell, minimum),
                        });
                    }
                }
            }
            EntitySpatialMembership::Oversized => {
                self.oversized.insert(entity_id);
            }
        }
    }

    pub(super) fn remove(&mut self, entity_id: i32, membership: &EntitySpatialMembership) {
        match membership {
            EntitySpatialMembership::Unindexed => {}
            EntitySpatialMembership::Cells(cells) => {
                for cell in cells {
                    let remove_cell = if let Some(indexed_entities) = self.cells.get_mut(cell) {
                        if let Some(index) = indexed_entities
                            .iter()
                            .position(|indexed| indexed.entity_id == entity_id)
                        {
                            indexed_entities.remove(index);
                        }
                        indexed_entities.is_empty()
                    } else {
                        false
                    };
                    if remove_cell {
                        self.cells.remove(cell);
                    }
                }
            }
            EntitySpatialMembership::Oversized => {
                self.oversized.remove(&entity_id);
            }
        }
    }

    /// Visits every fine-grained candidate exactly once.
    ///
    /// Returns `false` when the caller must use the exact section fallback.
    pub(super) fn try_visit_candidate_ids(
        &self,
        query: WorldAabb,
        mut visitor: impl FnMut(i32),
    ) -> bool {
        let Some(bounds) = SpatialCellBounds::for_aabb(query) else {
            return false;
        };
        let Some(bounds) = bounds else {
            return true;
        };
        if bounds.cell_count() > MAX_QUERY_CELLS {
            return false;
        }

        let query_minimum = SpatialCell(bounds.minimum);
        bounds.visit(|cell| {
            if let Some(indexed_entities) = self.cells.get(&cell) {
                for indexed in indexed_entities {
                    if indexed
                        .minimum_axes
                        .is_first_overlap_cell(cell, query_minimum)
                    {
                        visitor(indexed.entity_id);
                    }
                }
            }
        });
        for entity_id in &self.oversized {
            visitor(*entity_id);
        }
        true
    }
}

struct SpatialCellBounds {
    minimum: IVec3,
    maximum: IVec3,
}

impl SpatialCellBounds {
    /// Returns `None` for non-finite/out-of-range boxes and `Some(None)` for empty boxes.
    #[expect(
        clippy::option_option,
        reason = "the outer option selects exact fallback while the inner option represents an empty box"
    )]
    fn for_aabb(aabb: WorldAabb) -> Option<Option<Self>> {
        if aabb.is_empty() {
            return Some(None);
        }

        let minimum = IVec3::new(
            cell_coordinate(aabb.min_x())?,
            cell_coordinate(aabb.min_y())?,
            cell_coordinate(aabb.min_z())?,
        );
        let maximum = IVec3::new(
            exclusive_maximum_cell_coordinate(aabb.max_x())?,
            exclusive_maximum_cell_coordinate(aabb.max_y())?,
            exclusive_maximum_cell_coordinate(aabb.max_z())?,
        );
        Some(Some(Self { minimum, maximum }))
    }

    fn cell_count(&self) -> u64 {
        let x = i64::from(self.maximum.x) - i64::from(self.minimum.x) + 1;
        let y = i64::from(self.maximum.y) - i64::from(self.minimum.y) + 1;
        let z = i64::from(self.maximum.z) - i64::from(self.minimum.z) + 1;
        u64::try_from(x)
            .ok()
            .and_then(|x| u64::try_from(y).ok().and_then(|y| x.checked_mul(y)))
            .and_then(|xy| u64::try_from(z).ok().and_then(|z| xy.checked_mul(z)))
            .unwrap_or(u64::MAX)
    }

    fn visit(&self, mut visitor: impl FnMut(SpatialCell)) {
        for x in self.minimum.x..=self.maximum.x {
            for y in self.minimum.y..=self.maximum.y {
                for z in self.minimum.z..=self.maximum.z {
                    visitor(SpatialCell(IVec3::new(x, y, z)));
                }
            }
        }
    }
}

fn cell_coordinate(coordinate: f64) -> Option<i32> {
    if !coordinate.is_finite() {
        return None;
    }
    let cell = (coordinate.floor() / CELL_SIZE).floor();
    if cell < f64::from(i32::MIN) || cell > f64::from(i32::MAX) {
        return None;
    }
    Some(cell as i32)
}

fn exclusive_maximum_cell_coordinate(coordinate: f64) -> Option<i32> {
    if !coordinate.is_finite() {
        return None;
    }

    let mut cell = (coordinate.floor() / CELL_SIZE).floor();
    if coordinate % CELL_SIZE == 0.0 {
        cell -= 1.0;
    }
    if cell < f64::from(i32::MIN) || cell > f64::from(i32::MAX) {
        return None;
    }
    Some(cell as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate_ids(index: &EntitySpatialIndex, query: WorldAabb) -> Option<Vec<i32>> {
        let mut candidates = Vec::new();
        index
            .try_visit_candidate_ids(query, |entity_id| candidates.push(entity_id))
            .then_some(candidates)
    }

    #[test]
    fn exact_exclusive_maximum_selects_the_preceding_cell() {
        for (maximum, expected_cell) in [(2.0, 0), (0.0, -1), (-0.0, -1), (-2.0, -2)] {
            let minimum = maximum - CELL_SIZE;
            let membership = EntitySpatialMembership::for_bounding_box(WorldAabb::new(
                minimum, minimum, minimum, maximum, maximum, maximum,
            ));
            let EntitySpatialMembership::Cells(cells) = membership else {
                panic!("finite non-empty box should have indexed cells");
            };

            assert_eq!(
                cells.as_slice(),
                &[SpatialCell(IVec3::splat(expected_cell))],
                "exclusive maximum {maximum:?}"
            );
        }
    }

    #[test]
    fn canonical_overlap_cell_is_unique_for_small_signed_bounds() {
        let mut axis_bounds = Vec::new();
        for minimum in -1..=1 {
            for maximum in minimum..=1 {
                axis_bounds.push((minimum, maximum));
            }
        }

        let mut bounds = Vec::new();
        for &(minimum_x, maximum_x) in &axis_bounds {
            for &(minimum_y, maximum_y) in &axis_bounds {
                for &(minimum_z, maximum_z) in &axis_bounds {
                    bounds.push(SpatialCellBounds {
                        minimum: IVec3::new(minimum_x, minimum_y, minimum_z),
                        maximum: IVec3::new(maximum_x, maximum_y, maximum_z),
                    });
                }
            }
        }

        for entity in &bounds {
            for query in &bounds {
                let overlap_minimum = entity.minimum.max(query.minimum);
                let overlap_maximum = entity.maximum.min(query.maximum);
                let overlaps = overlap_minimum.x <= overlap_maximum.x
                    && overlap_minimum.y <= overlap_maximum.y
                    && overlap_minimum.z <= overlap_maximum.z;
                let mut canonical_cells = 0;
                if overlaps {
                    SpatialCellBounds {
                        minimum: overlap_minimum,
                        maximum: overlap_maximum,
                    }
                    .visit(|cell| {
                        let minimum_axes = MinimumAxes::for_cell(cell, SpatialCell(entity.minimum));
                        if minimum_axes.is_first_overlap_cell(cell, SpatialCell(query.minimum)) {
                            canonical_cells += 1;
                        }
                    });
                }

                assert_eq!(
                    canonical_cells,
                    i32::from(overlaps),
                    "entity {:?}..={:?}, query {:?}..={:?}",
                    entity.minimum,
                    entity.maximum,
                    query.minimum,
                    query.maximum
                );
            }
        }
    }

    #[test]
    fn candidate_collection_deduplicates_entities_spanning_cells() {
        let membership = EntitySpatialMembership::for_bounding_box(WorldAabb::new(
            1.0, 63.0, 1.0, 5.0, 67.0, 5.0,
        ));
        let mut index = EntitySpatialIndex::default();
        index.insert(7, &membership);

        let Some(candidates) =
            candidate_ids(&index, WorldAabb::new(1.5, 63.5, 1.5, 4.5, 66.5, 4.5))
        else {
            panic!("small finite query should use the spatial index");
        };

        assert_eq!(candidates.as_slice(), &[7]);
    }

    #[test]
    fn candidate_collection_emits_from_a_non_minimum_overlap_cell() {
        let membership = EntitySpatialMembership::for_bounding_box(WorldAabb::new(
            1.0, 63.0, 1.0, 5.0, 67.0, 5.0,
        ));
        let mut index = EntitySpatialIndex::default();
        index.insert(7, &membership);

        let Some(candidates) =
            candidate_ids(&index, WorldAabb::new(4.1, 66.1, 4.1, 4.9, 66.9, 4.9))
        else {
            panic!("small finite query should use the spatial index");
        };

        assert_eq!(candidates.as_slice(), &[7]);
    }

    #[test]
    fn dense_candidate_collection_emits_every_entity_once() {
        let membership = EntitySpatialMembership::for_bounding_box(WorldAabb::new(
            1.0, 63.0, 1.0, 5.0, 67.0, 5.0,
        ));
        let mut index = EntitySpatialIndex::default();
        for entity_id in 1..=64 {
            index.insert(entity_id, &membership);
        }

        let Some(candidates) =
            candidate_ids(&index, WorldAabb::new(1.5, 63.5, 1.5, 4.5, 66.5, 4.5))
        else {
            panic!("small finite query should use the spatial index");
        };

        assert_eq!(candidates, (1..=64).collect::<Vec<_>>());
    }

    #[test]
    fn large_query_selects_the_section_traversal_fallback() {
        let index = EntitySpatialIndex::default();
        let mut visited = false;

        assert!(!index.try_visit_candidate_ids(
            WorldAabb::new(0.0, 0.0, 0.0, 66.0, 66.0, 66.0),
            |_| {
                visited = true;
            },
        ));
        assert!(!visited);
    }
}
