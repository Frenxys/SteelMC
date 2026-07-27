use std::{
    array, hint,
    sync::atomic::{AtomicU64, Ordering},
};

use glam::DVec3;
use steel_utils::WorldAabb;

const SPATIAL_VALUE_COUNT: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct EntitySpatialState {
    position: DVec3,
    bounding_box: WorldAabb,
}

impl EntitySpatialState {
    pub(super) const fn new(position: DVec3, bounding_box: WorldAabb) -> Self {
        Self {
            position,
            bounding_box,
        }
    }

    pub(super) const fn position(self) -> DVec3 {
        self.position
    }

    pub(super) const fn bounding_box(self) -> WorldAabb {
        self.bounding_box
    }

    const fn to_bits(self) -> [u64; SPATIAL_VALUE_COUNT] {
        [
            self.position.x.to_bits(),
            self.position.y.to_bits(),
            self.position.z.to_bits(),
            self.bounding_box.min_x().to_bits(),
            self.bounding_box.min_y().to_bits(),
            self.bounding_box.min_z().to_bits(),
            self.bounding_box.max_x().to_bits(),
            self.bounding_box.max_y().to_bits(),
            self.bounding_box.max_z().to_bits(),
        ]
    }

    const fn from_bits(bits: [u64; SPATIAL_VALUE_COUNT]) -> Self {
        Self::new(
            DVec3::new(
                f64::from_bits(bits[0]),
                f64::from_bits(bits[1]),
                f64::from_bits(bits[2]),
            ),
            WorldAabb::new(
                f64::from_bits(bits[3]),
                f64::from_bits(bits[4]),
                f64::from_bits(bits[5]),
                f64::from_bits(bits[6]),
                f64::from_bits(bits[7]),
                f64::from_bits(bits[8]),
            ),
        )
    }
}

/// One coherently published entity position and bounding box.
///
/// The revision lets world indexes discard delayed callbacks without
/// regressing to an older spatial state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntitySpatialUpdate {
    revision: u64,
    state: EntitySpatialState,
}

impl EntitySpatialUpdate {
    const fn new(revision: u64, state: EntitySpatialState) -> Self {
        Self { revision, state }
    }

    /// Returns the monotonic spatial publication revision.
    #[must_use]
    pub const fn revision(self) -> u64 {
        self.revision
    }

    /// Returns the coherently published position.
    #[must_use]
    pub const fn position(self) -> DVec3 {
        self.state.position()
    }

    /// Returns the coherently published bounding box.
    #[must_use]
    pub const fn bounding_box(self) -> WorldAabb {
        self.state.bounding_box()
    }
}

/// Lock-free position and bounding-box publication for entity spatial queries.
///
/// Writers are serialized by `EntityBase::state`. Sequentially consistent
/// atomics keep every payload value between the same odd/even revision pair,
/// so readers cannot observe a box assembled from two movement commits.
pub(super) struct EntitySpatialSnapshot {
    revision: AtomicU64,
    values: [AtomicU64; SPATIAL_VALUE_COUNT],
}

impl EntitySpatialSnapshot {
    pub(super) fn new(state: EntitySpatialState) -> Self {
        Self {
            revision: AtomicU64::new(0),
            values: state.to_bits().map(AtomicU64::new),
        }
    }

    pub(super) fn load(&self) -> EntitySpatialUpdate {
        let (revision, state) = self.load_consistent(|| {
            let values = array::from_fn(|index| self.values[index].load(Ordering::SeqCst));
            EntitySpatialState::from_bits(values)
        });
        EntitySpatialUpdate::new(revision, state)
    }

    pub(super) fn position(&self) -> DVec3 {
        self.load_consistent(|| {
            DVec3::new(
                f64::from_bits(self.values[0].load(Ordering::SeqCst)),
                f64::from_bits(self.values[1].load(Ordering::SeqCst)),
                f64::from_bits(self.values[2].load(Ordering::SeqCst)),
            )
        })
        .1
    }

    pub(super) fn bounding_box(&self) -> WorldAabb {
        self.load_consistent(|| {
            WorldAabb::new(
                f64::from_bits(self.values[3].load(Ordering::SeqCst)),
                f64::from_bits(self.values[4].load(Ordering::SeqCst)),
                f64::from_bits(self.values[5].load(Ordering::SeqCst)),
                f64::from_bits(self.values[6].load(Ordering::SeqCst)),
                f64::from_bits(self.values[7].load(Ordering::SeqCst)),
                f64::from_bits(self.values[8].load(Ordering::SeqCst)),
            )
        })
        .1
    }

    fn load_consistent<T>(&self, load_value: impl Fn() -> T) -> (u64, T) {
        loop {
            let revision = self.revision.load(Ordering::SeqCst);
            if revision & 1 != 0 {
                hint::spin_loop();
                continue;
            }

            let value = load_value();
            if self.revision.load(Ordering::SeqCst) == revision {
                return (revision, value);
            }

            hint::spin_loop();
        }
    }

    pub(super) fn publish(&self, state: EntitySpatialState) -> EntitySpatialUpdate {
        let previous_revision = self.revision.load(Ordering::SeqCst);
        assert!(
            previous_revision & 1 == 0 && previous_revision <= u64::MAX - 2,
            "entity spatial publication revision exhausted or had an active writer"
        );
        self.revision.store(previous_revision + 1, Ordering::SeqCst);
        for (target, value) in self.values.iter().zip(state.to_bits()) {
            target.store(value, Ordering::SeqCst);
        }
        let revision = previous_revision + 2;
        self.revision.store(revision, Ordering::SeqCst);
        EntitySpatialUpdate::new(revision, state)
    }
}
