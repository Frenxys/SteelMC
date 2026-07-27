//! Dedicated worker pool for synchronous intra-tick gameplay computation.

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// Runs bounded gameplay computation that must complete within the current tick phase.
///
/// The pool deliberately exposes neither detached work nor its underlying Rayon pool. Callers
/// enter it synchronously through [`crate::world::World`], keeping tick ordering explicit.
pub struct GameplayComputePool {
    inner: ThreadPool,
}

impl GameplayComputePool {
    /// Creates a gameplay compute pool with the requested number of workers.
    ///
    /// A zero count is clamped to one. Server configuration resolves its automatic `0` value
    /// before constructing the pool.
    ///
    /// # Errors
    ///
    /// Returns an error when Rayon cannot create the worker threads.
    pub fn new(worker_threads: usize) -> Result<Self, ThreadPoolBuildError> {
        ThreadPoolBuilder::new()
            .num_threads(worker_threads.max(1))
            .thread_name(|index| format!("rayon-gameplay-{index}"))
            .build()
            .map(|inner| Self { inner })
    }

    pub(crate) fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.inner.install(operation)
    }

    pub(crate) fn worker_threads(&self) -> usize {
        self.inner.current_num_threads()
    }
}

pub(crate) fn gameplay_compute_threads_for_available(
    configured_threads: Option<usize>,
    available_threads: usize,
) -> usize {
    let available_threads = available_threads.max(1);
    if let Some(configured_threads) = configured_threads.filter(|&threads| threads > 0) {
        return configured_threads.min(available_threads);
    }

    ((available_threads / 2).max(2)).min(available_threads)
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::{GameplayComputePool, gameplay_compute_threads_for_available};

    #[test]
    fn configured_worker_count_is_positive_and_capped() {
        assert_eq!(gameplay_compute_threads_for_available(Some(1), 8), 1);
        assert_eq!(gameplay_compute_threads_for_available(Some(6), 8), 6);
        assert_eq!(gameplay_compute_threads_for_available(Some(16), 8), 8);
        assert_eq!(gameplay_compute_threads_for_available(Some(1), 0), 1);
    }

    #[test]
    fn automatic_worker_count_uses_half_the_available_threads() {
        assert_eq!(gameplay_compute_threads_for_available(None, 32), 16);
        assert_eq!(gameplay_compute_threads_for_available(Some(0), 8), 4);
        assert_eq!(gameplay_compute_threads_for_available(None, 3), 2);
        assert_eq!(gameplay_compute_threads_for_available(None, 2), 2);
        assert_eq!(gameplay_compute_threads_for_available(None, 1), 1);
        assert_eq!(gameplay_compute_threads_for_available(None, 0), 1);
    }

    #[test]
    fn pool_installs_work_on_named_workers() {
        let pool = GameplayComputePool::new(2).expect("gameplay compute pool should start");

        let (thread_count, thread_name) = pool.install(|| {
            (
                rayon::current_num_threads(),
                thread::current().name().map(str::to_owned),
            )
        });

        assert_eq!(thread_count, 2);
        assert!(
            thread_name.is_some_and(|name| name.starts_with("rayon-gameplay-")),
            "operation should run on a named gameplay worker"
        );
    }
}
