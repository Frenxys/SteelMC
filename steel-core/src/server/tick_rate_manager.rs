use std::time::Instant;

/// Number of tick samples to keep for averaging (matches vanilla).
const TICK_STATS_SPAN: usize = 100;

/// Nanoseconds per millisecond.
const NANOS_PER_MS: f64 = 1_000_000.0;

/// Nanoseconds per second.
const NANOS_PER_SEC: f64 = 1_000_000_000.0;

/// Milliseconds per second.
const MS_PER_SEC: f64 = 1000.0;

/// Smoothing factor for exponential moving average (matches vanilla's 0.8).
const TICK_TIME_SMOOTHING: f32 = 0.8;

/// Report data returned when a sprint finishes.
#[derive(Debug, Clone)]
pub struct SprintReport {
    /// Ticks per second achieved during the sprint.
    pub ticks_per_second: i32,
    /// Milliseconds per tick during the sprint.
    pub ms_per_tick: f64,
}

/// Manages the server tick rate, including freezing, stepping, and sprinting.
pub struct TickRateManager {
    /// The current tick rate in ticks per second.
    pub tick_rate: f32,
    /// The number of nanoseconds per tick based on the tick rate.
    pub nanoseconds_per_tick: u64,
    /// The current server tick count, including ticks where game elements are frozen.
    pub tick_count: u64,
    /// Whether the server is currently frozen.
    is_frozen: bool,
    /// The number of ticks to run while frozen (stepping).
    frozen_ticks_to_run: i32,
    /// Whether game elements should run this tick.
    run_game_elements: bool,

    // Sprinting
    /// The number of ticks remaining to sprint.
    remaining_sprint_ticks: i64,
    /// The total number of ticks scheduled for the current sprint.
    scheduled_current_sprint_ticks: i64,
    /// The start time of the current sprint tick.
    sprint_tick_start_time: Option<Instant>,
    /// The total time spent sprinting in nanoseconds.
    sprint_time_spent: i64,
    /// Whether the server was frozen before sprinting started.
    previous_is_frozen: bool,

    // Tick time tracking (vanilla-style)
    /// Rolling buffer of the last 100 tick times in nanoseconds.
    tick_times_nanos: [u64; TICK_STATS_SPAN],
    /// Number of valid entries currently held in `tick_times_nanos`.
    tick_time_sample_count: usize,
    /// Ring position that receives the next completed tick duration.
    next_tick_time_sample_index: usize,
    /// Aggregated sum of tick times for fast average calculation.
    aggregated_tick_times_nanos: u64,
    /// Exponentially smoothed tick time in milliseconds.
    smoothed_tick_time_ms: f32,
}

impl TickRateManager {
    /// Creates a new `TickRateManager` with the default tick rate (20.0 TPS).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tick_rate: 20.0,
            nanoseconds_per_tick: 50_000_000, // 1_000_000_000 / 20
            tick_count: 0,
            is_frozen: false,
            frozen_ticks_to_run: 0,
            run_game_elements: true,
            remaining_sprint_ticks: 0,
            scheduled_current_sprint_ticks: 0,
            sprint_tick_start_time: None,
            sprint_time_spent: 0,
            previous_is_frozen: false,
            tick_times_nanos: [0; TICK_STATS_SPAN],
            tick_time_sample_count: 0,
            next_tick_time_sample_index: 0,
            aggregated_tick_times_nanos: 0,
            smoothed_tick_time_ms: 0.0,
        }
    }

    /// Sets the tick rate.
    pub fn set_tick_rate(&mut self, rate: f32) {
        self.tick_rate = rate.max(1.0);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "result is always a positive sub-second nanosecond count, fits in u64"
        )]
        {
            self.nanoseconds_per_tick = (NANOS_PER_SEC / f64::from(self.tick_rate)) as u64;
        }
    }

    /// Returns the tick rate.
    #[must_use]
    pub const fn tick_rate(&self) -> f32 {
        self.tick_rate
    }

    /// Returns milliseconds per tick based on the current tick rate.
    #[must_use]
    pub fn milliseconds_per_tick(&self) -> f32 {
        self.nanoseconds_per_tick as f32 / NANOS_PER_MS as f32
    }

    /// Sets the frozen state of the server.
    pub const fn set_frozen(&mut self, frozen: bool) {
        self.is_frozen = frozen;
    }

    /// Returns whether the server is frozen.
    #[must_use]
    pub const fn is_frozen(&self) -> bool {
        self.is_frozen
    }

    /// Returns whether the server is currently stepping forward.
    #[must_use]
    pub const fn is_stepping_forward(&self) -> bool {
        self.frozen_ticks_to_run > 0
    }

    /// Returns the number of frozen ticks to run.
    #[must_use]
    pub const fn frozen_ticks_to_run(&self) -> i32 {
        self.frozen_ticks_to_run
    }

    /// Returns whether game elements should run this tick.
    #[must_use]
    pub const fn runs_normally(&self) -> bool {
        self.run_game_elements
    }

    /// Updates the state for the current tick.
    /// Call this at the start of each server tick.
    pub const fn tick(&mut self) {
        self.run_game_elements = !self.is_frozen || self.frozen_ticks_to_run > 0;
        if self.frozen_ticks_to_run > 0 {
            self.frozen_ticks_to_run -= 1;
        }
    }

    /// Increments the server tick count.
    pub const fn increment_tick_count(&mut self) {
        self.tick_count += 1;
    }

    // Stepping logic (for /tick step)

    /// Steps the game forward by the given number of ticks if paused.
    /// Returns true if stepping was started, false if the game is not frozen.
    pub const fn step_game_if_paused(&mut self, ticks: i32) -> bool {
        if !self.is_frozen {
            return false;
        }
        self.frozen_ticks_to_run = ticks;
        true
    }

    /// Stops the current step operation.
    /// Returns true if stepping was stopped, false if not stepping.
    pub const fn stop_stepping(&mut self) -> bool {
        if self.frozen_ticks_to_run > 0 {
            self.frozen_ticks_to_run = 0;
            true
        } else {
            false
        }
    }

    // Sprinting logic

    /// Returns whether the server is currently sprinting.
    #[must_use]
    pub const fn is_sprinting(&self) -> bool {
        self.scheduled_current_sprint_ticks > 0
    }

    /// Requests the game to sprint for a given number of ticks.
    /// Returns true if an existing sprint was interrupted.
    pub fn request_game_to_sprint(&mut self, ticks: i32) -> bool {
        let interrupted = self.remaining_sprint_ticks > 0;
        self.sprint_time_spent = 0;
        self.scheduled_current_sprint_ticks = i64::from(ticks);
        self.remaining_sprint_ticks = i64::from(ticks);
        self.previous_is_frozen = self.is_frozen;
        self.set_frozen(false);
        interrupted
    }

    /// Stops the current sprint.
    /// Returns the sprint report if a sprint was stopped, None otherwise.
    pub fn stop_sprinting(&mut self) -> Option<SprintReport> {
        if self.remaining_sprint_ticks > 0 {
            Some(self.finish_tick_sprint())
        } else {
            None
        }
    }

    /// Checks if the server should sprint this tick.
    /// Returns Some(report) when the sprint finishes, None otherwise.
    /// The bool indicates whether we should sprint (skip sleep).
    pub fn check_should_sprint_this_tick(&mut self) -> (bool, Option<SprintReport>) {
        if !self.run_game_elements {
            return (false, None);
        }
        if self.remaining_sprint_ticks > 0 {
            self.sprint_tick_start_time = Some(Instant::now());
            self.remaining_sprint_ticks -= 1;
            (true, None)
        } else if self.scheduled_current_sprint_ticks > 0 {
            // Sprint just finished
            (false, Some(self.finish_tick_sprint()))
        } else {
            (false, None)
        }
    }

    /// Ends the work for the current tick sprint.
    /// Call this at the end of each tick during a sprint.
    pub fn end_tick_work(&mut self) {
        if let Some(start) = self.sprint_tick_start_time.take() {
            self.sprint_time_spent += start.elapsed().as_nanos() as i64;
        }
    }

    /// Finishes the current tick sprint and returns the sprint report.
    fn finish_tick_sprint(&mut self) -> SprintReport {
        let completed_ticks = self.scheduled_current_sprint_ticks - self.remaining_sprint_ticks;
        let time_spent_ms = (self.sprint_time_spent.max(1) as f64) / NANOS_PER_MS;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "TPS fits well within i32 range"
        )]
        let ticks_per_second = (MS_PER_SEC * completed_ticks as f64 / time_spent_ms) as i32;
        let ms_per_tick = if completed_ticks == 0 {
            f64::from(self.milliseconds_per_tick())
        } else {
            time_spent_ms / completed_ticks as f64
        };

        self.scheduled_current_sprint_ticks = 0;
        self.sprint_time_spent = 0;
        self.remaining_sprint_ticks = 0;
        self.set_frozen(self.previous_is_frozen);

        SprintReport {
            ticks_per_second,
            ms_per_tick,
        }
    }

    // Tick time tracking methods (vanilla-style)

    /// Records the duration of a tick in nanoseconds.
    /// This should be called at the end of each server tick.
    pub fn record_tick_time(&mut self, tick_time_nanos: u64) {
        let tick_index = self.next_tick_time_sample_index;

        // Remove old value from aggregated sum, add new value
        self.aggregated_tick_times_nanos -= self.tick_times_nanos[tick_index];
        self.aggregated_tick_times_nanos += tick_time_nanos;
        self.tick_times_nanos[tick_index] = tick_time_nanos;
        self.next_tick_time_sample_index = (tick_index + 1) % TICK_STATS_SPAN;
        self.tick_time_sample_count = (self.tick_time_sample_count + 1).min(TICK_STATS_SPAN);

        // Update smoothed tick time (vanilla uses 80/20 exponential smoothing)
        let tick_time_ms = tick_time_nanos as f32 / NANOS_PER_MS as f32;
        self.smoothed_tick_time_ms = self.smoothed_tick_time_ms * TICK_TIME_SMOOTHING
            + tick_time_ms * (1.0 - TICK_TIME_SMOOTHING);
    }

    /// Returns the average tick time in nanoseconds over the last 100 ticks.
    #[must_use]
    pub fn get_average_tick_time_nanos(&self) -> u64 {
        self.aggregated_tick_times_nanos / self.tick_time_sample_count.max(1) as u64
    }

    /// Returns the average tick time in milliseconds over the last 100 ticks.
    #[must_use]
    pub fn get_average_mspt(&self) -> f32 {
        self.get_average_tick_time_nanos() as f32 / NANOS_PER_MS as f32
    }

    /// Returns the exponentially smoothed tick time in milliseconds.
    #[must_use]
    pub const fn get_smoothed_mspt(&self) -> f32 {
        self.smoothed_tick_time_ms
    }

    /// Returns the current TPS (ticks per second) based on average MSPT.
    /// Capped at the configured tick rate (default 20.0).
    #[must_use]
    pub fn get_tps(&self) -> f32 {
        let mspt = self.get_average_mspt();
        if mspt <= 0.0 {
            return self.tick_rate;
        }
        // TPS = 1000ms / mspt, but capped at the configured tick rate
        (1000.0 / mspt).min(self.tick_rate)
    }

    /// Returns a copy of the tick times array for percentile calculation.
    #[must_use]
    pub const fn get_tick_times_nanos(&self) -> [u64; TICK_STATS_SPAN] {
        self.tick_times_nanos
    }

    /// Returns the slowest completed tick in the rolling window, in nanoseconds.
    #[must_use]
    pub fn get_max_tick_time_nanos(&self) -> u64 {
        self.tick_times_nanos[..self.tick_time_sample_count]
            .iter()
            .copied()
            .max()
            .unwrap_or_default()
    }

    // Percentile methods (vanilla-style, used by /tick query)

    /// Returns the P50 (median) tick time in milliseconds.
    #[must_use]
    pub fn get_p50(&self) -> f32 {
        self.get_percentile(50)
    }

    /// Returns the P95 tick time in milliseconds.
    #[must_use]
    pub fn get_p95(&self) -> f32 {
        self.get_percentile(95)
    }

    /// Returns the P99 tick time in milliseconds.
    #[must_use]
    pub fn get_p99(&self) -> f32 {
        self.get_percentile(99)
    }

    /// Returns the number of tick samples currently available.
    #[must_use]
    pub const fn get_sample_count(&self) -> usize {
        self.tick_time_sample_count
    }

    /// Returns the tick time at a given percentile in milliseconds.
    fn get_percentile(&self, percentile: u8) -> f32 {
        let sample_count = self.get_sample_count();
        if sample_count == 0 {
            return 0.0;
        }

        // Copy and sort only the valid samples
        let mut sorted = self.tick_times_nanos;
        sorted[..sample_count].sort_unstable();

        let idx = (sample_count * percentile as usize / 100).min(sample_count - 1);
        sorted[idx] as f32 / NANOS_PER_MS as f32
    }
}

impl Default for TickRateManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_tick(manager: &mut TickRateManager, duration_nanos: u64) {
        manager.increment_tick_count();
        manager.record_tick_time(duration_nanos);
    }

    #[test]
    fn first_completed_tick_occupies_the_first_valid_sample() {
        let mut manager = TickRateManager::new();

        record_tick(&mut manager, 12_345_678);

        assert_eq!(manager.get_sample_count(), 1);
        assert_eq!(manager.get_tick_times_nanos()[0], 12_345_678);
        assert_eq!(manager.get_average_tick_time_nanos(), 12_345_678);
        let duration_ms = 12_345_678_f32 / 1_000_000_f32;
        assert_eq!(manager.get_p50().to_bits(), duration_ms.to_bits());
        assert_eq!(manager.get_p95().to_bits(), duration_ms.to_bits());
        assert_eq!(manager.get_p99().to_bits(), duration_ms.to_bits());
        assert_eq!(manager.get_max_tick_time_nanos(), 12_345_678);
    }

    #[test]
    fn current_tick_is_not_a_timing_sample_until_it_completes() {
        let mut manager = TickRateManager::new();
        record_tick(&mut manager, 20);

        manager.increment_tick_count();

        assert_eq!(manager.get_sample_count(), 1);
        assert_eq!(manager.get_average_tick_time_nanos(), 20);
        assert_eq!(manager.get_max_tick_time_nanos(), 20);
    }

    #[test]
    fn rolling_window_replaces_the_oldest_sample_and_recomputes_maximum() {
        let mut manager = TickRateManager::new();
        record_tick(&mut manager, 1_000);
        for duration in 2..=TICK_STATS_SPAN as u64 {
            record_tick(&mut manager, duration);
        }
        assert_eq!(manager.get_max_tick_time_nanos(), 1_000);

        record_tick(&mut manager, 101);

        let samples = manager.get_tick_times_nanos();
        assert_eq!(manager.get_sample_count(), TICK_STATS_SPAN);
        assert_eq!(samples[0], 101);
        assert_eq!(samples[1], 2);
        assert_eq!(manager.get_average_tick_time_nanos(), 51);
        assert_eq!(manager.get_max_tick_time_nanos(), 101);
    }

    #[test]
    fn empty_window_has_zero_average_percentiles_and_maximum() {
        let manager = TickRateManager::new();

        assert_eq!(manager.get_sample_count(), 0);
        assert_eq!(manager.get_average_tick_time_nanos(), 0);
        assert_eq!(manager.get_p50().to_bits(), 0.0_f32.to_bits());
        assert_eq!(manager.get_p95().to_bits(), 0.0_f32.to_bits());
        assert_eq!(manager.get_p99().to_bits(), 0.0_f32.to_bits());
        assert_eq!(manager.get_max_tick_time_nanos(), 0);
    }
}
