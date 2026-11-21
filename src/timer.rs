//! Timer / timeslice bookkeeping.
//!
//! The scheduler is driven by the PIT (see `rustos-drivers/src/pit.rs`):
//! channel 0 is programmed to [`TICK_HZ`] and raises IRQ 0 on every
//! tick. The kernel's timer IRQ handler calls [`Scheduler::tick`] (and
//! feeds this [`Timer`] if it wants uptime). This module owns the
//! arithmetic only — it never touches ports or the PIC.
//!
//! Default policy: 100 Hz ticks, 10 ms timeslice, i.e. one tick per
//! slice. A task that is still Running when its slice ends is
//! preempted in favour of the next Ready task.

/// PIT interrupt frequency, in Hz. Must match the value the kernel
/// programs into the PIT (see `rustos-drivers` `pit::init`).
pub const TICK_HZ: u32 = 100;

/// Length of one timeslice, in milliseconds.
pub const TIMESLICE_MS: u32 = 10;

/// How many ticks make one timeslice at `hz`. At least 1, so a very
/// slow tick rate still preempts (just with a longer slice).
pub const fn ticks_per_timeslice(hz: u32) -> u32 {
    if hz == 0 {
        return 1;
    }
    let t = TIMESLICE_MS * hz / 1000;
    if t == 0 {
        1
    } else {
        t
    }
}

/// Ticks in `ms` milliseconds at `hz`.
pub const fn ms_to_ticks(ms: u64, hz: u32) -> u64 {
    ms * hz as u64 / 1000
}

/// Milliseconds in `ticks` ticks at `hz` (0 if `hz` is 0).
pub const fn ticks_to_ms(ticks: u64, hz: u32) -> u64 {
    if hz == 0 {
        0
    } else {
        ticks * 1000 / hz as u64
    }
}

/// Counts ticks since boot and reports timeslice boundaries.
#[derive(Debug, Clone, Copy)]
pub struct Timer {
    hz: u32,
    ticks: u64,
    ticks_in_slice: u32,
}

impl Timer {
    pub fn new(hz: u32) -> Self {
        Timer {
            hz,
            ticks: 0,
            ticks_in_slice: 0,
        }
    }

    /// Record one PIT tick. Returns `true` when a timeslice boundary
    /// is reached (i.e. the running task's slice is over and the
    /// scheduler should consider preempting it).
    pub fn on_tick(&mut self) -> bool {
        self.ticks += 1;
        self.ticks_in_slice += 1;
        if self.ticks_in_slice >= ticks_per_timeslice(self.hz) {
            self.ticks_in_slice = 0;
            true
        } else {
            false
        }
    }

    /// Total ticks since boot.
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Uptime in milliseconds.
    pub fn uptime_ms(&self) -> u64 {
        ticks_to_ms(self.ticks, self.hz)
    }

    /// Ticks elapsed in the current (incomplete) timeslice.
    pub fn ticks_in_slice(&self) -> u32 {
        self.ticks_in_slice
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_per_timeslice_values() {
        assert_eq!(ticks_per_timeslice(100), 1); // 10 ms at 100 Hz
        assert_eq!(ticks_per_timeslice(1000), 10);
        assert_eq!(ticks_per_timeslice(10000), 100);
        assert_eq!(ticks_per_timeslice(50), 1); // clamped: 10 ms < 1 tick
        assert_eq!(ticks_per_timeslice(0), 1);
    }

    #[test]
    fn ms_tick_conversions() {
        assert_eq!(ms_to_ticks(1000, 100), 100);
        assert_eq!(ms_to_ticks(10, 100), 1);
        assert_eq!(ticks_to_ms(100, 100), 1000);
        assert_eq!(ticks_to_ms(50, 0), 0);
    }

    #[test]
    fn timer_slice_boundary_at_1000hz() {
        let mut timer = Timer::new(1000); // slice = 10 ticks
        for _ in 0..9 {
            assert!(!timer.on_tick());
        }
        assert!(timer.on_tick()); // 10th tick: boundary
        assert_eq!(timer.ticks(), 10);
        assert_eq!(timer.ticks_in_slice(), 0);
        assert_eq!(timer.uptime_ms(), 10);
    }

    #[test]
    fn timer_slice_boundary_at_100hz_is_every_tick() {
        let mut timer = Timer::new(100); // slice = 1 tick
        assert!(timer.on_tick());
        assert!(timer.on_tick());
        assert_eq!(timer.uptime_ms(), 20);
    }
}
