//! `rustos-scheduler` — preemptive round-robin scheduler for RustOS.
//!
//! Modules:
//!
//! * [`context`] — the saved CPU state of a task (`Context`) and the
//!   x86_64 `switch()` that swaps one task's registers/stack for
//!   another's. This is the only hardware-specific part.
//! * [`task`] — a [`Task`]: id, state, priority, entry point, stack,
//!   saved context, accounting.
//! * [`scheduler`] — the [`Scheduler`]: priority-ordered round-robin
//!   queues (implemented as a scan over one task list), `tick()` for
//!   timer-driven preemption, `yield_now()`, `block()`/`unblock()`,
//!   `exit()`.
//! * [`timer`] — tick bookkeeping: how many PIT ticks make one
//!   timeslice, uptime, slice-boundary detection.
//! * [`syscall`] — dispatch of the scheduler-related syscalls from
//!   `rustos-shell` (`SYS_YIELD = 24`, `SYS_EXIT = 60`); the numbers
//!   match `rustos-shell/src/syscall.rs` exactly.
//!
//! The crate is `#![no_std]` outside tests (`core` + `alloc` only, no
//! external crates), so it can link into `rustos-kernel`. All scheduling
//! decisions are pure logic over the task list and are unit-tested on
//! the host; only [`context::switch`] touches hardware, and it is
//! deliberately *not* exercised by the unit tests (calling it on the
//! host would jump to a fabricated stack).
//!
//! # Wiring into the kernel
//!
//! ```ignore
//! // rustos-kernel, after gdt::init() / interrupts::init_idt():
//! let mut sched = Scheduler::new();
//! // Stacks are kernel memory (e.g. pages from BootInfoFrameAllocator,
//! // or 'static arrays); prepare each one before creating the task:
//! let rsp = unsafe { Task::prepare_stack(stack_ptr, STACK_SIZE, entry as u64) };
//! sched.add_task("shell", 5, rsp, stack_top, entry);
//! // In the timer IRQ handler (IRQ 0, after the PICs are remapped —
//! // see README): `match sched.tick() { TickOutcome::Preempt(next) => switch_to(next), _ => {} }`
//! // and `switch_to` is where the kernel calls `context::switch(old_ctx, new_ctx)`.
//! ```

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod context;
pub mod scheduler;
pub mod syscall;
pub mod task;
pub mod timer;

pub use context::{prepare_initial_stack, switch, Context};
pub use scheduler::{Scheduler, TickOutcome};
pub use task::{Task, TaskId, TaskState};
pub use timer::{Timer, TICK_HZ, TIMESLICE_MS};
