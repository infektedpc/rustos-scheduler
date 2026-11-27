//! A single schedulable task: its identity, state, priority, stack,
//! entry point, and saved [`Context`](crate::context::Context).
//!
//! A task is created with [`Task::prepare_stack`] (lays out its stack
//! in kernel memory) followed by [`Task::new`] (records the prepared
//! stack pointer in its context). The scheduler owns tasks after that;
//! the kernel owns the backing stack memory and must keep it alive
//! until the task is `Terminated` and reaped.

use crate::context::Context;
use crate::timer::{ticks_per_timeslice, TICK_HZ};

/// Opaque task identifier. Assigned by the [`Scheduler`](crate::Scheduler),
/// starting at 1 (0 is reserved for "no task").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(pub u64);

impl TaskId {
    pub const NONE: TaskId = TaskId(0);
}

/// Lifecycle state of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Eligible to run; waiting for the scheduler to pick it.
    Ready,
    /// Currently executing on a CPU.
    Running,
    /// Waiting on something (I/O, another task, a timer). `blocked_on`
    /// says what — see [`Task::blocked_on`].
    Blocked,
    /// Finished (or killed). Never scheduled again.
    Terminated,
}

/// Entry point of a task: a plain function that never returns
/// (it loops, halts, or calls the exit syscall). Declared `fn()`,
/// not `fn() -> !`: a diverging fn item does not coerce to `fn()`
/// in this toolchain, so entries are `fn()` whose bodies diverge
/// (an infinite loop) instead.
pub type TaskEntry = fn();

/// Placeholder entry for ring-3 tasks (see [`Task::new_for_user`]):
/// the first switch to such a task `ret`s into the iretq stub, not
/// into this function, so it must never run. If it ever did, the
/// safe behaviour is to spin forever — returning would `ret` into
/// whatever sits above the (empty) stack.
fn user_entry_placeholder() {
    loop {
        core::hint::spin_loop();
    }
}

/// A task.
#[derive(Debug)]
pub struct Task {
    pub id: TaskId,
    /// Human-readable name, for logs and the shell's `ps`-style output.
    pub name: &'static str,
    pub state: TaskState,
    /// Priority, 0..=255. **Higher runs first.** Ties are broken
    /// round-robin. There is no aging yet, so a steady stream of
    /// high-priority Ready tasks can starve lower ones — documented
    /// in the README as a known limitation.
    pub priority: u8,
    /// Entry point (also stored in `context.rip`).
    pub entry: TaskEntry,
    /// Saved CPU context (registers + stack pointer).
    pub context: Context,
    /// One-past-the-end address of the task's stack.
    pub stack_top: u64,
    /// Size of the task's stack, in bytes.
    pub stack_size: usize,
    /// Ticks of CPU this task has consumed in total.
    pub ticks_run: u64,
    /// Ticks left in the current timeslice. Reset to a full slice
    /// whenever the task is (re)scheduled.
    pub quantum_left: u32,
    /// What a Blocked task is waiting on, if anything. Convention:
    /// 0 = I/O, 1 = timer/sleep, 2 = another task (join). The kernel
    /// and drivers define the codes; the scheduler only stores them.
    pub blocked_on: Option<u32>,
    /// Exit code, once Terminated.
    pub exit_code: Option<i32>,
    /// True for a ring-3 (userspace) task: its entry point is a raw
    /// program address (not a Rust fn), its `stack_top` is the top
    /// of its *kernel* stack (what `TSS.rsp[0]` points at when it
    /// runs), and its faults and syscall buffers are handled as
    /// user faults / user buffers (see `rustos-kernel`). False for
    /// kernel tasks.
    pub is_user: bool,
}

impl Task {
    /// Lay out a task stack in kernel memory and return the initial
    /// `rsp` to pass to [`Task::new`].
    ///
    /// # Safety
    ///
    /// `stack` must point at `stack_size` writable bytes that belong
    /// to this task alone and stay alive until the task terminates.
    pub unsafe fn prepare_stack(
        stack: *mut u8,
        stack_size: usize,
        entry: TaskEntry,
    ) -> Option<u64> {
        unsafe { crate::context::prepare_initial_stack_raw(stack, stack_size, entry as u64) }
    }

    /// Create a task. `initial_rsp` must be the value returned by
    /// [`Task::prepare_stack`] for this task's stack; `stack_top` is
    /// the one-past-the-end address of that stack.
    pub fn new(
        id: TaskId,
        name: &'static str,
        priority: u8,
        initial_rsp: u64,
        stack_top: u64,
        stack_size: usize,
        entry: TaskEntry,
    ) -> Self {
        Task {
            id,
            name,
            state: TaskState::Ready,
            priority,
            entry,
            context: Context::new(entry as u64, initial_rsp),
            stack_top,
            stack_size,
            ticks_run: 0,
            quantum_left: ticks_per_timeslice(TICK_HZ),
            blocked_on: None,
            exit_code: None,
            is_user: false,
        }
    }

    /// Create a ring-3 (userspace) task whose entry point is a raw
    /// address — a program's ELF entry — not a Rust function.
    /// `initial_rsp` must be the value returned by
    /// [`crate::context::prepare_ring3_stack`] for this task's
    /// *kernel* stack; `stack_top` is that kernel stack's
    /// one-past-the-end address (also what `TSS.rsp[0]` is set to
    /// when this task runs, so privilege transitions land on it).
    ///
    /// The `entry` field holds a placeholder that is never called:
    /// the first switch to this task `ret`s into the iretq stub
    /// (see [`crate::context::prepare_ring3_stack`]), not into
    /// `entry` — the real entry lives in `context.rip` (and in the
    /// iretq frame on the initial stack).
    pub fn new_for_user(
        id: TaskId,
        name: &'static str,
        priority: u8,
        initial_rsp: u64,
        stack_top: u64,
        stack_size: usize,
        entry_rip: u64,
    ) -> Self {
        Task {
            id,
            name,
            state: TaskState::Ready,
            priority,
            entry: user_entry_placeholder,
            context: Context::new(entry_rip, initial_rsp),
            stack_top,
            stack_size,
            ticks_run: 0,
            quantum_left: ticks_per_timeslice(TICK_HZ),
            blocked_on: None,
            exit_code: None,
            is_user: true,
        }
    }

    /// Start of the stack (lowest address).
    pub fn stack_bottom(&self) -> u64 {
        self.stack_top.saturating_sub(self.stack_size as u64)
    }

    /// True if `addr` lies inside this task's stack.
    pub fn stack_contains(&self, addr: u64) -> bool {
        addr >= self.stack_bottom() && addr < self.stack_top
    }

    /// True if the task can be scheduled right now.
    pub fn is_ready(&self) -> bool {
        self.state == TaskState::Ready
    }

    /// Reset the timeslice quantum (called when the task is scheduled).
    pub fn reset_quantum(&mut self) {
        self.quantum_left = ticks_per_timeslice(TICK_HZ);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_entry() {}

    /// A stack buffer owned by the test; returns (rsp, top, size).
    fn make_stack(entry: TaskEntry) -> (Vec<u8>, u64, u64, usize) {
        let mut buf = vec![0u8; 8192];
        let rsp = unsafe { Task::prepare_stack(buf.as_mut_ptr(), buf.len(), entry) }
            .expect("stack big enough");
        let top = buf.as_ptr() as u64 + buf.len() as u64;
        (buf, rsp, top, 8192)
    }

    #[test]
    fn new_task_initial_state_and_context() {
        let (_buf, rsp, top, size) = make_stack(dummy_entry);
        let task = Task::new(TaskId(1), "shell", 5, rsp, top, size, dummy_entry);
        assert_eq!(task.id, TaskId(1));
        assert_eq!(task.name, "shell");
        assert_eq!(task.state, TaskState::Ready);
        assert_eq!(task.priority, 5);
        assert_eq!(task.context.rip, dummy_entry as *const () as u64);
        assert_eq!(task.context.rsp, rsp);
        assert_eq!(task.context.rflags, 0x202);
        assert_eq!(task.ticks_run, 0);
        assert!(task.quantum_left > 0);
        assert_eq!(task.blocked_on, None);
        assert_eq!(task.exit_code, None);
        assert!(task.is_ready());
    }

    #[test]
    fn stack_bounds() {
        let (_buf, rsp, top, size) = make_stack(dummy_entry);
        let task = Task::new(TaskId(1), "t", 0, rsp, top, size, dummy_entry);
        assert_eq!(task.stack_bottom(), top - 8192);
        assert!(task.stack_contains(top - 1));
        assert!(task.stack_contains(task.stack_bottom()));
        assert!(!task.stack_contains(top));
        assert!(!task.stack_contains(0));
    }

    #[test]
    fn prepare_stack_returns_none_for_tiny_stack() {
        let mut buf = vec![0u8; 16];
        assert_eq!(
            unsafe { Task::prepare_stack(buf.as_mut_ptr(), buf.len(), dummy_entry) },
            None
        );
    }
}
