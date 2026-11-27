//! Round-robin scheduler with priorities.
//!
//! Policy:
//!
//! * Among `Ready` tasks, the **highest priority** runs first.
//! * Within one priority, tasks run **round-robin**: each pick starts
//!   scanning after the last-picked task, so a task is not picked twice
//!   while another same-priority task is still waiting.
//! * A `Running` task gets one timeslice ([`crate::timer`] — 10 ms at
//!   100 Hz). [`Scheduler::tick`] is called from the timer IRQ; when
//!   the slice expires and another task is Ready, the scheduler
//!   returns [`TickOutcome::Preempt`] and the kernel switches to it.
//! * `yield_now()` is the voluntary version of the same thing
//!   (syscall `SYS_YIELD`); `block()`/`unblock()` park a task until
//!   something wakes it (I/O completion, another task); `exit()`
//!   terminates it.
//!
//! The scheduler decides *which* task runs next and owns the state
//! transitions. It does **not** perform the register switch itself:
//! the kernel reads the old/new [`Context`](crate::context::Context)
//! out of the tasks and calls [`crate::context::switch`]. The
//! decision is returned as task ids so the pure logic stays
//! unit-testable on the host.
//!
//! Single CPU only (no SMP): one `current` task at a time. There is
//! no aging, so sustained high-priority work can starve low-priority
//! tasks — see the README.

use alloc::vec::Vec;

use crate::context::Context;
use crate::task::{Task, TaskEntry, TaskId, TaskState};

/// What [`Scheduler::tick`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickOutcome {
    /// Keep running the current task (slice not over, or nothing else
    /// is Ready).
    Continue,
    /// Slice over: switch to this task now.
    Preempt(TaskId),
}

/// The scheduler: owns all tasks and the currently-running one.
#[derive(Debug)]
pub struct Scheduler {
    tasks: Vec<Task>,
    current: Option<TaskId>,
    next_id: u64,
    /// Index into `tasks` just after the last-picked task; the
    /// round-robin scan starts here.
    rr_cursor: usize,
    /// Total timer ticks seen by `tick()`.
    ticks: u64,
    /// Total context switches performed by `switch_to()`.
    switches: u64,
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler {
            tasks: Vec::new(),
            current: None,
            next_id: 1,
            rr_cursor: 0,
            ticks: 0,
            switches: 0,
        }
    }

    /// Number of tasks known to the scheduler (any state).
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Number of tasks in `Ready` state.
    pub fn ready_count(&self) -> usize {
        self.tasks.iter().filter(|t| t.is_ready()).count()
    }

    /// True when at least one ring-3 (userspace) task is still alive
    /// (any state but `Terminated`). The kernel's idle loop uses
    /// this to decide who owns the keyboard: while a userspace
    /// program exists, its `read(fd 0)` is the only consumer of
    /// stdin (serial + the PS/2 key ring) — the idle loop must not
    /// drain or echo keystrokes itself, or it would steal bytes the
    /// program is waiting for and echo them twice (Batch 9). Once
    /// every user task has exited, the idle loop resumes echoing so
    /// the console stays live. Kernel tasks never count: the worker
    /// does not read stdin, so it must not silence the echo.
    pub fn has_live_user_task(&self) -> bool {
        self.tasks
            .iter()
            .any(|t| t.is_user && t.state != TaskState::Terminated)
    }

    /// The currently-running task, if any.
    pub fn current_id(&self) -> Option<TaskId> {
        self.current
    }

    pub fn current_task(&self) -> Option<&Task> {
        self.current.and_then(|id| self.task(id))
    }

    /// Look a task up by id.
    pub fn task(&self, id: TaskId) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }

    pub fn task_mut(&mut self, id: TaskId) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }

    /// All tasks, in creation order.
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// Total timer ticks processed.
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Total switches performed by `switch_to()`.
    pub fn switches(&self) -> u64 {
        self.switches
    }

    /// Create a task from a prepared stack and add it in `Ready` state.
    /// `initial_rsp` must come from [`Task::prepare_stack`]; see
    /// [`Task::new`]. Returns the assigned id.
    pub fn add_task(
        &mut self,
        name: &'static str,
        priority: u8,
        initial_rsp: u64,
        stack_top: u64,
        stack_size: usize,
        entry: TaskEntry,
    ) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        self.tasks.push(Task::new(
            id, name, priority, initial_rsp, stack_top, stack_size, entry,
        ));
        id
    }

    /// Create a ring-3 (userspace) task from a prepared kernel stack
    /// and add it in `Ready` state. `initial_rsp` must come from
    /// [`crate::context::prepare_ring3_stack`]; `entry_rip` is the
    /// program's ELF entry point (a raw address, not a Rust fn).
    /// Returns the assigned id. See [`Task::new_for_user`].
    pub fn add_user_task(
        &mut self,
        name: &'static str,
        priority: u8,
        initial_rsp: u64,
        stack_top: u64,
        stack_size: usize,
        entry_rip: u64,
    ) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        self.tasks.push(Task::new_for_user(
            id, name, priority, initial_rsp, stack_top, stack_size, entry_rip,
        ));
        id
    }

    /// Add a task that was already constructed (e.g. by tests).
    /// Its id is kept; the scheduler's id counter moves past it.
    pub fn add_existing(&mut self, task: Task) -> TaskId {
        let id = task.id;
        if id.0 >= self.next_id {
            self.next_id = id.0 + 1;
        }
        self.tasks.push(task);
        id
    }

    /// Pick the next task to run: highest-priority `Ready` task,
    /// round-robin within a priority, optionally skipping `exclude`.
    /// Pure: changes no state.
    pub fn pick_next(&self) -> Option<TaskId> {
        self.pick_next_excluding(None)
    }

    fn pick_next_excluding(&self, exclude: Option<TaskId>) -> Option<TaskId> {
        if self.tasks.is_empty() {
            return None;
        }
        let mut best: Option<(usize, u8)> = None; // (index, priority)
        for offset in 0..self.tasks.len() {
            let i = (self.rr_cursor + offset) % self.tasks.len();
            let t = &self.tasks[i];
            if t.state != TaskState::Ready {
                continue;
            }
            if Some(t.id) == exclude {
                continue;
            }
            match best {
                None => best = Some((i, t.priority)),
                Some((_, p)) if t.priority > p => best = Some((i, t.priority)),
                _ => {}
            }
        }
        best.map(|(i, _)| self.tasks[i].id)
    }

    /// Switch the CPU to `next` (which must be `Ready`, or already the
    /// current task). The previous `Running` task goes back to `Ready`
    /// with a fresh quantum. Returns the previous current task, if any.
    /// The kernel performs the actual register switch after this call
    /// (see `context_switch()`); this only moves the bookkeeping.
    pub fn switch_to(&mut self, next: TaskId) -> Option<TaskId> {
        let prev = self.current;
        if prev == Some(next) {
            return prev;
        }
        let next_idx = self.tasks.iter().position(|t| t.id == next)?;
        if self.tasks[next_idx].state != TaskState::Ready {
            return None;
        }
        if let Some(prev_id) = prev {
            if let Some(prev_task) = self.task_mut(prev_id) {
                if prev_task.state == TaskState::Running {
                    prev_task.state = TaskState::Ready;
                    prev_task.reset_quantum();
                }
            }
        }
        self.tasks[next_idx].state = TaskState::Running;
        self.tasks[next_idx].reset_quantum();
        self.rr_cursor = next_idx;
        self.current = Some(next);
        self.switches += 1;
        prev
    }

    /// Pick the next task and switch to it. Returns the newly-running
    /// task, or `None` if nothing is Ready (the kernel then idles).
    pub fn run_next(&mut self) -> Option<TaskId> {
        let next = self.pick_next()?;
        self.switch_to(next);
        Some(next)
    }

    /// The (from, to) contexts for a switch to `next`: the caller
    /// passes these to [`crate::context::switch`]. Returns `None` if
    /// there is no current task (first run — the kernel just loads the
    /// new task's context) or `next` is unknown/not Ready.
    ///
    /// # Safety
    ///
    /// The returned pointers borrow the scheduler's task list; the
    /// kernel must call `switch()` before mutating the scheduler
    /// again, and must not hold the pointers across a switch.
    pub unsafe fn context_switch(
        &mut self,
        next: TaskId,
    ) -> Option<(*mut Context, *const Context)> {
        let next_idx = self.tasks.iter().position(|t| t.id == next)?;
        if self.tasks[next_idx].state != TaskState::Ready {
            return None;
        }
        let to = &self.tasks[next_idx].context as *const Context;
        let from = match self.current {
            Some(cur) => {
                let cur_idx = self.tasks.iter().position(|t| t.id == cur)?;
                &mut self.tasks[cur_idx].context as *mut Context
            }
            None => core::ptr::null_mut(),
        };
        Some((from, to))
    }

    /// One timer tick (call from the timer IRQ handler, 100 Hz).
    /// Charges the running task one tick; on quantum expiry, if
    /// another task is Ready, returns `Preempt(next)`. The caller is
    /// responsible for `switch_to(next)` + the register switch.
    pub fn tick(&mut self) -> TickOutcome {
        self.ticks += 1;
        let Some(current) = self.current else {
            return TickOutcome::Continue;
        };
        // Charge the running task one tick. The borrow ends here so
        // the pick below can borrow `self` again.
        let expired = {
            let Some(task) = self.task_mut(current) else {
                return TickOutcome::Continue;
            };
            task.ticks_run += 1;
            if task.quantum_left > 0 {
                task.quantum_left -= 1;
            }
            if task.quantum_left > 0 {
                return TickOutcome::Continue;
            }
            // Slice over: fresh quantum for next time this task runs.
            task.reset_quantum();
            task.state = TaskState::Ready;
            true
        };
        let _ = expired;
        match self.pick_next_excluding(Some(current)) {
            Some(next) => TickOutcome::Preempt(next),
            None => {
                // Nothing else to run: stay on the current task.
                if let Some(task) = self.task_mut(current) {
                    task.state = TaskState::Running;
                }
                TickOutcome::Continue
            }
        }
    }

    /// Voluntary yield (syscall `SYS_YIELD`): give up the rest of the
    /// slice. Returns the newly-running task, or the current task if
    /// nothing else is Ready.
    pub fn yield_now(&mut self) -> Option<TaskId> {
        let current = self.current?;
        if let Some(next) = self.pick_next_excluding(Some(current)) {
            self.switch_to(next);
            Some(next)
        } else {
            if let Some(t) = self.task_mut(current) {
                t.reset_quantum();
            }
            Some(current)
        }
    }

    /// Park the current task: `Running` -> `Blocked`, waiting on
    /// `reason` (see [`Task::blocked_on`]). Returns false if no task
    /// is currently running.
    pub fn block_current(&mut self, reason: u32) -> bool {
        let Some(current) = self.current else {
            return false;
        };
        self.block(current, reason)
    }

    /// Park any task by id. If it was the current task, the CPU
    /// becomes idle until `run_next()` is called.
    pub fn block(&mut self, id: TaskId, reason: u32) -> bool {
        let was_current = self.current == Some(id);
        let Some(t) = self.task_mut(id) else {
            return false;
        };
        if t.state == TaskState::Terminated {
            return false;
        }
        t.state = TaskState::Blocked;
        t.blocked_on = Some(reason);
        if was_current {
            self.current = None;
        }
        true
    }

    /// Wake a blocked task: `Blocked` -> `Ready`. Returns false if
    /// the task does not exist or was not blocked.
    pub fn unblock(&mut self, id: TaskId) -> bool {
        let Some(t) = self.task_mut(id) else {
            return false;
        };
        if t.state != TaskState::Blocked {
            return false;
        }
        t.state = TaskState::Ready;
        t.blocked_on = None;
        true
    }

    /// Terminate the current task with `code` (syscall `SYS_EXIT`).
    /// Returns false if no task is currently running.
    pub fn exit_current(&mut self, code: i32) -> bool {
        let Some(current) = self.current else {
            return false;
        };
        self.exit(current, code)
    }

    /// Terminate any task by id. If it was the current task, the CPU
    /// becomes idle until `run_next()` is called.
    pub fn exit(&mut self, id: TaskId, code: i32) -> bool {
        let was_current = self.current == Some(id);
        let Some(t) = self.task_mut(id) else {
            return false;
        };
        if t.state == TaskState::Terminated {
            return false;
        }
        t.state = TaskState::Terminated;
        t.exit_code = Some(code);
        t.blocked_on = None;
        if was_current {
            self.current = None;
        }
        true
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::TaskId;

    fn idle_entry() {}

    fn other_entry() {}

    /// Build a scheduler with `n` same-priority tasks, each with its
    /// own stack buffer. Buffers are leaked on purpose: the tasks only
    /// store addresses, and the test needs the memory to stay alive.
    /// (Kernel stacks live in kernel memory for the same reason.)
    fn sched_with_tasks(names: &[(&'static str, u8)]) -> (Scheduler, Vec<TaskId>) {
        let mut sched = Scheduler::new();
        let mut ids = Vec::new();
        for (name, prio) in names {
            let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 8192]));
            let rsp = unsafe { Task::prepare_stack(buf.as_mut_ptr(), buf.len(), idle_entry) }
                .expect("stack");
            let top = buf.as_ptr() as u64 + buf.len() as u64;
            ids.push(sched.add_task(name, *prio, rsp, top, buf.len(), idle_entry));
        }
        (sched, ids)
    }

    #[test]
    fn add_task_assigns_incremental_ids_and_starts_ready() {
        let (sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5)]);
        assert_eq!(ids, vec![TaskId(1), TaskId(2)]);
        assert_eq!(sched.task_count(), 2);
        assert_eq!(sched.ready_count(), 2);
        assert_eq!(sched.current_id(), None);
        assert_eq!(sched.task(ids[0]).unwrap().state, TaskState::Ready);
    }

    #[test]
    fn add_user_task_stores_the_raw_entry_and_ring3_stack() {
        // A ring-3 task created through add_user_task: its
        // context.rip is the program's ELF entry (a raw address,
        // not a Rust fn), its context.rsp is the prepared ring-3
        // stack pointer, and its stack range is the kernel stack
        // (the one TSS.rsp[0] points at when it runs).
        use crate::context::{prepare_ring3_stack, Ring3Frame};
        let mut sched = Scheduler::new();
        let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 16384]));
        let frame = Ring3Frame::new(0x400040, 0x7fffd8, 5, 6);
        let rsp = prepare_ring3_stack(buf, frame).expect("stack big enough");
        let top = buf.as_ptr() as u64 + buf.len() as u64;
        let id = sched.add_user_task("hello", 5, rsp, top, buf.len(), 0x400040);
        let t = sched.task(id).unwrap();
        assert_eq!(t.id, id);
        assert_eq!(t.name, "hello");
        assert_eq!(t.state, TaskState::Ready);
        assert_eq!(t.context.rip, 0x400040);
        assert_eq!(t.context.rsp, rsp);
        assert_eq!(t.stack_top, top);
        assert_eq!(t.stack_bottom(), top - 16384);
        assert!(t.stack_contains(t.context.rsp));
        // The prepared stack really holds the iretq frame: after
        // the pops and ret, rsp lands on the frame's RIP slot.
        let frame_addr = t.context.rsp as usize + 7 * 8;
        assert_eq!(unsafe { *(frame_addr as *const u64) }, 0x400040);
    }

    #[test]
    fn has_live_user_task_tracks_userspace_lifetime() {
        // No tasks at all: nobody owns the keyboard.
        let mut sched = Scheduler::new();
        assert!(!sched.has_live_user_task());
        // Kernel tasks only (idle + worker): the idle loop keeps
        // echoing — the worker never reads stdin.
        let (mut sched, _ids) = sched_with_tasks(&[("idle", 5), ("worker", 5)]);
        assert!(!sched.has_live_user_task());
        // A live user task claims the keyboard.
        let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 16384]));
        use crate::context::{prepare_ring3_stack, Ring3Frame};
        let frame = Ring3Frame::new(0x800000, 0xbfff_e8, 5, 6);
        let rsp = prepare_ring3_stack(buf, frame).expect("stack big enough");
        let top = buf.as_ptr() as u64 + buf.len() as u64;
        let sh = sched.add_user_task("sh", 5, rsp, top, buf.len(), 0x800000);
        assert!(sched.has_live_user_task());
        // A second user task does not change it; terminating one
        // leaves it true while the other lives.
        let buf2 = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 16384]));
        let frame2 = Ring3Frame::new(0x400000, 0x7fff_e8, 5, 6);
        let rsp2 = prepare_ring3_stack(buf2, frame2).expect("stack big enough");
        let top2 = buf2.as_ptr() as u64 + buf2.len() as u64;
        let hello = sched.add_user_task("hello", 5, rsp2, top2, buf2.len(), 0x400000);
        assert!(sched.has_live_user_task());
        assert!(sched.exit(hello, 0));
        assert!(sched.has_live_user_task());
        // Terminating the last user task hands the keyboard back to
        // the idle loop.
        assert!(sched.exit(sh, 0));
        assert!(!sched.has_live_user_task());
    }

    #[test]
    fn add_task_stores_the_stack_range_it_was_given() {
        // The boot log prints `stack_bottom()-stack_top` for each
        // task (see the kernel's `init_tasks` snapshot) — exactly
        // what this pins. On the 2026-11-22 live boot the worker
        // printed `0x4444444402c0-0x4444444442c0`, which looked like
        // a fill value but was the real heap address (the kernel
        // heap starts at `0x444444440000`); the range, the size,
        // and the allocation base must all agree, or the print is
        // lying about the stack. If `add_task` were ever passed the
        // prepared `rsp` as the top, or a wrong size, this fails.
        let mut sched = Scheduler::new();
        let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 8192]));
        let rsp = unsafe { Task::prepare_stack(buf.as_mut_ptr(), buf.len(), idle_entry) }
            .expect("stack");
        let top = buf.as_ptr() as u64 + buf.len() as u64;
        let id = sched.add_task("w", 5, rsp, top, buf.len(), idle_entry);
        let t = sched.task(id).unwrap();
        assert_eq!(t.stack_top, top);
        assert_eq!(t.stack_bottom(), top - 8192);
        assert_eq!(t.stack_bottom(), buf.as_ptr() as u64);
        assert_eq!(t.stack_size, 8192);
        assert!(t.stack_contains(t.stack_bottom()));
        assert!(t.stack_contains(top - 1));
        assert!(!t.stack_contains(top));
        // The prepared rsp sits just below the top (8 slots down
        // from the 16-aligned top).
        assert!(t.stack_contains(t.context.rsp));
        assert_eq!(t.context.rsp, (top & !0xF) - 64);
    }

    #[test]
    fn pick_next_empty_is_none() {
        let mut sched = Scheduler::new();
        assert_eq!(sched.pick_next(), None);
        assert_eq!(sched.run_next(), None);
    }

    #[test]
    fn round_robin_fairness_within_one_priority() {
        let (mut sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5), ("c", 5)]);
        // Successive run_next() calls cycle a -> b -> c -> a ...
        assert_eq!(sched.run_next(), Some(ids[0]));
        assert_eq!(sched.run_next(), Some(ids[1]));
        assert_eq!(sched.run_next(), Some(ids[2]));
        assert_eq!(sched.run_next(), Some(ids[0]));
        assert_eq!(sched.run_next(), Some(ids[1]));
        assert_eq!(sched.switches(), 5);
    }

    #[test]
    fn idle_task_joins_round_robin_at_equal_priority() {
        // Batch 8 (2026-11-22): the kernel's idle task starved
        // forever once the worker and a userspace task were both
        // Ready at priority 5 while idle sat at 0 — pick_next always
        // prefers the higher priority, so kernel_main never ran
        // again (no "hello task pid 2 created", no "RustOS kernel
        // ready", no uptime). The kernel now creates idle at the
        // same priority as the others; with equal priorities the
        // round-robin must visit every task, idle included.
        let (mut sched, ids) = sched_with_tasks(&[("idle", 5), ("worker", 5), ("hello", 5)]);
        assert_eq!(sched.run_next(), Some(ids[0])); // idle
        assert_eq!(sched.run_next(), Some(ids[1])); // worker
        assert_eq!(sched.run_next(), Some(ids[2])); // hello
        assert_eq!(sched.run_next(), Some(ids[0])); // idle again
    }

    #[test]
    fn low_priority_task_starves_while_higher_priority_tasks_stay_ready() {
        // The other half of the Batch 8 lesson: a priority-0 task is
        // NEVER picked while any priority-5 task is Ready — the pick
        // scans for the highest priority, and round-robin only
        // applies within one priority. This is why the idle task
        // must not sit below the always-Ready tasks; the scheduler
        // offers no aging (see README).
        let (mut sched, ids) = sched_with_tasks(&[("idle", 0), ("worker", 5), ("hello", 5)]);
        assert_eq!(sched.run_next(), Some(ids[1])); // worker
        assert_eq!(sched.run_next(), Some(ids[2])); // hello
        assert_eq!(sched.run_next(), Some(ids[1])); // worker again
        assert_eq!(sched.run_next(), Some(ids[2])); // hello again
        assert_ne!(sched.current_id(), Some(ids[0]));
    }

    #[test]
    fn priority_ordering_highest_first() {
        let (mut sched, ids) = sched_with_tasks(&[("low", 1), ("high", 9), ("mid", 5)]);
        assert_eq!(sched.pick_next(), Some(ids[1])); // high, added second
        sched.run_next();
        // High is Running now; next pick is the highest Ready task.
        assert_eq!(sched.pick_next(), Some(ids[2])); // mid
    }

    #[test]
    fn priority_does_not_block_low_forever_once_high_is_gone() {
        let (mut sched, ids) = sched_with_tasks(&[("low", 1), ("high", 9)]);
        sched.run_next(); // high runs
        assert_eq!(sched.current_id(), Some(ids[1]));
        sched.exit_current(0);
        assert_eq!(sched.run_next(), Some(ids[0])); // low finally runs
    }

    #[test]
    fn tick_preempts_after_quantum_when_other_ready() {
        let (mut sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5)]);
        sched.run_next(); // a runs, quantum = 1 tick at 100 Hz
        assert_eq!(sched.tick(), TickOutcome::Preempt(ids[1]));
        // Caller switches; the preempted task is Ready again.
        sched.switch_to(ids[1]);
        assert_eq!(sched.current_id(), Some(ids[1]));
        assert_eq!(sched.task(ids[0]).unwrap().state, TaskState::Ready);
    }

    #[test]
    fn tick_does_not_preempt_when_alone() {
        let (mut sched, ids) = sched_with_tasks(&[("solo", 5)]);
        sched.run_next();
        assert_eq!(sched.tick(), TickOutcome::Continue);
        assert_eq!(sched.current_id(), Some(ids[0]));
        assert_eq!(sched.task(ids[0]).unwrap().state, TaskState::Running);
        // And again: quantum resets, so a lone task runs forever.
        assert_eq!(sched.tick(), TickOutcome::Continue);
        assert_eq!(sched.task(ids[0]).unwrap().ticks_run, 2);
    }

    #[test]
    fn tick_with_no_current_task_is_a_noop() {
        let (mut sched, _) = sched_with_tasks(&[("a", 5)]);
        assert_eq!(sched.tick(), TickOutcome::Continue);
        assert_eq!(sched.ticks(), 1);
    }

    #[test]
    fn yield_moves_to_next_and_back() {
        let (mut sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5)]);
        sched.run_next();
        assert_eq!(sched.yield_now(), Some(ids[1]));
        assert_eq!(sched.yield_now(), Some(ids[0]));
    }

    #[test]
    fn yield_with_no_other_ready_stays() {
        let (mut sched, ids) = sched_with_tasks(&[("solo", 5)]);
        sched.run_next();
        assert_eq!(sched.yield_now(), Some(ids[0]));
        assert_eq!(sched.current_id(), Some(ids[0]));
    }

    #[test]
    fn yield_with_no_current_task_is_none() {
        let mut sched = Scheduler::new();
        assert_eq!(sched.yield_now(), None);
    }

    #[test]
    fn block_and_unblock() {
        let (mut sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5)]);
        sched.run_next(); // a runs
        assert!(sched.block_current(0)); // a blocks on I/O
        assert_eq!(sched.current_id(), None);
        assert_eq!(sched.task(ids[0]).unwrap().state, TaskState::Blocked);
        assert_eq!(sched.task(ids[0]).unwrap().blocked_on, Some(0));
        // Blocked tasks are never picked.
        assert_eq!(sched.pick_next(), Some(ids[1]));
        assert_eq!(sched.run_next(), Some(ids[1]));
        // Unblocking a non-blocked task fails; unblocking a blocked one works.
        assert!(!sched.unblock(ids[1]));
        assert!(sched.unblock(ids[0]));
        assert_eq!(sched.task(ids[0]).unwrap().state, TaskState::Ready);
        assert_eq!(sched.task(ids[0]).unwrap().blocked_on, None);
        assert!(!sched.unblock(TaskId(999)));
    }

    #[test]
    fn exit_current_terminates_and_frees_the_cpu() {
        let (mut sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5)]);
        sched.run_next();
        assert!(sched.exit_current(42));
        assert_eq!(sched.current_id(), None);
        let a = sched.task(ids[0]).unwrap();
        assert_eq!(a.state, TaskState::Terminated);
        assert_eq!(a.exit_code, Some(42));
        assert_eq!(sched.pick_next(), Some(ids[1]));
        // Terminating twice fails; terminated tasks are never picked.
        assert!(!sched.exit(ids[0], 0));
        assert_eq!(sched.run_next(), Some(ids[1]));
    }

    #[test]
    fn exit_last_task_leaves_scheduler_empty() {
        let (mut sched, _ids) = sched_with_tasks(&[("solo", 5)]);
        sched.run_next();
        assert!(sched.exit_current(0));
        assert_eq!(sched.pick_next(), None);
        assert_eq!(sched.run_next(), None);
        assert_eq!(sched.ready_count(), 0);
    }

    #[test]
    fn switch_to_unknown_or_non_ready_is_rejected() {
        let (mut sched, ids) = sched_with_tasks(&[("a", 5)]);
        assert_eq!(sched.switch_to(TaskId(999)), None);
        assert_eq!(sched.current_id(), None);
        sched.run_next();
        // Switching to the already-running task is a no-op success.
        assert_eq!(sched.switch_to(ids[0]), Some(ids[0]));
        assert_eq!(sched.switches(), 1); // only the run_next counted
    }

    #[test]
    fn context_switch_returns_from_and_to_pointers() {
        let (mut sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5)]);
        // No current task yet: from is null, to is a's context.
        let (from, to) = unsafe { sched.context_switch(ids[0]) }.unwrap();
        assert!(from.is_null());
        assert!(!to.is_null());
        assert_eq!(unsafe { &*to }.rip, idle_entry as *const () as u64);
        sched.run_next(); // a runs
        let (from, to) = unsafe { sched.context_switch(ids[1]) }.unwrap();
        assert!(!from.is_null());
        assert!(!to.is_null());
        assert_eq!(unsafe { &*to }.rip, idle_entry as *const () as u64);
        // A terminated task has no switch target.
        sched.exit(ids[1], 0);
        assert!(unsafe { sched.context_switch(ids[1]) }.is_none());
        let _ = other_entry;
    }

    #[test]
    fn context_switch_must_run_before_switch_to() {
        // The kernel's switch sequence is context_switch() THEN
        // switch_to(): switch_to() moves the target to Running, and
        // context_switch() only hands out pointers for Ready tasks.
        // Calling them the other way round must yield no pointers.
        let (mut sched, ids) = sched_with_tasks(&[("a", 5), ("b", 5)]);
        sched.run_next(); // a runs
        sched.switch_to(ids[1]); // b is Running now
        assert!(unsafe { sched.context_switch(ids[1]) }.is_none());
        assert_eq!(sched.current_id(), Some(ids[1]));
    }
}
