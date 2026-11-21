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
}
