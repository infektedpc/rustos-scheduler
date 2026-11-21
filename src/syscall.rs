//! Scheduler-related syscalls, matching `rustos-shell/src/syscall.rs`.
//!
//! The shell defines the numbers; this module is the kernel side of
//! the two that touch the scheduler:
//!
//! | nr | name  | effect |
//! |----|-------|--------|
//! | 24 | yield | `Scheduler::yield_now()` — give up the rest of the slice |
//! | 60 | exit  | `Scheduler::exit_current(code)` — terminate this task |
//!
//! The other numbers (read = 0, write = 1, open = 2, close = 3) are
//! *not* scheduler work — they belong to the filesystem / console
//! drivers — so `dispatch_syscall` reports them as
//! `SyscallError::NotSchedulerSyscall` and the kernel's main dispatch
//! table routes them elsewhere. Keeping the numbers in one place (here
//! and in `rustos-shell`) is what stops the two sides drifting apart;
//! the test below asserts the values agree with the shell's table.

use crate::scheduler::Scheduler;
use crate::task::TaskId;

/// read(fd, buf, len) -> bytes read, or negative errno.
pub const SYS_READ: u64 = 0;
/// write(fd, buf, len) -> bytes written, or negative errno.
pub const SYS_WRITE: u64 = 1;
/// open(path, flags) -> fd, or negative errno.
pub const SYS_OPEN: u64 = 2;
/// close(fd) -> 0, or negative errno.
pub const SYS_CLOSE: u64 = 3;
/// yield() -> 0. Voluntarily give up the CPU.
pub const SYS_YIELD: u64 = 24;
/// exit(code) -> never returns.
pub const SYS_EXIT: u64 = 60;

/// Why a syscall could not be dispatched by the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyscallError {
    /// The number is not a scheduler syscall (e.g. read/write/open).
    /// Route it to the filesystem/console handler instead.
    NotSchedulerSyscall,
    /// The number is unknown entirely.
    UnknownSyscall,
    /// The syscall is valid but there is no current task (e.g. yield
    /// called before any task is running).
    NoCurrentTask,
}

/// Result of a dispatched syscall: the value the kernel puts in `rax`
/// on return to userspace, or an error.
pub type SyscallResult = Result<u64, SyscallError>;

/// Human-readable name for a syscall number, or "unknown".
pub fn syscall_name(n: u64) -> &'static str {
    match n {
        SYS_READ => "read",
        SYS_WRITE => "write",
        SYS_OPEN => "open",
        SYS_CLOSE => "close",
        SYS_YIELD => "yield",
        SYS_EXIT => "exit",
        _ => "unknown",
    }
}

/// Dispatch a scheduler syscall against `sched`.
///
/// * `SYS_YIELD` — yields; returns the id of the task now running.
/// * `SYS_EXIT` — terminates the current task with `arg0` as the exit
///   code; returns 0 (the caller never actually resumes — the kernel
///   must switch away immediately after this returns).
/// * `SYS_READ`/`SYS_WRITE`/`SYS_OPEN`/`SYS_CLOSE` — recognised but
///   not scheduler work: `Err(NotSchedulerSyscall)`.
/// * Anything else — `Err(UnknownSyscall)`.
pub fn dispatch_syscall(
    sched: &mut Scheduler,
    nr: u64,
    arg0: u64,
) -> SyscallResult {
    match nr {
        SYS_YIELD => match sched.yield_now() {
            Some(TaskId(id)) => Ok(id),
            None => Err(SyscallError::NoCurrentTask),
        },
        SYS_EXIT => {
            if sched.exit_current(arg0 as i32) {
                Ok(0)
            } else {
                Err(SyscallError::NoCurrentTask)
            }
        }
        SYS_READ | SYS_WRITE | SYS_OPEN | SYS_CLOSE => {
            Err(SyscallError::NotSchedulerSyscall)
        }
        _ => Err(SyscallError::UnknownSyscall),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::Task;

    fn idle_entry() {}

    fn sched_with_two() -> (Scheduler, TaskId, TaskId) {
        let mut sched = Scheduler::new();
        let mut ids = [TaskId(0), TaskId(0)];
        for (i, name) in ["a", "b"].iter().enumerate() {
            let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 8192]));
            let rsp =
                unsafe { Task::prepare_stack(buf.as_mut_ptr(), buf.len(), idle_entry) }
                    .expect("stack");
            let top = buf.as_ptr() as u64 + buf.len() as u64;
            ids[i] = sched.add_task(name, 5, rsp, top, buf.len(), idle_entry);
        }
        (sched, ids[0], ids[1])
    }

    #[test]
    fn numbers_match_the_shell_table() {
        // rustos-shell/src/syscall.rs: read=0, write=1, open=2,
        // close=3, yield=24, exit=60. If either side changes, this
        // test (and the shell's own) must change with it.
        assert_eq!(SYS_READ, 0);
        assert_eq!(SYS_WRITE, 1);
        assert_eq!(SYS_OPEN, 2);
        assert_eq!(SYS_CLOSE, 3);
        assert_eq!(SYS_YIELD, 24);
        assert_eq!(SYS_EXIT, 60);
        assert_eq!(syscall_name(SYS_YIELD), "yield");
        assert_eq!(syscall_name(SYS_EXIT), "exit");
        assert_eq!(syscall_name(999), "unknown");
    }

    #[test]
    fn yield_syscall_switches_task() {
        let (mut sched, a, b) = sched_with_two();
        sched.run_next();
        assert_eq!(sched.current_id(), Some(a));
        assert_eq!(dispatch_syscall(&mut sched, SYS_YIELD, 0), Ok(b.0));
        assert_eq!(sched.current_id(), Some(b));
    }

    #[test]
    fn exit_syscall_terminates_current_task() {
        let (mut sched, a, b) = sched_with_two();
        sched.run_next();
        assert_eq!(dispatch_syscall(&mut sched, SYS_EXIT, 3), Ok(0));
        assert_eq!(sched.task(a).unwrap().exit_code, Some(3));
        assert_eq!(sched.current_id(), None);
        assert_eq!(sched.run_next(), Some(b));
    }

    #[test]
    fn non_scheduler_syscalls_are_routed_elsewhere() {
        let (mut sched, _, _) = sched_with_two();
        sched.run_next();
        for nr in [SYS_READ, SYS_WRITE, SYS_OPEN, SYS_CLOSE] {
            assert_eq!(
                dispatch_syscall(&mut sched, nr, 0),
                Err(SyscallError::NotSchedulerSyscall),
                "nr {nr}"
            );
        }
        assert_eq!(
            dispatch_syscall(&mut sched, 999, 0),
            Err(SyscallError::UnknownSyscall)
        );
    }

    #[test]
    fn yield_and_exit_with_no_current_task_error() {
        let mut sched = Scheduler::new();
        assert_eq!(
            dispatch_syscall(&mut sched, SYS_YIELD, 0),
            Err(SyscallError::NoCurrentTask)
        );
        assert_eq!(
            dispatch_syscall(&mut sched, SYS_EXIT, 0),
            Err(SyscallError::NoCurrentTask)
        );
    }
}
