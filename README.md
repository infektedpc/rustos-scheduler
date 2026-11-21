# rustos-scheduler

Preemptive round-robin scheduler with priorities, timer ticks, and
x86_64 context switching for RustOS. `no_std` (`core` + `alloc` only,
no external crates), so it links straight into `rustos-kernel`.

## Layout

| File | What it is |
|---|---|
| `src/lib.rs` | Crate docs + re-exports |
| `src/context.rs` | `Context` (callee-saved regs + `rsp`/`rip`/`rflags`), `switch()` (push/pop/`ret` context switch), `prepare_initial_stack()` (first-run stack layout) |
| `src/task.rs` | `Task` (id, state, priority, entry, stack, saved context, accounting) |
| `src/scheduler.rs` | `Scheduler`: priority round-robin, `tick()`, `yield_now()`, `block()`/`unblock()`, `exit()`, `switch_to()` / `context_switch()` |
| `src/timer.rs` | Tick bookkeeping: 100 Hz, 10 ms timeslice, uptime, slice boundaries |
| `src/syscall.rs` | Scheduler syscalls, numbers matching `rustos-shell`: `SYS_YIELD = 24`, `SYS_EXIT = 60` (read/write/open/close are recognised but routed elsewhere) |

## Policy

- Highest priority runs first (priority is a `u8`, **higher wins**).
- Within one priority, round-robin: each pick scans from just after
  the last-picked task.
- A running task gets one 10 ms timeslice (1 tick at 100 Hz). When
  `tick()` is called from the timer IRQ and the slice expires with
  another task Ready, it returns `TickOutcome::Preempt(next)`.
- `yield_now()` is the voluntary version (`SYS_YIELD`).
- `block(reason)` parks a task until `unblock()` wakes it;
  `exit(code)` terminates it (`SYS_EXIT`).

## How the kernel wires it in

Prerequisite: the PICs must be remapped first (the kernel's IDT
currently only installs CPU exceptions — breakpoint, double fault,
page fault — and deliberately has no timer/keyboard handlers yet,
because the default IRQ routing collides with exceptions 0–15).
Once IRQ 0 reaches the handler, the wiring is:

```rust
use rustos_scheduler::{Scheduler, Task, TickOutcome, context};

// Once, at boot, after gdt::init() / interrupts::init_idt():
let mut sched = Scheduler::new();
// Stacks live in kernel memory (pages from BootInfoFrameAllocator,
// or 'static arrays). Prepare each stack, then create the task:
let rsp = unsafe { Task::prepare_stack(stack_ptr, STACK_SIZE, entry) }
    .expect("stack too small");
let shell = sched.add_task("shell", 5, rsp, stack_top, STACK_SIZE, entry);
sched.run_next(); // first task starts running

// In the timer IRQ handler (IRQ 0, PIT at 100 Hz):
match sched.tick() {
    TickOutcome::Preempt(next) => {
        // Bookkeeping first, then the register switch:
        let (from, to) = unsafe { sched.context_switch(next) }
            .expect("preempt target vanished");
        sched.switch_to(next);
        if !from.is_null() {
            unsafe { context::switch(from, to) };
        }
        // (If `from` is null this is the first run: just load `to`.)
    }
    TickOutcome::Continue => {}
}

// In the syscall handler (int 0x80 / syscall instruction):
let result = rustos_scheduler::syscall::dispatch_syscall(&mut sched, nr, arg0);
```

The scheduler only *decides* which task runs; the kernel performs
the register switch. That split is what makes the scheduling logic
unit-testable on the host without ever jumping to a fabricated stack.

## Build & test

```sh
cargo test
```

Stable Rust is enough (`core::arch::asm` is stable); no nightly,
no `x86_64-unknown-none` target needed for the tests. `cargo test`
runs the host unit tests: round-robin fairness, priority ordering,
timeslice expiry/preemption, yield, block/unblock, exit, empty-queue
behaviour, stack layout, and the syscall-number agreement with
`rustos-shell`.

## Limitations (honest)

- **Not live-boot-tested.** There is no QEMU on the build machine,
  so no task has ever actually run under this scheduler. The
  context-switch sequence is the standard push/pop/`ret` one and the
  stack layout it depends on is unit-tested, but the first real boot
  (timer IRQ → `tick()` → `switch()`) is still the proof.
- **No SMP.** One CPU, one current task. A second core would need
  per-CPU run queues and locking.
- **No aging / no starvation protection.** A steady stream of
  high-priority Ready tasks can starve lower-priority ones forever.
- **Cooperative tasks can still hog the CPU** if they disable
  interrupts or spin with interrupts off — preemption only happens
  on a timer tick.
- **No sleep/wake timers yet.** `block(reason)` parks a task, but
  nothing wakes a timer-blocked task automatically; the kernel must
  call `unblock()` (e.g. from an I/O completion or a sleep queue it
  owns). A sleep queue is a natural follow-up.
- **No memory isolation between tasks.** All tasks currently share
  the kernel page tables; per-task address spaces (and the
  user/kernel mode split) are follow-up work in `rustos-kernel`.
- **x86_64 only** for the actual switch; the pure scheduling logic
  compiles anywhere.
