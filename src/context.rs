//! Saved CPU context of a task, and the x86_64 context switch.
//!
//! A [`Context`] holds exactly the callee-saved registers the System V
//! ABI says a function must preserve across a call (`rbx`, `rbp`,
//! `r12`–`r15`), plus `rsp`, `rip`, and `rflags`. Caller-saved
//! registers (`rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`–`r11`) are *not*
//! saved: a context switch only ever happens at a call to `switch()`,
//! and the caller of `switch()` has already spilled anything live in
//! those registers (or does not care about them), exactly as with any
//! ordinary function call.
//!
//! # How `switch()` works
//!
//! `switch(from, to)` is the classic stack-based switch:
//!
//! 1. Push the callee-saved registers onto the current stack.
//! 2. Store the current `rsp` into `from.rsp`.
//! 3. Load `rsp` from `to.rsp`.
//! 4. Pop the callee-saved registers off the new stack.
//! 5. `ret` — which pops the return address the *other* task saved
//!    when it last called `switch()`, resuming it right after its own
//!    `switch()` call. (For a task that has never run, that return
//!    address is its entry point — see [`prepare_initial_stack`].)
//!
//! The function therefore "returns" twice in a sense: once into the
//! task being switched away from (later, when it is resumed) and once
//! into the task being switched to (now). Callers must treat it as an
//! ordinary call that may return much later, on a different stack.
//!
//! # Safety
//!
//! * Call with interrupts disabled, from kernel context, on x86_64.
//! * `from` and `to` must point at valid, distinct `Context`s.
//! * `to.rsp` must point at a stack laid out by
//!   [`prepare_initial_stack`] (first run) or saved by a previous
//!   `switch()` call (resumed run). A bogus `rsp` triple-faults.
//! * The switch is **not live-boot-tested** (no QEMU on the build
//!   machine); it is the standard push/pop/ret sequence, but the first
//!   real boot is the proof. The stack *layout* it depends on is
//!   unit-tested below.

use core::arch::asm;

/// Value of RFLAGS for a freshly created task: bit 1 (always 1) and
/// bit 9 (IF, interrupts enabled) set.
pub const INITIAL_RFLAGS: u64 = 0x202;

/// Saved CPU state of one task. `#[repr(C)]` so the field layout is
/// fixed and documented; `switch()` only touches `rsp` (offset 0) via
/// a memory operand, the rest is saved/restored with push/pop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Context {
    /// Stack pointer. Offset 0 — the only field `switch()` addresses.
    pub rsp: u64,
    /// Instruction pointer: entry point on first run, resume address after.
    pub rip: u64,
    pub rbx: u64,
    pub rbp: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rflags: u64,
}

impl Context {
    /// A fresh context for a task whose stack has already been prepared
    /// by [`prepare_initial_stack`]. `initial_rsp` is the value that
    /// function returned; `entry` is the task's entry point.
    pub fn new(entry: u64, initial_rsp: u64) -> Self {
        Context {
            rsp: initial_rsp,
            rip: entry,
            rbx: 0,
            rbp: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
            rflags: INITIAL_RFLAGS,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Context {
            rsp: 0,
            rip: 0,
            rbx: 0,
            rbp: 0,
            r12: 0,
            r13: 0,
            r14: 0,
            r15: 0,
            rflags: INITIAL_RFLAGS,
        }
    }
}

/// Number of `u64` slots [`prepare_initial_stack`] lays out on a new
/// task's stack: six zeroed callee-saved registers + the entry address
/// (the `ret` target).
pub const INITIAL_STACK_SLOTS: usize = 7;

/// Lay out a new task's stack so the first `switch()` to it lands in
/// its entry point.
///
/// The stack grows down from `stack_top` (one past the last byte).
/// Slots are written low-to-high as `r15, r14, r13, r12, rbp, rbx,
/// entry` — the reverse of the pop order in `switch()` — so after
/// `switch()` loads this `rsp` and pops six registers, `ret` pops
/// `entry` and jumps there, with `rsp` back at the (16-byte-aligned)
/// stack top. All six registers start at zero.
///
/// Returns the initial `rsp` (to store in the task's `Context`), or
/// `None` if the stack is too small (< 64 bytes, incl. alignment
/// slack).
///
/// # Safety
///
/// `stack` must point at `stack_size` writable bytes of task-stack
/// memory that no other task is using.
pub unsafe fn prepare_initial_stack_raw(
    stack: *mut u8,
    stack_size: usize,
    entry: u64,
) -> Option<u64> {
    if stack_size < 64 || stack.is_null() {
        return None;
    }
    // 16-byte-align the top down, then reserve the seven slots.
    let top = (stack as usize + stack_size) & !0xF;
    let rsp = top.checked_sub(INITIAL_STACK_SLOTS * 8)?;
    let slots = rsp as *mut u64;
    unsafe {
        // slots[0..6] = r15, r14, r13, r12, rbp, rbx = 0; slots[6] = entry.
        for i in 0..6 {
            slots.add(i).write(0);
        }
        slots.add(6).write(entry);
    }
    Some(rsp as u64)
}

/// Safe wrapper over [`prepare_initial_stack_raw`] for a `&mut [u8]`
/// stack buffer (the form host tests and kernel stack allocation use).
pub fn prepare_initial_stack(stack: &mut [u8], entry: u64) -> Option<u64> {
    let len = stack.len();
    let ptr = stack.as_mut_ptr();
    unsafe { prepare_initial_stack_raw(ptr, len, entry) }
}

/// Switch from the task described by `from` to the task described by
/// `to`. See the module docs for the sequence and the safety contract.
///
/// x86_64 only. On other targets this is a runtime `unimplemented!`
/// (the crate still compiles there so host tooling can use the pure
/// scheduling logic).
#[cfg(target_arch = "x86_64")]
pub unsafe fn switch(from: *mut Context, to: *const Context) {
    unsafe {
        asm!(
            "push rbx",
            "push rbp",
            "push r12",
            "push r13",
            "push r14",
            "push r15",
            "mov [{from}], rsp",
            "mov rsp, [{to}]",
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop rbp",
            "pop rbx",
            "ret",
            from = in(reg) from,
            to = in(reg) to,
            // No `nostack`: this asm pushes/pops the stack by design.
            // No clobbers listed: rbx/rbp/r12-r15 are saved and
            // restored by the asm itself; `from`/`to` live in
            // caller-saved registers the asm does not touch.
        );
    }
}

/// Non-x86_64 fallback: context switching is not implemented.
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn switch(from: *mut Context, to: *const Context) {
    let _ = (from, to);
    unimplemented!("rustos-scheduler context switch is x86_64-only");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_new_has_entry_rsp_and_flags() {
        let ctx = Context::new(0x1000, 0x8000);
        assert_eq!(ctx.rip, 0x1000);
        assert_eq!(ctx.rsp, 0x8000);
        assert_eq!(ctx.rflags, 0x202);
        assert_eq!(ctx.rbx, 0);
        assert_eq!(ctx.r15, 0);
    }

    #[test]
    fn prepare_stack_lays_out_regs_then_entry() {
        // A u64-backed buffer so the stack is at least 8-byte aligned.
        let mut stack = [0u64; 512];
        let buf = unsafe {
            core::slice::from_raw_parts_mut(stack.as_mut_ptr() as *mut u8, 512 * 8)
        };
        let top = buf.as_ptr() as usize + buf.len();
        let entry: u64 = 0xdead_beef;
        let rsp = prepare_initial_stack(buf, entry).expect("stack big enough");
        // rsp is 16-aligned top minus 7 slots.
        let aligned_top = top & !0xF;
        assert_eq!(rsp as usize, aligned_top - 7 * 8);
        // Slots, low to high: r15..rbx = 0, then the entry address.
        let slots = unsafe { core::slice::from_raw_parts(rsp as *const u64, 7) };
        assert_eq!(slots[..6], [0; 6]);
        assert_eq!(slots[6], entry);
        // After switch() pops 6 regs and rets, rsp is back at the top.
        assert_eq!(rsp as usize + 7 * 8, aligned_top);
    }

    #[test]
    fn prepare_stack_rejects_tiny_stacks() {
        let mut small = [0u8; 32];
        assert_eq!(prepare_initial_stack(&mut small, 0x1000), None);
        let mut empty: [u8; 0] = [];
        assert_eq!(prepare_initial_stack(&mut empty, 0x1000), None);
        // Exactly 64 bytes is enough.
        let mut exact = [0u8; 64];
        assert!(prepare_initial_stack(&mut exact, 0x1000).is_some());
    }

    #[test]
    fn initial_stack_slots_constant_matches_layout() {
        assert_eq!(INITIAL_STACK_SLOTS, 7);
        assert_eq!(INITIAL_RFLAGS, 0x202);
    }
}
