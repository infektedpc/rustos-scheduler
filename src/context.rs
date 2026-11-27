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
/// (the `ret` target) + one zeroed padding slot.
///
/// The padding slot is load-bearing, not spare: after `switch()` pops
/// the six registers and `ret`s into the entry, `rsp` must be 8 mod 16
/// (the System V ABI: `rsp + 8` is a multiple of 16 at function
/// entry). Without it `rsp` would land 0 mod 16 and the compiler's
/// stack-frame math — which assumes the ABI alignment — would be off
/// by 8, so a `movaps` on a stack local could #GP on the task's very
/// first run. Seven slots put `rsp` at `top - 56`, i.e. 0 mod 16
/// after the `ret`; eight put it at `top - 8`, i.e. 8 mod 16.
pub const INITIAL_STACK_SLOTS: usize = 8;

/// Lay out a new task's stack so the first `switch()` to it lands in
/// its entry point.
///
/// The stack grows down from `stack_top` (one past the last byte).
/// Slots are written low-to-high as `r15, r14, r13, r12, rbp, rbx,
/// entry, pad` — the reverse of the pop order in `switch()` — so after
/// `switch()` loads this `rsp` and pops six registers, `ret` pops
/// `entry` and jumps there, with `rsp` at `stack_top - 8` (8 mod 16:
/// the ABI entry alignment — see [`INITIAL_STACK_SLOTS`]). All six
/// registers and the pad start at zero.
///
/// Returns the initial `rsp` (to store in the task's `Context`), or
/// `None` if the stack is too small (< 64 bytes, incl. alignment
/// slack) or so misaligned that the laid-out stack would start
/// below the buffer.
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
    // 16-byte-align the top down, then reserve the slots. The
    // laid-out stack must start inside the buffer: a small,
    // misaligned buffer could otherwise push `rsp` below `stack`.
    let top = (stack as usize + stack_size) & !0xF;
    let rsp = top.checked_sub(INITIAL_STACK_SLOTS * 8)?;
    if rsp < stack as usize {
        return None;
    }
    let slots = rsp as *mut u64;
    unsafe {
        // slots[0..6] = r15, r14, r13, r12, rbp, rbx = 0; slots[6] =
        // entry; slots[7] = padding (see INITIAL_STACK_SLOTS).
        for i in 0..6 {
            slots.add(i).write(0);
        }
        slots.add(6).write(entry);
        slots.add(7).write(0);
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

/// The five words the CPU pops on `iretq` when returning to ring 3,
/// in pop order (low address to high): RIP, CS, RFLAGS, RSP, SS.
///
/// A ring-3 task's first entry is not a `ret` into a kernel function
/// but an `iretq` into user mode (see [`prepare_ring3_stack`]): the
/// frame carries the user entry point, the user code/data selectors
/// (RPL 3), RFLAGS with IF set, and the initial user stack pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ring3Frame {
    /// User entry point (the program's ELF `e_entry`).
    pub rip: u64,
    /// User code segment selector, RPL 3 (e.g. 0x33 for GDT index 6 —
    /// the kernel puts user code at index 6, one entry after user
    /// data, so `sysret` can derive it from the STAR base).
    pub cs: u64,
    /// RFLAGS for the first user instruction: [`INITIAL_RFLAGS`]
    /// (bit 1 set, IF set) — the timer must be able to preempt it.
    pub rflags: u64,
    /// Initial user stack pointer.
    pub rsp: u64,
    /// User data segment selector, RPL 3 (e.g. 0x2B for GDT index 5 —
    /// the kernel puts user data at index 5, one entry before user
    /// code, so `sysret` can derive both from the STAR base).
    pub ss: u64,
}

impl Ring3Frame {
    /// Build a ring-3 entry frame. `code_selector`/`data_selector`
    /// are the GDT indices of the user segments (6 and 5 in the
    /// kernel's GDT — code one entry after data, so `sysret` can
    /// derive both from the STAR base); the RPL 3 bits are added
    /// here, so callers pass plain indices.
    pub fn new(entry: u64, user_rsp: u64, code_index: u16, data_index: u16) -> Self {
        Ring3Frame {
            rip: entry,
            cs: ((code_index as u64) << 3) | 3,
            rflags: INITIAL_RFLAGS,
            rsp: user_rsp,
            ss: ((data_index as u64) << 3) | 3,
        }
    }

    /// The frame as five stack slots, in the order `iretq` pops them.
    pub fn as_slots(&self) -> [u64; 5] {
        [self.rip, self.cs, self.rflags, self.rsp, self.ss]
    }
}

/// Number of `u64` slots [`prepare_ring3_stack`] lays out on a new
/// ring-3 task's *kernel* stack: six zeroed callee-saved registers
/// + the `ret` target (the iretq stub) + the five-word iretq frame.
///
/// There is deliberately NO padding slot (unlike
/// [`INITIAL_STACK_SLOTS`]): after `switch()` pops the six registers
/// and `ret`s, `rsp` must point exactly at the frame — the stub is a
/// bare `iretq`, so the ABI entry alignment the padding serves for
/// kernel tasks does not apply here, and a pad would push the frame
/// one slot too far down for the `iretq` to find it.
pub const RING3_STACK_SLOTS: usize = 12;

/// Lay out a ring-3 task's kernel stack so the first `switch()` to
/// it lands in ring 3.
///
/// Slots low-to-high: `r15, r14, r13, r12, rbp, rbx` = 0, then the
/// return address ([`ring3_iretq_stub`]), then the iretq frame
/// ([`Ring3Frame::as_slots`]). `switch()` pops the six registers
/// and `ret`s into the stub with `rsp` pointing at the frame; the
/// stub's `iretq` pops RIP/CS/RFLAGS/RSP/SS and drops to ring 3 at
/// `frame.rip` with `frame.rsp` as the user stack. The same layout
/// works for `enter()` (a terminating switch to a never-run ring-3
/// task): its second half is the same pop/ret.
///
/// Returns the initial `rsp` (to store in the task's `Context`),
/// or `None` if the stack is too small (< 96 bytes, incl.
/// alignment slack) or so misaligned that the laid-out stack would
/// start below the buffer.
///
/// # Safety
///
/// `stack` must point at `stack_size` writable bytes of kernel-stack
/// memory that belong to this task alone and stay alive until the
/// task terminates.
pub unsafe fn prepare_ring3_stack_raw(
    stack: *mut u8,
    stack_size: usize,
    frame: Ring3Frame,
) -> Option<u64> {
    if stack_size < RING3_STACK_SLOTS * 8 || stack.is_null() {
        return None;
    }
    let top = (stack as usize + stack_size) & !0xF;
    let rsp = top.checked_sub(RING3_STACK_SLOTS * 8)?;
    if rsp < stack as usize {
        return None;
    }
    let slots = rsp as *mut u64;
    unsafe {
        for i in 0..6 {
            slots.add(i).write(0);
        }
        slots.add(6).write(ring3_iretq_stub as *const () as u64);
        let frame_slots = frame.as_slots();
        for (i, v) in frame_slots.iter().enumerate() {
            slots.add(7 + i).write(*v);
        }
    }
    Some(rsp as u64)
}

/// Safe wrapper over [`prepare_ring3_stack_raw`] for a `&mut [u8]`
/// stack buffer.
pub fn prepare_ring3_stack(stack: &mut [u8], frame: Ring3Frame) -> Option<u64> {
    let len = stack.len();
    let ptr = stack.as_mut_ptr();
    unsafe { prepare_ring3_stack_raw(ptr, len, frame) }
}

/// Enter ring 3: pop the `iretq` frame the CPU needs to drop to user
/// mode. Reached via `ret` from [`switch`]/[`enter`] when a
/// never-run ring-3 task is first entered: the task's initial stack
/// (see [`prepare_ring3_stack`]) has the frame directly above the
/// return address, so `rsp` points at it when this runs.
///
/// Naked, like [`switch`]: a single `iretq`, no prologue — any push
/// would shift `rsp` off the frame and the `iretq` would pop
/// garbage as RIP. Never returns (it jumps to ring 3).
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "C" fn ring3_iretq_stub() -> ! {
    core::arch::naked_asm!("iretq",);
}

/// Non-x86_64 fallback: no ring-3 entry.
#[cfg(not(target_arch = "x86_64"))]
pub unsafe extern "C" fn ring3_iretq_stub() -> ! {
    unimplemented!("rustos-scheduler ring-3 entry is x86_64-only");
}
///
/// x86_64 only. On other targets this is a runtime `unimplemented!`
/// (the crate still compiles there so host tooling can use the pure
/// scheduling logic).
///
/// This is a `#[naked]` function, and that is load-bearing, not
/// style: the entire contract of `switch()` is the exact stack
/// layout — six callee-saved pushes, store `rsp`, load the new
/// `rsp`, six pops, `ret`. A compiler-generated prologue (even one
/// `push`, for stack alignment) shifts the saved `rsp` by a slot,
/// and the resume `ret` then pops a data value instead of the return
/// address. That is exactly what happened on the first live boot of
/// Batch 7 (2026-11-22): the debug build emitted `push %rax` ahead
/// of the asm, the worker's first yield switched back into a `ret`
/// that jumped to `switch`'s own address (rax held the indirect-call
/// target), and the CPU looped there forever with interrupts off —
/// silent wedge, no fault, no further output. A naked function has
/// no prologue or epilogue by construction, so the asm is the whole
/// function and the layout cannot drift again. The arguments arrive
/// in `rdi`/`rsi` under the C ABI and are read directly; there are
/// no register operands, so the compiler emits nothing at all around
/// the asm. (`enter()` below is naked for the same reason: it reads
/// `rsp` values this function wrote.)
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "C" fn switch(from: *mut Context, to: *const Context) {
    core::arch::naked_asm!(
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "mov [rdi], rsp",
        "mov rsp, [rsi]",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "ret",
    );
}

/// Non-x86_64 fallback: context switching is not implemented.
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn switch(from: *mut Context, to: *const Context) {
    let _ = (from, to);
    unimplemented!("rustos-scheduler context switch is x86_64-only");
}

/// Enter a task's context without saving the current one: load `rsp`
/// from `to`, pop the callee-saved registers, `ret` into the task.
/// For a task that has never run, `ret` lands in its entry point
/// (see [`prepare_initial_stack`]); for a resumed task, right after
/// its own `switch()` call — the same second half as [`switch`].
///
/// Unlike [`switch`], this never returns to the caller: the current
/// context is abandoned, not saved. Use it only when the current
/// task is finished for good (the `exit` syscall: the exiting task
/// is `Terminated` and its stack is never resumed). The caller must
/// already have done the scheduler bookkeeping (outgoing task
/// `Terminated`, incoming task `Running`, `current` moved) and
/// dropped any scheduler lock — the incoming task may take that lock
/// on its first tick, and a lock held by an abandoned task would
/// spin forever.
///
/// Naked, like [`switch`]: it reads `rsp` values `switch()` wrote,
/// so the same no-prologue guarantee applies — and a prologue here
/// would be dead weight on a stack that is abandoned anyway. The
/// target arrives in `rdi` under the C ABI.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
pub unsafe extern "C" fn enter(to: *const Context) -> ! {
    core::arch::naked_asm!(
        "mov rsp, [rdi]",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "ret",
    );
}

/// Non-x86_64 fallback: context switching is not implemented.
#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn enter(to: *const Context) -> ! {
    let _ = to;
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
        // rsp is 16-aligned top minus 8 slots.
        let aligned_top = top & !0xF;
        assert_eq!(rsp as usize, aligned_top - 8 * 8);
        // Slots, low to high: r15..rbx = 0, then the entry address,
        // then the zeroed padding slot.
        let slots = unsafe { core::slice::from_raw_parts(rsp as *const u64, 8) };
        assert_eq!(slots[..6], [0; 6]);
        assert_eq!(slots[6], entry);
        assert_eq!(slots[7], 0);
        // After switch() pops 6 regs and rets, rsp is at top - 8:
        // 8 mod 16, the ABI entry alignment (rsp + 8 is a multiple
        // of 16 when control reaches the entry point).
        let rsp_after_ret = rsp as usize + 7 * 8;
        assert_eq!(rsp_after_ret, aligned_top - 8);
        assert_eq!(rsp_after_ret % 16, 8);
    }

    #[test]
    fn prepare_stack_rejects_tiny_stacks() {
        let mut small = [0u8; 32];
        assert_eq!(prepare_initial_stack(&mut small, 0x1000), None);
        let mut empty: [u8; 0] = [];
        assert_eq!(prepare_initial_stack(&mut empty, 0x1000), None);
        // Exactly 64 bytes is enough — but only when the buffer is
        // 16-byte aligned, so the laid-out stack fits inside it.
        #[repr(align(16))]
        struct Aligned64([u8; 64]);
        let mut exact = Aligned64([0u8; 64]);
        assert!(prepare_initial_stack(&mut exact.0, 0x1000).is_some());
        // A 64-byte window starting 8 mod 16 cannot hold the 8
        // aligned slots: it is rejected instead of writing below
        // the buffer. (The offset is picked from the real address,
        // so the misalignment is deterministic.)
        let mut backing = [0u64; 10];
        let base = backing.as_mut_ptr() as usize;
        let offset = if base % 16 == 0 { 8 } else { 0 };
        let raw = backing.as_mut_ptr() as *mut u8;
        let misaligned = unsafe { core::slice::from_raw_parts_mut(raw.add(offset), 64) };
        assert_eq!(misaligned.as_ptr() as usize % 16, 8);
        assert_eq!(prepare_initial_stack(misaligned, 0x1000), None);
    }

    #[test]
    fn initial_stack_slots_constant_matches_layout() {
        assert_eq!(INITIAL_STACK_SLOTS, 8);
        assert_eq!(INITIAL_RFLAGS, 0x202);
    }

    #[test]
    fn ring3_frame_slots_are_in_iretq_pop_order() {
        // GDT indices 5 (code) and 6 (data), RPL 3 added by the
        // constructor: 0x2B and 0x33 — the exact values the CPU
        // expects in the iretq frame's CS/SS slots.
        let frame = Ring3Frame::new(0x400040, 0x7fffd8, 5, 6);
        assert_eq!(frame.rip, 0x400040);
        assert_eq!(frame.cs, 0x2b);
        assert_eq!(frame.ss, 0x33);
        assert_eq!(frame.rflags, INITIAL_RFLAGS);
        assert_eq!(frame.rsp, 0x7fffd8);
        assert_eq!(frame.as_slots(), [0x400040, 0x2b, 0x202, 0x7fffd8, 0x33]);
    }

    #[test]
    fn ring3_frame_uses_the_kernel_gdt_user_indices() {
        // The kernel's GDT (Batch 11) puts user data at index 5
        // (0x2B) and user code at index 6 (0x33) — code exactly
        // one entry after data, so `sysret` can derive both from
        // the STAR base. The kernel passes
        // (USER_CODE_INDEX, USER_DATA_INDEX) = (6, 5) to the
        // constructor; pin the resulting frame here so a GDT
        // reorder that forgets this call site fails loudly.
        let frame = Ring3Frame::new(0x400000, 0x7fffe8, 6, 5);
        assert_eq!(frame.cs, 0x33);
        assert_eq!(frame.ss, 0x2b);
        assert_eq!(frame.as_slots(), [0x400000, 0x33, 0x202, 0x7fffe8, 0x2b]);
    }

    #[test]
    fn prepare_ring3_stack_lays_out_regs_stub_then_frame() {
        let mut stack = [0u64; 512];
        let buf = unsafe {
            core::slice::from_raw_parts_mut(stack.as_mut_ptr() as *mut u8, 512 * 8)
        };
        let top = buf.as_ptr() as usize + buf.len();
        let frame = Ring3Frame::new(0x400040, 0x7fffd8, 5, 6);
        let rsp = prepare_ring3_stack(buf, frame).expect("stack big enough");
        // rsp is 16-aligned top minus 12 slots, pointing at the
        // first callee-saved slot (r15).
        let aligned_top = top & !0xF;
        assert_eq!(rsp as usize, aligned_top - 12 * 8);
        let slots = unsafe { core::slice::from_raw_parts(rsp as *const u64, 12) };
        // Slots 0..6: the zeroed callee-saved registers.
        assert_eq!(slots[..6], [0; 6]);
        // Slot 6: the return address — the iretq stub.
        assert_eq!(slots[6], ring3_iretq_stub as *const () as u64);
        // Slots 7..12: the iretq frame, in pop order.
        assert_eq!(slots[7..12], [0x400040, 0x2b, 0x202, 0x7fffd8, 0x33]);
        // After switch() pops 6 regs and rets, rsp points exactly
        // at the frame (slot 7): rsp + 7*8 == the frame's address.
        // (No padding slot: a pad would push the frame one slot too
        // far down for the stub's iretq to find it.)
        let rsp_after_ret = rsp as usize + 7 * 8;
        assert_eq!(rsp_after_ret, aligned_top - 5 * 8);
        assert_eq!(unsafe { *(rsp_after_ret as *const u64) }, 0x400040);
    }

    #[test]
    fn prepare_ring3_stack_rejects_tiny_stacks() {
        let frame = Ring3Frame::new(0x400040, 0x7fffd8, 5, 6);
        let mut small = [0u8; 64];
        assert_eq!(prepare_ring3_stack(&mut small, frame), None);
        let mut empty: [u8; 0] = [];
        assert_eq!(prepare_ring3_stack(&mut empty, frame), None);
        // Exactly 96 bytes is enough — but only when the buffer is
        // 16-byte aligned, so the laid-out stack fits inside it.
        #[repr(align(16))]
        struct Aligned96([u8; 96]);
        let mut exact = Aligned96([0u8; 96]);
        assert!(prepare_ring3_stack(&mut exact.0, frame).is_some());
    }

    // Round-trip tests for switch()/enter() on the host.
    //
    // These exercise the exact resume path that wedged the first
    // live boot of Batch 7: a task saved by switch() must resume
    // right after its own switch() call, and enter() must land in a
    // previously-saved context. The stacks are real heap buffers
    // laid out by prepare_initial_stack, and the entries are real
    // functions that switch back — so a layout drift (like the
    // compiler prologue that caused the wedge) crashes or hangs
    // this test instead of silently passing. Each entry loops
    // forever after switching away, so a stray resume spins instead
    // of returning into the padding slot (address 0).
    //
    // The contexts live in statics because the entries take no
    // arguments. Each test has its own set (tests run in parallel),
    // and only its own test touches them.
    const fn zero_context() -> Context {
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

    static ROUNDS_B: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static mut CTX_MAIN_B: Context = zero_context();
    static mut CTX_B: Context = zero_context();

    fn task_b_entry() {
        ROUNDS_B.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        unsafe {
            switch(
                &mut *core::ptr::addr_of_mut!(CTX_B),
                &*core::ptr::addr_of!(CTX_MAIN_B),
            );
        }
        loop {
            core::hint::spin_loop();
        }
    }

    #[test]
    fn switch_round_trip_resumes_the_caller() {
        // 64 KiB stack, so a stray signal handler has room.
        let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 65536]));
        let rsp = prepare_initial_stack(buf, task_b_entry as fn() as u64)
            .expect("stack big enough");
        unsafe {
            CTX_B = Context::new(task_b_entry as fn() as u64, rsp);
            CTX_MAIN_B = zero_context();
        }
        unsafe {
            switch(
                &mut *core::ptr::addr_of_mut!(CTX_MAIN_B),
                &*core::ptr::addr_of!(CTX_B),
            );
        }
        // Resumed here only if B ran and switched back: B ran
        // exactly once, and the resume landed right after this
        // switch() call — not at some other address.
        assert_eq!(ROUNDS_B.load(core::sync::atomic::Ordering::SeqCst), 1);
        // The switch saved this thread's context into CTX_MAIN_B,
        // and B's context into CTX_B when it switched away.
        assert_ne!(unsafe { CTX_MAIN_B.rsp }, 0);
        assert_ne!(unsafe { CTX_B.rsp }, 0);
    }

    static ROUNDS_C: core::sync::atomic::AtomicUsize =
        core::sync::atomic::AtomicUsize::new(0);
    static mut CTX_MAIN_C: Context = zero_context();
    static mut CTX_C: Context = zero_context();

    fn task_c_entry() {
        ROUNDS_C.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        unsafe {
            enter(&*core::ptr::addr_of!(CTX_MAIN_C));
        }
    }

    #[test]
    fn enter_lands_in_a_saved_context() {
        let buf = alloc::boxed::Box::leak(alloc::boxed::Box::new([0u8; 65536]));
        let rsp = prepare_initial_stack(buf, task_c_entry as fn() as u64)
            .expect("stack big enough");
        unsafe {
            CTX_C = Context::new(task_c_entry as fn() as u64, rsp);
            CTX_MAIN_C = zero_context();
        }
        unsafe {
            switch(
                &mut *core::ptr::addr_of_mut!(CTX_MAIN_C),
                &*core::ptr::addr_of!(CTX_C),
            );
        }
        // Resumed here only if C ran and entered this context: C
        // ran exactly once, and it was abandoned (it never returns).
        assert_eq!(ROUNDS_C.load(core::sync::atomic::Ordering::SeqCst), 1);
        assert_ne!(unsafe { CTX_MAIN_C.rsp }, 0);
    }
}
