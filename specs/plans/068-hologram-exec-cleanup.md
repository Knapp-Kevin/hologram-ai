# Plan 068: hologram-exec Aggressive Cleanup

**Status:** Planned (after Plan 067 completion)
**Created:** 2026-04-16
**Depends on:** Plan 067 (ComputeBackend + ComputeMemory rewrite)

## Problem

hologram-exec has grown to 5700+ lines in tape.rs with:
- Dual-path GPU/CPU execution logic
- GpuBuffer/GpuInput/dispatch_*_chained infrastructure
- 2000+ line dispatch_kernel function with per-kernel-type match arms
- Lazy readback, threshold checks, gpu_bufs parallel array
- Profile counters, debug tracing interleaved with execution
- Multiple execution paths (execute_direct, level-based executor)

## Goal

After hologram-backend absorbs all dispatch logic, hologram-exec becomes
a thin executor crate: tape data structures + execution loop + buffer management.

**Target: tape.rs under 1000 lines.**

## What Moves to hologram-backend

| Code | Current Location | New Location |
|------|-----------------|-------------|
| Per-kernel CPU dispatch (2000+ lines) | tape.rs dispatch_kernel() | CpuBackend::dispatch() |
| Metal dispatch + MSL shaders | backend/metal.rs (1800+ lines) | MetalBackend in hologram-backend |
| WebGPU dispatch | backend/webgpu.rs | WebGpuBackend in hologram-backend |
| Float dispatch (matmul, conv, attention, norm) | float_dispatch/*.rs (~3000 lines) | Shared kernel impls in hologram-backend |
| GpuBuffer, GpuInput, KernelOutput types | backend/mod.rs | hologram-backend lib.rs |
| Hardware detection + thresholds | backend/hardware.rs | hologram-backend |

## What Stays in hologram-exec

| Code | Purpose |
|------|---------|
| TapeKernel enum | Instruction encoding |
| EnumTape struct | Tape data structure (instructions, levels, constants) |
| TapeBuilder | Graph → Tape compilation |
| execute() | Single-path execution loop (~50 lines) |
| BufferArena | Device-agnostic buffer slot management |
| KV cache management | KvCacheState, KvSlotRead/Write |
| Shape resolution | resolve_spatial_dims, resolve_matmul_dims |

## Cleanup Steps

### Step 1: Remove dual-path execution
- Delete dispatch_kernel_gpu() and all GPU chaining logic
- Delete gpu_bufs, has_gpu_inputs, try_gpu conditions
- Delete lazy readback code
- Delete GpuBuffer, GpuInput, dispatch_*_chained from backend/mod.rs

### Step 2: Remove dispatch_kernel
- Delete the entire 2000+ line dispatch_kernel function
- Replace with: `backend.dispatch(&instr.kernel, &inputs, &mut output, &params)`

### Step 3: Remove backend/ module from hologram-exec
- Delete backend/metal.rs, backend/webgpu.rs, backend/cpu.rs
- Delete backend/hardware.rs
- Delete backend/mod.rs
- Add dependency on hologram-backend instead

### Step 4: Move float_dispatch/ to hologram-backend
- Move matmul.rs, conv.rs, attention.rs, norm.rs, etc.
- These become the kernel implementations inside CpuBackend
- Remove float_dispatch/ module from hologram-exec

### Step 5: Simplify execute_direct
- Single loop: build inputs from arena, call backend.dispatch(), store output
- Remove profiling hooks (move to hologram-backend's dispatch)
- Remove eviction logic (move to BufferArena)
- ~50 lines total

### Step 6: Clean up TapeKernel
- Remove unused variants
- Ensure all variants map to hologram-backend ops
- Consider splitting into categories (float, ring, data_movement)

## Expected Result

| Metric | Before | After |
|--------|--------|-------|
| tape.rs lines | 5700+ | <1000 |
| backend/ module | 3000+ lines | Deleted (in hologram-backend) |
| float_dispatch/ module | 3000+ lines | Deleted (in hologram-backend) |
| Total hologram-exec lines | ~12000 | ~3000 |
| Execution paths | 3 (direct, level, parallel) | 1 |
