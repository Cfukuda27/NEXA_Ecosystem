#!/usr/bin/env python3
import sys
import struct
from axiom_ir import (parse_axm, eliminate_dead, jumptable_labels,
                      optimize_common, fold_immediates, merge_literal_writes)

REG_MAP = {
    "r0": "rax",
    "r1": "rbx",
    "r2": "rcx",
    "r3": "rdx",
    "r4": "rsi",
    "r5": "rdi",
}

# ---------- Encoding helpers ----------

# Smallest correct encoding to materialize a 64-bit immediate, chosen by range:
#
#   imm == 0                 ->  xor r32, r32        (2 bytes)
#       Writing a 32-bit register zero-extends into the full 64-bit register
#       (x86-64 rule), so `xor eax,eax` clears all of rax. No REX needed.
#
#   1 <= imm <= 0x7FFFFFFF   ->  mov r32, imm32       (5 bytes, B8+rd)
#       Also zero-extends to 64 bits. For non-negative values < 2^31 this is
#       bit-for-bit identical to the old REX.W C7 form's result, just shorter.
#
#   0x80000000 <= imm <= 0xFFFFFFFF  ->  mov r64, imm32  (7 bytes, REX.W C7 /0)
#       MUST stay on the sign-extending form: the old codegen used C7 across
#       the whole 0..0xFFFFFFFF range, so a value with bit 31 set sign-extends
#       to a negative 64-bit value (e.g. 0xFFFFFFFF -> -1). Preserve that.
#
#   imm > 0xFFFFFFFF (or out of the unsigned-32 range)  ->  mov r64, imm64
#       Full REX.W B8+rd imm64 (10 bytes).
#
# instr_size_mov_imm() below derives the first-pass size by measuring this very
# function, so the size model and the emitted bytes can never disagree.
_IMM_XOR = {        # xor r32, r32  (31 /r, modrm = C0|reg<<3|reg)
    "rax": b"\x31\xC0", "rcx": b"\x31\xC9", "rdx": b"\x31\xD2",
    "rbx": b"\x31\xDB", "rsi": b"\x31\xF6", "rdi": b"\x31\xFF",
}
_IMM_MOV32 = {      # mov r32, imm32  (B8+rd)
    "rax": b"\xB8", "rcx": b"\xB9", "rdx": b"\xBA",
    "rbx": b"\xBB", "rsi": b"\xBE", "rdi": b"\xBF",
}
_IMM_MOV64C7 = {    # mov r64, imm32 sign-extended  (REX.W C7 /0)
    "rax": b"\x48\xC7\xC0", "rcx": b"\x48\xC7\xC1", "rdx": b"\x48\xC7\xC2",
    "rbx": b"\x48\xC7\xC3", "rsi": b"\x48\xC7\xC6", "rdi": b"\x48\xC7\xC7",
}
_IMM_MOV64B8 = {    # mov r64, imm64  (REX.W B8+rd)
    "rax": b"\x48\xB8", "rcx": b"\x48\xB9", "rdx": b"\x48\xBA",
    "rbx": b"\x48\xBB", "rsi": b"\x48\xBE", "rdi": b"\x48\xBF",
}

def encode_mov_imm64(reg, imm):
    if imm < 0:
        imm &= 0xFFFFFFFFFFFFFFFF   # two's-complement: a negative loads as its
        #                            full 64-bit pattern (e.g. -1 -> 0xFFFF..FF)
    if imm == 0:
        if reg not in _IMM_XOR:
            raise ValueError(f"Unsupported reg for xor zero: {reg}")
        return _IMM_XOR[reg]
    if 0 < imm <= 0x7FFFFFFF:
        if reg not in _IMM_MOV32:
            raise ValueError(f"Unsupported reg for mov imm32: {reg}")
        return _IMM_MOV32[reg] + struct.pack("<I", imm)
    if imm <= 0xFFFFFFFF:
        # bit 31 may be set: use the 32-bit-dest form (B8+rd), which ZERO-extends
        # into the full 64-bit register, so an unsigned value like 0xDEADBEEF
        # stays positive (0x00000000DEADBEEF) instead of being sign-extended to a
        # negative i64. This matches encode_value's unsigned bit patterns.
        if reg not in _IMM_MOV32:
            raise ValueError(f"Unsupported reg for mov imm32 (zx): {reg}")
        return _IMM_MOV32[reg] + struct.pack("<I", imm)
    if reg not in _IMM_MOV64B8:
        raise ValueError(f"Unsupported reg for mov imm64: {reg}")
    return _IMM_MOV64B8[reg] + struct.pack("<Q", imm)


def instr_size_mov_imm(imm):
    """Byte size of the MOV-immediate encoding — measured from the encoder
    itself so the first-pass size and the second-pass bytes always agree."""
    return len(encode_mov_imm64("rax", imm))


def encode_sys_r0_r1():
    # exit(0): mov rax,60 ; mov rdi,0 ; syscall
    code  = encode_mov_imm64("rax", 60)   # 7 bytes
    code += encode_mov_imm64("rdi", 0)    # 7 bytes
    code += b"\x0F\x05"                   # 2 bytes  → total 16
    return code

# `abort` lowers to a clean exit syscall with a distinct, non-zero status.
# 134 is the conventional "aborted" exit code (128 + SIGABRT), but this is a
# real exit_group syscall — a controlled termination, NOT an actual signal —
# so the program ends deterministically instead of faulting. One-line change
# if a different status is preferred.
ABORT_EXIT_CODE = 134

def encode_abort():
    # exit(ABORT_EXIT_CODE): mov rax,60 ; mov rdi,<code> ; syscall
    code  = encode_mov_imm64("rax", 60)               # 7 bytes
    code += encode_mov_imm64("rdi", ABORT_EXIT_CODE)  # 7 bytes  (code < 2^32)
    code += b"\x0F\x05"                               # 2 bytes  → total 16
    return code

# When True, the 123-odd identical `write(1,buf,len)` syscall sequences are not
# emitted inline at every print site. Instead a single shared stub is emitted
# once and each site becomes a 5-byte `call __write_stub` (vs the 9-byte inline
# form) — classic code outlining, ~4 bytes saved per call. WRITE is already a
# register-clobbering barrier, so calling a stub is semantically identical.
OUTLINE_WRITE = True
_WRITE_STUB_LABEL = "__write_stub"

# Data-section minimization (both safe & backend-local):
#   DEDUP_DATA      — identical read-only .bytes blobs (e.g. the newline emitted
#                     once per println) share a single copy. Strings are never
#                     written, so a shared address is indistinguishable.
#   DROP_DEAD_DATA  — scalar slots that no surviving instruction loads, stores,
#                     or takes the address of are not emitted at all. Purely
#                     unreferenced, so removing them changes no behaviour.
DEDUP_DATA = True
DROP_DEAD_DATA = True

# MERGE_LITERAL_WRITES — adjacent string-literal prints each pay a full
# LEA r4,buf ; LDR r3,len ; (call) WRITE triple. A run of N back-to-back literal
# writes with no label/branch between them produces identical stdout bytes if
# their payloads are concatenated into one blob and emitted as a single WRITE,
# saving (N-1) triples (~15 bytes each). Pure compile-time, semantics-preserving:
# the bytes written, and their order, are unchanged. A label/branch between two
# writes ends the run, so no reachable entry point is ever skipped.
MERGE_LITERAL_WRITES = True

# FOLD_IMMEDIATES — the IR materializes every constant operand with its own
# `LDR r1,#imm` before a register-register `OP r0,r1`. x86 has immediate-operand
# forms for all of these (except DIV, which has no immediate divisor), so the
# load is folded directly into the op: `OP rax,imm`. One fewer instruction and
# one fewer register dependency per fold — a win on size AND speed. Only folded
# when r1 is dead afterwards (it always is in this IR: r1 is reloaded before
# every use) and the immediate fits the instruction's sign-extended field.
FOLD_IMMEDIATES = True

# PACK_SCALARS — read the front-end's __typemap sidecar and pack each scalar slot
# to its true width (i8/c8/bool->1, i16/c16->2, i32/c32->4, i64/ptr->8). Floats
# stay 8 (their bits live in 64-bit GP regs here); arrays/refs are never listed,
# so they stay 8 too. Loads sign-extend signed ints and zero-extend chars/bools,
# so a narrow slot reconstructs the exact 64-bit value — same result as before,
# fewer bytes of RAM. This is the data-section RAM floor for data-heavy programs;
# it is page-granular, so it only changes committed RAM when the data crosses a
# page boundary. Memory-safety is unchanged: each store touches exactly its slot.
PACK_SCALARS = True

# nexa type -> (byte width, kind) where kind 's'=signed int, 'u'=zero-extended
# (char/bool/ptr), 'f'=float. Floats and pointers stay 8 (no narrowing).
_TYPE_WS = {
    "i8": (1, "s"), "i16": (2, "s"), "i32": (4, "s"), "i64": (8, "s"),
    "c8": (1, "u"), "c16": (2, "u"), "c32": (4, "u"), "c64": (8, "u"),
    "bool": (1, "u"), "ptr": (8, "u"),
    "f8": (8, "f"), "f16": (8, "f"), "f32": (8, "f"), "f64": (8, "f"),
}

# ---- width-aware memory access (RIP-relative 'far' and rbp-relative 'near') ----
# modrm reg field selects the GP register; rm=101 is RIP-relative (mod 00) or
# [rbp+disp8] (mod 01). The same modrm works for 8/16/32/64-bit forms; only the
# opcode/prefix changes. 16/8-bit forms keep the value's full 64 bits correct via
# movsx/movzx on load and a width-matched mov on store.
def _store_bytes(reg, width, near_disp, rel32):
    num = _X86NUM[reg]
    far = near_disp is None
    modrm = bytes([(num << 3) | (0x05 if far else 0x45)])
    tail = struct.pack("<i", rel32) if far else struct.pack("<b", near_disp)
    if width == 8: return b"\x48\x89" + modrm + tail          # mov [m], r64
    if width == 4: return b"\x89" + modrm + tail              # mov [m], r32
    if width == 2: return b"\x66\x89" + modrm + tail          # mov [m], r16
    rex = b"\x40" if num >= 4 else b""                         # sil/dil need REX
    return rex + b"\x88" + modrm + tail                       # mov [m], r8

def _load_bytes(reg, width, kind, near_disp, rel32):
    # Extension is WIDTH-based, to reproduce exactly what the wide (8-byte) slot
    # held: the front-end's LDR materializes values via encode_mov_imm64, which
    # zero-extends anything below 2^31 and sign-extends only at the 32-bit
    # boundary. So 1/2-byte slots zero-extend (movzx); 4-byte slots sign-extend
    # at 32 (movsxd). 'kind' is unused here — type signedness does not change the
    # stored bit pattern, only the width does.
    num = _X86NUM[reg]
    far = near_disp is None
    modrm = bytes([(num << 3) | (0x05 if far else 0x45)])
    tail = struct.pack("<i", rel32) if far else struct.pack("<b", near_disp)
    if width == 8: return b"\x48\x8B" + modrm + tail                       # mov r64,[m]
    if width == 4: return b"\x48\x63" + modrm + tail                       # movsxd r64,[m]
    if width == 2: return b"\x48\x0F\xB7" + modrm + tail                   # movzx r64,word[m]
    return b"\x48\x0F\xB6" + modrm + tail                                  # movzx r64,byte[m]

# Reserve rbp as a pointer into the data section. Slots within a signed 8-bit
# displacement of rbp are accessed as [rbp+disp8] (4 B) instead of RIP-relative
# (7 B). Pure addressing change — the effective address is identical, so
# behaviour, the &-relocation model, and the manifest are all unaffected.
BASE_REG_ADDR = True

def encode_write_stub():
    # the shared write(1,rsi,rdx) body + ret, reached only via `call`
    return encode_write() + b"\xC3"

# WRITE: write(1, rsi, rdx) — fd=stdout. rsi (buffer addr) and rdx (length)
# must already be loaded (via LEA r4 / LDR r3). syscall also clobbers rcx and
# r11 per the kernel ABI, but our codegen always reloads operands, so that is
# harmless. Same 16-byte shape as the other syscalls.
def encode_write():
    code  = encode_mov_imm64("rax", 1)   # mov eax,1  (5)  sys_write == 1
    code += b"\x89\xC7"                   # mov edi,eax (2) fd=stdout, reuse the 1
    code += b"\x0F\x05"                   # syscall    (2)  -> total 9
    return code

# PUSH/POP single-register, 1 byte each (0x50+rd / 0x58+rd) for the low 8
# GPRs. Used by the inline itoa to reverse the extracted digits.
_PUSH_POP_RD = {"rax": 0, "rcx": 1, "rdx": 2, "rbx": 3,
                "rsp": 4, "rbp": 5, "rsi": 6, "rdi": 7}

def encode_push(reg):
    return bytes([0x50 + _PUSH_POP_RD[reg]])

def encode_pop(reg):
    return bytes([0x58 + _PUSH_POP_RD[reg]])

# READ: read(0, rsi, rdx) — fd=stdin. rsi (buffer addr) and rdx (count) must
# already be loaded (LEA r4 / LDR r3). Bytes read are returned in rax (ignored
# by input()). Same 16-byte shape as WRITE.
def encode_read():
    code  = encode_mov_imm64("rax", 0)   # 7 bytes  sys_read
    code += encode_mov_imm64("rdi", 0)   # 7 bytes  fd = stdin
    code += b"\x0F\x05"                  # 2 bytes  syscall  → total 16
    return code

# I2F: convert the signed integer in rax to an IEEE-754 double, leaving the
# 64-bit bit pattern back in rax.  cvtsi2sd xmm0, rax ; movq rax, xmm0.
def encode_i2f():
    return (b"\xF2\x48\x0F\x2A\xC0"   # cvtsi2sd xmm0, rax   (5 bytes)
            + b"\x66\x48\x0F\x7E\xC0") # movq rax, xmm0       (5 bytes) → 10

# F2I: truncate the IEEE double in rax (bit pattern) toward zero to a signed
# integer, leaving the integer in rax.  movq xmm0,rax ; cvttsd2si rax,xmm0.
def encode_f2i():
    return (b"\x66\x48\x0F\x6E\xC0"   # movq xmm0, rax        (5 bytes)
            + b"\xF2\x48\x0F\x2C\xC0") # cvttsd2si rax, xmm0   (5 bytes) → 10

# Double-precision arithmetic on the bit patterns in rax (r0) and rbx (r1),
# result bits back in rax.  Each: load both into xmm, op, store back.
def _encode_fbinop(op3):
    return (b"\x66\x48\x0F\x6E\xC0"   # movq xmm0, rax
            + b"\x66\x48\x0F\x6E\xCB" # movq xmm1, rbx
            + op3                      # <op>sd xmm0, xmm1   (4 bytes)
            + b"\x66\x48\x0F\x7E\xC0") # movq rax, xmm0       → 19 bytes
def encode_fadd(): return _encode_fbinop(b"\xF2\x0F\x58\xC1")  # addsd
def encode_fsub(): return _encode_fbinop(b"\xF2\x0F\x5C\xC1")  # subsd
def encode_fmul(): return _encode_fbinop(b"\xF2\x0F\x59\xC1")  # mulsd
def encode_fdiv(): return _encode_fbinop(b"\xF2\x0F\x5E\xC1")  # divsd

# LDRB dst, base, index:  movzx dst, byte [base + index]   (zero-extended)
_LDRB_RNUM = {"rax": 0, "rcx": 1, "rdx": 2, "rbx": 3,
              "rsp": 4, "rbp": 5, "rsi": 6, "rdi": 7}
def encode_ldrb(dst, base, index):
    d = _LDRB_RNUM[dst]; b = _LDRB_RNUM[base]; i = _LDRB_RNUM[index]
    modrm = 0x04 | (d << 3)        # mod=00, reg=dst, rm=100 (SIB follows)
    sib   = (i << 3) | b           # scale=1, index, base
    return b"\x48\x0F\xB6" + bytes([modrm, sib])

# STORB src, base, index:  mov byte [base + index], src_low8
def encode_storb(src, base, index):
    s = _LDRB_RNUM[src]; b = _LDRB_RNUM[base]; i = _LDRB_RNUM[index]
    modrm = 0x04 | (s << 3)
    sib   = (i << 3) | b
    return b"\x88" + bytes([modrm, sib])

# LDRQ dst, base, index:  mov dst, [base + index*8]   (64-bit, SIB scale=8)
# This is the runtime array-element load: every nexus slot is 8 bytes, so the
# element address is base + (element index)*8. scale=8 lets the CPU do the *8.
# Only r0..r5 (rax/rbx/rcx/rdx/rsi/rdi, all reg# < 8) are ever used here, so a
# bare REX.W (0x48) suffices and none of them is rsp(4)/rbp(5), avoiding the
# SIB no-index / disp32-base special cases.
def encode_ldrq(dst, base, index):
    d = _LDRB_RNUM[dst]; b = _LDRB_RNUM[base]; i = _LDRB_RNUM[index]
    modrm = 0x04 | (d << 3)              # mod=00, reg=dst, rm=100 (SIB follows)
    sib   = (0b11 << 6) | (i << 3) | b   # scale=8, index, base
    return b"\x48\x8B" + bytes([modrm, sib])

# STORQ src, base, index:  mov [base + index*8], src   (64-bit, SIB scale=8)
def encode_storq(src, base, index):
    s = _LDRB_RNUM[src]; b = _LDRB_RNUM[base]; i = _LDRB_RNUM[index]
    modrm = 0x04 | (s << 3)
    sib   = (0b11 << 6) | (i << 3) | b
    return b"\x48\x89" + bytes([modrm, sib])

def encode_mov_r0_r1():
    return b"\x48\x89\xD8"   # mov rax, rbx

# x86-64 register numbers for the low-8 registers this backend uses.
_X86NUM = {"rax": 0, "rcx": 1, "rdx": 2, "rbx": 3, "rsi": 6, "rdi": 7}

def encode_mov_rr(dst, src):
    """mov <dst>, <src> for any pair of the mapped registers — REX.W 89 /r,
    always 3 bytes. encode_mov_r0_r1() is just the rax<-rbx special case.
    Generalizing this (the lowering previously NOP'd anything but rax<-rbx) is
    what lets the register allocator shuttle values between the rax/rbx ALU
    pair and the rcx/rdx/rsi 'home' registers."""
    modrm = 0xC0 | (_X86NUM[src] << 3) | _X86NUM[dst]
    return b"\x48\x89" + bytes([modrm])

def encode_add_r0_r1():
    return b"\x48\x01\xD8"   # add rax, rbx

def encode_sub_r0_r1():
    return b"\x48\x29\xD8"   # sub rax, rbx

def encode_mul_r0_r1():
    return b"\x48\xF7\xEB"   # imul rbx

def encode_div_r0_r1():
    return b"\x48\x99" + b"\x48\xF7\xFB"  # cqo ; idiv rbx

def encode_cmp_r0_r1():
    return b"\x48\x39\xD8"   # cmp rax, rbx

# setcc al ; movzx rax, al   — materialize condition as 0/1 in rax.
# setcc opcodes (0F 9x C0 = set__ al):
#   setl  0F 9C   setg  0F 9F   setle 0F 9E   setge 0F 9D
#   sete  0F 94   setne 0F 95
_MOVZX_RAX_AL = b"\x48\x0F\xB6\xC0"   # movzx rax, al

def _setcc(opcode2):
    return bytes([0x0F, opcode2, 0xC0]) + _MOVZX_RAX_AL

def encode_setlt(): return _setcc(0x9C)   # signed <
def encode_setgt(): return _setcc(0x9F)   # signed >
def encode_setle(): return _setcc(0x9E)   # signed <=
def encode_setge(): return _setcc(0x9D)   # signed >=
def encode_seteq(): return _setcc(0x94)   # ==
def encode_setne(): return _setcc(0x95)   # !=

# Logical combine of two 0/1 values in rax, rbx -> 0/1 in rax.
# AND: and rax,rbx ; (result already 0/1 since inputs are 0/1)
# OR : or  rax,rbx
# XOR: xor rax,rbx
def encode_setand():
    return b"\x48\x21\xD8"   # and rax, rbx
def encode_setor():
    return b"\x48\x09\xD8"   # or rax, rbx
def encode_setxor():
    return b"\x48\x31\xD8"   # xor rax, rbx

# Shifts: x86 variable-count shift uses CL (low byte of rcx) as the count.
# We move the count (rbx) into rcx, then shl/sar rax, cl.
#   mov rcx, rbx   48 89 D9
#   shl rax, cl    48 D3 E0
#   sar rax, cl    48 D3 F8   (arithmetic / sign-preserving right shift)
def encode_shl_r0_r1():
    return b"\x48\x89\xD9" + b"\x48\xD3\xE0"   # mov rcx,rbx ; shl rax,cl
def encode_shr_r0_r1():
    return b"\x48\x89\xD9" + b"\x48\xD3\xF8"   # mov rcx,rbx ; sar rax,cl

# ---- immediate-operand forms:  OP rax, imm  (dst is always r0/rax) ----
# Folding `LDR r1,#imm ; OP r0,r1` into a single `OP rax,imm` removes the load
# (smaller AND fewer micro-ops/register deps -> faster). imm8 forms (48 83 /n ib)
# are 4 bytes; the rax-special id forms (48 <op> id) are 6. The immediate is
# sign-extended to 64 bits in every form, matching the old mov-then-op result.
def _alu_imm(ext, rax_op, imm):
    if -128 <= imm <= 127:
        return b"\x48\x83" + bytes([0xC0 | (ext << 3)]) + struct.pack("<b", imm)
    return b"\x48" + bytes([rax_op]) + struct.pack("<i", imm)
def encode_add_imm(imm): return _alu_imm(0, 0x05, imm)   # add rax, imm
def encode_or_imm(imm):  return _alu_imm(1, 0x0D, imm)   # or  rax, imm
def encode_and_imm(imm): return _alu_imm(4, 0x25, imm)   # and rax, imm
def encode_sub_imm(imm): return _alu_imm(5, 0x2D, imm)   # sub rax, imm
def encode_xor_imm(imm): return _alu_imm(6, 0x35, imm)   # xor rax, imm
def encode_cmp_imm(imm): return _alu_imm(7, 0x3D, imm)   # cmp rax, imm
def encode_mul_imm(imm):                                  # imul rax, rax, imm
    if -128 <= imm <= 127:
        return b"\x48\x6B\xC0" + struct.pack("<b", imm)
    return b"\x48\x69\xC0" + struct.pack("<i", imm)
def encode_shl_imm(imm):
    return b"\x48\xD1\xE0" if imm == 1 else b"\x48\xC1\xE0" + struct.pack("<B", imm & 0xFF)
def encode_shr_imm(imm):  # arithmetic (sign-preserving) right shift
    return b"\x48\xD1\xF8" if imm == 1 else b"\x48\xC1\xF8" + struct.pack("<B", imm & 0xFF)

# Dispatch table + foldability: the folded value must be representable as the
# instruction's sign-extended imm32 (shifts: 0..63), else keep the mov-then-op.
_ALU_IMM = {"ADD": encode_add_imm, "SUB": encode_sub_imm, "MUL": encode_mul_imm,
            "CMP": encode_cmp_imm, "SETAND": encode_and_imm, "SETOR": encode_or_imm,
            "SETXOR": encode_xor_imm, "SHL": encode_shl_imm, "SHR": encode_shr_imm}

# sizes
SETCC_SIZE = 3 + 4   # setcc al (3) + movzx rax,al (4) = 7
CMP_SIZE   = 3
LOGIC_SIZE = 3
SHIFT_SIZE = 6       # mov rcx,rbx (3) + shift rax,cl (3)

# ---------- IR ----------

class Instr:
    def __init__(self, kind, args, size):
        self.kind = kind
        self.args = args
        self.size = size
        self.short = None   # set on relaxable branches: minimal (rel8) size
        self.long  = None   # set on relaxable branches: full (rel32) size


# Relaxable control-transfer ops and their (short rel8, long rel32) byte sizes.
# JZ/JNZ include the leading `test rax,rax` (3 bytes); the jcc itself is the
# part that shrinks from a 6-byte rel32 form to a 2-byte rel8 form.
_RELAX_SIZES = {
    "JMP": (2, 5),   # EB rel8           / E9 rel32
    "JA":  (2, 6),   # 77 rel8           / 0F 87 rel32
    "JZ":  (5, 9),   # test + 74 rel8    / test + 0F 84 rel32
    "JNZ": (5, 9),   # test + 75 rel8    / test + 0F 85 rel32
}

def _mk_branch(kind, args):
    """Build a relaxable branch instruction, optimistically sized to its short
    (rel8) form. _relax_branches() will grow individual branches to rel32 only
    where the target is out of ±127 range."""
    s, l = _RELAX_SIZES[kind]
    ins = Instr(kind, args, s)
    ins.short, ins.long = s, l
    return ins


def _relax_branches(text):
    """Branch relaxation to a fixpoint: every JMP/JZ/JNZ/JA starts in its 2-byte
    (rel8) form; any whose target lands outside the signed-8-bit displacement is
    grown to the rel32 form. Sizes only ever grow, so this converges. This is
    exactly what a hand assembler does — near jumps cost 2 bytes, not 5/6/9."""
    while True:
        off, label_off, pos = 0, {}, []
        for inst in text:
            pos.append(off)
            if inst.kind == "LABEL":
                label_off[inst.args[0]] = off
            else:
                off += inst.size
        grow = []
        for idx, inst in enumerate(text):
            if inst.kind in _RELAX_SIZES and inst.size == inst.short:
                tgt = inst.args[0]
                if tgt in label_off:
                    rel = label_off[tgt] - (pos[idx] + inst.size)
                    if not (-128 <= rel <= 127):
                        grow.append(idx)
        if not grow:
            return
        for idx in grow:
            text[idx].size = text[idx].long

# ---------- Loop register allocation (x86-only, opt-in) ----------
#
# Keeps a loop's hottest scalar variables resident in the otherwise-idle home
# registers (r2=rcx, r3=rdx, r4=rsi) across iterations, so per-iteration reads
# and writes hit registers instead of memory. The ALU stays on r0/r1 (rax/rbx);
# we only rewrite the variable *accesses*:
#     LDR rX, V   ->  MOV rX, home[V]
#     STR rX, V   ->  MOV home[V], rX
# plus one entry-load before the loop header and a store-back at each loop exit
# (so memory is correct for code after the loop). This is primarily a *speed*
# optimization; the byte-size effect is small and never negative (we only
# promote when in-loop savings cover the entry/exit overhead).
#
# Correctness is guarded by a strict, conservative pattern match — anything that
# doesn't clearly fit is left exactly as the stock codegen emitted it. The
# manifest memory-safety model still applies unchanged: MOV propagates the
# manifest, and the entry-load / store-back are ordinary LDR/STR.
REG_ALLOC = True
REG_ALLOC_DEBUG = False   # set True to log each loop's promote/skip decision

# Body ops the allocator understands. Anything else in a loop body -> skip it.
_RA_OK_OPS = {"LABEL", "LDR", "STR", "MOV", "ADD", "SUB", "MUL", "DIV", "CMP",
              "SETLT", "SETGT", "SETLE", "SETGE", "SETEQ", "SETNE",
              "JZ", "JNZ", "JA", "JMP", "SHL", "SHR"}
_RA_BRANCH = {"JZ", "JNZ", "JA", "JMP"}
_RA_HOMES = ["r2", "r3", "r4"]   # rcx, rdx, rsi


def _ra_is_slot(tok):
    return isinstance(tok, str) and not tok.startswith("#") and not tok.startswith("__")


def allocate_loop_registers(ir):
    if not REG_ALLOC:
        return ir
    ir = [(op, list(a) if isinstance(a, (list, tuple)) else a) for op, a in ir]
    processed = set()
    while True:
        edit = _ra_try_one_loop(ir, processed)
        if edit is None:
            return ir
        ir = edit


def _ra_try_one_loop(ir, processed):
    n = len(ir)
    label_idx = {a: i for i, (op, a) in enumerate(ir) if op == "LABEL"}
    # global branch targets, by branch-instruction index
    branches = [(i, ir[i][1][0]) for i in range(n)
                if ir[i][0] in _RA_BRANCH and ir[i][1]]

    for bj, (op, a) in enumerate(ir):
        if op != "JMP" or not a:
            continue
        tgt = a[0]
        H = label_idx.get(tgt)
        if H is None or H >= bj:            # need an unconditional *back*-edge
            continue
        if tgt in processed:
            continue
        body = range(H + 1, bj)             # instructions between header and back-edge

        # ---- whitelist + no-barrier + no nested back-edge ----
        ok = True
        inner_labels = {ir[i][1] for i in body if ir[i][0] == "LABEL"}
        inner_labels.add(tgt)
        for i in body:
            o = ir[i][0]
            if o not in _RA_OK_OPS:
                ok = False; break
            if o in _RA_BRANCH:
                t = ir[i][1][0]
                ti = label_idx.get(t)
                if ti is not None and H <= ti <= bj and ti < i:
                    ok = False; break       # nested/backward internal branch
        if not ok:
            processed.add(tgt); continue

        # ---- single entry: nothing outside the loop may jump to any inner label ----
        if any((bi < H or bi > bj) and bt in inner_labels for bi, bt in branches):
            processed.add(tgt); continue

        # ---- exits: branches in body to labels outside [H..bj] ----
        exit_tgts = set()
        for i in body:
            if ir[i][0] in _RA_BRANCH:
                t = ir[i][1][0]
                if t not in inner_labels:
                    exit_tgts.add(t)
        # every exit label must be private to this loop (only this loop jumps to
        # it) and must not be an entry symbol
        bad = False
        for E in exit_tgts:
            if E in ("main",) or E.startswith("fn_") or E.startswith("isr_"):
                bad = True; break
            if any((bi < H or bi > bj) and bt == E for bi, bt in branches):
                bad = True; break
            if E not in label_idx:
                bad = True; break
        if bad:
            processed.add(tgt); continue

        # ---- available home registers ----
        used_regs = set()
        has_muldiv = has_shift = False
        for i in body:
            o, aa = ir[i]
            for x in (aa if isinstance(aa, list) else []):
                if x in _RA_HOMES:
                    used_regs.add(x)
            if o in ("MUL", "DIV"): has_muldiv = True
            if o in ("SHL", "SHR"): has_shift = True
        homes = [r for r in _RA_HOMES if r not in used_regs]
        if has_muldiv and "r3" in homes:
            homes.remove("r3")              # imul/idiv clobber rdx
        if has_shift and "r2" in homes:
            homes.remove("r2")              # shift count uses rcx
        if not homes:
            processed.add(tgt); continue

        # ---- candidate slots: every body access is LDR/STR via r0/r1 ----
        from collections import Counter
        cnt = Counter()
        disq = set()
        for i in body:
            o, aa = ir[i]
            if o == "LDR" and len(aa) == 2 and _ra_is_slot(aa[1]):
                if aa[0] in ("r0", "r1"): cnt[aa[1]] += 1
                else: disq.add(aa[1])
            elif o == "STR" and len(aa) == 2 and _ra_is_slot(aa[1]):
                if aa[0] in ("r0", "r1"): cnt[aa[1]] += 1
                else: disq.add(aa[1])
            else:
                # any other reference to a slot disqualifies it
                for x in (aa if isinstance(aa, list) else []):
                    if _ra_is_slot(x) and x not in ("r0", "r1", "r2", "r3", "r4", "r5"):
                        disq.add(x)
        cands = [(s, c) for s, c in cnt.most_common() if s not in disq and c >= 2]
        # net size guard: 4*c saved in-loop must cover 7 (entry) + 7*exits (store-back)
        overhead_per = 7 + 7 * max(len(exit_tgts), 1)
        chosen = []
        for s, c in cands:
            if len(chosen) >= len(homes):
                break
            if c * 4 - overhead_per > 0 or c >= 3:
                chosen.append(s)
        if not chosen:
            processed.add(tgt); continue

        home_of = {s: homes[k] for k, s in enumerate(chosen)}
        if REG_ALLOC_DEBUG:
            print("[reg-alloc] loop '%s': promote %s -> %s  (exits=%s)"
                  % (tgt, chosen, [home_of[s] for s in chosen], sorted(exit_tgts)))

        # ---- build the rewritten IR ----
        out = []
        for i in range(n):
            if i == H:
                # entry-loads go right before the header label
                for s in chosen:
                    out.append(("LDR", [home_of[s], s]))
                out.append(ir[i])
                continue
            o, aa = ir[i]
            if H < i < bj and o == "LDR" and len(aa) == 2 and aa[1] in home_of:
                out.append(("MOV", [aa[0], home_of[aa[1]]]))
            elif H < i < bj and o == "STR" and len(aa) == 2 and aa[1] in home_of:
                out.append(("MOV", [home_of[aa[1]], aa[0]]))
            else:
                out.append(ir[i])
            if o == "LABEL" and aa in exit_tgts:
                # store-backs at the top of each exit block: STR <home_reg>, <slot>
                for s in chosen:
                    out.append(("STR", [home_of[s], s]))
        processed.add(tgt)
        return out

    return None


# ---------- Parser + two-pass assembler ----------

def parse_axiom(path, text_base, data_base):
    # IR parsing, dead-code elimination, and the target-independent peepholes
    # are shared with the AVR backend -- see axiom_ir.py. From here on this file
    # only chooses x86-64 opcodes for the already-optimized IR.
    text_t, data_t, vectors = parse_axm(path)
    # Honor the front-end's '.volatile' sidecar: these slots are opaque to the
    # peephole optimizer (no load reuse / no store-elision). x86 has no mmio, but
    # an explicit `volatile` still applies (e.g. a slot shared with a handler).
    volatile_slots = set()
    for _n, _d, _r in data_t:
        if _d == ".volatile":
            volatile_slots |= set(_r)
    text_t = eliminate_dead(text_t, (), jumptable_labels(data_t))
    text_t = optimize_common(text_t, volatile_slots)
    if MERGE_LITERAL_WRITES:
        text_t, data_t = merge_literal_writes(text_t, data_t)
    text_t = allocate_loop_registers(text_t)
    if FOLD_IMMEDIATES:
        text_t = fold_immediates(text_t)

    text = []
    data = []
    data_offsets = {}
    text_offset = 0
    data_offset = 0

    # ---------- DATA LAYOUT (computed before the text pass so base-register
    # addressing can size each access; data offsets don't depend on text) ------
    ref_syms = set(jumptable_labels(data_t))
    for _op, _a in text_t:
        if _op in ("LDR", "STR", "LEA", "LDRB", "STORB") and isinstance(_a, (list, tuple)):
            for _x in _a:
                if isinstance(_x, str) and not _x.startswith("#") and _x not in REG_MAP:
                    ref_syms.add(_x)
    # Array slots {base}_0 .. {base}_N are stored contiguously and accessed as
    # `LEA {base}_0 ; STORQ/LOADQ` at a *register* index, so only {base}_0 ever
    # appears as a symbol operand. The other elements are reached purely by
    # computed address and would look "dead" to the dropper below — removing
    # them would collapse the array's storage and corrupt runtime indexing. So
    # if ANY element of an array group is referenced, keep the whole group.
    _arr_groups = {}
    for _nm, _d, _r in data_t:
        _base, _sep, _suf = _nm.rpartition("_")
        if _sep and _suf.isdigit():
            _arr_groups.setdefault(_base, []).append(_nm)
    for _members in _arr_groups.values():
        if any(_m in ref_syms for _m in _members):
            ref_syms.update(_members)
    # The manifest (built at compile time in the front-end) has already proven
    # that every scalar/array slot occupies a UNIQUE, non-overlapping byte
    # range. Each such slot is zero-initialized (.byte/.word/.dword/.qword 0):
    # its value is written at run time by a STR, never baked into the file. So
    # the file does not need to store those zeros at all. We split the data into
    # two regions:
    #
    #   * init region  — .bytes (string literals) and .qaddrs (jump tables).
    #     Real content; written into the ELF file.  [contributes to p_filesz]
    #   * zero region  — every scalar/array slot. Laid out AFTER the init region
    #     and materialized for free by the loader via p_memsz > p_filesz (the
    #     standard .bss mechanism). No bytes on disk, no runtime code, no runtime
    #     check — the shrink is a pure compile-time layout decision that rides on
    #     the manifest's non-overlap proof.
    #
    # Putting the zero region last is what lets a single contiguous p_filesz
    # truncate it away; the manifest guarantees nothing in the init region ever
    # aliases a slot, so reordering is safe by construction.
    sizes = {".byte": 1, ".word": 2, ".dword": 4, ".qword": 8}
    # Read the scalar type sidecar (if present): slot -> (width, kind).
    slot_ws = {}
    if PACK_SCALARS:
        for name, directive, rest in data_t:
            if directive == ".typemap":
                for entry in rest:
                    if "=" in entry:
                        sl, ty = entry.split("=", 1)
                        if ty in _TYPE_WS:
                            slot_ws[sl] = _TYPE_WS[ty]
    def slot_width(nm):
        return slot_ws.get(nm, (8, "s"))[0]
    def slot_kind(nm):
        return slot_ws.get(nm, (8, "s"))[1]

    init_entries = []     # (name, directive, size, payload)  -> file-backed
    zero_entries = []     # (name, directive, size)           -> BSS, not in file
    seen_blobs = {}       # dedup: byte-tuple -> canonical name (offset resolved later)
    dedup_alias = {}      # duplicate name    -> canonical name
    for name, directive, rest in data_t:
        if directive == ".qaddrs":
            labels = list(rest)
            init_entries.append((name, ".qaddrs", 8 * len(labels), labels))
        elif directive == ".bytes":
            if DROP_DEAD_DATA and name not in ref_syms:
                continue                 # never LEA'd (e.g. merged away) -> dead
            vals = [int(x) & 0xFF for x in rest]
            key = tuple(vals)
            if DEDUP_DATA and key in seen_blobs:
                dedup_alias[name] = seen_blobs[key]
                continue
            seen_blobs[key] = name
            init_entries.append((name, ".bytes", len(vals), vals))
        else:
            if directive not in sizes:
                continue                 # skips .typemap, .at, etc.
            if DROP_DEAD_DATA and name not in ref_syms:
                continue
            # pack to the slot's true width (default 8); arrays/floats/ptr stay 8
            w = slot_width(name) if PACK_SCALARS else sizes[directive]
            zero_entries.append((name, directive, w))

    # Only the init region is emitted into `data` (and therefore into the file).
    data = list(init_entries)
    data_offset = 0
    for nm, _d, sz, *_ in init_entries:
        data_offsets[nm] = data_offset
        data_offset += sz
    file_data_size = data_offset
    # Pack the zero region tight: widest slots first means every slot lands on a
    # natural boundary with zero padding (all sizes are powers of two).
    for nm, _d, sz in sorted(zero_entries, key=lambda e: -e[2]):
        data_offsets[nm] = data_offset      # lives only in memory (BSS)
        data_offset += sz
    bss_size = data_offset - file_data_size
    for dup, canon in dedup_alias.items():
        data_offsets[dup] = data_offsets[canon]

    # ---------- BASE-REGISTER SELECTION ----------
    # Count data-slot accesses, then pick the base offset K that places the most
    # accesses within a signed-8-bit displacement window [K-128 .. K+127].
    _acc = {}
    for _op, _a in text_t:
        if _op in ("LDR", "STR", "LEA") and isinstance(_a, (list, tuple)) and len(_a) == 2:
            s = _a[1]
            if s in data_offsets:
                _acc[s] = _acc.get(s, 0) + 1
    base_off = None
    near = {}
    if BASE_REG_ADDR and _acc:
        best_k, best_score = 0, -1
        for k in sorted({data_offsets[s] for s in _acc}):
            score = sum(c for s, c in _acc.items() if -128 <= data_offsets[s] - k <= 127)
            if score > best_score:
                best_score, best_k = score, k
        if best_score * 3 > 7:          # must beat the one-time 7-byte lea setup
            base_off = best_k
            near = {s: o - base_off for s, o in data_offsets.items()
                    if -128 <= o - base_off <= 127}

    # ---------- FIRST PASS — measure sizes ----------
    if base_off is not None:
        text.append(Instr("LEA_RBP", (base_off,), 7))   # lea rbp,[rip+data] (entry)
        text_offset += 7
    for op, args in text_t:
        if op == "LABEL":
            text.append(Instr("LABEL", (args,), 0))
            continue
        parts = [op] + list(args)

        if op == "JMP":
            # unconditional near jump — relaxable: EB rel8 (2) or E9 rel32 (5)
            text.append(_mk_branch("JMP", (parts[1],)))
            text_offset += text[-1].size

        elif op == "CALL":
            # near call — E8 rel32 = 5 bytes (pushes return address)
            text.append(Instr("CALL", (parts[1],), 5))
            text_offset += 5

        elif op == "RET":
            # near return — C3 = 1 byte (pops return address into rip)
            text.append(Instr("RET", (), 1))
            text_offset += 1

        elif op in ("JZ", "JNZ"):
            # conditional near jump on r0 == 0 / != 0.
            # test rax,rax (3) + jcc rel8 (2)  → 5, or jcc rel32 (6) → 9.
            text.append(_mk_branch(op, (parts[1],)))
            text_offset += text[-1].size

        elif op == "JA":
            # unsigned "jump if above" — reads flags from the PRECEDING CMP
            # (no test prefix). Relaxable: 77 rel8 (2) or 0F 87 rel32 (6).
            text.append(_mk_branch("JA", (parts[1],)))
            text_offset += text[-1].size

        elif op == "LEA":
            # LEA reg, label — RIP-relative effective address (48 8D /r = 7 B),
            # or [rbp+disp8] = 4 B when the data slot is base-register-near.
            lsize = 4 if parts[2] in near else 7
            text.append(Instr("LEA", (parts[1], parts[2]), lsize))
            text_offset += lsize

        elif op == "JMPIDX":
            # jmp [base + idx*8] — indexed indirect jump through a qword
            # table. FF /4 with SIB (scale 8) = 3 bytes.
            text.append(Instr("JMPIDX", (parts[1], parts[2]), 3))
            text_offset += 3

        elif op == "LDR":
            reg = parts[1]
            arg = parts[2]
            if arg.startswith("#"):
                raw_val = arg[1:]
                imm = int(raw_val, 16) if raw_val.startswith("0x") or raw_val.startswith("0X") else int(raw_val)
                size = instr_size_mov_imm(imm)
                text.append(Instr("LDR_IMM", (reg, imm), size))
            else:
                # width-aware load: movsxd for 4-byte, movzx for 1/2-byte, plain
                # mov for 8-byte. Size must match _load_bytes exactly.
                w = slot_width(arg)
                nr = arg in near
                if w == 8:   lsize = 4 if nr else 7
                elif w == 4: lsize = 4 if nr else 7      # movsxd, same length as mov r64
                else:        lsize = 5 if nr else 8      # movzx (3-byte opcode)
                text.append(Instr("LDR_LABEL", (reg, arg), lsize))
            text_offset += text[-1].size

        elif op == "STR":
            reg, label = parts[1], parts[2]
            w = slot_width(label)
            nr = label in near
            if w == 8:   ssize = 4 if nr else 7
            elif w == 4: ssize = 3 if nr else 6
            elif w == 2: ssize = 4 if nr else 7
            else:        ssize = (3 if nr else 6) + (1 if _X86NUM[REG_MAP[reg]] >= 4 else 0)
            text.append(Instr("STR", (reg, label), ssize))
            text_offset += ssize

        elif op == "MOV":
            text.append(Instr("MOV", (parts[1], parts[2]), 3))
            text_offset += 3

        elif op in ("ADD", "SUB", "MUL", "CMP", "SETAND", "SETOR", "SETXOR", "SHL", "SHR"):
            if len(parts) > 2 and parts[2].startswith("#"):
                imm = int(parts[2][1:])
                sz = len(_ALU_IMM[op](imm))
                text.append(Instr(op, (parts[1], parts[2]), sz))
            else:
                rr = {"ADD": 3, "SUB": 3, "MUL": 3, "CMP": CMP_SIZE,
                      "SETAND": LOGIC_SIZE, "SETOR": LOGIC_SIZE, "SETXOR": LOGIC_SIZE,
                      "SHL": SHIFT_SIZE, "SHR": SHIFT_SIZE}[op]
                text.append(Instr(op, (parts[1], parts[2]), rr))
            text_offset += text[-1].size

        elif op == "DIV":
            text.append(Instr("DIV", (parts[1], parts[2]), 5))
            text_offset += 5

        elif op in ("SETLT", "SETGT", "SETLE", "SETGE", "SETEQ", "SETNE"):
            # SET<cond> r0  — single register operand
            text.append(Instr(op, (parts[1],), SETCC_SIZE))
            text_offset += SETCC_SIZE

        elif op == "SYS":
            # exit(0) — size measured from the encoder (shorter now that the
            # immediate loads use compact encodings; also tracks the harness's
            # monkeypatched exit).
            text.append(Instr("SYS", (), len(encode_sys_r0_r1())))
            text_offset += text[-1].size

        elif op == "ABORT":
            # clean exit(ABORT_EXIT_CODE)
            text.append(Instr("ABORT", (), len(encode_abort())))
            text_offset += text[-1].size

        elif op == "WRITE":
            # write(1, rsi, rdx) — outlined to `call __write_stub` (5) or inline
            text.append(Instr("WRITE", (),
                              5 if OUTLINE_WRITE else len(encode_write())))
            text_offset += text[-1].size

        elif op == "READ":
            # read(0, rsi, rdx)
            text.append(Instr("READ", (), len(encode_read())))
            text_offset += text[-1].size

        elif op == "I2F":
            # cvtsi2sd xmm0,rax (5) + movq rax,xmm0 (5) = 10
            text.append(Instr("I2F", (parts[1],), 10))
            text_offset += 10

        elif op == "F2I":
            text.append(Instr("F2I", (parts[1],), 10))
            text_offset += 10

        elif op in ("FADD", "FSUB", "FMUL", "FDIV"):
            text.append(Instr(op, (parts[1], parts[2]), 19))
            text_offset += 19

        elif op == "LDRB":
            # LDRB dst, base, index  -> 5 bytes
            text.append(Instr("LDRB", (parts[1], parts[2], parts[3]), 5))
            text_offset += 5

        elif op == "STORB":
            # STORB src, base, index  -> 3 bytes (88 /r SIB)
            text.append(Instr("STORB", (parts[1], parts[2], parts[3]), 3))
            text_offset += 3

        elif op == "LDRQ":
            # LDRQ dst, base, index  -> 4 bytes (48 8B /r SIB, scale=8)
            text.append(Instr("LDRQ", (parts[1], parts[2], parts[3]), 4))
            text_offset += 4

        elif op == "STORQ":
            # STORQ src, base, index  -> 4 bytes (48 89 /r SIB, scale=8)
            text.append(Instr("STORQ", (parts[1], parts[2], parts[3]), 4))
            text_offset += 4

        elif op in ("PUSH", "POP"):
            # single-register push/pop — 1 byte
            text.append(Instr(op, (parts[1],), 1))
            text_offset += 1

        elif op in ("SEI", "CLI"):
            # global-interrupt enable/disable: AVR-only. x86 user-mode code
            # can't touch the interrupt flag, and there is no timer to fire,
            # so these are 0-byte no-ops (the .axm still assembles cleanly).
            pass

        elif op == "RETI":
            # return-from-interrupt: no x86 equivalent. An ISR is never
            # entered on x86 (no vector table), so a plain near ret keeps the
            # block well-formed if anything ever falls into it.
            text.append(Instr("RET", (), 1))
            text_offset += 1

        elif op == ".VECTOR":
            # interrupt vector-table binding — consumed by the AVR backend,
            # meaningless on x86. Skip it (emits nothing).
            pass

    # ---------- OUTLINED WRITE STUB ----------
    # If WRITE was outlined, emit the single shared stub once, after all user
    # code. It is reached only via `call`, ends in `ret`, and nothing falls into
    # it (the preceding instruction is always a RET or the exit syscall).
    if OUTLINE_WRITE and any(i.kind == "WRITE" for i in text):
        text.append(Instr("LABEL", (_WRITE_STUB_LABEL,), 0))
        text.append(Instr("WRITE_STUB", (), len(encode_write_stub())))

    # ---------- BRANCH RELAXATION ----------
    # Shrink every JMP/JZ/JNZ/JA to its 2-byte rel8 form where the target is in
    # range; grow back to rel32 only where it isn't. Must happen before label
    # addresses are computed, since it changes instruction sizes.
    _relax_branches(text)
    text_offset = sum(inst.size for inst in text if inst.kind != "LABEL")

    # ---------- ADDRESS COMPUTATION ----------
    # text_base and data_base are passed in from the ELF builder
    # so both passes agree on the same layout.
    symbol_addr = {name: data_base + off for name, off in data_offsets.items()}

    # Resolve code-label byte offsets by walking the sized instruction list.
    label_offset = {}
    _off = 0
    for inst in text:
        if inst.kind == "LABEL":
            label_offset[inst.args[0]] = _off
        else:
            _off += inst.size
    # Absolute address of each code label
    label_addr = {name: text_base + off for name, off in label_offset.items()}

    # ---------- SECOND PASS — encode ----------
    out      = bytearray()
    cur      = 0
    manifest = {}   # reg -> variable name

    for inst in text:
        if inst.kind == "LDR_IMM":
            reg, imm = inst.args
            # '__imm__' sentinel: register holds a value but no named variable.
            # This allows STR to proceed — the programmer loaded a literal and
            # is storing it into a named slot, which is always safe.
            manifest[reg] = "__imm__"
            encoded = encode_mov_imm64(REG_MAP[reg], imm)
            out += encoded
            cur += inst.size

        elif inst.kind == "LDR_LABEL":
            reg, label = inst.args
            if label not in symbol_addr:
                # Unmapped label = a memory-mapped (.at) register on AVR. There
                # is no such hardware on x86, so the read is a no-op — but we
                # still emit inst.size NOPs so every later code label keeps its
                # exact byte offset (jumps stay correct).
                out += b"\x90" * inst.size
                cur += inst.size
                continue
            # Loading a value into a register takes ownership, overwriting
            # whatever the register previously held. Sequential reuse of r0/r1
            # during expression evaluation is legitimate. Scratch temporaries
            # (__tN) are transient values, not named-variable bindings, so a
            # register loaded from one is free to store anywhere.
            if label.startswith("__t"):
                manifest[reg] = "__result__"
            else:
                manifest[reg] = label
            target     = symbol_addr[label]
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel32      = target - next_addr
            # width-aware load: sign/zero-extends a narrow slot to the full 64-bit
            # register, so a packed value reads back identical to an 8-byte slot.
            nd = near[label] if label in near else None
            out += _load_bytes(REG_MAP[reg], slot_width(label), slot_kind(label), nd, rel32)
            cur += inst.size

        elif inst.kind == "STR":
            reg, label = inst.args
            if label not in symbol_addr:
                # Memory-mapped (.at) register: no-op store on x86. Emit NOPs of
                # the reserved size so subsequent label offsets stay exact.
                out += b"\x90" * inst.size
                cur += inst.size
                continue
            if reg not in manifest:
                raise RuntimeError(
                    f"Memory Safety: {reg} is uninitialized, cannot store into {label}")
            held = manifest[reg]
            # '__imm__' (immediate loaded) and '__result__' (arithmetic result)
            # are transient values free to store into any slot. A named variable
            # may only be stored back into its own slot. Compiler-internal
            # scratch slots (names starting with '__', e.g. the itoa workspace
            # and the 1-byte write buffer) are transient by construction and may
            # be written from any register.
            if (held not in ("__imm__", "__result__")
                    and held != label
                    and not label.startswith("__")):
                raise RuntimeError(
                    f"Memory Safety: {reg} contains '{held}', cannot store into {label}")
            target     = symbol_addr[label]
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel32      = target - next_addr
            # width-aware store: writes exactly the slot's bytes, so a packed
            # slot never touches its neighbour — the non-overlap invariant holds.
            nd = near[label] if label in near else None
            out += _store_bytes(REG_MAP[reg], slot_width(label), nd, rel32)
            cur += inst.size

        elif inst.kind == "MOV":
            dst, src = inst.args
            out += encode_mov_rr(REG_MAP[dst], REG_MAP[src])
            # dst now holds exactly what src held (for the manifest's safety check)
            if src in manifest:
                manifest[dst] = manifest[src]
            else:
                manifest.pop(dst, None)
            cur += inst.size

        elif inst.kind in ("ADD", "SUB", "MUL", "SETAND", "SETOR", "SETXOR", "SHL", "SHR"):
            dst, src = inst.args
            manifest["r0"] = "__result__"
            if src.startswith("#"):
                out += _ALU_IMM[inst.kind](int(src[1:]))      # OP rax, imm
            elif dst == "r0" and src == "r1":
                out += {"ADD": encode_add_r0_r1, "SUB": encode_sub_r0_r1,
                        "MUL": encode_mul_r0_r1, "SETAND": encode_setand,
                        "SETOR": encode_setor, "SETXOR": encode_setxor,
                        "SHL": encode_shl_r0_r1, "SHR": encode_shr_r0_r1}[inst.kind]()
            else:
                out += b"\x90" * inst.size
            cur += inst.size

        elif inst.kind == "DIV":
            dst, src = inst.args
            if dst == "r0" and src == "r1":
                manifest["r0"] = "__result__"
                out += encode_div_r0_r1()
            else:
                out += b"\x90"
            cur += inst.size

        elif inst.kind == "CMP":
            dst, src = inst.args
            if src.startswith("#"):
                out += encode_cmp_imm(int(src[1:]))           # cmp rax, imm
            elif dst == "r0" and src == "r1":
                out += encode_cmp_r0_r1()
            else:
                out += b"\x90" * inst.size
            cur += inst.size

        elif inst.kind in ("SETLT", "SETGT", "SETLE", "SETGE", "SETEQ", "SETNE"):
            # result (0/1) lands in r0
            manifest["r0"] = "__result__"
            enc = {
                "SETLT": encode_setlt, "SETGT": encode_setgt,
                "SETLE": encode_setle, "SETGE": encode_setge,
                "SETEQ": encode_seteq, "SETNE": encode_setne,
            }[inst.kind]()
            out += enc
            cur += inst.size

        elif inst.kind == "LABEL":
            # zero-size marker, emits nothing
            pass

        elif inst.kind == "JMP":
            target = inst.args[0]
            if target not in label_addr:
                raise RuntimeError(f"unknown jump target: {target}")
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel        = label_addr[target] - next_addr
            if inst.size == 2:                          # EB rel8
                out += b"\xEB" + struct.pack("<b", rel)
            else:                                       # E9 rel32
                out += b"\xE9" + struct.pack("<i", rel)
            cur += inst.size

        elif inst.kind == "CALL":
            target = inst.args[0]
            if target not in label_addr:
                raise RuntimeError(f"unknown call target: {target}")
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel32      = label_addr[target] - next_addr
            out += b"\xE8" + struct.pack("<i", rel32)   # call rel32
            # the callee leaves its result in rax (r0); mark it storable so a
            # following STR r0, dest passes the memory-safety check.
            manifest["r0"] = "__result__"
            cur += inst.size

        elif inst.kind == "RET":
            out += b"\xC3"                              # ret
            cur += inst.size

        elif inst.kind in ("JZ", "JNZ"):
            target = inst.args[0]
            if target not in label_addr:
                raise RuntimeError(f"unknown jump target: {target}")
            # test rax, rax  (48 85 C0) — sets ZF if rax == 0
            test = b"\x48\x85\xC0"
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel        = label_addr[target] - next_addr
            if inst.size == 5:                          # test + jcc rel8
                op1 = 0x74 if inst.kind == "JZ" else 0x75
                out += test + bytes([op1]) + struct.pack("<b", rel)
            else:                                       # test + jcc rel32
                op2 = 0x84 if inst.kind == "JZ" else 0x85
                out += test + bytes([0x0F, op2]) + struct.pack("<i", rel)
            cur += inst.size

        elif inst.kind == "JA":
            # Unsigned "jump if above" on the flags set by the preceding CMP.
            # No test prefix. Relaxable: 77 rel8 (2) or 0F 87 rel32 (6).
            target = inst.args[0]
            if target not in label_addr:
                raise RuntimeError(f"unknown jump target: {target}")
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel        = label_addr[target] - next_addr
            if inst.size == 2:                          # 77 rel8
                out += b"\x77" + struct.pack("<b", rel)
            else:                                       # 0F 87 rel32
                out += b"\x0F\x87" + struct.pack("<i", rel)
            cur += inst.size

        elif inst.kind == "LEA":
            # lea <reg>, [rip+rel32] — RIP-relative effective address (48 8D /r).
            # ModRM mirrors LDR_LABEL: mod=00, rm=101 (RIP), reg = destination.
            reg, label = inst.args
            # LEA serves both data symbols (e.g. string literals) and code
            # labels (e.g. &fn -> a function pointer). RIP-relative math is the
            # same for both; only the lookup table differs.
            if label in symbol_addr:
                target = symbol_addr[label]
            elif label in label_addr:
                target = label_addr[label]
            else:
                raise RuntimeError(f"LEA references unknown symbol: {label}")
            manifest[reg] = "__result__"   # holds a computed address, free to use
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel32      = target - next_addr
            modrm_reg = {
                "rax": 0x05, "rcx": 0x0D, "rdx": 0x15, "rbx": 0x1D,
                "rsi": 0x35, "rdi": 0x3D,
            }[REG_MAP[reg]]
            if label in near:
                # lea <reg>, [rbp+disp8] — 4 bytes (data symbols only)
                bp = 0x45 | (_X86NUM[REG_MAP[reg]] << 3)
                out += b"\x48\x8D" + bytes([bp]) + struct.pack("<b", near[label])
            else:
                out += b"\x48\x8D" + bytes([modrm_reg]) + struct.pack("<i", rel32)
            cur += inst.size

        elif inst.kind == "LEA_RBP":
            # Entry setup: lea rbp, [rip+rel32] -> rbp = data_base + base_off.
            # Subsequent [rbp+disp8] accesses reach nearby data slots in 4 bytes.
            (k,) = inst.args
            target     = data_base + k
            instr_addr = text_base + cur
            next_addr  = instr_addr + inst.size
            rel32      = target - next_addr
            out += b"\x48\x8D" + bytes([0x2D]) + struct.pack("<i", rel32)  # lea rbp,[rip+d]
            cur += inst.size

        elif inst.kind == "JMPIDX":
            # jmp [base + idx*8] — FF /4 with a SIB byte (scale=8). Reads an
            # 8-byte absolute target from the table and jumps to it.
            rbase, ridx = inst.args
            x86num = {"rax": 0, "rcx": 1, "rdx": 2, "rbx": 3,
                      "rsp": 4, "rbp": 5, "rsi": 6, "rdi": 7}
            base  = x86num[REG_MAP[rbase]]
            index = x86num[REG_MAP[ridx]]
            sib = (0b11 << 6) | (index << 3) | base   # scale 8
            out += b"\xFF\x24" + bytes([sib])
            cur += inst.size

        elif inst.kind == "SYS":
            out += encode_sys_r0_r1()
            cur += inst.size

        elif inst.kind == "ABORT":
            out += encode_abort()
            cur += inst.size

        elif inst.kind == "WRITE":
            if OUTLINE_WRITE:
                # call __write_stub (E8 rel32) — same clobber profile as the
                # inline syscall, so the manifest is treated exactly as before.
                instr_addr = text_base + cur
                next_addr  = instr_addr + inst.size
                rel32      = label_addr[_WRITE_STUB_LABEL] - next_addr
                out += b"\xE8" + struct.pack("<i", rel32)
            else:
                out += encode_write()
            cur += inst.size

        elif inst.kind == "WRITE_STUB":
            out += encode_write_stub()
            cur += inst.size

        elif inst.kind == "READ":
            out += encode_read()
            cur += inst.size

        elif inst.kind == "I2F":
            # operates on rax in place; the IR always names r0 here
            (reg,) = inst.args
            if REG_MAP[reg] != "rax":
                raise RuntimeError("I2F only supported on r0 (rax)")
            manifest[reg] = "__result__"   # now holds float bits, transient
            out += encode_i2f()
            cur += inst.size

        elif inst.kind == "F2I":
            (reg,) = inst.args
            if REG_MAP[reg] != "rax":
                raise RuntimeError("F2I only supported on r0 (rax)")
            manifest[reg] = "__result__"
            out += encode_f2i()
            cur += inst.size

        elif inst.kind in ("FADD", "FSUB", "FMUL", "FDIV"):
            ra, rb = inst.args
            if REG_MAP[ra] != "rax" or REG_MAP[rb] != "rbx":
                raise RuntimeError(f"{inst.kind} only supported on r0, r1")
            manifest[ra] = "__result__"
            out += {"FADD": encode_fadd, "FSUB": encode_fsub,
                    "FMUL": encode_fmul, "FDIV": encode_fdiv}[inst.kind]()
            cur += inst.size

        elif inst.kind == "LDRB":
            dst, base, index = inst.args
            manifest[dst] = "__result__"
            out += encode_ldrb(REG_MAP[dst], REG_MAP[base], REG_MAP[index])
            cur += inst.size

        elif inst.kind == "STORB":
            src, base, index = inst.args
            out += encode_storb(REG_MAP[src], REG_MAP[base], REG_MAP[index])
            cur += inst.size

        elif inst.kind == "LDRQ":
            dst, base, index = inst.args
            manifest[dst] = "__result__"
            out += encode_ldrq(REG_MAP[dst], REG_MAP[base], REG_MAP[index])
            cur += inst.size

        elif inst.kind == "STORQ":
            src, base, index = inst.args
            out += encode_storq(REG_MAP[src], REG_MAP[base], REG_MAP[index])
            cur += inst.size

        elif inst.kind == "PUSH":
            (reg,) = inst.args
            out += encode_push(REG_MAP[reg])
            cur += inst.size

        elif inst.kind == "POP":
            (reg,) = inst.args
            # popped value is transient — free to store anywhere afterwards
            manifest[reg] = "__result__"
            out += encode_pop(REG_MAP[reg])
            cur += inst.size

    # Safety net: the first-pass size model and the second-pass encoders must
    # agree byte-for-byte, or every label/jump/RIP-relative offset past the
    # first mismatch is wrong. `cur` summed the declared sizes; len(out) is the
    # bytes actually emitted. They must be identical.
    assert len(out) == cur == text_offset, (
        f"codegen size drift: emitted {len(out)} bytes but sized {cur}/{text_offset}")

    # ---------- DATA SECTION ----------
    data_out = bytearray()
    for entry in data:
        if len(entry) == 4 and entry[1] == ".qaddrs":
            name, directive, size, labels = entry
            for lab in labels:
                if lab not in label_addr:
                    raise RuntimeError(
                        f"jump table '{name}' references unknown code label: {lab}")
                data_out += struct.pack("<Q", label_addr[lab])   # absolute address
        elif len(entry) == 4 and entry[1] == ".bytes":
            name, directive, size, vals = entry
            data_out += bytes(vals)                              # raw string bytes
        else:
            name, directive, size = entry
            data_out += b"\x00" * size

    return bytes(out), bytes(data_out), bss_size


# ---------- ELF64 writer ----------

# When True, headers + text + data are wrapped in ONE PT_LOAD segment marked
# RWX (PF_R|PF_W|PF_X). This is the smallest possible layout for a conformant
# ELF: the file is exactly  headers(120) + text + data  with NO padding, and
# only one program header instead of two.
#
# The two-segment design (False) keeps W^X — code stays read-execute, data
# stays read-write — but that requires the data to live on its own 4 KB page,
# which forces up to ~4095 bytes of zero padding between text and data in the
# file. For tiny programs that padding dwarfs the actual code (90%+ of a.out).
#
# Set to False if you need strict W^X (e.g. a hardened kernel that refuses
# writable+executable mappings); otherwise the single RWX segment is far
# smaller and runs fine on a standard Linux loader.
SINGLE_RWX_SEGMENT = True


def build_elf64(text_size, data_size, bss_size=0):
    """
    Build the ELF64 program headers and report the load addresses.

    `data_size` is the file-backed (initialized) data; `bss_size` is the
    zero-initialized tail that is NOT stored in the file and is materialized by
    the loader through p_memsz > p_filesz. The manifest proved that tail's
    layout safe at compile time, so the binary carries none of those zero bytes.

    Two layouts are supported (see SINGLE_RWX_SEGMENT):

      * single RWX segment — one PT_LOAD covering ELF header + PHDR + text +
        data, contiguous in both file and memory. Smallest output; data is
        writable (STR works) and executable, but no W^X.

      * two segments (W^X) — PT_LOAD RX over header+PHDRs+text and a separate
        PT_LOAD RW over the data. Keeps code non-writable and data
        non-executable at the cost of one extra PHDR and up to a full 4 KB of
        inter-segment alignment padding.

    Returns (headers_bytes, text_vaddr, data_vaddr, pad_between).
    """
    e_ident     = b"\x7fELF\x02\x01\x01" + b"\x00" * 9
    e_type      = 2        # ET_EXEC
    e_machine   = 0x3E     # x86-64
    e_version   = 1
    e_phoff     = 64       # PHDRs start right after ELF header
    e_shoff     = 0
    e_flags     = 0
    e_ehsize    = 64
    e_phentsize = 56
    e_shentsize = 0
    e_shnum     = 0
    e_shstrndx  = 0
    BASE        = 0x400000
    ALIGN       = 0x1000

    if SINGLE_RWX_SEGMENT:
        # ---- one PT_LOAD RWX: headers + text + data, fully contiguous ----
        e_phnum      = 1
        headers_size = 64 + 56                       # 120 bytes, the floor

        text_file_off = headers_size
        text_vaddr    = BASE + text_file_off
        e_entry       = text_vaddr

        # Data sits immediately after text in both the file and memory. Because
        # everything shares one segment whose p_offset(0) ≡ p_vaddr(BASE) mod
        # ALIGN, no congruence padding is ever needed.
        data_vaddr  = text_vaddr + text_size
        pad_between = 0
        seg_filesz  = headers_size + text_size + data_size
        seg_memsz   = seg_filesz + bss_size

        elf_header = struct.pack(
            "<16sHHIQQQIHHHHHH",
            e_ident, e_type, e_machine, e_version,
            e_entry, e_phoff, e_shoff, e_flags,
            e_ehsize, e_phentsize, e_phnum,
            e_shentsize, e_shnum, e_shstrndx
        )
        phdr = struct.pack(
            "<IIQQQQQQ",
            1,            # PT_LOAD
            7,            # PF_R | PF_W | PF_X
            0,            # file offset
            BASE,         # p_vaddr
            BASE,         # p_paddr
            seg_filesz,   # p_filesz  (file: headers + text + init data)
            seg_memsz,    # p_memsz   (mem:  + zero/BSS tail, loader-filled)
            ALIGN         # p_align
        )
        padding = b"\x00" * (headers_size - len(elf_header) - len(phdr))
        return elf_header + phdr + padding, text_vaddr, data_vaddr, pad_between

    # ---- two segments (W^X): RX over text, RW over data ----
    e_phnum      = 2
    headers_size = 64 + 56 * 2   # ELF header + 2 PHDRs = 176 bytes

    text_file_off  = headers_size
    text_vaddr     = BASE + text_file_off   # 0x4000B0
    e_entry        = text_vaddr

    # The kernel's mmap requires p_vaddr ≡ p_offset (mod p_align). Put data on
    # the next page boundary after text and pad the file so its offset is
    # congruent to that vaddr modulo ALIGN.
    text_end_off   = text_file_off + text_size
    data_vaddr     = (text_vaddr + text_size + (ALIGN - 1)) & ~(ALIGN - 1)
    target_mod     = data_vaddr % ALIGN
    data_file_off  = text_end_off
    if data_file_off % ALIGN != target_mod:
        delta = (target_mod - (data_file_off % ALIGN)) % ALIGN
        data_file_off += delta
    pad_between = data_file_off - text_end_off

    elf_header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        e_ident, e_type, e_machine, e_version,
        e_entry, e_phoff, e_shoff, e_flags,
        e_ehsize, e_phentsize, e_phnum,
        e_shentsize, e_shnum, e_shstrndx
    )
    rx_size = headers_size + text_size
    phdr_rx = struct.pack(
        "<IIQQQQQQ",
        1, 5, 0, BASE, BASE, rx_size, rx_size, ALIGN
    )
    phdr_rw = struct.pack(
        "<IIQQQQQQ",
        1, 6, data_file_off, data_vaddr, data_vaddr,
        data_size, data_size + bss_size, ALIGN
    )
    padding = b"\x00" * (headers_size - len(elf_header) - len(phdr_rx) - len(phdr_rw))
    return elf_header + phdr_rx + phdr_rw + padding, text_vaddr, data_vaddr, pad_between


# ---------- main ----------

def main():
    if len(sys.argv) != 2:
        print("usage: python3 axiom_x86.py <file.axm>")
        sys.exit(1)

    path = sys.argv[1]

    # --- Pass 1: dry run with placeholder addresses to measure text size ---
    # We use dummy bases; only the sizes matter here, not the rel32 values.
    DUMMY_TEXT = 0x400000
    DUMMY_DATA = 0x600000
    text_bytes, data_bytes, bss_size = parse_axiom(path, DUMMY_TEXT, DUMMY_DATA)
    text_size = len(text_bytes)
    data_size = len(data_bytes)        # file-backed (initialized) data only

    # --- Build ELF headers to find out real load addresses ---
    headers, real_text_base, real_data_base, pad_between = build_elf64(
        text_size, data_size, bss_size)

    # --- Pass 2: reassemble with correct addresses ---
    text_bytes, data_bytes, _ = parse_axiom(path, real_text_base, real_data_base)

    elf = headers + text_bytes + (b"\x00" * pad_between) + data_bytes

    with open("a.out", "wb") as f:
        f.write(elf)

    print(f"Assembled {path} -> a.out")
    print(f"  text: {text_size} bytes at 0x{real_text_base:X}")
    print(f"  data: {data_size} bytes at 0x{real_data_base:X} (file)"
          f" + {bss_size} bytes BSS (loader-zeroed, not in file)")
    print(f"  total ELF: {len(elf)} bytes  (memory image: {len(elf) + bss_size} bytes)")
    print(f"Run:  chmod +x a.out && ./a.out ; echo \"exit: $?\"")

if __name__ == "__main__":
    main()

'''
Update this assembler to include the manifest memory safety checks 
and be reprogammable for the backend upcodes keeping the same exact syntax
'''