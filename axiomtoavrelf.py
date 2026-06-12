#!/usr/bin/env python3
# ===========================================================================
# axiomtoavrelf.py  —  Axiom IR  ->  AVR (ATmega328P) ELF + Intel HEX
#
#   python3 nexatoaxiom.py   prog.nexa     # Nexa  -> Axiom (.axm)
#   python3 axiomtoavrelf.py prog.axm      # Axiom -> AVR ELF (a.elf) + a.hex
#   avr-objcopy -O ihex a.elf a.hex        # (a.hex is also written directly)
#   avrdude ... -U flash:w:a.hex           # upload
#
# Same .axm syntax as the x86 backend. AVR is an 8-bit Harvard MCU with no OS
# and no FPU, so the desktop-only parts of the IR cannot be lowered:
#   * Axiom registers r0..r4 are modelled as 16-bit AVR register PAIRS, so all
#     integer values are TRUNCATED TO 16 BITS.
#   * WRITE / READ (stdout/stdin syscalls) and the float ops (I2F/F2I/F*) have
#     no meaning on bare metal and raise a clear error.
#   * SYS / ABORT become a clean halt (cli + self-loop) — bare metal never
#     "exits".
#   * Variables live in SRAM (.bss, 2 bytes each); strings / jump tables are
#     rejected (they imply print()).
# The integer + control-flow + GPIO subset is fully supported and is the
# realistic AVR target (counters, conditionals, loops, port I/O).
# ===========================================================================

import sys
import struct
from axiom_ir import (parse_axm, eliminate_dead, jumptable_labels,
                      optimize_common)

# --- ATmega328P facts -------------------------------------------------------
RAMSTART = 0x0100        # first SRAM byte (data space)
RAMEND   = 0x08FF        # last SRAM byte -> initial stack pointer
SPL_IO   = 0x3D
SPH_IO   = 0x3E

# Axiom register -> (low AVR reg, high AVR reg).  r0..r4 cover everything the
# Nexa compiler emits (r0/r1 arithmetic, r3 length, r4 buffer/address).
AX_PAIR = {
    "r0": (24, 25),
    "r1": (22, 23),
    "r2": (20, 21),
    "r3": (18, 19),
    "r4": (26, 27),   # X pair
}

# I/O register names usable with OUT/IN (data-space minus 0x20 = I/O address).
IO_ADDR = {
    "PINB": 0x03, "DDRB": 0x04, "PORTB": 0x05,
    "PINC": 0x06, "DDRC": 0x07, "PORTC": 0x08,
    "PIND": 0x09, "DDRD": 0x0A, "PORTD": 0x0B,
    "SPL": SPL_IO, "SPH": SPH_IO,
}

# ATmega328P interrupt vectors: name -> slot number (slot 0 = RESET). Each slot
# is a 2-word (4-byte) entry, since the hardware spaces vectors 2 words apart on
# parts with >8 KB flash.
VECTOR_SLOT = {
    "RESET": 0, "INT0": 1, "INT1": 2,
    "PCINT0": 3, "PCINT1": 4, "PCINT2": 5, "WDT": 6,
    "TIMER2_COMPA": 7, "TIMER2_COMPB": 8, "TIMER2_OVF": 9,
    "TIMER1_CAPT": 10, "TIMER1_COMPA": 11, "TIMER1_COMPB": 12, "TIMER1_OVF": 13,
    "TIMER0_COMPA": 14, "TIMER0_COMPB": 15, "TIMER0_OVF": 16,
    "SPI_STC": 17, "USART_RX": 18, "USART_UDRE": 19, "USART_TX": 20,
    "ADC": 21, "EE_READY": 22, "ANALOG_COMP": 23, "TWI": 24, "SPM_READY": 25,
}

# --- instruction word encoders (every one verified against avr-as) ----------
def _ldi(rd, k):   return 0xE000 | ((k & 0xF0) << 4) | ((rd - 16) << 4) | (k & 0x0F)
def _movw(rd, rr): return 0x0100 | ((rd >> 1) << 4) | (rr >> 1)
def _two(base, rd, rr): return base | ((rr & 0x10) << 5) | ((rd & 0x1F) << 4) | (rr & 0x0F)
def _add(rd, rr):  return _two(0x0C00, rd, rr)
def _adc(rd, rr):  return _two(0x1C00, rd, rr)
def _sub(rd, rr):  return _two(0x1800, rd, rr)
def _sbc(rd, rr):  return _two(0x0800, rd, rr)
def _cp(rd, rr):   return _two(0x1400, rd, rr)
def _cpc(rd, rr):  return _two(0x0400, rd, rr)
def _and(rd, rr):  return _two(0x2000, rd, rr)
def _or(rd, rr):   return _two(0x2800, rd, rr)
def _eor(rd, rr):  return _two(0x2400, rd, rr)
def _mov(rd, rr):  return _two(0x2C00, rd, rr)
def _mul(rd, rr):  return _two(0x9C00, rd, rr)
def _clr(rd):      return _eor(rd, rd)
def _lsl(rd):      return _add(rd, rd)
def _rol(rd):      return _adc(rd, rd)
def _lsr(rd):      return 0x9406 | (rd << 4)
def _ror(rd):      return 0x9407 | (rd << 4)
def _asr(rd):      return 0x9405 | (rd << 4)
def _dec(rd):      return 0x940A | (rd << 4)
def _sbrc(rr, b):  return 0xFC00 | (rr << 4) | (b & 7)   # skip next if bit b clear
def _ori(rd, k):   return 0x6000 | ((k & 0xF0) << 4) | ((rd - 16) << 4) | (k & 0x0F)
def _cpi(rd, k):   return 0x3000 | ((k & 0xF0) << 4) | ((rd - 16) << 4) | (k & 0x0F)
def _sbiw(rd, k):  return 0x9700 | ((k & 0x30) << 2) | (((rd - 24) >> 1) << 4) | (k & 0x0F)
def _adiw(rd, k):  return 0x9600 | ((k & 0x30) << 2) | (((rd - 24) >> 1) << 4) | (k & 0x0F)
def _out(a, rr):   return 0xB800 | ((a & 0x30) << 5) | ((rr & 0x1F) << 4) | (a & 0x0F)
def _in(rr, a):    return 0xB000 | ((a & 0x30) << 5) | ((rr & 0x1F) << 4) | (a & 0x0F)
def _sbi(a, b):    return 0x9A00 | ((a & 0x1F) << 3) | (b & 0x07)   # set I/O bit
def _cbi(a, b):    return 0x9800 | ((a & 0x1F) << 3) | (b & 0x07)   # clear I/O bit
def _push(rr):     return 0x920F | (rr << 4)
def _pop(rr):      return 0x900F | (rr << 4)
def _lds_w(rd, addr): return (0x9000 | (rd << 4), addr & 0xFFFF)
def _sts_w(addr, rd): return (0x9200 | (rd << 4), addr & 0xFFFF)
# conditional branch words (k = signed word offset, target = PC + 1 + k)
def _brbs(s, k):   return 0xF000 | ((k & 0x7F) << 3) | s
def _brbc(s, k):   return 0xF400 | ((k & 0x7F) << 3) | s
def _breq(k): return _brbs(1, k)
def _brne(k): return _brbc(1, k)
def _brlt(k): return _brbs(4, k)
def _brge(k): return _brbc(4, k)
def _brcs(k): return _brbs(0, k)
def _brcc(k): return _brbc(0, k)
RET  = 0x9508
RETI = 0x9518          # return from interrupt (re-enables global interrupts)
SEI  = 0x9478          # set global interrupt enable
CLI  = 0x94F8          # clear global interrupt enable
RJMP_SELF = 0xCFFF      # rjmp .  (k = -1 -> jumps to itself)
def _rjmp(k):  return 0xC000 | (k & 0x0FFF)
def _rcall(k): return 0xD000 | (k & 0x0FFF)

def _jmp_words(waddr):
    hi = ((waddr >> 16) & 0x3F)
    return (0x940C | ((hi >> 1) << 4) | (hi & 1), waddr & 0xFFFF)
def _call_words(waddr):
    hi = ((waddr >> 16) & 0x3F)
    return (0x940E | ((hi >> 1) << 4) | (hi & 1), waddr & 0xFFFF)


class AvrError(Exception):
    pass


# ---------------------------------------------------------------------------
# IR parsing, dead-code elimination, and the target-independent peepholes are
# shared with the x86 backend -- see axiom_ir.py. Only the AVR-specific
# adiw/sbiw immediate fusion lives here, layered on top of optimize_common().
# ---------------------------------------------------------------------------
def optimize_text(text, volatile_slots=()):
    text = optimize_common(text, volatile_slots)
    # AVR-specific: fold 'LDR r1,#k ; ADD/SUB r0,r1' (0<=k<=63) into one
    # adiw/sbiw r0,k. r0 maps to the r24:25 pair, exactly what adiw/sbiw use.
    out = []
    i, n = 0, len(text)
    while i < n:
        if (i + 1 < n
                and text[i][0] == "LDR" and len(text[i][1]) == 2
                and text[i][1][0] == "r1" and text[i][1][1].startswith("#")
                and text[i+1][0] in ("ADD", "SUB") and text[i+1][1] == ["r0", "r1"]):
            v = text[i][1][1][1:]
            try:
                k = int(v, 16) if v.lower().startswith("0x") else int(v)
            except ValueError:
                k = -1
            if 0 <= k <= 63:
                out.append(("ADIWK" if text[i+1][0] == "ADD" else "SBIWK",
                            ["r0", str(k)]))
                i += 2
                continue
        out.append(text[i])
        i += 1
    return out


# ---------------------------------------------------------------------------
# Register promotion: keep a leaf function's scalar locals in CPU registers
# instead of SRAM, so the load/test/modify/store loop bodies lose their lds/sts
# entirely. Only LEAF functions (those that call nothing — not even the mul/div
# helpers) are eligible, and only slots touched solely by LDR/STR within that
# one function. The promotion register pairs (r8..r15) are not used anywhere
# else in the backend or its helpers, so the values are never clobbered.
# LDR/STR of a promoted slot become movw register moves.
# ---------------------------------------------------------------------------
PROMO_PAIRS = [(8, 9), (10, 11), (12, 13), (14, 15)]
# sbiw/adiw only work on r24/r26/r28/r30. r24 is the scratch (AX r0) and r26 is
# AX r4, so the freely-available sbiw-capable pairs are Y (r28:29) and Z (r30:31).
# A promoted *loop counter* placed here can be decremented/tested in place, with
# no movw shuffling through the working register.
SBIW_PAIRS = [(28, 29), (30, 31)]


def _is_counter(slot, idxs, text):
    """True iff `slot` is a pure loop counter: its value (always carried in r0)
    is only ever consumed by a zero-test branch (JZ/JNZ) or an sbiw/adiw that
    stores straight back to it. Tracks r0's contents because the redundant-load
    pass may have dropped an explicit reload, so a value reaching a CMP/ADD/etc.
    must still disqualify the slot."""
    n = len(text)
    seen = False
    r0 = False                                  # does r0 currently hold `slot`?
    for k in idxs:
        op, args = text[k]
        a = args if isinstance(args, (list, tuple)) else []
        if op == "LABEL":
            r0 = False                          # branch target: r0 unknown
            continue
        if r0:                                  # r0 holds slot — check the use
            if op in ("JZ", "JNZ"):
                continue                        # zero-test, r0 unchanged
            if op in ("SBIWK", "ADIWK") and a and a[0] == "r0":
                nn = text[k+1] if k + 1 < n else ("", None)
                if not (nn[0] == "STR" and nn[1] == ["r0", slot]):
                    return False                # must store straight back
                continue                        # still slot-derived
            if op == "STR" and len(a) == 2 and a[0] == "r0":
                if a[1] != slot:
                    return False                # slot value escaping elsewhere
                continue
            if op == "LDR" and len(a) == 2 and a[0] == "r0":
                r0 = (a[1] == slot)
                seen = seen or r0
                continue
            if "r0" in a:
                return False                    # CMP/ADD/MOV/... reads the value
            continue                            # op doesn't touch r0
        # r0 does not hold slot
        if op == "LDR" and len(a) == 2 and a[1] == slot:
            if a[0] != "r0":
                return False
            r0 = True; seen = True
        elif op == "STR" and len(a) == 2 and a[1] == slot:
            if a[0] != "r0":
                return False
            r0 = True; seen = True
        elif a and a[0] == "r0":
            r0 = False                          # r0 overwritten with non-slot data
    return seen


def invert_loops(ops):
    """Rotate a provably-nonzero in-place counter loop from while-at-top into
    do-while form: drop the top `TSTW;BZ`, and turn the closing `SBIWKR;JMP top`
    into `SBIWKR;BNZ top` (the sbiw already set Z). Saves a test + an
    unconditional jump per loop. Only fires when the counter's initial value is
    a nonzero constant (so the first iteration always runs) and the loop's top
    and exit labels have no other references (no break/continue into them)."""
    ref = {}
    for op, args in ops:
        if op in ("JMP", "BZ", "BNZ", "JZ", "JNZ", "JA", "CALL") and args:
            ref[args[0]] = ref.get(args[0], 0) + 1
    n = len(ops)
    remove, change = set(), {}
    for i in range(n):
        op, args = ops[i]
        if not (op == "LDIW" and i + 3 < n
                and ops[i+1][0] == "LABEL"
                and ops[i+2][0] == "TSTW" and ops[i+2][1] == [args[0]]
                and ops[i+3][0] == "BZ"):
            continue
        P = args[0]
        v = args[1][1:]
        try:
            init = int(v, 16) if v.lower().startswith("0x") else int(v)
        except ValueError:
            init = 0
        if init == 0:
            continue
        top = ops[i+1][1]
        exit_lbl = ops[i+3][1][0]
        j = next((k for k in range(i+4, n)
                  if ops[k][0] == "LABEL" and ops[k][1] == exit_lbl), None)
        if j is None or j < 2:
            continue
        if not (ops[j-1][0] == "JMP" and ops[j-1][1][0] == top
                and ops[j-2][0] in ("SBIWKR", "ADIWKR") and ops[j-2][1][0] == P):
            continue
        if ref.get(top, 0) != 1 or ref.get(exit_lbl, 0) != 1:
            continue                            # break/continue target -> bail
        if any(ops[k][0] in ("LDIW", "SBIWKR", "ADIWKR", "TSTW")
               and ops[k][1] and ops[k][1][0] == P
               for k in range(i+4, j-2)):
            continue                            # counter touched mid-body -> bail
        remove.add(i+2); remove.add(i+3)        # drop TSTW ; BZ
        change[j-1] = ("BNZ", [top])            # JMP top -> BNZ top
    if not remove and not change:
        return ops
    return [change.get(idx, e) for idx, e in enumerate(ops) if idx not in remove]


def _inplace_rewrite(ops, cpairs):
    """Collapse the promoted load/test/modify/store idioms for counter pairs in
    `cpairs` into in-place sbiw/adiw on the pair itself (no movw via r0)."""
    R0 = [24, 25]
    out, i, n = [], 0, len(ops)
    while i < n:
        op, args = ops[i]
        # init:  LDR r0,#imm ; RMOV[P <- r0]
        if (op == "LDR" and len(args) == 2 and args[0] == "r0" and args[1].startswith("#")
                and i + 1 < n and ops[i+1][0] == "RMOV"
                and tuple(ops[i+1][1][0:2]) in cpairs and ops[i+1][1][2:] == R0):
            out.append(("LDIW", [ops[i+1][1][0], args[1]])); i += 2; continue
        if op == "RMOV" and args[0:2] == R0 and tuple(args[2:]) in cpairs:
            P = args[2:]                                   # [plo, phi]
            nxt = ops[i+1] if i + 1 < n else ("", None)
            # test+modify (tight inner loop): RMOV[r0<-P];JZ/JNZ;SBIWK/ADIWK;RMOV[P<-r0]
            if (nxt[0] in ("JZ", "JNZ") and i + 3 < n
                    and ops[i+2][0] in ("SBIWK", "ADIWK") and ops[i+2][1][0] == "r0"
                    and ops[i+3][0] == "RMOV" and ops[i+3][1][0:2] == P and ops[i+3][1][2:] == R0):
                out.append(("TSTW", [P[0]]))
                out.append(("BZ" if nxt[0] == "JZ" else "BNZ", nxt[1]))
                out.append(("SBIWKR" if ops[i+2][0] == "SBIWK" else "ADIWKR",
                            [P[0], ops[i+2][1][1]]))
                i += 4; continue
            # test only:  RMOV[r0<-P] ; JZ/JNZ
            if nxt[0] in ("JZ", "JNZ"):
                out.append(("TSTW", [P[0]]))
                out.append(("BZ" if nxt[0] == "JZ" else "BNZ", nxt[1]))
                i += 2; continue
            # modify only:  RMOV[r0<-P] ; SBIWK/ADIWK ; RMOV[P<-r0]
            if (nxt[0] in ("SBIWK", "ADIWK") and nxt[1][0] == "r0" and i + 2 < n
                    and ops[i+2][0] == "RMOV" and ops[i+2][1][0:2] == P and ops[i+2][1][2:] == R0):
                out.append(("SBIWKR" if nxt[0] == "SBIWK" else "ADIWKR",
                            [P[0], nxt[1][1]]))
                i += 3; continue
        out.append((op, args)); i += 1
    return out


def promote_registers(text, mmio_names=()):
    mmio_names = set(mmio_names)
    # 1) split into regions delimited by 'fn_NAME' / 'isr_NAME' labels (main is
    #    the lead-in). An interrupt handler is its own region AND is never a
    #    promotion site: it can fire between any two main-line instructions, so a
    #    variable it touches must live in memory (a register copy could be stale
    #    or clobber state the interrupted code was using).
    cur = "main"
    region = {cur: []}
    order = [cur]
    idx_of = {}
    isr_regions = set()
    for k, e in enumerate(text):
        if e[0] == "LABEL" and (e[1].startswith("fn_") or e[1].startswith("isr_")):
            cur = e[1][3:] if e[1].startswith("fn_") else e[1]
            if cur not in region:
                region[cur] = []
                order.append(cur)
                if e[1].startswith("isr_"):
                    isr_regions.add(cur)
        region[cur].append(k)
        idx_of[k] = cur

    # 2) per function: slots it touches, and whether it is a leaf (no CALL)
    touched = {f: {} for f in region}     # slot -> [reads, writes]
    bad = {f: set() for f in region}      # slots disqualified (non LDR/STR use)
    is_leaf = {f: (f not in isr_regions) for f in region}
    for f in region:
        for k in region[f]:
            op, args = text[k]
            if op == "LABEL":
                continue                  # label name is a string, not operands
            if op in ("CALL", "MUL", "DIV", "SHL", "SHR"):
                # CALL invokes a user function; MUL/DIV/SHL/SHR invoke helper
                # routines. Either way another routine runs, so to keep the
                # promotion registers provably untouched we only promote in
                # functions that invoke nothing at all.
                is_leaf[f] = False
            elif op == "LDR" and len(args) == 2 and not args[1].startswith("#"):
                touched[f].setdefault(args[1], [0, 0])[0] += 1
            elif op == "STR" and len(args) == 2:
                touched[f].setdefault(args[1], [0, 0])[1] += 1
            else:
                # any other appearance of a name disqualifies it (LEA/OUT/...)
                for a in args:
                    if a in touched[f]:
                        bad[f].add(a)

    # a slot may only be promoted if it is used in exactly ONE region
    slot_regions = {}
    for f in region:
        for s in touched[f]:
            slot_regions.setdefault(s, set()).add(f)

    promote = {}          # slot -> (lo, hi) register pair
    counter_pairs = set() # pairs holding an in-place loop counter
    for f in order:
        if not is_leaf[f]:
            continue
        cand = [s for s in touched[f]
                if s not in bad[f] and s not in mmio_names
                and slot_regions.get(s) == {f}]
        counters = [s for s in cand if _is_counter(s, region[f], text)]
        free_sbiw = list(SBIW_PAIRS)
        free_other = list(PROMO_PAIRS)
        # counters first claim the sbiw-capable pairs (in-place sbiw/adiw)
        for s in counters:
            if free_sbiw:
                promote[s] = free_sbiw.pop(0)
                counter_pairs.add(promote[s])
            elif free_other:
                promote[s] = free_other.pop(0)   # counter, but only movw available
        for s in cand:
            if s in promote:
                continue
            if free_other:
                promote[s] = free_other.pop(0)
            elif free_sbiw:
                promote[s] = free_sbiw.pop(0)

    if not promote:
        return text, set()

    # 3) rewrite LDR/STR of promoted slots into register-pair moves (RMOV)
    AX = {"r0": (24, 25), "r1": (22, 23), "r2": (20, 21),
          "r3": (18, 19), "r4": (26, 27)}
    moved = []
    for op, args in text:
        if op == "LDR" and len(args) == 2 and args[1] in promote:
            dlo, dhi = AX[args[0]]
            slo, shi = promote[args[1]]
            moved.append(("RMOV", [dlo, dhi, slo, shi]))    # reg <- promoted slot
        elif op == "STR" and len(args) == 2 and args[1] in promote:
            dlo, dhi = promote[args[1]]
            slo, shi = AX[args[0]]
            moved.append(("RMOV", [dlo, dhi, slo, shi]))    # promoted slot <- reg
        else:
            moved.append((op, args))

    # drop a redundant movw that immediately undoes the previous one:
    #   movw A,B ; movw B,A   (B already equals A) -> keep only the first
    #   movw A,B ; movw A,B   (identical)          -> keep only the first
    out = []
    for e in moved:
        if (e[0] == "RMOV" and out and out[-1][0] == "RMOV"):
            a = out[-1][1]; b = e[1]
            if b == a or b == [a[2], a[3], a[0], a[1]]:
                continue
        out.append(e)
    if counter_pairs:
        out = _inplace_rewrite(out, counter_pairs)
        out = invert_loops(out)
    return out, set(promote)


# ---------------------------------------------------------------------------
# SRAM data layout — every scalar slot is a 16-bit cell (2 bytes)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SRAM data layout — scalars are 16-bit cells (2 bytes) by default. With the
# __typemap sidecar and PACK_SCALARS, byte-wide types (i8/c8/bool) take just one
# SRAM byte. SRAM is the scarce resource on a 2 KB MCU, so every byte is real
# (no pages here). Widest-first ordering keeps things tidy; AVR has no alignment
# requirement for lds/sts, so packing tight is free.
# ---------------------------------------------------------------------------
PACK_SCALARS = True

# byte width on AVR (everything wider than 16 bits is already truncated to a
# 16-bit pair) and whether a 1-byte load must sign-extend.
_AVR_W = {"i8": 1, "c8": 1, "bool": 1}        # all else -> 2
_AVR_SIGNED = {"i8"}                          # c8/bool zero-extend; i8 sign-extends

def layout_data(data):
    addr = {}
    mmio = set()
    width = {}        # slot -> 1 or 2 SRAM bytes
    signed = {}       # slot -> True if a 1-byte load must sign-extend
    # read the type sidecar first
    types = {}
    if PACK_SCALARS:
        for name, directive, rest in data:
            if directive == ".typemap":
                for e in rest:
                    if "=" in e:
                        sl, ty = e.split("=", 1)
                        types[sl] = ty
    def slot_bytes(name):
        return _AVR_W.get(types.get(name, ""), 2)

    pending_alias = []
    # widest first -> tight packing with zero waste (sizes are 2 then 1)
    scalars = [(n, d, r) for (n, d, r) in data
               if d in (".qword", ".dword", ".word", ".byte")]
    others  = [(n, d, r) for (n, d, r) in data
               if d not in (".qword", ".dword", ".word", ".byte", ".typemap")]
    cur = RAMSTART
    for name, directive, rest in sorted(scalars, key=lambda e: -slot_bytes(e[0])):
        w = slot_bytes(name)
        addr[name] = cur
        width[name] = w
        signed[name] = (types.get(name, "") in _AVR_SIGNED)
        cur += w
    for name, directive, rest in others:
        if directive == ".at":
            addr[name] = int(rest[0], 0)
            mmio.add(name)
        elif directive == ".space":
            n = int(rest[0]); addr[name] = cur; cur += n
        elif directive == ".alias":
            pending_alias.append((name, rest[0]))
        elif directive == ".bytes":
            raise AvrError(
                f"data '{name}': string literals (.bytes) are not supported on AVR "
                f"— they come from print()/strings, which the AVR target lacks. "
                f"Compile a program without print()/string output for AVR.")
        elif directive == ".qaddrs":
            raise AvrError(
                f"data '{name}': jump tables (.qaddrs) are not supported on AVR. "
                f"A dense match() lowers to a jump table; use fewer/sparser arms "
                f"so it lowers to a comparison chain instead.")
    for name, target in pending_alias:
        if target not in addr:
            raise AvrError(f"alias '{name}' -> unknown target '{target}'")
        addr[name] = addr[target]
        width[name] = width.get(target, 2)
        signed[name] = signed.get(target, False)
    if cur - 1 > RAMEND:
        raise AvrError(
            f"data does not fit in SRAM: needs {cur - RAMSTART} bytes, "
            f"ATmega328P has {RAMEND - RAMSTART + 1}. Relocations using large "
            f"byte offsets (e.g. &x + 4096) overflow an 8-bit MCU's RAM.")
    return addr, mmio, width, signed, cur - RAMSTART


# ---------------------------------------------------------------------------
# Translate Axiom IR -> a flat list of micro-ops:
#   ("w", word)        one literal instruction word
#   ("jmp", label)     absolute jmp   (2 words, resolved in pass 2)
#   ("call", label)    absolute call  (2 words, resolved in pass 2)
#   ("label", name)    zero-size marker
# Branches are emitted as literal words with fixed local offsets (no labels).
# ---------------------------------------------------------------------------
def bit_io_rewrite(text, data_addr, mmio):
    """Turn full-byte stores to a bit-addressable low-I/O register (data 0x20-
    0x3F) into atomic sbi/cbi when the register provably only ever holds {0,
    1<<b} for a single bit b. Also drops a redundant clear-to-reset of such a
    register in straight-line entry code (the chip resets these registers to 0,
    so an initial cbi of an already-zero bit is a no-op) -- this is exactly the
    assumption hand-written startup code makes."""
    cand = {n: data_addr[n] - 0x20 for n in mmio
            if data_addr.get(n) is not None and 0x20 <= data_addr[n] <= 0x3F}
    if not cand:
        return text
    # collect the constants stored to each candidate (all must be constant)
    vals = {n: set() for n in cand}
    ok = {n: True for n in cand}
    for i, (op, args) in enumerate(text):
        if op == "STR" and len(args) == 2 and args[1] in cand:
            R = args[1]
            if (i and text[i-1][0] == "LDR" and text[i-1][1][0] == args[0]
                    and text[i-1][1][1].startswith("#")):
                v = text[i-1][1][1][1:]
                try:
                    vals[R].add((int(v, 16) if v.lower().startswith("0x") else int(v)) & 0xFF)
                except ValueError:
                    ok[R] = False
            else:
                ok[R] = False               # stored a non-constant -> not bit-only
    elig = {}
    for n in cand:
        if not ok[n]:
            continue
        bits = {v.bit_length() - 1 for v in vals[n] if v}
        if all((v & (v - 1)) == 0 for v in vals[n] if v) and len(bits) == 1:
            elig[n] = (cand[n], bits.pop())     # (io_addr, bit)
    if not elig:
        return text
    # rewrite LDR r0,#v ; STR r0,R  ->  SBI/CBI
    out, i, n = [], 0, len(text)
    while i < n:
        op, args = text[i]
        if (op == "LDR" and i + 1 < n and len(args) == 2 and args[0] == "r0"
                and args[1].startswith("#")
                and text[i+1][0] == "STR" and text[i+1][1][0] == "r0"
                and text[i+1][1][1] in elig):
            io, b = elig[text[i+1][1][1]]
            v = args[1][1:]
            iv = (int(v, 16) if v.lower().startswith("0x") else int(v)) & 0xFF
            out.append(("CBI" if iv == 0 else "SBI", [io, b]))
            i += 2
            continue
        out.append((op, args))
        i += 1
    # drop a redundant clear-to-reset in the straight-line prologue: the first
    # sbi/cbi seen for an io, if it's a cbi, before any branch-target label.
    targets = {a[0] for o, a in out
               if o in ("JMP", "BZ", "BNZ", "JZ", "JNZ", "JA") and a}
    seen_target = False
    touched = set()
    final, drop = [], None
    for idx, (op, args) in enumerate(out):
        if op == "LABEL" and args in targets:
            seen_target = True
        if op in ("SBI", "CBI"):
            io = args[0]
            if io not in touched:
                touched.add(io)
                if op == "CBI" and not seen_target:
                    continue                    # reset already cleared this bit
        final.append((op, args))
    return final


def translate(text, data_addr, mmio, vectors=(), width=None, signed=None):
    prog = []
    used = {"mul": False, "div": False, "shl": False, "shr": False}
    width = width or {}
    signed = signed or {}

    def emit(*words):
        for x in words:
            prog.append(("w", x))

    def lo(reg): return AX_PAIR[reg][0]
    def hi(reg): return AX_PAIR[reg][1]

    def need_pair(reg, op):
        if reg not in AX_PAIR:
            raise AvrError(f"{op}: register {reg} is not mapped on AVR (use r0..r4)")

    # ---- interrupt vector table (emitted ONLY when interrupts are used) ----
    # Each slot is a forced 2-word jmp ("vjmp"), because the hardware spaces
    # vectors 2 words apart. Slot 0 (RESET) jumps to startup; bound ISRs jump to
    # their handler; any unused slot below the highest one jumps to a tiny
    # reti-only default. Programs with no interrupts skip this entirely, so they
    # keep starting execution at flash address 0 exactly as before.
    need_default = False
    if vectors:
        slotmap = {0: "__start"}
        maxslot = 0
        for name, lbl in vectors:
            if name not in VECTOR_SLOT:
                raise AvrError(f"unknown interrupt vector '{name}'")
            s = VECTOR_SLOT[name]
            if s == 0:
                raise AvrError("cannot bind a handler to RESET (vector 0)")
            slotmap[s] = lbl
            maxslot = max(maxslot, s)
        for s in range(maxslot + 1):
            tgt = slotmap.get(s)
            if tgt is None:
                tgt = "__vec_default"
                need_default = True
            prog.append(("ctl", "vjmp", tgt))
        prog.append(("label", "__start"))

    # ---- startup: set stack pointer to RAMEND, then fall into main ----
    emit(_ldi(16, RAMEND & 0xFF), _out(SPL_IO, 16))
    emit(_ldi(16, (RAMEND >> 8) & 0xFF), _out(SPH_IO, 16))

    UNSUPPORTED = {
        "WRITE": "WRITE (stdout) needs an OS/UART — not available on bare-metal AVR",
        "READ":  "READ (stdin) needs an OS/UART — not available on bare-metal AVR",
        "I2F": "floating point has no AVR FPU", "F2I": "floating point has no AVR FPU",
        "FADD": "floating point has no AVR FPU", "FSUB": "floating point has no AVR FPU",
        "FMUL": "floating point has no AVR FPU", "FDIV": "floating point has no AVR FPU",
        "JMPIDX": "jump tables are unsupported on AVR (see .qaddrs note)",
        "LDRB": "byte array indexing is not yet supported on AVR",
        "STORB": "byte array indexing is not yet supported on AVR",
    }

    SREG_IO = 0x3F
    HELPER_CLOBBER = {0, 1, 2, 3, 18, 19, 20, 21, 22, 23, 24, 25}

    # Per-region register-clobber sets, transitively closed over CALL. An ISR
    # that calls a function must save every register that function (and anything
    # it calls) can modify -- not just the registers named in the ISR's own body.
    # Functions here are caller-save (only recursive frames are preserved), so
    # without this an interrupt firing mid-statement in the caller would have its
    # live registers silently corrupted by the handler's callee.
    def _region_clobbers(t):
        clob = {"main": set()}; calls = {"main": set()}; cur = "main"
        for op, args in t:
            if op == "LABEL" and isinstance(args, str) and (
                    args.startswith("fn_") or args.startswith("isr_")):
                cur = args; clob.setdefault(cur, set()); calls.setdefault(cur, set())
                continue
            if op == "LABEL":
                continue
            rc = clob[cur]
            if op in ("MUL", "DIV", "SHL", "SHR"):
                rc |= HELPER_CLOBBER
            if op == "CALL" and isinstance(args, (list, tuple)) and args:
                calls[cur].add(args[0]); continue
            if isinstance(args, (list, tuple)):
                for a in args:
                    if a in AX_PAIR:
                        rc.update(AX_PAIR[a])          # abstract r0..r4 -> real pair
                    elif isinstance(a, int):
                        rc.add(a)                      # RMOV promoted-pair real regs
        changed = True
        while changed:                                 # transitive closure
            changed = False
            for f, cs in calls.items():
                for g in cs:
                    if g in clob and not clob[g] <= clob[f]:
                        clob[f] |= clob[g]; changed = True
        return clob
    clob_of = _region_clobbers(text)
    # conservative fallback if a callee can't be resolved: every register the
    # backend can write as a callee (AX pairs, mul/div helpers, promotion pairs).
    _ALL_CALLEE_CLOB = set(HELPER_CLOBBER) | {18,19,20,21,22,23,24,25,26,27,28,29,30,31}

    # ISR context-save state. An interrupt can fire between any two main-line
    # instructions, so a handler must leave every register and SREG exactly as
    # it found them (this is what avr-gcc's ISR() does). We track which real
    # registers the handler touches and wrap it with the minimal push/pop.
    in_isr = False
    isr_ins = 0          # prog index just after the isr_ label (prologue goes here)
    isr_clob = set()     # real registers the handler writes

    _ti = 0
    while _ti < len(text):
        op, args = text[_ti]
        _next = text[_ti + 1] if _ti + 1 < len(text) else (None, None)
        _ti += 1
        if op == "LABEL":
            prog.append(("label", args))
            if isinstance(args, str) and args.startswith("isr_"):
                in_isr, isr_ins, isr_clob = True, len(prog), set()
            continue

        if op in UNSUPPORTED:
            raise AvrError(f"opcode {op} cannot target AVR: {UNSUPPORTED[op]}")

        # collect the registers an interrupt handler clobbers, so we can save
        # exactly those (plus SREG) on entry and restore them before RETI.
        if in_isr:
            if op in ("MUL", "DIV", "SHL", "SHR"):
                isr_clob |= HELPER_CLOBBER
            if op == "CALL" and isinstance(args, (list, tuple)) and args:
                isr_clob |= clob_of.get(args[0], _ALL_CALLEE_CLOB)
            if isinstance(args, (list, tuple)):
                for a in args:
                    if a in AX_PAIR:
                        isr_clob.update(AX_PAIR[a])
                    elif isinstance(a, int):
                        isr_clob.add(a)

        if op == "RETI" and in_isr:
            # build prologue (push r24; save SREG; push other clobbered regs)
            # and epilogue (mirror), then the reti itself.
            extra = sorted(isr_clob - {24})
            prologue = [("w", _push(24)), ("w", _in(24, SREG_IO)), ("w", _push(24))]
            prologue += [("w", _push(r)) for r in extra]
            prog[isr_ins:isr_ins] = prologue
            for r in reversed(extra):
                emit(_pop(r))
            emit(_pop(24), _out(SREG_IO, 24), _pop(24))
            emit(RETI)
            in_isr = False
            continue

        if op == "LDR":
            reg, src = args[0], args[1]
            need_pair(reg, op)
            if src.startswith("#"):
                v = src[1:]
                imm = int(v, 16) if v.lower().startswith("0x") else int(v)
                imm &= 0xFFFF
                # 8-bit MMIO store coming next: a memory-mapped register write
                # only stores the LOW byte (sts uses lo(reg)), so when this
                # freshly-loaded immediate fits in a byte and is consumed solely
                # by that store, the high-byte load is dead — skip it. (r0..r4
                # are reloaded before any later 16-bit use, so this is safe.)
                _no, _na = _next
                if (_no == "STR" and _na and _na[0] == reg
                        and _na[1] in mmio and 0 <= imm <= 0xFF):
                    emit(_ldi(lo(reg), imm & 0xFF))
                else:
                    emit(_ldi(lo(reg), imm & 0xFF), _ldi(hi(reg), (imm >> 8) & 0xFF))
            else:
                if src not in data_addr:
                    raise AvrError(f"LDR: unknown data label '{src}'")
                a = data_addr[src]
                if src in mmio:
                    # 8-bit hardware register, zero-extended into the pair. Low
                    # I/O registers (data 0x20..0x5F, e.g. PINB/DDRB/PORTB) have
                    # a 1-word `in`; everything else needs a 2-word lds.
                    if 0x20 <= a <= 0x5F:
                        emit(_in(lo(reg), a - 0x20))
                    else:
                        emit(*_lds_w(lo(reg), a))
                    emit(_clr(hi(reg)))
                elif width.get(src, 2) == 1:
                    # packed 1-byte slot: load the low byte, zero-extend the high
                    # byte. The front-end materializes byte-wide values as their
                    # narrow two's-complement (e.g. i8 -3 -> 253) and the wide
                    # slot stored them zero-extended, so zero-extension here
                    # reproduces the exact value the 2-byte slot held.
                    emit(*_lds_w(lo(reg), a))
                    emit(_clr(hi(reg)))
                else:
                    emit(*_lds_w(lo(reg), a))
                    emit(*_lds_w(hi(reg), a + 1))

        elif op == "STR":
            reg, dst = args[0], args[1]
            need_pair(reg, op)
            if dst not in data_addr:
                raise AvrError(f"STR: unknown data label '{dst}'")
            a = data_addr[dst]
            if dst in mmio:
                # 8-bit hardware register. Low I/O (data 0x20..0x5F: PINB,
                # DDRB, PORTB, ...) uses a 1-word `out`; higher registers
                # (timer, etc.) use a 2-word sts.
                if 0x20 <= a <= 0x5F:
                    emit(_out(a - 0x20, lo(reg)))
                else:
                    emit(*_sts_w(a, lo(reg)))
            elif width.get(dst, 2) == 1:
                emit(*_sts_w(a, lo(reg)))        # packed: store the low byte only
            else:
                emit(*_sts_w(a, lo(reg)))
                emit(*_sts_w(a + 1, hi(reg)))

        elif op == "MOV":
            d, s = args
            need_pair(d, op); need_pair(s, op)
            emit(_movw(lo(d), lo(s)))

        elif op == "RMOV":
            # promoted-register move: movw dst_pair <- src_pair (reg numbers)
            dlo, dhi, slo, shi = (int(x) for x in args)
            emit(_movw(dlo, slo))

        elif op == "ADD":
            d, s = args
            emit(_add(lo(d), lo(s)), _adc(hi(d), hi(s)))
        elif op == "SUB":
            d, s = args
            emit(_sub(lo(d), lo(s)), _sbc(hi(d), hi(s)))
        elif op == "ADIWK":            # r0 += k  (0<=k<=63), fused from LDR+ADD
            emit(_adiw(24, int(args[1])))
        elif op == "SBIWK":            # r0 -= k  (0<=k<=63), fused from LDR+SUB
            emit(_sbiw(24, int(args[1])))

        # ---- in-place loop-counter ops (counter kept in an sbiw-capable pair) ----
        elif op == "SBI":              # set single bit of a low-I/O register
            emit(_sbi(int(args[0]), int(args[1])))
        elif op == "CBI":              # clear single bit of a low-I/O register
            emit(_cbi(int(args[0]), int(args[1])))
        elif op == "LDIW":             # load 16-bit immediate into pair [plo]
            plo = int(args[0]); v = args[1][1:]
            imm = (int(v, 16) if v.lower().startswith("0x") else int(v)) & 0xFFFF
            emit(_ldi(plo, imm & 0xFF), _ldi(plo + 1, (imm >> 8) & 0xFF))
        elif op == "TSTW":             # set Z from pair [plo] without changing it
            emit(_sbiw(int(args[0]), 0))
        elif op == "SBIWKR":           # pair[plo] -= k, in place
            emit(_sbiw(int(args[0]), int(args[1])))
        elif op == "ADIWKR":           # pair[plo] += k, in place
            emit(_adiw(int(args[0]), int(args[1])))
        elif op == "BZ":               # bare branch-if-zero (Z already set)
            prog.append(("ctl", "beq", args[0]))
        elif op == "BNZ":              # bare branch-if-nonzero (Z already set)
            prog.append(("ctl", "bne", args[0]))
        elif op == "CMP":
            d, s = args
            emit(_cp(lo(d), lo(s)), _cpc(hi(d), hi(s)))

        elif op == "MUL":
            used["mul"] = True
            prog.append(("ctl", "call", "__mul16"))
        elif op == "DIV":
            used["div"] = True
            prog.append(("ctl", "call", "__div16"))
        elif op == "SHL":
            used["shl"] = True
            prog.append(("ctl", "call", "__shl16"))
        elif op == "SHR":
            used["shr"] = True
            prog.append(("ctl", "call", "__shr16"))

        # ---- comparison/logical materialisation: r0 = 0/1 (after a CMP) ----
        elif op in ("SETEQ", "SETNE", "SETLT", "SETGE"):
            br = {"SETEQ": _breq, "SETNE": _brne,
                  "SETLT": _brlt, "SETGE": _brge}[op]
            # ldi lo,1 ; ldi hi,0 ; br<cond> +1 (keep 1) ; ldi lo,0
            emit(_ldi(lo("r0"), 1), _ldi(hi("r0"), 0), br(1), _ldi(lo("r0"), 0))
        elif op == "SETGT":
            # 1 iff r0 > r1  ==  not(r0<r1) and not(r0==r1).
            # ldi 1 ; if < -> 0 ; if == -> 0 ; else (>) skip the 'ldi 0'.
            emit(_ldi(lo("r0"), 1), _ldi(hi("r0"), 0),
                 _brlt(2), _breq(1), _rjmp(1), _ldi(lo("r0"), 0))
        elif op == "SETLE":
            # 1 iff r0<r1 or r0==r1
            emit(_ldi(lo("r0"), 1), _ldi(hi("r0"), 0),
                 _brlt(2), _breq(1), _ldi(lo("r0"), 0))
        elif op == "SETAND":
            emit(_and(lo("r0"), lo("r1")), _and(hi("r0"), hi("r1")))
        elif op == "SETOR":
            emit(_or(lo("r0"), lo("r1")), _or(hi("r0"), hi("r1")))
        elif op == "SETXOR":
            emit(_eor(lo("r0"), lo("r1")), _eor(hi("r0"), hi("r1")))

        elif op == "JMP":
            prog.append(("ctl", "jmp", args[0]))
        elif op == "CALL":
            prog.append(("ctl", "call", args[0]))
        elif op == "RET":
            emit(RET)

        elif op == "JZ":
            prog.append(("ctl", "brz", args[0]))
        elif op == "JNZ":
            prog.append(("ctl", "brnz", args[0]))
        elif op == "JA":
            prog.append(("ctl", "bra", args[0]))

        elif op == "OUT":
            # OUT rX, IOREG  — write the low byte of rX to an I/O register
            reg, io = args
            need_pair(reg, op)
            a = IO_ADDR.get(io)
            if a is None:
                a = int(io, 0)
            if a > 0x3F:
                raise AvrError(f"OUT: I/O address {io} out of range for 'out'")
            emit(_out(a, lo(reg)))
        elif op == "IN":
            reg, io = args
            need_pair(reg, op)
            a = IO_ADDR.get(io, None)
            if a is None:
                a = int(io, 0)
            emit(_in(lo(reg), a), _clr(hi(reg)))

        elif op == "PUSH":
            reg = args[0]; need_pair(reg, op)
            emit(_push(hi(reg)), _push(lo(reg)))
        elif op == "POP":
            reg = args[0]; need_pair(reg, op)
            emit(_pop(lo(reg)), _pop(hi(reg)))

        elif op == "LEA":
            # LEA rX, label — load a data SRAM address as a 16-bit value
            reg, label = args
            need_pair(reg, op)
            if label not in data_addr:
                raise AvrError(
                    f"LEA '{label}': only data addresses are supported on AVR "
                    f"(taking the address of a function/code label is not)")
            a = data_addr[label]
            emit(_ldi(lo(reg), a & 0xFF), _ldi(hi(reg), (a >> 8) & 0xFF))

        elif op == "SEI":
            emit(SEI)                       # enable global interrupts
        elif op == "CLI":
            emit(CLI)                       # disable global interrupts
        elif op == "RETI":
            emit(RETI)                      # return from interrupt handler

        elif op in ("SYS", "ABORT"):
            emit(CLI, RJMP_SELF)            # clean halt

        else:
            raise AvrError(f"unsupported Axiom opcode for AVR: {op}")

    # ---- default interrupt handler (only if some vector slot is unused) ----
    if need_default:
        prog.append(("label", "__vec_default"))
        emit(RETI)

    # ---- runtime helper routines (appended once, if referenced) ----
    if used["mul"]:
        prog.append(("label", "__mul16"))
        # r0(r25:24) = (r0 * r1) low 16 bits ; clobbers real r0..r3
        emit(_mul(24, 22), _movw(2, 0),
             _mul(24, 23), _add(3, 0),
             _mul(25, 22), _add(3, 0),
             _movw(24, 2), _clr(1), RET)
    if used["shl"]:
        prog.append(("label", "__shl16"))
        emit(_mov(18, 22), _cpi(18, 0), _breq(4),
             _lsl(24), _rol(25), _dec(18), _brne(-4), RET)
    if used["shr"]:
        prog.append(("label", "__shr16"))
        emit(_mov(18, 22), _cpi(18, 0), _breq(4),
             _asr(25), _ror(24), _dec(18), _brne(-4), RET)
    if used["div"]:
        prog.append(("label", "__div16"))
        # unsigned 16-bit restoring division: quotient -> r25:24
        emit(_clr(2), _clr(3), _ldi(18, 16),
             _lsl(24), _rol(25), _rol(2), _rol(3),
             _cp(2, 22), _cpc(3, 23), _brcs(3),
             _sub(2, 22), _sbc(3, 23), _ori(24, 1),
             _dec(18), _brne(-12), RET)
    return prog


# ---------------------------------------------------------------------------
# Assemble micro-ops -> flash bytes (two passes for jmp/call targets)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Assemble micro-ops -> flash bytes.
# Control-flow ops ("ctl", kind, label) are relaxed iteratively: every jump
# starts as its shortest form (rjmp/rcall/direct conditional branch) and is
# promoted to a long form (jmp/call/skip+jmp) only if the target is out of the
# short form's reach. This is the classic branch-relaxation fixpoint and is why
# a small program stays tiny — almost everything fits the 2-byte forms.
#   jmp   : rjmp k                       (1w)  | jmp  addr                 (2w)
#   call  : rcall k                      (1w)  | call addr                 (2w)
#   brz   : sbiw r24,0 ; breq target     (2w)  | sbiw ; brne+2 ; jmp addr  (4w)
#   brnz  : sbiw r24,0 ; brne target     (2w)  | sbiw ; breq+2 ; jmp addr  (4w)
#   bra   : (unsigned above, after CMP)        | brcs+3 ; breq+2 ; jmp addr(4w)
# rjmp/rcall reach +/-2048 words; conditional branches reach +/-64 words.
# ---------------------------------------------------------------------------
RJMP_RANGE = (-2048, 2047)
BR_RANGE   = (-64, 63)

def _fits(off, rng):
    return rng[0] <= off <= rng[1]

def assemble(prog):
    SBIW0 = _sbiw(24, 0)   # sets Z from r0 (r24:25) without changing it

    # which ctl ops are currently "short"? start optimistic (all short)
    short = {}
    for i, e in enumerate(prog):
        if e[0] == "ctl":
            short[i] = True

    def ctl_words(kind, is_short):
        if kind == "vjmp":
            return 2            # vector-table jmp: forced 2 words (hw spacing)
        if kind in ("jmp", "call"):
            return 1 if is_short else 2
        if kind in ("brz", "brnz"):
            return 2 if is_short else 4
        if kind in ("beq", "bne"):
            return 1 if is_short else 3      # bare cond branch | brXX+2 ; jmp
        if kind == "bra":
            return 4            # no short form (two conditions); always long
        raise AvrError(f"bad ctl kind {kind}")

    # relaxation fixpoint: recompute offsets, re-decide short/long, repeat
    for _ in range(64):
        label_word = {}
        w = 0
        for i, e in enumerate(prog):
            if e[0] == "label":
                label_word[e[1]] = w
            elif e[0] == "w":
                w += 1
            else:  # ctl
                w += ctl_words(e[1], short[i])
        changed = False
        w = 0
        for i, e in enumerate(prog):
            if e[0] == "label":
                pass
            elif e[0] == "w":
                w += 1
            else:
                kind, label = e[1], e[2]
                if label not in label_word:
                    raise AvrError(f"branch to unknown label '{label}'")
                tgt = label_word[label]
                if kind in ("jmp", "call"):
                    fit = _fits(tgt - (w + 1), RJMP_RANGE)
                elif kind in ("brz", "brnz"):
                    # conditional branch sits one word after the sbiw (at w+1)
                    fit = _fits(tgt - (w + 2), BR_RANGE)
                elif kind in ("beq", "bne"):
                    fit = _fits(tgt - (w + 1), BR_RANGE)
                else:
                    fit = False
                if fit != short.get(i, False):
                    short[i] = fit
                    changed = True
                w += ctl_words(kind, short[i])
        if not changed:
            break

    # final encode
    words = []
    w = 0
    for i, e in enumerate(prog):
        if e[0] == "label":
            continue
        if e[0] == "w":
            words.append(e[1] & 0xFFFF)
            w += 1
            continue
        kind, label = e[1], e[2]
        tgt = label_word[label]
        is_short = short[i]
        if kind == "vjmp":
            w1, w2 = _jmp_words(tgt); words += [w1, w2]; w += 2
        elif kind == "jmp":
            if is_short:
                words.append(_rjmp(tgt - (w + 1))); w += 1
            else:
                w1, w2 = _jmp_words(tgt); words += [w1, w2]; w += 2
        elif kind == "call":
            if is_short:
                words.append(_rcall(tgt - (w + 1))); w += 1
            else:
                w1, w2 = _call_words(tgt); words += [w1, w2]; w += 2
        elif kind in ("brz", "brnz"):
            words.append(SBIW0); w += 1
            if is_short:
                off = tgt - (w + 1)
                words.append(_breq(off) if kind == "brz" else _brne(off)); w += 1
            else:
                # branch over the 2-word jmp on the opposite condition
                words.append(_brne(2) if kind == "brz" else _breq(2)); w += 1
                w1, w2 = _jmp_words(tgt); words += [w1, w2]; w += 2
        elif kind in ("beq", "bne"):
            if is_short:
                off = tgt - (w + 1)
                words.append(_breq(off) if kind == "beq" else _brne(off)); w += 1
            else:
                words.append(_brne(2) if kind == "beq" else _breq(2)); w += 1
                w1, w2 = _jmp_words(tgt); words += [w1, w2]; w += 2
        elif kind == "bra":
            words.append(_brcs(3)); words.append(_breq(2)); w += 2
            w1, w2 = _jmp_words(tgt); words += [w1, w2]; w += 2
    return b"".join(struct.pack("<H", x) for x in words)


# ---------------------------------------------------------------------------
# Intel HEX (what avrdude uploads); also obtainable via avr-objcopy -O ihex
# ---------------------------------------------------------------------------
def to_ihex(flash):
    out = []
    for off in range(0, len(flash), 16):
        chunk = flash[off:off + 16]
        n = len(chunk)
        rec = [n, (off >> 8) & 0xFF, off & 0xFF, 0x00] + list(chunk)
        chk = (-sum(rec)) & 0xFF
        out.append(":" + "".join(f"{b:02X}" for b in rec) + f"{chk:02X}")
    out.append(":00000001FF")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Minimal AVR ELF32 (ET_EXEC, EM_AVR): one PT_LOAD of flash at vaddr 0,
# plus .text / .shstrtab section headers so objcopy/objdump are happy.
# ---------------------------------------------------------------------------
def build_elf(flash):
    EM_AVR = 83
    ehsize, phentsize, shentsize = 52, 32, 40
    phnum, shnum = 1, 3
    phoff = ehsize
    text_off = phoff + phentsize * phnum            # flash bytes start here
    shstr = b"\x00.text\x00.shstrtab\x00"
    shstr_off = text_off + len(flash)
    shoff = shstr_off + len(shstr)

    def elf_header():
        e_ident = b"\x7fELF" + bytes([1, 1, 1, 0]) + b"\x00" * 8   # 32-bit, LE
        return struct.pack("<16sHHIIIIIHHHHHH",
            e_ident, 2, EM_AVR, 1,        # ET_EXEC, EM_AVR, version
            0,                            # e_entry = 0 (reset vector)
            phoff, shoff, 0x85,           # e_flags: avr5 arch (common)
            ehsize, phentsize, phnum, shentsize, shnum, 2)

    phdr = struct.pack("<IIIIIIII",
        1,            # PT_LOAD
        text_off,     # p_offset
        0, 0,         # p_vaddr, p_paddr = 0 (flash)
        len(flash), len(flash),   # filesz, memsz
        5,            # PF_R | PF_X
        0x1000)       # align

    sh_null = b"\x00" * shentsize
    sh_text = struct.pack("<IIIIIIIIII",
        1, 1, 0x6, 0, text_off, len(flash), 0, 0, 2, 0)   # PROGBITS ALLOC+EXEC
    sh_shstr = struct.pack("<IIIIIIIIII",
        7, 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0)    # STRTAB

    return (elf_header() + phdr + flash + shstr +
            sh_null + sh_text + sh_shstr)


def main():
    if len(sys.argv) != 2:
        print("usage: python3 axiomtoavrelf.py <file.axm>")
        sys.exit(1)
    path = sys.argv[1]
    try:
        text, data, vectors = parse_axm(path)
        mmio_names = {d[0] for d in data if d[1] == ".at"}
        # Slots the optimizer must treat as opaque: explicit `volatile`
        # declarations (front-end '.volatile' sidecar) plus every mmio register,
        # which is volatile by definition. Protecting these is what makes a
        # variable shared with an interrupt handler safe.
        volatile_names = set()
        for d in data:
            if d[1] == ".volatile":
                volatile_names |= set(d[2])
        protected = mmio_names | volatile_names
        text = eliminate_dead(text, protected)
        text = optimize_text(text, protected)
        text, promoted = promote_registers(text, protected)
        data = [d for d in data if d[0] not in promoted]
        data_addr, mmio, width, signed, sram = layout_data(data)
        text = bit_io_rewrite(text, data_addr, mmio)
        prog = translate(text, data_addr, mmio, vectors, width, signed)
        flash = assemble(prog)
    except AvrError as e:
        print(f"avr-backend: error: {e}", file=sys.stderr)
        sys.exit(1)

    elf_path = (path.rsplit(".", 1)[0] or "a") + ".elf"
    hex_path = (path.rsplit(".", 1)[0] or "a") + ".hex"
    with open(elf_path, "wb") as f:
        f.write(build_elf(flash))
    with open(hex_path, "w") as f:
        f.write(to_ihex(flash))

    print(f"Assembled {path} -> {elf_path}  (+ {hex_path})")
    print(f"  target: ATmega328P   flash: {len(flash)} bytes   SRAM data: {sram} bytes")
    print(f"  upload: avrdude -c <prog> -p m328p -U flash:w:{hex_path}:i")
    print(f"  (or:    avr-objcopy -O ihex {elf_path} {hex_path})")


if __name__ == "__main__":
    main()