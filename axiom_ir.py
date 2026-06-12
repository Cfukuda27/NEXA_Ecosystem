"""
axiom_ir.py — shared Axiom IR front-end for the Nexa backends.

Both backends (axiom_x86.py and axiomtoavrelf.py) parse the *same* Axiom IR
and run the *same* target-independent optimization pipeline defined here:

    text, data, vectors = parse_axm(path)
    text = eliminate_dead(text, mmio_names, jumptable_labels(data))
    text = optimize_common(text)

After this point the two backends differ ONLY in how they lower the optimized
IR into machine code (opcodes) and wrap it in an ELF. Everything above the
opcode layer — the IR grammar, dead-code/dead-store elimination, and the
semantics-preserving peepholes — lives here and is byte-for-byte identical for
every target.

The IR is a flat list of (OP, args) tuples plus ("LABEL", name) markers. OP is
upper-cased; args is a list of operand tokens (registers r0..r4, slot names,
"#imm" literals, or labels). The abstract machine has five scratch registers
r0..r4 that every backend maps onto real registers.
"""

# ---------------------------------------------------------------------------
# Parser — turns .axm text into (text, data, vectors). Identical surface
# syntax for every target; the backends interpret the directives.
# ---------------------------------------------------------------------------
def parse_axm(path):
    with open(path) as f:
        lines = f.readlines()
    mode = "text"
    text = []     # (op, args)  and ("LABEL", name)
    data = []     # (name, directive, rest_tokens)
    vectors = []  # (vector_name, handler_label) from '.vector NAME label'
    for raw in lines:
        if "; Data section" in raw:
            mode = "data"
            continue
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if mode == "text":
            if line.endswith(":"):
                text.append(("LABEL", line[:-1].strip()))
                continue
            parts = line.replace(",", " ").split()
            if parts[0].lower() == ".vector":
                vectors.append((parts[1], parts[2]))      # name, handler label
                continue
            text.append((parts[0].upper(), parts[1:]))
        else:
            if ":" not in line:
                continue
            name, rest = line.split(":", 1)
            p = rest.strip().split()
            if not p:
                continue
            data.append((name.strip(), p[0].lower(), p[1:]))
    return text, data, vectors


def jumptable_labels(data):
    """Code labels referenced by a `.qaddrs` jump table (used by `match`). These
    are reachable entry points even though no branch op names them directly, so
    dead-code elimination must treat them as live."""
    labs = set()
    for entry in data:
        if len(entry) >= 3 and entry[1] == ".qaddrs":
            labs.update(entry[2])
    return labs


# ---------------------------------------------------------------------------
# Dead-code / dead-store elimination (target independent — operates purely on
# the abstract IR and its r0..r4 registers).
# ---------------------------------------------------------------------------
_REG_WRITERS = {"MOV", "ADD", "SUB", "MUL", "DIV", "SHL", "SHR",
                "SETEQ", "SETNE", "SETLT", "SETGT", "SETLE", "SETGE",
                "SETAND", "SETOR", "SETXOR", "IN", "POP", "LEA",
                "I2F", "F2I", "FADD", "FSUB", "FMUL", "FDIV", "LDRB", "LDRQ"}

_DCE_REGS = {"r0", "r1", "r2", "r3", "r4"}


def _imm_zero(tok):
    if not tok.startswith("#"):
        return False
    v = tok[1:]
    try:
        return (int(v, 16) if v.lower().startswith("0x") else int(v)) == 0
    except ValueError:
        return False


def _dce_unreachable(text, extra_entries=()):
    """Drop instructions that can never be reached: anything after an
    unconditional transfer (JMP/RET/SYS/ABORT/RETI) until the next label that
    is branched to, is a function/interrupt entry, or is a jump-table target."""
    extra = set(extra_entries)
    targets = set()
    for op, args in text:
        if op in ("JMP", "JZ", "JNZ", "JA", "CALL") and isinstance(args, (list, tuple)) and args:
            targets.add(args[0])
    TRANSFER = {"JMP", "RET", "SYS", "ABORT", "RETI"}
    out, reachable = [], True
    for op, args in text:
        if op == "LABEL":
            is_entry = (args in targets or args in extra or args == "main"
                        or args.startswith("fn_") or args.startswith("isr_"))
            if is_entry:
                reachable = True
            if reachable:
                out.append((op, args))
            continue
        if reachable:
            out.append((op, args))
            if op in TRANSFER:
                reachable = False
    return out


def _dce_def_use(op, args, slots):
    """Return (defs, uses) restricted to tracked values: the r0..r4 registers
    plus the confined scalar `slots` of the current function."""
    d, u = set(), set()
    a = args if isinstance(args, (list, tuple)) else ()
    if op == "LDR":
        if a[0] in _DCE_REGS: d.add(a[0])
        if len(a) > 1 and not a[1].startswith("#") and a[1] in slots: u.add(a[1])
    elif op == "STR":
        if a[1] in slots: d.add(a[1])
        if a[0] in _DCE_REGS: u.add(a[0])
    elif op == "MOV":
        if a[0] in _DCE_REGS: d.add(a[0])
        if a[1] in _DCE_REGS: u.add(a[1])
    elif op in ("ADD", "SUB", "SETAND", "SETOR", "SETXOR", "MUL", "DIV", "SHL", "SHR"):
        if a[0] in _DCE_REGS: d.add(a[0])
        for x in a[:2]:
            if x in _DCE_REGS: u.add(x)
    elif op == "CMP":
        for x in a[:2]:
            if x in _DCE_REGS: u.add(x)
    elif op in ("SETLT", "SETGT", "SETLE", "SETGE", "SETEQ", "SETNE", "LEA", "IN", "POP"):
        if a and a[0] in _DCE_REGS: d.add(a[0])
    elif op == "LDRQ":            # dst = mem[base + index*8]
        if a and a[0] in _DCE_REGS: d.add(a[0])
        for x in a[1:3]:
            if x in _DCE_REGS: u.add(x)
    elif op == "STORQ":          # mem[base + index*8] = src
        for x in a[:3]:
            if x in _DCE_REGS: u.add(x)
    elif op == "OUT":
        if len(a) > 1 and a[1] in _DCE_REGS: u.add(a[1])
    elif op == "PUSH":
        if a and a[0] in _DCE_REGS: u.add(a[0])
    elif op == "CALL":
        d |= _DCE_REGS                  # a call clobbers all scratch registers
    elif op in ("WRITE", "READ"):
        # syscalls take no IR operands but DO read rsi (buffer = r4) and rdx
        # (length = r3); without this, DCE deletes the length/address setup.
        u |= {"r3", "r4"}
    elif op in ("JZ", "JNZ", "JA", "JMP", "RET", "SYS", "ABORT", "SEI", "CLI", "RETI"):
        pass
    else:
        # unknown op (incl. WRITE/READ/float/byte ops): be conservative —
        # treat every register/slot arg as used so nothing feeding it is removed
        for x in a:
            if x in _DCE_REGS or x in slots: u.add(x)
    return d, u


def _dce_dead_values(text, mmio_names):
    """Per-function liveness: remove stores to confined scalar slots that are
    never read before being overwritten, and loads/moves whose destination
    register is dead. MMIO stores (side effects) and values shared across
    functions or with an ISR are never touched. A region containing a computed
    jump (JMPIDX) is skipped entirely — its control flow isn't modeled here."""
    call_targets = {args[0] for op, args in text
                    if op == "CALL" and isinstance(args, (list, tuple)) and args}
    starts = [i for i, (op, args) in enumerate(text)
              if op == "LABEL" and (args == "main" or args in call_targets
                                    or args.startswith("fn_") or args.startswith("isr_"))]
    if not starts:
        return text
    bounds = starts + [len(text)]

    slot_regions = {}
    for ri in range(len(starts)):
        for i in range(bounds[ri], bounds[ri + 1]):
            op, args = text[i]
            if op == "LDR" and isinstance(args, (list, tuple)) and len(args) == 2 and not args[1].startswith("#"):
                slot_regions.setdefault(args[1], set()).add(ri)
            elif op == "STR" and isinstance(args, (list, tuple)) and len(args) == 2:
                slot_regions.setdefault(args[1], set()).add(ri)
    confined = {sl: next(iter(rs)) for sl, rs in slot_regions.items()
                if len(rs) == 1 and sl not in mmio_names}

    drop = set()
    for ri in range(len(starts)):
        lo, hi = bounds[ri], bounds[ri + 1]
        if any(text[i][0] in ("JMPIDX", "WRITE", "READ") for i in range(lo, hi)):
            # computed jumps and the print/input (itoa / syscall) machinery use
            # registers and temporaries in ways this IR-level liveness can't
            # model safely -> leave the whole region untouched.
            continue
        slots = {sl for sl, r in confined.items() if r == ri}
        read_here = {sl for sl in slots
                     if any(text[i][0] == "LDR" and text[i][1][1] == sl
                            for i in range(lo, hi)
                            if isinstance(text[i][1], (list, tuple)) and len(text[i][1]) == 2)}
        label_idx = {text[i][1]: i for i in range(lo, hi) if text[i][0] == "LABEL"}

        def succ(i):
            op, args = text[i]
            if op == "JMP":
                t = label_idx.get(args[0]); return {t} if t is not None else set()
            if op in ("JZ", "JNZ", "JA"):
                s = set()
                if i + 1 < hi: s.add(i + 1)
                t = label_idx.get(args[0])
                if t is not None: s.add(t)
                return s
            if op in ("RET", "RETI", "SYS", "ABORT"):
                return set()
            return {i + 1} if i + 1 < hi else set()

        live_out = {i: set() for i in range(lo, hi)}
        defs, uses = {}, {}
        for i in range(lo, hi):
            defs[i], uses[i] = _dce_def_use(text[i][0], text[i][1], slots)
        changed = True
        while changed:
            changed = False
            for i in range(hi - 1, lo - 1, -1):
                op = text[i][0]
                if op in ("RET", "RETI"):
                    lo_out = set(_DCE_REGS)          # caller may use a return value
                elif op in ("SYS", "ABORT"):
                    lo_out = set()
                else:
                    lo_out = set()
                    for s in succ(i):
                        lo_out |= (uses[s] | (live_out[s] - defs[s]))
                if lo_out != live_out[i]:
                    live_out[i] = lo_out
                    changed = True

        for i in range(lo, hi):
            op, args = text[i]
            if op in ("LDR", "MOV"):
                dst = args[0]
                if dst in _DCE_REGS and dst not in live_out[i]:
                    drop.add(i)
            elif op == "STR" and args[1] in read_here and args[1] not in live_out[i]:
                drop.add(i)

    if not drop:
        return text
    return [e for i, e in enumerate(text) if i not in drop]


def eliminate_dead(text, mmio_names=(), extra_entries=()):
    """Reachability + liveness driven dead-code elimination, run to a fixpoint.
    `mmio_names` are slots that must never be removed (hardware side effects);
    `extra_entries` are reachable labels not named by any branch op (jump-table
    targets). x86 passes mmio_names=() (no memory-mapped I/O)."""
    mmio_names = set(mmio_names)
    while True:
        t2 = _dce_unreachable(text, extra_entries)
        t2 = _dce_dead_values(t2, mmio_names)
        if t2 == text:
            return text
        text = t2


# ---------------------------------------------------------------------------
# Target-independent peepholes (semantics preserving). The only job here is to
# delete instructions the front-end emitted redundantly; the resulting IR is
# still valid for every backend.
#
#  (1) compare-to-zero -> direct branch:
#        LDR rX,#0 ; CMP r0,rX ; SETEQ/SETNE r0 ; JZ/JNZ L   ->   JZ/JNZ L
#  (2) redundant-load elimination:
#        LDR rX,S ; ... ; LDR rX,S   (rX still holds S)  ->  drop the 2nd load
# ---------------------------------------------------------------------------
def optimize_common(text, volatile_slots=()):
    # `volatile_slots` are memory locations the optimizer must treat as opaque:
    # a volatile read may observe a value written by an interrupt handler or by
    # hardware between two ordinary instructions, so its load is never reused and
    # never dropped, and storing to it never lets a later load be elided.
    volatile_slots = set(volatile_slots)
    # ---- pass 1: compare-to-zero -> direct conditional branch ----
    out = []
    i, n = 0, len(text)
    while i < n:
        if (i + 3 < n
                and text[i][0] == "LDR" and len(text[i][1]) == 2
                and _imm_zero(text[i][1][1])
                and text[i+1][0] == "CMP" and text[i+1][1] == ["r0", text[i][1][0]]
                and text[i+2][0] in ("SETEQ", "SETNE") and text[i+2][1] == ["r0"]
                and text[i+3][0] in ("JZ", "JNZ")):
            seteq = text[i+2][0] == "SETEQ"
            jz = text[i+3][0] == "JZ"
            if seteq:
                jz = not jz
            out.append(("JZ" if jz else "JNZ", text[i+3][1]))
            i += 4
            continue
        out.append(text[i])
        i += 1
    text = out

    # ---- pass 2: redundant-load elimination via per-register value tracking ----
    holds = {}                      # reg -> slot label currently held
    out = []
    for op, args in text:
        if op == "LABEL":
            holds.clear()
            out.append((op, args))
            continue
        if op == "LDR" and len(args) == 2 and not args[1].startswith("#"):
            reg, slot = args
            if slot in volatile_slots:
                # opaque: must re-read every time, and never becomes "held"
                holds[reg] = None
                out.append((op, args))
                continue
            if holds.get(reg) == slot:
                continue            # already in that register — drop the load
            holds[reg] = slot
            out.append((op, args))
            continue
        if op == "LDR":             # LDR reg, #imm
            holds[args[0]] = None
            out.append((op, args))
            continue
        if op == "STR":
            reg, slot = args
            for r in list(holds):
                if holds[r] == slot and r != reg:
                    holds[r] = None     # memory changed; other copies are stale
            # a store to a volatile slot does not let a later load of it be
            # elided (the value may change again before we read it back)
            holds[reg] = None if slot in volatile_slots else slot
            out.append((op, args))
            continue
        if op in ("CALL", "WRITE", "READ"):
            holds.clear()               # clobbers registers (syscall / callee)
            out.append((op, args))
            continue
        if op in ("JMP", "JMPIDX", "RET", "RETI", "ABORT", "SYS"):
            out.append((op, args))
            holds.clear()               # unconditional transfer / barrier
            continue
        if op in ("CMP", "JZ", "JNZ", "JA", "OUT", "PUSH"):
            out.append((op, args))      # do not write a register
            continue
        if op in _REG_WRITERS and args:
            holds[args[0]] = None       # destination register clobbered
        out.append((op, args))
    return out

# ---------------------------------------------------------------------------
# Target-independent IR transforms shared by every backend (each backend must
# be able to lower the forms they introduce: OP r0,#imm for fold_immediates).
# ---------------------------------------------------------------------------

def _imm_to_s32(v):
    """Map the 64-bit value `LDR r1,#v` would produce to a signed-32 immediate,
    or None if it needs a full 64-bit load (not foldable)."""
    if 0 <= v <= 0xFFFFFFFF:
        return v if v <= 0x7FFFFFFF else v - 0x100000000
    if -0x80000000 <= v < 0:
        return v
    return None

# sizes
SETCC_SIZE = 3 + 4   # setcc al (3) + movzx rax,al (4) = 7
CMP_SIZE   = 3
LOGIC_SIZE = 3
SHIFT_SIZE = 6       # mov rcx,rbx (3) + shift rax,cl (3)

# ---------- IR ----------

def fold_immediates(text_t):
    """Fold `LDR r1,#imm ; OP r0,r1` -> `OP r0,#imm` (DIV excluded).

    r1 must be dead after the OP. In this IR r1 is a pure scratch register that
    is reloaded before every use, so it is dead at the end of any basic block; we
    verify it locally — if r1 is read before being rewritten/clobbered within the
    block, the fold is skipped. The byte output is unchanged: an immediate add is
    identical in result to loading then adding."""
    FOLDABLE = {"ADD", "SUB", "MUL", "CMP", "SETAND", "SETOR", "SETXOR", "SHL", "SHR"}
    BLOCK_END = {"JMP", "JZ", "JNZ", "JA", "RET", "RETI", "SYS", "ABORT", "JMPIDX", "CALL"}

    def r1_dead_after(idx):
        for k in range(idx + 1, len(text_t)):
            op, a = text_t[k]
            if op == "LABEL" or op in BLOCK_END:
                return True            # block ends without reading r1 (CALL clobbers it)
            a = a if isinstance(a, (list, tuple)) else ()
            if op in ("LDR", "MOV", "POP", "IN") and a and a[0] == "r1":
                return True            # r1 rewritten before any read -> dead
            # any other appearance of r1 as an operand is a read -> live
            if "r1" in a[1:] or (op not in ("LDR", "MOV", "POP", "IN") and "r1" in a):
                return False
        return True

    out, i, n = [], 0, len(text_t)
    while i < n:
        op, a = text_t[i]
        if (op == "LDR" and len(a) == 2 and a[0] == "r1" and a[1].startswith("#")
                and i + 1 < n and text_t[i+1][0] in FOLDABLE
                and list(text_t[i+1][1]) == ["r0", "r1"]):
            raw = a[1][1:]
            v = int(raw, 16) if raw[:2].lower() == "0x" else int(raw)
            s32 = _imm_to_s32(v)
            op2 = text_t[i+1][0]
            shift_ok = op2 not in ("SHL", "SHR") or (s32 is not None and 0 <= s32 < 64)
            if s32 is not None and shift_ok and r1_dead_after(i + 1):
                out.append((op2, ["r0", f"#{s32}"]))
                i += 2
                continue
        out.append(text_t[i]); i += 1
    return out


def merge_literal_writes(text_t, data_t):
    """Collapse runs of adjacent string-literal prints into one WRITE — but only
    where it is a net byte win.

    A literal print lowers to the fixed triple
        LEA r4, <strblob> ; LDR r3, #len ; WRITE
    Consecutive triples (nothing — not even a LABEL — between them) can be fused:
    concatenating payloads and issuing one WRITE emits byte-for-byte the same
    output. Each fused triple saves ~12-15 text bytes, but the concatenated blob
    costs data bytes — and crucially, a blob shared by other prints (e.g. the
    newline DEDUP'd across every println) would be DUPLICATED into the merge,
    losing more in data than is saved in text.

    So each run is merged only if  text_saved > data_added, where a component
    blob's bytes are reclaimed (not a real cost) when ALL of its references lie
    inside mergeable runs, counted once globally. Output bytes/order unchanged."""
    from collections import Counter
    blob = {name: [int(x) & 0xFF for x in rest]
            for (name, d, rest) in data_t if d == ".bytes"}

    def is_lit(i):
        return (i + 2 < len(text_t)
                and text_t[i][0] == "LEA"
                and len(text_t[i][1]) == 2 and text_t[i][1][0] == "r4"
                and text_t[i][1][1] in blob
                and text_t[i+1][0] == "LDR"
                and len(text_t[i+1][1]) == 2 and text_t[i+1][1][0] == "r3"
                and text_t[i+1][1][1].startswith("#")
                and text_t[i+2][0] == "WRITE")

    # global LEA reference count per blob, and references that sit inside runs
    refs = Counter(text_t[i][1][1] for i in range(len(text_t))
                   if is_lit(i))
    runs, i = [], 0
    while i < len(text_t):
        if is_lit(i):
            j = i
            while is_lit(j):
                j += 3
            if (j - i) // 3 > 1:
                runs.append(list(range(i, j, 3)))   # LEA index of each literal
            i = j
        else:
            i += 1
    in_runs = Counter()
    for r in runs:
        for k in r:
            in_runs[text_t[k][1][1]] += 1
    fully = {n for n in in_runs if refs[n] == in_runs[n]}   # all uses are mergeable

    TRIPLE = 12                       # conservative bytes saved per fused triple
    merge_at, reclaimed = {}, set()
    for r in runs:
        names = [text_t[k][1][1] for k in r]
        combined = []
        for n in names:
            combined += blob[n]
        data_added = len(combined)
        for n in set(names):          # reclaim a fully-consumed blob once
            if n in fully and n not in reclaimed:
                data_added -= len(blob[n])
        if (len(r) - 1) * TRIPLE > data_added:
            merge_at[r[0]] = (r[-1] + 3, combined)   # (end index exclusive, bytes)
            for n in set(names):
                if n in fully:
                    reclaimed.add(n)

    out, new_blobs, ctr, i = [], [], 0, 0
    while i < len(text_t):
        if i in merge_at:
            end, combined = merge_at[i]
            nm = f"__litmerge_{ctr}"; ctr += 1
            new_blobs.append((nm, ".bytes", [str(b) for b in combined]))
            out.append(("LEA", ["r4", nm]))
            out.append(("LDR", ["r3", f"#{len(combined)}"]))
            out.append(("WRITE", []))
            i = end
        else:
            out.append(text_t[i]); i += 1
    return out, data_t + new_blobs