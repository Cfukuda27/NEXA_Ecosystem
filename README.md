# Nexa & the Axiom Toolchain

Nexa is a small, statically-typed, memory-safe systems language with **no runtime
and no garbage collector**. Programs compile to a compact textual IR called
**Axiom assembly** (`.axm`), which is then assembled directly into a freestanding
**x86-64 Linux (Ubuntu) ELF executable** — no libc, no linker, no external
toolchain.

```
  .nexa  ──(nexatoaxiom.py)──▶  .axm  ──(axiom_x86.py)──▶  a.out (ELF)
  source        compiler        IR        assembler        executable
```

This README documents how to use the tools and how to write Nexa.

---

## 1. Requirements

* **Python 3** (3.10+ recommended) — the compiler and assembler are Python scripts.
* **Linux on x86-64** to *run* the produced ELF (it is a native Ubuntu binary).
  You can compile/assemble anywhere Python runs; you can only execute the output
  on x86-64 Linux.

No other dependencies. The tools are self-contained `.py` files.

---

## 2. The tools

| File | Role |
|------|------|
| `nexatoaxiom.py` | **Front-end compiler.** Parses a `.nexa` file (resolving `import`s), enforces types and memory safety, and emits Axiom assembly (`.axm`). |
| `axiom_x86.py` | **Back-end assembler.** Reads `.axm`, runs the IR optimizer, lowers to x86-64 machine code, and writes a complete ELF named `a.out`. |
| `axiom_ir.py` | **Shared IR layer** imported by `axiom_x86.py`: the `.axm` parser and the optimizer passes (dead-code elimination, common-subexpression peephole, immediate folding, literal-write merging). |

### 2.1 Compile a program

```bash
python3 nexatoaxiom.py myprog.nexa
# -> Compiled myprog.nexa -> myprog.axm
```

The output filename is the input with `.nexa` replaced by `.axm`. Any files the
program `import`s are pulled in automatically (they must sit next to the source).

### 2.2 Assemble to an executable

```bash
python3 axiom_x86.py myprog.axm
# -> Assembled myprog.axm -> a.out
#    text: <n> bytes ...   data: <n> bytes ...   total ELF: <n> bytes
```

The assembler always writes to `a.out` in the current directory.

### 2.3 Run it

```bash
chmod +x a.out
./a.out
echo "exit: $?"
```

### 2.4 End-to-end example

```bash
cat > hello.nexa <<'EOF'
fn main() -> void {
    print("Hello, Ubuntu ELF!");
}
EOF

python3 nexatoaxiom.py hello.nexa     # hello.nexa -> hello.axm
python3 axiom_x86.py   hello.axm      # hello.axm  -> a.out
chmod +x a.out && ./a.out             # prints: Hello, Ubuntu ELF!
```

### 2.5 Programs that read input

A program using `input()` reads from stdin. Feed it on the command line:

```bash
printf '6\n2.5\n' | ./a.out ; echo "exit: $?"
#        ^int  ^float — one value per line, in the order the program reads them
```

`input()` into an integer type does `atoi`; into a float type parses a float.
Reading text into a `c8` skips line-ending bytes, so text protocols should be
**length-prefixed** rather than newline-terminated.

---

## 3. The Nexa language

### 3.1 Design goals

Lower-level than C/Zig/Rust, memory-safe **by default with no `unsafe` escape
hatch**, no runtime or GC, all checks performed at compile time, strictly
procedural and structural (no classes, no closures, no inheritance).

### 3.2 Data types

| Category | Types | Notes |
|----------|-------|-------|
| Signed integers | `i8`, `i16`, `i32`, `i64` | bounds enforced (`i8` is −128..127, etc.) |
| Floats | `f8`, `f16`, `f32`, `f64` | |
| Characters | `c8`, `c16`, `c32`, `c64` | a `c8` is one byte; `'Q'` is a char literal |
| Boolean | `bool` | `true` / `false` |
| Pointer | `ptr` | an 8-byte address into value/code space |
| Arrays | `T[N]` | fixed-size; see the array rules below |

Physical storage widths: 1 byte (`i8`/`c8`/`bool`), 2 (`i16`/`c16`), 4
(`i32`/`c32`/`f32`), 8 (`i64`/`c64`/`f64`/`ptr`). `null` is a valid initializer
(zero). There are **no unsigned types** (`u8`…`u64` were removed).

### 3.3 Declarations and mutability

```nexa
let     x: i32 = 5;      // immutable — reassigning x is a compile error
let mut y: i32 = 5;      // mutable
const   MAX: i32 = 1000; // compile-time constant
```

### 3.4 Functions: `fn` (stack) vs `def` (heap)

```nexa
fn add(a: i32, b: i32) -> i32 {   // stack frame, re-entrant / recursion-safe
    let s: i32 = a + b;
    return s;
}

def scale(x: i32) -> i32 {        // single persistent heap frame, NOT recursion-safe
    let v: i32 = x * 2;
    return v;
}
```

All functions are **global** (no nested/local functions, no methods). `void` is
the no-return type. Every program needs exactly one `fn main() -> void`.

### 3.5 Operators

```
Arithmetic     +  -  *  /
Assignment-op  += -= *= /=        Step    ++  --
Comparison     ==  !=  <  >  <=  >=
Logic (bool)   &&  ||  ^^          Grouping ( )
Bitwise        <<  >>             (shift only; no native & | ~ in expressions)
```

There is **no `%` (modulo)**; compute it as `a - a / b * b`.

### 3.6 Control flow

```nexa
if (cond) {
    ...
}
else {
    ...
}

match(sel) {            // value match; arms are `if <literal> { }`, optional `else`
    if 1 {
        pick = 10;
    }
    if 2 {
        pick = 20;
    }
    else {
        pick = -1;
    }
}

while (i < 5) {
    i++;
}

loop {                  // infinite loop; exit with break
    kk++;
    if (kk >= 3) {
        break;
    }
}

for(let mut i: i32 = 0; i < 10; i++) {
    if (i / 2 * 2 != i) {
        continue;
    }
}
```

`break` and `continue` work in all loops. **All block bodies must be
multi-line** — `if (c) { x = 1; }` on one line is rejected; the `{`, the body,
and the `}` must be on separate lines.

### 3.7 Structs and `self`

```nexa
struct Device {
    let mut id: i32;
    let mut tag: c8;
    struct Sensor {          // structs nest
        let mut temp: i32;
        let mut voltage: i32;
    }
}
```

Struct fields are accessed with `.` and nested structs with `::`
(`Device::Sensor.temp`). A function can take a struct receiver via `self`:

```nexa
fn warm(self: Device, amount: i32) -> void {
    self::Sensor.temp = self::Sensor.temp + amount;
}

fn sync(self: (Device, Engine)) -> void {   // multi-root self searches the roots
    self::Sensor.temp += 5;
    self.rpm += 100;
}

fn debug_all(self) -> void {                 // universal self
    self::show_device();                     // function bridge: call via self
}
```

### 3.8 Strings, `print`, `println`

String literals appear inside `print`/`println`. Use `{name}` to interpolate a
variable, const, char, bool, or struct field:

```nexa
println("m={m} sh={sh} ch={ll}");
print("cmp={cmp} FLAG={FLAG}\n");       // print does not add a newline
println("temp={self::Sensor.temp}");
```

`println` appends a newline; `print` does not.

### 3.9 Input

```nexa
let n:  i32 = input();   // parses an integer from stdin
let f:  f64 = input();   // parses a float from stdin
```

### 3.10 Addresses and `&`

```nexa
let mut value: i32 = 42;
let pv: ptr = &value;    // address-of a variable -> ptr
let pf: i64 = &fact;     // address-of a function -> i64
```

Nexa also supports **byte-accurate relocation** of a variable's storage with
`&name = ...;`, and the compiler rejects relocations that would make two
variables overlap (see Memory Safety).

### 3.11 `abort` and `import`

```nexa
abort;                       // immediately exit the program (error model)
import "nexa_std.nexa";      // pull another .nexa file into this program
```

`import` brings the imported file's consts, structs, and functions directly into
scope. The imported file has **no `fn main()`** (it is a library).

### 3.12 Reserved words

```
null let mut const fn def self abort if else ifelse match for while loop
break continue print println input int float struct import return void
true false isr sei cli volatile
```

None of these may be used as a variable or const name.

---

## 4. Memory safety (enforced at compile time)

Nexa is memory-safe with **no unsafe escape**. The compiler rejects, among
others:

```nexa
let k: i32 = 5;  k = 9;     // error: 'k' is immutable (no 'mut')
&ms1 = &ms3;                // error: ms1 would overlap ms3
&ms1 = 1 + 2 + &ms3;        // error: ms3+3 lands inside ms3's 8 bytes
```

Every variable is guaranteed its own non-overlapping storage; relocations are
checked byte-accurately. There is no way to forge an aliasing pointer.

---

## 5. Practical language rules & gotchas

These follow from how the front-end parses Nexa. Keeping them in mind avoids
almost all "why won't this compile?" surprises.

* **One statement per line.** Block/`if`/loop/`match` bodies must be multi-line.
* **Comments are full-line only** (`// ...` on its own line). No trailing
  inline comments.
* **No function calls or array elements inside expressions or conditions.**
  Read an array element or call result into a scalar temp first, then use the
  temp:
  ```nexa
  let t: i32 = arr[i];     // ok
  if (t > 0) { ... }       // ok
  // if (arr[i] > 0)       // not allowed
  ```
* **Array index must be a simple variable or literal.** Arrays that are indexed
  at runtime must be 8-byte (`i64`) element type.
* **`for`-loop init must declare a fresh variable** (`for(let mut i ...)`), not
  reuse an existing one. If you need to start from an existing value, use a
  `while` loop instead.
* **Bare variable-to-variable assignment is rejected** for arithmetic types —
  write `x = y + 0;` instead of `x = y;`. (Booleans are assigned directly:
  `flag = true;`. A `bool` cannot be assigned from another `bool` *variable*; if
  you need that, model the flag as an `i64` 0/1 and compare with `== 1`.)
* **No `%` and no native `& | ~`** in expressions — only `<<` / `>>` for bits,
  and `&&` / `||` / `^^` are *boolean* operators, not bitwise.
* **Struct fields are declared `let mut name: T;`.**
* `input()` into a `c8` skips line-ending bytes; design text input as
  length-prefixed rather than newline-delimited.

---

## 6. The Axiom IR (`.axm`)

`.axm` is human-readable text. You normally never write it by hand, but
understanding it helps when debugging or extending the toolchain.

* **Comments** start with `;`. Blank lines are ignored.
* **Labels** end in a colon: `main:`.
* **Instructions** are `OP arg, arg`. Operands are registers `r0..r5`,
  immediates like `#100`, or symbol names.
* **Data directives** declare storage and bytes:
  * `name: .byte 0` / `.word` / `.dword` / `.qword` — a scalar slot (1/2/4/8 bytes)
  * `name: .bytes 72 101 108 ...` — a literal byte blob (e.g. a string)
  * `name: .space N` — N bytes of gap
  * `__typemap: .typemap a=i32 b=i8 s=i32` — a sidecar telling the backend each
    slot's source type, used to pack scalars to their real widths.

### 6.1 Registers

The abstract machine has registers `r0`–`r5`, mapped to x86-64 as:

| Axiom | x86-64 |
|-------|--------|
| `r0` | `rax` |
| `r1` | `rbx` |
| `r2` | `rcx` |
| `r3` | `rdx` |
| `r4` | `rsi` |
| `r5` | `rdi` |

### 6.2 Core opcodes (selection)

`LDR` (load reg ← imm/slot), `STR` (store slot ← reg), `LEA` (load address),
`MOV`, `ADD`/`SUB`/`MUL`/`DIV`, `SHL`/`SHR`, the `SETxx` comparison-to-register
ops (`SETEQ`/`SETNE`/`SETLT`/`SETGT`/`SETLE`/`SETGE`/…), `CMP`,
`JMP`/`JZ`/`JNZ`/`JA` (branches), `CALL`/`RET`, `WRITE`/`READ` (I/O via
syscalls), `SYS` (exit), and `ABORT`. Float ops (`FADD`/`FSUB`/`I2F`/…) and
firmware ops (`SEI`/`CLI`/`RETI` for interrupts) also exist.

---

## 7. What the assembler produces

`axiom_x86.py` lowers the (optimized) IR straight to machine code and writes a
minimal, conformant ELF:

* Load base address `0x400000`, 120-byte ELF+program headers.
* A single `PT_LOAD` segment marked **RWX** (smallest conformant layout; a
  W^X two-segment layout is also supported in the source).
* `.bytes` blobs are placed in the file image; scalar slots become BSS
  (loader-zeroed, not stored in the file).

### 7.1 Optimizations the backend runs

Before lowering, `axiom_x86.py` runs the `axiom_ir.py` pass pipeline, in order:

1. **`eliminate_dead`** — removes unreachable code and dead stores/values.
2. **`optimize_common`** — common-subexpression / redundant-load peephole
   (respects `volatile` slots, which are never cached in a register).
3. **`merge_literal_writes`** — fuses runs of adjacent literal `WRITE`s (e.g.
   consecutive `print` calls) into a single write over a concatenated blob when
   it is a net size win.
4. **`allocate_loop_registers`** — keeps hot loop values in registers.
5. **`fold_immediates`** — folds constant immediates.

The backend additionally performs two encoding optimizations during lowering:
**scalar packing** (uses the `.typemap` to store `i8`/`i16`/`i32` in their true
widths instead of full 8-byte slots, with width-correct `movsxd`/`movzx`
loads), and **base-register addressing** (when many data accesses cluster, it
loads a base into `rbp` once and uses compact `[rbp+disp8]` forms instead of
RIP-relative addressing).

---

## 8. Quick reference: the whole flow

```bash
# 1. write or edit  myprog.nexa   (and any imported .nexa beside it)
python3 nexatoaxiom.py myprog.nexa     # -> myprog.axm
python3 axiom_x86.py   myprog.axm      # -> a.out  (x86-64 Ubuntu ELF)
chmod +x a.out
./a.out ; echo "exit: $?"

# programs that read input:
printf '6\n2.5\n' | ./a.out
```

| Stage | Tool | In → Out |
|-------|------|----------|
| Compile | `nexatoaxiom.py` | `.nexa` → `.axm` |
| Assemble | `axiom_x86.py` (+ `axiom_ir.py`) | `.axm` → `a.out` (ELF) |
| Run | the OS | `./a.out` |

---

## 9. Self-hosting note (`axiom_x86.nexa` / `axiom_ir.nexa`)

The backend and IR optimizer have been **reimplemented in Nexa itself**, in two
source files mirroring `axiom_x86.py` and `axiom_ir.py`. Every pass was verified
byte-identical to the Python reference under fuzzing, and the two files compile
and assemble into a working native binary. These are the path toward a fully
self-hosting toolchain; the Python tools above remain the supported, complete
way to build programs today.
