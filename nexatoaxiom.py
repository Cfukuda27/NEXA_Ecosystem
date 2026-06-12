#!/usr/bin/env python3
import sys
import re
import os
import struct
import math

# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

TYPE_INFO = {
    "i8":  (".byte",  1),
    "i16": (".word",  2),
    "i32": (".dword", 4),
    "i64": (".qword", 8),
    "f8":  (".byte",  1),
    "f16": (".word",  2),
    "f32": (".dword", 4),
    "f64": (".qword", 8),
    "c8":  (".byte",  1),
    "c16": (".word",  2),
    "c32": (".dword", 4),
    "c64": (".qword", 8),
    "bool": (".byte", 1),
    "ptr": (".qword", 8),   # an address into value/code space (8 bytes)
}

INT_TYPES   = {"i8", "i16", "i32", "i64"}
FLOAT_TYPES = {"f8", "f16", "f32", "f64"}
CHAR_TYPES  = {"c8", "c16", "c32", "c64"}
BOOL_TYPES  = {"bool"}
PTR_TYPES   = {"ptr"}
ALL_TYPES   = INT_TYPES | FLOAT_TYPES | CHAR_TYPES | BOOL_TYPES | PTR_TYPES

INT_BOUNDS = {
    "i8":  (-128,                 127),
    "i16": (-32768,               32767),
    "i32": (-2147483648,          2147483647),
    "i64": (-9223372036854775808, 9223372036854775807),
}

REMOVED_UNSIGNED = {"u8", "u16", "u32", "u64"}

# Reserved words that cannot be used as variable or const names
RESERVED = {"null", "let", "mut", "const", "fn", "def", "self", "abort",
            "if", "else", "ifelse", "match", "for", "while", "loop",
            "break", "continue", "print", "println", "input", "int", "float",
            "struct", "import", "return", "void", "true", "false",
            "isr", "sei", "cli", "volatile"}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

TYPE_PAT = r"i8|i16|i32|i64|f8|f16|f32|f64|c8|c16|c32|c64|bool|ptr"
# A value can be: number, char, string, identifier (const/var ref),
# OR an arithmetic expression (digits, idents, + - * / ( ) and spaces).
# String and char literals are matched first so their contents aren't split.
VAL_PAT  = r"\"[^\"]*\"|'.'|[0-9A-Za-z_+\-*/()<>=!&|^ .]+?"
# An array size can be a number or a const name
SIZE_PAT = r"[0-9]+|[a-zA-Z_]\w*"

# Detects whether a RHS string is an arithmetic expression (has an operator
# outside of a leading unary minus) rather than a single literal/ident.
EXPR_OP_RE = re.compile(r"<<|>>|==|!=|<=|>=|&&|\|\||\^\^|[+\-*/()<>]")
SINGLE_NEG_NUM_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")

# const NAME: type = value;   (no let, no mut)
CONST_RE = re.compile(
    r"const\s+([a-zA-Z_]\w*)"
    r"\s*:\s*(" + TYPE_PAT + r")"
    r"\s*=\s*"
    r"(" + VAL_PAT + r")"
    r"\s*;"
)

# let [mut] name: type = value;
SCALAR_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)"
    r"\s*:\s*(" + TYPE_PAT + r")"
    r"\s*=\s*"
    r"(" + VAL_PAT + r")"
    r"\s*;"
)

# let [mut] name: type[N] = [...] or "string";
ARRAY_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)"
    r"\s*:\s*(" + TYPE_PAT + r")"
    r"\s*\[\s*(" + SIZE_PAT + r")\s*\]"
    r"\s*=\s*"
    r"(\[.*?\]|\"[^\"]*\")"
    r"\s*;"
)

# Scalar reassignment:  name = value;
REASSIGN_SCALAR_RE = re.compile(
    r"([a-zA-Z_]\w*)"
    r"\s*=\s*"
    r"(" + VAL_PAT + r")"
    r"\s*;"
)

# Array element reassignment:  name[N] = value;
REASSIGN_ARRAY_RE = re.compile(
    r"([a-zA-Z_]\w*)"
    r"\s*\[\s*(" + SIZE_PAT + r")\s*\]"
    r"\s*=\s*"
    r"(" + VAL_PAT + r")"
    r"\s*;"
)

# Array element READ into a scalar (the index may be a runtime variable):
#   let [mut] x: T = arr[i];      x = arr[i];
# A digit/const index reads the named slot directly; a variable index lowers
# to LEA arr_0 ; LDR idx ; LDRQ  (mem[base + idx*8]) and so requires an 8-byte
# (i64/ptr) element array. The '[' is not part of VAL_PAT, so these forms never
# collide with the scalar-decl / scalar-reassign / expression handlers.
ARRAY_READ_DECL_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*"
    r"([a-zA-Z_]\w*)\s*\[\s*(" + SIZE_PAT + r")\s*\]\s*;$")
ARRAY_READ_REASSIGN_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*=\s*"
    r"([a-zA-Z_]\w*)\s*\[\s*(" + SIZE_PAT + r")\s*\]\s*;$")

# Step ops — increment/decrement (postfix) and compound-assign.
# Scalar increment/decrement:  name++;   name--;
STEP_INCDEC_SCALAR_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*(\+\+|--)\s*;"
)
# Array element increment/decrement:  name[i]++;  name[i]--;
STEP_INCDEC_ARRAY_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*\[\s*(" + SIZE_PAT + r")\s*\]\s*(\+\+|--)\s*;"
)
# Scalar compound assign:  name += expr;  (also -= *= /=)
STEP_COMPOUND_SCALAR_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*(\+=|-=|\*=|/=)\s*(" + VAL_PAT + r")\s*;"
)
# Array element compound assign:  name[i] += expr;
STEP_COMPOUND_ARRAY_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*\[\s*(" + SIZE_PAT + r")\s*\]\s*(\+=|-=|\*=|/=)\s*(" + VAL_PAT + r")\s*;"
)

UNSIGNED_RE = re.compile(
    r"(?:let\s+(?:mut\s+)?|const\s+)[a-zA-Z_]\w*\s*:\s*(u8|u16|u32|u64)"
)

# Control flow.  Condition is everything between the parens; body opens with {.
IF_RE     = re.compile(r"if\s*\((.*)\)\s*\{$")
# Parenless form:  if <cond> {   (condition is everything up to the brace)
IF_NOPAREN_RE = re.compile(r"if\s+(.+?)\s*\{$")
IFELSE_RE = re.compile(r"ifelse\s*\((.*)\)\s*\{$")
ELSE_RE   = re.compile(r"else\s*\{$")
MATCH_RE  = re.compile(r"match\s*\(\s*([a-zA-Z_]\w*)\s*\)\s*\{$")
# Inside a match: `if <value> {` or `if <op> <value> {` (op defaults to ==)
MATCH_ARM_RE = re.compile(r"if\s*(==|!=|<=|>=|<|>)?\s*(" + VAL_PAT + r")\s*\{$")
CLOSE_RE  = re.compile(r"^\}$")

# Loops.  `while (cond) {` is a conditional loop; `loop {` is infinite.
# `break;` exits the innermost loop; `continue;` jumps to its top.
WHILE_RE    = re.compile(r"while\s*\((.*)\)\s*\{$")
LOOP_RE     = re.compile(r"loop\s*\{$")
BREAK_RE    = re.compile(r"break\s*;$")
CONTINUE_RE = re.compile(r"continue\s*;$")
# C-style for:  for(<init>; <cond>; <step>) {   — clauses never contain ';'.
FOR_RE      = re.compile(r"for\s*\(([^;]*);([^;]*);([^;]*)\)\s*\{$")
# abort; — hard, clean program exit. Usable anywhere (top level, conditionals,
# loops, match arms). No propagation, no catch.
ABORT_RE    = re.compile(r"abort\s*;$")
# sei; / cli; — enable / disable global (hardware) interrupts. Bare statements
# like abort;. On AVR these become the SEI/CLI instructions; on x86 they are
# 0-byte no-ops (user-mode code has no interrupt flag and there is no timer).
SEI_RE      = re.compile(r"sei\s*;$")
CLI_RE      = re.compile(r"cli\s*;$")
# isr <VECTOR> { ... } — an interrupt service routine bound to a named hardware
# vector (e.g. TIMER1_COMPA). Parsed like a void, parameter-less function whose
# emitted body ends in RETI and is wired into the interrupt vector table.
ISR_HDR_RE  = re.compile(r"isr\s+([A-Za-z_]\w*)\s*\{$")
# print("...");  — Python-style format string. {name} interpolates a variable
# or const; \n \t \\ \{ \} \" are escapes. Greedy capture spans the first quote
# to the last quote before the closing paren.
PRINT_RE    = re.compile(r'print\s*\(\s*"(.*)"\s*\)\s*;$')
# println("...");  — identical to print() but always appends a trailing
# newline after the formatted text. The string argument is optional:
# println(); emits a single blank line.
PRINTLN_RE  = re.compile(r'println\s*\(\s*(?:"(.*)")?\s*\)\s*;$')
# A function may declare `self` as an implicit, compile-time receiver. It is
# NEVER a positional value argument and carries zero runtime cost. Three forms:
#   self: Device            single-root  — self IS that struct (self.field)
#   self: (Device, Engine)  multi-root   — self is a namespace (self::Device.field)
#   self                    universal    — every global struct + global function
SELF_SINGLE_RE = re.compile(r"^self\s*:\s*([A-Za-z_]\w*)$")
SELF_MULTI_RE  = re.compile(
    r"^self\s*:\s*\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)$")
# Function bridge through self: `self::callee(args)` lowers to `callee(args)`.
SELF_CALL_DECL_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*"
    r"self::([a-zA-Z_]\w*)\s*\((.*)\)\s*;$")
SELF_CALL_REASSIGN_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*=\s*self::([a-zA-Z_]\w*)\s*\((.*)\)\s*;$")
SELF_CALL_STMT_RE = re.compile(r"self::([a-zA-Z_]\w*)\s*\((.*)\)\s*;$")

# ---- '&' reference / relocation (explicit address-space control) -------
# RHS reference:   let r: ptr = &x;   r = &func;   — r receives the ADDRESS of
#                  a variable or function (a pure reference, lowered to LEA).
# LHS relocation:  &a = <addr-expr>;  — relocate 'a' to a byte-accurate address
#                  computed from integer arithmetic and '&var' (a variable's
#                  address), e.g. &a = &c + 8;  The manifest tracks every
#                  variable's byte footprint and rejects any overlap, so two
#                  variables can never occupy the same memory.
REF_DECL_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(i64|ptr)\s*=\s*&\s*([a-zA-Z_]\w*)\s*;$")
REF_REASSIGN_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*=\s*&\s*([a-zA-Z_]\w*)\s*;$")
RELOC_RE = re.compile(
    r"&\s*([a-zA-Z_]\w*)\s*=\s*(.+?)\s*;$")
# A '&var' term inside a relocation's address expression.
ADDR_REF_RE = re.compile(r"&\s*([a-zA-Z_]\w*)")
# Builtin value-producing calls on the RHS of a declaration or assignment:
#   input()        — read one char from stdin
#   int(ch)        — digit char -> integer value  ('7' -> 7)
#   float(ch)      — digit char -> IEEE double     ('7' -> 7.0)
#   bool(ch)       — '0' -> false, any other char -> true
BUILTIN_DECL_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*"
    r"(input|int|float|bool)\s*\(\s*([a-zA-Z_]\w*)?\s*\)\s*;$"
)
BUILTIN_REASSIGN_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*=\s*"
    r"(input|int|float|bool)\s*\(\s*([a-zA-Z_]\w*)?\s*\)\s*;$"
)

# ---- functions (stack-linked; declared with `fn`) ----------------------
#   fn name(p1: t1, p2: t2) -> ret_type { ... return expr; }
#   fn name() -> void { ... }
#   fn main() { ... }      — the special entry point (no params, no arrow)
# Calls (whole-RHS or statement forms only; not inside larger expressions):
#   let x: T = name(args);   x = name(args);   name(args);
FN_DEF_RE = re.compile(
    r"fn\s+([a-zA-Z_]\w*)\s*\((.*)\)\s*->\s*(" + TYPE_PAT + r"|void)\s*\{\s*$"
)
FN_MAIN_RE = re.compile(r"fn\s+main\s*\(\s*\)\s*(?:->\s*void\s*)?\{\s*$")
# ---- heap functions (declared with `def`) -------------------------------
# Identical syntax to `fn`, but a `def` is a HEAP function: its frame lives in
# persistent (heap/static) storage rather than on the hardware stack. The
# caller does NOT save/restore a `def`'s frame window around a call, so a `def`
# keeps a single persistent frame — it is not re-entrant / not recursion-safe,
# in deliberate contrast to a `fn`. There is no `def main()`.
DEF_DEF_RE = re.compile(
    r"def\s+([a-zA-Z_]\w*)\s*\((.*)\)\s*->\s*(" + TYPE_PAT + r"|void)\s*\{\s*$"
)
RETURN_RE  = re.compile(r"return\b\s*(.*?)\s*;$")
# ---- import (file inclusion) --------------------------------------------
#   import "path/to/file.nexa";
# Brings another Nexa file's top-level structs, functions and consts into the
# current file (flat, direct access — call them as if local). Resolved against
# the importing file's directory; '~' expands to the user's home directory.
IMPORT_RE = re.compile(r'import\s+"([^"]*)"\s*;\s*$')
CALL_DECL_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*"
    r"([a-zA-Z_]\w*)\s*\((.*)\)\s*;$"
)
CALL_REASSIGN_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)\s*\((.*)\)\s*;$"
)
CALL_STMT_RE = re.compile(r"([a-zA-Z_]\w*)\s*\((.*)\)\s*;$")


# ---- structs -----------------------------------------------------------
#   struct Name { ... }      — fields are zero-initialised, must be mutable
#   field access:  Outer::Inner.field   ('::' steps through nested structs,
#                  '.' selects the field). A.field for a top-level struct.
STRUCT_OPEN_RE        = re.compile(r"struct\s+([A-Za-z_]\w*)\s*\{$")
STRUCT_FIELD_RE       = re.compile(r"let\s+mut\s+([A-Za-z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*;")
STRUCT_FIELD_IMMUT_RE = re.compile(r"let\s+([A-Za-z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*;")
FIELD_ACCESS_PAT      = r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*\.[A-Za-z_]\w*"
FIELD_ACCESS_RE       = re.compile(r"^" + FIELD_ACCESS_PAT + r"$")
# Generalized access: a root followed by one or more '.'/'::' segments. The
# convention is '::' for namespaces (structs / nested structs / functions) and
# '.' for fields, but both are accepted as navigation so mixed paths like
# 'self.Device::Sensor.temp' resolve. Used for self / alias / instance / struct.
GEN_ACCESS_PAT        = r"[A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)+"
GEN_ACCESS_RE         = re.compile(r"^" + GEN_ACCESS_PAT + r"$")
GEN_ACCESS_FIND_RE    = re.compile(r"([A-Za-z_]\w*)((?:(?:::|\.)[A-Za-z_]\w*)+)")
DECL_FROM_FIELD_RE    = re.compile(
    r"let\s+(mut\s+)?([A-Za-z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*(" + GEN_ACCESS_PAT + r")\s*;")
VAR_FROM_FIELD_RE     = re.compile(r"([A-Za-z_]\w*)\s*=\s*(" + GEN_ACCESS_PAT + r")\s*;")
# Field write / compound-assign with an arbitrary expression RHS:
#   self::Sensor.temp = self::Sensor.temp + amount;
#   self.Engine.rpm += 100;     foo.Engine.rpm = b * 10;
FIELD_ASSIGN_RE       = re.compile(
    r"(" + GEN_ACCESS_PAT + r")\s*(\+=|-=|\*=|/=|=)\s*(.+?)\s*;")
FIELD_WRITE_RE        = re.compile(r"(" + FIELD_ACCESS_PAT + r")\s*=\s*(.+?)\s*;")

# Scalar-LHS statements whose RHS is an arbitrary expression that may MIX field
# accesses (navigated with '::' / a final '.') with literals, consts and vars,
# e.g.  let t: i32 = A.x + B.y;   t = A.x * 2;   t += S::Inner.q;
# These capture a broad RHS; the handlers gate on the RHS actually containing a
# resolvable access, then lower each access to its mangled slot label before
# the expression engine runs (the field-LHS path already does this).
DECL_EXPR_ACCESS_RE     = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*(.+?)\s*;$")
REASSIGN_EXPR_ACCESS_RE = re.compile(r"([a-zA-Z_]\w*)\s*=\s*(.+?)\s*;$")
COMPOUND_EXPR_ACCESS_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*(\+=|-=|\*=|/=)\s*(.+?)\s*;$")

# Local self aliases (only legal where the function has NO receiver self):
#   let foo: self = self;        universal alias to the whole global namespace
#   let bar: i32  = self;        type-filtered alias (only that type is visible)
SELF_ALIAS_RE       = re.compile(r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*self\s*=\s*self\s*;$")
SELF_TYPED_ALIAS_RE = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*self\s*;$")
# Local struct instance:  let mut local_net: Network = Network;
STRUCT_INSTANCE_RE  = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*;$")
# Namespace-qualified call through self or a self-alias:  root::warm(10);
NS_CALL_DECL_RE     = re.compile(
    r"let\s+(mut\s+)?([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")\s*=\s*"
    r"([A-Za-z_]\w*)::([a-zA-Z_]\w*)\s*\((.*)\)\s*;$")
NS_CALL_REASSIGN_RE = re.compile(
    r"([a-zA-Z_]\w*)\s*=\s*([A-Za-z_]\w*)::([a-zA-Z_]\w*)\s*\((.*)\)\s*;$")
NS_CALL_STMT_RE     = re.compile(r"([A-Za-z_]\w*)::([a-zA-Z_]\w*)\s*\((.*)\)\s*;$")

# Identifier (for detecting const references)
IDENT_RE = re.compile(r"^[a-zA-Z_]\w*$")

# ---------------------------------------------------------------------------
# Float encoding
# ---------------------------------------------------------------------------

def encode_f8(v):
    if v == 0.0:
        return 0x00
    sign = 0
    if v < 0:
        sign = 1
        v = -v
    exp = math.floor(math.log2(v))
    biased = max(0, min(7, exp + 3))
    mantissa = min(15, int((v / (2 ** exp) - 1.0) * 16))
    return (sign << 7) | (biased << 4) | mantissa

def encode_f16(v):
    return struct.unpack(">H", struct.pack(">e", v))[0]

def encode_f32(v):
    return struct.unpack(">I", struct.pack(">f", v))[0]

def encode_f64(v):
    return struct.unpack(">Q", struct.pack(">d", v))[0]

FLOAT_ENCODERS = {
    "f8": encode_f8, "f16": encode_f16,
    "f32": encode_f32, "f64": encode_f64,
}

# ---------------------------------------------------------------------------
# Const table — name -> {'ty', 'value'(raw literal string), 'int_value'}
# ---------------------------------------------------------------------------

class ConstTable:
    def __init__(self):
        self.consts = {}   # name -> dict(ty, raw, int_value)

    def define(self, name, ty, raw, int_value, lineno):
        if name in self.consts:
            raise ValueError(
                f"Line {lineno}: const '{name}' is already defined — "
                f"all const names must be unique"
            )
        self.consts[name] = {'ty': ty, 'raw': raw, 'int_value': int_value}

    def is_const(self, name):
        return name in self.consts

    def raw_of(self, name):
        return self.consts[name]['raw']

    def int_of(self, name):
        return self.consts[name]['int_value']


# ---------------------------------------------------------------------------
# Expression engine — tokenize, shunting-yard, fold or emit
# ---------------------------------------------------------------------------

EXPR_TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<float>[0-9]+\.[0-9]+)
      | (?P<hex>0[xX][0-9a-fA-F]+)
      | (?P<int>[0-9]+)
      | (?P<char>'.')
      | (?P<bool>true|false)
      | (?P<ident>[a-zA-Z_]\w*)
      | (?P<op><<|>>|==|!=|<=|>=|&&|\|\||\^\^|[+\-*/()<>])
    )
""", re.VERBOSE)

# Precedence (higher binds tighter), C convention:
#   * /  >  + -  >  << >>  >  relational  >  equality  >  &&  >  ^^  >  ||
PRECEDENCE = {
    '*': 8, '/': 8,
    '+': 7, '-': 7,
    '<<': 6, '>>': 6,
    '<': 5, '>': 5, '<=': 5, '>=': 5,
    '==': 4, '!=': 4,
    '&&': 3,
    '^^': 2,
    '||': 1,
}

# Operators that produce a bool result (used for type-checking the target)
COMPARE_OPS = {'<', '>', '<=', '>=', '==', '!=', '&&', '||', '^^'}
ARITH_OPS   = {'+', '-', '*', '/'}
SHIFT_OPS   = {'<<', '>>'}


def expr_yields_bool(rpn):
    """An expression yields a bool if its last (top-level) operator is a comparison/logical op."""
    last_op = None
    for kind, val in rpn:
        if kind == 'op':
            last_op = val
    return last_op in COMPARE_OPS


def expr_has_shift(rpn):
    """True if the TOP-LEVEL operator is a shift (result is the shift value).
    A shift nested under a comparison is fine — it produces an integer that
    feeds the comparison, which yields a bool. Only a shift whose result is
    the expression's final value constrains the target to an integer type."""
    last_op = None
    for kind, val in rpn:
        if kind == 'op':
            last_op = val
    return last_op in SHIFT_OPS


def is_expression(raw):
    """True if raw is an arithmetic expression rather than a single literal."""
    raw = raw.strip()
    # String/char literals are never expressions
    if raw.startswith('"') or (raw.startswith("'") and raw.endswith("'")):
        return False
    # A bare negative or positive number is a single literal, not an expression
    if SINGLE_NEG_NUM_RE.match(raw):
        return False
    # Otherwise, the presence of any operator means it's an expression.
    # (a lone identifier has no operator -> not an expression -> handled as const ref)
    return bool(EXPR_OP_RE.search(raw))


def expr_tokenize(expr, lineno):
    tokens = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = EXPR_TOKEN_RE.match(expr, pos)
        if not m or m.end() == pos:
            raise ValueError(f"Line {lineno}: cannot parse expression near {expr[pos:]!r}")
        pos = m.end()
        if m.group('float'):
            tokens.append(('num', m.group('float')))
        elif m.group('hex'):
            tokens.append(('num', str(int(m.group('hex'), 16))))
        elif m.group('int'):
            tokens.append(('num', m.group('int')))
        elif m.group('char'):
            tokens.append(('char', m.group('char')))
        elif m.group('bool'):
            tokens.append(('num', '1' if m.group('bool') == 'true' else '0'))
        elif m.group('ident'):
            tokens.append(('ident', m.group('ident')))
        elif m.group('op'):
            tokens.append(('op', m.group('op')))
    return tokens


def expr_to_rpn(tokens, lineno):
    """Shunting-yard with unary-minus handling (encoded as 0 - x)."""
    output = []
    ops = []
    prev = None
    for kind, val in tokens:
        if kind in ('num', 'char', 'ident'):
            output.append((kind, val))
        elif kind == 'op':
            if val == '(':
                ops.append(val)
            elif val == ')':
                while ops and ops[-1] != '(':
                    output.append(('op', ops.pop()))
                if not ops:
                    raise ValueError(f"Line {lineno}: mismatched parentheses")
                ops.pop()
            else:
                if val == '-' and (prev is None or
                                   (prev[0] == 'op' and prev[1] != ')')):
                    output.append(('num', '0'))
                while (ops and ops[-1] != '(' and
                       PRECEDENCE.get(ops[-1], 0) >= PRECEDENCE[val]):
                    output.append(('op', ops.pop()))
                ops.append(val)
        prev = (kind, val)
    while ops:
        top = ops.pop()
        if top in '()':
            raise ValueError(f"Line {lineno}: mismatched parentheses")
        output.append(('op', top))
    return output


def expr_operand_value(kind, val, consts, manifest, lineno):
    """Resolve an operand to ('const', number) or ('var', name) or ('imm', number)."""
    if kind == 'num':
        return ('imm', float(val) if '.' in val else int(val))
    if kind == 'char':
        return ('imm', ord(val[1]))
    if kind == 'ident':
        if consts.is_const(val):
            return ('imm', consts.int_of(val))
        if val in manifest:
            if manifest[val]['is_array']:
                raise ValueError(
                    f"Line {lineno}: '{val}' is an array and cannot be used "
                    f"directly in an expression"
                )
            return ('var', val)
        # Array-element slot reference like arr_2 — generated internally by
        # step ops on array elements. The base array must exist and the slot
        # is a real data label, so load it directly.
        if '_' in val:
            base, _, idx = val.rpartition('_')
            if idx.isdigit() and base in manifest and manifest[base]['is_array']:
                return ('var', val)
        raise ValueError(
            f"Line {lineno}: '{val}' is not a defined const or variable"
        )
    raise ValueError(f"Line {lineno}: unexpected operand {val!r}")


def expr_is_constant(rpn, consts, manifest):
    """True if every operand is compile-time constant (no runtime variable)."""
    for kind, val in rpn:
        if kind == 'ident':
            if consts.is_const(val):
                continue
            if val in manifest:
                return False  # runtime variable
            # unknown -> let later resolution raise
            return False
    return True


def fold_constant_expr(rpn, consts, lineno):
    """Evaluate a fully-constant RPN expression to a single Python number."""
    stack = []
    for kind, val in rpn:
        if kind == 'num':
            stack.append(float(val) if '.' in val else int(val))
        elif kind == 'char':
            stack.append(ord(val[1]))
        elif kind == 'ident':
            if not consts.is_const(val):
                raise ValueError(f"Line {lineno}: '{val}' is not a defined const")
            stack.append(consts.int_of(val))
        elif kind == 'op':
            if len(stack) < 2:
                raise ValueError(f"Line {lineno}: malformed expression")
            b = stack.pop(); a = stack.pop()
            if val == '+':
                stack.append(a + b)
            elif val == '-':
                stack.append(a - b)
            elif val == '*':
                stack.append(a * b)
            elif val == '/':
                if b == 0:
                    raise ValueError(f"Line {lineno}: division by zero in constant expression")
                if isinstance(a, int) and isinstance(b, int):
                    stack.append(int(a / b))
                else:
                    stack.append(a / b)
            # relational / equality -> 1 or 0
            elif val == '<':
                stack.append(1 if a < b else 0)
            elif val == '>':
                stack.append(1 if a > b else 0)
            elif val == '<=':
                stack.append(1 if a <= b else 0)
            elif val == '>=':
                stack.append(1 if a >= b else 0)
            elif val == '==':
                stack.append(1 if a == b else 0)
            elif val == '!=':
                stack.append(1 if a != b else 0)
            # logical -> treat nonzero as true, result 1 or 0
            elif val == '&&':
                stack.append(1 if (a != 0 and b != 0) else 0)
            elif val == '||':
                stack.append(1 if (a != 0 or b != 0) else 0)
            elif val == '^^':
                stack.append(1 if ((a != 0) != (b != 0)) else 0)
            # bitwise shifts — integer operands only
            elif val == '<<':
                if not (isinstance(a, int) and isinstance(b, int)):
                    raise ValueError(f"Line {lineno}: shift operands must be integers")
                if b < 0:
                    raise ValueError(f"Line {lineno}: negative shift amount")
                stack.append(a << b)
            elif val == '>>':
                if not (isinstance(a, int) and isinstance(b, int)):
                    raise ValueError(f"Line {lineno}: shift operands must be integers")
                if b < 0:
                    raise ValueError(f"Line {lineno}: negative shift amount")
                stack.append(a >> b)   # arithmetic (sign-preserving) for Python ints
    if len(stack) != 1:
        raise ValueError(f"Line {lineno}: malformed expression")
    return stack[0]


# Map arithmetic op -> Axiom mnemonic (operate on r0 with r1)
AXIOM_OP = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV'}

# Float arithmetic op -> SSE-backed Axiom mnemonic (operands are IEEE bits)
AXIOM_FOP = {'+': 'FADD', '-': 'FSUB', '*': 'FMUL', '/': 'FDIV'}

# Bitwise shift op -> Axiom mnemonic (shift r0 by amount in r1)
AXIOM_SHIFT = {'<<': 'SHL', '>>': 'SHR'}

# Map relational/equality op -> Axiom SET mnemonic (after CMP r0, r1).
# These leave 1 or 0 in the destination register.
AXIOM_SET = {
    '<':  'SETLT',
    '>':  'SETGT',
    '<=': 'SETLE',
    '>=': 'SETGE',
    '==': 'SETEQ',
    '!=': 'SETNE',
    # logical ops: both operands are already 0/1; combine and re-normalize.
    # We lower these as: AND/OR/XOR the two values then SETNE against 0,
    # but for simplicity the assembler treats SETAND/SETOR/SETXOR as
    # "combine r0,r1 logically -> 0/1 in r0".
    '&&': 'SETAND',
    '||': 'SETOR',
    '^^': 'SETXOR',
}


def emit_runtime_expr(out, rpn, ty, consts, manifest, lineno):
    """
    Emit Axiom instructions to evaluate a runtime expression, result left in r0.

    Strategy: a stack machine over r0/r1. We keep a small spill stack in the
    data section is overkill for Tier-1, so we evaluate using the constraint
    that the assembler supports ADD/SUB/MUL/DIV r0, r1. For each binary op we
    need both operands in r0 (lhs) and r1 (rhs). We linearize the RPN so the
    left operand lands in r0 and the right in r1 by evaluating left fully,
    moving to r1 holding area via the variable's own slot when needed.

    For Tier-1 we support the common case: left-associative chains where each
    step is (accumulator op operand). Deeply nested right subtrees that need
    two live runtime temporaries are spilled to a scratch slot.
    """
    # We implement a simple stack VM that materializes intermediate results
    # into named scratch slots in the data section: __t0, __t1, ...
    scratch = []          # list of scratch slot names allocated
    vstack = []           # value stack: ('imm', n) | ('var', name) | ('tmp', slot)

    def load_into(reg, item):
        kind, v = item
        if kind == 'imm':
            enc = encode_scalar_for_type(v, ty)
            if ty in FLOAT_TYPES:
                hexd = TYPE_INFO[ty][1] * 2
                out.append(f"    LDR {reg}, #0x{enc:0{hexd}X}")
            else:
                out.append(f"    LDR {reg}, #{enc}")
        elif kind == 'var':
            out.append(f"    LDR {reg}, {v}")
        elif kind == 'tmp':
            out.append(f"    LDR {reg}, {v}")

    tmp_counter = [0]
    def new_tmp():
        slot = f"__t{tmp_counter[0]}"
        tmp_counter[0] += 1
        if slot not in scratch:
            scratch.append(slot)
        return slot

    # The final (top-level) operator's result is already in r0, so it does not
    # need to be spilled to a scratch slot and immediately reloaded. Only the
    # intermediate results of a multi-op expression are spilled (to preserve a
    # left operand while a later sub-result is computed). This keeps the emitted
    # Axiom lean: a simple 'a - 1' is LDR/LDR/SUB with no scratch round-trip.
    op_positions = [i for i, (k, _v) in enumerate(rpn) if k == 'op']
    last_op = op_positions[-1] if op_positions else -1

    for idx, (kind, val) in enumerate(rpn):
        if kind in ('num', 'char', 'ident'):
            vstack.append(expr_operand_value(kind, val, consts, manifest, lineno))
        elif kind == 'op':
            rhs = vstack.pop()
            lhs = vstack.pop()
            # load lhs->r0, rhs->r1
            load_into('r0', lhs)
            load_into('r1', rhs)
            if val in ARITH_OPS:
                if ty in FLOAT_TYPES:
                    out.append(f"    {AXIOM_FOP[val]} r0, r1")
                else:
                    out.append(f"    {AXIOM_OP[val]} r0, r1")
            elif val in SHIFT_OPS:
                # shift r0 by the amount in r1
                out.append(f"    {AXIOM_SHIFT[val]} r0, r1")
            elif val in ('&&', '||', '^^'):
                # logical: operands are already 0/1, combine directly
                out.append(f"    {AXIOM_SET[val]} r0, r1")
            else:
                # relational/equality: CMP sets flags, SET* materializes 0/1
                out.append(f"    CMP r0, r1")
                out.append(f"    {AXIOM_SET[val]} r0")
            if idx == last_op:
                # result stays in r0 — no spill/reload
                vstack.append(('acc', None))
            else:
                slot = new_tmp()
                out.append(f"    STR r0, {slot}")
                vstack.append(('tmp', slot))

    # final result is on top of vstack; ensure it's in r0
    if len(vstack) != 1:
        raise ValueError(f"Line {lineno}: malformed expression")
    final = vstack[0]
    if final[0] == 'acc':
        pass                       # already in r0 (the top-level op left it there)
    elif final[0] == 'tmp':
        out.append(f"    LDR r0, {final[1]}")
    else:
        # a bare operand (no ops) — just load it
        load_into('r0', final)

    return scratch


def encode_scalar_for_type(value, ty):
    """Encode a Python number as the integer bit pattern for the given type."""
    if ty in FLOAT_TYPES:
        return FLOAT_ENCODERS[ty](float(value))
    # integer / char / bool
    v = int(value)
    if ty in INT_TYPES:
        lo, hi = INT_BOUNDS[ty]
        # wrap silently here; bounds already checked for literals upstream
        if v < 0:
            v = v + (1 << (TYPE_INFO[ty][1] * 8))
    return v


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------

def encode_value(raw, ty, consts=None, lineno=0):
    """
    Encode a single literal value (or const reference) for the given type.
    If raw is a bare identifier, it must resolve to a defined const.
    """
    raw = raw.strip()

    # null keyword — formats the value space to all-bits-zero for any type.
    # For ints this is 0, for chars NUL, for floats the 0x0 bit pattern (0.0).
    # For bool this is false.
    if raw == "null":
        return 0

    # ptr type — a raw ptr literal can only be 'null'; a real address is taken
    # with the '&' reference operator (handled as its own statement form).
    if ty in PTR_TYPES:
        raise ValueError(
            f"Line {lineno}: a ptr can only be initialized to 'null' or to a "
            f"reference '&name' (got {raw!r})")

    # bool type — strictly true/false literals (stored as 1/0 underneath)
    if ty in BOOL_TYPES:
        if raw == "true":
            return 1
        if raw == "false":
            return 0
        # A const reference to another bool is allowed
        if IDENT_RE.match(raw) and consts is not None and consts.is_const(raw):
            const_entry = consts.consts[raw]
            if const_entry['ty'] not in BOOL_TYPES:
                raise ValueError(
                    f"Line {lineno}: const '{raw}' is {const_entry['ty']}, "
                    f"cannot assign to a bool"
                )
            return const_entry['int_value']
        raise ValueError(
            f"Line {lineno}: invalid bool value {raw!r} — "
            f"bool must be 'true' or 'false' (not a number)"
        )

    # Const reference?  (bare identifier, not a number/char/string)
    if IDENT_RE.match(raw) and not raw.lstrip('-').isdigit():
        if consts is None or not consts.is_const(raw):
            raise ValueError(
                f"Line {lineno}: '{raw}' is not a defined const — "
                f"a const must be defined before it is used"
            )
        # Use the const's underlying raw literal, re-encoded for THIS type
        const_raw = consts.raw_of(raw)
        return encode_value(const_raw, ty, consts, lineno)

    if ty in INT_TYPES:
        try:
            v = int(raw, 0) if raw.lower().startswith(("0x", "-0x", "0b", "0o")) else int(raw)
        except ValueError:
            raise ValueError(f"Line {lineno}: invalid integer literal for {ty}: {raw!r}")
        lo, hi = INT_BOUNDS[ty]
        if not (lo <= v <= hi):
            raise ValueError(
                f"Line {lineno}: value {v} out of range for {ty} (must be {lo} to {hi})"
            )
        if v < 0:
            v = v + (1 << (TYPE_INFO[ty][1] * 8))
        return v

    elif ty in FLOAT_TYPES:
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"Line {lineno}: invalid float literal for {ty}: {raw!r}")
        return FLOAT_ENCODERS[ty](v)

    elif ty in CHAR_TYPES:
        m = re.match(r"^'(.)'$", raw)
        if not m:
            raise ValueError(
                f"Line {lineno}: invalid char literal for {ty}: {raw!r} — "
                f"must be a single character in single quotes e.g. 'A'"
            )
        code = ord(m.group(1))
        if code > 0x7F:
            raise ValueError(
                f"Line {lineno}: invalid char literal for {ty}: {raw!r} — "
                f"only ASCII (0x00-0x7F) supported at this stage"
            )
        return code

    raise ValueError(f"Line {lineno}: unknown type: {ty!r}")


def resolve_size(raw_size, consts, lineno):
    """Resolve an array size that may be a number or a const name."""
    raw_size = raw_size.strip()
    if raw_size.isdigit():
        return int(raw_size)
    # const reference
    if not consts.is_const(raw_size):
        raise ValueError(
            f"Line {lineno}: array size '{raw_size}' is not a defined const — "
            f"a const must be defined before it is used"
        )
    entry = consts.consts[raw_size]
    if entry['ty'] not in INT_TYPES:
        raise ValueError(
            f"Line {lineno}: array size const '{raw_size}' must be an integer type, "
            f"not {entry['ty']}"
        )
    size = entry['int_value']
    if size <= 0:
        raise ValueError(
            f"Line {lineno}: array size const '{raw_size}' = {size} must be positive"
        )
    return size


# ---------------------------------------------------------------------------
# Array element parsing
# ---------------------------------------------------------------------------

def parse_array_elements(raw_list, ty, max_size, consts, lineno):
    raw_list = raw_list.strip()

    if raw_list.startswith('"'):
        if ty not in CHAR_TYPES:
            raise ValueError(
                f"Line {lineno}: string literal initializer only valid for char types, not {ty}"
            )
        s = raw_list[1:-1]
        if len(s) > max_size:
            raise ValueError(
                f"Line {lineno}: string \"{s}\" has {len(s)} characters but "
                f"{ty}[{max_size}] only allows {max_size}"
            )
        encoded = []
        for ch in s:
            code = ord(ch)
            if code > 0x7F:
                raise ValueError(
                    f"Line {lineno}: non-ASCII character {ch!r} in string literal"
                )
            encoded.append(code)
        encoded += [0] * (max_size - len(encoded))
        return encoded

    if not (raw_list.startswith('[') and raw_list.endswith(']')):
        raise ValueError(f"Line {lineno}: expected array initializer [...], got: {raw_list!r}")

    inner = raw_list[1:-1].strip()
    elements = [e.strip() for e in inner.split(',')] if inner else []

    if len(elements) > max_size:
        raise ValueError(
            f"Line {lineno}: array initializer has {len(elements)} elements but "
            f"[{max_size}] only allows {max_size}"
        )

    encoded = [encode_value(e, ty, consts, lineno) for e in elements]
    encoded += [0] * (max_size - len(encoded))
    return encoded


# ---------------------------------------------------------------------------
# Axiom emission helpers
# ---------------------------------------------------------------------------

def emit_ldr_str(out, encoded, ty, slot):
    byte_width = TYPE_INFO[ty][1]
    hex_digits = byte_width * 2
    if ty in FLOAT_TYPES:
        out.append(f"    LDR r0, #0x{encoded:0{hex_digits}X}")
    else:
        out.append(f"    LDR r0, #{encoded}")
    out.append(f"    STR r0, {slot}")


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def match_arm_const_int(op, val, consts):
    """If a match arm is `== <integer-valued constant>`, return that integer;
    otherwise return None. Only equality arms against an int/char/bool literal
    or an int/char/bool const are eligible for jump-table lowering — comparison
    arms (>, >=, <, <=, !=) and non-constant values fall back to the chain."""
    if op != '==':
        return None
    val = val.strip()
    if re.fullmatch(r"-?[0-9]+", val):
        return int(val)
    m = re.fullmatch(r"'(.)'", val)
    if m:
        return ord(m.group(1))
    if IDENT_RE.match(val) and consts.is_const(val):
        entry = consts.consts[val]
        if entry['ty'] in INT_TYPES or entry['ty'] in CHAR_TYPES or entry['ty'] in BOOL_TYPES:
            return entry['int_value']
    return None


def compile_t1_to_axiom(src):
    # --- volatile qualifier pre-pass ---------------------------------------
    # `let volatile [mut] name: ty = ...;` marks a slot as volatile: the
    # optimizers must never cache it in a register, never drop a store to it,
    # and never hoist it across a loop. This matters for firmware, where a
    # variable shared with an interrupt handler (or a hardware register) can
    # change between two ordinary statements. We record the name and strip the
    # `volatile ` token so every existing declaration regex matches unchanged;
    # volatility is then carried by name through a '.volatile' sidecar.
    volatile_names = set()
    _vol_decl = re.compile(r"^(\s*let\s+)volatile\s+((?:mut\s+)?)([A-Za-z_]\w*)")
    _vol_lines = []
    for _vl in src.splitlines():
        m = _vol_decl.match(_vl)
        if m:
            volatile_names.add(m.group(3))
            _vl = _vol_decl.sub(r"\1\2\3", _vl, count=1)
        _vol_lines.append(_vl)
    src = "\n".join(_vol_lines)

    consts = ConstTable()
    manifest = {}        # local variable manifest
    # Manual relocations from '&a = <addr-expr>;'. Recorded in source order and
    # resolved at layout time (after all variables are known) so the manifest's
    # byte-accurate memory map can reject any overlap.
    relocations = []     # list of (name, rhs_expr, lineno)
    relocated_names = set()
    # Memory-mapped I/O: '&name = <absolute address>;' pins a variable to a fixed
    # data-space byte address (a hardware register, e.g. AVR PORTB at 0x25), with
    # no storage of its own. Reads/writes of the variable then ARE reads/writes of
    # that register. The address is absolute (no '&var'), so it can sit anywhere,
    # including below the normal data segment where the I/O registers live.
    mmio = {}            # name -> absolute data-space address (int)
    declarations = []    # drives code + data emission (locals + reassigns)
    global_consts = []   # consts declared anywhere, emitted to data section

    # --- functions (stack-linked) -----------------------------------------
    # current_fn is None at top level, 'main' inside main, or a function name.
    # Each non-main function buffers its body into fn_bodies[name]; main's body
    # is the top-level `declarations` list. ALL_TYPES set is reused for params.
    ALL_TY = {"i8","i16","i32","i64","f8","f16","f32","f64",
              "c8","c16","c32","c64","bool"}

    def parse_params(params_s, lineno):
        """Return (regular_params, self_recv).

        self_recv is None, or one of:
            ('single', 'Device')          — self IS that struct
            ('multi', ('Device','Engine'))— self is a namespace over those roots
            ('universal',)                — self spans all global structs + fns
        A `self` receiver is implicit (never a positional argument)."""
        params_s = params_s.strip()
        if params_s == "":
            return [], None

        # split on top-level commas only (so 'self: (A, B)' stays intact)
        parts, depth, cur = [], 0, ""
        for chx in params_s:
            if chx == '(':
                depth += 1; cur += chx
            elif chx == ')':
                depth -= 1; cur += chx
            elif chx == ',' and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += chx
        parts.append(cur)

        out_params = []
        self_recv = None
        for part in parts:
            part = part.strip()
            if part == "":
                continue
            recv = None
            if part == "self":
                recv = ('universal',)
            else:
                ms = SELF_SINGLE_RE.match(part)
                mm = SELF_MULTI_RE.match(part)
                if mm:
                    roots = tuple(x.strip() for x in mm.group(1).split(","))
                    if len(set(roots)) != len(roots):
                        raise ValueError(
                            f"Line {lineno}: duplicate struct in multi-root self")
                    recv = ('multi', roots)
                elif ms:
                    recv = ('single', ms.group(1))
            if recv is not None:
                if self_recv is not None:
                    raise ValueError(
                        f"Line {lineno}: a function may declare 'self' only once")
                self_recv = recv
                continue
            pm = re.match(r"^([a-zA-Z_]\w*)\s*:\s*(" + TYPE_PAT + r")$", part)
            if not pm:
                raise ValueError(
                    f"Line {lineno}: malformed parameter '{part}' (expected "
                    f"'name: type', 'self: Struct', 'self: (A, B)', or 'self')")
            if pm.group(1) == "self":
                raise ValueError(
                    f"Line {lineno}: 'self' is a receiver, not a typed value "
                    f"parameter — use 'self', 'self: Struct', or 'self: (A, B)'")
            out_params.append((pm.group(1), pm.group(2)))
        return out_params, self_recv

    # Pre-scan for function signatures so calls may precede definitions.
    functions = {}   # name -> {'params': [(n,ty)], 'ret': ty|'void', 'kind': 'fn'|'def'}
    for _ln, _raw in enumerate(src.splitlines(), 1):
        _L = _raw.strip()
        _is_fn  = _L.startswith("fn ")  or _L.startswith("fn\t")
        _is_def = _L.startswith("def ") or _L.startswith("def\t")
        if _is_fn and not FN_MAIN_RE.match(_L):
            _m, _kind = FN_DEF_RE.match(_L), 'fn'
        elif _is_def:
            _m, _kind = DEF_DEF_RE.match(_L), 'def'
        elif _L.startswith("isr ") or _L.startswith("isr\t"):
            # interrupt handler: register a void, no-param pseudo-function named
            # 'isr_<VECTOR>' so it gets a body buffer and a place in fn_order.
            _mi = ISR_HDR_RE.match(_L)
            if not _mi:
                raise ValueError(
                    f"Line {_ln}: malformed interrupt header — expected "
                    f"'isr <VECTOR> {{'")
            _vec = _mi.group(1)
            _iname = "isr_" + _vec
            if _iname in functions:
                raise ValueError(
                    f"Line {_ln}: interrupt vector '{_vec}' already has a handler")
            functions[_iname] = {'params': [], 'ret': 'void',
                                 'self_recv': None, 'kind': 'isr',
                                 'vector': _vec}
            continue
        else:
            _m = None
        if _m:
            _name = _m.group(1)
            if _name in functions:
                raise ValueError(f"Line {_ln}: function '{_name}' already defined")
            if _name in RESERVED:
                raise ValueError(f"Line {_ln}: '{_name}' is a reserved word")
            _params, _self = parse_params(_m.group(2), _ln)
            functions[_name] = {'params': _params,
                                'ret': _m.group(3),
                                'self_recv': _self,
                                'kind': _kind}
    fn_bodies = {name: [] for name in functions}   # non-main bodies
    fn_order  = list(functions.keys())
    current_fn = [None]
    current_self = [None]   # struct name bound to 'self' inside the current fn
    in_main = False
    block_stack = []     # control-flow frames
    label_counter = [0]

    def new_label(prefix):
        label_counter[0] += 1
        return f"{prefix}_{label_counter[0]}"

    def innermost_loop(lineno):
        """Find the nearest enclosing loop frame for break/continue. Searches
        past any intervening if/match/arm frames to the closest loop."""
        for fr in reversed(block_stack):
            if fr['kind'] == 'loop':
                return fr
        raise ValueError(
            f"Line {lineno}: 'break'/'continue' is only valid inside a while/loop")

    # Declarations normally flow into the top-level `declarations` list, but a
    # match buffers each arm's body into its own list so the whole construct
    # can be analyzed (table vs. chain) before anything is emitted. `emit`
    # always appends to the currently-active sink; `decl_sink_stack` is pushed
    # when an arm/else body opens and popped when it closes.
    decl_sink_stack = [declarations]
    def emit(d):
        decl_sink_stack[-1].append(d)

    # Each entry: (table_label, [code-label per dense value]). Emitted to the
    # data section as a `.qaddrs` array the backend fills with real addresses.
    jump_tables = []

    # ---- print() state ----------------------------------------------------
    # String literal pool: list of (label, [byte ints]) emitted as `.bytes`.
    string_pool = []
    string_uid  = [0]
    print_uid   = [0]              # unique-id source for inline itoa labels
    needs_print_scratch = [False]  # emit the itoa workspace + 1-byte buffer?
    needs_itoa = [False]           # emit the shared fn___itoa routine?
    needs_float_io = [False]       # emit float/atoi/atof/dtoa workspace?
    needs_callret = [False]        # emit the call-result stash slot (__callret)?
    bool_str = [None]              # cached ("true","false") string labels

    def new_string_label(byte_list):
        string_uid[0] += 1
        lbl = f"__str_{string_uid[0]}"
        string_pool.append((lbl, byte_list))
        return lbl

    def bool_labels():
        """Return the shared (true_label, false_label) string-pool labels used
        to print a runtime bool, creating them once on first use."""
        if bool_str[0] is None:
            t = new_string_label([116, 114, 117, 101])         # "true"
            f = new_string_label([102, 97, 108, 115, 101])     # "false"
            bool_str[0] = (t, f)
        return bool_str[0]

    # ---- struct state -----------------------------------------------------
    # struct_path: names of the structs currently being defined (nesting).
    # struct_fields: mangled-label -> {'ty': type} for every declared field.
    # struct_field_order: emission order for the data section.
    struct_path        = []
    struct_fields      = {}
    struct_field_order = []
    declared_structs   = set()       # mangled struct paths, for duplicate checks
    top_level_structs  = set()       # global (root) struct names, for universal self
    self_aliases       = {}          # local alias name -> 'universal' | ('typed', ty)
    struct_instances   = {}          # local instance name -> struct type
    instance_field_ty  = {}          # instance field slot -> type

    def _split_access(full):
        """('self::Sensor.temp') -> ('self', ['Sensor','temp']) or None."""
        m = GEN_ACCESS_RE.match(full)
        if not m:
            return None
        rootm = re.match(r"^([A-Za-z_]\w*)", full)
        root = rootm.group(1)
        segs = re.findall(r"(?:::|\.)([A-Za-z_]\w*)", full[len(root):])
        return root, segs

    def resolve_access(full, lineno):
        """Resolve any self / alias / instance / struct access to (label, ty).
        Returns None if 'full' is not actually a navigated access we own."""
        parts = _split_access(full)
        if parts is None:
            return None
        root, segs = parts
        filt = None
        # ---- enforce the navigation rule -----------------------------------
        # '::' steps into a struct / nested struct / function (a namespace);
        # '.' may appear only once, as the final field access. This only
        # applies to accesses we actually own (self / alias / instance / struct).
        recognized = (root == "self" or root in self_aliases
                      or root in struct_instances or root in top_level_structs)
        if not recognized:
            return None
        seps = re.findall(r"(::|\.)[A-Za-z_]\w*", full[len(root):])
        if seps:
            if any(s == "." for s in seps[:-1]):
                raise ValueError(
                    f"Line {lineno}: in '{full}', '::' must be used to navigate into "
                    f"a struct/namespace — '.' only accesses the final field. "
                    f"Replace the inner '.' with '::'")
            if seps[-1] != ".":
                raise ValueError(
                    f"Line {lineno}: '{full}' must end by accessing a field with '.', "
                    f"e.g. {root}::...::field")
        if root == "self":
            recv = current_self[0] or ('universal',)
            if recv[0] == 'single':
                path = [recv[1]] + segs
            elif recv[0] == 'multi':
                roots = recv[1]
                if not segs:
                    raise ValueError(f"Line {lineno}: 'self' needs a member, e.g. self::Sensor.temp")
                # (a) explicit root-qualified:  self::Device::Sensor.temp
                if segs[0] in roots:
                    path = segs
                else:
                    # (b) search the roots for the named member (nested namespace
                    #     via '::' or field via '.') — no need to name the root.
                    hits = [R for R in roots
                            if "__".join([R] + segs) in struct_fields]
                    if len(hits) == 1:
                        path = [hits[0]] + segs
                    elif not hits:
                        raise ValueError(
                            f"Line {lineno}: '{full}' is not a member of this "
                            f"function's self roots {list(roots)}")
                    else:
                        raise ValueError(
                            f"Line {lineno}: '{full}' is ambiguous across self roots "
                            f"{hits} — qualify it, e.g. self::{hits[0]}::...")
            else:  # universal — navigate from the global namespace (name the struct)
                if not segs:
                    raise ValueError(f"Line {lineno}: 'self' needs a struct, e.g. self::Struct.field")
                r0 = segs[0]
                if r0 not in top_level_structs:
                    raise ValueError(f"Line {lineno}: '{r0}' is not a global struct")
                path = segs
        elif root in self_aliases:
            mode = self_aliases[root]
            if not segs:
                raise ValueError(f"Line {lineno}: '{root}' needs a struct, e.g. {root}::Struct.field")
            if segs[0] not in top_level_structs:
                raise ValueError(f"Line {lineno}: '{segs[0]}' is not a global struct")
            path = segs
            if isinstance(mode, tuple):
                filt = mode[1]
        elif root in struct_instances:
            path = [root] + segs
        elif root in top_level_structs or "__".join([root] + segs) in struct_fields:
            path = [root] + segs
        else:
            return None
        label = "__".join(path)
        if root in struct_instances:
            if label not in instance_field_ty:
                raise ValueError(f"Line {lineno}: '{full}' is not a field of instance '{root}'")
            fty = instance_field_ty[label]
        else:
            if label not in struct_fields:
                raise ValueError(f"Line {lineno}: unknown struct field '{full}'")
            fty = struct_fields[label]['ty']
        if filt is not None and fty != filt:
            raise ValueError(
                f"Line {lineno}: type-filtered alias '{root}: {filt}' cannot reach "
                f"the {fty} field '{full}'")
        return label, fty

    def mangle_access(access, lineno=0):
        r = resolve_access(access, lineno)
        if r is not None:
            return r[0]
        # plain nested-struct path with no self/alias/instance root
        return access.replace("::", "__").replace(".", "__")

    def lower_expr(expr, lineno):
        """Rewrite every field/self/alias/instance access in an expression
        string to its mangled slot label, so the expression engine (which
        treats the label as a runtime variable) can evaluate it."""
        def repl(m):
            full = m.group(0)
            r = resolve_access(full, lineno)
            return r[0] if r is not None else full
        return GEN_ACCESS_FIND_RE.sub(repl, expr)

    def _classify_index(arr, raw_index, lineno):
        """Resolve an array subscript to ('static', int) or ('dyn', var_name).

        A digit or const folds to a compile-time slot index (bounds-checked).
        A declared integer scalar becomes a runtime index, which requires the
        array's elements to be 8 bytes wide (i64/ptr) so the LEA+LDRQ/STORQ
        lowering — mem[base + index*8] — addresses the right element."""
        aentry = manifest[arr]
        if aentry['ty'] not in ("i64", "ptr"):
            raise ValueError(
                f"Line {lineno}: reading '{arr}[{raw_index}]' requires an 8-byte "
                f"element type (i64 or ptr); '{arr}' is {aentry['ty']}[]")
        if raw_index.isdigit() or consts.is_const(raw_index):
            idx = int(raw_index) if raw_index.isdigit() else resolve_size(raw_index, consts, lineno)
            if idx >= aentry['max_size'] or idx < 0:
                raise ValueError(
                    f"Line {lineno}: index {idx} out of bounds for "
                    f"'{arr}[{aentry['max_size']}]'")
            return ('static', idx)
        if raw_index in manifest and not manifest[raw_index]['is_array']:
            if manifest[raw_index]['ty'] not in INT_TYPES:
                raise ValueError(
                    f"Line {lineno}: array index '{raw_index}' must be an integer, "
                    f"not {manifest[raw_index]['ty']}")
            return ('dyn', raw_index)
        raise ValueError(
            f"Line {lineno}: array index '{raw_index}' is not a digit, const, "
            f"or declared integer variable")

    def access_uses_self(access):
        return (access == "self" or access.startswith("self.")
                or access.startswith("self::"))

    def check_self_context(access, lineno):
        # self is now valid in any scope (universal by default); nothing to do.
        return

    def resolve_source(rhs, ty, lineno):
        """Turn a field-write RHS into a load descriptor:
        ('imm', int)  — a literal / const folded to an integer
        ('slot', lbl) — a variable or another field, loaded as a value."""
        rhs = rhs.strip()
        if FIELD_ACCESS_RE.match(rhs):
            check_self_context(rhs, lineno)
            lbl = mangle_access(rhs)
            if lbl not in struct_fields:
                raise ValueError(f"Line {lineno}: unknown struct field '{rhs}'")
            return ('slot', lbl)
        if consts.is_const(rhs):
            return ('imm', consts.consts[rhs]['int_value'])
        if rhs in manifest and not manifest[rhs]['is_array']:
            return ('slot', rhs)
        # otherwise treat as a literal (number / 'c' / true / false / null)
        try:
            return ('imm', encode_value(rhs, ty, consts, lineno))
        except Exception:
            raise ValueError(
                f"Line {lineno}: cannot assign '{rhs}' to a struct field "
                f"(expected a literal, const, variable, or field)")

    ESCAPES = {'n': 10, 't': 9, 'r': 13, '\\': 92,
               '{': 123, '}': 125, '"': 34, '0': 0}

    def parse_format(fmt, lineno):
        """Split a print() format string into ordered segments:
        ('print_lit', label, length) for literal runs, and
        ('print_int'|'print_char', varname) for runtime interpolations.
        Const interpolations are folded into the literal bytes at compile time.
        """
        segs = []
        buf  = bytearray()
        def flush():
            if buf:
                segs.append(('print_lit', new_string_label(list(buf)), len(buf)))
                buf.clear()
        i = 0
        while i < len(fmt):
            c = fmt[i]
            if c == '\\':
                if i + 1 >= len(fmt):
                    raise ValueError(f"Line {lineno}: dangling '\\' in print string")
                e = fmt[i + 1]
                if e not in ESCAPES:
                    raise ValueError(f"Line {lineno}: unknown escape '\\{e}'")
                buf.append(ESCAPES[e]); i += 2; continue
            if c == '{':
                j = fmt.find('}', i + 1)
                if j == -1:
                    raise ValueError(f"Line {lineno}: unmatched '{{' in print string")
                name = fmt[i + 1:j].strip()
                i = j + 1
                # struct/self/alias/instance field interpolation
                _acc = resolve_access(name, lineno) if GEN_ACCESS_RE.match(name) else None
                if _acc is not None:
                    lbl, fty = _acc
                    if fty in INT_TYPES:
                        flush(); segs.append(('print_int', lbl)); needs_print_scratch[0] = True; needs_itoa[0] = True
                    elif fty in CHAR_TYPES:
                        flush(); segs.append(('print_char', lbl)); needs_print_scratch[0] = True
                    elif fty in FLOAT_TYPES:
                        flush(); segs.append(('print_float', lbl))
                        needs_print_scratch[0] = True; needs_float_io[0] = True
                    elif fty in BOOL_TYPES:
                        flush()
                        _t, _f = bool_labels()
                        segs.append(('print_bool', lbl, _t, _f))
                    else:
                        raise ValueError(f"Line {lineno}: cannot interpolate {fty} field "
                                         f"'{name}' (only int/char/float/bool fields)")
                    continue
                if consts.is_const(name):
                    entry = consts.consts[name]
                    ty, iv = entry['ty'], entry['int_value']
                    if ty in CHAR_TYPES:
                        buf.append(iv & 0xFF)
                    elif ty in BOOL_TYPES:
                        buf.extend(b"true" if iv else b"false")
                    elif ty in INT_TYPES:
                        buf.extend(str(iv).encode())
                    elif ty in FLOAT_TYPES:
                        # the const's data slot holds the IEEE bits at runtime;
                        # format it with the same dtoa routine as float vars
                        flush(); segs.append(('print_float', name))
                        needs_print_scratch[0] = True; needs_float_io[0] = True
                    else:
                        raise ValueError(f"Line {lineno}: cannot interpolate {ty} "
                                         f"const '{name}'")
                    continue
                if name in manifest:
                    if manifest[name]['is_array']:
                        raise ValueError(f"Line {lineno}: cannot interpolate whole array '{name}'")
                    ty = manifest[name]['ty']
                    if ty in INT_TYPES:
                        flush(); segs.append(('print_int', name)); needs_print_scratch[0] = True; needs_itoa[0] = True
                    elif ty in CHAR_TYPES:
                        flush(); segs.append(('print_char', name)); needs_print_scratch[0] = True
                    elif ty in BOOL_TYPES:
                        flush()
                        _t, _f = bool_labels()
                        segs.append(('print_bool', name, _t, _f))
                    elif ty in FLOAT_TYPES:
                        flush(); segs.append(('print_float', name))
                        needs_print_scratch[0] = True; needs_float_io[0] = True
                    else:
                        raise ValueError(f"Line {lineno}: cannot interpolate {ty} "
                                         f"variable '{name}'")
                    continue
                raise ValueError(f"Line {lineno}: unknown name '{name}' in print interpolation")
            if c == '}':
                raise ValueError(f"Line {lineno}: unmatched '}}' (use \\}} for a literal brace)")
            buf.append(ord(c) & 0xFF)
            i += 1
        flush()
        return segs

    # A match becomes a jump table only when it has at least this many arms,
    # all equality-against-constant, packed densely enough to be worthwhile.
    MIN_TABLE_ARMS = 3

    def lower_match(frame):
        """Emit a finished match: a jump table when every arm is
        `== <non-negative int/char const>` and the values are dense, otherwise
        the original linear comparison chain."""
        var, ty, end = frame['var'], frame['ty'], frame['end']
        arms, else_body = frame['arms'], frame['else_body']

        eligible = (
            len(arms) >= MIN_TABLE_ARMS and
            (ty in INT_TYPES or ty in CHAR_TYPES or ty in BOOL_TYPES) and
            all(a['const_int'] is not None and a['const_int'] >= 0 for a in arms)
        )
        if eligible:
            values = [a['const_int'] for a in arms]
            lo, hi = min(values), max(values)
            # Density guard: don't build a giant sparse table.
            if (hi - lo + 1) > max(64, 4 * len(arms)):
                eligible = False

        if eligible:
            table_label   = new_label("match_table")
            default_label = new_label("match_else") if else_body is not None else end
            for a in arms:
                a['label'] = new_label("arm_body")

            # O(1) dispatch: bias by lo, bounds-check, indexed indirect jump.
            emit(('comment', f"match({var}) -> jump table over dense values [{lo}..{hi}]"))
            emit(('jumptable_dispatch', var, lo, hi - lo, default_label, table_label))

            # Arm bodies (each falls out to the shared end).
            for a in arms:
                emit(('label', a['label']))
                for d in a['body']:
                    emit(d)
                emit(('jmp', end))
            if else_body is not None:
                emit(('label', default_label))
                for d in else_body:
                    emit(d)
            emit(('label', end))

            # Build the dense table (first arm wins on duplicate values).
            value_to_label = {}
            for a in arms:
                value_to_label.setdefault(a['const_int'], a['label'])
            entries = [value_to_label.get(v, default_label) for v in range(lo, hi + 1)]
            jump_tables.append((table_label, entries))
        else:
            # Linear comparison chain — identical lowering to the original.
            pending_next = None
            for a in arms:
                if pending_next is not None:
                    emit(('jmp', end))
                    emit(('label', pending_next))
                cond = f"{var} {a['op']} {a['val']}"
                rpn = expr_to_rpn(expr_tokenize(cond, frame['lineno']), frame['lineno'])
                emit(('cond_expr', rpn, frame['lineno']))
                arm_next = new_label("arm_next")
                emit(('jz', arm_next))
                for d in a['body']:
                    emit(d)
                pending_next = arm_next
            if else_body is not None:
                if pending_next is not None:
                    emit(('jmp', end))
                    emit(('label', pending_next))
                    pending_next = None
                for d in else_body:
                    emit(d)
            if pending_next is not None:
                emit(('jmp', end))
                emit(('label', pending_next))
            emit(('label', end))

    def handle_simple_statement(line, lineno):
        """Parse one non-control-flow statement (step op / reassignment /
        declaration) and emit it. Reused both for the main parse loop and
        for the init and step clauses of a for-loop."""

        def rhs_has_access(rhs):
            """True if RHS contains a struct/self/alias/instance field access we
            own (so it must be lowered before the expression engine runs).
            resolve_access raises on a navigation-rule violation, which we let
            propagate so bad paths still error cleanly."""
            for mm in GEN_ACCESS_FIND_RE.finditer(rhs):
                if resolve_access(mm.group(0), lineno) is not None:
                    return True
            return False

        def check_expr_target_ty(ty, name, rpn):
            """Validate that an expression's result type is compatible with the
            target's declared type (shared by the scalar-LHS expression forms)."""
            yields_bool = expr_yields_bool(rpn)
            if expr_has_shift(rpn) and ty not in INT_TYPES:
                raise ValueError(
                    f"Line {lineno}: shift operations are integer-only, "
                    f"cannot store result in {ty} '{name}'")
            if ty in BOOL_TYPES:
                if not yields_bool:
                    raise ValueError(
                        f"Line {lineno}: bool '{name}' needs a comparison/logical "
                        f"expression, not arithmetic")
            elif ty in CHAR_TYPES:
                raise ValueError(f"Line {lineno}: expressions are not allowed for {ty}")
            else:
                if yields_bool:
                    raise ValueError(
                        f"Line {lineno}: comparison result is a bool and cannot "
                        f"be stored in {ty} '{name}'")

        # --- return [expr]; ------------------------------------------------
        mret = RETURN_RE.match(line)
        if mret:
            expr = mret.group(1).strip()
            fn = current_fn[0]
            if fn is None:
                raise ValueError(f"Line {lineno}: 'return' outside of a function")
            if fn == 'main':
                if expr:
                    raise ValueError(
                        f"Line {lineno}: main has no return value — use 'return;' to exit early")
                emit(('return_void', 'main'))
                return True
            ret = functions[fn]['ret']
            if ret == 'void':
                if expr:
                    raise ValueError(
                        f"Line {lineno}: '{fn}' returns void; 'return;' takes no value")
                emit(('return_void', fn))
            else:
                if not expr:
                    raise ValueError(
                        f"Line {lineno}: '{fn}' must return a {ret} value")
                # lower any field/self/alias/instance accesses to slot labels
                # so 'return Account.balance;' / 'return A.x + B.y;' work
                rpn = expr_to_rpn(expr_tokenize(lower_expr(expr, lineno), lineno), lineno)
                emit(('return_expr', fn, ret, rpn))
            return True

        # --- '&' relocation:  &a = <addr-expr>;  (byte-accurate placement) --
        # Relocates 'a' to a computed byte address. The address expression mixes
        # integer arithmetic with '&var' (a variable's address), e.g.
        #   &a = &c + 8;     &a = 16 + &base;
        # The actual address can only be resolved once every variable's layout
        # is known, so the statement is recorded here (with light validation)
        # and resolved at layout time, where the manifest's byte map rejects any
        # range that overlaps another variable.
        m = RELOC_RE.match(line)
        if m and "&" in m.group(2):
            a, rhs = m.group(1), m.group(2)
            if consts.is_const(a):
                raise ValueError(
                    f"Line {lineno}: cannot relocate a const ('&{a} = ...')")
            if a not in manifest:
                raise ValueError(f"Line {lineno}: '{a}' is not a declared variable")
            if manifest[a]['is_array']:
                raise ValueError(
                    f"Line {lineno}: relocation of whole arrays is not supported ('{a}')")
            if not manifest[a]['mutable']:
                raise ValueError(
                    f"Line {lineno}: '{a}' is immutable — only a 'let mut' variable "
                    f"may be relocated with '&{a} = ...'")
            if a in relocated_names:
                raise ValueError(f"Line {lineno}: '{a}' has already been relocated")
            refs = ADDR_REF_RE.findall(rhs)
            if not refs:
                raise ValueError(
                    f"Line {lineno}: relocation address must reference a variable "
                    f"address, e.g. &{a} = &other + 8;")
            for rname in refs:
                if rname not in manifest and not consts.is_const(rname):
                    raise ValueError(
                        f"Line {lineno}: '&{rname}' in the address expression is not "
                        f"a declared variable or const")
            relocated_names.add(a)
            manifest[a]['relocated'] = True
            relocations.append((a, rhs, lineno))
            emit(('comment', f"relocate '{a}' to {rhs}  (resolved at layout time)"))
            return True

        # --- &name = <absolute address>;  (memory-mapped I/O register) ---------
        # No '&var' in the RHS: the address is an absolute data-space byte address
        # (a hardware register). The variable gets NO storage of its own — every
        # read/write of it becomes a load/store at that fixed address. This is how
        # GPIO and other peripherals are driven in pure Nexa (e.g. on an AVR,
        # &portb = 0x25; then `portb = 16;` writes the PORTB register).
        if m:   # RELOC_RE matched but RHS has no '&'
            a, rhs = m.group(1), m.group(2)
            if consts.is_const(a):
                raise ValueError(
                    f"Line {lineno}: cannot map a const to an address ('&{a} = ...')")
            if a not in manifest:
                raise ValueError(f"Line {lineno}: '{a}' is not a declared variable")
            if manifest[a]['is_array']:
                raise ValueError(
                    f"Line {lineno}: whole-array memory mapping is not supported ('{a}')")
            if not manifest[a]['mutable']:
                raise ValueError(
                    f"Line {lineno}: '{a}' is immutable — only a 'let mut' variable "
                    f"may be mapped to an address with '&{a} = ...'")
            if a in relocated_names:
                raise ValueError(f"Line {lineno}: '{a}' has already been relocated/mapped")
            rpn = expr_to_rpn(expr_tokenize(rhs, lineno), lineno)
            if not expr_is_constant(rpn, consts, manifest):
                raise ValueError(
                    f"Line {lineno}: address for '&{a} = ...' must be a constant "
                    f"(an absolute byte address) or reference '&other'")
            addr = int(fold_constant_expr(rpn, consts, lineno))
            if addr < 0:
                raise ValueError(
                    f"Line {lineno}: memory-mapped address for '{a}' is negative ({addr})")
            mmio[a] = addr
            relocated_names.add(a)
            manifest[a]['relocated'] = True
            manifest[a]['mmio'] = True
            emit(('comment',
                  f"map '{a}' to absolute address 0x{addr:X}  (memory-mapped I/O)"))
            return True

        # --- '&' reference (RHS):  let r: ptr = &x;   r = &func; -------------
        # r receives the ADDRESS of a variable or function (a pure reference).
        def check_ref_target(target, lineno):
            if target in manifest:
                if manifest[target]['is_array']:
                    raise ValueError(
                        f"Line {lineno}: cannot take the address of a whole array "
                        f"'{target}'")
                return 'var'
            if target in functions:
                return 'fn'
            raise ValueError(
                f"Line {lineno}: '&{target}' — '{target}' is not a declared "
                f"variable or function")

        m = REF_DECL_RE.match(line)
        if m:
            is_mut, name, ty, target = (m.group(1) is not None, m.group(2),
                                        m.group(3), m.group(4))
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            kind = check_ref_target(target, lineno)
            manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
            emit(('ref', name, target, kind))
            return True

        m = REF_REASSIGN_RE.match(line)
        if m:
            name, target = m.group(1), m.group(2)
            if consts.is_const(name):
                raise ValueError(f"Line {lineno}: '{name}' is a constant and cannot be modified")
            if name not in manifest or manifest[name]['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is not a declared scalar")
            if not manifest[name]['mutable']:
                raise ValueError(f"Line {lineno}: '{name}' is immutable — declare with 'let mut'")
            if manifest[name]['ty'] not in ('i64', 'ptr'):
                raise ValueError(
                    f"Line {lineno}: a reference '&{target}' is an address; store it "
                    f"in an i64 or ptr, not {manifest[name]['ty']} '{name}'")
            kind = check_ref_target(target, lineno)
            emit(('ref', name, target, kind))
            return True



        # --- local self alias (only where the function has NO receiver self) -
        #   let foo: self = self;      universal alias
        #   let bar: i32  = self;      type-filtered alias
        def guard_self_alias(name, lineno):
            if current_self[0] is not None:
                raise ValueError(
                    f"Line {lineno}: this function takes 'self' as a receiver, so it "
                    f"cannot also create a local self alias ('{name}') in the same "
                    f"scope (memory-safety rule)")
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest or name in self_aliases \
                    or name in struct_instances:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")

        m = SELF_ALIAS_RE.match(line)
        if m:
            name = m.group(2)
            guard_self_alias(name, lineno)
            self_aliases[name] = 'universal'
            emit(('comment', f"local universal self alias '{name}' (compile-time)"))
            return True

        m = SELF_TYPED_ALIAS_RE.match(line)
        if m:
            name, ty = m.group(2), m.group(3)
            guard_self_alias(name, lineno)
            self_aliases[name] = ('typed', ty)
            emit(('comment',
                  f"local type-filtered self alias '{name}: {ty}' (compile-time)"))
            return True

        # --- local struct instance:  let mut local_net: Network = Network; ----
        m = STRUCT_INSTANCE_RE.match(line)
        if m and m.group(3) in top_level_structs and m.group(4) == m.group(3):
            is_mut, name, sty = m.group(1) is not None, m.group(2), m.group(3)
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest or name in self_aliases \
                    or name in struct_instances:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            struct_instances[name] = sty
            # give the instance its own copy of every field of the struct,
            # initialised from the global instance.
            prefix = sty + "__"
            for fslot, finfo in list(struct_fields.items()):
                if fslot == sty or fslot.startswith(prefix):
                    suffix = fslot[len(sty):]            # '__id', '__Sensor__temp'
                    islot = name + suffix
                    instance_field_ty[islot] = finfo['ty']
                    manifest[islot] = {'ty': finfo['ty'], 'mutable': True, 'is_array': False}
                    emit(('store_value', islot, ('slot', fslot)))
            return True

        # --- struct field access (statements containing '.' / '::') ---
        # let [mut] x: ty = <access>;   (read a field/instance into a new var)
        m = DECL_FROM_FIELD_RE.match(line)
        if m:
            is_mut = m.group(1) is not None
            name, ty, access = m.group(2), m.group(3), m.group(4)
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            src = mangle_access(access, lineno)
            manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
            emit(('field_decl', name, ty, src))
            return True

        # <access> = <expr>;  or  <access> += <expr>;  (write/compound a field
        # or instance field with an arbitrary expression RHS)
        m = FIELD_ASSIGN_RE.match(line)
        if m and resolve_access(m.group(1), lineno) is not None:
            access, op, rhs = m.group(1), m.group(2), m.group(3)
            dest, fty = resolve_access(access, lineno)
            if fty not in (INT_TYPES | CHAR_TYPES | FLOAT_TYPES | PTR_TYPES | BOOL_TYPES):
                raise ValueError(f"Line {lineno}: cannot assign to {fty} field '{access}'")
            rhs_l = lower_expr(rhs, lineno)
            if op != '=':
                rhs_l = f"{dest} {op[0]} ({rhs_l})"     # x += e  ->  x = x + (e)
            rpn = expr_to_rpn(expr_tokenize(rhs_l, lineno), lineno)
            emit(('field_expr_store', dest, rpn, fty))
            return True

        # x = <access>;   (read a field/instance field into an existing var)
        m = VAR_FROM_FIELD_RE.match(line)
        if m and resolve_access(m.group(2), lineno) is not None:
            name, access = m.group(1), m.group(2)
            src = mangle_access(access, lineno)
            if consts.is_const(name):
                raise ValueError(f"Line {lineno}: '{name}' is a constant and cannot be modified")
            if name not in manifest or manifest[name]['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is not a declared scalar")
            if not manifest[name]['mutable']:
                raise ValueError(f"Line {lineno}: '{name}' is immutable — declare with 'let mut'")
            emit(('store_value', name, ('slot', src)))
            return True

        # --- Builtin calls: input() / int(ch) / float(ch) / bool(ch) ---
        # Validate the call's argument and target type, returning a normalized
        # ('builtin'-style) payload (fn, arg, target_ty).
        def check_builtin(fn, arg, target_ty, lineno):
            if fn == "input":
                if arg:
                    raise ValueError(f"Line {lineno}: input() takes no argument")
                # input() is type-directed:
                #   c8   -> read a single character (skipping CR/LF)
                #   iNN  -> read a whole integer token (multi-digit atoi)
                #   f64  -> read a whole float token (atof)
                #   bool -> read a token, '0'/empty -> false, else true
                if target_ty in CHAR_TYPES:
                    return
                if target_ty in INT_TYPES or target_ty in BOOL_TYPES:
                    needs_float_io[0] = True
                    return
                if target_ty in FLOAT_TYPES:
                    if target_ty != "f64":
                        raise ValueError(
                            f"Line {lineno}: input() into a float currently "
                            f"produces an f64 (got {target_ty})")
                    needs_float_io[0] = True
                    return
                raise ValueError(
                    f"Line {lineno}: input() cannot store into {target_ty}")
            # int/float/bool all convert a single char argument
            if not arg:
                raise ValueError(f"Line {lineno}: {fn}() needs a char argument, e.g. {fn}(ch)")
            if consts.is_const(arg):
                aty = consts.consts[arg]['ty']
            elif arg in manifest and not manifest[arg]['is_array']:
                aty = manifest[arg]['ty']
            else:
                raise ValueError(f"Line {lineno}: {fn}() argument '{arg}' is not a declared char")
            if aty not in CHAR_TYPES:
                raise ValueError(
                    f"Line {lineno}: {fn}() converts a char, but '{arg}' is {aty}")
            want = {"int": INT_TYPES, "float": FLOAT_TYPES, "bool": BOOL_TYPES}[fn]
            if target_ty not in want:
                raise ValueError(
                    f"Line {lineno}: {fn}() result cannot be stored in {target_ty}")
            if fn == "float" and target_ty != "f64":
                raise ValueError(
                    f"Line {lineno}: float() currently produces an f64 (got {target_ty})")

        m = BUILTIN_DECL_RE.match(line)
        if m:
            is_mut = m.group(1) is not None
            name, ty, fn, arg = m.group(2), m.group(3), m.group(4), m.group(5)
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            check_builtin(fn, arg, ty, lineno)
            manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
            emit(('builtin', name, ty, fn, arg))
            return True

        m = BUILTIN_REASSIGN_RE.match(line)
        if m:
            name, fn, arg = m.group(1), m.group(2), m.group(3)
            if consts.is_const(name):
                raise ValueError(f"Line {lineno}: '{name}' is a constant and cannot be modified")
            if name not in manifest or manifest[name]['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is not a declared scalar")
            if not manifest[name]['mutable']:
                raise ValueError(f"Line {lineno}: '{name}' is immutable — declare with 'let mut'")
            check_builtin(fn, arg, manifest[name]['ty'], lineno)
            emit(('builtin_set', name, fn, arg))
            return True

        # --- function calls (whole-RHS / statement forms only) -------------
        #   let x: T = name(args);   x = name(args);   name(args);
        # Only fire when the callee is a known function (so these regexes don't
        # shadow other statements). Arguments are simple expressions; nested
        # calls inside an argument are not supported.
        def split_call_args(args_s, lineno):
            args_s = args_s.strip()
            if args_s == "":
                return []
            if "(" in args_s or ")" in args_s:
                raise ValueError(
                    f"Line {lineno}: nested calls inside arguments are not supported")
            return [a.strip() for a in args_s.split(",")]

        def emit_call(callee, args_s, dest, dest_ty, lineno):
            info = functions[callee]
            params = info['params']
            args = split_call_args(args_s, lineno)
            # A receiver-self function may be called as func(self, ...) or
            # func(alias, ...): a leading 'self'/self-alias argument is the
            # receiver (compile-time context), so drop it before arity-checking.
            if (info.get('self_recv') is not None and args
                    and (args[0] == 'self' or args[0] in self_aliases)):
                args = args[1:]
            if len(args) != len(params):
                extra = ""
                if info.get('self_recv') is not None:
                    extra = " ('self' is an implicit receiver)"
                raise ValueError(
                    f"Line {lineno}: '{callee}' expects {len(params)} argument(s), "
                    f"got {len(args)}{extra}")
            arg_specs = []
            for (pname, pty), a in zip(params, args):
                rpn = expr_to_rpn(expr_tokenize(lower_expr(a, lineno), lineno), lineno)
                arg_specs.append((pname, pty, rpn))
            ret = info['ret']
            if dest is not None:
                if ret == 'void':
                    raise ValueError(
                        f"Line {lineno}: '{callee}' returns void — its result cannot be stored")
                if dest_ty is not None and dest_ty != ret:
                    raise ValueError(
                        f"Line {lineno}: '{callee}' returns {ret}, cannot store into {dest_ty}")
            emit(('callseq', callee, arg_specs, dest, ret))

        # --- namespace-qualified call:  ns::callee(args) -> callee(args) -----
        # ns is 'self' or a local self alias; the receiver is supplied by the
        # namespace context, so it lowers to a direct call.
        for _sc_re, _has_decl, _has_dest in (
                (NS_CALL_DECL_RE, True, True),
                (NS_CALL_REASSIGN_RE, False, True),
                (NS_CALL_STMT_RE, False, False)):
            m = _sc_re.match(line)
            if not m:
                continue
            if _has_decl:
                is_mut, sb_name, sb_ty, ns, callee, args_s = (
                    m.group(1) is not None, m.group(2), m.group(3),
                    m.group(4), m.group(5), m.group(6))
            elif _has_dest:
                sb_name, ns, callee, args_s = (m.group(1), m.group(2),
                                               m.group(3), m.group(4))
                sb_ty = None
            else:
                sb_name = None; sb_ty = None
                ns, callee, args_s = m.group(1), m.group(2), m.group(3)
            # the namespace must be 'self' or a declared local self alias
            if ns != "self" and ns not in self_aliases:
                continue
            if callee not in functions:
                raise ValueError(
                    f"Line {lineno}: {ns}::{callee}() — no global function '{callee}'")
            if _has_decl:
                if sb_name in RESERVED:
                    raise ValueError(f"Line {lineno}: '{sb_name}' is a reserved word")
                if consts.is_const(sb_name) or sb_name in manifest:
                    raise ValueError(f"Line {lineno}: '{sb_name}' is already defined")
                manifest[sb_name] = {'ty': sb_ty, 'mutable': is_mut, 'is_array': False}
                emit_call(callee, args_s, sb_name, sb_ty, lineno)
            elif _has_dest:
                if consts.is_const(sb_name):
                    raise ValueError(f"Line {lineno}: '{sb_name}' is a constant and cannot be modified")
                if sb_name not in manifest or manifest[sb_name]['is_array']:
                    raise ValueError(f"Line {lineno}: '{sb_name}' is not a declared scalar")
                if not manifest[sb_name]['mutable']:
                    raise ValueError(f"Line {lineno}: '{sb_name}' is immutable — declare with 'let mut'")
                emit_call(callee, args_s, sb_name, manifest[sb_name]['ty'], lineno)
            else:
                emit_call(callee, args_s, None, None, lineno)
            return True

        m = CALL_DECL_RE.match(line)
        if m and m.group(4) in functions:
            is_mut = m.group(1) is not None
            name, ty, callee, args_s = m.group(2), m.group(3), m.group(4), m.group(5)
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
            emit_call(callee, args_s, name, ty, lineno)
            return True

        m = CALL_REASSIGN_RE.match(line)
        if m and m.group(2) in functions:
            name, callee, args_s = m.group(1), m.group(2), m.group(3)
            if consts.is_const(name):
                raise ValueError(f"Line {lineno}: '{name}' is a constant and cannot be modified")
            if name not in manifest or manifest[name]['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is not a declared scalar")
            if not manifest[name]['mutable']:
                raise ValueError(f"Line {lineno}: '{name}' is immutable — declare with 'let mut'")
            emit_call(callee, args_s, name, manifest[name]['ty'], lineno)
            return True

        m = CALL_STMT_RE.match(line)
        if m and m.group(1) in functions:
            callee, args_s = m.group(1), m.group(2)
            emit_call(callee, args_s, None, None, lineno)
            return True


        # Validate a step-op target: must be a declared, mutable, numeric var.
        def validate_step_target(name, want_array, lineno):
            if consts.is_const(name):
                raise ValueError(
                    f"Line {lineno}: '{name}' is a constant and cannot be modified")
            if name not in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is not declared")
            entry = manifest[name]
            if not entry['mutable']:
                raise ValueError(
                    f"Line {lineno}: '{name}' is immutable — "
                    f"declare with 'let mut' to allow step operations")
            if want_array and not entry['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is a scalar, not an array")
            if (not want_array) and entry['is_array']:
                raise ValueError(
                    f"Line {lineno}: '{name}' is an array — use '{name}[i]' to step an element")
            ty = entry['ty']
            if ty not in INT_TYPES and ty not in FLOAT_TYPES:
                raise ValueError(
                    f"Line {lineno}: step operations require a numeric type, not {ty}")
            return entry

        # --- scalar-LHS statements with a field-access expression RHS --------
        # These run after all call/builtin/single-access forms, so they only
        # catch genuine expressions that MIX field accesses with operators /
        # literals / vars (gated on rhs_has_access). Each access is lowered to
        # its mangled slot label, then the normal expression engine evaluates
        # it — exactly as the field-LHS path already does.

        # let [mut] name: ty = A.x + B.y;   (declaration)
        m = DECL_EXPR_ACCESS_RE.match(line)
        if m and is_expression(m.group(4)) and rhs_has_access(m.group(4)):
            is_mut, name, ty, rhs = (m.group(1) is not None, m.group(2),
                                     m.group(3), m.group(4))
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            rpn = expr_to_rpn(expr_tokenize(lower_expr(rhs, lineno), lineno), lineno)
            check_expr_target_ty(ty, name, rpn)
            manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
            emit(('decl_expr', name, ty, rhs, rpn, is_mut))
            return True

        # name += S::Inner.q;   (compound-assign; also -= *= /=)
        m = COMPOUND_EXPR_ACCESS_RE.match(line)
        if m and rhs_has_access(m.group(3)):
            name, op, rhs = m.group(1), m.group(2), m.group(3)
            entry = validate_step_target(name, False, lineno)
            ty = entry['ty']
            rhs_l = lower_expr(rhs, lineno)
            rpn = expr_to_rpn(expr_tokenize(f"{name} {op[0]} ({rhs_l})", lineno), lineno)
            if expr_yields_bool(rpn):
                raise ValueError(
                    f"Line {lineno}: step operation RHS must be numeric, not a comparison")
            emit(('reassign_expr', name, ty, f"{name} {op} {rhs}", rpn))
            return True

        # name = A.x * 2;   (reassignment)
        m = REASSIGN_EXPR_ACCESS_RE.match(line)
        if m and is_expression(m.group(2)) and rhs_has_access(m.group(2)):
            name, rhs = m.group(1), m.group(2)
            if consts.is_const(name):
                raise ValueError(f"Line {lineno}: '{name}' is a constant and cannot be modified")
            if name not in manifest or manifest[name]['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is not a declared scalar")
            if not manifest[name]['mutable']:
                raise ValueError(f"Line {lineno}: '{name}' is immutable — declare with 'let mut'")
            ty = manifest[name]['ty']
            rpn = expr_to_rpn(expr_tokenize(lower_expr(rhs, lineno), lineno), lineno)
            check_expr_target_ty(ty, name, rpn)
            emit(('reassign_expr', name, ty, rhs, rpn))
            return True

        # --- Array element READ:  let [mut] x: T = arr[i];  (declaration) ---
        m = ARRAY_READ_DECL_RE.match(line)
        if m and m.group(4) in manifest and manifest[m.group(4)]['is_array']:
            is_mut, name, ty, arr, raw_index = (
                m.group(1) is not None, m.group(2), m.group(3),
                m.group(4), m.group(5))
            if name in RESERVED:
                raise ValueError(f"Line {lineno}: '{name}' is a reserved word")
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            aentry = manifest[arr]
            if ty != aentry['ty']:
                raise ValueError(
                    f"Line {lineno}: type mismatch — '{arr}' is {aentry['ty']}[], "
                    f"cannot read an element into a {ty}")
            idx_kind, idx = _classify_index(arr, raw_index, lineno)
            manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
            emit(('load_array', name, ty, arr, idx_kind, idx, True, is_mut))
            return True

        # --- Array element READ:  x = arr[i];  (reassignment) ---
        m = ARRAY_READ_REASSIGN_RE.match(line)
        if m and m.group(2) in manifest and manifest[m.group(2)]['is_array']:
            name, arr, raw_index = m.group(1), m.group(2), m.group(3)
            if consts.is_const(name):
                raise ValueError(f"Line {lineno}: '{name}' is a constant and cannot be modified")
            if name not in manifest or manifest[name]['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is not a declared scalar")
            if not manifest[name]['mutable']:
                raise ValueError(f"Line {lineno}: '{name}' is immutable — declare with 'let mut'")
            ty = manifest[name]['ty']
            aentry = manifest[arr]
            if ty != aentry['ty']:
                raise ValueError(
                    f"Line {lineno}: type mismatch — '{arr}' is {aentry['ty']}[], "
                    f"cannot read an element into a {ty}")
            idx_kind, idx = _classify_index(arr, raw_index, lineno)
            emit(('load_array', name, ty, arr, idx_kind, idx, False, None))
            return True

        # name++ / name--
        m = STEP_INCDEC_SCALAR_RE.match(line)
        if m:
            name, op = m.group(1), m.group(2)
            entry = validate_step_target(name, False, lineno)
            ty = entry['ty']
            arith = '+' if op == '++' else '-'
            # desugar to: name = name <arith> 1
            rpn = expr_to_rpn(expr_tokenize(f"{name} {arith} 1", lineno), lineno)
            emit(('reassign_expr', name, ty, f"{name}{op}", rpn))
            return True

        # name[i]++ / name[i]--
        m = STEP_INCDEC_ARRAY_RE.match(line)
        if m:
            name, raw_index, op = m.group(1), m.group(2), m.group(3)
            entry = validate_step_target(name, True, lineno)
            ty = entry['ty']
            index = int(raw_index) if raw_index.isdigit() else resolve_size(raw_index, consts, lineno)
            if index >= entry['max_size']:
                raise ValueError(
                    f"Line {lineno}: index {index} out of bounds for "
                    f"'{name}[{entry['max_size']}]'")
            slot = f"{name}_{index}"
            arith = '+' if op == '++' else '-'
            rpn = expr_to_rpn(expr_tokenize(f"{slot} {arith} 1", lineno), lineno)
            emit(('reassign_expr_slot', slot, ty, f"{name}[{index}]{op}", rpn))
            return True

        # name += expr  (and -= *= /=)
        m = STEP_COMPOUND_SCALAR_RE.match(line)
        if m:
            name, op, rhs = m.group(1), m.group(2), m.group(3)
            entry = validate_step_target(name, False, lineno)
            ty = entry['ty']
            arith = op[0]  # '+', '-', '*', '/'
            # desugar to: name = name <arith> (rhs)   — RHS parenthesized
            rpn = expr_to_rpn(expr_tokenize(f"{name} {arith} ({rhs})", lineno), lineno)
            if expr_yields_bool(rpn):
                raise ValueError(
                    f"Line {lineno}: step operation RHS must be numeric, not a comparison")
            emit(('reassign_expr', name, ty, f"{name} {op} {rhs}", rpn))
            return True

        # name[i] += expr  (and -= *= /=)
        m = STEP_COMPOUND_ARRAY_RE.match(line)
        if m:
            name, raw_index, op, rhs = m.group(1), m.group(2), m.group(3), m.group(4)
            entry = validate_step_target(name, True, lineno)
            ty = entry['ty']
            index = int(raw_index) if raw_index.isdigit() else resolve_size(raw_index, consts, lineno)
            if index >= entry['max_size']:
                raise ValueError(
                    f"Line {lineno}: index {index} out of bounds for "
                    f"'{name}[{entry['max_size']}]'")
            slot = f"{name}_{index}"
            arith = op[0]
            rpn = expr_to_rpn(expr_tokenize(f"{slot} {arith} ({rhs})", lineno), lineno)
            if expr_yields_bool(rpn):
                raise ValueError(
                    f"Line {lineno}: step operation RHS must be numeric, not a comparison")
            emit(('reassign_expr_slot', slot, ty, f"{name}[{index}] {op} {rhs}", rpn))
            return True

        # --- Array reassignment:  name[i] = value; ---
        m = REASSIGN_ARRAY_RE.match(line)
        if m:
            name      = m.group(1)
            raw_index = m.group(2)
            raw       = m.group(3)
            if consts.is_const(name):
                raise ValueError(
                    f"Line {lineno}: '{name}' is a constant and cannot be reassigned"
                )
            if name not in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is not declared")
            entry = manifest[name]
            if not entry['mutable']:
                raise ValueError(
                    f"Line {lineno}: '{name}' is immutable — "
                    f"declare with 'let mut' to allow reassignment"
                )
            if not entry['is_array']:
                raise ValueError(f"Line {lineno}: '{name}' is a scalar, not an array")
            ty = entry['ty']
            # Runtime variable index: not a digit and not a const, but a declared
            # scalar -> emit a computed store (LEA base ; LDR idx ; STORQ).
            is_dyn = (not raw_index.isdigit()
                      and not consts.is_const(raw_index)
                      and raw_index in manifest
                      and not manifest[raw_index]['is_array'])
            if is_dyn:
                if ty not in ("i64", "ptr"):
                    raise ValueError(
                        f"Line {lineno}: runtime index '{name}[{raw_index}]' requires an "
                        f"8-byte element type (i64 or ptr); '{name}' is {ty}[]")
                if manifest[raw_index]['ty'] not in INT_TYPES:
                    raise ValueError(
                        f"Line {lineno}: array index '{raw_index}' must be an integer, "
                        f"not {manifest[raw_index]['ty']}")
                # RHS may be a full expression; evaluate it at emit time into r0.
                rhs_l = lower_expr(raw, lineno)
                rpn = expr_to_rpn(expr_tokenize(rhs_l, lineno), lineno)
                emit(('store_array', name, ty, raw_index, raw, rpn))
                return True
            index = resolve_size(raw_index, consts, lineno) if not raw_index.isdigit() else int(raw_index)
            if index >= entry['max_size']:
                raise ValueError(
                    f"Line {lineno}: index {index} out of bounds for "
                    f"'{name}[{entry['max_size']}]'"
                )
            encoded = encode_value(raw, ty, consts, lineno)
            emit(('reassign_array', name, ty, index, raw, encoded))
            return True

        # --- Scalar reassignment:  name = value; ---
        m = REASSIGN_SCALAR_RE.match(line)
        if m:
            name = m.group(1)
            raw  = m.group(2)
            if consts.is_const(name):
                raise ValueError(
                    f"Line {lineno}: '{name}' is a constant and cannot be reassigned"
                )
            if name not in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is not declared")
            entry = manifest[name]
            if not entry['mutable']:
                raise ValueError(
                    f"Line {lineno}: '{name}' is immutable — "
                    f"declare with 'let mut' to allow reassignment"
                )
            if entry['is_array']:
                raise ValueError(
                    f"Line {lineno}: '{name}' is an array — "
                    f"use '{name}[i] = value' to reassign an element"
                )
            ty = entry['ty']
            # Expression on the RHS?
            if is_expression(raw):
                rpn = expr_to_rpn(expr_tokenize(raw, lineno), lineno)
                yields_bool = expr_yields_bool(rpn)
                # Shift expressions are integer-only — target must be an integer type.
                if expr_has_shift(rpn) and ty not in INT_TYPES:
                    raise ValueError(
                        f"Line {lineno}: shift operations are integer-only, "
                        f"cannot store result in {ty} '{name}'"
                    )
                # Type compatibility:
                #   bool target  <- comparison/logical expression (yields bool)
                #   numeric target <- arithmetic expression (yields number)
                if ty in BOOL_TYPES:
                    if not yields_bool:
                        raise ValueError(
                            f"Line {lineno}: bool '{name}' needs a comparison/logical "
                            f"expression, not arithmetic"
                        )
                elif ty in CHAR_TYPES:
                    raise ValueError(
                        f"Line {lineno}: expressions are not allowed for {ty}"
                    )
                else:
                    if yields_bool:
                        raise ValueError(
                            f"Line {lineno}: comparison result is a bool and cannot "
                            f"be stored in {ty} '{name}'"
                        )
                if expr_is_constant(rpn, consts, manifest):
                    folded = fold_constant_expr(rpn, consts, lineno)
                    encoded = encode_scalar_for_type(folded, ty)
                    emit(('reassign_scalar', name, ty, f"{raw} = {folded}", encoded))
                else:
                    emit(('reassign_expr', name, ty, raw, rpn))
                return True
            encoded = encode_value(raw, ty, consts, lineno)
            emit(('reassign_scalar', name, ty, raw, encoded))
            return True

        # --- Array declaration ---
        m = ARRAY_RE.match(line)
        if m:
            is_mut   = m.group(1) is not None
            name     = m.group(2)
            ty       = m.group(3)
            raw_size = m.group(4)
            raw_init = m.group(5)
            if name in RESERVED:
                raise ValueError(
                    f"Line {lineno}: '{name}' is a reserved word and cannot be a variable name"
                )
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            max_size = resolve_size(raw_size, consts, lineno)
            elements = parse_array_elements(raw_init, ty, max_size, consts, lineno)
            manifest[name] = {
                'ty': ty, 'mutable': is_mut,
                'is_array': True, 'max_size': max_size
            }
            emit(('array', name, ty, max_size, elements, is_mut))
            return True

        # --- Scalar declaration ---
        m = SCALAR_RE.match(line)
        if m:
            is_mut = m.group(1) is not None
            name   = m.group(2)
            ty     = m.group(3)
            raw    = m.group(4)
            if name in RESERVED:
                raise ValueError(
                    f"Line {lineno}: '{name}' is a reserved word and cannot be a variable name"
                )
            if consts.is_const(name) or name in manifest:
                raise ValueError(f"Line {lineno}: '{name}' is already defined")
            # Expression on the RHS?
            if is_expression(raw):
                rpn = expr_to_rpn(expr_tokenize(raw, lineno), lineno)
                yields_bool = expr_yields_bool(rpn)
                if expr_has_shift(rpn) and ty not in INT_TYPES:
                    raise ValueError(
                        f"Line {lineno}: shift operations are integer-only, "
                        f"cannot store result in {ty} '{name}'"
                    )
                if ty in BOOL_TYPES:
                    if not yields_bool:
                        raise ValueError(
                            f"Line {lineno}: bool '{name}' needs a comparison/logical "
                            f"expression, not arithmetic"
                        )
                elif ty in CHAR_TYPES:
                    raise ValueError(
                        f"Line {lineno}: expressions are not allowed for {ty}"
                    )
                else:
                    if yields_bool:
                        raise ValueError(
                            f"Line {lineno}: comparison result is a bool and cannot "
                            f"be stored in {ty} '{name}'"
                        )
                manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
                if expr_is_constant(rpn, consts, manifest):
                    folded = fold_constant_expr(rpn, consts, lineno)
                    encoded = encode_scalar_for_type(folded, ty)
                    emit(('scalar', name, ty, f"{raw} = {folded}", encoded, is_mut))
                else:
                    emit(('decl_expr', name, ty, raw, rpn, is_mut))
                return True
            encoded = encode_value(raw, ty, consts, lineno)
            manifest[name] = {'ty': ty, 'mutable': is_mut, 'is_array': False}
            emit(('scalar', name, ty, raw, encoded, is_mut))
            return True

        raise ValueError(f"Line {lineno}: unsupported syntax: {line!r}")

    for lineno, raw_line in enumerate(src.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue

        # Unsigned type check (applies everywhere)
        um = UNSIGNED_RE.match(line)
        if um:
            raise ValueError(
                f"Line {lineno}: unknown type '{um.group(1)}' — "
                f"unsigned types are not supported in Nexa. "
                f"Use i8/i16/i32/i64 instead."
            )

        # --- const declaration (allowed both globally and in main) ---
        m = CONST_RE.match(line)
        if m:
            name = m.group(1)
            ty   = m.group(2)
            raw  = m.group(3)
            if name in RESERVED:
                raise ValueError(
                    f"Line {lineno}: '{name}' is a reserved word and cannot be a const name"
                )
            # Reassignment-to-const guard: a const name can't already be a local
            if name in manifest:
                raise ValueError(
                    f"Line {lineno}: '{name}' is already a variable, "
                    f"cannot redefine as const"
                )
            int_value = encode_value(raw, ty, consts, lineno)
            consts.define(name, ty, raw, int_value, lineno)
            global_consts.append((name, ty, raw, int_value))
            emit(('const', name, ty, raw, int_value, current_fn[0] is not None))
            continue

        # ===================================================================
        # struct definitions — allowed in global scope AND inside fn main()
        # (a struct definition emits no runtime code; it only registers fields
        #  and reserves their zero-initialised data slots)
        # ===================================================================
        def open_struct(sname, lineno):
            if sname in RESERVED:
                raise ValueError(f"Line {lineno}: '{sname}' is a reserved word")
            path = "__".join(struct_path + [sname])
            if path in declared_structs:
                raise ValueError(f"Line {lineno}: struct '{sname}' already defined here")
            declared_structs.add(path)
            if len(struct_path) == 0:
                top_level_structs.add(sname)
            struct_path.append(sname)
            block_stack.append({'kind': 'struct', 'name': sname})

        # Inside a struct body: only field declarations, nested structs, and '}'.
        if block_stack and block_stack[-1]['kind'] == 'struct':
            mso = STRUCT_OPEN_RE.match(line)
            if mso:
                open_struct(mso.group(1), lineno)
                continue
            if line == "}":
                block_stack.pop()
                struct_path.pop()
                continue
            mfield = STRUCT_FIELD_RE.match(line)
            if mfield:
                fname, fty = mfield.group(1), mfield.group(2)
                mangled = "__".join(struct_path + [fname])
                if mangled in struct_fields:
                    raise ValueError(f"Line {lineno}: field '{fname}' already declared in this struct")
                struct_fields[mangled] = {'ty': fty}
                struct_field_order.append(mangled)
                # also expose the slot to the expression engine (as a runtime
                # operand) so fields can appear inside arbitrary expressions.
                manifest[mangled] = {'ty': fty, 'mutable': True, 'is_array': False}
                emit(('struct_field', mangled, fty))
                continue
            if STRUCT_FIELD_IMMUT_RE.match(line):
                raise ValueError(
                    f"Line {lineno}: struct fields must be mutable — use 'let mut'")
            raise ValueError(
                f"Line {lineno}: inside a struct only 'let mut' fields and nested "
                f"'struct' blocks are allowed, got: {line!r}")

        # Top-level struct opener (global scope or top of main).
        mso = STRUCT_OPEN_RE.match(line)
        if mso:
            open_struct(mso.group(1), lineno)
            continue

        # --- interrupt handler header:  isr <VECTOR> { -----------------------
        _miso = ISR_HDR_RE.match(line)
        if _miso:
            if current_fn[0] is not None:
                raise ValueError(
                    f"Line {lineno}: an 'isr' handler must be declared at top "
                    f"level — close the current function first")
            iname = "isr_" + _miso.group(1)
            current_fn[0] = iname
            current_self[0] = None
            in_main = False
            decl_sink_stack[:] = [fn_bodies[iname]]
            continue

        # --- function definition headers (top level only; no nesting) ------
        #     `fn name(...)` (stack) / `fn main()` (entry) / `def name(...)` (heap)
        _hdr_fn  = line.startswith("fn ")  or line.startswith("fn\t")
        _hdr_def = line.startswith("def ") or line.startswith("def\t")
        if _hdr_fn or _hdr_def:
            if current_fn[0] is not None:
                raise ValueError(
                    f"Line {lineno}: nested functions are not allowed — "
                    f"close the current function first")
            if _hdr_fn and FN_MAIN_RE.match(line):
                current_fn[0] = 'main'
                current_self[0] = None
                in_main = True
                decl_sink_stack[:] = [declarations]
                continue
            if _hdr_def and re.match(r"def\s+main\b", line):
                raise ValueError(
                    f"Line {lineno}: 'main' must be a stack function — use "
                    f"'fn main()', not 'def main()'")
            m = (DEF_DEF_RE if _hdr_def else FN_DEF_RE).match(line)
            if not m:
                _kw = "def" if _hdr_def else "fn"
                raise ValueError(
                    f"Line {lineno}: malformed function header — expected "
                    f"'{_kw} name(args) -> type {{'"
                    + ("" if _hdr_def else " or 'fn main() {'"))
            fname = m.group(1)
            current_fn[0] = fname
            in_main = False
            decl_sink_stack[:] = [fn_bodies[fname]]
            # bind self (compile-time receiver) if declared, validating roots
            recv = functions[fname].get('self_recv')
            if recv is not None:
                if recv[0] == 'single':
                    if recv[1] not in declared_structs:
                        raise ValueError(
                            f"Line {lineno}: 'self: {recv[1]}' refers to an unknown "
                            f"struct — declare the struct before the function")
                elif recv[0] == 'multi':
                    for r in recv[1]:
                        if r not in declared_structs:
                            raise ValueError(
                                f"Line {lineno}: 'self: (...)' root '{r}' is an "
                                f"unknown struct — declare it before the function")
                # universal: spans whatever globals exist; nothing to validate
                current_self[0] = recv
            else:
                current_self[0] = None
            # register parameters as function-scoped locals. Nexa keeps a flat
            # storage namespace, but a `fn` saves/restores its frame window on
            # the stack around every call, so the SAME parameter/local name may
            # be reused across functions as long as the type matches — each
            # activation is preserved independently on the stack.
            for pname, pty in functions[fname]['params']:
                if pname in RESERVED:
                    raise ValueError(f"Line {lineno}: parameter '{pname}' is a reserved word")
                if consts.is_const(pname):
                    raise ValueError(
                        f"Line {lineno}: parameter '{pname}' collides with a const")
                if pname in manifest:
                    ex = manifest[pname]
                    if ex.get('is_array') or ex['ty'] != pty or not ex.get('local'):
                        raise ValueError(
                            f"Line {lineno}: '{pname}' is already used elsewhere with a "
                            f"different type or kind — reuse across functions requires "
                            f"the same type")
                    # same-typed local in another function: share the slot
                    emit(('param', pname, pty))
                    continue
                manifest[pname] = {'ty': pty, 'mutable': True,
                                   'is_array': False, 'local': True}
                emit(('param', pname, pty))
            continue

        if current_fn[0] is None:
            raise ValueError(
                f"Line {lineno}: only const, struct, and fn declarations are "
                f"allowed at top level, got: {line!r}"
            )

        # ===================================================================
        # Control flow
        # ===================================================================
        def emit_condition(cond_src, false_label, ln):
            rpn = expr_to_rpn(expr_tokenize(lower_expr(cond_src, ln), ln), ln)
            emit(('cond_expr', rpn, ln))
            emit(('jz', false_label))

        in_match = bool(block_stack) and block_stack[-1]['kind'] == 'match'

        # If the top frame is a chain whose arm just closed (awaiting possible
        # ifelse/else), decide now based on the current line.
        if block_stack and block_stack[-1].get('kind') == 'chain' \
                and block_stack[-1].get('awaiting'):
            frame = block_stack[-1]
            if IFELSE_RE.match(line):
                me = IFELSE_RE.match(line)
                emit(('jmp', frame['end']))
                emit(('label', frame['pending_next']))
                cond = me.group(1)
                next_label = new_label("if_next")
                emit_condition(cond, next_label, lineno)
                frame['pending_next'] = next_label
                frame['awaiting'] = False
                continue
            elif ELSE_RE.match(line):
                emit(('jmp', frame['end']))
                emit(('label', frame['pending_next']))
                frame['pending_next'] = None
                frame['has_else'] = True
                frame['awaiting'] = False
                continue
            else:
                # chain is finished — finalize and pop, then process this line
                if frame['pending_next'] is not None:
                    emit(('label', frame['pending_next']))
                emit(('label', frame['end']))
                block_stack.pop()
                in_match = bool(block_stack) and block_stack[-1]['kind'] == 'match'

        # --- if (cond) { ---
        if not in_match:
            mi = IF_RE.match(line)
            if (not mi) and (not line.startswith("ifelse")):
                mi = IF_NOPAREN_RE.match(line)   # parenless: if <cond> {
            if mi and not line.startswith("ifelse"):
                cond = mi.group(1)
                chain_end  = new_label("if_end")
                next_label = new_label("if_next")
                emit_condition(cond, next_label, lineno)
                block_stack.append({
                    'kind': 'chain', 'end': chain_end,
                    'pending_next': next_label, 'has_else': False,
                    'awaiting': False
                })
                continue

        # --- match(var) { ---
        mm = MATCH_RE.match(line)
        if mm:
            var = mm.group(1)
            if var in manifest:
                if manifest[var]['is_array']:
                    raise ValueError(f"Line {lineno}: cannot match on an array '{var}'")
                mty = manifest[var]['ty']
            elif consts.is_const(var):
                mty = consts.consts[var]['ty']
            else:
                raise ValueError(f"Line {lineno}: match variable '{var}' is not declared")
            block_stack.append({
                'kind': 'match', 'var': var, 'ty': mty,
                'end': new_label("match_end"),
                'arms': [], 'else_body': None, 'lineno': lineno,
            })
            continue

        # --- match arms (only when directly inside a match, not in an arm) ---
        if in_match and line != "}":
            frame = block_stack[-1]
            mae = ELSE_RE.match(line)
            if mae:
                if frame['else_body'] is not None:
                    raise ValueError(f"Line {lineno}: match already has an else arm")
                frame['else_body'] = []
                decl_sink_stack.append(frame['else_body'])
                block_stack.append({'kind': 'arm'})
                continue
            maa = MATCH_ARM_RE.match(line)
            if maa:
                op  = maa.group(1) or '=='
                val = maa.group(2).strip()
                arm = {
                    'op': op, 'val': val, 'body': [],
                    'const_int': match_arm_const_int(op, val, consts),
                }
                frame['arms'].append(arm)
                decl_sink_stack.append(arm['body'])
                block_stack.append({'kind': 'arm'})
                continue
            raise ValueError(
                f"Line {lineno}: only 'if <value>' or 'else' arms are allowed "
                f"directly inside match, got: {line!r}")

        # --- closing brace: closes innermost block, or main ---
        if line == "}":
            if block_stack:
                frame = block_stack[-1]
                if frame['kind'] == 'arm':
                    # End of a match arm (or else) body: pop the frame and its
                    # declaration sink; the body has been buffered for lowering.
                    block_stack.pop()
                    decl_sink_stack.pop()
                    continue
                if frame['kind'] == 'chain':
                    # Don't finalize yet — an ifelse/else may follow. Mark the
                    # arm as closed; the next line decides continuation or end.
                    frame['awaiting'] = True
                    if frame.get('has_else'):
                        emit(('label', frame['end']))
                        block_stack.pop()
                    continue
                elif frame['kind'] == 'match':
                    block_stack.pop()
                    lower_match(frame)
                    continue
                elif frame['kind'] == 'loop':
                    # Close a while/loop/for. For a for-loop, the continue
                    # target ('cont') is the step: emit it here, then the
                    # back-edge re-tests the condition. while/loop have no step
                    # (cont == start), so they just take the back-edge.
                    block_stack.pop()
                    if frame.get('looptype') == 'for':
                        emit(('label', frame['cont']))
                        handle_simple_statement(frame['step'] + ";", frame['step_lineno'])
                    emit(('jmp', frame['start']))
                    emit(('label', frame['end']))
                    continue
            else:
                # closing the current function's body
                if current_fn[0] is None:
                    raise ValueError(f"Line {lineno}: unexpected '}}'")
                current_fn[0] = None
                current_self[0] = None
                in_main = False
                decl_sink_stack[:] = [declarations]
                continue

        # --- loop { ---   (infinite loop; only exits via break)
        if LOOP_RE.match(line):
            loop_start = new_label("loop_start")
            loop_end   = new_label("loop_end")
            emit(('label', loop_start))
            block_stack.append({'kind': 'loop', 'looptype': 'loop',
                                'start': loop_start, 'cont': loop_start,
                                'end': loop_end})
            continue

        # --- while (cond) { ---   (test-at-top conditional loop)
        mw = WHILE_RE.match(line)
        if mw:
            cond = mw.group(1)
            loop_start = new_label("while_start")   # re-evaluated each iteration
            loop_end   = new_label("while_end")
            emit(('label', loop_start))
            emit_condition(cond, loop_end, lineno)   # eval cond; JZ loop_end
            block_stack.append({'kind': 'loop', 'looptype': 'while',
                                'start': loop_start, 'cont': loop_start,
                                'end': loop_end})
            continue

        # --- for(init; cond; step) { ---
        mf = FOR_RE.match(line)
        if mf:
            init = mf.group(1).strip()
            cond = mf.group(2).strip()
            step = mf.group(3).strip()
            # init runs once, before the loop (declares/initializes the counter)
            handle_simple_statement(init + ";", lineno)
            for_start = new_label("for_start")     # condition re-check
            for_step  = new_label("for_step")      # continue target -> run step
            for_end   = new_label("for_end")       # break target
            emit(('label', for_start))
            emit_condition(cond, for_end, lineno)  # eval cond; JZ for_end
            block_stack.append({'kind': 'loop', 'looptype': 'for',
                                'start': for_start, 'cont': for_step,
                                'end': for_end, 'step': step, 'step_lineno': lineno})
            continue

        # --- break; / continue; ---
        if BREAK_RE.match(line):
            emit(('jmp', innermost_loop(lineno)['end']))
            continue
        if CONTINUE_RE.match(line):
            # while/loop: 'cont' is the top of the loop (re-tests the condition
            # for while).  for: 'cont' is the step, so the increment still runs.
            emit(('jmp', innermost_loop(lineno)['cont']))
            continue

        # --- abort; ---   hard, clean exit. Allowed anywhere, including inside
        # if/ifelse/else, match arms, and loops — emitted into the current sink.
        if ABORT_RE.match(line):
            emit(('abort',))
            continue

        # --- sei; / cli; ---  enable / disable global hardware interrupts.
        if SEI_RE.match(line):
            emit(('sei',))
            continue
        if CLI_RE.match(line):
            emit(('cli',))
            continue

        # --- println("..."); ---  like print() but always appends a newline.
        # The string argument is optional: println(); emits a blank line.
        mprintln = PRINTLN_RE.match(line)
        if mprintln:
            fmt = mprintln.group(1)
            if fmt is not None:
                for seg in parse_format(fmt, lineno):
                    emit(seg)
            nl_label = new_string_label([10])
            emit(('print_lit', nl_label, 1))
            continue

        # --- print("..."); ---  format string with {var} interpolation and
        # escape sequences. Emits one write per literal run / interpolation,
        # in order, into the current sink (so it nests in loops/ifs/arms).
        mprint = PRINT_RE.match(line)
        if mprint:
            for seg in parse_format(mprint.group(1), lineno):
                emit(seg)
            continue


        # --- simple statements: step ops, reassignments, declarations ---
        handle_simple_statement(line, lineno)

    # -----------------------------------------------------------------------
    # Emit Axiom assembly
    # -----------------------------------------------------------------------
    out = []
    out.append("; Axiom assembly generated from Tier-1")
    out.append("")

    # Tracks scratch slots needed by runtime expressions: slot_name -> type
    scratch_slots = {}
    all_scratch = {}

    # Global consts emitted first as a comment block for clarity
    out.append("; --- consts ---")
    for name, ty, raw, int_value in global_consts:
        byte_width = TYPE_INFO[ty][1]
        hex_digits = byte_width * 2
        if ty in CHAR_TYPES:
            out.append(f";   const {name}: {ty} = {raw}  (code = {int_value} = 0x{int_value:02X})")
        elif ty in FLOAT_TYPES:
            out.append(f";   const {name}: {ty} = {raw}  (bits = 0x{int_value:0{hex_digits}X})")
        else:
            out.append(f";   const {name}: {ty} = {raw}  (= {int_value})")
    out.append("")

    # Interrupt vector bindings (read by the AVR backend to build the vector
    # table; ignored by the x86 backend). Emitted before any code.
    _isr_vectors = [(functions[f]['vector'], f)
                    for f in fn_order if functions[f].get('kind') == 'isr']
    if _isr_vectors:
        out.append("; --- interrupt vectors ---")
        for _vec, _iname in _isr_vectors:
            out.append(f".vector {_vec} {_iname}")
        out.append("")

    def emit_one(decl):
        kind = decl[0]

        if kind == 'const':
            _, name, ty, raw, encoded, was_in_main = decl
            byte_width = TYPE_INFO[ty][1]
            hex_digits = byte_width * 2
            scope = "local" if was_in_main else "global"
            if ty in CHAR_TYPES:
                out.append(f"    ; const {name}: {ty} [{scope}] = {raw}  (code = {encoded} = 0x{encoded:02X})")
            elif ty in FLOAT_TYPES:
                out.append(f"    ; const {name}: {ty} [{scope}] = {raw}  (bits = 0x{encoded:0{hex_digits}X})")
            else:
                out.append(f"    ; const {name}: {ty} [{scope}] = {raw}")
            emit_ldr_str(out, encoded, ty, name)
            out.append("")

        elif kind == 'scalar':
            _, name, ty, raw, encoded, is_mut = decl
            byte_width = TYPE_INFO[ty][1]
            hex_digits = byte_width * 2
            mut_tag = " [mut]" if is_mut else " [immutable]"
            if ty in CHAR_TYPES:
                out.append(f"    ; {name}: {ty}{mut_tag} = {raw}  (code = {encoded} = 0x{encoded:02X})")
            elif ty in FLOAT_TYPES:
                out.append(f"    ; {name}: {ty}{mut_tag} = {raw}  (bits = 0x{encoded:0{hex_digits}X})")
            else:
                out.append(f"    ; {name}: {ty}{mut_tag} = {raw}")
            emit_ldr_str(out, encoded, ty, name)
            out.append("")

        elif kind == 'array':
            _, name, ty, max_size, elements, is_mut = decl
            byte_width = TYPE_INFO[ty][1]
            hex_digits = byte_width * 2
            mut_tag = " [mut]" if is_mut else " [immutable]"
            out.append(f"    ; {name}: {ty}[{max_size}]{mut_tag}")
            for idx, encoded in enumerate(elements):
                slot = f"{name}_{idx}"
                if ty in CHAR_TYPES:
                    out.append(f"    ; [{idx}] = 0x{encoded:02X} ({chr(encoded)!r} if printable)")
                elif ty in FLOAT_TYPES:
                    out.append(f"    ; [{idx}] bits = 0x{encoded:0{hex_digits}X}")
                else:
                    out.append(f"    ; [{idx}] = {encoded}")
                emit_ldr_str(out, encoded, ty, slot)
            out.append("")

        elif kind == 'reassign_scalar':
            _, name, ty, raw, encoded = decl
            byte_width = TYPE_INFO[ty][1]
            hex_digits = byte_width * 2
            if ty in FLOAT_TYPES:
                out.append(f"    ; {name} = {raw}  (bits = 0x{encoded:0{hex_digits}X})")
            else:
                out.append(f"    ; {name} = {raw}")
            emit_ldr_str(out, encoded, ty, name)
            out.append("")

        elif kind == 'reassign_array':
            _, name, ty, index, raw, encoded = decl
            slot = f"{name}_{index}"
            byte_width = TYPE_INFO[ty][1]
            hex_digits = byte_width * 2
            if ty in FLOAT_TYPES:
                out.append(f"    ; {name}[{index}] = {raw}  (bits = 0x{encoded:0{hex_digits}X})")
            else:
                out.append(f"    ; {name}[{index}] = {raw}")
            emit_ldr_str(out, encoded, ty, slot)
            out.append("")

        elif kind == 'load_array':
            # x = arr[i]  -> read an array element into a scalar slot. Both static
            # and runtime indices use mem[&arr_0 + index*8] (LDRQ marks r0 as a
            # transient result, so the store into a different slot is permitted).
            _, dest, ty, arr, idx_kind, idx, is_decl, _is_mut = decl
            out.append(f"    ; {dest} = {arr}[{idx}]"
                       + ("  (runtime index)" if idx_kind == 'dyn' else ""))
            out.append(f"    LEA r2, {arr}_0")
            if idx_kind == 'static':
                out.append(f"    LDR r3, #{idx}")
            else:
                out.append(f"    LDR r3, {idx}")
            out.append(f"    LDRQ r0, r2, r3")
            out.append(f"    STR r0, {dest}")
            out.append("")

        elif kind == 'store_array':
            # arr[i] = <expr>  -> compute the value, then store at a runtime index.
            _, arr, ty, idx, raw, rpn = decl
            out.append(f"    ; {arr}[{idx}] = {raw}  (runtime index)")
            scratch = emit_runtime_expr(out, rpn, ty, consts, manifest, 0)
            for s in scratch:
                scratch_slots[s] = ty
            # value is in r0; base/index go to r2/r3 (neither clobbers r0)
            out.append(f"    LEA r2, {arr}_0")
            out.append(f"    LDR r3, {idx}")
            out.append(f"    STORQ r0, r2, r3")
            out.append("")

        elif kind == 'decl_expr':
            _, name, ty, raw, rpn, is_mut = decl
            mut_tag = " [mut]" if is_mut else " [immutable]"
            out.append(f"    ; {name}: {ty}{mut_tag} = {raw}  (runtime expression)")
            scratch = emit_runtime_expr(out, rpn, ty, consts, manifest, 0)
            for s in scratch:
                scratch_slots[s] = ty
            out.append(f"    STR r0, {name}")
            out.append("")

        elif kind == 'reassign_expr':
            _, name, ty, raw, rpn = decl
            out.append(f"    ; {name} = {raw}  (runtime expression)")
            scratch = emit_runtime_expr(out, rpn, ty, consts, manifest, 0)
            for s in scratch:
                scratch_slots[s] = ty
            out.append(f"    STR r0, {name}")
            out.append("")

        elif kind == 'reassign_expr_slot':
            _, slot, ty, raw, rpn = decl
            out.append(f"    ; {raw}  (runtime expression)")
            scratch = emit_runtime_expr(out, rpn, ty, consts, manifest, 0)
            for s in scratch:
                scratch_slots[s] = ty
            out.append(f"    STR r0, {slot}")
            out.append("")

        elif kind == 'cond_expr':
            _, rpn, ln = decl
            out.append(f"    ; condition")
            # Evaluate condition; result (0/1) ends up in r0.
            scratch = emit_runtime_expr(out, rpn, 'i32', consts, manifest, ln)
            for s in scratch:
                scratch_slots[s] = 'i32'
            out.append("")

        elif kind == 'jz':
            _, target = decl
            out.append(f"    JZ {target}")
            out.append("")

        elif kind == 'jmp':
            _, target = decl
            out.append(f"    JMP {target}")
            out.append("")

        elif kind == 'label':
            _, name = decl
            out.append(f"{name}:")
            out.append("")

        elif kind == 'abort':
            out.append("    ; abort -> clean program exit")
            out.append("    ABORT")
            out.append("")

        elif kind == 'sei':
            out.append("    ; sei -> enable global interrupts")
            out.append("    SEI")
            out.append("")

        elif kind == 'cli':
            out.append("    ; cli -> disable global interrupts")
            out.append("    CLI")
            out.append("")

        elif kind == 'print_lit':
            _, label, length = decl
            out.append(f"    ; print literal ({length} bytes)")
            out.append(f"    LEA r4, {label}")
            out.append(f"    LDR r3, #{length}")
            out.append(f"    WRITE")
            out.append("")

        elif kind == 'print_char':
            _, var = decl
            out.append(f"    ; print char {var}")
            out.append(f"    LDR r0, {var}")
            out.append(f"    STR r0, __char1")
            out.append(f"    LEA r4, __char1")
            out.append(f"    LDR r3, #1")
            out.append(f"    WRITE")
            out.append("")

        elif kind == 'print_bool':
            # print "true" (nonzero) or "false" (zero) for a runtime bool
            _, var, true_lbl, false_lbl = decl
            print_uid[0] += 1
            u = print_uid[0]
            out.append(f"    ; print bool {var}  (true / false)")
            out.append(f"    LDR r0, {var}")
            out.append(f"    JZ bfalse_{u}")
            out.append(f"    LEA r4, {true_lbl}")
            out.append(f"    LDR r3, #4")
            out.append(f"    WRITE")
            out.append(f"    JMP bdone_{u}")
            out.append(f"bfalse_{u}:")
            out.append(f"    LEA r4, {false_lbl}")
            out.append(f"    LDR r3, #5")
            out.append(f"    WRITE")
            out.append(f"bdone_{u}:")
            out.append("")

        elif kind == 'print_int':
            _, var = decl
            out.append(f"    ; print int {var}  (-> shared fn___itoa)")
            # value into the workspace slot, then call the one shared routine.
            # The arg crosses a function boundary via memory (__pn), so DCE
            # never touches it; r0 is consumed immediately by the STR.
            out.append(f"    LDR r0, {var}")
            out.append(f"    STR r0, __pn")
            out.append(f"    CALL fn___itoa")
            out.append("")

        elif kind == 'print_float':
            _, var = decl
            out.append(f"    ; print float {var}  (-> shared fn___dtoa)")
            out.append(f"    LDR r0, {var}")
            out.append(f"    STR r0, __fp")
            out.append(f"    CALL fn___dtoa")
            out.append("")

        elif kind in ('builtin', 'builtin_set'):
            # ('builtin', name, ty, fn, arg) | ('builtin_set', name, fn, arg)
            if kind == 'builtin':
                _, name, _ty, fn, arg = decl
            else:
                _, name, fn, arg = decl
            if fn == 'input':
                print_uid[0] += 1
                u = print_uid[0]
                tty = _ty if kind == 'builtin' else manifest[name]['ty']

                def emit_read_byte(slot):
                    # zero the slot first so EOF (0 bytes read) reads as 0
                    out.append(f"    LDR r0, #0")
                    out.append(f"    STR r0, {slot}")
                    out.append(f"    LEA r4, {slot}")
                    out.append(f"    LDR r3, #1")
                    out.append(f"    READ")

                def emit_skip_ws(lbl):
                    # skip leading LF/CR/space, leaving the first significant
                    # character in __ic
                    out.append(f"{lbl}:")
                    emit_read_byte("__ic")
                    for code in (10, 13, 32):
                        out.append(f"    LDR r0, __ic")
                        out.append(f"    LDR r1, #{code}")
                        out.append(f"    SUB r0, r1")
                        out.append(f"    STR r0, __iflag")
                        out.append(f"    LDR r0, __iflag")
                        out.append(f"    JZ {lbl}")

                if tty in CHAR_TYPES:
                    out.append(f"    ; {name} = input()  (read 1 char, skipping CR/LF)")
                    out.append(f"input_read_{u}:")
                    out.append(f"    LEA r4, {name}")
                    out.append(f"    LDR r3, #1")
                    out.append(f"    READ")
                    out.append(f"    LDR r0, {name}")
                    out.append(f"    LDR r1, #10")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    JZ input_read_{u}")
                    out.append(f"    LDR r0, {name}")
                    out.append(f"    LDR r1, #13")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    JZ input_read_{u}")

                elif tty in INT_TYPES or tty in BOOL_TYPES:
                    out.append(f"    ; {name} = input()  (read integer token, atoi)")
                    out.append(f"    LDR r0, #0")
                    out.append(f"    STR r0, __iacc")
                    out.append(f"    LDR r0, #1")
                    out.append(f"    STR r0, __isign")
                    emit_skip_ws(f"in_lead_{u}")
                    # optional leading '-'
                    out.append(f"    LDR r0, __ic")
                    out.append(f"    LDR r1, #45")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_conv_{u}")
                    out.append(f"    LDR r0, #0")
                    out.append(f"    LDR r1, #1")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __isign")
                    emit_read_byte("__ic")
                    out.append(f"in_conv_{u}:")
                    out.append(f"    LDR r0, __ic")
                    out.append(f"    LDR r1, #48")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __idig")
                    # if digit < 0 -> done
                    out.append(f"    LDR r0, __idig")
                    out.append(f"    LDR r1, #0")
                    out.append(f"    CMP r0, r1")
                    out.append(f"    SETLT r0")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_done_{u}")
                    # if 9 < digit -> done
                    out.append(f"    LDR r0, #9")
                    out.append(f"    LDR r1, __idig")
                    out.append(f"    CMP r0, r1")
                    out.append(f"    SETLT r0")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_done_{u}")
                    # acc = acc*10 + digit
                    out.append(f"    LDR r0, __iacc")
                    out.append(f"    LDR r1, #10")
                    out.append(f"    MUL r0, r1")
                    out.append(f"    LDR r1, __idig")
                    out.append(f"    ADD r0, r1")
                    out.append(f"    STR r0, __iacc")
                    emit_read_byte("__ic")
                    out.append(f"    JMP in_conv_{u}")
                    out.append(f"in_done_{u}:")
                    if tty in BOOL_TYPES:
                        # bool: false iff the accumulated value is 0
                        out.append(f"    LDR r0, __iacc")
                        out.append(f"    LDR r1, #0")
                        out.append(f"    CMP r0, r1")
                        out.append(f"    SETNE r0")
                        out.append(f"    STR r0, {name}")
                    else:
                        out.append(f"    LDR r0, __iacc")
                        out.append(f"    LDR r1, __isign")
                        out.append(f"    MUL r0, r1")
                        out.append(f"    STR r0, {name}")

                elif tty in FLOAT_TYPES:
                    out.append(f"    ; {name} = input()  (read float token, atof)")
                    out.append(f"    LDR r0, #0")
                    out.append(f"    STR r0, __iacc")
                    out.append(f"    LDR r0, #1")
                    out.append(f"    STR r0, __isign")
                    emit_skip_ws(f"in_lead_{u}")
                    # optional leading '-'
                    out.append(f"    LDR r0, __ic")
                    out.append(f"    LDR r1, #45")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_int_{u}")
                    out.append(f"    LDR r0, #0")
                    out.append(f"    LDR r1, #1")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __isign")
                    emit_read_byte("__ic")
                    # integer part
                    out.append(f"in_int_{u}:")
                    out.append(f"    LDR r0, __ic")
                    out.append(f"    LDR r1, #48")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __idig")
                    out.append(f"    LDR r0, __idig")
                    out.append(f"    LDR r1, #0")
                    out.append(f"    CMP r0, r1")
                    out.append(f"    SETLT r0")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_afterint_{u}")
                    out.append(f"    LDR r0, #9")
                    out.append(f"    LDR r1, __idig")
                    out.append(f"    CMP r0, r1")
                    out.append(f"    SETLT r0")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_afterint_{u}")
                    out.append(f"    LDR r0, __iacc")
                    out.append(f"    LDR r1, #10")
                    out.append(f"    MUL r0, r1")
                    out.append(f"    LDR r1, __idig")
                    out.append(f"    ADD r0, r1")
                    out.append(f"    STR r0, __iacc")
                    emit_read_byte("__ic")
                    out.append(f"    JMP in_int_{u}")
                    out.append(f"in_afterint_{u}:")
                    # facc = (double)iacc
                    out.append(f"    LDR r0, __iacc")
                    out.append(f"    I2F r0")
                    out.append(f"    STR r0, __facc")
                    # if current char != '.', finish
                    out.append(f"    LDR r0, __ic")
                    out.append(f"    LDR r1, #46")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_ffin_{u}")
                    # fractional part
                    emit_read_byte("__ic")
                    out.append(f"    LDR r0, #0x3FF0000000000000")   # 1.0
                    out.append(f"    STR r0, __fscale")
                    out.append(f"in_frac_{u}:")
                    out.append(f"    LDR r0, __ic")
                    out.append(f"    LDR r1, #48")
                    out.append(f"    SUB r0, r1")
                    out.append(f"    STR r0, __idig")
                    out.append(f"    LDR r0, __idig")
                    out.append(f"    LDR r1, #0")
                    out.append(f"    CMP r0, r1")
                    out.append(f"    SETLT r0")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_ffin_{u}")
                    out.append(f"    LDR r0, #9")
                    out.append(f"    LDR r1, __idig")
                    out.append(f"    CMP r0, r1")
                    out.append(f"    SETLT r0")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JNZ in_ffin_{u}")
                    # scale /= 10.0
                    out.append(f"    LDR r0, __fscale")
                    out.append(f"    LDR r1, #0x4024000000000000")   # 10.0
                    out.append(f"    FDIV r0, r1")
                    out.append(f"    STR r0, __fscale")
                    # facc += (double)digit * scale
                    out.append(f"    LDR r0, __idig")
                    out.append(f"    I2F r0")
                    out.append(f"    LDR r1, __fscale")
                    out.append(f"    FMUL r0, r1")
                    out.append(f"    STR r0, __fdg")
                    out.append(f"    LDR r0, __facc")
                    out.append(f"    LDR r1, __fdg")
                    out.append(f"    FADD r0, r1")
                    out.append(f"    STR r0, __facc")
                    emit_read_byte("__ic")
                    out.append(f"    JMP in_frac_{u}")
                    out.append(f"in_ffin_{u}:")
                    # apply sign: if isign < 0, facc = 0.0 - facc
                    out.append(f"    LDR r0, __isign")
                    out.append(f"    LDR r1, #0")
                    out.append(f"    CMP r0, r1")
                    out.append(f"    SETLT r0")
                    out.append(f"    STR r0, __iflag")
                    out.append(f"    LDR r0, __iflag")
                    out.append(f"    JZ in_fstore_{u}")
                    out.append(f"    LDR r0, #0")
                    out.append(f"    LDR r1, __facc")
                    out.append(f"    FSUB r0, r1")
                    out.append(f"    STR r0, __facc")
                    out.append(f"in_fstore_{u}:")
                    out.append(f"    LDR r0, __facc")
                    out.append(f"    LDR r1, #0")
                    out.append(f"    ADD r0, r1")
                    out.append(f"    STR r0, {name}")
            elif fn == 'int':
                out.append(f"    ; {name} = int({arg})  (digit char -> value)")
                out.append(f"    LDR r0, {arg}")
                out.append(f"    LDR r1, #48")
                out.append(f"    SUB r0, r1")
                out.append(f"    STR r0, {name}")
            elif fn == 'bool':
                out.append(f"    ; {name} = bool({arg})  ('0' -> false, else true)")
                out.append(f"    LDR r0, {arg}")
                out.append(f"    LDR r1, #48")
                out.append(f"    CMP r0, r1")
                out.append(f"    SETNE r0")
                out.append(f"    STR r0, {name}")
            elif fn == 'float':
                out.append(f"    ; {name} = float({arg})  (digit char -> IEEE double)")
                out.append(f"    LDR r0, {arg}")
                out.append(f"    LDR r1, #48")
                out.append(f"    SUB r0, r1")
                out.append(f"    I2F r0")
                out.append(f"    STR r0, {name}")
            out.append("")

        elif kind == 'struct_field':
            # field storage is the zero-initialised data slot; no runtime code
            _, mangled, ty = decl
            out.append(f"    ; struct field {mangled}: {ty} (defaults to 0/null)")

        elif kind == 'field_decl':
            # let x = <field>;  — read a field's value into a new variable
            _, name, ty, src = decl
            out.append(f"    ; {name}: {ty} = {src}  (struct field read)")
            out.append(f"    LDR r0, {src}")
            out.append(f"    LDR r1, #0")
            out.append(f"    ADD r0, r1")
            out.append(f"    STR r0, {name}")
            out.append("")

        elif kind == 'store_value':
            # store a value into a slot (struct field write, or var = field)
            _, dest, src = decl
            if src[0] == 'imm':
                out.append(f"    ; {dest} = {src[1]}")
                out.append(f"    LDR r0, #{src[1]}")
                out.append(f"    STR r0, {dest}")
            else:  # ('slot', label)
                out.append(f"    ; {dest} = {src[1]}  (value copy)")
                out.append(f"    LDR r0, {src[1]}")
                out.append(f"    LDR r1, #0")
                out.append(f"    ADD r0, r1")
                out.append(f"    STR r0, {dest}")
            out.append("")

        elif kind == 'field_expr_store':
            # store an arbitrary expression result into a field / instance slot
            _, label, rpn, fty = decl
            out.append(f"    ; {label} = <expr>")
            scratch = emit_runtime_expr(out, rpn, fty, consts, manifest, 0)
            for s in scratch:
                scratch_slots[s] = fty
            out.append(f"    LDR r1, #0")     # normalize r0 -> transient result
            out.append(f"    ADD r0, r1")
            out.append(f"    STR r0, {label}")
            out.append("")

        elif kind == 'ref':
            # r = &target  — load the ADDRESS of a variable slot or a function
            # label (LEA), then store the reference value.
            _, name, target, tkind = decl
            if tkind == 'fn':
                out.append(f"    ; {name} = &{target}  (reference to function)")
                out.append(f"    LEA r0, fn_{target}")
            else:
                out.append(f"    ; {name} = &{target}  (reference to variable)")
                out.append(f"    LEA r0, {target}")
            out.append(f"    STR r0, {name}")
            out.append("")

        elif kind == 'comment':
            _, text = decl
            out.append(f"    ; {text}")

        elif kind == 'jumptable_dispatch':
            _, var, lo, rng, default_label, table_label = decl
            # idx = scrutinee - lo ; if (unsigned)idx > rng goto default ;
            # else jmp table[idx].  All values are non-negative here, so the
            # scrutinee's stored (zero-extended) value equals its true value.
            out.append(f"    ; dispatch: idx = {var} - {lo}; bounds 0..{rng}")
            out.append(f"    LDR r0, {var}")
            if lo != 0:
                out.append(f"    LDR r1, #{lo}")
                out.append(f"    SUB r0, r1")
            out.append(f"    LDR r1, #{rng}")
            out.append(f"    CMP r0, r1")
            out.append(f"    JA {default_label}")
            out.append(f"    LEA r1, {table_label}")
            out.append(f"    JMPIDX r1, r0")
            out.append("")

        elif kind == 'param':
            _, name, ty = decl
            out.append(f"    ; param {name}: {ty}  (set by caller before CALL)")

        elif kind == 'return_void':
            _, fn = decl
            tgt = 'main_ret' if fn == 'main' else f'{fn}_ret'
            out.append(f"    ; return")
            out.append(f"    JMP {tgt}")
            out.append("")

        elif kind == 'return_expr':
            _, fn, ty, rpn = decl
            out.append(f"    ; return <expr>  ({ty}) — result left in r0")
            scratch = emit_runtime_expr(out, rpn, ty, consts, manifest, 0)
            for s in scratch:
                scratch_slots[s] = ty
            out.append(f"    JMP {fn}_ret")
            out.append("")

        elif kind == 'callseq':
            _, callee, arg_specs, dest, ret = decl
            is_heap = functions[callee].get('kind') == 'def'
            # Save/restore the callee's frame only when it can actually be
            # re-entered (it is a `fn` AND it is recursive). Otherwise the frame
            # is never live across the call, so the push/pop is skipped.
            needs_save = (not is_heap) and (callee in recursive_fns)
            fslots = frame_slots.get(callee, []) if needs_save else []
            if is_heap:
                out.append(f"    ; call {callee}(...)  -> {ret}"
                           f"   [heap frame: persistent, no save/restore]")
            elif needs_save:
                out.append(f"    ; call {callee}(...)  -> {ret}"
                           f"   [stack frame: save/restore {len(fslots)} slot(s)]")
            else:
                out.append(f"    ; call {callee}(...)  -> {ret}"
                           f"   [non-recursive fn: frame save/restore elided]")
            # 1) For a STACK (`fn`) callee, the caller pushes the callee's frame
            #    window onto the hardware stack, preserving the current
            #    activation across the call (this is what makes recursion /
            #    re-entrancy correct). A HEAP (`def`) callee keeps a single
            #    persistent frame in static/heap storage — nothing is saved or
            #    restored, so it is intentionally not re-entrant.
            for s in fslots:
                out.append(f"    LDR r0, {s}")
                out.append(f"    PUSH r0")
            # 2) evaluate each argument into the callee's parameter slot
            for pslot, pty, rpn in arg_specs:
                scratch = emit_runtime_expr(out, rpn, pty, consts, manifest, 0)
                for s in scratch:
                    scratch_slots[s] = pty
                out.append(f"    LDR r1, #0")
                out.append(f"    ADD r0, r1")
                out.append(f"    STR r0, {pslot}")
            # 3) transfer control
            out.append(f"    CALL fn_{callee}")
            # 4) stash the result, restore the frame window (LIFO), recover it.
            #    A `def` has no saved window, so steps 1 and the restore below
            #    are both empty for it.
            if dest is not None:
                needs_callret[0] = True
                out.append(f"    STR r0, __callret")   # result returned in r0
            for s in reversed(fslots):
                out.append(f"    POP r0")
                out.append(f"    STR r0, {s}")
            if dest is not None:
                out.append(f"    LDR r0, __callret")
                out.append(f"    LDR r1, #0")          # normalize -> transient
                out.append(f"    ADD r0, r1")
                out.append(f"    STR r0, {dest}")
            out.append("")

    # ---- per-function stack frame -----------------------------------------
    # A `fn` is a STACK function: each activation's parameter+local window is
    # saved onto the hardware stack by the caller around every CALL and
    # restored on return. The static slots therefore behave as a fixed
    # register window that is spilled per call, which makes fn re-entrant and
    # recursion-safe — the live frame for every activation lives on the stack.
    # frame_slots[fn] is that window: the ordered set of data slots the fn owns
    # (its params and locals). Struct fields and consts are global shared
    # state, never part of a frame.
    def compute_frame_slots(fname):
        slots, seen = [], set()
        def add(s):
            # Each variable (including a relocated one) owns its storage, so it
            # is part of the frame window and is saved/restored normally.
            # A memory-mapped register is a fixed hardware address with no storage
            # of its own, so it is global state and never part of a frame.
            if s in mmio:
                return
            if s not in seen:
                seen.add(s); slots.append(s)
        for pname, _pty in functions[fname]['params']:
            add(pname)
        for d in fn_bodies[fname]:
            k = d[0]
            if k == 'scalar':       add(d[1])
            elif k == 'decl_expr':  add(d[1])
            elif k == 'builtin':    add(d[1])
            elif k == 'field_decl': add(d[1])
            elif k == 'ref':        add(d[1])
            elif k == 'param':      add(d[1])
            elif k == 'array':
                _, nm, _ty, ms, *_ = d
                for i in range(ms):
                    add(f"{nm}_{i}")
            elif k == 'callseq':
                dst = d[3]
                if dst is not None:
                    add(dst)
        return slots
    frame_slots = {fn: compute_frame_slots(fn) for fn in fn_order}

    # ---- which functions are actually recursive? -------------------------
    # A `fn`'s per-call frame save/restore exists ONLY to make recursion /
    # re-entrancy correct: it protects an outer activation's frame while an
    # inner activation of the SAME function runs. If a function can never be
    # re-entered while already active (it is not on a cycle in the call graph),
    # that save/restore is pure overhead and is elided. This keeps `fn`
    # semantics intact (a non-recursive fn behaves identically either way) while
    # removing the push/pop bloat — no need to reach for `def`.
    _callees = {}
    for _fn in fn_order:
        body = declarations if _fn == 'main' else fn_bodies.get(_fn, [])
        cs = set()
        for d in body:
            if d[0] == 'callseq':
                cs.add(d[1])
        _callees[_fn] = cs

    def _reaches_self(start):
        seen, stack = set(), list(_callees.get(start, ()))
        while stack:
            x = stack.pop()
            if x == start:
                return True
            if x not in seen:
                seen.add(x)
                stack.extend(_callees.get(x, ()))
        return False
    recursive_fns = {fn for fn in fn_order if _reaches_self(fn)}

    # ---- program layout: main (entry) first, then each function body ----
    out.append("main:")
    for decl in declarations:
        emit_one(decl)
    out.append("main_ret:")
    out.append("    SYS r0, r1")
    for fname in fn_order:
        _kw = functions[fname].get('kind', 'fn')
        if _kw == 'isr':
            # interrupt handler: label is the bare 'isr_<VECTOR>' (the vector
            # table jumps straight here), body runs, then RETI restores SREG and
            # re-enables interrupts. Never called via CALL, so no _ret/RET.
            vec = functions[fname]['vector']
            out.append("")
            out.append(f"; ---- isr {vec}  (interrupt handler) ----")
            out.append(f"{fname}:")
            for decl in fn_bodies[fname]:
                emit_one(decl)
            out.append("    RETI")
            continue
        sig_parts = []
        _recv = functions[fname].get('self_recv')
        if _recv is not None:
            if _recv[0] == 'single':
                sig_parts.append(f"self: {_recv[1]}")
            elif _recv[0] == 'multi':
                sig_parts.append("self: (" + ", ".join(_recv[1]) + ")")
            else:
                sig_parts.append("self")
        sig_parts += [f"{p}: {t}" for p, t in functions[fname]['params']]
        sig = ", ".join(sig_parts)
        _frame = "heap, persistent" if _kw == 'def' else "stack, re-entrant"
        out.append("")
        out.append(f"; ---- {_kw} {fname}({sig}) -> {functions[fname]['ret']}"
                   f"   [{_frame}] ----")
        out.append(f"fn_{fname}:")
        for decl in fn_bodies[fname]:
            emit_one(decl)
        out.append(f"{fname}_ret:")
        out.append("    RET")
    if needs_itoa[0]:
        # ---- shared signed-itoa routine (replaces 50+ inlined copies) ----
        # entry: value already stored in __pn; writes its decimal form to
        # stdout one byte at a time, then RETs. Clobbers r0/r1/r3/r4 and the
        # itoa scratch slots; PUSH/POP are balanced so the stack (and the
        # return address) is intact at RET.
        out.append("")
        out.append("; ---- shared signed-itoa: __pn -> decimal on stdout ----")
        out.append("fn___itoa:")
        out.append("    LDR r0, __pn")
        out.append("    LDR r1, #0")
        out.append("    CMP r0, r1")
        out.append("    SETLT r0")
        out.append("    STR r0, __sign")
        out.append("    LDR r0, __sign")
        out.append("    JZ itoa_pos")
        out.append("    LDR r0, #45")            # '-'
        out.append("    STR r0, __char1")
        out.append("    LEA r4, __char1")
        out.append("    LDR r3, #1")
        out.append("    WRITE")
        out.append("    LDR r0, #0")
        out.append("    LDR r1, __pn")
        out.append("    SUB r0, r1")
        out.append("    STR r0, __pn")
        out.append("itoa_pos:")
        out.append("    LDR r0, __pn")
        out.append("    JNZ itoa_conv")
        out.append("    LDR r0, #48")            # '0'
        out.append("    STR r0, __char1")
        out.append("    LEA r4, __char1")
        out.append("    LDR r3, #1")
        out.append("    WRITE")
        out.append("    JMP itoa_done")
        out.append("itoa_conv:")
        out.append("    LDR r0, #0")
        out.append("    STR r0, __cnt")
        out.append("itoa_loop:")
        out.append("    LDR r0, __pn")
        out.append("    LDR r1, #10")
        out.append("    DIV r0, r1")
        out.append("    STR r0, __q")
        out.append("    LDR r0, __q")
        out.append("    LDR r1, #10")
        out.append("    MUL r0, r1")
        out.append("    STR r0, __tmp")
        out.append("    LDR r0, __pn")
        out.append("    LDR r1, __tmp")
        out.append("    SUB r0, r1")
        out.append("    LDR r1, #48")
        out.append("    ADD r0, r1")
        out.append("    PUSH r0")
        out.append("    LDR r0, __cnt")
        out.append("    LDR r1, #1")
        out.append("    ADD r0, r1")
        out.append("    STR r0, __cnt")
        out.append("    LDR r0, __q")
        out.append("    STR r0, __pn")
        out.append("    LDR r0, __pn")
        out.append("    JNZ itoa_loop")
        out.append("itoa_print:")
        out.append("    POP r0")
        out.append("    STR r0, __char1")
        out.append("    LEA r4, __char1")
        out.append("    LDR r3, #1")
        out.append("    WRITE")
        out.append("    LDR r0, __cnt")
        out.append("    LDR r1, #1")
        out.append("    SUB r0, r1")
        out.append("    STR r0, __cnt")
        out.append("    LDR r0, __cnt")
        out.append("    JNZ itoa_print")
        out.append("itoa_done:")
        out.append("    RET")

    if needs_float_io[0]:
        # ---- shared dtoa routine (replaces the per-float inlined copy) ----
        # entry: float bits already in __fp; writes 'I.FFF' to stdout, then RETs.
        out.append("")
        out.append("; ---- shared dtoa: __fp -> decimal on stdout ----")
        out.append("fn___dtoa:")
        # sign bit: raw bits < 0 (signed) => negative
        out.append(f"    LDR r0, __fp")
        out.append(f"    LDR r1, #0")
        out.append(f"    CMP r0, r1")
        out.append(f"    SETLT r0")
        out.append(f"    STR r0, __fdflag")
        out.append(f"    LDR r0, __fdflag")
        out.append(f"    JZ fpos")
        out.append(f"    LDR r0, #45")          # '-'
        out.append(f"    STR r0, __char1")
        out.append(f"    LEA r4, __char1")
        out.append(f"    LDR r3, #1")
        out.append(f"    WRITE")
        out.append(f"    LDR r0, #0")           # fp = 0.0 - fp
        out.append(f"    LDR r1, __fp")
        out.append(f"    FSUB r0, r1")
        out.append(f"    STR r0, __fp")
        out.append(f"fpos:")
        # integer part = trunc(fp)
        out.append(f"    LDR r0, __fp")
        out.append(f"    F2I r0")
        out.append(f"    STR r0, __fip")
        # --- itoa on the (non-negative) integer part ---
        out.append(f"    LDR r0, __fip")
        out.append(f"    STR r0, __pn")
        out.append(f"    LDR r0, __pn")
        out.append(f"    JNZ fconv")
        out.append(f"    LDR r0, #48")          # "0"
        out.append(f"    STR r0, __char1")
        out.append(f"    LEA r4, __char1")
        out.append(f"    LDR r3, #1")
        out.append(f"    WRITE")
        out.append(f"    JMP fitoa_done")
        out.append(f"fconv:")
        out.append(f"    LDR r0, #0")
        out.append(f"    STR r0, __cnt")
        out.append(f"floop:")
        out.append(f"    LDR r0, __pn")
        out.append(f"    LDR r1, #10")
        out.append(f"    DIV r0, r1")
        out.append(f"    STR r0, __q")
        out.append(f"    LDR r0, __q")
        out.append(f"    LDR r1, #10")
        out.append(f"    MUL r0, r1")
        out.append(f"    STR r0, __tmp")
        out.append(f"    LDR r0, __pn")
        out.append(f"    LDR r1, __tmp")
        out.append(f"    SUB r0, r1")
        out.append(f"    LDR r1, #48")
        out.append(f"    ADD r0, r1")
        out.append(f"    PUSH r0")
        out.append(f"    LDR r0, __cnt")
        out.append(f"    LDR r1, #1")
        out.append(f"    ADD r0, r1")
        out.append(f"    STR r0, __cnt")
        out.append(f"    LDR r0, __q")
        out.append(f"    STR r0, __pn")
        out.append(f"    LDR r0, __pn")
        out.append(f"    JNZ floop")
        out.append(f"fprint:")
        out.append(f"    POP r0")
        out.append(f"    STR r0, __char1")
        out.append(f"    LEA r4, __char1")
        out.append(f"    LDR r3, #1")
        out.append(f"    WRITE")
        out.append(f"    LDR r0, __cnt")
        out.append(f"    LDR r1, #1")
        out.append(f"    SUB r0, r1")
        out.append(f"    STR r0, __cnt")
        out.append(f"    LDR r0, __cnt")
        out.append(f"    JNZ fprint")
        out.append(f"fitoa_done:")
        # '.'
        out.append(f"    LDR r0, #46")
        out.append(f"    STR r0, __char1")
        out.append(f"    LEA r4, __char1")
        out.append(f"    LDR r3, #1")
        out.append(f"    WRITE")
        # frac = fp - (double)fip
        out.append(f"    LDR r0, __fip")
        out.append(f"    I2F r0")
        out.append(f"    STR r0, __fipf")
        out.append(f"    LDR r0, __fp")
        out.append(f"    LDR r1, __fipf")
        out.append(f"    FSUB r0, r1")
        out.append(f"    STR r0, __ffrac")
        # fractional part as a rounded 6-digit integer:
        #   fracint = trunc(ffrac * 1e6 + 0.5), capped to 999999
        out.append(f"    LDR r0, __ffrac")
        out.append(f"    LDR r1, #0x412E848000000000")   # 1000000.0
        out.append(f"    FMUL r0, r1")
        out.append(f"    LDR r1, #0x3FE0000000000000")   # 0.5 (round)
        out.append(f"    FADD r0, r1")
        out.append(f"    F2I r0")
        out.append(f"    STR r0, __fracint")
        out.append(f"    LDR r0, #999999")               # cap (avoid 7 digits)
        out.append(f"    LDR r1, __fracint")
        out.append(f"    CMP r0, r1")
        out.append(f"    SETLT r0")
        out.append(f"    STR r0, __iflag")
        out.append(f"    LDR r0, __iflag")
        out.append(f"    JZ fcap_ok")
        out.append(f"    LDR r0, #999999")
        out.append(f"    STR r0, __fracint")
        out.append(f"fcap_ok:")
        # extract digits least-significant first into __fv5..__fv0
        out.append(f"    LDR r0, __fracint")
        out.append(f"    STR r0, __pn")
        for kdig in (5, 4, 3, 2, 1, 0):
            out.append(f"    ; fractional digit {kdig}")
            out.append(f"    LDR r0, __pn")
            out.append(f"    LDR r1, #10")
            out.append(f"    DIV r0, r1")
            out.append(f"    STR r0, __q")
            out.append(f"    LDR r0, __q")
            out.append(f"    LDR r1, #10")
            out.append(f"    MUL r0, r1")
            out.append(f"    STR r0, __tmp")
            out.append(f"    LDR r0, __pn")
            out.append(f"    LDR r1, __tmp")
            out.append(f"    SUB r0, r1")
            out.append(f"    STR r0, __fv{kdig}")
            out.append(f"    LDR r0, __fv{kdig}")
            out.append(f"    LDR r1, #48")
            out.append(f"    ADD r0, r1")
            out.append(f"    STR r0, __fc{kdig}")
            out.append(f"    LDR r0, __q")
            out.append(f"    STR r0, __pn")
        # tail-nonzero flags: nz5..nz1 (monotone from the end)
        out.append(f"    LDR r0, __fv5")
        out.append(f"    LDR r1, #0")
        out.append(f"    CMP r0, r1")
        out.append(f"    SETNE r0")
        out.append(f"    STR r0, __fnz5")
        for kdig in (4, 3, 2, 1):
            out.append(f"    LDR r0, __fv{kdig}")
            out.append(f"    LDR r1, #0")
            out.append(f"    CMP r0, r1")
            out.append(f"    SETNE r0")
            out.append(f"    LDR r1, __fnz{kdig+1}")
            out.append(f"    SETOR r0, r1")
            out.append(f"    STR r0, __fnz{kdig}")
        # always print digit 0
        out.append(f"    LDR r0, __fc0")
        out.append(f"    STR r0, __char1")
        out.append(f"    LEA r4, __char1")
        out.append(f"    LDR r3, #1")
        out.append(f"    WRITE")
        # digits 1..5 gated on their tail-nonzero flag (stop at first zero)
        for kdig in range(1, 6):
            out.append(f"    LDR r0, __fnz{kdig}")
            out.append(f"    JZ fdend")
            out.append(f"    LDR r0, __fc{kdig}")
            out.append(f"    STR r0, __char1")
            out.append(f"    LEA r4, __char1")
            out.append(f"    LDR r3, #1")
            out.append(f"    WRITE")
        out.append(f"fdend:")
        out.append("    RET")


    out.append("")
    out.append("; Data section")
    out.append("; NOTE: every slot is a full .qword (8 bytes) so that 64-bit")
    out.append("; LDR/STR is always safe. The declared type still governs bounds")
    out.append("; checking and semantics in the compiler; physical storage is uniform.")

    # ===================================================================
    # Byte-accurate data arena
    # Every variable / const / struct-field occupies a UNIQUE byte range.
    # The default layout packs them sequentially (8 bytes each; an array is
    # N*8). Manual relocations ('&a = <addr-expr>;') then override a variable's
    # offset to a computed byte address, and the manifest rejects any range that
    # overlaps another variable — so two variables can never share memory.
    # ===================================================================
    arena = []            # ordered: {'base': name, 'size': bytes, 'labels': [...]}
    _seen_bases = set()
    # typemap: scalar slot name -> nexa type. Emitted as a sidecar so the backend
    # can pack each slot to its true width (RAM floor) with width-correct,
    # sign/zero-extending loads/stores. Arrays, refs and pointers are omitted on
    # purpose -> the backend leaves them a full 8-byte qword (safe default).
    typemap = {}
    def note_type(name, ty):
        if name not in mmio and ty in ALL_TYPES:
            typemap[name] = ty
    def add_item(base, labels, size):
        if base in mmio:
            return          # memory-mapped register: fixed address, no storage
        if base in _seen_bases:
            return
        _seen_bases.add(base)
        arena.append({'base': base, 'size': size, 'labels': labels})

    for name, ty, raw, int_value in global_consts:
        add_item(name, [name], 8)
        note_type(name, ty)

    _all_body_decls = list(declarations)
    for _fn in fn_order:
        _all_body_decls += fn_bodies[_fn]

    for decl in _all_body_decls:
        kind = decl[0]
        if kind in ('scalar', 'decl_expr'):
            add_item(decl[1], [decl[1]], 8)
            note_type(decl[1], decl[2])
        elif kind == 'load_array':
            # ('load_array', dest, ty, arr, idx_kind, idx, is_decl, is_mut)
            if decl[6]:                       # is_decl -> new scalar slot
                add_item(decl[1], [decl[1]], 8)
                note_type(decl[1], decl[2])
        elif kind == 'array':
            _, name, ty, max_size, _, _ = decl
            add_item(name, [f"{name}_{i}" for i in range(max_size)], 8 * max_size)
            # arrays stay 8-byte stride -> not recorded (no packing)
        elif kind == 'builtin':
            add_item(decl[1], [decl[1]], 8)
            note_type(decl[1], decl[2])
        elif kind == 'struct_field':
            add_item(decl[1], [decl[1]], 8)
            note_type(decl[1], decl[2])
        elif kind == 'field_decl':
            add_item(decl[1], [decl[1]], 8)
            note_type(decl[1], decl[2])
        elif kind == 'ref':
            add_item(decl[1], [decl[1]], 8)
            # ref/pointer slot holds a 64-bit address -> leave at 8
        elif kind == 'param':
            add_item(decl[1], [decl[1]], 8)
            note_type(decl[1], decl[2])
        elif kind == 'callseq':
            dest = decl[3]
            if dest is not None:
                add_item(dest, [dest], 8)
                if len(decl) > 4:
                    note_type(dest, decl[4])

    for slot, _ty in instance_field_ty.items():
        add_item(slot, [slot], 8)
        note_type(slot, _ty)

    # default packed offsets (this is the layout when nothing is relocated)
    offset = {}
    _cursor = 0
    for it in arena:
        offset[it['base']] = _cursor
        _cursor += it['size']

    # resolve relocations in source order: fold the address expression (each
    # '&var' becomes that variable's current byte offset), then bounds- and
    # overlap-check the target range against every OTHER variable.
    def _fold_addr(expr, lineno):
        def repl(mm):
            nm = mm.group(1)
            if nm not in offset:
                raise ValueError(
                    f"Line {lineno}: '&{nm}' has no storage to take the address of")
            return str(offset[nm])
        subbed = ADDR_REF_RE.sub(repl, expr)
        rpn = expr_to_rpn(expr_tokenize(subbed, lineno), lineno)
        return int(fold_constant_expr(rpn, consts, lineno))

    for (a, rhs, lineno) in relocations:
        target = _fold_addr(rhs, lineno)
        if target < 0:
            raise ValueError(
                f"Line {lineno}: relocation address for '{a}' is negative ({target})")
        lo, hi = target, target + 8
        for it in arena:
            if it['base'] == a:
                continue
            b_lo = offset[it['base']]
            b_hi = b_lo + it['size']
            if lo < b_hi and b_lo < hi:
                raise ValueError(
                    f"Line {lineno}: cannot relocate '{a}' to bytes [{lo}..{hi-1}] — "
                    f"that range overlaps '{it['base']}' at [{b_lo}..{b_hi-1}]. "
                    f"Every variable must occupy a unique location.")
        offset[a] = target

    # emit the arena in ascending address order, filling gaps with .space
    out.append("; --- variable arena (byte-accurate; .space marks free gaps) ---")
    _items_sorted = sorted(arena, key=lambda it: offset[it['base']])
    _cur = 0
    _pad = 0
    for it in _items_sorted:
        base_off = offset[it['base']]
        if base_off > _cur:
            out.append(f"__pad_{_pad}: .space {base_off - _cur}")
            _pad += 1
        for lab in it['labels']:
            out.append(f"{lab}: .qword 0")
        _cur = base_off + it['size']

    # Scratch slots used by runtime expressions
    if scratch_slots:
        out.append("; expression scratch")
        for slot, ty in scratch_slots.items():
            out.append(f"{slot}: .qword 0")

    # call-result stash: holds a function's return value while the caller
    # restores the callee's frame window from the stack after the CALL.
    if needs_callret[0]:
        out.append("; function call-result stash")
        out.append("__callret: .qword 0")

    # print() workspace: signed-itoa registers + the 1-byte write buffer.
    if needs_print_scratch[0]:
        out.append("; print() workspace (itoa + 1-byte write buffer)")
        for slot in ("__pn", "__q", "__tmp", "__cnt", "__sign", "__char1"):
            out.append(f"{slot}: .qword 0")

    # float / token-input / dtoa workspace
    if needs_float_io[0]:
        out.append("; float + input(atoi/atof) + dtoa workspace")
        fio = ["__ic", "__idig", "__iflag", "__isign", "__iacc",
               "__fp", "__fip", "__fipf", "__ffrac", "__fscaled",
               "__fdgf", "__fscale", "__fdg", "__facc", "__fdflag", "__fracint"]
        fio += [f"__fv{i}" for i in range(6)]
        fio += [f"__fc{i}" for i in range(6)]
        fio += [f"__fnz{i}" for i in range(1, 6)]
        for slot in fio:
            out.append(f"{slot}: .qword 0")

    # String literal pool for print(): raw byte buffers.
    if string_pool:
        out.append("; string literals (print)")
        for label, byte_list in string_pool:
            out.append(f"{label}: .bytes " + " ".join(str(b) for b in byte_list))

    # Match jump tables: each is an array of code-label addresses. The backend
    # resolves the labels to absolute addresses when it lays out the binary.
    if jump_tables:
        out.append("; match jump tables (each entry is a resolved code address)")
        for table_label, entries in jump_tables:
            out.append(f"{table_label}: .qaddrs " + " ".join(entries))

    # Memory-mapped I/O registers: absolute data-space addresses, no storage.
    # The backend points the label straight at the address; LDR/STR become a
    # load/store of that hardware register.
    if mmio:
        out.append("; memory-mapped I/O registers (absolute addresses)")
        for name, addr in mmio.items():
            out.append(f"{name}: .at 0x{addr:X}")

    # Scalar type sidecar — the backend packs each listed slot to its true width
    # with sign/zero-correct loads. Backends that don't pack ignore '.typemap'.
    if typemap:
        out.append("; scalar type map (width-aware packing; ignored by non-packing backends)")
        out.append("__typemap: .typemap " +
                   " ".join(f"{n}={t}" for n, t in typemap.items()))

    # Volatile sidecar — slots the optimizer must treat as opaque (always
    # re-load on read, always store-through on write, never register-promote).
    # mmio registers are volatile by definition (a hardware register can change
    # underneath the program), so they are included automatically.
    volatile_slots = set(volatile_names) | set(mmio)
    if volatile_slots:
        out.append("; volatile slots (never cached in a register; ISR/mmio safe)")
        out.append("__volatile: .volatile " + " ".join(sorted(volatile_slots)))

    return "\n".join(out)


def _strip_main_block(lines):
    """Remove a top-level `fn main() {...}` block from imported source. A
    library is imported for its structs/functions/consts, never its entry
    point, so any main it carries is dropped. Brace counting ignores braces
    that appear inside string/char literals (e.g. print("{x}"))."""
    out = []
    i, n = 0, len(lines)
    while i < n:
        if FN_MAIN_RE.match(lines[i].strip()):
            depth = 0
            while i < n:
                code = re.sub(r'"[^"]*"', '', lines[i])
                code = re.sub(r"'.'", '', code)
                depth += code.count('{') - code.count('}')
                i += 1
                if depth == 0:
                    break
            out.append('// [import: dropped this file\'s fn main()]')
            continue
        out.append(lines[i])
        i += 1
    return out


def _resolve_import_path(raw_path, base_dir):
    """Resolve an import path the way the example expects: '~' expands to the
    home directory, absolute paths are used directly, and a relative path is
    taken relative to the importing file's directory (then the cwd)."""
    p = os.path.expanduser(raw_path)
    if os.path.isabs(p):
        candidates = [p]
    else:
        candidates = [os.path.join(base_dir, p), p]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.realpath(c)
    return None


def preprocess_imports(src, base_dir, seen):
    """Expand `import "path";` lines by splicing in the referenced file's
    source (recursively). `seen` holds the realpaths already included so a
    file is included at most once — this makes duplicate and circular imports
    safe (the second include becomes a no-op comment)."""
    out_lines = []
    for raw in src.splitlines():
        stripped = raw.strip()
        if stripped.startswith("//"):
            out_lines.append(raw)
            continue
        m = IMPORT_RE.match(stripped)
        if not m:
            if stripped.startswith("import ") or stripped == "import":
                raise ValueError(
                    f'malformed import — expected: import "path/to/file.nexa";')
            out_lines.append(raw)
            continue
        raw_path = m.group(1)
        if not raw_path.endswith(".nexa"):
            raise ValueError(
                f'import "{raw_path}" must reference a .nexa file '
                f'(imports only accept file_name.nexa)')
        resolved = _resolve_import_path(raw_path, base_dir)
        if resolved is None:
            raise ValueError(f'cannot find import "{raw_path}" '
                             f'(searched relative to {base_dir} and home)')
        if resolved in seen:
            out_lines.append(f'// [import skipped: "{raw_path}" already included]')
            continue
        seen.add(resolved)
        try:
            with open(resolved) as f:
                imported_src = f.read()
        except OSError as e:
            raise ValueError(f'cannot read import "{raw_path}": {e}')
        imported_src = "\n".join(_strip_main_block(imported_src.splitlines()))
        # the imported file's own imports resolve relative to ITS directory
        expanded = preprocess_imports(imported_src, os.path.dirname(resolved), seen)
        out_lines.append(f'// ===== begin import "{raw_path}" =====')
        out_lines.append(expanded)
        out_lines.append(f'// ===== end import "{raw_path}" =====')
    return "\n".join(out_lines)


def main():
    if len(sys.argv) != 2:
        print("usage: python3 nexatoaxiom.py <file.nexa>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        src = f.read()

    base_dir = os.path.dirname(os.path.realpath(path))
    try:
        # expand imports first, then compile the combined source
        src = preprocess_imports(src, base_dir, {os.path.realpath(path)})
        axm = compile_t1_to_axiom(src)
    except (ValueError, OSError) as err:
        # Frontend diagnostics (imports, immutability, type, scope, self, etc.)
        # print as a clean one-line compile error rather than a Python traceback.
        print(f"nexa: compile error: {err}", file=sys.stderr)
        sys.exit(1)

    out_path = path.rsplit(".", 1)[0] + ".axm"
    with open(out_path, "w") as f:
        f.write(axm)

    print(f"Compiled {path} -> {out_path}")

if __name__ == "__main__":
    main()

'''
Requirements: 1) Lower level than C/zig especially rust 2) New compilation target easier than mlir 3) memory safe by default without any unsafe scopes
4) No runtimes or garbage collection, compilation checked 5) Nested structs 6) self variable passing between functions and structs 7) only global functions
8) stack vs heap functions using fn and def 9) import something.nexa; for importing files 10) Strictly procedural and structural paradigms
11) mutable vs immutable declarations 12) Slices for vectors 13) Error handling model using abort keyword to exit the program

Data Types: i8, i16, i32, i64 / f8, f16, f32, f64 / c8, c16, c32, c64 / arrays / let / mut / const / void / bool (true/false) /
struct / fn / def / match() / if / ifelse / else / for(){} / while() / loop {} / self / Vector[] / ptr / abort / import <>;

Comparison: == means if equals to, && means and, || means or, != means does not equal to

Bitwise Operations: &= and op, |= or op, ^= xor op, << take number on left shift left so many units by whats on the right,
>> take number on right shift right so many units by whats on the right.



Completed:
    Data Types/Reserved Words: i8, i16, i32, i64 / f8, f16, f32, f64 / c8, c16, c32, c64 / arrays / mut / const / null / bool (true/false) /
    Math / Comparison / Step Ops / Bitwise Ops / if / ifelse / else / match() / break / continue / while() / loop {} / for() {} /
    abort / syscalls / print / input / input converts / struct / nested structs / :: / . / print(float) / input(float) / convert(types) /
    fn / return / void / println(); / self / def / import ;

    Math: +, -, *, /, &, =
    
    Comparison: ==, &&, ||, ^^, (), <=, >=, <, >

    Step Ops: ++, --, +=, -=, *=, /=

    Bitwise Operations: <<, >>

    Memory Safety: working manifest safety and enforcement at compile time

    Notes: Work on building nexustoaxiom.py which is nexa but without import and has an embedded file read function for future high level language compiling

In Development:
    Data Types and Progress:

    Math:

    Comparison:

    Step Ops:

    Bitwise Operations:

    Memory Safety:


-Command history (Obsolete from mlir to llvm pipeline): (.nexa -> uppder.mlir -> lower.mlir -> llvm.ll -> .o -> exe)
python3 nexa_compiler.py test.nexa
Wrote MLIR to test.mlir
mlir-opt test.mlir \
        --convert-to-llvm \
        --reconcile-unrealized-casts \
        -o test.lowered.mlir
mlir-translate test.lowered.mlir --mlir-to-llvmir -o test.ll
llc test.ll -filetype=obj -o test.o
clang test.o -o test


-Updated Command History:
python3 nexatoaxiom.py test.nexa
Compiled test.nexa -> test.axm

python3 axiom_x86.py test.axm
Assembled test.axm -> a.out
  text: 3900 bytes at 0x4000B0
  data: 800 bytes at 0x401000
  total ELF: 4896 bytes
Run:  chmod +x a.out && ./a.out ; echo "exit: $?"

chmod +x a.out
./a.out
echo $?
'''