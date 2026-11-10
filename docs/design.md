# Design

## Targets

| Item | Value |
|---|---|
| Board | KV260 (Zynq UltraScale+ XCK26) |
| Models | SmolVLA (`lerobot/smolvla_base`) with 2 cameras; SparseDrive-S with 6 cameras |

XCK26 resources (K26 SOM data sheet): 1,248 DSP48E2, 117,120 LUTs, 144 BRAM36, 64 URAM.

The forward pass of both models runs on the NPU.

## Hardware description language

The RTL is written in SystemVerilog.

## Architecture

| Block | Function |
|---|---|
| Sequencer | Fetches the instruction stream from DDR, issues instructions to the engines, synchronizes with the PS, holds the control registers |
| DMA | Transfers between DDR and the on-chip memories, with format conversion and gather-accumulate |
| GEMM | Block dot-product array: `C[m][n] (+)= Σ_k A[m][k] · W[n][k]` |
| VEC | Programmable vector unit, 8 FP32 lanes with an integer path |

The PL clock is 250 MHz (pl_clk0, IOPLL 2,000 MHz / 8).

### GEMM array

- 2 × 8 BDU2 units cover 4 tokens × 8 columns: 1,024 MACs per cycle on 512 DSP48E2.
- Each cycle reads one ABUF word (4 tokens × 1 block) and one WBUF word (8 columns × 1 block) and computes 32 block dot products. Activations are shared along a row of the array, weights along a column.
- A BDU2 computes 2 tokens × 1 column on a cascade of 32 DSP48E2. Each DSP computes `(a1 · 2^18 + a0) × w` with the pre-adder inputs `A = a1 << 18`, `D = a0` and `B = w`. The cascade output is `P = S1 · 2^18 + S0`; `S0 = sext(P[17:0])` and `S1 = (P − S0) >> 18`.
- The inputs of DSP k arrive k cycles late. Skew registers are kept per array row (activations) and per column (weights) and shared by the BDU2s of that row or column.
- Pipeline: DSP 4 + cascade 31 + separation 1 + integer-to-FP32 conversion 2 + FP32 addition 6 = 44 cycles to the adder output. Throughput is one block per cycle.
- Each BDU2 has 8 FP32 accumulators. The loop order is fixed in hardware: `for g (8-column group): for qg (32 tokens): for kb: for i (accumulator 0..7)`. At the end of the K loop each 4-token × 8-column tile is written to FBUF as one word. With `acc = 1` the accumulators start from the old FBUF value.

### Vector unit

- An FBUF word (4 tokens × 8 columns) passes through the 8 lanes in 4 cycles, one token per cycle.
- A VEC instruction starts a VEC program in CBUF. The library programs are in CBUF [0, 256), present from configuration; other programs are loaded into CBUF by DMA.
- Each lane has two FP32 adders, one FP32 multiplier and one 24 × 16-bit multiplier for table interpolation.
- Row scalars RS0 and RS1 (one per token of a quad) are computed by a scalar unit from the reductions of a pass. Reductions and row scalars are held in 2 row contexts, so the scalar unit works on one context while the lanes process passes of the other.
- With q, the final pass is quantized to MXINT8 per 4 FBUF words and written to ABUF instead of FBUF.

### DMA engine

- One AXI4 master, 128 bits, INCR bursts of at most 16 beats that do not cross a 4 KB boundary.
- One instruction at a time.
- Index lists are read up to 48 beats ahead of their use.
- Stores issue up to four write bursts ahead of their data beats.
- The gather-accumulate datapath has 16 lanes.
- Gather-accumulate requests the coefficients of a slot group when an entry that is not null needs them and none are pending, and requests the next group's coefficients when it starts to use a group's; null entries do not wait for coefficients, and two null entries of one list beat are consumed in one cycle.

### On-chip memories

| Name | Implementation | Word | Depth | Contents |
|---|---|---|---|---|
| ABUF | URAM | 1,056 bits: 4 tokens × 32 INT8, then 4 E8M0 scales | 8,192 | MXINT8 activations (GEMM A operand) |
| WBUF | BRAM | 1,152 bits: 8 columns × 32 × 4 bits, then 8 E8M0 scales | 2,048 | GEMM W operand |
| FBUF | URAM | 1,024 bits: 32 FP32, element `t·8 + c` for token t and column c | 4,096 | GEMM output and accumulation, VEC input and output |
| PBUF | BRAM | 1,024 bits: 32 FP32 | 512 | Normalization γ and β, biases, RoPE cos and sin |
| TBUF | BRAM, one copy per lane pair (4 copies) | 64 bits: `{s[31:0], v[31:0]}` | 1,024 | Nonlinear-function tables |
| CBUF | BRAM | 128 bits | 1,024 | VEC programs: [0, 256) library (initial contents), [256, 1024) loaded by DMA; DMA can write every word |
| IBUF | FIFO | 128 bits | 64 | Instruction prefetch |

The VEC scalar register file has 32 registers of 32 bits. The row mask RM has 512 bits, one per quad index of a PASS.

## Number formats

| Name | Element | Block | Scale | Use |
|---|---|---|---|---|
| MXFP4 | E2M1 (4 bits) | 32 along K | E8M0 (8 bits) | GEMM W operand, mode FP4 |
| MXINT8 | INT8 (Q1.6) | 32 along K | E8M0 | GEMM A operand (ABUF) |
| W8 | MXINT8 split into a high-nibble plane and a low-nibble plane | 32 | E8M0 | GEMM W operand, modes W8_HI and W8_LO |
| FP32 | IEEE 754 single precision | — | — | GEMM accumulation and output, VEC, gather-accumulate (FBUF, PBUF) |

An E8M0 scale byte holds `e + 127`.

Encoding:

- `E` is the exponent of the largest magnitude in the block, read from its FP32 exponent field (equal to `floor(log2(amax))`). For an all-zero block, `e = 0`.
- MXINT8: `e = E`; element `q = RNE(x × 2^(6 − e))`, saturated to [−128, 127]; value `q × 2^(e − 6)`.
- MXFP4: `e = E − 2`; element rounded to the nearest E2M1 value {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, ties to even, saturating at ±6; value `v × 2^e`.
- A K that is not a multiple of 32 is zero-padded to the next multiple, in both activations and weights.

Block dot product:

- Each element product is the exact integer product of the INT8 activation and the 4-bit weight value in the table below.
- The 32 products are summed exactly as integers (17 bits).
- The sum `s` is converted exactly to the FP32 value `s × 2^(e_a + e_w + SBIAS)`.
- Block results are accumulated in FP32 with round-to-nearest-even, in block order `kb = 0, 1, 2, …`.

| Weight mode | 4-bit value | SBIAS |
|---|---|---|
| FP4 | E2M1 magnitude × 2, with sign | −7 |
| W8_HI | High nibble, signed −8..7 | −8 |
| W8_LO | Low nibble, unsigned 0..15 | −12 |

A W8 matrix multiply runs a W8_HI pass and then a W8_LO pass that accumulates onto the same output.

FP32 arithmetic everywhere rounds to nearest even, flushes subnormal inputs and results to zero, and turns a −0 result into +0. NaN and infinity are not handled; inputs are finite.

## Control

- Instructions are 128 bits, fixed length.
- The compiler unrolls one inference into a single instruction stream with no loops or branches. Quantities that depend on the input (rows to read, instances to keep, coefficients) are data in DDR (index lists, coefficient matrices), so the stream itself does not depend on the input.
- The DMA, GEMM and VEC engines run concurrently. Each engine accepts one instruction at a time and executes its instructions in order.
- The sequencer fetches the stream from DDR into a 64-instruction FIFO and issues in order. Each instruction carries one wait bit per engine; a set bit holds the instruction until every instruction already issued to that engine has completed. There are no finer-grained dependencies. An instruction whose engine is still busy also holds every instruction behind it.
- Every VEC instruction runs a program stored in the VEC program memory (CBUF). These programs have loops and branches. Input-dependent index lists and coefficients inside the forward pass are produced by VEC programs.
- Synchronization with the PS: SIGNAL writes an id and a value to a register and raises an interrupt; WAIT_HOST stops issue until the HOST_GO register, written by the PS, reaches a given value; HALT ends the stream and raises an interrupt. An undefined opcode stops the sequencer and raises an error interrupt.

## Instruction set

```
[127:120] opcode   [119:116] wait   [115:0] operands
```

The engines are numbered 0 DMA, 1 GEMM, 2 VEC. Wait bit e names engine e; bit 3 is reserved.

| Opcode | Name | Engine |
|---|---|---|
| 0 | NOP | — |
| 1 | DMA_LD | DMA |
| 2 | DMA_ST | DMA |
| 4 | GEMM | GEMM |
| 5 | VEC | VEC |
| 6 | SIGNAL | Sequencer |
| 7 | HALT | Sequencer |
| 8 | WAIT_HOST | Sequencer |

Opcode 3 and opcodes above 8 are illegal.

Operands:

```
DMA_LD / DMA_ST : [115:88] ddr_addr  [87:85] mem  [84:72] addr  [71:60] rows  [59:48] row_words  [47:28] ddr_stride  [27:25] kind  [24:19] tok  [18] acc  [17] flush
GEMM            : [111:92] a_base  [91:82] a_stride  [81:72] kb  [71:62] m  [61:42] w_base  [41:32] w_stride  [31:24] n8  [23:4] c_base  [3:2] wmode  [1] acc
VEC             : [115:108] entry  [107] q  [106:95] src  [94:83] src2  [82:70] dst  [69:61] rows_q  [60:53] n8  [51:43] p_base  [42:32] aux  [31:0] imm
SIGNAL          : [111:104] id  [31:0] value
WAIT_HOST       : [31:0] value
```

Operand bits that no field names are 0.

| Field | Codes |
|---|---|
| mem | 0 ABUF, 1 WBUF, 2 FBUF, 3 PBUF, 4 TBUF, 5 CBUF; 6 and 7 are not used |
| kind | 0 RAW, 1 ROW, 2 ROW_INDEXED, 3 SPLIT8, 4 T32_SPLIT8, 5 GACC; 6 and 7 are not used |
| wmode | 0 FP4, 1 W8_HI, 2 and 3 W8_LO |

### GEMM

`C[m][n] (+)= Σ_k A[m][k] · W[n][k]` with M = `m`, N = 8 · `n8`, K = 32 · `kb`.

- A: ABUF word `a_base + q · a_stride + b` holds tokens 4q..4q+3 of block b. Byte `j·32 + k` is element k of token j; bytes 128..131 are the scales of tokens 0..3.
- W: WBUF word `w_base + g · w_stride + b` holds columns 8g..8g+7 of block b. Nibble `c·32 + k` (bits `c·128 + 4k`) is element k of column c; bytes 128..135 are the scales of columns 0..7.
- C: FBUF word `c_base + q · n8 + g` holds tokens 4q..4q+3 × columns 8g..8g+7.
- `m` is a multiple of 32 from 32 to 992; `1 ≤ kb ≤ a_stride`; `1 ≤ n8 ≤ 255`.
- `acc = 1` accumulates onto the existing C.

### VEC

A VEC instruction loads its operand fields into scalar registers (see "VEC programs") and starts the program at CBUF word 4 · entry.

FBUF row addressing: word `base + q · n8 + g` for quad q and word g of the row. A PBUF row vector is dense: word `p_base + g/4`, slice `g mod 4`, replicated to the 4 tokens.

The library programs:

| Entry | Name | Inputs → output | Notes |
|---|---|---|---|
| 0 | ADD | src, src2 → dst | |
| 1 | MUL | src, src2 → dst | |
| 2 | MAX | src, src2 → dst | Element-wise max |
| 3 | ADDP | src, PBUF row vector → dst | |
| 4 | MULP | src, PBUF row vector → dst | |
| 5 | SCALE | src × imm (FP32) → dst | |
| 6 | COPY | src → dst | |
| 7 | RELU | src → dst | max(x, 0) |
| 8 | ACT | src → dst | `x · T(\|x\|)` with the table T in TBUF [512, 1024): σ for SiLU, Φ̃ for GELU |
| 9 | ACT_MUL | src (gate), src2 (up) → dst | ACT(gate) × up |
| 10 | ROPE | src → dst | rotate-half, head_dim 64; aux = number of heads (≤ 15); cos tiles at p_base, 4 words per quad; sin at p_base + rows_q · 4; without q, dst ≠ src |
| 12 | SOFTMAX | src → dst | aux = number of valid columns; later columns give 0 |
| 22 | LAYERNORM | src → dst | ε = imm; γ at p_base, β at p_base + n8/4; aux = element count |
| 34 | RMSNORM | src → dst | ε = imm; γ at p_base; aux = element count |

With q, the final pass is quantized and written to ABUF: dst is an ABUF address and n8 is a multiple of 4. The multi-pass programs write their intermediate passes to a scratch area in FBUF of the size of the output, named by src2 for LAYERNORM and RMSNORM and by `imm[11:0]` for ACT, ACT_MUL, SOFTMAX and ROPE; without q, the scratch area is dst.

Pass structure (→ writes; the last write goes to ABUF with q):

| Program | Passes |
|---|---|
| ADD, MUL, MAX | `x + x2 →`; `x × x2 →`; `max(x, x2) →` |
| ADDP, MULP, SCALE, COPY, RELU | `x + P →`; `x × P →`; `x × imm →`; `x →`; `max(x, 0) →` |
| LAYERNORM | 1: Σx; 2: Σ(x − mean)²; 3: `(x − mean) × rstd →` scratch; 4: `scratch × γ →` scratch; 5: `scratch + β →` |
| RMSNORM | 1: Σx²; 2: `x × rstd →` scratch; 3: `scratch × γ →` |
| ACT | 1: `T(\|x\|) →` scratch; 2: `x < 0 ? −(x × scratch) + x : x × scratch →` |
| ACT_MUL | 1, 2 as ACT into scratch; 3: `scratch × up →` |
| SOFTMAX | 1: max over the valid columns; 2: `2^((x − max) × log2e) →` scratch and Σ, 0 in the other columns; 3: `scratch × (1/Σ) →` |
| ROPE (per head; words g and g + 4 pair up) | 1: `x2 × sin →` scratch[g]; 2: `x1 × cos − scratch[g] →` dst[g]; 3: `x1 × sin →` scratch[g + 4]; 4: `x2 × cos + scratch[g + 4] →` dst[g + 4] |

LAYERNORM, RMSNORM and SOFTMAX process the quads in groups of 2, one row context per quad of the group, with the passes of a group in pass-major order (pass 1 of every quad, then pass 2, …) and a remainder of quads processed one at a time in context 0; the other programs run each pass over all quads. Passes follow the execution rules of VEC programs.

## VEC programs

A VEC instruction loads its operand fields into scalar registers:

| Field | Register |
|---|---|
| src | r1 |
| src2 | r2 |
| dst | r3 |
| rows_q | r4 |
| n8 | r5 |
| p_base | r6 |
| aux | r7 |
| imm | r8 |
| q | r9 |

r0 is always 0. r10–r31, RS0, RS1 and rn are not initialized. Programs start at CBUF words that are multiples of 4. A program ends with RET, which waits until every row context is idle and then clears the engine's busy state without waiting for the drain.

Instruction word, `[127:124]` kind: 0 PASS, 1 SALU, 2 SMEM, 3 BR, 4 RET, 5 NOP, 7 TRANSPOSE8; 6 and 8–15 act as NOP.

### PASS

A PASS streams `rq` quads × `rnw` words through the lanes, 4 cycles per word. Read and write addresses come from registers; each quad adds the row stride `rn8` (`rn8/4` for ABUF writes). The low 12 bits of rx and rx2, 13 of rd (12 for FBUF writes), 9 of rq, 10 of rnw, 8 of rn8 and 9 of rpb are used.

| Bits | Field | Meaning |
|---|---|---|
| [123:119] | rx | Register: x start (FBUF word) |
| [118:114] | rx2 | Register: x2 start |
| [113:109] | rd | Register: write start (FBUF or ABUF) |
| [108:104] | rq | Register: number of quads |
| [103:99] | rnw | Register: words per row |
| [98:94] | rn8 | Register: row stride (words) |
| [93:89] | rpb | Register: PBUF start |
| [88:84] | rimm | Register: immediate for b, c0 or sel = IMM, and ε for seps |
| [83:79] | rimm2 | Register: comparison value |
| [78] | x2_en | Read x2; with x2_en = 0, x2 is x |
| [77:76] | mg | Word map group: 2^mg words |
| [75:74] | ms | Word map stride: 2^ms words |
| [73:72] / [71:70] / [69:68] | mx / mx2 / md | Word index of the x read / the x2 read / the write: 0 w; 1 m(w); 2 m(w) + 2^mg; 3 as 0 |
| [67:66] | pa | PBUF value for c0 = PBUF and b = PBUF: 0 NONE; 1 ROWVEC (word `rpb + w/4`, slice `w mod 4` for every token); 2 TILE (word `rpb + q·4 + (w mod 4)`, slice t for token t); 3 as NONE |
| [65:63] | c0 | 0 ZERO, 1 X2, 2 PBUF, 3 RS0, 4 IMM; 5–7 as ZERO |
| [62:60] | b | 0 ONE, 1 X2, 2 PBUF, 3 RS1, 4 IMM, 5 PRE (square); 6 and 7 as ONE |
| [59:58] | nl | 0 NONE, 1 ACT, 2 EXP2, 3 TAB01 |
| [57:56] | po | Post operation: 0 ADD, 1 SUB, 2 MAX, 3 SIGNFIX |
| [55] | pv | Post operand c1: 0 zero, 1 x2 |
| [54:53] | red | Reduction: 0 NONE, 1 SUM, 2 MAX; 3 as NONE |
| [52:51] | wr | Write: 0 NONE, 1 FBUF, 2 ABUF (quantized), 3 FINAL (ABUF with q, else FBUF) |
| [50:48] | sf | Row scalar function after the pass: 0 NONE, 1 IDENTITY, 2 NEGATE, 3 RECIPROCAL, 4 RSQRT; 5–7 as NONE |
| [47] | st | Row scalar target: 0 RS1, 1 RS0 |
| [46] | srn | Multiply the reduction by rn before the function |
| [45] | seps | Add ε (the rimm value) before the function |
| [44:25] | px | Permutation of x: a 4-bit selector for each bit of the source element index, bit 4 at [44:41] down to bit 0 at [28:25] |
| [24:22] | cmp | 0 NONE, 1 LT, 2 LE, 3 EQ, 4 NE, 5 GT, 6 GE; 7 as NONE |
| [21:19] | cl | Left operand of the comparison: 0 x, 1 x2, 2 COLUMN (0..7), 3 POSITION (token · 8 + column, 0..31), 4 ROW_COLUMN (8 · w + column, with w the word index in the row); 5–7 as x |
| [18] | cr | Right operand of the comparison: 0 the rimm2 value, 1 x2 |
| [17] | cmp_int | Integer comparison (0: FP32); COLUMN, POSITION and ROW_COLUMN always compare as integers |
| [16:14] | sel | Value for lanes whose predicate is false: 0 NONE, 1 ZERO, 2 X2, 3 IMM, 4 NULL (0xFFFFFFFF), 5 RS1; 6 and 7 as NONE |
| [13:11] | ialu | Integer path: 0 NONE, 1 ADD, 2 SUB, 3 AND, 4 OR, 5 XOR, 6 SHR1 (logical shift right by 1), 7 RSUB (c0 − x) |
| [10] | c0_pred | c0 is 0 in lanes whose predicate is false |
| [9] | x_s1 | x and x2 read starts advance by 1 word per quad |
| [8] | dst_s1 | FBUF write start advances by 1 word per quad |
| [7] | ctx | Row context of the reduction, the row scalar and the RS0/RS1 reads |
| [6:5] | rm | Row mask: 0 NONE, 1 SKIP (quads whose RM bit is 0 are skipped), 2 SET (RM bit q is written for each quad q of the pass); 3 as NONE |
| [4:0] | — | Reserved |

The word map is `m(w) = ((w >> mg) << ms) + (w mod 2^mg)`: groups of 2^mg consecutive words placed 2^ms words apart; mode 2 takes the next group of each stride. A PASS that uses mode 1 has ms ≥ mg, and one that uses mode 2 has ms > mg.

The permutation gives, for output position `p = token · 8 + column` with bits p4..p0, bit k of the source element index: selector bits [2:0] = 0 take p_k, 1–5 take p4, p3, p2, p1, p0, 6 gives 0 and 7 gives 1; selector bit [3] inverts the result. All selectors 0 is the identity; for example, column c of each token broadcast to all columns sets bits 2..0 to the constants of c, and swapping tokens t and t xor 1 inverts bit 3.

Lane operation (x, x2 after permutation):

```
pred = cmp == NONE ? 1 : rel(left, right)
c0'  = c0_pred && !pred ? 0 : c0
ialu == NONE:  pre = x + c0' ;  prod = pre × b ;  y0 = nl ? table(prod) : post(prod)
ialu != NONE:  y0 = ialu(x, bits(c0'))
y    = sel == NONE ? y0 : (pred ? y0 : sel value)
```

y is the value written and reduced.

- `rel` compares the left operand selected by cl with the right operand selected by cr. FP32 comparison uses a sign-magnitude key, with −0 and subnormal values equal to +0. With cmp_int, and always when cl is COLUMN, POSITION or ROW_COLUMN, the comparison is int32 signed.
- `post(prod)`: ADD gives `prod + c1`, SUB `prod − c1`, MAX `max(prod, c1)`, and SIGNFIX `−prod + x` when the sign bit of x is set, else `prod`; c1 is 0 or x2 as selected by pv.
- The integer path bypasses the multiplier, the nonlinear table and the post-adder.
- TAB01 interpolates the first half of TBUF at the fraction of prod in [0, 1).
- Row scalar after a PASS with red and sf ≠ NONE, for each token: `r` is the token's Σ (red = SUM) or max (red = MAX); `v = r × rn` with srn (rn set by SMEM RN), then `v + ε` with seps; RS[st] of the context receives f(v), where f is the identity, the negation (the sign bit flipped, +0 for ±0), the reciprocal or the reciprocal square root. For example, a mean is NEGATE with srn into RS0 after a SUM, so that adding RS0 subtracts it.
- NaN and the value 0xFFFFFFFF must not enter the FP32 path or an FP32 comparison; integer and NULL entries go through the integer path and sel only. A MAX reduction orders NULL below every other value.
- Integer–FP32 conversion uses magic constants: `float(i) = bits(0x4B400000 + i) − 12582912.0` for |i| < 2^22; `rint(x) = bits(x + 12582912.0) − 0x4B400000`; `floor(x) = rint(x) − (rint(x) > x)`.

### TRANSPOSE8

| Bits | Field | Meaning |
|---|---|---|
| [123:119] | rx | Register: source FBUF base |
| [118:114] | rsw | Register: source word stride |
| [113:109] | rd | Register: destination FBUF base |
| [108:104] | rt | Register: tile count |
| [103:99] | rdt | Register: destination tile stride |
| [98:94] | rst | Register: source tile stride |
| [93:89] | rdw | Register: destination word stride |
| [88:87] | ror | Column rotation k: 0–2; 3 as 0 |
| [86:0] | — | Reserved |

Bases and strides use the low 12 bits of their registers and are measured in FBUF words. The tile count uses the low 9 bits. Each tile contains four independent 8×8 matrices of 32-bit elements. For tile q, source row r is read from `rx + q·rst + r·rsw`, and destination row c is written to `rd + q·rdt + c·rdw`. All word addresses wrap at 4,096 words. Element `(t, c)` of source row r is at bits `32·(8·t + c) +: 32`, where t selects one of the four matrices.

Destination row c, element `(t, j)` receives source row `ror(j, k)`, element `(t, c)`, where `ror(j, k)` rotates the 3-bit column index j right by k bits. Every 32-bit element is copied unchanged. The source and destination word sets are disjoint, and destination word addresses are distinct. A zero tile count performs no reads or writes.

TRANSPOSE8 waits for preceding vector operations to drain and completes all destination writes before advancing to the next instruction. The two tile buffers hold 8,192 bits each in eight skewed distributed-RAM banks. A registered output word and address remain stable during a denied FBUF write. A tile is released after its final word is written. RS0, RS1 and the scalar register file are unchanged.

### SALU

```
[123:119] op  [118:114] rd  [113:109] ra  [108:104] rb  [103] use_imm  [31:0] imm
```

`rd = ra op (use_imm ? imm : rb)`. op: 0 ADD, 1 SUB, 2 AND, 3 OR, 4 XOR, 5 SHL, 6 SHR, 7 SRA, 8 MUL (low 32 bits), 9 SLT (signed), 10 SLTU, 11 SEQ, 12 MOV (`rd = use_imm ? imm : rb`); 13–31 as MOV. Shifts use the low 5 bits of the second operand. Writes to r0 are discarded.

### SMEM

```
[123:119] op  [118:114] rd  [113:109] ra  [108:104] rb  [103] col_imm  [102:98] col  [97] which (0 RS1, 1 RS0)  [96] ctx  [31:0] imm (signed word offset)
```

| op | Name | Operation |
|---|---|---|
| 0 | SLD | `rd ← FBUF[ra + imm]` lane col (col_imm ? col : rb[4:0]) |
| 1 | RSMOV | `rd ← RS[which][col[1:0]]` of row context ctx |
| 2 | RN | `rn ← 1 / float(ra)`, ra < 2^14; ra = 0 counts as 1 |
| 3–31 | — | No operation |

The FBUF address `ra + imm` wraps at 4,096 words.

### BR

```
[123:119] cond (0 ALWAYS, 1 Z: ra == 0, 2 NZ: ra != 0, 3–31 never)  [113:109] ra  [31:0] target (CBUF word; bits [9:0] are used)
```

SALU, SMEM and BR place op or cond, rd, ra, rb and the immediate at the same bits.

RET and NOP have no operands.

### Execution rules

- A PASS without rm = SET that is followed by a PASS does not wait for the pipeline to drain: the next PASS starts while its writes complete, and a read of an FBUF word whose write is still pending waits for the write. Every other PASS waits for the pipeline to drain at its end.
- A PASS with rm = SET writes RM bit q, for each quad q < rq, with 1 when any element of the words that quad q produces is nonzero and 0 otherwise; the other RM bits are unchanged.
- A PASS with rm = SKIP does nothing for a quad q whose RM bit is 0: no reads, no writes, no reduction; the read and write starts advance as usual. The reduction accumulator index counts the words that are issued. RM is unchanged until the next PASS with rm = SET, and its bits are undefined before the first one.
- The issue stage passes over up to two skipped quads per cycle and ends a PASS as soon as every remaining quad is skipped.
- A row context is reducing from the start of a PASS with red in that context until every word of the PASS has been reduced; if the PASS has sc, the context then holds the scalar program until its result is written to RS0 or RS1 of the context. The scalar unit runs one program at a time, in the order the reductions complete.
- A PASS with red waits at its start until its context is neither reducing nor holding a scalar program, then clears the context's accumulators.
- A PASS that reads RS0 or RS1 (c0 = RS0, b = RS1 or sel = RS1) waits at its start while a scalar program of its context is outstanding.
- SMEM waits for preceding passes to drain; RSMOV also waits for an outstanding scalar program of its context, and RN waits for the scalar unit to be idle.
- The rimm and rimm2 values are captured when a PASS is issued.
- A PASS must not have rq = 0 or rnw = 0.
- sel = X2 and po = SIGNFIX are not used in the same pass.

## Vector arithmetic

Each lane computes `pre = x + c0`, `prod = pre × b`, then either `post(prod)` ("Lane operation" in "VEC programs") or a nonlinear function of `prod`. Every operation rounds to FP32 under the rules in "Number formats".

Nonlinear functions use piecewise-linear interpolation in tables:

| Function | Use | Domain | Segments |
|---|---|---|---|
| 2^f | exp in softmax: `2^t` with `n = floor(t)`, `f = t − n`, result `2^f × 2^n` | f ∈ [0, 1) | 512 |
| σ(\|x\|) | SiLU(x) = x · σ(x) | \|x\| ∈ [0, 16); 1.0 at and above 16 | 512 |
| Φ̃(\|x\|) | GELU (tanh approximation) = x · Φ̃(x), Φ̃(x) = (1 + tanh(√(2/π)(x + 0.044715x³))) / 2 | same as σ | 512 |

- Table entries: `v[i] = fp32(F(i·h))`, `s[i] = fp32(v[i+1] − v[i])`. The tables share one 1,024-entry memory: [0, 512) holds 2^f, [512, 1024) holds σ or Φ̃.
- The input is converted to a 25-bit fixed-point value: |x| truncated to UQ4.21 for σ and Φ̃; for 2^t, t × 2^25 rounded toward −∞ gives n = floor(t) and the fraction f in its low 25 bits; t ≤ −256 gives 0 and t ≥ 256 gives +∞. `idx` is the top 9 bits of that value and `frac` the next 16 bits.
- `y = v[idx] + RNE(s[idx] × frac / 2^16)`. For 2^t the result is positive and multiplied by 2^n by exponent addition; it flushes to zero below 2^−126 and becomes +∞ at 2^128 and above.
- For x < 0, σ(−|x|) = 1 − σ(|x|), so SiLU(x) = −(x · σ(|x|)) + x. GELU uses the same identity with Φ̃.

Scalar functions, for a positive normal `a` (about 2^−100 to 2^100), with four Newton iterations:

- `1/a`: `y0 = bits(0x7EF311C7 − bits(a))`; `y ← y × (2 − a × y)`.
- `1/√a`: `y0 = bits(0x5F375A86 − (bits(a) >> 1))`; `y ← y × (1.5 − ((a × y) × y) × 0.5)`.

Row reductions:

- Sum: each group of 8 elements is added as `((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))`; the result of group j is added to accumulator `j mod 8`; the 8 accumulators are combined with the same tree.
- Max: independent of order; values are compared by their sign-magnitude key without flushing, so −0 is below +0.

Row operations:

- LayerNorm: `mean = sum(x) × (1/n)`; `var = sum((x − mean)²) × (1/n)`; `rstd = 1/√(var + ε)`; `y = ((x − mean) × rstd) × γ`, then `+ β`.
- RMSNorm: `rstd = 1/√(sum(x²) × (1/n) + ε)`; `y = (x × rstd) × γ`.
- Softmax over the first L columns: `m = max(x)`; `e_i = 2^((x_i − m) × log2e)` with log2e rounded to FP32 (0x3FB8AA3B); columns at and beyond L give 0; `y_i = e_i × (1/sum(e))`.
- RoPE (rotate-half pairs x1, x2): `y1 = x1 × cos − (x2 × sin)`, `y2 = x2 × cos + (x1 × sin)`; the bracketed product is rounded first.
- `1/n` is computed with the `1/a` function above.
- Quantization to ABUF follows the MXINT8 encoding per token and 32-column block.

## DMA

### Words in DDR

An on-chip word is stored little-endian and padded with zeros to a multiple of 16 bytes (one 128-bit AXI beat).

| mem | Word | Bytes in DDR |
|---|---|---|
| ABUF | 1,056 bits: bytes 0..127 = 4 tokens × 32 INT8 (byte `j·32 + k`), bytes 128..131 = 4 scales, 12 zero bytes | 144 |
| WBUF | 1,152 bits | 144 |
| FBUF, PBUF | 1,024 bits | 128 |
| TBUF | 64 bits | 16 |
| CBUF | 128 bits | 16 |

DDR addresses and strides count 16-byte units: `ddr_addr`, `ddr_stride`, the entries of index lists and gather lists, and the coefficient address of a gather list. `addr` is the on-chip word address.

| kind | DMA_LD | DMA_ST |
|---|---|---|
| RAW | Words of mem | Words of ABUF or FBUF |
| ROW | Row format into ABUF or FBUF | Row format from ABUF or FBUF |
| ROW_INDEXED | Row format through an index list into ABUF or FBUF | — |
| SPLIT8 | W8 planes into WBUF | — |
| T32_SPLIT8 | W8 planes of the transpose into WBUF | — |
| GACC | Gather-accumulate into FBUF | — |

- mem names the memory read or written: WBUF for SPLIT8 and T32_SPLIT8, FBUF for GACC.
- `rows` and `row_words` are at least 1.
- acc and flush are 0 except for GACC.
- DMA_ST reads ABUF through the write port of ABUF in the cycles without a write from VEC or DMA, so it does not wait for the GEMM engine.

### RAW (2D)

Row r = 0..rows−1, word w = 0..row_words−1: DDR byte `16 · (ddr_addr + r · ddr_stride) + w · (word bytes)` ↔ on-chip `addr + r · row_words + w`.

### Index list (ROW_INDEXED)

DMA_LD of kind ROW_INDEXED reads a list of `rows · row_words` 32-bit entries at `ddr_addr`. Entry e is the DDR address of token e, and the tokens are packed in order as in row format. The entry 0xFFFFFFFF reads nothing and writes a zero token (for MXINT8, an all-zero block with scale byte 127).

### Row format (ROW, ROW_INDEXED)

A token is contiguous in DDR:

| Format | One token | Size |
|---|---|---|
| MXINT8 (to and from ABUF) | `tok` blocks of 32 bytes, then `tok` scale bytes, zero-padded to 16 bytes | `32·tok + 16·ceil(tok/16)` bytes |
| FP32 (to and from FBUF) | `32·tok` FP32 values | `128·tok` bytes |

- `tok` is the number of 32-element blocks per token: 1 to 32 for MXINT8 and 1 to 8 for FP32.
- `row_words · (token size / 16) < 65,536`.
- DMA_LD of kind ROW: row r, token w is at DDR byte `16 · (ddr_addr + r · ddr_stride) + w · (token size)`; each 4 consecutive tokens are packed into one quad of on-chip words. `rows · row_words` is a multiple of 4.
- DMA_ST of kind ROW: each on-chip quad is unpacked into 4 tokens.

### SPLIT8

Converts an ABUF-format matrix into the two W8 planes in WBUF (mem = WBUF). Rows 2p and 2p + 1 (8 tokens) are taken together at the same word position w. Token j (0..3) of row 2p + r becomes column c = 4·r + j. The high nibble of each INT8 (signed, `q = 16·hi + lo`) goes to the HI plane, the low nibble to the LO plane; scales are copied. HI word `addr + p · W + w`, LO word `addr + nW + p · W + w`, with W = row_words and `nW = rows · W / 2`. `rows` is even.

### T32_SPLIT8

Converts an ABUF-format matrix into the W8 planes of its transpose (mem = WBUF). For each group of 8 rows (32 tokens) r8 and word position w, the 32-token × 32-dimension block is transposed: dimension d becomes column `c = d mod 8` of column group `cg = d / 8`, and the token becomes the K position. The token scales are aligned: `emax` is the largest `e_t` among the tokens of the block that have a nonzero element (`e = 0` if none); each element becomes `RNE(q / 2^(emax − e_t))`, which is 0 when `emax − e_t ≥ 9`; all 8 output scales are `emax`. The word for (w, cg, r8) is `addr + (4·w + cg) · KBt + r8` with `KBt = rows / 8`; the LO plane adds `nW = rows · W / 2`. `rows` is a multiple of 8.

### GACC

Builds the output rows of one quad (4 tokens) in FBUF (mem = FBUF), `out[t][ch]` with `ch = 0..32·row_words − 1`, from MXINT8 rows in row format:

```
out[t][ch] = Σ_{s = 0..S−1} Σ_{k = 0..3} w[t][kb(ch)][s] · bw[s][t][k] · row(addr[s][t][k])[ch]      kb(ch) = ch / 32
```

| Operand | Meaning |
|---|---|
| ddr_addr | List address. The first 16 bytes are a header: [31:0] address of w for this quad, the other bytes 0 |
| ddr_stride | Stride between the w areas of consecutive blocks kb |
| rows | Number of slots S (≤ 2047) |
| row_words | Blocks per token, K / 32 (1 to 8) |
| addr | FBUF address of the output quad row (4 · row_words words) |

- Entries follow the header, 8 bytes each: [31:0] row address (0xFFFFFFFF adds nothing), [63:32] bw (FP32). Order: `for s: for t: for k`.
- `w[t][kb][s]` is element `t·8 + (s mod 8)` of the FBUF-format word at DDR byte `16 · (w address + kb · ddr_stride + 8 · (s/8))`.
- acc = 0 starts the accumulators at 0; flush = 1 writes them to addr at the end. Larger S, and lists per camera, are chained with acc = 1.
- Output word g of the quad row holds channels 8g..8g+7 of the 4 tokens (element `t·8 + c`).

## Gather-accumulate arithmetic

For each entry, in list order:

- Coefficient `c = RNE(w × bw)`.
- The row's block scale is applied to `c` by exponent addition: `c' = c × 2^(e_row − 6)`; beyond the FP32 range c' is the largest finite value of the same sign, and below 2^−126 it is 0.
- `acc = RNE(acc + RNE(q × c'))`, where the INT8 element `q` converts to FP32 exactly.
- A null entry adds nothing.

## Control registers

AXI-Lite, byte offsets:

| Offset | Name | Contents |
|---|---|---|
| 0x00 | CTRL (W) | bit 0 START: run INSTR_COUNT instructions from INSTR_BASE (a START while running restarts the stream); bit 1 SOFT_RESET: stop issue and fetch, empty the IBUF, discard the instruction words still in flight and clear running, halted and error (registers, counters and engines are kept) |
| 0x04 | STATUS (R) | bit 0 running, 1 halted, 2 error, 3 waiting for host, 4 + e engine e busy (4 DMA, 5 GEMM, 6 VEC), 7 IBUF empty |
| 0x08 | INSTR_BASE | DDR address of the instruction stream (256-byte aligned) |
| 0x0C | INSTR_COUNT | Number of instructions |
| 0x10 | IRQ_STATUS (R, write 1 to clear) | bit 0 SIGNAL, 1 HALT, 2 illegal instruction, 3 AXI error |
| 0x14 | IRQ_ENABLE | Interrupt output = IRQ_STATUS & IRQ_ENABLE |
| 0x18 | SIGNAL | [7:0] last id, [31:16] count |
| 0x1C | SIGNAL_VALUE | Last value |
| 0x20 | Issued instructions | Since START |
| 0x24 | Cycles | From START to HALT |
| 0x28 + 4e | Busy cycles of engine e | 0x28 DMA, 0x2C GEMM, 0x30 VEC |
| 0x34 | Issue-stall cycles | An instruction is waiting but not issued |
| 0x38 | Starve cycles | No instruction is waiting to issue |
| 0x3C | TRACE_BASE | Trace destination in DDR (16-byte aligned) |
| 0x40 | TRACE_CTRL | bit 0 enable, bit 2 statistics records (both set before START); read bit 1 idle |
| 0x44 | TRACE_COUNT (R) | Trace records written, of both types |
| 0x48 | HOST_GO (R/W) | Value written by the PS; WAIT_HOST proceeds when HOST_GO ≥ value; cleared by START |

### Execution trace

With trace enabled, the sequencer writes one 16-byte completion record per instruction to DDR from TRACE_BASE, through the write channel of the instruction-fetch AXI master. GEMM, VEC and DMA instructions are recorded when they complete; the other instructions, including NOP and illegal opcodes, when they issue. Bits [119:116] of every record hold its type.

Completion record (type 0):

```
[31:0] instruction index (issue order)  [63:32] issue cycle  [95:64] completion cycle  [99:96] source (0 DMA, 1 GEMM, 2 VEC, 3 sequencer)  [107:100] opcode  [111:108] wait  [119:116] 0  [127:112] other bits 0
```

With TRACE_CTRL bit 2 set, the completion record of a GEMM, VEC or DMA instruction is followed by a statistics record (type 1) of the same instruction:

```
[31:0] instruction index  [35:32] source  [39:36] 0  [63:40] A  [87:64] B  [111:88] C  [115:112] 0  [119:116] 1  [127:120] D
```

A, B and C are 24-bit counters and D an 8-bit counter; all saturate, are cleared when the engine accepts the command and are captured when it completes:

| Source | A | B | C | D |
|---|---|---|---|---|
| DMA | AXI read beats | AXI write beats | cycles with an on-chip read or write request not granted | null index-list and gather entries |
| GEMM | cycles with the array fed | 0 | 0 | 0 |
| VEC | cycles in the issue state and in TRANSPOSE8 | cycles in the drain states | cycles a PASS, an SMEM or RET waits for a row context or the scalar unit | PASS instructions started |

Each source holds one pending completion record with its statistics record. One record per cycle moves into the 64-entry record FIFO: the completion record of the selected source, then its statistics record when enabled; when several sources are pending, the order is DMA, GEMM, VEC, sequencer. Issue is held while the source's record slot is occupied or the record FIFO holds 56 or more records. TRACE_COUNT counts the records of both types.

## Board integration

| Item | Configuration |
|---|---|
| Block design | Zynq UltraScale+ PS (KV260 SOM preset) and the NPU, referenced as a module through a Verilog wrapper |
| Control registers | M_AXI_HPM0_FPD (128 bits) → SmartConnect → NPU AXI-Lite (32 bits), at 0xA000_0000, 4 KB |
| DMA | NPU AXI4 master 0 → S_AXI_HP0_FPD (128 bits), connected directly; AxCACHE 0011, AxPROT, AxQOS, AxLOCK, AxID and AxUSER 0 |
| Instruction fetch and trace | NPU AXI4 master 1 → S_AXI_HP1_FPD (128 bits), connected like master 0 |
| AXI register stage | Both masters are registered at the NPU boundary: AR, AW and W through two-entry skid buffers with registered ready, R and B through input registers; RREADY and BREADY are always 1 |
| Clock | pl_clk0, 250 MHz |
| Interrupt | pl_ps_irq0[0], GIC SPI 89, level high |

Linux (Ubuntu 22.04 on the board) loads the design with `xmutil loadapp npu` from `/lib/firmware/xilinx/npu/`:

- `npu.bit.bin` (from bootgen).
- `npu.dtbo`: assigns `npu.bit.bin` and the four PL resets to the fpga-region, sets PL clock 0 to 250 MHz, and exposes the control registers (0xA000_0000, 4 KB, interrupt SPI 89) as a `generic-uio` device.
- `shell.json`: `XRT_FLAT`, one slot.

## PS software

- OS: Ubuntu 22.04 on the board.
- Register access: UIO (`/dev/uioN`).
- Shared memory: u-dma-buf, one 320 MB buffer from CMA with a 32-bit DMA mask, loaded at boot. udev gives the `plugdev` group read and write access to `/dev/uio*` and `/dev/udmabuf*`. The physical addresses used in instructions are inside this buffer.
- The board program runs its frames either on the NPU or, with `--device iss`, on the instruction-set simulator through the same driver interface.
- Tracing: the runtime enables the execution trace on request, with TRACE_BASE at the `trace` region of the image (32 bytes per instruction of the longest stream), waits for the trace to drain after HALT and checks TRACE_COUNT against the issued count. `hotarubi_run --trace DIR` (and `--trace-stats` for statistics records) writes per frame `<frame>.trace.bin` with the records and `<frame>.counters.json` with the stream name, the trace mode, every counter, the wall time of the frame split into input placement, NPU run, signal handling and output reading (nanoseconds), the SHA-256 of the image, the bitstream and the input file, and the PL clock when readable.
- The driver (register access, start, interrupt wait, counters) and the runtime are written in C++23 and cross-compiled on the host (see "Build").
- The PS places the inputs in DDR, starts the instruction stream, and runs the processing outside the forward pass:
  - SmolVLA: camera inputs `camera1` = SO-100 `top`, `camera2` = SO-100 `wrist`, `camera3` unused; image resize with padding to 512 × 512 (aspect-preserving bilinear, `align_corners = False`, zero padding at the top left) and scaling from [0, 1] to [−1, 1]; im2col of the 16 × 16 patches into 768-element rows with K = c·256 + ky·16 + kx (patch s = py·32 + px on the 32 × 32 grid), placed in the order p = (i·4 + j)·64 + hq·8 + wq for patch s = (4·hq + i)·32 + 4·wq + j (i, j = 0..3; hq, wq = 0..7), and MXINT8 encoding; lookup of the language embedding rows; state zero-padded to 32 values; noise.
  - SparseDrive-S: images, timestamps, ego pose, projection matrices and image sizes; detection of the start of a scene; hierarchical planning selection and collision rescoring after the forward pass.

## Quantization

### Activations

- The input of every matrix multiply is MXINT8 with blocks along K. Conv inputs have blocks along channels. SmolVLA's pixel input is MXINT8.
- Attention: Q, K and V have blocks along head_dim; P has blocks along keys. K becomes the W8 operand of QKᵀ through SPLIT8, and V the W8 operand of PV through T32_SPLIT8.
- Feature maps read by the gather-accumulate are MXINT8 with blocks along channels.
- Gather-accumulate coefficients and accumulation, normalization, softmax, ReLU and residual additions are FP32.

### Weights

Weights are quantized once on the host. Biases and normalization parameters stay FP32. SmolVLA's language embedding table is not quantized; the host looks up its rows.

W8 (MXINT8, round to nearest):

- SmolVLA: `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`, and the vision patch embedding.
- SparseDrive-S: Linear layers with K < 32 or N ≤ 64; the first conv (Cin = 3, 7 × 7), as im2col with K ordered (c, ky, kx) in rows of 8 where kx = 7 is zero, K = 192; the attention input projections for Q, K and V.

Every other Linear layer and conv is quantized to MXFP4 by GPTQ. SparseDrive-S folds BatchNorm into the conv weights before quantization.

GPTQ:

- The FP32 model runs on the calibration data. For each layer, `H = (1/N) Σ x xᵀ` over all N input rows. For a conv, x is the im2col column ordered (kh, kw, Cin), so each 32-element block lies along Cin within one tap. SmolVLA's action expert accumulates the inputs of all 10 denoising steps.
- 1% of the mean of the diagonal of H is added to its diagonal. Columns whose diagonal entry of H is zero get zero weight.
- Columns are quantized in order (no activation ordering), with lazy updates in blocks of 128 columns.
- At the first column of each 32-column group, the group's scale is set from the group's current, error-compensated weights by the MXFP4 encoding.
- Each column is rounded to E2M1 × 2^e, and its error is propagated to the remaining columns through the upper Cholesky factor of H⁻¹.

Calibration data:

- SmolVLA: 96 frames from the even-numbered episodes of `lerobot/svla_so100_pickplace`, chosen with seed 11, each with the images of both of its cameras; frame i uses noise seed i.
- SparseDrive-S: all 323 frames of nuScenes `mini_train` (8 scenes) in scene order, resetting the temporal state at the start of each scene.

## Toolchain

The NPU toolchain is written in C++23:

| Component | Contents |
|---|---|
| Instruction-set simulator | The bit-exact arithmetic of each unit, execution of instruction streams, a cycle model that assigns issue and completion cycles, writes trace records of both types and drives the cycle counters, and the hazard check of the executed streams |
| Compiler | Generation of instruction streams, their scheduling, the DDR image and the JSON listings |
| Trace analyzer | Validation, reports, kernel tables, timelines, differences and calibration of execution traces; prediction of traces with the cycle model |
| RTL testbenches | Verilator testbenches |
| Runtime | Driver and runtime on the board, cross-compiled for aarch64 |

### Compiler output

The compiler writes a directory with `image.bin`, `meta.json` and one `<stream>.listing.json` per instruction stream. `meta.json` holds the base address, the main stream, the stream table, the listing file names, the parameters and the regions of the image; the `trace` region, allocated after the streams, holds 32 bytes per instruction of the longest stream. A listing holds the base address, the tag table and one entry per instruction with its index, engine, opcode, wait bits, hexadecimal word, tag index, note, and the on-chip ranges it reads and writes. Tags are the scopes open when the instruction was emitted, joined with `/`.

### Scheduling

Kernels emit instructions in program order with the on-chip ranges each instruction reads and writes. Two instructions conflict when one writes a range the other reads or writes, when both are DMA transfers and at least one is a store whose DDR extent overlaps the other's extent (the extent of an index-list or GACC transfer is all of DDR), when both are GACC transfers, when both are VEC commands and one of them runs a program outside the library, or when one of them is SIGNAL, WAIT_HOST or HALT. The compiler reorders each stream so that conflicting instructions keep their program order, choosing at every step, among the first `schedule-window` unscheduled instructions in program order whose conflicting predecessors are all scheduled, the one with the earliest predicted issue cycle under the sequencer rules and the cycle model, with program order as the tie-break. The wait bits are then recomputed for the new order: an instruction waits for an engine when a conflicting instruction of that engine was issued after the last point at which that engine was known to be idle. `--schedule-window 0` keeps the program order; the window is recorded in `meta.json`.

### Cycle model

The simulator assigns each instruction an issue cycle and a completion cycle with the sequencer rules: in-order issue one cycle after the previous issue at the earliest, an instruction waits for the completion of every engine named by its wait bits and of its own engine. The first instruction issues at `start_cycles`. The cost of a command is a linear function of the activity counted while the simulator executes it:

| Engine | Cycles |
|---|---|
| GEMM | `n8 · (m/32) · kb · 8 + gemm_fixed_cycles` |
| DMA | `dma_fixed_cycles + read beats · dma_read_cycles_per_kilobeat / 1024 + read bursts · dma_read_burst_cycles + write beats · dma_write_cycles_per_kilobeat / 1024 + write bursts · dma_write_burst_cycles + gather entries · dma_gather_entry_cycles + gathered rows · dma_gather_row_cycles + coefficient fetches · dma_gather_fetch_cycles` |
| VEC | `vec_fixed_cycles + passes · vec_pass_cycles + issue + drain + scalar + SALU · vec_salu_cycles + SMEM · vec_smem_cycles + branches · vec_branch_cycles` |

For gather-accumulate, read beats and bursts count the index list and the coefficients, the beats of gathered rows are counted only in the statistics record, and coefficient fetches are the coefficient requests made while an entry waits for them.

For VEC, `issue = (words + stalled words) · vec_word_cycles − passes · vec_issue_tail_cycles + transposes · vec_transpose_cycles + tiles · vec_transpose_tile_cycles + skip cycles`, where words counts the issued words, a run of k skipped quads costs ⌈k/2⌉ cycles at the start of a PASS, one less when the PASS directly follows a PASS that did not drain, max(0, ⌊k/2⌋ + c − 4) cycles between two issued words, where c is the number of FBUF reads of a word (2 with a second operand), and ⌊(k − 1)/2⌋ cycles at the end of a PASS that does not drain, less 40 and at least 0 at the end of a PASS that drains (a PASS whose quads are all skipped costs the start amount, less 40 and at least 0 when it drains), a word whose read finds a write of an earlier overlapped pass pending is stalled until 10 words after the issue of that write, `drain = draining passes · vec_drain_cycles + draining quantizing passes · vec_drain_quantized_cycles`, and `scalar = context waits · vec_word_cycles + RN · 138`. A reduction completes 16 words after the issue of its last word; a scalar program then takes 11 cycles per arithmetic micro-operation (the four tokens share one pipelined fp32 adder and multiplier), 2 per immediate write and 2 to start, after the previous program of the scalar unit; a PASS or SMEM that needs the context waits the remaining words. A row scalar has 7 arithmetic micro-operations for the sum tree of red = SUM, 1 for srn, 1 for seps, 12 for RECIPROCAL and 20 for RSQRT, and 1 immediate write, 1 more for RECIPROCAL and RSQRT, and 1 more when red = MAX is followed by arithmetic. RN has 12 arithmetic micro-operations and 2 immediate writes.

The statistics of a predicted record are the read and write beats, the cycles between issue and the start of the data phase and the null entries for DMA; the array cycles for GEMM; and the issue, drain and scalar cycles and the pass count for VEC. The host's response to SIGNAL and WAIT_HOST takes no time in the model, and the trace back-pressure and the contention for the FBUF write port between the engines are not modelled. The default parameters are the values that the trace analyzer calibrates from board traces of both models with statistics records, except vec_word_cycles, vec_transpose_cycles, vec_transpose_tile_cycles and dma_gather_fetch_cycles, which are fitted to Verilator runs of the RTL. `hotarubi_trace predict` and `hotarubi_run --device iss` take `--timing NAME=VALUE,...` to override any parameter. The scheduler estimates the cost of an instruction with the same model from the activity implied by its fields (the activity of a library program by executing it; another VEC command as one pass over its words).

### Hazard check

While executing a stream, the simulator records the on-chip words every command reads and writes and reports a hazard for each pair of commands on different engines whose accesses conflict (a write and a read or two writes of the same word) without an ordering between them. A command is ordered after an earlier command when a wait bit of it or of any command issued between them names the earlier command's engine after that command was issued, or when the engines are in order and the earlier command's engine has issued a later command (a command issues only when its engine is free). `hotarubi_trace predict` and `hotarubi_run --device iss` fail on a stream with hazards and list them. `hotarubi_trace predict` also checks that the ranges listed for every instruction cover the words it actually reads and writes (the fetch of a library program from the ROM part of CBUF is implied) and fails otherwise; with `--accesses` it writes `<stream>.accesses.jsonl`, one line per instruction with the extent of its actual reads and writes in each memory.

## Build

CMake is the single entry point:

- It builds the toolchain and the Verilator testbenches, and runs the tests with CTest.
- It runs the Vivado steps (synthesis, implementation, bitstream) through custom targets that call Tcl scripts in `fpga/`.
- The list of RTL sources is defined once in CMake and used by both Verilator and Vivado.
- The runtime is cross-compiled for aarch64 and linked statically.

## RTL verification

The RTL is simulated with Verilator. The testbenches are written in C++.

The kernel bench runs a window of a compiled instruction stream on the RTL: it loads the image into the simulated DDR, appends the chosen preamble instructions, the window and a HALT as a new stream after the image, records the trace with the image's trace region, and writes a compiled directory (`image.bin` with the DDR contents before the run, `meta.json`, `kernel.listing.json`) together with `kernel.trace.bin` and `kernel.counters.json`.
