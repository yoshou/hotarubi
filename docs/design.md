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
- The fixed operations (VEC op 0–15) are programs in CBUF [0, 256), present from configuration. VEC op 16 (RUN) starts a program loaded into CBUF.
- Each lane has two FP32 adders, one FP32 multiplier and one 24 × 16-bit multiplier for table interpolation.
- Row scalars RS0 and RS1 (one per token of a quad) are computed by a 4-way scalar unit after a pass has drained.
- With the Q flag, the final pass is quantized to MXINT8 per 4 FBUF words and written to ABUF instead of FBUF.

### DMA engine

- One AXI4 master, 128 bits, INCR bursts of at most 16 beats that do not cross a 4 KB boundary.
- One instruction at a time.
- The gather-accumulate datapath has 8 lanes.

### On-chip memories

| Name | Implementation | Word | Depth | Contents |
|---|---|---|---|---|
| ABUF | URAM | 1,056 bits: 4 tokens × 32 INT8, then 4 E8M0 scales | 8,192 | MXINT8 activations (GEMM A operand) |
| WBUF | BRAM | 1,152 bits: 8 columns × 32 × 4 bits, then 8 E8M0 scales | 2,048 | GEMM W operand |
| FBUF | URAM | 1,024 bits: 32 FP32, element `t·8 + c` for token t and column c | 4,096 | GEMM output and accumulation, VEC input and output |
| PBUF | BRAM | 1,024 bits: 32 FP32 | 512 | Normalization γ and β, biases, RoPE cos and sin |
| TBUF | BRAM, one copy per lane pair (4 copies) | 64 bits: `{s[31:0], v[31:0]}` | 1,024 | Nonlinear-function tables |
| CBUF | BRAM | 128 bits | 1,024 | VEC programs: [0, 256) fixed operations (initial contents), [256, 1024) loaded by DMA; DMA can write every word |
| IBUF | FIFO | 128 bits | 64 | Instruction prefetch |

The VEC scalar register file has 32 registers of 32 bits.

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
- The VEC engine also runs programs stored in its program memory (CBUF), started by a VEC RUN instruction. These programs have loops and branches. Input-dependent index lists and coefficients inside the forward pass are produced by VEC programs.
- Synchronization with the PS: SIGNAL writes an id and a value to a register and raises an interrupt; WAIT_HOST stops issue until the HOST_GO register, written by the PS, reaches a given value; HALT ends the stream and raises an interrupt. An undefined opcode stops the sequencer and raises an error interrupt.

## Instruction set

```
[127:120] opcode   [119:116] wait (bit 0 DMA, bit 1 reserved, bit 2 GEMM, bit 3 VEC)   [115:112] flags   [111:0] operands
```

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
DMA_LD / DMA_ST : [111:80] ddr_addr  [79:77] mem  [76:57] dst/src  [56:45] rows  [44:33] row_words  [32:9] ddr_stride  [8:7] mode  [6:0] tok
GEMM            : [111:92] a_base  [91:82] a_stride  [81:72] kb  [71:62] m  [61:42] w_base  [41:32] w_stride  [31:24] n8  [23:4] c_base  [3:2] wmode  [1] acc
VEC             : [111:107] op  [106:95] src  [94:83] src2  [82:70] dst  [69:61] rows_q  [60:53] n8  [52:43] p_base  [42:32] aux  [31:0] imm
SIGNAL          : [111:104] id  [31:0] value
WAIT_HOST       : [31:0] value
```

| Field | Codes |
|---|---|
| mem | 0 ABUF, 1 WBUF, 2 FBUF, 3 PBUF, 4 TBUF, 5 CBUF; 6 and 7 are not used |
| mode | 0 RAW, 1 SPLIT8, 2 T32_SPLIT8, 3 GACC |
| wmode | 0 FP4, 1 W8_HI, 2 and 3 W8_LO |
| flags, DMA_LD and DMA_ST | bit 0 I (index list), bit 1 R (row format) |
| flags, GACC | bit 0 acc, bit 1 flush |
| flags, VEC | bit 0 Q |

### GEMM

`C[m][n] (+)= Σ_k A[m][k] · W[n][k]` with M = `m`, N = 8 · `n8`, K = 32 · `kb`.

- A: ABUF word `a_base + q · a_stride + b` holds tokens 4q..4q+3 of block b. Byte `j·32 + k` is element k of token j; bytes 128..131 are the scales of tokens 0..3.
- W: WBUF word `w_base + g · w_stride + b` holds columns 8g..8g+7 of block b. Nibble `c·32 + k` (bits `c·128 + 4k`) is element k of column c; bytes 128..135 are the scales of columns 0..7.
- C: FBUF word `c_base + q · n8 + g` holds tokens 4q..4q+3 × columns 8g..8g+7.
- `m` is a multiple of 32 from 32 to 992; `1 ≤ kb ≤ a_stride`; `1 ≤ n8 ≤ 255`.
- `acc = 1` accumulates onto the existing C.

### VEC

FBUF row addressing: word `base + q · n8 + g` for quad q and word g of the row. A PBUF row vector is dense: word `p_base + g/4`, slice `g mod 4`, replicated to the 4 tokens.

| op | Name | Inputs → output | Notes |
|---|---|---|---|
| 0 | ADD | src, src2 → dst | |
| 1 | MUL | src, src2 → dst | |
| 2 | ADDP | src, PBUF row vector → dst | |
| 3 | MULP | src, PBUF row vector → dst | |
| 4 | SCALE | src × imm (FP32) → dst | |
| 5 | COPY | src → dst | |
| 6 | LAYERNORM | src → dst | ε = imm; γ at p_base, β at p_base + n8/4; aux = element count (0 means 8 · n8) |
| 7 | RMSNORM | src → dst | ε = imm; γ at p_base; aux as LAYERNORM |
| 8 | GELU | src → dst | Φ̃ table in TBUF [512, 1024) |
| 9 | SILU | src → dst | σ table in TBUF [512, 1024) |
| 10 | SILU_MUL | src (gate), src2 (up) → dst | SiLU(gate) × up |
| 11 | SOFTMAX | src → dst | aux = number of valid columns; later columns give 0 |
| 12 | ROPE | src → dst | rotate-half, head_dim 64; aux = number of heads (≤ 15); cos tiles at p_base, 4 words per quad; sin at p_base + rows_q · 4; without Q, dst ≠ src |
| 13 | QUANT | FBUF src → ABUF dst | Same as COPY with Q; n8 is a multiple of 4 |
| 14 | RELU | src → dst | max(x, 0) |
| 15 | MAX | src, src2 → dst | Element-wise max |
| 16 | RUN | — | Starts the CBUF program at aux |
| 17–31 | — | — | Runs the program of op − 16; with Q, the scratch area is `imm[11:0]` |

Q flag: the final pass is quantized and written to ABUF; dst is an ABUF address and n8 is a multiple of 4. The FBUF scratch area for intermediate passes is src2 for LAYERNORM, RMSNORM and SCALE, and `imm[11:0]` for the other operations.

Pass structure (→ writes; the last write goes to ABUF with Q):

| Operation | Passes |
|---|---|
| ADD, MUL | `x + x2 →`; `x × x2 →` |
| ADDP, MULP, SCALE, COPY | `x + P →`; `x × P →`; `x × imm →`; `x →` |
| LAYERNORM | 1: Σx; 2: Σ(x − mean)²; 3: `(x − mean) × rstd →` scratch; 4: `scratch × γ →` scratch; 5: `scratch + β →` |
| RMSNORM | 1: Σx²; 2: `x × rstd →` scratch; 3: `scratch × γ →` |
| GELU, SILU | 1: `σ(|x|) →` scratch; 2: `x < 0 ? −(x × scratch) + x : x × scratch →` |
| SILU_MUL | 1, 2 as SILU into scratch; 3: `scratch × up →` |
| SOFTMAX | 1: max; 2: `2^((x − max) × log2e) →` scratch and Σ; 3: `scratch × (1/Σ) →` |
| ROPE (per head; words g and g + 4 pair up) | 1: `x2 × sin →` scratch[g]; 2: `x1 × cos − scratch[g] →` dst[g]; 3: `x1 × sin →` scratch[g + 4]; 4: `x2 × cos + scratch[g + 4] →` dst[g + 4] |
| RELU, MAX | `max(x, 0) →`; `max(x, x2) →` |

LAYERNORM, RMSNORM and SOFTMAX run their passes quad by quad; the other operations run each pass over all quads. Passes follow the execution rules of VEC programs.

## VEC programs

RUN loads its operand fields into scalar registers:

| Field | Register |
|---|---|
| src | r1 |
| src2 | r2 |
| dst | r3 |
| rows_q | r4 |
| n8 | r5 |
| p_base | r6 |
| aux[9:0] | pc (start address, CBUF word) |
| 0 | r7 |
| imm | r8 |
| scratch (as in the Q-flag rule; dst without Q) | r9 |
| Q flag | r10 |

For op 0–15 the same registers are loaded, except `r7 = aux` and pc is the start of the operation's program in CBUF [0, 256). r0 is always 0. r11–r31, RS0, RS1 and rn are not initialized. Programs do not write r10. A program ends with RET, which clears the engine's busy state without waiting for the drain.

Instruction word, `[127:124]` kind: 0 PASS, 1 SALU, 2 SMEM, 3 BR, 4 RET, 5 NOP, 7 TRANSPOSE8; 6 and 8–15 act as NOP.

### PASS

A PASS streams `rq` quads × `rnw` words through the lanes, 4 cycles per word. Read and write addresses come from registers; each quad adds the row stride `rn8` (`rn8/4` for ABUF writes). The low 12 bits of rx and rx2, 13 of rd (12 for FBUF writes), 9 of rq, 10 of rnw, 8 of rn8 and 10 of rpb are used.

| Bits | Field | Meaning |
|---|---|---|
| [123:119] | rx | Register: x start (FBUF word) |
| [118:114] | rx2 | Register: x2 start |
| [113:109] | rd | Register: write start (FBUF or ABUF) |
| [108:104] | rq | Register: number of quads |
| [103:99] | rnw | Register: words per row |
| [98:94] | rn8 | Register: row stride (words) |
| [93:89] | rpb | Register: PBUF start |
| [88:84] | rimm | Register: immediate for b, c0 or sel = IMM, and ε for sc = RSTD |
| [83:79] | rimm2 | Register: comparison value, or valid-column count for mask_en (low 16 bits) |
| [78] | x2_en | Read x2; with x2_en = 0, x2 is x |
| [77:76] / [75:74] / [73:72] | wx / wx2 / wd | Word-index mapping: 0 NORM `w`; 1 X1 `h·8 + gg`; 2 X2 `h·8 + 4 + gg` (h = w[5:2], gg = w[1:0]); 3 S4 `4·w` |
| [71:70] | pa | PBUF access: 0 NONE; 1 ROWVEC (word `rpb + w/4`, slice `w mod 4`); 2 TILE (word `rpb + q·4 + gg`); 3 as NONE |
| [69] | pa_at_mul | PBUF used at the multiplier (0: at the pre-adder) |
| [68:66] | c0 | 0 ZERO, 1 X2, 2 PBUF, 3 RS0, 4 IMM; 5–7 as ZERO |
| [65:63] | b | 0 ONE, 1 X2, 2 PBUF, 3 RS1, 4 IMM, 5 PRE (square), 6 LOG2E; 7 as ONE |
| [62:61] | nl | 0 NONE, 1 ACT, 2 EXP2, 3 TAB01 |
| [60:59] | c1 | 0 ZERO, 1 X2, 2 MAX, 3 SIGNFIX |
| [58] | neg_c1 | Negate c1; with c1 = MAX, compare with 0 instead of x2 |
| [57] | mask_en | Columns at and beyond the rimm2 value give 0 |
| [56:55] | red | Reduction: 0 NONE, 1 SUM, 2 MAX; 3 as NONE |
| [54:53] | wr | Write: 0 NONE, 1 FBUF, 2 ABUF (quantized), 3 FINAL (ABUF if r10 = Q, else FBUF) |
| [52:50] | sc | Row scalar after the pass: 0 NONE, 1 MEAN, 2 RSTD, 3 NEGMAX, 4 and 7 RECIPSUM, 5 SUM, 6 MAX |
| [49:47] | px | Permutation of x: 0 NONE; 1 BCAST (column pcol of each token to all 8 columns); 2 TSWAP1 (token t ↔ t xor 1); 3 TSWAP2 (t ↔ t xor 2); 4 T4 (output lane l ← token `l mod 4`, column `l/4`); 5 T4B (output lane l ← token `(l/4) mod 4`, column `(l mod 4) + 4·(l/16)`); 6 and 7 as NONE |
| [46:44] | — | Reserved |
| [43:41] | pcol | Column for BCAST |
| [40:38] | cmp | 0 NONE, 1 LT, 2 LE, 3 EQ, 4 NE, 5 GT, 6 GE; 7 as NONE |
| [37:36] | cmp_src | 0 IMM2 (x vs rimm2), 1 X2 (x vs x2), 2 LANE (lane number vs rimm2), 3 X2_IMM2 (x2 vs rimm2) |
| [35] | cmp_int | Integer comparison (0: FP32) |
| [34:32] | sel | Value for lanes whose predicate is false: 0 NONE, 1 ZERO, 2 X2, 3 IMM, 4 NULL (0xFFFFFFFF), 5 RS0, 6 RS1; 7 as ZERO |
| [31:29] | ialu | Integer path: 0 NONE, 1 ADD, 2 SUB, 3 AND, 4 OR, 5 XOR, 6 SHR1 (logical shift right by 1), 7 RSUB (c0 − x) |
| [28] | dst_s1 | FBUF write start advances by 1 word per quad |
| [27] | c0_pred | c0 is 0 in lanes whose predicate is false |
| [26] | nodrain | Do not wait for the pipeline to drain before the next instruction |
| [25] | x_s1 | x and x2 read starts advance by 1 word per quad |

Lane operation (x, x2 after permutation):

```
pred = cmp == NONE ? 1 : rel(left, right)
c0'  = c0_pred && !pred ? 0 : c0
ialu == NONE:  pre = x + c0' ;  prod = pre × b ;  y0 = nl ? table(prod) : post(prod)
ialu != NONE:  y0 = ialu(x, bits(c0'))
y2   = sel == NONE ? y0 : (pred ? y0 : sel value)
y    = mask_en && invalid column ? 0 : y2
```

- `rel` compares x (x2 for X2_IMM2, the lane number for LANE) with rimm2 (IMM2, X2_IMM2, LANE) or x2 (X2). FP32 comparison uses a sign-magnitude key, with −0 and subnormal values equal to +0. With cmp_int, and always for LANE, int32 signed comparison. For LANE, the lane number is the column 0..7 (cmp_int = 0) or the word position 0..31 = token · 8 + column (cmp_int = 1).
- `post(prod)`: c1 = ZERO or X2 gives `prod + c1`, or `prod − c1` with neg_c1; c1 = MAX gives `max(prod, x2)`, or `max(prod, 0)` with neg_c1; c1 = SIGNFIX gives `−prod + x` when the sign bit of x is set, else `prod`.
- The integer path bypasses the multiplier, the nonlinear table and the post-adder.
- TAB01 interpolates the first half of TBUF at the fraction of prod in [0, 1).
- An invalid column is one whose position `8·w + column` in the row is at or beyond the rimm2 value. For a MAX reduction, invalid columns count as −∞; for a SUM reduction, as 0.
- Row scalars after the pass (rn set by SMEM RN, ε = rimm value): MEAN `RS0 = −(Σ × rn)`; RSTD `RS1 = 1/√(Σ × rn + ε)`; NEGMAX `RS0 = −max`; RECIPSUM `RS1 = 1/Σ`; SUM `RS1 = Σ`; MAX `RS1 = max`. The negation flips the sign bit and gives +0 for ±0.
- NaN and the value 0xFFFFFFFF must not enter the FP32 path or an FP32 comparison; integer and NULL entries go through the integer path and sel only.
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
| [88] | pack | 0 NONE; 1 ZIP4 |
| [87:0] | — | Reserved |

Bases and strides use the low 12 bits of their registers and are measured in FBUF words. The tile count uses the low 9 bits. Each tile contains four independent 8×8 matrices of 32-bit elements. For tile q, source row r is read from `rx + q·rst + r·rsw`, and destination row c is written to `rd + q·rdt + c·rdw`. All word addresses wrap at 4,096 words. Element `(t, c)` of source row r is at bits `32·(8·t + c) +: 32`, where t selects one of the four matrices.

With pack = NONE, destination row c, element `(t, r)` receives source row r, element `(t, c)`. With pack = ZIP4, the destination columns are selected in the order 0, 4, 1, 5, 2, 6, 3, 7 from the transposed row. Every 32-bit element is copied unchanged. The source and destination word sets are disjoint, and destination word addresses are distinct. A zero tile count performs no reads or writes.

TRANSPOSE8 waits for preceding vector operations to drain and completes all destination writes before advancing to the next instruction. The two tile buffers hold 8,192 bits each in eight skewed distributed-RAM banks. A registered output word and address remain stable during a denied FBUF write. A tile is released after its final word is written. RS0, RS1 and the scalar register file are unchanged.

### SALU

```
[123:119] op  [118:114] rd  [113:109] ra  [108:104] rb  [103] use_imm  [31:0] imm
```

`rd = ra op (use_imm ? imm : rb)`. op: 0 ADD, 1 SUB, 2 AND, 3 OR, 4 XOR, 5 SHL, 6 SHR, 7 SRA, 8 MUL (low 32 bits), 9 SLT (signed), 10 SLTU, 11 SEQ, 12 MOV (`rd = use_imm ? imm : rb`); 13–31 as MOV. Shifts use the low 5 bits of the second operand. Writes to r0 are discarded.

### SMEM

```
[123:121] op  [120:116] rd  [115:111] ra  [110:106] rb  [105] col_imm  [104:100] col  [99] which (0 RS1, 1 RS0)  [98] neg  [31:0] imm (signed word offset)
```

| op | Name | Operation |
|---|---|---|
| 0 | SLD | `rd ← FBUF[ra + imm]` lane col (col_imm ? col : rb[4:0]) |
| 1, 3, 7 | — | No operation |
| 2 | RSLD | `RS[which][t] ← FBUF[ra + imm]` token t, column col[2:0], t = 0..3; negated if neg; a value with a zero exponent field becomes +0 |
| 4 | RSMOV | `rd ← RS[which][col[1:0]]` |
| 5 | SMOVR | `RS[which][t] ← ra` for all t |
| 6 | RN | `rn ← 1 / float(ra)`, ra < 2^14; ra = 0 counts as 1 |

The FBUF address `ra + imm` wraps at 4,096 words.

### BR

```
[123:122] cond (0 ALWAYS, 1 Z: ra == 0, 2 NZ: ra != 0, 3 never)  [121:117] ra  [15:0] target (CBUF word; bits [9:0] are used)
```

RET and NOP have no operands.

### Execution rules

- A PASS waits for the pipeline to drain at its end unless nodrain is set, or unless it has no red and no sc, does not write ABUF, the next instruction is a PASS, and it does not read PBUF at the multiplier while the next PASS reads PBUF at the pre-adder. In the latter case the next PASS starts while the writes of this one complete, and a read of an FBUF word whose write is still pending waits for the write. With nodrain, sc has no effect.
- SMEM waits for preceding passes to drain.
- PASS, RSLD and SMOVR instructions that write RS0 or RS1 wait for the drain.
- The rimm and rimm2 values are captured when a PASS is issued.
- A PASS must not have rq = 0 or rnw = 0.
- sel = X2 and c1 = SIGNFIX are not used in the same pass.
- pa = TILE is used with pa_at_mul = 1.
- A PASS that follows a nodrain PASS with red does not use red.
- A PASS that follows a nodrain PASS reading PBUF at the multiplier does not read PBUF at the pre-adder.
- The last PASS before RET does not set nodrain.

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

`ddr_addr` and `ddr_stride` are multiples of 16 bytes. DMA_ST reads ABUF or FBUF and treats every mode as RAW. DMA_ST reads ABUF through the write port of ABUF in the cycles without a write from VEC or DMA, so it does not wait for the GEMM engine. DMA writes to TBUF and CBUF wait until the VEC engine is idle.

### RAW (2D)

Row r = 0..rows−1, word w = 0..row_words−1: DDR `ddr_addr + r · ddr_stride + w · (word bytes)` ↔ on-chip `dst + r · row_words + w`.

### Index list (flag I)

`ddr_addr` points to a list of `rows · row_words` 32-bit entries. Entry e is the DDR address, in 16-byte units, of transfer unit e (a word for RAW, a token for row format). On-chip units go to `dst + e` in order. The entry 0xFFFFFFFF reads nothing and writes zero (for MXINT8, an all-zero block with scale byte 127). For DMA_ST, the entries give the write addresses, and the entry 0xFFFFFFFF skips its unit.

### Row format (flag R)

A token is contiguous in DDR:

| Format | One token | Size |
|---|---|---|
| MXINT8 (to and from ABUF) | `tok` blocks of 32 bytes, then `tok` scale bytes, zero-padded to 16 bytes | `32·tok + 16·ceil(tok/16)` bytes |
| FP32 (to and from FBUF) | `32·tok` FP32 values | `128·tok` bytes |

- `tok` ([6:0]) is the number of 32-element blocks per token: at most 32 for MXINT8 and 8 for FP32. 0 counts as 1 and values above 32 as 32.
- `row_words · (token size / 16) < 65,536`.
- DMA_LD with R: row r, token w is at `ddr_addr + r · ddr_stride + w · (token size)`; each 4 consecutive tokens are packed into one quad of on-chip words. `rows · row_words` is a multiple of 4.
- DMA_ST with R: each on-chip quad is unpacked into 4 tokens.

### SPLIT8

Converts an ABUF-format matrix into the two W8 planes in WBUF (mem = WBUF). Rows 2p and 2p + 1 (8 tokens) are taken together at the same word position w. Token j (0..3) of row 2p + r becomes column c = 4·r + j. The high nibble of each INT8 (signed, `q = 16·hi + lo`) goes to the HI plane, the low nibble to the LO plane; scales are copied. HI word `dst + p · W + w`, LO word `dst + nW + p · W + w`, with W = row_words and `nW = rows · W / 2`. `rows` is even. Flags I and R are not used.

### T32_SPLIT8

Converts an ABUF-format matrix into the W8 planes of its transpose (mem = WBUF). For each group of 8 rows (32 tokens) r8 and word position w, the 32-token × 32-dimension block is transposed: dimension d becomes column `c = d mod 8` of column group `cg = d / 8`, and the token becomes the K position. The token scales are aligned: `emax` is the largest `e_t` among the tokens of the block that have a nonzero element (`e = 0` if none); each element becomes `RNE(q / 2^(emax − e_t))`, which is 0 when `emax − e_t ≥ 9`; all 8 output scales are `emax`. The word for (w, cg, r8) is `dst + (4·w + cg) · KBt + r8` with `KBt = rows / 8`; the LO plane adds `nW = rows · W / 2`. `rows` is a multiple of 8. Flags I and R are not used.

### GACC (mode 3)

Builds the output rows of one quad (4 tokens), `out[t][ch]` with `ch = 0..32·row_words − 1`, from MXINT8 rows in row format:

```
out[t][ch] = Σ_{s = 0..S−1} Σ_{k = 0..3} w[t][kb(ch)][s] · bw[s][t][k] · row(addr[s][t][k])[ch]      kb(ch) = ch / 32
```

| Operand | Meaning |
|---|---|
| ddr_addr | List address. The first 16 bytes are a header: [31:0] address of w (16-byte units), [47:32] stride between the w areas of consecutive blocks kb (words), [63:48] words per quad row of w (S / 8) |
| rows | Number of slots S (≤ 2047) |
| row_words | Blocks per token, K / 32 (1 to 8) |
| dst | FBUF address of the output quad row (4 · row_words words) |
| ddr_stride | [15:0] quad number q of the output |

- Entries follow the header, 8 bytes each: [31:0] row address (16-byte units; 0xFFFFFFFF adds nothing), [63:32] bw (FP32). Order: `for s: for t: for k`.
- `w[t][kb][s]` is element `t·8 + (s mod 8)` of the FBUF-format word at `w address + (kb · stride + q · quad-row words + s/8) · 128` bytes.
- flags: acc = 0 starts the accumulators at 0; flush = 1 writes them to dst at the end. Larger S, and lists per camera, are chained with acc = 1.
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
| 0x00 | CTRL (W) | bit 0 START: run INSTR_COUNT instructions from INSTR_BASE (a START while running restarts the stream); bit 1 SOFT_RESET: stop issue and fetch, empty the IBUF and clear running, halted and error (registers, counters and engines are kept) |
| 0x04 | STATUS (R) | bit 0 running, 1 halted, 2 error, 3 waiting for host, 4 GEMM busy, 5 VEC busy, 6 DMA busy, 7 IBUF empty |
| 0x08 | INSTR_BASE | DDR address of the instruction stream (256-byte aligned) |
| 0x0C | INSTR_COUNT | Number of instructions |
| 0x10 | IRQ_STATUS (R, write 1 to clear) | bit 0 SIGNAL, 1 HALT, 2 illegal instruction, 3 AXI error |
| 0x14 | IRQ_ENABLE | Interrupt output = IRQ_STATUS & IRQ_ENABLE |
| 0x18 | SIGNAL | [7:0] last id, [31:16] count |
| 0x1C | SIGNAL_VALUE | Last value |
| 0x20 | Issued instructions | Since START |
| 0x24 | Cycles | From START to HALT |
| 0x28 / 0x2C / 0x30 | GEMM / VEC / DMA busy cycles | |
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
[31:0] instruction index (issue order)  [63:32] issue cycle  [95:64] completion cycle  [99:96] source (0 DMA, 1 GEMM, 2 VEC, 3 sequencer)  [107:100] opcode  [111:108] wait  [115:112] flags  [119:116] 0  [127:120] 0
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
| VEC | cycles in the issue state and in TRANSPOSE8 | cycles in the drain states | cycles waiting for the scalar unit | PASS instructions started |

Each source holds one pending completion record with its statistics record. One record per cycle moves into the 64-entry record FIFO: the completion record of the selected source, then its statistics record when enabled; when several sources are pending, the order is DMA, GEMM, VEC, sequencer. Issue is held while the source's record slot is occupied or the record FIFO holds 56 or more records. TRACE_COUNT counts the records of both types.

## Board integration

| Item | Configuration |
|---|---|
| Block design | Zynq UltraScale+ PS (KV260 SOM preset) and the NPU, referenced as a module through a Verilog wrapper |
| Control registers | M_AXI_HPM0_FPD (128 bits) → SmartConnect → NPU AXI-Lite (32 bits), at 0xA000_0000, 4 KB |
| DMA | NPU AXI4 master 0 → SmartConnect → S_AXI_HP0_FPD (128 bits) |
| Instruction fetch and trace | NPU AXI4 master 1 → SmartConnect → S_AXI_HP1_FPD (128 bits) |
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

The compiler writes a directory with `image.bin`, `meta.json` and one `<stream>.listing.json` per instruction stream. `meta.json` holds the base address, the main stream, the stream table, the listing file names, the parameters and the regions of the image; the `trace` region, allocated after the streams, holds 32 bytes per instruction of the longest stream. A listing holds the base address, the tag table and one entry per instruction with its index, engine, opcode, wait bits, flags, hexadecimal word, tag index, note, and the on-chip ranges it reads and writes. Tags are the scopes open when the instruction was emitted, joined with `/`.

### Scheduling

Kernels emit instructions in program order with the on-chip ranges each instruction reads and writes. Two instructions conflict when one writes a range the other reads or writes, when both are DMA transfers and at least one is a store whose DDR extent overlaps the other's extent (the extent of an index-list or GACC transfer is all of DDR), when both are GACC transfers, when both are VEC commands and one of them is RUN, or when one of them is SIGNAL, WAIT_HOST or HALT. The compiler reorders each stream so that conflicting instructions keep their program order, choosing at every step, among the first `schedule-window` unscheduled instructions in program order whose conflicting predecessors are all scheduled, the one with the earliest predicted issue cycle under the sequencer rules and the cycle model, with program order as the tie-break. The wait bits are then recomputed for the new order: an instruction waits for an engine when a conflicting instruction of that engine was issued after the last point at which that engine was known to be idle. `--schedule-window 0` keeps the program order; the window is recorded in `meta.json`.

### Cycle model

The simulator assigns each instruction an issue cycle and a completion cycle with the sequencer rules: in-order issue one cycle after the previous issue at the earliest, an instruction waits for the completion of every engine named by its wait bits and of its own engine, and a DMA load to TBUF or CBUF starts its data phase when the VEC engine is free. The first instruction issues at `start_cycles`. The cost of a command is a linear function of the activity counted while the simulator executes it:

| Engine | Cycles |
|---|---|
| GEMM | `n8 · (m/32) · kb · 8 + gemm_fixed_cycles` |
| DMA | `dma_fixed_cycles + read beats · dma_read_cycles_per_kilobeat / 1024 + read bursts · dma_read_burst_cycles + write beats · dma_write_cycles_per_kilobeat / 1024 + write bursts · dma_write_burst_cycles + gather entries · dma_gather_entry_cycles + gathered rows · dma_gather_row_cycles` |
| VEC | `vec_fixed_cycles + passes · vec_pass_cycles + issue + drain + scalar + SALU · vec_salu_cycles + SMEM · vec_smem_cycles + branches · vec_branch_cycles` |

For VEC, `issue = (words + stalled words) · vec_word_cycles − passes · vec_issue_tail_cycles + transposes · vec_transpose_cycles + tiles · vec_transpose_tile_cycles`, where a word whose read finds a write of an earlier overlapped pass pending is stalled until 10 words after the issue of that write, `drain = draining passes · vec_drain_cycles + reducing passes · vec_drain_reduction_cycles + quantizing passes · vec_drain_quantized_cycles`, and the scalar cost follows the scalar unit's programs: each arithmetic micro-operation costs `vec_scalar_arithmetic_cycles`, each immediate write `vec_scalar_immediate_cycles`, and each program start `vec_scalar_start_cycles`, with the micro-operation counts of the RTL:

| Program | Arithmetic | Immediate |
|---|---:|---:|
| MEAN | 8 | 1 |
| RSTD | 29 | 2 |
| NEGMAX, MAX | 0 | 1 |
| RECIPSUM | 19 | 2 |
| SUM | 7 | 1 |
| RN | 12 | 2 |

The statistics of a predicted record are the read and write beats, the cycles between issue and the start of the data phase and the null entries for DMA; the array cycles for GEMM; and the issue, drain and scalar cycles and the pass count for VEC. The host's response to SIGNAL and WAIT_HOST takes no time in the model, and the trace back-pressure and the contention for the FBUF write port between the engines are not modelled. The default parameters are the values that the trace analyzer calibrates from board traces with statistics records. `hotarubi_trace predict` and `hotarubi_run --device iss` take `--timing NAME=VALUE,...` to override any parameter. The scheduler estimates the cost of an instruction with the same model from the activity implied by its fields (the activity of a fixed VEC operation by executing its ROM program; a RUN command as one pass over its words).

### Hazard check

While executing a stream, the simulator records the on-chip words every command reads and writes and reports a hazard for each pair of commands on different engines whose accesses conflict (a write and a read or two writes of the same word) without an ordering between them. A command is ordered after an earlier command when a wait bit of it or of any command issued between them names the earlier command's engine after that command was issued, when the engines are in order and the earlier command's engine has issued a later command (a command issues only when its engine is free), or when the earlier command is a VEC command and the later one is a DMA write to TBUF or CBUF. `hotarubi_trace predict` and `hotarubi_run --device iss` fail on a stream with hazards and list them. `hotarubi_trace predict` also checks that the ranges listed for every instruction cover the words it actually reads and writes (the fetch of a fixed VEC operation from the ROM part of CBUF is implied) and fails otherwise; with `--accesses` it writes `<stream>.accesses.jsonl`, one line per instruction with the extent of its actual reads and writes in each memory.

## Build

CMake is the single entry point:

- It builds the toolchain and the Verilator testbenches, and runs the tests with CTest.
- It runs the Vivado steps (synthesis, implementation, bitstream) through custom targets that call Tcl scripts in `fpga/`.
- The list of RTL sources is defined once in CMake and used by both Verilator and Vivado.
- The runtime is cross-compiled for aarch64 and linked statically.

## RTL verification

The RTL is simulated with Verilator. The testbenches are written in C++.

The kernel bench runs a window of a compiled instruction stream on the RTL: it loads the image into the simulated DDR, appends the chosen preamble instructions, the window and a HALT as a new stream after the image, records the trace with the image's trace region, and writes a compiled directory (`image.bin` with the DDR contents before the run, `meta.json`, `kernel.listing.json`) together with `kernel.trace.bin` and `kernel.counters.json`.
