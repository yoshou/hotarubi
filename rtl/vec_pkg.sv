package vec_pkg;

  import isa_pkg::*;

  localparam int Lanes = 8;
  localparam int Hw    = Lanes * 32;

  localparam int PreLatency    = 7;
  localparam int MulLatency    = 7;
  localparam int FixLatency    = 2;
  localparam int TabLatency    = 2;
  localparam int InterpLatency = 4;
  localparam int PostLatency   = 7;
  localparam int FinLatency    = 1;
  localparam int StageB        = PreLatency;
  localparam int StageNl       = PreLatency + MulLatency;
  localparam int StageTabAddr  = StageNl + FixLatency;
  localparam int StageTabData  = StageTabAddr + TabLatency;
  localparam int StageC1       = StageTabData + InterpLatency;
  localparam int StageY        = StageC1 + PostLatency + FinLatency;
  localparam int MemLatency    = 3;
  localparam int CbufLatency   = 2;
  localparam int NumContexts   = 2;
  localparam int CtxIdxW       = 1;
  localparam int AccAw         = CtxIdxW + 5;
  localparam int CmdRegs       = int'(VecRegQ);

  typedef enum logic [1:0] {
    C1Zero    = 2'd0,
    C1X2      = 2'd1,
    C1Max     = 2'd2,
    C1Signfix = 2'd3
  } c1_e;

  typedef struct packed {
    sf_e  fn;
    logic rs0;
    logic rn;
    logic eps;
    logic max;
  } sc_t;

  typedef struct packed {
    logic [4:0] rx, rx2, rd, rq, rnw, rn8, rpb, rimm, rimm2;
  } pass_regs_t;

  typedef struct packed {
    pass_regs_t  r;
    logic        x2_en;
    logic [1:0]  mg, ms;
    mm_e         mx, mx2, md;
    pa_e         pa;
    c0_e         c0;
    b_e          b;
    nl_e         nl;
    po_e         po;
    logic        pv;
    red_e        red;
    wr_e         wr;
    sf_e         sf;
    logic        st;
    logic        srn;
    logic        seps;
    logic [19:0] px;
    cmp_e        cmp;
    cl_e         cl;
    logic        cr;
    logic        cmp_int;
    sel_e        sel;
    ia_e         ialu;
    logic        c0_pred;
    logic        x_s1;
    logic        dst_s1;
    logic [CtxIdxW-1:0] ctx;
    rm_e         rm;
  } pass_t;

  function automatic mm_e dec_map(input logic [1:0] v);
    return (v > MmUpper) ? MmNone : mm_e'(v);
  endfunction

  function automatic pass_t dec_pass(input logic [127:0] w);
    pass_t d;
    logic  unused_w;
    unused_w  = ^{w[ProgKindMsb:ProgKindLsb], w[PassRmLsb-1:0]};
    d.r       = pass_regs_t'(w[PassRxMsb:PassRimm2Lsb]);
    d.x2_en   = w[PassX2EnLsb];
    d.mg      = w[PassMgMsb:PassMgLsb];
    d.ms      = w[PassMsMsb:PassMsLsb];
    d.mx      = dec_map(w[PassMxMsb:PassMxLsb]);
    d.mx2     = dec_map(w[PassMx2Msb:PassMx2Lsb]);
    d.md      = dec_map(w[PassMdMsb:PassMdLsb]);
    d.pa      = (w[PassPaMsb:PassPaLsb] > PaTile) ? PaNone : pa_e'(w[PassPaMsb:PassPaLsb]);
    d.c0      = (w[PassC0Msb:PassC0Lsb] > C0Imm) ? C0Zero : c0_e'(w[PassC0Msb:PassC0Lsb]);
    d.b       = (w[PassBMsb:PassBLsb] > BPre) ? BOne : b_e'(w[PassBMsb:PassBLsb]);
    d.nl      = nl_e'(w[PassNlMsb:PassNlLsb]);
    d.po      = po_e'(w[PassPoMsb:PassPoLsb]);
    d.pv      = w[PassPvLsb];
    d.red     = (w[PassRedMsb:PassRedLsb] > RedMax) ? RedNone : red_e'(w[PassRedMsb:PassRedLsb]);
    d.wr      = wr_e'(w[PassWrMsb:PassWrLsb]);
    d.sf      = (w[PassSfMsb:PassSfLsb] > SfRsqrt) ? SfNone : sf_e'(w[PassSfMsb:PassSfLsb]);
    d.st      = w[PassStLsb];
    d.srn     = w[PassSrnLsb];
    d.seps    = w[PassSepsLsb];
    d.px      = w[PassPxMsb:PassPxLsb];
    d.cmp     = (w[PassCmpMsb:PassCmpLsb] > CmpGe) ? CmpNone : cmp_e'(w[PassCmpMsb:PassCmpLsb]);
    d.cl      = (w[PassClMsb:PassClLsb] > ClRowColumn) ? ClX : cl_e'(w[PassClMsb:PassClLsb]);
    d.cr      = w[PassCrLsb];
    d.cmp_int = w[PassCmpIntLsb];
    d.sel     = (w[PassSelMsb:PassSelLsb] > SelRs1) ? SelNone : sel_e'(w[PassSelMsb:PassSelLsb]);
    d.ialu    = ia_e'(w[PassIaluMsb:PassIaluLsb]);
    d.c0_pred = w[PassC0PredLsb];
    d.x_s1    = w[PassXS1Lsb];
    d.dst_s1  = w[PassDstS1Lsb];
    d.ctx     = w[PassCtxMsb:PassCtxLsb];
    d.rm      = (w[PassRmMsb:PassRmLsb] > RmSet) ? RmNone : rm_e'(w[PassRmMsb:PassRmLsb]);
    return d;
  endfunction

  typedef struct packed {
    logic        valid;
    logic [1:0]  ph;
    logic [CtxIdxW-1:0] ctx;
    c0_e         c0;
    b_e          b;
    nl_e         nl;
    c1_e         c1;
    logic        neg_c1;
    logic [9:0]  w;
    red_e        red;
    wr_e         wr;
    logic [12:0] waddr;
    logic [2:0]  j;
    logic [1:0]  qw;
    logic [1:0]  pa_slice;
    logic        pa_tile;
    logic [19:0] px;
    cmp_e        cmp;
    cl_e         cl;
    logic        cr;
    logic        cmp_int;
    sel_e        sel;
    ia_e         ialu;
    logic        c0_pred;
    logic        rm_set;
    logic        row_end;
    logic [8:0]  row;
    logic [31:0] imm;
    logic [31:0] imm2;
  } ctrl_t;

  localparam logic [31:0] FpOne          = 32'h3F800000;
  localparam logic [31:0] FpTwo          = 32'h40000000;
  localparam logic [31:0] FpHalf         = 32'h3F000000;
  localparam logic [31:0] FpOnePointFive = 32'h3FC00000;
  localparam logic [31:0] FpNegInf       = 32'hFF800000;

endpackage
