package vec_pkg;

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
  localparam int CtxW          = 2;
  localparam int CtxIdxW       = (NumContexts > 1) ? $clog2(NumContexts) : 1;
  localparam int AccAw         = CtxIdxW + 5;

  typedef enum logic [4:0] {
    OpAdd       = 5'd0,
    OpMul       = 5'd1,
    OpAddP      = 5'd2,
    OpMulP      = 5'd3,
    OpScale     = 5'd4,
    OpCopy      = 5'd5,
    OpLayerNorm = 5'd6,
    OpRmsNorm   = 5'd7,
    OpGelu      = 5'd8,
    OpSilu      = 5'd9,
    OpSiluMul   = 5'd10,
    OpSoftmax   = 5'd11,
    OpRope      = 5'd12,
    OpQuant     = 5'd13,
    OpRelu      = 5'd14,
    OpMax       = 5'd15,
    OpRun       = 5'd16
  } op_e;

  typedef enum logic [1:0] {
    PaNone   = 2'd0,
    PaRowvec = 2'd1,
    PaTile   = 2'd2
  } pa_e;

  typedef enum logic [2:0] {
    C0Zero = 3'd0,
    C0X2   = 3'd1,
    C0Pbuf = 3'd2,
    C0Rs0  = 3'd3,
    C0Imm  = 3'd4
  } c0_e;

  typedef enum logic [2:0] {
    BOne   = 3'd0,
    BX2    = 3'd1,
    BPbuf  = 3'd2,
    BRs1   = 3'd3,
    BImm   = 3'd4,
    BPre   = 3'd5,
    BLog2e = 3'd6
  } b_e;

  typedef enum logic [1:0] {
    NlNone  = 2'd0,
    NlAct   = 2'd1,
    NlExp2  = 2'd2,
    NlTab01 = 2'd3
  } nl_e;

  typedef enum logic [1:0] {
    C1Zero    = 2'd0,
    C1X2      = 2'd1,
    C1Max     = 2'd2,
    C1Signfix = 2'd3
  } c1_e;

  typedef enum logic [1:0] {
    RedNone = 2'd0,
    RedSum  = 2'd1,
    RedMax  = 2'd2
  } red_e;

  typedef enum logic [1:0] {
    WrNone  = 2'd0,
    WrFbuf  = 2'd1,
    WrAbuf  = 2'd2,
    WrFinal = 2'd3
  } wr_e;

  typedef enum logic [2:0] {
    ScNone     = 3'd0,
    ScMean     = 3'd1,
    ScRstd     = 3'd2,
    ScNegmax   = 3'd3,
    ScRecipsum = 3'd4,
    ScSum      = 3'd5,
    ScMax      = 3'd6
  } sc_e;

  typedef enum logic [1:0] {
    MapNorm = 2'd0,
    MapX1   = 2'd1,
    MapX2   = 2'd2,
    MapS4   = 2'd3
  } rw_e;

  typedef enum logic [2:0] {
    PxNone   = 3'd0,
    PxBcast  = 3'd1,
    PxTswap1 = 3'd2,
    PxTswap2 = 3'd3,
    PxT4     = 3'd4,
    PxT4b    = 3'd5
  } px_e;

  typedef enum logic [2:0] {
    CmpNone = 3'd0,
    CmpLt   = 3'd1,
    CmpLe   = 3'd2,
    CmpEq   = 3'd3,
    CmpNe   = 3'd4,
    CmpGt   = 3'd5,
    CmpGe   = 3'd6
  } cmp_e;

  typedef enum logic [1:0] {
    CmpSrcImm2   = 2'd0,
    CmpSrcX2     = 2'd1,
    CmpSrcLane   = 2'd2,
    CmpSrcX2Imm2 = 2'd3
  } cs_e;

  typedef enum logic [2:0] {
    SelNone = 3'd0,
    SelZero = 3'd1,
    SelX2   = 3'd2,
    SelImm  = 3'd3,
    SelNull = 3'd4,
    SelRs0  = 3'd5,
    SelRs1  = 3'd6
  } sel_e;

  typedef enum logic [1:0] {
    RmNone = 2'd0,
    RmSkip = 2'd1,
    RmSet  = 2'd2
  } rm_e;

  typedef enum logic [2:0] {
    IaluNone = 3'd0,
    IaluAdd  = 3'd1,
    IaluSub  = 3'd2,
    IaluAnd  = 3'd3,
    IaluOr   = 3'd4,
    IaluXor  = 3'd5,
    IaluShr1 = 3'd6,
    IaluRsub = 3'd7
  } ia_e;

  typedef struct packed {
    logic [4:0] rx, rx2, rd, rq, rnw, rn8, rpb, rimm, rimm2;
    logic       x2_en;
    rw_e        wx, wx2, wd;
    pa_e        pa;
    logic       pa_at_mul;
    c0_e        c0;
    b_e         b;
    nl_e        nl;
    c1_e        c1;
    logic       neg_c1;
    logic       mask_en;
    red_e       red;
    wr_e        wr;
    sc_e        sc;
    px_e        px;
    logic [2:0] pcol;
    cmp_e       cmp;
    cs_e        cmp_src;
    logic       cmp_int;
    sel_e       sel;
    ia_e        ialu;
    logic       dst_s1;
    logic       x_s1;
    logic       c0_pred;
    logic       nodrain;
    logic [CtxW-1:0] ctx;
    rm_e        rm;
  } pass_t;

  function automatic pass_t dec_pass(input logic [127:0] w);
    pass_t d;
    logic  unused_w;
    unused_w = ^{w[127:124], w[46:44], w[20:0]};
    d.rx        = w[123:119];
    d.rx2       = w[118:114];
    d.rd        = w[113:109];
    d.rq        = w[108:104];
    d.rnw       = w[103:99];
    d.rn8       = w[98:94];
    d.rpb       = w[93:89];
    d.rimm      = w[88:84];
    d.rimm2     = w[83:79];
    d.x2_en     = w[78];
    d.wx        = rw_e'(w[77:76]);
    d.wx2       = rw_e'(w[75:74]);
    d.wd        = rw_e'(w[73:72]);
    d.pa        = pa_e'(w[71:70]);
    d.pa_at_mul = w[69];
    d.c0        = c0_e'(w[68:66]);
    d.b         = b_e'(w[65:63]);
    d.nl        = nl_e'(w[62:61]);
    d.c1        = c1_e'(w[60:59]);
    d.neg_c1    = w[58];
    d.mask_en   = w[57];
    d.red       = red_e'(w[56:55]);
    d.wr        = wr_e'(w[54:53]);
    d.sc        = sc_e'(w[52:50]);
    d.px        = px_e'(w[49:47]);
    d.pcol      = w[43:41];
    d.cmp       = cmp_e'(w[40:38]);
    d.cmp_src   = cs_e'(w[37:36]);
    d.cmp_int   = w[35];
    d.sel       = sel_e'(w[34:32]);
    d.ialu      = ia_e'(w[31:29]);
    d.dst_s1    = w[28];
    d.c0_pred   = w[27];
    d.nodrain   = w[26];
    d.x_s1      = w[25];
    d.ctx       = w[24:23];
    d.rm        = (w[22:21] == 2'd3) ? RmNone : rm_e'(w[22:21]);
    return d;
  endfunction

  localparam logic [3:0] KindPass   = 4'd0;
  localparam logic [3:0] KindSalu   = 4'd1;
  localparam logic [3:0] KindSmem   = 4'd2;
  localparam logic [3:0] KindBranch = 4'd3;
  localparam logic [3:0] KindRet    = 4'd4;
  localparam logic [3:0] KindTranspose = 4'd7;

  localparam logic [4:0] SaluAdd  = 5'd0;
  localparam logic [4:0] SaluSub  = 5'd1;
  localparam logic [4:0] SaluAnd  = 5'd2;
  localparam logic [4:0] SaluOr   = 5'd3;
  localparam logic [4:0] SaluXor  = 5'd4;
  localparam logic [4:0] SaluShl  = 5'd5;
  localparam logic [4:0] SaluShr  = 5'd6;
  localparam logic [4:0] SaluSra  = 5'd7;
  localparam logic [4:0] SaluMul  = 5'd8;
  localparam logic [4:0] SaluSlt  = 5'd9;
  localparam logic [4:0] SaluSltu = 5'd10;
  localparam logic [4:0] SaluSeq  = 5'd11;

  localparam logic [2:0] SmemSld   = 3'd0;
  localparam logic [2:0] SmemRsld  = 3'd2;
  localparam logic [2:0] SmemRsmov = 3'd4;
  localparam logic [2:0] SmemSmovr = 3'd5;
  localparam logic [2:0] SmemRn    = 3'd6;

  typedef struct packed {
    logic        valid;
    logic [1:0]  ph;
    logic [CtxW-1:0] ctx;
    c0_e         c0;
    b_e          b;
    nl_e         nl;
    c1_e         c1;
    logic        neg_c1;
    logic [7:0]  mask;
    red_e        red;
    wr_e         wr;
    logic [12:0] waddr;
    logic [2:0]  j;
    logic [1:0]  qw;
    logic        pa_at_mul;
    logic [9:0]  pa_addr;
    logic [1:0]  pa_slice;
    logic        pa_tile;
    px_e         px;
    logic [2:0]  pcol;
    cmp_e        cmp;
    cs_e         cmp_src;
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
  localparam logic [31:0] FpLog2e        = 32'h3FB8AA3B;
  localparam logic [31:0] FpNegInf       = 32'hFF800000;
  localparam logic [31:0] NullEntry      = 32'hFFFFFFFF;

endpackage
