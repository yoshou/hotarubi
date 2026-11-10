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
  localparam int CtxIdxW       = 1;
  localparam int AccAw         = CtxIdxW + 5;
  localparam int CmdRegs       = 9;

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
    BOne  = 3'd0,
    BX2   = 3'd1,
    BPbuf = 3'd2,
    BRs1  = 3'd3,
    BImm  = 3'd4,
    BPre  = 3'd5
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

  typedef enum logic [1:0] {
    PoAdd     = 2'd0,
    PoSub     = 2'd1,
    PoMax     = 2'd2,
    PoSignfix = 2'd3
  } po_e;

  typedef enum logic [2:0] {
    SfNone       = 3'd0,
    SfIdentity   = 3'd1,
    SfNegate     = 3'd2,
    SfReciprocal = 3'd3,
    SfRsqrt      = 3'd4
  } sf_e;

  typedef struct packed {
    sf_e  fn;
    logic rs0;
    logic rn;
    logic eps;
    logic max;
  } sc_t;

  typedef enum logic [1:0] {
    MmNone  = 2'd0,
    MmGroup = 2'd1,
    MmUpper = 2'd2
  } mm_e;

  typedef enum logic [2:0] {
    CmpNone = 3'd0,
    CmpLt   = 3'd1,
    CmpLe   = 3'd2,
    CmpEq   = 3'd3,
    CmpNe   = 3'd4,
    CmpGt   = 3'd5,
    CmpGe   = 3'd6
  } cmp_e;

  typedef enum logic [2:0] {
    ClX         = 3'd0,
    ClX2        = 3'd1,
    ClColumn    = 3'd2,
    ClPosition  = 3'd3,
    ClRowColumn = 3'd4
  } cl_e;

  typedef enum logic [2:0] {
    SelNone = 3'd0,
    SelZero = 3'd1,
    SelX2   = 3'd2,
    SelImm  = 3'd3,
    SelNull = 3'd4,
    SelRs1  = 3'd5
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
    return (v == 2'd3) ? MmNone : mm_e'(v);
  endfunction

  function automatic pass_t dec_pass(input logic [127:0] w);
    pass_t d;
    logic  unused_w;
    unused_w  = ^{w[127:124], w[4:0]};
    d.r       = pass_regs_t'(w[123:79]);
    d.x2_en   = w[78];
    d.mg      = w[77:76];
    d.ms      = w[75:74];
    d.mx      = dec_map(w[73:72]);
    d.mx2     = dec_map(w[71:70]);
    d.md      = dec_map(w[69:68]);
    d.pa      = (w[67:66] == 2'd3) ? PaNone : pa_e'(w[67:66]);
    d.c0      = (w[65:63] > 3'd4) ? C0Zero : c0_e'(w[65:63]);
    d.b       = (w[62:60] > 3'd5) ? BOne : b_e'(w[62:60]);
    d.nl      = nl_e'(w[59:58]);
    d.po      = po_e'(w[57:56]);
    d.pv      = w[55];
    d.red     = (w[54:53] == 2'd3) ? RedNone : red_e'(w[54:53]);
    d.wr      = wr_e'(w[52:51]);
    d.sf      = (w[50:48] > 3'd4) ? SfNone : sf_e'(w[50:48]);
    d.st      = w[47];
    d.srn     = w[46];
    d.seps    = w[45];
    d.px      = w[44:25];
    d.cmp     = (w[24:22] == 3'd7) ? CmpNone : cmp_e'(w[24:22]);
    d.cl      = (w[21:19] > 3'd4) ? ClX : cl_e'(w[21:19]);
    d.cr      = w[18];
    d.cmp_int = w[17];
    d.sel     = (w[16:14] > 3'd5) ? SelNone : sel_e'(w[16:14]);
    d.ialu    = ia_e'(w[13:11]);
    d.c0_pred = w[10];
    d.x_s1    = w[9];
    d.dst_s1  = w[8];
    d.ctx     = w[7];
    d.rm      = (w[6:5] == 2'd3) ? RmNone : rm_e'(w[6:5]);
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

  localparam logic [4:0] SmemSld   = 5'd0;
  localparam logic [4:0] SmemRsmov = 5'd1;
  localparam logic [4:0] SmemRn    = 5'd2;

  localparam logic [4:0] BrAlways  = 5'd0;
  localparam logic [4:0] BrZero    = 5'd1;
  localparam logic [4:0] BrNonzero = 5'd2;

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
  localparam logic [31:0] NullEntry      = 32'hFFFFFFFF;

endpackage
