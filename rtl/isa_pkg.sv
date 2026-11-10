package isa_pkg;

  localparam int OpcodeMsb = 127;
  localparam int OpcodeLsb = 120;
  localparam int WaitMsb = 119;
  localparam int WaitLsb = 116;

  localparam logic [7:0] OpNop = 8'd0;
  localparam logic [7:0] OpDmaLoad = 8'd1;
  localparam logic [7:0] OpDmaStore = 8'd2;
  localparam logic [7:0] OpGemm = 8'd4;
  localparam logic [7:0] OpVec = 8'd5;
  localparam logic [7:0] OpSignal = 8'd6;
  localparam logic [7:0] OpHalt = 8'd7;
  localparam logic [7:0] OpWaitHost = 8'd8;

  localparam int Engines = 3;
  localparam int EngDma = 0;
  localparam int EngGemm = 1;
  localparam int EngVec = 2;

  localparam logic [2:0] MemAbuf = 3'd0;
  localparam logic [2:0] MemWbuf = 3'd1;
  localparam logic [2:0] MemFbuf = 3'd2;
  localparam logic [2:0] MemPbuf = 3'd3;
  localparam logic [2:0] MemTbuf = 3'd4;
  localparam logic [2:0] MemCbuf = 3'd5;
  localparam logic [31:0] NullEntry = 32'hFFFFFFFF;

  localparam int DmaDdrAddrMsb = 115;
  localparam int DmaDdrAddrLsb = 88;
  localparam int DmaMemMsb = 87;
  localparam int DmaMemLsb = 85;
  localparam int DmaAddrMsb = 84;
  localparam int DmaAddrLsb = 72;
  localparam int DmaRowsMsb = 71;
  localparam int DmaRowsLsb = 60;
  localparam int DmaRowWordsMsb = 59;
  localparam int DmaRowWordsLsb = 48;
  localparam int DmaDdrStrideMsb = 47;
  localparam int DmaDdrStrideLsb = 28;
  localparam int DmaKindMsb = 27;
  localparam int DmaKindLsb = 25;
  localparam int DmaTokMsb = 24;
  localparam int DmaTokLsb = 19;
  localparam int DmaAccMsb = 18;
  localparam int DmaAccLsb = 18;
  localparam int DmaFlushMsb = 17;
  localparam int DmaFlushLsb = 17;
  localparam logic [2:0] DmaKindRaw = 3'd0;
  localparam logic [2:0] DmaKindRow = 3'd1;
  localparam logic [2:0] DmaKindRowIndexed = 3'd2;
  localparam logic [2:0] DmaKindSplit8 = 3'd3;
  localparam logic [2:0] DmaKindT32Split8 = 3'd4;
  localparam logic [2:0] DmaKindGacc = 3'd5;

  localparam int GemmABaseMsb = 111;
  localparam int GemmABaseLsb = 92;
  localparam int GemmAStrideMsb = 91;
  localparam int GemmAStrideLsb = 82;
  localparam int GemmKbMsb = 81;
  localparam int GemmKbLsb = 72;
  localparam int GemmMMsb = 71;
  localparam int GemmMLsb = 62;
  localparam int GemmWBaseMsb = 61;
  localparam int GemmWBaseLsb = 42;
  localparam int GemmWStrideMsb = 41;
  localparam int GemmWStrideLsb = 32;
  localparam int GemmN8Msb = 31;
  localparam int GemmN8Lsb = 24;
  localparam int GemmCBaseMsb = 23;
  localparam int GemmCBaseLsb = 4;
  localparam int GemmWmodeMsb = 3;
  localparam int GemmWmodeLsb = 2;
  localparam int GemmAccMsb = 1;
  localparam int GemmAccLsb = 1;
  localparam logic [1:0] WeightModeFp4 = 2'd0;
  localparam logic [1:0] WeightModeW8Hi = 2'd1;
  localparam logic [1:0] WeightModeW8Lo = 2'd2;

  localparam int VecEntryMsb = 115;
  localparam int VecEntryLsb = 108;
  localparam int VecQMsb = 107;
  localparam int VecQLsb = 107;
  localparam int VecSrcMsb = 106;
  localparam int VecSrcLsb = 95;
  localparam int VecSrc2Msb = 94;
  localparam int VecSrc2Lsb = 83;
  localparam int VecDstMsb = 82;
  localparam int VecDstLsb = 70;
  localparam int VecRowsQMsb = 69;
  localparam int VecRowsQLsb = 61;
  localparam int VecN8Msb = 60;
  localparam int VecN8Lsb = 53;
  localparam int VecPBaseMsb = 51;
  localparam int VecPBaseLsb = 43;
  localparam int VecAuxMsb = 42;
  localparam int VecAuxLsb = 32;
  localparam int VecImmMsb = 31;
  localparam int VecImmLsb = 0;
  localparam logic [4:0] VecRegSrc = 5'd1;
  localparam logic [4:0] VecRegSrc2 = 5'd2;
  localparam logic [4:0] VecRegDst = 5'd3;
  localparam logic [4:0] VecRegRowsQ = 5'd4;
  localparam logic [4:0] VecRegN8 = 5'd5;
  localparam logic [4:0] VecRegPBase = 5'd6;
  localparam logic [4:0] VecRegAux = 5'd7;
  localparam logic [4:0] VecRegImm = 5'd8;
  localparam logic [4:0] VecRegQ = 5'd9;
  localparam int VecEntryWords = 4;
  localparam int VecRomEntries = 64;
  localparam int VecRomWords = 256;

  localparam int SignalIdMsb = 111;
  localparam int SignalIdLsb = 104;
  localparam int SignalValueMsb = 31;
  localparam int SignalValueLsb = 0;
  localparam int WaitHostValueMsb = 31;
  localparam int WaitHostValueLsb = 0;

  localparam int ProgKindMsb = 127;
  localparam int ProgKindLsb = 124;
  localparam logic [3:0] KindPass = 4'd0;
  localparam logic [3:0] KindSalu = 4'd1;
  localparam logic [3:0] KindSmem = 4'd2;
  localparam logic [3:0] KindBranch = 4'd3;
  localparam logic [3:0] KindRet = 4'd4;
  localparam logic [3:0] KindNop = 4'd5;
  localparam logic [3:0] KindTranspose = 4'd7;

  localparam int ProgOpMsb = 123;
  localparam int ProgOpLsb = 119;
  localparam int ProgRdMsb = 118;
  localparam int ProgRdLsb = 114;
  localparam int ProgRaMsb = 113;
  localparam int ProgRaLsb = 109;
  localparam int ProgRbMsb = 108;
  localparam int ProgRbLsb = 104;
  localparam int ProgUseImmMsb = 103;
  localparam int ProgUseImmLsb = 103;
  localparam int ProgImmMsb = 31;
  localparam int ProgImmLsb = 0;
  localparam int ProgSmemColMsb = 102;
  localparam int ProgSmemColLsb = 98;
  localparam int ProgSmemWhichMsb = 97;
  localparam int ProgSmemWhichLsb = 97;
  localparam int ProgSmemCtxMsb = 96;
  localparam int ProgSmemCtxLsb = 96;
  localparam logic [4:0] ProgRegZero = 5'd0;
  localparam logic SmemWhichRs1 = 1'd0;
  localparam logic SmemWhichRs0 = 1'd1;

  localparam logic [4:0] SaluAdd = 5'd0;
  localparam logic [4:0] SaluSub = 5'd1;
  localparam logic [4:0] SaluAnd = 5'd2;
  localparam logic [4:0] SaluOr = 5'd3;
  localparam logic [4:0] SaluXor = 5'd4;
  localparam logic [4:0] SaluShl = 5'd5;
  localparam logic [4:0] SaluShr = 5'd6;
  localparam logic [4:0] SaluSra = 5'd7;
  localparam logic [4:0] SaluMul = 5'd8;
  localparam logic [4:0] SaluSlt = 5'd9;
  localparam logic [4:0] SaluSltu = 5'd10;
  localparam logic [4:0] SaluSeq = 5'd11;
  localparam logic [4:0] SaluMov = 5'd12;
  localparam logic [4:0] SmemSld = 5'd0;
  localparam logic [4:0] SmemRsmov = 5'd1;
  localparam logic [4:0] SmemRn = 5'd2;
  localparam logic [4:0] BrAlways = 5'd0;
  localparam logic [4:0] BrZero = 5'd1;
  localparam logic [4:0] BrNonzero = 5'd2;

  localparam int PassRxMsb = 123;
  localparam int PassRxLsb = 119;
  localparam int PassRx2Msb = 118;
  localparam int PassRx2Lsb = 114;
  localparam int PassRdMsb = 113;
  localparam int PassRdLsb = 109;
  localparam int PassRqMsb = 108;
  localparam int PassRqLsb = 104;
  localparam int PassRnwMsb = 103;
  localparam int PassRnwLsb = 99;
  localparam int PassRn8Msb = 98;
  localparam int PassRn8Lsb = 94;
  localparam int PassRpbMsb = 93;
  localparam int PassRpbLsb = 89;
  localparam int PassRimmMsb = 88;
  localparam int PassRimmLsb = 84;
  localparam int PassRimm2Msb = 83;
  localparam int PassRimm2Lsb = 79;
  localparam int PassX2EnMsb = 78;
  localparam int PassX2EnLsb = 78;
  localparam int PassMgMsb = 77;
  localparam int PassMgLsb = 76;
  localparam int PassMsMsb = 75;
  localparam int PassMsLsb = 74;
  localparam int PassMxMsb = 73;
  localparam int PassMxLsb = 72;
  localparam int PassMx2Msb = 71;
  localparam int PassMx2Lsb = 70;
  localparam int PassMdMsb = 69;
  localparam int PassMdLsb = 68;
  localparam int PassPaMsb = 67;
  localparam int PassPaLsb = 66;
  localparam int PassC0Msb = 65;
  localparam int PassC0Lsb = 63;
  localparam int PassBMsb = 62;
  localparam int PassBLsb = 60;
  localparam int PassNlMsb = 59;
  localparam int PassNlLsb = 58;
  localparam int PassPoMsb = 57;
  localparam int PassPoLsb = 56;
  localparam int PassPvMsb = 55;
  localparam int PassPvLsb = 55;
  localparam int PassRedMsb = 54;
  localparam int PassRedLsb = 53;
  localparam int PassWrMsb = 52;
  localparam int PassWrLsb = 51;
  localparam int PassSfMsb = 50;
  localparam int PassSfLsb = 48;
  localparam int PassStMsb = 47;
  localparam int PassStLsb = 47;
  localparam int PassSrnMsb = 46;
  localparam int PassSrnLsb = 46;
  localparam int PassSepsMsb = 45;
  localparam int PassSepsLsb = 45;
  localparam int PassPxMsb = 44;
  localparam int PassPxLsb = 25;
  localparam int PassCmpMsb = 24;
  localparam int PassCmpLsb = 22;
  localparam int PassClMsb = 21;
  localparam int PassClLsb = 19;
  localparam int PassCrMsb = 18;
  localparam int PassCrLsb = 18;
  localparam int PassCmpIntMsb = 17;
  localparam int PassCmpIntLsb = 17;
  localparam int PassSelMsb = 16;
  localparam int PassSelLsb = 14;
  localparam int PassIaluMsb = 13;
  localparam int PassIaluLsb = 11;
  localparam int PassC0PredMsb = 10;
  localparam int PassC0PredLsb = 10;
  localparam int PassXS1Msb = 9;
  localparam int PassXS1Lsb = 9;
  localparam int PassDstS1Msb = 8;
  localparam int PassDstS1Lsb = 8;
  localparam int PassCtxMsb = 7;
  localparam int PassCtxLsb = 7;
  localparam int PassRmMsb = 6;
  localparam int PassRmLsb = 5;

  typedef enum logic [1:0] {
    MmNone = 2'd0,
    MmGroup = 2'd1,
    MmUpper = 2'd2
  } mm_e;

  typedef enum logic [1:0] {
    PaNone = 2'd0,
    PaRowVec = 2'd1,
    PaTile = 2'd2
  } pa_e;

  typedef enum logic [2:0] {
    C0Zero = 3'd0,
    C0X2 = 3'd1,
    C0Pbuf = 3'd2,
    C0Rs0 = 3'd3,
    C0Imm = 3'd4
  } c0_e;

  typedef enum logic [2:0] {
    BOne = 3'd0,
    BX2 = 3'd1,
    BPbuf = 3'd2,
    BRs1 = 3'd3,
    BImm = 3'd4,
    BPre = 3'd5
  } b_e;

  typedef enum logic [1:0] {
    NlNone = 2'd0,
    NlAct = 2'd1,
    NlExp2 = 2'd2,
    NlTab01 = 2'd3
  } nl_e;

  typedef enum logic [1:0] {
    PoAdd = 2'd0,
    PoSub = 2'd1,
    PoMax = 2'd2,
    PoSignfix = 2'd3
  } po_e;

  typedef enum logic [1:0] {
    RedNone = 2'd0,
    RedSum = 2'd1,
    RedMax = 2'd2
  } red_e;

  typedef enum logic [1:0] {
    WrNone = 2'd0,
    WrFbuf = 2'd1,
    WrAbuf = 2'd2,
    WrFinal = 2'd3
  } wr_e;

  typedef enum logic [2:0] {
    SfNone = 3'd0,
    SfIdentity = 3'd1,
    SfNegate = 3'd2,
    SfReciprocal = 3'd3,
    SfRsqrt = 3'd4
  } sf_e;

  typedef enum logic [2:0] {
    CmpNone = 3'd0,
    CmpLt = 3'd1,
    CmpLe = 3'd2,
    CmpEq = 3'd3,
    CmpNe = 3'd4,
    CmpGt = 3'd5,
    CmpGe = 3'd6
  } cmp_e;

  typedef enum logic [2:0] {
    ClX = 3'd0,
    ClX2 = 3'd1,
    ClColumn = 3'd2,
    ClPosition = 3'd3,
    ClRowColumn = 3'd4
  } cl_e;

  typedef enum logic [2:0] {
    SelNone = 3'd0,
    SelZero = 3'd1,
    SelX2 = 3'd2,
    SelImm = 3'd3,
    SelNull = 3'd4,
    SelRs1 = 3'd5
  } sel_e;

  typedef enum logic [2:0] {
    IaluNone = 3'd0,
    IaluAdd = 3'd1,
    IaluSub = 3'd2,
    IaluAnd = 3'd3,
    IaluOr = 3'd4,
    IaluXor = 3'd5,
    IaluShr1 = 3'd6,
    IaluRsub = 3'd7
  } ia_e;

  typedef enum logic [1:0] {
    RmNone = 2'd0,
    RmSkip = 2'd1,
    RmSet = 2'd2
  } rm_e;

  localparam int PermSelectorBits = 4;
  localparam int PermPositionBits = 5;
  localparam int PermInvertBit = 3;
  localparam logic [2:0] PermSame = 3'd0;
  localparam logic [2:0] PermPos4 = 3'd1;
  localparam logic [2:0] PermPos3 = 3'd2;
  localparam logic [2:0] PermPos2 = 3'd3;
  localparam logic [2:0] PermPos1 = 3'd4;
  localparam logic [2:0] PermPos0 = 3'd5;
  localparam logic [2:0] PermZero = 3'd6;
  localparam logic [2:0] PermOne = 3'd7;

  localparam int TransposeRxMsb = 123;
  localparam int TransposeRxLsb = 119;
  localparam int TransposeRswMsb = 118;
  localparam int TransposeRswLsb = 114;
  localparam int TransposeRdMsb = 113;
  localparam int TransposeRdLsb = 109;
  localparam int TransposeRtMsb = 108;
  localparam int TransposeRtLsb = 104;
  localparam int TransposeRdtMsb = 103;
  localparam int TransposeRdtLsb = 99;
  localparam int TransposeRstMsb = 98;
  localparam int TransposeRstLsb = 94;
  localparam int TransposeRdwMsb = 93;
  localparam int TransposeRdwLsb = 89;
  localparam int TransposeRorMsb = 88;
  localparam int TransposeRorLsb = 87;

  localparam int RegAddrBits = 8;
  localparam int RegByteBits = 2;
  localparam logic [5:0] RegCtrl = 6'd0;
  localparam logic [5:0] RegStatus = 6'd1;
  localparam logic [5:0] RegInstrBase = 6'd2;
  localparam logic [5:0] RegInstrCount = 6'd3;
  localparam logic [5:0] RegIrqStatus = 6'd4;
  localparam logic [5:0] RegIrqEnable = 6'd5;
  localparam logic [5:0] RegSignal = 6'd6;
  localparam logic [5:0] RegSignalValue = 6'd7;
  localparam logic [5:0] RegIssued = 6'd8;
  localparam logic [5:0] RegCycles = 6'd9;
  localparam logic [5:0] RegDmaCycles = 6'd10;
  localparam logic [5:0] RegGemmCycles = 6'd11;
  localparam logic [5:0] RegVecCycles = 6'd12;
  localparam logic [5:0] RegStallCycles = 6'd13;
  localparam logic [5:0] RegStarveCycles = 6'd14;
  localparam logic [5:0] RegTraceBase = 6'd15;
  localparam logic [5:0] RegTraceCtrl = 6'd16;
  localparam logic [5:0] RegTraceCount = 6'd17;
  localparam logic [5:0] RegHostGo = 6'd18;

  localparam int CtrlStart = 0;
  localparam int CtrlSoftReset = 1;
  localparam int StatusRunning = 0;
  localparam int StatusHalted = 1;
  localparam int StatusError = 2;
  localparam int StatusWaitingHost = 3;
  localparam int StatusBusyLsb = 4;
  localparam int StatusIbufEmpty = 7;
  localparam int IrqSignal = 0;
  localparam int IrqHalt = 1;
  localparam int IrqIllegal = 2;
  localparam int IrqAxiError = 3;
  localparam int IrqBits = 4;
  localparam int SignalRegIdMsb = 7;
  localparam int SignalRegIdLsb = 0;
  localparam int SignalRegCountMsb = 31;
  localparam int SignalRegCountLsb = 16;
  localparam int TraceCtrlEnable = 0;
  localparam int TraceCtrlIdle = 1;
  localparam int TraceCtrlStatistics = 2;

  localparam int TraceRecordBits = 128;
  localparam int TraceIndexMsb = 31;
  localparam int TraceIndexLsb = 0;
  localparam int TraceTypeMsb = 119;
  localparam int TraceTypeLsb = 116;
  localparam int TraceIssueMsb = 63;
  localparam int TraceIssueLsb = 32;
  localparam int TraceCompletionMsb = 95;
  localparam int TraceCompletionLsb = 64;
  localparam int TraceSourceMsb = 99;
  localparam int TraceSourceLsb = 96;
  localparam int TraceOpcodeMsb = 107;
  localparam int TraceOpcodeLsb = 100;
  localparam int TraceWaitMsb = 111;
  localparam int TraceWaitLsb = 108;
  localparam int TraceStatSourceMsb = 35;
  localparam int TraceStatSourceLsb = 32;
  localparam int TraceStatAMsb = 63;
  localparam int TraceStatALsb = 40;
  localparam int TraceStatBMsb = 87;
  localparam int TraceStatBLsb = 64;
  localparam int TraceStatCMsb = 111;
  localparam int TraceStatCLsb = 88;
  localparam int TraceStatDMsb = 127;
  localparam int TraceStatDLsb = 120;
  localparam logic [3:0] TraceTypeCompletion = 4'd0;
  localparam logic [3:0] TraceTypeStatistics = 4'd1;
  localparam int TraceSourceSequencer = 3;

endpackage
