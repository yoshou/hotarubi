module vec_engine
  import vec_pkg::*;
  import vec_rom_pkg::*;
(
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          cmd_valid_i,
  output logic          cmd_ready_o,
  input  logic [4:0]    cmd_op_i,
  input  logic          cmd_q_i,
  input  logic [11:0]   cmd_src_i,
  input  logic [11:0]   cmd_src2_i,
  input  logic [12:0]   cmd_dst_i,
  input  logic [8:0]    cmd_rows_q_i,
  input  logic [7:0]    cmd_n8_i,
  input  logic [9:0]    cmd_p_base_i,
  input  logic [10:0]   cmd_aux_i,
  input  logic [31:0]   cmd_imm_i,
  output logic          busy_o,
  output logic          fbuf_rd_req_o,
  input  logic          fbuf_rd_grant_i,
  output logic [11:0]   fbuf_raddr_o,
  input  logic [1023:0] fbuf_rdata_i,
  output logic          fbuf_wr_req_o,
  input  logic          fbuf_wr_grant_i,
  output logic [11:0]   fbuf_waddr_o,
  output logic [1023:0] fbuf_wdata_o,
  output logic [9:0]    pbuf_a_addr_o,
  input  logic [1023:0] pbuf_a_rdata_i,
  output logic [9:0]    tab_addr_o [Lanes],
  input  logic [63:0]   tab_data_i [Lanes],
  output logic          abuf_wr_req_o,
  input  logic          abuf_wr_grant_i,
  output logic [12:0]   abuf_waddr_o,
  output logic [1055:0] abuf_wdata_o,
  output logic [9:0]    cbuf_addr_o,
  input  logic [127:0]  cbuf_rdata_i,
  output logic          stat_issue_o,
  output logic          stat_drain_o,
  output logic          stat_scalar_o,
  output logic          stat_pass_o
);

  function automatic logic uses_imm(input logic [4:0] op);
    return (op == 5'(OpLayerNorm)) || (op == 5'(OpRmsNorm)) || (op == 5'(OpScale));
  endfunction

  typedef enum logic [4:0] {
    StIdle,
    StInit,
    StFetch,
    StFwait,
    StDecode,
    StPrd,
    StSegSetup,
    StIssue,
    StDrain,
    StSrd,
    StSex,
    StMulw,
    StMdrain,
    StMrd,
    StMwait,
    StMex,
    StMrs,
    StRnwait,
    StBex,
    StNext,
    StTransposeStart,
    StTransposeWait
  } st_e;

  st_e          st;
  logic [9:0]   pc;
  logic [127:0] ins;
  logic [3:0]   kind;
  logic [3:0]   cnt;
  logic [2:0]   pf_cnt;
  pass_t        pd_n;
  logic         pre_on, pre_done;
  logic [3:0]   pre_cnt;
  logic [11:0]  pn_x, pn_x2;
  logic [12:0]  pn_d;
  logic [8:0]   pn_rq;
  logic [9:0]   pn_nw;
  logic [7:0]   pn_n8;
  logic [9:0]   pn_pb;
  logic [31:0]  pn_imm, pn_imm2;
  pass_t        pd;
  logic         qf;
  logic [11:0]  c_src, c_src2, c_scr;
  logic [12:0]  c_dst;
  logic [8:0]   c_rows_q;
  logic [7:0]   c_n8;
  logic [9:0]   c_p_base;
  logic [10:0]  c_aux;
  logic [31:0]  c_imm;

  assign cmd_ready_o   = (st == StIdle);
  assign busy_o        = (st != StIdle);
  assign stat_issue_o  = (st == StIssue) || (st == StTransposeWait);
  assign stat_drain_o  = (st == StDrain) || (st == StMdrain);
  assign stat_scalar_o = ((st == StSegSetup) && pass_wait) || (st == StRnwait) ||
                         ((st == StDecode) && (kind == KindRet) && !ret_ok);
  assign stat_pass_o   = (st == StSegSetup) && !pass_wait;

  (* ram_style = "distributed" *) logic [31:0] rf [32];
  logic        rf_we;
  logic [4:0]  rf_wa, ra_q, rb_q;
  logic [31:0] rf_wd;

  always_ff @(posedge clk_i) begin
    if (rf_we) begin
      rf[rf_wa] <= rf_wd;
    end
  end

  logic [31:0] rf_a, rf_b;

  assign rf_a = (ra_q == 5'd0) ? 32'd0 : rf[ra_q];
  assign rf_b = (rb_q == 5'd0) ? 32'd0 : rf[rb_q];

  logic [11:0] p_x, p_x2;
  logic [12:0] p_d;
  logic [8:0]  p_rq;
  logic [9:0]  p_nw;
  logic [7:0]  p_n8;
  logic [9:0]  p_pb;
  logic [31:0] p_imm, p_imm2;
  logic [11:0] tr_src_tile_stride, tr_dst_tile_stride, tr_dst_word_stride;
  logic        tr_start, tr_busy, tr_rd, tr_wr, tr_mode;
  logic [11:0] tr_ra, tr_wa;
  logic [1023:0] tr_wd;
  wr_e         wr_eff;

  assign wr_eff = (pd.wr == WrFinal) ? (qf ? WrAbuf : WrFbuf) : pd.wr;

  logic [11:0] rb_src, rb_src2, rb_dst;
  logic [12:0] rb_abuf;
  logic [9:0]  pt_row;
  logic [8:0]  q;
  logic [9:0]  w;
  logic        slot_ph;
  logic [2:0]  slot_j;
  logic [1:0]  gap;

  function automatic logic [9:0] word_of(input rw_e sel, input logic [9:0] wn);
    case (sel)
      MapX1:   return {3'd0, wn[5:2], 1'b0, wn[1:0]};
      MapX2:   return {3'd0, wn[5:2], 1'b1, wn[1:0]};
      MapS4:   return {wn[7:0], 2'b00};
      default: return wn;
    endcase
  endfunction

  logic [9:0] wa_word, wb_word, ww_word;

  assign wa_word = word_of(pd.wx, w);
  assign wb_word = word_of(pd.wx2, w);
  assign ww_word = word_of(pd.wd, w);

  logic wf_afull, df_afull, af_afull;
  logic pend_full, pend_clear;
  logic can_issue;

  assign can_issue = !wf_afull & !df_afull & !af_afull & !pend_full;

  logic        rd_v, rd_is_entry, rd_is_hold;
  logic [11:0] rd_addr;
  ctrl_t       rd_ctrl;
  logic [7:0]  mask8;

  always_comb begin
    for (int c = 0; c < 8; c++) begin
      mask8[c] = pd.mask_en & ((16'({w[9:0], 3'd0}) + 16'(c)) >= p_imm2[15:0]);
    end
  end

  always_comb begin
    rd_is_hold  = (slot_ph == 1'b0) & pd.x2_en;
    rd_is_entry = !rd_is_hold;
    rd_v        = (st == StIssue) & can_issue & !(rd_is_entry & (gap != 2'd0)) & pend_clear;
    if (slot_ph == 1'b0) begin
      rd_addr = rb_src + 12'(wa_word);
    end else begin
      rd_addr = rb_src2 + 12'(wb_word);
    end
    rd_ctrl           = '0;
    rd_ctrl.valid     = 1'b1;
    rd_ctrl.ctx       = pd.ctx;
    rd_ctrl.c0        = pd.c0;
    rd_ctrl.b         = pd.b;
    rd_ctrl.nl        = pd.nl;
    rd_ctrl.c1        = pd.c1;
    rd_ctrl.neg_c1    = pd.neg_c1;
    rd_ctrl.mask      = mask8;
    rd_ctrl.red       = pd.red;
    rd_ctrl.wr        = wr_eff;
    rd_ctrl.waddr     = (wr_eff == WrAbuf) ? (rb_abuf + 13'(ww_word[9:2])) :
                                             13'(rb_dst + 12'(ww_word));
    rd_ctrl.j         = slot_j;
    rd_ctrl.qw        = ww_word[1:0];
    rd_ctrl.pa_at_mul = pd.pa_at_mul;
    rd_ctrl.pa_tile   = (pd.pa == PaTile);
    case (pd.pa)
      PaRowvec: rd_ctrl.pa_addr = p_pb + 10'(w[9:2]);
      PaTile:   rd_ctrl.pa_addr = p_pb + pt_row + 10'(w[1:0]);
      default:  rd_ctrl.pa_addr = 10'd0;
    endcase
    rd_ctrl.pa_slice = w[1:0];
    rd_ctrl.px       = pd.px;
    rd_ctrl.pcol     = pd.pcol;
    rd_ctrl.cmp      = pd.cmp;
    rd_ctrl.cmp_src  = pd.cmp_src;
    rd_ctrl.cmp_int  = pd.cmp_int;
    rd_ctrl.sel      = pd.sel;
    rd_ctrl.ialu     = pd.ialu;
    rd_ctrl.c0_pred  = pd.c0_pred;
    rd_ctrl.imm      = p_imm;
    rd_ctrl.imm2     = p_imm2;
  end

  logic        sm_rd_v;
  logic [11:0] sm_addr;

  assign fbuf_rd_req_o = rd_v | sm_rd_v | tr_rd;

  logic rd_fire, sm_rd_fire;

  assign rd_fire    = rd_v & fbuf_rd_grant_i;
  assign sm_rd_fire = sm_rd_v & fbuf_rd_grant_i;

  logic lane_busy, red_busy, sc_busy, quant_busy, wf_empty, af_empty, phs_v;
  logic pass_wait, ret_ok;
  logic rdpipe_busy;
  logic drained, drained_r;

  assign drained = !rdpipe_busy & !phs_v & !lane_busy & !red_busy & wf_empty & !quant_busy &
                   af_empty;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      drained_r <= 1'b0;
    end else begin
      drained_r <= drained;
    end
  end

  assign tr_mode  = (st == StTransposeWait);
  assign tr_start = (st == StTransposeStart) && drained_r;

  logic next_is_pass, next_pa_pre, overlap_ok;

  assign next_is_pass = (pf_cnt >= 3'(CbufLatency)) && (cbuf_rdata_i[127:124] == KindPass);
  assign next_pa_pre  = (pa_e'(cbuf_rdata_i[71:70]) != PaNone) && !cbuf_rdata_i[69];
  assign overlap_ok   = next_is_pass && !(pd.pa_at_mul && next_pa_pre);

  vec_transpose #(
    .ReadLatency(MemLatency)
  ) u_transpose (
    .clk_i,
    .rst_ni,
    .start_i          (tr_start),
    .src_i            (p_x),
    .dst_i            (p_d[11:0]),
    .tiles_i          (p_rq),
    .src_tile_stride_i(tr_src_tile_stride),
    .src_word_stride_i(p_x2),
    .dst_tile_stride_i(tr_dst_tile_stride),
    .dst_word_stride_i(tr_dst_word_stride),
    .pack_zip4_i      (ins[88]),
    .busy_o           (tr_busy),
    .rd_req_o         (tr_rd),
    .rd_grant_i       (fbuf_rd_grant_i),
    .rd_addr_o        (tr_ra),
    .rd_data_i        (fbuf_rdata_i),
    .wr_req_o         (tr_wr),
    .wr_grant_i       (fbuf_wr_grant_i),
    .wr_addr_o        (tr_wa),
    .wr_data_o        (tr_wd)
  );

  logic               sc_start, sc_recip_n, sc_running;
  sc_e                sc_prog;
  logic [13:0]        sc_nelem;
  logic [CtxIdxW-1:0] sc_ctx;
  logic               sc_rs_we, sc_rs_sel;
  logic [31:0]        sc_rs_val [4];
  logic               rs_we0, rs_we1;
  logic [31:0]        rs_val [4];
  logic [31:0]        rs_pick [4];
  logic [31:0]        rn;
  logic [31:0]        rs0 [NumContexts][4];
  logic [31:0]        rs1 [NumContexts][4];
  logic [CtxW-1:0]    sm_ctx_field;
  logic [CtxIdxW-1:0] sm_ctx, pd_ctx;

  assign sm_ctx_field = ins[97:96];
  assign sm_ctx       = sm_ctx_field[CtxIdxW-1:0];
  assign pd_ctx       = pd.ctx[CtxIdxW-1:0];

  logic [NumContexts-1:0] ctx_reducing, ctx_issued, ctx_sc_pending, ctx_idle, sc_outstanding,
                          ctx_done;
  logic [8:0]             ctx_pending [NumContexts];
  sc_e                    ctx_prog [NumContexts];
  logic [31:0]            ctx_eps [NumContexts];
  logic                   pass_reads_rs, pass_last_fire, ctx_all_idle, sm_ready;
  logic                   sum_done, max_done;
  logic [CtxIdxW-1:0]     sum_done_ctx, max_done_ctx;
  logic [CtxIdxW-1:0]     rq_mem [NumContexts];
  logic [CtxIdxW-1:0]     rq_wp, rq_rp;
  logic [CtxIdxW:0]       rq_count;
  logic                   rq_empty, rq_push, rq_pop, done_any;
  logic [CtxIdxW-1:0]     done_ctx;

  assign pass_reads_rs  = (pd.c0 == C0Rs0) | (pd.b == BRs1) | (pd.sel == SelRs0) |
                          (pd.sel == SelRs1);
  assign pass_last_fire = rd_fire & rd_is_entry & (w == p_nw - 10'd1) & (q == p_rq - 9'd1);
  assign rq_empty       = (rq_count == '0);
  assign ctx_all_idle   = &ctx_idle;
  assign ret_ok         = ctx_all_idle & !sc_busy & !sc_start & !sc_running & rq_empty;

  always_comb begin
    for (int k = 0; k < NumContexts; k++) begin
      sc_outstanding[k] = (ctx_reducing[k] & (ctx_prog[k] != ScNone)) | ctx_sc_pending[k];
      ctx_idle[k]       = !ctx_reducing[k] & !ctx_sc_pending[k];
      ctx_done[k]       = ctx_reducing[k] & ctx_issued[k] & (ctx_pending[k] == 9'd0);
    end
    pass_wait = (pass_reads_rs & sc_outstanding[pd_ctx]) |
                ((pd.red != RedNone) & !ctx_idle[pd_ctx]);
    done_any = 1'b0;
    done_ctx = '0;
    for (int k = NumContexts - 1; k >= 0; k--) begin
      if (ctx_done[k]) begin
        done_any = 1'b1;
        done_ctx = CtxIdxW'(k);
      end
    end
    rq_push = done_any & (ctx_prog[done_ctx] != ScNone);
    case (sm_op)
      SmemRn:  sm_ready = rq_empty & !sc_busy & !sc_start & !sc_running;
      SmemSld: sm_ready = 1'b1;
      default: sm_ready = !sc_outstanding[sm_ctx];
    endcase
  end

  assign rq_pop = !sc_busy & !sc_start & !sc_running & !rq_empty;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ctx_reducing   <= '0;
      ctx_issued     <= '0;
      ctx_sc_pending <= '0;
      rq_wp          <= '0;
      rq_rp          <= '0;
      rq_count       <= '0;
      sc_running     <= 1'b0;
      for (int k = 0; k < NumContexts; k++) begin
        ctx_pending[k] <= 9'd0;
        ctx_prog[k]    <= ScNone;
      end
    end else begin
      for (int k = 0; k < NumContexts; k++) begin
        ctx_pending[k] <= ctx_pending[k] +
                          ((rd_fire & rd_is_entry & (pd.red != RedNone) & (pd_ctx == CtxIdxW'(k)))
                           ? 9'd4 : 9'd0) -
                          ((sum_done & (sum_done_ctx == CtxIdxW'(k))) ? 9'd1 : 9'd0) -
                          ((max_done & (max_done_ctx == CtxIdxW'(k))) ? 9'd1 : 9'd0);
        if (red_clear & (pd_ctx == CtxIdxW'(k))) begin
          ctx_reducing[k] <= 1'b1;
          ctx_issued[k]   <= 1'b0;
          ctx_prog[k]     <= pd.sc;
          ctx_eps[k]      <= p_imm;
        end
        if (pass_last_fire & (pd.red != RedNone) & (pd_ctx == CtxIdxW'(k))) begin
          ctx_issued[k] <= 1'b1;
        end
        if (done_any & (done_ctx == CtxIdxW'(k))) begin
          ctx_reducing[k]   <= 1'b0;
          ctx_sc_pending[k] <= (ctx_prog[k] != ScNone);
        end
        if (sc_rs_we & sc_running & (sc_ctx == CtxIdxW'(k))) begin
          ctx_sc_pending[k] <= 1'b0;
        end
      end
      if (rq_push) begin
        rq_mem[rq_wp] <= done_ctx;
        rq_wp         <= rq_wp + 1'b1;
      end
      if (rq_pop) begin
        rq_rp      <= rq_rp + 1'b1;
        sc_ctx     <= rq_mem[rq_rp];
        sc_running <= 1'b1;
      end
      if (sc_rs_we & sc_running) begin
        sc_running <= 1'b0;
      end
      rq_count <= rq_count + (CtxIdxW + 1)'(rq_push) - (CtxIdxW + 1)'(rq_pop);
    end
  end

  always_ff @(posedge clk_i) begin
    if (sc_rs_we & sc_running) begin
      for (int r = 0; r < 4; r++) begin
        if (sc_rs_sel) begin
          rs1[sc_ctx][r] <= sc_rs_val[r];
        end else begin
          rs0[sc_ctx][r] <= sc_rs_val[r];
        end
      end
    end
    if (rs_we0) begin
      for (int t = 0; t < 4; t++) begin
        rs0[sm_ctx][t] <= rs_val[t];
      end
    end
    if (rs_we1) begin
      for (int t = 0; t < 4; t++) begin
        rs1[sm_ctx][t] <= rs_val[t];
      end
    end
  end
  logic [31:0]   op_a, op_b;
  logic [31:0]   mul_p1, mul_p2, mul_p3;
  logic [1023:0] mword;

  logic [4:0]  ins_rd, ins_ra, ins_rb, sa_op, sm_col;
  logic        sa_imm, sm_colimm, sm_which, sm_neg;
  logic [2:0]  sm_op;
  logic [31:0] ins_imm;
  logic [1:0]  br_cond;
  logic [9:0]  br_tgt;

  assign ins_rd    = (kind == KindSmem) ? ins[120:116] : ins[118:114];
  assign ins_ra    = (kind == KindSmem) ? ins[115:111] :
                     (kind == KindBranch) ? ins[121:117] : ins[113:109];
  assign ins_rb    = (kind == KindSmem) ? ins[110:106] : ins[108:104];
  assign sa_op     = ins[123:119];
  assign sa_imm    = ins[103];
  assign sm_op     = ins[123:121];
  assign sm_colimm = ins[105];
  assign sm_col    = sm_colimm ? ins[104:100] : op_b[4:0];
  assign sm_which  = ins[99];
  assign sm_neg    = ins[98];
  assign ins_imm   = ins[31:0];
  assign br_cond   = ins[123:122];
  assign br_tgt    = ins[9:0];

  function automatic logic [31:0] salu(input logic [4:0] op, input logic [31:0] a,
                                       input logic [31:0] b, input logic [31:0] mulres);
    case (op)
      SaluAdd:  return a + b;
      SaluSub:  return a - b;
      SaluAnd:  return a & b;
      SaluOr:   return a | b;
      SaluXor:  return a ^ b;
      SaluShl:  return a << b[4:0];
      SaluShr:  return a >> b[4:0];
      SaluSra:  return $signed(a) >>> b[4:0];
      SaluMul:  return mulres;
      SaluSlt:  return 32'($signed(a) < $signed(b));
      SaluSltu: return 32'(a < b);
      SaluSeq:  return 32'(a == b);
      default:  return b;
    endcase
  endfunction

  function automatic logic [31:0] ftz32(input logic [31:0] f);
    return (f[30:23] == 8'd0) ? 32'd0 : f;
  endfunction

  function automatic logic [31:0] fneg32(input logic [31:0] f);
    return (f[30:23] == 8'd0) ? 32'd0 : {~f[31], f[30:0]};
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      st          <= StIdle;
      sc_start    <= 1'b0;
      gap         <= 2'd0;
      rf_we       <= 1'b0;
      sm_rd_v     <= 1'b0;
      rs_we0      <= 1'b0;
      rs_we1      <= 1'b0;
      cbuf_addr_o <= 10'd0;
      pf_cnt      <= 3'd0;
      pre_on      <= 1'b0;
      pre_done    <= 1'b0;
    end else begin
      sc_start <= 1'b0;
      rf_we    <= 1'b0;
      rs_we0   <= 1'b0;
      rs_we1   <= 1'b0;
      gap      <= (rd_fire & rd_is_entry) ? 2'd3 : (gap != 2'd0) ? gap - 2'd1 : 2'd0;
      if (rq_pop) begin
        sc_start   <= 1'b1;
        sc_recip_n <= 1'b0;
        sc_prog    <= ctx_prog[rq_mem[rq_rp]];
      end
      if (pf_cnt != 3'd7) begin
        pf_cnt <= pf_cnt + 3'd1;
      end
      if (sm_rd_fire) begin
        sm_rd_v <= 1'b0;
      end
      if (st == StIssue || st == StDrain || st == StNext) begin
        if (!pre_on && !pre_done && pf_cnt >= 3'(CbufLatency) &&
            cbuf_rdata_i[127:124] == KindPass) begin
          pre_on  <= 1'b1;
          pd_n    <= dec_pass(cbuf_rdata_i);
          pre_cnt <= 4'd0;
        end else if (pre_on) begin
          case (pre_cnt)
            4'd0: begin
              ra_q <= pd_n.rx;
              rb_q <= pd_n.rx2;
            end
            4'd1: begin
              ra_q  <= pd_n.rd;
              rb_q  <= pd_n.rq;
              pn_x  <= rf_a[11:0];
              pn_x2 <= rf_b[11:0];
            end
            4'd2: begin
              ra_q  <= pd_n.rnw;
              rb_q  <= pd_n.rn8;
              pn_d  <= rf_a[12:0];
              pn_rq <= rf_b[8:0];
            end
            4'd3: begin
              ra_q  <= pd_n.rpb;
              rb_q  <= pd_n.rimm;
              pn_nw <= rf_a[9:0];
              pn_n8 <= rf_b[7:0];
            end
            4'd4: begin
              ra_q   <= pd_n.rimm2;
              rb_q   <= 5'd0;
              pn_pb  <= rf_a[9:0];
              pn_imm <= rf_b;
            end
            default: begin
              pn_imm2  <= rf_a;
              pre_on   <= 1'b0;
              pre_done <= 1'b1;
            end
          endcase
          pre_cnt <= pre_cnt + 4'd1;
        end
      end
      case (st)
        StIdle: begin
          if (cmd_valid_i) begin
            c_src    <= cmd_src_i;
            c_src2   <= cmd_src2_i;
            c_dst    <= cmd_dst_i;
            c_rows_q <= cmd_rows_q_i;
            c_n8     <= cmd_n8_i;
            c_p_base <= cmd_p_base_i;
            c_imm    <= cmd_imm_i;
            qf       <= cmd_q_i;
            c_scr    <= !cmd_q_i ? cmd_dst_i[11:0] :
                        uses_imm(cmd_op_i) ? cmd_src2_i : cmd_imm_i[11:0];
            if (cmd_op_i == 5'(OpRun)) begin
              pc    <= cmd_aux_i[9:0];
              c_aux <= 11'd0;
            end else begin
              pc    <= RomPc[cmd_op_i[3:0]];
              c_aux <= cmd_aux_i;
            end
            cnt <= 4'd1;
            st  <= StInit;
          end
        end
        StInit: begin
          rf_we <= 1'b1;
          rf_wa <= 5'(cnt);
          case (cnt)
            4'd1:    rf_wd <= 32'(c_src);
            4'd2:    rf_wd <= 32'(c_src2);
            4'd3:    rf_wd <= 32'(c_dst);
            4'd4:    rf_wd <= 32'(c_rows_q);
            4'd5:    rf_wd <= 32'(c_n8);
            4'd6:    rf_wd <= 32'(c_p_base);
            4'd7:    rf_wd <= 32'(c_aux);
            4'd8:    rf_wd <= c_imm;
            4'd9:    rf_wd <= 32'(c_scr);
            default: rf_wd <= 32'(qf);
          endcase
          cnt <= cnt + 4'd1;
          if (cnt == 4'd10) begin
            st <= StFetch;
          end
        end
        StFetch: begin
          cbuf_addr_o <= pc;
          pf_cnt      <= 3'd0;
          st          <= StFwait;
        end
        StFwait: begin
          if (pf_cnt >= 3'(CbufLatency)) begin
            ins  <= cbuf_rdata_i;
            kind <= cbuf_rdata_i[127:124];
            st   <= StDecode;
          end
        end
        StDecode: begin
          cbuf_addr_o <= pc + 10'd1;
          pf_cnt      <= 3'd0;
          pre_done    <= 1'b0;
          pre_on      <= 1'b0;
          case (kind)
            KindPass, KindTranspose: begin
              pd  <= dec_pass(ins);
              cnt <= 4'd0;
              st  <= StPrd;
            end
            KindSalu, KindBranch: begin
              ra_q <= ins_ra;
              rb_q <= ins_rb;
              st   <= StSrd;
            end
            KindSmem: begin
              ra_q <= ins_ra;
              rb_q <= ins_rb;
              st   <= StSrd;
            end
            KindRet: begin
              if (ret_ok) begin
                st <= StIdle;
              end
            end
            default: st <= StNext;
          endcase
        end
        StPrd: begin
          case (cnt)
            4'd0: begin
              ra_q <= pd.rx;
              rb_q <= pd.rx2;
            end
            4'd1: begin
              ra_q <= pd.rd;
              rb_q <= pd.rq;
              p_x  <= rf_a[11:0];
              p_x2 <= rf_b[11:0];
            end
            4'd2: begin
              ra_q <= pd.rnw;
              rb_q <= pd.rn8;
              p_d  <= rf_a[12:0];
              p_rq <= rf_b[8:0];
            end
            4'd3: begin
              ra_q <= pd.rpb;
              rb_q <= pd.rimm;
              p_nw <= rf_a[9:0];
              p_n8 <= rf_b[7:0];
              tr_dst_tile_stride <= rf_a[11:0];
              tr_src_tile_stride <= rf_b[11:0];
            end
            4'd4: begin
              ra_q  <= pd.rimm2;
              rb_q  <= 5'd0;
              p_pb  <= rf_a[9:0];
              p_imm <= rf_b;
              tr_dst_word_stride <= rf_a[11:0];
            end
            default: begin
              p_imm2 <= rf_a;
            end
          endcase
          cnt <= cnt + 4'd1;
          if (cnt == 4'd5) begin
            st <= (kind == KindTranspose) ? StTransposeStart : StSegSetup;
          end
        end
        StTransposeStart: begin
          if (drained_r) begin
            st <= StTransposeWait;
          end
        end
        StTransposeWait: begin
          if (!tr_busy) begin
            st <= StNext;
          end
        end
        StSegSetup: begin
          if (!pass_wait) begin
            w       <= 10'd0;
            slot_ph <= 1'b0;
            slot_j  <= 3'd0;
            q       <= 9'd0;
            rb_src  <= p_x;
            rb_src2 <= p_x2;
            rb_dst  <= p_d[11:0];
            rb_abuf <= p_d;
            pt_row  <= 10'd0;
            st      <= StIssue;
          end
        end
        StIssue: begin
          if (rd_fire) begin
            if (rd_is_hold) begin
              slot_ph <= 1'b1;
            end else begin
              slot_ph <= 1'b0;
              slot_j  <= slot_j + 3'd1;
              if (w == p_nw - 10'd1) begin
                w <= 10'd0;
                if (q == p_rq - 9'd1) begin
                  st <= (pd.nodrain || overlap_ok) ? StNext : StDrain;
                end else begin
                  q       <= q + 9'd1;
                  rb_src  <= rb_src + (pd.x_s1 ? 12'd1 : 12'(p_n8));
                  rb_src2 <= rb_src2 + (pd.x_s1 ? 12'd1 : 12'(p_n8));
                  rb_dst  <= rb_dst + (pd.dst_s1 ? 12'd1 : 12'(p_n8));
                  rb_abuf <= rb_abuf + 13'(p_n8[7:2]);
                  pt_row  <= pt_row + 10'd4;
                end
              end else begin
                w <= w + 10'd1;
              end
            end
          end
        end
        StDrain: begin
          if (drained) begin
            st <= StNext;
          end
        end
        StSrd: begin
          op_a <= rf_a;
          op_b <= (kind == KindSalu && sa_imm) ? ins_imm : rf_b;
          if (kind == KindSmem) begin
            ra_q <= ins_rd;
            st   <= StMdrain;
          end else if (kind == KindBranch) begin
            st <= StBex;
          end else begin
            st <= StSex;
          end
        end
        StSex: begin
          if (sa_op == SaluMul) begin
            cnt <= 4'd0;
            st  <= StMulw;
          end else begin
            rf_we <= (ins_rd != 5'd0);
            rf_wa <= ins_rd;
            rf_wd <= salu(sa_op, op_a, op_b, 32'd0);
            st    <= StNext;
          end
        end
        StMulw: begin
          cnt <= cnt + 4'd1;
          if (cnt == 4'd3) begin
            rf_we <= (ins_rd != 5'd0);
            rf_wa <= ins_rd;
            rf_wd <= mul_p3;
            st    <= StNext;
          end
        end
        StBex: begin
          if (br_cond == 2'd0 || (br_cond == 2'd1 && op_a == 32'd0) ||
              (br_cond == 2'd2 && op_a != 32'd0)) begin
            pc <= br_tgt;
            st <= StFetch;
          end else begin
            st <= StNext;
          end
        end
        StMdrain: begin
          sm_addr <= op_a[11:0] + ins_imm[11:0];
          if (drained && sm_ready) begin
            case (sm_op)
              SmemSld, SmemRsld: begin
                sm_rd_v <= 1'b1;
                st      <= StMrd;
              end
              SmemRn: begin
                sc_start   <= 1'b1;
                sc_recip_n <= 1'b1;
                sc_nelem   <= op_a[13:0];
                st         <= StRnwait;
              end
              default: st <= StMex;
            endcase
          end
        end
        StMrd: begin
          if (sm_rd_fire) begin
            cnt <= 4'd0;
            st  <= StMwait;
          end
        end
        StMwait: begin
          cnt <= cnt + 4'd1;
          if (cnt == 4'(MemLatency - 1)) begin
            mword <= fbuf_rdata_i;
            st    <= StMex;
          end
        end
        StMex: begin
          case (sm_op)
            SmemSld: begin
              rf_we <= (ins_rd != 5'd0);
              rf_wa <= ins_rd;
              rf_wd <= mword[sm_col*32 +: 32];
            end
            SmemRsld: begin
              for (int t = 0; t < 4; t++) begin
                rs_pick[t] <= mword[(t*8 + 32'(sm_col[2:0]))*32 +: 32];
              end
            end
            SmemRsmov: begin
              rf_we <= (ins_rd != 5'd0);
              rf_wa <= ins_rd;
              rf_wd <= sm_which ? rs0[sm_ctx][sm_col[1:0]] : rs1[sm_ctx][sm_col[1:0]];
            end
            SmemSmovr: begin
              if (sm_which) begin
                rs_we0 <= 1'b1;
              end else begin
                rs_we1 <= 1'b1;
              end
              for (int t = 0; t < 4; t++) begin
                rs_val[t] <= op_a;
              end
            end
            default: begin
            end
          endcase
          st <= (sm_op == SmemRsld) ? StMrs : StNext;
        end
        StMrs: begin
          if (sm_which) begin
            rs_we0 <= 1'b1;
          end else begin
            rs_we1 <= 1'b1;
          end
          for (int t = 0; t < 4; t++) begin
            rs_val[t] <= sm_neg ? fneg32(rs_pick[t]) : ftz32(rs_pick[t]);
          end
          st <= StNext;
        end
        StRnwait: begin
          if (!sc_start && !sc_busy) begin
            st <= StNext;
          end
        end
        StNext: begin
          if (pre_on) begin
          end else if (pre_done) begin
            pc          <= pc + 10'd1;
            cbuf_addr_o <= pc + 10'd2;
            pf_cnt      <= 3'd0;
            pre_done    <= 1'b0;
            pd          <= pd_n;
            p_x         <= pn_x;
            p_x2        <= pn_x2;
            p_d         <= pn_d;
            p_rq        <= pn_rq;
            p_nw        <= pn_nw;
            p_n8        <= pn_n8;
            p_pb        <= pn_pb;
            p_imm       <= pn_imm;
            p_imm2      <= pn_imm2;
            kind        <= KindPass;
            st          <= StSegSetup;
          end else begin
            pc <= pc + 10'd1;
            if (pf_cnt >= 3'(CbufLatency)) begin
              ins  <= cbuf_rdata_i;
              kind <= cbuf_rdata_i[127:124];
              st   <= StDecode;
            end else begin
              st <= StFwait;
            end
          end
        end
        default: st <= StIdle;
      endcase
    end
  end

  always_ff @(posedge clk_i) begin
    mul_p1 <= op_a * op_b;
    mul_p2 <= mul_p1;
    mul_p3 <= mul_p2;
  end

  logic red_clear;

  assign red_clear = (st == StSegSetup) & !pass_wait & (pd.red != RedNone);

  typedef struct packed {
    logic  v;
    logic  entry;
    logic  hold;
    logic  dual;
    ctrl_t c;
  } rtag_t;

  rtag_t rt [MemLatency];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < MemLatency; i++) begin
        rt[i] <= '0;
      end
    end else begin
      rt[0].v     <= rd_fire;
      rt[0].entry <= rd_is_entry;
      rt[0].hold  <= rd_is_hold;
      rt[0].dual  <= pd.x2_en;
      rt[0].c     <= rd_ctrl;
      for (int i = 1; i < MemLatency; i++) begin
        rt[i] <= rt[i-1];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    fbuf_raddr_o <= tr_mode ? tr_ra : sm_rd_v ? sm_addr : rd_addr;
  end

  always_comb begin
    rdpipe_busy = 1'b0;
    for (int i = 0; i < MemLatency; i++) begin
      rdpipe_busy |= rt[i].v;
    end
  end

  logic  ent_v0, hold_v;
  ctrl_t ent_c0;

  assign ent_v0 = rt[MemLatency-1].v & rt[MemLatency-1].entry;
  assign hold_v = rt[MemLatency-1].v & rt[MemLatency-1].hold;
  assign ent_c0 = rt[MemLatency-1].c;

  logic [1023:0] xhold;

  always_ff @(posedge clk_i) begin
    if (hold_v) begin
      xhold <= fbuf_rdata_i;
    end
  end

  logic [1023:0] x_word, x2_word;
  logic [255:0]  pa_slice_w;

  assign x_word     = rt[MemLatency-1].dual ? xhold : fbuf_rdata_i;
  assign x2_word    = fbuf_rdata_i;
  assign pa_slice_w = pbuf_a_rdata_i[ent_c0.pa_slice*256 +: 256];

  logic [1023:0] wq_x, wq_x2;
  logic [255:0]  wq_pa;
  ctrl_t         wq_c;
  logic          ph0_v, ph1_v, ph2_v, ph3_v;
  logic [1:0]    ph_code;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ph0_v   <= 1'b0;
      ph1_v   <= 1'b0;
      ph2_v   <= 1'b0;
      ph3_v   <= 1'b0;
      ph_code <= 2'd3;
    end else begin
      ph0_v   <= ent_v0;
      ph1_v   <= ph0_v;
      ph2_v   <= ph1_v;
      ph3_v   <= ph2_v;
      ph_code <= ent_v0 ? 2'd0 : ph0_v ? 2'd1 : ph1_v ? 2'd2 : 2'd3;
    end
  end

  always_ff @(posedge clk_i) begin
    if (ent_v0) begin
      wq_x  <= x_word;
      wq_x2 <= x2_word;
      wq_pa <= pa_slice_w;
      wq_c  <= ent_c0;
    end
  end

  assign phs_v = ph0_v | ph1_v | ph2_v | ph3_v;

  logic          ent_v;
  ctrl_t         ent_c;
  logic [1023:0] ent_xw, ent_x2w;
  logic [255:0]  ent_pa;

  assign ent_v = phs_v;

  always_comb begin
    ent_c    = wq_c;
    ent_xw   = wq_x;
    ent_x2w  = wq_x2;
    ent_pa   = wq_pa;
    ent_c.ph = ph_code;
  end

  logic [Hw-1:0] ent_x, ent_x2;

  always_comb begin
    for (int l = 0; l < Lanes; l++) begin
      logic [1:0] t;
      logic [4:0] si;
      t = ent_c.ph;
      case (ent_c.px)
        PxBcast:  si = {t, ent_c.pcol};
        PxTswap1: si = {t ^ 2'b01, 3'(l)};
        PxTswap2: si = {t ^ 2'b10, 3'(l)};
        PxT4:     si = {2'(l), {t, 1'(l >> 2)}};
        PxT4b:    si = {{t[0], 1'(l >> 2)}, {t[1], 2'(l)}};
        default:  si = {t, 3'(l)};
      endcase
      ent_x[l*32 +: 32]  = ent_xw[si*32 +: 32];
      ent_x2[l*32 +: 32] = ent_x2w[{t, 3'(l)}*32 +: 32];
    end
  end

  function automatic logic [Hw-1:0] expand(input logic [1023:0] wdat, input logic tile,
                                           input logic [1:0] ph, input logic [1:0] slice);
    return tile ? wdat[ph*256 +: 256] : wdat[slice*256 +: 256];
  endfunction

  logic          e0_v, e1_v;
  ctrl_t         e0_c, e1_c;
  logic [Hw-1:0] e0_x, e0_x2, e1_x, e1_x2, e1_c0raw;
  logic [255:0]  e0_pa;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      e0_v <= 1'b0;
      e1_v <= 1'b0;
    end else begin
      e0_v <= ent_v;
      e1_v <= e0_v;
    end
  end

  always_ff @(posedge clk_i) begin
    e0_c  <= ent_c;
    e0_x  <= ent_x;
    e0_x2 <= ent_x2;
    e0_pa <= ent_pa;
  end

  function automatic logic [31:0] fkey(input logic [31:0] f);
    logic [31:0] z;
    z = (f[30:23] == 8'd0) ? 32'd0 : f;
    return z[31] ? ~z : (z ^ 32'h80000000);
  endfunction

  function automatic logic cmp_rel(input cmp_e rel, input logic lt, input logic eq);
    case (rel)
      CmpLt:   return lt;
      CmpLe:   return lt | eq;
      CmpEq:   return eq;
      CmpNe:   return !eq;
      CmpGt:   return !lt & !eq;
      CmpGe:   return !lt;
      default: return 1'b1;
    endcase
  endfunction

  logic [Lanes-1:0] e0_pred, e1_pred;
  logic [Hw-1:0]    e0_c0raw;

  always_comb begin
    for (int l = 0; l < Lanes; l++) begin
      logic [31:0] xl, x2l, lhs, rhs, kl, kr, c0v;
      logic        lt, eq;
      xl  = e0_x[l*32 +: 32];
      x2l = e0_x2[l*32 +: 32];
      lhs = (e0_c.cmp_src == CmpSrcX2Imm2) ? x2l :
            (e0_c.cmp_src == CmpSrcLane) ? (e0_c.cmp_int ? 32'({e0_c.ph, 3'(l)}) : 32'(l)) : xl;
      rhs = (e0_c.cmp_src == CmpSrcX2) ? x2l : e0_c.imm2;
      kl  = (e0_c.cmp_int | (e0_c.cmp_src == CmpSrcLane)) ? (lhs ^ 32'h80000000) : fkey(lhs);
      kr  = (e0_c.cmp_int | (e0_c.cmp_src == CmpSrcLane)) ? (rhs ^ 32'h80000000) : fkey(rhs);
      lt  = (kl < kr);
      eq  = (kl == kr);
      e0_pred[l] = (e0_c.cmp == CmpNone) ? 1'b1 : cmp_rel(e0_c.cmp, lt, eq);
      case (e0_c.c0)
        C0X2:    c0v = x2l;
        C0Pbuf:  c0v = e0_pa[l*32 +: 32];
        C0Rs0:   c0v = rs0[e0_c.ctx[CtxIdxW-1:0]][e0_c.ph];
        C0Imm:   c0v = e0_c.imm;
        default: c0v = 32'd0;
      endcase
      e0_c0raw[l*32 +: 32] = c0v;
    end
  end

  always_ff @(posedge clk_i) begin
    e1_c     <= e0_c;
    e1_x     <= e0_x;
    e1_x2    <= e0_x2;
    e1_pred  <= e0_pred;
    e1_c0raw <= e0_c0raw;
  end

  logic [Hw-1:0] c0_in, y_int_in;

  always_comb begin
    for (int l = 0; l < Lanes; l++) begin
      logic [31:0] xl, c0l;
      xl  = e1_x[l*32 +: 32];
      c0l = (e1_c.c0_pred & !e1_pred[l]) ? 32'd0 : e1_c0raw[l*32 +: 32];
      c0_in[l*32 +: 32] = c0l;
      case (e1_c.ialu)
        IaluAdd:  y_int_in[l*32 +: 32] = xl + c0l;
        IaluSub:  y_int_in[l*32 +: 32] = xl - c0l;
        IaluAnd:  y_int_in[l*32 +: 32] = xl & c0l;
        IaluOr:   y_int_in[l*32 +: 32] = xl | c0l;
        IaluXor:  y_int_in[l*32 +: 32] = xl ^ c0l;
        IaluShr1: y_int_in[l*32 +: 32] = {1'b0, xl[31:1]};
        IaluRsub: y_int_in[l*32 +: 32] = c0l - xl;
        default:  y_int_in[l*32 +: 32] = 32'd0;
      endcase
    end
  end

  ctrl_t cd [StageY];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < StageY; i++) begin
        cd[i] <= '0;
      end
    end else begin
      cd[0]       <= e1_c;
      cd[0].valid <= e1_v;
      for (int i = 1; i < StageY; i++) begin
        cd[i] <= cd[i-1];
      end
    end
  end

  always_comb begin
    lane_busy = e0_v | e1_v;
    for (int i = 0; i < StageY; i++) begin
      lane_busy |= cd[i].valid;
    end
  end

  logic [31:0]      acc_max [NumContexts][4];
  logic [31:0]      sc_acc_max [4];
  logic [4:0]       sc_acc_addr [2];
  logic [AccAw-1:0] acc_rd_addr [2];
  logic [31:0]      acc_rd_data [2];

  always_comb begin
    for (int r = 0; r < 4; r++) begin
      sc_acc_max[r] = acc_max[sc_ctx][r];
    end
    for (int p = 0; p < 2; p++) begin
      acc_rd_addr[p] = {sc_ctx, sc_acc_addr[p]};
    end
  end
  logic [Lanes-1:0] pred_d [StageY];
  logic [Hw-1:0]    yint_d [StageY];

  always_ff @(posedge clk_i) begin
    pred_d[0] <= e1_pred;
    yint_d[0] <= y_int_in;
    for (int i = 1; i < StageY; i++) begin
      pred_d[i] <= pred_d[i-1];
      yint_d[i] <= yint_d[i-1];
    end
  end

  logic [Hw-1:0] d1 [StageB];
  logic [Hw-1:0] d2 [StageY];
  logic [Hw-1:0] d2_in;

  assign d2_in = (e1_c.c1 == C1Signfix) ? e1_x : e1_x2;

  always_ff @(posedge clk_i) begin
    d1[0] <= e1_x2;
    for (int i = 1; i < StageB; i++) begin
      d1[i] <= d1[i-1];
    end
    d2[0] <= d2_in;
    for (int i = 1; i < StageY; i++) begin
      d2[i] <= d2[i-1];
    end
  end

  ctrl_t c_pa;

  assign c_pa = cd[(StageB - MemLatency) - 1];

  always_ff @(posedge clk_i) begin
    pbuf_a_addr_o <= (c_pa.valid & c_pa.pa_at_mul) ? c_pa.pa_addr : rd_ctrl.pa_addr;
  end

  ctrl_t         c_b;
  logic [Hw-1:0] b_ext, pa_mul;
  logic [31:0]   rs1_b;

  assign c_b    = cd[StageB - 1];
  assign pa_mul = expand(pbuf_a_rdata_i, c_b.pa_tile, c_b.ph, c_b.pa_slice);
  assign rs1_b  = rs1[c_b.ctx[CtxIdxW-1:0]][c_b.ph];

  always_comb begin
    for (int l = 0; l < Lanes; l++) begin
      case (c_b.b)
        BX2:     b_ext[l*32 +: 32] = d1[StageB-1][l*32 +: 32];
        BPbuf:   b_ext[l*32 +: 32] = pa_mul[l*32 +: 32];
        BRs1:    b_ext[l*32 +: 32] = rs1_b;
        BImm:    b_ext[l*32 +: 32] = c_b.imm;
        BLog2e:  b_ext[l*32 +: 32] = FpLog2e;
        default: b_ext[l*32 +: 32] = FpOne;
      endcase
    end
  end

  ctrl_t c_nl, c_c1, c_ny, c_y;

  assign c_nl = cd[StageNl - 1];
  assign c_c1 = cd[StageC1 - 1];
  assign c_ny = cd[StageY - 2];
  assign c_y  = cd[StageY - 1];

  logic [Hw-1:0] y_lane;

  for (genvar l = 0; l < Lanes; l++) begin : g_lane
    vec_lane u (
      .clk_i,
      .rst_ni,
      .in_valid_i(e1_v),
      .x_i       (e1_x[l*32 +: 32]),
      .c0_i      (c0_in[l*32 +: 32]),
      .b_ext_i   (b_ext[l*32 +: 32]),
      .b_sel_i   (c_b.b),
      .nl_i      (c_nl.nl),
      .tab_addr_o(tab_addr_o[l]),
      .tab_data_i(tab_data_i[l]),
      .c1_ext_i  (d2[StageC1-1][l*32 +: 32]),
      .c1_sel_i  (c_c1.c1),
      .neg_c1_i  (c_c1.neg_c1),
      .nl_c1_i   (c_c1.nl),
      .nl_y_i    (c_ny.nl),
      .y_o       (y_lane[l*32 +: 32])
    );
  end

  logic [Hw-1:0] y_sel, y_wr, y_red;

  always_comb begin
    for (int l = 0; l < Lanes; l++) begin
      logic [31:0] y0, alt;
      y0 = (c_y.ialu != IaluNone) ? yint_d[StageY-1][l*32 +: 32] : y_lane[l*32 +: 32];
      case (c_y.sel)
        SelX2:   alt = d2[StageY-1][l*32 +: 32];
        SelImm:  alt = c_y.imm;
        SelNull: alt = NullEntry;
        SelRs0:  alt = rs0[c_y.ctx[CtxIdxW-1:0]][c_y.ph];
        SelRs1:  alt = rs1[c_y.ctx[CtxIdxW-1:0]][c_y.ph];
        default: alt = 32'd0;
      endcase
      y_sel[l*32 +: 32] = (c_y.sel != SelNone && !pred_d[StageY-1][l]) ? alt : y0;
      if (c_y.mask[l]) begin
        y_wr[l*32 +: 32]  = 32'd0;
        y_red[l*32 +: 32] = (c_y.red == RedMax) ? FpNegInf : 32'd0;
      end else begin
        y_wr[l*32 +: 32]  = y_sel[l*32 +: 32];
        y_red[l*32 +: 32] = y_sel[l*32 +: 32];
      end
    end
  end

  vec_reduce u_red (
    .clk_i,
    .rst_ni,
    .clear_i       (red_clear),
    .clear_ctx_i   (pd_ctx),
    .in_valid_i    (c_y.valid),
    .ctx_i         (c_y.ctx[CtxIdxW-1:0]),
    .ph_i          (c_y.ph),
    .red_i         (c_y.red),
    .j_i           (c_y.j),
    .y_i           (y_red),
    .acc_rd_addr_i (acc_rd_addr),
    .acc_rd_data_o (acc_rd_data),
    .acc_max_o     (acc_max),
    .sum_done_o    (sum_done),
    .sum_done_ctx_o(sum_done_ctx),
    .max_done_o    (max_done),
    .max_done_ctx_o(max_done_ctx),
    .busy_o        (red_busy)
  );

  vec_scalar u_sc (
    .clk_i,
    .rst_ni,
    .start_i      (sc_start),
    .prog_i       (sc_prog),
    .recip_count_i(sc_recip_n),
    .nelem_i      (sc_nelem),
    .eps_i        (ctx_eps[sc_ctx]),
    .acc_rd_addr_o(sc_acc_addr),
    .acc_rd_data_i(acc_rd_data),
    .acc_max_i    (sc_acc_max),
    .rs_we_o      (sc_rs_we),
    .rs_sel_o     (sc_rs_sel),
    .rs_val_o     (sc_rs_val),
    .rn_o         (rn),
    .busy_o       (sc_busy)
  );

  logic [Hw-1:0] yq [3];

  always_ff @(posedge clk_i) begin
    if (c_y.valid && c_y.ph != 2'd3) begin
      yq[c_y.ph] <= y_wr;
    end
  end

  logic [1023:0] y_word;
  logic          word_v;

  assign y_word = {y_wr, yq[2], yq[1], yq[0]};
  assign word_v = c_y.valid & (c_y.ph == 2'd3);

  logic          wf_push, wf_pop;
  logic [1035:0] wf_din, wf_dout;
  logic [5:0]    wf_count;

  assign wf_push = word_v & (c_y.wr == WrFbuf);
  assign wf_din  = {c_y.waddr[11:0], y_word};
  assign wf_pop  = fbuf_wr_req_o & fbuf_wr_grant_i & !tr_mode;

  sfifo #(
    .Width      (12 + 1024),
    .Depth      (32),
    .AfullMargin(StageY / 4 + MemLatency + 5)
  ) u_wf (
    .clk_i,
    .rst_ni,
    .clear_i(1'b0),
    .push_i (wf_push),
    .din_i  (wf_din),
    .pop_i  (wf_pop),
    .dout_o (wf_dout),
    .empty_o(wf_empty),
    .afull_o(wf_afull),
    .count_o(wf_count)
  );

  assign fbuf_wr_req_o = tr_mode ? tr_wr : !wf_empty;
  assign fbuf_waddr_o  = tr_mode ? tr_wa : wf_dout[1035:1024];
  assign fbuf_wdata_o  = tr_mode ? tr_wd : wf_dout[1023:0];

  localparam int PendDepth = 32;
  localparam int PendAw    = $clog2(PendDepth);

  logic [11:0]          pend_addr [PendDepth];
  logic [PendDepth-1:0] pend_v;
  logic [PendAw-1:0]    pend_wp, pend_rp;
  logic [PendAw:0]      pend_count;
  logic                 pend_push, pend_pop, pend_hit, pend_hit_r, pend_known, pend_empty_r;

  assign pend_push  = rd_fire & rd_is_entry & (wr_eff == WrFbuf);
  assign pend_pop   = wf_pop;
  assign pend_full  = (pend_count == (PendAw + 1)'(PendDepth));
  assign pend_clear = pend_empty_r | (pend_known & !pend_hit_r);

  always_comb begin
    pend_hit = 1'b0;
    for (int i = 0; i < PendDepth; i++) begin
      pend_hit |= pend_v[i] & (pend_addr[i] == rd_addr);
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pend_v       <= '0;
      pend_wp      <= '0;
      pend_rp      <= '0;
      pend_count   <= '0;
      pend_hit_r   <= 1'b0;
      pend_known   <= 1'b0;
      pend_empty_r <= 1'b1;
    end else begin
      pend_hit_r   <= pend_hit;
      pend_known   <= (st == StIssue) & !rd_fire;
      pend_empty_r <= (pend_count == '0) & !pend_push;
      if (pend_push) begin
        pend_v[pend_wp] <= 1'b1;
        pend_wp         <= pend_wp + 1'b1;
      end
      if (pend_pop) begin
        pend_v[pend_rp] <= 1'b0;
        pend_rp         <= pend_rp + 1'b1;
      end
      pend_count <= pend_count + (PendAw + 1)'(pend_push) - (PendAw + 1)'(pend_pop);
    end
  end

  always_ff @(posedge clk_i) begin
    if (pend_push) begin
      pend_addr[pend_wp] <= rd_ctrl.waddr[11:0];
    end
  end

  logic       qv;
  logic       q1_v, q2_v;
  logic [1:0] q1_ph, q2_ph, q1_qw, q2_qw;
  logic [7:0] ex1 [Lanes];
  logic [7:0] em2;

  assign qv = c_y.valid & (c_y.wr == WrAbuf);

  function automatic logic [7:0] max2(input logic [7:0] a, input logic [7:0] b);
    return (a > b) ? a : b;
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      q1_v <= 1'b0;
      q2_v <= 1'b0;
    end else begin
      q1_v <= qv;
      q2_v <= q1_v;
    end
  end

  always_ff @(posedge clk_i) begin
    q1_ph <= c_y.ph;
    q1_qw <= c_y.qw;
    for (int l = 0; l < Lanes; l++) begin
      ex1[l] <= y_wr[l*32 + 23 +: 8];
    end
    q2_ph <= q1_ph;
    q2_qw <= q1_qw;
    em2   <= max2(max2(max2(ex1[0], ex1[1]), max2(ex1[2], ex1[3])),
                  max2(max2(ex1[4], ex1[5]), max2(ex1[6], ex1[7])));
  end

  logic [7:0] bm [4];
  logic [7:0] bm_new;

  assign bm_new = ((q2_qw == 2'd0) || (em2 > bm[q2_ph])) ? em2 : bm[q2_ph];

  always_ff @(posedge clk_i) begin
    if (q2_v) begin
      bm[q2_ph] <= bm_new;
    end
  end

  logic        ef_push;
  logic [31:0] ef_din, ef_dout;
  logic        ef_empty, ef_afull, ef_pop;
  logic [4:0]  ef_count;

  assign ef_push = q2_v & (q2_ph == 2'd3) & (q2_qw == 2'd3);
  assign ef_din  = {bm_new, bm[2], bm[1], bm[0]};

  sfifo #(
    .Width      (32),
    .Depth      (16),
    .AfullMargin(8)
  ) u_ef (
    .clk_i,
    .rst_ni,
    .clear_i(1'b0),
    .push_i (ef_push),
    .din_i  (ef_din),
    .pop_i  (ef_pop),
    .dout_o (ef_dout),
    .empty_o(ef_empty),
    .afull_o(ef_afull),
    .count_o(ef_count)
  );

  logic [1038:0] df_dout;
  logic          df_empty, df_pop;
  logic [5:0]    df_count;

  sfifo #(
    .Width      (13 + 2 + 1024),
    .Depth      (32),
    .AfullMargin(StageY / 4 + MemLatency + 5)
  ) u_df (
    .clk_i,
    .rst_ni,
    .clear_i(1'b0),
    .push_i (qv & (c_y.ph == 2'd3)),
    .din_i  ({c_y.waddr, c_y.qw, y_word}),
    .pop_i  (df_pop),
    .dout_o (df_dout),
    .empty_o(df_empty),
    .afull_o(df_afull),
    .count_o(df_count)
  );

  logic [12:0]   df_addr;
  logic [1:0]    df_qw;
  logic [1023:0] df_y;

  assign df_addr = df_dout[1038:1026];
  assign df_qw   = df_dout[1025:1024];
  assign df_y    = df_dout[1023:0];

  typedef struct packed {
    logic        zero;
    logic        sign;
    logic [23:0] mant;
    logic [4:0]  sh;
  } cvt1_t;

  typedef struct packed {
    logic       zero;
    logic       sign;
    logic [7:0] ip;
    logic       g;
    logic       stk;
  } cvt_t;

  function automatic cvt1_t to_int8_a1(input logic [31:0] f, input logic [7:0] emax);
    cvt1_t      o;
    logic [8:0] sh;
    o.mant = (f[30:23] == 8'd0) ? 24'd0 : {1'b1, f[22:0]};
    sh     = 9'(emax) - 9'(f[30:23]) + 9'd17;
    o.sign = f[31];
    o.zero = (sh >= 9'd25) || (o.mant == 24'd0);
    o.sh   = sh[4:0];
    return o;
  endfunction

  function automatic cvt_t to_int8_a2(input cvt1_t a);
    cvt_t        o;
    logic [23:0] low;
    o.sign = a.sign;
    o.zero = a.zero;
    o.ip   = 8'(a.mant >> a.sh);
    low    = a.mant & ((24'd1 << a.sh) - 24'd1);
    o.g    = low[a.sh - 1];
    o.stk  = |(low & ((24'd1 << (a.sh - 1)) - 24'd1));
    return o;
  endfunction

  function automatic logic [7:0] to_int8_b(input cvt_t a);
    logic [8:0] mag;
    mag = a.zero ? 9'd0 : (9'(a.ip) + (9'(a.g) & (9'(a.stk) | 9'(a.ip[0]))));
    if (a.sign) begin
      return (mag >= 9'd128) ? 8'h80 : 8'(-mag);
    end else begin
      return (mag >= 9'd128) ? 8'h7F : 8'(mag);
    end
  endfunction

  logic [2:0]    cv_ph;
  logic [1023:0] cv_w;
  logic [12:0]   cv_addr;
  logic [1:0]    cv_qw;
  logic [31:0]   cv_e;

  assign df_pop = !df_empty & !ef_empty & !af_afull & (cv_ph == 3'd0 || cv_ph == 3'd4);
  assign ef_pop = df_pop & (df_qw == 2'd3);

  logic          cv_v;
  logic [1:0]    cv_t;
  logic [Hw-1:0] cv_in;

  assign cv_v  = (cv_ph != 3'd0);
  assign cv_t  = 2'(cv_ph - 3'd1);
  assign cv_in = cv_w[cv_t*256 +: 256];

  cvt1_t       cb [Lanes];
  cvt_t        ca [Lanes];
  logic        cb_v, cb_last, ca_v, ca_last;
  logic [1:0]  cb_t, ca_t, cb_qw, ca_qw;
  logic [12:0] cb_addr, ca_addr;
  logic [31:0] cb_e, ca_e;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      cv_ph <= 3'd0;
      cb_v  <= 1'b0;
      ca_v  <= 1'b0;
    end else begin
      cv_ph <= df_pop ? 3'd1 : (cv_ph == 3'd0 || cv_ph == 3'd4) ? 3'd0 : cv_ph + 3'd1;
      cb_v  <= cv_v;
      ca_v  <= cb_v;
    end
  end

  always_ff @(posedge clk_i) begin
    if (df_pop) begin
      cv_w    <= df_y;
      cv_addr <= df_addr;
      cv_qw   <= df_qw;
      cv_e    <= ef_dout;
    end
    for (int l = 0; l < Lanes; l++) begin
      cb[l] <= to_int8_a1(cv_in[l*32 +: 32], cv_e[cv_t * 8 +: 8]);
    end
    cb_t    <= cv_t;
    cb_qw   <= cv_qw;
    cb_addr <= cv_addr;
    cb_e    <= cv_e;
    cb_last <= (cv_t == 2'd3) & (cv_qw == 2'd3);
    for (int l = 0; l < Lanes; l++) begin
      ca[l] <= to_int8_a2(cb[l]);
    end
    ca_t    <= cb_t;
    ca_qw   <= cb_qw;
    ca_addr <= cb_addr;
    ca_e    <= cb_e;
    ca_last <= cb_last;
  end

  logic [1023:0] aw;
  logic          af_push;
  logic [12:0]   af_addr;
  logic [31:0]   af_e;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      af_push <= 1'b0;
    end else begin
      af_push <= ca_v & ca_last;
    end
  end

  always_ff @(posedge clk_i) begin
    if (ca_v) begin
      for (int l = 0; l < Lanes; l++) begin
        aw[ca_t * 256 + ca_qw * 64 + l * 8 +: 8] <= to_int8_b(ca[l]);
      end
    end
    if (ca_v & ca_last) begin
      af_addr <= ca_addr;
      af_e    <= ca_e;
    end
  end

  logic [1068:0] af_din;

  always_comb begin
    af_din[1068:1056] = af_addr;
    for (int r = 0; r < 4; r++) begin
      af_din[1024 + r*8 +: 8] = (af_e[r*8 +: 8] == 8'd0) ? 8'd127 : af_e[r*8 +: 8];
    end
    af_din[1023:0] = aw;
  end

  logic [1068:0] af_dout;
  logic          af_pop;
  logic [4:0]    af_count;

  assign af_pop = abuf_wr_req_o & abuf_wr_grant_i;

  sfifo #(
    .Width      (13 + 1056),
    .Depth      (16),
    .AfullMargin(8)
  ) u_af (
    .clk_i,
    .rst_ni,
    .clear_i(1'b0),
    .push_i (af_push),
    .din_i  (af_din),
    .pop_i  (af_pop),
    .dout_o (af_dout),
    .empty_o(af_empty),
    .afull_o(af_afull),
    .count_o(af_count)
  );

  assign abuf_wr_req_o = !af_empty;
  assign abuf_waddr_o  = af_dout[1068:1056];
  assign abuf_wdata_o  = af_dout[1055:0];
  assign quant_busy    = q1_v | q2_v | !df_empty | !ef_empty | cv_v | cb_v | ca_v | af_push;

  logic unused_engine;

  assign unused_engine = ^{c_y, c_ny, c_nl, c_c1, c_b, c_pa, rn, wf_count, ef_count, ef_afull,
                           df_count, af_count, sm_ctx_field, pd.ctx};

endmodule
