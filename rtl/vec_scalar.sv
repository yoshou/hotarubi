module vec_scalar
  import vec_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        start_i,
  input  sc_t         prog_i,
  input  logic        recip_count_i,
  input  logic [13:0] nelem_i,
  input  logic [31:0] eps_i,
  output logic [4:0]  acc_rd_addr_o [2],
  input  logic [31:0] acc_rd_data_i [2],
  input  logic [31:0] acc_max_i [4],
  output logic        rs_we_o,
  output logic        rs_sel_o,
  output logic [31:0] rs_val_o [4],
  output logic [31:0] rn_o,
  output logic        busy_o
);

  localparam int Step = 7;

  logic [31:0] ma, mb, my, aa, ab, ay;
  logic        m_valid, a_valid;

  fp32_mul u_m (
    .clk_i,
    .rst_ni,
    .in_valid_i (1'b1),
    .a_i        (ma),
    .b_i        (mb),
    .out_valid_o(m_valid),
    .y_o        (my)
  );

  fp32_add u_a (
    .clk_i,
    .rst_ni,
    .in_valid_i (1'b1),
    .a_i        (aa),
    .b_i        (ab),
    .out_valid_o(a_valid),
    .y_o        (ay)
  );

  function automatic logic [31:0] fneg(input logic [31:0] f);
    return (f[30:0] == 31'd0) ? 32'd0 : {~f[31], f[30:0]};
  endfunction

  function automatic logic [31:0] magic_recip(input logic [31:0] a);
    return 32'h7EF311C7 - a;
  endfunction

  function automatic logic [31:0] magic_rsqrt(input logic [30:0] a_hi);
    return 32'h5F375A86 - {1'b0, a_hi};
  endfunction

  function automatic logic [31:0] float_n(input logic [13:0] n);
    logic [3:0]  p;
    logic [22:0] sh;
    p = 4'd0;
    for (int i = 13; i >= 0; i--) begin
      if (n[i]) begin
        p = 4'(i);
        break;
      end
    end
    sh = 23'(37'(n) << (23 - p));
    return {1'b0, 8'(8'd127 + 8'(p)), sh[22:0]};
  endfunction

  logic [31:0] t0 [4], t1 [4], t2 [4], t3 [4], yv [4];

  typedef enum logic [1:0] {
    PhIdle,
    PhIssue,
    PhWait,
    PhWrite
  } ph_e;

  ph_e        ph;
  logic [5:0] step;
  logic [2:0] wait_cnt;
  logic [1:0] tok;
  sf_e        cur_fn;
  logic       cur_rs0, cur_max;
  logic       cur_rn;
  logic [2:0] ns;
  logic [5:0] nstart;
  logic [2:0] nlen;
  logic [5:0] b_load, b_scale, b_bias, b_seed, b_write;
  logic [5:0] n_load, n_scale, n_bias, n_seed, n_newton, n_write;
  logic       n_iterative;

  always_comb begin
    n_iterative = recip_count_i || (prog_i.fn == SfReciprocal) || (prog_i.fn == SfRsqrt);
    n_load      = (!recip_count_i && !prog_i.max) ? 6'd7 : 6'd0;
    n_scale     = n_load + 6'(!recip_count_i && prog_i.max &&
                              (prog_i.rn || prog_i.eps || n_iterative));
    n_bias      = n_scale + 6'(!recip_count_i && prog_i.rn);
    n_seed      = n_bias + 6'(!recip_count_i && prog_i.eps);
    n_newton    = n_seed + 6'(n_iterative);
    n_write     = n_newton + ((recip_count_i || prog_i.fn == SfReciprocal) ? 6'd12 :
                              (prog_i.fn == SfRsqrt) ? 6'd20 : 6'd0);
  end

  typedef enum logic [2:0] {
    DstT0,
    DstT1,
    DstT2,
    DstT3,
    DstY,
    DstC0s,
    DstBs,
    DstRn
  } dst_e;

  logic [1:0]  opk;
  dst_e        dst;
  logic [31:0] oa, ob, imm [4];
  logic        last;

  assign acc_rd_addr_o[0] = {tok, step[1:0], 1'b0};
  assign acc_rd_addr_o[1] = {tok, step[1:0], 1'b1};

  logic iterative, recip;

  assign recip     = cur_rn || (cur_fn == SfReciprocal);
  assign iterative = recip || (cur_fn == SfRsqrt);

  always_comb begin
    opk  = 2'd0;
    dst  = DstT0;
    last = 1'b0;
    oa   = 32'd0;
    ob   = 32'd0;
    for (int r = 0; r < 4; r++) begin
      imm[r] = 32'd0;
    end
    if (step < b_load) begin
      opk = 2'd1;
      case (step)
        6'd0, 6'd1, 6'd2, 6'd3: begin
          dst = dst_e'(step[1:0]);
          oa  = acc_rd_data_i[0];
          ob  = acc_rd_data_i[1];
        end
        6'd4: begin
          dst = DstT0;
          oa  = t0[tok];
          ob  = t1[tok];
        end
        6'd5: begin
          dst = DstT1;
          oa  = t2[tok];
          ob  = t3[tok];
        end
        default: begin
          dst = DstT0;
          oa  = t0[tok];
          ob  = t1[tok];
        end
      endcase
    end else if (step < b_scale) begin
      opk = 2'd3;
      dst = DstT0;
      for (int r = 0; r < 4; r++) begin
        imm[r] = acc_max_i[r];
      end
    end else if (step < b_bias) begin
      opk = 2'd2;
      dst = DstT0;
      oa  = t0[tok];
      ob  = rn_o;
    end else if (step < b_seed) begin
      opk = 2'd1;
      dst = DstT0;
      oa  = t0[tok];
      ob  = eps_i;
    end else if (step < nstart) begin
      opk = 2'd3;
      dst = DstY;
      for (int r = 0; r < 4; r++) begin
        imm[r] = recip ? magic_recip(t0[r]) : magic_rsqrt(t0[r][31:1]);
      end
    end else if (step < b_write) begin
      opk = 2'd2;
      dst = DstT1;
      if (recip) begin
        case (ns)
          3'd0: begin
            oa = t0[tok];
            ob = yv[tok];
          end
          3'd1: begin
            opk = 2'd1;
            oa  = FpTwo;
            ob  = fneg(t1[tok]);
          end
          default: begin
            dst = DstY;
            oa  = yv[tok];
            ob  = t1[tok];
          end
        endcase
      end else begin
        case (ns)
          3'd0: begin
            oa = t0[tok];
            ob = yv[tok];
          end
          3'd1: begin
            oa = t1[tok];
            ob = yv[tok];
          end
          3'd2: begin
            oa = t1[tok];
            ob = FpHalf;
          end
          3'd3: begin
            opk = 2'd1;
            oa  = FpOnePointFive;
            ob  = fneg(t1[tok]);
          end
          default: begin
            dst = DstY;
            oa  = yv[tok];
            ob  = t1[tok];
          end
        endcase
      end
    end else begin
      opk  = 2'd3;
      dst  = cur_rn ? DstRn : cur_rs0 ? DstC0s : DstBs;
      last = 1'b1;
      for (int r = 0; r < 4; r++) begin
        logic [31:0] v;
        v = iterative ? yv[r] : (cur_max && step == 6'd0) ? acc_max_i[r] : t0[r];
        imm[r] = (cur_fn == SfNegate) ? fneg(v) : v;
      end
    end
  end

  logic [31:0] imm_r [4];
  logic [1:0]  opk_r;
  dst_e        dst_r;
  logic        last_r;

  always_ff @(posedge clk_i) begin
    if (ph == PhIssue) begin
      ma <= oa;
      mb <= ob;
      aa <= oa;
      ab <= ob;
      if (tok == 2'd0) begin
        for (int r = 0; r < 4; r++) begin
          imm_r[r] <= imm[r];
        end
        opk_r  <= opk;
        dst_r  <= dst;
        last_r <= last;
      end
    end
  end

  logic [31:0] res;
  logic        step_done, write_rs;

  assign res       = (opk_r == 2'd1) ? ay : my;
  assign step_done = (ph == PhWrite) && ((opk_r == 2'd3) || (tok == 2'd3));
  assign write_rs  = (dst_r == DstC0s) || (dst_r == DstBs);
  assign busy_o    = (ph != PhIdle);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ph   <= PhIdle;
      step <= 6'd0;
      tok  <= 2'd0;
    end else begin
      case (ph)
        PhIdle: begin
          if (start_i) begin
            ph     <= PhIssue;
            step   <= 6'd0;
            tok    <= 2'd0;
            cur_fn  <= prog_i.fn;
            cur_rs0 <= prog_i.rs0;
            cur_max <= prog_i.max;
            cur_rn  <= recip_count_i;
            ns      <= 3'd0;
            b_load  <= n_load;
            b_scale <= n_scale;
            b_bias  <= n_bias;
            b_seed  <= n_seed;
            nstart  <= n_newton;
            b_write <= n_write;
            nlen    <= (!recip_count_i && prog_i.fn == SfRsqrt) ? 3'd5 : 3'd3;
          end
        end
        PhIssue: begin
          if (opk == 2'd3) begin
            ph <= PhWrite;
          end else if (tok == 2'd3) begin
            tok      <= 2'd0;
            ph       <= PhWait;
            wait_cnt <= 3'(Step - 5);
          end else begin
            tok <= tok + 2'd1;
          end
        end
        PhWait: begin
          if (wait_cnt == 3'd0) begin
            ph <= PhWrite;
          end else begin
            wait_cnt <= wait_cnt - 1'b1;
          end
        end
        PhWrite: begin
          if (step_done) begin
            tok  <= 2'd0;
            step <= step + 6'd1;
            if (step + 6'd1 == nstart) begin
              ns <= 3'd0;
            end else begin
              ns <= (ns == nlen - 3'd1) ? 3'd0 : ns + 3'd1;
            end
            ph <= last_r ? PhIdle : PhIssue;
          end else begin
            tok <= tok + 2'd1;
          end
        end
        default: ph <= PhIdle;
      endcase
    end
  end

  always_ff @(posedge clk_i) begin
    if (ph == PhIdle && start_i && recip_count_i) begin
      for (int r = 0; r < 4; r++) begin
        t0[r] <= float_n(nelem_i);
      end
    end
    if (ph == PhWrite) begin
      if (opk_r == 2'd3) begin
        for (int r = 0; r < 4; r++) begin
          case (dst_r)
            DstT0:   t0[r] <= imm_r[r];
            DstT1:   t1[r] <= imm_r[r];
            DstT2:   t2[r] <= imm_r[r];
            DstT3:   t3[r] <= imm_r[r];
            DstY:    yv[r] <= imm_r[r];
            DstRn:   rn_o  <= imm_r[r];
            default: begin
            end
          endcase
        end
      end else begin
        case (dst_r)
          DstT0:   t0[tok] <= res;
          DstT1:   t1[tok] <= res;
          DstT2:   t2[tok] <= res;
          DstT3:   t3[tok] <= res;
          DstY:    yv[tok] <= res;
          DstRn:   rn_o    <= res;
          default: begin
          end
        endcase
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      rs_we_o <= 1'b0;
    end else begin
      rs_we_o <= step_done && write_rs;
    end
  end

  always_ff @(posedge clk_i) begin
    rs_sel_o <= (dst_r == DstBs);
    if (ph == PhWrite && write_rs) begin
      if (opk_r == 2'd3) begin
        for (int r = 0; r < 4; r++) begin
          rs_val_o[r] <= imm_r[r];
        end
      end else begin
        rs_val_o[tok] <= res;
      end
    end
  end

  logic unused_valid;

  assign unused_valid = ^{m_valid, a_valid};

endmodule
