module vec_scalar
  import vec_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        start_i,
  input  sc_e         prog_i,
  input  logic        recip_count_i,
  input  logic [13:0] nelem_i,
  input  logic [31:0] eps_i,
  input  logic [31:0] acc_sum_i [4][8],
  input  logic [31:0] acc_max_i [4],
  input  logic        ext_we0_i,
  input  logic        ext_we1_i,
  input  logic [31:0] ext_val_i [4],
  output logic [31:0] c0s_o [4],
  output logic [31:0] bs_o [4],
  output logic [31:0] rn_o,
  output logic        busy_o
);

  localparam int Step = 7;

  logic [31:0] ma [4], mb [4], my [4], aa [4], ab [4], ay [4];
  logic [3:0]  m_valid, a_valid;

  for (genvar r = 0; r < 4; r++) begin : g_u
    fp32_mul u_m (
      .clk_i,
      .rst_ni,
      .in_valid_i (1'b1),
      .a_i        (ma[r]),
      .b_i        (mb[r]),
      .out_valid_o(m_valid[r]),
      .y_o        (my[r])
    );
    fp32_add u_a (
      .clk_i,
      .rst_ni,
      .in_valid_i (1'b1),
      .a_i        (aa[r]),
      .b_i        (ab[r]),
      .out_valid_o(a_valid[r]),
      .y_o        (ay[r])
    );
  end

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

  logic [31:0] t0 [4], t1 [4], t2 [4], t3 [4], av [4], yv [4];

  typedef enum logic [1:0] {
    PhIdle,
    PhIssue,
    PhWait,
    PhWrite
  } ph_e;

  ph_e        ph;
  logic [5:0] step;
  logic [2:0] wait_cnt;
  sc_e        cur;
  logic       cur_rn;
  logic [2:0] ns;
  logic [5:0] nstart;
  logic [2:0] nlen;

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
  logic [31:0] oa [4], ob [4], imm [4];
  logic        last;

  function automatic logic tree_step(input logic [5:0] s);
    return (s <= 6'd6);
  endfunction

  always_comb begin
    opk  = 2'd0;
    dst  = DstT0;
    last = 1'b0;
    for (int r = 0; r < 4; r++) begin
      oa[r]  = 32'd0;
      ob[r]  = 32'd0;
      imm[r] = 32'd0;
    end
    if (cur_rn) begin
      if (step == 6'd0) begin
        opk = 2'd3;
        dst = DstY;
        for (int r = 0; r < 4; r++) begin
          imm[r] = magic_recip(float_n(nelem_i));
        end
      end else if (step <= 6'd12) begin
        case (ns)
          3'd0: begin
            opk = 2'd2;
            dst = DstT1;
            for (int r = 0; r < 4; r++) begin
              oa[r] = av[r];
              ob[r] = yv[r];
            end
          end
          3'd1: begin
            opk = 2'd1;
            dst = DstT1;
            for (int r = 0; r < 4; r++) begin
              oa[r] = FpTwo;
              ob[r] = fneg(t1[r]);
            end
          end
          default: begin
            opk = 2'd2;
            dst = DstY;
            for (int r = 0; r < 4; r++) begin
              oa[r] = yv[r];
              ob[r] = t1[r];
            end
          end
        endcase
      end else begin
        opk = 2'd3;
        dst = DstRn;
        for (int r = 0; r < 4; r++) begin
          imm[r] = yv[r];
        end
        last = 1'b1;
      end
    end else if (cur == ScNegmax) begin
      opk = 2'd3;
      dst = DstC0s;
      for (int r = 0; r < 4; r++) begin
        imm[r] = fneg(acc_max_i[r]);
      end
      last = 1'b1;
    end else if (cur == ScMax) begin
      opk = 2'd3;
      dst = DstBs;
      for (int r = 0; r < 4; r++) begin
        imm[r] = acc_max_i[r];
      end
      last = 1'b1;
    end else if (tree_step(step)) begin
      opk = 2'd1;
      case (step)
        6'd0: begin
          dst = DstT0;
          for (int r = 0; r < 4; r++) begin
            oa[r] = acc_sum_i[r][0];
            ob[r] = acc_sum_i[r][1];
          end
        end
        6'd1: begin
          dst = DstT1;
          for (int r = 0; r < 4; r++) begin
            oa[r] = acc_sum_i[r][2];
            ob[r] = acc_sum_i[r][3];
          end
        end
        6'd2: begin
          dst = DstT2;
          for (int r = 0; r < 4; r++) begin
            oa[r] = acc_sum_i[r][4];
            ob[r] = acc_sum_i[r][5];
          end
        end
        6'd3: begin
          dst = DstT3;
          for (int r = 0; r < 4; r++) begin
            oa[r] = acc_sum_i[r][6];
            ob[r] = acc_sum_i[r][7];
          end
        end
        6'd4: begin
          dst = DstT0;
          for (int r = 0; r < 4; r++) begin
            oa[r] = t0[r];
            ob[r] = t1[r];
          end
        end
        6'd5: begin
          dst = DstT1;
          for (int r = 0; r < 4; r++) begin
            oa[r] = t2[r];
            ob[r] = t3[r];
          end
        end
        default: begin
          dst = DstT0;
          for (int r = 0; r < 4; r++) begin
            oa[r] = t0[r];
            ob[r] = t1[r];
          end
        end
      endcase
    end else begin
      case (cur)
        ScMean: begin
          if (step == 6'd7) begin
            opk = 2'd2;
            dst = DstT0;
            for (int r = 0; r < 4; r++) begin
              oa[r] = t0[r];
              ob[r] = rn_o;
            end
          end else begin
            opk = 2'd3;
            dst = DstC0s;
            for (int r = 0; r < 4; r++) begin
              imm[r] = fneg(t0[r]);
            end
            last = 1'b1;
          end
        end
        ScSum: begin
          opk = 2'd3;
          dst = DstBs;
          for (int r = 0; r < 4; r++) begin
            imm[r] = t0[r];
          end
          last = 1'b1;
        end
        ScRstd: begin
          if (step == 6'd7) begin
            opk = 2'd2;
            dst = DstT0;
            for (int r = 0; r < 4; r++) begin
              oa[r] = t0[r];
              ob[r] = rn_o;
            end
          end else if (step == 6'd8) begin
            opk = 2'd1;
            dst = DstT0;
            for (int r = 0; r < 4; r++) begin
              oa[r] = t0[r];
              ob[r] = eps_i;
            end
          end else if (step == 6'd9) begin
            opk = 2'd3;
            dst = DstY;
            for (int r = 0; r < 4; r++) begin
              imm[r] = magic_rsqrt(t0[r][31:1]);
            end
          end else if (step <= 6'd29) begin
            case (ns)
              3'd0: begin
                opk = 2'd2;
                dst = DstT1;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = t0[r];
                  ob[r] = yv[r];
                end
              end
              3'd1: begin
                opk = 2'd2;
                dst = DstT1;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = t1[r];
                  ob[r] = yv[r];
                end
              end
              3'd2: begin
                opk = 2'd2;
                dst = DstT1;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = t1[r];
                  ob[r] = FpHalf;
                end
              end
              3'd3: begin
                opk = 2'd1;
                dst = DstT1;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = FpOnePointFive;
                  ob[r] = fneg(t1[r]);
                end
              end
              default: begin
                opk = 2'd2;
                dst = DstY;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = yv[r];
                  ob[r] = t1[r];
                end
              end
            endcase
          end else begin
            opk = 2'd3;
            dst = DstBs;
            for (int r = 0; r < 4; r++) begin
              imm[r] = yv[r];
            end
            last = 1'b1;
          end
        end
        default: begin
          if (step == 6'd7) begin
            opk = 2'd3;
            dst = DstY;
            for (int r = 0; r < 4; r++) begin
              imm[r] = magic_recip(t0[r]);
            end
          end else if (step <= 6'd19) begin
            case (ns)
              3'd0: begin
                opk = 2'd2;
                dst = DstT1;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = t0[r];
                  ob[r] = yv[r];
                end
              end
              3'd1: begin
                opk = 2'd1;
                dst = DstT1;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = FpTwo;
                  ob[r] = fneg(t1[r]);
                end
              end
              default: begin
                opk = 2'd2;
                dst = DstY;
                for (int r = 0; r < 4; r++) begin
                  oa[r] = yv[r];
                  ob[r] = t1[r];
                end
              end
            endcase
          end else begin
            opk = 2'd3;
            dst = DstBs;
            for (int r = 0; r < 4; r++) begin
              imm[r] = yv[r];
            end
            last = 1'b1;
          end
        end
      endcase
    end
  end

  logic [31:0] oa_r [4], ob_r [4], imm_r [4];
  logic [1:0]  opk_r;
  dst_e        dst_r;
  logic        last_r;

  always_ff @(posedge clk_i) begin
    if (ph == PhIssue) begin
      for (int r = 0; r < 4; r++) begin
        oa_r[r]  <= oa[r];
        ob_r[r]  <= ob[r];
        imm_r[r] <= imm[r];
      end
      opk_r  <= opk;
      dst_r  <= dst;
      last_r <= last;
    end
  end

  always_comb begin
    for (int r = 0; r < 4; r++) begin
      ma[r] = oa_r[r];
      mb[r] = ob_r[r];
      aa[r] = oa_r[r];
      ab[r] = ob_r[r];
    end
  end

  logic [31:0] res [4];

  always_comb begin
    for (int r = 0; r < 4; r++) begin
      res[r] = (opk_r == 2'd1) ? ay[r] : (opk_r == 2'd2) ? my[r] : imm_r[r];
    end
  end

  assign busy_o = (ph != PhIdle);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      ph   <= PhIdle;
      step <= 6'd0;
    end else begin
      case (ph)
        PhIdle: begin
          if (!start_i) begin
            if (ext_we0_i) begin
              for (int r = 0; r < 4; r++) begin
                c0s_o[r] <= ext_val_i[r];
              end
            end
            if (ext_we1_i) begin
              for (int r = 0; r < 4; r++) begin
                bs_o[r] <= ext_val_i[r];
              end
            end
          end else begin
            ph     <= PhIssue;
            step   <= 6'd0;
            cur    <= prog_i;
            cur_rn <= recip_count_i;
            ns     <= 3'd0;
            nstart <= recip_count_i ? 6'd1 : (prog_i == ScRstd) ? 6'd10 : 6'd8;
            nlen   <= (!recip_count_i && prog_i == ScRstd) ? 3'd5 : 3'd3;
            if (recip_count_i) begin
              for (int r = 0; r < 4; r++) begin
                av[r] <= float_n(nelem_i);
              end
            end
          end
        end
        PhIssue: begin
          if (opk == 2'd3) begin
            ph <= PhWrite;
          end else begin
            ph       <= PhWait;
            wait_cnt <= 3'(Step - 1);
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
          for (int r = 0; r < 4; r++) begin
            case (dst_r)
              DstT0:   t0[r]    <= res[r];
              DstT1:   t1[r]    <= res[r];
              DstT2:   t2[r]    <= res[r];
              DstT3:   t3[r]    <= res[r];
              DstY:    yv[r]    <= res[r];
              DstC0s:  c0s_o[r] <= res[r];
              DstBs:   bs_o[r]  <= res[r];
              default: rn_o     <= res[r];
            endcase
          end
          step <= step + 6'd1;
          if (step + 6'd1 == nstart) begin
            ns <= 3'd0;
          end else begin
            ns <= (ns == nlen - 3'd1) ? 3'd0 : ns + 3'd1;
          end
          ph <= last_r ? PhIdle : PhIssue;
        end
      endcase
    end
  end

  logic unused_valid;

  assign unused_valid = ^{m_valid, a_valid};

endmodule
