module vec_lane
  import vec_pkg::*;
(
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        in_valid_i,
  input  logic [31:0] x_i,
  input  logic [31:0] c0_i,
  input  logic [31:0] b_ext_i,
  input  b_e          b_sel_i,
  input  nl_e         nl_i,
  output logic [9:0]  tab_addr_o,
  input  logic [63:0] tab_data_i,
  input  logic [31:0] c1_ext_i,
  input  c1_e         c1_sel_i,
  input  logic        neg_c1_i,
  input  nl_e         nl_c1_i,
  input  nl_e         nl_y_i,
  output logic [31:0] y_o
);

  logic [31:0] x_r, c0_r, pre;
  logic        v_r;
  logic        pre_valid, prod_valid, d_valid, y0_valid;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      v_r <= 1'b0;
    end else begin
      v_r <= in_valid_i;
    end
  end

  always_ff @(posedge clk_i) begin
    x_r  <= x_i;
    c0_r <= c0_i;
  end

  fp32_add u_pre (
    .clk_i,
    .rst_ni,
    .in_valid_i (v_r),
    .a_i        (x_r),
    .b_i        (c0_r),
    .out_valid_o(pre_valid),
    .y_o        (pre)
  );

  logic [31:0] bsel, prod;

  always_comb begin
    case (b_sel_i)
      BOne:    bsel = FpOne;
      BPre:    bsel = pre;
      default: bsel = b_ext_i;
    endcase
  end

  logic [31:0] pre_r, bsel_r;

  always_ff @(posedge clk_i) begin
    pre_r  <= pre;
    bsel_r <= bsel;
  end

  fp32_mul u_mul (
    .clk_i,
    .rst_ni,
    .in_valid_i (1'b1),
    .a_i        (pre_r),
    .b_i        (bsel_r),
    .out_valid_o(prod_valid),
    .y_o        (prod)
  );

  logic        sg;
  logic [7:0]  ex;
  logic [23:0] mant;

  assign sg   = prod[31];
  assign ex   = prod[30:23];
  assign mant = (ex == 8'd0) ? 24'd0 : {1'b1, prod[22:0]};

  logic [24:0] fx_act;
  logic        sat_act;

  always_comb begin
    sat_act = (ex >= 8'd131);
    if (ex == 8'd130) begin
      fx_act = {mant, 1'b0};
    end else if (ex == 8'd129) begin
      fx_act = {1'b0, mant};
    end else if (ex >= 8'd105) begin
      fx_act = 25'({1'b0, mant} >> (8'd129 - ex));
    end else begin
      fx_act = 25'd0;
    end
  end

  logic [32:0] mag;
  logic        inexact, big;

  always_comb begin
    big = (ex >= 8'd135);
    if (ex >= 8'd125) begin
      mag     = 33'(mant) << (ex - 8'd125);
      inexact = 1'b0;
    end else if (ex >= 8'd101) begin
      mag     = 33'(33'(mant) >> (8'd125 - ex));
      inexact = |(mant & ((24'd1 << (8'd125 - ex)) - 24'd1));
    end else begin
      mag     = 33'd0;
      inexact = (mant != 24'd0);
    end
  end

  logic        f1_act, f1_sg, f1_sat, f1_big, f1_inexact;
  logic [24:0] f1_fx;
  logic [32:0] f1_mag;

  always_ff @(posedge clk_i) begin
    f1_act     <= (nl_i == NlAct);
    f1_sg      <= sg;
    f1_sat     <= sat_act;
    f1_big     <= big;
    f1_inexact <= inexact;
    f1_fx      <= fx_act;
    f1_mag     <= mag;
  end

  logic [33:0] ft;

  assign ft = f1_sg ? -(34'(f1_mag) + 34'(f1_inexact)) : 34'(f1_mag);

  logic [15:0] frac_tadr;
  logic [8:0]  n_tadr;
  logic        sat_tadr, zero_tadr, inf_tadr;

  always_ff @(posedge clk_i) begin
    if (f1_act) begin
      tab_addr_o <= {1'b1, f1_fx[24:16]};
      frac_tadr  <= f1_fx[15:0];
      sat_tadr   <= f1_sat;
      zero_tadr  <= 1'b0;
      inf_tadr   <= 1'b0;
      n_tadr     <= 9'd0;
    end else begin
      tab_addr_o <= {1'b0, ft[24:16]};
      frac_tadr  <= ft[15:0];
      n_tadr     <= ft[33:25];
      sat_tadr   <= 1'b0;
      zero_tadr  <= f1_big & f1_sg;
      inf_tadr   <= f1_big & ~f1_sg;
    end
  end

  logic [15:0] frac_d [TabLatency];

  always_ff @(posedge clk_i) begin
    frac_d[0] <= frac_tadr;
    for (int i = 1; i < TabLatency; i++) begin
      frac_d[i] <= frac_d[i-1];
    end
  end

  logic [31:0] d, v_d [InterpLatency];
  logic [63:0] tab_r;

  always_ff @(posedge clk_i) begin
    tab_r <= tab_data_i;
  end

  fp32_ufmul u_int (
    .clk_i,
    .rst_ni,
    .in_valid_i (1'b1),
    .s_i        (tab_r[63:32]),
    .frac_i     (frac_d[TabLatency-1]),
    .out_valid_o(d_valid),
    .y_o        (d)
  );

  always_ff @(posedge clk_i) begin
    v_d[0] <= tab_r[31:0];
    for (int i = 1; i < InterpLatency; i++) begin
      v_d[i] <= v_d[i-1];
    end
  end

  localparam int DProd = StageC1 - StageNl;

  logic [31:0] prod_d [DProd];

  always_ff @(posedge clk_i) begin
    prod_d[0] <= prod;
    for (int i = 1; i < DProd; i++) begin
      prod_d[i] <= prod_d[i-1];
    end
  end

  logic [31:0] pa, pb, prod_c1, y0;
  logic        xneg;

  assign prod_c1 = prod_d[DProd-1];
  assign xneg    = c1_ext_i[31];

  logic [31:0] max_b;
  logic        max_ge;

  assign max_b = neg_c1_i ? 32'd0 : c1_ext_i;

  always_comb begin
    if (prod_c1[30:0] == 31'd0 && max_b[30:0] == 31'd0) begin
      max_ge = 1'b1;
    end else if (prod_c1[31] != max_b[31]) begin
      max_ge = !prod_c1[31];
    end else if (!prod_c1[31]) begin
      max_ge = (prod_c1[30:0] >= max_b[30:0]);
    end else begin
      max_ge = (prod_c1[30:0] <= max_b[30:0]);
    end
  end

  always_comb begin
    if (nl_c1_i != NlNone) begin
      pa = v_d[InterpLatency-1];
      pb = d;
    end else if (c1_sel_i == C1Max) begin
      pa = max_ge ? prod_c1 : max_b;
      pb = 32'd0;
    end else begin
      pa = (c1_sel_i == C1Signfix && xneg) ? {~prod_c1[31], prod_c1[30:0]} : prod_c1;
      case (c1_sel_i)
        C1Zero:    pb = 32'd0;
        C1Signfix: pb = xneg ? c1_ext_i : 32'd0;
        default:   pb = neg_c1_i ? {~c1_ext_i[31], c1_ext_i[30:0]} : c1_ext_i;
      endcase
    end
  end

  logic [31:0] pa_r, pb_r;

  always_ff @(posedge clk_i) begin
    pa_r <= pa;
    pb_r <= pb;
  end

  fp32_add u_post (
    .clk_i,
    .rst_ni,
    .in_valid_i (1'b1),
    .a_i        (pa_r),
    .b_i        (pb_r),
    .out_valid_o(y0_valid),
    .y_o        (y0)
  );

  localparam int DFlag = (StageC1 + PostLatency) - StageTabAddr;

  logic [11:0] flag_d [DFlag];

  always_ff @(posedge clk_i) begin
    flag_d[0] <= {n_tadr, sat_tadr, zero_tadr, inf_tadr};
    for (int i = 1; i < DFlag; i++) begin
      flag_d[i] <= flag_d[i-1];
    end
  end

  logic [8:0] n_y;
  logic       sat_y, zero_y, inf_y;

  assign n_y    = flag_d[DFlag-1][11:3];
  assign sat_y  = flag_d[DFlag-1][2];
  assign zero_y = flag_d[DFlag-1][1];
  assign inf_y  = flag_d[DFlag-1][0];

  logic signed [10:0] eo;

  assign eo = $signed({3'b000, y0[30:23]}) + $signed({{2{n_y[8]}}, n_y});

  always_ff @(posedge clk_i) begin
    case (nl_y_i)
      NlExp2: begin
        if (zero_y | (eo <= 11'sd0) | (y0[30:23] == 8'd0)) begin
          y_o <= 32'd0;
        end else if (inf_y | (eo >= 11'sd255)) begin
          y_o <= 32'h7F800000;
        end else begin
          y_o <= {1'b0, eo[7:0], y0[22:0]};
        end
      end
      NlAct:   y_o <= sat_y ? FpOne : y0;
      default: y_o <= y0;
    endcase
  end

  logic unused_valid;

  assign unused_valid = ^{pre_valid, prod_valid, d_valid, y0_valid};

endmodule
