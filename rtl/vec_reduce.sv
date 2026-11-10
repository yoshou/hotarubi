module vec_reduce
  import isa_pkg::*;
  import vec_pkg::*;
(
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic               clear_i,
  input  logic [CtxIdxW-1:0] clear_ctx_i,
  input  logic               in_valid_i,
  input  logic [CtxIdxW-1:0] ctx_i,
  input  logic [1:0]         ph_i,
  input  red_e               red_i,
  input  logic [2:0]         j_i,
  input  logic [Hw-1:0]      y_i,
  input  logic [AccAw-1:0]   acc_rd_addr_i [2],
  output logic [31:0]        acc_rd_data_o [2],
  output logic [31:0]        acc_max_o [NumContexts][4],
  output logic               sum_done_o,
  output logic [CtxIdxW-1:0] sum_done_ctx_o,
  output logic               max_done_o,
  output logic [CtxIdxW-1:0] max_done_ctx_o,
  output logic               busy_o
);

  localparam int LAdd       = 6;
  localparam int LTree      = 3 * LAdd;
  localparam int AccEntries = 1 << AccAw;

  logic [Hw-1:0]      y_r;
  logic               in_valid_r;
  logic [1:0]         ph_r;
  logic [CtxIdxW-1:0] ctx_r;
  logic [2:0]         j_r;
  red_e               red_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_valid_r <= 1'b0;
    end else begin
      in_valid_r <= in_valid_i;
    end
  end

  always_ff @(posedge clk_i) begin
    y_r   <= y_i;
    ph_r  <= ph_i;
    ctx_r <= ctx_i;
    j_r   <= j_i;
    red_r <= red_i;
  end

  logic [31:0] l1 [4], l2 [2], l3;
  logic [3:0]  l1_valid;
  logic [1:0]  l2_valid;
  logic        l3_valid, acc_valid;

  for (genvar i = 0; i < 4; i++) begin : g_l1
    fp32_add u (
      .clk_i,
      .rst_ni,
      .in_valid_i (1'b1),
      .a_i        (y_r[(2*i)*32 +: 32]),
      .b_i        (y_r[(2*i+1)*32 +: 32]),
      .out_valid_o(l1_valid[i]),
      .y_o        (l1[i])
    );
  end

  for (genvar i = 0; i < 2; i++) begin : g_l2
    fp32_add u (
      .clk_i,
      .rst_ni,
      .in_valid_i (1'b1),
      .a_i        (l1[2*i]),
      .b_i        (l1[2*i+1]),
      .out_valid_o(l2_valid[i]),
      .y_o        (l2[i])
    );
  end

  fp32_add u_l3 (
    .clk_i,
    .rst_ni,
    .in_valid_i (1'b1),
    .a_i        (l2[0]),
    .b_i        (l2[1]),
    .out_valid_o(l3_valid),
    .y_o        (l3)
  );

  logic               v_d   [LTree + LAdd];
  logic [2:0]         j_d   [LTree];
  logic [1:0]         p_d   [LTree];
  logic [CtxIdxW-1:0] c_d   [LTree];
  logic               sum_d [LTree];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < LTree + LAdd; i++) begin
        v_d[i] <= 1'b0;
      end
    end else begin
      v_d[0] <= in_valid_r & (red_r != RedNone);
      for (int i = 1; i < LTree + LAdd; i++) begin
        v_d[i] <= v_d[i-1];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    j_d[0]   <= j_r;
    p_d[0]   <= ph_r;
    c_d[0]   <= ctx_r;
    sum_d[0] <= (red_r == RedSum);
    for (int i = 1; i < LTree; i++) begin
      j_d[i]   <= j_d[i-1];
      p_d[i]   <= p_d[i-1];
      c_d[i]   <= c_d[i-1];
      sum_d[i] <= sum_d[i-1];
    end
  end

  logic               tree_v;
  logic [2:0]         tree_j;
  logic [1:0]         tree_p;
  logic [CtxIdxW-1:0] tree_c;

  assign tree_v = v_d[LTree-1] & sum_d[LTree-1];
  assign tree_j = j_d[LTree-1];
  assign tree_p = p_d[LTree-1];
  assign tree_c = c_d[LTree-1];

  logic [31:0]        acc_in, acc_out;
  logic [2:0]         j_w [LAdd];
  logic [1:0]         p_w [LAdd];
  logic [CtxIdxW-1:0] c_w [LAdd];
  logic               v_w [LAdd];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < LAdd; i++) begin
        v_w[i] <= 1'b0;
      end
    end else begin
      v_w[0] <= tree_v;
      for (int i = 1; i < LAdd; i++) begin
        v_w[i] <= v_w[i-1];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    j_w[0] <= tree_j;
    p_w[0] <= tree_p;
    c_w[0] <= tree_c;
    for (int i = 1; i < LAdd; i++) begin
      j_w[i] <= j_w[i-1];
      p_w[i] <= p_w[i-1];
      c_w[i] <= c_w[i-1];
    end
  end

  (* ram_style = "distributed" *) logic [31:0] acc_mem [AccEntries];
  logic [AccEntries-1:0] acc_vld;
  logic [AccAw-1:0]      acc_ra, acc_wa;
  logic                  acc_we;

  assign acc_ra = {tree_c, tree_p, tree_j};
  assign acc_wa = {c_w[LAdd-1], p_w[LAdd-1], j_w[LAdd-1]};
  assign acc_we = v_w[LAdd-1];
  assign acc_in = acc_vld[acc_ra] ? acc_mem[acc_ra] : 32'd0;

  fp32_add u_acc (
    .clk_i,
    .rst_ni,
    .in_valid_i (1'b1),
    .a_i        (acc_in),
    .b_i        (l3),
    .out_valid_o(acc_valid),
    .y_o        (acc_out)
  );

  always_ff @(posedge clk_i) begin
    if (acc_we) begin
      acc_mem[acc_wa] <= acc_out;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      acc_vld <= '0;
    end else begin
      if (acc_we) begin
        acc_vld[acc_wa] <= 1'b1;
      end
      if (clear_i) begin
        for (int e = 0; e < 32; e++) begin
          acc_vld[{clear_ctx_i, 5'(e)}] <= 1'b0;
        end
      end
    end
  end

  for (genvar p = 0; p < 2; p++) begin : g_acc_rd
    assign acc_rd_data_o[p] =
        acc_vld[acc_rd_addr_i[p]] ? acc_mem[acc_rd_addr_i[p]] : 32'd0;
  end

  assign sum_done_o     = acc_we;
  assign sum_done_ctx_o = c_w[LAdd-1];

  function automatic logic [31:0] key(input logic [31:0] f);
    return f[31] ? ~f : (f ^ 32'h80000000);
  endfunction

  function automatic logic [31:0] fmax(input logic [31:0] a, input logic [31:0] b);
    return (key(a) >= key(b)) ? a : b;
  endfunction

  logic [31:0]        m1 [4], m2 [2], m3;
  logic               mv [4];
  logic [1:0]         mp [3];
  logic [CtxIdxW-1:0] mc [3];

  always_ff @(posedge clk_i) begin
    for (int i = 0; i < 4; i++) begin
      m1[i] <= fmax(y_r[(2*i)*32 +: 32], y_r[(2*i+1)*32 +: 32]);
    end
    for (int i = 0; i < 2; i++) begin
      m2[i] <= fmax(m1[2*i], m1[2*i+1]);
    end
    m3    <= fmax(m2[0], m2[1]);
    mp[0] <= ph_r;
    mp[1] <= mp[0];
    mp[2] <= mp[1];
    mc[0] <= ctx_r;
    mc[1] <= mc[0];
    mc[2] <= mc[1];
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < 4; i++) begin
        mv[i] <= 1'b0;
      end
    end else begin
      mv[0] <= in_valid_r & (red_r == RedMax);
      for (int i = 1; i < 4; i++) begin
        mv[i] <= mv[i-1];
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (mv[2]) begin
      acc_max_o[mc[2]][mp[2]] <= fmax(acc_max_o[mc[2]][mp[2]], m3);
    end
    if (clear_i) begin
      for (int r = 0; r < 4; r++) begin
        acc_max_o[clear_ctx_i][r] <= FpNegInf;
      end
    end
  end

  assign max_done_o     = mv[2];
  assign max_done_ctx_o = mc[2];

  always_comb begin
    busy_o = in_valid_r;
    for (int i = 0; i < LTree + LAdd; i++) begin
      busy_o |= v_d[i];
    end
    for (int i = 0; i < LAdd; i++) begin
      busy_o |= v_w[i];
    end
    for (int i = 0; i < 4; i++) begin
      busy_o |= mv[i];
    end
  end

  logic unused_valid;

  assign unused_valid = ^{l1_valid, l2_valid, l3_valid, acc_valid};

endmodule
