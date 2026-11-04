module vec_reduce
  import vec_pkg::*;
(
  input  logic          clk_i,
  input  logic          rst_ni,
  input  logic          clear_i,
  input  logic          in_valid_i,
  input  logic [1:0]    ph_i,
  input  red_e          red_i,
  input  logic [2:0]    j_i,
  input  logic [Hw-1:0] y_i,
  output logic [31:0]   acc_sum_o [4][8],
  output logic [31:0]   acc_max_o [4],
  output logic          busy_o
);

  localparam int LAdd  = 6;
  localparam int LTree = 3 * LAdd;

  logic [Hw-1:0] y_r;
  logic          in_valid_r;
  logic [1:0]    ph_r;
  logic [2:0]    j_r;
  red_e          red_r;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      in_valid_r <= 1'b0;
    end else begin
      in_valid_r <= in_valid_i;
      y_r        <= y_i;
      ph_r       <= ph_i;
      j_r        <= j_i;
      red_r      <= red_i;
    end
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

  logic       v_d   [LTree + LAdd];
  logic [2:0] j_d   [LTree];
  logic [1:0] p_d   [LTree];
  logic       sum_d [LTree];

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
    sum_d[0] <= (red_r == RedSum);
    for (int i = 1; i < LTree; i++) begin
      j_d[i]   <= j_d[i-1];
      p_d[i]   <= p_d[i-1];
      sum_d[i] <= sum_d[i-1];
    end
  end

  logic       tree_v;
  logic [2:0] tree_j;
  logic [1:0] tree_p;

  assign tree_v = v_d[LTree-1] & sum_d[LTree-1];
  assign tree_j = j_d[LTree-1];
  assign tree_p = p_d[LTree-1];

  logic [31:0] acc_in, acc_out;
  logic [2:0]  j_w [LAdd];
  logic [1:0]  p_w [LAdd];
  logic        v_w [LAdd];

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
    for (int i = 1; i < LAdd; i++) begin
      j_w[i] <= j_w[i-1];
      p_w[i] <= p_w[i-1];
    end
  end

  assign acc_in = acc_sum_o[tree_p][tree_j];

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
    if (clear_i) begin
      for (int r = 0; r < 4; r++) begin
        for (int k = 0; k < 8; k++) begin
          acc_sum_o[r][k] <= 32'd0;
        end
      end
    end else if (v_w[LAdd-1]) begin
      acc_sum_o[p_w[LAdd-1]][j_w[LAdd-1]] <= acc_out;
    end
  end

  function automatic logic [31:0] key(input logic [31:0] f);
    return f[31] ? ~f : (f ^ 32'h80000000);
  endfunction

  function automatic logic [31:0] fmax(input logic [31:0] a, input logic [31:0] b);
    return (key(a) >= key(b)) ? a : b;
  endfunction

  logic [31:0] m1 [4], m2 [2], m3;
  logic        mv [4];
  logic [1:0]  mp [3];

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
    if (clear_i) begin
      for (int r = 0; r < 4; r++) begin
        acc_max_o[r] <= FpNegInf;
      end
    end else if (mv[2]) begin
      acc_max_o[mp[2]] <= fmax(acc_max_o[mp[2]], m3);
    end
  end

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
