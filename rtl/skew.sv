module skew #(
  parameter int Width = 16
) (
  input  logic              clk_i,
  input  logic [32*Width-1:0] in_i,
  output logic [Width-1:0]  out_o [32]
);

  logic [Width-1:0] pipe [32][32];

  always_ff @(posedge clk_i) begin
    for (int k = 1; k < 32; k++) begin
      pipe[k][0] <= in_i[k*Width +: Width];
      for (int s = 1; s < k; s++) begin
        pipe[k][s] <= pipe[k][s-1];
      end
    end
  end

  always_comb begin
    out_o[0] = in_i[0 +: Width];
    for (int k = 1; k < 32; k++) begin
      out_o[k] = pipe[k][k-1];
    end
  end

endmodule
