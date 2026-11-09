module reg_slice #(
  parameter int Width = 32
) (
  input  logic             clk_i,
  input  logic             rst_ni,
  input  logic             in_valid_i,
  output logic             in_ready_o,
  input  logic [Width-1:0] in_data_i,
  output logic             out_valid_o,
  input  logic             out_ready_i,
  output logic [Width-1:0] out_data_o
);

  logic             full, load;
  logic [Width-1:0] skid;

  assign in_ready_o = !full;
  assign load       = !out_valid_o || out_ready_i;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      out_valid_o <= 1'b0;
      full        <= 1'b0;
    end else if (load) begin
      out_valid_o <= full || in_valid_i;
      full        <= 1'b0;
    end else if (in_valid_i && !full) begin
      full <= 1'b1;
    end
  end

  always_ff @(posedge clk_i) begin
    if (load) begin
      out_data_o <= full ? skid : in_data_i;
    end
    if (!load && in_valid_i && !full) begin
      skid <= in_data_i;
    end
  end

endmodule
