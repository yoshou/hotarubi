module sfifo #(
  parameter int Width       = 32,
  parameter int Depth       = 64,
  parameter int AfullMargin = 32
) (
  input  logic                   clk_i,
  input  logic                   rst_ni,
  input  logic                   clear_i,
  input  logic                   push_i,
  input  logic [Width-1:0]       din_i,
  input  logic                   pop_i,
  output logic [Width-1:0]       dout_o,
  output logic                   empty_o,
  output logic                   afull_o,
  output logic [$clog2(Depth):0] count_o
);

  localparam int AddrWidth = $clog2(Depth);

  (* ram_style = "distributed" *) logic [Width-1:0] mem [Depth];
  logic [AddrWidth-1:0] wptr, rptr;

  always_ff @(posedge clk_i) begin
    if (push_i) begin
      mem[wptr] <= din_i;
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wptr    <= '0;
      rptr    <= '0;
      count_o <= '0;
    end else if (clear_i) begin
      wptr    <= '0;
      rptr    <= '0;
      count_o <= '0;
    end else begin
      if (push_i) begin
        wptr <= wptr + 1'b1;
      end
      if (pop_i) begin
        rptr <= rptr + 1'b1;
      end
      count_o <= count_o + (AddrWidth+1)'(push_i) - (AddrWidth+1)'(pop_i);
    end
  end

  assign dout_o  = mem[rptr];
  assign empty_o = (count_o == '0);
  assign afull_o = (count_o >= (AddrWidth+1)'(Depth - AfullMargin));

endmodule
