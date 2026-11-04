module dma_burst (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        req_valid_i,
  output logic        req_ready_o,
  input  logic [31:0] req_addr_i,
  input  logic [15:0] req_beats_i,
  input  logic [1:0]  req_tag_i,
  output logic        b_valid_o,
  input  logic        b_ready_i,
  output logic [31:0] b_addr_o,
  output logic [3:0]  b_len_o,
  output logic [1:0]  b_tag_o,
  output logic        idle_o
);

  logic        busy;
  logic [1:0]  tag;
  logic [31:0] cur;
  logic [15:0] rem;
  logic [8:0]  to4k;
  logic [15:0] len;

  assign to4k = 9'd256 - {1'b0, cur[11:4]};

  always_comb begin
    len = rem;
    if (len > 16'd16) begin
      len = 16'd16;
    end
    if (len > 16'(to4k)) begin
      len = 16'(to4k);
    end
  end

  assign req_ready_o = !busy;
  assign b_valid_o   = busy;
  assign b_addr_o    = cur;
  assign b_len_o     = 4'(len - 16'd1);
  assign b_tag_o     = tag;
  assign idle_o      = !busy;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      busy <= 1'b0;
      cur  <= '0;
      rem  <= '0;
    end else if (!busy) begin
      if (req_valid_i && req_beats_i != 16'd0) begin
        busy <= 1'b1;
        cur  <= req_addr_i;
        rem  <= req_beats_i;
        tag  <= req_tag_i;
      end
    end else if (b_ready_i) begin
      cur <= cur + 32'({len, 4'b0});
      rem <= rem - len;
      if (rem == len) begin
        busy <= 1'b0;
      end
    end
  end

endmodule
