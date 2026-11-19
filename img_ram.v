module img_ram (
    input wire clk,
    input wire wr_en,
    input wire [9:0] wr_addr,
    input wire [7:0] wr_data,
    input wire [9:0] rd_addr,
    output reg [7:0] rd_data
);
  (* ram_style = "block" *) reg [7:0] mem[0:783];
  always @(posedge clk) begin
    if (wr_en) mem[wr_addr] <= wr_data;
    rd_data <= mem[rd_addr];
  end
endmodule
