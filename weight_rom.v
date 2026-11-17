module weight_rom #(
    prev_neurons = 784,
    neurons = 64,
    FILE = "mem/l1_w.mem"
) (
    input wire clk,
    input wire [$clog2(prev_neurons)-1:0] addr,
    output reg [neurons*8-1:0] row
);
  (* rom_style = "block" *) reg [neurons*8-1:0] mem[0:prev_neurons-1];
  initial $readmemh(FILE, mem);
  always @(posedge clk) row <= mem[addr];
endmodule
