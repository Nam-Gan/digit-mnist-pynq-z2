module neuron #(
    in_depth = 8,
    weight_depth = 8,
    bias_depth = 32
) (
    input wire clk,
    input wire load,
    input wire en,
    input wire signed [bias_depth-1:0] bias,
    input wire signed [weight_depth-1:0] w,
    input wire [in_depth-1:0] x,
    output reg signed [bias_depth-1:0] acc
);
  always @(posedge clk) begin
    if (load) acc <= bias;
    else if (en) acc <= acc + $signed({1'b0, x}) * $signed(w);
    else acc <= acc;
  end
endmodule


