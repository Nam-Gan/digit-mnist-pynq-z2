module requant #(
    activation_depth = 8,
    SHIFT = 8
) (
    input wire signed [31:0] acc,
    output wire [activation_depth-1:0] activation
);
  wire signed [31:0] v;
  assign v = acc >>> SHIFT;
  assign activation = (v < 0) ? 8'd0 : ((v > 255) ? 8'd255 : v[7:0]);

endmodule

