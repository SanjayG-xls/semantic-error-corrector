module mux (
    input  wire a,
    input  wire b,
    input  wire sel,
    output wire y
);

    // Bug 2: Incomplete sensitivity list
    always @(posedge sel) begin
        if (sel)
            y = b;
        else
            y = a;
    end

endmodule