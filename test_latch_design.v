module buggy_mux (
    input  wire a,
    input  wire b,
    input  wire sel,
    output wire y      // Bug 1: Data type error
);

    // Bug 2: Incomplete sensitivity list
    always @(a or b) begin
        if (sel)
            y = b;
        else
            y = a;
    end

endmodule