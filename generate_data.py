import json
import re

# 1. Expanded Clean VLSI Verilog Examples
clean_modules = [
    {
        "name": "D-Flip Flop",
        "type": "sequential_blocking",
        "code": "module d_ff (input clk, input d, output reg q);\n  always @(posedge clk) begin\n    q <= d;\n  end\nendmodule"
    },
    {
        "name": "2-to-1 Multiplexer",
        "type": "unintended_latch",
        "code": "module mux21 (input a, input b, input sel, output reg y);\n  always @(*) begin\n    if (sel)\n      y = a;\n    else\n      y = b;\n  end\nendmodule"
    },
    {
        "name": "AND Gate Logic",
        "type": "multi_driven_net",
        "code": "module and_gate (input a, input b, output wire y);\n  assign y = a & b;\nendmodule"
    }
]

dataset = []
system_prompt = "Analyze the following Verilog code for semantic or logical errors. Identify the bug, explain the issue, and provide the corrected code."

# 2. Advanced Bug Injection Engine
for module in clean_modules:
    clean_code = module["code"]
    bug_type = module["type"]
    
    # --- BUG 1: Sequential Blocking Assignment ---
    if bug_type == "sequential_blocking":
        buggy_code = clean_code.replace("<=", "=")
        expected_output = f"### Error Analysis\n* **Error Type:** Semantic Error (Race Condition)\n* **Description:** Blocking assignments (`=`) used inside a sequential `always @(posedge clk)` block. This causes simulation race conditions. Use non-blocking (`<=`) for sequential logic.\n\n### Corrected Code\n```verilog\n{clean_code}\n```"
        dataset.append({"instruction": system_prompt, "input": buggy_code, "output": expected_output})

    # --- BUG 2: Unintended Latch (Missing Else) ---
    elif bug_type == "unintended_latch":
        # Remove the else branch to create a latch
        buggy_code = re.sub(r'else\s+y = b;', '', clean_code)
        expected_output = f"### Error Analysis\n* **Error Type:** Logical Error (Unintended Latch)\n* **Description:** The `if` statement inside the combinational `always @(*)` block is missing an `else` branch. If `sel` is 0, `y` holds its previous value, inferring an unwanted latch instead of pure combinational logic.\n\n### Corrected Code\n```verilog\n{clean_code}\n```"
        dataset.append({"instruction": system_prompt, "input": buggy_code, "output": expected_output})

    # --- BUG 3: Multi-Driven Net ---
    elif bug_type == "multi_driven_net":
        # Add a conflicting continuous assignment
        buggy_code = clean_code.replace("endmodule", "  assign y = a | b; // Conflicting driver\nendmodule")
        expected_output = f"### Error Analysis\n* **Error Type:** Semantic Error (Multi-Driven Net)\n* **Description:** The output `y` is being driven by multiple continuous assignments simultaneously (both AND and OR logic). This will resolve to an unknown 'X' state in simulation and fail synthesis.\n\n### Corrected Code\n```verilog\n{clean_code}\n```"
        dataset.append({"instruction": system_prompt, "input": buggy_code, "output": expected_output})

# 3. Save to JSON
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Successfully generated advanced dataset.json with {len(dataset)} complex VLSI bug examples!")