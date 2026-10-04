import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 1. Configuration
base_model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
adapter_dir = "./verilog-lora-model"

print("Booting up Industry-Grade VLSI AI...")

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Merge with your heavily-trained Emilgoh adapters
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

# 2. The Unseen Test Case: A buggy SPI FSM
# This is a classic industry bug. It compiles fine, but infers a latch because 
# the combinational block doesn't cover the default state.
industry_test_code = """
module spi_fsm (input enable, input [1:0] state, output reg [1:0] next_state);
  localparam WAIT = 2'b00, SHIFT = 2'b01, DONE = 2'b10;
  
  always @(*) begin
    case (state)
      WAIT: if (enable) next_state = SHIFT; else next_state = WAIT;
      SHIFT: next_state = DONE;
      DONE: next_state = WAIT;
      // Missing default case!
    endcase
  end
endmodule
"""

prompt = f"Instruction: Analyze the following complex hierarchical Verilog module for semantic or logical errors. Identify the bug, explain the issue, and provide the corrected code.\nInput:\n{industry_test_code}\nOutput:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n--- AI VLSI Analysis ---")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        temperature=0.1, 
        do_sample=False
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_only = generated_text.split("Output:\n")[-1]
print(response_only)