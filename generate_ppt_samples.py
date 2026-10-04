import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Boot up the AI
base_model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
adapter_dir = "./verilog-lora-model"

print("Booting up VLSI AI for PPT Samples...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

# Three distinct textbook VLSI bugs
samples = [
    {
        "title": "Sample 1: Sequential Race Condition",
        "code": """module flip_flop (input clk, input d, output reg q);
  always @(posedge clk) begin
    q = d; // Bug: Blocking assignment
  end
endmodule"""
    },
    {
        "title": "Sample 2: Unintended Combinational Latch",
        "code": """module mux (input sel, input a, input b, output reg y);
  always @(*) begin
    if (sel)
      y = a;
    // Bug: Missing else branch
  end
endmodule"""
    },
    {
        "title": "Sample 3: FSM State Deadlock",
        "code": """module simple_fsm (input clk, input rst_n, input step, output reg state);
  always @(*) begin
    case (state)
      1'b0: if (step) state = 1'b1;
      1'b1: state = 1'b0;
      // Bug: Missing default case
    endcase
  end
endmodule"""
    }
]

print("\n" + "="*50)
for i, sample in enumerate(samples):
    print(f"\n{sample['title']}")
    print("-" * 30)
    print(f"Input Code:\n{sample['code']}\n")
    
    prompt = f"Instruction: Analyze the following Verilog code for semantic or logical errors. Identify the bug, explain the issue, and provide the corrected code.\nInput:\n{sample['code']}\nOutput:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = generated_text.split("Output:\n")[-1]
    
    print("AI Analysis Output:")
    print(response_only.strip())
    print("\n" + "="*50)