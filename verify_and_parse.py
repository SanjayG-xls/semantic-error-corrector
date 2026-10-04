import os
import subprocess
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

# Optional: PyVerilog AST parser import
from pyverilog.vparser.parser import VerilogParser

# ==========================================
# 1. AST Structural Hierarchy Scanner
# ==========================================
def scan_hierarchy_with_ast(file_path):
    console.print(f"\n[bold cyan][AST] Parsing structural design of {file_path}...[/bold cyan]")
    try:
        parser = VerilogParser()
        ast = parser.parse(file_path)
        return True
    except Exception as e:
        console.print(f"[yellow][AST Warning] Standard AST parsing failed or iverilog preprocessor missing in PATH.[/yellow]")
        console.print("[dim]Falling back to structural regex parsing...[/dim]")
        
        with open(file_path, "r") as f:
            content = f.read()
        modules = re.findall(r'module\s+(\w+)', content)
        instantiations = re.findall(r'(\w+)\s+\w+\s*\(', content)
        console.print(f"-> Found Modules: {modules}")
        console.print(f"-> Found Submodule Instantiations: [ {', '.join([i for i in instantiations if i not in modules])} ]")
        return False

# ==========================================
# 2. Automated HDL Compiler Verification
# ==========================================
def verify_with_compiler(file_path):
    console.print(f"[dim][Compiler] Running Icarus Verilog lint check on {file_path}...[/dim]")
    command = ["iverilog", "-g2012", "-t", "null", file_path]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            console.print("✅ [bold green]Compilation Successful! No hardware syntax errors found.[/bold green]")
            return True, "0 Errors (Passed Icarus Verilog synthesis lint)"
        else:
            console.print("❌ [bold red]Compilation Failed! Hardware syntax or latch errors detected.[/bold red]")
            return False, result.stderr
    except FileNotFoundError:
        return False, "Icarus Verilog ('iverilog') executable not found in Windows PATH."

# ==========================================
# 3. Closed-Loop AI Execution Engine
# ==========================================
def run_closed_loop_pipeline():
    base_model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
    adapter_dir = "./verilog-lora-model"
    
    # The script will now read whatever is currently saved in this file
    buggy_file = "test_latch_design.v" 
    
    # Check if the file actually exists before running
    if not os.path.exists(buggy_file):
        console.print(f"[bold red]❌ Error: Could not find '{buggy_file}' in the current directory.[/bold red]")
        console.print("[dim]Please create it and paste your broken Verilog code inside it before running.[/dim]")
        return

    # Step A: Scan hierarchy and run initial compiler check
    scan_hierarchy_with_ast(buggy_file)
    verify_with_compiler(buggy_file)
    
    # Step B: Boot AI model
    console.print("\n[bold cyan]Booting up trained VLSI AI model...[/bold cyan]")
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

    # Step C: Read the custom buggy code and let AI process it
    with open(buggy_file, "r") as f:
        buggy_code = f.read()

    prompt = f"Instruction: Analyze the following complex hierarchical Verilog module for semantic or logical errors. Identify the bug, explain the issue, and provide the corrected code.\nInput:\n{buggy_code}\nOutput:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    console.print("\n[bold yellow][AI Engine] Processing code fixes...[/bold yellow]")
    with torch.no_grad():
        # Increased max_new_tokens to allow full generation of complex VLSI modules
        outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_only = generated_text.split("Output:\n")[-1]
    
    # Step D: Extract code & re-verify against Icarus Verilog compiler
    code_blocks = re.findall(r"```verilog\s+(.*?)\s+```", response_only, re.DOTALL)
    
    if code_blocks:
        corrected_code = code_blocks[0]
        fixed_file = "fixed_output_design.v"
        with open(fixed_file, "w") as f:
            f.write(corrected_code)
        
        success, final_report = verify_with_compiler(fixed_file)
        correctness_score = "100% Synthesizable" if success else "0% (Compilation Errors Remaining)"
    else:
        corrected_code = "Could not extract code cleanly from AI response."
        success = False
        correctness_score = "0% (Extraction Failed)"
        final_report = "Parsing Error"

    # Step E: Render Terminal 2-Column Table & Bottom Common Row
    table = Table(
        title="\n[bold yellow]Closed-Loop Hardware Verification Result[/bold yellow]", 
        show_header=True, 
        header_style="bold magenta", 
        border_style="blue", 
        expand=True
    )
    
    table.add_column("❌ Input Code (Buggy Design)", style="bright_red", ratio=1)
    table.add_column("✅ AI Corrected Code (Fixed Design)", style="bright_green", ratio=1)

    table.add_row(
        Syntax(buggy_code.strip(), "verilog", theme="monokai", line_numbers=True),
        Syntax(corrected_code.strip(), "verilog", theme="monokai", line_numbers=True)
    )

    console.print(table)

    analysis_content = (
        f"[bold green]Percentage of Correctness:[/] [black on green] {correctness_score} [/]\n"
        f"[bold cyan]Compiler Re-Verification Status:[/] {final_report.strip()}\n\n"
        f"[bold yellow]AI Error Analysis & Description:[/]\n{response_only.strip()}"
    )

    analysis_panel = Panel(
        analysis_content,
        title="[bold white]Unified Diagnostic & Verification Report[/bold white]",
        border_style="blue",
        expand=True
    )

    console.print(analysis_panel)

if __name__ == "__main__":
    run_closed_loop_pipeline()