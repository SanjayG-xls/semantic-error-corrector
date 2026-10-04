# Verilog ML Bug Detection & Verification 🚀

An end-to-end Machine Learning pipeline and custom EDA-style GUI (Synopsys Verdi classic theme) for automatically detecting and patching semantic bugs in Verilog code. It runs closed-loop inference using a fine-tuned DeepSeek model (via LoRA) and verifies the synthesized output directly using the Icarus Verilog (`iverilog`) compiler.

## Features ✨
* **Custom Verdi-Themed GUI**: Built with PySide6, featuring a pitch-black `nWave` timing diagram viewer built with Matplotlib.
* **DeepSeek Integration**: Uses `deepseek-ai/deepseek-coder-1.3b-instruct` loaded in 4-bit precision alongside a custom LoRA adapter for highly accurate RTL fixes.
* **Closed-loop Verification**: Patched code is automatically parsed and linted using `iverilog` to ensure 100% synthesizability.
* **AMS Workspace Tree**: File hierarchy tree and side-by-side code editors with Verilog syntax highlighting.

## Prerequisites 🛠️
Before running the project, you need the following installed on your system:
1. **Python 3.10+**
2. **Icarus Verilog (`iverilog`)**: You must have Icarus Verilog installed and added to your system's `PATH` variable. 
   - *Windows users*: You can download the setup from [bleyer.org/icarus](http://bleyer.org/icarus/).

## Setup Instructions 💻 (For your friend)

1. **Clone the repository** (or download the project folder):
   ```bash
   git clone <your-repo-link-here>
   cd verilog-ml-project
   ```

2. **Create a virtual environment**:
   ```cmd
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Command Prompt (`cmd`)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **PowerShell**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Mac/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install all dependencies**:
   Run the following command to install everything from the `requirements.txt` file (includes PySide6, PyTorch, Transformers, Peft, etc.):
   ```cmd
   pip install -r requirements.txt
   ```
   *(Note: The first time you install PyTorch with CUDA support, the download may be large).*

5. **Ensure the LoRA model is present**:
   Make sure the `verilog-lora-model/` folder (which contains the custom AI weights) is present in the main directory.

## How to Run & Use 🚀

1. Ensure your virtual environment is activated (you should see `(.venv)` in your terminal).
2. Launch the GUI:
   ```cmd
   python gui_app.py
   ```
3. Once the GUI opens:
   - Paste your buggy Verilog code into the left editor panel.
   - Click the **"▶ Run Closed-Loop Parse & Verify Pipeline"** button.
   - *Note: The very first time you run this, it will download the base `deepseek-coder` model (~2.7GB) from HuggingFace. This happens automatically in the background and might take a few minutes.*
4. Review the results:
   - The verified, patched code will appear in the right editor.
   - Check the **AI Diagnostics & Metrics** tab at the bottom for the model's reasoning.
   - View the timing diagrams in the **nWave Timing Diagram View** tab.
